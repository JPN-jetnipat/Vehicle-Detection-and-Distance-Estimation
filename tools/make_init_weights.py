#!/usr/bin/env python3
"""Stage 3 assembly: graft a distilled backbone into a YOLOv11 checkpoint.

Port of the YOLOv5 version from the archived project, rewritten for the
ultralytics checkpoint format.

Neck/head policy (fairness, applied identically to every JEPA arm): start from
the COCO `yolo11s.pt` and replace ONLY backbone layers 0-6 with the distilled
weights. All arms therefore share COCO init for layers 7-10 (P5 Conv, C3k2,
SPPF, C2PSA), the neck and the head, and differ exclusively in backbone
initialization.

The output keeps nc=80 exactly like the stock `yolo11s.pt`, so the fine-tune
path is byte-identical to the BASE arm's: ultralytics reshapes the head to 5
classes at train time, the same way and at the same point, for every arm.

Usage:
  python tools/make_init_weights.py \
      --distilled runs_jepa/stage2/distill_t1/backbone_distilled.pt \
      --base yolo11s.pt --out weights/init_t1.pt
  python tools/make_init_weights.py --verify weights/init_t1.pt \
      --distilled runs_jepa/stage2/distill_t1/backbone_distilled.pt --base yolo11s.pt
"""
import argparse
from copy import deepcopy
from pathlib import Path

import torch

DISTILLED_LAYERS = 7  # layers 0..6 (through the P4 C3k2) - keep in sync with jepa_distill/student.py


def load_ckpt(path):
    """torch>=2.6 flipped torch.load's weights_only default to True, which
    refuses to unpickle the DetectionModel object inside an ultralytics
    checkpoint. These files are ours / Meta's, not untrusted input."""
    return torch.load(path, map_location="cpu", weights_only=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--distilled", required=True, help="backbone_distilled.pt from stage 2")
    ap.add_argument("--base", default="yolo11s.pt",
                    help="COCO checkpoint providing layers 7-10 + neck + head")
    ap.add_argument("--out", help="where to write the assembled init (omit with --verify)")
    ap.add_argument("--verify", help="instead of writing, check an existing init file")
    args = ap.parse_args()
    if not args.out and not args.verify:
        ap.error("need --out (to build) or --verify (to check)")

    dist = load_ckpt(args.distilled)
    bstate = dist["backbone"]

    base_ck = load_ckpt(args.base)
    base_model = base_ck.get("ema") or base_ck["model"]
    base_state = {k: v.clone() for k, v in base_model.float().state_dict().items()}

    def same(a, b):
        """float tensors were round-tripped through half() at save time, so
        compare them loosely; integer buffers (num_batches_tracked) exactly."""
        if a.shape != b.shape:
            return False
        if a.is_floating_point() and b.is_floating_point():
            return torch.allclose(a.float(), b.float(), atol=1e-3)
        return bool(torch.equal(a.long(), b.long()))

    if args.verify:
        got = load_ckpt(args.verify)
        gmodel = got.get("ema") or got["model"]
        gstate = gmodel.float().state_dict()
        bad_backbone = [k for k, v in bstate.items()
                        if k not in gstate or not same(gstate[k], v)]
        rest = [k for k in base_state
                if k.split(".")[1].isdigit() and int(k.split(".")[1]) >= DISTILLED_LAYERS
                and not same(gstate[k], base_state[k])]
        print(f"backbone tensors matching the distilled file: "
              f"{len(bstate) - len(bad_backbone)}/{len(bstate)}")
        print(f"non-backbone tensors differing from {args.base}: {len(rest)} (expected 0)")
        print(f"provenance: {got.get('provenance')}")
        raise SystemExit(0 if not bad_backbone and not rest else 1)

    model = base_model.float()
    own = model.state_dict()
    replaced, skipped = 0, []
    for k, v in bstate.items():
        layer_idx = int(k.split(".")[1])
        assert layer_idx < DISTILLED_LAYERS, f"unexpected layer in distilled state: {k}"
        if k in own and own[k].shape == v.shape:
            own[k] = v.to(own[k].dtype)
            replaced += 1
        else:
            skipped.append(k)
    if skipped:
        raise SystemExit(
            f"FLAG: {len(skipped)} distilled keys did not match the base model "
            f"(width/scale mismatch? distilled from a different yaml? {skipped[:3]}) "
            f"- refusing to write a silently-broken init.")

    # Every trainable backbone tensor in layers 0-6 must have been covered.
    # A partial graft is the failure mode that would look fine and score like BASE.
    expected = [k for k in own if k.split(".")[0] == "model"
                and k.split(".")[1].isdigit() and int(k.split(".")[1]) < DISTILLED_LAYERS]
    missed = [k for k in expected if k not in bstate]
    if missed:
        raise SystemExit(
            f"FLAG: {len(missed)}/{len(expected)} tensors in layers 0-{DISTILLED_LAYERS - 1} "
            f"were NOT in the distilled file, e.g. {missed[:3]} - the graft would be "
            f"partial and the arm would silently be part-COCO. Refusing to write.")

    model.load_state_dict(own)

    out_ck = {
        "model": deepcopy(model).half(),
        "ema": None,
        "updates": None,
        "optimizer": None,
        "best_fitness": None,
        "epoch": -1,
        "date": base_ck.get("date"),
        "version": base_ck.get("version"),
        "train_args": base_ck.get("train_args", {}),
        "provenance": {
            "distilled": str(args.distilled),
            "base": str(args.base),
            "distilled_layers": f"0-{DISTILLED_LAYERS - 1}",
            "distilled_epoch": dist.get("epoch"),
            "git_hash": dist.get("git_hash"),
        },
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(out_ck, args.out)
    print(f"replaced {replaced} backbone tensors (layers 0-{DISTILLED_LAYERS - 1}, "
          f"all {len(expected)} covered), everything else from {args.base} -> {args.out}")


if __name__ == "__main__":
    main()
