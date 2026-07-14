#!/usr/bin/env python3
"""Stage 3 assembly: graft a distilled backbone into a YOLOv5 checkpoint.

Neck/head policy (fairness, brief section 7 - applied identically to T1 and
T2 arms): start from the COCO yolov5s.pt and replace ONLY backbone layers
0-6 with the distilled weights. All arms therefore share COCO neck/head
init and differ exclusively in backbone initialization.
(--neck-head random exists for a sanity ablation; not part of the core table.)

Usage:
  python tools/make_init_weights.py \
      --distilled runs_jepa/stage2/distill_t1/backbone_distilled.pt \
      --base yolov5s.pt --out weights/init_t1.pt
"""
import argparse, sys
from copy import deepcopy
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "yolov5"))

DISTILLED_LAYERS = 7  # layers 0..6 (through P4 C3) - keep in sync with jepa_distill/student.py


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--distilled", required=True, help="backbone_distilled.pt from stage 2")
    ap.add_argument("--base", default="yolov5s.pt", help="COCO checkpoint providing neck/head (+layers 7-9)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--neck-head", choices=["coco", "random"], default="coco")
    ap.add_argument("--seed", type=int, default=0, help="seed for --neck-head random")
    args = ap.parse_args()

    from models.yolo import DetectionModel

    dist = torch.load(args.distilled, map_location="cpu")
    bstate = dist["backbone"]

    base_ck = torch.load(args.base, map_location="cpu")
    model = base_ck["model"].float()
    if args.neck_head == "random":
        torch.manual_seed(args.seed)
        model = DetectionModel(cfg=model.yaml, ch=3)

    own = model.state_dict()
    replaced, skipped = 0, []
    for k, v in bstate.items():
        layer_idx = int(k.split(".")[1])
        assert layer_idx < DISTILLED_LAYERS, f"unexpected layer in distilled state: {k}"
        if k in own and own[k].shape == v.shape:
            own[k] = v
            replaced += 1
        else:
            skipped.append(k)
    if skipped:
        raise SystemExit(f"FLAG: {len(skipped)} distilled keys did not match the base model "
                         f"(scale mismatch? {skipped[:3]}) - refusing to write a silently-broken init.")
    model.load_state_dict(own)

    out_ck = {"model": deepcopy(model).half(), "optimizer": None, "best_fitness": None,
              "epoch": -1, "date": None,
              "provenance": {"distilled": args.distilled, "base": args.base,
                             "neck_head": args.neck_head,
                             "distilled_epoch": dist.get("epoch"), "git_hash": dist.get("git_hash")}}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(out_ck, args.out)
    print(f"replaced {replaced} backbone tensors (layers 0-{DISTILLED_LAYERS - 1}), "
          f"neck/head = {args.neck_head} -> {args.out}")


if __name__ == "__main__":
    main()
