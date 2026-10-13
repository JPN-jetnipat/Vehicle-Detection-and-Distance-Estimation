"""
Export a plain YOLO checkpoint from a DANN checkpoint.

Ultralytics pickles the model object into the checkpoint. A DANN checkpoint
therefore carries a reference to DANNDetectionModel, and loading it from any
script that cannot import dann_core will fail. Stripping the domain heads
and rebuilding a standard DetectionModel makes the result loadable by
eval_all_splits.py, the yolo CLI, or anything else, with no DANN code on the
path.

The domain heads are discarded on purpose: they exist only to shape the
backbone during training and play no part in inference.

Usage:
    python strip_domain_heads.py \\
        --src runs/detect/dann_v3/weights/best.pt \\
        --dst runs/detect/dann_v3/weights/best_clean.pt
"""

import argparse
from copy import deepcopy

import torch
from ultralytics.nn.tasks import DetectionModel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    opt = ap.parse_args()

    ckpt = torch.load(opt.src, map_location="cpu", weights_only=False)
    dann_model = ckpt.get("ema") or ckpt["model"]

    clean = DetectionModel(dann_model.yaml, nc=dann_model.yaml["nc"], verbose=False)

    state = dann_model.float().state_dict()
    detector_state = {k: v for k, v in state.items() if not k.startswith("domain_")}
    missing, unexpected = clean.load_state_dict(detector_state, strict=False)

    if missing:
        raise SystemExit(f"detector weights missing after strip: {missing[:5]} ...")
    print(f"dropped {len(state) - len(detector_state)} domain-head tensors")

    clean.args = getattr(dann_model, "args", None)
    clean.names = dann_model.names
    clean.nc = dann_model.nc
    clean.stride = dann_model.stride

    out = {
        "model": deepcopy(clean).half(),
        "ema": None,
        "updates": None,
        "optimizer": None,
        "train_args": ckpt.get("train_args", {}),
        "date": ckpt.get("date"),
        "version": ckpt.get("version"),
        "epoch": -1,
        "best_fitness": ckpt.get("best_fitness"),
    }
    torch.save(out, opt.dst)
    print(f"wrote {opt.dst}")


if __name__ == "__main__":
    main()
