"""
Pre-flight checks for the DANN setup. Run this BEFORE launching training.

Both previous attempts failed silently and were only diagnosed after a full
training run. Each check below targets one of those failures directly:

  Guard 1  preprocessing parity between domains
           v1 produced NaN domain loss because the target pipeline applied
           ImageNet mean/std normalization while Ultralytics only divides
           by 255, putting the two domains on different scales.

  Guard 2  adversarial gradient actually reaches the backbone
           v2 had a .detach() in the path from the backbone to the GRL. The
           domain classifier trained fine, the run completed, and the whole
           thing was a no-adversary control by accident.

  Guard 3  domain heads see features with sane statistics

Usage:
    python check_dann_setup.py \\
        --weights baseline/runs/detect/srconly_day/weights/best.pt \\
        --source-data baseline/bdd100k_day.yaml \\
        --target-images "C:/.../dataset/yolo/images/train_night"
"""

import argparse
import sys

import torch
from ultralytics import YOLO
from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.data.utils import check_det_dataset
from ultralytics.utils import DEFAULT_CFG

from dann_core import (BACKBONE_END, GLOBAL_LAYER, LOCAL_LAYER,
                       DANNDetectionModel, GlobalDomainHead, LocalDomainHead,
                       grad_reverse)

PASS = "PASS"
FAIL = "FAIL"
failures = []


def report(name, ok, detail=""):
    tag = PASS if ok else FAIL
    print(f"[{tag}] {name}")
    if detail:
        for line in detail.splitlines():
            print(f"       {line}")
    if not ok:
        failures.append(name)


def make_loader(args, img_path, data, batch=4):
    dataset = build_yolo_dataset(args, img_path, batch, data, mode="train",
                                 rect=False, stride=32)
    return build_dataloader(dataset, batch, workers=0, shuffle=True, rank=-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--source-data", required=True, help="source yaml, e.g. bdd100k_day.yaml")
    ap.add_argument("--target-images", required=True, help="target image folder")
    ap.add_argument("--batch", type=int, default=4)
    opt = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}\n")

    # ------------------------------------------------------------------
    # Build model
    # ------------------------------------------------------------------
    # YOLO() is the stable public entry point. attempt_load_one_weight has
    # moved between Ultralytics versions, so it is deliberately not used.
    ckpt_model = YOLO(opt.weights).model
    data = check_det_dataset(opt.source_data)

    model = DANNDetectionModel(ckpt_model.yaml, nc=data["nc"], verbose=False)
    model.load(ckpt_model)
    model = model.to(device).train()

    report("model built with domain heads attached", True,
           f"local head on layer {LOCAL_LAYER}, "
           f"global head on layer {GLOBAL_LAYER}, backbone ends at {BACKBONE_END}")

    # ------------------------------------------------------------------
    # Guard 1: preprocessing parity
    # ------------------------------------------------------------------
    args = DEFAULT_CFG
    args.imgsz = 640
    args.mosaic = 0.0          # disable mosaic for a clean statistics check
    args.copy_paste = 0.0
    args.task = "detect"
    args.mode = "train"

    src_loader = make_loader(args, data["train"], data, opt.batch)
    tgt_loader = make_loader(args, opt.target_images, data, opt.batch)

    s = next(iter(src_loader))["img"].to(device).float() / 255.0
    t = next(iter(tgt_loader))["img"].to(device).float() / 255.0

    in_range = (s.min() >= 0.0 and s.max() <= 1.0 and
                t.min() >= 0.0 and t.max() <= 1.0)
    mean_gap = abs(s.mean().item() - t.mean().item())

    detail = (f"source  min={s.min():.3f} max={s.max():.3f} mean={s.mean():.3f}\n"
              f"target  min={t.min():.3f} max={t.max():.3f} mean={t.mean():.3f}\n"
              f"mean gap = {mean_gap:.3f} (night is genuinely darker, "
              f"so a gap up to ~0.15 is expected)")
    report("Guard 1: both domains in [0,1] with the same preprocessing",
           in_range and mean_gap < 0.25, detail)

    # ------------------------------------------------------------------
    # Guard 2: adversarial gradient reaches the backbone
    # ------------------------------------------------------------------
    model.zero_grad(set_to_none=True)
    model.dann_lambda = 1.0

    f_local, f_global = model.forward_backbone(t)
    local_logits = model.domain_local(grad_reverse(f_local, 1.0))
    global_logits = model.domain_global(grad_reverse(f_global, 1.0))
    d_loss = (LocalDomainHead.loss(local_logits, 1.0)
              + GlobalDomainHead.loss(global_logits, 1.0, 3.0))

    # Detection loss deliberately excluded: any gradient the backbone
    # receives here can only have come through the GRL.
    d_loss.backward()

    backbone_with_grad = 0
    backbone_grad_sum = 0.0
    for i in range(BACKBONE_END + 1):
        for p in model.model[i].parameters():
            if p.grad is not None:
                backbone_with_grad += 1
                backbone_grad_sum += p.grad.abs().sum().item()

    head_grad_sum = sum(p.grad.abs().sum().item()
                        for p in model.domain_local.parameters() if p.grad is not None)

    connected = backbone_with_grad > 0 and backbone_grad_sum > 1e-8
    detail = (f"backbone tensors receiving gradient : {backbone_with_grad}\n"
              f"backbone gradient magnitude         : {backbone_grad_sum:.6e}\n"
              f"local head gradient magnitude       : {head_grad_sum:.6e}\n"
              f"domain loss                         : {d_loss.item():.6f}")
    if not connected:
        detail += ("\n-> the adversary is disconnected. Look for a .detach(), "
                   "a torch.no_grad() block, or a requires_grad_(False) "
                   "between the backbone and the GRL.")
    report("Guard 2: adversarial gradient reaches the backbone", connected, detail)

    # ------------------------------------------------------------------
    # Guard 3: feature and loss sanity
    # ------------------------------------------------------------------
    finite = torch.isfinite(f_local).all() and torch.isfinite(f_global).all()
    loss_finite = torch.isfinite(d_loss)
    detail = (f"local  feat shape={tuple(f_local.shape)} "
              f"mean={f_local.mean():.4f} std={f_local.std():.4f}\n"
              f"global feat shape={tuple(f_global.shape)} "
              f"mean={f_global.mean():.4f} std={f_global.std():.4f}")
    report("Guard 3: features and domain loss are finite",
           bool(finite and loss_finite), detail)

    # ------------------------------------------------------------------
    # Guard 4: lambda=0 must leave the backbone untouched
    # ------------------------------------------------------------------
    model.zero_grad(set_to_none=True)
    f_local0, f_global0 = model.forward_backbone(t)
    l0 = (LocalDomainHead.loss(model.domain_local(grad_reverse(f_local0, 0.0)), 1.0)
          + GlobalDomainHead.loss(model.domain_global(grad_reverse(f_global0, 0.0)), 1.0, 3.0))
    l0.backward()

    zero_sum = sum(p.grad.abs().sum().item()
                   for i in range(BACKBONE_END + 1)
                   for p in model.model[i].parameters() if p.grad is not None)
    head_sum0 = sum(p.grad.abs().sum().item()
                    for p in model.domain_local.parameters() if p.grad is not None)

    detail = (f"backbone gradient at lambda=0 : {zero_sum:.6e} (should be ~0)\n"
              f"head gradient at lambda=0     : {head_sum0:.6e} (should be > 0, "
              f"the heads keep learning during warmup)")
    report("Guard 4: warmup isolates the backbone but still trains the heads",
           zero_sum < 1e-6 and head_sum0 > 1e-8, detail)

    # ------------------------------------------------------------------
    print()
    if failures:
        print(f"{len(failures)} check(s) failed: {', '.join(failures)}")
        print("Do not start training until these pass.")
        sys.exit(1)
    print("All checks passed. Safe to start training.")


if __name__ == "__main__":
    main()