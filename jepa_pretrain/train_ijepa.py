#!/usr/bin/env python3
"""Stage 1: I-JEPA pretraining of ViT-B/16 on unlabeled BDD100K train images
(the T2 teacher). Faithful to the official facebookresearch/ijepa training
loop (vendored at third_party/ijepa): multi-block MaskCollator, EMA target
encoder with stop-gradient, targets = masked OUTPUT of the target encoder,
loss only on target positions. Additions for this project:
  - per-image oversampling of night / dawn-dusk / rainy (attribute index)
  - collapse alarm (std of target embeddings)
  - checkpoint/auto-resume, TensorBoard + CSV, config+git provenance
  - bf16/fp16/fp32 autocast switch (Kaggle P100 has no bf16)

Loss note: the brief and paper say L2; the official code uses smooth_l1.
Default here: l2 (per brief). `loss: smooth_l1` in the config switches to the
official behavior. Recorded as a documented deviation either way.

Usage:
  python jepa_pretrain/train_ijepa.py --config configs/exp/pretrain_t2.yaml
  nohup python jepa_pretrain/train_ijepa.py --config configs/exp/pretrain_t2.yaml > pretrain_t2.log 2>&1 &
Smoke test (~2 min CPU):  add --smoke
"""
import argparse, copy, json, math, random, subprocess, sys, time
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "third_party" / "ijepa"))

import src.models.vision_transformer as vit                      # noqa: E402
from src.masks.multiblock import MaskCollator                    # noqa: E402
from src.masks.utils import apply_masks                          # noqa: E402
from src.utils.tensors import repeat_interleave_batch            # noqa: E402
from src.utils.schedulers import WarmupCosineSchedule, CosineWDSchedule  # noqa: E402

from jepa_distill.data import ImageListDataset                   # noqa: E402


class ImagesOnly(ImageListDataset):
    def __getitem__(self, i):
        return super().__getitem__(i)[0]


def build_sample_weights(names, attr_index, over):
    """Multiplicative weights; logs expected epoch composition."""
    attrs = json.load(open(attr_index))
    ws = []
    for n in names:
        a = attrs.get(n, {})
        w = 1.0
        if a.get("timeofday") == "night":
            w *= over.get("night", 1.0)
        elif a.get("timeofday") == "dawn/dusk":
            w *= over.get("dawn_dusk", 1.0)
        if a.get("weather") == "rainy":
            w *= over.get("rainy", 1.0)
        ws.append(w)
    tot = sum(ws)
    comp = {}
    for key, sel in [("night", lambda a: a.get("timeofday") == "night"),
                     ("dawn/dusk", lambda a: a.get("timeofday") == "dawn/dusk"),
                     ("rainy", lambda a: a.get("weather") == "rainy")]:
        comp[key] = round(sum(w for n, w in zip(names, ws) if sel(attrs.get(n, {}))) / tot, 4)
    return ws, comp


def param_groups(encoder, predictor):
    """Official policy: no weight decay on biases / 1-D params."""
    return [
        {"params": [p for n, p in encoder.named_parameters() if "bias" not in n and len(p.shape) != 1]},
        {"params": [p for n, p in predictor.named_parameters() if "bias" not in n and len(p.shape) != 1]},
        {"params": [p for n, p in encoder.named_parameters() if "bias" in n or len(p.shape) == 1],
         "WD_exclude": True, "weight_decay": 0},
        {"params": [p for n, p in predictor.named_parameters() if "bias" in n or len(p.shape) == 1],
         "WD_exclude": True, "weight_decay": 0},
    ]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--device", default=None)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    run_dir = REPO / cfg.get("project", "runs_jepa/stage1") / cfg["name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device else cfg.get("device", "cuda:0") if torch.cuda.is_available() else "cpu")
    dt = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[cfg.get("amp_dtype", "bf16")]
    if dt == torch.bfloat16 and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        print("WARNING: no bf16 on this GPU - falling back to fp16"); dt = torch.float16

    try:
        git = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        git = "unknown"
    seed = cfg.get("seed", 0)
    torch.manual_seed(seed); random.seed(seed)

    # ---- data (train split only; val NEVER enters pretraining) ----
    size = cfg.get("image_size", 224)
    ds = ImagesOnly(cfg["train_list"], cfg["images_dir"], train=True, size=size,
                    limit=64 if args.smoke else 0)
    weights, comp = build_sample_weights(ds.names, cfg["attr_index"], cfg.get("oversample", {}))
    print(f"{len(ds)} images; oversample={cfg.get('oversample', {})}; "
          f"expected epoch composition: {comp}")
    sampler = WeightedRandomSampler(weights, num_samples=len(ds), replacement=True)
    # masking spec per brief section 6 == official defaults for I-JEPA
    collator = MaskCollator(input_size=(size, size), patch_size=cfg.get("patch_size", 16),
                            enc_mask_scale=tuple(cfg.get("enc_mask_scale", (0.85, 1.0))),
                            pred_mask_scale=tuple(cfg.get("pred_mask_scale", (0.15, 0.2))),
                            aspect_ratio=tuple(cfg.get("aspect_ratio", (0.75, 1.5))),
                            nenc=1, npred=4, allow_overlap=False, min_keep=10)
    micro = cfg.get("micro_batch", 128)
    dl = DataLoader(ds, batch_size=micro, sampler=sampler, collate_fn=collator, drop_last=True,
                    num_workers=0 if args.smoke else cfg.get("workers", 8),
                    pin_memory=device.type == "cuda")
    accum = max(1, cfg.get("batch_size", 256) // micro)

    # ---- models ----
    arch = cfg.get("arch", "vit_base")
    encoder = vit.__dict__[arch](img_size=[size], patch_size=cfg.get("patch_size", 16)).to(device)
    predictor = vit.vit_predictor(num_patches=encoder.patch_embed.num_patches,
                                  embed_dim=encoder.embed_dim,
                                  predictor_embed_dim=cfg.get("pred_embed_dim", 384),
                                  depth=cfg.get("pred_depth", 6),
                                  num_heads=encoder.num_heads).to(device)
    target_encoder = copy.deepcopy(encoder)          # init identical (brief section 6)
    for p in target_encoder.parameters():
        p.requires_grad = False                      # stop-gradient: EMA only

    # ---- optimization (official schedules, LR scaled to our batch) ----
    epochs = 1 if args.smoke else cfg.get("epochs", 300)
    ipe = len(dl) // accum
    eff_batch = micro * accum
    peak_lr = cfg.get("lr", 1e-3 * eff_batch / 2048)  # linear scaling from paper's 1e-3@2048
    opt = torch.optim.AdamW(param_groups(encoder, predictor))
    sched = WarmupCosineSchedule(opt, warmup_steps=int(cfg.get("warmup_epochs", 15) * ipe),
                                 start_lr=cfg.get("start_lr", 2e-4) * eff_batch / 2048,
                                 ref_lr=peak_lr,
                                 final_lr=cfg.get("final_lr", 1e-6),
                                 T_max=int(1.25 * epochs * ipe))
    wd_sched = CosineWDSchedule(opt, ref_wd=cfg.get("wd", 0.04), final_wd=cfg.get("final_wd", 0.4),
                                T_max=int(1.25 * epochs * ipe))
    ema = cfg.get("ema", (0.996, 1.0))
    momentum = (ema[0] + i * (ema[1] - ema[0]) / (ipe * epochs * 1.25)
                for i in range(int(ipe * epochs * 1.25) + 1))
    scaler = torch.cuda.amp.GradScaler(enabled=(dt == torch.float16))

    (run_dir / "resolved_config.json").write_text(json.dumps(
        {"config": cfg, "git_hash": git, "argv": sys.argv, "amp_dtype": str(dt),
         "eff_batch": eff_batch, "peak_lr": peak_lr, "ipe": ipe,
         "epoch_composition": comp}, indent=2))

    # ---- resume ----
    start_epoch = 0
    latest = run_dir / "ckpt_latest.pt"
    if latest.exists() and not args.no_resume and not args.smoke:
        ck = torch.load(latest, map_location="cpu")
        encoder.load_state_dict(ck["encoder"]); predictor.load_state_dict(ck["predictor"])
        target_encoder.load_state_dict(ck["target_encoder"]); opt.load_state_dict(ck["optimizer"])
        start_epoch = ck["epoch"] + 1
        for _ in range(start_epoch * ipe):           # fast-forward schedulers + EMA
            sched.step(); wd_sched.step(); next(momentum)
        print(f"resumed at epoch {start_epoch}")

    tb = SummaryWriter(str(run_dir / "tb"))
    csv_path = run_dir / "metrics.csv"
    if not csv_path.exists():
        csv_path.write_text("epoch,loss,target_std,lr,wd,imgs_per_s\n")

    print(f"arch={arch} eff_batch={eff_batch} peak_lr={peak_lr:.2e} epochs={epochs} ipe={ipe}")
    t0, seen = time.time(), 0
    for epoch in range(start_epoch, epochs):
        running, std_running, t_ep = 0.0, 0.0, time.time()
        opt.zero_grad()
        for it, (imgs, masks_enc, masks_pred) in enumerate(dl):
            imgs = imgs.to(device, non_blocking=True)
            masks_enc = [m.to(device) for m in masks_enc]
            masks_pred = [m.to(device) for m in masks_pred]
            with torch.autocast(device_type=device.type, dtype=dt, enabled=dt != torch.float32):
                with torch.no_grad():                              # -- targets
                    h = target_encoder(imgs)
                    h = F.layer_norm(h, (h.size(-1),))             # official: LN over feature dim
                    tstd = h.std(dim=0).mean().item()              # collapse alarm signal
                    h = apply_masks(h, masks_pred)
                    h = repeat_interleave_batch(h, len(imgs), repeat=len(masks_enc))
                z = encoder(imgs, masks_enc)                       # -- context
                z = predictor(z, masks_enc, masks_pred)
                loss = (F.smooth_l1_loss(z, h) if cfg.get("loss", "l2") == "smooth_l1"
                        else F.mse_loss(z, h)) / accum
            scaler.scale(loss).backward()
            running += loss.item() * accum; std_running += tstd; seen += len(imgs)
            if (it + 1) % accum == 0:
                lr = sched.step(); wd = wd_sched.step()
                scaler.step(opt); scaler.update(); opt.zero_grad()
                with torch.no_grad():                              # -- EMA (official)
                    m = next(momentum)
                    for pq, pk in zip(encoder.parameters(), target_encoder.parameters()):
                        pk.data.mul_(m).add_((1.0 - m) * pq.detach().data)
                gstep = epoch * ipe + (it + 1) // accum
                if gstep % 50 == 0:
                    avg, astd = running / (it + 1), std_running / (it + 1)
                    ips = seen / (time.time() - t0)
                    print(f"e{epoch} s{gstep} loss {avg:.4f} tstd {astd:.4f} lr {lr:.2e} {ips:.0f} img/s", flush=True)
                    tb.add_scalar("loss", avg, gstep); tb.add_scalar("target_std", astd, gstep)
                    if astd < 0.01:
                        print("!!! COLLAPSE ALARM: target embedding std ~ 0. "
                              "Audit stop-grad + EMA wiring. Stopping is recommended.", flush=True)
        avg = running / max(1, len(dl)); astd = std_running / max(1, len(dl))
        with open(csv_path, "a") as f:
            f.write(f"{epoch},{avg:.5f},{astd:.5f},{opt.param_groups[0]['lr']:.3e},"
                    f"{opt.param_groups[0].get('weight_decay', 0):.4f},{seen / (time.time() - t0):.1f}\n")
        if not args.smoke:
            torch.save({"encoder": encoder.state_dict(), "predictor": predictor.state_dict(),
                        "target_encoder": target_encoder.state_dict(),
                        "optimizer": opt.state_dict(), "epoch": epoch, "config": cfg,
                        "git_hash": git}, latest)
            if (epoch + 1) % cfg.get("keep_every", 50) == 0:
                torch.save({"target_encoder": target_encoder.state_dict(), "epoch": epoch,
                            "config": cfg}, run_dir / f"target_encoder_e{epoch + 1}.pt")
        print(f"epoch {epoch}: loss {avg:.4f} tstd {astd:.4f} ({(time.time() - t_ep) / 60:.1f} min)", flush=True)
    if not args.smoke:
        torch.save({"target_encoder": target_encoder.state_dict(), "epoch": epochs - 1,
                    "config": cfg}, run_dir / "target_encoder_final.pt")
        print(f"done -> {run_dir / 'target_encoder_final.pt'} (this is the T2 teacher)")


if __name__ == "__main__":
    main()
