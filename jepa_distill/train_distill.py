#!/usr/bin/env python3
"""Stage 2: distill a frozen I-JEPA teacher into the YOLOv5 backbone.

Design + sign-off: docs/STAGE2_DISTILL_DESIGN.md. Config-driven like stage 3;
checkpoints every --ckpt-every iters; auto-resume; TensorBoard + CSV; cheap
day-vs-night linear probe on pooled student P4 features every K epochs.

Usage:
  python jepa_distill/train_distill.py --config configs/exp/distill_t1.yaml
  nohup python jepa_distill/train_distill.py --config configs/exp/distill_t1.yaml > distill_t1.log 2>&1 &
Smoke test (CPU ok, ~2 min):
  python jepa_distill/train_distill.py --config configs/exp/distill_t1.yaml --smoke
"""
import argparse, csv, json, math, random, subprocess, sys, time
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from jepa_distill.data import ImageListDataset
from jepa_distill.student import BackboneStudent, cosine_distill_loss, smooth_l1_distill_loss
from jepa_distill.teacher import build_teacher


def amp_dtype(name, device):
    if name == "bf16":
        if device.type == "cuda" and not torch.cuda.is_bf16_supported():
            print("WARNING: bf16 unsupported (e.g. Kaggle P100) - falling back to fp16")
            return torch.float16
        return torch.bfloat16
    return {"fp16": torch.float16, "fp32": torch.float32}[name]


def cosine_lr(step, total, warmup, peak, floor):
    if step < warmup:
        return peak * step / max(1, warmup)
    p = (step - warmup) / max(1, total - warmup)
    return floor + 0.5 * (peak - floor) * (1 + math.cos(math.pi * p))


@torch.no_grad()
def pooled_feats(student, ds, idxs, device, bs=64):
    feats = []
    for i in range(0, len(idxs), bs):
        batch = torch.stack([ds[j][0] for j in idxs[i:i + bs]]).to(device)
        f = student._forward_backbone(batch)          # (B, C, g, g), pre-projection
        feats.append(f.mean(dim=(2, 3)).float().cpu())
    return torch.cat(feats)


def day_night_probe(student, list_file, images_dir, attr_index, device, n_per_class=750, seed=0):
    """Train a linear day/night classifier on pooled P4 features. Returns val acc."""
    attrs = json.load(open(attr_index))
    ds = ImageListDataset(list_file, images_dir, train=False)
    day = [i for i, n in enumerate(ds.names) if attrs.get(n, {}).get("timeofday") == "daytime"]
    night = [i for i, n in enumerate(ds.names) if attrs.get(n, {}).get("timeofday") == "night"]
    rng = random.Random(seed)
    day, night = rng.sample(day, min(n_per_class, len(day))), rng.sample(night, min(n_per_class, len(night)))
    idxs = day + night
    y = torch.tensor([0] * len(day) + [1] * len(night))
    student.eval()
    X = pooled_feats(student, ds, idxs, device)
    student.train()
    perm = torch.randperm(len(idxs), generator=torch.Generator().manual_seed(seed))
    X, y = X[perm], y[perm]
    n_tr = int(0.8 * len(idxs))
    X = (X - X[:n_tr].mean(0)) / (X[:n_tr].std(0) + 1e-6)
    clf = torch.nn.Linear(X.shape[1], 2)
    opt = torch.optim.Adam(clf.parameters(), lr=1e-2, weight_decay=1e-4)
    for _ in range(300):
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(clf(X[:n_tr]), y[:n_tr])
        loss.backward(); opt.step()
    with torch.no_grad():
        acc = (clf(X[n_tr:]).argmax(1) == y[n_tr:]).float().mean().item()
    return acc


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", default=None, help="explicit ckpt path (default: auto-detect latest)")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--device", default=None)
    ap.add_argument("--smoke", action="store_true", help="64 images, 1 epoch, no ckpt - wiring test")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    run_dir = REPO / cfg.get("project", "runs_jepa/stage2") / cfg["name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    requested = args.device if args.device else cfg.get("device", "cuda:0")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print(f"WARNING: requested '{requested}' but torch.cuda.is_available() is False - "
              f"falling back to CPU. This will be 100-300x slower for this workload. "
              f"Check nvidia-smi before letting this continue.", flush=True)
        device = torch.device("cpu")
    else:
        device = torch.device(requested)
    print(f"using device: {device}", flush=True)
    dt = amp_dtype(cfg.get("amp_dtype", "bf16"), device)

    try:
        git = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        git = "unknown"
    (run_dir / "resolved_config.json").write_text(json.dumps(
        {"config": cfg, "git_hash": git, "argv": sys.argv, "amp_dtype": str(dt)}, indent=2))

    seed = cfg.get("seed", 0)
    torch.manual_seed(seed); random.seed(seed)

    ds = ImageListDataset(cfg["train_list"], cfg["images_dir"], train=True,
                          limit=64 if args.smoke else 0)
    dl = DataLoader(ds, batch_size=cfg.get("micro_batch", 64), shuffle=True, drop_last=True,
                    num_workers=0 if args.smoke else cfg.get("workers", 8), pin_memory=device.type == "cuda")
    accum = max(1, cfg.get("batch_size", 128) // cfg.get("micro_batch", 64))

    teacher = build_teacher(cfg["teacher"]["kind"], **cfg["teacher"].get("kwargs", {})).to(device)
    if dt != torch.float32 and device.type == "cuda":
        teacher = teacher.to(dt)
    student = BackboneStudent(cfg.get("model_yaml", "yolo11s.yaml"),
                              nc=cfg.get("nc", 5), teacher_dim=teacher.embed_dim,
                              student_init=cfg.get("student_init", "random")).to(device)
    print(f"teacher: {cfg['teacher']['kind']} dim={teacher.embed_dim} grid={teacher.grid}; "
          f"student P4: {student.p4_channels}ch {student.p4_grid}x{student.p4_grid}; "
          f"eff.batch={cfg.get('batch_size',128)} (micro {cfg.get('micro_batch',64)} x{accum})")

    loss_fn = smooth_l1_distill_loss if cfg.get("loss", "cosine") == "smooth_l1" else cosine_distill_loss
    epochs = 1 if args.smoke else cfg.get("epochs", 30)
    steps_per_epoch = len(dl) // accum
    total = epochs * steps_per_epoch
    peak, floor = cfg.get("lr", 1e-3), cfg.get("lr_floor", 1e-5)
    warm = cfg.get("warmup_epochs", 3) * steps_per_epoch
    opt = torch.optim.AdamW(student.parameters(), lr=peak, weight_decay=cfg.get("weight_decay", 0.05))
    scaler = torch.cuda.amp.GradScaler(enabled=(dt == torch.float16))

    start_epoch, gstep = 0, 0
    latest = run_dir / "ckpt_latest.pt"
    resume_from = args.resume or (str(latest) if latest.exists() and not args.no_resume else None)
    if resume_from and not args.smoke:
        ck = torch.load(resume_from, map_location="cpu", weights_only=False)
        student.load_state_dict(ck["student"]); opt.load_state_dict(ck["optimizer"])
        start_epoch, gstep = ck["epoch"] + 1, ck["gstep"]
        print(f"resumed from {resume_from} at epoch {start_epoch}")

    tb = SummaryWriter(str(run_dir / "tb"))
    csv_path = run_dir / "metrics.csv"
    if not csv_path.exists():
        csv_path.write_text("epoch,step,loss,lr,imgs_per_s,probe_acc\n")

    student.train()
    t_start = time.time(); n_seen = 0
    for epoch in range(start_epoch, epochs):
        running, t_ep = 0.0, time.time()
        opt.zero_grad()
        for it, (imgs, _) in enumerate(dl):
            imgs = imgs.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=dt, enabled=dt != torch.float32):
                with torch.no_grad():
                    tt = teacher(imgs.to(dt) if device.type == "cuda" and dt != torch.float32 else imgs)
                loss = loss_fn(student(imgs), tt) / accum
            scaler.scale(loss).backward()
            running += loss.item() * accum; n_seen += imgs.shape[0]
            if (it + 1) % accum == 0:
                lr = cosine_lr(gstep, total, warm, peak, floor)
                for g in opt.param_groups:
                    g["lr"] = lr
                scaler.step(opt); scaler.update(); opt.zero_grad()
                gstep += 1
                if gstep % 50 == 0:
                    ips = n_seen / (time.time() - t_start)
                    avg = running / (it + 1)
                    print(f"e{epoch} s{gstep}/{total} loss {avg:.4f} lr {lr:.2e} {ips:.0f} img/s", flush=True)
                    tb.add_scalar("loss", avg, gstep); tb.add_scalar("lr", lr, gstep)
                if not args.smoke and gstep % cfg.get("ckpt_every", 500) == 0:
                    torch.save({"student": student.state_dict(), "optimizer": opt.state_dict(),
                                "epoch": epoch, "gstep": gstep, "config": cfg}, latest)
        probe = ""
        if not args.smoke and (epoch + 1) % cfg.get("probe_every", 5) == 0:
            acc = day_night_probe(student, cfg["train_list"], cfg["images_dir"],
                                  cfg["attr_index"], device)
            probe = f"{acc:.4f}"
            print(f"[probe] epoch {epoch}: day/night linear acc = {acc:.4f}", flush=True)
            tb.add_scalar("probe/day_night_acc", acc, epoch)
        avg = running / max(1, len(dl))
        with open(csv_path, "a") as f:
            f.write(f"{epoch},{gstep},{avg:.5f},{opt.param_groups[0]['lr']:.3e},"
                    f"{n_seen / (time.time() - t_start):.1f},{probe}\n")
        if not args.smoke:
            torch.save({"student": student.state_dict(), "optimizer": opt.state_dict(),
                        "epoch": epoch, "gstep": gstep, "config": cfg}, latest)
            torch.save({"backbone": student.backbone_state_dict(), "config": cfg,
                        "epoch": epoch, "git_hash": git}, run_dir / "backbone_distilled.pt")
        print(f"epoch {epoch} done in {(time.time() - t_ep) / 60:.1f} min, loss {avg:.4f}", flush=True)
    if args.smoke:
        print("smoke test OK - wiring verified. NOTHING was saved (by design); "
              "re-run without --smoke for the real distillation.")
    else:
        print(f"done. backbone -> {run_dir / 'backbone_distilled.pt'}")


if __name__ == "__main__":
    main()
