"""
Entry point for the DANN runs.

Three arms are needed to make a claim about the adversary:

  A  srconly_day     source-only baseline, already trained (test_night 0.471)
  B  finetune_noadv  best.pt fine-tuned with lambda fixed at 0
  C  dann_v3         best.pt fine-tuned with the lambda schedule active

Evidence that DANN works is  C > B,  not  C > A. Arm C also gets 30 extra
epochs of fine-tuning, so comparing it against A alone cannot separate the
adversary from the extra training. Arm B is what v2 accidentally became when
.detach() severed the gradient path -- this time it is a deliberate control.

Run arm B first. If B already closes most of the gap, the story is about
fine-tuning schedule, not domain adaptation, and that is worth knowing
before spending time tuning lambda.

Usage (from the DANN folder):

    python train_dann.py --arm noadv
    python train_dann.py --arm dann
"""

import argparse
from pathlib import Path

from dann_trainer import DANNTrainer

# --------------------------------------------------------------------------
# Paths -- edit these if the folder layout changes
# --------------------------------------------------------------------------
BASELINE_WEIGHTS = "baseline/runs/detect/srconly_day/weights/best.pt"
SOURCE_DATA = "baseline/bdd100k_day.yaml"
TARGET_IMAGES = (
    "C:/Users/USER/OneDrive/เดสก์ท็อป/baby/bdd100k_split_handoff/"
    "dataset/yolo/images/train_night"
)

# --------------------------------------------------------------------------
# Training configuration
#
# Augmentation values are copied unchanged from the baseline config so the
# only difference between arms is the adversary. lr0 is the one deliberate
# departure: the baseline trained from COCO weights at the Ultralytics
# default, but these arms fine-tune an already-converged checkpoint, and
# restarting at 0.01 would wash out the 100 epochs already invested.
# --------------------------------------------------------------------------
COMMON = dict(
    model=BASELINE_WEIGHTS,
    data=SOURCE_DATA,
    epochs=20,
    seed=42,
    imgsz=640,
    batch=16,
    workers=2,
    cache=False,
    optimizer="SGD",      # pinned, not "auto" -- see note below
    # lr0 was 0.001 in the first attempt. Both arms peaked at epoch 2 and
    # got steadily worse for the remaining 28 epochs (fitness fell every
    # epoch after that) -- that is catastrophic forgetting from too high a
    # learning rate on an already-converged checkpoint, not a DANN result.
    # 0.0001 is ten times lower.
    lr0=0.0001,
    lrf=0.01,
    cos_lr=True,           # smoother decay than linear, reduces oscillation
    warmup_epochs=3.0,
    patience=10,           # stop if fitness does not improve for 10 epochs
    exist_ok=True,
    # augmentation identical to the baseline
    hsv_v=0.25,
    close_mosaic=20,
    box=9.0,
    copy_paste=0.2,
    cls=0.7,
    scale=0.7,
    translate=0.15,
)

# Why optimizer is pinned to SGD: Ultralytics 8.4.120 resolves
# optimizer="auto" to MuSGD, and the resolution depends on the installed
# version. Pinning it means arm B and arm C are guaranteed to share an
# optimizer, and that a rerun months from now reproduces today's numbers.

ARMS = {
    "noadv": dict(
        name="finetune_noadv",
        lambda_max=0.0,       # adversary disabled -- the control arm
        warmup_epochs=20,     # covers the whole run (COMMON["epochs"]), lambda stays 0
    ),
    "dann": dict(
        name="dann_v3",
        lambda_max=1.0,
        warmup_epochs=3,
    ),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=list(ARMS))
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--lambda-max", type=float, default=None)
    ap.add_argument("--local-weight", type=float, default=1.0)
    ap.add_argument("--global-weight", type=float, default=1.0)
    ap.add_argument("--focal-gamma", type=float, default=3.0)
    opt = ap.parse_args()

    arm = ARMS[opt.arm]
    cfg = dict(COMMON)
    cfg["name"] = arm["name"]
    if opt.epochs is not None:
        cfg["epochs"] = opt.epochs
    if opt.batch is not None:
        cfg["batch"] = opt.batch

    if not Path(BASELINE_WEIGHTS).exists():
        raise SystemExit(f"baseline weights not found: {BASELINE_WEIGHTS}")

    DANNTrainer.target_img_path = TARGET_IMAGES
    DANNTrainer.lambda_max = (opt.lambda_max if opt.lambda_max is not None
                              else arm["lambda_max"])
    DANNTrainer.warmup_epochs = arm["warmup_epochs"]
    DANNTrainer.local_weight = opt.local_weight
    DANNTrainer.global_weight = opt.global_weight
    DANNTrainer.focal_gamma = opt.focal_gamma

    print(f"arm            : {opt.arm} -> runs/detect/{cfg['name']}")
    print(f"lambda_max     : {DANNTrainer.lambda_max}")
    print(f"warmup_epochs  : {DANNTrainer.warmup_epochs}")
    print(f"target images  : {TARGET_IMAGES}")
    print(f"epochs / batch : {cfg['epochs']} / {cfg['batch']}")
    print()

    trainer = DANNTrainer(overrides=cfg)
    trainer.train()


if __name__ == "__main__":
    main()