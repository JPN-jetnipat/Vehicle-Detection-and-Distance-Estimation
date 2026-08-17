#!/usr/bin/env python3
"""Train one YOLO11s arm from a hyperparameter recipe YAML (configs/hyp/*.yaml).

One config = one arm = one output dir (runs/detect/<name>/), matching the
old project's convention. model/data are NOT baked into the hyp YAMLs on
purpose - they're the same across every arm (yolo11s.pt, the shared
5-class dataset), only the training recipe differs, so they're script args
instead with sensible defaults.

Every key in the hyp YAML is passed straight through to ultralytics'
model.train(**kwargs) - nothing is silently added or renamed. If a key
isn't in the YAML, ultralytics' own default applies; the YAMLs are
deliberately lean (only the values that were actually agreed on), not a
full dump of every hyperparameter.

RAM note (32GB shared host, not A40 VRAM - same constraint the old
project's RUNBOOK hit before): every finalized hyp YAML already sets
workers=2 and cache=false. Don't override those from the command line.

Usage:
  python tools/train_yolo.py --hyp configs/hyp/set1_japan_night_aug.yaml --name set1_japan_night_aug
  python tools/train_yolo.py --hyp configs/hyp/baseline.yaml --name baseline_default
  python tools/train_yolo.py --hyp configs/hyp/set2_field_imbalance.yaml --name set2_field_imbalance

  # resume after an interruption (auto-finds runs/detect/<name>/weights/last.pt):
  python tools/train_yolo.py --hyp configs/hyp/set1_japan_night_aug.yaml --name set1_japan_night_aug --resume

Logging: this script only prints ultralytics' own progress output. Redirect
it to a file yourself so it survives a dropped session, same as the old
project's convention:
  nohup python tools/train_yolo.py --hyp configs/hyp/set1_japan_night_aug.yaml \
      --name set1_japan_night_aug > set1_japan_night_aug.log 2>&1 &
ultralytics also writes its own structured outputs (results.csv, args.yaml,
weights/) into runs/detect/<name>/ regardless - that's the primary record,
the .log file is just for following along live.
"""
import argparse
import sys
from pathlib import Path

import yaml


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hyp", required=True, help="path to a configs/hyp/*.yaml recipe")
    ap.add_argument("--name", required=True, help="run name - output lands in <project>/<name>/")
    ap.add_argument("--model", default="yolo11s.pt", help="starting weights (default: COCO-pretrained yolo11s.pt)")
    ap.add_argument("--data", default="configs/data/bdd100k_vehicle5.yaml", help="dataset yaml")
    ap.add_argument("--project", default="runs/detect", help="parent dir for run outputs")
    ap.add_argument("--resume", action="store_true",
                    help="resume from <project>/<name>/weights/last.pt if it exists")
    args = ap.parse_args()

    # Force --project to an absolute path anchored at THIS repo, not cwd and
    # not ultralytics' own global runs_dir setting (~/.config/Ultralytics/
    # settings.json). Found the hard way: on a machine that had previously
    # run training from an old checkout, a relative project="runs/detect"
    # resolved against that stale global setting and silently wrote results
    # into the OLD project's directory (doubled path and all:
    # <old_repo>/runs/detect/runs/detect/<name>). Same class of bug as the
    # dataset-symlink/split-path issues this project already hit once -
    # implicit path resolution against something other than "this repo,
    # right here" is not safe to rely on across machines/teammates.
    repo_root = Path(__file__).resolve().parent.parent
    if not Path(args.project).is_absolute():
        args.project = str(repo_root / args.project)

    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit("ultralytics not installed - pip install ultralytics")

    hyp_path = Path(args.hyp)
    if not hyp_path.exists():
        sys.exit(f"FLAG: hyp file not found: {hyp_path}")
    cfg = yaml.safe_load(hyp_path.read_text()) or {}
    if not isinstance(cfg, dict) or not cfg:
        sys.exit(f"FLAG: {hyp_path} did not parse into a non-empty dict of hyperparameters")

    # safety net - these two are the whole point of the RAM fix, refuse to
    # run without them rather than silently training with ultralytics'
    # defaults (workers=8, which is what crashed the old project's server).
    if cfg.get("workers") != 2:
        sys.exit(f"FLAG: {hyp_path} has workers={cfg.get('workers')!r}, expected 2 (RAM constraint). Refusing to run.")
    if cfg.get("cache") not in (False, "false", None) or str(cfg.get("cache")).lower() in ("true", "ram", "disk"):
        if cfg.get("cache") not in (False,):
            sys.exit(f"FLAG: {hyp_path} has cache={cfg.get('cache')!r}, expected false (RAM constraint). Refusing to run.")

    resume_ckpt = Path(args.project) / args.name / "weights" / "last.pt"
    if args.resume and resume_ckpt.exists():
        print(f"resuming from {resume_ckpt}")
        model = YOLO(str(resume_ckpt))
        cfg["resume"] = True
    else:
        if args.resume:
            print(f"--resume given but {resume_ckpt} doesn't exist yet - starting fresh")
        model = YOLO(args.model)

    print(f"=== arm: {args.name} ===")
    print(f"model={args.model}  data={args.data}")
    print("hyperparameters from", hyp_path, ":")
    print(yaml.dump(cfg, sort_keys=False))

    model.train(data=args.data, name=args.name, project=args.project, **cfg)


if __name__ == "__main__":
    main()
