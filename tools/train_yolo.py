#!/usr/bin/env python3
"""Config-driven launcher for YOLOv5 detection fine-tuning (stage 3).

One experiment = one yaml in configs/exp/ = one run dir. The launcher:
  1. reads the experiment yaml,
  2. writes the resolved config + git commit hash into the run dir,
  3. builds a per-experiment data yaml (so label fraction = just a
     different train list, everything else identical),
  4. execs yolov5/train.py with the mapped arguments.

Auto-resume: if <project>/<name>/weights/last.pt exists, training resumes
from it automatically (pass --no-resume to force a fresh start; the old
run dir is never deleted - rename it yourself first).

Usage:
  python tools/train_yolo.py --config configs/exp/baseline_coco_100.yaml
  nohup python tools/train_yolo.py --config configs/exp/baseline_coco_100.yaml > baseline_100.log 2>&1 &
"""
import argparse, json, subprocess, sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]


def git_hash():
    try:
        return subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                                       text=True).strip()
    except Exception:
        return "unknown"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--device", default=None, help="override cfg device (e.g. 0 or cpu)")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    name = cfg["name"]
    project = cfg.get("project", "runs_jepa/stage3")
    run_dir = REPO / project / name
    run_dir.mkdir(parents=True, exist_ok=True)

    # ---- provenance dump ----
    (run_dir / "resolved_config.json").write_text(json.dumps(
        {"config_file": str(args.config), "config": cfg, "git_hash": git_hash(),
         "argv": sys.argv}, indent=2))

    # ---- per-experiment data yaml (identical across arms except nothing;
    #      label fraction changes only the train list) ----
    base_data = yaml.safe_load(open(REPO / cfg["data_config"]))
    base_data["train"] = cfg["train_list"]
    base_data["val"] = cfg.get("val_list", "splits/modelsel.txt")
    data_yaml = run_dir / "data.yaml"
    base_data["path"] = str(REPO / base_data["path"])
    data_yaml.write_text(yaml.safe_dump(base_data, sort_keys=False))

    last = run_dir / "weights" / "last.pt"
    resume = last.exists() and not args.no_resume

    cmd = [sys.executable, str(REPO / "yolov5" / "train.py")]
    if resume:
        print(f"auto-resuming from {last}")
        cmd += ["--resume", str(last)]
    else:
        cmd += [
            "--data", str(data_yaml),
            "--weights", cfg.get("weights", ""),
            "--cfg", cfg.get("model_cfg", ""),
            "--hyp", cfg.get("hyp", str(REPO / "yolov5/data/hyps/hyp.scratch-low.yaml")),
            "--img", str(cfg.get("imgsz", 640)),
            "--batch-size", str(cfg.get("batch_size", 32)),
            "--epochs", str(cfg.get("epochs", 50)),
            "--seed", str(cfg.get("seed", 0)),
            "--project", str(REPO / project),
            "--name", name,
            "--exist-ok",
            "--workers", str(cfg.get("workers", 2)),
        ]
        if cfg.get("freeze"):
            cmd += ["--freeze", str(cfg["freeze"])]
    dev = args.device if args.device is not None else cfg.get("device")
    if dev is not None:
        cmd += ["--device", str(dev)]
    print("exec:", " ".join(cmd), flush=True)
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
