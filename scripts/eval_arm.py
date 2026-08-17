"""Evaluate an already-trained arm's checkpoint on its eval_splits, without retraining.

Usage:
    python scripts/eval_arm.py --config configs/experiments/set1_night_aug.yaml
    python scripts/eval_arm.py --config configs/experiments/set1_night_aug.yaml --weights runs/detect/set1_night_aug/weights/best.pt

Reuses train_arm.py's metrics_to_rows/append_metrics/export_excel, so results
land in the same results/metrics_all_arms.csv|xlsx under the config's
arm_name, alongside rows from actual training runs.

Must be run with the repo root as the working directory (same as train_arm.py).
"""

import argparse
from datetime import datetime, timezone
from pathlib import Path

import yaml
from ultralytics import YOLO

import train_arm

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--weights",
        type=Path,
        default=None,
        help="Path to a .pt checkpoint. Defaults to runs/detect/<train.name>/weights/best.pt from the config.",
    )
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    arm_name = cfg["arm_name"]
    train_kwargs = cfg["train"]

    weights = args.weights or ROOT / "runs" / "detect" / train_kwargs["name"] / "weights" / "best.pt"
    if not weights.exists():
        raise FileNotFoundError(f"No weights at {weights} - train this arm first, or pass --weights explicitly.")

    model = YOLO(weights)
    timestamp = datetime.now(timezone.utc).isoformat()
    rows: list[dict] = []
    for split in cfg["eval_splits"]:
        split_metrics = model.val(
            data=split["data"],
            split="val",
            imgsz=train_kwargs["imgsz"],
            batch=train_kwargs["batch"],
        )
        rows.extend(train_arm.metrics_to_rows(arm_name, split["name"], split_metrics, timestamp))

    train_arm.append_metrics(rows)
    train_arm.export_excel()
    print(f"Evaluated {weights} -> appended {len(rows)} metric rows for arm '{arm_name}' to {train_arm.METRICS_CSV}")
    print(f"Updated {train_arm.METRICS_XLSX}")


if __name__ == "__main__":
    main()
