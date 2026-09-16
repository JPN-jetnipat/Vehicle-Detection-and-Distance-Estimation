"""Evaluate an already-trained arm's checkpoint on its eval_splits, without retraining.

Usage:
    python scripts/eval_arm.py --config configs/experiments/set1_night_aug.yaml
    python scripts/eval_arm.py --config configs/experiments/set1_night_aug.yaml --weights runs/detect/set1_night_aug/weights/best.pt

Reuses train_arm.py's metrics_to_rows/append_metrics/export_excel, so results
land in the same results/metrics_all_arms.csv|xlsx under the config's
arm_name, alongside rows from actual training runs.

Scores with both scorers, exactly like train_arm.py does, so a checkpoint
re-evaluated here is directly comparable to one scored during training:
ultralytics' own AP implementation, then pycocotools (the standard COCO
protocol, and the only scorer that reports mAP75 - ultralytics' val() exposes
only mAP50 and mAP50-95). Rows are tagged with their scorer, so the two never
get conflated.

Must be run with the repo root as the working directory (same as train_arm.py).
"""

import argparse
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO

import eval_arm_coco
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
    timestamp = train_arm.now_th()
    rows: list[dict] = []
    for split in cfg["eval_splits"]:
        split_metrics = model.val(
            data=split["data"],
            split="val",
            imgsz=train_kwargs["imgsz"],
            batch=train_kwargs["batch"],
        )
        rows.extend(train_arm.metrics_to_rows(arm_name, split["name"], split_metrics, timestamp))

    coco_device = "0" if torch.cuda.is_available() else "cpu"
    for split in cfg["eval_splits"]:
        print(f"\n[{arm_name}] scoring {split['name']} ({split['data']}) with pycocotools ...")
        coco_result, n_images = eval_arm_coco.score_split(
            model, split["data"], train_kwargs["imgsz"], coco_device, train_kwargs["batch"]
        )
        eval_arm_coco.print_split_table(n_images, coco_result)
        rows.extend(eval_arm_coco.coco_result_to_rows(arm_name, split["name"], coco_result, timestamp))

    train_arm.append_metrics(rows)
    train_arm.export_excel()
    print(f"Evaluated {weights} -> appended {len(rows)} metric rows for arm '{arm_name}' to {train_arm.METRICS_CSV}")
    print(f"Updated {train_arm.METRICS_XLSX}")


if __name__ == "__main__":
    main()
