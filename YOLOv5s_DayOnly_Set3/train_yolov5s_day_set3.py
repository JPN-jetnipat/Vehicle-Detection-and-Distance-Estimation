#!/usr/bin/env python3
"""Train yolov5su (Ultralytics YOLOv5s, anchor-free head) on DAYTIME-ONLY
BDD100K 5-class data with hyperparameter Set 3 (set3.yaml).

Same structure as YOLO11s_TrainByDefault/train_yolo11s_default.py.
Differences vs YOLO11 Set 3 run: model (yolov5su.pt) and data (day only).
All hyperparameters come from set3.yaml.

Usage (run from inside YOLOv5s_DayOnly_Set3/, AFTER prepare_train_yaml_day.py):
  python prepare_train_yaml_day.py
  python train_yolov5s_day_set3.py
"""
from pathlib import Path

from ultralytics import YOLO

HERE = Path(__file__).resolve().parent  # YOLOv5s_DayOnly_Set3/
DATA_YAML = HERE / "data_bdd100k_vehicle5_day.yaml"
CFG_YAML = HERE / "set3.yaml"


def main():
    for f in (DATA_YAML, CFG_YAML):
        if not f.exists():
            raise SystemExit(f"FLAG: {f} not found")

    model = YOLO("yolov5su.pt")  # COCO-pretrained, auto-downloads

    model.train(
        cfg=str(CFG_YAML),   # all hyperparameters from Set 3
        data=str(DATA_YAML),
        project=str(HERE),
        name="set3_combined_v5_day",
    )


if __name__ == "__main__":  # required on Windows (dataloader workers)
    main()
