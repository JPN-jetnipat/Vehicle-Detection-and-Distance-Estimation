#!/usr/bin/env python3
r"""Experiment 1: score the PLAIN (untrained, COCO-pretrained) yolo11s.pt
against our dataset using pycocotools, with settings matched to the team's
scorer (conf=0.001, iou=0.6, max_det=300, fixed 1280x720 image size,
mAP50/mAP50-95/mAP75) - same matching used for the post-training eval, so
the two experiments are directly comparable to each other AND to the team.

Reuses the COCO-id-remapped eval set already built by build_coco_eval_set.py
(dataset_yolo_coco/, splits_eval/*.txt) - no need to rebuild anything, that
script already remapped our 5-class labels to real COCO ids (car=2,
truck=7, bus=5, motor=3, bike=1) so a COCO-pretrained model's raw output
can be scored against them directly.

Self-contained: lives in YOLO11s_Evaluation/. Requires
build_coco_eval_set.py to have been run first.

Usage:
  python eval_pycocotools_baseline.py
  python eval_pycocotools_baseline.py --weights yolo11s.pt
"""
import argparse
from pathlib import Path

import pandas as pd
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from ultralytics import YOLO

HERE = Path(__file__).resolve().parent
SPLITS_OUT = HERE / "splits_eval"
OUT_CSV = HERE / "yolo11s_baseline_eval_pycocotools.csv"

# our class name -> real COCO class id (must match build_coco_eval_set.py's OUR_TO_COCO)
CLASS_TO_COCO = {"car": 2, "truck": 7, "bus": 5, "motor": 3, "bike": 1}

# fixed to match the team's eval_arm_coco.py
CONF_THRES = 0.001
IOU_THRES = 0.6
MAX_DET = 300
IMG_WIDTH = 1280
IMG_HEIGHT = 720

RUNS = [
    ("val", "eval_val"),
    ("test_all", "eval_test_all"),
    ("test_day", "eval_test_day"),
    ("test_night", "eval_test_night"),
    ("test_dawndusk", "eval_test_dawndusk"),
    ("rainy", "eval_test_rainy"),  # requires add_weather_rainy_coco.py to have been run first
]


def label_path_for(img_path: Path) -> Path:
    # dataset_yolo_coco/images/<split>/<name> -> dataset_yolo_coco/labels/<split>/<name>.txt
    parts = list(img_path.parts)
    parts[parts.index("images")] = "labels"
    return Path(*parts).with_suffix(".txt")


def build_gt(image_paths: list[Path]) -> tuple[dict, dict]:
    images, annotations = [], []
    stem_to_id = {}
    ann_id = 1
    for img_id, img_path in enumerate(image_paths, 1):
        images.append({"id": img_id, "file_name": img_path.name, "width": IMG_WIDTH, "height": IMG_HEIGHT})
        stem_to_id[img_path.stem] = img_id

        lbl_path = label_path_for(img_path)
        if lbl_path.exists():
            for line in lbl_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                cls = int(parts[0])  # already a real COCO id (remapped by build_coco_eval_set.py)
                cx, cy, bw, bh = (float(v) for v in parts[1:5])
                x = (cx - bw / 2) * IMG_WIDTH
                y = (cy - bh / 2) * IMG_HEIGHT
                w_px, h_px = bw * IMG_WIDTH, bh * IMG_HEIGHT
                annotations.append({
                    "id": ann_id, "image_id": img_id, "category_id": cls,
                    "bbox": [x, y, w_px, h_px], "area": w_px * h_px, "iscrowd": 0,
                })
                ann_id += 1

    # only our 5 real COCO categories - keeps "all" restricted to these,
    # ignoring the other 75 COCO classes the pretrained model might also predict
    categories = [{"id": coco_id, "name": name} for name, coco_id in CLASS_TO_COCO.items()]
    gt = {"images": images, "annotations": annotations, "categories": categories}
    return gt, stem_to_id


BATCH = 8  # small chunk size for an 8GB GPU - matches the team's manual-chunking
           # workaround: passing the full path list to predict() does NOT
           # actually cap memory to the batch size, it was OOM'ing even on
           # their 46GB A40.


def build_predictions(model: YOLO, list_path: Path, stem_to_id: dict) -> list[dict]:
    # Read the list ourselves (UTF-8) and pass paths as a Python list, not a
    # txt-file path - Ultralytics' own file loader reads txt sources with
    # Path.read_text() (no encoding=), which defaults to the Windows locale
    # codepage (cp874 for Thai Windows) and crashes on our UTF-8 paths.
    # Passing an in-memory list bypasses that code path entirely.
    image_paths = [l.strip() for l in list_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    dets = []
    for start in range(0, len(image_paths), BATCH):
        chunk = image_paths[start:start + BATCH]
        results = model.predict(
            source=chunk,
            conf=CONF_THRES,
            iou=IOU_THRES,
            max_det=MAX_DET,
            batch=len(chunk),
            stream=False,
            verbose=False,
            save=False,
        )
        for r in results:
            stem = Path(r.path).stem
            img_id = stem_to_id.get(stem)
            if img_id is None:
                continue
            boxes = r.boxes
            for xyxy, conf, cls in zip(boxes.xyxy.tolist(), boxes.conf.tolist(), boxes.cls.tolist()):
                x1, y1, x2, y2 = xyxy
                dets.append({
                    "image_id": img_id, "category_id": int(cls),  # raw COCO id, 0-79
                    "bbox": [x1, y1, x2 - x1, y2 - y1], "score": conf,
                })
    return dets


def eval_one(model: YOLO, list_stem: str) -> dict:
    list_path = SPLITS_OUT / f"{list_stem}_images.txt"
    if not list_path.exists():
        raise SystemExit(f"missing {list_path} - run build_coco_eval_set.py first")

    image_paths = [Path(l.strip()) for l in list_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"\n{'=' * 60}\n{list_stem}  ({len(image_paths)} images)\n{'=' * 60}", flush=True)

    gt_dict, stem_to_id = build_gt(image_paths)
    coco_gt = COCO()
    coco_gt.dataset = gt_dict
    coco_gt.createIndex()

    n_images = len(gt_dict["images"])
    # ground-truth instance count per class - same regardless of predictions,
    # matches the "Instances" column in the reference screenshot
    n_instances_by_cat = {}
    for ann in gt_dict["annotations"]:
        n_instances_by_cat[ann["category_id"]] = n_instances_by_cat.get(ann["category_id"], 0) + 1

    dets = build_predictions(model, list_path, stem_to_id)
    if not dets:
        raise SystemExit(f"FLAG: no predictions produced for {list_stem} - check weights/model output")
    coco_dt = coco_gt.loadRes(dets)

    def stats_for(cat_id) -> dict:
        ev = COCOeval(coco_gt, coco_dt, iouType="bbox")
        if cat_id is not None:
            ev.params.catIds = [cat_id]
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
        n_inst = n_instances_by_cat.get(cat_id, sum(n_instances_by_cat.values())) if cat_id is not None \
            else sum(n_instances_by_cat.values())
        return {"Images": n_images, "Instances": n_inst,
                "mAP50": ev.stats[1], "mAP50-95": ev.stats[0], "mAP75": ev.stats[2]}

    out = {cls_name: stats_for(coco_id) for cls_name, coco_id in CLASS_TO_COCO.items()}
    out["all"] = stats_for(None)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="yolo11s.pt",
                     help="pretrained COCO weights - ultralytics auto-downloads if not found locally")
    args = ap.parse_args()

    model = YOLO(args.weights)

    per_run = {}
    for col_label, list_stem in RUNS:
        per_run[col_label] = eval_one(model, list_stem)

    rows = []
    for cls_name in list(CLASS_TO_COCO.keys()) + ["all"]:
        for metric in ("Images", "Instances", "mAP50", "mAP50-95", "mAP75"):
            row = {"class": cls_name, "metric": metric}
            for col_label, _ in RUNS:
                row[col_label] = per_run[col_label][cls_name][metric]
            rows.append(row)

    df = pd.DataFrame(rows)
    # Images/Instances are counts (int) - only round the mAP rows to keep the CSV readable
    is_count_row = df["metric"].isin(["Images", "Instances"])
    for col in df.columns[2:]:
        df[col] = df[col].astype(object)  # keep mixed int/float per-cell, avoid column-wide upcast to float
        df.loc[is_count_row, col] = df.loc[is_count_row, col].astype(int)
        df.loc[~is_count_row, col] = df.loc[~is_count_row, col].astype(float).round(4)
    df.to_csv(OUT_CSV, index=False)
    pd.set_option("display.width", 140)
    print(f"\n{'=' * 60}\ncombined summary (pycocotools, team-matched settings)\n{'=' * 60}")
    print(df.to_string(index=False))
    print(f"\nSaved: {OUT_CSV}")


if __name__ == "__main__":
    main()