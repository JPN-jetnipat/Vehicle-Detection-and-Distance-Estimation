#!/usr/bin/env python3
r"""Evaluate the trained yolo11s (best.pt) using pycocotools, with settings
matched to the team's scorer (tools/eval_arm_coco.py) so numbers are
directly comparable, not just "also pycocotools":

  - conf=0.001, iou=0.6, max_det=300 (fixed for every arm on their side -
    this matters more than using pycocotools at all: different NMS
    thresholds change which detections even reach scoring)
  - image size fixed at 1280x720 (BDD100K's actual, constant image size -
    matches their hardcoded IMG_WIDTH/IMG_HEIGHT instead of reading each
    image's header)
  - reports mAP50, mAP50-95, mAP75 (their three metrics - dropped AR100
    from the earlier version of this script, since they don't report it)

Still self-contained: lives in YOLO11s_TestTrained/, reuses splits_local/*.txt
from prepare_test_lists.py. Does NOT depend on their train_arm.py or
config yaml - this is our own standalone counterpart, matched on the
scoring settings that actually affect the numbers.

Usage:
  python eval_pycocotools.py
  python eval_pycocotools.py --weights "..\run1-2\weights\best.pt"
"""
import argparse
from pathlib import Path

import pandas as pd
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from ultralytics import YOLO

HERE = Path(__file__).resolve().parent
SPLITS_OUT = HERE / "splits_local"
OUT_CSV = HERE / "yolo11s_trained_eval_pycocotools.csv"

CLASSES = ["car", "truck", "bus", "motor", "bike"]  # id 0..4, matches labels as-is

# fixed to match the team's eval_arm_coco.py - do not make these per-run
# configurable, that would defeat the point of matching their numbers
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
    ("rainy", "eval_test_rainy"),  # requires add_weather_rainy.py to have been run first
]


def label_path_for(img_path: Path) -> Path:
    # dataset/yolo/images/<split>/<name> -> dataset/yolo/labels/<split>/<name>.txt
    parts = list(img_path.parts)
    parts[parts.index("images")] = "labels"
    return Path(*parts).with_suffix(".txt")


def yolo_line_to_xywh(parts: list[str], w: int, h: int) -> list[float]:
    cx, cy, bw, bh = (float(v) for v in parts[1:5])
    return [(cx - bw / 2) * w, (cy - bh / 2) * h, bw * w, bh * h]


def build_gt(image_paths: list[Path]) -> tuple[dict, dict]:
    """Fixed 1280x720 (BDD100K's real, constant size) - no per-image PIL read."""
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
                cls = int(parts[0])
                box = yolo_line_to_xywh(parts, IMG_WIDTH, IMG_HEIGHT)
                annotations.append({
                    "id": ann_id, "image_id": img_id, "category_id": cls,
                    "bbox": box, "area": box[2] * box[3], "iscrowd": 0,
                })
                ann_id += 1

    categories = [{"id": i, "name": name} for i, name in enumerate(CLASSES)]
    gt = {"images": images, "annotations": annotations, "categories": categories}
    return gt, stem_to_id


BATCH = 8  # small chunk size for an 8GB GPU - matches the team's manual-chunking
           # workaround (see their run_inference_to_dir comment): passing the
           # full path list to predict() does NOT actually cap memory to the
           # batch size, it was OOM'ing even on their 46GB A40.


def build_predictions(model: YOLO, list_path: Path, stem_to_id: dict) -> list[dict]:
    # Same encoding workaround as eval_pycocotools_baseline.py - read the
    # list ourselves (UTF-8) and pass it as a Python list, bypassing
    # Ultralytics' own txt-file loader (which crashes on Thai paths).
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
                    "image_id": img_id, "category_id": int(cls),
                    "bbox": [x1, y1, x2 - x1, y2 - y1], "score": conf,
                })
    return dets


def eval_one(model: YOLO, list_stem: str) -> dict:
    list_path = SPLITS_OUT / f"{list_stem}_images.txt"
    if not list_path.exists():
        raise SystemExit(f"missing {list_path} - run prepare_test_lists.py first")

    image_paths = [Path(l.strip()) for l in list_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"\n{'=' * 60}\n{list_stem}  ({len(image_paths)} images)\n{'=' * 60}", flush=True)

    gt_dict, stem_to_id = build_gt(image_paths)
    coco_gt = COCO()
    coco_gt.dataset = gt_dict
    coco_gt.createIndex()

    n_images = len(gt_dict["images"])
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

    out = {cls_name: stats_for(cat_id) for cat_id, cls_name in enumerate(CLASSES)}
    out["all"] = stats_for(None)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(HERE.parent / "run1" / "weights" / "best.pt"))
    args = ap.parse_args()

    weights_path = Path(args.weights)
    if not weights_path.exists():
        raise SystemExit(f"FLAG: {weights_path} not found - pass --weights to point at your trained best.pt")

    model = YOLO(str(weights_path))

    per_run = {}
    for col_label, list_stem in RUNS:
        per_run[col_label] = eval_one(model, list_stem)

    rows = []
    for cls_name in CLASSES + ["all"]:
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