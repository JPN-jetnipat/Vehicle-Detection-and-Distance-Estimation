#!/usr/bin/env python3
r"""Evaluate a YOLOv5 checkpoint (original yolov5/ repo format) with pycocotools,
using EXACTLY the same settings as
YOLO11s_TrainByDefault/YOLO11s_TestTrained/eval_pycocotools.py:

  - conf=0.001, iou=0.6, max_det=300, imgsz=640 (rect letterbox, stride 32)
  - GT image size fixed at 1280x720
  - same image lists (reads the YOLO11s splits_local/ folder by default)
  - same CSV layout: class x metric x split, mAP50 / mAP50-95 / mAP75

Why not reuse the YOLO11s script directly: ultralytics.YOLO() cannot load
checkpoints produced by the original yolov5 train.py, so inference here goes
through yolov5's DetectMultiBackend + non_max_suppression instead.

Usage (run from inside YOLOv5s_TestTrained/):
  python eval_pycocotools_v5.py                                  # trained run1/best.pt
  python eval_pycocotools_v5.py --weights ..\run1\weights\last.pt
  python eval_pycocotools_v5.py --weights yolov5s.pt --coco-remap   # COCO zero-shot reference
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
YOLOV5 = REPO / "yolov5"
sys.path.insert(0, str(YOLOV5))
sys.path.insert(0, str(YOLOV5.parent))  # yolov5/val.py in this repo imports geometry/

from models.common import DetectMultiBackend          # noqa: E402
from utils.augmentations import letterbox             # noqa: E402
from utils.general import non_max_suppression, scale_boxes  # noqa: E402
from utils.torch_utils import select_device           # noqa: E402

DEFAULT_SPLITS = REPO / "YOLO11s_TrainByDefault" / "YOLO11s_TestTrained" / "splits_local"
CLASSES = ["car", "truck", "bus", "motor", "bike"]
COCO_TO_OURS = {2: 0, 7: 1, 5: 2, 3: 3, 1: 4}  # car, truck, bus, motorcycle, bicycle

# fixed to match the YOLO11s eval - do not change per run
CONF_THRES = 0.001
IOU_THRES = 0.6
MAX_DET = 300
IMGSZ = 640
IMG_WIDTH, IMG_HEIGHT = 1280, 720
BATCH = 8

RUNS = [
    ("val", "eval_val"),
    ("test_all", "eval_test_all"),
    ("test_day", "eval_test_day"),
    ("test_night", "eval_test_night"),
    ("test_dawndusk", "eval_test_dawndusk"),
    ("rainy", "eval_test_rainy"),
]


def label_path_for(img_path: Path) -> Path:
    parts = list(img_path.parts)
    parts[len(parts) - 1 - parts[::-1].index("images")] = "labels"
    return Path(*parts).with_suffix(".txt")


def build_gt(image_paths):
    images, annotations, stem_to_id, ann_id = [], [], {}, 1
    for img_id, p in enumerate(image_paths, 1):
        images.append({"id": img_id, "file_name": p.name, "width": IMG_WIDTH, "height": IMG_HEIGHT})
        stem_to_id[p.stem] = img_id
        lbl = label_path_for(p)
        if not lbl.exists():
            continue
        for line in lbl.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if not parts:
                continue
            cx, cy, bw, bh = (float(v) for v in parts[1:5])
            box = [(cx - bw / 2) * IMG_WIDTH, (cy - bh / 2) * IMG_HEIGHT, bw * IMG_WIDTH, bh * IMG_HEIGHT]
            annotations.append({"id": ann_id, "image_id": img_id, "category_id": int(parts[0]),
                                "bbox": box, "area": box[2] * box[3], "iscrowd": 0})
            ann_id += 1
    cats = [{"id": i, "name": n} for i, n in enumerate(CLASSES)]
    return {"images": images, "annotations": annotations, "categories": cats}, stem_to_id


def read_image(path: Path):
    # np.fromfile + imdecode: works with Thai characters in Windows paths
    return cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)


@torch.no_grad()
def predict(model, device, image_paths, stem_to_id, coco_remap):
    dets = []
    for start in range(0, len(image_paths), BATCH):
        chunk = image_paths[start:start + BATCH]
        im0s = [read_image(p) for p in chunk]
        # rect letterbox (auto=True) = same as ultralytics predict for same-shape batches
        ims = [letterbox(im0, IMGSZ, stride=model.stride, auto=True)[0] for im0 in im0s]
        x = np.stack([im[:, :, ::-1].transpose(2, 0, 1) for im in ims])  # BGR->RGB, HWC->CHW
        x = torch.from_numpy(np.ascontiguousarray(x)).to(device).float() / 255.0
        pred = model(x)
        pred = non_max_suppression(pred, CONF_THRES, IOU_THRES, max_det=MAX_DET)
        for p, im0, det in zip(chunk, im0s, pred):
            img_id = stem_to_id.get(p.stem)
            if img_id is None or not len(det):
                continue
            det[:, :4] = scale_boxes(x.shape[2:], det[:, :4], im0.shape).round()
            for *xyxy, conf, cls in det.tolist():
                cls = int(cls)
                if coco_remap:
                    if cls not in COCO_TO_OURS:
                        continue
                    cls = COCO_TO_OURS[cls]
                x1, y1, x2, y2 = xyxy
                dets.append({"image_id": img_id, "category_id": cls,
                             "bbox": [x1, y1, x2 - x1, y2 - y1], "score": conf})
        if start % (BATCH * 100) == 0:
            print(f"  {start}/{len(image_paths)}", flush=True)
    return dets


def eval_one(model, device, list_path: Path, coco_remap):
    image_paths = [Path(l.strip()) for l in list_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"\n{'=' * 60}\n{list_path.name}  ({len(image_paths)} images)\n{'=' * 60}", flush=True)

    gt, stem_to_id = build_gt(image_paths)
    coco_gt = COCO()
    coco_gt.dataset = gt
    coco_gt.createIndex()
    n_inst = {}
    for a in gt["annotations"]:
        n_inst[a["category_id"]] = n_inst.get(a["category_id"], 0) + 1

    dets = predict(model, device, image_paths, stem_to_id, coco_remap)
    if not dets:
        raise SystemExit(f"FLAG: no predictions for {list_path.name}")
    coco_dt = coco_gt.loadRes(dets)

    def stats_for(cat_id):
        ev = COCOeval(coco_gt, coco_dt, iouType="bbox")
        if cat_id is not None:
            ev.params.catIds = [cat_id]
        ev.evaluate(); ev.accumulate(); ev.summarize()
        n = n_inst.get(cat_id, 0) if cat_id is not None else sum(n_inst.values())
        return {"Images": len(image_paths), "Instances": n,
                "mAP50": ev.stats[1], "mAP50-95": ev.stats[0], "mAP75": ev.stats[2]}

    out = {name: stats_for(i) for i, name in enumerate(CLASSES)}
    out["all"] = stats_for(None)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(HERE.parent / "run1" / "weights" / "best.pt"))
    ap.add_argument("--splits-dir", default=str(DEFAULT_SPLITS))
    ap.add_argument("--coco-remap", action="store_true",
                    help="weights are COCO-80 (e.g. yolov5s.pt): map 5 vehicle ids, drop the rest")
    ap.add_argument("--device", default="")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    splits_dir = Path(args.splits_dir)
    device = select_device(args.device)
    model = DetectMultiBackend(args.weights, device=device)
    model.eval()
    print(f"weights: {args.weights}\nclasses in model: {len(model.names)}  coco_remap={args.coco_remap}")
    if not args.coco_remap and len(model.names) != len(CLASSES):
        raise SystemExit("FLAG: model is not 5-class - add --coco-remap for COCO weights")

    runs = [(c, s) for c, s in RUNS if (splits_dir / f"{s}_images.txt").exists()]
    skipped = [c for c, s in RUNS if (c, s) not in runs]
    if skipped:
        print(f"WARNING: list files missing, skipped: {skipped}")
    if not runs:
        raise SystemExit(f"FLAG: no *_images.txt in {splits_dir} - run prepare_test_lists.py first")

    per_run = {c: eval_one(model, device, splits_dir / f"{s}_images.txt", args.coco_remap) for c, s in runs}

    rows = []
    for cls in CLASSES + ["all"]:
        for m in ("Images", "Instances", "mAP50", "mAP50-95", "mAP75"):
            row = {"class": cls, "metric": m}
            for c, _ in runs:
                v = per_run[c][cls][m]
                row[c] = int(v) if m in ("Images", "Instances") else round(float(v), 4)
            rows.append(row)
    df = pd.DataFrame(rows)

    tag = "coco_reference" if args.coco_remap else "trained"
    out_csv = Path(args.out) if args.out else HERE / f"yolov5s_{tag}_eval_pycocotools.csv"
    df.to_csv(out_csv, index=False)
    pd.set_option("display.width", 140)
    print(f"\n{'=' * 60}\nsummary (pycocotools, same settings as YOLO11s eval)\n{'=' * 60}")
    print(df.to_string(index=False))
    print(f"\nSaved: {out_csv}")


if __name__ == "__main__":
    main()
