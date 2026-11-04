#!/usr/bin/env python3
"""Evaluate the trained model with pycocotools on test_day / test_night /
test_dawndusk, plus test_all (all three pooled).
Reports images, instances, mAP50, mAP50-95, mAP75, AP small/medium/large,
all-class and per-class.

Image lists are read from each eval yaml (its `val:` entry: a folder,
a .txt of image paths, or a list).

Usage (run from inside YOLOv5s_DayOnly_Set3/, after training):
  python eval_coco_v5_day.py
"""
import contextlib
import csv
import io
from pathlib import Path

import yaml
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from ultralytics import YOLO

# ---- EDIT THESE: same eval yamls as the YOLO11 set3_combined arm ----
REPO_ROOT = Path(r"C:\Users\USER\Vehicle-Detection-and-Distance-Estimation")
EVAL_DIR = REPO_ROOT / "DANN" / "baseline"
EVAL_SPLITS = {
    "val_day": EVAL_DIR / "bdd100k_test_day.yaml",
    "val_night": EVAL_DIR / "bdd100k_test_night.yaml",
    "val_dawndusk": EVAL_DIR / "bdd100k_test_dawndusk.yaml",
}
# ----------------------------------------------------------------------------

HERE = Path(__file__).resolve().parent
WEIGHTS = HERE / "set3_combined_v5_day" / "weights" / "best.pt"
OUT_CSV = HERE / "yolov5s_dayonly_set3_coco.csv"
IMGSZ, CONF, IOU = 640, 0.001, 0.7  # keep identical to the YOLO11 evaluation
CHUNK = 32  # images per predict call (a list source is loaded fully into RAM)
IMG_EXT = {".jpg", ".jpeg", ".png"}


def label_path(img: Path) -> Path:
    """images/<split>/x.jpg -> labels/<split>/x.txt (works on Windows and Linux)."""
    parts = list(img.parts)
    i = len(parts) - 1 - parts[::-1].index("images")
    parts[i] = "labels"
    return Path(*parts).with_suffix(".txt")


def resolve_entry(entry, root: Path) -> list:
    """Turn a yaml val entry (dir, txt, or list) into image paths."""
    if isinstance(entry, list):
        return [p for e in entry for p in resolve_entry(e, root)]
    p = Path(entry)
    if not p.is_absolute():
        p = root / p
    if p.suffix == ".txt":
        out = []
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            q = Path(line)
            if not q.is_absolute():
                q = p.parent / q if line.startswith(".") else root / q
            out.append(q)
        return out
    return sorted(q for q in p.rglob("*") if q.suffix.lower() in IMG_EXT)


def images_from_yaml(yaml_path: Path) -> list:
    cfg = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    root = Path(cfg.get("path", ".") or ".")
    if not root.is_absolute():
        root = (yaml_path.parent / root).resolve()
    entry = cfg.get("val") or cfg.get("test")
    if entry is None:
        raise SystemExit(f"FLAG: no val/test entry in {yaml_path}")
    imgs = resolve_entry(entry, root)
    missing = [p for p in imgs if not p.exists()]
    if missing:
        raise SystemExit(f"FLAG: {len(missing)} images not found, e.g. {missing[0]} - "
                         f"check paths in {yaml_path}")
    return imgs


def collect(model, split: str, yaml_path: Path):
    """Return COCO images, GT annotations, detections for one split."""
    imgs = images_from_yaml(yaml_path)
    print(f"{split}: {len(imgs)} images")
    images, anns, dets, ids = [], [], [], {}

    # Ground truth: YOLO txt -> COCO (absolute xywh)
    for i, p in enumerate(imgs, 1):
        with Image.open(p) as im:
            w, h = im.size
        ids[p.name] = i
        images.append({"id": i, "file_name": p.name, "width": w, "height": h})
        lp = label_path(p)
        if not lp.exists():
            continue
        for line in lp.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            c, cx, cy, bw, bh = map(float, line.split()[:5])
            bw, bh = bw * w, bh * h
            anns.append({"id": len(anns) + 1, "image_id": i, "category_id": int(c) + 1,
                         "bbox": [cx * w - bw / 2, cy * h - bh / 2, bw, bh],
                         "area": bw * bh, "iscrowd": 0})
    print(f"  GT boxes: {len(anns)}")

    # Predictions, in chunks to keep RAM low
    for start in range(0, len(imgs), CHUNK):
        chunk = imgs[start:start + CHUNK]
        results = model.predict([str(p) for p in chunk], imgsz=IMGSZ, conf=CONF, iou=IOU,
                                verbose=False)
        for p, r in zip(chunk, results):
            iid = ids[p.name]
            for (x1, y1, x2, y2), sc, c in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(),
                                               r.boxes.cls.tolist()):
                dets.append({"image_id": iid, "category_id": int(c) + 1,
                             "bbox": [x1, y1, x2 - x1, y2 - y1], "score": sc})
        done = min(start + CHUNK, len(imgs))
        if done % 640 == 0 or done == len(imgs):
            print(f"  predicted {done}/{len(imgs)}")
    return images, anns, dets


def merge(parts):
    """Merge several (images, anns, dets) with re-numbered ids."""
    images, anns, dets = [], [], []
    for im_list, an_list, dt_list in parts:
        off = len(images)
        for im in im_list:
            images.append({**im, "id": im["id"] + off})
        for an in an_list:
            anns.append({**an, "id": len(anns) + 1, "image_id": an["image_id"] + off})
        for dt in dt_list:
            dets.append({**dt, "image_id": dt["image_id"] + off})
    return images, anns, dets


def coco_eval(names, split, images, anns, dets) -> list:
    cats = [{"id": k + 1, "name": v} for k, v in names.items()]
    gt = COCO()
    gt.dataset = {"images": images, "annotations": anns, "categories": cats}
    with contextlib.redirect_stdout(io.StringIO()):  # silence pycocotools logs
        gt.createIndex()
        E = COCOeval(gt, gt.loadRes(dets), "bbox")
        E.evaluate()
        E.accumulate()
        E.summarize()

    prec = E.eval["precision"]  # [IoU, recall, class, area, maxDet]

    def ap(k, t=None, a=0):
        x = prec[:, :, k, a, -1] if t is None else prec[t, :, k, a, -1]
        x = x[x > -1]
        return float(x.mean()) if x.size else float("nan")

    rows = [{"split": split, "class": "all",
             "images": len(images), "instances": len(anns),
             "mAP50": E.stats[1], "mAP50-95": E.stats[0], "mAP75": E.stats[2],
             "AP_small": E.stats[3], "AP_medium": E.stats[4], "AP_large": E.stats[5]}]
    for k, cid in enumerate(E.params.catIds):
        cls_anns = [a for a in anns if a["category_id"] == cid]
        rows.append({"split": split, "class": names[cid - 1],
                     "images": len({a["image_id"] for a in cls_anns}),
                     "instances": len(cls_anns),
                     "mAP50": ap(k, 0), "mAP50-95": ap(k), "mAP75": ap(k, 5),
                     "AP_small": ap(k, a=1), "AP_medium": ap(k, a=2), "AP_large": ap(k, a=3)})
    return rows


def main():
    if not WEIGHTS.exists():
        raise SystemExit(f"FLAG: {WEIGHTS} not found - train first")
    model = YOLO(str(WEIGHTS))

    rows, parts = [], []
    for split, f in EVAL_SPLITS.items():
        if not f.exists():
            raise SystemExit(f"FLAG: {f} not found - check EVAL_SPLITS")
        data = collect(model, split, f)
        parts.append(data)
        rows += coco_eval(model.names, split, *data)

    # test_all = day + night + dawndusk pooled (no re-prediction)
    rows += coco_eval(model.names, "test_all", *merge(parts))

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print("\n=== Summary ===")
    print(f"{'split':<14}{'class':<7}{'images':>7}{'inst':>8}"
          f"{'mAP50':>9}{'mAP50-95':>10}{'mAP75':>9}{'APs':>8}")
    for r in rows:
        print(f"{r['split']:<14}{r['class']:<7}{r['images']:>7}{r['instances']:>8}"
              f"{r['mAP50']:>9.4f}{r['mAP50-95']:>10.4f}{r['mAP75']:>9.4f}{r['AP_small']:>8.4f}")
    print(f"Saved {OUT_CSV}")


if __name__ == "__main__":
    main()