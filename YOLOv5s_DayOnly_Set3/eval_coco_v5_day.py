#!/usr/bin/env python3
"""Evaluate the trained model with pycocotools on test_day / test_night /
test_dawndusk. Reports all-class and per-class mAP50, mAP50-95, mAP75.

If the team already has a pycocotools scorer used for YOLO11, prefer that
one (swap the weights path) so the numbers are directly comparable.

Image lists are read from each eval yaml (its `val:` entry: a folder,
a .txt of image paths, or a list).

Usage (run from inside YOLOv5s_DayOnly_Set3/, after training):
  python eval_coco_v5_day.py
"""
import csv
from pathlib import Path

import yaml
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from ultralytics import YOLO

# ---- EDIT THESE: same eval yamls as the YOLO11 set3_combined arm ----
REPO_ROOT = Path(r"C:\Users\USER\OneDrive\เดสก์ท็อป\baby\bdd100k_split_handoff")
EVAL_SPLITS = {
    "val_day": REPO_ROOT / "dataset" / "bdd100k_test_day.yaml",
    "val_night": REPO_ROOT / "dataset" / "bdd100k_test_night.yaml",
    "val_dawndusk": REPO_ROOT / "dataset" / "bdd100k_test_dawndusk.yaml",
}
# ----------------------------------------------------------------------------

HERE = Path(__file__).resolve().parent
WEIGHTS = HERE / "set3_combined_v5_day" / "weights" / "best.pt"
OUT_CSV = HERE / "yolov5s_dayonly_set3_coco.csv"
IMGSZ, CONF, IOU = 640, 0.001, 0.7  # keep identical to the YOLO11 evaluation


def label_path(img: Path) -> Path:
    """images/<split>/x.jpg -> labels/<split>/x.txt (works on Windows and Linux)."""
    parts = list(img.parts)
    i = len(parts) - 1 - parts[::-1].index("images")
    parts[i] = "labels"
    return Path(*parts).with_suffix(".txt")


IMG_EXT = {".jpg", ".jpeg", ".png"}


def resolve_entry(entry, root: Path, base: Path) -> list:
    """Turn a yaml val entry (dir, txt, or list) into image paths."""
    if isinstance(entry, list):
        return [p for e in entry for p in resolve_entry(e, root, base)]
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
    imgs = resolve_entry(entry, root, yaml_path.parent)
    missing = [p for p in imgs if not p.exists()]
    if missing:
        raise SystemExit(f"FLAG: {len(missing)} images not found, e.g. {missing[0]} - "
                         f"check paths in {yaml_path}")
    return imgs


def evaluate(model, split: str, yaml_path: Path) -> list:
    imgs = images_from_yaml(yaml_path)
    print(f"{split}: {len(imgs)} images")

    cats = [{"id": k + 1, "name": v} for k, v in model.names.items()]
    images, anns, dets, ids = [], [], [], {}

    # Ground truth: YOLO txt -> COCO (absolute xywh)
    for i, p in enumerate(imgs, 1):
        w, h = Image.open(p).size
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

    # Predictions
    for r in model.predict([str(p) for p in imgs], imgsz=IMGSZ, conf=CONF, iou=IOU,
                           stream=True, verbose=False):
        iid = ids[Path(r.path).name]
        for (x1, y1, x2, y2), s, c in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(),
                                          r.boxes.cls.tolist()):
            dets.append({"image_id": iid, "category_id": int(c) + 1,
                         "bbox": [x1, y1, x2 - x1, y2 - y1], "score": s})

    gt = COCO()
    gt.dataset = {"images": images, "annotations": anns, "categories": cats}
    gt.createIndex()
    E = COCOeval(gt, gt.loadRes(dets), "bbox")
    E.evaluate()
    E.accumulate()
    E.summarize()

    prec = E.eval["precision"]  # [IoU, recall, class, area, maxDet]

    def ap(k, t=None):
        x = prec[:, :, k, 0, -1] if t is None else prec[t, :, k, 0, -1]
        x = x[x > -1]
        return float(x.mean()) if x.size else float("nan")

    rows = [{"split": split, "class": "all",
             "mAP50": E.stats[1], "mAP50-95": E.stats[0], "mAP75": E.stats[2]}]
    for k, cid in enumerate(E.params.catIds):
        rows.append({"split": split, "class": model.names[cid - 1],
                     "mAP50": ap(k, 0), "mAP50-95": ap(k), "mAP75": ap(k, 5)})
    return rows


def main():
    if not WEIGHTS.exists():
        raise SystemExit(f"FLAG: {WEIGHTS} not found - train first")
    model = YOLO(str(WEIGHTS))

    rows = []
    for split, f in EVAL_SPLITS.items():
        if not f.exists():
            raise SystemExit(f"FLAG: {f} not found - check EVAL_SPLITS")
        rows += evaluate(model, split, f)

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    for r in rows:
        print(f"{r['split']:<14} {r['class']:<6} mAP50={r['mAP50']:.4f} "
              f"mAP50-95={r['mAP50-95']:.4f} mAP75={r['mAP75']:.4f}")
    print(f"Saved {OUT_CSV}")


if __name__ == "__main__":
    main()
