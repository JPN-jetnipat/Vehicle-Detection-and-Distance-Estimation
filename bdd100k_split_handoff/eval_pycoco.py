"""
eval_pycoco.py

pycocotools evaluation of 3 models on the existing TEST set, per weather and
per timeofday. Scoring settings are matched to the team's scorer
(tools/eval_arm_coco.py, eval_pycocotools.py) so numbers are comparable:

    conf=0.001, iou=0.6, max_det=300      (fixed for every model)
    image size fixed at 1280x720          (BDD100K constant size, no header read)
    metrics: mAP50, mAP50-95, mAP75       (+ AP small/medium/large as extra columns)
    pycocotools==2.0.11

Model A (COCO-pretrained) is NOT class-filtered at inference: all 80 classes go
through NMS and max_det, then predictions are mapped back to the 5 project
classes and the rest are dropped (same as eval_pycocotools_baseline.py).

Each model runs inference once on the whole test set (cached in ./pycoco_cache);
each group is scored by restricting COCOeval to that group's image ids.

Run:  python split_by_weather.py     (once, makes weather_splits/raw_all.txt)
      python eval_pycoco.py
"""

import contextlib
import csv
import io
import json
import os
from pathlib import Path

import pandas as pd
import yaml
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# ---------------------------- CONFIG ----------------------------
ROOT = Path(__file__).resolve().parent
ALL_LIST = ROOT / "weather_splits" / "raw_all.txt"
ATTR_CSV = ROOT / "dataset" / "yolo" / "attr_index.csv"
DATA_YAML = ROOT / "configs" / "data" / "bdd100k_vehicle5.yaml"
CACHE_DIR = ROOT / "pycoco_cache"
OUT_CSV = ROOT / "pycoco_results.csv"

# kind: "coco" = COCO-pretrained 80-class model, "raw" = 5-class model on raw images,
#       "enh"  = 5-class model on enhanced images (needs ENH_REPLACE)
MODELS = {
    "A_coco_pretrained": {"weights": "yolo11s.pt", "kind": "coco"},
    "B_finetuned_100ep": {"weights": r"C:\Users\USER\OneDrive\เดสก์ท็อป\baby\YOLO11s_TrainByDefault\run1\weights\best.pt", "kind": "raw"},
    "C_set1_lowlight":   {"weights": r"C:\Users\USER\OneDrive\เดสก์ท็อป\baby\Augment_method1\runs\detect\method1_lowlight\weights\best.pt", "kind": "raw"},
}
ENH_REPLACE = None   # (text in raw image path, replacement) - only for kind "enh"

# Fixed to match the team's scorer. Do not change per run.
CONF_THRES = 0.001
IOU_THRES = 0.6
MAX_DET = 300
IMG_WIDTH = 1280
IMG_HEIGHT = 720

IMGSZ = 640
DEVICE = 0            # GPU index, or "cpu"
BATCH = 8             # manual chunk size (passing the full list does not cap memory)
PER_CLASS = False     # True adds per-class mAP50-95 columns (slower)

# COCO class id -> project class id (car0 truck1 bus2 motor3 bike4)
COCO_TO_PROJECT = {2: 0, 7: 1, 5: 2, 3: 3, 1: 4}
# ----------------------------------------------------------------

SMALL, MEDIUM = 32 ** 2, 96 ** 2


def read_list(txt):
    return [l.strip() for l in open(txt, encoding="utf-8") if l.strip()]


def img_to_label(img_path):
    s = os.path.normpath(str(img_path))
    sep = os.sep
    i = s.rfind(sep + "images" + sep)
    lab = s[:i] + sep + "labels" + sep + s[i + len(sep + "images" + sep):]
    return os.path.splitext(lab)[0] + ".txt"


def load_names():
    names = yaml.safe_load(open(DATA_YAML, encoding="utf-8"))["names"]
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]
    return names


def load_attrs():
    attrs = {}
    with open(ATTR_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            attrs[row["name"]] = (row["timeofday"], row["weather"])
    return attrs


def build_gt(images, names):
    """YOLO labels -> COCO ground truth at fixed 1280x720. Images with no boxes are kept."""
    cache = CACHE_DIR / "gt_1280x720.json"
    files = [os.path.basename(p) for p in images]
    if cache.exists():
        d = json.load(open(cache, encoding="utf-8"))
        if [im["file_name"] for im in d["images"]] == files:
            print("[cache] ground truth loaded")
            return d

    print("Building COCO ground truth from YOLO labels...")
    imgs, anns, aid = [], [], 1
    for i, p in enumerate(images, 1):
        imgs.append({"id": i, "file_name": os.path.basename(p), "width": IMG_WIDTH, "height": IMG_HEIGHT})
        lab = img_to_label(p)
        if not os.path.exists(lab):
            continue
        for ln in open(lab, encoding="utf-8"):
            parts = ln.split()
            if len(parts) < 5:
                continue
            c = int(parts[0])
            cx, cy, w, h = (float(v) for v in parts[1:5])
            bw, bh = w * IMG_WIDTH, h * IMG_HEIGHT
            anns.append({"id": aid, "image_id": i, "category_id": c,
                         "bbox": [(cx - w / 2) * IMG_WIDTH, (cy - h / 2) * IMG_HEIGHT, bw, bh],
                         "area": bw * bh, "iscrowd": 0})
            aid += 1
    d = {"images": imgs, "annotations": anns,
         "categories": [{"id": i, "name": n} for i, n in enumerate(names)]}
    CACHE_DIR.mkdir(exist_ok=True)
    json.dump(d, open(cache, "w", encoding="utf-8"))
    print(f"  images={len(imgs)}  boxes={len(anns)}")
    return d


def predict(mname, cfg, images, stem_to_id):
    """One inference pass per model, in chunks; detections cached as COCO json."""
    cache = CACHE_DIR / f"pred_{mname}_iou{IOU_THRES}_max{MAX_DET}.json"
    if cache.exists():
        print(f"[cache] predictions loaded for {mname}")
        return json.load(open(cache, encoding="utf-8"))

    from ultralytics import YOLO

    src = list(images)
    if cfg["kind"] == "enh":
        old, new = ENH_REPLACE
        src = [p.replace(old, new) for p in src]

    model = YOLO(cfg["weights"])
    dets = []
    for start in range(0, len(src), BATCH):
        chunk = src[start:start + BATCH]
        results = model.predict(source=chunk, conf=CONF_THRES, iou=IOU_THRES, max_det=MAX_DET,
                                imgsz=IMGSZ, device=DEVICE, batch=len(chunk),
                                stream=False, verbose=False, save=False)
        for r in results:
            img_id = stem_to_id.get(Path(r.path).stem)
            if img_id is None or r.boxes is None:
                continue
            for xyxy, conf, cls in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist()):
                c = int(cls)
                if cfg["kind"] == "coco":
                    if c not in COCO_TO_PROJECT:
                        continue          # drop the other 75 COCO classes after NMS
                    c = COCO_TO_PROJECT[c]
                x1, y1, x2, y2 = xyxy
                dets.append({"image_id": img_id, "category_id": c, "score": conf,
                             "bbox": [x1, y1, x2 - x1, y2 - y1]})
        if (start // BATCH) % 250 == 0:
            print(f"  {mname}: {start}/{len(src)} images", flush=True)
    CACHE_DIR.mkdir(exist_ok=True)
    json.dump(dets, open(cache, "w", encoding="utf-8"))
    print(f"  {mname}: {len(dets)} detections saved")
    return dets


def coco_eval(gt, dt, img_ids, cat_id=None):
    E = COCOeval(gt, dt, "bbox")
    E.params.imgIds = list(img_ids)
    if cat_id is not None:
        E.params.catIds = [cat_id]
    with contextlib.redirect_stdout(io.StringIO()):
        E.evaluate()
        E.accumulate()
        E.summarize()
    return [float("nan") if v < 0 else float(v) for v in E.stats]


def make_groups(images, attrs, stem_to_id):
    groups = {"all": [stem_to_id[Path(p).stem] for p in images]}
    for p in images:
        a = attrs.get(os.path.basename(p))
        if a is None:
            continue
        tod, weather = a
        iid = stem_to_id[Path(p).stem]
        groups.setdefault(f"weather={weather}", []).append(iid)
        groups.setdefault(f"timeofday={tod}", []).append(iid)
    return groups


def gt_size_counts(gt_dict, ids):
    ids = set(ids)
    s = m = l = 0
    for a in gt_dict["annotations"]:
        if a["image_id"] in ids:
            if a["area"] < SMALL:
                s += 1
            elif a["area"] < MEDIUM:
                m += 1
            else:
                l += 1
    return s, m, l


def main():
    images = read_list(ALL_LIST)
    names = load_names()
    attrs = load_attrs()
    print(f"test images: {len(images)}   classes: {names}")

    gt_dict = build_gt(images, names)
    stem_to_id = {Path(im["file_name"]).stem: im["id"] for im in gt_dict["images"]}
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO()
        gt.dataset = gt_dict
        gt.createIndex()

    groups = make_groups(images, attrs, stem_to_id)
    sizes = {g: gt_size_counts(gt_dict, ids) for g, ids in groups.items()}

    rows = []
    for mname, cfg in MODELS.items():
        if cfg["kind"] != "coco" and not os.path.exists(cfg["weights"]):
            print(f"[skip] weights not found: {cfg['weights']}")
            continue
        if cfg["kind"] == "enh" and ENH_REPLACE is None:
            print(f"[skip] {mname}: kind 'enh' needs ENH_REPLACE")
            continue

        print(f"\n##### {mname} #####")
        dets = predict(mname, cfg, images, stem_to_id)
        if not dets:
            print("  no detections, skipped")
            continue
        with contextlib.redirect_stdout(io.StringIO()):
            dt = gt.loadRes(dets)

        for g, ids in groups.items():
            st = coco_eval(gt, dt, ids)
            s, m, l = sizes[g]
            row = {"model": mname, "group": g, "n_images": len(ids),
                   "gt_small": s, "gt_medium": m, "gt_large": l,
                   "mAP50": st[1], "mAP50-95": st[0], "mAP75": st[2],
                   "AP_small": st[3], "AP_medium": st[4], "AP_large": st[5]}
            if PER_CLASS:
                for ci, cname in enumerate(names):
                    row[f"mAP50-95_{cname}"] = coco_eval(gt, dt, ids, cat_id=ci)[0]
            rows.append(row)
            print(f"  {g:<24} n={len(ids):<6} mAP50={st[1]:.4f}  mAP50-95={st[0]:.4f}  "
                  f"APs={st[3]:.4f}  APm={st[4]:.4f}  APl={st[5]:.4f}", flush=True)

    if not rows:
        print("No results.")
        return
    df = pd.DataFrame(rows).round(4)
    df.to_csv(OUT_CSV, index=False)
    for metric in ("mAP50-95", "mAP50", "mAP75", "AP_small", "AP_medium", "AP_large"):
        print(f"\n===== {metric} =====")
        print(df.pivot(index="group", columns="model", values=metric).to_string())
    print("\n===== images / GT boxes by size =====")
    print(df.drop_duplicates("group")[["group", "n_images", "gt_small", "gt_medium", "gt_large"]].to_string(index=False))
    print(f"\nSaved: {OUT_CSV}")


if __name__ == "__main__":
    main()
