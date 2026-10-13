"""
eval_weather.py

Evaluate the 3 models on every per-weather split made by split_by_weather.py.
No retraining; only the evaluation set changes.

Model kinds:
    "raw"  : fine-tuned on the 5 vehicle classes, evaluated on raw images
    "enh"  : Set1 model, evaluated on the enhanced images (needs ENH_REPLACE in split script)
    "coco" : COCO-pretrained 80-class model. Labels are remapped to COCO ids
             (car=2, truck=7, bus=5, motor=3, bike=1) in a temp copy, and only
             those 5 classes are scored.

Run:  python eval_weather.py
"""

import glob
import os
import shutil
from pathlib import Path

import pandas as pd
import yaml
from ultralytics import YOLO

# ---------------------------- CONFIG ----------------------------
ROOT = Path(__file__).resolve().parent
SPLIT_DIR = ROOT / "weather_splits"
DATA_YAML = ROOT / "configs" / "data" / "bdd100k_vehicle5.yaml"
COCO_TMP = ROOT / "_coco_remap"

MODELS = {
    "A_coco_pretrained": {"weights": "yolo11s.pt", "kind": "coco"},
    "B_finetuned_100ep": {"weights": r"C:\Users\USER\OneDrive\เดสก์ท็อป\baby\YOLO11s_TrainByDefault\run1\weights\best.pt", "kind": "raw"},
    "C_set1_lowlight":   {"weights": r"C:\Users\USER\OneDrive\เดสก์ท็อป\baby\Augment_method1\runs\detect\method1_lowlight\weights\best.pt", "kind": "raw"},
}

# project class id (from bdd100k_vehicle5.yaml) -> COCO class id
# README order: car(0) truck(1) bus(2) motor(3) bike(4)
TO_COCO = {0: 2, 1: 7, 2: 5, 3: 3, 4: 1}

IMGSZ, BATCH, DEVICE, WORKERS = 640, 16, 0, 0
CONF, IOU = 0.001, 0.7
OUT_CSV = ROOT / "weather_results.csv"
# ----------------------------------------------------------------


def img_to_label(img_path):
    s = os.path.normpath(str(img_path))
    sep = os.sep
    i = s.rfind(sep + "images" + sep)
    lab = s[:i] + sep + "labels" + sep + s[i + len(sep + "images" + sep):]
    return os.path.splitext(lab)[0] + ".txt"


def read_list(txt):
    return [l.strip() for l in open(txt, encoding="utf-8") if l.strip()]


def build_coco_copy(all_txt):
    """Hardlink (or copy) test images and write COCO-id labels. Returns {orig path: new path}."""
    img_dir = COCO_TMP / "images" / "test"
    lab_dir = COCO_TMP / "labels" / "test"
    img_dir.mkdir(parents=True, exist_ok=True)
    lab_dir.mkdir(parents=True, exist_ok=True)
    mapping = {}
    for p in read_list(all_txt):
        name = os.path.basename(p)
        dst = img_dir / name
        if not dst.exists():
            try:
                os.link(p, dst)
            except OSError:
                shutil.copy2(p, dst)
        lines = []
        src_lab = img_to_label(p)
        if os.path.exists(src_lab):
            for ln in open(src_lab, encoding="utf-8"):
                parts = ln.split()
                if parts:
                    parts[0] = str(TO_COCO[int(parts[0])])
                    lines.append(" ".join(parts))
        (lab_dir / (os.path.splitext(name)[0] + ".txt")).write_text("\n".join(lines), encoding="utf-8")
        mapping[p] = str(dst)
    return mapping


def coco_yaml_for(weather_txt, mapping, coco_names, key):
    new_paths = [mapping[p] for p in read_list(weather_txt)]
    txt = COCO_TMP / f"list_{key}.txt"
    txt.write_text("\n".join(new_paths), encoding="utf-8")
    yml = COCO_TMP / f"data_{key}.yaml"
    yml.write_text(yaml.safe_dump({
        "path": COCO_TMP.as_posix(), "train": txt.as_posix(), "val": txt.as_posix(),
        "nc": len(coco_names), "names": coco_names}, allow_unicode=True), encoding="utf-8")
    return str(yml), len(new_paths)


def main():
    rows = []
    mapping = None

    for mname, cfg in MODELS.items():
        weights, kind = cfg["weights"], cfg["kind"]
        if not os.path.exists(weights) and kind != "coco":
            print(f"[skip] weights not found: {weights}")
            continue

        tag = "enh" if kind == "enh" else "raw"
        yamls = sorted(glob.glob(str(SPLIT_DIR / f"{tag}_*.yaml")))
        if not yamls:
            print(f"[skip] {mname}: no {tag}_*.yaml in {SPLIT_DIR}")
            continue

        print(f"\n########## {mname} ({weights}, kind={kind}) ##########")
        model = YOLO(weights)

        if kind == "coco" and mapping is None:
            print("Building COCO-remapped copy of the test set (one time)...")
            mapping = build_coco_copy(SPLIT_DIR / "raw_all.txt")

        for yml in yamls:
            weather = Path(yml).stem.replace(f"{tag}_", "")
            txt = yml.replace(".yaml", ".txt")
            n_img = len(read_list(txt))
            extra = {}
            data = yml
            if kind == "coco":
                data, n_img = coco_yaml_for(txt, mapping, [model.names[i] for i in sorted(model.names)], weather)
                extra = {"classes": sorted(TO_COCO.values())}

            print(f"\n=== {mname} | weather={weather} | images={n_img} ===")
            try:
                m = model.val(data=data, imgsz=IMGSZ, batch=BATCH, device=DEVICE, workers=WORKERS,
                              conf=CONF, iou=IOU, split="val", plots=False, verbose=False,
                              name=f"val_{mname}_{weather}", **extra)
            except Exception as e:
                print(f"[error] {mname}/{weather}: {e}")
                continue

            rows.append({
                "model": mname, "weather": weather, "n_images": n_img,
                "mAP50-95": round(float(m.box.map), 4),
                "mAP50": round(float(m.box.map50), 4),
                "precision": round(float(m.box.mp), 4),
                "recall": round(float(m.box.mr), 4),
            })

    if not rows:
        print("No results.")
        return
    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print("\n===== mAP50-95 =====")
    print(df.pivot(index="weather", columns="model", values="mAP50-95").to_string())
    print("\n===== mAP50 =====")
    print(df.pivot(index="weather", columns="model", values="mAP50").to_string())
    print("\n===== images per weather =====")
    print(df.groupby("weather")["n_images"].max().to_string())
    print(f"\nSaved: {OUT_CSV}")


if __name__ == "__main__":
    main()