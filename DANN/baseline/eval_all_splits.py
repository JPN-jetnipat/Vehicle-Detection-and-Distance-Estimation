"""
Run Ultralytics validation across all evaluation splits for one trained
model and print/save per-class results in a consistent format.

Usage (from the DANN folder, e.g. C:\\...\\baby\\DANN):

    python eval_all_splits.py --model runs/detect/srconly_day/weights/best.pt

Add --tag to label the output file, e.g. for comparing against a later
DANN run:

    python eval_all_splits.py --model runs/detect/dann_v3/weights/best.pt --tag dann_v3

Splits evaluated (edit SPLITS below to add/remove):
    val_day, test_day, val_night, test_night, val_dawndusk, test_dawndusk

Each split needs a matching bdd100k_<split>.yaml in the current folder,
e.g. bdd100k_val_day.yaml, bdd100k_test_night.yaml, etc.
"""

import argparse
import csv
from pathlib import Path

from ultralytics import YOLO

SPLITS = [
    "val_day",
    "test_day",
    "val_night",
    "test_night",
    "val_dawndusk",
    "test_dawndusk",
]

CLASS_NAMES = ["car", "truck", "bus", "motor", "bike"]


def fmt(x):
    return f"{x:.3f}"


def run_one(model, split_name):
    data_yaml = f"bdd100k_{split_name}.yaml"
    if not Path(data_yaml).exists():
        print(f"  SKIP {split_name}: {data_yaml} not found")
        return None

    metrics = model.val(data=data_yaml, verbose=False, plots=False)

    # Overall (all-class) row
    box = metrics.box
    n_images = metrics.nt_per_image.sum() if hasattr(metrics, "nt_per_image") else None
    overall = {
        "class": "all",
        "images": n_images,
        "instances": int(box.nt_per_class.sum()) if hasattr(box, "nt_per_class") else None,
        "P": box.mp,
        "R": box.mr,
        "mAP50": box.map50,
        "mAP50-95": box.map,
    }

    rows = [overall]
    # Per-class rows, in the model's class-index order
    ap50_per_class = box.ap50  # array over classes present in val set
    ap_per_class = box.ap
    p_per_class = box.p
    r_per_class = box.r
    nt_per_class = box.nt_per_class if hasattr(box, "nt_per_class") else None

    for i, cls_idx in enumerate(box.ap_class_index):
        name = model.names[int(cls_idx)]
        rows.append({
            "class": name,
            "images": None,  # Ultralytics doesn't expose per-class image count directly
            "instances": int(nt_per_class[cls_idx]) if nt_per_class is not None else None,
            "P": p_per_class[i],
            "R": r_per_class[i],
            "mAP50": ap50_per_class[i],
            "mAP50-95": ap_per_class[i],
        })

    return rows


def print_table(split_name, rows):
    print(f"\n=== {split_name} ===\n")
    print(f"{'Class':>10} {'Instances':>10} {'P':>8} {'R':>8} {'mAP50':>8} {'mAP50-95':>10}")
    for r in rows:
        inst = r["instances"] if r["instances"] is not None else "-"
        print(f"{r['class']:>10} {inst:>10} "
              f"{fmt(r['P']):>8} {fmt(r['R']):>8} "
              f"{fmt(r['mAP50']):>8} {fmt(r['mAP50-95']):>10}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="path to best.pt")
    ap.add_argument("--tag", default=None, help="label for the output CSV, e.g. srconly_day")
    args = ap.parse_args()

    model = YOLO(args.model)
    tag = args.tag or Path(args.model).parent.parent.name  # e.g. runs/detect/<name>/weights/best.pt -> <name>

    all_rows = []
    for split_name in SPLITS:
        result = run_one(model, split_name)
        if result is None:
            continue
        print_table(split_name, result)
        for r in result:
            all_rows.append({"split": split_name, **r})

    out_csv = Path(f"eval_summary_{tag}.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "class", "images", "instances", "P", "R", "mAP50", "mAP50-95"])
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)

    print(f"\nSaved summary to {out_csv.resolve()}")


if __name__ == "__main__":
    main()
