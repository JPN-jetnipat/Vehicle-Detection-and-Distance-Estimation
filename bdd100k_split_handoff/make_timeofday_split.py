"""
Materialize a timeofday-filtered split for the DANN experiments.

Reads the existing split handoff artifacts and creates new image/label
folders containing only images of the requested timeofday. Uses hardlinks
so no disk space is duplicated.

Run from the bdd100k_split_handoff root:

    python make_timeofday_split.py --timeofday daytime --suffix day
    python make_timeofday_split.py --timeofday night   --suffix night

Creates, for suffix "day":
    dataset/yolo/images/train_day   dataset/yolo/labels/train_day
    dataset/yolo/images/val_day     dataset/yolo/labels/val_day
"""

import argparse
import csv
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

ATTR_INDEX = ROOT / "dataset" / "yolo" / "attr_index.csv"
SRC_IMAGES = ROOT / "dataset" / "yolo" / "images" / "train"
SRC_LABELS = ROOT / "dataset" / "yolo" / "labels" / "train"

# Both the training pool and the project validation set are drawn from the
# original BDD train split, so their images/labels both live under train/.
SPLIT_FILES = {
    "train": ROOT / "splits" / "train_100.txt",
    "val": ROOT / "splits" / "val.txt",
}


def load_timeofday(path):
    """Map image basename -> timeofday."""
    mapping = {}
    with open(path, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            mapping[row["name"]] = row["timeofday"]
    return mapping


def read_split(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def link_one(src, dst):
    """Hardlink src -> dst, falling back to a copy if linking is unsupported."""
    if dst.exists():
        return "skipped"
    try:
        os.link(src, dst)
        return "linked"
    except OSError:
        shutil.copy2(src, dst)
        return "copied"


def materialize(names, suffix, split_name):
    out_images = ROOT / "dataset" / "yolo" / "images" / f"{split_name}_{suffix}"
    out_labels = ROOT / "dataset" / "yolo" / "labels" / f"{split_name}_{suffix}"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    stats = {"linked": 0, "copied": 0, "skipped": 0}
    missing_image = 0
    missing_label = 0

    for name in names:
        src_img = SRC_IMAGES / name
        if not src_img.exists():
            missing_image += 1
            continue

        stem = Path(name).stem
        src_lbl = SRC_LABELS / f"{stem}.txt"
        if not src_lbl.exists():
            # No label file means no annotation was produced for this image.
            missing_label += 1
            continue

        stats[link_one(src_img, out_images / name)] += 1
        link_one(src_lbl, out_labels / f"{stem}.txt")

    print(f"  {split_name}_{suffix}: {sum(stats.values())} images "
          f"(linked={stats['linked']} copied={stats['copied']} "
          f"already_there={stats['skipped']})")
    if missing_image:
        print(f"  WARNING: {missing_image} images listed in split but not on disk")
    if missing_label:
        print(f"  WARNING: {missing_label} images had no label file, excluded")

    return sum(stats.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeofday", required=True,
                    choices=["daytime", "night", "dawn/dusk"],
                    help="timeofday value to keep, as written in attr_index.csv")
    ap.add_argument("--suffix", required=True,
                    help="folder suffix, e.g. 'day' -> train_day / val_day")
    args = ap.parse_args()

    for p in (ATTR_INDEX, SRC_IMAGES, SRC_LABELS, *SPLIT_FILES.values()):
        if not p.exists():
            sys.exit(f"ERROR: expected path not found: {p}")

    print(f"Loading {ATTR_INDEX.name} ...")
    tod = load_timeofday(ATTR_INDEX)
    print(f"  {len(tod)} entries")

    total = 0
    for split_name, split_path in SPLIT_FILES.items():
        names = read_split(split_path)
        kept = [n for n in names if tod.get(n) == args.timeofday]
        print(f"{split_path.name}: {len(names)} total -> "
              f"{len(kept)} with timeofday='{args.timeofday}'")
        total += materialize(kept, args.suffix, split_name)

    print(f"\nDone. {total} images materialized under suffix '{args.suffix}'.")


if __name__ == "__main__":
    main()
