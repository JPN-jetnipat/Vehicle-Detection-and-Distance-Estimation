#!/usr/bin/env python3
"""Build a COCO-class-id eval set so a plain (COCO-pretrained) yolo11s.pt
can be scored against our BDD 5-class labels with correct, comparable mAP.

Self-contained: lives in YOLO11s_Evaluation/ alongside the other eval files.
Reads source data (images/labels/attr_index) from the main bdd100k_split_handoff
repo via REPO_ROOT below - does NOT touch anything in that repo, all new
files (yolo_coco labels, eval list txts) are written under THIS folder
(YOLO11s_Evaluation/), so everything for this experiment stays together
and organized. Images are NOT duplicated - the list files just point back
at the real image files already materialized in the main repo.

Usage (run from inside YOLO11s_Evaluation/):
  python build_coco_eval_set.py
"""
import json
import os
from collections import defaultdict
from pathlib import Path

# ---- EDIT THIS if your bdd100k_split_handoff folder is somewhere else ----
REPO_ROOT = Path(r"C:\Users\USER\OneDrive\เดสก์ท็อป\baby\bdd100k_split_handoff")
# ---------------------------------------------------------------------------

HERE = Path(__file__).resolve().parent          # YOLO11s_Evaluation/
YOLO = REPO_ROOT / "dataset" / "yolo"            # source (read-only)
SPLITS_SRC = REPO_ROOT / "splits"                # source (read-only)

OUT = HERE / "dataset_yolo_coco"                 # everything new goes here
SPLITS_OUT = HERE / "splits_eval"

# our class id -> COCO class id (see configs/data/bdd100k_vehicle5.yaml for our ids)
# car=0->2, truck=1->7, bus=2->5, motor=3->3, bike=4->1
OUR_TO_COCO = {0: 2, 1: 7, 2: 5, 3: 3, 4: 1}

TIMEOFDAY_TO_STRATUM = {"daytime": "day", "night": "night", "dawn/dusk": "dawndusk"}


def remap_label_file(src: Path, dst: Path) -> None:
    if not src.exists():
        dst.write_text("", encoding="utf-8")  # background image, keep it as an empty label too
        return
    out_lines = []
    for line in src.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        cls = int(parts[0])
        if cls not in OUR_TO_COCO:
            raise SystemExit(f"FLAG: unexpected class id {cls} in {src} - not in OUR_TO_COCO")
        parts[0] = str(OUR_TO_COCO[cls])
        out_lines.append(" ".join(parts))
    dst.write_text("\n".join(out_lines) + ("\n" if out_lines else ""), encoding="utf-8")


def build_subset(names, src_split: str, out_stem: str) -> Path:
    """src_split is 'train' or 'val' (which dataset/yolo/{images,labels} folder
    the real files live in, inside the MAIN repo).

    Ultralytics finds an image's label by string-replacing '/images/' with
    '/labels/' in the image's own path - so the remapped (COCO-id) label only
    gets picked up if the image path also lives under a matching .../images/<out_stem>/
    folder here in YOLO11s_Evaluation/. We therefore hardlink the images too
    (not copies - zero extra disk, same file on disk, just a second folder
    entry) so everything for this experiment is self-contained in one place
    instead of being scattered across the main repo.
    """
    img_out = OUT / "images" / out_stem
    lbl_out = OUT / "labels" / out_stem
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    src_img_dir = YOLO / "images" / src_split
    src_lbl_dir = YOLO / "labels" / src_split

    list_path = SPLITS_OUT / f"{out_stem}_images.txt"
    written = []
    n_missing = 0
    n_total = len(names)
    print(f"{out_stem}: processing {n_total} images...", flush=True)
    for i, name in enumerate(names, 1):
        stem = Path(name).stem
        src_img = src_img_dir / name
        if not src_img.exists():
            n_missing += 1
            continue
        dst_img = img_out / name
        if not dst_img.exists():
            os.link(src_img, dst_img)
        remap_label_file(src_lbl_dir / f"{stem}.txt", lbl_out / f"{stem}.txt")
        written.append(str(dst_img.resolve()))
        if i % 500 == 0 or i == n_total:
            print(f"  {out_stem}: {i}/{n_total} done", flush=True)
    list_path.write_text("\n".join(written) + "\n", encoding="utf-8")
    if n_missing:
        print(f"  WARNING: {n_missing} names for {out_stem} had no source image, skipped")
    print(f"{out_stem}: {len(written)} images -> {list_path}")
    return list_path


def main():
    if not YOLO.exists():
        raise SystemExit(f"FLAG: {YOLO} not found - check REPO_ROOT at the top of this script")

    SPLITS_OUT.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)

    attr = json.loads((YOLO / "attr_index.json").read_text(encoding="utf-8"))

    # 1) project's own VAL (2.5% carve-out, lives under images/labels/train)
    val_names = [l.strip() for l in (SPLITS_SRC / "val.txt").read_text(encoding="utf-8").splitlines() if l.strip()]
    build_subset(val_names, "train", "eval_val")

    # 2) TEST, split by timeofday (lives under images/labels/val = BDD's official val)
    test_names = [l.strip() for l in (SPLITS_SRC / "test.txt").read_text(encoding="utf-8").splitlines() if l.strip()]
    build_subset(test_names, "val", "eval_test_all")

    buckets = defaultdict(list)
    n_missing_attr = 0
    for name in test_names:
        a = attr.get(name)
        if a is None:
            n_missing_attr += 1
            continue
        stratum = TIMEOFDAY_TO_STRATUM.get(a["timeofday"])
        if stratum is None:
            continue  # undefined etc - shouldn't happen in test_clean, but be safe
        buckets[stratum].append(name)
    if n_missing_attr:
        print(f"WARNING: {n_missing_attr} test names not found in attr_index.json")

    for stratum, names in buckets.items():
        build_subset(names, "val", f"eval_test_{stratum}")

    print(f"\nDone. Everything written under: {HERE}")
    print(f"  labels (COCO-id remapped): {OUT / 'labels'}")
    print(f"  image-list txts:           {SPLITS_OUT}")
    print("Next: python eval_yolo11s_baseline.py")


if __name__ == "__main__":
    main()