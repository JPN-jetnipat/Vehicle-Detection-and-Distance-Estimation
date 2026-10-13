#!/usr/bin/env python3
r"""Add a "rainy" weather split to YOLO11s_Evaluation/, for Experiment 1
(plain COCO-pretrained yolo11s.pt) - matching the friend's result
screenshot, which includes a rainy (n=737) breakdown.

Same COCO-id remap as build_coco_eval_set.py (car=2, truck=7, bus=5,
motor=3, bike=1) - required because the baseline model predicts using raw
COCO ids, not our 0-4 ids.

Run AFTER build_coco_eval_set.py (reuses REPO_ROOT / OUR_TO_COCO / output
locations, just adds one more subset next to the existing ones).

Usage (run from inside YOLO11s_Evaluation/):
  python add_weather_rainy_coco.py
"""
import json
import os
from pathlib import Path

# ---- EDIT THIS if your bdd100k_split_handoff folder is somewhere else ----
REPO_ROOT = Path(r"C:\Users\USER\OneDrive\เดสก์ท็อป\baby\bdd100k_split_handoff")
# ---------------------------------------------------------------------------

HERE = Path(__file__).resolve().parent
YOLO = REPO_ROOT / "dataset" / "yolo"
SPLITS_SRC = REPO_ROOT / "splits"
YOLO_COCO = HERE / "dataset_yolo_coco"
SPLITS_OUT = HERE / "splits_eval"

# must match build_coco_eval_set.py's OUR_TO_COCO
OUR_TO_COCO = {0: 2, 1: 7, 2: 5, 3: 3, 4: 1}
WEATHER = "rainy"


def remap_label_file(src: Path, dst: Path) -> None:
    if not src.exists():
        dst.write_text("", encoding="utf-8")
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


def main():
    if not YOLO.exists():
        raise SystemExit(f"FLAG: {YOLO} not found - check REPO_ROOT at the top of this script")

    attr = json.loads((YOLO / "attr_index.json").read_text(encoding="utf-8"))
    test_names = [l.strip() for l in (SPLITS_SRC / "test.txt").read_text(encoding="utf-8").splitlines() if l.strip()]

    rainy_names = [n for n in test_names if attr.get(n, {}).get("weather") == WEATHER]

    out_stem = "eval_test_rainy"
    img_out = YOLO_COCO / "images" / out_stem
    lbl_out = YOLO_COCO / "labels" / out_stem
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    src_img_dir = YOLO / "images" / "val"
    src_lbl_dir = YOLO / "labels" / "val"

    written, missing = [], 0
    for name in rainy_names:
        stem = Path(name).stem
        src_img = src_img_dir / name
        if not src_img.exists():
            missing += 1
            continue
        dst_img = img_out / name
        if not dst_img.exists():
            os.link(src_img, dst_img)
        remap_label_file(src_lbl_dir / f"{stem}.txt", lbl_out / f"{stem}.txt")
        written.append(str(dst_img.resolve()))

    SPLITS_OUT.mkdir(parents=True, exist_ok=True)
    list_path = SPLITS_OUT / f"{out_stem}_images.txt"
    list_path.write_text("\n".join(written) + "\n", encoding="utf-8")
    if missing:
        print(f"WARNING: {missing} rainy names had no matching image, skipped")
    print(f"rainy: {len(written)} images -> {list_path}")
    print("Next: python eval_pycocotools_baseline.py  (rainy is now included automatically)")


if __name__ == "__main__":
    main()
