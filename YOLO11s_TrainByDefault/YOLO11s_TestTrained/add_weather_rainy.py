#!/usr/bin/env python3
r"""Add a "rainy" weather split to YOLO11s_TestTrained/splits_local/, so
eval_pycocotools.py can score the trained model on it too - matching the
friend's result screenshot, which includes a rainy (n=737) breakdown
alongside timeofday.

Filters splits/test.txt by attr_index.json's "weather" field == "rainy".
No COCO remap needed here - the trained model already predicts using our
own 5-class ids, same as prepare_test_lists.py's other splits.

Run AFTER prepare_test_lists.py (reuses the same REPO_ROOT / dataset
locations, just adds one more list file next to the existing ones).

Usage (run from inside YOLO11s_TestTrained/):
  python add_weather_rainy.py
"""
import json
from pathlib import Path

# ---- EDIT THIS if your bdd100k_split_handoff folder is somewhere else ----
REPO_ROOT = Path(r"C:\Users\USER\OneDrive\เดสก์ท็อป\baby\bdd100k_split_handoff")
# ---------------------------------------------------------------------------

HERE = Path(__file__).resolve().parent
YOLO = REPO_ROOT / "dataset" / "yolo"
SPLITS_SRC = REPO_ROOT / "splits"
SPLITS_OUT = HERE / "splits_local"

WEATHER = "rainy"  # BDD100K's own label string, as stored in attr_index.json


def main():
    if not YOLO.exists():
        raise SystemExit(f"FLAG: {YOLO} not found - check REPO_ROOT at the top of this script")

    SPLITS_OUT.mkdir(parents=True, exist_ok=True)
    attr = json.loads((YOLO / "attr_index.json").read_text(encoding="utf-8"))
    test_names = [l.strip() for l in (SPLITS_SRC / "test.txt").read_text(encoding="utf-8").splitlines() if l.strip()]

    img_dir = YOLO / "images" / "val"  # TEST = BDD official val
    written, missing_img, missing_attr = [], 0, 0
    for name in test_names:
        a = attr.get(name)
        if a is None:
            missing_attr += 1
            continue
        if a.get("weather") != WEATHER:
            continue
        img = img_dir / name
        if not img.exists():
            missing_img += 1
            continue
        written.append(str(img.resolve()))

    out_path = SPLITS_OUT / "eval_test_rainy_images.txt"
    out_path.write_text("\n".join(written) + "\n", encoding="utf-8")
    if missing_attr:
        print(f"WARNING: {missing_attr} test names not found in attr_index.json")
    if missing_img:
        print(f"WARNING: {missing_img} rainy names had no matching image, skipped")
    print(f"rainy: {len(written)} images -> {out_path}")
    print("Next: python eval_pycocotools.py  (rainy is now included automatically)")


if __name__ == "__main__":
    main()
