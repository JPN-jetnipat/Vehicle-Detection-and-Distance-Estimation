"""Build side-by-side RAW | PREPROCESSED previews, to eyeball before preprocessing the whole dataset.

Samples N images from every (weather, timeofday) condition in
scripts/preprocess_router.py's ROUTING_TABLE, runs each through that
condition's pipeline, and writes one composite image per sample - raw on the
left, preprocessed on the right, labelled - into a single flat output folder.

Filenames are <weather>_<timeofday>_<nn>_<stem>.jpg so the folder sorts by
condition when browsed.

Usage (run from the repo root):
    python scripts/preview_preprocessing.py
    python scripts/preview_preprocessing.py --n-per-condition 5
    python scripts/preview_preprocessing.py --output-dir results/preprocessing_preview
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from preprocess_router import ROUTING_TABLE, get_pipeline  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
YOLO_DIR = ROOT / "dataset" / "yolo"
ATTR_INDEX_PATH = YOLO_DIR / "attr_index.json"
IMAGE_SRC_DIR = YOLO_DIR / "images" / "train"

LABEL_BAR_H = 44
DIVIDER_W = 4
FONT = cv2.FONT_HERSHEY_SIMPLEX

# BDD100K's weather tags are scene-level and noisy - a "rainy" frame often shows
# only a wet road with no droplets or streaks in view. These filenames were
# checked by eye against the image content so the rain preview actually shows
# rain; they're placed first, and the rest of the quota is filled normally.
CURATED_FIRST = {
    ("rainy", "daytime"): [
        "01705c36-8993db9f.jpg",   # windshield droplets + wet reflective road, headlights on
        "040dfa50-4082932d.jpg",   # windshield droplets, wet road, wiper visible
        "03112119-a36f2a78.jpg",   # windshield droplets, wet road
    ],
}


def apply_pipeline(img, pipeline):
    for fn in pipeline:
        img = fn(img)
    return img


def make_comparison(raw, processed, condition, pipeline_str):
    """Stack raw|processed side by side under a labelled bar."""
    h, w = raw.shape[:2]
    canvas = np.full((h + LABEL_BAR_H, w * 2 + DIVIDER_W, 3), 32, np.uint8)
    canvas[LABEL_BAR_H:, :w] = raw
    canvas[LABEL_BAR_H:, w + DIVIDER_W:] = processed

    cv2.putText(canvas, "RAW", (12, 31), FONT, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, "PREPROCESSED", (w + DIVIDER_W + 12, 31), FONT, 0.9, (120, 255, 120), 2, cv2.LINE_AA)

    caption = f"{condition}   |   {pipeline_str}"
    size = cv2.getTextSize(caption, FONT, 0.62, 1)[0]
    cv2.putText(canvas, caption, (canvas.shape[1] - size[0] - 12, 30), FONT, 0.62, (200, 200, 200), 1, cv2.LINE_AA)
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-per-condition", type=int, default=10, help="samples per condition (default: 10)")
    parser.add_argument("--output-dir", default="results/preprocessing_preview", help="single flat output folder")
    args = parser.parse_args()

    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    attr_index = json.loads(ATTR_INDEX_PATH.read_text(encoding="utf-8"))
    by_condition = {}
    for filename, attrs in attr_index.items():
        key = ((attrs.get("weather") or "").strip().lower(), (attrs.get("timeofday") or "").strip().lower())
        by_condition.setdefault(key, []).append(filename)

    total = 0
    for weather in ROUTING_TABLE:
        for timeofday in ROUTING_TABLE[weather]:
            candidates = sorted(by_condition.get((weather, timeofday), []))
            curated = CURATED_FIRST.get((weather, timeofday), [])
            ordered = curated + [f for f in candidates if f not in curated]

            pipeline = get_pipeline(weather, timeofday)
            pipeline_str = " -> ".join(fn.__name__ for fn in pipeline) or "raw (no-op)"
            condition = f"{weather}/{timeofday}"
            slug = f"{weather}_{timeofday}".replace(" ", "-").replace("/", "-")

            written = 0
            for filename in ordered:
                if written >= args.n_per_condition:
                    break
                src = IMAGE_SRC_DIR / filename
                if not src.exists():
                    continue
                raw = cv2.imread(str(src))
                if raw is None:
                    continue
                processed = apply_pipeline(raw.copy(), pipeline)
                composite = make_comparison(raw, processed, condition, pipeline_str)
                cv2.imwrite(str(out_dir / f"{slug}_{written + 1:02d}_{Path(filename).stem}.jpg"), composite)
                written += 1

            total += written
            note = "" if written == args.n_per_condition else f"  (พบแค่ {written} ภาพในเครื่อง)"
            print(f"{condition:26s} [{pipeline_str:28s}] {written:2d}{note}")

    print(f"\nเขียน {total} ภาพเปรียบเทียบ -> {out_dir}")


if __name__ == "__main__":
    main()
