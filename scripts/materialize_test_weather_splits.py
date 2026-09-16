"""Split dataset/yolo/images/val (+labels) into per-weather test subsets.

Sibling to materialize_test_splits.py (which buckets the same held-out test
pool by timeofday). This buckets by weather instead, so eval can report mAP
per weather condition in addition to per timeofday - see
scripts/preprocess_router.py's ROUTING_TABLE, which is conditioned on both.

dataset/yolo/images/val holds BDD100K's official val set, treated as the
held-out TEST pool. dataset/yolo/splits/test.txt lists the 8,841 image ids
in that pool. This script buckets those ids by weather using
dataset/yolo/attr_index.csv, and copies matching image+label pairs into
dataset/yolo/images/test_<bucket>/ and dataset/yolo/labels/test_<bucket>/.

foggy has only 13 images in the whole test pool (BDD100K itself is
fog-sparse) - test_foggy is still built so foggy can be reported like every
other condition, but treat its mAP as a rough signal, not a reliable number;
call this out wherever it's reported (a single image can swing it by
several points).

Run once; re-run is safe (skips files that already exist at the destination).
"""

import csv
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
YOLO_DIR = ROOT / "dataset" / "yolo"
ATTR_INDEX = YOLO_DIR / "attr_index.csv"
TEST_LIST = YOLO_DIR / "splits" / "test.txt"
SRC_IMAGES = YOLO_DIR / "images" / "val"
SRC_LABELS = YOLO_DIR / "labels" / "val"

BUCKETS = {
    "clear": "test_clear",
    "overcast": "test_overcast",
    "snowy": "test_snowy",
    "partly cloudy": "test_partlycloudy",
    "rainy": "test_rainy",
    "foggy": "test_foggy",
}

LOW_COUNT_WARN_THRESHOLD = 50


def load_weather(attr_index_path: Path) -> dict[str, str]:
    lookup = {}
    with attr_index_path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["split"] == "val":
                lookup[row["name"]] = row["weather"]
    return lookup


def main() -> None:
    test_ids = TEST_LIST.read_text(encoding="utf-8").splitlines()
    test_ids = [line.strip() for line in test_ids if line.strip()]
    weather = load_weather(ATTR_INDEX)

    counts = {bucket: 0 for bucket in BUCKETS.values()}
    missing_attr = []
    missing_files = []

    for image_name in test_ids:
        w = weather.get(image_name)
        bucket = BUCKETS.get(w)
        if bucket is None:
            missing_attr.append(image_name)
            continue

        stem = Path(image_name).stem
        src_img = SRC_IMAGES / image_name
        src_lbl = SRC_LABELS / f"{stem}.txt"
        if not src_img.exists() or not src_lbl.exists():
            missing_files.append(image_name)
            continue

        dst_img_dir = YOLO_DIR / "images" / bucket
        dst_lbl_dir = YOLO_DIR / "labels" / bucket
        dst_img_dir.mkdir(parents=True, exist_ok=True)
        dst_lbl_dir.mkdir(parents=True, exist_ok=True)

        dst_img = dst_img_dir / image_name
        dst_lbl = dst_lbl_dir / f"{stem}.txt"
        if not dst_img.exists():
            shutil.copy2(src_img, dst_img)
        if not dst_lbl.exists():
            shutil.copy2(src_lbl, dst_lbl)

        counts[bucket] += 1

    print(f"test.txt entries: {len(test_ids)}")
    for bucket, n in counts.items():
        flag = "  <- n too low for a reliable mAP, report with caveat" if n < LOW_COUNT_WARN_THRESHOLD else ""
        print(f"  {bucket}: {n}{flag}")
    if missing_attr:
        print(f"WARNING: {len(missing_attr)} ids had no weather in attr_index (skipped)")
    if missing_files:
        print(f"WARNING: {len(missing_files)} ids missing source image/label (skipped)")


if __name__ == "__main__":
    main()
