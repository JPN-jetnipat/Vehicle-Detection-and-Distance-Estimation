"""Split dataset/yolo/images/val (+labels) into per-timeofday test subsets.

dataset/yolo/images/val holds BDD100K's official val set, which this
project's split scheme (see bdd100k_split_handoff/README.md) treats as the
held-out TEST pool. dataset/yolo/splits/test.txt lists the 8,841 image ids
in that pool that survive the class/undefined-attribute cleanup. This script
buckets those ids by timeofday (day / night / dawn-dusk) using
dataset/yolo/attr_index.csv, and copies the matching image+label pairs into
dataset/yolo/images/test_<bucket>/ and dataset/yolo/labels/test_<bucket>/.

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
    "daytime": "test_day",
    "night": "test_night",
    "dawn/dusk": "test_dawndusk",
}


def load_timeofday(attr_index_path: Path) -> dict[str, str]:
    lookup = {}
    with attr_index_path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["split"] == "val":
                lookup[row["name"]] = row["timeofday"]
    return lookup


def main() -> None:
    test_ids = TEST_LIST.read_text(encoding="utf-8").splitlines()
    test_ids = [line.strip() for line in test_ids if line.strip()]
    timeofday = load_timeofday(ATTR_INDEX)

    counts = {bucket: 0 for bucket in BUCKETS.values()}
    missing_attr = []
    missing_files = []

    for image_name in test_ids:
        tod = timeofday.get(image_name)
        bucket = BUCKETS.get(tod)
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
        print(f"  {bucket}: {n}")
    if missing_attr:
        print(f"WARNING: {len(missing_attr)} ids had no timeofday in attr_index (skipped)")
    if missing_files:
        print(f"WARNING: {len(missing_files)} ids missing source image/label (skipped)")


if __name__ == "__main__":
    main()
