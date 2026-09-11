"""Build an offline, weather/timeofday-conditioned preprocessed copy of the BDD100K dataset.

Usage:
    python scripts/build_preprocessed_dataset.py --dry-run
    python scripts/build_preprocessed_dataset.py --splits train,val --max-per-bucket 5
    python scripts/build_preprocessed_dataset.py

Reads images from dataset/yolo/ (read-only, never modified) and writes
photometrically-preprocessed copies to dataset/yolo_preprocessed/. Label
.txt files are copied byte-for-byte (preprocessing here is purely
photometric, so boxes never move).

Routing per image is (weather, timeofday) -> ordered filter list, looked up
in scripts/preprocess_router.py from scripts/preprocess_router.py's
ROUTING_TABLE, using dataset/yolo/attr_index.json for the two attributes.

Split-naming trap (see bdd100k_split_handoff/README.md and project memory):
splits/val.txt (the real ~1542-image validation carve-out) lists filenames
that physically live under images/train/, NOT images/val/. images/val/ is
BDD's official val set, used here as the held-out test pool and already
bucketed into images/test_{day,night,dawndusk}/ by materialize_test_splits.py.
SUB_SPLITS below encodes the correct source directory for each case.

Must be run with the repo root as the working directory.
"""

import argparse
import json
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import cv2

from preprocess_router import bucket_label, get_pipeline

ROOT = Path(__file__).resolve().parent.parent
YOLO_DIR = ROOT / "dataset" / "yolo"
ATTR_INDEX_PATH = YOLO_DIR / "attr_index.json"
OUTPUT_ROOT = ROOT / "dataset" / "yolo_preprocessed"
REPORTS_DIR = ROOT / "results" / "preprocessing_reports"

# output_name -> (image_src_dir, label_src_dir, file_list_mode)
# file_list_mode is either ("split_file", <path under dataset/yolo/splits/>)
# or ("glob", None) to list every file already in image_src_dir.
SUB_SPLITS = {
    "train": (YOLO_DIR / "images" / "train", YOLO_DIR / "labels" / "train", ("split_file", "train_100.txt")),
    "val": (YOLO_DIR / "images" / "train", YOLO_DIR / "labels" / "train", ("split_file", "val.txt")),
    "test_day": (YOLO_DIR / "images" / "test_day", YOLO_DIR / "labels" / "test_day", ("glob", None)),
    "test_night": (YOLO_DIR / "images" / "test_night", YOLO_DIR / "labels" / "test_night", ("glob", None)),
    "test_dawndusk": (YOLO_DIR / "images" / "test_dawndusk", YOLO_DIR / "labels" / "test_dawndusk", ("glob", None)),
}


def load_attr_index() -> dict:
    with ATTR_INDEX_PATH.open(encoding="utf-8") as f:
        return json.load(f)


def list_filenames(image_src_dir: Path, file_list_mode) -> list:
    mode, arg = file_list_mode
    if mode == "split_file":
        split_path = YOLO_DIR / "splits" / arg
        names = split_path.read_text(encoding="utf-8").splitlines()
        return [n.strip() for n in names if n.strip()]
    if mode == "glob":
        return sorted(p.name for p in image_src_dir.glob("*.jpg"))
    raise ValueError(f"unknown file_list_mode: {mode}")


def apply_pipeline(img, pipeline):
    for fn in pipeline:
        img = fn(img)
    return img


def process_split(split_name: str, max_per_bucket, dry_run: bool, overwrite: bool, attr_index: dict) -> dict:
    image_src_dir, label_src_dir, file_list_mode = SUB_SPLITS[split_name]
    filenames = list_filenames(image_src_dir, file_list_mode)

    out_images_dir = OUTPUT_ROOT / "images" / split_name
    out_labels_dir = OUTPUT_ROOT / "labels" / split_name
    if not dry_run:
        out_images_dir.mkdir(parents=True, exist_ok=True)
        out_labels_dir.mkdir(parents=True, exist_ok=True)

    expected_counts = Counter()
    processed_counts = Counter()
    pipeline_by_bucket = {}
    missing_attr = []
    missing_image = []
    missing_label = []

    for filename in filenames:
        attrs = attr_index.get(filename, {})
        weather = attrs.get("weather")
        timeofday = attrs.get("timeofday")
        if filename not in attr_index:
            missing_attr.append(filename)

        bucket = bucket_label(weather, timeofday)
        pipeline = get_pipeline(weather, timeofday)
        pipeline_by_bucket[bucket] = [fn.__name__ for fn in pipeline]
        expected_counts[bucket] += 1

        if max_per_bucket is not None and processed_counts[bucket] >= max_per_bucket:
            continue

        src_img = image_src_dir / filename
        if not src_img.exists():
            missing_image.append(filename)
            continue

        stem = Path(filename).stem
        src_lbl = label_src_dir / f"{stem}.txt"
        dst_img = out_images_dir / filename
        dst_lbl = out_labels_dir / f"{stem}.txt"

        if not dry_run:
            if overwrite or not dst_img.exists():
                img = cv2.imread(str(src_img))
                if img is None:
                    missing_image.append(filename)
                    continue
                img = apply_pipeline(img, pipeline)
                cv2.imwrite(str(dst_img), img)
            if src_lbl.exists():
                if overwrite or not dst_lbl.exists():
                    shutil.copy2(src_lbl, dst_lbl)
            else:
                missing_label.append(filename)

        processed_counts[bucket] += 1

    return {
        "split": split_name,
        "total_files_listed": len(filenames),
        "total_expected": sum(expected_counts.values()),
        "total_processed": sum(processed_counts.values()),
        "buckets": {
            bucket: {
                "pipeline": pipeline_by_bucket[bucket],
                "expected": expected_counts[bucket],
                "processed": processed_counts[bucket],
            }
            for bucket in sorted(expected_counts)
        },
        "missing_attr_count": len(missing_attr),
        "missing_image_count": len(missing_image),
        "missing_label_count": len(missing_label),
        "missing_attr_sample": missing_attr[:10],
        "missing_image_sample": missing_image[:10],
        "missing_label_sample": missing_label[:10],
    }


def print_summary(report: dict) -> None:
    print(f"\n=== {report['split']} ===")
    print(f"listed: {report['total_files_listed']}  expected: {report['total_expected']}  processed: {report['total_processed']}")
    if report["missing_attr_count"]:
        print(f"  WARNING: {report['missing_attr_count']} files missing from attr_index.json (treated as undefined/raw)")
    if report["missing_image_count"]:
        print(f"  WARNING: {report['missing_image_count']} source images not found on disk")
    if report["missing_label_count"]:
        print(f"  WARNING: {report['missing_label_count']} label .txt files not found")
    for bucket, stats in report["buckets"].items():
        pipeline_str = " -> ".join(stats["pipeline"]) or "raw"
        print(f"  {bucket:30s} pipeline=[{pipeline_str:30s}] expected={stats['expected']:6d} processed={stats['processed']:6d}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--splits",
        default="all",
        help=f"comma-separated sub-splits to process, or 'all'. choices: {', '.join(SUB_SPLITS)}",
    )
    parser.add_argument(
        "--max-per-bucket",
        type=int,
        default=None,
        help="only process up to N images per (weather,timeofday) bucket - for sanity-checking before a full run",
    )
    parser.add_argument("--dry-run", action="store_true", help="compute routing/counts only, write nothing")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="reprocess/overwrite files that already exist in the output dir (default: skip existing, safe to re-run)",
    )
    args = parser.parse_args()

    if args.splits == "all":
        split_names = list(SUB_SPLITS)
    else:
        split_names = [s.strip() for s in args.splits.split(",") if s.strip()]
        unknown = [s for s in split_names if s not in SUB_SPLITS]
        if unknown:
            sys.exit(f"unknown split(s): {unknown}. choices: {list(SUB_SPLITS)}")

    attr_index = load_attr_index()

    all_reports = {}
    for split_name in split_names:
        report = process_split(split_name, args.max_per_bucket, args.dry_run, args.overwrite, attr_index)
        print_summary(report)
        all_reports[split_name] = report

    if not args.dry_run:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        report_path = REPORTS_DIR / f"preprocess_{'-'.join(split_names)}_{timestamp}.json"
        report_path.write_text(json.dumps(all_reports, indent=2), encoding="utf-8")
        print(f"\nWrote report: {report_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
