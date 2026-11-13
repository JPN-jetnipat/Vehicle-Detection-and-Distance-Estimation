"""Convert the raw ExDark dataset into a YOLO-format test split for our 5-class vehicle models.

Raw ExDark layout (as downloaded):
    <images root>/<Class>/<image>.(jpg|png|...)
    <annotations root>/<Class>/<image>.<ext>.txt     # "% bbGt version=3" header, then
                                                      # "<Class> left top width height 0 0 0 0 0 0 0"
Folder nesting doesn't matter - both roots are searched recursively.

Only the 4 ExDark classes that exist in our BDD label set are kept; everything
else (People, Dog, Cup, ...) is dropped. ExDark has no truck class, so truck
has 0 ground-truth instances and is not scored.

    ExDark     -> ours
    Car        -> 0 car
    Bus        -> 2 bus
    Motorbike  -> 3 motor
    Bicycle    -> 4 bike

By default only images with at least one vehicle box are written (the other
~5k ExDark images are mostly indoor scenes of cups/chairs/pets that a road
vehicle detector was never meant to see). Pass --keep-background to also
write those as empty-label images, which punishes false positives harder.

Every image is re-encoded as a plain JPEG with EXIF stripped, so the pixel
grid the model sees is exactly the one ExDark's box coordinates refer to
(some ExDark photos carry EXIF rotation tags that would otherwise be applied
on load and misalign the boxes).

Output (paths follow the images/ <-> labels/ convention ultralytics and
eval_arm_coco.py rely on):
    dataset/exdark_yolo/images/test/*.jpg
    dataset/exdark_yolo/labels/test/*.txt
    dataset/exdark_test.yaml

Usage (from repo root):
    python scripts/prepare_exdark.py --images /data/ExDark/ExDark --annotations /data/ExDark/ExDark_Annno
"""

from __future__ import annotations

import argparse
import shutil
from collections import Counter
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "dataset" / "exdark_yolo"
YAML_PATH = ROOT / "dataset" / "exdark_test.yaml"

CLASS_MAP = {"car": 0, "bus": 2, "motorbike": 3, "bicycle": 4}
NAMES = {0: "car", 1: "truck", 2: "bus", 3: "motor", 4: "bike"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

YAML_TEMPLATE = """\
# ExDark low-light test set, converted by scripts/prepare_exdark.py - eval only.
# Only car/bus/motor/bike exist in ExDark; truck has no ground truth.
path: dataset/exdark_yolo

# train: unused (this yaml is for eval only) - set to satisfy ultralytics' required-key check
train: images/test
val: images/test
test: images/test

names:
  0: car
  1: truck
  2: bus
  3: motor
  4: bike
"""


def index_annotations(anno_root: Path) -> dict[str, Path]:
    """Map lower-cased image filename (e.g. '2015_00001.jpg') -> its annotation file."""
    index = {}
    for txt in anno_root.rglob("*.txt"):
        index[txt.name[: -len(".txt")].lower()] = txt
    return index


def find_annotation(image: Path, index: dict[str, Path]) -> Path | None:
    hit = index.get(image.name.lower())
    if hit:
        return hit
    # Some annotation files are named by stem only, or with a different extension
    stem = image.stem.lower()
    for ext in ("", ".jpg", ".jpeg", ".png", ".JPG"):
        hit = index.get(stem + ext.lower())
        if hit:
            return hit
    return None


def parse_annotation(anno: Path, W: int, H: int, unknown: Counter) -> list[str]:
    """Return YOLO label lines for the vehicle boxes in one ExDark annotation file."""
    lines = []
    for raw in anno.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = raw.split()
        if not parts or parts[0].startswith("%") or len(parts) < 5:
            continue
        cls = CLASS_MAP.get(parts[0].lower())
        if cls is None:
            unknown[parts[0]] += 1
            continue
        left, top, w, h = (float(v) for v in parts[1:5])
        # clamp to the image - a few ExDark boxes spill a pixel or two past the border
        x1, y1 = max(0.0, left), max(0.0, top)
        x2, y2 = min(float(W), left + w), min(float(H), top + h)
        if x2 - x1 < 1 or y2 - y1 < 1:
            continue
        cx, cy = (x1 + x2) / 2 / W, (y1 + y2) / 2 / H
        lines.append(f"{cls} {cx:.6f} {cy:.6f} {(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True, type=Path, help="ExDark image root (the folder with Bicycle/, Bus/, Car/ ...)")
    parser.add_argument("--annotations", required=True, type=Path, help="ExDark annotation root (ExDark_Annno/)")
    parser.add_argument("--keep-background", action="store_true", help="Also write images with no vehicle boxes, as empty labels.")
    args = parser.parse_args()

    for p in (args.images, args.annotations):
        if not p.is_dir():
            raise SystemExit(f"Not a directory: {p}")

    anno_index = index_annotations(args.annotations)
    if not anno_index:
        raise SystemExit(f"No .txt annotation files found under {args.annotations}")
    images = sorted(p for p in args.images.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    if not images:
        raise SystemExit(f"No images found under {args.images}")
    print(f"Found {len(images)} images and {len(anno_index)} annotation files")

    img_out, lbl_out = OUT_DIR / "images" / "test", OUT_DIR / "labels" / "test"
    # rebuild from scratch so a rerun never leaves stale images behind
    shutil.rmtree(OUT_DIR, ignore_errors=True)
    img_out.mkdir(parents=True)
    lbl_out.mkdir(parents=True)

    stats = Counter()
    boxes = Counter()
    unknown = Counter()
    used_stems: set[str] = set()
    for image in images:
        anno = find_annotation(image, anno_index)
        if anno is None:
            stats["no_annotation"] += 1
            continue
        try:
            with Image.open(image) as im:
                im = im.convert("RGB")  # convert() drops EXIF, so no rotation is applied later
                W, H = im.size
                lines = parse_annotation(anno, W, H, unknown)
                if not lines and not args.keep_background:
                    stats["no_vehicle"] += 1
                    continue
                stem = image.stem
                if stem.lower() in used_stems:  # same filename in two class folders
                    stem = f"{image.parent.name}_{stem}"
                used_stems.add(stem.lower())
                im.save(img_out / f"{stem}.jpg", quality=95)
        except OSError as e:
            print(f"skip unreadable image {image}: {e}")
            stats["unreadable"] += 1
            continue
        (lbl_out / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        stats["written"] += 1
        for line in lines:
            boxes[NAMES[int(line.split()[0])]] += 1

    YAML_PATH.write_text(YAML_TEMPLATE, encoding="utf-8")

    print(f"\nWrote {stats['written']} images to {img_out}")
    print(f"Skipped: {stats['no_vehicle']} with no vehicle boxes, {stats['no_annotation']} with no annotation file, "
          f"{stats['unreadable']} unreadable")
    print("Vehicle boxes: " + ", ".join(f"{k}={boxes[k]}" for k in ("car", "bus", "motor", "bike")))
    print(f"Dropped non-vehicle boxes: {dict(unknown)}")
    print(f"Dataset yaml: {YAML_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
