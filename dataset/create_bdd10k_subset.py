from pathlib import Path
import random
import shutil

random.seed(42)

BASE = Path(__file__).resolve().parent

# Source (verified BDD100K)
SRC_TRAIN_IMAGES = BASE / "raw" / "BDD100K" / "bdd100k" / "bdd100k" / "images" / "100k" / "train"
SRC_VAL_IMAGES   = BASE / "raw" / "BDD100K" / "bdd100k" / "bdd100k" / "images" / "100k" / "val"

SRC_TRAIN_LABELS = BASE / "labels_bdd" / "train"
SRC_VAL_LABELS   = BASE / "labels_bdd" / "val"

# Destination (custom BDD10K)
DST_ROOT = BASE / "bdd10k_custom"

DST_TRAIN_IMAGES = DST_ROOT / "images" / "train"
DST_VAL_IMAGES   = DST_ROOT / "images" / "val"

DST_TRAIN_LABELS = DST_ROOT / "labels" / "train"
DST_VAL_LABELS   = DST_ROOT / "labels" / "val"

for d in [
    DST_TRAIN_IMAGES,
    DST_VAL_IMAGES,
    DST_TRAIN_LABELS,
    DST_VAL_LABELS,
]:
    d.mkdir(parents=True, exist_ok=True)

TRAIN_COUNT = 7000
VAL_COUNT = 1000


def build_subset(src_images, src_labels, dst_images, dst_labels, count):

    images = list(src_images.rglob("*.jpg"))

    print(f"\nFound {len(images)} images in {src_images}")

    selected = random.sample(images, count)

    copied = 0

    for img in selected:

        label = src_labels / f"{img.stem}.txt"

        if not label.exists():
            continue

        shutil.copy2(img, dst_images / img.name)
        shutil.copy2(label, dst_labels / label.name)

        copied += 1

    print(f"Copied {copied} image/label pairs")


build_subset(
    SRC_TRAIN_IMAGES,
    SRC_TRAIN_LABELS,
    DST_TRAIN_IMAGES,
    DST_TRAIN_LABELS,
    TRAIN_COUNT,
)

build_subset(
    SRC_VAL_IMAGES,
    SRC_VAL_LABELS,
    DST_VAL_IMAGES,
    DST_VAL_LABELS,
    VAL_COUNT,
)

print("\nBDD10K custom subset created successfully.")