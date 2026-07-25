import json
from pathlib import Path

# =====================================================
# Paths
# =====================================================

BASE_DIR = Path(__file__).resolve().parent

ROOT = BASE_DIR / "raw" / "BDD100K"

TRAIN_JSON = (
    ROOT
    / "bdd100k_labels_release"
    / "bdd100k"
    / "labels"
    / "bdd100k_labels_images_train.json"
)

VAL_JSON = (
    ROOT
    / "bdd100k_labels_release"
    / "bdd100k"
    / "labels"
    / "bdd100k_labels_images_val.json"
)

BDD10K_TRAIN_IMAGES = (
    ROOT
    / "bdd100k"
    / "bdd100k"
    / "images"
    / "10k"
    / "train"
)

BDD10K_VAL_IMAGES = (
    ROOT
    / "bdd100k"
    / "bdd100k"
    / "images"
    / "10k"
    / "val"
)

TRAIN_OUTPUT = BASE_DIR / "labels_bdd10k" / "train"
VAL_OUTPUT = BASE_DIR / "labels_bdd10k" / "val"

TRAIN_OUTPUT.mkdir(parents=True, exist_ok=True)
VAL_OUTPUT.mkdir(parents=True, exist_ok=True)

IMG_W = 1280
IMG_H = 720

# =====================================================
# Class Mapping
# =====================================================

CLASS_MAP = {
    "car": 0,
    "bus": 1,
    "truck": 2
}

# =====================================================
# Build image whitelist
# =====================================================

train_images = {p.name for p in BDD10K_TRAIN_IMAGES.glob("*.jpg")}
val_images = {p.name for p in BDD10K_VAL_IMAGES.glob("*.jpg")}

print(f"BDD10K train images: {len(train_images)}")
print(f"BDD10K val images  : {len(val_images)}")

# =====================================================
# Converter
# =====================================================

def convert_json(json_path, image_whitelist, output_dir):

    print("\n===================================")
    print(f"Converting {json_path.name}")
    print("===================================")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    image_count = 0
    box_count = 0

    for item in data:

        image_name = item["name"]

        if image_name not in image_whitelist:
            continue

        txt_path = output_dir / image_name.replace(".jpg", ".txt")

        lines = []

        for obj in item.get("labels", []):

            category = obj.get("category", "").lower()

            if category not in CLASS_MAP:
                continue

            if "box2d" not in obj:
                continue

            box = obj["box2d"]

            x1 = float(box["x1"])
            y1 = float(box["y1"])
            x2 = float(box["x2"])
            y2 = float(box["y2"])

            x_center = ((x1 + x2) / 2) / IMG_W
            y_center = ((y1 + y2) / 2) / IMG_H

            width = (x2 - x1) / IMG_W
            height = (y2 - y1) / IMG_H

            cls = CLASS_MAP[category]

            lines.append(
                f"{cls} "
                f"{x_center:.6f} "
                f"{y_center:.6f} "
                f"{width:.6f} "
                f"{height:.6f}"
            )

            box_count += 1

        txt_path.write_text("\n".join(lines))

        image_count += 1

    print(f"Images processed : {image_count}")
    print(f"Boxes converted  : {box_count}")
    print(f"Output folder    : {output_dir}")

# =====================================================
# Run
# =====================================================

convert_json(
    TRAIN_JSON,
    train_images,
    TRAIN_OUTPUT
)

convert_json(
    VAL_JSON,
    val_images,
    VAL_OUTPUT
)

print("\nBDD10K conversion complete.")