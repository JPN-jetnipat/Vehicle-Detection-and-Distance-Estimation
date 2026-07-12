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

TRAIN_OUTPUT = BASE_DIR / "labels_bdd" / "train"
VAL_OUTPUT = BASE_DIR / "labels_bdd" / "val"

TRAIN_OUTPUT.mkdir(parents=True, exist_ok=True)
VAL_OUTPUT.mkdir(parents=True, exist_ok=True)

IMG_W = 1280
IMG_H = 720

# =====================================================
# Class Mapping
# BDD100K -> KITTI
# =====================================================

CLASS_MAP = {
    "car": 0,      # Car
    "bus": 1,      # Van
    "truck": 2     # Truck
}

# =====================================================
# Conversion Function
# =====================================================

def convert_json(json_path, output_dir):

    print("\n===================================")
    print(f"Converting: {json_path.name}")
    print("===================================")

    if not json_path.exists():
        raise FileNotFoundError(
            f"\nCould not find:\n{json_path}"
        )

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Loaded {len(data)} images")

    image_count = 0
    box_count = 0

    for item in data:

        image_name = item["name"]

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

            x_center = ((x1 + x2) / 2.0) / IMG_W
            y_center = ((y1 + y2) / 2.0) / IMG_H

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
# Run Conversion
# =====================================================

convert_json(TRAIN_JSON, TRAIN_OUTPUT)
convert_json(VAL_JSON, VAL_OUTPUT)

print("\n===================================")
print("BDD100K -> YOLO conversion complete")
print("===================================")