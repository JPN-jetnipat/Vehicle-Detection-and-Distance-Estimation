import json
import shutil
from pathlib import Path

# =====================================================
# PATHS
# =====================================================

PROJECT_ROOT = Path(r"D:\Vehicle Detection and Distance Estimation\vehicle_distance_project")

BDD10K_IMAGES = PROJECT_ROOT / "dataset" / "bdd10k_custom" / "images"
BDD10K_LABELS = PROJECT_ROOT / "dataset" / "bdd10k_custom" / "labels"

TRAIN_JSON = (
    PROJECT_ROOT
    / "dataset"
    / "raw"
    / "BDD100K"
    / "bdd100k_labels_release"
    / "bdd100k"
    / "labels"
    / "bdd100k_labels_images_train.json"
)

VAL_JSON = (
    PROJECT_ROOT
    / "dataset"
    / "raw"
    / "BDD100K"
    / "bdd100k_labels_release"
    / "bdd100k"
    / "labels"
    / "bdd100k_labels_images_val.json"
)

OUTPUT_DAY = PROJECT_ROOT / "dataset" / "bdd10k_day"
OUTPUT_NIGHT = PROJECT_ROOT / "dataset" / "bdd10k_night"

# =====================================================
# CREATE FOLDERS
# =====================================================

for root in [OUTPUT_DAY, OUTPUT_NIGHT]:
    (root / "images" / "train").mkdir(parents=True, exist_ok=True)
    (root / "images" / "val").mkdir(parents=True, exist_ok=True)
    (root / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (root / "labels" / "val").mkdir(parents=True, exist_ok=True)

# =====================================================
# LOAD METADATA
# =====================================================

def load_timeofday_map(json_path):

    with open(json_path, "r") as f:
        data = json.load(f)

    result = {}

    for item in data:

        filename = item["name"]

        timeofday = item["attributes"].get("timeofday", "unknown")

        result[filename] = timeofday

    return result

train_map = load_timeofday_map(TRAIN_JSON)
val_map = load_timeofday_map(VAL_JSON)

# =====================================================
# COPY FILES
# =====================================================

def process_split(split_name, meta):

    img_dir = BDD10K_IMAGES / split_name
    lbl_dir = BDD10K_LABELS / split_name

    copied_day = 0
    copied_night = 0

    for img_path in img_dir.glob("*.jpg"):

        name = img_path.name

        if name not in meta:
            continue

        timeofday = meta[name]

        label_path = lbl_dir / f"{img_path.stem}.txt"

        if not label_path.exists():
            continue

        if timeofday == "daytime":

            shutil.copy2(
                img_path,
                OUTPUT_DAY / "images" / split_name / name
            )

            shutil.copy2(
                label_path,
                OUTPUT_DAY / "labels" / split_name / label_path.name
            )

            copied_day += 1

        elif timeofday == "night":

            shutil.copy2(
                img_path,
                OUTPUT_NIGHT / "images" / split_name / name
            )

            shutil.copy2(
                label_path,
                OUTPUT_NIGHT / "labels" / split_name / label_path.name
            )

            copied_night += 1

    print(
        f"{split_name}: "
        f"day={copied_day}, "
        f"night={copied_night}"
    )

# =====================================================
# RUN
# =====================================================

process_split("train", train_map)
process_split("val", val_map)

print("Done.")