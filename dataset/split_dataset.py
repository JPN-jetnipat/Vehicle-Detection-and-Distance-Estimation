# TODO: Train/Val/Test split

from pathlib import Path
import random
import shutil

random.seed(42)

IMAGE_DIR = Path(
    "dataset/raw/KITTI/data_object_image_2/training/image_2"
)

LABEL_DIR = Path(
    "dataset/labels_all"
)

OUTPUT_IMAGES = Path("dataset/images")
OUTPUT_LABELS = Path("dataset/labels")

for split in ["train", "val", "test"]:
    (OUTPUT_IMAGES / split).mkdir(parents=True, exist_ok=True)
    (OUTPUT_LABELS / split).mkdir(parents=True, exist_ok=True)

image_files = sorted(IMAGE_DIR.glob("*.png"))

print(f"Total Images: {len(image_files)}")

random.shuffle(image_files)

n = len(image_files)

# train : validate : test = 70:15:15
train_end = int(0.7 * n)
val_end = int(0.85 * n)

train_files = image_files[:train_end]
val_files = image_files[train_end:val_end]
test_files = image_files[val_end:]

# train_files = image_files[0:5236]
# val_files = image_files[5236:6358]
# test_files = image_files[6358:7481]

splits = {
    "train": train_files,
    "val": val_files,
    "test": test_files
}

for split_name, files in splits.items():

    print(split_name, len(files))

    for img_file in files:

        label_file = LABEL_DIR / f"{img_file.stem}.txt"

        shutil.copy(
            img_file,
            OUTPUT_IMAGES / split_name / img_file.name
        )

        if label_file.exists():

            shutil.copy(
                label_file,
                OUTPUT_LABELS / split_name / label_file.name
            )

print("Dataset Split Complete")