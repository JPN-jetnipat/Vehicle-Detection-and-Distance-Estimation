from pathlib import Path

for split in ["train", "val", "test"]:

    image_count = len(
        list(
            Path(f"dataset/images/{split}").glob("*.png")
        )
    )

    label_count = len(
        list(
            Path(f"dataset/labels/{split}").glob("*.txt")
        )
    )

    print(
        f"{split}: "
        f"{image_count} images, "
        f"{label_count} labels"
    )