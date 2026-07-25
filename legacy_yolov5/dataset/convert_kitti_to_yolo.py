# TODO: Convert KITTI labels to YOLO format
from pathlib import Path
from PIL import Image
from collections import Counter

# ==========================
# CONFIG
# ==========================

IMAGE_DIR = Path(
    "dataset/raw/KITTI/data_object_image_2/training/image_2"
)

LABEL_DIR = Path(
    "dataset/raw/KITTI/data_object_label_2/training/label_2"
)

# counter = Counter()

# for file in LABEL_DIR.glob("*.txt"):
#     with open(file, "r") as f:
#         for line in f:
#             cls = line.split()[0]
#             counter[cls] += 1

# print(counter)

OUTPUT_LABEL_DIR = Path("dataset/labels_all")

OUTPUT_LABEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# ==========================
# CLASS MAPPING
# ==========================

CLASS_MAP = {
    "Car": 0,
    "Van": 1,
    "Truck": 2,
    "Pedestrian": 3,
    "Person_sitting": 4,
    "Cyclist": 5,
    "Tram": 6,
    "Misc": 7
}

print(CLASS_MAP)

# ==========================
# CONVERT FUNCTION
# ==========================

def convert_bbox_to_yolo(
    x1,
    y1,
    x2,
    y2,
    img_w,
    img_h
):
    bbox_width = x2 - x1
    bbox_height = y2 - y1

    center_x = x1 + bbox_width / 2
    center_y = y1 + bbox_height / 2

    center_x /= img_w
    center_y /= img_h

    bbox_width /= img_w
    bbox_height /= img_h

    return (
        center_x,
        center_y,
        bbox_width,
        bbox_height
    )

# ==========================
# MAIN LOOP
# ==========================

label_files = list(
    LABEL_DIR.glob("*.txt")
)

print(
    f"Found {len(label_files)} KITTI labels"
)

for label_file in label_files:

    image_file = (
        IMAGE_DIR /
        f"{label_file.stem}.png"
    )

    if not image_file.exists():
        continue

    img = Image.open(image_file)

    img_w, img_h = img.size

    yolo_lines = []

    with open(label_file, "r") as f:

        lines = f.readlines()

    for line in lines:

        parts = line.strip().split()

        if len(parts) < 15:
            continue

        obj_class = parts[0]

        if obj_class == "DontCare":
            continue

        if obj_class not in CLASS_MAP:
            print(f"Unknown class: {obj_class}")
            continue

        class_id = CLASS_MAP[obj_class]

        x1 = float(parts[4])
        y1 = float(parts[5])
        x2 = float(parts[6])
        y2 = float(parts[7])

        (
            cx,
            cy,
            bw,
            bh
        ) = convert_bbox_to_yolo(
            x1,
            y1,
            x2,
            y2,
            img_w,
            img_h
        )

        yolo_lines.append(
            f"{class_id} "
            f"{cx:.6f} "
            f"{cy:.6f} "
            f"{bw:.6f} "
            f"{bh:.6f}"
        )

    output_file = (
        OUTPUT_LABEL_DIR /
        f"{label_file.stem}.txt"
    )

    with open(output_file, "w") as f:

        f.write(
            "\n".join(yolo_lines)
        )

print(
    "Conversion Complete"
)