"""
count_weather.py

Count how many images of each BDD100K weather type are in your existing split.
Run from the folder that contains your split (bdd100k_split_handoff).

Usage:
    python count_weather.py
"""

import glob
import json
import os
import sys
from collections import Counter

# ---------------------------- CONFIG ----------------------------
# Folder of images, OR a .txt file with one image path per line.
# Use raw strings (r"...") for Windows paths.
SPLIT_SRC = r"C:\path\to\your\split\images\test"

# BDD100K label files that hold the weather attribute.
LABEL_JSONS = [
    r"C:\path\to\bdd100k\labels\bdd100k_labels_images_val.json",
    r"C:\path\to\bdd100k\labels\bdd100k_labels_images_train.json",
]
# ----------------------------------------------------------------

IMG_EXTS = ("*.jpg", "*.jpeg", "*.png")


def load_images(src):
    if os.path.isdir(src):
        files = []
        for ext in IMG_EXTS:
            files += glob.glob(os.path.join(src, ext))
        return files
    if os.path.isfile(src):
        with open(src, encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]
    return []


def main():
    weather_of = {}
    for jp in LABEL_JSONS:
        if not os.path.exists(jp):
            print(f"[skip] not found: {jp}")
            continue
        with open(jp, encoding="utf-8") as f:
            for r in json.load(f):
                weather_of[r["name"]] = (r.get("attributes") or {}).get("weather", "undefined")
        print(f"[ok] loaded {jp}")

    if not weather_of:
        print("ERROR: no label json loaded. Fix LABEL_JSONS.")
        sys.exit(1)

    imgs = load_images(SPLIT_SRC)
    if not imgs:
        print(f"ERROR: no images found at {SPLIT_SRC}")
        sys.exit(1)

    counts = Counter(weather_of.get(os.path.basename(p), "NOT_FOUND") for p in imgs)

    print(f"\nTotal images in split: {len(imgs)}\n")
    print(f"{'weather':<16}{'count':>8}{'percent':>10}")
    for w, n in counts.most_common():
        print(f"{w:<16}{n:>8}{n / len(imgs) * 100:>9.1f}%")


if __name__ == "__main__":
    main()
