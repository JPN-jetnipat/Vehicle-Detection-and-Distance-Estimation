"""
split_by_weather.py

Split the existing TEST set (splits/test.txt) by weather, using
dataset/yolo/attr_index.csv. No image outside test.txt is ever used.

Outputs (in ./weather_splits):
    raw_<weather>.txt / .yaml   one list + one data yaml per weather
    raw_all.txt / .yaml         the whole test set (sanity baseline)
    enh_*                       same lists for the Set1 images (only if ENH_REPLACE is set)

Run from anywhere:  python split_by_weather.py
"""

import csv
import os
from collections import Counter, defaultdict
from pathlib import Path

import yaml

# ---------------------------- CONFIG ----------------------------
ROOT = Path(__file__).resolve().parent            # bdd100k_split_handoff
TEST_LIST = ROOT / "splits" / "test.txt"
ATTR_CSV = ROOT / "dataset" / "yolo" / "attr_index.csv"
DATA_YAML = ROOT / "configs" / "data" / "bdd100k_vehicle5.yaml"
OUT_DIR = ROOT / "weather_splits"

# Set1 (low-light compensation) images. Set to None if not needed yet.
# Format: (text in the raw image path, replacement text), e.g.
#   (r"dataset\yolo\images", r"dataset\set1\images")
# The matching labels folder must exist next to the enhanced images folder.
ENH_REPLACE = None
# ----------------------------------------------------------------


def load_attrs(csv_path):
    attrs = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            attrs[row["name"]] = (row["timeofday"], row["weather"])
    return attrs


def load_names(yaml_path):
    cfg = yaml.safe_load(open(yaml_path, encoding="utf-8"))
    names = cfg["names"]
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]
    return names


def build_image_index():
    """filename -> full path, searched in dataset/yolo/images/test_* folders."""
    index = {}
    for folder in sorted((ROOT / "dataset" / "yolo" / "images").glob("test_*")):
        for p in folder.iterdir():
            if p.suffix.lower() in (".jpg", ".jpeg", ".png"):
                index[p.name] = p
    return index


def read_list(txt_path):
    """test.txt may hold bare filenames; resolve them to full paths."""
    index = build_image_index()
    print(f"images found in dataset/yolo/images/test_*: {len(index)}")
    paths = []
    for line in open(txt_path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        p = Path(line)
        if p.parent == Path("."):            # bare filename
            p = index.get(p.name, ROOT / p.name)
        elif not p.is_absolute():
            p = ROOT / p
        paths.append(p)
    return paths


def write_split(tag, weather, paths, names):
    key = weather.replace("/", "-").replace(" ", "_")
    txt = OUT_DIR / f"{tag}_{key}.txt"
    txt.write_text("\n".join(str(p) for p in paths), encoding="utf-8")
    yml = OUT_DIR / f"{tag}_{key}.yaml"
    yml.write_text(
        yaml.safe_dump(
            {
                "path": str(OUT_DIR.as_posix()),
                "train": txt.as_posix(),   # dummy, unused during val
                "val": txt.as_posix(),
                "nc": len(names),
                "names": names,
            },
            allow_unicode=True,
        ),
        encoding="utf-8",
    )


def main():
    OUT_DIR.mkdir(exist_ok=True)
    attrs = load_attrs(ATTR_CSV)
    names = load_names(DATA_YAML)
    test_imgs = read_list(TEST_LIST)
    print(f"attr_index rows : {len(attrs)}")
    print(f"test.txt images : {len(test_imgs)}")
    print(f"class names     : {names}")

    missing_file = [p for p in test_imgs if not p.exists()]
    print(f"missing on disk : {len(missing_file)}")
    if missing_file:
        print("  first missing:", missing_file[:3])

    groups = defaultdict(list)
    cross = Counter()
    unmatched = 0
    for p in test_imgs:
        a = attrs.get(p.name)
        if a is None:
            unmatched += 1
            continue
        tod, weather = a
        groups[weather].append(p)
        cross[(weather, tod)] += 1
    print(f"unmatched names : {unmatched}\n")

    # ---- counts ----
    tods = sorted({t for _, t in cross})
    print(f"{'weather':<14}{'total':>8}" + "".join(f"{t:>12}" for t in tods))
    for w, ps in sorted(groups.items(), key=lambda x: -len(x[1])):
        print(f"{w:<14}{len(ps):>8}" + "".join(f"{cross[(w, t)]:>12}" for t in tods))
    print(f"{'TOTAL':<14}{sum(len(v) for v in groups.values()):>8}")

    # ---- write splits ----
    write_split("raw", "all", test_imgs, names)
    for w, ps in groups.items():
        write_split("raw", w, ps, names)

    if ENH_REPLACE is not None:
        old, new = ENH_REPLACE
        enh_all = [Path(str(p).replace(old, new)) for p in test_imgs]
        bad = [p for p in enh_all if not p.exists()]
        print(f"\nenhanced images missing: {len(bad)}")
        write_split("enh", "all", enh_all, names)
        for w, ps in groups.items():
            write_split("enh", w, [Path(str(p).replace(old, new)) for p in ps], names)

    print(f"\nWritten to {OUT_DIR}")


if __name__ == "__main__":
    main()