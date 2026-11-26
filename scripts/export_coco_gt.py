"""Export the BDD100K test splits' ground truth as COCO-format JSON files to share with the team.

One file per split, written to results/coco_gt/test_<split>.json, plus
test_all.json (the full 8,841-image test pool = day + night + dawndusk).

The JSON is built with exactly the same rules as build_coco() in
scripts/eval_arm_coco.py, so a prediction file scored against it gives the
same AP our own eval reports:
  - images sorted by file name, image_id = 1, 2, 3, ... in that order
  - category_id is 0-based: 0 car, 1 truck, 2 bus, 3 motor, 4 bike
  - every BDD image is 1280x720; bbox = [x, y, w, h] in pixels (top-left corner)

Images are not needed - the file names come from the label files
(<stem>.txt -> <stem>.jpg), so this also runs on a laptop that only has
dataset/yolo/labels. If ultralytics/pycocotools are installed (the server),
pass --verify to cross-check every split against eval_arm_coco.build_coco().

Usage (from repo root):
    python scripts/export_coco_gt.py
    python scripts/export_coco_gt.py --verify      # on the server
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LABELS_ROOT = ROOT / "dataset" / "yolo" / "labels"
OUT_DIR = ROOT / "results" / "coco_gt"

NAMES = ["car", "truck", "bus", "motor", "bike"]  # must match dataset/bdd100k_vehicle.yaml
IMG_WIDTH, IMG_HEIGHT = 1280, 720
SPLITS = ["day", "night", "dawndusk", "clear", "overcast", "partlycloudy", "rainy", "snowy", "foggy"]


def yolo_line_to_xywh(parts: list[str], W: int, H: int) -> list[float]:
    cx, cy, w, h = (float(v) for v in parts[1:5])
    return [(cx - w / 2) * W, (cy - h / 2) * H, w * W, h * H]


def build_coco_dict(label_files: list[Path]) -> dict:
    """label_files may come from several dirs (test_all); image names must be unique."""
    by_name = {f.stem + ".jpg": f for f in label_files}
    if len(by_name) != len(label_files):
        raise ValueError("duplicate image names across label dirs")
    images, anns = [], []
    aid = 1
    for iid, name in enumerate(sorted(by_name), start=1):
        images.append({"id": iid, "file_name": name, "width": IMG_WIDTH, "height": IMG_HEIGHT})
        for line in by_name[name].read_text().splitlines():
            p = line.split()
            if not p:
                continue
            box = yolo_line_to_xywh(p, IMG_WIDTH, IMG_HEIGHT)
            anns.append({"id": aid, "image_id": iid, "category_id": int(p[0]),
                         "bbox": box, "area": box[2] * box[3], "iscrowd": 0})
            aid += 1
    return {"images": images, "annotations": anns,
            "categories": [{"id": i, "name": n} for i, n in enumerate(NAMES)]}


def verify(split: str, coco: dict) -> None:
    """Compare against the real eval_arm_coco.build_coco() (needs ultralytics + pycocotools)."""
    import sys
    sys.path.insert(0, str(ROOT / "scripts"))
    import eval_arm_coco

    names = eval_arm_coco.load_names(ROOT / "dataset" / f"bdd100k_test_{split}.yaml")
    assert names == NAMES, f"class list differs: {names}"
    image_names = [im["file_name"] for im in coco["images"]]
    ref, name_to_id = eval_arm_coco.build_coco(image_names, LABELS_ROOT / f"test_{split}", names,
                                               eval_arm_coco.IMG_WIDTH, eval_arm_coco.IMG_HEIGHT)
    assert all(name_to_id[im["file_name"]] == im["id"] for im in coco["images"]), "image_id mismatch"
    assert ref.dataset["annotations"] == coco["annotations"], "annotation mismatch"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--verify", action="store_true", help="cross-check against eval_arm_coco.build_coco()")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    jobs = {f"test_{s}": sorted((LABELS_ROOT / f"test_{s}").glob("*.txt")) for s in SPLITS}
    jobs["test_all"] = sorted(jobs["test_day"] + jobs["test_night"] + jobs["test_dawndusk"])

    print(f"{'split':<18}{'images':>8}{'boxes':>8}" + "".join(f"{n:>7}" for n in NAMES))
    for split, files in jobs.items():
        if not files:
            raise SystemExit(f"no label files for {split} under {LABELS_ROOT}")
        coco = build_coco_dict(files)
        if args.verify and split != "test_all":
            verify(split.removeprefix("test_"), coco)
        (args.out_dir / f"{split}.json").write_text(json.dumps(coco))
        counts = [sum(a["category_id"] == i for a in coco["annotations"]) for i in range(len(NAMES))]
        print(f"{split:<18}{len(coco['images']):>8}{len(coco['annotations']):>8}" + "".join(f"{c:>7}" for c in counts))
    print(f"\nwrote {len(jobs)} files to {args.out_dir}" + ("  (verified against eval_arm_coco)" if args.verify else ""))


if __name__ == "__main__":
    main()
