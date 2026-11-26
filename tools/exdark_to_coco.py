#!/usr/bin/env python3
"""Convert ExDark (Exclusively Dark) into an EVAL-ONLY vehicle test set.

ExDark is never trained or tuned on. It is an external, out-of-domain night test:
every BDD-trained arm is scored on it zero-shot, all arms in ONE
evaluation/score_exdark.py run (same discipline as score_slices.py).

Input (raw download, kept outside the repo, e.g. ~/datasets/ExDark/):
  ExDark/<Class>/<image>             12 class folders of images
  ExDark_Annno/<Class>/<image>.txt   bbGt v3 annotations: "Class l t w h ..." in pixels
  imageclasslist.txt                 name, class(1-12), light(1-10), in/out(1-2), split(1-3)

Output:
  dataset/exdark/images/<image>      flat symlinks to the raw files (or a .png
                                     conversion if OpenCV cannot read the original)
  dataset/exdark/gt_vehicle5.json    COCO GT, REAL per-image width/height,
                                     category ids = BDD ids (car 0, truck 1, bus 2,
                                     motor 3, bike 4)
  dataset/exdark/meta.csv            one row per image: attributes, sizes, counts, status
  splits/exdark/*.txt                eval pool + slices (image names only)

Class mapping (ExDark -> ours). ExDark has NO truck class:
  Car -> car(0) · Bus -> bus(2) · Motorbike -> motor(3) · Bicycle -> bike(4)
  Boat, Bottle, Cat, Chair, Cup, Dog, People, Table -> ignored
  Truck predictions are handled in the scorer (--truck-as-car, default on).

Images are EXCLUDED from the pool (and listed in meta.csv with the reason) if:
  - OpenCV's decoded size differs from the raw size. OpenCV applies EXIF
    rotation, the MATLAB annotation tool did not, so the boxes would be rotated
    relative to what the detector sees.
  - the image or its annotation file is missing / unreadable.

Usage (server, ~/vehicle-detection, venv active):
  python tools/exdark_to_coco.py --raw ~/datasets/ExDark
"""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import cv2
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]

OUR_NAMES = ["car", "truck", "bus", "motor", "bike"]
EXDARK_TO_OURS = {"car": 0, "bus": 2, "motorbike": 3, "bicycle": 4}
EXDARK_CLASSES = ["Bicycle", "Boat", "Bottle", "Bus", "Car", "Cat", "Chair", "Cup",
                  "Dog", "Motorbike", "People", "Table"]                       # codes 1..12
LIGHT = ["low", "ambient", "object", "single", "weak", "strong", "screen",
         "window", "shadow", "twilight"]                                      # codes 1..10
INOUT = ["indoor", "outdoor"]                                                 # codes 1..2
SPLIT = ["train", "val", "test"]                                              # codes 1..3


def index_files(root, strip_suffix=""):
    """lowercased basename (minus strip_suffix) -> path. Aborts on collisions."""
    idx = {}
    for p in root.rglob("*"):
        if not p.is_file() or p.name.startswith("."):
            continue
        key = p.name.lower()
        if strip_suffix and key.endswith(strip_suffix):
            key = key[: -len(strip_suffix)]
        if key in idx:
            raise SystemExit(f"FLAG: name collision {idx[key]} vs {p} - flat layout unsafe")
        idx[key] = p
    return idx


def read_classlist(path):
    rows = []
    for line in path.read_text(errors="replace").splitlines():
        p = line.split()
        if len(p) < 5 or not p[1].isdigit():
            continue                                   # header / blank
        rows.append({"name": p[0], "ex_class": EXDARK_CLASSES[int(p[1]) - 1],
                     "light": LIGHT[int(p[2]) - 1], "inout": INOUT[int(p[3]) - 1],
                     "official_split": SPLIT[int(p[4]) - 1]})
    return rows


def read_boxes(path, W, H):
    """bbGt v3 -> list of (our_class, [x, y, w, h]) clipped to the image."""
    out, ignored, clipped = [], Counter(), 0
    for line in path.read_text(errors="replace").splitlines():
        if not line.strip() or line.startswith("%"):
            continue
        p = line.split()
        cls = p[0].lower()
        if cls not in EXDARK_TO_OURS:
            ignored[p[0]] += 1
            continue
        l, t, w, h = (float(v) for v in p[1:5])
        x0, y0 = max(0.0, l), max(0.0, t)
        x1, y1 = min(float(W), l + w), min(float(H), t + h)
        if (x0, y0, x1, y1) != (l, t, l + w, t + h):
            clipped += 1
        if x1 - x0 <= 1 or y1 - y0 <= 1:
            continue
        out.append((EXDARK_TO_OURS[cls], [x0, y0, x1 - x0, y1 - y0]))
    return out, ignored, clipped


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", required=True, help="folder holding ExDark/, ExDark_Annno/, imageclasslist.txt")
    ap.add_argument("--out-dir", default="dataset/exdark")
    ap.add_argument("--splits-dir", default="splits/exdark")
    args = ap.parse_args()

    raw = Path(args.raw).expanduser()
    img_idx = index_files(raw / "ExDark")
    ann_idx = index_files(raw / "ExDark_Annno", strip_suffix=".txt")
    rows = read_classlist(raw / "imageclasslist.txt")
    print(f"classlist {len(rows)} rows · images found {len(img_idx)} · annotations {len(ann_idx)}")

    out_dir = ROOT / args.out_dir
    img_out = out_dir / "images"
    img_out.mkdir(parents=True, exist_ok=True)

    images, anns, aid = [], [], 1
    ignored_total, clipped_total, status = Counter(), 0, Counter()
    for r in rows:
        key = r["name"].lower()
        src, ann = img_idx.get(key), ann_idx.get(key)
        r.update(width="", height="", n_car=0, n_bus=0, n_motor=0, n_bike=0, eval_name="")
        if src is None or ann is None:
            r["status"] = "missing_image" if src is None else "missing_annotation"
            status[r["status"]] += 1
            continue
        W, H = Image.open(src).size                   # raw size, no EXIF rotation
        im = cv2.imread(str(src))
        name = src.name
        if im is None:                                # format OpenCV cannot read -> png copy
            name = src.stem + ".png"
            Image.open(src).convert("RGB").save(img_out / name)
            im = cv2.imread(str(img_out / name))
        if im is None:
            r["status"] = "unreadable"; status["unreadable"] += 1
            continue
        if im.shape[:2] != (H, W):
            r["status"] = "exif_rotated"; status["exif_rotated"] += 1
            continue
        link = img_out / name
        if not link.exists():
            link.symlink_to(src.resolve())

        boxes, ign, clp = read_boxes(ann, W, H)
        ignored_total.update(ign); clipped_total += clp
        iid = len(images) + 1
        images.append({"id": iid, "file_name": name, "width": W, "height": H})
        for k, b in boxes:
            anns.append({"id": aid, "image_id": iid, "category_id": k, "bbox": b,
                         "area": b[2] * b[3], "iscrowd": 0})
            aid += 1
            r[f"n_{OUR_NAMES[k]}"] += 1
        r.update(width=W, height=H, eval_name=name, status="ok")
        status["ok"] += 1

    gt = {"images": images, "annotations": anns,
          "categories": [{"id": i, "name": n} for i, n in enumerate(OUR_NAMES)]}
    (out_dir / "gt_vehicle5.json").write_text(json.dumps(gt))

    with open(out_dir / "meta.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    # ---- slices (eval pool = every status=ok image, vehicle or not) ----
    ok = [r for r in rows if r["status"] == "ok"]
    has_veh = lambda r: (r["n_car"] + r["n_bus"] + r["n_motor"] + r["n_bike"]) > 0
    slices = {"all": ok, "has_vehicle": [r for r in ok if has_veh(r)]}
    for v in INOUT:
        slices[v] = [r for r in ok if r["inout"] == v]
    slices["outdoor_vehicle"] = [r for r in slices["outdoor"] if has_veh(r)]
    for v in LIGHT:
        slices[f"light_{v}"] = [r for r in ok if r["light"] == v]
    for v in SPLIT:
        slices[f"official_{v}"] = [r for r in ok if r["official_split"] == v]

    sdir = ROOT / args.splits_dir
    sdir.mkdir(parents=True, exist_ok=True)
    for s, rs in slices.items():
        (sdir / f"{s}.txt").write_text("".join(r["eval_name"] + "\n" for r in rs))

    # ---- report ----
    inst = Counter(a["category_id"] for a in anns)
    print(f"\nstatus: {dict(status)}")
    print(f"vehicle instances: " + ", ".join(f"{OUR_NAMES[k]} {inst.get(k, 0)}" for k in range(5)))
    print(f"ignored non-vehicle boxes: {sum(ignored_total.values())} {dict(ignored_total)}")
    print(f"boxes clipped to image bounds: {clipped_total}")
    print("slices: " + ", ".join(f"{s} {len(rs)}" for s, rs in slices.items()))
    print(f"-> {out_dir / 'gt_vehicle5.json'}, {out_dir / 'meta.csv'}, {sdir}/")


if __name__ == "__main__":
    main()
