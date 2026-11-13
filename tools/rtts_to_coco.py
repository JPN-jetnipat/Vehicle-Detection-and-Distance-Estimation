#!/usr/bin/env python3
"""Convert RTTS (RESIDE-beta, real fog) into an EVAL-ONLY vehicle test set.

Same role and output layout as tools/exdark_to_coco.py: never trained or tuned on,
scored zero-shot with evaluation/score_exdark.py (the scorer is dataset-agnostic:
pass --gt / --images-dir / --slices-dir / --out).

Input (raw download, outside the repo, e.g. ~/datasets/RTTS/RTTS/):
  Annotations/<id>.xml        VOC 2007 boxes: <object><name>car</name><difficult>0</difficult>
                              <bndbox><xmin>..</xmin><ymin>..</ymin><xmax>..</xmax><ymax>..</ymax>
  JPEGImages/<id>.png|.jpg
  ImageSets/Main/test.txt     image list

Output:
  dataset/rtts/images/<image>     flat symlinks to the raw files
  dataset/rtts/gt_vehicle5.json   COCO GT, real per-image width/height, BDD ids
                                  (car 0, truck 1, bus 2, motor 3, bike 4)
  dataset/rtts/meta.csv           one row per image: size, counts, status
  splits/rtts/*.txt               eval pool + slices

Class mapping (RTTS -> ours). RTTS has NO truck class:
  car -> car(0) · bus -> bus(2) · motorbike -> motor(3) · bicycle -> bike(4) · person -> ignored
  Score with --truck-as-car (default), same as ExDark.

VOC "difficult" boxes are kept with iscrowd=1, so COCOeval ignores them (a missed
difficult box is not a miss, a detection on one is not a false positive), which is
the VOC convention. Their count is printed.

Coordinates: VOC is 1-based inclusive; converted to 0-based x, y, w, h and clipped.

Images are EXCLUDED (status in meta.csv) if OpenCV's decoded size differs from the
raw size (EXIF rotation) or the image/annotation is missing/unreadable.

Usage (server, ~/vehicle-detection, venv active):
  python tools/rtts_to_coco.py --raw ~/datasets/RTTS/RTTS
"""
import argparse
import csv
import json
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import cv2
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]

OUR_NAMES = ["car", "truck", "bus", "motor", "bike"]
RTTS_TO_OURS = {"car": 0, "bus": 2, "motorbike": 3, "motorcycle": 3, "bicycle": 4}


def read_boxes(xml_path, W, H):
    out, ignored, clipped, difficult = [], Counter(), 0, 0
    root = ET.parse(xml_path).getroot()
    for obj in root.findall("object"):
        name = (obj.findtext("name") or "").strip().lower()
        if name not in RTTS_TO_OURS:
            ignored[name] += 1
            continue
        bb = obj.find("bndbox")
        x0, y0, x1, y1 = (float(bb.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax"))
        x0, y0 = x0 - 1, y0 - 1                       # VOC 1-based inclusive -> 0-based
        cx0, cy0, cx1, cy1 = max(0.0, x0), max(0.0, y0), min(float(W), x1), min(float(H), y1)
        if (cx0, cy0, cx1, cy1) != (x0, y0, x1, y1):
            clipped += 1
        if cx1 - cx0 <= 1 or cy1 - cy0 <= 1:
            continue
        d = int(float(obj.findtext("difficult") or 0)) == 1
        difficult += d
        out.append((RTTS_TO_OURS[name], [cx0, cy0, cx1 - cx0, cy1 - cy0], d))
    return out, ignored, clipped, difficult


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", required=True, help="folder holding Annotations/, JPEGImages/, ImageSets/")
    ap.add_argument("--out-dir", default="dataset/rtts")
    ap.add_argument("--splits-dir", default="splits/rtts")
    args = ap.parse_args()

    raw = Path(args.raw).expanduser()
    imgs = {p.stem: p for p in (raw / "JPEGImages").iterdir() if p.is_file() and not p.name.startswith(".")}
    anns = {p.stem: p for p in (raw / "Annotations").glob("*.xml")}
    lst = raw / "ImageSets" / "Main" / "test.txt"
    ids = [l.strip() for l in lst.read_text().splitlines() if l.strip()] if lst.exists() else sorted(anns)
    print(f"list {len(ids)} ids · images {len(imgs)} · annotations {len(anns)}")

    out_dir = ROOT / args.out_dir
    img_out = out_dir / "images"
    img_out.mkdir(parents=True, exist_ok=True)

    images, coco_anns, rows, aid = [], [], [], 1
    status, ignored_total, clipped_total, difficult_total = Counter(), Counter(), 0, 0
    for i in ids:
        r = {"id": i, "width": "", "height": "", "n_car": 0, "n_bus": 0, "n_motor": 0, "n_bike": 0,
             "n_difficult": 0, "eval_name": ""}
        src, ann = imgs.get(i), anns.get(i)
        if src is None or ann is None:
            r["status"] = "missing_image" if src is None else "missing_annotation"
            status[r["status"]] += 1; rows.append(r); continue
        W, H = Image.open(src).size
        im = cv2.imread(str(src))
        if im is None:
            r["status"] = "unreadable"; status["unreadable"] += 1; rows.append(r); continue
        if im.shape[:2] != (H, W):
            r["status"] = "exif_rotated"; status["exif_rotated"] += 1; rows.append(r); continue
        link = img_out / src.name
        if not link.exists():
            link.symlink_to(src.resolve())
        boxes, ign, clp, dif = read_boxes(ann, W, H)
        ignored_total.update(ign); clipped_total += clp; difficult_total += dif
        iid = len(images) + 1
        images.append({"id": iid, "file_name": src.name, "width": W, "height": H})
        for k, b, d in boxes:
            coco_anns.append({"id": aid, "image_id": iid, "category_id": k, "bbox": b,
                              "area": b[2] * b[3], "iscrowd": int(d)})
            aid += 1
            if not d:
                r[f"n_{OUR_NAMES[k]}"] += 1
        r.update(width=W, height=H, n_difficult=dif, eval_name=src.name, status="ok")
        status["ok"] += 1; rows.append(r)

    gt = {"images": images, "annotations": coco_anns,
          "categories": [{"id": i, "name": n} for i, n in enumerate(OUR_NAMES)]}
    (out_dir / "gt_vehicle5.json").write_text(json.dumps(gt))
    with open(out_dir / "meta.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    ok = [r for r in rows if r["status"] == "ok"]
    has_veh = [r for r in ok if r["n_car"] + r["n_bus"] + r["n_motor"] + r["n_bike"] > 0]
    sdir = ROOT / args.splits_dir
    sdir.mkdir(parents=True, exist_ok=True)
    for s, rs in {"all": ok, "has_vehicle": has_veh}.items():
        (sdir / f"{s}.txt").write_text("".join(r["eval_name"] + "\n" for r in rs))

    inst = Counter(a["category_id"] for a in coco_anns if not a["iscrowd"])
    print(f"\nstatus: {dict(status)}")
    print("vehicle instances (not difficult): " + ", ".join(f"{OUR_NAMES[k]} {inst.get(k, 0)}" for k in range(5)))
    print(f"difficult vehicle boxes (ignored by COCOeval): {difficult_total}")
    print(f"ignored non-vehicle boxes: {sum(ignored_total.values())} {dict(ignored_total)}")
    print(f"boxes clipped to image bounds: {clipped_total}")
    print(f"slices: all {len(ok)}, has_vehicle {len(has_veh)}")
    sizes = Counter((r["width"], r["height"]) for r in ok)
    print(f"distinct image sizes: {len(sizes)}, most common: {sizes.most_common(3)}")
    print(f"-> {out_dir / 'gt_vehicle5.json'}, {out_dir / 'meta.csv'}, {sdir}/")


if __name__ == "__main__":
    main()
