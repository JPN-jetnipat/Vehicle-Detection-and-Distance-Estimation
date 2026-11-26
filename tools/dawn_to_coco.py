#!/usr/bin/env python3
"""Convert DAWN (fog / rain / snow / sand) into an EVAL-ONLY vehicle test set.

Same role and output layout as tools/exdark_to_coco.py and tools/rtts_to_coco.py: never
trained or tuned on, scored zero-shot with evaluation/score_exdark.py
(--gt dataset/dawn/gt_vehicle5.json --images-dir dataset/dawn/images --slices-dir splits/dawn).

Input (raw download, outside the repo, e.g. ~/datasets/DAWN/DAWN/):
  <Cond>/<image>.jpg                       Cond in Fog, Rain, Sand, Snow (images anywhere under it)
  <Cond>/<Cond>_PASCAL_VOC/<image>.xml     VOC boxes with class names   <- used
  <Cond>/<Cond>_YOLO_darknet/<image>.txt   YOLO boxes (class ids only)  <- used only as a
                                           cross-check of the XML pixel convention

Class mapping (case-insensitive). DAWN HAS truck, so all 5 BDD classes are scored:
  car -> car(0) · truck -> truck(1) · bus -> bus(2) · motorcycle/motorbike/motor -> motor(3)
  · bicycle/bike -> bike(4) · person/pedestrian/... -> ignored. Unknown names are printed.
  Score DAWN with --no-truck-as-car (it has real truck labels).

Pixel convention: VOC is officially 1-based, but tools differ. The script compares each XML
box with the matching YOLO box and uses whichever convention (0- or 1-based) agrees better,
and prints both errors so the choice is visible.

Flat image names are "<cond>__<file>" so files from different conditions cannot collide.
Images are EXCLUDED (status in meta.csv) if OpenCV's decoded size differs from the raw size
(EXIF rotation), or the image/annotation is missing/unreadable.

Usage (server, ~/vehicle-detection, venv active):
  python tools/dawn_to_coco.py --raw ~/datasets/DAWN/DAWN
"""
import argparse
import csv
import json
import statistics
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import cv2
from PIL import Image
from ultralytics.data.utils import IMG_FORMATS

ROOT = Path(__file__).resolve().parents[1]

OUR_NAMES = ["car", "truck", "bus", "motor", "bike"]
NAME_MAP = {"car": 0, "cars": 0, "truck": 1, "trucks": 1, "bus": 2, "buses": 2,
            "motorcycle": 3, "motorbike": 3, "motor": 3, "motorcycles": 3,
            "bicycle": 4, "bike": 4, "bicycles": 4}
KNOWN_IGNORED = {"person", "people", "pedestrian", "rider", "cyclist"}
CONDS = ["fog", "rain", "sand", "snow"]


def parse_xml(p):
    root = ET.parse(p).getroot()
    objs = []
    for o in root.findall("object"):
        bb = o.find("bndbox")
        if bb is None:
            continue
        name = (o.findtext("name") or "").strip().lower()
        x0, y0, x1, y1 = (float(bb.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax"))
        objs.append((name, x0, y0, x1, y1))
    return objs


def parse_yolo(p, W, H):
    out = []
    for line in p.read_text(errors="replace").splitlines():
        q = line.split()
        if len(q) < 5:
            continue
        cx, cy, w, h = (float(v) for v in q[1:5])
        out.append(((cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", required=True, help="folder holding Fog/, Rain/, Sand/, Snow/")
    ap.add_argument("--out-dir", default="dataset/dawn")
    ap.add_argument("--splits-dir", default="splits/dawn")
    args = ap.parse_args()
    raw = Path(args.raw).expanduser()

    # ---- index files per condition ----
    items = []                                   # (cond, stem, img, xml, yolo)
    for cdir in sorted(p for p in raw.iterdir() if p.is_dir()):
        cond = cdir.name.lower()
        imgs, xmls, ymls = {}, {}, {}
        for p in cdir.rglob("*"):
            if not p.is_file() or p.name.startswith("."):
                continue
            ext = p.suffix.lower().lstrip(".")
            key = p.stem.lower()
            if ext == "xml":
                xmls[key] = p
            elif ext == "txt" and "yolo" in str(p.parent).lower():
                ymls[key] = p
            elif ext in ("jpg", "jpeg", "png", "bmp", "tif", "tiff", "webp", "gif"):
                imgs[key] = p
        for key in sorted(set(imgs) | set(xmls)):
            items.append((cond, key, imgs.get(key), xmls.get(key), ymls.get(key)))
        print(f"{cdir.name}: images {len(imgs)} · xml {len(xmls)} · yolo txt {len(ymls)}")

    out_dir = ROOT / args.out_dir
    img_out = out_dir / "images"
    img_out.mkdir(parents=True, exist_ok=True)

    # ---- pass 1: read everything, measure the XML pixel convention against YOLO ----
    parsed, err0, err1, status = [], [], [], Counter()
    for cond, key, img, xml, yml in items:
        rec = {"cond": cond, "key": key, "img": img, "status": "ok"}
        if img is None or xml is None:
            rec["status"] = "missing_image" if img is None else "missing_annotation"
            parsed.append(rec); continue
        W, H = Image.open(img).size
        name = f"{cond}__{img.name}"
        if img.suffix.lower().lstrip(".") not in IMG_FORMATS:
            name = f"{cond}__{img.stem}.png"
            Image.open(img).convert("RGB").save(img_out / name)
            dec = cv2.imread(str(img_out / name))
        else:
            dec = cv2.imread(str(img))
        if dec is None:
            rec["status"] = "unreadable"; parsed.append(rec); continue
        if dec.shape[:2] != (H, W):
            rec["status"] = "exif_rotated"; parsed.append(rec); continue
        objs = parse_xml(xml)
        if yml is not None:
            yb = parse_yolo(yml, W, H)
            if len(yb) == len(objs):
                for (_, x0, y0, x1, y1), (a0, b0, a1, b1) in zip(objs, yb):
                    err0.append(abs(x0 - a0) + abs(y0 - b0) + abs(x1 - a1) + abs(y1 - b1))
                    err1.append(abs(x0 - 1 - a0) + abs(y0 - 1 - b0) + abs(x1 - a1) + abs(y1 - b1))
        rec.update(W=W, H=H, name=name, objs=objs)
        parsed.append(rec)

    one_based = bool(err0) and statistics.median(err1) < statistics.median(err0)
    if err0:
        print(f"\nXML vs YOLO box check ({len(err0)} boxes, median sum of |corner diffs| px): "
              f"as 0-based {statistics.median(err0):.2f} · as 1-based {statistics.median(err1):.2f} "
              f"-> using {'1-based' if one_based else '0-based'}")
    else:
        print("\nXML vs YOLO box check: no comparable pairs, using 0-based")

    # ---- pass 2: build COCO ----
    images, anns, rows, aid = [], [], [], 1
    unknown, ignored, clipped = Counter(), Counter(), 0
    for rec in parsed:
        row = {"cond": rec["cond"], "file": rec["img"].name if rec["img"] else rec["key"],
               "width": rec.get("W", ""), "height": rec.get("H", ""),
               **{f"n_{n}": 0 for n in OUR_NAMES}, "eval_name": "", "status": rec["status"]}
        status[rec["status"]] += 1
        if rec["status"] != "ok":
            rows.append(row); continue
        W, H = rec["W"], rec["H"]
        link = img_out / rec["name"]
        if not link.exists():
            link.symlink_to(rec["img"].resolve())
        iid = len(images) + 1
        images.append({"id": iid, "file_name": rec["name"], "width": W, "height": H})
        for name, x0, y0, x1, y1 in rec["objs"]:
            if name not in NAME_MAP:
                (ignored if name in KNOWN_IGNORED else unknown)[name] += 1
                continue
            if one_based:
                x0, y0 = x0 - 1, y0 - 1
            c0, d0, c1, d1 = max(0.0, x0), max(0.0, y0), min(float(W), x1), min(float(H), y1)
            if (c0, d0, c1, d1) != (x0, y0, x1, y1):
                clipped += 1
            if c1 - c0 <= 1 or d1 - d0 <= 1:
                continue
            k = NAME_MAP[name]
            anns.append({"id": aid, "image_id": iid, "category_id": k,
                         "bbox": [c0, d0, c1 - c0, d1 - d0], "area": (c1 - c0) * (d1 - d0), "iscrowd": 0})
            aid += 1
            row[f"n_{OUR_NAMES[k]}"] += 1
        row["eval_name"] = rec["name"]
        rows.append(row)

    gt = {"images": images, "annotations": anns,
          "categories": [{"id": i, "name": n} for i, n in enumerate(OUR_NAMES)]}
    (out_dir / "gt_vehicle5.json").write_text(json.dumps(gt))
    with open(out_dir / "meta.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    ok = [r for r in rows if r["status"] == "ok"]
    veh = lambda r: sum(r[f"n_{n}"] for n in OUR_NAMES) > 0
    slices = {"all": ok, "has_vehicle": [r for r in ok if veh(r)]}
    for c in sorted({r["cond"] for r in ok}):
        slices[c] = [r for r in ok if r["cond"] == c]
    sdir = ROOT / args.splits_dir
    sdir.mkdir(parents=True, exist_ok=True)
    for s, rs in slices.items():
        (sdir / f"{s}.txt").write_text("".join(r["eval_name"] + "\n" for r in rs))

    inst = Counter(a["category_id"] for a in anns)
    print(f"\nstatus: {dict(status)}")
    print("vehicle instances: " + ", ".join(f"{OUR_NAMES[k]} {inst.get(k, 0)}" for k in range(5)))
    print(f"ignored (person etc.): {sum(ignored.values())} {dict(ignored)}")
    print(f"UNKNOWN class names (not scored - check these): {dict(unknown) or 'none'}")
    print(f"boxes clipped to image bounds: {clipped}")
    print("slices: " + ", ".join(f"{s} {len(rs)}" for s, rs in slices.items()))
    for c in sorted({r["cond"] for r in ok}):
        cnt = Counter()
        for r in slices[c]:
            for n in OUR_NAMES:
                cnt[n] += r[f"n_{n}"]
        print(f"  {c:<5} boxes: " + ", ".join(f"{n} {cnt[n]}" for n in OUR_NAMES))
    print(f"-> {out_dir / 'gt_vehicle5.json'}, {out_dir / 'meta.csv'}, {sdir}/")


if __name__ == "__main__":
    main()
