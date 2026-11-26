#!/usr/bin/env python3
"""Convert UA-DETRAC (test split, every Nth frame) into an EVAL-ONLY vehicle test set.

Same role and output layout as tools/dawn_to_coco.py / rtts_to_coco.py: never trained or
tuned on, scored zero-shot with evaluation/score_exdark.py
(--gt dataset/detrac/gt_vehicle5.json --images-dir dataset/detrac/images
 --slices-dir splits/detrac --truck car).

Input (outside the repo):
  --images   ~/datasets/UA-DETRAC/test_every10/MVI_xxxxx/imgNNNNN.jpg
             (every 10th frame of the 40 test sequences, extracted from DETRAC-Images.zip)
  --xml-dir  ~/datasets/UA-DETRAC/DETRAC-Annos   (searched recursively for MVI_*.xml;
             only the 40 test XMLs should be unzipped there)

UA-DETRAC facts used here: fixed roadside cameras, 960x540, 25 fps. One XML per sequence:
  <sequence_attribute sence_weather="sunny|cloudy|rainy|night"/>   (sic: "sence")
  <ignored_region><box left top width height/>...</ignored_region>  (areas NOT labelled)
  <frame num="1"><target_list><target><box .../><attribute vehicle_type="car|van|bus|others"/>
Frame num N <-> imgNNNNN.jpg (1-based). Boxes are pixel left/top/width/height.

Class mapping (scored with the 5 BDD ids so arms need no remapping):
  car, van -> car(0) · bus -> bus(2) · others (mostly trucks, also tankers etc.) -> IGNORED
  (iscrowd=1 boxes, so a detection on them is neither TP nor FP). DETRAC has no
  truck/motor/bike labels: truck/motor/bike AP are n/a; score with --truck car (truck
  predictions -> car, then re-NMS).
Ignored regions and "others" boxes are written as iscrowd=1 boxes for BOTH car and bus,
because COCOeval only lets a crowd box absorb detections of its own category.

Frames that the XML does not list (DETRAC omits frames with no labelled vehicle in some
sequences) are EXCLUDED (status no_annotation) rather than treated as empty, since we
cannot tell "no vehicle" from "not annotated". Also excluded: decoded-size mismatch,
unreadable images. Counts are printed and saved in meta.csv.

Slices: all, has_vehicle, sunny, cloudy, rainy, night, day (= sunny+cloudy+rainy, same
cameras as night -> gives a car night/day mAP75 ratio outside BDD).

Usage (server, ~/vehicle-detection, venv active):
  python tools/detrac_to_coco.py --images ~/datasets/UA-DETRAC/test_every10 \
      --xml-dir ~/datasets/UA-DETRAC/DETRAC-Annos
"""
import argparse
import csv
import json
import re
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import cv2
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]

OUR_NAMES = ["car", "truck", "bus", "motor", "bike"]
TYPE_MAP = {"car": 0, "van": 0, "bus": 2}
TYPE_IGNORED = {"others"}
CROWD_CATS = (0, 2)                      # car, bus
WEATHERS = ["sunny", "cloudy", "rainy", "night"]
IMG_RE = re.compile(r"^img(\d+)\.jpg$", re.IGNORECASE)


def box_of(el):
    return tuple(float(el.get(k)) for k in ("left", "top", "width", "height"))


def parse_seq(xml_path):
    root = ET.parse(xml_path).getroot()
    attr = root.find("sequence_attribute")
    weather = (attr.get("sence_weather") or attr.get("scene_weather") or "").lower() if attr is not None else ""
    ignored = [box_of(b) for b in root.findall("ignored_region/box")]
    frames = {}
    for fr in root.findall("frame"):
        objs = []
        for t in fr.findall("target_list/target"):
            b, a = t.find("box"), t.find("attribute")
            vt = (a.get("vehicle_type") if a is not None else "") or ""
            objs.append((vt.strip().lower(), box_of(b)))
        frames[int(fr.get("num"))] = objs
    return weather, ignored, frames


def clip(b, W, H):
    x, y, w, h = b
    x0, y0, x1, y1 = max(0.0, x), max(0.0, y), min(float(W), x + w), min(float(H), y + h)
    return (x0, y0, x1, y1), (x0, y0, x1, y1) != (x, y, x + w, y + h)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", required=True, help="folder of MVI_xxxxx/ frame folders")
    ap.add_argument("--xml-dir", required=True, help="folder searched for MVI_*.xml")
    ap.add_argument("--out-dir", default="dataset/detrac")
    ap.add_argument("--splits-dir", default="splits/detrac")
    args = ap.parse_args()
    img_root, xml_root = Path(args.images).expanduser(), Path(args.xml_dir).expanduser()

    xmls = {p.stem: p for p in xml_root.rglob("MVI_*.xml")}
    seq_dirs = {p.name: p for p in img_root.iterdir() if p.is_dir() and p.name.startswith("MVI_")}
    print(f"XML sequences {len(xmls)} · image sequences {len(seq_dirs)}")
    if set(seq_dirs) - set(xmls):
        raise SystemExit(f"FLAG: image folders without XML: {sorted(set(seq_dirs) - set(xmls))[:5]}")
    if set(xmls) - set(seq_dirs):
        print(f"  note: XML without image folder (not extracted?): {sorted(set(xmls) - set(seq_dirs))}")

    out_dir = ROOT / args.out_dir
    img_out = out_dir / "images"
    img_out.mkdir(parents=True, exist_ok=True)

    images, anns, rows, aid = [], [], [], 1
    status, types, clipped = Counter(), Counter(), 0
    seq_weather = {}
    for seq in sorted(seq_dirs):
        weather, ignored, frames = parse_seq(xmls[seq])
        if weather not in WEATHERS:
            raise SystemExit(f"FLAG: {seq} has unexpected weather '{weather}'")
        seq_weather[seq] = weather
        for img in sorted(seq_dirs[seq].iterdir()):
            m = IMG_RE.match(img.name)
            if not m:
                continue
            num = int(m.group(1))
            name = f"{seq}__{img.name}"
            row = {"seq": seq, "frame": num, "weather": weather, "width": "", "height": "",
                   **{f"n_{n}": 0 for n in OUR_NAMES}, "n_ignored_vehicle": 0,
                   "n_ignored_region": len(ignored), "eval_name": "", "status": "ok"}
            if num not in frames:
                row["status"] = "no_annotation"
            else:
                W, H = Image.open(img).size
                dec = cv2.imread(str(img))
                row.update(width=W, height=H)
                if dec is None:
                    row["status"] = "unreadable"
                elif dec.shape[:2] != (H, W):
                    row["status"] = "size_mismatch"
            status[row["status"]] += 1
            if row["status"] != "ok":
                rows.append(row); continue

            link = img_out / name
            if not link.exists():
                link.symlink_to(img.resolve())
            iid = len(images) + 1
            images.append({"id": iid, "file_name": name, "width": W, "height": H})

            def add(cat, xyxy, crowd):
                nonlocal aid
                x0, y0, x1, y1 = xyxy
                anns.append({"id": aid, "image_id": iid, "category_id": cat,
                             "bbox": [x0, y0, x1 - x0, y1 - y0], "area": (x1 - x0) * (y1 - y0),
                             "iscrowd": int(crowd)})
                aid += 1

            for b in ignored:
                xyxy, _ = clip(b, W, H)
                if xyxy[2] - xyxy[0] > 1 and xyxy[3] - xyxy[1] > 1:
                    for c in CROWD_CATS:
                        add(c, xyxy, True)
            for vt, b in frames[num]:
                types[vt] += 1
                xyxy, was_clipped = clip(b, W, H)
                clipped += was_clipped
                if xyxy[2] - xyxy[0] <= 1 or xyxy[3] - xyxy[1] <= 1:
                    continue
                if vt in TYPE_MAP:
                    add(TYPE_MAP[vt], xyxy, False)
                    row[f"n_{OUR_NAMES[TYPE_MAP[vt]]}"] += 1
                else:                                  # "others" and anything unknown
                    for c in CROWD_CATS:
                        add(c, xyxy, True)
                    row["n_ignored_vehicle"] += 1
            row["eval_name"] = name
            rows.append(row)

    gt = {"images": images, "annotations": anns,
          "categories": [{"id": i, "name": n} for i, n in enumerate(OUR_NAMES)]}
    (out_dir / "gt_vehicle5.json").write_text(json.dumps(gt))
    with open(out_dir / "meta.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    ok = [r for r in rows if r["status"] == "ok"]
    slices = {"all": ok, "has_vehicle": [r for r in ok if r["n_car"] + r["n_bus"] > 0]}
    for wx in WEATHERS:
        slices[wx] = [r for r in ok if r["weather"] == wx]
    slices["day"] = [r for r in ok if r["weather"] != "night"]
    sdir = ROOT / args.splits_dir
    sdir.mkdir(parents=True, exist_ok=True)
    for s, rs in slices.items():
        (sdir / f"{s}.txt").write_text("".join(r["eval_name"] + "\n" for r in rs))

    unknown = {k: v for k, v in types.items() if k not in TYPE_MAP and k not in TYPE_IGNORED}
    print(f"\nstatus: {dict(status)}")
    print(f"vehicle_type counts (raw): {dict(types)}")
    print(f"UNKNOWN vehicle types (treated as ignored - check these): {unknown or 'none'}")
    print(f"scored boxes: car {sum(r['n_car'] for r in ok)} (car+van) · bus {sum(r['n_bus'] for r in ok)}"
          f" · ignored vehicles ('others') {sum(r['n_ignored_vehicle'] for r in ok)}")
    print(f"boxes clipped to image bounds: {clipped}")
    print("sequences per weather: " + ", ".join(
        f"{wx} {sum(1 for v in seq_weather.values() if v == wx)}" for wx in WEATHERS))
    for s, rs in slices.items():
        print(f"  {s:<11} images {len(rs):>5} · car {sum(r['n_car'] for r in rs):>6} · bus {sum(r['n_bus'] for r in rs):>5}")
    print(f"-> {out_dir / 'gt_vehicle5.json'}, {out_dir / 'meta.csv'}, {sdir}/")


if __name__ == "__main__":
    main()
