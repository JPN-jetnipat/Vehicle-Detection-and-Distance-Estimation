#!/usr/bin/env python3
"""Score BDD-trained arms ZERO-SHOT on ExDark (external night test set).

Same protocol as evaluation/score_slices.py - NMS constants, 6dp normalized
round-trip, COCOeval defaults (maxDets 100), per-class COCOeval - and the
constants are IMPORTED from score_slices.py so the two cannot drift apart.
Differences, all forced by ExDark:

  1. Real per-image width/height (ExDark sizes vary; BDD is always 1280x720).
     The detector's decoded size is checked against the GT size per image.
  2. ExDark has no truck class. Default --truck-as-car: truck predictions are
     relabelled car, then car boxes are re-NMS'd at IOU_THRES so a car+truck
     double detection of one object does not become one TP + one FP.
     --no-truck-as-car (strict): truck predictions are dropped instead.
     Report the default; run strict once as a sensitivity check.
  3. 4 scored classes (car, bus, motor, bike). truck AP is n/a, not 0.

Absolute ExDark numbers are NOT comparable to BDD numbers (different images,
annotators and box conventions). Only between-arm deltas on ExDark mean anything,
and only for arms scored in the same run.

Usage (server):
  python evaluation/score_exdark.py \
      --arm base_100=weights/external/base_100_friend.pt \
      --arm t1b_10=runs/detect/t1b_10/weights/best.pt ... \
      --pair t1b_10:base_10 --pair v5_t1b_100:v5_base_100 \
      --device 0 --out results/exdark_scores.csv
"""
import argparse
import contextlib
import csv
import io
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import torch
from pycocotools.coco import COCO
from torchvision.ops import nms
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evaluation"))
from score_slices import (CONF_THRES, IOU_THRES, MAX_DET,  # noqa: E402
                          MIN_IMAGES_PER_SLICE, MIN_INSTANCES_PER_CLASS, evaluate)

CAR, TRUCK = 0, 1
HEADLINE = ["all", "has_vehicle", "outdoor", "outdoor_vehicle", "indoor", "official_test",
            "light_low", "light_ambient", "light_object", "light_single", "light_weak",
            "light_strong", "light_twilight"]


def run_inference(model, gt, images_dir, n_classes, imgsz, device, batch, truck_as_car):
    imgs = gt.dataset["images"]
    dets, size_mismatch, merged, dropped = [], 0, 0, 0
    for start in range(0, len(imgs), batch):
        chunk = imgs[start:start + batch]
        results = model.predict(source=[str(images_dir / im["file_name"]) for im in chunk],
                                imgsz=imgsz, conf=CONF_THRES, iou=IOU_THRES, max_det=MAX_DET,
                                device=device, batch=len(chunk), verbose=False)
        for im, r in zip(chunk, results):
            W, H = im["width"], im["height"]
            if tuple(r.orig_shape) != (H, W):
                size_mismatch += 1
                continue
            per = []
            if r.boxes is not None and len(r.boxes):
                for (cx, cy, w, h), c, k in zip(r.boxes.xywhn.tolist(), r.boxes.conf.tolist(),
                                                r.boxes.cls.tolist()):
                    k = int(k)
                    if k >= n_classes:
                        continue
                    if k == TRUCK:
                        if not truck_as_car:
                            dropped += 1
                            continue
                        k, merged = CAR, merged + 1
                    cx, cy, w, h = (float(f"{v:.6f}") for v in (cx, cy, w, h))
                    per.append((k, [(cx - w / 2) * W, (cy - h / 2) * H, w * W, h * H],
                                float(f"{c:.6f}")))
            if truck_as_car and per:
                cars = [p for p in per if p[0] == CAR]
                if cars:
                    xyxy = torch.tensor([[b[0], b[1], b[0] + b[2], b[1] + b[3]] for _, b, _ in cars])
                    keep = nms(xyxy, torch.tensor([s for _, _, s in cars]), IOU_THRES).tolist()
                    per = [p for p in per if p[0] != CAR] + [cars[i] for i in keep]
                per = sorted(per, key=lambda p: -p[2])[:MAX_DET]
            for k, b, s in per:
                dets.append({"image_id": im["id"], "category_id": k, "bbox": b, "score": s})
        done = start + len(chunk)
        if done % (batch * 40) == 0 or done == len(imgs):
            print(f"    inference {done}/{len(imgs)}", flush=True)
    if size_mismatch:
        raise SystemExit(f"FLAG: {size_mismatch} images decoded at a size different from GT - "
                         "rerun tools/exdark_to_coco.py (EXIF filter) before scoring")
    note = f"{merged} truck->car relabels" if truck_as_car else f"{dropped} truck predictions dropped"
    print(f"    {note}")
    return dets


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", action="append", required=True, metavar="NAME=WEIGHTS")
    ap.add_argument("--pair", action="append", default=[], metavar="ARM:CONTROL",
                    help="print ARM minus CONTROL; repeatable")
    ap.add_argument("--gt", default="dataset/exdark/gt_vehicle5.json")
    ap.add_argument("--images-dir", default="dataset/exdark/images")
    ap.add_argument("--slices-dir", default="splits/exdark")
    ap.add_argument("--truck-as-car", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--out", default="results/exdark_scores.csv")
    args = ap.parse_args()

    gt = COCO()
    gt.dataset = json.loads((ROOT / args.gt).read_text())
    with contextlib.redirect_stdout(io.StringIO()):
        gt.createIndex()
    class_names = [c["name"] for c in sorted(gt.dataset["categories"], key=lambda c: c["id"])]
    name_to_id = {im["file_name"]: im["id"] for im in gt.dataset["images"]}
    inst_by_img = defaultdict(lambda: [0] * len(class_names))
    for a in gt.dataset["annotations"]:
        inst_by_img[a["image_id"]][a["category_id"]] += 1
    print(f"ExDark eval pool: {len(name_to_id)} images, {len(gt.dataset['annotations'])} vehicle boxes, "
          f"truck_as_car={args.truck_as_car}")

    slices = {}
    for f in sorted((ROOT / args.slices_dir).glob("*.txt")):
        names = [l.strip() for l in open(f) if l.strip()]
        missing = [n for n in names if n not in name_to_id]
        if missing:
            raise SystemExit(f"FLAG: {f.name} lists {len(missing)} images not in GT, e.g. {missing[:3]}")
        slices[f.stem] = [name_to_id[n] for n in names]

    rows, summaries = [], {}
    timestamp = datetime.now(timezone.utc).isoformat()
    for spec in args.arm:
        arm, wpath = spec.split("=", 1)
        wpath = Path(wpath) if Path(wpath).is_absolute() else ROOT / wpath
        if not wpath.exists():
            raise SystemExit(f"FLAG: no weights at {wpath}")
        print(f"\n=== {arm} ({wpath}) ===")
        dets = run_inference(YOLO(str(wpath)), gt, ROOT / args.images_dir, len(class_names),
                             args.imgsz, args.device, args.batch, args.truck_as_car)
        if not dets:
            raise SystemExit(f"FLAG: no detections for {arm}")
        with contextlib.redirect_stdout(io.StringIO()):
            dt = gt.loadRes(dets)
        summaries[arm] = {}
        for sname, ids in slices.items():
            inst = [sum(inst_by_img[i][c] for i in ids) for c in range(len(class_names))]
            reportable = len(ids) >= MIN_IMAGES_PER_SLICE
            ov = evaluate(gt, dt, ids)
            summaries[arm][sname] = ov
            for m, v in ov.items():
                rows.append({"arm": arm, "slice": sname, "class": "all", "metric": m, "value": v,
                             "n_images": len(ids), "n_instances": sum(inst),
                             "reportable": reportable, "truck_as_car": args.truck_as_car,
                             "timestamp": timestamp})
            for ci, cname in enumerate(class_names):
                if inst[ci] == 0 and cname == "truck":
                    continue
                pc = evaluate(gt, dt, ids, cat_id=ci)
                summaries[arm][f"{sname}::{cname}"] = pc
                for m, v in pc.items():
                    rows.append({"arm": arm, "slice": sname, "class": cname, "metric": m,
                                 "value": v, "n_images": len(ids), "n_instances": inst[ci],
                                 "reportable": reportable and inst[ci] >= MIN_INSTANCES_PER_CLASS,
                                 "truck_as_car": args.truck_as_car, "timestamp": timestamp})
            flag = "" if reportable else "   <-- too few images, DO NOT REPORT"
            print(f"  {sname:<16} n={len(ids):>5} inst={sum(inst):>6}  mAP50={ov['mAP50']}  "
                  f"mAP75={ov['mAP75']}  mAP50-95={ov['mAP50-95']}{flag}")

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    out.with_suffix(".json").write_text(json.dumps(summaries, indent=2))

    fmt = lambda v: "n/a" if v is None else f"{v:.4f}"
    print("\n" + "=" * 78 + "\nExDark mAP50 all-class / car mAP50 / car mAP75\n" + "=" * 78)
    for s in HEADLINE:
        if s not in slices:
            continue
        print(f"{s}  (n={len(slices[s])})")
        for a in summaries:
            print(f"    {a:<18} all {fmt(summaries[a][s]['mAP50'])}   "
                  f"car {fmt(summaries[a].get(s + '::car', {}).get('mAP50'))} / "
                  f"{fmt(summaries[a].get(s + '::car', {}).get('mAP75'))}")
    if args.pair:
        print("\n" + "=" * 78 + "\npairs: ARM - CONTROL, points (all mAP50 | car mAP50 | car mAP75)\n" + "=" * 78)
        for p in args.pair:
            a, b = p.split(":")
            for s in ("all", "outdoor_vehicle", "light_low", "light_twilight"):
                if s not in slices:
                    continue
                d = []
                for key, m in ((s, "mAP50"), (s + "::car", "mAP50"), (s + "::car", "mAP75")):
                    va, vb = summaries[a][key][m], summaries[b][key][m]
                    d.append("n/a" if va is None or vb is None else f"{(va - vb) * 100:+.2f}")
                print(f"  {a} - {b:<14} {s:<16} " + " | ".join(d))
    print(f"\nwrote {len(rows)} rows -> {out} (+ .json)")


if __name__ == "__main__":
    main()
