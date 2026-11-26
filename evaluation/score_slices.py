#!/usr/bin/env python3
"""Score one or more trained arms on every test slice, with pycocotools.

Protocol is a deliberate byte-for-byte match of `eval_arm_coco.py` (the scorer
that produced the existing BASE/M1/M2 numbers), so rows from the two are
comparable:

  - NMS fixed for every arm: conf 0.001, iou 0.6, max_det 300
  - GT boxes built from the YOLO .txt labels at a fixed 1280x720
  - category_id == class index (0..4)
  - predictions round-tripped through 6-decimal normalized xywh, exactly as
    eval_arm_coco.py writes and re-reads them
  - COCOeval left at its default maxDets [1, 10, 100] - NOT raised to 300
  - per-class AP via a separate COCOeval with params.catIds = [ci]

Two deliberate differences, neither of which changes a number:

  1. **One inference pass, not one per slice.** eval_arm_coco.py re-runs the
     detector for every split. With 19 slices over the same 8,841 images that
     would be 19x the GPU time for identical predictions. This runs inference
     once over the full test set, then restricts `COCOeval.params.imgIds` per
     slice. Restricting imgIds is exactly equivalent to building a per-slice
     GT, since annotations and detections for excluded images take no part in
     the accumulation.
  2. **Slices come from list files**, not from a directory of images, because
     the slice yamls point `val:` at a list rather than a folder.

Usage:
  python evaluation/score_slices.py \
      --arm t1_jepa_100=runs/detect/t1_jepa_100/weights/best.pt \
      --arm base_100=runs/detect/base_100/weights/best.pt \
      --device 0 --out results/slice_scores.csv

Score every arm you intend to compare in ONE invocation, so they cannot
silently differ in protocol.
"""
import argparse
import contextlib
import csv
import io
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]

# Fixed for every arm. Do NOT make these per-arm configurable - that would
# defeat the entire point of the comparison.
CONF_THRES = 0.001
IOU_THRES = 0.6
MAX_DET = 300
IMG_WIDTH = 1280
IMG_HEIGHT = 720

# Reporting guardrails (docs/MASTER-RECORD.md section 1).
MIN_INSTANCES_PER_CLASS = 800    # below this, a per-class AP is indicative only
MIN_IMAGES_PER_SLICE = 100       # below this, do not report a slice mAP at all

HEADLINE = ["overall", "day", "night", "dawndusk", "clear", "rainy", "snowy",
            "adverse", "night_adverse", "day_adverse", "night_clear"]


def yolo_to_xywh(cx, cy, w, h, W=IMG_WIDTH, H=IMG_HEIGHT):
    return [(cx - w / 2) * W, (cy - h / 2) * H, w * W, h * H]


def build_gt(image_names, labels_dir, class_names):
    images, anns, name_to_id, aid = [], [], {}, 1
    per_image_counts = {}
    for iid, name in enumerate(sorted(image_names), start=1):
        name_to_id[name] = iid
        images.append({"id": iid, "file_name": name,
                       "width": IMG_WIDTH, "height": IMG_HEIGHT})
        gt = Path(labels_dir) / (Path(name).stem + ".txt")
        if not gt.exists():
            raise FileNotFoundError(f"missing GT label file: {gt}")
        counts = [0] * len(class_names)
        for line in gt.read_text().splitlines():
            if not line.strip():
                continue
            p = line.split()
            k = int(p[0])
            box = yolo_to_xywh(*(float(v) for v in p[1:5]))
            anns.append({"id": aid, "image_id": iid, "category_id": k, "bbox": box,
                         "area": box[2] * box[3], "iscrowd": 0})
            counts[k] += 1
            aid += 1
        per_image_counts[name] = counts
    gt_coco = COCO()
    gt_coco.dataset = {"images": images, "annotations": anns,
                       "categories": [{"id": i, "name": n} for i, n in enumerate(class_names)]}
    with contextlib.redirect_stdout(io.StringIO()):
        gt_coco.createIndex()
    return gt_coco, name_to_id, per_image_counts


def run_inference(model, image_paths, name_to_id, n_classes, imgsz, device, batch):
    """Chunked predict - passing the whole list to predict() does not actually
    cap memory to `batch` and has OOM'd a 46 GB A40 (eval_arm_coco.py's note).
    Detections are formatted to 6dp normalized and parsed back, matching the
    file round-trip eval_arm_coco.py performs."""
    dets, oob, n_done = [], 0, 0
    for start in range(0, len(image_paths), batch):
        chunk = image_paths[start:start + batch]
        results = model.predict(source=[str(p) for p in chunk], imgsz=imgsz,
                                conf=CONF_THRES, iou=IOU_THRES, max_det=MAX_DET,
                                device=device, batch=len(chunk), stream=False, verbose=False)
        for r in results:
            name = Path(r.path).name
            if r.boxes is not None and len(r.boxes):
                for (cx, cy, w, h), c, k in zip(r.boxes.xywhn.tolist(),
                                                r.boxes.conf.tolist(),
                                                r.boxes.cls.tolist()):
                    if int(k) >= n_classes:
                        oob += 1
                        continue
                    # identical 6dp round-trip to the .txt path
                    cx, cy, w, h = (float(f"{v:.6f}") for v in (cx, cy, w, h))
                    dets.append({"image_id": name_to_id[name], "category_id": int(k),
                                 "bbox": yolo_to_xywh(cx, cy, w, h),
                                 "score": float(f"{c:.6f}")})
        n_done += len(chunk)
        if n_done % (batch * 40) == 0 or n_done == len(image_paths):
            print(f"    inference {n_done}/{len(image_paths)}", flush=True)
    if oob:
        print(f"    note: dropped {oob} predictions with class id >= {n_classes}")
    return dets


def evaluate(gt_coco, dt_coco, img_ids, cat_id=None):
    ev = COCOeval(gt_coco, dt_coco, iouType="bbox")
    ev.params.imgIds = list(img_ids)
    if cat_id is not None:
        ev.params.catIds = [cat_id]
    with contextlib.redirect_stdout(io.StringIO()):
        ev.evaluate(); ev.accumulate(); ev.summarize()
    s = ev.stats
    out = {}
    for key, i in (("mAP50-95", 0), ("mAP50", 1), ("mAP75", 2)):
        v = float(s[i])
        out[key] = None if v < 0 else round(v, 5)   # COCOeval returns -1 for "no GT"
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", action="append", required=True, metavar="NAME=WEIGHTS",
                    help="repeatable; e.g. --arm t1_jepa_100=runs/detect/t1_jepa_100/weights/best.pt")
    ap.add_argument("--slices-dir", default="splits/test_slices")
    ap.add_argument("--test-list", default="splits/test.txt")
    ap.add_argument("--images-dir", default="dataset/yolo/images/val")
    ap.add_argument("--labels-dir", default="dataset/yolo/labels/val")
    ap.add_argument("--data-config", default="configs/data/bdd100k_vehicle5.yaml")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--out", default="results/slice_scores.csv")
    args = ap.parse_args()

    import yaml
    names = yaml.safe_load(open(ROOT / args.data_config))["names"]
    class_names = [names[k] for k in sorted(names)] if isinstance(names, dict) else list(names)

    images_dir = ROOT / args.images_dir
    all_names = [Path(l.strip()).name for l in open(ROOT / args.test_list) if l.strip()]
    print(f"test pool: {len(all_names)} images, {len(class_names)} classes {class_names}")

    slices = {}
    for f in sorted((ROOT / args.slices_dir).glob("*.txt")):
        slices[f.stem] = [Path(l.strip()).name for l in open(f) if l.strip()]
    slices = {"overall": all_names, **slices}
    print(f"slices: {len(slices)}")

    print("building GT ...")
    gt_coco, name_to_id, per_image_counts = build_gt(all_names, ROOT / args.labels_dir, class_names)
    print(f"  {len(gt_coco.dataset['annotations']):,} GT instances")

    rows, timestamp = [], datetime.now(timezone.utc).isoformat()
    summaries = {}

    for spec in args.arm:
        if "=" not in spec:
            ap.error(f"--arm must be NAME=WEIGHTS, got {spec!r}")
        arm, wpath = spec.split("=", 1)
        wpath = Path(wpath) if Path(wpath).is_absolute() else ROOT / wpath
        if not wpath.exists():
            raise SystemExit(f"FLAG: no weights at {wpath}")

        print(f"\n=== {arm} ({wpath}) ===")
        model = YOLO(str(wpath))
        dets = run_inference(model, [images_dir / n for n in all_names],
                             name_to_id, len(class_names), args.imgsz, args.device, args.batch)
        if not dets:
            raise SystemExit(f"FLAG: no detections for {arm} - check weights/thresholds")
        print(f"    {len(dets):,} detections")
        with contextlib.redirect_stdout(io.StringIO()):
            dt_coco = gt_coco.loadRes(dets)

        summaries[arm] = {}
        for sname, snames in slices.items():
            ids = [name_to_id[n] for n in snames]
            inst = [0] * len(class_names)
            for n in snames:
                for i, c in enumerate(per_image_counts[n]):
                    inst[i] += c
            total_inst = sum(inst)

            ov = evaluate(gt_coco, dt_coco, ids)
            summaries[arm][sname] = ov
            reportable = len(snames) >= MIN_IMAGES_PER_SLICE
            for metric, value in ov.items():
                rows.append({"arm": arm, "slice": sname, "class": "all", "metric": metric,
                             "value": value, "n_images": len(snames), "n_instances": total_inst,
                             "reportable": reportable, "timestamp": timestamp})
            for ci, cname in enumerate(class_names):
                pc = evaluate(gt_coco, dt_coco, ids, cat_id=ci)
                summaries[arm][f"{sname}::{cname}"] = pc
                for metric, value in pc.items():
                    rows.append({"arm": arm, "slice": sname, "class": cname, "metric": metric,
                                 "value": value, "n_images": len(snames), "n_instances": inst[ci],
                                 "reportable": reportable and inst[ci] >= MIN_INSTANCES_PER_CLASS,
                                 "timestamp": timestamp})
            flag = "" if reportable else "   <-- too few images, DO NOT REPORT"
            print(f"  {sname:<15} n={len(snames):>5} inst={total_inst:>7}  "
                  f"mAP50={ov['mAP50']}  mAP75={ov['mAP75']}  mAP50-95={ov['mAP50-95']}{flag}")

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\nwrote {len(rows)} rows -> {out}")

    # ---- headline table ----
    arms = list(summaries)
    print("\n" + "=" * 78)
    print("mAP50, 'all' classes" + (f"   (delta vs {arms[0]})" if len(arms) > 1 else ""))
    print("=" * 78)
    hdr = f"{'slice':<15}" + "".join(f"{a:>18}" for a in arms)
    print(hdr); print("-" * len(hdr))
    for s in HEADLINE:
        if s not in summaries[arms[0]]:
            continue
        line = f"{s:<15}"
        base = summaries[arms[0]][s]["mAP50"]
        for a in arms:
            v = summaries[a][s]["mAP50"]
            cell = f"{v:.4f}" if v is not None else "n/a"
            if a != arms[0] and v is not None and base is not None:
                cell += f" ({(v - base) * 100:+.2f})"
            line += f"{cell:>18}"
        print(line)

    # ---- the question four training recipes could not move ----
    print("\n" + "=" * 78)
    print("car day->night mAP75  (the localization collapse; BASE/M1/M2/M3 all -9.6 to -9.8)")
    print("=" * 78)
    for a in arms:
        d = summaries[a].get("day::car", {}).get("mAP75")
        n = summaries[a].get("night::car", {}).get("mAP75")
        if d is None or n is None:
            print(f"  {a:<20} n/a")
        else:
            print(f"  {a:<20} day {d:.4f}  night {n:.4f}   delta {(n - d) * 100:+.2f} pts")

    (out.with_suffix(".json")).write_text(json.dumps(summaries, indent=2))
    print(f"\nfull per-slice/per-class summary -> {out.with_suffix('.json')}")


if __name__ == "__main__":
    main()
