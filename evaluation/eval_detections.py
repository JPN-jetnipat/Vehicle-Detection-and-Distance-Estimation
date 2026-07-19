#!/usr/bin/env python3
"""THE one evaluator (brief section 8): every arm - ours, the friend's
weights, DANN teammate's if wanted - gets scored by this script so numbers
are directly comparable.

Inputs are YOLO-format files on disk (no torch needed - pure CPU, runs
anywhere):
  GT:    <gt-labels>/<stem>.txt   lines: cls cx cy w h          (normalized)
  preds: <preds>/<stem>.txt       lines: cls cx cy w h conf     (normalized)
Missing pred file = no detections for that image (fine). Missing GT file
is an error (converter always writes one, even empty).

Metrics per slice (overall / night / dawn-dusk / rainy): mAP@0.5,
mAP@0.75 (localization crispness), mAP@0.5:0.95, via pycocotools
(101-pt COCO protocol). Per-class AP is reported for the overall slice.
Fog is deliberately NOT a slice (too rare in BDD100K).

Usage:
  python evaluation/eval_detections.py \
      --image-list splits/val_final.txt --gt-labels dataset/yolo/labels/val \
      --preds runs_jepa/preds/baseline_coco_100_val \
      --attr-index dataset/yolo/attr_index.json \
      --data-config configs/data/bdd100k_all10.yaml \
      --out runs_jepa/metrics/baseline_coco_100_val
"""
import argparse, contextlib, csv, io, json
from pathlib import Path

import yaml
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

SLICES = {  # slice name -> (attr key, matching values)
    "overall":   None,
    "night":     ("timeofday", {"night"}),
    "dawn_dusk": ("timeofday", {"dawn/dusk"}),
    "rainy":     ("weather", {"rainy"}),
}


def load_names(data_config):
    names = yaml.safe_load(open(data_config))["names"]
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]
    return names


def yolo_line_to_xywh(parts, W, H):
    cx, cy, w, h = (float(v) for v in parts[1:5])
    return [(cx - w / 2) * W, (cy - h / 2) * H, w * W, h * H]


def build_coco(image_names, gt_dir, names, W, H):
    images, anns = [], []
    name_to_id = {}
    aid = 1
    for iid, name in enumerate(sorted(image_names), start=1):
        name_to_id[name] = iid
        images.append({"id": iid, "file_name": name, "width": W, "height": H})
        gt = Path(gt_dir) / (Path(name).stem + ".txt")
        if not gt.exists():
            raise FileNotFoundError(f"missing GT label file: {gt}")
        for line in gt.read_text().splitlines():
            p = line.split()
            box = yolo_line_to_xywh(p, W, H)
            anns.append({"id": aid, "image_id": iid, "category_id": int(p[0]),
                         "bbox": box, "area": box[2] * box[3], "iscrowd": 0})
            aid += 1
    gt_coco = COCO()
    gt_coco.dataset = {"images": images, "annotations": anns,
                       "categories": [{"id": i, "name": n} for i, n in enumerate(names)]}
    with contextlib.redirect_stdout(io.StringIO()):
        gt_coco.createIndex()
    return gt_coco, name_to_id


def load_preds(image_names, pred_dir, name_to_id, W, H, n_classes):
    dets = []
    found = 0
    oob = 0  # predictions in classes outside our list (e.g. the friend's
             # nc=8 head has 5 dead KITTI classes that may fire at low conf)
    for name in image_names:
        pf = Path(pred_dir) / (Path(name).stem + ".txt")
        if not pf.exists():
            continue
        found += 1
        for line in pf.read_text().splitlines():
            if not line.strip():
                continue  # blank line (e.g. a trailing newline in an empty-detections file)
            p = line.split()
            if len(p) != 6:
                raise ValueError(f"{pf}: expected 'cls cx cy w h conf', got: {line}")
            if int(p[0]) >= n_classes:
                oob += 1
                continue
            dets.append({"image_id": name_to_id[name], "category_id": int(p[0]),
                         "bbox": yolo_line_to_xywh(p, W, H), "score": float(p[5])})
    if oob:
        print(f"note: dropped {oob} predictions with class id >= {n_classes} (untrained/dead classes)")
    return dets, found


def evaluate_slice(gt_coco, dt_coco, img_ids, cat_id=None):
    ev = COCOeval(gt_coco, dt_coco, iouType="bbox")
    ev.params.imgIds = img_ids
    if cat_id is not None:
        ev.params.catIds = [cat_id]
    with contextlib.redirect_stdout(io.StringIO()):
        ev.evaluate(); ev.accumulate(); ev.summarize()
    s = ev.stats
    return {"mAP50_95": round(float(s[0]), 5), "mAP50": round(float(s[1]), 5),
            "mAP75": round(float(s[2]), 5)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image-list", required=True, help="txt of image names/paths (one per line)")
    ap.add_argument("--gt-labels", required=True)
    ap.add_argument("--preds", required=True)
    ap.add_argument("--attr-index", required=True)
    ap.add_argument("--data-config", required=True)
    ap.add_argument("--img-width", type=int, default=1280)
    ap.add_argument("--img-height", type=int, default=720)
    ap.add_argument("--out", required=True, help="output prefix -> .json + .csv")
    args = ap.parse_args()

    names = load_names(args.data_config)
    image_names = [Path(l.strip()).name for l in open(args.image_list) if l.strip()]
    attrs = json.load(open(args.attr_index))
    W, H = args.img_width, args.img_height

    gt_coco, name_to_id = build_coco(image_names, args.gt_labels, names, W, H)
    dets, found = load_preds(image_names, args.preds, name_to_id, W, H, len(names))
    print(f"{len(image_names)} images, pred files for {found}, {len(dets)} detections")
    if not dets:
        raise SystemExit("no detections found - wrong --preds dir?")
    with contextlib.redirect_stdout(io.StringIO()):
        dt_coco = gt_coco.loadRes(dets)

    results = {"n_images": {}, "slices": {}, "per_class_overall": {}}
    for sl, rule in SLICES.items():
        if rule is None:
            ids = [name_to_id[n] for n in image_names]
        else:
            key, vals = rule
            ids = [name_to_id[n] for n in image_names
                   if attrs.get(n, {}).get(key) in vals]
        results["n_images"][sl] = len(ids)
        if not ids:
            results["slices"][sl] = None
            continue
        results["slices"][sl] = evaluate_slice(gt_coco, dt_coco, ids)
        print(f"[{sl:9s}] n={len(ids):5d}  " + "  ".join(
            f"{k}={v}" for k, v in results["slices"][sl].items()))
    all_ids = [name_to_id[n] for n in image_names]
    for ci, cname in enumerate(names):
        results["per_class_overall"][cname] = evaluate_slice(gt_coco, dt_coco, all_ids, cat_id=ci)

    # Per-class ground-truth instance counts, straight from the COCO object
    # already built above (no re-reading label files needed). This is what
    # powers the YOLOv5-style summary table printed below.
    n_instances = {cname: len(gt_coco.getAnnIds(catIds=[ci])) for ci, cname in enumerate(names)}
    results["per_class_instances"] = n_instances

    n_total_images = results["n_images"]["overall"]
    print(f"\n{'Class':>10} {'Images':>8} {'Instances':>10} {'mAP50':>8} {'mAP75':>8} {'mAP50-95':>10}")
    o = results["slices"]["overall"]
    print(f"{'all':>10} {n_total_images:>8} {sum(n_instances.values()):>10} "
          f"{o['mAP50']:>8} {o['mAP75']:>8} {o['mAP50_95']:>10}")
    for cname in names:
        m = results["per_class_overall"][cname]
        print(f"{cname:>10} {n_total_images:>8} {n_instances[cname]:>10} "
              f"{m['mAP50']:>8} {m['mAP75']:>8} {m['mAP50_95']:>10}")
    print("(note: no P/R columns - those are single-confidence-threshold numbers "
          "YOLOv5 computes internally; mAP50/75/50-95 above integrate across all "
          "thresholds, which is the metric the project brief actually asks for)")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    meta = {"args": vars(args), "classes": names}
    out.with_suffix(".json").write_text(json.dumps({**meta, **results}, indent=2))
    with open(out.with_suffix(".csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["row_type", "name", "n_images", "n_instances", "mAP50", "mAP75", "mAP50_95"])
        for sl, m in results["slices"].items():
            if m:
                w.writerow(["slice", sl, results["n_images"][sl], "", m["mAP50"], m["mAP75"], m["mAP50_95"]])
        for cname in names:
            m = results["per_class_overall"][cname]
            w.writerow(["class", cname, n_total_images, n_instances[cname],
                        m["mAP50"], m["mAP75"], m["mAP50_95"]])
    print(f"\nwrote {out}.json and {out}.csv")


if __name__ == "__main__":
    main()
