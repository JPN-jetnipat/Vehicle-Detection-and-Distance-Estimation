"""Task 1 (Step 2): motor/bike AP50 + box counts on every BDD test split, and the
prediction files the team bootstrap needs.

For each --arm NAME=WEIGHTS this:
  1. runs inference ONCE on the 8,841-image test pool (test_day + test_night +
     test_dawndusk image folders); the weather splits are the same images regrouped,
     so they are scored from these predictions without re-running the model
  2. for every split in results/coco_gt/ (9 splits + test_all) writes
     <save-preds>/<NAME>_<split>.json   e.g. BASE_v11_test_night.json
     as a COCO results list whose image_ids match results/coco_gt/<split>.json
  3. scores overall / motor / bike AP50 with pycocotools (same COCOeval call and
     NMS constants as scripts/eval_arm_coco.py) and appends rows to --out

Box counts per split: gt_motor / gt_bike = ground-truth boxes; pred_motor / pred_bike =
predicted boxes with score >= --count-conf (default 0.25, ultralytics' default
display threshold - the raw files keep everything down to conf 0.001).

Append "@pre" to WEIGHTS to run that arm on the preprocessed images
(dataset/yolo_preprocessed/images/...) instead of the raw ones - for condprep_v1.

Usage (server, repo root):
    python scripts/score_bdd_motorbike.py \
        --arm BASE_v11=runs/detect/set3_combined/weights/best.pt \
        --arm Set1_v11=runs/detect/set1_night_aug/weights/best.pt \
        --arm Preprocessed_v11=runs/detect/condprep_v1/weights/best.pt@pre \
        --save-preds results/predictions --out results/motorbike_bdd.csv
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from pycocotools.coco import COCO
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import eval_arm_coco as E  # noqa: E402  (same NMS constants, inference and COCOeval call)

GT_DIR = ROOT / "results" / "coco_gt"
TIME_SPLITS = ["test_day", "test_night", "test_dawndusk"]  # together = the whole test pool
SPLITS = TIME_SPLITS + ["test_clear", "test_overcast", "test_partlycloudy", "test_rainy",
                        "test_snowy", "test_foggy", "test_all"]
E_NAMES = ["car", "truck", "bus", "motor", "bike"]
MOTOR, BIKE = 3, 4


def load_gt(split: str) -> COCO:
    gt = COCO()
    gt.dataset = json.loads((GT_DIR / f"{split}.json").read_text())
    with contextlib.redirect_stdout(io.StringIO()):
        gt.createIndex()
    return gt


def predict_pool(weights: Path, preprocessed: bool, imgsz: int, device: str, batch: int) -> dict:
    """file_name -> list of (cls, [x, y, w, h] px, score) for all 8,841 test images."""
    root = ROOT / "dataset" / ("yolo_preprocessed" if preprocessed else "yolo") / "images"
    model = YOLO(str(weights))
    by_name = {}
    with tempfile.TemporaryDirectory(prefix="score_bdd_mb_") as tmp:
        for split in TIME_SPLITS:
            img_dir = root / split
            paths = sorted(p for p in img_dir.iterdir() if p.is_file())
            if not paths:
                raise SystemExit(f"no images in {img_dir}")
            print(f"    inference {split}: {len(paths)} images from {img_dir}", flush=True)
            E.run_inference_to_dir(model, paths, Path(tmp), imgsz, device, batch)
            for p in paths:
                dets = []
                for line in (Path(tmp) / f"{p.stem}.txt").read_text().splitlines():
                    q = line.split()
                    if q and int(q[0]) < len(E_NAMES):
                        dets.append((int(q[0]), E.yolo_line_to_xywh(q, E.IMG_WIDTH, E.IMG_HEIGHT), float(q[5])))
                by_name[p.name] = dets
    return by_name


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", action="append", required=True, metavar="NAME=WEIGHTS[@pre]")
    ap.add_argument("--save-preds", type=Path, default=ROOT / "results" / "predictions")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "motorbike_bdd.csv")
    ap.add_argument("--count-conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0")
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()

    gts = {s: load_gt(s) for s in SPLITS}
    for s, g in gts.items():
        names = [c["name"] for c in sorted(g.dataset["categories"], key=lambda c: c["id"])]
        assert names == E_NAMES, f"{s}: unexpected classes {names}"
    args.save_preds.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).isoformat()
    rows = []

    for spec in args.arm:
        name, wspec = spec.split("=", 1)
        preprocessed = wspec.endswith("@pre")
        weights = Path(wspec.removesuffix("@pre"))
        weights = weights if weights.is_absolute() else ROOT / weights
        if not weights.exists():
            raise SystemExit(f"no weights at {weights}")
        print(f"\n=== {name} ({weights}{', preprocessed images' if preprocessed else ''}) ===", flush=True)
        pool = predict_pool(weights, preprocessed, args.imgsz, args.device, args.batch)

        print(f"  {'split':<18}{'n_img':>6}{'mAP50':>8}{'motor':>8}{'bike':>8}"
              f"{'gt_m':>6}{'gt_b':>6}{'pred_m':>8}{'pred_b':>8}")
        for split in SPLITS:
            gt = gts[split]
            dets = []
            for im in gt.dataset["images"]:
                if im["file_name"] not in pool:
                    raise SystemExit(f"{split}: no predictions for {im['file_name']}")
                for k, box, s in pool[im["file_name"]]:
                    dets.append({"image_id": im["id"], "category_id": k, "bbox": box, "score": s})
            (args.save_preds / f"{name}_{split}.json").write_text(json.dumps(dets))

            ids = gt.getImgIds()
            r = {"n_images": len(ids)}
            with contextlib.redirect_stdout(io.StringIO()):
                dt = gt.loadRes(dets)
            r["mAP50"] = E.evaluate_slice(gt, dt, ids)["mAP50"]
            for cname, cid in (("motor", MOTOR), ("bike", BIKE)):
                n_gt = len(gt.getAnnIds(catIds=[cid]))
                # COCOeval returns -1 when a class has no GT in the slice (e.g. foggy bike)
                r[f"AP50_{cname}"] = E.evaluate_slice(gt, dt, ids, cat_id=cid)["mAP50"] if n_gt else None
                r[f"gt_{cname}"] = n_gt
                r[f"pred_{cname}"] = sum(1 for d in dets if d["category_id"] == cid and d["score"] >= args.count_conf)
            fmt = lambda v: "  n/a" if v is None else f"{v:.3f}"
            print(f"  {split:<18}{r['n_images']:>6}{fmt(r['mAP50']):>8}{fmt(r['AP50_motor']):>8}"
                  f"{fmt(r['AP50_bike']):>8}{r['gt_motor']:>6}{r['gt_bike']:>6}{r['pred_motor']:>8}{r['pred_bike']:>8}",
                  flush=True)
            rows.append({"arm": name, "weights": str(weights), "images": "preprocessed" if preprocessed else "raw",
                         "split": split, **r, "count_conf": args.count_conf, "timestamp": timestamp})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    new = not args.out.exists()
    with open(args.out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if new:
            w.writeheader()
        w.writerows(rows)
    print(f"\nappended {len(rows)} rows -> {args.out}")
    print(f"prediction files -> {args.save_preds}/<arm>_<split>.json")


if __name__ == "__main__":
    main()
