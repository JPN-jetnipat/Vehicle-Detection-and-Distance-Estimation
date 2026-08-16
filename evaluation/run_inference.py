#!/usr/bin/env python3
"""Run a YOLO11 .pt on an image list and dump predictions for the evaluator.

Same NMS settings for EVERY arm (fairness): conf 0.001, IoU 0.6, max 300
dets - the standard mAP protocol. Output: one txt per image,
lines "cls cx cy w h conf" normalized to the ORIGINAL image size, i.e. the
exact format evaluation/eval_detections.py consumes.

Only needs `ultralytics` (pip) - no vendored repo to import (the old
YOLOv5-era version of this script needed the vendored yolov5/ repo for
letterbox/NMS/scale_boxes; ultralytics does all of that internally and just
hands back boxes already rescaled to the original image size).

--image-list entries can be bare filenames OR full paths - only the
basename is used. Pass --images-dir pointing at the single flat directory
that basename actually lives in (dataset/yolo/images/train or .../val, per
tools/materialize_images.py's layout) - same basename-only convention
evaluation/eval_detections.py already uses, so both scripts agree no matter
which split-list variant (bare-name splits/*.txt or full-path
dataset/yolo/splits/*.txt) you point this at.

Usage:
  python evaluation/run_inference.py --weights runs/detect/default_100/weights/best.pt \
      --image-list splits/test.txt --images-dir dataset/yolo/images/val \
      --out-dir runs/preds/default_100_test --device 0
"""
import argparse
from pathlib import Path

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--image-list", required=True,
                    help="txt of image names/paths, one per line - only the basename is used")
    ap.add_argument("--images-dir", required=True,
                    help="flat dir containing those images, e.g. dataset/yolo/images/val")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf-thres", type=float, default=0.001)
    ap.add_argument("--iou-thres", type=float, default=0.6)
    ap.add_argument("--max-det", type=int, default=300)
    ap.add_argument("--device", default="cpu", help="'cpu' or CUDA index like '0'")
    ap.add_argument("--half", action="store_true", help="fp16 inference (GPU only)")
    ap.add_argument("--batch", type=int, default=32, help="inference batch size (throughput only, not part of the mAP protocol)")
    args = ap.parse_args()

    images_dir = Path(args.images_dir)
    names = [Path(l.strip()).name for l in open(args.image_list) if l.strip()]
    paths = [images_dir / n for n in names]
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} of {len(paths)} images from --image-list not found under "
                                 f"{images_dir}, e.g. {missing[0]}")

    print(f"loading {args.weights} ...")
    model = YOLO(args.weights)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    results = model.predict(
        source=[str(p) for p in paths],
        imgsz=args.imgsz,
        conf=args.conf_thres,
        iou=args.iou_thres,
        max_det=args.max_det,
        device=args.device,
        half=args.half,
        batch=args.batch,
        stream=True,     # generator - one Results object per image, low memory for 10k+ lists
        verbose=False,
    )

    n_det = 0
    for i, r in enumerate(results):
        lines = []
        if r.boxes is not None and len(r.boxes):
            xywhn = r.boxes.xywhn.tolist()   # normalized [cx, cy, w, h], already rescaled to ORIGINAL image size
            conf = r.boxes.conf.tolist()
            cls = r.boxes.cls.tolist()
            for (cx, cy, w, h), c, k in zip(xywhn, conf, cls):
                lines.append(f"{int(k)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {c:.6f}")
        stem = Path(r.path).stem
        (out / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        n_det += len(lines)
        if (i + 1) % 500 == 0:
            print(f"{i + 1}/{len(paths)} images, {n_det} detections so far", flush=True)

    print(f"done: {len(paths)} images, {n_det} detections -> {out}")


if __name__ == "__main__":
    main()
