#!/usr/bin/env python3
"""Run a YOLOv5 .pt on an image list and dump predictions for the evaluator.

Same NMS settings for EVERY arm (fairness): conf 0.001, IoU 0.6, max 300
dets - the standard mAP protocol. Output: one txt per image,
lines "cls cx cy w h conf" normalized to the ORIGINAL image size, i.e. the
exact format evaluation/eval_detections.py consumes.

Needs the vendored yolov5/ repo (imported, not subprocessed) + torch.
Runs fine on CPU for small lists; use --device 0 on the server
(~4 min for 10k val images at 640 on the A40).

Usage:
  python evaluation/run_inference.py --weights runs_jepa/stage3/baseline_coco_100/weights/best.pt \
      --image-list splits/val_final.txt --images-root dataset/yolo \
      --out-dir runs_jepa/preds/baseline_coco_100_val --device 0
"""
import argparse, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "yolov5"))

import numpy as np
import torch
from models.common import DetectMultiBackend          # noqa: E402
from utils.augmentations import letterbox            # noqa: E402
from utils.general import non_max_suppression, scale_boxes, xyxy2xywhn  # noqa: E402
from utils.torch_utils import select_device          # noqa: E402
import cv2                                            # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--image-list", required=True,
                    help="txt of image paths; relative entries resolved against --images-root")
    ap.add_argument("--images-root", default=".", help="e.g. dataset/yolo (list has ./images/val/x.jpg)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf-thres", type=float, default=0.001)
    ap.add_argument("--iou-thres", type=float, default=0.6)
    ap.add_argument("--max-det", type=int, default=300)
    ap.add_argument("--device", default="cpu", help="'cpu' or CUDA index like '0'")
    ap.add_argument("--half", action="store_true", help="fp16 inference (GPU only)")
    args = ap.parse_args()

    device = select_device(args.device)
    model = DetectMultiBackend(args.weights, device=device, fp16=args.half)
    stride = int(model.stride)
    model.eval()

    paths = []
    for line in open(args.image_list):
        line = line.strip()
        if not line:
            continue
        p = Path(line)
        paths.append(p if p.is_absolute() else Path(args.images_root) / p)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    n_det = 0
    with torch.no_grad():
        for i, ip in enumerate(paths):
            im0 = cv2.imread(str(ip))
            if im0 is None:
                raise FileNotFoundError(ip)
            im = letterbox(im0, args.imgsz, stride=stride, auto=False)[0]
            im = im.transpose((2, 0, 1))[::-1]  # HWC BGR -> CHW RGB
            im = torch.from_numpy(np.ascontiguousarray(im)).to(device)
            im = (im.half() if model.fp16 else im.float()) / 255.0
            pred = model(im[None])
            pred = non_max_suppression(pred, args.conf_thres, args.iou_thres,
                                       max_det=args.max_det)[0]
            lines = []
            if len(pred):
                pred[:, :4] = scale_boxes(im.shape[1:], pred[:, :4], im0.shape).round()
                boxes = xyxy2xywhn(pred[:, :4], w=im0.shape[1], h=im0.shape[0])
                for (cx, cy, w, h), conf, cls in zip(boxes.tolist(),
                                                     pred[:, 4].tolist(),
                                                     pred[:, 5].tolist()):
                    lines.append(f"{int(cls)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {conf:.6f}")
            (out / (ip.stem + ".txt")).write_text("\n".join(lines) + ("\n" if lines else ""))
            n_det += len(lines)
            if (i + 1) % 500 == 0:
                print(f"{i + 1}/{len(paths)} images, {n_det} detections so far", flush=True)
    print(f"done: {len(paths)} images, {n_det} detections -> {out}")


if __name__ == "__main__":
    main()
