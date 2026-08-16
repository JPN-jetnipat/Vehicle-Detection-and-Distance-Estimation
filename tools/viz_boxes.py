#!/usr/bin/env python3
"""Overlay YOLO-format labels on images to eyeball converter correctness.

Usage:
  python tools/viz_boxes.py --images-dir dataset/yolo/images/val \
      --labels-dir dataset/yolo/labels/val --data-config configs/data/bdd100k_vehicle5.yaml \
      --num 12 --out-dir viz_out
"""
import argparse, random
from pathlib import Path

import yaml
from PIL import Image, ImageDraw

COLORS = ["#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231", "#911eb4",
          "#46f0f0", "#f032e6", "#bcf60c", "#fabebe"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images-dir", required=True)
    ap.add_argument("--labels-dir", required=True)
    ap.add_argument("--data-config", required=True)
    ap.add_argument("--num", type=int, default=12)
    ap.add_argument("--names", nargs="*", help="specific image filenames instead of random sample")
    ap.add_argument("--out-dir", default="viz_out")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    names = yaml.safe_load(open(args.data_config))["names"]
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]

    labels = sorted(Path(args.labels_dir).glob("*.txt"))
    if args.names:
        chosen = [Path(args.labels_dir) / (Path(n).stem + ".txt") for n in args.names]
    else:
        nonempty = [p for p in labels if p.stat().st_size > 0]
        chosen = random.Random(args.seed).sample(nonempty, min(args.num, len(nonempty)))

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for lp in chosen:
        ip = Path(args.images_dir) / (lp.stem + ".jpg")
        if not ip.exists():
            print(f"skip (no image): {ip}")
            continue
        img = Image.open(ip).convert("RGB")
        d = ImageDraw.Draw(img)
        W, H = img.size
        for line in lp.read_text().splitlines():
            cid, cx, cy, w, h = line.split()
            cid, cx, cy, w, h = int(cid), float(cx) * W, float(cy) * H, float(w) * W, float(h) * H
            x1, y1, x2, y2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
            c = COLORS[cid % len(COLORS)]
            d.rectangle([x1, y1, x2, y2], outline=c, width=2)
            d.text((x1 + 2, max(0, y1 - 12)), names[cid], fill=c)
        img.save(out / f"{lp.stem}_viz.jpg", quality=92)
        print(f"wrote {out / (lp.stem + '_viz.jpg')}")


if __name__ == "__main__":
    main()
