#!/usr/bin/env python3
"""BDD100K label JSON -> YOLO txt converter.

Class list is read from a YOLO data yaml (`names:`), so swapping in the
friend's Bdd10k_vehicle.yaml later requires no code change - just re-run.

Categories present in the JSON but absent from `names` are skipped (counted
in the stats). Images with zero kept boxes still get an EMPTY .txt file
(YOLO treats them as background images) so every arm sees identical data.

BDD100K images are uniformly 1280x720; we assume that and offer
--verify-sizes N to spot-check a random sample with PIL.

Usage (see RUNBOOK.md section 3):
  python tools/bdd_to_yolo.py \
      --labels-json dataset/raw/bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_val.json \
      --data-config configs/data/bdd100k_all10.yaml \
      --out-dir dataset/yolo/labels/val \
      --stats-out dataset/yolo/convert_stats_val.json
"""
import argparse, json, os, random, sys
from pathlib import Path

import yaml


def load_names(data_config: str):
    with open(data_config) as f:
        cfg = yaml.safe_load(f)
    names = cfg["names"]
    if isinstance(names, dict):  # {0: car, 1: bus, ...}
        names = [names[k] for k in sorted(names)]
    return {n: i for i, n in enumerate(names)}


def convert(entry, name_to_id, img_w, img_h):
    """Return (lines, n_kept, n_skipped_cat, n_degenerate)."""
    lines, kept, skip_cat, degen = [], 0, 0, 0
    for lab in entry.get("labels") or []:
        box = lab.get("box2d")
        if box is None:
            continue  # poly2d / lane / drivable entries
        cid = name_to_id.get(lab["category"])
        if cid is None:
            skip_cat += 1
            continue
        x1 = min(max(box["x1"], 0.0), img_w); x2 = min(max(box["x2"], 0.0), img_w)
        y1 = min(max(box["y1"], 0.0), img_h); y2 = min(max(box["y2"], 0.0), img_h)
        if x2 - x1 < 1.0 or y2 - y1 < 1.0:
            degen += 1
            continue
        cx = (x1 + x2) / 2.0 / img_w
        cy = (y1 + y2) / 2.0 / img_h
        w = (x2 - x1) / img_w
        h = (y2 - y1) / img_h
        lines.append(f"{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        kept += 1
    return lines, kept, skip_cat, degen


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels-json", required=True)
    ap.add_argument("--data-config", required=True, help="YOLO data yaml providing `names:`")
    ap.add_argument("--out-dir", required=True, help="output dir for YOLO .txt labels")
    ap.add_argument("--stats-out", default=None, help="write conversion stats JSON here")
    ap.add_argument("--img-width", type=int, default=1280)
    ap.add_argument("--img-height", type=int, default=720)
    ap.add_argument("--images-dir", default=None, help="needed only for --verify-sizes")
    ap.add_argument("--verify-sizes", type=int, default=0, metavar="N",
                    help="PIL-check N random images really are img-width x img-height")
    args = ap.parse_args()

    name_to_id = load_names(args.data_config)
    print(f"classes ({len(name_to_id)}): {list(name_to_id)}")

    with open(args.labels_json) as f:
        data = json.load(f)
    print(f"{len(data)} entries in {args.labels_json}")

    if args.verify_sizes:
        from PIL import Image
        assert args.images_dir, "--verify-sizes requires --images-dir"
        rng = random.Random(0)
        sample = rng.sample(data, min(args.verify_sizes, len(data)))
        bad = 0
        for e in sample:
            p = Path(args.images_dir) / e["name"]
            if not p.exists():
                continue
            wh = Image.open(p).size
            if wh != (args.img_width, args.img_height):
                bad += 1
                print(f"  SIZE MISMATCH {e['name']}: {wh}", file=sys.stderr)
        if bad:
            sys.exit(f"FLAG: {bad} images differ from {args.img_width}x{args.img_height} - do not assume fixed size.")
        print(f"size check ok on {len(sample)} sampled images")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    totals = {"images": 0, "boxes_kept": 0, "boxes_skipped_category": 0,
              "boxes_degenerate": 0, "empty_label_files": 0}
    per_class = {n: 0 for n in name_to_id}
    id_to_name = {i: n for n, i in name_to_id.items()}
    for e in data:
        lines, kept, skip, degen = convert(e, name_to_id, args.img_width, args.img_height)
        (out / (Path(e["name"]).stem + ".txt")).write_text("\n".join(lines) + ("\n" if lines else ""))
        totals["images"] += 1
        totals["boxes_kept"] += kept
        totals["boxes_skipped_category"] += skip
        totals["boxes_degenerate"] += degen
        if not lines:
            totals["empty_label_files"] += 1
        for ln in lines:
            per_class[id_to_name[int(ln.split()[0])]] += 1
    stats = {"args": vars(args), "totals": totals, "per_class": per_class}
    print(json.dumps(stats["totals"], indent=2))
    print(json.dumps(per_class, indent=2))
    if args.stats_out:
        Path(args.stats_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.stats_out).write_text(json.dumps(stats, indent=2))
        print(f"stats -> {args.stats_out}")


if __name__ == "__main__":
    main()
