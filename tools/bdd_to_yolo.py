#!/usr/bin/env python3
"""BDD100K label JSON -> YOLO txt converter.

Class list is read from a YOLO data yaml (`names:`), so swapping in the
friend's Bdd10k_vehicle.yaml later requires no code change - just re-run.

Categories present in the JSON but absent from `names` are skipped (counted
in the stats). Images with zero kept boxes still get an EMPTY .txt file
(YOLO treats them as background images) so every arm sees identical data.

BDD100K images are uniformly 1280x720; we assume that and offer
--verify-sizes N to spot-check a sample with PIL.

RAM: the train JSON is ~1.4 GB; json.load would spike to ~6-8 GB, which is
unsafe on the shared 32 GB server. We therefore stream entries one at a time
with ijson (in requirements.txt) - peak RAM stays ~tens of MB. Falls back to
json.load (with a loud warning) only if ijson is missing.

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

try:
    import ijson
except ImportError:
    ijson = None


def iter_entries(path):
    """Yield top-level array elements one at a time (RAM-safe)."""
    if ijson is not None:
        with open(path, "rb") as f:
            yield from ijson.items(f, "item")
    else:
        print("WARNING: ijson not installed - falling back to json.load "
              "(needs ~6-8 GB RAM for the train JSON). pip install ijson", file=sys.stderr)
        with open(path) as f:
            yield from json.load(f)


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
        # float() also handles decimal.Decimal, which ijson yields for numbers
        x1 = min(max(float(box["x1"]), 0.0), img_w); x2 = min(max(float(box["x2"]), 0.0), img_w)
        y1 = min(max(float(box["y1"]), 0.0), img_h); y2 = min(max(float(box["y2"]), 0.0), img_h)
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


    sizes_to_check = args.verify_sizes
    if sizes_to_check:
        from PIL import Image
        assert args.images_dir, "--verify-sizes requires --images-dir"

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    totals = {"images": 0, "boxes_kept": 0, "boxes_skipped_category": 0,
              "boxes_degenerate": 0, "empty_label_files": 0}
    per_class = {n: 0 for n in name_to_id}
    id_to_name = {i: n for n, i in name_to_id.items()}
    checked = 0
    for e in iter_entries(args.labels_json):
        if sizes_to_check and checked < sizes_to_check:
            ip = Path(args.images_dir) / e["name"]
            if ip.exists():
                wh = Image.open(ip).size
                if wh != (args.img_width, args.img_height):
                    sys.exit(f"FLAG: {e['name']} is {wh}, not "
                             f"{args.img_width}x{args.img_height} - do not assume fixed size.")
                checked += 1
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
    if sizes_to_check:
        print(f"size check ok on {checked} images")
    print(f"{totals['images']} entries in {args.labels_json}")
    stats = {"args": vars(args), "totals": totals, "per_class": per_class}
    print(json.dumps(stats["totals"], indent=2))
    print(json.dumps(per_class, indent=2))
    if args.stats_out:
        Path(args.stats_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.stats_out).write_text(json.dumps(stats, indent=2))
        print(f"stats -> {args.stats_out}")


if __name__ == "__main__":
    main()
