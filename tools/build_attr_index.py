#!/usr/bin/env python3
"""Build the image -> {timeofday, weather, scene} attribute index.

Consumed by (a) the I-JEPA pretraining oversampler (stage 1) and
(b) the eval slicer (night / dawn-dusk / rainy slices).

The BDD train JSON is ~1.4 GB (it embeds lane/drivable polylines);
json.load needs >6 GB RAM. Files larger than --stream-threshold-mb are
therefore parsed with a streaming regex scan that extracts only the
image-level `"name" ... "attributes"` pairs (key order weather/scene/
timeofday as in the 2018 release). The result is count-checked; on any
mismatch we exit nonzero rather than emit a silently short index.

Output: <out-prefix>.csv (name,split,timeofday,weather,scene) + same as JSON.

Usage:
  python tools/build_attr_index.py \
      --train-json dataset/raw/bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_train.json \
      --val-json   dataset/raw/bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_val.json \
      --out-prefix dataset/yolo/attr_index
"""
import argparse, csv, json, os, re, sys
from collections import Counter
from pathlib import Path

# image-level entries look like: {"name": "xxx.jpg", "attributes": {"weather": "...", "scene": "...", "timeofday": "..."}, "timestamp": ...
ENTRY_RE = re.compile(
    r'"name":\s*"([^"]+)",\s*"attributes":\s*\{\s*'
    r'"weather":\s*"([^"]*)",\s*"scene":\s*"([^"]*)",\s*"timeofday":\s*"([^"]*)"')


def iter_entries_streaming(path, chunk_mb=8):
    buf = ""
    keep = 4096  # overlap so entries split across chunk boundaries aren't lost
    with open(path, "r", encoding="utf-8") as f:
        while True:
            chunk = f.read(chunk_mb * 1024 * 1024)
            if not chunk:
                break
            buf += chunk
            last = 0
            for m in ENTRY_RE.finditer(buf):
                yield m.group(1), m.group(2), m.group(3), m.group(4)
                last = m.end()
            buf = buf[max(last, len(buf) - keep):]


def iter_entries_json(path):
    with open(path) as f:
        for e in json.load(f):
            a = e.get("attributes", {})
            yield (e["name"], a.get("weather", "undefined"),
                   a.get("scene", "undefined"), a.get("timeofday", "undefined"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-json")
    ap.add_argument("--val-json")
    ap.add_argument("--out-prefix", required=True)
    ap.add_argument("--stream-threshold-mb", type=int, default=500)
    ap.add_argument("--expect-counts", default="",
                    help="optional 'train=69863,val=10000' hard check")
    args = ap.parse_args()

    expect = {}
    for kv in filter(None, args.expect_counts.split(",")):
        k, v = kv.split("=")
        expect[k] = int(v)

    rows = []
    for split, path in (("train", args.train_json), ("val", args.val_json)):
        if not path:
            continue
        size_mb = os.path.getsize(path) / 1e6
        streaming = size_mb > args.stream_threshold_mb
        it = iter_entries_streaming(path) if streaming else iter_entries_json(path)
        n = 0
        for name, weather, scene, tod in it:
            rows.append({"name": name, "split": split, "timeofday": tod,
                         "weather": weather, "scene": scene})
            n += 1
        print(f"{split}: {n} entries ({'streaming' if streaming else 'json.load'}, {size_mb:.0f} MB)")
        if split in expect and n != expect[split]:
            sys.exit(f"FLAG: {split} count {n} != expected {expect[split]}")
        if n == 0:
            sys.exit(f"FLAG: parsed 0 entries from {path}")

    names = [r["name"] for r in rows]
    if len(set(names)) != len(names):
        sys.exit("FLAG: duplicate image names in index")

    out = Path(args.out_prefix)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(f"{out}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["name", "split", "timeofday", "weather", "scene"])
        w.writeheader()
        w.writerows(rows)
    with open(f"{out}.json", "w") as f:
        json.dump({r["name"]: {k: r[k] for k in ("split", "timeofday", "weather", "scene")} for r in rows}, f)

    for split in ("train", "val"):
        sub = [r for r in rows if r["split"] == split]
        if not sub:
            continue
        print(f"[{split}] timeofday: {dict(Counter(r['timeofday'] for r in sub))}")
        print(f"[{split}] weather:   {dict(Counter(r['weather'] for r in sub))}")
    print(f"wrote {out}.csv and {out}.json ({len(rows)} rows)")


if __name__ == "__main__":
    main()
