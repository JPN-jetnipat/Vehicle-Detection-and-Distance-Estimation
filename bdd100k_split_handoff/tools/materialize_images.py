#!/usr/bin/env python3
"""Flatten a (possibly messily-nested) BDD image folder into
dataset/yolo/images/<split>/ via per-file symlinks (no copying - images
aren't duplicated, just re-exposed under one flat directory so the paths
written by tools/make_splits.py resolve).

Why this exists: some BDD100K mirrors (e.g. the Kaggle solesensei copy the
old project's RUNBOOK.md flags as a "may differ" risk) don't ship
images/100k/train/ as one flat folder of 70,000 .jpg - they nest them into
arbitrary subfolders (testA/testB/trainA/trainB alongside some loose files).
A single `ln -s .../images/100k/train dataset/yolo/images/train` (the old
RUNBOOK's command, which assumes a flat folder) then only resolves whatever
fraction happens to sit at the top level - silently, since a broken symlink
target just looks like a missing file later, at training time, not now.

This scans ALL subfolders under --raw-split-dir recursively, one level or
many, and symlinks every *.jpg found by its basename. Exits nonzero on a
duplicate basename (would indicate real data corruption, not just messy
nesting) rather than silently overwriting.

Usage:
  python tools/materialize_images.py \
      --raw-split-dir dataset/raw/bdd100k/bdd100k/images/100k/train \
      --out-dir dataset/yolo/images/train
"""
import argparse, sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-split-dir", required=True, help="e.g. dataset/raw/.../images/100k/train")
    ap.add_argument("--out-dir", required=True, help="e.g. dataset/yolo/images/train")
    ap.add_argument("--ext", default=".jpg")
    args = ap.parse_args()

    src_root = Path(args.raw_split_dir).resolve()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    seen = {}
    n_linked = n_exists = n_conflict = 0
    for f in src_root.rglob(f"*{args.ext}"):
        if not f.is_file():
            continue
        name = f.name
        if name in seen and seen[name] != f:
            sys.exit(f"FLAG: duplicate basename {name} found at both {seen[name]} and {f} "
                      f"- this is real data ambiguity, not just messy nesting. Investigate before continuing.")
        seen[name] = f
        link = out / name
        if link.exists() or link.is_symlink():
            n_exists += 1
            continue
        link.symlink_to(f)
        n_linked += 1

    print(f"scanned {src_root}: {len(seen)} unique images found")
    print(f"{n_linked} new symlinks created, {n_exists} already existed in {out}")
    if n_conflict:
        print(f"WARNING: {n_conflict} conflicts (see above)")


if __name__ == "__main__":
    main()
