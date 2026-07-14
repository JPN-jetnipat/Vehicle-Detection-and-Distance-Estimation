#!/usr/bin/env python3
"""Generate all fixed, committed split lists (seed-deterministic).

From BDD100K *train* (~70k) this produces, stratified by timeofday:
  splits/modelsel.txt        model-selection set (default 2500) - stands in
                             for val during development; BDD val is touched
                             only for final scoring.
  splits/pretrain.txt        SSL/distillation pool = train MINUS modelsel.
                             (Conservative choice: excluding modelsel from
                             unlabeled pretraining keeps model selection
                             untainted; costs ~2.5k of 70k images.)
  splits/train_100.txt       fine-tune pool = same as pretrain.txt (labels used)
  splits/train_50.txt        stratified 50% of train_100
  splits/train_25.txt        stratified 25%
  splits/train_10.txt        stratified 10%
  splits/val_final.txt       all BDD val names (final scoring only)

Lists contain YOLO-resolvable relative paths: ./images/<split>/<name>.jpg
(relative to the data-yaml `path:` = dataset/yolo). The same lists are also
mirrored as bare-name .txt into the repo's committed `splits/` dir - the
dataset/ dir is gitignored, so tracked copies live in the repo and are
materialized into dataset/yolo/splits/ by this script on any machine.

Usage:
  python tools/make_splits.py --attr-index dataset/yolo/attr_index.json \
      --repo-splits splits --yolo-splits dataset/yolo/splits --seed 42
"""
import argparse, json, random
from collections import defaultdict
from pathlib import Path

FRACTIONS = {"train_100": 1.0, "train_50": 0.5, "train_25": 0.25, "train_10": 0.10}


def stratified_sample(names_by_stratum, k_total, rng):
    """Proportional allocation per stratum, largest-remainder rounding."""
    total = sum(len(v) for v in names_by_stratum.values())
    alloc, remainders = {}, []
    for s, names in names_by_stratum.items():
        exact = k_total * len(names) / total
        alloc[s] = int(exact)
        remainders.append((exact - int(exact), s))
    short = k_total - sum(alloc.values())
    for _, s in sorted(remainders, reverse=True)[:short]:
        alloc[s] += 1
    out = []
    for s, names in names_by_stratum.items():
        out.extend(rng.sample(sorted(names), min(alloc[s], len(names))))
    return sorted(out)


def write_lists(names, stem, split_dir_repo, split_dir_yolo, img_subdir):
    Path(split_dir_repo).mkdir(parents=True, exist_ok=True)
    Path(split_dir_yolo).mkdir(parents=True, exist_ok=True)
    (Path(split_dir_repo) / f"{stem}.txt").write_text("\n".join(names) + "\n")
    (Path(split_dir_yolo) / f"{stem}.txt").write_text(
        "\n".join(f"./images/{img_subdir}/{n}" for n in names) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--attr-index", required=True)
    ap.add_argument("--repo-splits", default="splits")
    ap.add_argument("--yolo-splits", default="dataset/yolo/splits")
    ap.add_argument("--modelsel-size", type=int, default=2500)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    attrs = json.load(open(args.attr_index))
    rng = random.Random(args.seed)

    train = sorted(n for n, a in attrs.items() if a["split"] == "train")
    val = sorted(n for n, a in attrs.items() if a["split"] == "val")
    by_tod = defaultdict(list)
    for n in train:
        by_tod[attrs[n]["timeofday"]].append(n)
    print("train timeofday strata:", {k: len(v) for k, v in by_tod.items()})

    modelsel = set(stratified_sample(by_tod, args.modelsel_size, rng))
    pool = [n for n in train if n not in modelsel]
    pool_by_tod = defaultdict(list)
    for n in pool:
        pool_by_tod[attrs[n]["timeofday"]].append(n)

    write_lists(sorted(modelsel), "modelsel", args.repo_splits, args.yolo_splits, "train")
    write_lists(pool, "pretrain", args.repo_splits, args.yolo_splits, "train")
    write_lists(val, "val_final", args.repo_splits, args.yolo_splits, "val")

    for stem, frac in FRACTIONS.items():
        if frac == 1.0:
            subset = pool
        else:
            subset = stratified_sample(pool_by_tod, round(len(pool) * frac), rng)
        write_lists(subset, stem, args.repo_splits, args.yolo_splits, "train")
        tod = defaultdict(int)
        for n in subset:
            tod[attrs[n]["timeofday"]] += 1
        print(f"{stem}: {len(subset)} images, timeofday={dict(tod)}")

    manifest = {"seed": args.seed, "modelsel_size": args.modelsel_size,
                "counts": {"train_total": len(train), "modelsel": len(modelsel),
                           "pool": len(pool), "val_final": len(val)}}
    (Path(args.repo_splits) / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    print("manifest:", json.dumps(manifest))


if __name__ == "__main__":
    main()
