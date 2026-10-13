#!/usr/bin/env python3
"""Generate all fixed, committed split lists (seed-deterministic) for the
new project scope (2026-08).

Rules encoded here (agreed 2026-08-14):
  - Drop any image with weather == "undefined" or timeofday == "undefined",
    from BOTH the BDD train pool and the BDD val pool, before anything else.
  - BDD's official val -> this project's TEST set, used as-is (no sampling),
    touched only for final scoring. Same discipline as the old project's
    val_final.txt, renamed to make the new role explicit.
  - This project's VAL is carved OUT OF BDD train: stratify the cleaned
    train pool by timeofday (daytime / night / dawn-dusk) and take 2.5% of
    EACH stratum independently, so val mirrors train's timeofday mix rather
    than being dominated by daytime. Weather is NOT a stratification axis
    (matches the old project's modelsel precedent - timeofday only).
  - Everything else cleaned-train minus val = the pool used for BOTH the
    unlabeled JEPA pretraining stage (pretrain.txt) and the 100%-label
    fine-tuning arm (train_100.txt) - same list, two names, exactly like
    the old project (a script either ignores or uses the labels; the pool
    of images is identical either way).
  - train_50 / train_25 / train_10: stratified-by-timeofday subsets of that
    pool, for the label-fraction ablation grid (full replication, per
    project decision 2026-08-14).

This does NOT do class filtering (car/truck/bus/motor/bike) - that's
tools/bdd_to_yolo.py's job, working from these name lists.

Usage:
  python tools/build_attr_index.py \
      --train-json dataset/raw/bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_train.json \
      --val-json   dataset/raw/bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_val.json \
      --out-prefix dataset/yolo/attr_index
  python tools/make_splits.py --attr-index dataset/yolo/attr_index.json \
      --repo-splits splits --yolo-splits dataset/yolo/splits --seed 42
"""
import argparse, json, random
from collections import Counter, defaultdict
from pathlib import Path

FRACTIONS = {"train_100": 1.0, "train_50": 0.5, "train_25": 0.25, "train_10": 0.10}


def stratified_sample(names_by_stratum, k_total, rng):
    """Proportional allocation per stratum, largest-remainder rounding.

    Used only for the train_50/25/10 label-fraction subsets, where we want
    an exact total count k_total spread across strata in proportion to
    their sizes. (The val 2.5%-per-stratum carve-out below is simpler and
    doesn't need this - each stratum is sampled independently to its own
    exact 2.5%, no shared total to hit.)
    """
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
    """Write both the bare-name repo copy and the YOLO-resolvable path list.

    img_subdir is "train" or "val" - which BDD image folder these names
    physically live in (independent of which new split they've been
    assigned to: this project's val/pretrain/train_* all point at "train"
    images; test points at "val" images).
    """
    Path(split_dir_repo).mkdir(parents=True, exist_ok=True)
    Path(split_dir_yolo).mkdir(parents=True, exist_ok=True)
    (Path(split_dir_repo) / f"{stem}.txt").write_text("\n".join(names) + "\n")
    images_root = Path(split_dir_yolo).resolve().parent / "images"
    (Path(split_dir_yolo) / f"{stem}.txt").write_text(
        "\n".join(str(images_root / img_subdir / n) for n in names) + "\n")


def clean(names, attrs):
    """Drop weather==undefined or timeofday==undefined."""
    kept = [n for n in names
            if attrs[n]["weather"] != "undefined" and attrs[n]["timeofday"] != "undefined"]
    dropped = len(names) - len(kept)
    return kept, dropped


def by_timeofday(names, attrs):
    d = defaultdict(list)
    for n in names:
        d[attrs[n]["timeofday"]].append(n)
    return d


def report(label, names, attrs):
    tod = Counter(attrs[n]["timeofday"] for n in names)
    wx = Counter(attrs[n]["weather"] for n in names)
    print(f"{label}: {len(names)} images | timeofday={dict(tod)} | weather={dict(wx)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--attr-index", required=True)
    ap.add_argument("--repo-splits", default="splits")
    ap.add_argument("--yolo-splits", default="dataset/yolo/splits")
    ap.add_argument("--val-fraction", type=float, default=0.025,
                    help="fraction taken from EACH timeofday stratum of train, into val")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    attrs = json.load(open(args.attr_index))
    rng = random.Random(args.seed)

    raw_train = sorted(n for n, a in attrs.items() if a["split"] == "train")
    raw_val = sorted(n for n, a in attrs.items() if a["split"] == "val")

    train, n_dropped_train = clean(raw_train, attrs)
    test, n_dropped_test = clean(raw_val, attrs)
    print(f"train: {len(raw_train)} raw -> dropped {n_dropped_train} (weather/timeofday undefined) -> {len(train)} clean")
    print(f"test:  {len(raw_val)} raw -> dropped {n_dropped_test} (weather/timeofday undefined) -> {len(test)} clean")

    # --- val: 2.5% of EACH timeofday stratum of cleaned train, independently ---
    train_by_tod = by_timeofday(train, attrs)
    print("clean train timeofday strata:", {k: len(v) for k, v in train_by_tod.items()})

    val = []
    pool_by_tod = {}
    for tod, names in train_by_tod.items():
        names_sorted = sorted(names)
        n_val = round(len(names_sorted) * args.val_fraction)
        this_val = set(rng.sample(names_sorted, n_val))
        val.extend(this_val)
        pool_by_tod[tod] = [n for n in names_sorted if n not in this_val]
    val = sorted(val)
    pool = sorted(n for names in pool_by_tod.values() for n in names)

    # --- write test (BDD val, cleaned, untouched) ---
    write_lists(test, "test", args.repo_splits, args.yolo_splits, "val")

    # --- write val (carved from train) ---
    write_lists(val, "val", args.repo_splits, args.yolo_splits, "train")

    # --- write pretrain / train_100 (same pool, two names - unlabeled vs labeled use) ---
    write_lists(pool, "pretrain", args.repo_splits, args.yolo_splits, "train")

    for stem, frac in FRACTIONS.items():
        if frac == 1.0:
            subset = pool
        else:
            subset = stratified_sample(pool_by_tod, round(len(pool) * frac), rng)
        write_lists(subset, stem, args.repo_splits, args.yolo_splits, "train")
        report(f"  {stem}", subset, attrs)

    report("test", test, attrs)
    report("val ", val, attrs)
    report("pool/pretrain/train_100", pool, attrs)

    manifest = {
        "seed": args.seed,
        "val_fraction_per_timeofday_stratum": args.val_fraction,
        "counts": {
            "train_raw": len(raw_train), "train_dropped_undefined": n_dropped_train,
            "train_clean": len(train),
            "val": len(val), "pool_pretrain_train100": len(pool),
            "test_raw": len(raw_val), "test_dropped_undefined": n_dropped_test,
            "test_clean": len(test),
        },
        "val_by_timeofday": dict(Counter(attrs[n]["timeofday"] for n in val)),
        "test_by_timeofday": dict(Counter(attrs[n]["timeofday"] for n in test)),
        "pool_by_timeofday": dict(Counter(attrs[n]["timeofday"] for n in pool)),
    }
    (Path(args.repo_splits) / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    print("manifest:", json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
