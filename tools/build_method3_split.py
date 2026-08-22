#!/usr/bin/env python3
"""Build the Method 3 training split: low-light augmentation x IRFS oversampling.

WHAT METHOD 3 IS
----------------
    Method 1 (Baby) : low-light augmentation applied to every training image,
                      trained on ORIGINAL + AUGMENTED (a 2x pool).
    Method 2 (JPN)  : IRFS instance-aware repeat-factor oversampling, restricted
                      to bike/motor, applied to the original pool.
    Method 3 (Field): both, composed -- the pool is ORIGINAL + AUGMENTED, and
                      IRFS repeat factors are then applied on top of that pool.

This script is stage 2. Stage 1 is `tools/augment_lowlight.py`, which must have
already written the low-light copies and their image list.

WHY THE COMPOSITION IS CLEAN (worth stating in the write-up)
------------------------------------------------------------
The low-light augmentation is pixel-level: it never moves, adds or removes a
box. So the augmented pool has EXACTLY the same class distribution as the
original pool, and doubling the pool leaves both IRFS frequency terms unchanged:

    f_(i,c) = (2 * images_with_c) / (2 * N)          = unchanged
    f_(b,c) = (2 * boxes_of_c) / (2 * total_boxes)   = unchanged
    => r_c   = max(1, sqrt(t / sqrt(f_i_c * f_b_c))) = unchanged

So Method 3's per-class repeat factors are provably identical to Method 2's
(bike 2.417, motor 3.388 at t = 0.122). Method 3 therefore differs from
Method 2 in exactly one thing -- half its samples are night-degraded -- and
from Method 1 in exactly one thing -- bike/motor images are oversampled.
That is what makes the three-arm ablation interpretable. This script ASSERTS
that identity when `--method2-report` is given, so a silent drift in either
arm's inputs gets caught rather than quietly reinterpreted.

The sampling math itself is not reimplemented here: it is imported from
`tools/compute_repeat_factors.py`, a verbatim copy of the Method 2 arm's module.

OUTPUTS
-------
    dataset/yolo/splits/train_100_method3.txt      repeated image list
    configs/data/bdd100k_vehicle5_method3.yaml     dataset yaml pointing at it
    results/sampling_reports/method3_distribution.json

Usage (from the repo root):
    python tools/build_method3_split.py
    python tools/build_method3_split.py --dry-run          # counts only, writes nothing
    python tools/build_method3_split.py \
        --method2-report ../irfs_augmentation/irfs_distribution.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import compute_repeat_factors as rf  # noqa: E402  (needs the sys.path line above)

REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_ORIG_LIST = "dataset/yolo/splits/train_100.txt"
DEFAULT_AUG_LIST = "dataset/yolo/splits/train_100_lowlight.txt"
DEFAULT_ORIG_LABELS = "dataset/yolo/labels/train"
DEFAULT_AUG_LABELS = "dataset/lowlight/labels/train"
DEFAULT_BASE_DATA_YAML = "configs/data/bdd100k_vehicle5.yaml"
DEFAULT_SPLIT_OUT = "dataset/yolo/splits/train_100_method3.txt"
DEFAULT_DATA_YAML_OUT = "configs/data/bdd100k_vehicle5_method3.yaml"
DEFAULT_REPORT_OUT = "results/sampling_reports/method3_distribution.json"

# t = 0.122 and the bike/motor restriction are NOT this script's choices --
# they are the Method 2 arm's settings, carried over so the two arms'
# oversampling is identical. Do not retune them here without retuning Method 2.
DEFAULT_THRESHOLD = 0.122
DEFAULT_TARGET_CLASSES = ["bike", "motor"]
DEFAULT_SEED = 42


def resolve(path_str: str) -> Path:
    p = Path(path_str).expanduser()
    return p if p.is_absolute() else (REPO_ROOT / p)


def repo_rel(path: Path) -> str:
    """Repo-root-relative POSIX path. os.path.abspath, not Path.resolve(): see
    the same function in tools/augment_lowlight.py -- resolve() would follow a
    dataset/lowlight -> /disk2/... symlink and bake an absolute path into the
    generated split list and dataset yaml."""
    p = Path(os.path.abspath(str(path)))
    try:
        return p.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return p.as_posix()


def read_list(path: Path) -> list[str]:
    if not path.exists():
        sys.exit(f"FLAG: list not found: {repo_rel(path)}")
    return [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def merge_counts(a: rf.ClassCounts, b: rf.ClassCounts) -> rf.ClassCounts:
    overlap = set(a.per_image_classes) & set(b.per_image_classes)
    if overlap:
        sys.exit(f"FLAG: {len(overlap)} image paths appear in BOTH lists, e.g. {sorted(overlap)[:3]}. "
                 "The augmented copies must live under a different directory than the originals.")
    return rf.ClassCounts(
        image_names=a.image_names + b.image_names,
        per_image_classes={**a.per_image_classes, **b.per_image_classes},
        class_names=a.class_names,
    )


def check_twin_consistency(
    counts_orig: rf.ClassCounts, counts_aug: rf.ClassCounts, max_report: int = 5
) -> None:
    """Every augmented image must carry the same labels as the original it came from.

    This is the check that catches a half-finished stage-1 run: if a label copy
    was missed, the augmented twin reads as a background image, its repeat factor
    silently drops to 1, and Method 3 quietly under-samples the rare classes it
    exists to boost.
    """
    by_stem_orig = {Path(n).stem: c for n, c in counts_orig.per_image_classes.items()}
    mismatched: list[str] = []
    for name, classes in counts_aug.per_image_classes.items():
        stem = Path(name).stem
        if stem not in by_stem_orig:
            mismatched.append(f"{name}: no original with this stem")
        elif by_stem_orig[stem] != classes:
            mismatched.append(f"{name}: labels {classes} != original {by_stem_orig[stem]}")
    if mismatched:
        print(f"FLAG: {len(mismatched)} augmented images have labels that differ from their original:")
        for line in mismatched[:max_report]:
            print(f"  {line}")
        sys.exit("Rerun tools/augment_lowlight.py (it resumes; existing images are kept) before continuing.")


def write_split(image_names: list[str], repeat_counts: dict[str, int], out_path: Path) -> int:
    lines: list[str] = []
    for name in image_names:  # preserve input ordering, same as Method 2's builder
        lines.extend([name] * repeat_counts[name])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(lines)


def write_data_yaml(base: dict, base_path: Path, split_out: Path, out_path: Path) -> None:
    """Mirror configs/data/bdd100k_vehicle5.yaml, swapping only `train:`.

    `path:` and `val:` are copied verbatim so Method 3 validates against the
    same split as every other arm -- the only thing that changes between arms
    is the training pool.
    """
    train_rel = Path(repo_rel(split_out))
    path_rel = Path(base["path"])
    try:
        train_value = train_rel.relative_to(path_rel).as_posix()
    except ValueError:
        sys.exit(f"FLAG: {repo_rel(split_out)} is not under the dataset root '{base['path']}' "
                 f"declared in {base_path.name}; ultralytics would not resolve it.")

    lines = [
        "# Method 3 (Field) - low-light augmentation on every image  x  IRFS bike/motor oversampling.",
        "# AUTO-GENERATED by tools/build_method3_split.py - edit that script, not this file.",
        f"# train: built from {DEFAULT_ORIG_LIST} + {DEFAULT_AUG_LIST}, then IRFS-repeated.",
        f"# path/val/names copied unchanged from {base_path.name} so every arm validates identically.",
        f"path: {base['path']}",
        f"train: {train_value}",
        f"val: {base['val']}",
        "",
        "names:",
    ]
    for class_id, class_name in sorted(base["names"].items()):
        lines.append(f"  {class_id}: {class_name}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize(counts: rf.ClassCounts, repeat_counts: dict[str, int], subset_prefix: str | None = None):
    """Per-class image/instance totals before and after repetition."""
    images_before = {c: 0 for c in counts.class_names}
    inst_before = {c: 0 for c in counts.class_names}
    images_after = {c: 0 for c in counts.class_names}
    inst_after = {c: 0 for c in counts.class_names}
    n_before = n_after = 0
    for name, classes in counts.per_image_classes.items():
        if subset_prefix is not None and not name.startswith(subset_prefix):
            continue
        rep = repeat_counts[name]
        n_before += 1
        n_after += rep
        for c, k in classes.items():
            images_before[c] += 1
            inst_before[c] += k
            images_after[c] += rep
            inst_after[c] += k * rep
    return images_before, inst_before, images_after, inst_after, n_before, n_after


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--orig-list", default=DEFAULT_ORIG_LIST)
    ap.add_argument("--aug-list", default=DEFAULT_AUG_LIST)
    ap.add_argument("--orig-labels", default=DEFAULT_ORIG_LABELS)
    ap.add_argument("--aug-labels", default=DEFAULT_AUG_LABELS)
    ap.add_argument("--base-data-yaml", default=DEFAULT_BASE_DATA_YAML)
    ap.add_argument("--split-out", default=DEFAULT_SPLIT_OUT)
    ap.add_argument("--data-yaml-out", default=DEFAULT_DATA_YAML_OUT)
    ap.add_argument("--report-out", default=DEFAULT_REPORT_OUT)
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                    help=f"t in r_c = max(1, sqrt(t / sqrt(f_i_c * f_b_c))). Default {DEFAULT_THRESHOLD}, "
                         "matching the Method 2 arm. Changing it breaks comparability with Method 2.")
    ap.add_argument("--target-classes", nargs="+", default=DEFAULT_TARGET_CLASSES,
                    help="Classes eligible for oversampling; everything else is forced to r_c = 1.")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Seed for stochastic rounding.")
    ap.add_argument("--method2-report", default=None,
                    help="Path to Method 2's irfs_distribution.json. If given, this script asserts that "
                         "Method 3's per-class repeat factors match it exactly (they must -- see the "
                         "module docstring) and fails loudly if they do not.")
    ap.add_argument("--dry-run", action="store_true", help="Print the numbers, write nothing.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    base_path = resolve(args.base_data_yaml)
    if not base_path.exists():
        sys.exit(f"FLAG: base data yaml not found: {repo_rel(base_path)}")
    base = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    class_names = {int(k): v for k, v in base["names"].items()}

    orig_list = resolve(args.orig_list)
    aug_list = resolve(args.aug_list)
    orig_labels = resolve(args.orig_labels)
    aug_labels = resolve(args.aug_labels)

    for d in (orig_labels, aug_labels):
        if not d.is_dir():
            sys.exit(f"FLAG: labels dir not found: {repo_rel(d)}. Run tools/augment_lowlight.py first.")

    counts_orig = rf.load_class_counts(orig_labels, orig_list, class_names)
    counts_aug = rf.load_class_counts(aug_labels, aug_list, class_names)

    n_orig, n_aug = len(counts_orig.image_names), len(counts_aug.image_names)
    print(f"Original pool : {n_orig:>7,} images  ({repo_rel(orig_list)})")
    print(f"Low-light pool: {n_aug:>7,} images  ({repo_rel(aug_list)})")
    if n_orig != n_aug:
        print(f"FLAG: pools differ in size by {abs(n_orig - n_aug)} images. Method 1's recipe augments "
              "EVERY training image, so these should match. Stage 1 probably did not finish -- rerun "
              "tools/augment_lowlight.py (it resumes) before continuing.")
        sys.exit(1)

    check_twin_consistency(counts_orig, counts_aug)
    print("Twin check    : every low-light copy carries its original's labels. OK")

    counts = merge_counts(counts_orig, counts_aug)

    class_factors = rf.compute_irfs_factors(counts, args.threshold, args.target_classes)
    image_r = rf.compute_image_repeat_factors(counts, class_factors)
    repeat_counts = rf.compute_repeat_counts(image_r, seed=args.seed)

    # --- The identity claim from the module docstring, actually checked -----
    if args.method2_report:
        m2_path = Path(args.method2_report).expanduser()
        if not m2_path.is_absolute():
            m2_path = (REPO_ROOT / m2_path).resolve()
        if not m2_path.exists():
            sys.exit(f"FLAG: --method2-report not found: {m2_path}")
        m2 = json.loads(m2_path.read_text(encoding="utf-8"))
        drift = []
        for cid, cname in sorted(class_names.items()):
            ours = class_factors[cid]["r_c"]
            theirs = m2["per_class"].get(cname, {}).get("repeat_factor_r_c")
            if theirs is None:
                drift.append(f"{cname}: absent from Method 2's report")
            elif abs(ours - theirs) > 1e-9:
                drift.append(f"{cname}: Method 3 r_c={ours:.6f} vs Method 2 r_c={theirs:.6f}")
        if drift:
            print("FLAG: repeat factors do NOT match Method 2's. They must, if both arms share t, the "
                  "target classes and the same underlying labels. Investigate before training:")
            for line in drift:
                print(f"  {line}")
            sys.exit(1)
        print(f"Method 2 check: per-class repeat factors identical to {m2_path.name}. OK")

    # --- Report -------------------------------------------------------------
    img_b, inst_b, img_a, inst_a, n_b, n_a = summarize(counts, repeat_counts)

    print()
    print(f"{'class':<8}{'r_c':>8}{'imgs before':>14}{'imgs after':>13}{'inst before':>14}{'inst after':>13}")
    print("-" * 70)
    for cid, cname in sorted(class_names.items()):
        print(f"{cname:<8}{class_factors[cid]['r_c']:>8.3f}{img_b[cid]:>14,}{img_a[cid]:>13,}"
              f"{inst_b[cid]:>14,}{inst_a[cid]:>13,}")
    print("-" * 70)
    print(f"{'TOTAL':<8}{'':>8}{n_b:>14,}{n_a:>13,}")
    print()
    print(f"Samples per epoch: {n_a:,}")
    print(f"  vs Method 1 (orig + low-light, no oversampling): {n_b:,}")
    print(f"  vs Method 2 (IRFS on originals only)           : ~{n_a // 2:,}")

    report = {
        "arm": "method3_lowlight_irfs",
        "composition": "Method 1 (low-light on every image, original + augmented pool) "
                       "x Method 2 (IRFS oversampling restricted to bike/motor)",
        "mode": "irfs",
        "threshold": args.threshold,
        "seed": args.seed,
        "restricted_to": sorted(args.target_classes),
        "excluded_classes_forced_r1": sorted(set(class_names.values()) - set(args.target_classes)),
        "pool": {
            "original_images": n_orig,
            "lowlight_images": n_aug,
            "combined_images_before_repeat": n_b,
            "samples_per_epoch_after_repeat": n_a,
        },
        "per_class": {
            cname: {
                **{k: v for k, v in class_factors[cid].items() if k != "r_c"},
                "repeat_factor_r_c": class_factors[cid]["r_c"],
                "image_count_before": img_b[cid],
                "instance_count_before": inst_b[cid],
                "image_count_after": img_a[cid],
                "instance_count_after": inst_a[cid],
            }
            for cid, cname in sorted(class_names.items())
        },
    }

    if args.dry_run:
        print("\n--dry-run: nothing written.")
        return

    split_out = resolve(args.split_out)
    n_lines = write_split(counts.image_names, repeat_counts, split_out)
    assert n_lines == n_a, f"split line count {n_lines} != computed total {n_a}"

    data_yaml_out = resolve(args.data_yaml_out)
    write_data_yaml(base, base_path, split_out, data_yaml_out)

    report_out = resolve(args.report_out)
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print()
    print(f"split       -> {repo_rel(split_out)}  ({n_lines:,} lines)")
    print(f"data yaml   -> {repo_rel(data_yaml_out)}")
    print(f"report      -> {repo_rel(report_out)}")
    print()
    print("Next (from the repo root):")
    print("  python tools/train_yolo.py --hyp configs/hyp/set3_combined_smoke.yaml \\")
    print(f"      --data {repo_rel(data_yaml_out)} --name set3_combined_smoke")
    print("  nohup python tools/train_yolo.py --hyp configs/hyp/set3_combined.yaml \\")
    print(f"      --data {repo_rel(data_yaml_out)} --name set3_combined > set3_combined.log 2>&1 &")


if __name__ == "__main__":
    main()
