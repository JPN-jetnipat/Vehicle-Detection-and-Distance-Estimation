"""Render per-split mAP comparison tables for one arm against a baseline arm.

Separate from train_arm.py on purpose: training only needs to append rows to
results/metrics_all_arms.csv, and reporting/formatting is a different concern
that you want to re-run on demand (re-print after a later arm finishes, tweak
which arm is the baseline, regenerate for a report) without re-training or
re-evaluating anything.

Reads results/metrics_all_arms.csv (already populated by train_arm.py /
eval_arm.py) and prints one table per scope - overall (all vehicle classes
pooled) plus one per individual class (car/truck/bus/motor/bike) - each table
showing mAP50 / mAP75 / mAP50-95 for every val_* split, arm vs. baseline vs.
delta.

Usage:
    python scripts/report_arm_results.py --arm condprep_v1
    python scripts/report_arm_results.py --arm condprep_v1 --baseline set3_combined
    python scripts/report_arm_results.py --arm condprep_v1 --scorer pycocotools
    python scripts/report_arm_results.py --arm condprep_v1 --out results/reports/condprep_v1_vs_set3.md
"""

import argparse
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_METRICS_CSV = ROOT / "results" / "metrics_all_arms.csv"

METRIC_TYPES = ["mAP50", "mAP75", "mAP50-95"]
CLASSES = ["car", "truck", "bus", "motor", "bike"]

# Ordered timeofday splits, then weather splits. Matches every arm config's
# eval_splits (see configs/experiments/*.yaml). n is the fixed size of the
# held-out test pool per split (same regardless of arm/preprocessing - only
# the images' pixels differ, not the count) - from
# materialize_test_splits.py / materialize_test_weather_splits.py's own
# printed counts. Recompute with `ls dataset/yolo/images/test_<split>|wc -l`
# if the underlying split lists ever change.
SPLIT_ORDER = [
    ("val_day", 4264),
    ("val_night", 3863),
    ("val_dawndusk", 714),
    ("val_clear", 5345),
    ("val_overcast", 1239),
    ("val_partlycloudy", 738),
    ("val_snowy", 769),
    ("val_rainy", 737),
    ("val_foggy", 13),
]
LOW_N_THRESHOLD = 50


def load_latest_values(csv_path: Path, scorer: str) -> dict:
    """Return {(arm, dataset_split, metric_name): value}, keeping only the
    most recent timestamp per key (a re-run of the same arm/split appends a
    new row rather than replacing the old one)."""
    latest = {}
    with csv_path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["scorer"] != scorer:
                continue
            key = (row["arm_name"], row["dataset_split"], row["metric_name"])
            prev = latest.get(key)
            if prev is None or row["timestamp"] > prev[1]:
                latest[key] = (float(row["value"]), row["timestamp"])
    return {k: v[0] for k, v in latest.items()}


def metric_name_for(metric_type: str, scope: str) -> str:
    return metric_type if scope == "overall" else f"{metric_type}_{scope}"


def format_value(v):
    return f"{v:.3f}" if v is not None else "   —  "


def format_delta(arm_v, base_v):
    if arm_v is None or base_v is None:
        return "   —  "
    d = arm_v - base_v
    return f"{d:+.3f}"


def build_rows(scope: str, values: dict, arm: str, baseline: str):
    """Return (header, rows) as lists of plain string cells - one row per split."""
    header = ["dataset_split", "n"]
    for mt in METRIC_TYPES:
        header += [f"{mt}\n{baseline}", f"{mt}\n{arm}", f"{mt}\ndelta"]

    rows = []
    for split, n in SPLIT_ORDER:
        row = [split, str(n)]
        for mt in METRIC_TYPES:
            mname = metric_name_for(mt, scope)
            base_v = values.get((baseline, split, mname))
            arm_v = values.get((arm, split, mname))
            row += [format_value(base_v), format_value(arm_v), format_delta(arm_v, base_v)]
        note = "  (n too low for a reliable mAP)" if n < LOW_N_THRESHOLD else ""
        rows.append((row, note))
    return header, rows


def render_scope_table_text(scope: str, values: dict, arm: str, baseline: str) -> str:
    """Plain fixed-width table for terminal viewing."""
    label = "OVERALL (all vehicle classes)" if scope == "overall" else scope
    header, rows = build_rows(scope, values, arm, baseline)
    header = [h.split("\n")[-1] if "\n" in h else h for h in header]  # single-line header for text mode
    # group header: metric type names above the baseline/arm/delta sub-header
    group_header = ["", ""] + sum(([mt, "", ""] for mt in METRIC_TYPES), [])

    widths = [max(len(header[i]), *(len(r[0][i]) for r in rows)) for i in range(len(header))]

    def fmt_row(cells):
        return "  ".join(c.ljust(w) for c, w in zip(cells, widths))

    lines = [f"=== {label} ===", ""]
    lines.append(fmt_row(group_header).rstrip())
    lines.append(fmt_row(header))
    lines.append("  ".join("-" * w for w in widths))
    for row, note in rows:
        lines.append(fmt_row(row) + note)
    return "\n".join(lines)


def render_scope_table_md(scope: str, values: dict, arm: str, baseline: str) -> str:
    """Markdown pipe-table, for a report file meant to be viewed rendered."""
    label = "OVERALL (all vehicle classes)" if scope == "overall" else scope
    header, rows = build_rows(scope, values, arm, baseline)
    header = [h.replace("\n", " ") for h in header]
    lines = [f"### {label}", ""]
    lines.append(" | ".join(header))
    lines.append(" | ".join("---" for _ in header))
    for row, note in rows:
        lines.append(" | ".join(row) + note)
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arm", required=True, help="arm_name to report (must already have rows in the metrics csv)")
    parser.add_argument("--baseline", default="set3_combined", help="arm_name to compare against (default: set3_combined)")
    # pycocotools by default: it is the only scorer that reports mAP75 (ultralytics'
    # val() exposes just mAP50 and mAP50-95), and it is the cross-team-comparable
    # one - the set2 collaborator scores with the same COCO protocol.
    parser.add_argument("--scorer", default="pycocotools", choices=["ultralytics", "pycocotools"])
    parser.add_argument("--metrics-csv", default=str(DEFAULT_METRICS_CSV))
    parser.add_argument("--out", default=None, help="optional path to also write the report (e.g. results/reports/<arm>.md)")
    args = parser.parse_args()

    csv_path = Path(args.metrics_csv)
    values = load_latest_values(csv_path, args.scorer)

    have_arm = any(k[0] == args.arm for k in values)
    have_base = any(k[0] == args.baseline for k in values)
    if not have_arm:
        raise SystemExit(f"no rows found for arm '{args.arm}' (scorer={args.scorer}) in {csv_path}")
    if not have_base:
        raise SystemExit(f"no rows found for baseline '{args.baseline}' (scorer={args.scorer}) in {csv_path}")

    text_sections = [f"{args.arm} vs {args.baseline}  (scorer={args.scorer})", ""]
    for scope in ["overall"] + CLASSES:
        text_sections.append(render_scope_table_text(scope, values, args.arm, args.baseline))
        text_sections.append("")
    print("\n".join(text_sections))

    if args.out:
        out_path = ROOT / args.out if not Path(args.out).is_absolute() else Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.suffix.lower() == ".md":
            md_sections = [f"# {args.arm} vs {args.baseline}  (scorer={args.scorer})", ""]
            for scope in ["overall"] + CLASSES:
                md_sections.append(render_scope_table_md(scope, values, args.arm, args.baseline))
                md_sections.append("")
            out_path.write_text("\n".join(md_sections), encoding="utf-8")
        else:
            out_path.write_text("\n".join(text_sections), encoding="utf-8")
        print(f"\nWrote report: {out_path}")


if __name__ == "__main__":
    main()
