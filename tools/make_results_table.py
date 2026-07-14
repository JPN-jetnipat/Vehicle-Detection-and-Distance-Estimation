#!/usr/bin/env python3
"""Aggregate metrics JSONs (from evaluation/eval_detections.py) into the
final ablation table: rows = arms, columns = slices x metrics.

Point it at runs_jepa/metrics/; it groups by filename. Emits Markdown +
CSV. Add --require to fail if an expected arm is missing (for the final
paper table).

Usage:
  python tools/make_results_table.py --metrics-dir runs_jepa/metrics --out results/ablation
"""
import argparse, csv, json
from pathlib import Path

SLICE_ORDER = ["overall", "night", "dawn_dusk", "rainy"]
METRICS = ["mAP50", "mAP75", "mAP50_95"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metrics-dir", required=True)
    ap.add_argument("--out", required=True, help="output prefix -> .md + .csv")
    ap.add_argument("--require", nargs="*", default=[],
                    help="metric-file stems that must exist (e.g. baseline_coco_100_val)")
    args = ap.parse_args()

    files = sorted(Path(args.metrics_dir).glob("*.json"))
    missing = [r for r in args.require if not any(f.stem == r for f in files)]
    if missing:
        raise SystemExit(f"missing required metrics: {missing}")
    if not files:
        raise SystemExit(f"no metrics JSONs in {args.metrics_dir}")

    rows = []
    for f in files:
        d = json.loads(f.read_text())
        row = {"arm": f.stem}
        for sl in SLICE_ORDER:
            m = (d.get("slices") or {}).get(sl)
            for k in METRICS:
                row[f"{sl}/{k}"] = (None if m is None else m.get(k))
            row[f"{sl}/n"] = (d.get("n_images") or {}).get(sl)
        rows.append(row)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cols = ["arm"] + [f"{sl}/{k}" for sl in SLICE_ORDER for k in METRICS + ["n"]]
    with open(out.with_suffix(".csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader(); w.writerows(rows)

    md = ["| arm | " + " | ".join(f"{sl} {k}" for sl in SLICE_ORDER for k in METRICS) + " |",
          "|" + "---|" * (1 + len(SLICE_ORDER) * len(METRICS))]
    for r in rows:
        cells = [r["arm"]] + [("-" if r[f"{sl}/{k}"] is None else f"{r[f'{sl}/{k}']:.3f}")
                              for sl in SLICE_ORDER for k in METRICS]
        md.append("| " + " | ".join(cells) + " |")
    out.with_suffix(".md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"\nwrote {out}.md and {out}.csv")


if __name__ == "__main__":
    main()
