"""
Find which epoch Ultralytics should have picked as "best" for each run,
based on the fitness metric (mAP50-95 weighted with mAP50, matching the
default Ultralytics fitness function), and compare against what actually
got saved to weights/best.pt.

Usage:
    python find_best_epoch.py dann_v3 finetune_noadv
"""

import sys
import csv
import torch
from pathlib import Path


def fitness(row):
    # Ultralytics default: 0.1 * mAP50 + 0.9 * mAP50-95
    map50 = float(row["metrics/mAP50(B)"])
    map5095 = float(row["metrics/mAP50-95(B)"])
    return 0.1 * map50 + 0.9 * map5095


def main():
    names = sys.argv[1:]
    if not names:
        sys.exit("usage: python find_best_epoch.py <run_name> [<run_name> ...]")

    for name in names:
        csv_path = Path(f"runs/detect/{name}/results.csv")
        if not csv_path.exists():
            print(f"{name}: results.csv not found at {csv_path}")
            continue

        rows = list(csv.DictReader(open(csv_path)))
        best_row = max(rows, key=fitness)
        print(f"\n=== {name} ===")
        print(f"  total epochs logged : {len(rows)}")
        print(f"  best epoch by fitness: {best_row['epoch']}")
        print(f"  mAP50    at that epoch: {best_row['metrics/mAP50(B)']}")
        print(f"  mAP50-95 at that epoch: {best_row['metrics/mAP50-95(B)']}")
        print(f"  fitness              : {fitness(best_row):.5f}")

        ckpt_path = Path(f"runs/detect/{name}/weights/best.pt")
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            print(f"  best.pt recorded epoch: {ckpt.get('epoch')}")
            print(f"  best.pt date          : {ckpt.get('date')}")
            print(f"  best.pt best_fitness  : {ckpt.get('best_fitness')}")


if __name__ == "__main__":
    main()
