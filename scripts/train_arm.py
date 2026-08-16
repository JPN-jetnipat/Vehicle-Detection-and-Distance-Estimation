"""Train one experiment arm and log mAP metrics for cross-arm comparison.

Usage:
    python scripts/train_arm.py --config configs/experiments/set1_night_aug.yaml

Reads an arm's hyperparameters from --config, trains via ultralytics
YOLO.train(), then runs YOLO.val() on each dataset listed under
`eval_splits` (plus the in-training val split) and appends the results to
results/metrics_all_arms.csv as long-format rows:
arm_name, dataset_split, metric_name, value, timestamp.

Must be run with the repo root as the working directory (the dataset yamls'
`path:` fields are relative to it).
"""

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path

import yaml
from tqdm import tqdm
from ultralytics import YOLO

import resolve_splits

ROOT = Path(__file__).resolve().parent.parent
METRICS_CSV = ROOT / "results" / "metrics_all_arms.csv"
CSV_FIELDS = ["arm_name", "dataset_split", "metric_name", "value", "timestamp"]


def append_metrics(rows: list[dict]) -> None:
    METRICS_CSV.parent.mkdir(parents=True, exist_ok=True)
    write_header = not METRICS_CSV.exists()
    with METRICS_CSV.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def metrics_to_rows(arm_name: str, dataset_split: str, metrics, timestamp: str) -> list[dict]:
    rows = [
        {"arm_name": arm_name, "dataset_split": dataset_split, "metric_name": "mAP50", "value": metrics.box.map50, "timestamp": timestamp},
        {"arm_name": arm_name, "dataset_split": dataset_split, "metric_name": "mAP50-95", "value": metrics.box.map, "timestamp": timestamp},
    ]
    for i, class_id in enumerate(metrics.box.ap_class_index):
        class_name = metrics.names[int(class_id)]
        _, _, ap50, ap = metrics.box.class_result(i)
        rows.append({"arm_name": arm_name, "dataset_split": dataset_split, "metric_name": f"mAP50_{class_name}", "value": ap50, "timestamp": timestamp})
        rows.append({"arm_name": arm_name, "dataset_split": dataset_split, "metric_name": f"mAP50-95_{class_name}", "value": ap, "timestamp": timestamp})
    return rows


def make_epoch_progress_callback(pbar: tqdm):
    def _on_fit_epoch_end(trainer) -> None:
        pbar.update(1)
        postfix = {
            k.split("/")[-1].rstrip("(B)"): f"{v:.4f}"
            for k, v in (trainer.metrics or {}).items()
            if k in ("metrics/mAP50(B)", "metrics/mAP50-95(B)")
        }
        if postfix:
            pbar.set_postfix(postfix)

    return _on_fit_epoch_end


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    arm_name = cfg["arm_name"]
    train_kwargs = cfg["train"]

    resolve_splits.main()

    model = YOLO(cfg["model"])
    with tqdm(total=train_kwargs["epochs"], desc=f"[{arm_name}] train", unit="epoch") as epoch_pbar:
        model.add_callback("on_fit_epoch_end", make_epoch_progress_callback(epoch_pbar))
        train_results = model.train(data=cfg["data"], **train_kwargs)
    timestamp = datetime.now(timezone.utc).isoformat()

    rows = metrics_to_rows(arm_name, "val", train_results, timestamp)

    best_weights = model.trainer.save_dir / "weights" / "best.pt"
    eval_model = YOLO(best_weights)

    for split in tqdm(cfg["eval_splits"], desc=f"[{arm_name}] eval splits", unit="split"):
        split_metrics = eval_model.val(
            data=split["data"],
            split="val",
            imgsz=train_kwargs["imgsz"],
            batch=train_kwargs["batch"],
        )
        rows.extend(metrics_to_rows(arm_name, split["name"], split_metrics, timestamp))

    append_metrics(rows)
    print(f"Appended {len(rows)} metric rows for arm '{arm_name}' to {METRICS_CSV}")


if __name__ == "__main__":
    main()
