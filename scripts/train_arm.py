"""Train one experiment arm and log mAP metrics for cross-arm comparison.

Usage:
    python scripts/train_arm.py --config configs/experiments/set1_night_aug.yaml

Reads an arm's hyperparameters from --config, trains via ultralytics
YOLO.train(), then runs YOLO.val() on each dataset listed under
`eval_splits` (plus the in-training val split) and appends the results to
results/metrics_all_arms.csv as long-format rows:
arm_name, dataset_split, metric_name, value, timestamp. A pivoted, more
readable copy is kept in sync at results/metrics_all_arms.xlsx.

Per-epoch training progress is written to results/logs/<arm_name>.log as
plain text (loss + mAP per epoch). Ultralytics also writes its own
per-epoch runs/detect/<name>/results.csv automatically - that one has more
columns (precision/recall/lr) if you need the full picture.

Must be run with the repo root as the working directory (the dataset yamls'
`path:` fields are relative to it).
"""

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path

import yaml
from openpyxl import Workbook
from tqdm import tqdm
from ultralytics import YOLO

import resolve_splits

ROOT = Path(__file__).resolve().parent.parent
METRICS_CSV = ROOT / "results" / "metrics_all_arms.csv"
METRICS_XLSX = ROOT / "results" / "metrics_all_arms.xlsx"
LOGS_DIR = ROOT / "results" / "logs"
CSV_FIELDS = ["arm_name", "dataset_split", "scorer", "metric_name", "value", "timestamp"]


def append_metrics(rows: list[dict]) -> None:
    METRICS_CSV.parent.mkdir(parents=True, exist_ok=True)
    write_header = not METRICS_CSV.exists()
    with METRICS_CSV.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def export_excel() -> None:
    """Rebuild metrics_all_arms.xlsx as a wide pivot (one row per run/split/scorer) from the full CSV history."""
    with METRICS_CSV.open(encoding="utf-8") as f:
        long_rows = list(csv.DictReader(f))

    pivot: dict[tuple[str, str, str, str], dict[str, float]] = {}
    metric_names: list[str] = []
    for row in long_rows:
        key = (row["timestamp"], row["arm_name"], row["dataset_split"], row.get("scorer", "ultralytics"))
        pivot.setdefault(key, {})[row["metric_name"]] = float(row["value"])
        if row["metric_name"] not in metric_names:
            metric_names.append(row["metric_name"])
    # Stable order: overall metrics first, then per-class, alphabetically after that.
    priority = {"mAP50": 0, "mAP50-95": 1, "mAP75": 2}
    metric_names.sort(key=lambda m: (priority.get(m, 3), m))

    wb = Workbook()
    ws = wb.active
    ws.title = "metrics"
    ws.append(["timestamp", "arm_name", "dataset_split", "scorer", *metric_names])
    for key in sorted(pivot):
        timestamp, arm_name, dataset_split, scorer = key
        values = pivot[key]
        ws.append([timestamp, arm_name, dataset_split, scorer, *(values.get(m) for m in metric_names)])
    ws.freeze_panes = "A2"
    for col_idx in range(1, 5 + len(metric_names)):
        ws.column_dimensions[ws.cell(row=1, column=col_idx).column_letter].width = 14

    METRICS_XLSX.parent.mkdir(parents=True, exist_ok=True)
    wb.save(METRICS_XLSX)


def metrics_to_rows(arm_name: str, dataset_split: str, metrics, timestamp: str, scorer: str = "ultralytics") -> list[dict]:
    rows = [
        {"arm_name": arm_name, "dataset_split": dataset_split, "scorer": scorer, "metric_name": "mAP50", "value": metrics.box.map50, "timestamp": timestamp},
        {"arm_name": arm_name, "dataset_split": dataset_split, "scorer": scorer, "metric_name": "mAP50-95", "value": metrics.box.map, "timestamp": timestamp},
    ]
    for i, class_id in enumerate(metrics.box.ap_class_index):
        class_name = metrics.names[int(class_id)]
        _, _, ap50, ap = metrics.box.class_result(i)
        rows.append({"arm_name": arm_name, "dataset_split": dataset_split, "scorer": scorer, "metric_name": f"mAP50_{class_name}", "value": ap50, "timestamp": timestamp})
        rows.append({"arm_name": arm_name, "dataset_split": dataset_split, "scorer": scorer, "metric_name": f"mAP50-95_{class_name}", "value": ap, "timestamp": timestamp})
    return rows


def make_epoch_progress_callback(pbar: tqdm):
    def _on_fit_epoch_end(trainer) -> None:
        # Ultralytics fires this callback once more after the loop, for a final
        # re-validation on best.pt (trainer.epoch == trainer.epochs there) - not a real epoch.
        if trainer.epoch >= trainer.epochs:
            return
        pbar.update(1)
        postfix = {
            k.split("/")[-1].rstrip("(B)"): f"{v:.4f}"
            for k, v in (trainer.metrics or {}).items()
            if k in ("metrics/mAP50(B)", "metrics/mAP50-95(B)")
        }
        if postfix:
            pbar.set_postfix(postfix)

    return _on_fit_epoch_end


def format_per_class(validator) -> str:
    """Compact 'class:mAP50/mAP50-95 ...' string from a validator's DetMetrics (None if unavailable)."""
    box = getattr(getattr(validator, "metrics", None), "box", None)
    if box is None or not len(box.ap_class_index):
        return ""
    names = validator.metrics.names
    parts = []
    for i, class_id in enumerate(box.ap_class_index):
        _, _, ap50, ap = box.class_result(i)
        parts.append(f"{names[int(class_id)]}={ap50:.3f}/{ap:.3f}")
    return " ".join(parts)


def make_epoch_log_callback(log_path: Path):
    def _on_fit_epoch_end(trainer) -> None:
        losses = trainer.label_loss_items(trainer.tloss) if trainer.tloss is not None else {}
        metrics = trainer.metrics or {}
        metric_parts = [
            f"{k.split('/')[-1].rstrip('(B)')}={v:.4f}"
            for k, v in metrics.items()
            if k in ("metrics/mAP50(B)", "metrics/mAP50-95(B)", "metrics/precision(B)", "metrics/recall(B)")
        ]
        per_class = format_per_class(trainer.validator)
        if per_class:
            metric_parts.append(f"per_class(mAP50/mAP50-95)=[{per_class}]")
        timestamp = datetime.now(timezone.utc).isoformat()
        # See note in make_epoch_progress_callback: this extra firing is a final
        # re-validation on best.pt after training ends, not a new epoch.
        if trainer.epoch >= trainer.epochs:
            parts = ["final validation (best.pt)", *metric_parts]
        else:
            loss_parts = [f"{k.split('/')[-1]}={v:.5f}" for k, v in losses.items()]
            parts = [f"epoch {trainer.epoch + 1}/{trainer.epochs}", *loss_parts, *metric_parts]
        with log_path.open("a", encoding="utf-8") as f:
            f.write(f"[{timestamp}] {' | '.join(parts)}\n")

    return _on_fit_epoch_end


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    arm_name = cfg["arm_name"]
    train_kwargs = cfg["train"]

    resolve_splits.main()

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOGS_DIR / f"{arm_name}.log"
    log_path.write_text(f"[{datetime.now(timezone.utc).isoformat()}] starting arm '{arm_name}' ({train_kwargs['epochs']} epochs)\n", encoding="utf-8")

    model = YOLO(cfg["model"])
    with tqdm(total=train_kwargs["epochs"], desc=f"[{arm_name}] train", unit="epoch") as epoch_pbar:
        model.add_callback("on_fit_epoch_end", make_epoch_progress_callback(epoch_pbar))
        model.add_callback("on_fit_epoch_end", make_epoch_log_callback(log_path))
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
        with log_path.open("a", encoding="utf-8") as f:
            f.write(f"[{timestamp}] eval {split['name']}: mAP50={split_metrics.box.map50:.4f} mAP50-95={split_metrics.box.map:.4f}\n")

    append_metrics(rows)
    export_excel()
    print(f"Appended {len(rows)} metric rows for arm '{arm_name}' to {METRICS_CSV}")
    print(f"Updated {METRICS_XLSX}")
    print(f"Per-epoch log: {log_path}")


if __name__ == "__main__":
    main()
