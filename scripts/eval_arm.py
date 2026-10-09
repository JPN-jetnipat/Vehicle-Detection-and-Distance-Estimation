"""Evaluate an already-trained arm's checkpoint on its eval_splits, without retraining.

Usage:
    python scripts/eval_arm.py --config configs/experiments/set1_night_aug.yaml
    python scripts/eval_arm.py --config configs/experiments/set1_night_aug.yaml --weights runs/detect/set1_night_aug/weights/best.pt
    python scripts/eval_arm.py --config configs/experiments/set1_night_aug.yaml --splits val_clear val_overcast val_snowy

Reuses train_arm.py's metrics_to_rows/append_metrics/export_excel, so results
land in the same results/metrics_all_arms.csv|xlsx under the config's
arm_name, alongside rows from actual training runs.

Scores with both scorers, exactly like train_arm.py does, so a checkpoint
re-evaluated here is directly comparable to one scored during training:
ultralytics' own AP implementation, then pycocotools (the standard COCO
protocol, and the only scorer that reports mAP75 - ultralytics' val() exposes
only mAP50 and mAP50-95). Rows are tagged with their scorer, so the two never
get conflated.

--plots-only reruns ultralytics' val() just to regenerate each split's plots
(<run_dir>/eval/<split>/), without the pycocotools pass and without appending
any rows to the metrics CSV. --samples N instead only draws the first N images
of each split one per file (<run_dir>/test_pred_separate/<split>/), which is
cheap enough to run on the local laptop.

--splits restricts scoring to the named eval_splits (e.g. only the weather
splits an arm is missing). Other splits' existing rows are left untouched.

Must be run with the repo root as the working directory (same as train_arm.py).
"""

import argparse
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO

import eval_arm_coco
import train_arm

ROOT = Path(__file__).resolve().parent.parent


def save_samples(model: YOLO, splits: list[dict], out_root: Path, n: int, imgsz: int) -> None:
    """Draw predictions on the first n images (by filename) of each split, one file per image."""
    for split in splits:
        data = yaml.safe_load((ROOT / split["data"]).read_text(encoding="utf-8"))
        image_dir = ROOT / data["path"] / data["val"]
        images = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})[:n]
        if not images:
            print(f"[{split['name']}] no images found in {image_dir}, skipping")
            continue
        # line_width=1 keeps boxes and labels small enough to read on crowded street scenes
        model.predict(
            [str(p) for p in images],
            imgsz=imgsz,
            conf=0.25,
            save=True,
            project=str(out_root),
            name=split["name"],
            exist_ok=True,
            line_width=1,
            verbose=False,
        )
        print(f"[{split['name']}] saved {len(images)} images to {out_root / split['name']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--weights",
        type=Path,
        default=None,
        help="Path to a .pt checkpoint. Defaults to runs/detect/<train.name>/weights/best.pt from the config.",
    )
    parser.add_argument(
        "--plots-only",
        action="store_true",
        help="Regenerate each split's val plots only; skip pycocotools and write no metric rows.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=0,
        metavar="N",
        help="Only save per-image predictions for the first N images of each split, then exit.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=None,
        metavar="NAME",
        help="Only evaluate these eval_splits by name (e.g. val_clear val_snowy). Defaults to all.",
    )
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    arm_name = cfg["arm_name"]
    train_kwargs = cfg["train"]

    if args.splits:
        known = {split["name"] for split in cfg["eval_splits"]}
        unknown = sorted(set(args.splits) - known)
        if unknown:
            raise ValueError(f"Unknown split(s) {unknown} - {args.config} defines {sorted(known)}")
        cfg["eval_splits"] = [split for split in cfg["eval_splits"] if split["name"] in args.splits]

    weights = args.weights or ROOT / "runs" / "detect" / train_kwargs["name"] / "weights" / "best.pt"
    if not weights.exists():
        raise FileNotFoundError(f"No weights at {weights} - train this arm first, or pass --weights explicitly.")

    model = YOLO(weights)
    run_dir = weights.resolve().parent.parent

    if args.samples:
        save_samples(model, cfg["eval_splits"], run_dir / "test_pred_separate", args.samples, train_kwargs["imgsz"])
        return

    timestamp = train_arm.now_th()
    rows: list[dict] = []
    for split in cfg["eval_splits"]:
        split_metrics = model.val(
            data=split["data"],
            split="val",
            imgsz=train_kwargs["imgsz"],
            batch=train_kwargs["batch"],
            # weights live at <run_dir>/weights/best.pt; save plots next to them
            project=str(run_dir / "eval"),
            name=split["name"],
            exist_ok=True,
        )
        rows.extend(train_arm.metrics_to_rows(arm_name, split["name"], split_metrics, timestamp))

    if args.plots_only:
        print(f"Saved plots for {len(cfg['eval_splits'])} splits under {run_dir / 'eval'} (no metric rows written)")
        return

    coco_device = "0" if torch.cuda.is_available() else "cpu"
    for split in cfg["eval_splits"]:
        print(f"\n[{arm_name}] scoring {split['name']} ({split['data']}) with pycocotools ...")
        coco_result, n_images = eval_arm_coco.score_split(
            model, split["data"], train_kwargs["imgsz"], coco_device, train_kwargs["batch"]
        )
        eval_arm_coco.print_split_table(n_images, coco_result)
        rows.extend(eval_arm_coco.coco_result_to_rows(arm_name, split["name"], coco_result, timestamp))

    train_arm.append_metrics(rows)
    train_arm.export_excel()
    print(f"Evaluated {weights} -> appended {len(rows)} metric rows for arm '{arm_name}' to {train_arm.METRICS_CSV}")
    print(f"Updated {train_arm.METRICS_XLSX}")


if __name__ == "__main__":
    main()
