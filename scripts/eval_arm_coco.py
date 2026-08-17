"""Score an already-trained arm's checkpoint with the pycocotools-based shared
evaluator (evaluation/eval_detections.py + evaluation/run_inference.py),
instead of ultralytics' own model.val().

Combines the two evaluators discussed with the set2 collaborator: standard
COCO-protocol scoring (pycocotools) and NMS settings fixed identically for
every arm (conf=0.001, iou=0.6, max_det=300) for fairness, while still
writing into the same results/metrics_all_arms.csv|xlsx used by
train_arm.py/eval_arm.py - rows are tagged scorer="pycocotools" (vs
"ultralytics" for the model.val()-based rows) so the two are never
conflated in the comparison table.

Raw prediction files are written to a temp directory and deleted once
scoring is done - nothing is kept on disk beyond the usual CSV/xlsx rows.

Usage:
    python scripts/eval_arm_coco.py --config configs/experiments/set1_night_aug.yaml
    python scripts/eval_arm_coco.py --config configs/experiments/set1_night_aug.yaml --weights runs/detect/set1_night_aug/weights/best.pt --device 0

Must be run with the repo root as the working directory (same as
train_arm.py/eval_arm.py).
"""

import argparse
import contextlib
import io
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import yaml
from ultralytics import YOLO
from ultralytics.data.utils import check_det_dataset

import train_arm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "evaluation"))
import eval_detections as coco_eval  # noqa: E402

# Fixed for every arm, matching evaluation/run_inference.py's defaults - do
# not make these per-arm configurable, that would defeat the fairness point.
CONF_THRES = 0.001
IOU_THRES = 0.6
MAX_DET = 300
IMG_WIDTH = 1280
IMG_HEIGHT = 720


def resolve_split_dirs(data_yaml: str) -> tuple[Path, Path]:
    """(images_dir, gt_labels_dir) for an eval_splits entry's data yaml."""
    resolved = check_det_dataset(data_yaml)
    images_dir = Path(resolved["val"])
    sep = os.sep
    labels_dir = Path(str(images_dir).replace(f"{sep}images{sep}", f"{sep}labels{sep}"))
    return images_dir, labels_dir


def run_inference_to_dir(model: YOLO, image_paths: list[Path], out_dir: Path, imgsz: int, device: str, batch: int) -> None:
    results = model.predict(
        source=[str(p) for p in image_paths],
        imgsz=imgsz,
        conf=CONF_THRES,
        iou=IOU_THRES,
        max_det=MAX_DET,
        device=device,
        batch=batch,
        stream=True,
        verbose=False,
    )
    for r in results:
        lines = []
        if r.boxes is not None and len(r.boxes):
            xywhn = r.boxes.xywhn.tolist()
            conf = r.boxes.conf.tolist()
            cls = r.boxes.cls.tolist()
            for (cx, cy, w, h), c, k in zip(xywhn, conf, cls):
                lines.append(f"{int(k)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {c:.6f}")
        stem = Path(r.path).stem
        (out_dir / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))


def score_split(model: YOLO, data_yaml: str, imgsz: int, device: str, batch: int) -> tuple[dict, int]:
    images_dir, gt_labels_dir = resolve_split_dirs(data_yaml)
    image_names = sorted(p.name for p in images_dir.iterdir() if p.is_file())
    names = coco_eval.load_names(data_yaml)

    with tempfile.TemporaryDirectory(prefix="eval_arm_coco_") as tmp:
        preds_dir = Path(tmp)
        run_inference_to_dir(model, [images_dir / n for n in image_names], preds_dir, imgsz, device, batch)

        gt_coco, name_to_id = coco_eval.build_coco(image_names, gt_labels_dir, names, IMG_WIDTH, IMG_HEIGHT)
        dets, _found = coco_eval.load_preds(image_names, preds_dir, name_to_id, IMG_WIDTH, IMG_HEIGHT, len(names))
    if not dets:
        raise SystemExit(f"no detections for {data_yaml} - check weights/thresholds")
    with contextlib.redirect_stdout(io.StringIO()):
        dt_coco = gt_coco.loadRes(dets)

    all_ids = [name_to_id[n] for n in image_names]
    overall = coco_eval.evaluate_slice(gt_coco, dt_coco, all_ids)
    per_class = {}
    for ci, cname in enumerate(names):
        per_class[cname] = coco_eval.evaluate_slice(gt_coco, dt_coco, all_ids, cat_id=ci)
    return {"overall": overall, "per_class": per_class}, len(image_names)


def coco_result_to_rows(arm_name: str, dataset_split: str, result: dict, timestamp: str) -> list[dict]:
    rows = []
    for metric_key, metric_name in (("mAP50", "mAP50"), ("mAP50_95", "mAP50-95"), ("mAP75", "mAP75")):
        rows.append({"arm_name": arm_name, "dataset_split": dataset_split, "scorer": "pycocotools",
                     "metric_name": metric_name, "value": result["overall"][metric_key], "timestamp": timestamp})
    for cname, m in result["per_class"].items():
        for metric_key, metric_name in (("mAP50", f"mAP50_{cname}"), ("mAP50_95", f"mAP50-95_{cname}"), ("mAP75", f"mAP75_{cname}")):
            rows.append({"arm_name": arm_name, "dataset_split": dataset_split, "scorer": "pycocotools",
                         "metric_name": metric_name, "value": m[metric_key], "timestamp": timestamp})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--weights", type=Path, default=None, help="Defaults to runs/detect/<train.name>/weights/best.pt")
    parser.add_argument("--device", default="cpu", help="'cpu' or CUDA index like '0'")
    parser.add_argument("--batch", type=int, default=16)
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    arm_name = cfg["arm_name"]
    train_kwargs = cfg["train"]

    weights = args.weights or ROOT / "runs" / "detect" / train_kwargs["name"] / "weights" / "best.pt"
    if not weights.exists():
        raise FileNotFoundError(f"No weights at {weights} - train this arm first, or pass --weights explicitly.")

    model = YOLO(weights)
    timestamp = datetime.now(timezone.utc).isoformat()
    rows: list[dict] = []
    for split in cfg["eval_splits"]:
        print(f"[{arm_name}] scoring {split['name']} ({split['data']}) with pycocotools ...")
        result, n_images = score_split(model, split["data"], train_kwargs["imgsz"], args.device, args.batch)
        print(f"  n={n_images}  mAP50={result['overall']['mAP50']}  mAP75={result['overall']['mAP75']}  mAP50-95={result['overall']['mAP50_95']}")
        rows.extend(coco_result_to_rows(arm_name, split["name"], result, timestamp))

    train_arm.append_metrics(rows)
    train_arm.export_excel()
    print(f"\nAppended {len(rows)} pycocotools-scored rows for arm '{arm_name}' to {train_arm.METRICS_CSV}")
    print(f"Updated {train_arm.METRICS_XLSX}")


if __name__ == "__main__":
    main()
