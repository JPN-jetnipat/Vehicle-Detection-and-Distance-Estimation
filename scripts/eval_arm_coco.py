"""Score an already-trained arm's checkpoint with a pycocotools-based
scorer, instead of ultralytics' own model.val().

Combines the two evaluators discussed with the set2 collaborator: standard
COCO-protocol scoring (pycocotools, adapted from their eval_detections.py)
and NMS settings fixed identically for every arm (conf=0.001, iou=0.6,
max_det=300) for fairness, while still writing into the same
results/metrics_all_arms.csv|xlsx used by train_arm.py/eval_arm.py - rows
are tagged scorer="pycocotools" (vs "ultralytics" for the model.val()-based
rows) so the two are never conflated in the comparison table.

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
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import yaml
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from ultralytics import YOLO
from ultralytics.data.utils import check_det_dataset

import train_arm

ROOT = Path(__file__).resolve().parent.parent

# Fixed for every arm - do not make these per-arm configurable, that would
# defeat the fairness point.
CONF_THRES = 0.001
IOU_THRES = 0.6
MAX_DET = 300
IMG_WIDTH = 1280
IMG_HEIGHT = 720


def load_names(data_config) -> list[str]:
    names = yaml.safe_load(open(data_config))["names"]
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]
    return names


def yolo_line_to_xywh(parts: list[str], W: int, H: int) -> list[float]:
    cx, cy, w, h = (float(v) for v in parts[1:5])
    return [(cx - w / 2) * W, (cy - h / 2) * H, w * W, h * H]


def build_coco(image_names: list[str], gt_dir, names: list[str], W: int, H: int):
    images, anns = [], []
    name_to_id = {}
    aid = 1
    for iid, name in enumerate(sorted(image_names), start=1):
        name_to_id[name] = iid
        images.append({"id": iid, "file_name": name, "width": W, "height": H})
        gt = Path(gt_dir) / (Path(name).stem + ".txt")
        if not gt.exists():
            raise FileNotFoundError(f"missing GT label file: {gt}")
        for line in gt.read_text().splitlines():
            p = line.split()
            box = yolo_line_to_xywh(p, W, H)
            anns.append({"id": aid, "image_id": iid, "category_id": int(p[0]),
                         "bbox": box, "area": box[2] * box[3], "iscrowd": 0})
            aid += 1
    gt_coco = COCO()
    gt_coco.dataset = {"images": images, "annotations": anns,
                       "categories": [{"id": i, "name": n} for i, n in enumerate(names)]}
    with contextlib.redirect_stdout(io.StringIO()):
        gt_coco.createIndex()
    return gt_coco, name_to_id


def load_preds(image_names: list[str], pred_dir, name_to_id: dict, W: int, H: int, n_classes: int):
    dets = []
    found = 0
    oob = 0  # predictions in classes outside our list (guards against a
             # mismatched/stale weights file firing on dead class ids)
    for name in image_names:
        pf = Path(pred_dir) / (Path(name).stem + ".txt")
        if not pf.exists():
            continue
        found += 1
        for line in pf.read_text().splitlines():
            if not line.strip():
                continue
            p = line.split()
            if len(p) != 6:
                raise ValueError(f"{pf}: expected 'cls cx cy w h conf', got: {line}")
            if int(p[0]) >= n_classes:
                oob += 1
                continue
            dets.append({"image_id": name_to_id[name], "category_id": int(p[0]),
                         "bbox": yolo_line_to_xywh(p, W, H), "score": float(p[5])})
    if oob:
        print(f"note: dropped {oob} predictions with class id >= {n_classes} (untrained/dead classes)")
    return dets, found


def evaluate_slice(gt_coco, dt_coco, img_ids: list[int], cat_id: int | None = None) -> dict:
    ev = COCOeval(gt_coco, dt_coco, iouType="bbox")
    ev.params.imgIds = img_ids
    if cat_id is not None:
        ev.params.catIds = [cat_id]
    with contextlib.redirect_stdout(io.StringIO()):
        ev.evaluate(); ev.accumulate(); ev.summarize()
    s = ev.stats
    return {"mAP50_95": round(float(s[0]), 5), "mAP50": round(float(s[1]), 5),
            "mAP75": round(float(s[2]), 5)}


def resolve_split_dirs(data_yaml: str) -> tuple[Path, Path]:
    """(images_dir, gt_labels_dir) for an eval_splits entry's data yaml."""
    resolved = check_det_dataset(data_yaml)
    images_dir = Path(resolved["val"])
    sep = os.sep
    labels_dir = Path(str(images_dir).replace(f"{sep}images{sep}", f"{sep}labels{sep}"))
    return images_dir, labels_dir


def run_inference_to_dir(model: YOLO, image_paths: list[Path], out_dir: Path, imgsz: int, device: str, batch: int) -> None:
    # Chunk manually - passing the full path list straight to model.predict()
    # with batch=N does NOT actually cap memory use to N images at a time (it
    # allocates as if for the whole list regardless of `batch`), which OOM'd
    # even on a 46GB A40. One predict() call per chunk keeps memory bounded.
    for start in range(0, len(image_paths), batch):
        chunk = image_paths[start:start + batch]
        results = model.predict(
            source=[str(p) for p in chunk],
            imgsz=imgsz,
            conf=CONF_THRES,
            iou=IOU_THRES,
            max_det=MAX_DET,
            device=device,
            batch=len(chunk),
            stream=False,
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
    names = load_names(data_yaml)

    with tempfile.TemporaryDirectory(prefix="eval_arm_coco_") as tmp:
        preds_dir = Path(tmp)
        run_inference_to_dir(model, [images_dir / n for n in image_names], preds_dir, imgsz, device, batch)

        gt_coco, name_to_id = build_coco(image_names, gt_labels_dir, names, IMG_WIDTH, IMG_HEIGHT)
        dets, _found = load_preds(image_names, preds_dir, name_to_id, IMG_WIDTH, IMG_HEIGHT, len(names))
    if not dets:
        raise SystemExit(f"no detections for {data_yaml} - check weights/thresholds")
    with contextlib.redirect_stdout(io.StringIO()):
        dt_coco = gt_coco.loadRes(dets)

    all_ids = [name_to_id[n] for n in image_names]
    overall = evaluate_slice(gt_coco, dt_coco, all_ids)
    per_class = {}
    n_instances = {}
    for ci, cname in enumerate(names):
        per_class[cname] = evaluate_slice(gt_coco, dt_coco, all_ids, cat_id=ci)
        n_instances[cname] = len(gt_coco.getAnnIds(catIds=[ci]))
    return {"overall": overall, "per_class": per_class, "n_instances": n_instances}, len(image_names)


def print_split_table(n_images: int, result: dict) -> None:
    n_total_instances = sum(result["n_instances"].values())
    o = result["overall"]
    print(f"{'Class':>10} {'Images':>8} {'Instances':>10} {'mAP50':>8} {'mAP75':>8} {'mAP50-95':>10}")
    print(f"{'all':>10} {n_images:>8} {n_total_instances:>10} "
          f"{o['mAP50']:>8} {o['mAP75']:>8} {o['mAP50_95']:>10}")
    for cname, m in result["per_class"].items():
        print(f"{cname:>10} {n_images:>8} {result['n_instances'][cname]:>10} "
              f"{m['mAP50']:>8} {m['mAP75']:>8} {m['mAP50_95']:>10}")


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
        print(f"\n[{arm_name}] scoring {split['name']} ({split['data']}) with pycocotools ...")
        result, n_images = score_split(model, split["data"], train_kwargs["imgsz"], args.device, args.batch)
        print_split_table(n_images, result)
        rows.extend(coco_result_to_rows(arm_name, split["name"], result, timestamp))

    train_arm.append_metrics(rows)
    train_arm.export_excel()
    print(f"\nAppended {len(rows)} pycocotools-scored rows for arm '{arm_name}' to {train_arm.METRICS_CSV}")
    print(f"Updated {train_arm.METRICS_XLSX}")


if __name__ == "__main__":
    main()
