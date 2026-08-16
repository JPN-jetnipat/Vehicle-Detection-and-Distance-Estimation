# Vehicle Detection — YOLOv11s on BDD100K

Nighttime / harsh-weather vehicle detection. This folder is the standalone
detection-baseline project: it does not depend on the old
`(Claude)Vehicle-Detection-and-Distance-Estimation` folder for anything.

For exact commands (environment setup, training, evaluation), see
**RUNBOOK.md**. For getting this dataset/repo onto the university Jupyter
GPU server, see **docs/dataset-upload-guide.md** (raw data) and
**docs/SERVER_MIGRATION.md** (updating an existing old-project checkout on
the server to match this repo).

## Current scope (2026-08)

- **Detector:** YOLOv11s (`ultralytics` pip package — no vendored repo).
- **Dataset:** BDD100K, 2018 Kaggle mirror ([solesensei/solesensei_bdd100k](https://www.kaggle.com/datasets/solesensei/solesensei_bdd100k)).
- **Classes (5):** car, truck, bus, motor, bike — see `configs/data/bdd100k_vehicle5.yaml`.
- **Split methodology** (full detail + exact counts in `bdd100k_split_handoff/README.md`
  and `splits/MANIFEST.json`):
  - Any image with `weather == undefined` or `timeofday == undefined` is dropped
    entirely, from both the train and val pools, before anything else.
  - **TEST** = BDD's official val set (cleaned) — used as-is, touched only for
    final scoring.
  - **VAL** = 2.5% of *each* timeofday stratum (day / night / dawn-dusk) of BDD's
    official train set, sampled independently per stratum so val mirrors train's
    timeofday mix. The remaining 97.5% is the train pool.
  - `train_100/50/25/10` = that pool at 100/50/25/10% of labels, stratified by
    timeofday (only `train_100` is used by the current 4-method plan; the
    label-fraction splits are there if you want that ablation later).
  - Weather retained (rainy, snowy, clear, overcast, partly cloudy, foggy — no
    undefined) as image-level metadata (`dataset/yolo/attr_index.json`) for
    slicing results, not as a stratification axis.
  - Everything is seed=42, deterministic — `tools/make_splits.py` regenerates
    byte-identical splits from `dataset/yolo/attr_index.json`.

## The 4-method comparison

| # | arm | weights | hyp recipe |
|---|---|---|---|
| 1 | vanilla pretrained | COCO `yolo11s.pt`, no fine-tuning | — (inference only) |
| 2 | default | fine-tuned from COCO | `configs/hyp/default.yaml` |
| 3 | set1 (low-light compensation) | fine-tuned from COCO | `configs/hyp/set1_japan_night_aug.yaml` |
| 4 | set2 (class & scene diversity) | fine-tuned from COCO | `configs/hyp/set2_field_imbalance.yaml` |

All four get scored the same way, per time-of-day (day/night/dawn-dusk) plus
overall and rainy, on `splits/test.txt`. See RUNBOOK.md for exact commands.

**Open flag for the team, not yet resolved here:** method 1 (vanilla
pretrained) uses COCO's 80-class head, whose class ids/order don't match our
5-class scheme (COCO does have `car`/`truck`/`bus`/`motorcycle`/`bicycle` as
separate classes, just different ids). Decide and document how that
comparison is scored — e.g. filter+remap COCO's predictions to the 5 classes
before handing them to `evaluation/eval_detections.py` — before running
method 1 for the record.

## Folder layout

```
configs/data/      dataset yaml (classes, split file pointers)
configs/hyp/        per-arm hyperparameter recipes (methods 2-4)
dataset/raw/        raw BDD100K download (images + label JSONs) - gitignored
dataset/yolo/        converted YOLO-format labels, attr index, flat image dirs, splits actually read at train time
splits/              canonical portable split lists (bare filenames) + MANIFEST.json
tools/               dataset prep + training launcher scripts
evaluation/          inference + scoring scripts
docs/                setup/server guides
bdd100k_split_handoff/   self-contained package for a teammate to reproduce the exact same split on their own machine
```

## Not in this folder (on purpose, for now)

Left behind in the old project folder — architecture-agnostic where noted,
so it's still usable when that phase actually starts:

- **Distance estimation / camera calibration** (`geometry/`) — this repo is
  scoped to detection only right now.
- **I-JEPA pretraining research code** (`jepa_pretrain/` + vendored
  `third_party/ijepa/`) — doesn't touch YOLO internals, portable as-is. The
  separate distillation-into-backbone step (`jepa_distill/`) *is*
  YOLOv5-specific (hard-indexed to a YOLOv5 backbone layer) and will need a
  rewrite for YOLOv11 when that phase starts.
- Teammate's old YOLOv5 weights (`weights/friend_best.pt`) — architecture
  mismatch, won't load into a YOLOv11 model.
