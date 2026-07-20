# Handoff to Friend — Baseline YOLOv5 for DANN Work

*Written 2026-07-19. Friend has stopped training his own YOLOv5 and will
build his domain-adversarial (DANN) work on top of our trained model
instead. This is the package to give him, and why each piece is needed.*

## What this is

The COCO-initialized YOLOv5s baseline from our I-JEPA ablation project,
fine-tuned on the full BDD100K training pool (62,020 images, stratified by
time-of-day) using the friend's own recipe (640px, batch 64-equivalent, 100
epochs, default hyps). This is **arm 1 of 3** in our eventual comparison —
if the I-JEPA-distilled arms (T1/T2, in progress) beat it, a stronger
checkpoint may follow later. Worth telling him this up front so he doesn't
build around it assuming it's final.

## Files to send him

1. **`runs_jepa/stage3/baseline_coco_100/weights/best.pt`** (~14 MB) — the
   trained model. Suggest renaming when sending it, e.g.
   `baseline_coco100_car_bus_truck.pt`, since more checkpoints will follow
   and "best.pt" alone is ambiguous once there are several.
2. **`configs/data/bdd100k_vehicle3.yaml`** — the exact class list/order
   this model was trained with: `0: car, 1: bus, 2: truck`.
3. **`runs_jepa/stage3/baseline_coco_100/resolved_config.json`** — full
   provenance (git commit, seed, recipe) for the record, in case it matters
   for his write-up later.

## Important correction to make when handing this over

His own ChatGPT-generated checklist (2026-07-19) lists 8 classes (Car, Van,
Truck, Pedestrian, Person_sitting, Cyclist, Tram, Misc) — that's his *old*
`bdd100k_vehicle.yaml`. This checkpoint uses only 3. If he interprets our
model's output through his 8-class list, class ids get mislabeled (id 1 =
"bus" here, "Van" there) even though nothing crashes to warn him. Point him
at `bdd100k_vehicle3.yaml` instead of his old yaml.

## Good news on his other 3 checklist points

- **Label format / class ids:** his own `convert_bdd100k_to_yolo.py` already
  uses `CLASS_MAP = {"car": 0, "bus": 1, "truck": 2}` — identical ids to
  ours (we deliberately copied his mapping — see REPO_SURVEY.md FLAG 1). If
  he still has his own converted labels, they should already be compatible.
  For a guaranteed-identical match, he can instead run our
  `tools/bdd_to_yolo.py` with `bdd100k_vehicle3.yaml` against his own
  already-downloaded raw BDD100K JSON — no data transfer needed either way.
- **Dataset split:** no special file needed. To reproduce our val numbers he
  just needs BDD100K's actual official val folder (10,000 images), scored
  as a whole — not a custom subset of it.
- **Image preprocessing (640px / letterbox / normalize):** automatic in any
  standard YOLOv5 loading path (`detect.py`, `val.py`, or our own
  `evaluation/run_inference.py`). Nothing to hand over for this.
- **Batch size / epochs / optimizer / LR:** correctly identified as *not*
  needed to use the model, only to retrain it (which he's no longer doing).
  Included anyway in `resolved_config.json` for the record.

## Performance context (our evaluator, BDD100K val, 10,000 images)

| slice | mAP50 | mAP75 | mAP50-95 |
|---|---|---|---|
| overall | 0.623 | 0.454 | 0.423 |
| night | 0.593 | 0.422 | 0.393 |
| dawn/dusk | 0.660 | 0.488 | 0.452 |
| rainy | 0.647 | 0.465 | 0.438 |

Full per-class breakdown: `runs_jepa/metrics/baseline_coco_100_val.json`.
