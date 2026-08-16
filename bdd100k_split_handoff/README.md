# BDD100K split - shared setup

This is the exact class list + train/val/test split used for the vehicle
detection project, so your results and mine are directly comparable
(same test set, same val set, no leakage between them).

Rules baked into these files (2026-08-14):
- Classes: car(0), truck(1), bus(2), motor(3), bike(4) - everything else
  (person, rider, traffic light, traffic sign, train) is dropped.
- Images with zero boxes after that filter are KEPT as background
  (empty label file), not dropped.
- Any image with weather == "undefined" or timeofday == "undefined" is
  dropped entirely, from both pools, before splitting.
- TEST = BDD's official val set (cleaned), used as-is. Touch it for final
  scoring only - never for development/model-selection.
- VAL = 2.5% of EACH timeofday stratum (day/night/dawn-dusk) of BDD's
  official train set, sampled independently per stratum so val mirrors
  train's timeofday mix. The remaining 97.5% is the train pool.
- train_100/50/25/10 = that pool at 100/50/25/10% of labels (stratified by
  timeofday), for the label-fraction ablation. train_100 and pretrain.txt
  are the SAME image list - pretrain.txt is for JEPA's unlabeled stage,
  train_100.txt is the same images used with labels for fine-tuning.
- Everything is seed=42, deterministic.

## What you need that ISN'T in this folder

You need your own copy of BDD100K's official detection labels + images
(train+val). Not included here - it's ~1.6GB of JSON + ~6GB of images, and
it's a licensed public research dataset, so pull your own copy from
wherever we've been getting it (same Kaggle mirror, or bdd-data.berkeley.edu
directly) rather than me forwarding a copy peer-to-peer.

You need `pyyaml` and `ijson` (`pip install pyyaml ijson`).

## Setup (run from wherever this folder lives, as the working directory)

**1. Build the attribute index** - this ALSO hard-verifies your copy of the
dataset matches mine (69,863 train / 10,000 val). If this fails, your
download is a different/partial copy - stop and fix that first, don't
proceed with a mismatched split:

```bash
python tools/build_attr_index.py \
  --train-json <path-to>/bdd100k_labels_images_train.json \
  --val-json   <path-to>/bdd100k_labels_images_val.json \
  --out-prefix dataset/yolo/attr_index \
  --expect-counts train=69863,val=10000
```

**2. Get the split.** Simplest and safest: just use the `splits/*.txt` files
already in this folder directly - skip regenerating. If you want to
double-check your data matches mine bit-for-bit first, you can regenerate
into a throwaway folder and diff:

```bash
python tools/make_splits.py --attr-index dataset/yolo/attr_index.json \
  --repo-splits splits_check --yolo-splits dataset/yolo/splits --seed 42
diff -r splits splits_check   # no output = byte-identical, you're good
```

**3. Convert labels to YOLO format** (per split you actually need; train
takes ~1 min, val ~10 sec):

```bash
python tools/bdd_to_yolo.py --labels-json <train.json> \
  --data-config configs/data/bdd100k_vehicle5.yaml \
  --out-dir dataset/yolo/labels/train --stats-out dataset/yolo/convert_stats_train.json
python tools/bdd_to_yolo.py --labels-json <val.json> \
  --data-config configs/data/bdd100k_vehicle5.yaml \
  --out-dir dataset/yolo/labels/val --stats-out dataset/yolo/convert_stats_val.json
```

Sanity numbers you should get (train): car 713166, truck 29969, bus 11672,
motor 3002, bike 7209, 48 degenerate boxes, 564 empty label files.
Val: car 102504, truck 4244, bus 1597, motor 452, bike 1007.

**4. Materialize images** - your copy's `images/100k/train/` may or may not
be flat (mine wasn't - it was split across 5 subfolders on the mirror we
used). This handles either case:

```bash
python tools/materialize_images.py \
  --raw-split-dir <path-to>/images/100k/train --out-dir dataset/yolo/images/train
python tools/materialize_images.py \
  --raw-split-dir <path-to>/images/100k/val --out-dir dataset/yolo/images/val
```

Done - `dataset/yolo/` now mirrors mine, and `splits/*.txt` /
`dataset/yolo/splits/*.txt` point at real images + labels either way.
