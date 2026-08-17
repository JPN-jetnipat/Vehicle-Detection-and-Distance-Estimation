# RUNBOOK — YOLOv11s vehicle detection on BDD100K

All commands run **from this folder's root**, on the university GPU server
(A40, CUDA 12.4, 32 GB shared host RAM) inside a JupyterLab terminal, unless
marked `[mac]`. Long jobs: always `tmux` or `nohup … &` — VPN/Jupyter
sessions drop.

**Before launching ANY job:**
```bash
nvidia-smi        # who else is on the GPU + free VRAM
free -h           # system RAM - the server has only 32 GB SHARED (has crashed before!)
df -h ~           # disk headroom before large downloads/extractions
```

**Night-shift protocol (GPU-sharing policy):** long jobs run overnight,
announced in the GPU users' group line first. Ultralytics checkpoints every
epoch (`last.pt`), so a multi-night job is safe to stop and resume:
```bash
# EVENING - announce, then:
tmux attach -t train || tmux new -s train
source .venv/bin/activate
nvidia-smi && free -h
nohup python tools/train_yolo.py --hyp configs/hyp/default.yaml --name default_100 > default_100.log 2>&1 &
# detach: Ctrl+b d

# MORNING - stop gracefully, freeing the GPU:
pgrep -af train_yolo.py     # note the PID
kill <PID>                   # plain kill = SIGTERM; NEVER kill -9
nvidia-smi                   # confirm GPU is free

# NEXT EVENING - same launch command with --resume added; picks up from
# <project>/<name>/weights/last.pt automatically.
```

**RAM rule, enforced by the script itself, not just documented:**
`tools/train_yolo.py` hard-refuses to launch (`FLAG:` + nonzero exit) unless
the hyp yaml has `workers: 2` and `cache: false` — every file in
`configs/hyp/` already sets both. Don't override them on the command line.

---

## 1. Environment — ~10 min

torch/torchvision/ultralytics/numpy are pinned exactly in requirements.txt
to Japan's env (torch==2.6.0, ultralytics==8.4.120 — updated 2026-08-17,
superseding an earlier snapshot of his that showed 2.12.0/8.4.65; his
header now explicitly claims this is the env that produced his real
training results). See requirements.txt's header for why exact pins matter
here (it's the actual fix for the evaluator mAP-discrepancy issue, not just
a version bump) — worth a live confirm with him before a long run, since
his own numbers have moved once already.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
nvidia-smi   # confirm CUDA 12.4 still current (top-right corner)
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# expect: torch.cuda.is_available() == True, device == NVIDIA A40
pip freeze | grep -iE "torch|ultralytics"   # sanity check it matches the pin above
```
Kaggle notebooks: skip the torch line (preinstalled); `pip install -r requirements.txt` minus torch.

## 2. Dataset onto the server — one-time

If this is a fresh server checkout: follow **docs/dataset-upload-guide.md** to
get `dataset/raw/bdd100k/` + `dataset/raw/bdd100k_labels_release/` onto the
server (chunked upload — the raw set is a few GB, too big for one browser
upload over VPN).

If the server already has the **old** project's checkout: follow
**docs/SERVER_MIGRATION.md** instead — it reuses the raw images already up
there rather than re-uploading ~4.6 GB.

**Verify layout (either path):**
```bash
ls dataset/raw/bdd100k/bdd100k/images/100k/          # train/ val/ (ignore test/ - unused, official test has no labels)
ls dataset/raw/bdd100k/bdd100k/images/100k/train | wc -l   # ~70,000 (may be nested in subfolders - fine, step 3 handles it)
ls dataset/raw/bdd100k/bdd100k/images/100k/val   | wc -l   # = 10,000
ls dataset/raw/bdd100k_labels_release/bdd100k/labels/       # train + val label JSONs
```

## 3. Data plumbing — CPU-only, ~10 min total, no GPU needed

Canonical class config: `configs/data/bdd100k_vehicle5.yaml` (car, truck,
bus, motor, bike). Both the converter and the attr-index script stream the
1.4 GB train JSON (peak RAM ~tens of MB) — safe for the shared 32 GB server.

**3.1 Attribute index** (~2 min) — this ALSO hard-verifies your copy of the
dataset matches the committed splits (69,863 train / 10,000 val). If it
fails, your download is a different/partial copy — stop and fix that before
continuing, don't proceed with a mismatched split:
```bash
python tools/build_attr_index.py \
  --train-json dataset/raw/bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_train.json \
  --val-json   dataset/raw/bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_val.json \
  --out-prefix dataset/yolo/attr_index \
  --expect-counts train=69863,val=10000
```

**3.2 Convert labels** (~3 min each):
```bash
python tools/bdd_to_yolo.py --labels-json dataset/raw/bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_train.json \
  --data-config configs/data/bdd100k_vehicle5.yaml --out-dir dataset/yolo/labels/train \
  --stats-out dataset/yolo/convert_stats_train.json
python tools/bdd_to_yolo.py --labels-json dataset/raw/bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_val.json \
  --data-config configs/data/bdd100k_vehicle5.yaml --out-dir dataset/yolo/labels/val \
  --stats-out dataset/yolo/convert_stats_val.json
```
Expected (from the committed convert_stats_*.json): train - car 713166 /
truck 29969 / bus 11672 / motor 3002 / bike 7209 boxes, 48 degenerate, 564
empty label files. Val - car 102504 / truck 4244 / bus 1597 / motor 452 /
bike 1007.

**3.3 Materialize flat image dirs** (handles nested mirror layouts like
`train/trainA/`, `train/trainB/` automatically):
```bash
python tools/materialize_images.py \
  --raw-split-dir dataset/raw/bdd100k/bdd100k/images/100k/train --out-dir dataset/yolo/images/train
python tools/materialize_images.py \
  --raw-split-dir dataset/raw/bdd100k/bdd100k/images/100k/val --out-dir dataset/yolo/images/val
```

**3.4 Splits** — regenerate rather than trusting a git-synced copy (splits
are gitignored on purpose since they're deterministic from attr_index; seed
42 guarantees byte-identical output):
```bash
python tools/make_splits.py --attr-index dataset/yolo/attr_index.json \
  --repo-splits splits --yolo-splits dataset/yolo/splits --seed 42
```
Sanity check against the committed manifest: `diff <(python -c "import json;print(json.load(open('splits/MANIFEST.json')))") ...`
or just eyeball the printed counts against `splits/MANIFEST.json` — val
1,542 / test 8,841 / train_100 60,186.

**3.5 Eyeball a few overlays** (optional sanity check):
```bash
python tools/viz_boxes.py --images-dir dataset/yolo/images/val \
  --labels-dir dataset/yolo/labels/val --data-config configs/data/bdd100k_vehicle5.yaml \
  --num 8 --out-dir viz_out
```

## 4. The 4 methods

**Method 1 — vanilla pretrained (no training).** See the open flag in
README.md before treating this number as final — COCO's class ids don't
match ours 1:1, decide the remap/filter approach with the team first.
```bash
python evaluation/run_inference.py --weights yolo11s.pt \
  --image-list splits/test.txt --images-dir dataset/yolo/images/val \
  --out-dir runs/preds/vanilla_test --device 0
```
(`yolo11s.pt` auto-downloads via ultralytics on first use if not already present.)

**Method 2 — default:**
```bash
nohup python tools/train_yolo.py --hyp configs/hyp/default.yaml --name default_100 > default_100.log 2>&1 &
tail -f default_100.log
```

**Method 3 — set1 (low-light compensation):**
```bash
nohup python tools/train_yolo.py --hyp configs/hyp/set1_japan_night_aug.yaml --name set1_100 > set1_100.log 2>&1 &
```

**Method 4 — set2 (class & scene diversity):**
```bash
nohup python tools/train_yolo.py --hyp configs/hyp/set2_field_imbalance.yaml --name set2_100 > set2_100.log 2>&1 &
```
Each of 2-4 trains from COCO `yolo11s.pt`, 100 epochs, on `splits/train_100.txt`,
validating against `splits/val.txt` during training (`configs/data/bdd100k_vehicle5.yaml`).
Runs land in `runs/detect/<name>/weights/{last,best}.pt`. Relaunch the same
command with `--resume` added to continue after a stop/crash.

## 5. Evaluation — same two commands per arm, on `splits/test.txt` only

**Never touch `splits/test.txt` except for this final scoring pass.** Use
`splits/val.txt` (this project's own 2.5% carve-out) for any earlier
development/sanity checks instead.

```bash
# repeat for each arm's best.pt (vanilla / default_100 / set1_100 / set2_100)
python evaluation/run_inference.py --weights runs/detect/default_100/weights/best.pt \
  --image-list splits/test.txt --images-dir dataset/yolo/images/val \
  --out-dir runs/preds/default_100_test --device 0

python evaluation/eval_detections.py \
  --image-list splits/test.txt --gt-labels dataset/yolo/labels/val \
  --preds runs/preds/default_100_test \
  --attr-index dataset/yolo/attr_index.json \
  --data-config configs/data/bdd100k_vehicle5.yaml \
  --out runs/metrics/default_100_test
```
Reports mAP50 / mAP75 / mAP50-95 sliced by overall / daytime / night /
dawn-dusk / rainy, plus per-class AP — exactly the "test for each time of
day" breakdown the project plan calls for.

## 6. Final comparison table

```bash
python tools/make_results_table.py --metrics-dir runs/metrics --out results/ablation \
  --require vanilla_test default_100_test set1_100_test set2_100_test
```
Writes `results/ablation.md` (paste straight into the writeup) and `.csv`.

---

## Conventions

- One arm = one `--name` = one output dir: `runs/detect/<name>/`.
- Never touch `splits/test.txt` except for final scoring (Section 5).
- Never train on the raw `images/100k/test/` folder — unused, no labels, not
  referenced by any split.
- `workers: 2` / `cache: false` on every arm — enforced by `tools/train_yolo.py`,
  not optional.
- Runs projected to take longer than ~2 nights of shared A40 time: flag it
  with the team before launching, same as before.
