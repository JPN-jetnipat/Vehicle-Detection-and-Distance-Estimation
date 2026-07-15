# RUNBOOK — I-JEPA pretraining → YOLOv5 nighttime vehicle detection (Path B)

All commands run **on the university GPU server** (A40 46 GB, CUDA 12.4) inside a
JupyterLab terminal, unless marked `[mac]`. I (Claude) write scripts/configs;
**you execute everything**. Sections marked ⏳ land with their milestone.

**Before launching ANY job:**
```bash
nvidia-smi        # who else is on the GPU + free VRAM
free -h           # system RAM - the server has only 32 GB SHARED (crashed once already!)
df -h ~           # disk headroom before large downloads/extractions
```
Long jobs: always `tmux` or `nohup … &` — VPN/Jupyter sessions drop.

**RAM rules (staff directive, 2026-07 — the server crashed on a 100k run):**
- `workers: 2` everywhere (already the default in all our configs/scripts).
- Detection training: per-step batch 16 (configs updated); YOLOv5 internally
  accumulates gradients to nominal batch 64, so results stay comparable to
  the friend's batch-64 recipe.
- **NEVER pass `--cache` / `--cache ram`** to yolov5 train.py — it tries to
  load the whole dataset into RAM. Our wrapper never does; don't add it by hand.
- The converter + attr-index scripts stream the 1.4 GB JSON (ijson) — safe.

---

## 0. One-time git setup

Repo syncs Mac ↔ server via the private GitHub remote (you handle push/pull).

```bash
git clone <your-private-remote-url> vehicle-jepa && cd vehicle-jepa
```

## 1. Environment (server) — ~10 min

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# expect: 2.5.1+cu124 True NVIDIA A40
python -c "import torch; print('bf16 ok:', torch.cuda.is_bf16_supported())"   # expect True on A40
```

Kaggle notebooks: skip the torch line (preinstalled); `pip install -r requirements.txt`
minus torch. Use `--amp-dtype fp16` on P100 (no bf16).

## 2. Dataset onto the server (~9.9 GB) — one-time

**Path A — Kaggle CLI (try first; friend hit 401s in June, may be fixed):**
```bash
df -h ~                                    # need ≥ 25 GB free (zip + extracted)
mkdir -p ~/.config/kaggle                  # put kaggle.json (from kaggle.com → Settings → API) here
chmod 600 ~/.config/kaggle/kaggle.json
kaggle datasets download -d solesensei/solesensei_bdd100k -p dataset/raw/
unzip -q dataset/raw/solesensei_bdd100k.zip -d dataset/raw/
```

**Path B — fallback if CLI still 401s:** chunked zip upload through the Jupyter
file browser; full tested procedure in `docs/dataset-upload-guide.md`.

**Verify layout (either path):**
```bash
ls dataset/raw/bdd100k/bdd100k/images/100k/          # train/ val/ test/
ls dataset/raw/bdd100k/bdd100k/images/100k/train | wc -l   # ≈ 70_000
ls dataset/raw/bdd100k/bdd100k/images/100k/val   | wc -l   # = 10_000
ls dataset/raw/bdd100k_labels_release/bdd100k/labels/       # train + val JSONs
```
⚠️ FLAG me if counts mismatch. The `images/10k/` folder is the segmentation
subset — ignore it entirely.

## 3. Data plumbing (Milestone 2) — READY. ~10 min total, CPU-only, no GPU needed.

Class list: **RESOLVED (2026-07-13)** — canonical config is
`configs/data/bdd100k_vehicle3.yaml` (3 vehicle classes: car/bus/truck, ids
matching the friend's protocol). `bdd100k_all10.yaml` is reference-only.

Both the converter and the attr-index script STREAM the 1.4 GB train JSON
(peak RAM ~tens of MB) — safe for the shared 32 GB server.

**3.1 Attribute index** (~2 min):
```bash
python tools/build_attr_index.py \
  --train-json dataset/raw/bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_train.json \
  --val-json   dataset/raw/bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_val.json \
  --out-prefix dataset/yolo/attr_index \
  --expect-counts train=69863,val=10000    # confirmed full official train set (FLAG 4 resolved 2026-07-14)
```
Confirmed 2026-07-14: server has the complete official set (69,863 train / 10,000
val) — an earlier partial local copy had shown only 64,520; that's resolved.
splits/ were regenerated against the full 69,863 (still seed 42).

**3.2 Convert labels** (~3 min each):
```bash
python tools/bdd_to_yolo.py --labels-json dataset/raw/bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_train.json \
  --data-config configs/data/bdd100k_vehicle3.yaml --out-dir dataset/yolo/labels/train \
  --stats-out dataset/yolo/convert_stats_train.json
python tools/bdd_to_yolo.py --labels-json dataset/raw/bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_val.json \
  --data-config configs/data/bdd100k_vehicle3.yaml --out-dir dataset/yolo/labels/val \
  --stats-out dataset/yolo/convert_stats_val.json \
  --images-dir dataset/raw/bdd100k/bdd100k/images/100k/val --verify-sizes 50
```
Val sanity numbers (vehicle3 config): 10,000 images, 108,345 boxes kept
(car 102,504 / bus 1,597 / truck 4,244), 77,178 non-vehicle boxes skipped,
3 degenerate, 96 empty (background) label files.
Train sanity numbers (full 69,863, verified on server 2026-07-14): 754,807
boxes kept (car 713,166 / bus 11,672 / truck 29,969), 47 degenerate, 606 empty
label files. Splits: modelsel 2,500 / pool 67,363 (train_100 67,363, train_50
33,682, train_25 16,841, train_10 6,736).

**3.3 Materialize split lists + image symlinks**:
```bash
python tools/make_splits.py --attr-index dataset/yolo/attr_index.json \
  --repo-splits splits --yolo-splits dataset/yolo/splits --seed 42
# should print counts identical to splits/MANIFEST.json (committed)
mkdir -p dataset/yolo/images
ln -sfn "$(pwd)/dataset/raw/bdd100k/bdd100k/images/100k/train" dataset/yolo/images/train
ln -sfn "$(pwd)/dataset/raw/bdd100k/bdd100k/images/100k/val"   dataset/yolo/images/val
```

**3.4 Eyeball a few overlays** (open the jpgs in the Jupyter file browser):
```bash
python tools/viz_boxes.py --images-dir dataset/raw/bdd100k/bdd100k/images/100k/val \
  --labels-dir dataset/yolo/labels/val --data-config configs/data/bdd100k_vehicle3.yaml \
  --num 8 --out-dir viz_out
```

## 4. Baseline arm — COCO-init YOLOv5 (Milestone 3) — READY (provisional recipe)

Recipe = the friend's (RESOLVED 2026-07-13): yolov5s, 640 px, batch 64,
100 epochs, default hyps. Arms T1/T2 copy the same config, ONLY `weights:`
differs.

**4.1 COCO weights** (~15 MB; if the server has no internet, download on the
Mac from https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt
and upload via Jupyter):
```bash
ls yolov5s.pt || wget https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt
```

**4.2 Train** (A40: roughly 9–14 h for 100 epochs @ per-step batch 16 / 640 px, ~6 GB
VRAM — check `nvidia-smi` first; use tmux):
```bash
nohup python tools/train_yolo.py --config configs/exp/baseline_coco_100.yaml > baseline_100.log 2>&1 &
tail -f baseline_100.log
```
Auto-resumes from `runs_jepa/stage3/baseline_coco_100/weights/last.pt` if
relaunched after a kill. The 10% arm is `configs/exp/baseline_coco_10.yaml`
(~1–1.5 h).

**4.3 Predict + score on the model-selection split** (development loop —
NEVER on val until the final table):
```bash
python evaluation/run_inference.py \
  --weights runs_jepa/stage3/baseline_coco_100/weights/best.pt \
  --image-list dataset/yolo/splits/modelsel.txt --images-root dataset/yolo \
  --out-dir runs_jepa/preds/baseline_coco_100_modelsel --device 0
python evaluation/eval_detections.py \
  --image-list splits/modelsel.txt --gt-labels dataset/yolo/labels/train \
  --preds runs_jepa/preds/baseline_coco_100_modelsel \
  --attr-index dataset/yolo/attr_index.json \
  --data-config configs/data/bdd100k_vehicle3.yaml \
  --out runs_jepa/metrics/baseline_coco_100_modelsel
```
(Final scoring later: same two commands with `splits/val_final.txt`,
`dataset/yolo/labels/val`, ~5–10 min GPU inference for 10k images.)

**4.4 Re-score the friend's weights** (his 10k-subset best.pt is at
`weights/friend_best.pt`; upload it to the server — it is gitignored):
```bash
python evaluation/run_inference.py --weights weights/friend_best.pt \
  --image-list dataset/yolo/splits/modelsel.txt --images-root dataset/yolo \
  --out-dir runs_jepa/preds/friend_best_modelsel --device 0
python evaluation/eval_detections.py --image-list splits/modelsel.txt \
  --gt-labels dataset/yolo/labels/train --preds runs_jepa/preds/friend_best_modelsel \
  --attr-index dataset/yolo/attr_index.json --data-config configs/data/bdd100k_vehicle3.yaml \
  --out runs_jepa/metrics/friend_best_modelsel
```
His nc=8 head has 5 dead classes; the evaluator drops those predictions
automatically (it prints how many).
## 5. T1 path — ViT-H/14 distill → fine-tune (Milestone 4) — READY

Design signed off 2026-07-12 (docs/STAGE2_DISTILL_DESIGN.md).

**5.1 Teacher checkpoint** (~2.5 GB; if the server has no internet, run the
same command on the Mac and upload the folder via Jupyter):
```bash
huggingface-cli download facebook/ijepa_vith14_1k --local-dir weights/ijepa_vith14_1k
```

**5.2 Smoke test the wiring** (~2 min, CPU or GPU — run before the real thing):
```bash
python jepa_distill/train_distill.py --config configs/exp/distill_t1.yaml --smoke
```
Expect: teacher/student shape printout (teacher dim=1280 grid=16, student P4
256ch 14x14), loss starting near 1.0 and moving. FLAG me anything odd.

**5.3 Distill** (A40: ~1.5–3 h, ~8–12 GB VRAM; check nvidia-smi, use tmux):
```bash
nohup python jepa_distill/train_distill.py --config configs/exp/distill_t1.yaml > distill_t1.log 2>&1 &
tail -f distill_t1.log
```
Watch: loss trending down; `[probe]` day/night accuracy every 5 epochs should
sit well above 0.5 and climb/plateau. Auto-resumes from ckpt_latest.pt.

**5.4 Assemble the stage-3 init** (seconds):
```bash
python tools/make_init_weights.py \
  --distilled runs_jepa/stage2/distill_t1/backbone_distilled.pt \
  --base yolov5s.pt --out weights/init_t1.pt
```
Policy (fairness): distilled backbone layers 0–6 + COCO neck/head — every arm
shares neck/head init and differs only in backbone.

**5.5 Fine-tune + score — identical to section 4** but with configs
`t1_distill_100.yaml` / `t1_distill_10.yaml` (only name+weights differ from
baseline), then run_inference + eval_detections on modelsel. Same runtimes.
## 6. T2 path — I-JEPA ViT-B/16 pretraining on BDD (Milestone 5) — READY

Official ijepa code is vendored at `third_party/ijepa` (no download needed).

**6.1 Smoke test** (~2 min):
```bash
python jepa_pretrain/train_ijepa.py --config configs/exp/pretrain_t2.yaml --smoke
```
Expect: composition printout (night/dawn-dusk/rainy shares roughly doubled vs
raw), loss finite and moving, `tstd` (target std) well above 0.

**6.2 Pretrain** (A40: ~10–15 h for 300 epochs, ~20 GB VRAM at micro-batch 128
— drop `micro_batch` to 64 if the GPU is busy; check nvidia-smi, use tmux):
```bash
nohup python jepa_pretrain/train_ijepa.py --config configs/exp/pretrain_t2.yaml > pretrain_t2.log 2>&1 &
tail -f pretrain_t2.log
```
Watch every few hours: loss decreasing (NOT instantly ~0 - that's collapse);
`tstd` staying comfortably above 0.01 (the script prints a COLLAPSE ALARM
otherwise). Auto-resumes. To extend 300→600 epochs later: raise `epochs` in
the config and relaunch — it resumes and continues.
⚠️ If the first-hour img/s projects total time > 2 days, STOP and flag me.

**6.3 Distill T2 → backbone** (same machinery as T1, ~30–60 min):
```bash
python jepa_distill/train_distill.py --config configs/exp/distill_t2.yaml --smoke   # wiring check
nohup python jepa_distill/train_distill.py --config configs/exp/distill_t2.yaml > distill_t2.log 2>&1 &
```

**6.4 Assemble + fine-tune + score** — same as 5.4/5.5 with t2 names:
```bash
python tools/make_init_weights.py \
  --distilled runs_jepa/stage2/distill_t2/backbone_distilled.pt \
  --base yolov5s.pt --out weights/init_t2.pt
nohup python tools/train_yolo.py --config configs/exp/t2_distill_100.yaml > t2_100.log 2>&1 &
# then run_inference + eval_detections on modelsel, as in 4.3
```
## 7. Label-fraction grid + final eval (Milestones 6–7) — READY

**7.1 The 10% arms** (~1–1.5 h each; Kaggle-suitable, see appendix):
`baseline_coco_10.yaml`, `t1_distill_10.yaml`, `t2_distill_10.yaml` via
tools/train_yolo.py — identical procedure to their 100% versions.

**7.2 FINAL scoring — the only time BDD val is touched.** For every finished
arm (and weights/friend_best.pt), run inference + eval with the val lists:
```bash
python evaluation/run_inference.py --weights <arm>/weights/best.pt \
  --image-list dataset/yolo/splits/val_final.txt --images-root dataset/yolo \
  --out-dir runs_jepa/preds/<arm>_val --device 0
python evaluation/eval_detections.py --image-list splits/val_final.txt \
  --gt-labels dataset/yolo/labels/val --preds runs_jepa/preds/<arm>_val \
  --attr-index dataset/yolo/attr_index.json --data-config configs/data/bdd100k_vehicle3.yaml \
  --out runs_jepa/metrics/<arm>_val
```

**7.3 Ablation table:**
```bash
python tools/make_results_table.py --metrics-dir runs_jepa/metrics --out results/ablation
```

---

## Appendix — running small jobs on Kaggle (free tier)

Good for: the 10% arms, re-scoring weights, eval inference. NOT for 100%
arms (40–70 h on a T4 — exceeds quota) or Stage-1 pretraining.

1. New notebook → Settings: GPU (T4 preferred; P100 works — scripts fall back
   to fp16 automatically), Internet ON.
2. Add data → search `solesensei_bdd100k` → attach (no upload needed; mounts
   read-only at `/kaggle/input/solesensei_bdd100k/`).
3. First cell:
```
!git clone https://<your-token>@github.com/<you>/<repo>.git proj
%cd proj
!pip -q install -r requirements.txt 2>&1 | tail -1   # torch is preinstalled - do NOT reinstall
!mkdir -p dataset/raw && ln -s /kaggle/input/solesensei_bdd100k/bdd100k dataset/raw/bdd100k \
  && ln -s /kaggle/input/solesensei_bdd100k/bdd100k_labels_release dataset/raw/bdd100k_labels_release
```
   ⚠️ Verify the exact folder names inside /kaggle/input first (`!ls`) — the
   mirror's nesting may differ; adjust the symlinks accordingly, and FLAG me
   if label counts mismatch RUNBOOK section 3 (that also answers FLAG 4).
4. Run RUNBOOK section 3 (attr index, convert, splits, symlinks), then e.g.:
```
!python tools/train_yolo.py --config configs/exp/baseline_coco_10.yaml --device 0
```
5. Sessions cap at ~12 h. `runs_jepa/` lives in /kaggle/working (persisted on
   Save & Run All / manual save). To resume after a session death: re-run the
   setup cells, restore runs_jepa from the previous version's output, relaunch
   — the wrapper auto-resumes from last.pt.

---

## Conventions (every run, every stage)

- One experiment = one config = one output dir: `runs_jepa/<stage>/<exp_name>/`
  containing resolved config dump, git hash, seed, checkpoints, TensorBoard
  events, metrics CSV/JSON.
- Every training script supports `--resume` / auto-detects latest checkpoint.
- bf16 autocast + grad accumulation on the A40; leave VRAM headroom (shared GPU).
- Never touch BDD **val** except for final scoring. Never train on the 20k test folder.
- Runs projected > ~2 days wall-clock get FLAGged before launch.
