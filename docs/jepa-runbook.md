# JEPA RUNBOOK — exact commands

Companion to `docs/jepa-plan.md` (what and why). This file is only how.
All commands run **from the repo root**, on the A40 server inside `tmux`,
`.venv` active. Phase numbering matches the plan.

Phase 0 (the port) is **done** — this file starts at the first command you
actually need to run on the server.

---

## 0. What Phase 0 left you

```
third_party/ijepa/          vendored official Meta I-JEPA code (unchanged)
jepa_pretrain/train_ijepa.py   Stage 1 — I-JEPA pretraining     (patched: adverse-scope oversampling)
jepa_distill/data.py           shared 224px view                (unchanged)
jepa_distill/teacher.py        frozen ViT teacher wrappers      (patched: torch 2.6 load)
jepa_distill/student.py        YOLOv11 backbone student         (REWRITTEN)
jepa_distill/train_distill.py  Stage 2 — distillation loop      (patched)
tools/make_init_weights.py     Stage 3 — graft into yolo11s.pt  (REWRITTEN)
tools/patch_jepa_port.py       the idempotent patcher, kept for provenance
evaluation/score_slices.py     Phase 4 — all 19 slices, one inference pass  (added 2026-09-10)
configs/jepa/{pretrain_t2,distill_t1,distill_t2}.yaml
```

Verified in a clean torch 2.9 / ultralytics 8.4 environment on 2026-09-08:

- student taps yolo11s layer 6 → **256ch @ 14×14** at 224px input (identical to
  the YOLOv5s tap the archived code used), gradient reaches all 63 parameters
  of layers 0–6, and `backbone_state_dict()` emits 126 tensors spanning layers
  0–6 (63 params + 63 BN buffers).
- `make_init_weights.py` grafts all 126, leaves every non-backbone tensor
  byte-identical to `yolo11s.pt`, round-trips through `YOLO(...)`, and **refuses**
  both failure modes it was built to catch: a partial graft (dropped layer 3 →
  exit 1) and a scale mismatch (yolo11m backbone into yolo11s → exit 1).
- the widened oversampling moves night-or-adverse from **54.7%** of the pool to
  **71.1%** of sampled epochs, and rejects a mistyped config key.

Not yet verified, because it needs the dataset and the teacher checkpoint —
that is step 1 below.

## 1. Environment (once per server checkout)

```bash
source .venv/bin/activate
pip install -r requirements.txt        # picks up the new tensorboard pin
python -c "import torch, tensorboard; print(torch.__version__, torch.cuda.is_available())"
```

## 2. Wiring smoke tests — CPU is fine, ~2 min each. Do NOT skip.

```bash
# Stage 2 wiring (needs the teacher tar from step 3 first if you run T1;
# for a teacher-free check, run the student self-test):
python jepa_distill/student.py
# expect: "student P4 256ch 14x14, proj out (2, 1280, 14, 14), grads 63/63"

# Stage 1 wiring — 64 images, 1 epoch, saves nothing:
python jepa_pretrain/train_ijepa.py --config configs/jepa/pretrain_t2.yaml --smoke
# expect: composition printout (night_or_adverse raw 0.5466 -> sampled 0.7112),
#         finite loss that moves, tstd well above 0
```

**Gate:** if `--smoke` prints a missing-images warning above 1% it exits by
design — that means the server's copy of BDD is incomplete, not that the code
is broken. Fix the data first.

## 3. T1 arm — Meta's ViT-H/14, no pretraining needed

**3.1 Fetch the TARGET encoder.** The HuggingFace conversion exports the
*context* encoder; the archived project burned a full run on it.

```bash
df -h ~     # the tar is 9.6 GB, NOT the ~2.5 GB an earlier draft of this file said
wget https://dl.fbaipublicfiles.com/ijepa/IN1K-vit.h.14-300e.pth.tar -P weights/
```

**Disk (measured 2026-09-09):** `/` was at 97% (32 GB free) before this download,
so the tar alone takes a third of the headroom. Budget for the rest:

| item | size | lifetime |
|---|---|---|
| `IN1K-vit.h.14-300e.pth.tar` | 9.6 GB | **delete after the T1 distill** - it is re-downloadable in <3 min at 60 MB/s and is never needed again once `backbone_distilled.pt` exists |
| stage-2 `ckpt_latest.pt` (student+optimizer) | ~30 MB | overwritten each epoch |
| stage-1 `ckpt_latest.pt` (encoder+predictor+target+optimizer) | ~1.6 GB | overwritten each epoch |
| stage-1 `target_encoder_e{50,100,...}.pt` | ~350 MB x 6 | `keep_every: 50` - raise it to 100 if disk is tight |
| each YOLO fine-tune run (`save_period: 10`) | ~250 MB | keep |

Re-check `df -h ~` before launching stage 1 - it is the only job here that can
fill a shared disk, and a disk-full at hour 12 of 15 loses the run.

**3.2 Smoke, then distill** (~3 h, ~4 GB VRAM at the archived project's measured
189 img/s):

```bash
python jepa_distill/train_distill.py --config configs/jepa/distill_t1.yaml --smoke
# expect: "teacher: t2_ijepa dim=1280 grid=16; student P4: 256ch 14x14"

nvidia-smi && free -h
nohup python jepa_distill/train_distill.py --config configs/jepa/distill_t1.yaml > distill_t1.log 2>&1 &
tail -f distill_t1.log
```

Watch: cosine loss falling from ~1.0 toward ~0.2, and `[probe]` day/night linear
accuracy every 5 epochs sitting well above 0.5. Auto-resumes from
`runs_jepa/stage2/distill_t1/ckpt_latest.pt`.

**3.3 Graft, then verify the graft:**

```bash
python tools/make_init_weights.py \
  --distilled runs_jepa/stage2/distill_t1/backbone_distilled.pt \
  --base yolo11s.pt --out weights/init_t1.pt

python tools/make_init_weights.py \
  --distilled runs_jepa/stage2/distill_t1/backbone_distilled.pt \
  --base yolo11s.pt --verify weights/init_t1.pt
# expect: 126/126 backbone tensors match, 0 non-backbone tensors differ
```

**3.4 Fine-tune — the set3 recipe, with `model:` as the ONLY change.**

```bash
nohup python tools/train_yolo.py --hyp configs/hyp/set3_combined.yaml \
  --model weights/init_t1.pt --name t1_jepa_100 > t1_jepa_100.log 2>&1 &
```

`tools/train_yolo.py` already takes `--model` (default `yolo11s.pt`), so the
hyp yaml stays byte-identical to the BASE arm's — verified 2026-09-08. The
`--data` default is `configs/data/bdd100k_vehicle5.yaml`; pass `--data` to
switch to the 10%-label yaml.

## 4. T2 arm — your own I-JEPA pretraining (the contribution)

**4.1 Announce it on the GPU group line first — this is a 10–15 h job.**

```bash
nvidia-smi && free -h && df -h ~
nohup python jepa_pretrain/train_ijepa.py --config configs/jepa/pretrain_t2.yaml > pretrain_t2.log 2>&1 &
tail -f pretrain_t2.log
```

Watch, every few hours:
- loss decreasing but **not** collapsing to ~0 instantly;
- `tstd` (target-embedding std) staying well above 0.01 — the script prints
  `COLLAPSE ALARM` and tells you to stop if it doesn't;
- img/s in the first hour. **If it projects past 2 days, stop** and cut to 150
  epochs or `arch: vit_small` rather than silently overrunning.

Auto-resumes. To extend 300 → 600 epochs later, raise `epochs` and relaunch.

**4.2 Distill + graft** (~30–60 min; ViT-B is ~5× lighter than ViT-H):

```bash
python jepa_distill/train_distill.py --config configs/jepa/distill_t2.yaml --smoke
nohup python jepa_distill/train_distill.py --config configs/jepa/distill_t2.yaml > distill_t2.log 2>&1 &

python tools/make_init_weights.py \
  --distilled runs_jepa/stage2/distill_t2/backbone_distilled.pt \
  --base yolo11s.pt --out weights/init_t2.pt
python tools/make_init_weights.py \
  --distilled runs_jepa/stage2/distill_t2/backbone_distilled.pt \
  --base yolo11s.pt --verify weights/init_t2.pt
```

**4.3 Fine-tune** — same as 3.4 with `weights/init_t2.pt` and `--name t2_jepa_100`.

## 5. The 10%-label grid

**Clear two blockers first.**

```bash
# (a) the 10% data yaml does not exist yet
sed 's#^train: splits/train_100.txt#train: splits/train_10.txt#' \
  configs/data/bdd100k_vehicle5.yaml > configs/data/bdd100k_vehicle5_10.yaml
grep '^train:' configs/data/bdd100k_vehicle5_10.yaml     # must read splits/train_10.txt

# (b) PIN THE OPTIMIZER in the recipe before any 10% run
grep -n 'optimizer' configs/hyp/set3_combined.yaml
```

`train_10.txt` is 6,019 images; ultralytics computes
`ceil(6019 / max(batch, nbs=64)) x 100 epochs = 9,500` iterations, **below** its 10,000
threshold, so `optimizer: auto` silently selects AdamW while every 100%-label arm ran
MuSGD. Unpinned, this column compares optimizers, not label fractions
(`MASTER-RECORD.md` §3.5).

Then the same commands with `--data configs/data/bdd100k_vehicle5_10.yaml` and names
`base_10 / t1_jepa_10 / t2_jepa_10`. ~3–4 h each. This is where SSL pretraining usually
shows its gain — after the T1@100% result it is the **core** of the deliverable, not an
extension.

## 6. Scoring — all 19 test slices, one scorer, fixed NMS

```bash
python tools/make_test_slices.py          # regenerate if attr_index ever changes
```

Then score **every arm you intend to compare in one invocation** — separate
invocations can silently differ in protocol:

```bash
python evaluation/score_slices.py \
  --arm base_100=weights/external/base_100_friend.pt \
  --arm t1_jepa_100=runs/detect/t1_jepa_100/weights/best.pt \
  --device 0 --out results/slice_scores_t1.csv
```

It runs inference once over all 8,841 test images and restricts `COCOeval.params.imgIds`
per slice, so 19 slices cost one inference pass, not 19. NMS is hard-pinned at
conf 0.001 / iou 0.6 / max_det 300 and is deliberately not configurable. Writes a
720-row CSV plus a per-slice/per-class JSON, and prints the headline table and the
car day→night mAP75 line.

**To measure the noise floor for free, add each arm's `last.pt` as an extra `--arm`.**
The within-arm best/last gap bounds run-to-run variance, which is what decides whether
a 1-point between-arm delta means anything (`MASTER-RECORD.md` §4.2).

Headline slices:

```
overall  day  night  dawndusk  clear  rainy  snowy  adverse  night_adverse  day_adverse  night_clear
```

Do **not** report a mAP for `foggy` (13 images / 144 boxes) — quote the count.
Do **not** report per-class numbers on `night_rainy` / `night_snowy` — motor and
bike are single digits there.

Also report, for every arm, **car day→night mAP75**. BASE/M1/M2/M3 all sat at
−9.6 to −9.8 points. Whether a JEPA-initialized backbone moves that number is
the sharpest question in the study.

---

## Conventions inherited from the main RUNBOOK

- One arm = one `--name` = one run dir. Never delete a run; rename aborted ones.
- `workers: 2` / `cache: false` everywhere — enforced by `train_yolo.py`.
- `splits/test.txt` and its slices are for final scoring only; develop on
  `splits/val.txt`.
- Anything projected over ~2 nights of shared A40 time gets announced first.
