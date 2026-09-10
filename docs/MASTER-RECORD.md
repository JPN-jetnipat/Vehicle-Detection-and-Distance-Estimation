# Master record — nighttime / adverse-environment vehicle detection

**Owner:** Kanade · **Last updated:** 2026-09-09
**Purpose:** the single ordered record of what was run, what was found, why each choice
was made, and how to defend it. Other project docs hold the detail; this one holds the
argument. If a claim isn't in here with its defence, don't put it in the write-up.

**Reading order:** §1 setup → §2 results in chronological order → §3 decisions and their
defences → §4 limitations to disclose → §5 current state.

---

## 1. Fixed experimental setup

Everything below is held constant across every arm. Arms differ in exactly one thing at
a time; that discipline is what makes any of the numbers mean anything.

| | value |
|---|---|
| detector | YOLOv11s (`yolo11s.pt`, ultralytics pip, pinned 8.4.120) |
| dataset | BDD100K, 2018 Kaggle mirror |
| classes (5) | car, truck, bus, motor, bike |
| splits (seed 42, deterministic) | train 60,186 / val 1,542 / test 8,841 |
| unlabeled pool | `splits/pretrain.txt` = the same 60,186 train images |
| recipe | `set3_combined` (see naming hazard below) |
| scorer | pycocotools, NMS fixed at conf 0.001 / iou 0.6 / max_det 300 |
| optimizer | MuSGD, momentum 0.9, warmup_bias_lr 0.0 (see §3.5) |

**Split methodology.** Any image with `weather == undefined` or `timeofday == undefined`
is dropped first. TEST = BDD's official val set, cleaned, touched only for final scoring.
VAL = 2.5% of *each* timeofday stratum of BDD's official train set, sampled per stratum
so val mirrors train's mix. Remainder is the train pool. Weather is retained as
image-level metadata for slicing, not as a stratification axis.

### ⚠️ Naming hazard — three different things are called "set 3"

Get this wrong in the write-up and the whole methods section becomes unreadable.

| name | what it is | in use? |
|---|---|---|
| **`set3_combined`** | the **shared cross-team hyperparameter recipe** — hsv_v 0.25, close_mosaic 20, box 9.0, copy_paste 0.2, cls 0.7, scale 0.7, translate 0.15, patience 0, seed 42, imgsz 640, batch 16, workers 2, cache false | ✅ **this is the recipe** |
| `set3_combined` (as a *run name*) | in this repo it also labels **Method 3** of the augmentation ablation; in the teammates' pipeline the same string labels the **no-augmentation baseline** | ⚠️ ambiguous — always say which |
| `set3_merged_night_localization` | a **different, earlier** hyperparameter-search arm (dfl 1.8, cls 0.5, scale 0.55, hsv_s 0.60) | ❌ not the recipe |

### What "no augmentation" means now

The **custom** augmentation line is closed — synthetic low-light generation and IRFS
oversampling are not being pursued (§2.2 says why). The **built-in ultralytics**
augmentations that are part of `set3_combined` (mosaic, scale, translate, hsv) remain,
because they are part of the recipe that every arm shares. Do not describe the project as
"no augmentation" without that distinction.

### Test slices (built 2026-09-08, `tools/make_test_slices.py`)

Scope moved from "low light" to "adverse environment", so the test set is now sliceable by
weather as well as time of day. Deterministic; the generator asserts that the time-of-day
slices and the weather slices each partition all 8,841 images before writing anything.
Counts were verified cell-by-cell against a teammate's independent table — **all 54 cells
matched exactly.**

| slice | images | instances | | slice | images | instances |
|---|---|---|---|---|---|---|
| day | 4,264 | 51,930 | | night_clear | 3,274 | 31,550 |
| night | 3,863 | 36,751 | | night_rainy | 286 | 2,535 |
| dawndusk | 714 | 8,512 | | night_snowy | 273 | 2,374 |
| clear | 5,345 | 56,667 | | **night_adverse** | 566 | 4,986 |
| overcast | 1,239 | 15,505 | | day_clear | 1,764 | 21,352 |
| partlycloudy | 738 | 9,256 | | day_rainy | 396 | 4,356 |
| snowy | 769 | 8,080 | | day_snowy | 422 | 4,842 |
| rainy | 737 | 7,541 | | **day_adverse** | 823 | 9,252 |
| foggy | **13** | 144 | | **adverse** | 1,519 | 15,765 |
| | | | | benign | 7,322 | 81,428 |

**Reporting rules, decided in advance:**
- **Never report a mAP for `foggy`** — 13 images. Quote the count. It folds into `adverse`
  where it contributes ~1%.
- **No per-class numbers on `night_rainy` / `night_snowy`** — motor and bike are single
  digits there. Per-class needs ≥800 instances.
- `night` and `rainy` overlap, so improving on both says nothing about *which* factor was
  fixed. The triple that separates them is **`day_adverse`** (bad weather, good light) vs
  **`night_clear`** (good weather, bad light) vs **`night_adverse`** (both).

---

## 2. Results, in the order they were produced

### 2.1 Hyperparameter search — Sets 1 / 2 / 3

Scored on `splits/test.txt` with `evaluation/eval_detections.py`.

| metric | Set 1 | Set 2 | Set 3 |
|---|---|---|---|
| overall mAP50 | 0.5392 | **0.5521** | 0.5445 |
| overall mAP50-95 | 0.3409 | **0.3444** | 0.3399 |
| bike mAP50 | 0.4270 | **0.4605** | 0.4538 |
| motor mAP50 | 0.3744 | **0.3918** | 0.3754 |
| dawn/dusk mAP50 | 0.5696 | 0.5703 | **0.5912** |

- **Set 2 is the strongest single arm** of the three (48 of 75 per-class cells).
- **A column-swap error was caught and corrected on 2026-08-21.** Set 1 and Set 2 were
  transposed in the original table. Caught because Set 3's real `set2_100_test.csv`
  matched the doc's "Set 1" column 4 for 4. Independently re-derived. **Keep this in the
  write-up as a data-integrity note** — it is evidence the numbers were checked, not a
  weakness.
- **Finding B — `copy_paste: 0.2` never executed.** `CopyPaste.__call__` returns early
  when `segments` is empty, and BDD in YOLO detect format is bbox-only. Silent no-op,
  verified empirically. It is still in the recipe only to keep the config byte-identical
  across arms. **It must never be reported as an active augmentation.**
- **Finding F — `dfl: 1.8` did not fix the mAP75 collapse** (§2.3). Set 3's day→night car
  mAP75 delta was −10.08, slightly *worse* than Set 1 (−9.78) and Set 2 (−9.88), despite
  dfl being raised specifically to target it.

### 2.2 Four-arm augmentation ablation — a negative result with a mechanism

All four: yolo11s, 100 epochs, seed 42, identical `set3_combined` recipe.

| arm | pool | oversampling | samples/epoch |
|---|---|---|---|
| BASE | original only | none | 60,186 |
| M1 | original + low-light copy of every image | none | 120,372 |
| M2 | original only | IRFS bike ×2.42 / motor ×3.39 | 68,763 |
| M3 | original + low-light copy | same IRFS factors | 137,519 |

`all` mAP50 on `test.txt`:

| slice | BASE | M1 | M2 | M3 |
|---|---|---|---|---|
| daytime | 0.5407 | 0.5320 | **0.5503** | 0.5399 |
| night | **0.5625** | 0.5506 | 0.5586 | 0.5610 |
| dawn/dusk | 0.5739 | 0.5611 | **0.5795** | 0.5758 |

**M1 ranks last on all three slices, including night.** The intervention designed to fix
night made night worse by 1.20 points. These rows carry 8,512–51,930 instances, so unlike
the rare-class numbers they are not noise.

**Two mechanisms, both consistent with the data:**
1. A gamma curve models *dusk*, not night. Real night is point light sources, headlight
   glare and extreme local contrast; global gamma + Poisson–Gaussian noise reproduces
   uniform dimming. Predicts help at dawn/dusk (observed) and none at true night (observed).
2. The transform was applied to images that were *already* night. ~40% of BDD's train split
   is night, and a night image at gamma 3 is nearly black — off-distribution for any test
   image. Roughly 24,000 of the 60,186 augmented copies may have been actively harmful.

**Rare classes — IRFS did do its job.** On the overall test set, M3 vs M1: motor +3.43,
bike +2.35, at a cost of −0.3 to −0.9 on the common classes. Still n=1 and under the
800-instance threshold; treat as indicative.

### 2.3 The finding that motivates the JEPA phase

Car day → night, **mAP75** (47,715 / 35,439 instances):

| BASE | M1 | M2 | M3 |
|---|---|---|---|
| −9.71 | −9.71 | −9.64 | −9.82 |

Night *detection* is nearly intact (mAP50 −3.0 to −3.5). Night **localization** collapses,
identically, across four independent training-data recipes — and Set 3's dfl-targeted
attempt made it marginally worse (−10.08).

**This is the strongest result in the study, and it is the argument for JEPA.** Four data
recipes and one loss-weight intervention could not move it, which points away from data
distribution and toward either representation quality or input resolution. `imgsz: 768` is
the untried resolution lever (rejected so far on shared-RAM grounds); a better backbone
initialization is the representation lever, and that is what the JEPA phase tests.

### 2.4 JEPA Stage 2 — T1 distillation (2026-09-09)

Meta I-JEPA **ViT-H/14 target encoder** → yolo11s backbone layers 0–6, tapped at P4.
Full detail in `claude/jepa-t1-distill-record.md`.

| metric | value |
|---|---|
| final cosine loss | 0.17975 (converged; 0.18155 at e24, LR at its 1e-5 floor) |
| % of total drop by epoch 4 | 94% |
| day/night linear probe | 0.9867 (e4) → 0.9833 (e29), **flat throughout** |
| throughput / wall clock / VRAM | 134 img/s · 3.7 h · 4,348 MiB |
| graft | 126/126 backbone tensors into `weights/init_t1.pt`, 0 non-backbone tensors changed |

**None of these are results.** They are methods detail. See §3.7 and §4.4.

---

## 3. Decisions and how to defend each one

### 3.1 Why JEPA at all
Four training-data recipes left the night mAP75 collapse untouched (§2.3). Data-side fixes
were tried and measured, and they failed. That makes "is this a *representation* problem?"
the natural next question, and self-supervised pretraining is the standard way to ask it.
**Defence: the motivation is an owned measurement, not a literature hunch.**

### 3.2 Why I-JEPA, not V-JEPA
BDD100K's detection set is 70k stills drawn from 70k *different* videos; the raw video
release is ~1.8 TB and the detector is single-frame. V-JEPA has no input to consume here.
`DriveJEPA.pdf` uses V-JEPA for trajectory planning — different task, different modality.
**Defence: cite it as related work, never as a method we could have followed.**

### 3.3 Why distillation, not swapping in the ViT
Three options were considered. (a) Replace YOLO's backbone with the ViT — breaks the
"arms differ only in init" contrast, needs a new neck, slow at 640px. (b) Distil frozen
ViT features into the conv backbone, then fine-tune normally — chosen. (c) ViT as
auxiliary supervision during detection fine-tuning — more novel, much harder to defend,
no code. **Defence for (b): it isolates backbone initialization as the single variable.**

### 3.4 Why the P4 tap at layer 6
Verified empirically, not assumed: yolo11s layer 6 outputs **256ch @ 14×14** at 224px
input — an exact grid match for a ViT-B/16 teacher and one bilinear resize from ViT-H/14's
16×16. It is also byte-for-byte the same shape the archived YOLOv5s pipeline used, so the
design note carries over unchanged. Layers 7–10 (P5 Conv, C3k2, SPPF, **C2PSA**), neck and
head take COCO weights in **every** arm, identically. **Note for the write-up: C2PSA is an
attention block YOLOv5 didn't have; it sits after the tap and is COCO-initialized in all
arms, so it is not part of what JEPA touches.**

### 3.5 Why the optimizer is pinned
`optimizer: auto` in ultralytics 8.4.x selects MuSGD, momentum 0.9, warmup_bias_lr 0.0
whenever iterations > 10,000 — not the SGD/0.937/0.1 printed as defaults. Every arm so far
ran MuSGD without that being an explicit choice. **The knock-on that matters: the threshold
is on iterations, not epochs.** `train_10.txt` (6,019 images) at 100 epochs = 9,500
iterations, *below* the threshold, so on `auto` a 10%-label arm silently trains with AdamW
at a different LR. **A label-fraction ablation left on `auto` would compare optimizers, not
label fractions.** Pin it explicitly before running the 10% grid.

### 3.6 Why 30 distillation epochs
**Honest provenance: a budget default**, carried from the archived project's
`STAGE2_DISTILL_DESIGN.md` sign-off item (d), chosen so the stage fits one GPU session.
No paper, no ablation behind it.
**Earned defence:** the objective converged — the last 10 epochs moved the loss 0.0065, the
last 5 moved it 0.0018, with LR at its 1e-5 floor throughout. Show the curve.
**Do not claim** I-JEPA's 300–600 epochs as the source (that's *pretraining*, a different
stage), and **do not claim** "15 epochs would have sufficed" — the cosine schedule is
defined over 30, so epoch 15 sits mid-schedule at LR 5.3e-4, not converged.

### 3.7 Why the distillation metrics are not evidence of quality
- **Cosine distance isn't comparable across architectures.** The archived YOLOv5 run also
  finished near 0.186 with a 0.98 probe. That is a coincidence — different students have
  different achievable floors — not a replication.
- **A falling cosine loss can be earned cheaply**, by matching brightness and gross layout
  without learning anything semantic.
- **The day/night probe is uninformative here.** It saturated at epoch 4 and never moved
  for 25 epochs while the loss kept falling. Raw pixel brightness already predicts
  day/night. Report it as a non-collapse sanity check and nothing more.

### 3.8 Why the graft is verified, and what `--verify` actually checks
`tools/make_init_weights.py --verify` confirms all 126 backbone tensors match the distilled
file and that **zero** non-backbone tensors differ from stock `yolo11s.pt`. It refuses two
specific failure modes, both tested: a partial graft (a missing layer → exit 1) and a scale
mismatch (yolo11m weights into yolo11s → exit 1). Both would otherwise produce an arm that
is silently part-COCO and scores near BASE.

**Fixed 2026-09-09:** the first `--verify` run reported 125/126. Cause was the checker, not
the graft — checkpoints are stored fp16 (same as stock `yolo11s.pt`), and an *absolute*
1e-3 tolerance falsely flags BatchNorm `running_var` buffers whose post-training magnitudes
reach the hundreds, where fp16 spacing is ~0.12. Reproduced deliberately, then fixed to
compare against the same fp16 round-trip **exactly** — stricter than the old check, and it
still catches a 1%-corrupted weight. Re-run `--verify` after pulling; it must print 126/126.

### 3.9 Why the arms stay separate and every result is kept
Each arm = one `--name` = its own `runs/detect/<name>/`. `tools/train_yolo.py --model`
defaults to `yolo11s.pt`, so the plain YOLOv11s path is completely unaffected by any of the
JEPA work — `weights/init_t1.pt` is an *additional* file, not a replacement, and the recipe
yaml is byte-identical for both. **The one real risk to "keep every result": the recipe
sets `exist_ok: true`, so re-running the same `--name` overwrites that run directory
silently. Always give a new run a new name.** `runs/` is gitignored, so results live only
on the server disk — copy each run's `results.csv` and `args.yaml` into a tracked
`results/` directory as it finishes.

---

## 4. Limitations to disclose, not hide

1. **Test-set leakage.** The `set3_combined` recipe was chosen from `test.txt` observations,
   which the runbook reserves for final scoring. **Every** `test.txt` number in this project
   — including the JEPA arms — carries the same optimistic bias. It is identical across
   arms, so *between-arm deltas remain fair*; absolute numbers do not. Say this explicitly.
2. **n = 1 per arm.** Seed variance is unquantified, and the augmentation ablation's spread
   on `all` rows was ±1 point. One repeated BASE run at a different seed would establish the
   noise floor and settle whether a 1-point JEPA gain means anything. One night of GPU time;
   the cheapest credibility available.
3. **T1 vs T2 confounds scale with domain.** ViT-H/14 (630M, ImageNet) vs ViT-B/16 (86M,
   BDD). Cannot be fixed within budget. Disclose.
4. **T2 will be data-starved and there is no warm start.** Meta released I-JEPA weights only
   for ViT-H/14, ViT-H/16-448 and ViT-g/16 — **no ViT-B**. So T2 trains from scratch on
   60,186 images where the paper used 1.28M, ~20× less. Either frame that as the research
   question, or warm-start from MAE `vit-mae-base` (same arch and patch size) as a
   documented deviation.
5. **Distillation length was not optimized.** Loss convergence ≠ optimal transfer; a student
   can keep improving downstream after the matching loss plateaus. Testing costs one full
   fine-tune per length (~2 nights each). Also `backbone_distilled.pt` is overwritten each
   epoch, so no intermediate snapshots exist — an ablation needs a fresh run.
6. **Two evaluators existed historically.** M3 was scored with `eval_detections.py`;
   M1/M2/BASE with pycocotools. Instance counts matched exactly across all four, but score
   one common arm through both before publishing any cross-pipeline delta.
7. **Rare-class thresholds.** Any per-class number resting on <800 instances is indicative
   only. The most-quoted figure from the ablation — M3 motor at dawn/dusk, +5.57 — rests on
   **22 boxes.**

---

## 5. Current state and next steps

**Done:** splits frozen · 19 test slices built and cross-verified · hyperparameter search ·
four-arm augmentation ablation (closed, negative result) · JEPA code ported to YOLOv11 and
verified · T1 distillation converged · `weights/init_t1.pt` grafted.

**Deliverable being built — one table:**

| arm (backbone init) | 100% labels | 10% labels |
|---|---|---|
| BASE — COCO `yolo11s.pt` | ✅ done | ⬜ |
| T1 — I-JEPA ViT-H/14 target encoder, distilled | 🔧 fine-tune next | ⬜ |
| T2 — I-JEPA ViT-B/16 pretrained on BDD, distilled | ⬜ | ⬜ |

Scored on: overall · day · night · dawndusk · clear · rainy · snowy · adverse ·
night_adverse · day_adverse · night_clear. Plus **car day→night mAP75 for every arm** —
whether a JEPA backbone moves the −9.7 is the sharpest question in the study.

**Immediate order:**
1. Fine-tune T1 at 100% labels (`--model weights/init_t1.pt --name t1_jepa_100`), ~2 nights.
2. Build `evaluation/score_slices.py` so Phase 3 isn't blocked (not yet started).
3. **Pin the optimizer explicitly before any 10%-label run** (§3.5).
4. T2: Stage-1 I-JEPA pretrain (10–15 h) → distil → graft → fine-tune.
5. Seed repeat of BASE to establish the noise floor (§4.2).

**Note on expectations:** BASE at 100% labels is already strong, and the augmentation
ablation showed how hard it is to beat. If JEPA wins anywhere it is most likely the
**10%-label column**, which is where self-supervised pretraining reliably shows its gains.
Treat the 10% grid as core, not optional.
