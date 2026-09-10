# Master record — nighttime / adverse-environment vehicle detection

**Owner:** Kanade · **Last updated:** 2026-09-10
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

> `set3_combined` = **Set 2 + `hsv_v` 0.4→0.25 + `close_mosaic` 10→20 + `box` 7.5→9.0**.
> Re-examined 2026-09-10 against Set 2 and confirmed as the recipe — see §3.11 for the
> per-class decomposition and the defence. **Do not switch recipes mid-study.**
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

**None of these are results.** They are methods detail. See §3.7 and §4.5.

---

### 2.5 JEPA Stage 3 — T1 fine-tuned at 100% labels (2026-09-10)

`runs/detect/t1_jepa_100/weights/best.pt` against the BASE arm, both scored in a
**single** `evaluation/score_slices.py` invocation so the two cannot silently differ
in protocol.

**Comparator provenance.** BASE here is `weights/external/base_100_friend.pt`, a
checkpoint received from a teammate. It is the same BASE arm already in §2.2: through
the new slice scorer it returns day **0.54074** / night **0.56255** / dawndusk
**0.5739**, against §2.2's `eval_arm_coco.py` values of 0.5407 / 0.5625 / 0.5739 —
exact to four decimals. Record its `sha256sum` and an `args.yaml` diff here to close
the point beyond argument.

| slice | BASE mAP50 | T1 mAP50 | Δ | BASE mAP75 | T1 mAP75 | Δ |
|---|---|---|---|---|---|---|
| overall | 0.5477 | 0.5412 | −0.66 | 0.3474 | 0.3364 | **−1.10** |
| day | 0.5407 | 0.5388 | −0.20 | 0.3494 | 0.3470 | −0.24 |
| **night** | 0.5625 | 0.5399 | **−2.27** | 0.3460 | 0.3128 | **−3.32** |
| night_clear | 0.5655 | 0.5445 | −2.10 | 0.3380 | 0.3090 | −2.90 |
| night_adverse | 0.5423 | 0.5066 | −3.57 | 0.3981 | 0.3424 | −5.56 |
| day_adverse | 0.5262 | 0.5213 | −0.48 | 0.3517 | 0.3506 | −0.11 |
| snowy | 0.4813 | 0.4496 | −3.17 | 0.3219 | 0.2963 | −2.56 |
| rainy | 0.5503 | 0.5574 | +0.71 | 0.3871 | 0.3762 | −1.09 |
| dawndusk | 0.5739 | 0.5745 | +0.06 | 0.3728 | 0.3606 | −1.21 |

Detections emitted: BASE 622,150 · T1 **637,128** (+2.4%).

**car day→night mAP75: −9.71 → −9.60.** Unmoved. Five training-data recipes and one
backbone initialisation have now failed to shift this number (§2.3).

**Three patterns, stated before any interpretation:**

1. Daytime is essentially untouched (−0.20); the damage is concentrated at night and
   in snow.
2. mAP75 falls harder than mAP50 in every damaged slice — the arm got *less* precise
   at localization, which is the axis JEPA was introduced to fix.
3. T1 emits **more** boxes, not fewer.

**What this is NOT yet.** −0.66 overall sits inside the ±1-point spread the
augmentation ablation showed on `all` rows (§4.2), and both arms are n=1. Until a
noise floor exists, write this as **"indistinguishable from BASE, trending down"** —
not "worse". §5 lists two ways to get that floor without training anything.

**Candidate mechanism — inference, not measurement.** Two properties of the T1 setup
would each predict damage concentrated in mAP75 and in the hardest slices:

- T1's layers 0–6 never saw COCO (`student_init: random`, §3.10), so this arm is
  *JEPA instead of COCO*, and those layers feed a neck, head and C2PSA block that are
  still COCO-initialised and were trained expecting COCO-shaped P4 features.
- The distillation target is a 14×14 grid over a 224px image — about one cell per
  16×16 pixel block. It supervises *what is present*, carrying little information
  about *where an edge precisely lies*.

Both are consistent with the table; neither is tested. §5 says what would test them.

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

### 3.10 Why the distillation student started from random weights
`configs/jepa/distill_t1.yaml` sets `student_init: random`, so T1's layers 0–6 carry no
COCO information at all. This was deliberate, and it makes T1 the clean form of the
scientific question: **can self-supervised pretraining replace ImageNet/COCO
initialisation?** The alternative, `student_init: coco`, asks a different and more
practical question: **can it improve on COCO?** Both are legitimate. They are not the
same arm and must never be reported under one name.

**What §2.5 costs this choice.** A random-init backbone is grafted onto a
COCO-initialised neck, head and C2PSA (§3.4). That mismatch is one plausible reading of
§2.5's damage pattern — but it is a hypothesis, and §2.5's numbers are the honest
answer to the question T1 was actually built to ask. If a COCO-init student is run
later it is a **new arm (T1b)**, not a rerun of T1.

---

### 3.11 Why the recipe is `set3_combined` and not Set 2
Set 2 is the strongest arm of the §2.1 search, and `set3_combined` is **Set 2 plus
exactly three changes**: `hsv_v` 0.4→0.25, `close_mosaic` 10→20, `box` 7.5→9.0.
Everything else — `cls` 0.7, `scale` 0.7, `translate` 0.15, `copy_paste` 0.2, `dfl` 1.5,
`mixup` 0.0 — is identical. Head-to-head on the test slices (2026-09-10; Set 2 via
`eval_detections.py`, `set3_combined` via pycocotools, protocol-identical per §4.6):

| slice | Set 2 | set3_combined | Δ |
|---|---|---|---|
| overall mAP50 | **0.5521** | 0.5478 | −0.44 |
| day mAP50 | **0.5497** | 0.5407 | −0.90 |
| night mAP50 | 0.5555 | **0.5626** | **+0.71** |
| night mAP75 | 0.3399 | **0.3460** | **+0.61** |
| dawndusk mAP50 | 0.5703 | **0.5739** | +0.37 |
| rainy mAP50 | **0.5740** | 0.5503 | −2.37 |

**Why this table does not justify switching — and the reason is not "we already
started".** `all` is an *unweighted* mean over five classes, so a class with 30 instances
votes as loudly as `car` with 91,118. Decomposed, every gap above is carried by classes
under this project's own ≥800-instance threshold:

- **Day's −0.90:** `motor` contributes 0.59 of it (**262 instances**), `bike` 0.13 (494)
  — 80% of Set 2's daytime win from two under-threshold classes.
- **Night's +0.71:** `bus` contributes 0.36 (**297 instances**), `truck` 0.16 (736) — 74%.
- **Rainy's −2.37:** rests on `motor` at **30 instances** and `bike` at 57.

**On `car`, the only class with adequate support anywhere, the two are a tie:** day
−0.12, night −0.16, dawn/dusk −0.49 mAP50, night mAP75 **+0.05**. Both arms are n=1 and
every gap here is inside the ±1-point band of §4.2.

**Four further reasons the choice stands:**

1. Both recipes were selected while looking at `test.txt` (§4.1), so this table cannot
   legitimately arbitrate between them. The honest tie-break is `splits/val.txt`, and
   only `set3_combined` has a val score (all mAP50 0.619, `set3_combined_val.txt`).
   There is no Set 2 val run to compare against.
2. Three knobs moved at once, so this is not an ablation — no per-knob claim is available
   in either direction.
3. It is the **agreed cross-team recipe**, byte-identical across Methods 1, 2 and 3.
   Switching breaks comparability with the teammate's DANN arm.
4. BASE, M1, M2, M3 and T1 all ran `set3_combined`. Switching invalidates all five and
   costs ~4 nights of retraining to return to the current position.

**The defence that does not lean on test numbers — use this one in the write-up.**
`hsv_v: 0.25` applies *less* brightness jitter than Set 2's 0.4. §2.2 found that
aggressive brightness manipulation of already-dark BDD images pushes them
off-distribution and *hurt* night (M1 ranked last on every time-of-day slice). Lower
`hsv_v` is that same mechanism read forwards, and this is a night-focused study. It is
the only argument here that survives a reviewer asking whether the recipe was picked off
the test set.

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
6. **Two evaluators existed historically — partially RESOLVED 2026-09-10.** M3 was
   scored with `eval_detections.py`; M1/M2/BASE with pycocotools via `eval_arm_coco.py`;
   `evaluation/score_slices.py` is a third entry point. The common arm has now been
   scored through **both pycocotools paths and they agree exactly**: `score_slices.py`
   returns BASE day 0.54074 / night 0.56255 / dawndusk 0.5739 against `eval_arm_coco.py`'s
   0.5407 / 0.5625 / 0.5739. Deltas between any two pycocotools-scored arms are therefore
   safe, and that includes every JEPA arm. **Still open, but narrow:**
   `eval_detections.py` — which scored M3 and the whole Set 1/2/3 search — has not been
   reconciled *empirically*, but by inspection it is protocol-identical: its predictions
   come from `run_inference.py` at the same conf 0.001 / iou 0.6 / max_det 300, its GT is
   built the same way at 1280×720 with `category_id == class index`, and it leaves
   `COCOeval` maxDets at default. Expect agreement. Confirm it by scoring one common arm
   through both before publishing a cross-pipeline delta.
7. **Rare-class thresholds.** Any per-class number resting on <800 instances is indicative
   only. The most-quoted figure from the ablation — M3 motor at dawn/dusk, +5.57 — rests on
   **22 boxes.**
8. **`COCOeval` maxDets is left at its default 100 while NMS runs at max_det 300.**
   Deliberate — it matches `eval_arm_coco.py` exactly, which is what makes §4.6's
   agreement meaningful — but any test image holding more than 100 ground-truth
   vehicles has its recall clipped, and dense night scenes are where that bites. The
   clip is identical for every arm, so between-arm deltas stay fair. Count how many
   test images exceed 100 GT boxes and disclose the figure.

---

## 5. Current state and next steps

**Done:** splits frozen · 19 test slices built and cross-verified · hyperparameter search ·
four-arm augmentation ablation (closed, negative result) · JEPA code ported to YOLOv11 and
verified · T1 distillation converged · `weights/init_t1.pt` grafted ·
**`evaluation/score_slices.py` built and cross-validated against `eval_arm_coco.py` (§4.6)** ·
**T1 fine-tuned and scored at 100% labels (§2.5)**.

**Deliverable being built — one table:**

| arm (backbone init) | 100% labels | 10% labels |
|---|---|---|
| BASE — COCO `yolo11s.pt` | ✅ done | ⬜ |
| T1 — I-JEPA ViT-H/14 distilled into a **random-init** student | ✅ done — §2.5 | ⬜ |
| T1b — same teacher, **COCO-init** student (§3.10) | ⬜ not started | ⬜ |
| T2 — I-JEPA ViT-B/16 pretrained on BDD, distilled | ⬜ not started | ⬜ |

Scored on: overall · day · night · dawndusk · clear · rainy · snowy · adverse ·
night_adverse · day_adverse · night_clear. Plus **car day→night mAP75 for every arm**.

### Immediate order

**Free — no GPU training required:**

1. `sha256sum weights/external/base_100_friend.pt`, and diff its `args.yaml` against the
   T1 run's. Record both in §2.5.
2. **Get a noise floor out of work already done.** Score `last.pt` alongside `best.pt`
   for both arms in one `score_slices.py` call — the within-arm best/last gap bounds run
   variance. Then read the last 20 rows of each `results.csv`: post-`close_mosaic`
   epoch-to-epoch val fluctuation is a second free estimate. **§2.5's verdict depends on
   this**, and it costs inference time only.
3. Copy `results.csv` and `args.yaml` for `t1_jepa_100` into the tracked `results/`
   directory — `runs/` is gitignored and `exist_ok: true` will overwrite it (§3.9).
4. Count test images with more than 100 GT boxes (§4.8).

**Cheap — about one night of shared A40 in total:**

5. **Pin the optimizer explicitly (§3.5) before anything else in this block.**
   `train_10.txt` works out to 9,500 iterations, below ultralytics' 10,000 threshold, so
   `optimizer: auto` silently switches to AdamW and the label-fraction column would be
   comparing optimizers instead of label fractions.
6. Create `configs/data/bdd100k_vehicle5_10.yaml` (`train: splits/train_10.txt`). **It
   does not exist yet and it blocks every 10% run.**
7. Run `base_10` and `t1_jepa_10` (~3–4 h each) and score both.

**The fork — decide with the 10% numbers in hand.** T2 (the stated contribution;
~15 h pretrain + ~1 h distill + ~2 nights fine-tune) or T1b (§3.10; ~3.7 h distill +
~2 nights). There is not budget for both plus the 10% grid.

**Note on expectations, restated after §2.5.** BASE at 100% labels was already hard to
beat, and T1 did not beat it. The 10% column is where self-supervised pretraining
reliably shows its gain, which makes it the **core** of the deliverable now rather than
an extension.

**Still open, and still the strongest result in the study:** the −9.7 car day→night
mAP75 collapse. Five training-data recipes and one backbone initialisation have failed
to move it. `imgsz: 768` is the untried resolution lever (§2.3), and it is now the most
likely source of a positive finding if budget allows one more BASE run.
