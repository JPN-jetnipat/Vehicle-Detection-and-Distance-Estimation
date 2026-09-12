# Master record — nighttime / adverse-environment vehicle detection

**Owner:** Kanade · **Last updated:** 2026-09-12
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

### ⚠️ Naming hazard — FOUR different things are called "set 3"

Get this wrong in the write-up and the whole methods section becomes unreadable.
This is not hypothetical: it caused a live mix-up on 2026-09-10, when the teammate's
`set3_combined` config was mistaken for this repo's `set3_combined` run.

| name | what it is | in use? |
|---|---|---|
| **`configs/hyp/set3_combined.yaml`** | the **shared cross-team hyperparameter recipe** — hsv_v 0.25, close_mosaic 20, box 9.0, copy_paste 0.2, cls 0.7, scale 0.7, translate 0.15, patience 0, seed 42, imgsz 640, batch 16, workers 2, cache false | ✅ **this is the recipe** |
| `runs/detect/m3_lowlight_irfs/` **(this repo)** | **Method 3** of the augmentation ablation — the recipe above run on the 137,519-image low-light × IRFS pool. **Renamed 2026-09-10 from `set3_combined`** to end the collision; `runs/` is gitignored so this rename is server-side only, and the dir's own `args.yaml` still records `name: set3_combined` as provenance | ✅ a real arm — call it **M3** |
| `set3_combined` **(teammate's pipeline)** | the **BASE / no-augmentation control** — the same recipe on the plain 60,186 pool. This produced `weights/external/base_100_friend.pt` | ✅ a real arm — call it **BASE** |
| `configs/archive/hyp/set3_merged_night_localization.yaml` | a **different, earlier** hyperparameter-search arm (dfl 1.8, cls 0.5, scale 0.55, hsv_s 0.60) | ❌ not the recipe, line closed |

**The recipe is genuinely shared** — the teammate's config and `set3_combined.yaml` are
byte-identical apart from `save_period`, and T1 used it too. What differs between the two
`set3_combined` *arms* is the **data pool**, which is the entire point of the ablation.

**The discriminator — one command, use it before trusting any run named `set3_combined`:**

```bash
grep -E "^(data|model):" runs/detect/<name>/args.yaml
```

- `data: .../bdd100k_vehicle5_method3.yaml` → **M3** (augmented pool; dir now `m3_lowlight_irfs/`)
- `data: .../bdd100k_vehicle5.yaml` or `bdd100k_vehicle.yaml` → **BASE** (plain pool)

**Rule for the write-up and for every scoring run:** never label an arm `set3_combined`.
Use `BASE`, `M1`, `M2`, `M3`, `T1`, `T2`. The string `set3_combined` refers to the recipe
file and nothing else. Run directories keep their historical names for provenance — do not
rename them — but the `--arm` labels passed to `evaluation/score_slices.py` must be the
unambiguous ones, because those strings become the permanent column headers in
`results/*.csv`.

> **Recipe provenance:** `set3_combined` = **Set 2 + `hsv_v` 0.4→0.25 + `close_mosaic`
> 10→20 + `box` 7.5→9.0**. Re-examined 2026-09-10 against Set 2 and confirmed as the
> recipe — see §3.11 for the per-class decomposition and the defence.
> **Do not switch recipes mid-study.**

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

### 2.2.1 ⚠️ REVISION (2026-09-11) — the augmentation line was judged on the wrong slices

When this ablation was written up, only the **time-of-day** slices existed. M3 was
re-scored on 2026-09-11 across all 19 slices (arm `m3_lowlight_irfs`, same scorer, same
NMS, same job as the T1 noise floor). Against BASE, mAP50, `all`:

| slice | Δ vs BASE | floor | ratio | |
|---|---|---|---|---|
| **day_adverse** | **+3.39** | 0.52 | 6.5× | bad weather, good light |
| **snowy** | **+2.60** | 0.65 | 4.0× | |
| **adverse** | **+1.21** | 0.23 | 5.3× | rain+snow+fog |
| rainy | +0.89 | 0.10–0.39 | ~2× | |
| night_clear | +0.45 | 0.46 | 1.0× | inside noise |
| overall | −0.17 | 0.31 | 0.5× | inside noise |
| day | −0.11 | 0.58 | 0.2× | inside noise |
| night | −0.16 | 0.36 | 0.4× | inside noise |
| clear | −0.62 | 0.31 | 2.0× | weak |
| night_adverse | −2.78 | 2.18 | 1.3× | weak |

**The negative result stands exactly as stated — and it was stated about the wrong
thing.** On time of day, M3 ≈ BASE (every one of day / night / dawndusk is inside the
noise floor). But on *weather*, M3 is clearly ahead: **+3.39 on `day_adverse` at 6.5×
the floor**, +2.60 on snow, +1.21 on adverse overall.

**Reading that is consistent with §2.2's own mechanism.** The mechanism paragraph above
says a gamma curve models *dusk*, not night. Push that one step further: gamma darkening
plus Poisson–Gaussian noise is a **contrast-and-degradation** transform, and rain, snow
and fog are contrast-and-degradation conditions. So the augmentation was doing something
real the whole time — it just was not doing the thing it was named after. It fails at
night (where the "already dark, made darker" problem of §2.2 bites) and succeeds in
daytime bad weather. `night_adverse` −2.78 is where both effects meet, which fits.

**What to claim, and what not to.** Claim: *"low-light augmentation did not improve
night performance, but the same transform improved adverse-weather performance in
daylight — it behaves as a degradation augmentation, not a night augmentation."* Do not
claim a weather-robustness method: this is n=1, `day_adverse` is 823 images, the noise
floor is borrowed from a different arm's run, and the recipe was never designed or tuned
for weather. It is an observation that reframes a closed line, not a new result.

**This does not reopen the augmentation line for the JEPA deliverable** — scope is still
night — but it is the most interesting thing in §2.2 and belongs in the write-up.

#### The prediction that would confirm or kill this — registered 2026-09-11, before the data

M3 = low-light copies **and** IRFS oversampling, so its weather gain could come from
either. M1 and M2 separate them, and they have never been scored on the weather slices.
Stating the outcome in advance, so this is a test and not a story fitted after the fact:

| arm | what it isolates | **prediction if the degradation-augmentation reading is right** |
|---|---|---|
| **M1** (low-light only) | the transform | **should show the `day_adverse` / `snowy` / `adverse` gains** |
| **M2** (IRFS only) | the oversampling | **should NOT show them** |

**If M2 shows the gains too, the mechanism above is wrong** — the effect would be coming
from rare-class oversampling, not from the transform — and §2.2.1's reframing must be
retracted, not reworded. If M1 shows them and M2 does not, the reading is supported and
can be stated as a mechanism rather than a reading.

Cost: one `score_slices.py` job, ~40 min, no training. **Blocker:** neither M1 nor M2 has
a run directory on this server (§5) — both came from the teammate, like BASE. Request
`best.pt` for each, and confirm which data config each used rather than assuming M1 is
low-light-only and M2 is IRFS-only.

**If the weights cannot be obtained:** say in §2.2 that M1/M2 numbers are reported from
the teammate's scoring and were never re-verified on this scorer, and keep §2.2.1's
weather claim scoped to M3 alone. Weaker, but honest, and it costs nothing.

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
distribution and toward either representation quality or input resolution.

**Update 2026-09-11 — now measured across seven arms, one scorer, one job:**

| arm | day | night | Δ |
|---|---|---|---|
| base_100 (BASE) | 0.5076 | 0.4105 | **−9.71** |
| t1_best | 0.5061 | 0.4102 | −9.60 |
| t1_last | 0.5053 | 0.4093 | −9.60 |
| t1_e90 | 0.5051 | 0.4102 | −9.48 |
| t1_e80 | 0.5058 | 0.4101 | −9.56 |
| set2_100 | 0.5088 | 0.4100 | −9.87 |
| m3_lowlight_irfs (M3) | 0.5013 | 0.4031 | −9.82 |

Full range **−9.48 to −9.87: 0.39 points**, against a within-run wobble of 0.12 among
the four T1 checkpoints. Four training pools, two hyperparameter recipes and a
JEPA-initialised backbone all land on the same number. **This is not a property of any
arm — it is a property of the setup**, and it is the finding the write-up should be
built around. The remaining untested lever is input resolution (`imgsz: 768`); every
data-side and representation-side lever tried so far has left it untouched. `imgsz: 768` is
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

### 2.5.1 Verdict — the gap is real, and it is night-specific (2026-09-11)

A noise floor now exists, obtained without training anything. `save_period: 10` left
four checkpoints of the *same* `t1_jepa_100` run — epoch 80, epoch 90, best, last — all
in the mosaic-free tail (`close_mosaic: 20`). Scoring all four in one
`evaluation/score_slices.py` pass gives a within-run spread; `evaluation/noise_floor_report.py`
compares each T1-vs-BASE delta against it. Because the floor is a *lower* bound (§4.2),
a delta is only taken seriously at **>2× floor**, and headlined at **>4×**.

| slice | Δ mAP50 | floor | ratio | verdict |
|---|---|---|---|---|
| **night** | **−2.27** | 0.36 | **6.3×** | holds clearly |
| **snowy** | **−3.17** | 0.65 | **4.9×** | holds clearly |
| **night_clear** | **−2.10** | 0.46 | **4.6×** | holds clearly |
| adverse | −0.84 | 0.23 | 3.7× | holds |
| clear | −0.95 | 0.31 | 3.1× | holds |
| overall | −0.66 | 0.31 | 2.1× | holds, weakly |
| night_adverse | −3.57 | 2.18 | 1.6× | **weak — do not headline** |
| rainy | +0.71 | 0.10–0.39 | 1.8× | weak |
| **day** | **−0.20** | 0.58 | 0.3× | **inside noise — no effect** |
| **day_adverse** | **−0.48** | 0.52 | 0.9× | **inside noise — no effect** |
| dawndusk | +0.06 | 0.27 | 0.2× | inside noise |

mAP75 says the same thing louder: night −3.32 (floor 1.31), night_clear −2.90 (1.12),
day −0.24 (0.56, inside noise).

**So: T1 did not fail uniformly. Daytime is untouched — genuinely, measurably
untouched — and every point of loss is concentrated at night and in snow.** That is a
much sharper statement than "T1 scored 0.66 lower", and it is the one to write up.

**What this still is not.** The floor is a *within-run* spread: four checkpoints, one
seed, one data order. Between-seed variance includes head re-initialisation and
different augmentation draws and is normally larger. So a delta that **fails** this test
is definitely not a finding, while one that passes is *consistent with* being real
rather than proven. Do not call this a seed study. §4.2 still stands; the repeated BASE
run is still the thing that would settle it.

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

### 2.6 The label-fraction grid — JEPA's best case, and it lost worse (2026-09-12)

`base_10` and `t1_jepa_10` on `splits/train_10.txt` (6,019 images), optimizer pinned
(§3.5), all four arms scored in **one** `score_slices.py` job.

**The comparison that matters is `t1_10` vs `base_10`** — not against `base_100`, which
is what the tool's delta column prints.

| slice | T1 − BASE @100% | T1 − BASE @10% | gap widened by | cost of dropping to 10% labels |
|---|---|---|---|---|
| overall | −0.65 | **−3.29** | −2.64 | −11.15 |
| day | −0.19 | **−2.76** | −2.57 | −10.59 |
| **night** | −2.26 | **−4.40** | −2.14 | −12.02 |
| dawndusk | +0.06 | **−5.17** | −5.23 | −11.83 |
| clear | −0.95 | −2.22 | −1.27 | −12.38 |
| rainy | +0.71 | **−5.40** | −6.11 | −8.55 |
| snowy | −3.17 | −2.76 | +0.41 | −9.27 |
| adverse | −0.84 | **−4.94** | −4.10 | −8.65 |
| **night_adverse** | −3.57 | **−9.34** | −5.77 | −9.61 |
| day_adverse | −0.49 | −1.43 | −0.94 | −8.77 |
| night_clear | −2.10 | −3.11 | −1.01 | −12.13 |

Consistent with the training logs' own val numbers (`splits/val.txt`, `all` mAP50):
`base_10` **0.485**, `t1_jepa_10` **0.447**.

**This was JEPA's best case and it is the worst result in the study.** §5 predicted the
10% column as the place self-supervised pretraining reliably pays: with 6,019 labels the
supervision can no longer overwrite whatever the backbone started as, so initialisation
quality should dominate. It does dominate — in the wrong direction. T1's deficit grows
**5× overall** (−0.65 → −3.29) and **2× at night** (−2.26 → −4.40), and `day`, which was
inside the noise floor at 100% labels, becomes a clear −2.76 loss.

**What this licenses saying — and what it does not.**

✅ *"A YOLOv11s backbone initialised by distilling Meta's I-JEPA ViT-H/14 into a
randomly-initialised student is a worse starting point for BDD100K vehicle detection than
COCO pretraining, and the disadvantage grows as labelled data becomes scarce."*

❌ *"JEPA does not work for vehicle detection."* T1 replaces COCO rather than adding to it
(`student_init: random`, §3.10). The widening-with-scarcity pattern is the textbook
signature of **a worse initialisation**, not of a useless objective: at 60k labels
fine-tuning can repair what the backbone lacks, at 6k it cannot. Nothing here separates
"the I-JEPA features are bad" from "not having COCO is bad" — **T1b is exactly that
experiment**, and this result promotes it from optional to the single most informative
run left (§5).

**Caveat carried forward.** The noise floor in §2.5.1 was measured on the `t1_jepa_100`
run. A 10%-label run trains on a tenth of the data and should be *more* variable, so that
floor is not transferable. `t1_jepa_10` has its own `epoch80/90/best/last` checkpoints;
measuring a 10%-specific floor costs ~30 min of scoring and no training, and should be
done before §2.6 is quoted in the write-up. That said, −3.29 and −9.34 are far outside
any plausible floor — the conclusion is not in doubt, only its stated precision.

### 2.6.1 A cleaner invariant than −9.7: the night/day mAP75 **ratio**

§2.3 reported the car day→night mAP75 gap as constant near −9.7 across seven arms. The
10% arms show that framing was slightly wrong — the *absolute* gap shrinks when overall
performance falls:

| arm | day | night | absolute Δ | **night/day ratio** |
|---|---|---|---|---|
| base_100 | 0.5076 | 0.4105 | −9.71 | **0.809** |
| t1_100 | 0.5061 | 0.4102 | −9.59 | **0.811** |
| base_10 | 0.4571 | 0.3742 | −8.29 | **0.819** |
| t1_10 | 0.4499 | 0.3687 | −8.12 | **0.820** |

The absolute delta moves 1.6 points between label fractions; **the ratio moves 0.011.**
Night mAP75 is ~81% of day mAP75 regardless of arm *or* label fraction — across four
training pools, two hyperparameter recipes, two backbone initialisations and a 10×
change in labelled data.

**Report the ratio, not the absolute delta.** The absolute gap compresses toward zero as
performance drops, so it is partly an artefact of scale; the ratio is not. This is the
most robust quantity the project has produced and it is the right anchor for the
write-up. It also sharpens the claim: night degradation is **multiplicative**, which is
what one expects from a sensing/resolution limit rather than from a data-distribution or
initialisation problem — and it is consistent with every intervention tried so far
failing to move it.

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
**Verified at source level 2026-09-11**, not inferred. `BaseTrainer.build_optimizer`
in the installed ultralytics 8.4.120:

```python
# engine/trainer.py:1120-1122
lr_fit = round(0.002 * 5 / (4 + nc), 6)
name, lr, momentum = ("MuSGD", 0.01, 0.9) if iterations > 10000 else ("AdamW", lr_fit, 0.9)
self.args.warmup_bias_lr = 0.0
```

So `optimizer: auto` selects **MuSGD, lr0 0.01, momentum 0.9, warmup_bias_lr 0.0** when
iterations > 10,000 — not the SGD / 0.937 / 0.1 printed as the yaml defaults. Every arm so
far ran MuSGD without that ever being an explicit choice. Confirmed against
`t1_jepa_100.log`: `optimizer: MuSGD(lr=0.01, momentum=0.9)`.

**The knock-on that matters: the threshold is on iterations, not epochs**, and iterations
use `nbs=64`, not `batch`:

    iterations = ceil(N_images / max(batch, nbs=64)) x epochs

| arm | images | iterations | `auto` picks | lr0 |
|---|---|---|---|---|
| 100% labels | 60,186 | ceil(60186/64)×100 = **94,100** | MuSGD | **0.01** |
| 10% labels | 6,019 | ceil(6019/64)×100 = **9,500** | **AdamW** | **0.001111** |

`lr_fit` at nc=5 is `0.002 × 5 / 9 = 0.001111`. So an unpinned 10% arm would have trained
with **a different optimizer at a 9× lower learning rate** — and `warmup_bias_lr` would
still have been forced to 0.0, so that one value is safe either way. **A label-fraction
ablation left on `auto` would have compared optimizers, not label fractions**, and the
BASE@100 → BASE@10 drop would have been uninterpretable.

**Resolution:** `configs/hyp/set3_combined_10.yaml` pins `optimizer: MuSGD`, `lr0: 0.01`,
`momentum: 0.9`, `warmup_bias_lr: 0.0` — all four now confirmed against the source above.
`configs/hyp/set3_combined.yaml` is deliberately left on `auto` so the completed 100%
arms keep a byte-identical recipe file; at 94,100 iterations `auto` resolves to exactly
the pinned values, so the two columns are comparable.
**Smoke-tested 2026-09-11:** the pinned recipe at 2 epochs (~190 iterations, deep in
AdamW territory) printed `MuSGD(lr=0.01, momentum=0.9)` with parameter groups 81/88/87,
identical to the 100% runs. The pin holds.

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

**Re-measured 2026-09-11** through `evaluation/score_slices.py`, both arms in one job,
all 19 slices — so this no longer depends on two scripts agreeing. mAP50, `all`, with
the T1 run's noise floor for scale:

| slice | Set 2 | set3_combined | Δ (Set 2 −) | floor | |
|---|---|---|---|---|---|
| overall | **0.5521** | 0.5477 | +0.44 | 0.31 | weak |
| day | **0.5497** | 0.5407 | +0.89 | 0.58 | weak |
| **night** | 0.5558 | **0.5625** | −0.67 | 0.36 | holds |
| **night_clear** | 0.5579 | **0.5655** | −0.75 | 0.46 | holds |
| **snowy** | 0.4694 | **0.4813** | −1.18 | 0.65 | holds |
| dawndusk | 0.5702 | **0.5739** | −0.37 | 0.27 | weak |
| clear | 0.5507 | **0.5539** | −0.33 | 0.31 | inside noise |
| rainy | **0.5740** | 0.5503 | +2.37 | 0.10–0.39 | holds |
| adverse | **0.5215** | 0.5175 | +0.40 | 0.23 | weak |
| night_adverse | **0.5458** | 0.5423 | +0.36 | 2.18 | inside noise |
| day_adverse | **0.5289** | 0.5262 | +0.27 | 0.52 | inside noise |

**The split is clean and it runs along exactly the axis this project is scoped on.**
Set 2 is ahead in **daylight** (+0.89) and **rain** (+2.37). `set3_combined` is ahead at
**night** (−0.67), **night_clear** (−0.75) and in **snow** (−1.18), each clearing the
floor. Overall (+0.44 to Set 2) is weak and is the average of those opposing effects, so
it is the least informative number in the table — which is why picking a recipe on
`overall` would have been the wrong call.

For a night-and-adverse-conditions study, `set3_combined` is the right recipe on the
measurement, not merely on the inertia. (Earlier per-class decomposition, from the
2026-09-10 two-script comparison, is retained below.)

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
2. **n = 1 per arm — PARTIALLY ADDRESSED 2026-09-11, still open.** A *within-run* floor
   now exists, from four checkpoints of the same `t1_jepa_100` run (§2.5.1): roughly
   **0.3 points on the large slices**, rising to 2.2 on `night_adverse` (566 images).
   That is enough to rule deltas out, and it retired several apparent effects (day,
   day_adverse, dawndusk all fell inside it). **It is a lower bound, not the noise
   floor.** Between-seed variance also includes head re-initialisation, augmentation
   draws and data order, and is normally larger. A repeated BASE run at a different seed
   is still the only thing that converts "consistent with real" into "real" — one night
   of GPU time, still the cheapest credibility available. Until then, no JEPA claim
   should rest on a delta below ~2× the within-run floor.
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

**Done:** splits frozen · 19 slices built and cross-verified · hyperparameter search ·
augmentation ablation (negative on time-of-day, **positive on weather — §2.2.1**) · JEPA
ported to YOLOv11 · T1 distilled, grafted, fine-tuned and scored at **both** label
fractions · noise floor established without extra training (§2.5.1) · recipe choice
re-measured on one scorer (§3.11) · optimizer pin source-verified (§3.5) · **the
night/day mAP75 ratio identified as the project's most robust invariant (§2.6.1)** ·
configs for closed lines archived.

| arm (backbone init) | 100% labels | 10% labels |
|---|---|---|
| BASE — COCO `yolo11s.pt` | ✅ 0.5477 | ✅ 0.4362 |
| T1 — I-JEPA ViT-H/14 → **random-init** student | ✅ 0.5412 (−0.65) | ✅ 0.4033 (**−3.29**) |
| **T1b — same teacher, COCO-init student (§3.10)** | ⬜ | ⬜ **← do this next** |
| T2 — I-JEPA ViT-B/16 pretrained on BDD | ⬜ | ⬜ |

### The fork is decided: T1b before T2

The 10% grid was the condition most favourable to self-supervised pretraining and T1 lost
there by 5× its 100% margin (§2.6). The pattern — deficit widening as labels grow scarce
— is the signature of **a worse initialisation**, and T1's backbone is the only one in
the study that never saw COCO. Three reasons T1b now outranks T2:

1. **It tests the actual hypothesis.** T1b (`student_init: coco`) separates "the I-JEPA
   features are unhelpful" from "not having COCO is harmful". Nothing else does.
2. **T2 inherits the same design.** `configs/jepa/distill_t2.yaml` also uses a
   random-init student. If random-init is what sank T1, T2 fails identically and 10–15 h
   of Stage-1 pretraining is spent proving it twice.
3. **It is cheaper**, and a positive T1b is the only path left to a *positive* JEPA
   result: "JEPA refines COCO features" is a defensible contribution; "JEPA replaces COCO
   badly" is already established.

**Run T1b at 10% labels FIRST.** §2.6 shows the initialisation effect is ~5× larger at
10% than at 100%, so the 10% column is both the most sensitive test and the cheapest:
~3.7 h distillation + ~4 h fine-tune = one day, versus two nights at 100%. Only if
T1b@10% beats T1@10% materially is the 100% run worth booking.

### Immediate order

1. **Measure a 10%-specific noise floor** (~30 min, no training). §2.5.1's floor came
   from the 100% run and does not transfer. `t1_jepa_10` has its own
   `epoch80/90/best/last`; strip and score them exactly as before. Do this before
   quoting §2.6.
2. **T1b:** set `student_init: coco` in a **new** config (`configs/jepa/distill_t1b.yaml`
   — do not edit `distill_t1.yaml`, T1's provenance depends on it), distil, graft to
   `weights/init_t1b.pt`, `--verify`, then fine-tune at 10% labels as `t1b_jepa_10`.
3. **Seed repeat of BASE@100** — still the highest-value single night. Every verdict in
   §2.5.1 and §3.11 rests on a within-run lower bound.
4. **M1 / M2 weights from the teammate** — the §2.2.1 prediction is registered and
   waiting. ~40 min of scoring once they arrive.
5. Copy every run's `results.csv` / `args.yaml` into tracked `results/runs/<name>/`
   (§3.9), and count test images with >100 GT boxes (§4.8).

### ⚠️ Server housekeeping — checked 2026-09-12

- **Disk at 98% (24 GB free).** T2's Stage-1 pretraining and its checkpoints will not
  fit comfortably. Clear space before any T2 decision; `runs/detect/*/weights/epoch*.pt`
  are ~57 MB each and several arms hold ten apiece.
- **The A40 is shared and contended** — a co-tenant VLLM process was holding 38 GB and
  100% utilisation during these runs. `base_10` took 7.7 h and `t1_jepa_10` 4.1 h for
  identical workloads; **that difference is contention, not a property of either arm.**
  Never quote wall-clock as a result.
- Host RAM is under pressure (17 GB of swap in use). Keep `workers: 2` / `cache: false`.

### What the deliverable actually is now

An honest, well-instrumented negative result with one robust positive finding:

1. Night localization degrades **multiplicatively** — night mAP75 ≈ 0.81 × day mAP75 —
   and that ratio is invariant across four training pools, two recipes, two backbone
   initialisations and a 10× change in labelled data (§2.6.1).
2. Data-side interventions do not move it (§2.2), though the low-light transform turns
   out to help *daytime adverse weather* (§2.2.1).
3. A representation-side intervention does not move it either, and replacing COCO with
   distilled I-JEPA actively hurts, increasingly so as labels grow scarce (§2.5, §2.6).
4. Untested levers, in order of promise: **T1b** (JEPA *plus* COCO), then **`imgsz: 768`**
   — a resolution lever, which is what §2.6.1's multiplicative signature points at.
