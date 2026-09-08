# JEPA implementation plan (v3, 2026-09-08)

Supersedes the old project's `PROJECT_PLAN_new.md`. Written against the **current**
`Vehicle Detection` repo (YOLOv11s / BDD100K / set3 recipe / pycocotools), after
reading both that repo and the archived `(Claude)Vehicle-Detection-and-Distance-Estimation`.

---

## 0. Where the project actually stands

**Locked and reusable:**

| thing | state |
|---|---|
| detector | YOLOv11s (`yolo11s.pt`), ultralytics pip |
| dataset | BDD100K, 5 classes (car/truck/bus/motor/bike) |
| splits | frozen, seed 42 — train 60,186 / val 1,542 / test 8,841 |
| recipe | `set3_combined` (epochs 100, imgsz 640, batch 16, box 9.0, cls 0.7, scale 0.7, translate 0.15, hsv_v 0.25, close_mosaic 20, patience 0) |
| scorer | pycocotools (`eval_arm_coco.py`), NMS fixed conf 0.001 / iou 0.6 / max_det 300 |
| unlabeled pool | `splits/pretrain.txt` = the same 60,186 train images |

**Closed:** the augmentation line. M1 (low-light) / M2 (IRFS) / M3 (both) vs BASE is a
negative result with a mechanism — nothing to redo. Per your note, augmentation is out
of the plan from here.

**The finding that motivates JEPA — and it's already measured.** Car day→night mAP75:

| BASE | M1 | M2 | M3 |
|---|---|---|---|
| −9.71 | −9.71 | −9.64 | −9.82 |

Four different training-data recipes, one number that will not move. Night detection is
nearly intact (mAP50 −3.0 to −3.5); night **localization** collapses, identically,
regardless of what you feed the model. That is not a data-distribution problem — you
already tried fixing it with data, four ways. It reads as a *representation* problem,
which is exactly the claim JEPA makes. **This is your motivation paragraph, and unlike
most senior projects it is backed by a measurement you already own.**

**Already-written JEPA code in the old folder (it ran, on the A40, and it worked):**

| file | reusable? |
|---|---|
| `third_party/ijepa/` (vendored official Meta code) | ✅ as-is |
| `jepa_pretrain/train_ijepa.py` (Stage 1, 236 lines) | ✅ as-is — no YOLO dependency at all |
| `jepa_distill/data.py`, `teacher.py`, `train_distill.py` | ✅ as-is |
| `jepa_distill/student.py` | ⚠️ hard-coded to YOLOv5 — needs a ~15-line rewrite |
| `tools/make_init_weights.py` | ⚠️ YOLOv5 checkpoint format — needs a rewrite |
| `configs/exp/pretrain_t2.yaml`, `distill_t2.yaml` | ⚠️ repoint paths, update oversampling |

Evidence it works: the T1 distillation completed twice on the A40 — 30 epochs, 3.0 h,
189 img/s, 3.7 GB VRAM, cosine loss 1.007 → 0.186, day/night linear probe 0.98.

---

## 1. The design decision, restated

JEPA hands you a **ViT encoder**. YOLOv11 wants a **CSP-conv backbone**. Three ways to
connect them:

| option | verdict |
|---|---|
| (a) Replace YOLO's backbone with the ViT | breaks the "arms differ only in init" comparison, needs a new neck, slow at 640px. No. |
| (b) **Distill** frozen ViT features into the YOLO backbone, then fine-tune normally | ✅ chosen. Arms differ *only* in backbone init. Code already exists. |
| (c) ViT as auxiliary supervision during detection fine-tuning | more novel, much harder to defend, no code. Future work. |

**I-JEPA, not V-JEPA.** V-JEPA needs video; BDD100K's detection set is 70k stills from
70k *different* videos, and the raw video release is ~1.8 TB. `DriveJEPA.pdf` in the
project uses V-JEPA for trajectory planning, not detection — different task, different
inputs. Cite it as related work, don't copy it.

**Tap point — verified today, not assumed.** yolo11s at 224px input:

```
layer 5  Conv   →  (256, 14, 14)     stride 16
layer 6  C3k2   →  (256, 14, 14)     ← P4, the tap
layer 7  Conv   →  (512,  7,  7)
```

256 channels at 14×14 — **byte-for-byte the same shape as the YOLOv5s tap the old code
used**, and layer index 6 is the same too. The student port is essentially a
constructor swap.

---

## 2. The arms

| arm | backbone init | Stage-1 cost | why it's in the table |
|---|---|---|---|
| **BASE** | COCO `yolo11s.pt` | — | control; **you already have this run and scored** |
| **T1** | Meta's I-JEPA ViT-H/14 (IN1K) target encoder, distilled | none | cheap JEPA arm — no pretraining at all |
| **T2** | I-JEPA ViT-B/16 pretrained by you on the 60,186 BDD images with adverse-condition oversampling, distilled | 10–15 h | **this is your contribution** |

Everything else — recipe, splits, scorer, NMS — byte-identical across all three.

### Two honesty notes you must put in the write-up

1. **T2 is data-starved and there is no way around it.** Meta released I-JEPA weights only
   for ViT-H/14, ViT-H/16-448, and ViT-g/16 — **no ViT-B**. So T2 trains from scratch on
   60k images where the paper used 1.28M: ~20× less data. Two ways to handle it:
   - *(default)* Frame it as the question: "does domain-matched SSL on a small in-domain
     corpus beat ImageNet SSL at 20× less data?" A no is still a publishable answer.
   - *(optional upgrade)* Warm-start the ViT-B/16 from **MAE** (`facebook/vit-mae-base` —
     same arch, same patch size, drops straight in) and continue with the I-JEPA
     objective. Cheaper and probably stronger, at the cost of one documented deviation.
2. **T1 vs T2 confounds model scale with domain.** ViT-H/14 (630M) vs ViT-B/16 (86M).
   Already flagged in the old repo's `RESEARCH_VULNERABILITIES.md` §1; it cannot be fixed
   inside your budget. Disclose it, don't paper over it.

### The recommendation that matters most

Run the **10%-label arms**. `splits/train_10.txt` (6,019 images) already exists, and each
run is ~1/10 the cost of a full arm.

Why: BASE@100% is already strong, and the augmentation ablation showed how hard it is to
beat. Self-supervised pretraining reliably shows its gains **where labels are scarce** —
that is the standard result in every SSL paper, I-JEPA included. If JEPA wins anywhere in
your project, the 10% column is where it wins. A 3×2 table (BASE/T1/T2 × 100%/10%) is a
much better deliverable than three numbers at 100%, and the extra three runs cost roughly
one night total.

---

## 3. Step by step

### Phase 0 — port the code (≈1 day, CPU only, no GPU, no data)

1. `mkdir jepa/` in the current repo; copy over `third_party/ijepa/`,
   `jepa_pretrain/train_ijepa.py`, `jepa_distill/{data,teacher,train_distill}.py`.
2. **Rewrite `jepa_distill/student.py`** for YOLOv11. Replaces the YOLOv5
   `DetectionModel` construction with:
   ```python
   from ultralytics import YOLO
   full = YOLO("yolo11s.yaml").model      # or yolo11s.pt to start from COCO
   self.layers = full.model[:7]           # layers 0..6, through the P4 C3k2
   ```
   Everything else in that file (`proj` 1×1 conv 256→D_t, `cosine_distill_loss`,
   `backbone_state_dict`) works unchanged. Keep `P4_LAYER = 6`.
3. **Rewrite `tools/make_init_weights.py`** for the ultralytics checkpoint format
   (load `yolo11s.pt`, replace only `model.0.*`–`model.6.*`, keep COCO weights for
   layers 7–10 incl. SPPF + **C2PSA**, neck and head, save back). Keep the existing
   hard refusal on any key-shape mismatch — that guard is what stops a silently
   broken init.
4. Repoint the configs: `train_list: splits/pretrain.txt`,
   `images_dir: dataset/yolo/images/train`, `attr_index: dataset/yolo/attr_index.json`.
5. **Widen the Stage-1 oversampling to the new scope.** Old config oversampled night /
   dawn-dusk / rainy. Scope is now adverse *environment*, so:
   ```yaml
   oversample:
     night: 2.0
     dawn_dusk: 2.0
     rainy: 2.0
     snowy: 2.0      # new
     foggy: 2.0      # new (only 129 train images — effectively free)
   ```
   `build_sample_weights` needs one added branch for the extra weather keys.
6. Smoke-test both stages on CPU (`--smoke` exists in both scripts, ~2 min each).
   Expected printout: teacher dim 1280 / grid 16, student P4 256ch 14×14, loss near 1.0
   and moving.

**Gate:** don't touch the GPU until both smoke tests pass.

### Phase 1 — T1 arm (the cheap one; do this first)

7. Fetch Meta's original checkpoint — **the target encoder, not the HuggingFace
   conversion**, which exports the *context* encoder (this was FLAG 5 in the old repo
   and it cost a wasted run):
   ```
   wget https://dl.fbaipublicfiles.com/ijepa/IN1K-vit.h.14-300e.pth.tar -P weights/
   ```
8. Distill → `jepa/runs/stage2/distill_t1_yolo11/backbone_distilled.pt`.
   Budget: ~3 h, ~4 GB VRAM (measured on the old run). Watch cosine loss falling from
   ~1.0 toward 0.2, and the day/night probe every 5 epochs.
9. Assemble `weights/init_t1.pt` = distilled layers 0–6 + COCO everything else.
10. Fine-tune with the **exact `set3_combined` recipe**, only `model:` changes to
    `weights/init_t1.pt`. ~2 nights of shared A40.

**Gate:** if T1@100% lands within noise of BASE, that is *expected*, not failure — go
straight to the 10% arms before spending 15 h on T2's pretraining.

### Phase 2 — T2 arm (the contribution)

11. Stage-1 I-JEPA pretrain, ViT-B/16 @224, 300 epochs on the 60,186-image pool with the
    widened oversampling. Budget 10–15 h, ~20 GB VRAM at micro_batch 128 (drop to 64 if
    the card is contended). Auto-resumes; announce it on the GPU group line first.
    **Watch `tstd`** — the script prints a COLLAPSE ALARM if target-embedding std → 0.
    If the first hour projects past 2 days, stop and re-scope.
12. Distill (~30–60 min, ViT-B is ~5× lighter than ViT-H) → assemble `weights/init_t2.pt`.
13. Fine-tune with the same set3 recipe. ~2 nights.

### Phase 3 — the label-fraction grid

14. Three more runs on `splits/train_10.txt`: BASE@10%, T1@10%, T2@10%. ~3–4 h each,
    Kaggle-suitable. Same recipe otherwise.

### Phase 4 — scoring (new, per the scope change — **done today, see §4**)

15. Score every arm's `best.pt` with `eval_arm_coco.py` on **all** the new test slices.
    Never re-tune on test; `splits/val.txt` stays the development set.
16. Headline table: arm × {overall, day, night, dawn/dusk, clear, rainy, snowy, adverse,
    night_adverse} at 100% and 10% labels.
17. Report the day→night mAP75 gap for every arm — the −9.7 number is the one thing four
    recipes couldn't move, so "did JEPA move it?" is the sharpest question you can ask.

---

## 4. Test slices — built and verified (2026-09-08)

Scope moved from "low light" to "adverse environment", so the held-out test set is now
sliceable by weather as well as time of day.

**Verification first:** your friend's table was checked cell-by-cell against
`dataset/yolo/attr_index.json` + `splits/{train_100,val,test}.txt`. **All 54 cells match
exactly** — train 60,186 / val 1,542 / test 8,841, and every time-of-day × weather cell.
The `test_day` / `test_night` / `test_dawndusk` folders you already have are consistent
with it.

New generator: **`tools/make_test_slices.py`** — deterministic, no seed, pure function of
`test.txt` + `attr_index.json`. It writes, per slice: a portable bare-filename list
(`splits/test_slices/<slug>.txt`), the path list ultralytics reads
(`dataset/yolo/splits/test_<slug>.txt`), and a data yaml
(`configs/data/bdd100k_test_<slug>.yaml`). It asserts that the time-of-day slices and the
weather slices each partition all 8,841 images before writing anything.

| slice | images | instances | use |
|---|---|---|---|
| day | 4,264 | 51,930 | existing |
| night | 3,863 | 36,751 | existing |
| dawndusk | 714 | 8,512 | existing |
| clear | 5,345 | 56,667 | weather reference |
| overcast | 1,239 | 15,505 | |
| partlycloudy | 738 | 9,256 | |
| snowy | 769 | 8,080 | **adverse** |
| rainy | 737 | 7,541 | **adverse** |
| foggy | **13** | 144 | ⚠️ **do not report mAP** — quote the count only |
| adverse (rain+snow+fog) | 1,519 | 15,765 | **headline** |
| benign (clear+overcast+partly) | 7,322 | 81,428 | its complement |
| night_clear | 3,274 | 31,550 | dark, good weather |
| night_rainy | 286 | 2,535 | |
| night_snowy | 273 | 2,374 | |
| night_adverse | 566 | 4,986 | **headline** — dark *and* bad weather |
| day_clear | 1,764 | 21,352 | easiest case |
| day_rainy | 396 | 4,356 | |
| day_snowy | 422 | 4,842 | |
| day_adverse | 823 | 9,252 | **bad weather, good light** |

**Why the crossed slices earn their place.** `night` and `rainy` overlap, so a model that
improves on both tells you nothing about *which* factor it fixed. The pair
`day_adverse` (823 imgs, weather bad / light fine) vs `night_clear` (3,274 imgs, light bad
/ weather fine) separates them cleanly, and `night_adverse` is where both hit at once.
That triple is the actual argument your scope change is asking for.

**Reporting discipline:**
- Foggy is 13 images / 144 boxes. Quote the count, never a mAP. It is folded into
  `adverse` where it contributes ~1%.
- `night_rainy` and `night_snowy` (~2,400–2,500 instances each) are usable at the `all`
  level but **not per-class** — motor/bike there will be single digits. Per-class numbers
  need the ≥800-instance rule from `set3-notes.md`.
- Same scorer, same NMS, every slice, every arm. The old project got burned by mixing two
  evaluators; don't repeat it.

---

## 5. Budget and risk

| item | GPU time |
|---|---|
| Phase 0 port + smoke tests | 0 (CPU) |
| T1 distill | ~3 h |
| T1 fine-tune @100% | ~2 nights |
| T2 I-JEPA pretrain | 10–15 h |
| T2 distill | ~1 h |
| T2 fine-tune @100% | ~2 nights |
| three @10% arms | ~1 night total |
| all scoring passes | ~2–3 h |

≈ 6 nights of shared A40 plus a day of pretraining, on a contended card. Announce
anything over ~2 nights on the group line, as before.

**Abort criteria, decided now so they're not rationalized later:**
- Stage-1 `tstd` collapses → stop, audit the stop-grad/EMA wiring, don't burn 15 h.
- Stage-1 projected wall-clock > 2 days after the first hour → cut to 150 epochs or
  ViT-S/16 rather than silently overrunning.
- Distillation cosine loss flat near 1.0 after 5 epochs → wiring bug, not a training
  problem. Check the shared-view assumption in `data.py` first.
- T1@100% ≈ BASE within noise → **not a failure**; proceed to the 10% grid, which is
  where the signal should be.

**Known limitation carried forward:** the set3 recipe was chosen from `test.txt`
observations (`set3-notes.md` §5), so every `test.txt` number — including the JEPA arms —
carries the same optimistic bias. It is identical across arms, so *between-arm* deltas
are still fair. Say this in the write-up.

**Still unquantified:** seed variance. Every arm is n=1, and the augmentation ablation's
spread on `all` rows was ±1 point. One repeated BASE run at a different seed would
establish the noise floor and tell you whether a 1-point JEPA gain means anything. It is
one night, and it is the cheapest credibility you can buy.
