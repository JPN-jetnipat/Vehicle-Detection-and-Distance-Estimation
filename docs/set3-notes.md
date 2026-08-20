# Set 3 — merged night-localization arm

Decision record for `configs/hyp/set3_merged_night_localization.yaml`.
Written 2026-08-20 from the Set 1 vs Set 2 `splits/test.txt` results.

---

## 1. What the Set 1 / Set 2 comparison actually showed

| Metric | Set 1 | Set 2 | Δ (S1 − S2), pts |
|---|---|---|---|
| overall mAP50 | 0.5521 | 0.5392 | **+1.29** |
| overall mAP75 | 0.3490 | 0.3479 | +0.11 |
| overall mAP50-95 | 0.34443 | 0.34093 | +0.35 |
| bike mAP50 | 0.4605 | 0.4270 | **+3.35** |
| motor mAP50 | 0.3918 | 0.3744 | **+1.74** |
| truck mAP50 | 0.5813 | 0.5732 | +0.81 |
| car mAP50 | 0.7893 | 0.7879 | +0.14 |

Set 1 wins 48 of 75 per-class metric cells.

**Set 2 failed on its own target metric.** It existed to improve rare classes; those are the classes that regressed most.

### `copy_paste: 0.2` never executed

`ultralytics/data/augment.py`, `CopyPaste.__call__`:

```python
if len(labels["instances"].segments) == 0 or self.p == 0:
    return labels
```

`tools/bdd_to_yolo.py` emits bbox-only labels, so `Instances.segments` is empty and the transform returns unchanged. Verified empirically on 8.4.124: image unmodified, instance count unchanged.

Set 2's headline hypothesis was therefore **never tested**. Its real delta from Set 1 was only `cls 0.5→0.7`, `scale 0.5→0.7`, `translate 0.1→0.15`, plus the reversion of Set 1's `hsv_v` / `close_mosaic` / `box`.

Confirm for yourself: `runs/detect/set2_100/train_batch0.jpg` will show no duplicated/mirrored instances.

### The night failure is localization, not detection

Day → night degradation for **car** (96.4% of all night instances):

| Metric | Set 1 | Set 2 |
|---|---|---|
| car mAP50 | −3.43 | −3.34 |
| car **mAP75** | **−9.88** | **−9.78** |
| car mAP50-95 | −6.96 | −6.92 |

The model still finds the car at night; it can't put a tight box on it. This is the most reproducible signal in the results, it is nearly identical across both arms, and **neither arm targeted it**. It is also the finding that matters most downstream — box edge accuracy is what the pinhole distance estimate consumes.

### Reporting caveat: per-slice "all" rows are class-mix confounded

Instance composition of `splits/test.txt`:

| Slice | n | car | truck | bus | motor | bike |
|---|---|---|---|---|---|---|
| daytime | 51930 | 91.88% | 4.88% | 1.79% | 0.50% | 0.95% |
| night | 36751 | **96.43%** | 2.00% | 0.81% | 0.25% | 0.51% |
| rainy | 7541 | 92.08% | 5.01% | 1.75% | 0.40% | 0.76% |
| dawn_dusk | 8512 | 93.56% | 3.99% | 1.49% | 0.26% | 0.69% |

Night "all" mAP50 reads *higher* than daytime (+0.58) purely because of this mix, not because night is easier. **Report per-class day-vs-night, not per-slice "all".** dawn_dusk motor (22 instances) and rainy bike (57) swing ±6 pts between the two arms — treat as noise.

---

## 2. Set 3 design

| Param | default | Set 1 | Set 2 | **Set 3** | Why |
|---|---|---|---|---|---|
| `hsv_v` | 0.4 | 0.25 | 0.4 | **0.25** | inherited winner |
| `hsv_s` | 0.7 | 0.7 | 0.7 | **0.60** | NEW — extends the hsv_v hypothesis to the channel neither arm touched |
| `close_mosaic` | 10 | 20 | 10 | **20** | inherited winner |
| `box` | 7.5 | 9.0 | 7.5 | **9.0** | inherited winner |
| `dfl` | 1.5 | 1.5 | 1.5 | **1.8** | NEW — direct lever on the −9.9 pt night mAP75 collapse; restores box:dfl = 5.0 |
| `cls` | 0.5 | 0.5 | 0.7 | **0.5** | reverted — raising it cost mAP50 on every class |
| `translate` | 0.1 | 0.1 | 0.15 | **0.15** | kept from Set 2 — the lowest-risk of its three changes |
| `scale` | 0.5 | 0.5 | 0.7 | **0.55** | pulled back — 0.7 shrinks objects to 0.3×, below the P3 stride-8 floor |
| `copy_paste` | 0.0 | 0.0 | 0.2 | **0.0** | was inert; removing it stops it reading as a tested variable |

**Loss-balance reasoning.** Set 1 raised `box` alone, shifting the regression branch's internal balance from box:dfl 5.0 → 6.0: more weight on the IoU term, none on the distribution that produces the box edge. Set 3's 9.0/1.8 restores 5.0 while keeping total regression weight ~20% above default. Regression:cls ratio — default 18.0, Set 1 21.0, Set 2 12.9, **Set 3 21.6**.

`cos_lr` was considered and left out: it is a convenience change, not a hypothesis, and Set 3 already moves six variables. Candidate for a later arm.

---

## 3. Correction: `optimizer: auto` is not a no-op

None of `default` / `set1` / `set2` specify an optimizer, so all three ran `auto`. In `BaseTrainer.build_optimizer`:

```python
iterations = ceil(len(dataset) / max(batch, nbs)) * epochs
           = ceil(60186 / max(16, 64)) * 100 = 94,100      # > 10,000
name, lr, momentum = ("MuSGD", 0.01, 0.9)
self.args.warmup_bias_lr = 0.0
```

**Every arm so far trained with MuSGD, momentum 0.9, warmup_bias_lr 0.0** — not SGD, and not the `momentum: 0.937` / `warmup_bias_lr: 0.1` printed as defaults in `cfg/default.yaml`. `auto` silently overrides both, and logs that it is ignoring your `lr0`/`momentum`.

Set 3 pins these four values explicitly to reproduce that exactly:

```yaml
optimizer: MuSGD
lr0: 0.01
momentum: 0.9
warmup_bias_lr: 0.0
```

This changes nothing today and protects the arm from the `auto` heuristic being retuned in a future release — it has already moved once (this branch selected plain SGD before MuSGD was introduced). Checked identical in 8.4.65 and the pinned 8.4.120.

Two consequences worth carrying into the writeup:

- The optimizer is **iteration-count dependent, not epoch dependent**. `train_10.txt` (6,019 imgs) at 100 epochs is `ceil(6019/64)*100 = 9,500` iterations — **below the 10,000 threshold**, so on `auto` it would silently train with **AdamW at a different learning rate**. A label-fraction ablation left on `auto` would not be comparing label fractions; it would be comparing optimizers. Pin the optimizer on those arms too.
- "SGD with momentum 0.937" is what the printed defaults would lead you to write in a methods section. It would be wrong.

---

## 4. GPU-server constraints (Twipol 2026-07-14, Apichon)

| Constraint | Status |
|---|---|
| `batch` 8 or 16 | ✅ 16 |
| `workers: 2` always | ✅ 2 — and `train_yolo.py` hard-refuses otherwise |
| never `--cache` / `--cache ram` | ✅ `cache: false` |
| 32 GB session cap | ✅ **Set 3 adds zero RAM over Set 1/2.** Every change (`hsv_s`, `scale`, `translate`, `box`, `dfl`, `cls`, `close_mosaic`) is either a loss coefficient or a CPU-side augmentation parameter operating on a fixed-size canvas. `scale: 0.55` is *below* Set 2's 0.7. The only change that would have cost memory was `imgsz: 768`, rejected for that reason. |
| server reset / power cut for AC maintenance | ⚠️ **newly addressed** — `save_period: 10`. `last.pt` every epoch covers a clean stop/resume, but not a power loss *during* `torch.save`, which leaves a truncated `last.pt` that `--resume` cannot load. Epoch snapshots cost disk only, no RAM. Check `df -h ~` first. |
| "start small, progressively increase" | ⚠️ **run `set3_smoke.yaml` first** — the generic `smoke_test.yaml` runs ultralytics defaults, so it does not exercise the MuSGD pin (at 2 epochs `auto` picks AdamW instead) or the `save_period` write path. |
| GPU sharing / run overnight / announce first | ➡️ unchanged, follow RUNBOOK's night-shift protocol. Set 3 is the same 100 epochs on `train_100.txt` as the other arms, so budget the same ~2 nights and flag it in the group line. |
| `nohup` | ✅ already the RUNBOOK convention. |

**Resume gotcha:** on `--resume`, ultralytics restores args **from the checkpoint**, not from the YAML. Editing `set3_merged_night_localization.yaml` mid-run has no effect on a resumed run. If a hyperparameter needs to change, the run has to restart.

---

## 5. Open flag — `test.txt` has now been used for model selection

RUNBOOK §5: *"Never touch `splits/test.txt` except for this final scoring pass."*

Set 3's design is derived from `test.txt` observations (the day/night mAP75 gap, the per-class Set 1 vs Set 2 deltas). That is model selection on the test set. A `test.txt` score for Set 3 is therefore optimistically biased and is not comparable, as a clean held-out number, to the Set 1 / Set 2 scores.

Cheapest honest options, in order of preference:

1. Develop and compare Set 3 on `splits/val.txt` (1,542 imgs, already carved out and stratified by time-of-day), then take **one** final `test.txt` reading for the writeup.
2. Report the `test.txt` number but state plainly that Set 3 was configured after inspecting test-set results. Examiners generally accept a disclosed post-hoc arm; they do not accept an undisclosed one.

Worth raising with the team before the run, since it also affects how Set 3 is framed relative to the four planned methods.

---

## 6. Follow-ups, in priority order

1. **Set 3 with `dfl: 1.5`** — the one ablation that isolates the localization lever this whole arm rests on. Everything else in Set 3 is inherited from an arm that already ran.
2. **Set 3 at `imgsz: 768`** — strongest untried lever for night mAP75. VRAM cost, not host-RAM cost; still needs a `free -h` / `nvidia-smi` check at `workers: 2`. Breaks eval comparability, so it is its own arm.
3. **Seed repeat of the winner** — every arm is n=1. The ~1.3 pt overall gap is probably real; class-level gaps on <800-instance classes are within plausible seed noise. One repeat gives an error bar.
4. **Rare classes** — the real fix is instance oversampling in the dataloader, or polygon labels from BDD100K's 10K instance-seg subset so `copy_paste` actually fires. Not `cls` gain.

---

## 7. Commands

```bash
# 1. smoke test first (2 epochs, real settings) - watch `free -h` in a second pane
python tools/train_yolo.py --hyp configs/hyp/set3_smoke.yaml --name set3_smoke
#    PASS: optimizer line says MuSGD (not "optimizer=auto found ..."),
#          "Closing dataloader mosaic" appears before epoch 2,
#          weights/ has last.pt, best.pt AND epoch0.pt, RAM well clear of 32 GB
rm -rf runs/detect/set3_smoke

# 2. the real arm - announce in the GPU group line first, launch in the evening
nvidia-smi && free -h && df -h ~
nohup python tools/train_yolo.py --hyp configs/hyp/set3_merged_night_localization.yaml \
    --name set3_100 > set3_100.log 2>&1 &

# 3. resume next evening (same command + --resume)
nohup python tools/train_yolo.py --hyp configs/hyp/set3_merged_night_localization.yaml \
    --name set3_100 --resume > set3_100.resume.log 2>&1 &

# 4. score on val.txt during development (see §5 before scoring on test.txt)
python evaluation/run_inference.py --weights runs/detect/set3_100/weights/best.pt \
  --image-list splits/val.txt --images-dir dataset/yolo/images/train \
  --out-dir runs/preds/set3_100_val --device 0
```

Note on step 4: `splits/val.txt` is carved out of BDD's *train* pool, so `--images-dir` is `dataset/yolo/images/train`, not `.../val` as in RUNBOOK §5.
