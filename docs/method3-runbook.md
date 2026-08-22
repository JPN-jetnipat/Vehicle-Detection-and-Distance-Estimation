# Method 3 (Field) runbook — low-light augmentation × IRFS oversampling

All commands run **from the repo root**, same as `RUNBOOK.md`. This document
covers only the Method 3 arm; environment setup, dataset plumbing and the
night-shift GPU protocol are in `RUNBOOK.md` §1–§3 and still apply.

---

## 1. What this arm is

| | pool | oversampling | samples/epoch |
|---|---|---|---|
| **Method 1** (Baby) | original + low-light copy of every image | none | 120,372 |
| **Method 2** (JPN) | original only | IRFS on bike/motor | 68,763 |
| **Method 3** (Field) | original + low-light copy of every image | IRFS on bike/motor | **~137,500** |

Method 3 is the composition, not a third independent idea: it takes Method 1's
pool exactly as Method 1 defines it, and applies Method 2's sampler to it
exactly as Method 2 defines it. Same `yolo11s.pt` start, same recipe
(`configs/hyp/set3_combined.yaml`), same val split. The only thing that varies
across the three arms is the training pool — which is what makes the ablation
readable.

### Why the composition is clean (put this in the write-up)

The low-light augmentation is **pixel-level only** — gamma, sensor noise, motion
blur. No box moves, none are added or removed. So the augmented pool has an
identical class distribution to the original, and doubling the pool leaves both
IRFS frequency terms unchanged:

```
f_(i,c) = (2 · images_containing_c) / (2 · N)        = unchanged
f_(b,c) = (2 · boxes_of_class_c)   / (2 · total)     = unchanged
  ⇒ r_c = max(1, √(t / √(f_i_c · f_b_c)))            = unchanged
```

So Method 3's per-class repeat factors are **provably identical** to Method 2's
(bike **2.417**, motor **3.388** at t = 0.122). `tools/build_method3_split.py`
verifies this numerically rather than asserting it — pass
`--method2-report <path to irfs_distribution.json>` and it fails loudly if the
two arms ever drift apart.

Concretely, this means:

- Method 3 vs **Method 2** differs in exactly one thing: half its samples are
  night-degraded.
- Method 3 vs **Method 1** differs in exactly one thing: bike/motor images are
  repeated.

### Provenance

| file | origin |
|---|---|
| `tools/augment_lowlight.py` | Method 1's script; augmentation math unchanged, plumbing adapted (see §6) |
| `tools/compute_repeat_factors.py` | **verbatim** copy of Method 2's module — do not edit locally |
| `tools/build_method3_split.py` | new; the composition step |
| `configs/hyp/set3_combined.yaml` | the agreed cross-team recipe, byte-identical to Methods 1 and 2 |

---

## 2. Cost before you start

**Disk — measured on 40 real BDD train images, not estimated:**

| `--jpeg-quality` | avg augmented image | 60,186 images |
|---|---|---|
| **95** (default, matches Method 1) | 183 KB | **10.5 GB** |
| 90 | 118 KB | 6.8 GB |
| 85 | 89 KB | 5.1 GB |

Originals average 54 KB; darkening plus sensor noise is what costs the 3.4×
— noise is the least compressible thing you can put in a JPEG. Stay at **95**
unless disk forces otherwise: dropping quality changes the compression
artifacts relative to Method 1's images and becomes a confound you have to
disclose.

```bash
df -h ~     # want ≥ 12 GB free before stage 1
```

**Time.** Stage 1 is single-threaded CPU, roughly 40–90 min for the full
60,186 images depending on the host. It touches no GPU — run it while
something else has the A40.

**Training.** ~137,500 samples/epoch is **≈2.0× Method 2's** epoch time and
**≈1.14× Method 1's**. `RUNBOOK.md`'s convention says runs projected past ~2
nights of shared A40 time get announced to the team first — this arm is the
one most likely to cross that line, so time it off Method 1's measured
epoch time and announce before launching.

---

## 3. Stage 1 — materialize the low-light copies

Probe first, on 500 images, to confirm the real per-image size on this machine
before committing 10 GB:

```bash
python tools/augment_lowlight.py --limit 500 \
    --dst-root dataset/lowlight_probe \
    --out-list dataset/yolo/splits/train_100_lowlight_probe.txt \
    --log-csv ''
du -sh dataset/lowlight_probe          # ×120 ≈ the full-run footprint
python tools/viz_boxes.py --images-dir dataset/lowlight_probe/images/train \
    --labels-dir dataset/lowlight_probe/labels/train \
    --data-config configs/data/bdd100k_vehicle5.yaml --num 8 --out-dir viz_out
```

Eyeball those 8 overlays. Boxes must sit exactly where they do on the
originals — if they don't, something non-pixel-level crept in and the whole
premise of §1 breaks. Then delete `dataset/lowlight_probe/` and run for real:

```bash
nohup python tools/augment_lowlight.py > lowlight_aug.log 2>&1 &
tail -f lowlight_aug.log
```

Writes:

```
dataset/lowlight/images/train/*.jpg              60,186 augmented images
dataset/lowlight/labels/train/*.txt              labels, copied unchanged
dataset/yolo/splits/train_100_lowlight.txt       repo-relative image list
results/sampling_reports/lowlight_aug_log.csv    which gates fired per image
```

The log prints a **live size projection** every 500 images — check it once and
kill the job early if it is heading somewhere you don't have room for.

Expected in the summary: gamma 100.0% (forced by design — see §6), noise ≈50%,
blur ≈50%.

**If it dies partway (power cut, full disk), just rerun the same command.** It
skips images already written and reproduces byte-identical output for them,
because each image's RNG is seeded from `(seed, filename)` rather than from one
sequential stream.

---

## 4. Stage 2 — build the Method 3 split

```bash
python tools/build_method3_split.py --dry-run          # look before writing
python tools/build_method3_split.py \
    --method2-report ../irfs_augmentation/irfs_distribution.json
```

(Drop `--method2-report` if you don't have your teammate's file to hand; the
cross-check is optional but it is the cheapest guard against the two arms
silently diverging.)

Three checks run before anything is written, each of which has a real failure
mode behind it:

| check | what it catches |
|---|---|
| pool sizes match | stage 1 stopped early — Method 3 would quietly train on a partial low-light pool |
| twin labels match | a label copy was missed — the twin reads as a background image, its repeat factor drops to 1, and the arm under-samples the exact classes it exists to boost |
| `r_c` matches Method 2 | t, the target classes, or the underlying labels drifted between arms — the ablation stops being an ablation |

Writes:

```
dataset/yolo/splits/train_100_method3.txt        ~137,500 lines
configs/data/bdd100k_vehicle5_method3.yaml       dataset yaml
results/sampling_reports/method3_distribution.json
```

Expected table (motor/bike ≈ 2× Method 2's after-counts, everything else ≈ 2×
its before-counts):

```
class        r_c   imgs before   imgs after   inst before   inst after
car        1.000       119,090      136,070     1,234,758    1,407,300
truck      1.000        31,846       36,996        50,456       58,582
bus        1.000        14,710       17,692        19,044       22,910
motor      3.388         3,702       12,526         4,848       16,372
bike       2.417         6,498       16,280        10,660       27,000
```

Exact totals will land within a few hundred of these — stochastic rounding is
seeded but drawn per pool entry, so the doubled pool is ~2× Method 2 in
expectation, not to the line.

---

## 5. Train

**Smoke test first** — `configs/hyp/smoke_test.yaml` does not exercise any of
this arm's specific risks:

```bash
python tools/train_yolo.py --hyp configs/hyp/set3_combined_smoke.yaml \
    --data configs/data/bdd100k_vehicle5_method3.yaml --name set3_combined_smoke
free -h    # second pane — stay well clear of 32 GB
```

Verify in the output:

- the scan line says **~137,500 images**. 60,186 means the method3 data yaml
  wasn't picked up; 120,372 means the repeated lines got dropped.
- `Scanning dataset/lowlight/labels/train...` with **0 corrupt** — this is the
  proof that ultralytics resolved the low-light images to the low-light labels.
- 2 epochs finish; `runs/detect/set3_combined_smoke/weights/` holds `last.pt`,
  `best.pt` and at least one `epoch*.pt`.

Then delete `runs/detect/set3_combined_smoke/` and launch the real arm
(announce on the GPU line first, per `RUNBOOK.md`):

```bash
tmux attach -t train || tmux new -s train
source .venv/bin/activate
nvidia-smi && free -h
nohup python tools/train_yolo.py --hyp configs/hyp/set3_combined.yaml \
    --data configs/data/bdd100k_vehicle5_method3.yaml \
    --name set3_combined > set3_combined.log 2>&1 &
# detach: Ctrl+b d ; resume next evening: same command + --resume
```

`--name set3_combined` is passed on the command line, **not** set in the hyp
yaml — `tools/train_yolo.py` already passes `name=` into `model.train()`, so a
`name:` key in the yaml would raise `TypeError: got multiple values for
keyword argument 'name'`. Same for `model` and `data`.

---

## 6. Evaluation

Identical to `RUNBOOK.md` §5 — nothing about this arm changes the protocol, and
`configs/data/bdd100k_vehicle5_method3.yaml` deliberately copies `val:` from
`bdd100k_vehicle5.yaml` so every arm validates against the same split.

```bash
python evaluation/run_inference.py --weights runs/detect/set3_combined/weights/best.pt \
  --image-list splits/test.txt --images-dir dataset/yolo/images/val \
  --out-dir runs/preds/set3_combined_test --device 0

python evaluation/eval_detections.py \
  --image-list splits/test.txt --gt-labels dataset/yolo/labels/val \
  --preds runs/preds/set3_combined_test \
  --attr-index dataset/yolo/attr_index.json \
  --data-config configs/data/bdd100k_vehicle5.yaml \
  --out runs/metrics/set3_combined_test
```

The metrics that decide whether the composition earned its 2× cost:

- **bike / motor mAP50** — did IRFS still pay off once half the pool is
  degraded? (Method 2 alone: bike 0.4605, motor 0.3918.)
- **night mAP50 per class** — did low-light augmentation still pay off once
  bike/motor are oversampled?
- **car mAP75, daytime → night delta** — Finding C in
  `docs/set3-notes.md`: a −9.9 pt collapse that no arm has fixed yet. Method 3
  is the first arm whose *training data* is night-degraded rather than just its
  loss weights, so this is the number most likely to move.

---

## 7. Flags and gotchas

**`copy_paste: 0.2` is inert and stays anyway.** `CopyPaste.__call__` returns
immediately when `labels["instances"].segments` is empty, and BDD in YOLO
detect format is bbox-only (`docs/set3-notes.md` Finding B, verified
empirically). It is kept in `set3_combined.yaml` **only** so this arm's recipe
stays byte-identical to Methods 1 and 2 — dropping it would make Method 3's
config differ from theirs in a second place. Do not report it as an active
augmentation in the write-up.

**`optimizer: auto` resolves to MuSGD here — same as every other arm.** Per
`docs/set3-notes.md` §3 the threshold is on *iterations*, not epochs:
`ceil(137526 / 64) × 100 ≈ 214,900 > 10,000` → MuSGD, lr 0.01, momentum 0.9,
`warmup_bias_lr 0.0`. Methods 1 (≈188,100) and 2 (≈107,500) clear it too, so
all three arms match. Nothing to pin — but if anyone ever runs this arm at a
smaller label fraction, re-check that number before trusting the comparison.

**A label cache appears at `dataset/lowlight/labels/train.cache`.** Ultralytics
writes it next to the *first* label file in sorted order, which for this arm is
in the low-light tree. It is hash-validated against the split, so regenerating
the split invalidates it automatically; delete it freely. It does not collide
with `dataset/yolo/labels/train.cache`, which the other arms use.

**Name collision with the earlier "Set 3".** `configs/hyp/set3_combined.yaml`
(this arm) is not `configs/hyp/set3_merged_night_localization.yaml` (the
hyperparameter-search arm in `docs/set3-notes.md`). Different files, different
run dirs, no runtime collision — but say "set3_combined" in full whenever you
write about either.

**Test-set leakage caveat still applies.** `docs/set3-notes.md` §5: the recipe
in `set3_combined.yaml` was chosen from `test.txt` observations, so a
`test.txt` score for this arm carries the same optimistic bias as the earlier
ones. Method 3 doesn't make that worse, but it doesn't fix it either — develop
on `splits/val.txt` and disclose the post-hoc framing.

**Two deliberate deviations from Method 1's script**, both plumbing, neither
touching the augmentation distribution:

1. *Repo-relative output paths* instead of absolute. Method 1's script wrote
   absolute paths, which don't survive the Mac → GPU-server move this project
   actually makes; this repo's own `train_100.txt` is repo-relative and
   `tools/train_yolo.py` already assumes cwd = repo root.
2. *Per-image RNG seeded from `(seed, filename)`* instead of one global
   sequential stream. Same operators, same gates, same parameter ranges, same
   distribution — but resumable, which matters given the AC-maintenance power
   cuts in `docs/set3-notes.md` §4. With a global stream, resuming at image
   40,000 silently draws a different sequence.

Everything else — gamma U(2, 3.5) force-applied, Poisson–Gaussian noise at
p = 0.5, motion blur at p = 0.5, order gamma → noise → blur, seed 42 — is
Method 1's, unchanged. Citations (NightAug/2PCNet CVPR 2023, MAET ICCV 2021,
Foi et al. TIP 2008) are in the script docstring.

---

## 8. Results table

Once all arms have scored:

```bash
python tools/make_results_table.py --metrics-dir runs/metrics --out results/ablation \
  --require vanilla_test default_100_test method1_lowlight_test \
            method2_irfs_sampling_test set3_combined_test
```

The three-way read the ablation is built to support:

| | low-light aug | IRFS | isolates |
|---|---|---|---|
| Method 1 | ✓ | | effect of night-degraded data alone |
| Method 2 | | ✓ | effect of rare-class oversampling alone |
| **Method 3** | ✓ | ✓ | whether the two **compose** or interfere |

The interesting outcome is not "Method 3 wins". It is whether Method 3's gain
over the default arm is bigger, equal to, or smaller than Method 1's gain plus
Method 2's gain. Sub-additive would say the two interventions are fixing the
same underlying weakness; super-additive would say low-light augmentation makes
the extra bike/motor samples *more* informative rather than just more numerous.
Either result is worth reporting.
