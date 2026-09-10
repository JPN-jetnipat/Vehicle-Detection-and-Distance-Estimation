# configs/archive — closed experimental lines

Configs in here belong to lines of work that are **finished and will not be re-run**.
They are kept because they are the provenance for numbers already written into
`docs/MASTER-RECORD.md`. Nothing here should be passed to `tools/train_yolo.py`
for a new arm.

Archived 2026-09-10, after the JEPA phase made the live config set much smaller.

## ⚠️ Read this before you go looking for `set3_combined.yaml`

**`configs/hyp/set3_combined.yaml` is NOT in this folder and must never be moved here.**

Its name comes from the Method 3 *run*, but the file itself is the **live shared
cross-team recipe** used by BASE, M1, M2, M3, T1, and (via `set3_combined_10.yaml`)
every 10%-label arm. It is byte-identical across all of them, which is the whole
basis of "arms differ in exactly one thing". Moving or editing it invalidates every
result in the project. See `docs/MASTER-RECORD.md` §1 "naming hazard" and §3.11.

## What's in here

### `hyp/` — the hyperparameter search (closed, decided)

| file | what it was | where its numbers live |
|---|---|---|
| `default.yaml` | the first fine-tune arm, ultralytics defaults + the RAM fix | `README.md` arm table |
| `set1_japan_night_aug.yaml` | Set 1 — low-light augmentation hypothesis | MASTER-RECORD §2.1 |
| `set2_field_imbalance.yaml` | Set 2 — class/scale diversity. **Strongest single search arm** | MASTER-RECORD §2.1, §3.11 |
| `set3_merged_night_localization.yaml` | Set 3 — dfl 1.8 night-localization attempt. **Not** the recipe | MASTER-RECORD §2.1 (Finding F) |

The search concluded in `set3_combined` (still live). §3.11 records why that recipe
was kept over Set 2 and why it is not revisited: on `car` — the only class with
adequate instance support — the two are a tie, and every `all`-level gap between them
is carried by classes under the ≥800-instance reporting threshold.

### `hyp/` — smoke configs for closed lines

`set3_smoke.yaml`, `set3_combined_smoke.yaml`, `smoke_test.yaml` — 2-epoch wiring
tests for arms that are done. The live smoke config is
`configs/hyp/set3_combined_10_smoke.yaml`, which is **not** archived because it still
has a job: proving the pinned optimizer is accepted before a 10%-label run.

### `data/` — the augmentation line (closed, negative result)

`bdd100k_vehicle5_method3.yaml` — the Method 3 training pool: the original 60,186
images plus a synthetic low-light copy of each, with IRFS repeat factors on bike and
motor. This is the only *augmentation-specific* config the project had.

**Why it's closed.** M1 (low-light copies) ranked **last** on every time-of-day slice
including night, the slice it was designed to fix. The mechanism is in MASTER-RECORD
§2.2: a gamma curve models dusk, not night, and ~40% of BDD's train split is already
night, so roughly 24,000 of the augmented copies were near-black and off-distribution.
The negative result stands on its own and is reported; it does not need re-running.

## What deliberately stayed live in `configs/`

| path | why |
|---|---|
| `hyp/set3_combined.yaml` | the shared recipe — see the warning above |
| `hyp/set3_combined_10.yaml` | same recipe with the optimizer pinned, for the 10% arms (§3.5) |
| `hyp/set3_combined_10_smoke.yaml` | verifies the optimizer pin before a real 10% run |
| `data/bdd100k_vehicle5.yaml` | the canonical 5-class dataset config |
| `data/bdd100k_vehicle5_10.yaml` | the 10%-label variant |
| `data/bdd100k_test_*.yaml` | the 19 test slices — used by every scoring pass |
| `jepa/*.yaml` | Stage-1 pretrain and Stage-2 distillation, both still active |

## The augmentation *tools* were not archived

`tools/augment_lowlight.py`, `tools/build_method3_split.py` and
`tools/compute_repeat_factors.py` stay in `tools/`. They regenerate
`results/sampling_reports/`, which is tracked and cited in the write-up, and the
augmentation line's reproducibility depends on them. Archive them only if you also
stop claiming M1–M3 are reproducible.
