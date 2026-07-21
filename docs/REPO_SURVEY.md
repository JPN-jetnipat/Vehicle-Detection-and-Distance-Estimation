# Repo Survey — Milestone 1 (2026-07-12)

Survey of the working copy (friend's reconstructed baseline pipeline) against the handoff brief.

## What the repo actually contains

| Path | Contents | Status vs. brief |
|---|---|---|
| `yolov5/` | Vanilla Ultralytics YOLOv5 clone (converted from submodule, 93a043e). Stock `data/*.yaml`, stock `data/hyps/`, unmodified `train.py`/`val.py`. | No BDD data config, no custom hyperparameters. |
| `evaluation/metrics.py` | MAE/RMSE stubs (4 lines) for **distance** estimation. | No detection eval protocol exists. |
| `geometry/` | KITTI calibration parser + pinhole distance estimator. | Out of scope (distance estimation dropped). |
| `docs/NEXT_STEPS.txt` | KITTI-era plan ("Implement convert_kitti_to_yolo.py" still pending). | Stale — predates the KITTI→BDD pivot. |
| `docs/dataset-upload-guide.md` | Chunked zip upload to the ICT GPU server (cngpu-vm001) via JupyterLab. Notes that the **Kaggle CLI failed with persistent 401s** (kaggle-cli#888). | Conflicts with brief §9 (kaggle CLI on server). See FLAG 2. |
| `dataset/raw/` | Partial local BDD100K: val **10,000** (complete), train **1,160**, test **293**; both label JSONs present; `10k` seg folder present (ignore). ~15 GB. Gitignored. | Good enough for local converter unit tests. Full data lives on the server. |
| `.gitignore` | Excludes `runs/`, `*.pt`, `*.pth`, `dataset/`. | Friend's weights/configs were never committed. |

Git history: 7 commits, ending `cbe0301 "baseline: final version"` (Jun 13) and `d0ba7eb` (docs update). No training-run artifacts anywhere in history.

## Answers to the four survey questions

1. **Data converter:** does not exist in the repo — neither KITTI nor BDD. We must write `tools/bdd_to_yolo.py` ourselves (Milestone 2), plus the attribute index.
2. **Class list:** not defined anywhere in the repo. See FLAG 1 and the provisional decision below.
3. **Training hyperparameters:** none beyond Ultralytics defaults. The friend's actual 10k-subset runs (numbers in brief §3) were produced from files not present here — presumably on the server. Until recovered, our baseline recipe = YOLOv5 defaults (`hyp.scratch-low.yaml`, 640 px), documented as provisional.
4. **Eval protocol:** none for detection. We write our own evaluator (brief §8) — already the plan ("one evaluator for everything"); his weights get re-scored by our script whenever we obtain them.

## FLAGS (raised 2026-07-12, decisions recorded)

**FLAG 1 — friend's class list / recipe / weights missing from repo.**
Decision (Kanade, 2026-07-12): proceed with a provisional default of **all 10 BDD detection classes** (`car, bus, truck, person, rider, bike, motor, traffic light, traffic sign, train`), clearly marked provisional; safest for later re-scoring of his weights. Swap to his exact list when his files arrive. **Action for Kanade: ask him for (a) his data yaml + class list, (b) hyp yaml / train command, (c) best.pt, (d) model scale (yolov5s?).**
Update 2026-07-12: friend says the class list is in his `Bdd10k_vehicle.yaml` (file itself not yet received; not in repo or git history — name implies a vehicle subset, so expect the provisional all-10 list to change). Items (b)–(d) still pending.
**RESOLVED 2026-07-13:** friend's files received (now in `dataset/` + `scripts/`; his weights at `weights/friend_best.pt`, gitignored).
His `bdd100k_vehicle.yaml` lists 8 KITTI-legacy names but his converter populates only 3: car→0 ("Car"), bus→1 ("Van"), truck→2 ("Truck") — everything else dropped. Our canonical config `configs/data/bdd100k_vehicle3.yaml` keeps his ids exactly, with truthful BDD names. His 100k train command: `--img 640 --batch 64 --epochs 100 --weights yolov5s.pt --workers 8`, default hyps (adopted into all exp configs). His 10k command was not saved; his 10k subset was a random 7000/1000 split, seed 42 (not used by us).

**FLAG 2 — Kaggle CLI vs. chunked upload.**
Brief §9 says kaggle CLI on the server; the friend's guide reports it broken (401s, kaggle-cli#888). RUNBOOK documents the CLI path first (worth retrying — may be fixed) with the friend's chunked-upload procedure as the tested fallback.

**FLAG 3 — model scale unconfirmed.**
No configs exist to confirm yolov5s (brief §7). Provisional default: **yolov5s**. Confirm with the friend (folded into FLAG 1 action).
**RESOLVED 2026-07-13:** `best.pt` (14.4 MB) embeds `yolov5s`, depth 0.33 / width 0.50 → **yolov5s confirmed**.


**FLAG 4 — train label JSON is short: 64,520 entries vs official 69,863.**
The local copy of `bdd100k_labels_images_train.json` (from the friend's zip of the
Kaggle mirror) contains 64,520 image entries; the official BDD100K detection train
set has 69,863. Val is complete (10,000). Impact: ~7.6% fewer train images than the
brief's "~70k" everywhere (pretraining pool, fine-tune pool). All committed splits/
lists were generated from the 64,520-entry JSON (see splits/MANIFEST.json); the
RUNBOOK's attr-index step hard-checks this count on the server and stops on mismatch.
Action for Kanade: when downloading on the server, if the Kaggle-CLI copy has the full
69,863, tell me — splits must then be regenerated (same seed, one command).

**Update 2026-07-19:** concretely hit during the Stage-2 (T1 distillation) smoke
test — `jepa_distill` raised `FileNotFoundError` for an image
(`00091078-cedbfea7.jpg`) listed in `splits/pretrain.txt` (derived from the
label JSON) but absent from `dataset/raw/bdd100k/bdd100k/images/100k/train/`
on the server. This is the JSON-vs-disk gap FLAG 4 already worried about, now
showing up as an actual missing file, not just a suspicious total count.
`jepa_distill/data.py` was hardened to skip missing files with a loud warning
and to hard-stop if more than 1% of any image list is missing (rather than
silently train on a shrunk dataset). Diagnostic requested from Kanade, not
yet run: exact file count in the train images folder, and how many of
`pretrain.txt`'s 62,020 names are missing from it (`comm -23` against a
sorted `ls`). Val is independently confirmed complete — all 10,000 files
loaded successfully during baseline val scoring.

**FLAG 5 — GitHub remote is Public, not Private.**
The brief (section 1, item 7) specifies "a private GitHub remote." The actual
repo (`github.com/JPN-jetnipat/Vehicle-Detection-and-Distance-Estimation`) is
set to **Public** — confirmed 2026-07-14 from the repo page badge. Anyone on
the internet can currently read the full codebase (data itself is NOT
exposed — `dataset/`, `runs_jepa/`, weights, and `kaggle.json` are all
gitignored — but all source code, configs, and the design docs are visible).
Not necessarily a problem for a class project, but it should be a conscious
choice rather than a default. Action for Kanade/JPN: decide together whether
to flip the repo to Private (Settings -> General -> Danger Zone -> Change
visibility) or leave it Public; either is fine, just pick one deliberately.

**FLAG 5 — the HuggingFace I-JEPA checkpoint is the CONTEXT encoder (2026-07-21).**
Verified by reading HF's `convert_ijepa_to_hf.py`: it loads
`torch.hub.load_state_dict_from_url(...)["encoder"]` from Meta's
`IN1K-vit.h.14-300e.pth.tar`. That is the context (x) encoder; the brief's
non-negotiable §12 and I-JEPA paper App. A.1 specify the **target** encoder for
downstream use. The two differ (target = EMA of context), though they converge
substantially after 300 epochs.
Run `distill_t1` (2026-07-21) therefore used the CONTEXT encoder: loss
1.007→0.186, probe 0.98, 3.0 h. Remedy available at zero code cost:
`configs/exp/distill_t1_target.yaml` loads Meta's original .tar via the
`t2_ijepa` teacher (`target_encoder` key, arch vit_huge, patch 14).
Also fixed here: `jepa_distill/teacher.py` now puts `third_party/ijepa` on
sys.path itself (train_distill.py did not, so any original-format teacher —
including every T2 run — would have ImportError'd).


**FLAG 5 — HuggingFace `ijepa_vith14_1k` is the CONTEXT encoder, not the target encoder.**
Verified 2026-07-21 by reading transformers' `convert_ijepa_to_hf.py`: it loads
`checkpoint["encoder"]` from Meta's `IN1K-vit.h.14-300e.pth.tar`. Meta's tar contains
both `encoder` (context, x-encoder) and `target_encoder` (EMA, y-encoder). The brief
§12 non-negotiable requires distilling from the TARGET encoder (I-JEPA paper App. A.1).
Decision (Kanade, 2026-07-21): re-distill from the true target encoder using
`configs/exp/distill_t1_target.yaml` (loads Meta's original tar via the vendored ijepa
ViT, `ckpt["target_encoder"]`), and use THAT for the T1 arm. The completed
context-encoder run (`runs_jepa/stage2/distill_t1`, final loss 0.1856, day/night probe
0.980) is retained as `weights/init_t1_ctx.pt` for an optional context-vs-target footnote.

## Reuse verdict

Reusable: the vanilla `yolov5/` tree itself (train/val/detect), the upload guide, the partial local dataset for tests. Everything else in the brief's pipeline (converter, attribute index, splits, evaluator, all I-JEPA/distillation code) must be built new — which also keeps us cleanly separable from the DANN teammate: we only add files, never edit his.
