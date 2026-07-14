# What Has Been Done So Far — In Plain Words

*(Last updated: 2026-07-12, after Milestones 1–3 groundwork. This document explains what exists in the repo and why, without jargon. The technical spec lives in the handoff brief; the commands live in RUNBOOK.md.)*

## The project in one paragraph

Vehicle detectors work well in daylight and get much worse at night — our own team's numbers show the accuracy dropping by a big margin on night images. My contribution is testing one idea: before teaching the detector (YOLOv5) to find vehicles, first let a separate model (I-JEPA) study tens of thousands of unlabeled driving photos — many of them night and rainy scenes — so it learns what these scenes generally look like. Then we transfer that knowledge into the detector's "backbone" (its feature-extracting front half) and check: does a detector that starts from this knowledge beat one that starts from the standard COCO starting point, especially at night and when we only have a little labeled data?

## Step 1 — I read the starter repo (Milestone 1)

The plan assumed my teammate's repo contained a working BDD100K training setup we could reuse. It turned out to hold a plain, unmodified copy of YOLOv5 plus some leftover code from an earlier plan that used a different dataset (KITTI). There was no script to convert BDD100K's labels into YOLO's format, no list of which object classes he trained on, no training settings, and no evaluation code for detection. So almost everything had to be built from scratch. This is written up in `docs/REPO_SURVEY.md`, along with a list of open questions ("FLAGs") — the biggest one being that we still need his class list file (`Bdd10k_vehicle.yaml`), his training command, and his trained model file.

I also pinned down the software environment (`requirements.txt`) so the same package versions run on the university GPU server, and started `RUNBOOK.md` — the step-by-step command list Kanade follows on the server, since Claude only writes code and never runs training itself.

## Step 2 — I built the data pipeline and tested it (Milestone 2)

Four small tools now live in `tools/`:

**The label converter** (`bdd_to_yolo.py`). BDD100K ships its annotations as one big JSON file; YOLO wants one small text file per image with box coordinates scaled 0–1. The converter does this translation. The class list is read from a config file, so when the teammate's real class list arrives we swap one file and re-run — no code changes. I ran it on the 10,000 validation images and drew the resulting boxes back onto the photos to check them by eye: the boxes sit exactly on the cars, signs and lights, including in night scenes.

**The attribute index** (`build_attr_index.py`). Every BDD image is tagged with time of day and weather. This tool collects those tags into one lookup table. It matters twice: during pretraining we want to show the model *more* night/rainy images than average, and during evaluation we want separate scores for night, dawn/dusk and rainy subsets. (One technical note: the training label file is 1.4 GB and crashed a small machine when loaded whole, so the tool now reads it as a stream — this also makes it safer on the shared server.)

**The split maker** (`make_splits.py`). Fair experiments need everyone to use exactly the same image lists. This tool carves out, with a fixed random seed: a 2,500-image "model selection" set (used for all development decisions, so the real 10,000-image validation set stays untouched until the very final scores — this avoids accidentally tuning to our test set), the pretraining pool, and the 100% / 50% / 25% / 10% label-fraction lists. Each smaller list keeps the same day/night mix as the full set, so a 10% experiment isn't accidentally all daytime. The lists are committed to git, so every experiment — mine and future ones — uses identical data.

**The box viewer** (`viz_boxes.py`) — the little tool used for the eyeball check above.

One surprise worth knowing: our copy of the training labels covers 64,520 images, but the official dataset has 69,863. We're missing about 7.6%. Everything is built and double-checked around the 64,520 figure, and the RUNBOOK stops with a warning if the server copy turns out different.

## Step 3 — I built the measuring equipment (Milestone 3, first half)

Before training anything, we need a referee. Two scripts in `evaluation/`:

**The prediction runner** (`run_inference.py`) takes any trained model file and produces its detections on a given image list — always with identical settings, so no model gets an unfair advantage from different thresholds.

**The scorer** (`eval_detections.py`) compares those detections against the ground truth and reports the standard accuracy scores (mAP at different strictness levels), broken out by overall / night / dawn-dusk / rainy. Every model in the project — ours, the teammate's, all ablation arms — gets scored by this one script, so all numbers in the final table are directly comparable. I verified it works by feeding it artificial predictions (correct boxes, slightly shaken, with some removed and some fake ones added) and checking the scores behave sensibly.

There is also a training launcher (`tools/train_yolo.py`): each experiment is described by one small config file, and the launcher records everything needed to reproduce it (the exact settings, the git version of the code, the random seed) into the run folder, and can resume automatically if the server connection drops mid-training. The first two experiment configs — the COCO-baseline at 100% and at 10% of the labels — are ready in `configs/exp/`; RUNBOOK section 4 has the exact commands and rough runtimes (about 4–6 hours for the big one).

## What's next

First the baseline gets trained on the server (RUNBOOK §4) — that's ablation row one. Then the "T1" path: distilling knowledge from Meta's released I-JEPA model into the YOLOv5 backbone, which proves the transfer machinery works. Then the thesis piece, "T2": pretraining our own I-JEPA on BDD100K's night-heavy driving images and running the same transfer. **Update 2026-07-13:** the teammate's files arrived. They settled the three open questions: he detects only three vehicle types (car, bus, truck — everything else is ignored), he used the small model size (yolov5s), and his training settings are now copied into our experiment configs (640-pixel images, batch 64, 100 epochs). All our experiment arms now follow his exact protocol, so the final comparison table will be apples-to-apples — including his own trained model, which we will re-score with our own referee script.

## Update 2026-07-13 (second): the full pipeline is now code-complete

The last missing piece — Stage 1, pretraining our own I-JEPA on BDD100K's driving images — is written (`jepa_pretrain/`), reusing Meta's official implementation (vendored into `third_party/ijepa`) for the two most bug-prone parts: the block-masking logic and the slow-moving "target encoder" update. Our additions: night/dawn-dusk/rainy images are sampled about twice as often (the whole point is a night-savvy teacher), a "collapse alarm" that warns if the model starts outputting constant embeddings (the classic failure mode of this training style), and the same resume/logging machinery as everything else. There's also now a results aggregator that turns all evaluation outputs into the final comparison table, and a RUNBOOK appendix for running the small (10%-labels) experiments on free Kaggle GPUs. Every stage of the project now exists as runnable code; what remains is executing RUNBOOK sections 1–7 on the GPU server and reading the numbers.
