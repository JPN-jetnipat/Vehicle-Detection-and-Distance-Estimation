# Updating the Jupyter server to match this project

The old project (`(Claude)Vehicle-Detection-and-Distance-Estimation`,
probably cloned on the server as something like `vehicle-jepa/`) already has
the ~4.6 GB raw BDD100K dataset uploaded. This guide reuses that instead of
re-doing the slow chunked upload, and gets a clean checkout of *this*
(YOLOv11s / 5-class) project running alongside or in place of it.

## Decision to make first: new folder, or edit in place?

**Recommended: a fresh checkout in a new folder next to the old one.** The
codebase changed enough (dropped the vendored `yolov5/` repo, new class
scheme, new scripts) that surgically editing the old checkout risks leftover
stale files quietly getting picked up. A fresh clone is unambiguous. The old
checkout isn't touched, so you can still reference/diff against it, and only
delete it once the new one is confirmed working (Section 5).

This also means you need to decide, with your teammate: does this project
get a **new GitHub repo**, or a **new branch/folder in the existing one**
(`JPN-jetnipat/Vehicle-Detection-and-Distance-Estimation`)? Either works —
just make it a deliberate choice, since the old plan flagged repo visibility
(public/private) as something to decide on purpose, not by default.

## Step 1 — Push this local folder to GitHub (on your Mac)

```bash
cd "/path/to/Vehicle Detection"
git remote add origin <your-new-or-existing-remote-url>
git branch -M main          # or a dedicated branch, if reusing the old repo
git push -u origin main
```
(This repo was already `git init`'d with an initial commit locally — see the
bottom of this project's setup notes if you need to redo that.)

## Step 2 — Clone it on the server

In a Jupyter terminal, next to (not inside) the old checkout:
```bash
cd ~
git clone <your-remote-url> vehicle-detection
cd vehicle-detection
```

## Step 3 — Reuse the already-uploaded raw dataset (no re-upload)

The raw images + label JSONs don't change with the class scheme — only how
they get *converted* does. Copy (not move, in case anything still points at
the old copy) the raw folders across, server-side, which is fast since it's
local disk-to-disk, not a network transfer:

```bash
cp -r ~/vehicle-jepa/dataset/raw/bdd100k ~/vehicle-detection/dataset/raw/
cp -r ~/vehicle-jepa/dataset/raw/bdd100k_labels_release ~/vehicle-detection/dataset/raw/
```
(Adjust `~/vehicle-jepa` to whatever the old checkout is actually named on
your server — check with `ls ~`.)

If disk space is tight, `mv` instead of `cp` works too (frees the old copy);
just make sure nothing else still needs it first.

**Verify:**
```bash
ls ~/vehicle-detection/dataset/raw/bdd100k/bdd100k/images/100k/
du -sh ~/vehicle-detection/dataset/raw/
```

## Step 4 — Environment + data plumbing

Follow **RUNBOOK.md** Sections 1 and 3 from here — new venv (this project
uses `ultralytics`/YOLOv11, not the old vendored `yolov5/` repo, so don't
reuse the old `.venv`), then attribute index → convert labels → materialize
images → make splits, all against `configs/data/bdd100k_vehicle5.yaml`
(5 classes: car/truck/bus/motor/bike — this is why re-conversion is
necessary even though the raw data is reused; the old converted labels were
3-class and are not compatible).

Section 3.1's `--expect-counts train=69863,val=10000` check will confirm the
copied raw data is the complete set before you spend time converting it.

## Step 5 — Once the new checkout is confirmed working

Run one short training smoke-test (a couple epochs) to confirm the pipeline
runs end-to-end on the server before committing to full runs. Once satisfied:

- Keep the old checkout around as long as you might want to reference its
  docs/results, or
- Free the disk space it's using: `du -sh ~/vehicle-jepa` first to see what
  you'd reclaim, then `rm -rf ~/vehicle-jepa` once you're sure nothing in it
  is still needed (its raw dataset copy is now redundant with Step 3 above
  either way).
