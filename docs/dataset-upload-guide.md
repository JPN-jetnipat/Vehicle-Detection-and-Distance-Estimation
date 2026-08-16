# Uploading a Large Dataset to the ICT GPU Server (cngpu-vm001)

This guide covers getting a large dataset (BDD100K) from your laptop onto the
university GPU server via Jupyter Hub, when the dataset is too big for a
single reliable browser upload. Applies to a fresh server checkout that
doesn't have the raw dataset up yet — if the server already has the *old*
project's copy of BDD100K, use `docs/SERVER_MIGRATION.md` instead and skip
this whole upload.

**Context:** the ICT-Server manual (VPN + Jupyter Hub setup) walks through a
Spark/Hive pipeline for structured/tabular data — that's not needed here.
Once you're on a GPU node (confirm with `nvidia-smi` in the Jupyter
terminal), you're just working with a normal Linux box with a GPU attached.
This guide is only about getting the image dataset onto that box.

**Why not the Kaggle API?** We tried it and it wasn't reliable at the time of
writing — both the new access-token method (`kagglehub`/`kaggle` CLI has an
open bug, [kaggle-cli#888](https://github.com/Kaggle/kaggle-cli/issues/888))
and the legacy `kaggle.json` method returned persistent 401s even with
freshly regenerated credentials. If you already have the dataset downloaded
locally (e.g. via the Kaggle *website's* Download button, which just uses
your logged-in browser session), skip straight to Step 1 below.

---

## Prerequisites

- Connected to the ICT-Server VPN, logged into Jupyter Hub, and have a terminal open on your GPU node.
- Dataset already downloaded to your own computer.

---

## Step 1 & 2 — Trim, zip, and split the dataset locally

Only zip what you actually need for training: the images and detection
labels — **not** `bdd100k_seg` (that's for the segmentation task).

A single multi-GB upload through a browser over VPN is slow and prone to
stalling. Splitting into chunks means a hiccup only costs you one chunk, not
the whole transfer — so we zip and split in the same step.

### macOS / Linux

```bash
cd "/path/to/your/dataset/raw"
zip -r -X ~/Desktop/dataset.zip bdd100k bdd100k_labels_release -x "*.DS_Store"
cd ~/Desktop
split -b 500m dataset.zip dataset_part_
ls -la dataset_part_*
```

- `-X` strips extra macOS metadata; `-x "*.DS_Store"` excludes `.DS_Store` files.
- `500m` = ~500MB per chunk. Drop to `250m` if your connection is especially unstable.
- `split` names parts alphabetically (`dataset_part_aa`, `_ab`, `_ac`, ...) by default — that's fine, wildcards expand in the correct order later.

### Windows

Windows doesn't have `zip`/`split` built in. Use **7-Zip** instead (free —
[7-zip.org](https://www.7-zip.org/)). It can zip *and* split into volumes in
one command, and — important — as long as you use `-tzip` (real zip format,
not 7z's own format), the split pieces are just raw byte-chunks, so they
reassemble on the Linux server with the exact same `cat` command used below.

Open **PowerShell** in the folder containing `bdd100k` and
`bdd100k_labels_release`, then (adjust the path to `7z.exe` if it's not on
your PATH — typically `C:\Program Files\7-Zip\7z.exe`):

```powershell
7z a -tzip -v500m "$env:USERPROFILE\Desktop\dataset.zip" bdd100k bdd100k_labels_release
```

This produces `dataset.zip.001`, `dataset.zip.002`, etc. on your Desktop, each ~500MB.

---

## Step 3 — Upload the chunks

In the Jupyter file browser, use the upload button (or drag-and-drop) to
upload each chunk file individually into your home directory on the server —
`dataset_part_aa`, `dataset_part_ab`, ... (macOS/Linux) or `dataset.zip.001`,
`dataset.zip.002`, ... (Windows/7-Zip). Uploading one at a time (rather than
all at once) makes it easier to retry just the failed one if a chunk stalls.

If an upload does stall, refresh the Jupyter browser tab to cancel it (that
kills the stuck network request without affecting your notebooks), delete
any partial file it left behind in the file browser, and retry that chunk.

---

## Step 4 — Reassemble and extract on the server

In the **Jupyter terminal** — run each command separately, one at a time,
waiting for the prompt to return before the next one.

```bash
cd ~
cat dataset_part_* > dataset.zip        # macOS/Linux split naming
# or, if you used 7-Zip on Windows:
cat dataset.zip.0* > dataset.zip
```

```bash
ls -la dataset.zip
```
(sanity check — size should roughly match the original)

```bash
unzip dataset.zip -d ~/bdd100k
```

---

## Step 5 — Verify nothing got corrupted

File count and size checks alone won't catch a byte-level corruption from
the split/reassemble process — spot-check actual image decoding too.

**Count real files** (excluding macOS `__MACOSX` junk):
```bash
find ~/bdd100k -type f -not -path "*__MACOSX*" | wc -l
```

**Check total size:**
```bash
du -sh ~/bdd100k
```

**Verify a random sample of images actually decode:**
```bash
python3 -c "
from PIL import Image
import glob, random
files = [f for f in glob.glob('$HOME/bdd100k/**/*.jpg', recursive=True) if '__MACOSX' not in f]
print('found', len(files), 'jpgs')
sample = random.sample(files, 20)
bad = 0
for f in sample:
    try:
        Image.open(f).verify()
    except Exception as e:
        print('CORRUPT:', f, e)
        bad += 1
print('checked 20, corrupt:', bad)
"
```
(`pip install Pillow` first if needed.)

**Verify label JSONs aren't truncated:**
```bash
find ~/bdd100k -name "*.json" -exec python3 -c "import json,sys; json.load(open(sys.argv[1])); print(sys.argv[1], 'OK')" {} \;
```
A parse error here means that JSON got cut off during upload — re-transfer it.

---

## Step 6 — Clean up

Free up disk space once you've confirmed the extraction is good:

```bash
rm -f dataset_part_* dataset.zip.0* dataset.zip
```

Optionally remove the macOS resource-fork junk from the extracted data (cosmetic only, doesn't affect training):
```bash
find ~/bdd100k -name "__MACOSX" -type d -exec rm -rf {} +
find ~/bdd100k -name "._*" -delete
```
