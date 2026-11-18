# External test sets (ExDark, RTTS, DAWN, UA-DETRAC): how to reproduce our setup

Use this to score your own models on the same three external datasets with exactly the same
images, labels and scoring rules as our results (`results/exdark*.txt`, `results/rtts.txt`,
`results/dawn.txt`).

**Role of these datasets: test only (zero-shot).** No image is used for training, fine-tuning,
tuning or picking a checkpoint, so every usable image is test data. There is no
train/val/test split. Do not fine-tune on them, or the numbers stop being comparable.

---

## 1. What you need

| item | value |
|---|---|
| repo / branch | `JPN-jetnipat/Vehicle-Detection-and-Distance-Estimation`, branch `field-yolov11`, at or after commit `2a9f1f3` |
| Python env | `requirements.txt`: Python 3.11, torch 2.6.0, torchvision 0.21.0, ultralytics 8.4.120, pycocotools 2.0.11 |
| extra (download only) | `pip install gdown` (for ExDark; not in `requirements.txt`) |
| disk | about 3.5 GB free while unzipping (ExDark ~1.5 GB, RTTS ~1 GB, DAWN ~0.14 GB) |
| your models | Ultralytics `.pt` files trained on BDD100K with our 5 classes in this order: `0 car, 1 truck, 2 bus, 3 motor, 4 bike` |

**Code to use (all in the repo):**

| file | what it does |
|---|---|
| `tools/exdark_to_coco.py` | ExDark raw → COCO ground truth + slice lists |
| `tools/rtts_to_coco.py` | RTTS raw → COCO ground truth + slice lists |
| `tools/dawn_to_coco.py` | DAWN raw → COCO ground truth + slice lists |
| `evaluation/score_exdark.py` | scores any of the three (pass `--gt / --images-dir / --slices-dir`) |
| `evaluation/score_slices.py` | must be present: `score_exdark.py` imports its scoring constants from it |
| `splits/exdark/`, `splits/rtts/`, `splits/dawn/` | our image lists (committed). Your converter output must match them exactly, see §5 |

Raw datasets go **outside** the repo (we use `~/datasets/<NAME>/`). The converters write
`dataset/<name>/` inside the repo (gitignored: symlinks to the raw images, `gt_vehicle5.json`,
`meta.csv`) and regenerate `splits/<name>/`.

---

## 2. Scoring rules (same for all three, fixed, do not change)

- Inference at `imgsz 640`, `conf 0.001`, NMS IoU `0.6`, `max_det 300` (constants imported from
  `evaluation/score_slices.py`, the same as our BDD test scoring).
- pycocotools COCOeval with default `maxDets [1, 10, 100]`. We report mAP50, mAP75 and
  mAP50-95: all classes (plain average over classes) and per class.
- Ground truth uses each image's **real width/height** (these datasets vary in size).
- Classes are mapped to the BDD ids above; any other class (person, dog, train, …) is not scored.
- **Truck handling** (`--truck`):
  - `car` (default) for **ExDark and RTTS** (no truck class): truck predictions are relabelled
    car, then car boxes are re-NMS'd at IoU 0.6.
  - `keep` for **DAWN** (has real truck labels): predictions are scored as-is.
  - The scorer refuses any mode except `keep` when the ground truth contains truck boxes.
- Reporting guardrails: a slice with fewer than 100 images is marked "DO NOT REPORT"; a class
  with fewer than 800 boxes in a slice is indicative only.
- **Score every model you want to compare in ONE run** (repeat `--arm`), and include a BASE
  model in that run, so all rows share one protocol.

---

## 3. Per dataset: download, exclusions, convert

### 3.1 ExDark (real low-light photos)
Source: https://github.com/cs-chan/Exclusively-Dark-Image-Dataset · cite Loh & Chan, CVIU 2019 ·
non-commercial research use.

```bash
mkdir -p ~/datasets/ExDark && cd ~/datasets/ExDark
git clone --depth 1 https://github.com/cs-chan/Exclusively-Dark-Image-Dataset.git repo
pip install gdown
# gdown >= 6 takes bare file IDs (no --fuzzy); this pulls the image zip and the annotation zip
for id in $(grep -rhoE "drive\.google\.com/file/d/[^/]+" repo | sort -u | sed 's#.*/d/##'); do gdown "$id"; done
for z in *.zip; do unzip -q "$z"; done
cp repo/Groundtruth/imageclasslist.txt .
rm -rf *.zip __MACOSX
# expected layout: ~/datasets/ExDark/{ExDark/<12 class folders>, ExDark_Annno/, imageclasslist.txt}

cd ~/vehicle-detection && source .venv/bin/activate
python tools/exdark_to_coco.py --raw ~/datasets/ExDark
```

- **Class mapping:** Car→car, Bus→bus, Motorbike→motor, Bicycle→bike. No truck. The other 8
  classes are ignored.
- **Excluded: 1 image** (OpenCV's decoded size differs from the raw size because of EXIF
  rotation, so its boxes would not line up). See it with
  `grep exif_rotated dataset/exdark/meta.csv`.
- **Kept:** images with no vehicle (7,362 − 2,587 = 4,775). They catch false alarms.
- **Expected converter output:**
  `status: {'ok': 7362, 'exif_rotated': 1}` ·
  `vehicle instances: car 2927, truck 0, bus 706, motor 1072, bike 1120` ·
  `boxes clipped to image bounds: 11` · slices `all 7362, has_vehicle 2587, …, official_test 2563`.
- **Slices:** `all`, `has_vehicle`, `indoor`, `outdoor`, `outdoor_vehicle`, 10 × `light_*`,
  `official_train/val/test` (ExDark's own split, kept only as slices).
  Too small to report: `light_shadow` (56 images); `light_low`, `light_screen`, `light_window`
  and `indoor` have very few vehicle boxes.

### 3.2 RTTS (real fog / haze)
Source: RESIDE-β, https://sites.google.com/view/reside-dehaze-datasets/reside-β · VOC 2007 format.

```bash
mkdir -p ~/datasets/RTTS && cd ~/datasets/RTTS
# the bit.ly link on the RESIDE page (https://bit.ly/3c4gl3z) points to a UTexas Box share;
# this is that share's direct-download address:
curl -L -o RTTS.zip "https://utexas.box.com/shared/static/2yekra41udg9rgyzi3ysi513cps621qz"
file RTTS.zip            # must say "Zip archive" (~1 GB)
unzip -q RTTS.zip && rm RTTS.zip
# expected layout: ~/datasets/RTTS/RTTS/{Annotations, JPEGImages, ImageSets/Main}

cd ~/vehicle-detection && source .venv/bin/activate
python tools/rtts_to_coco.py --raw ~/datasets/RTTS/RTTS
```

- **Class mapping:** car→car, bus→bus, motorbike→motor, bicycle→bike. No truck. person ignored.
- **Excluded images: none.**
- **VOC "difficult" boxes are kept but ignored by the scoring** (stored with `iscrowd=1`): missing
  one is not a miss, detecting one is not a false alarm. That's the VOC convention. There are
  8,187 of them (~27% of vehicle boxes).
- VOC coordinates are 1-based; the converter shifts them to 0-based.
- **Expected converter output:** `status: {'ok': 4322}` ·
  `vehicle instances (not difficult): car 18414, truck 0, bus 1838, motor 862, bike 534` ·
  `difficult vehicle boxes: 8187` · `person: 11366` · `clipped: 4` ·
  `slices: all 4322, has_vehicle 3260`.

### 3.3 DAWN (fog, rain, sandstorm, snow)
Source: https://data.mendeley.com/datasets/766ygrbt8y/3 · CC BY-NC 3.0.

```bash
mkdir -p ~/datasets/DAWN && cd ~/datasets/DAWN
wget -O DAWN.zip "https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/766ygrbt8y-3.zip"
file DAWN.zip            # must say "Zip archive" (~132 MB)
unzip -q DAWN.zip && rm DAWN.zip
for z in $(find . -name "*.zip"); do unzip -q -o "$z" -d "$(dirname "$z")" && rm "$z"; done   # 4 nested zips
# expected layout: ~/datasets/DAWN/DAWN/{Fog,Rain,Sand,Snow}/ each with images,
#                  <Cond>_PASCAL_VOC/ (used) and <Cond>_YOLO_darknet/ (cross-check only)

cd ~/vehicle-detection && source .venv/bin/activate
python tools/dawn_to_coco.py --raw ~/datasets/DAWN/DAWN
```

- **Class mapping:** car, truck, bus, motorcycle→motor, bicycle→bike. **Has truck: all 5
  classes scored.** person (477 boxes) and 1 "train" box are not scored.
- **Excluded images: none.**
- The converter checks the XML pixel convention against the YOLO txt files. It should print
  `as 0-based 0.00 · as 1-based 2.00 -> using 0-based`.
- Flat image names are `<cond>__<file>` (no collisions across conditions).
- **Expected converter output:** `status: {'ok': 1027}` ·
  `vehicle instances: car 6454, truck 646, bus 161, motor 81, bike 26` ·
  `slices: all 1027, has_vehicle 1026, fog 300, rain 200, sand 323, snow 204`.
- Only car has 800+ boxes; read car, not the all-class mean (rain/snow have 0–4 motor/bike boxes).

### 3.4 UA-DETRAC (fixed roadside traffic cameras: sunny, cloudy, rainy, night)
Source: author page https://sites.google.com/view/daweidu/projects/ua-detrac (Google Drive).
Cite Wen et al., CVIU 2020. We use the **40 test sequences, every 10th frame** (25 fps video,
neighbouring frames are near-duplicates; also keeps COCOeval within the server's RAM).

```bash
mkdir -p ~/datasets/UA-DETRAC && cd ~/datasets/UA-DETRAC
gdown 1fQC6CEWOeL9pnJ4ZdD7U2gromcnWDRgm      # DETRAC-Images.zip, 9.85 GB, all 100 sequences flat
gdown --folder https://drive.google.com/drive/folders/1am_x3ywLJnABq4REtzbi0XTu5csFrp8v   # DETRAC-Annos/
unzip -tq DETRAC-Images.zip | tail -1        # "No errors detected"
cd DETRAC-Annos && unzip -q DETRAC-Test-Annotations-XML.zip && cd ..   # test XMLs only (40)
python -I ~/scripts/detrac_extract.py DETRAC-Images.zip DETRAC-Annos test_every10 10
#   expect: test seqs in XML: 40 | found in zip: 40 | images: 5640 | MISSING: none (416 MB)
rm DETRAC-Images.zip

cd ~/vehicle-detection && source .venv/bin/activate
python tools/detrac_to_coco.py --images ~/datasets/UA-DETRAC/test_every10 \
    --xml-dir ~/datasets/UA-DETRAC/DETRAC-Annos
```
`~/scripts/detrac_extract.py` keeps frames 1, 11, 21, … of the sequences that have a test XML
(a 20-line zipfile script; copy in §6).

- **Class mapping:** car + van → car, bus → bus. `others` (mostly trucks) and each sequence's
  ignored regions become `iscrowd=1` boxes for car and bus, so detections there count neither as
  TP nor FP. No truck/motor/bike labels: score with **`--truck car`**; motor/bike AP are n/a.
- **Excluded:** frames the XML does not list (no labelled vehicles, or not annotated; cannot tell
  which), status `no_annotation` in `meta.csv`.
- **Slices:** `all`, `has_vehicle`, `sunny` (8 seq), `cloudy` (11), `rainy` (9), `night` (12),
  `day` (= sunny + cloudy + rainy). Day and night come from the same kind of cameras, so the scorer
  prints a **car night/day mAP75 ratio** (compare with BDD's 0.81).
- Absolute numbers are not comparable with papers that train on UA-DETRAC (they use all frames,
  their own class sets and often the DETRAC protocol, not COCOeval).

---

## 4. Score

Run on a GPU, in tmux. Replace the `--arm` lines with your models, and keep our BASE in the
same run as the reference.

```bash
cd ~/vehicle-detection && source .venv/bin/activate

# ExDark (truck -> car is the default)
python evaluation/score_exdark.py \
  --gt dataset/exdark/gt_vehicle5.json --images-dir dataset/exdark/images --slices-dir splits/exdark \
  --arm base_100=weights/external/base_100_friend.pt \
  --arm my_model=path/to/my_model.pt \
  --pair my_model:base_100 \
  --device 0 --out results/exdark_mine.csv 2>&1 | tee results/exdark_mine.txt

# RTTS (truck -> car is the default)
python evaluation/score_exdark.py \
  --gt dataset/rtts/gt_vehicle5.json --images-dir dataset/rtts/images --slices-dir splits/rtts \
  --arm base_100=weights/external/base_100_friend.pt \
  --arm my_model=path/to/my_model.pt \
  --pair my_model:base_100 \
  --device 0 --out results/rtts_mine.csv 2>&1 | tee results/rtts_mine.txt

# DAWN (has real truck labels: --truck keep is required)
python evaluation/score_exdark.py \
  --gt dataset/dawn/gt_vehicle5.json --images-dir dataset/dawn/images --slices-dir splits/dawn \
  --truck keep \
  --arm base_100=weights/external/base_100_friend.pt \
  --arm my_model=path/to/my_model.pt \
  --pair my_model:base_100 \
  --device 0 --out results/dawn_mine.csv 2>&1 | tee results/dawn_mine.txt
```

The first output line shows the pool and the truck mode, for example
`eval pool (dataset/dawn/gt_vehicle5.json): 1027 images, 7368 scored vehicle boxes (+0 ignored), truck=keep`.

---

## 5. Check that your setup matches ours

**a) Same images.** After converting, the regenerated lists must be identical to the committed ones:
```bash
git status splits/exdark splits/rtts splits/dawn splits/detrac     # must show no changes
```

**b) Same scores.** Scoring our BASE weights must reproduce these `all`-slice numbers exactly:

| dataset | model | mAP50 | mAP75 | mAP50-95 | car mAP50 | car mAP75 |
|---|---|---|---|---|---|---|
| ExDark | `base_100_friend.pt` (v11) | 0.50995 | 0.29809 | 0.29312 | 0.73497 | 0.46162 |
| ExDark | `v5_base_100_friend.pt` | 0.50922 | 0.28479 | 0.28246 | 0.73506 | 0.44619 |
| RTTS | `base_100_friend.pt` (v11) | 0.45774 | 0.30658 | 0.28391 | 0.77559 | 0.56574 |
| RTTS | `v5_base_100_friend.pt` | 0.44676 | 0.29376 | 0.27672 | 0.76812 | 0.56177 |
| DAWN | `base_100_friend.pt` (v11) | 0.40822 | 0.28702 | 0.26847 | 0.78256 | 0.55335 |
| DAWN | `v5_base_100_friend.pt` | 0.38339 | 0.26748 | 0.24890 | 0.77343 | 0.54362 |

If (a) passes but (b) differs, check the library versions against `requirements.txt` first.

---

## 6. Download gotchas we hit

- **ExDark:** `gdown --fuzzy` fails on gdown 6.x. Pass the bare file ID, as above.
- **RTTS:** the bit.ly link lands on a Box web page, not the zip; `?dl=1` does not work on Box.
  Use the `shared/static/` address above. Always run `file *.zip` before unzipping.
- **DAWN:** the zip contains 4 more zips (one per condition); unzip those too.
- If a download is refused, download on a laptop and upload it (JupyterLab upload button), then
  continue from the unzip step.
- **UA-DETRAC:** the 9.85 GB images zip has no Train/Test folders (all 100 sequences flat); the
  test XML file names decide which 40 are test. If Drive says "too many users", put a browser
  `cookies.txt` at `~/.cache/gdown/cookies.txt`. A full disk mid-download leaves a `.part` file;
  `gdown -c <id>` resumes it. Extraction script used for the every-10th-frame subset:

```python
import re, sys, zipfile, pathlib
zip_path, xml_dir, out_dir, step = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
test_seqs = {p.stem for p in pathlib.Path(xml_dir).rglob('MVI_*.xml')}
pat = re.compile(r'^DETRAC-Images/(MVI_\d+)/img(\d+)\.jpg$')
out, per = pathlib.Path(out_dir), {}
with zipfile.ZipFile(zip_path) as z:
    for name in z.namelist():
        m = pat.match(name)
        if not m or m.group(1) not in test_seqs or (int(m.group(2)) - 1) % step:
            continue
        dst = out / m.group(1) / pathlib.PurePosixPath(name).name
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(z.read(name))
        per[m.group(1)] = per.get(m.group(1), 0) + 1
print(f'test seqs in XML: {len(test_seqs)} | found in zip: {len(per)} | images: {sum(per.values())}')
miss = test_seqs - per.keys()
print('MISSING:', sorted(miss) if miss else 'none')
```
