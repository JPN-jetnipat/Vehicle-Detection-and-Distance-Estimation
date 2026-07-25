# Vehicle Detection and Distance Estimation — Project Summary

Monocular vehicle detection + distance-to-camera estimation pipeline. A YOLOv5 detector
locates vehicles in a single RGB frame; a pinhole-camera geometry module converts each
detected bounding box into an estimated real-world distance. Built and trained locally on
Windows with an NVIDIA GPU (CUDA confirmed via `torch.cuda`), Python virtualenv (`.venv`).

## 1. Pipeline architecture

```
image ──> YOLOv5s detector ──> bounding boxes + class ──> pinhole distance model ──> distance (m)
                                                              (needs focal length,
                                                               known real-world object width)
```

- **Detection**: `yolov5/` — a vendored copy of the Ultralytics YOLOv5 repo (`train.py`,
  `val.py`, `detect.py`), used as-is (no custom model architecture changes found).
- **Distance estimation**: `geometry/distance_estimator.py` implements the classic pinhole
  formula `D = f * W / w`, where `f` = focal length in pixels, `W` = assumed real-world
  object width in meters, `w` = detected bounding-box width in pixels.
  Assumed real widths per class: `Car: 1.8m`, `Van: 2.0m`, `Truck: 2.5m`.
- **Calibration**: `geometry/calibration.py` parses KITTI calibration files (`P0–P3`,
  `R0_rect`, `Tr_velo_to_cam`) and extracts `fx` from the `P2` projection matrix. This only
  works for KITTI images, which ship per-image calibration files — BDD100K has none, so the
  distance module currently only has a real calibration source for the KITTI split.
- **Evaluation**: `evaluation/metrics.py` — bare `MAE`/`RMSE` functions for eventual
  distance-accuracy evaluation. No ground-truth-distance evaluation has been run yet (would
  need KITTI 3D box depth or LiDAR-derived ground truth to compare against).

## 2. Datasets

Three public autonomous-driving datasets were used, all converted to YOLO-format labels
(normalized `class cx cy w h` per line).

### KITTI (object detection benchmark)
- Source: `dataset/raw/KITTI/` (`data_object_image_2`, `data_object_label_2`,
  `data_object_calib`).
- Native classes (8): `Car, Van, Truck, Pedestrian, Person_sitting, Cyclist, Tram, Misc`
  (`DontCare` dropped during conversion).
- Converted via `dataset/convert_kitti_to_yolo.py` (parses KITTI's 15-column label format,
  converts `x1,y1,x2,y2` → normalized YOLO xywh).
- Split (`dataset/kitti.yaml`, `path: ../dataset`): **train 5,236 / val 1,122 / test 1,123**
  images.
- Note: `dataset/kitti.yaml` has an `include: [Car, Truck, Van]` key expressing intent to
  train on only 3 of the 8 classes, but this key is **not** consumed by stock YOLOv5
  (`utils/general.py::check_dataset`), and the conversion script keeps all 8 classes. So the
  actual trained model has **8 output classes**, 5 of which (Pedestrian, Person_sitting,
  Cyclist, Tram, Misc) are present but not the intended focus — worth being upfront about
  when quoting the headline KITTI mAP number.

### BDD100K (full, 100K images)
- Source: `dataset/raw/BDD100K/` — images (`images/100k/{train,val}`) + label JSONs
  (`bdd100k_labels_images_{train,val}.json`).
- Full size: **70,000 train / 10,000 val** images (720p, 1280×720).
- Converted via `dataset/convert_bdd100k_to_yolo.py`: only 3 BDD100K categories are kept —
  `car → 0 (Car)`, `bus → 1 (Van)`, `truck → 2 (Truck)` (BDD's "bus" is deliberately remapped
  onto the KITTI "Van" id so the class ids line up across datasets). Pedestrians/cyclists/etc.
  from BDD100K are discarded entirely.
- Config: `dataset/bdd100k_vehicle.yaml`.
- **Known issue**: the training run against the full 100K set (see §3) was launched with
  `--data ../dataset/dataset/bdd100k_vehicle.yaml` (double `dataset/dataset`), a path that
  does not exist in this repo. That run's very low, flat-from-epoch-0 metrics are consistent
  with the data config not resolving correctly — treat that particular result as invalid
  rather than a real "full-BDD100K" benchmark.

### BDD10K (custom subset of BDD100K)
- Built by `dataset/create_bdd10k_subset.py`: a fixed random sample (`seed=42`) of the full
  BDD100K train/val split — **7,000 train / 1,000 val** requested, **6,988 train / 1,000 val**
  actually copied (some images had no matching label after conversion).
- Config: `dataset/bdd10k_custom.yaml`. Same 3-class mapping as above (Car/Van/Truck only).
- Rationale: full 100K-image BDD100K training is slow locally; this subset was used to
  iterate faster while still being much more diverse than KITTI (multiple US cities, weather,
  time of day).

### BDD10K day / night split
- Built by `scripts/split_day_night.py`, which reads the `attributes.timeofday` field from
  the original BDD100K label JSON and splits the BDD10K subset accordingly.
- **Day**: `dataset/bdd10k_day/` — 3,675 train / 509 val images (`bdd10k_day.yaml`).
- **Night**: `dataset/bdd10k_night/` — 2,819 train / 418 val images (`bdd10k_night.yaml`),
  prepared but **no completed training run exists yet** for the night-only split (goal is
  presumably to compare day-only vs. night-only vs. mixed detection performance).

## 3. Model & training setup

- **Model**: YOLOv5s (small), pretrained COCO weights (`yolov5s.pt`) as the starting
  checkpoint for every run (standard transfer-learning fine-tune, not trained from scratch).
- **Framework versions**: `torch==2.12.0`, `torchvision==0.27.0`, `ultralytics==8.4.65`
  (see `requirements.txt`); YOLOv5 code itself is the vendored repo under `yolov5/`.
- **Common hyperparameters** (YOLOv5 defaults, unchanged): `lr0=0.01`, `lrf=0.01`,
  `momentum=0.937`, `weight_decay=5e-4`, SGD optimizer, `imgsz=640`, mosaic augmentation
  on, standard HSV/translate/scale/flip augmentation, no mixup/copy-paste.
- **Image size**: 640×640 for all runs. **Batch size**: 8. **Workers**: 0–2.

### Training runs and results

All numbers are the final-epoch validation metrics from each run's `results.csv`
(precision/recall/mAP are for the YOLO detection task, not distance accuracy — no
distance-error evaluation has been run yet).

| Run (`runs/train/…`) | Dataset | Train/Val imgs | Epochs | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|---|---|---|---|---|
| `kitti_vehicle_100e2` | KITTI (8 classes, see note above) | 5,236 / 1,122 | 100 | 0.971 | 0.943 | **0.980** | 0.820 |
| `bdd10k_vehicle_100e` | BDD10K custom subset, day+night mixed (3 classes) | 6,988 / 1,000 | 100 | 0.725 | 0.489 | 0.542 | 0.366 |
| `bdd10k_day_only` | BDD10K day-only subset (3 classes) | 3,675 / 509 | 50 | 0.736 | 0.503 | 0.560 | 0.377 |
| `bdd100k_vehicle_100e_bs8` | Full BDD100K (3 classes) | 70,000 / 10,000 | 100 | 0.125 | 0.153 | 0.107 | 0.065 | ⚠️ bad `--data` path (see §2) |
| `kitti_smoketest`, `gpu_test*`, `sanity_bdd100k*` | various | — | 1–few | — | — | — | — | early smoke tests, not real results |

Interpretation to be upfront about when asking others for advice:
- **KITTI is the strongest result by far (mAP@0.5 ≈ 0.98)**, but KITTI is a small,
  geographically narrow, daylight-biased German-city dataset with forward-facing highway/
  street scenes — it's the "easy mode" benchmark here, not evidence the detector generalizes.
- **BDD10K day+night and day-only sit around mAP@0.5 ≈ 0.54–0.56**, with recall notably
  lower than precision (~0.49–0.50 recall vs. ~0.73 precision) — the model is conservative,
  missing a meaningful fraction of vehicles rather than producing lots of false positives.
  This is the more realistic real-world number given BDD100K's diversity (weather, dusk/dawn,
  varied camera mounts).
- The full-BDD100K run is not trustworthy as-is due to the path bug above and should be
  re-run before being cited anywhere.
- No run has yet isolated night-only performance despite the dataset being prepared.

## 4. Distance estimation approach

Purely geometric, not learned: `distance = focal_length_px * real_world_width_m /
bbox_width_px`, using per-class fixed real-world widths (Car 1.8 m, Van 2.0 m, Truck 2.5 m)
and a focal length pulled from KITTI's per-image `P2` calibration matrix. This is the classic
single-camera "similar triangles" approach (same idea as monocular ADAS distance estimators),
with known limitations worth mentioning if asking for feedback:
- Assumes a fixed canonical object width per class, which doesn't account for viewing angle
  (a car photographed at an angle has a foreshortened bounding-box width, so the estimate
  breaks down outside near-frontal/near-rear views).
- No BDD100K calibration data exists, so distance estimation is only actually validated
  against KITTI's known focal length; BDD100K frames would need an assumed/estimated focal
  length instead of a measured one.
- `evaluation/metrics.py` (MAE/RMSE) is implemented but not yet wired up to any ground-truth
  distance comparison — there's no reported distance-accuracy number yet, only detection mAP.

## 5. What's not done yet (from `docs/NEXT_STEPS.txt` + repo state)

1. Fix the double `dataset/dataset/...` path bug and re-run the full-BDD100K training.
2. Train + evaluate the night-only BDD10K split (data is ready, run is missing).
3. Decide/implement the intended KITTI 3-class filtering (`include:` key in `kitti.yaml` is
   currently a no-op).
4. Wire up end-to-end distance evaluation (MAE/RMSE against real KITTI 3D/depth ground truth).
5. `geometry/calibration.py` only supports KITTI-style calibration files — no path yet for
   estimating/assuming focal length on cameras without calibration files (e.g. BDD100K, or a
   live camera feed).
