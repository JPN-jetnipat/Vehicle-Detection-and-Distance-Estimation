# BDD100K test ground truth (COCO format)

Shared ground truth for scoring every model on the same BDD100K test pool.
Created with `python scripts/export_coco_gt.py`.

## Files

| File | Images | Boxes | motor | bike |
|---|---|---|---|---|
| test_all.json (day + night + dawndusk) | 8841 | 97193 | 377 | 739 |
| test_day.json | 4264 | 51930 | 262 | 494 |
| test_night.json | 3863 | 36751 | 93 | 186 |
| test_dawndusk.json | 714 | 8512 | 22 | 59 |
| test_clear.json | 5345 | 56667 | 200 | 391 |
| test_overcast.json | 1239 | 15505 | 72 | 141 |
| test_partlycloudy.json | 738 | 9256 | 65 | 69 |
| test_rainy.json | 737 | 7541 | 30 | 57 |
| test_snowy.json | 769 | 8080 | 9 | 81 |
| test_foggy.json | 13 | 144 | 1 | 0 |

Don't trust the AP from foggy (1 motor box, 0 bike) or snowy (9 motor boxes).
Report those splits with a caveat.

## Conventions (keep these the same, or scores won't match)

- **Class IDs are 0-based:** `0 car, 1 truck, 2 bus, 3 motor, 4 bike`
- **Images:** every image is 1280x720. `image_id` counts from 1 in file-name sort order. Map your predictions by `file_name`.
- **bbox:** `[x, y, w, h]` in pixels, where x, y is the top-left corner.

## Prediction file format (for the bootstrap)

Send a COCO results JSON file, one per model per split:

```json
[{"image_id": 1, "category_id": 3, "bbox": [x, y, w, h], "score": 0.87}, ...]
```

Use these inference settings for every model: `conf=0.001, iou=0.6, max_det=300, imgsz=640`.

Score with `scripts/eval_arm_coco.py`, which uses pycocotools COCOeval defaults.
