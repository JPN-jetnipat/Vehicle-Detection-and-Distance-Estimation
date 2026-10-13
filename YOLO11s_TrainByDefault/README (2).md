# YOLO11s Training — BDD100K Vehicle Detection

**โฟลเดอร์:** `YOLO11s_TrainByDefault\`

เทรน `yolo11s.pt` บน BDD100K 5-class vehicle dataset ด้วย **default hyperparameters ของ Ultralytics ทั้งหมด** (ไม่แก้ค่าใดๆ)

---

## 1. Setup

| | |
|---|---|
| Base model | `yolo11s.pt` (pretrained COCO) |
| Dataset | BDD100K, 5 classes: car, truck, bus, motor, bike |
| Train set | 60,186 ภาพ (train pool ตาม `splits/train_100.txt`) |
| Val set | 1,542 ภาพ (2.5% stratified carve-out จาก train, ตาม `splits/val.txt`) |
| Hardware | RTX 3060 Ti (8GB VRAM), local Windows |

### Hyperparameters (default ทั้งหมด, ไม่แก้ไข)

| Param | ค่า |
|---|---|
| epochs | 100 |
| batch | 16 |
| imgsz | 640 |
| optimizer | auto |
| lr0 | 0.01 |
| lrf | 0.01 |
| momentum | 0.937 |
| weight_decay | 0.0005 |
| warmup_epochs | 3.0 |
| box / cls / dfl loss weight | 7.5 / 0.5 / 1.5 |
| augmentation | default (mosaic=1.0, hsv_h/s/v=0.015/0.7/0.4, fliplr=0.5, translate=0.1, scale=0.5) |

### ไฟล์ที่ใช้
- `prepare_train_yaml.py` — สร้าง absolute-path image list + data.yaml (เพราะ yaml เดิมของ repo มีปัญหา relative path ชนกับ global datasets folder ของ Ultralytics)
- `train_yolo11s_default.py` — รันเทรนจริง ไม่ override hyperparameter ใดๆ นอกจาก path ที่เก็บผล

### วิธีรัน
```powershell
cd "C:\Users\USER\OneDrive\เดสก์ท็อป\baby\YOLO11s_TrainByDefault"
python prepare_train_yaml.py
python train_yolo11s_default.py
```

**ผลลัพธ์เทรน:** `run1-2\weights\best.pt` (checkpoint ที่ mAP ดีที่สุด), `run1-2\results.csv`, กราฟต่างๆ

---

## 2. ผล Validation ตอนจบเทรน (val set, 1,542 ภาพ)

| Class | P | R | mAP50 | mAP50-95 |
|---|---|---|---|---|
| car | 0.763 | 0.761 | 0.800 | 0.488 |
| truck | 0.667 | 0.610 | 0.646 | 0.471 |
| bus | 0.720 | 0.553 | 0.639 | 0.489 |
| motor | 0.799 | 0.385 | 0.457 | 0.205 |
| bike | 0.621 | 0.504 | 0.546 | 0.264 |
| **all** | **0.714** | **0.562** | **0.617** | **0.383** |

---

## 3. ผล Test (แยกตาม time-of-day)

**โฟลเดอร์:** `YOLO11s_TrainByDefault\YOLO11s_TestTrained\` — เทส `best.pt` กับ TEST set (8,841 ภาพ, BDD official val, ไม่เคยใช้ระหว่างเทรนเลย) แยกตาม day/night/dawndusk

### ไฟล์ที่ใช้
- `prepare_test_lists.py` — สร้าง image list แยกตาม timeofday
- `eval_trained_model.py` — รัน eval, รวมผลเป็นตาราง + CSV

```powershell
cd "...\YOLO11s_TrainByDefault\YOLO11s_TestTrained"
python prepare_test_lists.py
python eval_trained_model.py
```

### test_all (8,841 ภาพ)

| Class | P | R | mAP50 | mAP50-95 |
|---|---|---|---|---|
| car | 0.759 | 0.753 | 0.792 | 0.483 |
| truck | 0.656 | 0.578 | 0.625 | 0.456 |
| bus | 0.690 | 0.523 | 0.604 | 0.457 |
| motor | 0.676 | 0.332 | 0.403 | 0.190 |
| bike | 0.538 | 0.440 | 0.448 | 0.219 |
| **all** | **0.664** | **0.525** | **0.574** | **0.361** |

### test_day (4,264 ภาพ)

| Class | P | R | mAP50 | mAP50-95 |
|---|---|---|---|---|
| car | 0.783 | 0.768 | 0.807 | 0.511 |
| truck | 0.658 | 0.577 | 0.626 | 0.458 |
| bus | 0.715 | 0.512 | 0.606 | 0.452 |
| motor | 0.657 | 0.324 | 0.405 | 0.198 |
| bike | 0.540 | 0.413 | 0.414 | 0.200 |
| **all** | **0.671** | **0.519** | **0.572** | **0.364** |

### test_night (3,863 ภาพ)

| Class | P | R | mAP50 | mAP50-95 |
|---|---|---|---|---|
| car | 0.711 | 0.738 | 0.768 | 0.439 |
| truck | 0.637 | 0.577 | 0.604 | 0.441 |
| bus | 0.616 | 0.522 | 0.571 | 0.443 |
| motor | 0.762 | 0.379 | 0.430 | 0.199 |
| bike | 0.505 | 0.500 | 0.496 | 0.257 |
| **all** | **0.646** | **0.543** | **0.574** | **0.356** |

### test_dawndusk (714 ภาพ)

| Class | P | R | mAP50 | mAP50-95 |
|---|---|---|---|---|
| car | 0.764 | 0.764 | 0.800 | 0.504 |
| truck | 0.674 | 0.597 | 0.664 | 0.486 |
| bus | 0.669 | 0.606 | 0.676 | 0.534 |
| motor | 0.598 | 0.270 | 0.342 | 0.112 |
| bike | 0.701 | 0.555 | 0.596 | 0.278 |
| **all** | **0.681** | **0.559** | **0.616** | **0.383** |

ผลลัพธ์เต็ม: `YOLO11s_TestTrained\yolo11s_trained_eval.csv`

---

## 4. ข้อสังเกต

- **car** ทำได้ดีที่สุดในทุกชุด (mAP50 0.77–0.81)
- **motor** แย่ที่สุดในทุกชุด (mAP50-95 0.11–0.21) — วัตถุขนาดเล็ก, จำนวน instance น้อยในเทรนเซ็ต
- **mAP50 ระหว่าง test_day (0.572) กับ test_night (0.574) แทบไม่ต่างกันเลย** — โมเดลเห็นภาพกลางคืนจริงระหว่างเทรน (train pool มีทั้ง day/night/dawndusk ปนกัน) จึงไม่มีช่องว่างสภาพแสงเหมือนตอนที่เทส `yolo11s.pt` แบบ pretrained COCO ล้วนๆ (ตอนนั้น gap ~23%)
- **test_dawndusk** (714 ภาพ, กลุ่มเล็กสุด) ได้ mAP50 สูงสุด (0.616) — sample size เล็ก ควรตีความอย่างระมัดระวัง
