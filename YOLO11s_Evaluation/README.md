# YOLO11s Baseline Evaluation — BDD100K Vehicle Detection

สรุปสิ่งที่ทำในขั้นตอนนี้ของโปรเจกต์: เตรียม dataset, เทส `yolo11s.pt` (pretrained บน COCO, **ยังไม่ผ่านการเทรน** กับ BDD100K) เป็น baseline ก่อนเริ่มงาน DANN

---

## 1. เป้าหมาย

วัดประสิทธิภาพของ YOLO11s แบบ pretrained ล้วนๆ (ไม่ fine-tune) กับงานตรวจจับยานพาหนะบน BDD100K โดยเฉพาะดูช่องว่างของประสิทธิภาพระหว่างสภาพแสงกลางวัน / กลางคืน / พลบค่ำ เพื่อใช้เป็นตัวเลขอ้างอิง (reference point) เทียบกับโมเดล DANN ที่จะเทรนในขั้นต่อไป

**สำคัญ:** ขั้นตอนทั้งหมดในเอกสารนี้ **ไม่มีการเทรนโมเดลใดๆ ทั้งสิ้น** เป็นการวัดผล (evaluation) ล้วนๆ

---

## 2. โครงสร้าง dataset และ split

ใช้ split ที่มากับ `bdd100k_split_handoff` (seed=42, deterministic) ตรงกับ `splits/MANIFEST.json`:

| Split | จำนวนภาพ | ที่มา |
|---|---|---|
| train pool | 60,186 | 97.5% ของ BDD train (หลังตัด undefined) |
| **val** (dev set) | 1,542 | 2.5% stratified-by-timeofday carve-out จาก BDD train |
| **test** (final scoring) | 8,841 | BDD official val set ทั้งหมด (หลังตัด undefined 1,159 ภาพจาก 10,000) |

**Timeofday breakdown ของ test set:**

| Stratum | จำนวนภาพ |
|---|---|
| test_day (daytime) | 4,264 |
| test_night (night) | 3,863 |
| test_dawndusk (dawn/dusk) | 714 |

Classes (5 คลาส, ตัดจาก BDD ดั้งเดิม 10 คลาส): `car, truck, bus, motor, bike` — ตัด person/rider/traffic light/traffic sign/train ออก

### ขั้นตอนเตรียม dataset (ตาม README ของ handoff repo)
1. `build_attr_index.py` — สร้าง index timeofday/weather/scene ของทุกภาพ, ยืนยันจำนวนภาพตรง (train=69,863, val=10,000)
2. `bdd_to_yolo.py` — แปลง label จาก BDD JSON เป็น YOLO format (train + val)
3. `materialize_images.py` — จัดเรียงไฟล์ภาพให้ตรงกับ label (แก้จาก symlink เป็น hardlink เนื่องจาก Windows ไม่ให้สิทธิ์ symlink กับ user ทั่วไป)

ทำทั้งหมดบนเครื่อง local (Windows, RTX 3060 Ti) ที่ `C:\Users\USER\OneDrive\เดสก์ท็อป\baby\bdd100k_split_handoff`

---

## 3. ปัญหา class-id mismatch และวิธีแก้

`yolo11s.pt` แบบ pretrained เทรนมาจาก **COCO (80 คลาส)** ซึ่งใช้ index คนละชุดกับ label ของเรา:

| คลาสของเรา | id ของเรา | id ใน COCO |
|---|---|---|
| car | 0 | 2 |
| truck | 1 | 7 |
| bus | 2 | 5 |
| motor | 3 | 3 |
| bike | 4 | 1 |

ถ้า eval ตรงๆ โดยไม่ remap, ระบบจะเทียบ id ผิดชุดกัน (เช่น โมเดลทายถูกว่าเป็นรถแต่ output id=2 ซึ่ง label ของเราตีความว่าเป็น "bus") ทำให้ mAP ที่ได้ไม่มีความหมาย

**วิธีแก้:** เขียนสคริปต์ `build_coco_eval_set.py` สร้างชุด eval คู่ขนาน (`dataset_yolo_coco/`) ที่:
- Hardlink ภาพเดิม (ไม่ duplicate ข้อมูล)
- Remap label id ของเราให้ตรงกับ COCO id ตามตารางด้านบน
- ไม่แตะ dataset เดิม (`dataset/yolo/`) เลย — ยังใช้เทรน DANN ต่อได้ปกติ

ไฟล์ทั้งหมดของงาน eval นี้ (สคริปต์, yaml config, output) เก็บรวมไว้ในโฟลเดอร์ `YOLO11s_Evaluation/` แยกต่างหาก ไม่ปนกับ repo หลัก

### เครื่องมือที่ใช้
- `build_coco_eval_set.py` — สร้าง eval set + remap class id
- `eval_yolo11s_baseline.py` — รัน `model.val()` ทั้ง 5 ชุด แล้วรวมผลเป็นตารางเดียว
- `bdd100k_coco80.yaml` — data config อ้างอิง COCO 80 คลาสเต็ม

---

## 4. ผลการทดสอบ (yolo11s.pt, ไม่ fine-tune)

รันบน RTX 3060 Ti, CUDA 12.1, Ultralytics 8.4.75

### 4.1 ภาพรวม (all classes, mAP50 / mAP50-95)

| Split | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| val (dev set) | 0.523 | 0.359 | 0.363 | 0.224 |
| test_all | 0.505 | 0.358 | 0.357 | 0.219 |
| **test_day** | **0.542** | **0.379** | **0.383** | **0.240** |
| **test_night** | **0.401** | **0.319** | **0.295** | **0.173** |
| test_dawndusk | 0.515 | 0.359 | 0.377 | 0.231 |

**ข้อสังเกตหลัก:** กลางคืน (test_night) แย่กว่ากลางวัน (test_day) ชัดเจนมาก — mAP50 ตกจาก 0.383 → 0.295 คิดเป็น **ลดลง ~23%** และ mAP50-95 ตกจาก 0.240 → 0.173 (**ลดลง ~28%**) นี่คือช่องว่างประสิทธิภาพที่งาน DANN ตั้งใจจะลดลง

### 4.2 แยกรายคลาส

**car**

| Split | P | R | mAP50 | mAP50-95 |
|---|---|---|---|---|
| val | 0.754 | 0.476 | 0.570 | 0.332 |
| test_all | 0.752 | 0.468 | 0.561 | 0.327 |
| test_day | 0.784 | 0.534 | 0.618 | 0.378 |
| test_night | 0.636 | 0.391 | 0.466 | 0.243 |
| test_dawndusk | 0.821 | 0.491 | 0.600 | 0.368 |

**truck**

| Split | P | R | mAP50 | mAP50-95 |
|---|---|---|---|---|
| val | 0.305 | 0.444 | 0.338 | 0.251 |
| test_all | 0.311 | 0.434 | 0.328 | 0.244 |
| test_day | 0.331 | 0.460 | 0.354 | 0.265 |
| test_night | 0.226 | 0.344 | 0.223 | 0.162 |
| test_dawndusk | 0.353 | 0.414 | 0.356 | 0.267 |

**bus**

| Split | P | R | mAP50 | mAP50-95 |
|---|---|---|---|---|
| val | 0.484 | 0.422 | 0.404 | 0.303 |
| test_all | 0.426 | 0.380 | 0.355 | 0.275 |
| test_day | 0.476 | 0.374 | 0.366 | 0.285 |
| test_night | 0.298 | 0.414 | 0.331 | 0.252 |
| test_dawndusk | 0.479 | 0.386 | 0.383 | 0.304 |

**motor (motorcycle)**

| Split | P | R | mAP50 | mAP50-95 |
|---|---|---|---|---|
| val | 0.347 | 0.200 | 0.188 | 0.087 |
| test_all | 0.451 | 0.260 | 0.251 | 0.116 |
| test_day | 0.499 | 0.298 | 0.291 | 0.138 |
| test_night | 0.327 | 0.172 | 0.191 | 0.087 |
| test_dawndusk | 0.327 | 0.136 | 0.148 | 0.038 |

**bike (bicycle)**

| Split | P | R | mAP50 | mAP50-95 |
|---|---|---|---|---|
| val | 0.724 | 0.252 | 0.317 | 0.145 |
| test_all | 0.584 | 0.248 | 0.288 | 0.134 |
| test_day | 0.622 | 0.229 | 0.286 | 0.134 |
| test_night | 0.516 | 0.274 | 0.264 | 0.122 |
| test_dawndusk | 0.593 | 0.371 | 0.398 | 0.178 |

### 4.3 สรุปเชิงคุณภาพ

- **car** คือคลาสที่โมเดลทำได้ดีที่สุดในทุกชุด (มีจำนวน instance เยอะที่สุดในเทรนของ COCO ด้วย) แต่ยัง drop ชัดเจนตอนกลางคืน (mAP50-95: 0.378 → 0.243)
- **motor** คือคลาสที่แย่ที่สุดโดยรวม (mAP50-95 อยู่แค่ 0.04–0.14 ทุกชุด) — เป็นวัตถุขนาดเล็ก ตรวจจับยากอยู่แล้วแม้กลางวัน
- **truck** และ **bus** มี pattern คล้ายกัน คือกลางคืนแย่กว่ากลางวันชัดเจนแต่ไม่มากเท่า car
- **dawndusk** (714 ภาพ, กลุ่มเล็กสุด) บางครั้งได้ค่าดีกว่า test_day ในบางคลาส/metric (เช่น bike) แต่ต้องระวังการตีความเพราะ sample size เล็ก ผลจึง noise สูงกว่า

---

## 5. ไฟล์ที่เกี่ยวข้อง

ทั้งหมดอยู่ในโฟลเดอร์ `YOLO11s_Evaluation/`:

| ไฟล์ | หน้าที่ |
|---|---|
| `build_coco_eval_set.py` | สร้าง eval set + remap class id เป็น COCO |
| `eval_yolo11s_baseline.py` | รัน eval ทั้ง 5 ชุด, print แยกทีละชุด, รวม CSV |
| `bdd100k_coco80.yaml` | data config อ้างอิง COCO 80 คลาส |
| `YOLO11s_Test.ipynb` | notebook รวม pipeline ทั้งหมด |
| `dataset_yolo_coco/` | eval set ที่ remap แล้ว (auto-generated) |
| `splits_eval/` | image-list txt ของแต่ละชุด (auto-generated) |
| `yolo11s_baseline_eval.csv` | ผลลัพธ์ตารางรวม (output หลัก) |

---

## 6. ขั้นต่อไป

ใช้ตัวเลขในเอกสารนี้เป็น **baseline (ก่อนเทรน)** เทียบกับผลของ DANN v3 หลังเทรนเสร็จ — โดยเฉพาะดูว่า DANN ช่วยลดช่องว่าง day/night ของ mAP50 ได้มากน้อยแค่ไหนเทียบกับ 23% ที่เห็นในโมเดล baseline นี้
