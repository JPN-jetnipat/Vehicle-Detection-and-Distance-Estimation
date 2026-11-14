"""
Check that the YOLOv5su day-only checkpoint was trained on the same daytime images
as the YOLO11s source-only baseline. Run from the baby folder:
    python DANN/check_day_split.py
"""
from pathlib import Path

BASE = Path(r"C:\Users\USER\OneDrive\เดสก์ท็อป\baby")
YOLO11S_DIR = BASE / "bdd100k_split_handoff" / "dataset" / "yolo" / "images" / "train_day"
V5SU_LIST = BASE / "YOLOv5s_DayOnly_Set3" / "splits_local" / "train_day_images.txt"

a = {p.name for p in YOLO11S_DIR.iterdir()}
b = {Path(l.strip()).name for l in V5SU_LIST.read_text(encoding="utf-8").splitlines() if l.strip()}

print(f"YOLO11s train_day : {len(a)}")
print(f"v5su train list   : {len(b)}")
print(f"in both           : {len(a & b)}")
print(f"only YOLO11s      : {len(a - b)}")
print(f"only v5su         : {len(b - a)}")
print("RESULT:", "SAME images" if a == b else "DIFFERENT images -> retrain arm A on bdd100k_day.yaml")
