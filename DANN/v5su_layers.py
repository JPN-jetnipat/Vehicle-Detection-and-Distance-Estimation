"""
Patch the DANN layer map for YOLOv5su without editing dann_core.py.
Import this module before train_dann / dann_trainer / check_dann_setup.

YOLOv5su backbone: layers 0-9, all sequential.
  4 = C3   (stride 8)  -> local head, same index as YOLO11s
  9 = SPPF (stride 32) -> global head, last backbone layer (YOLO11s uses 10, C2PSA)
"""
import dann_core

dann_core.LOCAL_LAYER = 4
dann_core.GLOBAL_LAYER = 9
dann_core.BACKBONE_END = 9

V5SU_WEIGHTS = ("C:/Users/USER/OneDrive/เดสก์ท็อป/baby/YOLOv5s_DayOnly_Set3/"
                "set3_combined_v5_day/weights/best.pt")

print("DANN layer map : yolov5su (local=4, global=9, backbone_end=9)")
