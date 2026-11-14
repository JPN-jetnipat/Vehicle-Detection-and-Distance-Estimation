"""
Same DANN arms as train_dann.py, on the YOLOv5su day-only checkpoint.
Hyperparameters, source data and target images come from train_dann.py unchanged;
only the starting weights, the layer map and the run names differ.

Usage (from the DANN folder):
    python train_dann_v5su.py --arm noadv
    python train_dann_v5su.py --arm dann
"""
import v5su_layers                     # must come first: patches dann_core
import train_dann as td

td.BASELINE_WEIGHTS = v5su_layers.V5SU_WEIGHTS
td.COMMON["model"] = v5su_layers.V5SU_WEIGHTS
td.ARMS["noadv"]["name"] = "finetune_noadv_v5su"
td.ARMS["dann"]["name"] = "dann_v3_v5su"

print("source weights :", td.COMMON["model"])
print("source data    :", td.COMMON["data"])

if __name__ == "__main__":
    td.main()
