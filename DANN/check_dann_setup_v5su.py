"""
Run check_dann_setup.py unchanged, with the YOLOv5su layer map patched in.
Arguments are passed through, e.g.:
    python check_dann_setup_v5su.py --weights <v5su best.pt> --target-images <train_night>
"""
import runpy
import sys

import v5su_layers  # patches dann_core before the checker imports it

sys.argv[0] = "check_dann_setup.py"
runpy.run_path("check_dann_setup.py", run_name="__main__")
