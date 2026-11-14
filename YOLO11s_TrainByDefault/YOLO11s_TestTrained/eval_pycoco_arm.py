"""
Score any trained checkpoint with the team-matched pycocotools scorer
(eval_pycocotools.py: conf=0.001, iou=0.6, max_det=300) without editing it.
Only the output file name changes, so each arm gets its own CSV.
The DANN folder is added to sys.path so DANN checkpoints (saved as
dann_core.DANNDetectionModel) can be unpickled. Domain heads are not used at inference.
Usage: python eval_pycoco_arm.py --weights <best.pt> --tag <name>
"""
import argparse
import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DANN_DIR = HERE.parent.parent / "DANN"
sys.path.insert(0, str(DANN_DIR))

ap = argparse.ArgumentParser()
ap.add_argument("--weights", required=True, help="path to best.pt")
ap.add_argument("--tag", required=True, help="label for the output CSV")
args = ap.parse_args()

spec = importlib.util.spec_from_file_location("eval_pycocotools", HERE / "eval_pycocotools.py")
scorer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scorer)

scorer.OUT_CSV = HERE / f"eval_pycocotools_{args.tag}.csv"
print(f"weights : {args.weights}")
print(f"output  : {scorer.OUT_CSV}")

sys.argv = [sys.argv[0], "--weights", args.weights]
scorer.main()
