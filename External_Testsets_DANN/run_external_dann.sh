#!/usr/bin/env bash
# Score DANN vs finetune_noadv (YOLO11s and YOLOv5su) on DAWN, RTTS and ExDark.
# Zero-shot test only, same protocol as docs/external-testsets.md.
set -u -o pipefail

REPO="$HOME/vd-ext"
W="$HOME/weights"
OUT="$REPO/results"
TAG="dann"                 # output files: results/<dataset>_${TAG}.csv / .txt
DEVICE=0
BATCH_ARG=""               # set to "--batch 8" if CUDA runs out of memory

# DANN checkpoints are pickled with dann_core, so it must be importable
export PYTHONPATH="$HOME/Vehicle-Detection-and-Distance-Estimation/DANN:${PYTHONPATH:-}"

cd "$REPO" || exit 1
mkdir -p "$OUT"

ARMS=(
  "dann_v3_v11=$W/dann_v3/weights/best.pt"
  "finetune_noadv_v11=$W/finetune_noadv/weights/best.pt"
  "dann_v3_v5su=$W/dann_v3_v5su/weights/best.pt"
  "finetune_noadv_v5su=$W/finetune_noadv_v5su/weights/best.pt"
)

# Preflight checks
python -c "import dann_core" 2>/dev/null || { echo "ERROR: dann_core not importable, check PYTHONPATH"; exit 1; }
ARM_ARGS=()
for a in "${ARMS[@]}"; do
  [ -f "${a#*=}" ] || { echo "ERROR: missing weights ${a#*=}"; exit 1; }
  ARM_ARGS+=(--arm "$a")
done

# Record environment and weights for reproducibility
{
  echo "date: $(date -Is)"
  echo "repo commit: $(git rev-parse --short HEAD)"
  python -c "import torch,torchvision,ultralytics; print('torch',torch.__version__,'| torchvision',torchvision.__version__,'| ultralytics',ultralytics.__version__)"
  pip show pycocotools | grep -i '^version'
  for a in "${ARMS[@]}"; do md5sum "${a#*=}"; done
} > "$OUT/external_${TAG}_env.txt"

# Run each dataset (smallest first)
for d in dawn rtts exdark; do
  if [ ! -f "dataset/$d/gt_vehicle5.json" ]; then
    echo "SKIP $d: dataset/$d/gt_vehicle5.json not found"; continue
  fi
  EXTRA=""
  [ "$d" = "dawn" ] && EXTRA="--truck keep"      # DAWN has real truck labels
  echo "=== $d: started $(date +%T) ==="
  python evaluation/score_exdark.py \
    --gt "dataset/$d/gt_vehicle5.json" --images-dir "dataset/$d/images" --slices-dir "splits/$d" \
    $EXTRA $BATCH_ARG "${ARM_ARGS[@]}" \
    --device "$DEVICE" --out "$OUT/${d}_${TAG}.csv" 2>&1 | tee "$OUT/${d}_${TAG}.txt"
  echo "=== $d: exit code ${PIPESTATUS[0]} ==="
done

echo "DONE. Result files:"
ls -lh "$OUT"/*_${TAG}.* "$OUT/external_${TAG}_env.txt"
