#!/bin/bash
# MEDPSeg-R inference on the ATM'26 test and val splits with one epoch's checkpoint.
# Usage: bash run_infer.sh [epoch]   (default 035, the final model). Finished cases are skipped.
# Needs ATM26_TEST_IMAGES and ATM26_VAL_IMAGES (folders with <id>_0000.mha).
set -u
cd "$(dirname "$0")"
PY=${PY:-python}
EP=${1:-035}
CK=runs/medpseg_r_seed20251014/checkpoints/epoch_${EP}.ckpt
[ -f "$CK" ] || { echo "missing $CK"; exit 1; }
OUT=outputs/epoch${EP}
mkdir -p "$OUT"
$PY predict_medpseg_r.py --weight "$CK" --splits test val --out-root "$OUT" --batch-size 4 2>&1 | tee -a "$OUT/infer.log"
