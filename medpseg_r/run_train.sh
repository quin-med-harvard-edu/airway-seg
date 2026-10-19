#!/bin/bash
# MEDPSeg-R training launcher (single GPU). Re-running resumes from the newest last*.ckpt.
#   1. runs the data tests,
#   2. benchmarks micro-batch 12, then 8, then 6 (the largest that fits wins),
#   3. trains with that micro-batch (effective batch 48 via gradient accumulation).
# Prerequisite: scripts/00_extract_train.py and scripts/01_prepare_data.py have written data/prepared/.
set -u
cd "$(dirname "$0")"
PY=${PY:-python}
DATA=${MEDPSEG_R_DATA:-data}
RUN=runs/medpseg_r_seed20251014
mkdir -p "$RUN"
LOG="$RUN/launcher.log"
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

nvidia-smi | tee -a "$LOG" || { say "nvidia-smi failed: no GPU visible"; exit 1; }
[ -f "$DATA/prepared/index.json" ] || { say "missing $DATA/prepared/index.json: run the data scripts first"; exit 1; }

say "running data tests"
$PY tests/test_medpseg_r.py "$DATA/prepared/index.json" 2>&1 | grep -v -i warn | tee -a "$LOG" | tail -3
grep -q "ALL TESTS PASSED" "$LOG" || { say "data tests failed; not training"; exit 1; }

MB=""
if ls "$RUN"/checkpoints/last*.ckpt >/dev/null 2>&1 && [ -f "$RUN/micro_batch.txt" ]; then
  MB=$(cat "$RUN/micro_batch.txt"); say "resuming with micro-batch $MB"
else
  for cand in 12 8 6; do
    say "benchmark micro-batch $cand"
    if $PY train_medpseg_r.py --benchmark-steps 60 --micro-batch "$cand" --out "$RUN/bench_mb$cand" \
         > "$RUN/bench_mb$cand.log" 2>&1; then
      grep BENCHMARK "$RUN/bench_mb$cand.log" | tee -a "$LOG"
      MB=$cand; break
    else
      say "micro-batch $cand failed: $(grep -i -m1 'out of memory\|error' "$RUN/bench_mb$cand.log")"
    fi
  done
  [ -n "$MB" ] || { say "no micro-batch fits; see $RUN/bench_mb*.log"; exit 1; }
  echo "$MB" > "$RUN/micro_batch.txt"
fi

say "starting training, micro-batch $MB (effective batch 48), log: $RUN/train.log"
$PY train_medpseg_r.py --micro-batch "$MB" --out "$RUN" >> "$RUN/train.log" 2>&1
status=$?
say "training exited with status $status"
[ -f "$RUN/BEST_CHECKPOINT.txt" ] && say "best: $(head -1 "$RUN/BEST_CHECKPOINT.txt")"
exit $status
