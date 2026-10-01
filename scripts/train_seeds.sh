#!/usr/bin/env bash
# Train one method's seed replicates (default SEEDS="42 123 456") back-to-back,
# one 8-GPU job at a time. Between runs, wait (bounded) for GPU memory to be released.
#
# Usage (from anywhere; the script cd's into $VERL):
#   VERL=/path/to/verl STUDENT_MODEL=/path/to/hf_model bash scripts/train_seeds.sh               # GRPO
#   METHOD=opd VERL=... STUDENT_MODEL=... bash scripts/train_seeds.sh                            # OPD (Qwen3-8B teacher)
#   SEEDS="42" VERL=... STUDENT_MODEL=... bash scripts/train_seeds.sh                            # a single seed
set -uo pipefail
export PATH=/usr/local/nvidia/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/nvidia/lib64:${LD_LIBRARY_PATH:-}

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # this package
VERL=${VERL:?set VERL to your verl checkout root}
METHOD=${METHOD:-grpo}                                    # grpo | opd
SEEDS=${SEEDS:-"42 123 456"}
LAUNCHER=$HERE/scripts/run_qwen3_1p7b_tml_opd_dapo_math_seed.sh
export LOGS_ROOT=${LOGS_ROOT:-$HERE/runs}
mkdir -p "$LOGS_ROOT"
cd "$VERL" || { echo "[train] FATAL: no verl dir $VERL"; exit 1; }

gpu_used_max(){ nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | sort -n | tail -1; }

for S in $SEEDS; do
  TS=$(date -u +%m%d_%H%M)
  RUN_LOG=$LOGS_ROOT/launch_1p7b_${METHOD}_seed${S}_${TS}.log
  echo "[train $(date -u +%FT%TZ)] launching SEED=$S ALGOS=$METHOD TOTAL_TRAINING_STEPS=200 -> $RUN_LOG"
  SEED=$S ALGOS=$METHOD TEACHERS=8B TOTAL_TRAINING_STEPS=200 SAVE_FREQ=20 bash "$LAUNCHER" > "$RUN_LOG" 2>&1
  RC=$?
  echo "[train $(date -u +%FT%TZ)] SEED=$S finished (exit=$RC)"
  # ensure GPUs are released before the next run (up to ~10 min)
  for w in $(seq 1 40); do
    u=$(gpu_used_max)
    if [ -n "$u" ] && [ "$u" -lt 5000 ]; then
      echo "[train $(date -u +%FT%TZ)] GPUs free (max_used=${u}MiB) after SEED=$S"
      break
    fi
    sleep 15
  done
  sleep 20
done
echo "[train $(date -u +%FT%TZ)] all seeds done ($SEEDS)"
