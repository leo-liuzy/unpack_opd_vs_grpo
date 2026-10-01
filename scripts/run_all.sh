#!/usr/bin/env bash
# ===========================================================================
# Top-level reproduction wrapper for the 1.7B OPD-vs-GRPO suite.
#
# This is a DOCUMENTED SEQUENCE, not a turnkey script — it will only launch
# anything if you export RUN=1. Read it first, set the two paths below, then:
#     RUN=1 VERL=/path/to/verl STUDENT_MODEL=/path/to/hf_model bash scripts/run_all.sh
# By default (RUN unset) it just prints the commands it would run.
#
# Prereqs (see README.md, docs/ENVIRONMENT.md and docs/CODE_PATCH.md):
#   - verl checked out at the pinned commit, patch applied, ttrl_math copied in
#   - reward deps installed (latex2sympy2_extended math_verify pylatexenc)
#   - an SFT student checkpoint (not released; see docs/SFT.md)
#   - 8 GPUs with >= 80 GB each (single node)
# Data defaults to this package's dataset/; outputs go to <package>/runs
# (override with LOGS_ROOT).
# ===========================================================================
set -uo pipefail

# ---- EDIT THESE ----
VERL=${VERL:-/path/to/verl}                           # verl checkout root
STUDENT_MODEL=${STUDENT_MODEL:-/path/to/student_hf_model}
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"  # this package

run(){ echo "+ $*"; [[ "${RUN:-0}" == "1" ]] && "$@"; }

[[ -d "$VERL" ]] || { echo "FATAL: set VERL to your verl checkout"; exit 1; }
export VERL STUDENT_MODEL

echo "=== 1) GRPO — seeds 42, 123, 456 (8 GPUs each, serial) ==="
run env METHOD=grpo bash "$HERE/scripts/train_seeds.sh"

echo "=== 2) OPD — seeds 42, 123, 456 (4 train + 4 teacher, Qwen3-8B) ==="
run env METHOD=opd bash "$HERE/scripts/train_seeds.sh"

echo "=== 3) Eval checkpoints {20,40,60,80,120,160,200} of each run ==="
run env METHOD=grpo bash "$HERE/scripts/eval_seeds.sh"
run env METHOD=opd  bash "$HERE/scripts/eval_seeds.sh"

echo "Done. (RUN=${RUN:-0}; set RUN=1 to actually launch.)"
