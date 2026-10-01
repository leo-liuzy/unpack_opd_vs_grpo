#!/usr/bin/env bash
# ===========================================================================
# Gated post-training eval of one method's seed runs (default SEEDS="42 123 456").
#
# Usage:
#   VERL=/path/to/verl bash scripts/eval_seeds.sh               # GRPO
#   METHOD=opd VERL=/path/to/verl bash scripts/eval_seeds.sh    # OPD
#
# GATE: block until EVERY seed run has a completed ckpt/global_step_200/actor,
#       AND no verl process is alive, AND GPUs are free (<5 GB max used),
#       held for a 3-poll stable streak (avoids racing a mid-teardown state).
# EVAL: for each seed run, for each step in STEPS_TO_EVAL, run
#       run_standalone_eval.sh once (STEPS=<one step>, EVAL_OUT_DIR set) so each
#       eval gets a deterministic dir, is fault-isolated (the tool is `set -e`,
#       so a shared multi-step call would abort the rest on one failure), and is
#       resumable (skip if validation/metrics.json exists). 3 seeds x 7 steps = 21.
#
# The eval tool's defaults are the protocol we report:
#   n16 / T0.6 / top_p0.95 / top_k20 / max_tokens31744 / prompt1024 / model_len32768
#   / bf16 / gpu_mem0.8 / ttrl_math reward / dataset/eval aime24+25+amc23
# so no decoding overrides are needed. Each eval merges the FSDP shards to
# <ckpt>/global_step_N/actor/hf_merged (reused if present) then one val_only pass.
#
# Must run from the verl checkout so `verl` resolves to the repo package
# (from ~ it resolves to a namespace stub without verl.model_merger).
# ===========================================================================
set -uo pipefail
export PATH=/usr/local/nvidia/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/nvidia/lib64:${LD_LIBRARY_PATH:-}

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # this package
VERL=${VERL:?set VERL to your verl checkout root}
EVAL_SH=$HERE/scripts/run_standalone_eval.sh
LOGS=${LOGS_ROOT:-$HERE/runs}
METHOD=${METHOD:-grpo}                                    # grpo | opd
case "$METHOD" in
  grpo) RUN_TAG=grpo ;;
  opd)  RUN_TAG=opd_t8b_k1pg ;;
  *) echo "FATAL: METHOD must be grpo or opd"; exit 1 ;;
esac
SEEDS=${SEEDS:-"42 123 456"}
N_SEEDS=$(wc -w <<<"$SEEDS" | tr -d ' ')
GLOB_PREFIX="$LOGS/qwen3-1p7b-ot3sft_${RUN_TAG}_think_lr2e-6_dapo17k"
STEPS_TO_EVAL=${STEPS_TO_EVAL:-"20 40 60 80 120 160 200"}
mkdir -p "$LOGS"
ORCH_LOG=$LOGS/eval_${METHOD}_seeds_orch.log
NV=nvidia-smi

say(){ echo "[eval-orch $(date -u +%FT%TZ)] $*" | tee -a "$ORCH_LOG"; }

cd "$VERL" || { say "FATAL: no verl dir $VERL"; exit 1; }
[[ -f "$EVAL_SH" ]] || { say "FATAL: no eval tool $EVAL_SH"; exit 1; }

gpu_used_max(){ $NV --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | sort -n | tail -1; }
verl_alive(){ pgrep -f 'verl[.]trainer[.]main_ppo' >/dev/null 2>&1; }

# Populate RUNS[] with the (latest) run dir per seed in SEEDS that has a
# completed step-200 checkpoint. Empty entry => that seed not done yet.
RUNS=()
find_runs(){
  RUNS=()
  local S d latest
  for S in $SEEDS; do
    latest=""
    for d in $(ls -dt ${GLOB_PREFIX}_seed${S}_*/ 2>/dev/null); do
      if [[ -d "${d}ckpt/global_step_200/actor" ]]; then latest="${d%/}"; break; fi
    done
    [[ -n "$latest" ]] && RUNS+=("$latest")
  done
}

# ---- GATE ----
say "eval orchestrator started; steps={$STEPS_TO_EVAL}; waiting for seeds {$SEEDS} to finish 200 steps"
streak=0; polls=0; CAP=2880   # 2880 * 30s = 24h
while :; do
  find_runs
  n=${#RUNS[@]}
  used=$(gpu_used_max); alive=$(verl_alive && echo 1 || echo 0)
  if [[ "$n" -eq "$N_SEEDS" && "$alive" -eq 0 && -n "$used" && "$used" -lt 5000 ]]; then
    streak=$((streak+1))
    say "gate: $n runs @step200, verl=0, gpu_used_max=${used}MiB (streak=$streak/3)"
    [[ $streak -ge 3 ]] && break
  else
    [[ $((polls % 20)) -eq 0 ]] && say "waiting: runs_done=$n/$N_SEEDS verl=$alive gpu_used_max=${used:-?}MiB"
    streak=0
  fi
  polls=$((polls+1)); [[ $polls -ge $CAP ]] && { say "FATAL: gate timeout ~24h"; exit 1; }
  sleep 30
done
say "GATE PASSED. runs:"
for r in "${RUNS[@]}"; do say "  - $(basename "$r")"; done
sleep 10

# ---- EVAL ----
fails=0; done_n=0
for RUN in "${RUNS[@]}"; do
  CKPT_ROOT="$RUN/ckpt"
  for STEP in $STEPS_TO_EVAL; do
    OUT="$RUN/eval_step${STEP}_trainprompt_think_n16"
    if [[ -f "$OUT/validation/metrics.json" ]]; then
      say "SKIP $(basename "$RUN") step$STEP (metrics.json exists)"; done_n=$((done_n+1)); continue
    fi
    if [[ ! -d "$CKPT_ROOT/global_step_${STEP}/actor" ]]; then
      say "WARN missing ckpt $(basename "$RUN") global_step_${STEP} -- skipping"; fails=$((fails+1)); continue
    fi
    say "EVAL $(basename "$RUN") step$STEP -> $OUT"
    CKPT_ROOT="$CKPT_ROOT" STEPS="$STEP" EVAL_OUT_DIR="$OUT" \
      bash "$EVAL_SH" > "$OUT.run.log" 2>&1
    rc=$?
    if [[ $rc -eq 0 && -f "$OUT/validation/metrics.json" ]]; then
      say "OK   $(basename "$RUN") step$STEP (rc=0)"; done_n=$((done_n+1))
    else
      say "FAIL $(basename "$RUN") step$STEP rc=$rc metrics=$([[ -f "$OUT/validation/metrics.json" ]] && echo yes || echo NO) -- see $OUT.run.log"
      fails=$((fails+1))
    fi
    # ensure GPUs released before the next eval
    for w in $(seq 1 40); do u=$(gpu_used_max); [[ -n "$u" && "$u" -lt 5000 ]] && break; sleep 15; done
  done
done

say "=== ALL EVALS DONE (ok=$done_n fails=$fails) ==="

# ---- summary table (macro = unweighted mean over aime24/aime25/amc23 acc/mean@16) ----
say "summary (val-core acc/mean@16):"
for RUN in "${RUNS[@]}"; do
  for STEP in $STEPS_TO_EVAL; do
    M="$RUN/eval_step${STEP}_trainprompt_think_n16/validation/metrics.json"
    [[ -f "$M" ]] || continue
    line=$(python3 - "$M" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))["metrics"]
def find(sub):
    for k,v in d.items():
        if "acc/mean@16" in k and sub in k.lower():
            try: return float(v)
            except: return None
    return None
a24=find("aime24"); a25=find("aime25"); amc=find("amc23")
vals=[x for x in (a24,a25,amc) if x is not None]
macro=sum(vals)/len(vals) if vals else None
def f(x): return "n/a" if x is None else f"{x:.4f}"
print(f"macro={f(macro)}  aime24={f(a24)} aime25={f(a25)} amc23={f(amc)}")
PY
)
    say "  $(basename "$RUN") step$STEP: $line"
  done
done
say "=== eval orchestrator exiting ==="
