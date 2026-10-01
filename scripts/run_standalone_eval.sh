#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Standalone post-training evaluation of saved checkpoints.
#
# AIME24 + AIME25 + AMC23 (\boxed{} system prompt, this package's dataset/eval/),
# with these decoding defaults:
#
#   n=16, temperature 0.6, top_p 0.95, top_k 20,
#   max prompt 1024, max generation 31744, max_model_len 32768, bf16.
#
# Reported metric per benchmark: val-core/<dataset>/acc/mean@16.
# The original plot's macro score is the unweighted mean over the three.
#
# Each run is a single `trainer.val_only=True` pass: one validation sweep, then
# return before any training step. Distillation is off (no teacher is loaded).
#
# CHECKPOINTS — verl saves sharded FSDP state by default
# (actor.checkpoint.save_contents=['model','optimizer','extra']), which is NOT
# loadable as a HF model. Each step is therefore merged to HF form first via
# `python -m verl.model_merger`, into <ckpt>/global_step_N/actor/hf_merged.
# Merging is skipped when that directory already exists.
#
# Usage:
#   # every saved step of one run
#   CKPT_ROOT=<LOGS_ROOT>/<exp_name>/ckpt \
#     bash <package>/scripts/run_standalone_eval.sh
#
#   # specific steps
#   CKPT_ROOT=.../ckpt STEPS="40 80 120" bash .../run_standalone_eval.sh
#
#   # an arbitrary HF model (e.g. the SFT init, as a step-0 reference)
#   EVAL_MODEL_PATH=/path/to/hf_model EVAL_TAG=init bash .../run_standalone_eval.sh
# ---------------------------------------------------------------------------

set -euo pipefail

# CUDA-13 torch against a 12.8 driver: the forward-compat libcuda must come first
# or torch sees zero devices. Only set by an interactive rc file otherwise.
CUDA_COMPAT_DIR=${CUDA_COMPAT_DIR:-/usr/local/cuda/compat}
if [ -d "$CUDA_COMPAT_DIR" ]; then
    export LD_LIBRARY_PATH="$CUDA_COMPAT_DIR:${LD_LIBRARY_PATH:-}"
fi

# Root of this reproduction package (parent of scripts/).
REPRO_DIR=${REPRO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}

# ---- what to evaluate ----
CKPT_ROOT=${CKPT_ROOT:-}
STEPS=${STEPS:-}
EVAL_MODEL_PATH=${EVAL_MODEL_PATH:-}
EVAL_TAG=${EVAL_TAG:-}

# ---- eval decoding protocol ----
# n=16 per request; the original standalone protocol used 64.
VAL_ROLLOUT_N=${VAL_ROLLOUT_N:-16}
VAL_TEMPERATURE=${VAL_TEMPERATURE:-0.6}
VAL_TOP_P=${VAL_TOP_P:-0.95}
VAL_TOP_K=${VAL_TOP_K:-20}
VAL_MAX_TOKENS=${VAL_MAX_TOKENS:-31744}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
VAL_MAX_SAMPLES=${VAL_MAX_SAMPLES:--1}
ENABLE_THINKING=${ENABLE_THINKING:-true}
EVAL_DTYPE=${EVAL_DTYPE:-bfloat16}

# ---- data ----
TRAIN_FILE=${TRAIN_FILE:-$REPRO_DIR/dataset/train/dapo_math_17k_train_17661.parquet}
VAL_AIME24=${VAL_AIME24:-$REPRO_DIR/dataset/eval/aime24.parquet}
VAL_AIME25=${VAL_AIME25:-$REPRO_DIR/dataset/eval/aime25.parquet}
VAL_AMC23=${VAL_AMC23:-$REPRO_DIR/dataset/eval/amc23.parquet}

# ---- engine ----
GPUS=${GPUS:-0,1,2,3,4,5,6,7}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.8}
WANDB_PROJECT=${WANDB_PROJECT:-opd-tml-dapo-math-eval}
EVAL_GROUP_NAME=${EVAL_GROUP_NAME:-standalone-eval}
DRY_RUN=${DRY_RUN:-0}

# ===========================================================================
# Invariants
# ===========================================================================
for f in "$TRAIN_FILE" "$VAL_AIME24" "$VAL_AIME25" "$VAL_AMC23"; do
    [ -e "$f" ] || { echo "FATAL: dataset file missing: $f" >&2; exit 1; }
done

if [ -z "$CKPT_ROOT" ] && [ -z "$EVAL_MODEL_PATH" ]; then
    echo "FATAL: set CKPT_ROOT (a training run's ckpt/ dir) or EVAL_MODEL_PATH." >&2
    exit 1
fi
if [ -n "$CKPT_ROOT" ] && [ -n "$EVAL_MODEL_PATH" ]; then
    echo "FATAL: CKPT_ROOT and EVAL_MODEL_PATH are mutually exclusive." >&2
    exit 1
fi

N_GPUS=$(awk -F, '{print NF}' <<<"$GPUS")
export CUDA_VISIBLE_DEVICES="$GPUS"

VISIBLE=$(python3 -c "import torch; print(torch.cuda.device_count() if torch.cuda.is_available() else 0)" 2>/dev/null || echo 0)
if [ "$VISIBLE" != "$N_GPUS" ]; then
    echo "FATAL: torch sees $VISIBLE CUDA devices, expected $N_GPUS (GPUS=$GPUS)." >&2
    echo "       If 0, CUDA forward-compat libs are missing from LD_LIBRARY_PATH=$LD_LIBRARY_PATH" >&2
    exit 1
fi

# The engine must hold the full prompt plus the longest generation. Qwen3-1.7B
# has max_position_embeddings=32768, so this budget must not be padded past it.
MAX_RESPONSE_LENGTH=$VAL_MAX_TOKENS
MAX_MODEL_LEN=$(( MAX_PROMPT_LENGTH + VAL_MAX_TOKENS ))
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-$MAX_MODEL_LEN}
if [ "$MAX_NUM_BATCHED_TOKENS" -lt "$MAX_MODEL_LEN" ]; then
    echo "FATAL: MAX_NUM_BATCHED_TOKENS=$MAX_NUM_BATCHED_TOKENS < max_model_len=$MAX_MODEL_LEN" >&2
    exit 1
fi

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=true
export HYDRA_FULL_ERROR=1
# verl has no trainer.group_name; wandb reads the run group from the environment.
export WANDB_RUN_GROUP="$EVAL_GROUP_NAME"

# Set LOGGER='["console"]' to run without a wandb account.
LOGGER=${LOGGER:-'["console","wandb"]'}

# ---------------------------------------------------------------------------
# eval_one <label> <hf_model_path>
# ---------------------------------------------------------------------------
eval_one() {
    local label="$1"
    local model_path="$2"
    # A dry run never performs the merge, so the model dir legitimately does not
    # exist yet; only enforce these on a real run.
    if [ "$DRY_RUN" != "1" ]; then
        [ -d "$model_path" ] || { echo "FATAL: model dir missing: $model_path" >&2; exit 1; }

        # vLLM hard-errors if max_model_len exceeds the checkpoint's positional limit.
        local max_pos
        max_pos=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['max_position_embeddings'])" \
            "$model_path/config.json") || { echo "FATAL: cannot read max_position_embeddings from $model_path/config.json" >&2; exit 1; }
        if [ "$MAX_MODEL_LEN" -gt "$max_pos" ]; then
            echo "FATAL: max_model_len=$MAX_MODEL_LEN exceeds max_position_embeddings=$max_pos for $model_path" >&2
            exit 1
        fi
    fi

    # EVAL_OUT_DIR lets the caller place results inside the training run's own
    # directory (eval_step<N>_trainprompt_think_n16). Without it, results go to
    # a shared eval root.
    local exp_name out_base
    if [ -n "${EVAL_OUT_DIR:-}" ]; then
        out_base="$EVAL_OUT_DIR"
        exp_name="$(basename "$out_base")"
    else
        exp_name="eval_${label}_n${VAL_ROLLOUT_N}_$(date +%m%d_%H%M%S)"
        out_base="${EVAL_OUT_ROOT:-$REPRO_DIR/runs/standalone_eval}/$exp_name"
    fi

    echo "=================================================================="
    echo "  standalone eval : $label"
    echo "  model           : $model_path"
    echo "  decoding        : n=$VAL_ROLLOUT_N temp=$VAL_TEMPERATURE top_p=$VAL_TOP_P top_k=$VAL_TOP_K"
    echo "  lengths         : prompt=$MAX_PROMPT_LENGTH gen=$VAL_MAX_TOKENS max_model_len=$MAX_MODEL_LEN"
    echo "  dtype           : $EVAL_DTYPE   gpus=$N_GPUS"
    echo "  out             : $out_base"
    echo "=================================================================="

    local ARGS=(
        algorithm.adv_estimator=grpo
        algorithm.use_kl_in_reward=False

        data.train_files="$TRAIN_FILE"
        "data.val_files=[$VAL_AIME24,$VAL_AIME25,$VAL_AMC23]"
        data.val_max_samples=$VAL_MAX_SAMPLES
        data.max_prompt_length=$MAX_PROMPT_LENGTH
        data.max_response_length=$MAX_RESPONSE_LENGTH
        data.shuffle=False
        data.filter_overlong_prompts=True
        data.truncation=error
        # Empty dict under Hydra struct mode: the nested key must be appended.
        +data.apply_chat_template_kwargs.enable_thinking=$ENABLE_THINKING

        actor_rollout_ref.model.path="$model_path"
        actor_rollout_ref.model.use_remove_padding=True
        actor_rollout_ref.actor.fsdp_config.model_dtype=$EVAL_DTYPE
        actor_rollout_ref.actor.fsdp_config.param_offload=True
        # FSDPActorConfig validates this even under val_only, where the actor is
        # never updated. Any value satisfies it; nothing reads it on this path.
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1

        # Same validation as the actor above: these are required to construct the
        # configs even though val_only never computes log probs.
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1

        actor_rollout_ref.rollout.name=vllm
        actor_rollout_ref.rollout.mode=async
        actor_rollout_ref.rollout.tensor_model_parallel_size=1
        actor_rollout_ref.rollout.gpu_memory_utilization=$GPU_MEM_UTIL
        actor_rollout_ref.rollout.max_model_len=$MAX_MODEL_LEN
        actor_rollout_ref.rollout.max_num_batched_tokens=$MAX_NUM_BATCHED_TOKENS
        actor_rollout_ref.rollout.val_kwargs.n=$VAL_ROLLOUT_N
        actor_rollout_ref.rollout.val_kwargs.temperature=$VAL_TEMPERATURE
        actor_rollout_ref.rollout.val_kwargs.top_p=$VAL_TOP_P
        actor_rollout_ref.rollout.val_kwargs.top_k=$VAL_TOP_K
        actor_rollout_ref.rollout.val_kwargs.do_sample=True

        distillation.enabled=False

        custom_reward_function.path="pkg://verl.utils.reward_score.ttrl_math"
        custom_reward_function.name=reward_func

        trainer.project_name="$WANDB_PROJECT"
        trainer.experiment_name="$exp_name"
        trainer.default_local_dir="$out_base/ckpt"
        trainer.validation_data_dir="$out_base/validation"
        trainer.log_val_generations=2
        trainer.n_gpus_per_node=$N_GPUS
        trainer.nnodes=1
        trainer.total_epochs=1
        trainer.test_freq=1
        trainer.save_freq=-1
        trainer.val_before_train=True
        trainer.val_only=True
        "trainer.logger=$LOGGER"
    )

    if [ "$DRY_RUN" = "1" ]; then
        printf 'python3 -m verl.trainer.main_ppo'
        for a in "${ARGS[@]}"; do printf ' \\\n    %q' "$a"; done
        printf '\n\n'
        return
    fi

    mkdir -p "$out_base"

    # Provenance record (eval_config.json).
    # Note on lengths: this verl version has no val-side length knob (SamplingConfig
    # carries no max_tokens), so the generation budget IS data.max_response_length.
    # max_tokens and max_response_len are therefore the same number here.
    LABEL="$label" MODEL_PATH="$model_path" OUT_BASE="$out_base" EXP_NAME="$exp_name" \
    A24="$VAL_AIME24" A25="$VAL_AIME25" AMC="$VAL_AMC23" \
    N="$VAL_ROLLOUT_N" TEMP="$VAL_TEMPERATURE" TOPP="$VAL_TOP_P" TOPK="$VAL_TOP_K" \
    MAXPROMPT="$MAX_PROMPT_LENGTH" MAXRESP="$MAX_RESPONSE_LENGTH" MAXMODEL="$MAX_MODEL_LEN" \
    VMS="$VAL_MAX_SAMPLES" THINK="$ENABLE_THINKING" DT="$EVAL_DTYPE" GMU="$GPU_MEM_UTIL" \
    NG="$N_GPUS" GP="$GPUS" \
    python3 -c '
import json, os, subprocess, datetime
def git():
    try:
        return subprocess.check_output(["git","rev-parse","HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return None
e = os.environ
cfg = {
    "label": e["LABEL"],
    "model_path": e["MODEL_PATH"],
    "eval_which": "custom",
    "val_files": {"AIME24": e["A24"], "AIME25": e["A25"], "AMC23": e["AMC"]},
    "val_kwargs": {"n": int(e["N"]), "temperature": float(e["TEMP"]), "top_p": float(e["TOPP"]),
                   "top_k": int(e["TOPK"]), "do_sample": "True", "max_tokens": int(e["MAXRESP"])},
    "max_prompt_len": int(e["MAXPROMPT"]),
    "max_response_len": int(e["MAXRESP"]),
    "max_model_len": int(e["MAXMODEL"]),
    "val_max_samples": int(e["VMS"]),
    "enable_thinking": e["THINK"],
    "dtype": e["DT"],
    "gpu_memory_utilization": float(e["GMU"]),
    "reward_function": "verl.utils.reward_score.ttrl_math",
    "n_gpus": int(e["NG"]),
    "gpus": e["GP"],
    "experiment_name": e["EXP_NAME"],
    "out_base": e["OUT_BASE"],
    "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "git_commit": git(),
}
with open(os.path.join(e["OUT_BASE"], "eval_config.json"), "w") as f:
    json.dump(cfg, f, indent=2)
print("[eval_config] wrote", os.path.join(e["OUT_BASE"], "eval_config.json"))
'

    ray stop --force 2>/dev/null || true

    # Keep this eval's stdout so the metric dict survives the run. verl computes the full
    # val-core/val-aux dict but only prints it -- validation_data_dir persists generations
    # and nothing else -- so write_eval_metrics.py lifts it into validation/metrics.json.
    # pipefail is set, so a failing trainer still fails the pipeline.
    python3 -m verl.trainer.main_ppo "${ARGS[@]}" 2>&1 | tee "$out_base/eval.log"

    python3 "$(dirname "${BASH_SOURCE[0]}")/write_eval_metrics.py" \
        --eval-log "$out_base/eval.log" --out-dir "$out_base"
}

# ---------------------------------------------------------------------------
# Merge a sharded FSDP actor checkpoint into a loadable HF model directory.
# ---------------------------------------------------------------------------
merge_ckpt() {
    local actor_dir="$1"
    local target="$actor_dir/hf_merged"
    if [ -d "$target" ]; then
        echo "[merge] reusing $target" >&2
        printf '%s' "$target"
        return
    fi
    echo "[merge] $actor_dir -> $target" >&2
    if [ "$DRY_RUN" = "1" ]; then
        printf '%s' "$target"
        return
    fi
    python3 -m verl.model_merger merge \
        --backend fsdp \
        --local_dir "$actor_dir" \
        --target_dir "$target" >&2
    printf '%s' "$target"
}

if [ -n "$EVAL_MODEL_PATH" ]; then
    eval_one "${EVAL_TAG:-custom}" "$EVAL_MODEL_PATH"
    exit 0
fi

[ -d "$CKPT_ROOT" ] || { echo "FATAL: CKPT_ROOT not a directory: $CKPT_ROOT" >&2; exit 1; }

# Default to every saved step, in numeric order.
if [ -z "$STEPS" ]; then
    STEPS=$(/bin/ls -d "$CKPT_ROOT"/global_step_*/ 2>/dev/null \
        | sed 's#.*/global_step_##; s#/##' | sort -n | tr '\n' ' ')
fi
[ -n "${STEPS// /}" ] || { echo "FATAL: no global_step_* checkpoints under $CKPT_ROOT" >&2; exit 1; }

echo "Evaluating steps: $STEPS"
RUN_TAG=$(basename "$(dirname "$CKPT_ROOT")")

for step in $STEPS; do
    actor_dir="$CKPT_ROOT/global_step_${step}/actor"
    [ -d "$actor_dir" ] || { echo "FATAL: missing $actor_dir" >&2; exit 1; }
    hf_dir=$(merge_ckpt "$actor_dir")
    eval_one "${RUN_TAG}_step${step}" "$hf_dir"
done
