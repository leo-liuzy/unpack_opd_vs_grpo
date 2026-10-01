#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Launcher for the 1.7B OPD-vs-GRPO suite (GRPO and OPD; pick with ALGOS).
#
# Notes:
#   * A `SEED` env var (default 123) is exposed and threaded through the four
#     verl-level RNG knobs that actually vary a run:
#         data.seed
#         data.data_loader_seed
#         actor_rollout_ref.actor.data_loader_seed
#         actor_rollout_ref.actor.fsdp_config.seed
#   * `_seed${SEED}` is embedded in EXP_NAME so the run dir and wandb name
#     make the seed obvious without opening the config.
#   * The rollout engine (vLLM) has NO seed knob under verl's rollout config,
#     so nucleus-sampling stream diversity remains process-random. This is the
#     reproducibility floor.
#
# Run from the verl checkout root. Required: STUDENT_MODEL. Data defaults to
# this package's dataset/ dir; outputs go to LOGS_ROOT (default <package>/runs).
#
# Usage:
#   STUDENT_MODEL=/path/to/hf_model SEED=123 ALGOS=grpo bash .../run_qwen3_1p7b_tml_opd_dapo_math_seed.sh
#   STUDENT_MODEL=/path/to/hf_model SEED=123 ALGOS=opd TEACHERS=8B bash .../run_qwen3_1p7b_tml_opd_dapo_math_seed.sh
#   DRY_RUN=1 ... bash .../run_qwen3_1p7b_tml_opd_dapo_math_seed.sh   # print, don't launch
# ---------------------------------------------------------------------------

set -euo pipefail

CUDA_COMPAT_DIR=${CUDA_COMPAT_DIR:-/usr/local/cuda/compat}
if [ -d "$CUDA_COMPAT_DIR" ]; then
    export LD_LIBRARY_PATH="$CUDA_COMPAT_DIR:${LD_LIBRARY_PATH:-}"
fi
# Ray auto-detects GPUs by spawning nvidia-smi. On some GPU containers (e.g.
# Kubernetes/GKE device-plugin setups) nvidia-smi lives at
# /usr/local/nvidia/bin and libnvidia-ml.so (which nvidia-smi dynamically loads)
# lives at /usr/local/nvidia/lib64 -- neither is in the default non-interactive
# PATH/LD_LIBRARY_PATH. Without this block, Ray's local instance init reports
# 0 GPUs and verl fails with "Total available GPUs 0 is less than 8".
# CRITICAL: /usr/local/nvidia/lib64 must come AFTER /usr/local/cuda/compat --
# both contain a libcuda.so and the compat one MUST be resolved first (or torch
# gets the stock 12.8 libcuda against a 13 build and reports 0 devices).
if [ -d /usr/local/nvidia/lib64 ]; then
    export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:/usr/local/nvidia/lib64"
fi
if [ -d /usr/local/nvidia/bin ]; then
    export PATH="/usr/local/nvidia/bin:${PATH:-}"
fi

# Root of this reproduction package (parent of scripts/).
REPRO_DIR=${REPRO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}

# ---- jobs to run, sequentially ----
ALGOS=${ALGOS:-"opd"}
TEACHERS=${TEACHERS:-"8B"}

# ---- student / data ----
# STUDENT_MODEL: HF dir of the SFT'd Qwen3-1.7B-Base student (see README §2).
STUDENT_MODEL=${STUDENT_MODEL:-}
TRAIN_FILE=${TRAIN_FILE:-$REPRO_DIR/dataset/train/dapo_math_17k_train_17661.parquet}
# In-run validation is disabled (TEST_FREQ=-1); verl still requires val files.
VAL_AIME24=${VAL_AIME24:-$REPRO_DIR/dataset/eval/aime24.parquet}
VAL_AIME25=${VAL_AIME25:-$REPRO_DIR/dataset/eval/aime25.parquet}
VAL_AMC23=${VAL_AMC23:-$REPRO_DIR/dataset/eval/amc23.parquet}

# ---- training hyperparameters (match the original DAPO-Math runs) ----
GPUS=${GPUS:-0,1,2,3,4,5,6,7}
LR=${LR:-2e-6}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-64}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-64}
MICRO_BS=${MICRO_BS:-4}
ROLLOUT_N=${ROLLOUT_N:-8}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-7168}
USE_DYNAMIC_BSZ=${USE_DYNAMIC_BSZ:-True}
PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-32768}
TRAIN_TEMPERATURE=${TRAIN_TEMPERATURE:-1.0}
TRAIN_TOP_P=${TRAIN_TOP_P:-1.0}
TRAIN_TOP_K=${TRAIN_TOP_K:--1}
ENABLE_THINKING=${ENABLE_THINKING:-true}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-120}
EPOCHS=${EPOCHS:-40}
SAVE_FREQ=${SAVE_FREQ:-20}
TEST_FREQ=${TEST_FREQ:--1}

# ---- SEED (replicates: 123, 456) ----
# Only integer values are honored.
SEED=${SEED:-123}
if ! [[ "$SEED" =~ ^-?[0-9]+$ ]]; then
    echo "FATAL: SEED must be an integer, got: $SEED" >&2; exit 1
fi

# ---- in-training validation rollout ----
VAL_ROLLOUT_N=${VAL_ROLLOUT_N:-16}
VAL_TEMPERATURE=${VAL_TEMPERATURE:-0.7}
VAL_TOP_P=${VAL_TOP_P:-0.95}
VAL_TOP_K=${VAL_TOP_K:--1}
DTYPE=${DTYPE:-fp32}

# ---- GRPO baseline knobs ----
GRPO_NORM_ADV_BY_STD=${GRPO_NORM_ADV_BY_STD:-False}
GRPO_ROLLOUT_IS=${GRPO_ROLLOUT_IS:-token}
GRPO_ROLLOUT_IS_THRESHOLD=${GRPO_ROLLOUT_IS_THRESHOLD:-2.0}
GRPO_USE_TORCH_COMPILE=${GRPO_USE_TORCH_COMPILE:-False}
GRPO_CLIP_RATIO_HIGH=${GRPO_CLIP_RATIO_HIGH:-0.28}

# ---- PG OPD (Thinking Machines) objective knobs ----
DISTILLATION_LOSS_MODE=${DISTILLATION_LOSS_MODE:-k1}
USE_POLICY_GRADIENT=${USE_POLICY_GRADIENT:-True}
POLICY_LOSS_MODE=${POLICY_LOSS_MODE:-vanilla}
CLIP_RATIO_LOW=${CLIP_RATIO_LOW:-0.2}
CLIP_RATIO_HIGH=${CLIP_RATIO_HIGH:-0.2}
LOSS_MAX_CLAMP=${LOSS_MAX_CLAMP:-10.0}
LOG_PROB_MIN_CLAMP=${LOG_PROB_MIN_CLAMP:--10.0}
USE_TASK_REWARDS=${USE_TASK_REWARDS:-False}

# ---- engine / memory ----
ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.5}
ROLLOUT_MAX_NUM_BATCHED_TOKENS=${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-32768}
TEACHER_DTYPE=${TEACHER_DTYPE:-float32}
TEACHER_GPU_MEM_UTIL=${TEACHER_GPU_MEM_UTIL:-0.8}
TEACHER_REPLICAS=${TEACHER_REPLICAS:-}
TEACHER_MAX_NUM_BATCHED_TOKENS=${TEACHER_MAX_NUM_BATCHED_TOKENS:-32768}

# ---- logging ----
WANDB_PROJECT=${WANDB_PROJECT:-opd-tml-dapo-math}
RUN_TS=${RUN_TS:-$(date +%m%d_%H%M)}

# ---- resuming an existing run ----
RESUME_FROM_PATH=${RESUME_FROM_PATH:-}
RESUME_MODE=${RESUME_MODE:-}
LOGS_ROOT=${LOGS_ROOT:-$REPRO_DIR/runs}
DRY_RUN=${DRY_RUN:-0}

# ===========================================================================
# Invariants
# ===========================================================================
for f in "$TRAIN_FILE" "$VAL_AIME24" "$VAL_AIME25" "$VAL_AMC23"; do
    [ -e "$f" ] || { echo "FATAL: dataset file missing: $f" >&2; exit 1; }
done
[ -n "$STUDENT_MODEL" ] || { echo "FATAL: set STUDENT_MODEL to the SFT student HF dir (see README)" >&2; exit 1; }
[ -d "$STUDENT_MODEL" ] || { echo "FATAL: student model dir missing: $STUDENT_MODEL" >&2; exit 1; }

TOTAL_GPUS=$(awk -F, '{print NF}' <<<"$GPUS")
[ "$TOTAL_GPUS" -gt 0 ] || { echo "FATAL: GPUS is empty" >&2; exit 1; }
export CUDA_VISIBLE_DEVICES="$GPUS"

VISIBLE=$(python3 -c "import torch; print(torch.cuda.device_count() if torch.cuda.is_available() else 0)" 2>/dev/null || echo 0)
if [ "$VISIBLE" != "$TOTAL_GPUS" ]; then
    echo "FATAL: torch sees $VISIBLE CUDA devices, expected $TOTAL_GPUS (GPUS=$GPUS)." >&2
    echo "       If 0, the CUDA forward-compat libs are missing from LD_LIBRARY_PATH:" >&2
    echo "       LD_LIBRARY_PATH=$LD_LIBRARY_PATH" >&2
    exit 1
fi

if [ "$DISTILLATION_LOSS_MODE" = "k1" ] && [ "$USE_POLICY_GRADIENT" != "True" ]; then
    echo "FATAL: loss_mode=k1 requires USE_POLICY_GRADIENT=True (DistillationLossConfig rejects k1 with direct backprop)." >&2
    exit 1
fi
MAX_NUM_TOKENS=$(( MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH + 1 ))
ROLLOUT_MAX_MODEL_LEN=$(( MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH + 1 ))

if [ "$PPO_MAX_TOKEN_LEN_PER_GPU" -lt $(( MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH )) ]; then
    echo "FATAL: PPO_MAX_TOKEN_LEN_PER_GPU=$PPO_MAX_TOKEN_LEN_PER_GPU < prompt+response=$(( MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH ))" >&2
    exit 1
fi
if [ "$ROLLOUT_MAX_NUM_BATCHED_TOKENS" -lt "$ROLLOUT_MAX_MODEL_LEN" ]; then
    echo "FATAL: ROLLOUT_MAX_NUM_BATCHED_TOKENS=$ROLLOUT_MAX_NUM_BATCHED_TOKENS < rollout max_model_len=$ROLLOUT_MAX_MODEL_LEN" >&2
    exit 1
fi

# Set LOGGER='["console"]' to run without a wandb account.
LOGGER=${LOGGER:-'["console","wandb"]'}

JOBS=()
for A in $ALGOS; do
    case "$A" in
        grpo) JOBS+=("grpo:none") ;;
        opd)  for T in $TEACHERS; do JOBS+=("opd:$T"); done ;;
        *) echo "FATAL: unknown algo '$A' (use grpo or opd)" >&2; exit 1 ;;
    esac
done
[ ${#JOBS[@]} -gt 0 ] || { echo "FATAL: no jobs selected (ALGOS='$ALGOS')" >&2; exit 1; }

CKPT_GB_EACH=${CKPT_GB_EACH:-21}
if [ "$SAVE_FREQ" -gt 0 ]; then
    mkdir -p "$LOGS_ROOT"
    NEED_GB=$(( (TOTAL_TRAINING_STEPS / SAVE_FREQ) * ${#JOBS[@]} * CKPT_GB_EACH ))
    AVAIL_GB=$(df -BG --output=avail "$LOGS_ROOT" | tail -1 | tr -dc '0-9')
    echo "[preflight] checkpoints: $(( TOTAL_TRAINING_STEPS / SAVE_FREQ )) x ${#JOBS[@]} run(s) ~= ${NEED_GB} GB; ${AVAIL_GB} GB free on $LOGS_ROOT"
    if [ "$NEED_GB" -gt "$AVAIL_GB" ]; then
        echo "FATAL: need ~${NEED_GB} GB of checkpoints but only ${AVAIL_GB} GB free on $LOGS_ROOT." >&2
        exit 1
    fi
fi

for JOB in "${JOBS[@]}"; do
    ALGO="${JOB%%:*}"
    T="${JOB##*:}"

    if [ "$ALGO" = "grpo" ]; then
        TEACHER_GPUS=0
        TRAINER_GPUS=$TOTAL_GPUS
        RUN_TAG="grpo"
        OBJECTIVE_DESC="grpo (outcome reward via ttrl_math)"
    else
        case "$T" in
            4B)  TEACHER_MODEL="Qwen/Qwen3-4B";  TEACHER_TP=1; N_REPLICAS=4 ;;
            8B)  TEACHER_MODEL="Qwen/Qwen3-8B";  TEACHER_TP=2; N_REPLICAS=2 ;;
            32B) TEACHER_MODEL="Qwen/Qwen3-32B"; TEACHER_TP=4; N_REPLICAS=1 ;;
            *) echo "FATAL: unknown teacher '$T' (use 4B, 8B or 32B)" >&2; exit 1 ;;
        esac
        N_REPLICAS=${TEACHER_REPLICAS:-$N_REPLICAS}
        TEACHER_GPUS=$(( TEACHER_TP * N_REPLICAS ))
        TRAINER_GPUS=$(( TOTAL_GPUS - TEACHER_GPUS ))

        if [ $(( TEACHER_GPUS % TEACHER_TP )) -ne 0 ]; then
            echo "FATAL: teacher pool $TEACHER_GPUS not divisible by tp=$TEACHER_TP" >&2; exit 1
        fi
        if [ "$TRAINER_GPUS" -le 0 ]; then
            echo "FATAL: teacher pool ($TEACHER_GPUS) leaves $TRAINER_GPUS GPUs for training out of $TOTAL_GPUS." >&2
            exit 1
        fi
        TEACHER_REPLICAS_EFF=$N_REPLICAS
        RUN_TAG="opd_t$(echo "$T" | tr 'A-Z' 'a-z')_${DISTILLATION_LOSS_MODE}pg"
        OBJECTIVE_DESC="loss_mode=$DISTILLATION_LOSS_MODE use_policy_gradient=$USE_POLICY_GRADIENT use_task_rewards=$USE_TASK_REWARDS"
    fi

    for _n in $(( TRAIN_BATCH_SIZE * ROLLOUT_N )) $(( PPO_MINI_BATCH_SIZE * ROLLOUT_N )); do
        if [ $(( _n % TRAINER_GPUS )) -ne 0 ]; then
            echo "FATAL: $_n sequences do not divide $TRAINER_GPUS trainer GPUs (job=$JOB)." >&2
            echo "       train_batch_size=$TRAIN_BATCH_SIZE ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE rollout_n=$ROLLOUT_N" >&2
            exit 1
        fi
    done

    # SEED tag embedded before RUN_TS so seed is visible in log dir + wandb name.
    EXP_NAME="qwen3-1p7b-ot3sft_${RUN_TAG}_think_lr${LR}_dapo17k_seed${SEED}_${RUN_TS}"
    LOGDIR="$LOGS_ROOT/$EXP_NAME"

    # Refuse to overwrite an existing run.
    if [ -e "$LOGDIR" ] && [ -z "$RESUME_FROM_PATH" ] && [ -z "$RESUME_MODE" ]; then
        echo "FATAL: run dir already exists (would overwrite): $LOGDIR" >&2
        echo "       Rename it, choose a new SEED / RUN_TS, or set RESUME_FROM_PATH." >&2
        exit 1
    fi

    echo "=================================================================="
    echo "  experiment   : $EXP_NAME"
    echo "  student      : $STUDENT_MODEL"
    echo "  seed         : $SEED  (actor.data_loader_seed / actor.fsdp.seed; data.seed left null)"
    if [ "$ALGO" = "grpo" ]; then
        echo "  teacher      : (none)"
    else
        echo "  teacher      : $TEACHER_MODEL (tp=$TEACHER_TP, replicas=$TEACHER_REPLICAS_EFF, dtype=$TEACHER_DTYPE)"
    fi
    echo "  gpus         : $TRAINER_GPUS train + $TEACHER_GPUS teacher = $TOTAL_GPUS"
    echo "  objective    : $OBJECTIVE_DESC"
    echo "  lengths      : prompt=$MAX_PROMPT_LENGTH resp=$MAX_RESPONSE_LENGTH (rollout ctx=$ROLLOUT_MAX_MODEL_LEN)"
    echo "  steps        : $TOTAL_TRAINING_STEPS (save every $SAVE_FREQ, test_freq=$TEST_FREQ)"
    if [ "$TEST_FREQ" -gt 0 ]; then
        echo "  in-train eval: n=$VAL_ROLLOUT_N temp=$VAL_TEMPERATURE top_p=$VAL_TOP_P top_k=$VAL_TOP_K gen<=$MAX_RESPONSE_LENGTH"
    fi
    echo "=================================================================="

    ARGS=(
        algorithm.adv_estimator=grpo
        algorithm.use_kl_in_reward=False

        data.train_files="$TRAIN_FILE"
        "data.val_files=[$VAL_AIME24,$VAL_AIME25,$VAL_AMC23]"
        data.train_batch_size=$TRAIN_BATCH_SIZE
        data.max_prompt_length=$MAX_PROMPT_LENGTH
        data.max_response_length=$MAX_RESPONSE_LENGTH
        data.filter_overlong_prompts=True
        data.truncation=error
        data.shuffle=False
        +data.apply_chat_template_kwargs.enable_thinking=$ENABLE_THINKING

        # ---- SEED overrides. data.seed is deliberately LEFT at its default
        # (`null`), matching our runs; data.shuffle=False anyway. ----
        actor_rollout_ref.actor.data_loader_seed=$SEED
        actor_rollout_ref.actor.fsdp_config.seed=$SEED

        actor_rollout_ref.model.path="$STUDENT_MODEL"
        actor_rollout_ref.model.use_remove_padding=True
        actor_rollout_ref.model.enable_gradient_checkpointing=True

        actor_rollout_ref.actor.optim.lr=$LR
        actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$MICRO_BS
        actor_rollout_ref.actor.use_dynamic_bsz=$USE_DYNAMIC_BSZ
        actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$PPO_MAX_TOKEN_LEN_PER_GPU
        actor_rollout_ref.actor.ppo_epochs=1
        actor_rollout_ref.actor.use_kl_loss=False
        actor_rollout_ref.actor.loss_agg_mode=token-mean
        actor_rollout_ref.actor.fsdp_config.model_dtype=$DTYPE
        actor_rollout_ref.actor.fsdp_config.param_offload=False
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=False

        actor_rollout_ref.rollout.name=vllm
        actor_rollout_ref.rollout.mode=async
        actor_rollout_ref.rollout.n=$ROLLOUT_N
        actor_rollout_ref.rollout.temperature=$TRAIN_TEMPERATURE
        actor_rollout_ref.rollout.top_p=$TRAIN_TOP_P
        actor_rollout_ref.rollout.top_k=$TRAIN_TOP_K
        actor_rollout_ref.rollout.tensor_model_parallel_size=1
        actor_rollout_ref.rollout.gpu_memory_utilization=$ROLLOUT_GPU_MEM_UTIL
        actor_rollout_ref.rollout.max_model_len=$ROLLOUT_MAX_MODEL_LEN
        actor_rollout_ref.rollout.max_num_batched_tokens=$ROLLOUT_MAX_NUM_BATCHED_TOKENS
        actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=$USE_DYNAMIC_BSZ
        actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$PPO_MAX_TOKEN_LEN_PER_GPU
        actor_rollout_ref.rollout.val_kwargs.n=$VAL_ROLLOUT_N
        actor_rollout_ref.rollout.val_kwargs.temperature=$VAL_TEMPERATURE
        actor_rollout_ref.rollout.val_kwargs.top_p=$VAL_TOP_P
        actor_rollout_ref.rollout.val_kwargs.top_k=$VAL_TOP_K
        actor_rollout_ref.rollout.val_kwargs.do_sample=True

        custom_reward_function.path="pkg://verl.utils.reward_score.ttrl_math"
        custom_reward_function.name=reward_func

        trainer.balance_batch=True
        trainer.project_name="$WANDB_PROJECT"
        trainer.experiment_name="$EXP_NAME"
        trainer.default_local_dir="$LOGDIR/ckpt"
        trainer.validation_data_dir="$LOGDIR/validation"
        trainer.n_gpus_per_node=$TRAINER_GPUS
        trainer.nnodes=1
        trainer.total_epochs=$EPOCHS
        trainer.total_training_steps=$TOTAL_TRAINING_STEPS
        trainer.save_freq=$SAVE_FREQ
        trainer.test_freq=$TEST_FREQ
        trainer.val_before_train=False
        "trainer.logger=$LOGGER"
    )

    if [ -n "$RESUME_FROM_PATH" ]; then
        [ -d "$RESUME_FROM_PATH" ] || { echo "FATAL: RESUME_FROM_PATH does not exist: $RESUME_FROM_PATH" >&2; exit 1; }
        [ -d "$RESUME_FROM_PATH/actor" ] || { echo "FATAL: no actor/ under $RESUME_FROM_PATH" >&2; exit 1; }
        case "$RESUME_FROM_PATH" in
            *global_step_*) : ;;
            *) echo "FATAL: RESUME_FROM_PATH must name a global_step_N directory: $RESUME_FROM_PATH" >&2; exit 1 ;;
        esac
        _resume_step=${RESUME_FROM_PATH##*global_step_}
        if [ "$_resume_step" -ge "$TOTAL_TRAINING_STEPS" ]; then
            echo "FATAL: resuming at step $_resume_step but TOTAL_TRAINING_STEPS=$TOTAL_TRAINING_STEPS; nothing to do." >&2
            exit 1
        fi
        ARGS+=(
            trainer.resume_mode=resume_path
            trainer.resume_from_path="$RESUME_FROM_PATH"
        )
    elif [ -n "$RESUME_MODE" ]; then
        ARGS+=(trainer.resume_mode="$RESUME_MODE")
    fi

    if [ "$ALGO" = "grpo" ]; then
        ARGS+=(
            distillation.enabled=False
            algorithm.norm_adv_by_std_in_grpo=$GRPO_NORM_ADV_BY_STD
            algorithm.rollout_correction.rollout_is=$GRPO_ROLLOUT_IS
            algorithm.rollout_correction.rollout_is_threshold=$GRPO_ROLLOUT_IS_THRESHOLD
            actor_rollout_ref.rollout.calculate_log_probs=True
            actor_rollout_ref.actor.fsdp_config.use_torch_compile=$GRPO_USE_TORCH_COMPILE
            actor_rollout_ref.actor.clip_ratio_high=$GRPO_CLIP_RATIO_HIGH
        )
    else
        ARGS+=(
            distillation.enabled=True
            distillation.n_gpus_per_node=$TEACHER_GPUS
            distillation.nnodes=1
            distillation.teacher_models.teacher_model.model_path="$TEACHER_MODEL"
            distillation.teacher_models.teacher_model.num_replicas=$TEACHER_REPLICAS_EFF
            distillation.teacher_models.teacher_model.inference.name=vllm
            distillation.teacher_models.teacher_model.inference.dtype=$TEACHER_DTYPE
            distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size=$TEACHER_TP
            distillation.teacher_models.teacher_model.inference.gpu_memory_utilization=$TEACHER_GPU_MEM_UTIL
            distillation.teacher_models.teacher_model.inference.max_model_len=$MAX_NUM_TOKENS
            distillation.teacher_models.teacher_model.inference.max_num_batched_tokens=$TEACHER_MAX_NUM_BATCHED_TOKENS
            distillation.distillation_loss.loss_mode=$DISTILLATION_LOSS_MODE
            distillation.distillation_loss.use_policy_gradient=$USE_POLICY_GRADIENT
            distillation.distillation_loss.policy_loss_mode=$POLICY_LOSS_MODE
            distillation.distillation_loss.clip_ratio_low=$CLIP_RATIO_LOW
            distillation.distillation_loss.clip_ratio_high=$CLIP_RATIO_HIGH
            distillation.distillation_loss.loss_max_clamp=$LOSS_MAX_CLAMP
            distillation.distillation_loss.log_prob_min_clamp=$LOG_PROB_MIN_CLAMP
            distillation.distillation_loss.use_task_rewards=$USE_TASK_REWARDS
        )
    fi

    if [ "$DRY_RUN" = "1" ]; then
        printf 'python3 -m verl.trainer.main_ppo'
        for a in "${ARGS[@]}"; do printf ' \\\n    %q' "$a"; done
        printf '\n\n'
        continue
    fi

    ray stop --force 2>/dev/null || true
    python3 -m verl.trainer.main_ppo "${ARGS[@]}"
done
