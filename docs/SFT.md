# Student SFT checkpoint

We do not release the SFT student checkpoint. This page describes how we made
it so you can train your own and pass it as `STUDENT_MODEL`.

The recipe mirrors tinker-cookbook's `off_policy_reasoning.py`, the off-policy
SFT step of Thinking Machines' on-policy distillation post.

## Dataset

- Source: [`open-thoughts/OpenThoughts3-1.2M`](https://huggingface.co/datasets/open-thoughts/OpenThoughts3-1.2M),
  all 1,200,000 rows (all domains).
- Every row is a single `human` -> `gpt` turn. We map it to two string columns:
  `prompt` (the human message) and `response` (the gpt message, verbatim,
  including its `<think>...</think>` trace).
- We shuffle the rows once with a fixed seed and write a single parquet file.
- There is no filtering. Sequences longer than 16384 tokens are truncated on the
  right at train time.

## Training config

We use full fine-tuning (no LoRA) of `Qwen/Qwen3-1.7B-Base` with verl's
single-turn prompt/response SFT trainer (`verl.trainer.fsdp_sft_trainer`, verl
v0.7.0) on 8 GPUs. The loss covers response tokens only.

| setting | value |
|---|---|
| chat template | Qwen3 template applied to the prompt, `enable_thinking=true` (the responses already contain `<think>`, so this avoids a second wrapper) |
| max length | 16384, `truncation=right` |
| global batch | 128 sequences |
| steps | 3000 (≈384k sequences, about one third of an epoch) |
| optimizer | AdamW, betas (0.9, 0.95), weight decay 0, grad clip 1.0 |
| LR schedule | no warmup, linear decay from peak to 0 over the 3000 steps |
| peak LR | **1e-5** for the runs in this package and the blog post; we **recommend 1e-4** (see below) |
| precision / sharding | bf16 params, FSDP2, gradient checkpointing, remove-padding |
| checkpoint | HF-format model at step 3000 is the student (`STUDENT_MODEL`) |

Two details matter if you use a different trainer:
- The schedule must decay over the 3000 steps actually run, not over the full
  epoch.
- verl v0.7.0's trainer has no `linear` option and sizes its schedule by epoch,
  so we added both.

## Learning rate: 1e-5 vs 1e-4

All GRPO and OPD runs in this package start from the **lr 1e-5** SFT checkpoint.
We later found that **lr 1e-4** gives a noticeably stronger SFT init, so we
recommend it for new runs. We repeated the GRPO-vs-OPD comparison from the 1e-4
init, and the conclusion about GRPO vs OPD does not change. Absolute scores
shift with the init, so compare against your own step-0 eval
(`EVAL_MODEL_PATH=<student> EVAL_TAG=init bash scripts/run_standalone_eval.sh`).
