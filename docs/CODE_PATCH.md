# Code base, patch, and OPD file map

## Where the code comes from

The base is **public upstream verl** (`https://github.com/volcengine/verl.git`) at
commit **`7aed6b2`** (`verl 0.8.0.dev0`). At this commit upstream already contains the
OPD (on-policy distillation) trainer (`verl/trainer/distillation/`,
`verl/workers/config/distillation.py`), so no fork is needed.

```
HEAD:   7aed6b230776f963fa09509c10d9c3a767d1102c  [ci] fix: FSDP actor/critic ci fail (#6550)
```

On top of that commit, our runs add:

| What | Where in this package | Where it goes in verl |
|---|---|---|
| 1-file patch to the distillation loss | `patches/losses_global_batch_info.patch` | `verl/trainer/distillation/losses.py` (`git apply`) |
| `ttrl_math` reward module (new) | `code_snapshot/verl/utils/reward_score/ttrl_math/` | `verl/utils/reward_score/ttrl_math/` (copy) |
| Launcher, eval tool, train/eval seed scripts | `scripts/` | run in place from this package (no copy needed) |

```bash
git clone https://github.com/volcengine/verl.git && cd verl
git checkout 7aed6b230776f963fa09509c10d9c3a767d1102c
git apply <package>/patches/losses_global_batch_info.patch
cp -r <package>/code_snapshot/verl/utils/reward_score/ttrl_math verl/utils/reward_score/
pip3 install --no-deps -e .
```

The other files under `code_snapshot/` are reference copies of the upstream OPD
sources. `trainer/distillation/losses.py` already has the patch applied; the rest
are byte-identical to upstream `7aed6b2`, so you can verify your checkout with `diff -r`.

## The patch — `patches/losses_global_batch_info.patch`

One change, in `verl/trainer/distillation/losses.py`, inside the policy-gradient
branch of `distillation_loss()`. It reads the global-batch normalizers
(`dp_size`, `batch_num_tokens`, `global_batch_size`) **directly from the batch**
instead of copying them out of `config.global_batch_info` — the latter lags by one
micro-batch, because `ppo_loss` fills that dict only *after*
`distillation_ppo_loss` has already called this function. Apply with:

```bash
cd <verl>
git apply <package>/patches/losses_global_batch_info.patch
```

## OPD file map (what implements OPD)

| Path | Role |
|---|---|
| `verl/trainer/distillation/losses.py` | Loss dispatch (`get_distillation_loss_fn`), the reverse-KL estimator `compute_distillation_loss_reverse_kl_estimator` (registered for names `kl,k1,abs,mse,k2,low_var_kl,k3`), and the policy-gradient wrapping used when `use_policy_gradient=True`. **This is the file the patch touches.** |
| `verl/trainer/distillation/{fsdp,megatron}/losses.py` | Backend-specific loss impls (e.g. `compute_forward_kl_topk`) + registry. FSDP is the backend used here. |
| `verl/trainer/distillation/__init__.py` | Re-exports the loss registry. |
| `verl/workers/config/distillation.py` | `DistillationConfig` dataclass: `enabled`, `loss_mode`, `use_policy_gradient`, `policy_loss_mode`, `use_task_rewards`, `clip_ratio_{low,high}`, `loss_max_clamp`, `log_prob_min_clamp`, `teacher_models.*`, `n_gpus_per_node`, ... All the `distillation.*` overrides in `configs/opd_8b_seed123_dryrun.txt` bind here. |
| `verl/utils/reward_score/ttrl_math/` (**added**, not upstream) | The `\boxed{}` math grader (`reward_func`, `compute_score`, `grader.py`, `math_utils.py`, `math_normalize.py`). Used as GRPO's outcome reward **and** as the eval reward. |

The GRPO runs set `distillation.enabled=False` and use the stock verl GRPO path
(`adv_estimator=grpo`) with the `ttrl_math` reward; the OPD runs set
`distillation.enabled=True` and the `distillation.*` knobs above. Both are driven by
the single launcher `scripts/run_qwen3_1p7b_tml_opd_dapo_math_seed.sh` via the
`ALGOS` env knob.

## What "loss_mode=k1" means

`loss_mode=k1` selects `compute_distillation_loss_reverse_kl_estimator` with the
**k1** estimator of the reverse KL between student and teacher (the log-ratio
estimator, which can be negative — hence `log_prob_min_clamp`/`loss_max_clamp`). With
`use_policy_gradient=True` the negative distillation loss is used as a per-token
reward (the On-Policy Distillation recipe), and `use_task_rewards=False` means **no**
outcome/task reward is mixed in — OPD is pure distillation from the teacher.
