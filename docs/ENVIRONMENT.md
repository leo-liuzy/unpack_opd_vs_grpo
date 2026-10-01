# Environment

## Hardware
- 1 node, **8 GPUs** with ≥80 GB each (we used H200s).
- GRPO: 8 GPUs training. OPD: 4 training + 4 serving the Qwen3-8B teacher (vLLM).

## Environment — use verl's released environment
Use verl's **official released environment**. Follow verl's install guide:
<https://verl.readthedocs.io/en/latest/start/install.html>. Options:

- **Prebuilt Docker (recommended)** — use verl's published **vLLM** image on
  DockerHub: `verlai/verl:vllm011.latest`. Then install verl from the checkout:
  `pip3 install --no-deps -e .`.
- **From source** — into a matching CUDA/torch env:
  ```
  git clone https://github.com/volcengine/verl && cd verl
  pip3 install -e ".[vllm]"
  ```
  (verl requires **vLLM ≥ 0.18**.)

Check out verl at the pinned commit with the patch and the `ttrl_math` module
added — see `docs/CODE_PATCH.md`. The verl source is **not** part of the image; it
always lives in the checkout.

The table below is the exact stack **our** runs used; treat it as the target versions. Any current verl release
environment that satisfies verl's guide should reproduce the suite — the OPD/GRPO
behavior does not depend on these exact patch versions.

## Code base
Public upstream verl at commit `7aed6b2` (`verl 0.8.0.dev0`), plus
`patches/losses_global_batch_info.patch` and the added `ttrl_math` reward module.
Details and the OPD file map are in `docs/CODE_PATCH.md`.

> Run every training/eval command **from the verl checkout root**. Launched from
> a home directory, `import verl` resolves to a PEP-420 namespace stub whose
> `verl.model_merger` is missing, and the eval's FSDP-shard merge step fails.

## Pinned package versions (as run)
| package | version |
|---|---|
| verl | 0.8.0.dev0 |
| torch | 2.11.0+cu130 |
| vllm | 0.20.2 |
| transformers | 5.3.0 |
| ray | 2.55.1 |
| flash_attn | 2.8.3 |
| latex2sympy2_extended | 1.11.0 |
| math_verify | 0.9.0 |
| pylatexenc | 2.10 |
| sympy | 1.14.0 |
| antlr4-python3-runtime | 4.9.3 |

## Non-interactive launch environment (Ray must see the GPUs)
On some container setups (e.g. Kubernetes GPU device plugins that mount the driver
under `/usr/local/nvidia`), a **non-interactive** launch (detached process, `nohup`,
ssh `bash -c`) does not inherit the NVIDIA userspace paths that an interactive login
shell sets up, and Ray then initializes with **0 GPUs**. If `nvidia-smi` already works
in your launch shell, skip this. Export these before any launch/eval (the
training, seed, and eval scripts in `scripts/` already do this at the top):

```bash
export PATH=/usr/local/nvidia/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/nvidia/lib64:${LD_LIBRARY_PATH:-}
```

`nvidia-smi` should list all 8 GPUs after this. (The FSDP-shard merge in the eval
tool additionally relies on `/usr/local/cuda/compat` being reachable via
`LD_LIBRARY_PATH` when torch's CUDA is newer than the driver; the scripts add it if it exists.)

## Inference engine
Both training rollouts and the eval use **vLLM** (`rollout.name=vllm`), so verl's
vLLM environment (`verlai/verl:vllm011.latest`) is all you need — this is the engine
our runs used.

> Caveat specific to our pinned stack: on the exact versions above (vLLM 0.20.2 /
> torch 2.11.0 / prebuilt flash-attn 2.8.3), adding `sglang` on top pulled torch
> 2.11 → 2.13 and broke the prebuilt flash-attn. Stay on the vLLM environment.

## Reward-grader dependencies (easy to miss)
The `ttrl_math` reward (`verl/utils/reward_score/ttrl_math/`, used for both GRPO's
outcome reward and all scored evals) imports `latex2sympy2_extended`, `math_verify`,
and `pylatexenc`. verl's images may **not** ship these — training
then crashes at `RewardLoopWorker.__init__` with
`ModuleNotFoundError: No module named 'latex2sympy2_extended'` *after* vLLM has
already loaded weights (so an early "weights loaded" log looks healthy). Install:

```bash
python -m pip install --break-system-packages latex2sympy2_extended math_verify pylatexenc
```

`--break-system-packages` is only needed on PEP-668 externally-managed Python
installs (drop it inside a venv/conda env). `sympy` and
`antlr4-python3-runtime` are typically already present.

Sanity check the grader before a long run:
```python
from verl.utils.reward_score.ttrl_math import reward_func   # must import cleanly
```
