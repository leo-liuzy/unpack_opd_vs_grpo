# Dataset prompts (train & eval) — verbatim

These are the exact prompts baked into the parquet files used by this suite.
Reward grading (both training and eval) uses `verl.utils.reward_score.ttrl_math`
(`reward_func`), a `\boxed{}`/answer-matching math grader.

---

## Train — DAPO-Math-17k (`dataset/train/dapo_math_17k_train_17661.parquet`)

- 17,661 rows, `data_source="math_dapo"`, `ability="MATH"`,
`reward_model.style="rule-lighteval/MATH_v2"`, ground-truth in
`reward_model.ground_truth`.
- Prompt format = the standard **DAPO "Answer:" template**, delivered as a single
`user` message (no system message). The template wraps each problem as:

```
Solve the following math problem step by step. The last line of your response
should be of the form Answer: $Answer (without quotes) where $Answer is the answer
to the problem.

{problem}

Remember to put your answer on its own line after "Answer:".
```

Concrete row 0 (`prompt` field):

```json
[
  {
    "role": "user",
    "content": "Solve the following math problem step by step. The last line of your response should be of the form Answer: $Answer (without quotes) where $Answer is the answer to the problem.\n\nA triangle has side lengths of $x$, $75$, and $100$, where $x < 75$, and altitudes of lengths $y$, $28$, and $60$, where $y < 28$. What is the value of $x + y$?\n\nRemember to put your answer on its own line after \"Answer:\"."
  }
]
```

`reward_model = {"ground_truth": "56", "style": "rule-lighteval/MATH_v2"}`

**Provenance:** the public **DAPO-Math-17k** dataset, preprocessed into verl parquet
format with the DAPO prompt above, then 256 rows held out
(`dataset/train/heldout_256.parquet`) leaving 17,661 training rows.

---



## Eval — AIME24 / AIME25 / AMC23 (`dataset/eval/*.parquet`)

- 30 + 30 + 83 = **143 prompts**. `data_source ∈ {AIME24, AIME25, AMC23}`,
`ability="math"`, `reward_model.style="rule"`, ground-truth in
`reward_model.ground_truth`.
- Prompt format = a `\boxed{}` **system prompt** + the bare problem as the user
message:

```
System: Please reason step by step, and put your final answer within \boxed{}.
User:   {problem}
```

Concrete AIME24 row 0 (`prompt` field):

```json
[
  {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
  {"role": "user",   "content": "Every morning Aya goes for a $9$-kilometer-long walk ... Find the number of minutes the walk takes her, including the $t$ minutes spent in the coffee shop."}
]
```

`reward_model = {"ground_truth": "204", "style": "rule"}`

**Provenance / builder:** the shipped `dataset/eval/*.parquet` are ready to use.
They were produced by `dataset/build_opd_eval_train_prompt.py` from source parquets
in which the instruction `"... Please reason step by step, and put your final answer within \boxed{}."` is a **suffix of the user message** (i.e. the problem formatted
with the prompt string below). The builder asserts that exact suffix, strips it off,
and re-emits it as a standalone `system` message. Rebuilding is only needed if you
start from your own source files:

```bash
python dataset/build_opd_eval_train_prompt.py \
    --source-dir <dir with aime24/aime25/amc23.parquet, suffix format> \
    --output-dir <out dir>
```

The builder validates row counts (30/30/83), schema, and that every non-prompt
column is byte-identical to the source.

---



## Central prompt string (reference)

The math prompt string the source eval sets were formatted with (before the builder
moved the instruction into the system message):

```python
PROMPT = "{problem}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}."
```

