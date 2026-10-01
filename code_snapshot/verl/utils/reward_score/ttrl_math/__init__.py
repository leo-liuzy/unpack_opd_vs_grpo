# Copyright 2024 PRIME team and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except Exception in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Provides a math answer grading function with high recall.
Based on HF math_verify, verl, open reasoner zero, etc.
"""

from latex2sympy2_extended import latex2sympy
from sympy import simplify
from sympy.parsing.sympy_parser import parse_expr
import traceback

from .math_utils import extract_boxed_answer, is_latex_equal, grade_answer_mathd, grade_answer_sympy, timeout_ours

"""
This code is adapted from Entropy Machanism Recipe (https://github.com/volcengine/verl/tree/main/recipe/entropy/).
"""

def extract_answer(passage: str) -> str:
    if "\\boxed" in passage:
        return extract_boxed_answer(passage)
    return None


def grade(model_answer: str, gt_answer: str, fast: bool = True):
    if "\\boxed" in gt_answer:
        gt_answer = extract_answer(gt_answer)
    correct = grade_answer_mathd(model_answer, gt_answer) or grade_answer_sympy(model_answer, gt_answer)
    if not fast:
        # This mode further uses math_verify to recall originally false positives.
        # Will be a bit slower, and sensitive to bad inputs.
        correct = correct or is_latex_equal(
            model_answer,
            gt_answer,
        )
    return correct

@timeout_ours(timeout_seconds=10)
def simplify_expression_string(expression_string: str) -> str:
    try:
        sympy_expr = parse_expr(expression_string, transformations="all", evaluate=False)
        simplified_expr = simplify(sympy_expr)
        return str(simplified_expr)
    except TimeoutError:
        return expression_string
    except Exception as e:
        try:
            sympy_expr = latex2sympy(expression_string)
            simplified_expr = simplify(sympy_expr)
            return str(simplified_expr)
        except TimeoutError:
            return expression_string
        except Exception as e:
            return expression_string

def compute_score(model_response, gt_answer, fast=False):
    model_answer = extract_answer(model_response)

    if model_answer is None:
        return {
            "score": 0.0,
            "format_score": 0.0,
            "acc": False,
            "extracted_gt": gt_answer,
            "pred": "",
            "feedback": "",
        }
        # return 0.0, 0.0  # Cannot even parse anything.
    is_correct = False
    if isinstance(gt_answer, float) or isinstance(gt_answer, int):
        gt_answer = str(gt_answer)
    if isinstance(gt_answer, str):
        is_correct = grade(model_answer, gt_answer, fast)
    elif isinstance(gt_answer, list):
        is_correct = False
        for gt in gt_answer:
            is_correct |= grade(model_answer, gt, fast)
    if is_correct:
        return {
            "score": 1.0,
            "format_score": 1.0,
            "acc": True,
            "extracted_gt": gt_answer,
            "pred": model_answer,
            "feedback": "",
        }
    else:
        return {
            "score": 0.0,
            "format_score": 1.0,
            "acc": False,
            "extracted_gt": gt_answer,
            "pred": model_answer,
            "feedback": "",
        }


# Recognized values for LLM_FEEDBACK_LEVEL when this reward function is used as
# the custom reward in an SDPO(gt) run. ttrl_math does NOT make LLM judge calls
# (those live in verl.utils.reward_score.feedback); the env var here only
# selects between "no feedback emitted" (the OPD-paper default) and "GT-letter
# feedback emitted for the SDPO reprompt machinery" (sdpo_gt on the OPD-paper
# setup). Any other value is a wiring bug — fail loudly rather than silently
# fall back to no-feedback.
_TTRL_MATH_VALID_FEEDBACK_LEVELS = {"little", "answer-only"}


def reward_func(
    data_source, solution_str, ground_truth, extra_info=None, sandbox_fusion_url=None, concurrent_semaphore=None
):
    try:
        res = compute_score(solution_str, str(ground_truth))

        if isinstance(res, dict):
            import os

            feedback_level = os.environ.get("LLM_FEEDBACK_LEVEL", "little")
            assert feedback_level in _TTRL_MATH_VALID_FEEDBACK_LEVELS, (
                f"ttrl_math.reward_func only supports LLM_FEEDBACK_LEVEL in "
                f"{sorted(_TTRL_MATH_VALID_FEEDBACK_LEVELS)}; got "
                f"{feedback_level!r}. ttrl_math has no LLM judge; for "
                f"LLM-based feedback levels use "
                f"verl.utils.reward_score.feedback instead."
            )

            assert "feedback" in res, (
                "ttrl_math.compute_score must always return a 'feedback' key "
                "(possibly empty) so the SDPO reward_extra_info channel sees a "
                "consistent shape across batch items."
            )

            is_validate = bool((extra_info or {}).get("validate", False))

            # SDPO(gt) parity with the MMLU-Pro T2 treatment (FB_LEVEL=
            # answer-only): inject "The correct answer is: \boxed{<gt>}" into
            # the teacher reprompt for INCORRECT training rollouts. We do NOT
            # emit feedback during validation (keeps val acc clean) and we do
            # NOT emit feedback for correct rollouts (the GT would tell the
            # teacher nothing the student doesn't already know, and matches the
            # feedback/math.py correctness_feedback semantics). The GT is
            # wrapped in \boxed{} to match the answer surface ttrl_math grades
            # against and that the student is trained to emit.
            if feedback_level == "answer-only" and not is_validate and not res["acc"]:
                res["feedback"] = f"The correct answer is: \\boxed{{{ground_truth}}}"
            return res
        elif isinstance(res, (int, float, bool)):
            return float(res)
        else:
            return float(res[0])
    except Exception as e:
        print(f"[ERROR] Error in process_completion for task : {str(e)}")
        traceback.print_exc()
        raise