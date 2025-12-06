# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# try:
#     from math_verify.errors import TimeoutException
#     from math_verify.metric import math_metric
#     from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig
# except ImportError:
#     print("To use Math-Verify, please install it first by running `pip install math-verify`.")
from math_verify.errors import TimeoutException
from math_verify.metric import math_metric
from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig

def compute_score(model_output: str, ground_truth: str, timeout_score: float = 0) -> bool:
    verify_func = math_metric(
        gold_extraction_target=(LatexExtractionConfig(),),
        pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
    )
    ret_score = 0.0

    # Wrap the ground truth in \boxed{} format for verification
    ground_truth_boxed = "\\boxed{" + ground_truth + "}"
    try:
        ret_score, _ = verify_func([ground_truth_boxed], [model_output])
    except Exception:
        pass
    except TimeoutException:
        ret_score = timeout_score

    return ret_score

if __name__ == "__main__":
    math_response= "I need to find the radius of a circle given information about its area and circumference.\nLet me set up the problem with the given information.\nGiven information:\n- Area of circle = \$x\$ square units\n- Circumference of circle = \$y\$ units\n- \$x + y = 80\pi\$\nLet \$r\$ be the radius of the circle.\nUsing the formulas for area and circumference:\n- Area: \$x = \pi r^2\$\n- Circumference: \$y = 2\pi r\$\nSubstituting into the given equation:\n\$x + y = 80\pi\$\n\$\pi r^2 + 2\pi r = 80\pi\$\nI can factor out \$\pi\$ from the left side:\n\$\pi(r^2 + 2r) = 80\pi\$\nDividing both sides by \$\pi\$:\n\$r^2 + 2r = 80\$\nRearranging to standard form:\n\$r^2 + 2r - 80 = 0\$\nNow I'll solve this quadratic equation using the quadratic formula or factoring.\nLooking for two numbers that multiply to \$-80\$ and add to \$2\$:\n- The numbers are \$10\$ and \$-8\$ since \$10 \times (-8) = -80\$ and \$10 + (-8) = 2\$\nSo I can factor:\n\$(r + 10)(r - 8) = 0\$\nThis gives me:\n\$r + 10 = 0\$ or \$r - 8 = 0\$\n\$r = -10\$ or \$r = 8\$\nSince the radius must be positive, \$r = 8\$.\nLet me verify this answer:\n- Area: \$x = \pi(8)^2 = 64\pi\$\n- Circumference: \$y = 2\pi(8) = 16\pi\$\n- Sum: \$x + y = 64\pi + 16\pi = 80\pi\$ ✓\nTherefore, the radius of the circle is \\boxed{8} units."
    gt = "8"
    # boxed = last_boxed_only_string(math_response)
    # print('Boxed:',boxed)
    math_score = compute_score(math_response, gt)
    print(math_score)