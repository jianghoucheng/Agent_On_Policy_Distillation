# Copyright 2024 PRIME team and/or its affiliates
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

import json
import re
import traceback
from verl.utils.reward_score.livecodebench import lcb_compute_score, prepare_unit_test_data
import os, pickle
from verl.utils.reward_score.livecodebench.lcb_runner.benchmarks.code_generation import CodeGenerationProblem
from verl.utils.reward_score.livecodebench.lcb_runner.evaluation.compute_code_generation_metrics import codegen_metrics
from verl.utils.reward_score.livecodebench.lcb_runner.evaluation.pass_k_utils import extract_instance_results
from math_verify import parse, verify
import tempfile
import subprocess
from contextlib import contextmanager
import signal
import ast
import numpy as np
from verl.utils.reward_score.sandbox_fusion.utils import check_correctness
from typing import Optional

IMPORT_PROMPT='''from typing import *

from functools import *
from collections import *
from itertools import *
from heapq import *
from bisect import *
from string import *
from operator import *
from math import *
import math
import datetime
inf = float('inf')

'''

livecodebench_dir = os.environ.get("LIVECODEBENCH_DATA_PATH", None)
# if livecodebench_dir is None:
#     raise ValueError("LIVECODEBENCH_DATA_PATH is not set")

def last_boxed_only_string(string: str) -> Optional[str]:
    """Extract the last LaTeX boxed expression from a string.

    Args:
        string: Input string containing LaTeX code

    Returns:
        The last boxed expression or None if not found
    """
    idx = string.rfind("\\boxed{")
    if idx < 0:
        return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0

    # while i < len(string):
    for i in range(idx+4, len(string)):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    return string[idx : right_brace_idx+1] if right_brace_idx is not None else None


@contextmanager
def timeout_run(seconds):
    def signal_handler(signum, frame):
        raise TimeoutError("代码执行超时")
    
    # 注册信号处理器
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    
    try:
        yield
    finally:
        signal.alarm(0)

def convert_function_to_class_method(raw_code: str, function_name: str) -> str:
    # 解析原始代码为 AST
    tree = ast.parse(raw_code)
    target_func = None
    new_body = []
    # 遍历顶层节点，保留非目标函数的代码
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            target_func = node
        else:
            new_body.append(node)
    
    if target_func is None:
        return None

    if not (target_func.args.args and target_func.args.args[0].arg == "self"):
        self_arg = ast.arg(arg="self", annotation=None)
        target_func.args.args.insert(0, self_arg)    
    class_def = ast.ClassDef(
        name="Solution",
        bases=[],
        keywords=[],
        body=[target_func],
        decorator_list=[]
    )
    
    new_body.append(class_def)
    tree.body = new_body
    
    # 使用 ast.unparse 将 AST 转换为代码字符串（Python 3.9+支持）
    new_code = ast.unparse(tree)
    return new_code


def math_verify_reward_function(solution_str, ground_truth):

    ground_truth = [ground_truth] if isinstance(ground_truth, str) else ground_truth
    
    # 0 in case parsing cannot be completed
    try:
        math_verify_parsed = parse(solution_str, parsing_timeout=5)
    except Exception:
        return_dict ={
        "score": -1.0,
        "acc": False,
        "pred": None,
        }
        return return_dict
    
    # 0 if parsing is problematic
    if len(math_verify_parsed) < 2:
        return_dict ={
        "score": -1.0,
        "acc": False,
        "pred": None,
        }
        return return_dict
    
    # We perform a quick string match first
    if math_verify_parsed[1] in ground_truth:
        return_dict ={
        "score": 1.0,
        "acc": True,
        "pred": math_verify_parsed[1],
        }
        return return_dict
    
    # We now fallback to semantic verification
    for gt in ground_truth:
        try:
            if verify(
                parse(f"\\boxed{{{gt}}}", parsing_timeout=5),
                math_verify_parsed,
                timeout_seconds=5,
            ):
                return_dict ={
                "score": 1.0,
                "acc": True,
                "pred": math_verify_parsed[1],
                }
                return return_dict
        except Exception:
            continue

    
    # Very unlikely to be correct after the above matches
    return  {"score": -1.0, "acc": False, "pred": math_verify_parsed[1]}



def compute_score(completion, test_cases, task=None, timeout=30, is_long_penalty=False, is_binary_reward=False, is_power4_reward=False):
    # try to get code solution from completion. if the completion is pure code, this will not take effect.
    # solution = completion.split('```python')[-1].split('```')[0]

    if "</think>" in completion:
        solution_str = completion.split("</think>")[1]
    else:
        solution_str = completion
    test_cases = str(test_cases)
    if 'import_prefix' in test_cases:
        solutions = re.findall(r"```python\n(.*?)```", solution_str, re.DOTALL)
        if len(solutions) == 0:
            return {"score": -1.0, "acc": False, "pred": None}
        try:
            solution = solutions[-1]
            # Model outputs may escape newlines/tabs (e.g., "\\n    if valid"), which break parsing.
            solution = solution.replace("\\n", "\n").replace("\\t", "\t")
            tree = ast.parse(solution)
            try:
                test_cases = json.loads(test_cases)
            except:
                tmp = json.dumps(test_cases)
                test_cases = json.loads(tmp)
            solution = test_cases["import_prefix"] + solution
            test_code = [x for x in test_cases['test_code'].split("\n") if x != ""]
            unit_test_result = []
            unit_test_metadata = []
            for i in range(1, len(test_code)):
                cur_solution = solution
                cur_solution += "\n" + test_code[0] + test_code[i]
                cur_solution += "\ncheck({})".format(test_cases['entry_point'])
                try:
                    # 执行代码的逻辑
                    success = False
                    message = None
                    ## Add Sandbox Fusion API
                    metrics = check_correctness(
                            sandbox_fusion_url="https://sd4a8bq819ma6giel1p30.apigateway-cn-beijing.volceapi.com/run_code",
                            in_outs={'inputs':["prefix"],"outputs":["prefix"]},
                            generation=cur_solution,
                            timeout=timeout
                        )
                    if metrics[1][0]['api_response']['run_result']['return_code'] == 0:
                         unit_test_result.append(True)
                         unit_test_metadata.append(f"成功")
                    else:
                         unit_test_result.append(False)
                         unit_test_metadata.append(f"执行错误: {metrics[1][0]['stderr']}")
                except TimeoutError:
                    print("代码执行超时")
                    traceback.print_exc(10)
                    unit_test_result.append(False)
                    unit_test_metadata.append("代码执行超时")
                except Exception as e:
                    print(f"执行异常: {str(e)}")
                    unit_test_result.append(False)
                    unit_test_metadata.append("执行异常")
                    
            if is_binary_reward:
                return {"score": 1.0 if all(unit_test_result) else -1.0, "acc": all(unit_test_result), "pred": solution}
            else:
                if is_power4_reward:
                    return {"score": sum(unit_test_result)/len(unit_test_result)**4, "acc": all(unit_test_result), "pred": solution}
                else:
                    return  {"score": sum(unit_test_result)/len(unit_test_result), "acc": all(unit_test_result), "pred": solution}

        except Exception as e:
            traceback.print_exc(10)
            return {"score": -1.0, "acc": False, "pred": None}

    elif "inputs" in test_cases:
        try:
            solutions = re.findall(r"```python\n(.*?)```", solution_str, re.DOTALL)
            if len(solutions) == 0:
                return {"score": -1.0, "acc": False, "pred": None}
            else:
                solution = solutions[-1]
                try:
                    solution = solution.replace("\\n", "\n").replace("\\t", "\t")
                    tree = ast.parse(solution)
                except:
                    traceback.print_exc(10)
                    return {"score": -1.0, "acc": False, "pred": None}

            if isinstance(test_cases, str):
                try:
                    input_output = json.loads(test_cases)
                except:
                    tmp = json.dumps(test_cases)
                    input_output = json.loads(tmp)
            elif isinstance(test_cases, dict):
                input_output = test_cases
                test_cases = json.dumps(test_cases)
                
            else:
                assert False
            if "fn_name" in input_output and "class Solution" not in solution:
                solution = convert_function_to_class_method(solution, input_output["fn_name"])
                if not isinstance(solution, str):
                    return  {"score": -1.0, "acc": False, "pred": None}
                
            # Add Sandbox Fusion API
            metrics = check_correctness(
                sandbox_fusion_url = "https://sd4a8bq819ma6giel1p30.apigateway-cn-beijing.volceapi.com/run_code",
                in_outs=json.loads(json.dumps(test_cases)),
                generation=solution,
                timeout=timeout,
            )

            metrics = list(metrics)
            # print(metrics)
            fixed = []
            for e in metrics[0]:
                if isinstance(e, np.ndarray):
                    e = e.item(0)
                if isinstance(e, np.bool_):
                    e = bool(e)
                fixed.append(e)
            metrics[0] = fixed

            if is_binary_reward:
                return {"score": 1.0 if sum(metrics[0]) == len(metrics[0]) else -1.0, "acc": sum(metrics[0]) == len(metrics[0]), "pred": solution}
            else:
                if is_power4_reward:
                    return {"score": (sum((x if x in [False, True] else False) for x in metrics[0])/len(metrics[0]))**4, "acc": sum(metrics[0]) == len(metrics[0]), "pred": solution}
                else:
                    return {"score": sum((x if x in [False, True] else False) for x in metrics[0])/len(metrics[0]), "acc": sum(metrics[0]) == len(metrics[0]), "pred": solution}

        except Exception as e:
            traceback.print_exc(10)
            return {"score": -1.0, "acc": False, "pred": solution}
    else:
        try:
            last_boxed = last_boxed_only_string(solution_str)
            return math_verify_reward_function(last_boxed, test_cases)
        except:
            traceback.print_exc(10)
            return {"score": -1.0, "acc": False, "pred": None}    
