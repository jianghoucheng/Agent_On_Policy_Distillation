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
import logging
import re
from typing import Any

import datasets
import requests

from verl.tools.base_tool import OpenAIFunctionToolSchema
from verl.tools.sandbox_fusion_tools import SandboxFusionTool
from verl.tools.schemas import ToolResponse
from verl.utils.dataset import RLHFDataset
from verl.utils.reward_score import math_dapo
from verl.utils.rollout_trace import rollout_trace_op

logger = logging.getLogger(__name__)


class CustomSandboxFusionTool(SandboxFusionTool):
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self.code_pattern = re.compile(r"```python(.*?)```", re.DOTALL)
        self.use_local_flask = config.get("use_local_flask", False)
        self.local_flask_timeout = config.get("local_flask_timeout", self.default_timeout)

    @rollout_trace_op
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[str, float, dict]:
        code = parameters["code"]
        matches = self.code_pattern.findall(code)
        if matches:
            code = matches[0].strip()

        # NOTE: some script may not explicitly print result, we need to add a print statement to the end of the script
        lines = code.split("\n")
        for i, line in reversed(list(enumerate(lines))):
            if line == "":
                continue
            if not lines[i].startswith("print"):
                lines[i] = f"print({line})"
            break
        code = "\n".join(lines)

        timeout = parameters.get("timeout", self.default_timeout)
        language = parameters.get("language", self.default_language)
        if not isinstance(code, str):
            code = str(code)

        result = await self.execution_pool.execute.remote(self.execute_code, instance_id, code, timeout, language)
        # sandbox has no score or metrics, use Nones
        return result, None, None

    def execute_code(self, instance_id, code, timeout=30, language="python"):
        if self.use_local_flask:
            return self._execute_via_local_flask(code, timeout, language)
        return super().execute_code(instance_id, code, timeout=timeout, language=language)

    def _execute_via_local_flask(self, code: str, timeout: int, language: str) -> ToolResponse:
        request_timeout = min(timeout, self.local_flask_timeout)
        payload = {"code": code, "language": language}
        try:
            response = requests.post(self.sandbox_fusion_url, json=payload, timeout=request_timeout)
            response.raise_for_status()
            data = response.json()
        except requests.Timeout:
            logger.warning("Local sandbox execution timed out after %s seconds", request_timeout)
            return ToolResponse(text="Execution timed out in local sandbox")
        except requests.RequestException as exc:
            logger.error("Local sandbox request failed: %s", exc)
            return ToolResponse(text=f"Local sandbox request failed: {exc}")

        if isinstance(data, dict):
            if "error" in data:
                errmsg = data.get("error", "Unknown error")
                stderr = data.get("stderr", "")
                stdout = data.get("stdout", "")
                combined = f"{stdout}{stderr}\n{errmsg}".strip()
                return ToolResponse(text=combined or errmsg)

            stdout = data.get("stdout", "") or ""
            stderr = data.get("stderr", "") or ""
            combined = f"{stdout}{stderr}".strip()
            if not combined:
                return ToolResponse(text="no stdout here")
            return ToolResponse(text=combined)

        logger.error("Unexpected response from local sandbox: %s", data)
        return ToolResponse(text="Unexpected response from local sandbox")


answer_format = """\nThe answer format must be: \\boxed{'The final answer goes here.'}"""


class CustomRLHFDataset(RLHFDataset):
    """Custom dataset class to process Maxwell-Jia/AIME_2024, yentinglin/aime_2025 datasets."""

    def _read_files_and_tokenize(self):
        dataframes = []
        for parquet_file in self.data_files:
            # read parquet files and cache
            dataframe = datasets.load_dataset(parquet_file)["train"]
            data_source = "/".join(parquet_file.split("/")[-2:])
            if data_source in ["Maxwell-Jia/AIME_2024", "yentinglin/aime_2025"]:
                dataframe = dataframe.map(
                    self.map_fn, fn_kwargs={"data_source": data_source}, remove_columns=dataframe.column_names
                )
            else:
                dataframe = dataframe.map(self.map_fn2, num_proc=16)
            dataframes.append(dataframe)
        self.dataframe: datasets.Dataset = datasets.concatenate_datasets(dataframes)

        print(f"dataset len: {len(self.dataframe)}")

    def map_fn(self, row: dict, *, data_source: str = None):
        if data_source == "Maxwell-Jia/AIME_2024":
            problem, answer = row["Problem"], row["Answer"]
        elif data_source == "yentinglin/aime_2025":
            problem, answer = row["problem"], row["answer"]

        prompt = problem + answer_format
        data = {
            "data_source": data_source.split("/")[1].lower(),  # aime_2024, aime_2025
            "prompt": [{"role": "user", "content": prompt}],
            "ability": "MATH",
            "reward_model": {"ground_truth": str(answer)},
            "agent_name": "tool_agent",
        }
        return data

    def map_fn2(self, row: dict):
        content = row["prompt"][0]["content"]
        row["prompt"][0]["content"] = content + answer_format
        row["agent_name"] = "tool_agent"
        return row


def compute_score(data_source, solution_str, ground_truth, extra_info, **kwargs):
    # use \\boxed{...} answer
    result = math_dapo.compute_score(solution_str, ground_truth, strict_box_verify=True)

    # encourage model to call tools
    num_turns = extra_info["num_turns"]
    if result["score"] < 0:
        tool_call_reward = (num_turns - 2) / 2 * 0.1
        result["score"] = min(-0.6, result["score"] + tool_call_reward)

    if result["pred"] is None:
        result["pred"] = ""

    return result

# def compute_score_opd(
#     data_source,
#     solution_str,
#     ground_truth,
#     extra_info,
#     call_bonus: float = 0.1,
#     success_bonus: float = 0.2,
#     extra_success_bonus: float = 0.1,
# ):
#     result = _math_score(solution_str, ground_truth)

#     extra_info = extra_info or {}
#     num_calls = int(extra_info.get("num_tool_calls", 0) or 0)
#     num_success = int(extra_info.get("num_tool_success", 0) or 0)

    
#     if result["score"] < 0:
#         call_reward = call_bonus if num_calls > 0 else 0.0
#         success_reward = 0.0
#         if num_success > 0:
#             success_reward = success_bonus + extra_success_bonus * max(num_success - 1, 0)
#         total_bonus = call_reward + success_reward
#         result["score"] = min(-0.5, result["score"] + total_bonus)

#     return result
