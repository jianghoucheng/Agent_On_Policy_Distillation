# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
"""
Offline sampling + filtering pipeline for ReTool-style distillation data.

Usage example:
      CUDA_VISIBLE_DEVICES=0,1,2,3 python recipe/retool/offline_distill_sampling.py \
    --input_parquet /data0/user10/ReTool-SFT/data/train-00000-of-00001.parquet \
    --output_parquet /data0/user10/ReTool-Distill/data/train-00000-of-00001.parquet \
    --teacher_model JoeYing/ReTool-Qwen-32B \
    --sandbox_config recipe/retool/sandbox_fusion_tool_config.yaml \
    --tensor_parallel_size 4 \
    --gpu_memory_utilization 0.9 \
    --temperature 0.7 \
    --top_p 0.95 \
    --max_new_tokens 16384 \
    --max_attempts 5
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from verl.tools.schemas import OpenAIFunctionToolSchema
from verl.utils.reward_score.sandbox_fusion.utils import _process_single_case

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline distillation sampler for ReTool.")
    parser.add_argument("--input_parquet", type=Path, required=True, help="Base prompt parquet file.")
    parser.add_argument("--output_parquet", type=Path, required=True, help="Output parquet path.")
    parser.add_argument("--teacher_model", type=str, default="JoeYing/ReTool-Qwen-32B", help="HF teacher id.")
    parser.add_argument("--teacher_model_path", type=str, default=None, help="Optional local teacher path.")
    parser.add_argument("--teacher_tokenizer_path", type=str, default=None, help="Optional local tokenizer path.")
    parser.add_argument("--sandbox_config", type=Path, required=True, help="Sandbox fusion YAML config.")
    parser.add_argument("--max_turns", type=int, default=8, help="Max multi-turn iterations per sample.")
    parser.add_argument("--max_samples", type=int, default=0, help="0 for full dataset, else random subset.")
    parser.add_argument("--temperature", type=float, default=0.7, help="Teacher sampling temperature.")
    parser.add_argument("--top_p", type=float, default=0.95, help="Teacher top-p.")
    parser.add_argument("--max_new_tokens", type=int, default=16384, help="Max new tokens per assistant turn.")
    parser.add_argument("--tensor_parallel_size", type=int, default=4, help="vLLM TP degree.")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9, help="vLLM GPU util.")
    parser.add_argument("--max_attempts", type=int, default=5, help="Max retries per sample for format compliance.")
    return parser.parse_args()


def _load_dataset(parquet_path: Path, max_samples: int) -> list[dict]:
    logger.info("Loading base prompts from %s", parquet_path)
    df = pd.read_parquet(parquet_path)
    records = df.to_dict(orient="records")
    if max_samples and max_samples < len(records):
        records = random.sample(records, max_samples)
    return records


class SandboxFusionClient:
    """Lightweight client replicating CustomSandboxFusionTool behaviour."""

    def __init__(self, config_path: Path):
        cfg = yaml.safe_load(config_path.read_text())
        tool_cfg = cfg["tools"][0]["config"]
        self.url = tool_cfg["sandbox_fusion_url"]
        self.timeout = tool_cfg.get("default_timeout", 30)
        self.memory_limit = tool_cfg.get("memory_limit_mb", 1024)
        self.language = tool_cfg.get("default_language", "python")
        self.headers = tool_cfg.get("headers")

    @staticmethod
    def _prepare_code(code: str) -> str:
        pattern = re.compile(r"```python(.*?)```", re.DOTALL)
        matches = pattern.findall(code)
        if matches:
            code = matches[0].strip()

        lines = code.split("\n")
        for i in range(len(lines) - 1, -1, -1):
            if lines[i].strip() == "":
                continue
            if not lines[i].lstrip().startswith("print"):
                lines[i] = f"print({lines[i]})"
            break
        return "\n".join(lines)

    def run_code(self, code: str) -> str:
        prepared = self._prepare_code(code)
        result, metadata = _process_single_case(
            case_index=0,
            stdin_data=None,
            expected_output=None,
            sandbox_fusion_url=self.url,
            generation=prepared,
            timeout=self.timeout,
            memory_limit_mb=self.memory_limit,
            language=self.language,
        )
        status = metadata.get("run_status")
        stdout = metadata.get("stdout", "")
        stderr = metadata.get("stderr", "")
        if status != "Finished":
            return f"[sandbox-error] status={status}, stderr={stderr}"
        return (stdout + stderr).strip()


class HermesToolParser:
    """Parse <tool_call> tokens used in ReTool Hermes format."""

    _pattern = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)

    def extract(self, text: str) -> tuple[str, list[dict]]:
        calls = []
        for match in self._pattern.findall(text):
            try:
                payload = json.loads(match)
                calls.append(payload)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to parse tool call payload %s (%s)", match, exc)
        cleaned = self._pattern.sub("", text).strip()
        return cleaned, calls


def _build_initial_messages(row: dict, tool_schema: list[dict]) -> list[dict[str, Any]]:
    first_user_content = None
    if "messages" in row:
        for msg in row["messages"]:
            if msg.get("role") == "user" and msg.get("content"):
                first_user_content = msg["content"]
                break
    if first_user_content is None and row.get("prompt"):
        first_user_content = row["prompt"][0]["content"]
    if first_user_content is None:
        raise ValueError("Input row does not contain a user prompt.")
    return [{"role": "user", "content": first_user_content, "tool_calls": None}]


def _append_tool_message(messages: list[dict[str, Any]], content: str) -> None:
    messages.append({"role": "tool", "content": content, "tool_calls": None})


def _append_assistant_message(messages: list[dict[str, Any]], content: str, tool_calls: Optional[list[dict]]) -> None:
    payload = {"role": "assistant", "content": content, "tool_calls": tool_calls or None}
    messages.append(payload)


def _build_prompt_text(
    tokenizer: AutoTokenizer,
    messages: list[dict[str, Any]],
    tool_schema: list[dict],
) -> str:
    kwargs = {"add_generation_prompt": True, "tokenize": False}
    if tool_schema:
        return tokenizer.apply_chat_template(messages, tools=tool_schema, **kwargs)
    return tokenizer.apply_chat_template(messages, **kwargs)


def _generate_assistant_output(
    llm: LLM,
    tokenizer: AutoTokenizer,
    messages: list[dict[str, Any]],
    sampling_params: SamplingParams,
    tool_schema: list[dict],
) -> str:
    prompt_text = _build_prompt_text(tokenizer, messages, tool_schema)
    outputs = llm.generate(prompt_text, sampling_params)
    if not outputs or not outputs[0].outputs:
        return ""
    text = outputs[0].outputs[0].text
    return text.strip()


def _normalize_tool_calls(raw_calls: Optional[list[dict[str, Any]]]) -> Optional[list[dict[str, Any]]]:
    if not raw_calls:
        return None
    normalized: list[dict[str, Any]] = []
    for call in raw_calls:
        if call is None:
            continue
        if isinstance(call, str):
            try:
                call = json.loads(call)
            except Exception:  # noqa: BLE001
                logger.warning("Failed to parse tool call payload string: %s", call)
                continue
        if not isinstance(call, dict):
            continue
        function = call.get("function") if isinstance(call.get("function"), dict) else {}
        name = function.get("name") or call.get("name") or "code_interpreter"
        arguments = function.get("arguments") or call.get("arguments") or {}
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except Exception:  # noqa: BLE001
                arguments = {"code": arguments}
        if not isinstance(arguments, dict):
            arguments = {"code": str(arguments)}
        normalized.append({"type": "function", "function": {"name": name, "arguments": arguments}})
    return normalized or None


def _has_boxed_answer(text: str) -> bool:
    return text is not None and "\\boxed" in text


def _append_final_answer_message(messages: list[dict[str, Any]]) -> None:
    last_assistant = None
    for msg in reversed(messages):
        if msg.get("role") == "assistant":
            last_assistant = msg
            break
    if last_assistant is None:
        return
    content = last_assistant.get("content", "")
    marker = content.rfind("\\boxed")
    if marker == -1:
        return
    boxed = content[marker:].strip()
    if not boxed:
        return
    if content.strip() == boxed:
        return
    messages.append({"role": "assistant", "content": boxed, "tool_calls": None})


def _format_sft_record(messages: list[dict[str, Any]], tool_schema: list[dict[str, Any]]) -> dict[str, Any]:
    normalized_messages = []
    for msg in messages:
        tool_calls = msg.get("tool_calls")
        if isinstance(tool_calls, list):
            tool_calls = np.array(tool_calls, dtype=object)
        normalized_messages.append(
            {
                "role": msg["role"],
                "content": msg.get("content", ""),
                "tool_calls": tool_calls,
            }
        )
    return {
        "messages": np.array(normalized_messages, dtype=object),
        "tools": deepcopy(tool_schema),
    }


def _sample_teacher_dialog(
    row: dict,
    *,
    llm: LLM,
    tokenizer: AutoTokenizer,
    sampling_params: SamplingParams,
    tool_schema: list[dict[str, Any]],
    sandbox_client: SandboxFusionClient,
    parser: HermesToolParser,
    max_turns: int,
) -> Optional[tuple[list[dict[str, Any]], int]]:
    messages = _build_initial_messages(row, tool_schema)
    tool_turns = 0

    for _ in range(max_turns):
        assistant_raw = _generate_assistant_output(llm, tokenizer, messages, sampling_params, tool_schema)
        if not assistant_raw:
            break

        assistant_clean, raw_calls = parser.extract(assistant_raw)
        normalized_calls = _normalize_tool_calls(raw_calls)
        _append_assistant_message(messages, assistant_clean, normalized_calls)

        if normalized_calls:
            tool_turns += 1
            call = normalized_calls[-1]
            arguments = call["function"].get("arguments", {})
            code = arguments.get("code")
            if code is None:
                logger.warning("Tool call missing 'code' argument.")
                return None
            if not isinstance(code, str):
                code = json.dumps(code)
            tool_output = sandbox_client.run_code(code)
            _append_tool_message(messages, tool_output)
            continue

        break

    if messages[-1]["role"] != "assistant":
        logger.warning("Teacher did not produce a final response; retrying.")
        return None

    return messages, tool_turns


def _load_tool_schema(config_path: Path) -> list[dict]:
    yaml_cfg = yaml.safe_load(config_path.read_text())
    schemas = []
    for entry in yaml_cfg["tools"]:
        schema = OpenAIFunctionToolSchema.model_validate(entry["tool_schema"])
        schemas.append(schema.model_dump(exclude_unset=True, exclude_none=True))
    return schemas


def run_pipeline(args: argparse.Namespace) -> None:
    random.seed(0)
    torch.manual_seed(0)

    tokenizer_src = args.teacher_tokenizer_path or args.teacher_model
    tokenizer_kwargs = {}
    if args.teacher_tokenizer_path is not None:
        tokenizer_kwargs["local_files_only"] = True
        tokenizer_kwargs["trust_remote_code"] = True
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_src, **tokenizer_kwargs)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = "bfloat16" if torch.cuda.is_available() else "float32"
    teacher_model_src = args.teacher_model_path or args.teacher_model
    llm = LLM(
        model=teacher_model_src,
        tokenizer=tokenizer_src,
        trust_remote_code=True,
        dtype=dtype,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    dataset = _load_dataset(args.input_parquet, args.max_samples)
    sandbox_client = SandboxFusionClient(args.sandbox_config)
    tool_schema = _load_tool_schema(args.sandbox_config)
    parser = HermesToolParser()

    accepted_records: list[dict[str, Any]] = []
    preview_path = args.output_parquet.with_suffix(".preview.jsonl")
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    preview_f = preview_path.open("w", encoding="utf-8")
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_new_tokens,
        stop=tokenizer.eos_token,
    )

    for row in tqdm(dataset, desc="Sampling", total=len(dataset)):
        success = False
        for attempt in range(1, args.max_attempts + 1):
            outcome = _sample_teacher_dialog(
                row,
                llm=llm,
                tokenizer=tokenizer,
                sampling_params=sampling_params,
                tool_schema=tool_schema,
                sandbox_client=sandbox_client,
                parser=parser,
                max_turns=args.max_turns,
            )
            if outcome is None:
                continue

            messages, tool_turns = outcome
            final_response = messages[-1].get("content", "")

            if tool_turns == 0:
                logger.info("Attempt %d skipped (no tool calls).", attempt)
                continue

            if not _has_boxed_answer(final_response):
                logger.info("Attempt %d skipped (missing \\boxed answer).", attempt)
                continue

            _append_final_answer_message(messages)
            record = _format_sft_record(messages, tool_schema)
            accepted_records.append(record)
            preview_payload = {
                "messages": [
                    {
                        "role": msg["role"],
                        "content": msg["content"],
                        "tool_calls": msg["tool_calls"].tolist() if isinstance(msg["tool_calls"], np.ndarray) else None,
                    }
                    for msg in record["messages"]
                ]
            }
            preview_f.write(json.dumps(preview_payload, ensure_ascii=False) + "\n")
            logger.info(
                "Accepted sample #%d after %d attempt(s) (tool_turns=%d, messages=%d)",
                len(accepted_records),
                attempt,
                tool_turns,
                len(messages),
            )
            success = True
            break

        if not success:
            logger.info("Skipping sample after %d attempts due to format issues.", args.max_attempts)

    if not accepted_records:
        logger.warning("No samples passed filtering. Nothing to save.")
        preview_f.close()
        return

    output_dir = args.output_parquet.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(accepted_records).to_parquet(args.output_parquet, index=False)
    preview_f.close()
    logger.info(
        "Saved %d filtered samples to %s (preview: %s)",
        len(accepted_records),
        args.output_parquet,
        preview_path,
    )


if __name__ == "__main__":
    cli_args = _parse_arguments()
    run_pipeline(cli_args)
