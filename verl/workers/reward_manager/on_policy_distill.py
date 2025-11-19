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
"""Reverse-KL reward manager for on-policy distillation."""

from __future__ import annotations

import logging
from typing import Any

import torch
from torch import nn
from transformers import AutoModelForCausalLM

from verl import DataProto
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager

logger = logging.getLogger(__name__)


def _as_torch_dtype(dtype_like: str | torch.dtype) -> torch.dtype:
    if isinstance(dtype_like, torch.dtype):
        return dtype_like
    dtype_like = dtype_like.lower()
    if dtype_like in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if dtype_like in {"fp16", "float16", "half"}:
        return torch.float16
    if dtype_like in {"fp32", "float32", "f32"}:
        return torch.float32
    if dtype_like in {"fp64", "float64", "f64"}:
        return torch.float64
    raise ValueError(f"Unsupported dtype string '{dtype_like}'")


def _segment_response_spans(mask: torch.Tensor) -> list[tuple[int, int]]:
    """Return (start, end) indices for contiguous 1-runs inside mask."""
    mask = mask.to(torch.int32)
    padded = torch.nn.functional.pad(mask, (1, 1))
    diff = torch.diff(padded)
    start_indices = torch.nonzero(diff == 1, as_tuple=True)[0]
    end_indices = torch.nonzero(diff == -1, as_tuple=True)[0]
    return list(zip(start_indices.tolist(), end_indices.tolist()))


def _extract_assistant_messages(raw_messages: Any) -> list[Any]:
    """Best-effort helper to fetch assistant messages."""
    if raw_messages is None:
        return []
    if isinstance(raw_messages, dict) and "messages" in raw_messages:
        raw_messages = raw_messages["messages"]
    assistant_msgs: list[Any] = []
    for msg in raw_messages:
        role = getattr(msg, "role", None)
        if role is None and isinstance(msg, dict):
            role = msg.get("role")
        if role == "assistant":
            assistant_msgs.append(msg)
    return assistant_msgs


def _has_tool_call(message: Any) -> bool:
    """Detect whether the assistant message ended with a tool call."""
    if message is None:
        return False
    tool_calls = getattr(message, "tool_calls", None)
    if tool_calls is None and isinstance(message, dict):
        tool_calls = message.get("tool_calls")
    return bool(tool_calls)


@register("on_policy_distill")
class OnPolicyDistillRewardManager(AbstractRewardManager):
    """Compute reverse-KL token rewards using a stronger teacher model."""

    def __init__(
        self,
        tokenizer,
        num_examine: int,
        *,
        teacher_model_id: str,
        teacher_revision: str | None = None,
        teacher_dtype: str | torch.dtype = "bfloat16",
        teacher_device: str = "cuda",
        teacher_max_batch_size: int = 1,
        share_tokenizer: bool = True,
        tool_reward_weight: float = 1.2,
        final_answer_weight: float = 1.0,
        min_step_tokens: int = 1,
        epsilon: float = 1e-6,
        **_,
    ) -> None:
        if teacher_model_id is None:
            raise ValueError("`teacher_model_id` must be specified for on-policy distillation.")

        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.epsilon = epsilon
        self.tool_reward_weight = tool_reward_weight
        self.final_answer_weight = final_answer_weight
        self.min_step_tokens = max(1, int(min_step_tokens))

        # Teacher components -------------------------------------------------
        logger.info(
            "Loading teacher model '%s' (revision=%s, dtype=%s)",
            teacher_model_id,
            teacher_revision,
            teacher_dtype,
        )
        dtype = _as_torch_dtype(teacher_dtype)
        if not share_tokenizer:
            raise NotImplementedError(
                "share_tokenizer=False is not yet supported. Please use teacher/student pairs that share the same tokenizer."
            )
        self.teacher_tokenizer = tokenizer

        teacher_device = torch.device(teacher_device if torch.cuda.is_available() else "cpu")
        if teacher_device.type == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested for teacher but not available, falling back to CPU.")
            teacher_device = torch.device("cpu")
        self.teacher_device = teacher_device
        self.teacher_model: nn.Module = AutoModelForCausalLM.from_pretrained(
            teacher_model_id,
            revision=teacher_revision,
            torch_dtype=dtype,
            trust_remote_code=True,
        ).to(self.teacher_device)
        self.teacher_model.eval()
        self.teacher_max_batch_size = max(1, int(teacher_max_batch_size))

    def __call__(self, data: DataProto, return_dict: bool = False) -> torch.Tensor | dict[str, Any]:
        batch = data.batch

        required_keys = ["input_ids", "attention_mask", "responses", "response_mask"]
        for key in required_keys:
            if key not in batch:
                raise KeyError(f"Missing '{key}' in batch when computing on-policy distillation reward.")
        if "rollout_log_probs" not in batch:
            raise KeyError(
                "rollout_log_probs not found. Enable `actor_rollout_ref.rollout.calculate_log_probs=True` "
                "so that on-policy distillation can recover student token probabilities."
            )

        input_ids = batch["input_ids"].to(torch.long)
        attention_mask = batch["attention_mask"].to(torch.long)
        response_mask = batch["response_mask"].to(torch.long)
        responses = batch["responses"]
        response_length = responses.shape[1]

        rollout_log_probs = batch["rollout_log_probs"].to(torch.float32)
        if rollout_log_probs.shape[-1] != response_length:
            raise ValueError(
                "rollout_log_probs length mismatch: "
                f"{rollout_log_probs.shape[-1]} (log probs) vs {response_length} (responses)."
            )

        teacher_log_probs = self._batched_teacher_log_probs(input_ids, attention_mask)[:, -response_length:]
        reward_tensor = torch.zeros_like(responses, dtype=torch.float32)

        raw_messages = data.non_tensor_batch.get("messages")
        step_logs: list[list[dict[str, Any]]] = []
        batch_size = rollout_log_probs.shape[0]
        for idx in range(batch_size):
            sample_messages = None if raw_messages is None else raw_messages[idx]
            reward_tensor[idx], per_sample_log = self._compute_sample_reward(
                student_log_probs=rollout_log_probs[idx],
                teacher_log_probs=teacher_log_probs[idx],
                response_mask=response_mask[idx],
                messages=sample_messages,
            )
            step_logs.append(per_sample_log)

        if return_dict:
            return {"reward_tensor": reward_tensor, "reward_extra_info": {"step_logs": step_logs}}
        return reward_tensor

    def _batched_teacher_log_probs(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Compute log p_teacher(token) for every token in the batch."""
        outputs: list[torch.Tensor] = []
        total = input_ids.shape[0]
        for start in range(0, total, self.teacher_max_batch_size):
            end = min(start + self.teacher_max_batch_size, total)
            chunk_ids = input_ids[start:end].to(self.teacher_device)
            chunk_mask = attention_mask[start:end].to(self.teacher_device)

            with torch.no_grad():
                logits = self.teacher_model(input_ids=chunk_ids, attention_mask=chunk_mask).logits
                token_log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)
                next_token_ids = chunk_ids[:, 1:].unsqueeze(-1)
                gathered = torch.gather(token_log_probs, dim=-1, index=next_token_ids).squeeze(-1)
                pad = torch.zeros(gathered.size(0), 1, device=gathered.device, dtype=gathered.dtype)
                gathered = torch.cat([pad, gathered], dim=1)
            outputs.append(gathered.cpu())
        return torch.cat(outputs, dim=0)

    def _compute_sample_reward(
        self,
        *,
        student_log_probs: torch.Tensor,
        teacher_log_probs: torch.Tensor,
        response_mask: torch.Tensor,
        messages: Any,
    ) -> tuple[torch.Tensor, list[dict[str, Any]]]:
        """Compute token-level reverse-KL reward for a single sample."""
        spans = _segment_response_spans(response_mask)
        assistant_messages = _extract_assistant_messages(messages)
        per_token_reward = torch.zeros_like(student_log_probs, dtype=torch.float32)
        step_logs: list[dict[str, Any]] = []

        for step_idx, (start, end) in enumerate(spans):
            if end - start < self.min_step_tokens:
                continue

            segment_student = student_log_probs[start:end]
            segment_teacher = teacher_log_probs[start:end]
            segment_reward = segment_student - segment_teacher

            weight = 1.0
            if step_idx == len(spans) - 1:
                weight *= self.final_answer_weight
            if step_idx < len(assistant_messages) and _has_tool_call(assistant_messages[step_idx]):
                weight *= self.tool_reward_weight

            segment_reward = segment_reward * weight
            per_token_reward[start:end] = segment_reward

            step_logs.append(
                {
                    "step": step_idx,
                    "token_span": (int(start), int(end)),
                    "avg_token_reward": float(segment_reward.mean().detach().cpu()),
                    "weight": float(weight),
                    "token_count": int(end - start),
                }
            )

        return per_token_reward, step_logs
