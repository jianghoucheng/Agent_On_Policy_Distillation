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
import os
from contextlib import contextmanager, nullcontext
from typing import Any, Hashable

import numpy as np
import torch
from torch import nn
from transformers import AutoModelForCausalLM

from verl import DataProto
from verl.utils.import_utils import is_vllm_available
from verl.utils.reward_score import math_dapo
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager

logger = logging.getLogger(__name__)

# Shared cache so train/val reward managers reuse the same teacher instance.
_TEACHER_CACHE: dict[tuple[Hashable, ...], dict[str, Any]] = {}

try:  # Optional dependency when running under Ray.
    import ray

    def _get_assigned_cuda_device() -> torch.device | None:
        try:
            ctx = ray.get_runtime_context()
            gpu_ids = ctx.get_resource_ids().get("GPU")
            if not gpu_ids:
                return None
            gpu_id = int(gpu_ids[0][0])
            return torch.device(f"cuda:{gpu_id}")
        except Exception:  # noqa: BLE001
            return None

except Exception:  # noqa: BLE001

    ray = None

    def _get_assigned_cuda_device() -> torch.device | None:  # type: ignore[override]
        return None


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


def _resolve_teacher_device(requested: str | torch.device) -> torch.device:
    if isinstance(requested, torch.device):
        device = requested
    else:
        lowered = requested.lower()
        if lowered in {"auto", "cuda:auto"}:
            device = _get_assigned_cuda_device() or torch.device("cpu")
            if device.type != "cuda":
                if torch.cuda.is_available():
                    device = torch.device("cuda:0")
                    logger.info(
                        "teacher_device=auto but no Ray GPU assigned; using first visible CUDA device %s instead.",
                        device,
                    )
                else:
                    logger.warning("Could not infer assigned CUDA device for teacher, falling back to CPU.")
        else:
            device = torch.device(requested)

    if device.type == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested for teacher but not available, falling back to CPU.")
        return torch.device("cpu")
    return device


def _infer_input_device(model: nn.Module, fallback: torch.device) -> torch.device:
    """Get a device to place inputs on when accelerate device_map is used."""
    device_map = getattr(model, "hf_device_map", None)
    if device_map:
        first_shard = next(iter(device_map.values()))
        if isinstance(first_shard, (list, tuple)):
            first_shard = first_shard[0]
        return torch.device(first_shard)
    return fallback


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


@contextmanager
def _temporary_cuda_visible_devices(devices: str | None):
    """Context manager to temporarily set CUDA_VISIBLE_DEVICES."""
    if devices is None:
        yield
        return
    old_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = devices
    try:
        yield
    finally:
        if old_devices is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = old_devices


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
        teacher_device_map: str | dict | None = None,
        teacher_max_memory: dict | str | None = None,
        teacher_init_on_call: bool = False,
        reuse_teacher: bool = True,
        teacher_visible_devices: str | None = None,
        teacher_use_remote: bool = True,
        teacher_num_gpus: int = 1,
        teacher_max_batch_size: int = 1,
        share_tokenizer: bool = True,
        teacher_backend: str = "hf",
        teacher_vllm_gpu_memory_utilization: float | None = None,
        teacher_vllm_max_model_len: int | None = None,
        teacher_vllm_enforce_eager: bool = False,
        teacher_vllm_use_remote: bool = True,
        tool_reward_weight: float = 1.2,
        final_answer_weight: float = 1.0,
        min_step_tokens: int = 1,
        epsilon: float = 1e-6,
        reward_scale: float = 1.0,
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
        self.reward_scale = float(reward_scale)

        # Teacher components -------------------------------------------------
        if not share_tokenizer:
            raise NotImplementedError(
                "share_tokenizer=False is not yet supported. Please use teacher/student pairs that share the same tokenizer."
            )
        self.teacher_tokenizer = tokenizer

        self.teacher_model_id = teacher_model_id
        self.teacher_revision = teacher_revision
        self.teacher_dtype = _as_torch_dtype(teacher_dtype)
        self.teacher_backend = teacher_backend.lower().strip()
        if self.teacher_backend not in {"hf", "vllm"}:
            raise ValueError(f"Unsupported teacher_backend '{teacher_backend}'. Expected 'hf' or 'vllm'.")
        self.teacher_device_request = teacher_device
        self.teacher_device_map = teacher_device_map
        self.teacher_max_memory = teacher_max_memory
        self.teacher_init_on_call = teacher_init_on_call
        self.reuse_teacher = reuse_teacher
        self.teacher_visible_devices = teacher_visible_devices
        self.teacher_use_remote = bool(
            teacher_use_remote and ray is not None and ray.is_initialized() and self.teacher_backend == "hf"
        )
        self.teacher_num_gpus = max(1, int(teacher_num_gpus))
        self.teacher_max_batch_size = max(1, int(teacher_max_batch_size))
        self.teacher_vllm_gpu_memory_utilization = teacher_vllm_gpu_memory_utilization
        self.teacher_vllm_max_model_len = teacher_vllm_max_model_len
        self.teacher_vllm_enforce_eager = bool(teacher_vllm_enforce_eager)
        self.teacher_vllm_use_remote = bool(
            teacher_vllm_use_remote and ray is not None and ray.is_initialized() and self.teacher_backend == "vllm"
        )

        # Teacher runtime state
        self.teacher_device: torch.device | None = None
        self.teacher_input_device: torch.device | None = None
        self.teacher_model: nn.Module | None = None
        self.teacher_vllm = None
        self.teacher_vllm_actor = None
        self._teacher_vllm_sampling_params = None
        self.teacher_actor = None
        self._teacher_loaded = False
        self._teacher_cache_key = (
            self.teacher_model_id,
            self.teacher_revision,
            str(self.teacher_dtype),
            str(self.teacher_device_request),
            str(self.teacher_device_map),
            str(self.teacher_visible_devices),
            str(self.teacher_use_remote),
            str(self.teacher_num_gpus),
            str(self.teacher_backend),
            str(self.teacher_vllm_gpu_memory_utilization),
            str(self.teacher_vllm_max_model_len),
            str(self.teacher_vllm_enforce_eager),
            str(self.teacher_vllm_use_remote),
        )

        if not self.teacher_init_on_call:
            self._load_teacher_if_needed()

    def __call__(self, data: DataProto, return_dict: bool = False) -> torch.Tensor | dict[str, Any]:
        batch = data.batch

        # Validation/test: if ground-truth answers are present, compute rule-based accuracy reward (no tool bonus).
        if self.num_examine and "reward_model" in data.non_tensor_batch:
            responses = batch["responses"]
            response_mask = batch.get("response_mask")
            reward_tensor = torch.zeros_like(responses, dtype=torch.float32)
            scores: list[float] = []
            preds: list[str] = []
            reward_model_field = data.non_tensor_batch["reward_model"]
            if isinstance(reward_model_field, dict):
                gts = reward_model_field.get("ground_truth")
            else:
                gts = None

            if gts is None:
                gts = [None] * responses.shape[0]
            if isinstance(gts, str):
                gts = [gts] * responses.shape[0]

            for resp_tokens, gt in zip(responses, gts, strict=True):
                if response_mask is not None:
                    mask = response_mask[len(scores)]
                    valid_ids = resp_tokens[mask.to(torch.bool)].tolist()
                else:
                    valid_ids = resp_tokens.tolist()

                text = self.tokenizer.decode(valid_ids, skip_special_tokens=False)
                if gt is None:
                    scores.append(0.0)
                    preds.append("")
                    continue

                result = math_dapo.compute_score(text, gt, strict_box_verify=True)
                if result.get("pred") is None:
                    result["pred"] = ""

                scores.append(float(result.get("score", 0.0)))
                preds.append(result["pred"])

            if return_dict:
                return {
                    "reward_tensor": reward_tensor,
                    "reward_extra_info": {
                        "score": np.array(scores, dtype=np.float32),
                        "pred": np.array(preds, dtype=object),
                    },
                }
            return reward_tensor

        # If reward model worker already provided teacher log-probs, use them directly.
        if "teacher_log_probs" in batch:
            teacher_log_probs = batch["teacher_log_probs"].to(torch.float32)
            self._teacher_loaded = True  # no local teacher needed
        else:
            self._load_teacher_if_needed()
            teacher_log_probs = None

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

        if teacher_log_probs is None:
            teacher_log_probs = self._batched_teacher_log_probs(input_ids, attention_mask)[:, -response_length:]
        else:
            teacher_log_probs = teacher_log_probs[:, -response_length:]
        reward_tensor = torch.zeros_like(responses, dtype=torch.float32)

        raw_messages = data.non_tensor_batch.get("messages")
        step_logs: list[list[dict[str, Any]]] = []
        step_reward_means: list[float] = []
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
            if per_sample_log:
                mean_reward = float(
                    sum(item.get("avg_token_reward", 0.0) for item in per_sample_log) / max(1, len(per_sample_log))
                )
            else:
                mean_reward = 0.0
            step_reward_means.append(mean_reward)

        if return_dict:
            return {
                "reward_tensor": reward_tensor * self.reward_scale,
                "reward_extra_info": {"step_reward_mean": np.array(step_reward_means, dtype=np.float32)},
            }
        return reward_tensor * self.reward_scale

    def _clear_vllm_teacher_cache(self) -> None:
        """Clear cached vLLM teacher to allow reloading after failures."""
        self.teacher_vllm = None
        self.teacher_vllm_actor = None
        self._teacher_loaded = False
        if self._teacher_cache_key in _TEACHER_CACHE:
            cached = _TEACHER_CACHE[self._teacher_cache_key]
            cached.pop("vllm", None)
            cached.pop("vllm_actor", None)
            cached.pop("vllm_sampling_params", None)

    def _load_teacher_if_needed(self) -> None:
        """Load or reuse the teacher model."""
        if self._teacher_loaded:
            return

        if self.reuse_teacher and self._teacher_cache_key in _TEACHER_CACHE:
            cached = _TEACHER_CACHE[self._teacher_cache_key]
            self.teacher_model = cached.get("model")
            self.teacher_device = cached.get("device")
            self.teacher_input_device = cached.get("input_device")
            self.teacher_actor = cached.get("actor")
            self.teacher_vllm = cached.get("vllm")
            self.teacher_vllm_actor = cached.get("vllm_actor")
            self._teacher_vllm_sampling_params = cached.get("vllm_sampling_params")
            self._teacher_loaded = True
            if self.teacher_backend == "vllm":
                logger.info(
                    "Reusing cached vLLM teacher model '%s' (tp=%s, visible_devices=%s, remote=%s).",
                    self.teacher_model_id,
                    self.teacher_num_gpus,
                    self.teacher_visible_devices,
                    self.teacher_vllm_actor is not None,
                )
            elif self.teacher_actor is None:
                logger.info(
                    "Reusing cached teacher model '%s' on device %s (device_map=%s).",
                    self.teacher_model_id,
                    self.teacher_input_device,
                    self.teacher_device_map,
                )
            else:
                logger.info(
                    "Reusing cached remote teacher model '%s' on %s GPUs.",
                    self.teacher_model_id,
                    self.teacher_num_gpus,
                )
            return

        if self.teacher_backend == "vllm" and self.teacher_vllm_use_remote:
            resolved_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(
                "Loading teacher model '%s' (backend=%s, revision=%s, dtype=%s, remote vLLM tp=%s, visible_devices=%s)",
                self.teacher_model_id,
                self.teacher_backend,
                self.teacher_revision,
                self.teacher_dtype,
                self.teacher_num_gpus,
                self.teacher_visible_devices,
            )
        else:
            resolved_device = _resolve_teacher_device(self.teacher_device_request)
            logger.info(
                "Loading teacher model '%s' (backend=%s, revision=%s, dtype=%s, device=%s, device_map=%s, visible_devices=%s)",
                self.teacher_model_id,
                self.teacher_backend,
                self.teacher_revision,
                self.teacher_dtype,
                resolved_device,
                self.teacher_device_map,
                self.teacher_visible_devices,
            )

        if self.teacher_backend == "vllm":
            self._load_vllm_teacher()
            return

        model_kwargs: dict[str, Any] = {
            "pretrained_model_name_or_path": self.teacher_model_id,
            "revision": self.teacher_revision,
            "torch_dtype": self.teacher_dtype,
            "trust_remote_code": True,
        }
        if self.teacher_device_map is not None:
            model_kwargs["device_map"] = self.teacher_device_map
        if self.teacher_max_memory is not None:
            model_kwargs["max_memory"] = self.teacher_max_memory

        if self.teacher_use_remote:
            assert ray is not None and ray.is_initialized()

            @ray.remote(num_gpus=self.teacher_num_gpus, num_cpus=1)
            class _TeacherActor:
                def __init__(self, kwargs: dict[str, Any], device_request: str | torch.device):
                    resolved = _resolve_teacher_device(device_request)
                    model = AutoModelForCausalLM.from_pretrained(**kwargs)
                    if kwargs.get("device_map") is None:
                        model = model.to(resolved)
                    model.eval()
                    self.model = model
                    self.input_device = _infer_input_device(model, resolved)
                    self.dtype = kwargs.get("torch_dtype", torch.bfloat16)

                def compute_log_probs(self, ids: list[list[int]], masks: list[list[int]], max_bs: int) -> list[list[float]]:
                    import time

                    t0 = time.time()
                    ids_tensor = torch.tensor(ids, dtype=torch.long, device=self.input_device)
                    mask_tensor = torch.tensor(masks, dtype=torch.long, device=self.input_device)
                    outputs: list[torch.Tensor] = []
                    total = ids_tensor.shape[0]
                    max_len = int(mask_tensor.sum(dim=1).max().item())
                    ids_tensor = ids_tensor[:, :max_len]
                    mask_tensor = mask_tensor[:, :max_len]
                    print(
                        f"[TeacherActor] start compute_log_probs: batch={total}, max_len={max_len}, "
                        f"max_bs={max_bs}, device={self.input_device}, dtype={self.dtype}",
                        flush=True,
                    )
                    for s in range(0, total, max_bs):
                        e = min(s + max_bs, total)
                        chunk_ids = ids_tensor[s:e]
                        chunk_mask = mask_tensor[s:e]
                        chunk_t0 = time.time()
                        autocast_ctx = (
                            torch.autocast(device_type=self.input_device.type, dtype=self.dtype)
                            if self.input_device.type in {"cuda", "cpu"}
                            else nullcontext()
                        )
                        with torch.inference_mode(), autocast_ctx:
                            logits = self.model(input_ids=chunk_ids, attention_mask=chunk_mask).logits
                            print(
                                f"[TeacherActor] chunk {s}-{e} logits shape={logits.shape} "
                                f"(seq_len={chunk_ids.shape[1]}, batch={chunk_ids.shape[0]})",
                                flush=True,
                            )
                            token_log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)
                            next_ids = chunk_ids[:, 1:].unsqueeze(-1)
                            gathered = torch.gather(token_log_probs, dim=-1, index=next_ids).squeeze(-1)
                            pad = torch.zeros(gathered.size(0), 1, device=gathered.device, dtype=gathered.dtype)
                            gathered = torch.cat([pad, gathered], dim=1)
                        outputs.append(gathered.cpu())
                        chunk_elapsed = time.time() - chunk_t0
                        print(
                            f"[TeacherActor] chunk {s}-{e} done in {chunk_elapsed:.2f}s "
                            f"({chunk_elapsed / max(1, chunk_ids.shape[0]):.2f}s per sample)",
                            flush=True,
                        )
                    total_elapsed = time.time() - t0
                    print(f"[TeacherActor] finished compute_log_probs in {total_elapsed:.2f}s", flush=True)
                    return torch.cat(outputs, dim=0).tolist()

            self.teacher_actor = _TeacherActor.remote(model_kwargs, self.teacher_device_request)
            self.teacher_model = None
            self.teacher_device = None
            self.teacher_input_device = None
            self._teacher_loaded = True

            if self.reuse_teacher:
                _TEACHER_CACHE[self._teacher_cache_key] = {
                    "actor": self.teacher_actor,
                }
            logger.info("Teacher model loaded in remote actor using %s GPUs.", self.teacher_num_gpus)
            return

        with _temporary_cuda_visible_devices(self.teacher_visible_devices):
            teacher_model = AutoModelForCausalLM.from_pretrained(**model_kwargs)
            if self.teacher_device_map is None:
                teacher_model = teacher_model.to(resolved_device)
            teacher_model.eval()

        input_device = _infer_input_device(teacher_model, resolved_device)

        self.teacher_model = teacher_model
        self.teacher_device = resolved_device
        self.teacher_input_device = input_device
        self._teacher_loaded = True

        if self.reuse_teacher:
            _TEACHER_CACHE[self._teacher_cache_key] = {
                "model": teacher_model,
                "device": resolved_device,
                "input_device": input_device,
            }

    def _load_vllm_teacher(self) -> None:
        """Initialize a vLLM teacher for fast log-prob computation."""
        if not is_vllm_available():
            raise ImportError(
                "vLLM backend requested for on-policy distillation, but the 'vllm' package is not available."
            )

        try:
            from verl.third_party.vllm import LLM
            from vllm import SamplingParams
        except Exception as exc:  # noqa: BLE001
            raise ImportError("Failed to import vLLM. Please ensure vllm>=0.7.0 is installed.") from exc

        if self.teacher_device_request not in {"cuda", "auto", "cuda:auto"}:
            logger.warning(
                "vLLM teacher only supports CUDA. Requested teacher_device=%s; proceeding with CUDA devices.",
                self.teacher_device_request,
            )

        llm_kwargs: dict[str, Any] = {
            "model": self.teacher_model_id,
            "revision": self.teacher_revision,
            "tensor_parallel_size": self.teacher_num_gpus,
            "dtype": self._vllm_dtype_name(self.teacher_dtype),
            "trust_remote_code": True,
            "max_model_len": self.teacher_vllm_max_model_len,
            "enforce_eager": self.teacher_vllm_enforce_eager,
        }
        if self.teacher_vllm_gpu_memory_utilization is not None:
            llm_kwargs["gpu_memory_utilization"] = self.teacher_vllm_gpu_memory_utilization

        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=1,  # minimum allowed; we only care about prompt_logprobs
            logprobs=1,
            prompt_logprobs=1,
        )

        if self.teacher_vllm_use_remote:
            assert ray is not None and ray.is_initialized()

            @ray.remote(num_gpus=self.teacher_num_gpus, num_cpus=1)
            class _VLLMTeacherActor:
                def __init__(self, kwargs: dict[str, Any], visible_devices: str | None):
                    from verl.workers.reward_manager.on_policy_distill import _temporary_cuda_visible_devices
                    from verl.third_party.vllm import LLM
                    from vllm import SamplingParams

                    sampling_params = SamplingParams(
                        temperature=0.0,
                        max_tokens=1,
                        logprobs=1,
                        prompt_logprobs=1,
                    )
                    with _temporary_cuda_visible_devices(visible_devices):
                        self.vllm = LLM(**kwargs)
                    self.sampling_params = sampling_params

                def compute_log_probs(
                    self, prompts: list[list[int]], max_len: int, max_bs: int
                ) -> list[list[float]]:
                    import time

                    t0 = time.time()
                    outputs = []
                    total = len(prompts)
                    for s in range(0, total, max_bs):
                        e = min(s + max_bs, total)
                        chunk_prompts = prompts[s:e]
                        chunk_t0 = time.time()
                        payloads = [{"prompt_token_ids": ids} for ids in chunk_prompts]
                        res = self.vllm.generate(
                            prompts=payloads,
                            sampling_params=self.sampling_params,
                            use_tqdm=False,
                        )
                        chunk_log_probs = torch.zeros(len(res), max_len, dtype=torch.float32)
                        for idx, item in enumerate(res):
                            token_log_probs = OnPolicyDistillRewardManager._extract_vllm_prompt_logprobs_static(item)
                            seq_len = min(token_log_probs.numel(), max_len)
                            chunk_log_probs[idx, :seq_len] = token_log_probs[:seq_len]
                        outputs.append(chunk_log_probs)
                        logger.info(
                            "[VLLMTeacherActor] chunk %s-%s processed in %.2fs (bs=%s, max_len=%s)",
                            s,
                            e,
                            time.time() - chunk_t0,
                            len(chunk_prompts),
                            max_len,
                        )
                    logger.info(
                        "[VLLMTeacherActor] finished compute_log_probs in %.2fs", time.time() - t0
                    )
                    return torch.cat(outputs, dim=0).tolist()

            self.teacher_vllm_actor = _VLLMTeacherActor.remote(llm_kwargs, self.teacher_visible_devices)
            self.teacher_vllm = None
            self._teacher_vllm_sampling_params = sampling_params
            self._teacher_loaded = True

            if self.reuse_teacher:
                _TEACHER_CACHE[self._teacher_cache_key] = {
                    "vllm_actor": self.teacher_vllm_actor,
                    "vllm_sampling_params": sampling_params,
                }
            logger.info(
                "vLLM teacher loaded in remote actor for '%s' (tp=%d, max_model_len=%s, gpu_mem_util=%s, visible_devices=%s).",
                self.teacher_model_id,
                self.teacher_num_gpus,
                self.teacher_vllm_max_model_len,
                self.teacher_vllm_gpu_memory_utilization,
                self.teacher_visible_devices,
            )
            return

        with _temporary_cuda_visible_devices(self.teacher_visible_devices):
            teacher_vllm = LLM(**llm_kwargs)

        self.teacher_vllm = teacher_vllm
        self._teacher_vllm_sampling_params = sampling_params
        self._teacher_loaded = True

        if self.reuse_teacher:
            _TEACHER_CACHE[self._teacher_cache_key] = {
                "vllm": teacher_vllm,
                "vllm_sampling_params": sampling_params,
            }

        logger.info(
            "vLLM teacher loaded for '%s' (tp=%d, max_model_len=%s, gpu_mem_util=%s, visible_devices=%s).",
            self.teacher_model_id,
            self.teacher_num_gpus,
            self.teacher_vllm_max_model_len,
            self.teacher_vllm_gpu_memory_utilization,
            self.teacher_visible_devices,
        )

    @staticmethod
    def _vllm_dtype_name(dtype: torch.dtype) -> str:
        if dtype == torch.float16:
            return "float16"
        if dtype == torch.bfloat16:
            return "bfloat16"
        if dtype == torch.float32:
            return "float32"
        return str(dtype)

    def _batched_teacher_log_probs(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Compute log p_teacher(token) for every token in the batch."""
        if self.teacher_backend == "vllm":
            return self._vllm_teacher_log_probs(input_ids, attention_mask)

        if self.teacher_actor is not None:
            ids_list = input_ids.cpu().tolist()
            mask_list = attention_mask.cpu().tolist()
            result = ray.get(
                self.teacher_actor.compute_log_probs.remote(ids_list, mask_list, self.teacher_max_batch_size)
            )
            return torch.tensor(result, dtype=torch.float32)

        assert self.teacher_model is not None and self.teacher_input_device is not None

        # Trim away trailing padding to avoid needless compute on teacher.
        max_seq_len = int(attention_mask.sum(dim=1).max().item())
        if max_seq_len < attention_mask.shape[1]:
            input_ids = input_ids[:, :max_seq_len]
            attention_mask = attention_mask[:, :max_seq_len]

        outputs: list[torch.Tensor] = []
        total = input_ids.shape[0]
        for start in range(0, total, self.teacher_max_batch_size):
            end = min(start + self.teacher_max_batch_size, total)
            chunk_ids = input_ids[start:end].to(self.teacher_input_device)
            chunk_mask = attention_mask[start:end].to(self.teacher_input_device)

            autocast_ctx = (
                torch.autocast(device_type=self.teacher_input_device.type, dtype=self.teacher_dtype)
                if self.teacher_input_device.type in {"cuda", "cpu"}
                else nullcontext()
            )
            with torch.inference_mode(), autocast_ctx:
                logits = self.teacher_model(input_ids=chunk_ids, attention_mask=chunk_mask).logits
                token_log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)
                next_token_ids = chunk_ids[:, 1:].unsqueeze(-1)
                gathered = torch.gather(token_log_probs, dim=-1, index=next_token_ids).squeeze(-1)
                pad = torch.zeros(gathered.size(0), 1, device=gathered.device, dtype=gathered.dtype)
                gathered = torch.cat([pad, gathered], dim=1)
            outputs.append(gathered.cpu())
        return torch.cat(outputs, dim=0)

    def _vllm_teacher_log_probs(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Compute teacher log probs using vLLM prompt log-prob capability."""
        if (self.teacher_vllm is None and self.teacher_vllm_actor is None) or self._teacher_vllm_sampling_params is None:
            raise RuntimeError("vLLM teacher not initialized. Call `_load_teacher_if_needed` first.")

        # Trim trailing padding for efficiency.
        max_seq_len = int(attention_mask.sum(dim=1).max().item())
        if max_seq_len < attention_mask.shape[1]:
            input_ids = input_ids[:, :max_seq_len]
            attention_mask = attention_mask[:, :max_seq_len]
        target_len = max_seq_len

        outputs: list[torch.Tensor] = []
        total = input_ids.shape[0]
        for start in range(0, total, self.teacher_max_batch_size):
            end = min(start + self.teacher_max_batch_size, total)
            chunk_ids = input_ids[start:end]
            chunk_mask = attention_mask[start:end]
            seq_lens = chunk_mask.sum(dim=1).tolist()
            prompts: list[list[int]] = []
            for sample_ids, seq_len in zip(chunk_ids, seq_lens):
                prompts.append(sample_ids[: int(seq_len)].tolist())

            if target_len == 0:
                outputs.append(torch.zeros(chunk_ids.size(0), 0, dtype=torch.float32))
                continue

            if self.teacher_vllm_actor is not None:
                try:
                    result = ray.get(
                        self.teacher_vllm_actor.compute_log_probs.remote(
                            prompts=prompts, max_len=target_len, max_bs=self.teacher_max_batch_size
                        )
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning("vLLM actor failed (%s); reloading teacher and retrying once.", exc)
                    self._clear_vllm_teacher_cache()
                    self._load_teacher_if_needed()
                    if self.teacher_vllm_actor is None:
                        raise
                    result = ray.get(
                        self.teacher_vllm_actor.compute_log_probs.remote(
                            prompts=prompts, max_len=target_len, max_bs=self.teacher_max_batch_size
                        )
                    )
                outputs.append(torch.tensor(result, dtype=torch.float32))
            else:
                prompt_payloads = [{"prompt_token_ids": ids} for ids in prompts]
                vllm_outputs = self.teacher_vllm.generate(
                    prompts=prompt_payloads,
                    sampling_params=self._teacher_vllm_sampling_params,
                    use_tqdm=False,
                )

                chunk_log_probs = torch.zeros(len(vllm_outputs), target_len, dtype=torch.float32)
                for sample_idx, sample_output in enumerate(vllm_outputs):
                    token_log_probs = self._extract_vllm_prompt_logprobs(sample_output)
                    seq_len = min(token_log_probs.numel(), target_len)
                    chunk_log_probs[sample_idx, :seq_len] = token_log_probs[:seq_len]
                outputs.append(chunk_log_probs)

        return torch.cat(outputs, dim=0) if outputs else torch.zeros(0, 0, dtype=torch.float32)

    @staticmethod
    def _extract_vllm_prompt_logprobs(sample_output: Any) -> torch.Tensor:
        """Convert vLLM prompt_logprobs into a 1D tensor aligned with prompt_token_ids."""
        prompt_token_ids = getattr(sample_output, "prompt_token_ids", None)
        prompt_logprobs = getattr(sample_output, "prompt_logprobs", None)
        if prompt_token_ids is None or prompt_logprobs is None:
            raise RuntimeError(
                "vLLM output missing prompt_logprobs. Ensure SamplingParams(prompt_logprobs=1) is supported."
            )

        if len(prompt_logprobs) != len(prompt_token_ids):
            logger.warning(
                "Mismatch between prompt_token_ids (%d) and prompt_logprobs (%d); truncating to smallest length.",
                len(prompt_token_ids),
                len(prompt_logprobs),
            )
        length = min(len(prompt_token_ids), len(prompt_logprobs))
        collected: list[float] = []
        for token_id, token_logprob in zip(prompt_token_ids[:length], prompt_logprobs[:length]):
            if token_logprob is None:
                collected.append(0.0)
                continue
            chosen = token_logprob.get(token_id)
            if chosen is None:
                collected.append(0.0)
            else:
                logprob_value = getattr(chosen, "logprob", None)
                collected.append(float(logprob_value if logprob_value is not None else chosen))
        return torch.tensor(collected, dtype=torch.float32)

    @staticmethod
    def _extract_vllm_prompt_logprobs_static(sample_output: Any) -> torch.Tensor:
        # Static variant for use inside Ray actors where classmethods are not bound.
        prompt_token_ids = getattr(sample_output, "prompt_token_ids", None)
        prompt_logprobs = getattr(sample_output, "prompt_logprobs", None)
        if prompt_token_ids is None or prompt_logprobs is None:
            raise RuntimeError(
                "vLLM output missing prompt_logprobs. Ensure SamplingParams(prompt_logprobs=1) is supported."
            )

        length = min(len(prompt_token_ids), len(prompt_logprobs))
        collected: list[float] = []
        for token_id, token_logprob in zip(prompt_token_ids[:length], prompt_logprobs[:length]):
            if token_logprob is None:
                collected.append(0.0)
                continue
            chosen = token_logprob.get(token_id)
            if chosen is None:
                collected.append(0.0)
            else:
                logprob_value = getattr(chosen, "logprob", None)
                collected.append(float(logprob_value if logprob_value is not None else chosen))
        return torch.tensor(collected, dtype=torch.float32)

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
