"""Recipe-local reward manager for on-policy distillation (reverse KL).

This keeps changes scoped to the recipe while reusing the core OPD logic.
It masks tool-return tokens via `response_mask` and applies optional
tool/final-answer weighting.
"""

from __future__ import annotations

import torch

from verl.workers.reward_manager.on_policy_distill import (
    OnPolicyDistillRewardManager,
    _extract_assistant_messages,
    _has_tool_call,
    _segment_response_spans,
)
from verl.workers.reward_manager.registry import register


@register("retool_opd")
class RetoolOPDRewardManager(OnPolicyDistillRewardManager):
    """Reverse-KL reward manager scoped to the retool recipe."""

    def _compute_sample_reward(
        self,
        *,
        student_log_probs: torch.Tensor,
        teacher_log_probs: torch.Tensor,
        response_mask: torch.Tensor,
        messages,
    ) -> tuple[torch.Tensor, list[dict]]:
        # Mask out tool-return/observation tokens (response_mask == 0) up front.
        response_mask_bool = response_mask.to(torch.bool)
        base_reward = (student_log_probs - teacher_log_probs) * response_mask_bool

        spans = _segment_response_spans(response_mask_bool)
        assistant_messages = _extract_assistant_messages(messages)
        per_token_reward = torch.zeros_like(student_log_probs, dtype=torch.float32)
        step_logs: list[dict] = []

        for step_idx, (start, end) in enumerate(spans):
            if end - start < self.min_step_tokens:
                continue

            weight = 1.0
            if step_idx == len(spans) - 1:
                weight *= self.final_answer_weight
            if step_idx < len(assistant_messages) and _has_tool_call(assistant_messages[step_idx]):
                weight *= self.tool_reward_weight

            segment_reward = base_reward[start:end] * weight
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


def main():
    """Entry point so we can `python -m recipe.retool.opd_reward ...` and still use the PPO launcher."""
    from verl.trainer.main_ppo import main as ppo_main

    return ppo_main()


if __name__ == "__main__":
    main()
