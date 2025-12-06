# Retool
[ReTool: Reinforcement Learning for Strategic Tool Use in LLMs](https://arxiv.org/abs/2504.11536)

## Overview
- Base model: [Qwen/Qwen2.5-32B-Instruct](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct)
- SFT dataset: [JoeYing/ReTool-SFT](https://huggingface.co/datasets/JoeYing/ReTool-SFT)
- RL dataset: [BytedTsinghua-SIA/DAPO-Math-17k](https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k)
- Val dataset: [yentinglin/aime_2025](https://huggingface.co/datasets/yentinglin/aime_2025)

## SFT
1. Data preparation
```bash
python3 recipe/retool/retool_sft_preprocess.py
```

2. Training
```bash
bash recipe/retool/run_qwen2-32b_sft.sh
```

After 6 epoches, validation metrics:
- val-core/aime_2025/acc/mean@30: 0.24
- val-aux/num_turns/mean: 7.2

## RL

### GRPO
```bash
bash recipe/retool/run_qwen2-32b_dapo.sh
```

After 150 steps, validation metrics:
- val-core/aime_2025/acc/mean@30: 0.6
- val-aux/num_turns/mean: 10

### PPO

```bash
bash recipe/retool/run_qwen2-32b_ppo.sh
```

After 250 steps, validation metrics:
- val-core/aime_2025/acc/mean@30: 0.55
- val-aux/num_turns/mean: 8.3

## On-policy distillation
- Student: `JiangHoucheng/multiturn-sft-qwen-3-4b`
- Teacher: `JoeYing/ReTool-DeepSeek-R1-Distill-Qwen-32B` (only used for log-prob scoring)

```bash
bash recipe/retool/run_qwen3_4b_opd.sh
```

The script reuses the GRPO data/reward stack but turns on `algorithm.use_kl_in_reward` so that the per-token reverse KL from the teacher is subtracted from the student advantages. Only the student-generated response tokens contribute to this penalty because `response_mask` trims out tool returns before the PPO loss is computed.

## Local sandbox option
You can run the tool-calling sandbox locally for debugging/troubleshooting:

1. Start the Flask runner:

```bash
python3 recipe/retool/local_sandbox_server.py
```

2. Set `recipe/retool/sandbox_fusion_tool_config.yaml`:
   - `sandbox_fusion_url: http://localhost:8080/execute`
   - `use_local_flask: true`
   - (optional) adjust `local_flask_timeout`.

Switch the flag back to `false` to return to the remote Sandbox Fusion backend; the rest of the tooling stack keeps the same interface and outputs.
