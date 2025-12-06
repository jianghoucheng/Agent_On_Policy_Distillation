set -x
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export VLLM_USE_V1=${VLLM_USE_V1:-1}

# ================= data/model/tool =================
HDFS_ROOT=${HDFS_ROOT:-$PWD}
DATA_ROOT=${DATA_ROOT:-$PWD}

dapo_math_17k=$DATA_ROOT/dataset/BytedTsinghua-SIA/DAPO-Math-17k
aime_2024=$DATA_ROOT/dataset/Maxwell-Jia/AIME_2024
aime_2025=$DATA_ROOT/dataset/yentinglin/aime_2025

student_model_path=JiangHoucheng/multiturn-sft-qwen-3-4b
teacher_model_path=JoeYing/ReTool-DeepSeek-R1-Distill-Qwen-32B

train_files="['$dapo_math_17k']"
test_files="['$aime_2024', '$aime_2025']"

# tool
tool_config_path=recipe/retool/sandbox_fusion_tool_config.yaml

# wandb
project_name=retool_opd
experiment_name=qwen3_4b_opd
default_local_dir=$DATA_ROOT/checkpoint/$experiment_name

# ================= algorithm =================
adv_estimator=grpo
# teacher_kl_coef=0.002
# filter groups for reward variance control
enable_filter_groups=False
filter_groups_metric=seq_reward
filter_groups_reward_std_threshold=0.0

# use_kl_in_reward=False
use_kl_loss=False
kl_loss_coef=0.0

clip_ratio_low=0.2
clip_ratio_high=0.28

max_turns=16
max_prompt_length=2048
max_response_length=16384
actor_lr=1e-6

train_batch_size=64
ppo_mini_batch_size=16
n_resp_per_prompt=16
n_resp_per_prompt_val=32

# token-level KL gating
token_kl_gamma=1.0
token_kl_beta_min=0.0
token_kl_beta_max=0.05

# ================= performance =================
infer_tp=4   # vllm tensor parallelism for the rollout model
train_sp=4   # training sequence parallelism on the student
offload=True # enable FSDP offload for memory headroom during long sequences

actor_max_token_len_per_gpu=$(( (max_prompt_length + max_response_length) * 1 ))
log_prob_max_token_len_per_gpu=$(( actor_max_token_len_per_gpu * 4 ))

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=$adv_estimator \
    algorithm.use_kl_in_reward=$use_kl_in_reward \
    algorithm.kl_ctrl.kl_coef=$teacher_kl_coef \
    +algorithm.filter_groups.enable=$enable_filter_groups \
    +algorithm.filter_groups.metric=$filter_groups_metric \
    +algorithm.filter_groups.reward_std_threshold=$filter_groups_reward_std_threshold \
    +algorithm.token_kl_reg.enable=True \
    +algorithm.token_kl_reg.gamma=$token_kl_gamma \
    +algorithm.token_kl_reg.beta_min=$token_kl_beta_min \
    +algorithm.token_kl_reg.beta_max=$token_kl_beta_max \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.return_raw_chat=True \
    data.train_batch_size=$train_batch_size \
    data.max_prompt_length=$max_prompt_length \
    data.max_response_length=$max_response_length \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.custom_cls.path=recipe/retool/retool.py \
    data.custom_cls.name=CustomRLHFDataset \
    custom_reward_function.path=recipe/retool/retool.py \
    custom_reward_function.name=compute_score \
    actor_rollout_ref.model.path=$student_model_path \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=$actor_lr \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_batch_size \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$actor_max_token_len_per_gpu \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=$train_sp \
    actor_rollout_ref.actor.fsdp_config.param_offload=$offload \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=$offload \
    +actor_rollout_ref.ref.model.path=$teacher_model_path \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$log_prob_max_token_len_per_gpu \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$infer_tp \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=$max_turns \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=$max_turns \
    actor_rollout_ref.rollout.multi_turn.tool_config_path=$tool_config_path \
    actor_rollout_ref.rollout.multi_turn.format=hermes \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.75 \
    actor_rollout_ref.rollout.n=$n_resp_per_prompt \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.6 \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.n=$n_resp_per_prompt_val \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$project_name \
    trainer.experiment_name=$experiment_name \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.val_before_train=False \
    trainer.log_val_generations=20 \
    trainer.save_freq=30 \
    trainer.default_local_dir=$default_local_dir \
    trainer.test_freq=10 \
    trainer.total_epochs=1 $@
