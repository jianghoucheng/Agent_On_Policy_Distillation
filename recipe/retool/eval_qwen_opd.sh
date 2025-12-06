set -x
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4,5,6,7}
export VLLM_USE_V1=${VLLM_USE_V1:-1}

if [[ -z "${NUM_GPUS:-}" ]]; then
    if [[ -n "$CUDA_VISIBLE_DEVICES" ]]; then
        IFS=',' read -r -a __gpu_list <<< "$CUDA_VISIBLE_DEVICES"
        NUM_GPUS=${#__gpu_list[@]}
    else
        NUM_GPUS=1
    fi
fi

# ================= data/model/tool =================
HDFS_ROOT=${HDFS_ROOT:-$PWD}
DATA_ROOT=${DATA_ROOT:-$PWD}

aime_2024=$DATA_ROOT/dataset/Maxwell-Jia/AIME_2024
aime_2025=$DATA_ROOT/dataset/yentinglin/aime_2025

student_model_path=JoeYing/ReTool-DeepSeek-R1-Distill-Qwen-32B

eval_files="['$aime_2024', '$aime_2025']"

# tool
tool_config_path=recipe/retool/sandbox_fusion_tool_config.yaml

# wandb / logging
project_name=${PROJECT_NAME:-retool_opd_eval}
experiment_name=${EXPERIMENT_NAME:-qwen_opd_eval}
default_local_dir=$DATA_ROOT/checkpoint/$experiment_name
eval_dump_dir=$default_local_dir/eval_generations
mkdir -p "$default_local_dir" "$eval_dump_dir"

# ================= algorithm / rollout =================
adv_estimator=grpo
max_turns=16
max_prompt_length=2048
max_response_length=4096
actor_lr=1e-6

train_batch_size=1
ppo_mini_batch_size=1
n_resp_per_prompt=1
n_resp_per_prompt_val=1

num_gpus=${NUM_GPUS:-4}
infer_tp=${INFER_TP:-$num_gpus}
train_sp=${TRAIN_SP:-$num_gpus}
offload=${FSDP_OFFLOAD:-True}

actor_max_token_len_per_gpu=$(( (max_prompt_length + max_response_length) * 1 ))

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=$adv_estimator \
    algorithm.use_kl_in_reward=False \
    +algorithm.filter_groups.enable=False \
    +algorithm.token_kl_reg.enable=False \
    data.train_files="$eval_files" \
    data.val_files="$eval_files" \
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
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$infer_tp \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=$max_turns \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=$max_turns \
    actor_rollout_ref.rollout.multi_turn.tool_config_path=$tool_config_path \
    actor_rollout_ref.rollout.multi_turn.format=hermes \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.n=$n_resp_per_prompt \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.6 \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.n=$n_resp_per_prompt_val \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$project_name \
    trainer.experiment_name=$experiment_name \
    trainer.n_gpus_per_node=$num_gpus \
    trainer.nnodes=1 \
    trainer.val_before_train=True \
    trainer.val_only=True \
    trainer.total_epochs=0 \
    trainer.test_freq=0 \
    trainer.save_freq=0 \
    trainer.log_val_generations=20 \
    trainer.validation_data_dir=$eval_dump_dir \
    trainer.default_local_dir=$default_local_dir \
    "$@"
