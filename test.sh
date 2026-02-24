#!/bin/bash

#SBATCH --chdir=/fsx/zyhang/verl/
#SBATCH --gres=gpu:4
#SBATCH --mem 128G
#SBATCH -c 64
#SBATCH --job-name=minimal_rl_grpo
#SBATCH --output=/fsx/zyhang/verl/slurm/minimal_rl_grpo.stdout
#SBATCH --error=/fsx/zyhang/verl/slurm/minimal_rl_grpo.stderr


set -x

# export VLLM_ATTENTION_BACKEND=XFORMERS
project_name="test"
algorithm=grpo
rollout_n=4
# for mean@K computation
k_max=16
#experiment_name=${model}-${algorithm}-${data}-n${n}
experiment_name="initial_grpo_baseline"
ROOT_DIR=/fsx/zyhang/verl
CHECKPOINT_PATH=/fsx/zyhang/checkpoints/
MODEL_PATH=/fsx/zyhang/Qwen/Qwen2.5-7B-Instruct
DATA_PATH=./


mkdir -p logs/${project_name}

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=$algorithm \
    data.train_files=$DATA_PATH/data/gsm8k/train.parquet \
    data.val_files=$DATA_PATH/data/gsm8k/test.parquet \
    data.train_batch_size=1024 \
    data.max_prompt_length=512 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.model.enable_gradient_checkpointing=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=${rollout_n} \
    actor_rollout_ref.rollout.val_kwargs.n=${k_max} \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=${project_name} \
    trainer.experiment_name=${experiment_name} \
    trainer.n_gpus_per_node=4 \
    trainer.val_before_train=True \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.max_actor_ckpt_to_keep=5 \
    trainer.max_critic_ckpt_to_keep=5 \
    trainer.default_local_dir=$CHECKPOINT_PATH/${project_name}/${experiment_name} \
    trainer.test_freq=10 \
    trainer.total_epochs=1 2>&1 | tee logs/${project_name}/${experiment_name}.log


python3 -m verl.model_merger merge \
    --backend fsdp \
    --local_dir $CHECKPOINT_PATH/${project_name}/${experiment_name}/global_step_1/actor \
    --target_dir $CHECKPOINT_PATH/${project_name}/${experiment_name}/global_step_1/actor/huggingface