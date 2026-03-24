#!/bin/bash

#SBATCH --chdir=/fsx/zyhang/verl/
#SBATCH --qos=h200_mrs_2_high
#SBATCH --nodes 4
#SBATCH --tasks-per-node 8
#SBATCH --cpus-per-task 24
#SBATCH --gpus-per-node 8
#SBATCH --mem 500G
#SBATCH --time=48:00:00
#SBATCH --job-name=ebpo_qwen3_14b_rl_dapo_coding_fsdp
#SBATCH --output=/fsx/zyhang/verl/slurm/ebpo_qwen3_14b_rl_dapo_coding.stdout
#SBATCH --error=/fsx/zyhang/verl/slurm/ebpo_qwen3_14b_rl_dapo_coding.stderr


set -x

export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"
export VLLM_USE_V1=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_ENGINE_ITERATION_TIMEOUT_S=100000000000
export NCCL_SOCKET_IFNAME=eth0
export NCCL_DEBUG=WARN

project_name="ebpo"
algorithm=grpo
rollout_n=4
k_max=8
loss_agg_mode="token-mean"
clip_ratio_high=0.28
enable_overlong_buffer=True
overlong_buffer_len=$((1024 * 4))
overlong_penalty_factor=-1.0
enable_filter_groups=False
filter_groups_metric=acc
max_num_gen_batches=10
experiment_name="ebpo_qwen3_14b_rl_dapo_coding_fsdp_multi_nodes"
ROOT_DIR=/fsx/zyhang/verl
CHECKPOINT_PATH=/fsx/zyhang/checkpoints
MODEL_PATH=/fsx/zyhang/Qwen/Qwen3-14B
DATA_PATH=/fsx/zyhang/verl/recipe/ebpo/prime-rl


mkdir -p logs/${project_name}
rm -rf $CHECKPOINT_PATH/${project_name}/${experiment_name}


export CUDA_DEVICE_MAX_CONNECTIONS=1 # For megatron communication/computation overlapping


mkdir -p logs/${project_name}
# rm -rf $CHECKPOINT_PATH/${project_name}/${experiment_name}

nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)

head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

# 处理IPv6或多个IP的情况
if [[ "$head_node_ip" == *" "* ]]; then
  IFS=' ' read -ra ADDR <<<"$head_node_ip"
  if [[ ${#ADDR[0]} -gt 16 ]]; then
    head_node_ip=${ADDR[1]}
  else
    head_node_ip=${ADDR[0]}
  fi
  echo "IPV6 address detected. We split the IPV4 address as $head_node_ip"
fi

port=6379
ip_head=$head_node_ip:$port
export RAY_ADDRESS=$ip_head
echo "IP Head: $ip_head"

# -----------start Ray Head ----------
echo "Starting Ray HEAD at $head_node"
srun --nodes=1 --ntasks=1 -w "$head_node" \
  ray start --head --node-ip-address=$head_node_ip --port=$port \
    --num-cpus $SLURM_CPUS_PER_TASK --num-gpus $SLURM_GPUS_PER_NODE --block &

sleep 20
# -----------start Ray Worker ----------
worker_num=$((SLURM_JOB_NUM_NODES - 1))

for ((i = 1; i <= worker_num; i++)); do
  node_i=${nodes_array[$i]}
  echo "Starting Ray WORKER $i at $node_i"
  srun --nodes=1 --ntasks=1 -w "$node_i" \
    ray start --address $ip_head \
      --num-cpus $SLURM_CPUS_PER_TASK --num-gpus $SLURM_GPUS_PER_NODE --block &
  sleep 5
done
sleep 30
for i in {1..20}; do
  worker_cnt=$(ray status | grep GPU | grep -o "[0-9.]\+/[0-9.]\+ GPU" | head -n 1 | cut -d/ -f2)
  if [[ "$worker_cnt" == "16.0 GPU" ]]; then
    echo "All workers connected!"
    break
  fi

  echo "current GPUs: ($worker_cnt) Waiting for workers... ($i)"
  sleep 5
done

ray status

PYTHONUNBUFFERED=1 srun --overlap --nodes=1 --ntasks=1 -w "$head_node" \
  bash -c "
    export NCCL_SOCKET_IFNAME=eth0
    export GLOO_SOCKET_IFNAME=eth0
    python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=$algorithm \
    algorithm.kl_ctrl.kl_coef=0.001 \
    algorithm.norm_adv_by_std_in_grpo=False \
    algorithm.filter_groups.enable=${enable_filter_groups} \
    algorithm.filter_groups.max_num_gen_batches=${max_num_gen_batches} \
    algorithm.filter_groups.metric=${filter_groups_metric} \
    data.train_files="$DATA_PATH/train_coding_by_source.parquet" \
    data.val_files="$DATA_PATH/valid_coding.parquet" \
    data.shuffle=False \
    data.train_batch_size=512 \
    data.max_prompt_length=4096 \
    data.max_response_length=8192 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.weight_decay=0.01 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
    actor_rollout_ref.rollout.n=${rollout_n} \
    actor_rollout_ref.rollout.val_kwargs.n=${k_max} \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    reward_model.overlong_buffer.enable=${enable_overlong_buffer} \
    reward_model.overlong_buffer.len=${overlong_buffer_len} \
    reward_model.overlong_buffer.penalty_factor=${overlong_penalty_factor} \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=${project_name} \
    trainer.experiment_name=${experiment_name} \
    trainer.default_local_dir=$CHECKPOINT_PATH/${project_name}/${experiment_name} \
    trainer.n_gpus_per_node=8 \
    trainer.val_before_train=True \
    trainer.nnodes=4 \
    trainer.save_freq=10 \
    trainer.test_freq=5 \
    trainer.total_epochs=20 2>&1 | tee logs/${project_name}/${experiment_name}.log
  "

wait


# Path to experiment directory
base_dir="$CHECKPOINT_PATH/${project_name}/${experiment_name}"

# Get the global_step_* directory with the largest number
latest_step_dir=$(find "$base_dir" -maxdepth 1 -type d -name "global_step_*" \
  | awk -F'_' '{ print $0, $(NF) }' \
  | sort -k2 -n \
  | tail -n 1 \
  | awk '{ print $1 }')

# If no match is found
if [[ -z "$latest_step_dir" ]]; then
  echo "No global_step_* directory found in $base_dir"
  exit 1
fi

echo "Latest step: $latest_step_dir"

# Run the merge command
python3 -m verl.model_merger merge \
    --backend fsdp \
    --local_dir "$latest_step_dir/actor" \
    --target_dir "$latest_step_dir/actor/huggingface"
