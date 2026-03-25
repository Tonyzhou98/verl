#!/bin/bash
# Merge FSDP checkpoints to HuggingFace format
# Usage: bash merge_fsdp_checkpoint.sh /fsx/zyhang/checkpoints/ebpo/ebpo_qwen3_8b_rl_reinforce_pp_fsdp_multi_nodes/global_step_20

set -x

if [ -z "$1" ]; then
    echo "Usage: bash merge_fsdp_checkpoint.sh <checkpoint_path>"
    echo "Example: bash merge_fsdp_checkpoint.sh /fsx/zyhang/checkpoints/ebpo/ebpo_qwen3_8b_rl_reinforce_pp_fsdp_multi_nodes/global_step_20"
    exit 1
fi

CKPT_PATH="$1"

cd /home/zyhang/ebpo/verl

srun --qos=h200_mrs_2_high --gres=gpu:1 --cpus-per-task=64 --mem=500G --time=4-00:00:00 python3 -m verl.model_merger merge \
    --backend fsdp \
    --local_dir "${CKPT_PATH}/actor" \
    --target_dir "${CKPT_PATH}/actor/huggingface"
