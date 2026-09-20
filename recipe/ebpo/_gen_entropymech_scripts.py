#!/usr/bin/env python3
"""Generate EntropyMech (Clip-Cov) random-ordering training scripts for
Qwen3-8B and Qwen3-14B.

We reuse the validated SLURM + Ray harness from the existing baseline scripts
and only swap the launcher (recipe.entropy.main_entropy) and the loss config
(clip-cov + KL off), keeping the ebpo experimental setup (data, batch sizes,
6 epochs ~= 210 steps, keep-final FSDP, merge deferred to eval).
"""
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
# Use an existing instruct-model random script as the harness donor.
DONOR = "reinforce_pp_fsdp_math_17k_random_qwen3_8b_multi_nodes.sh"

# The full python invocation block for EntropyMech (clip-cov).
ENTROPY_PY = r'''PYTHONUNBUFFERED=1 srun --overlap --nodes=1 --ntasks=1 -w "$head_node" \
  bash -c "
    export NCCL_SOCKET_IFNAME=eth0
    export GLOO_SOCKET_IFNAME=eth0
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate ebpo
    python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_ctrl.kl_coef=0.0 \
    data.train_files="$DATA_PATH/dapo-math-17k-unique.parquet" \
    data.val_files="$DATA_PATH/aime-2024.parquet" \
    data.shuffle=True \
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
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.clip_ratio_low=1.0 \
    actor_rollout_ref.actor.clip_ratio_high=1.0 \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.policy_loss.loss_mode=clip_cov \
    actor_rollout_ref.actor.policy_loss.clip_cov_ratio=0.0002 \
    actor_rollout_ref.actor.policy_loss.clip_cov_lb=1.0 \
    actor_rollout_ref.actor.policy_loss.clip_cov_ub=5.0 \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
    actor_rollout_ref.rollout.n=${rollout_n} \
    actor_rollout_ref.rollout.val_kwargs.n=${k_max} \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=${project_name} \
    trainer.experiment_name=${experiment_name} \
    trainer.default_local_dir=$CHECKPOINT_PATH/${project_name}/${experiment_name} \
    trainer.n_gpus_per_node=8 \
    trainer.val_before_train=True \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.max_actor_ckpt_to_keep=1 \
    trainer.test_freq=5 \
    trainer.total_epochs=6 2>&1 | tee logs/${project_name}/${experiment_name}.log
  "'''

MODELS = [
    ("8b",  "/fsx/zyhang/Qwen/Qwen3-8B"),
    ("14b", "/fsx/zyhang/Qwen/Qwen3-14B"),
]


def build(size, model_path):
    with open(os.path.join(HERE, DONOR)) as f:
        text = f.read()

    new_exp = f"ebpo_qwen3_{size}_rl_entropymech_clipcov_random_fsdp_multi_nodes"

    # header names
    text = re.sub(r"#SBATCH --qos=\S+", "#SBATCH --qos=h200_mrs_1_high", text)
    text = re.sub(r"#SBATCH --nodes 4", "#SBATCH --nodes 1", text)
    text = text.replace('"16.0 GPU"', '"8.0 GPU"')
    text = re.sub(r"_micro_batch_size_per_gpu=2\b", "_micro_batch_size_per_gpu=8", text)
    text = re.sub(r"#SBATCH --job-name=\S+", f"#SBATCH --job-name={new_exp}", text)
    text = re.sub(r"#SBATCH --output=\S+", f"#SBATCH --output=/fsx/zyhang/verl/slurm/{new_exp}.stdout", text)
    text = re.sub(r"#SBATCH --error=\S+", f"#SBATCH --error=/fsx/zyhang/verl/slurm/{new_exp}.stderr", text)

    # experiment name + model path
    text = re.sub(r'experiment_name="[^"]*"', f'experiment_name="{new_exp}"', text)
    text = re.sub(r"MODEL_PATH=\S+", f"MODEL_PATH={model_path}", text)
    text = re.sub(r"\nalgorithm=\S+", "\nalgorithm=grpo", text)

    # replace the whole python invocation block (from the PYTHONUNBUFFERED srun
    # line through its closing `  "`) with the EntropyMech block.
    pat = re.compile(r'PYTHONUNBUFFERED=1 srun --overlap.*?\n  "', re.DOTALL)
    text, n = pat.subn(lambda _m: ENTROPY_PY, text)
    assert n == 1, f"python block not matched for {size}"

    dst = f"entropymech_clipcov_fsdp_math_17k_random_qwen3_{size}_multi_nodes.sh"
    with open(os.path.join(HERE, dst), "w") as f:
        f.write(text)
    return dst, new_exp


for size, mp in MODELS:
    d, e = build(size, mp)
    print(f"WROTE {d}\n      exp={e}  model={mp}")
