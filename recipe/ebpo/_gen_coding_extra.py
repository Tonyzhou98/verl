#!/usr/bin/env python3
"""Add coding-task baselines (rloo, reinforce++, entropymech-clipcov) for Qwen3-8B,
using the SAME coding setup as the existing coding scripts (ordered data
train_coding_by_source.parquet, shuffle=False, epochs=1, k_max=1), built from the
grpo coding donor with the proven runnable-harness fixes applied.
"""
import os, re

HERE = os.path.dirname(os.path.abspath(__file__))
DONOR = "grpo_fsdp_coding_qwen3_8b_multi_nodes.sh"

# tag -> (algorithm/adv_estimator, is_clipcov)
METHODS = {
    "rloo": ("rloo", False),
    "reinforce_pp": ("reinforce_plus_plus", False),
    "entropymech_clipcov": ("grpo", True),
}

CLIPCOV_BLOCK = (
    "    actor_rollout_ref.actor.entropy_coeff=0 \\\n"
    "    actor_rollout_ref.actor.clip_ratio_low=1.0 \\\n"
    "    actor_rollout_ref.actor.clip_ratio_high=1.0 \\\n"
    "    actor_rollout_ref.actor.clip_ratio_c=10.0 \\\n"
    "    actor_rollout_ref.actor.policy_loss.loss_mode=clip_cov \\\n"
    "    actor_rollout_ref.actor.policy_loss.clip_cov_ratio=0.0002 \\\n"
    "    actor_rollout_ref.actor.policy_loss.clip_cov_lb=1.0 \\\n"
    "    actor_rollout_ref.actor.policy_loss.clip_cov_ub=5.0 \\\n"
)

with open(os.path.join(HERE, DONOR)) as f:
    donor_text = f.read()

for tag, (algo, clipcov) in METHODS.items():
    text = donor_text
    new_exp = f"ebpo_qwen3_8b_rl_{tag}_coding_fsdp_multi_nodes"

    # --- runnable-harness fixes (same as math runs) ---
    text = re.sub(r"#SBATCH --qos=\S+", "#SBATCH --qos=h200_mrs_1_high", text)
    text = re.sub(r"#SBATCH --nodes 4", "#SBATCH --nodes 1", text)
    text = re.sub(r"trainer\.nnodes=4", "trainer.nnodes=1", text)
    text = text.replace('"16.0 GPU"', '"8.0 GPU"')
    text = re.sub(r"_micro_batch_size_per_gpu=2\b", "_micro_batch_size_per_gpu=8", text)
    text = text.replace(
        "export NCCL_DEBUG=WARN\n",
        "export NCCL_DEBUG=WARN\n\nsource ~/miniconda3/etc/profile.d/conda.sh\nconda activate ebpo\n",
    )
    text = text.replace(
        "    export GLOO_SOCKET_IFNAME=eth0\n",
        "    export GLOO_SOCKET_IFNAME=eth0\n    source ~/miniconda3/etc/profile.d/conda.sh\n    conda activate ebpo\n",
    )
    # bound checkpoints on disk
    text = re.sub(r"(\n(\s*)trainer\.save_freq=\d+ \\\n)",
                  r"\1\2trainer.max_actor_ckpt_to_keep=1 \\\n", text)

    # --- method-specific ---
    text = re.sub(r"\nalgorithm=grpo\b", f"\nalgorithm={algo}", text)

    if clipcov:
        text = text.replace("actor_rollout_ref.actor.use_kl_loss=True",
                            "actor_rollout_ref.actor.use_kl_loss=False")
        text = text.replace("actor_rollout_ref.actor.kl_loss_coef=0.001",
                            "actor_rollout_ref.actor.kl_loss_coef=0.0")
        text = text.replace("algorithm.kl_ctrl.kl_coef=0.001",
                            "algorithm.kl_ctrl.kl_coef=0.0")
        anchor = "    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \\\n"
        assert anchor in text, "loss_agg_mode anchor not found"
        text = text.replace(anchor, anchor + CLIPCOV_BLOCK)

    # --- names ---
    text = re.sub(r'experiment_name="[^"]*"', f'experiment_name="{new_exp}"', text)
    text = re.sub(r"#SBATCH --job-name=\S+", f"#SBATCH --job-name={new_exp}", text)
    text = re.sub(r"#SBATCH --output=\S+", f"#SBATCH --output=/fsx/zyhang/verl/slurm/{new_exp}.stdout", text)
    text = re.sub(r"#SBATCH --error=\S+", f"#SBATCH --error=/fsx/zyhang/verl/slurm/{new_exp}.stderr", text)

    dst = f"{tag}_fsdp_coding_qwen3_8b_multi_nodes.sh"
    with open(os.path.join(HERE, dst), "w") as f:
        f.write(text)
    print(f"WROTE {dst}  (algorithm={algo}, clipcov={clipcov})")
