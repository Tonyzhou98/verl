#!/usr/bin/env python3
"""Generate random-ordering (shuffle=True) variants of the Table-1 baseline
training scripts.

For each source script we:
  - set data.shuffle=True
  - point train_files at the unsorted dapo-math-17k-unique.parquet
  - set trainer.total_epochs=6  (~210 steps, near the ~200-step roof)
  - bound FSDP checkpoints during training (trainer.max_actor_ckpt_to_keep=1)
  - rename experiment_name / SBATCH fields (so we never clobber existing runs)
  - replace the tail merge with a merge-then-verify-then-delete-FSDP block
Everything else (model path, optimizer, algo-specific flags) is preserved.
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

# (source filename, method token, size, is_base_model)
SOURCES = [
    ("grpo_fsdp_math_17k_sorted_qwen3_8b_base_multi_nodes.sh",         "grpo",         "8b",  True),
    ("drgrpo_fsdp_math_17k_sorted_qwen3_8b_base_multi_nodes.sh",       "drgrpo",       "8b",  True),
    ("dapo_fsdp_math_17k_sorted_qwen3_8b_base_multi_nodes.sh",         "dapo",         "8b",  True),
    ("reinforce_pp_fsdp_math_17k_naive_qwen3_8b_multi_nodes.sh",       "reinforce_pp", "8b",  False),
    ("rloo_fsdp_math_17k_naive_qwen3_8b_multi_nodes.sh",              "rloo",         "8b",  False),
    ("spo_fsdp_math_17k_naive_qwen3_8b_multi_nodes.sh",               "spo",          "8b",  False),
    ("grpo_fsdp_math_17k_sorted_qwen3_14b_base_multi_nodes.sh",        "grpo",         "14b", True),
    ("drgrpo_fsdp_math_17k_sorted_qwen3_14b_base_multi_nodes.sh",      "drgrpo",       "14b", True),
    ("dapo_fsdp_math_17k_sorted_qwen3_14b_base_multi_nodes.sh",        "dapo",         "14b", True),
    ("reinforce_pp_fsdp_math_17k_naive_qwen3_14b_multi_nodes.sh",      "reinforce_pp", "14b", False),
    ("rloo_fsdp_math_17k_naive_qwen3_14b_multi_nodes.sh",             "rloo",         "14b", False),
    ("spo_fsdp_math_17k_naive_qwen3_14b_multi_nodes.sh",              "spo",          "14b", False),
]

OLD_MERGE = '''python3 -m verl.model_merger merge \\
    --backend fsdp \\
    --local_dir "$latest_step_dir/actor" \\
    --target_dir "$latest_step_dir/actor/huggingface"'''

NEW_MERGE = '''# ------------------------------------------------------------------
# FSDP->HF merge is deferred to eval time (eval_math.sh / eval_coding.sh
# auto-merge and then prune the FSDP shards). Training keeps only the final
# FSDP checkpoint on disk (trainer.max_actor_ckpt_to_keep=1).
# ------------------------------------------------------------------
echo "Final FSDP checkpoint ready for eval-time merge: $latest_step_dir/actor"'''


def transform(text, method, size, is_base):
    new_exp = f"ebpo_qwen3_{size}_rl_{method}_random_fsdp_multi_nodes"
    model_path = f"/fsx/zyhang/Qwen/Qwen3-{'8B' if size == '8b' else '14B'}"

    # 0. QOS: use h200_mrs_1_high (h200_mrs_2_high is rejected; shared not desired)
    text = re.sub(r"#SBATCH --qos=\S+", "#SBATCH --qos=h200_mrs_1_high", text)

    # Table-1 uses the plain Qwen3-{8B,14B} for ALL methods (not -base, not instruct).
    text = re.sub(r"MODEL_PATH=\S+", f"MODEL_PATH={model_path}", text)

    # 0b. single-node (4 nodes unavailable): keep global batch=512 -> 4x prompts/GPU
    text = re.sub(r"#SBATCH --nodes 4", "#SBATCH --nodes 1", text)
    text = re.sub(r"trainer\.nnodes=4", "trainer.nnodes=1", text)
    text = text.replace('"16.0 GPU"', '"8.0 GPU"')

    # 0c. more prompts/GPU: raise per-GPU micro-batch (H200 has headroom on 1 node)
    text = re.sub(r"_micro_batch_size_per_gpu=2\b", "_micro_batch_size_per_gpu=8", text)

    # 0d. activate the training conda env (else srun uses base python3.9 w/o verl deps)
    text = text.replace(
        "export NCCL_DEBUG=WARN\n",
        "export NCCL_DEBUG=WARN\n\nsource ~/miniconda3/etc/profile.d/conda.sh\nconda activate ebpo\n",
    )
    text = text.replace(
        "    export GLOO_SOCKET_IFNAME=eth0\n",
        "    export GLOO_SOCKET_IFNAME=eth0\n    source ~/miniconda3/etc/profile.d/conda.sh\n    conda activate ebpo\n",
    )

    # 1. random ordering
    text, n = re.subn(r"data\.shuffle=False", "data.shuffle=True", text)
    assert n == 1, f"shuffle not found ({method} {size})"

    # 2. unsorted data file
    text, n = re.subn(r"dapo-math-17k-unique[-\w]*\.parquet",
                      "dapo-math-17k-unique.parquet", text)
    assert n >= 1, f"train_file not found ({method} {size})"

    # 3. epochs -> 6
    text, n = re.subn(r"trainer\.total_epochs=\d+", "trainer.total_epochs=6", text)
    assert n == 1, f"total_epochs not found ({method} {size})"

    # 3b. checkpoint every 50 steps
    text = re.sub(r"trainer\.save_freq=\d+", "trainer.save_freq=50", text)

    # 4. experiment_name
    text, n = re.subn(r'experiment_name="[^"]*"', f'experiment_name="{new_exp}"', text)
    assert n == 1, f"experiment_name not found ({method} {size})"

    # 5. SBATCH job-name / output / error
    text = re.sub(r"#SBATCH --job-name=\S+", f"#SBATCH --job-name={new_exp}", text)
    text = re.sub(r"#SBATCH --output=\S+", f"#SBATCH --output=/fsx/zyhang/verl/slurm/{new_exp}.stdout", text)
    text = re.sub(r"#SBATCH --error=\S+", f"#SBATCH --error=/fsx/zyhang/verl/slurm/{new_exp}.stderr", text)

    # 6. bound FSDP checkpoints during training (insert after save_freq line)
    text, n = re.subn(r"(\n(\s*)trainer\.save_freq=\d+ \\\n)",
                      r"\1\2trainer.max_actor_ckpt_to_keep=1 \\\n", text)
    assert n == 1, f"save_freq anchor not found ({method} {size})"

    # 7. cleanup block
    assert OLD_MERGE in text, f"merge block not found ({method} {size})"
    text = text.replace(OLD_MERGE, NEW_MERGE)

    return text, new_exp


def main():
    written = []
    for src, method, size, is_base in SOURCES:
        sp = os.path.join(HERE, src)
        if not os.path.exists(sp):
            print(f"MISSING SOURCE: {src}", file=sys.stderr)
            sys.exit(1)
        with open(sp) as f:
            text = f.read()
        out, new_exp = transform(text, method, size, is_base)
        dst = (src.replace("_sorted_", "_random_").replace("_naive_", "_random_")
                  .replace("_8b_base_", "_8b_").replace("_14b_base_", "_14b_"))
        dp = os.path.join(HERE, dst)
        with open(dp, "w") as f:
            f.write(out)
        written.append((dst, new_exp))
        print(f"WROTE {dst}\n      exp={new_exp}")
    print(f"\nTotal: {len(written)} scripts")


if __name__ == "__main__":
    main()
