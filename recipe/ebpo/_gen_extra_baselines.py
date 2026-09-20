#!/usr/bin/env python3
"""Generate random-ordering training scripts for the two extra advantage-estimator
baselines requested by reviewers: James-Stein shrinkage (Zeng et al.) and BNPO
(Xiao et al.). They are drop-in adv_estimator swaps on the standard main_ppo
harness, so we reuse the reinforce_pp random script (instruct model) as donor.
"""
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))

# (method token used in adv_estimator + names)
METHODS = ["shrinkage_js", "bnpo"]
SIZES = [
    ("8b", "reinforce_pp_fsdp_math_17k_random_qwen3_8b_multi_nodes.sh"),
    ("14b", "reinforce_pp_fsdp_math_17k_random_qwen3_14b_multi_nodes.sh"),
]


def build(method, size, donor):
    with open(os.path.join(HERE, donor)) as f:
        text = f.read()
    tag = method.replace("_", "")  # shrinkagejs / bnpo (clean names)
    new_exp = f"ebpo_qwen3_{size}_rl_{tag}_random_fsdp_multi_nodes"

    text = re.sub(r"\nalgorithm=\S+", f"\nalgorithm={method}", text)
    text = re.sub(r"trainer\.save_freq=\d+", "trainer.save_freq=50", text)
    text = re.sub(r'experiment_name="[^"]*"', f'experiment_name="{new_exp}"', text)
    text = re.sub(r"#SBATCH --job-name=\S+", f"#SBATCH --job-name={new_exp}", text)
    text = re.sub(r"#SBATCH --output=\S+", f"#SBATCH --output=/fsx/zyhang/verl/slurm/{new_exp}.stdout", text)
    text = re.sub(r"#SBATCH --error=\S+", f"#SBATCH --error=/fsx/zyhang/verl/slurm/{new_exp}.stderr", text)

    dst = f"{method}_fsdp_math_17k_random_qwen3_{size}_multi_nodes.sh"
    with open(os.path.join(HERE, dst), "w") as f:
        f.write(text)
    return dst, new_exp


for method in METHODS:
    for size, donor in SIZES:
        d, e = build(method, size, donor)
        print(f"WROTE {d}\n      exp={e}")
