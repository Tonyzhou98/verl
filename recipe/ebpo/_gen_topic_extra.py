#!/usr/bin/env python3
"""Generate topic-clustering (bertopic) variants of the two new baselines
(shrinkage_js, bnpo) for 8B and 14B, from the validated random-ordering scripts.
Only changes: train data -> bertopic parquet, data.shuffle=False, and rename
_random_ -> _topic_ (experiment_name, SBATCH job-name/output/error, filename).
"""
import os, re

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = [
    "shrinkage_js_fsdp_math_17k_random_qwen3_8b_multi_nodes.sh",
    "shrinkage_js_fsdp_math_17k_random_qwen3_14b_multi_nodes.sh",
    "bnpo_fsdp_math_17k_random_qwen3_8b_multi_nodes.sh",
    "bnpo_fsdp_math_17k_random_qwen3_14b_multi_nodes.sh",
]

for src in SRC:
    with open(os.path.join(HERE, src)) as f:
        text = f.read()
    assert "dapo-math-17k-unique.parquet" in text, f"train file not found in {src}"
    assert "data.shuffle=True" in text, f"shuffle=True not found in {src}"
    # topic-clustered training data + preserve its ordering (no shuffle)
    text = text.replace("dapo-math-17k-unique.parquet",
                        "dapo-math-17k-unique-bertopic.parquet")
    text = text.replace("data.shuffle=True", "data.shuffle=False")
    # rename random -> topic across experiment_name, SBATCH names, output/error
    text = text.replace("_random_", "_topic_")
    dst = src.replace("_random_", "_topic_")
    with open(os.path.join(HERE, dst), "w") as f:
        f.write(text)
    print(f"WROTE {dst}")
