#!/bin/bash
# Evaluate a model checkpoint on math benchmarks
# Usage: bash eval_math.sh <model_path> [output_file]
# Example: bash eval_math.sh /fsx/zyhang/checkpoints/ebpo/ebpo_qwen3_8b_rl_ebpo_fsdp_multi_nodes/global_step_300/actor/huggingface

set -x

if [ -z "$1" ]; then
    echo "Usage: bash eval_math.sh <model_path> [output_file]"
    echo "Example: bash eval_math.sh /fsx/zyhang/checkpoints/ebpo/.../actor/huggingface"
    exit 1
fi

MODEL_PATH="$1"
# Extract experiment name and step from path
# e.g. /fsx/.../ebpo_qwen3_8b_rl_reinforce_pp_fsdp_multi_nodes/global_step_20/actor/huggingface
# -> eval_results_ebpo_qwen3_8b_rl_reinforce_pp_fsdp_multi_nodes_global_step_20.txt
STEP_DIR=$(basename $(dirname $(dirname "$MODEL_PATH")))  # global_step_20
EXP_DIR=$(basename $(dirname $(dirname $(dirname "$MODEL_PATH"))))  # ebpo_qwen3_8b_rl_...
OUTPUT_FILE="${2:-eval_results_${EXP_DIR}_${STEP_DIR}.txt}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "Model: $MODEL_PATH" | tee "$OUTPUT_FILE"
echo "Date: $(date)" | tee -a "$OUTPUT_FILE"
echo "==========================================" | tee -a "$OUTPUT_FILE"

for benchmark in math_500.jsonl aime_2024_problems.parquet aime_2025_problems.parquet amc23.parquet olympiadbench.parquet; do
    echo "" | tee -a "$OUTPUT_FILE"
    echo ">>> Benchmark: $benchmark" | tee -a "$OUTPUT_FILE"
    echo "------------------------------------------" | tee -a "$OUTPUT_FILE"
    python3 "$SCRIPT_DIR/ebpo_eval.py" \
        --model_path "$MODEL_PATH" \
        --test_file "$benchmark" 2>&1 | tee -a "$OUTPUT_FILE"
done

echo "" | tee -a "$OUTPUT_FILE"
echo "==========================================" | tee -a "$OUTPUT_FILE"
echo "Evaluation complete. Results saved to: $OUTPUT_FILE"
