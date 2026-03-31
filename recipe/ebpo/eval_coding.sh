#!/bin/bash
# Evaluate a model checkpoint on coding benchmarks (apps, codecontests, taco, codeforces)
# Reports avg@8 per data source
#
# Usage: bash eval_coding.sh <model_path> [output_file]
# Example: bash eval_coding.sh /fsx/zyhang/checkpoints/ebpo/.../actor/huggingface

set -x

if [ -z "$1" ]; then
    echo "Usage: bash eval_coding.sh <model_path> [output_file]"
    echo "Example: bash eval_coding.sh /fsx/zyhang/checkpoints/ebpo/.../actor/huggingface"
    exit 1
fi

MODEL_PATH="$1"
STEP_DIR=$(basename $(dirname $(dirname "$MODEL_PATH")))
EXP_DIR=$(basename $(dirname $(dirname $(dirname "$MODEL_PATH"))))
OUTPUT_FILE="${2:-eval_coding_${EXP_DIR}_${STEP_DIR}.txt}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_PATH="/fsx/zyhang/verl/recipe/ebpo/prime-rl/valid_coding.parquet"

echo "Model: $MODEL_PATH" | tee "$OUTPUT_FILE"
echo "Date: $(date)" | tee -a "$OUTPUT_FILE"
echo "==========================================" | tee -a "$OUTPUT_FILE"

python3 "$SCRIPT_DIR/coding_eval.py" \
    --model_path "$MODEL_PATH" \
    --test_file "$DATA_PATH" \
    --n_samples 8 2>&1 | tee -a "$OUTPUT_FILE"

echo "" | tee -a "$OUTPUT_FILE"
echo "==========================================" | tee -a "$OUTPUT_FILE"
echo "Evaluation complete. Results saved to: $OUTPUT_FILE"
