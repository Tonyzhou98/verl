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

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ---- Resolve model path: auto-merge FSDP -> HF if needed ----
# Accepts a merged HF dir, a global_step_* dir, or an FSDP actor dir.
# If handed an unmerged FSDP checkpoint, merge it inline (verl.model_merger) first.
resolve_model_path() {
    local p="$1"
    if [[ -d "$p/actor" ]]; then p="$p/actor"; fi
    if compgen -G "$p/*.safetensors" >/dev/null 2>&1; then echo "$p"; return; fi
    if [[ -d "$p/huggingface" ]] && compgen -G "$p/huggingface/*.safetensors" >/dev/null 2>&1; then
        echo "$p/huggingface"; return
    fi
    if compgen -G "$p/model_world_size_*.pt" >/dev/null 2>&1; then
        local hf="$p/huggingface"
        echo "Merging FSDP checkpoint -> $hf" >&2
        python3 -m verl.model_merger merge --backend fsdp --local_dir "$p" --target_dir "$hf" 1>&2
        if compgen -G "$hf/*.safetensors" >/dev/null 2>&1; then
            echo "$hf"; return
        fi
        echo "ERROR: merge produced no safetensors in $hf" >&2; exit 1
    fi
    echo "$p"   # assume a plain HF model dir/name
}

MODEL_PATH="$(resolve_model_path "$1")"
# Extract output filename from path
# Case 1: checkpoint path like .../experiment_name/global_step_20/actor/huggingface
# Case 2: raw model path like /fsx/zyhang/Qwen/Qwen3-8B
if [[ "$MODEL_PATH" == */actor/huggingface ]]; then
    STEP_DIR=$(basename $(dirname $(dirname "$MODEL_PATH")))  # global_step_20
    EXP_DIR=$(basename $(dirname $(dirname $(dirname "$MODEL_PATH"))))  # ebpo_qwen3_8b_rl_...
    OUTPUT_FILE="${2:-eval_results_${EXP_DIR}_${STEP_DIR}.txt}"
else
    MODEL_NAME=$(basename "$MODEL_PATH")
    OUTPUT_FILE="${2:-eval_results_${MODEL_NAME}.txt}"
fi

echo "Model: $MODEL_PATH"
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
