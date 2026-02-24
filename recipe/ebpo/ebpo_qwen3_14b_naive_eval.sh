MODEL_PATH=/lustre-storage/fsx/zyhang/verl/checkpoints/ebpo/ebpo_qwen3_14b_rl_ebpo_fsdp_multi_nodes/global_step_260/actor/huggingface

python3 ebpo_eval.py \
--model_path $MODEL_PATH \
--test_file math_500.jsonl

python3 ebpo_eval.py \
--model_path $MODEL_PATH \
--test_file aime_2024_problems.parquet

python3 ebpo_eval.py \
--model_path $MODEL_PATH \
--test_file aime_2025_problems.parquet


python3 ebpo_eval.py \
--model_path $MODEL_PATH \
--test_file amc23.parquet

python3 ebpo_eval.py \
--model_path $MODEL_PATH \
--test_file olympiadbench.parquet