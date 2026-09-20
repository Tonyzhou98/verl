"""
Evaluate a coding model on the PRIME-RL coding benchmarks.

Reports avg@N (average score over N samples) per data source.

Usage:
    python coding_eval.py --model_path <path> --test_file <parquet> [--n_samples 8]
"""

import argparse
from collections import defaultdict

import pyarrow.parquet as pq
import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from verl.utils.reward_score.prime_code import compute_score


def load_data(test_file):
    """Load coding evaluation data from parquet, handling nested columns."""
    pf = pq.ParquetFile(test_file)
    rows = []
    for i in range(pf.metadata.num_row_groups):
        rg = pf.read_row_group(i)
        for j in range(len(rg)):
            row = {}
            for col_name in rg.schema.names:
                row[col_name] = rg.column(col_name)[j].as_py()
            rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--test_file", type=str, required=True)
    parser.add_argument("--n_samples", type=int, default=8, help="Number of samples per problem for avg@N")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--max_tokens", type=int, default=20480)
    parser.add_argument("--max_problems", type=int, default=0, help="Limit number of problems (0=all, useful for debugging)")
    args = parser.parse_args()
    print(args)

    # Load data
    rows = load_data(args.test_file)
    if args.max_problems > 0:
        rows = rows[:args.max_problems]
        print(f"[DEBUG] Limited to {len(rows)} problems")
    print(f"Loaded {len(rows)} problems")

    # Group by data source
    source_indices = defaultdict(list)
    for i, row in enumerate(rows):
        source_indices[row["data_source"]].append(i)

    for src, indices in sorted(source_indices.items()):
        print(f"  {src}: {len(indices)} problems")

    # Initialize model
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=int(torch.cuda.device_count()),
        gpu_memory_utilization=0.95,
        trust_remote_code=True,
        enforce_eager=True,
    )
    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        top_p=args.top_p,
        n=args.n_samples,
    )

    # Build prompts using chat template
    prompts = []
    for row in rows:
        messages = row["prompt"]  # list of {content, role} dicts
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompts.append(prompt_text)

    # Generate
    print(f"\nGenerating {args.n_samples} samples per problem ({len(prompts)} problems)...")
    outputs = llm.generate(prompts, sampling_params)

    # Compute avg token length per generation
    total_token_len = 0
    total_samples = 0
    for output in outputs:
        for sample in output.outputs:
            total_token_len += len(sample.token_ids)
            total_samples += 1
    avg_token_len = total_token_len / total_samples if total_samples > 0 else 0

    # Score each sample using prime_code reward function
    source_scores = defaultdict(list)  # data_source -> list of per-problem avg scores
    source_token_lens = defaultdict(list)  # data_source -> list of per-problem avg token lengths

    for i, (row, output) in enumerate(zip(rows, outputs)):
        data_source = row["data_source"]
        ground_truth = row["reward_model"]["ground_truth"]

        problem_scores = []
        problem_token_lens = []
        for sample in output.outputs:
            solution = sample.text.strip()
            problem_token_lens.append(len(sample.token_ids))
            try:
                result = compute_score(solution, ground_truth, continuous=True)
                # compute_score returns (success, metadata) tuple
                if isinstance(result, tuple):
                    score = float(result[0])
                elif isinstance(result, dict):
                    score = float(result.get("score", result.get("reward", 0.0)))
                else:
                    score = float(result)
            except Exception:
                score = 0.0

            problem_scores.append(score)

        avg_score = sum(problem_scores) / len(problem_scores)
        source_scores[data_source].append(avg_score)
        source_token_lens[data_source].append(sum(problem_token_lens) / len(problem_token_lens))

        if (i + 1) % 50 == 0:
            print(f"  Scored {i + 1}/{len(rows)} problems")

    # Report results
    print(f"\n{'='*60}")
    print(f"Model: {args.model_path}")
    print(f"Test file: {args.test_file}")
    print(f"Metric: avg@{args.n_samples}")
    print(f"{'='*60}")

    all_scores = []
    for src in sorted(source_scores.keys()):
        scores = source_scores[src]
        token_lens = source_token_lens[src]
        avg = sum(scores) / len(scores)
        avg_tl = sum(token_lens) / len(token_lens)
        all_scores.extend(scores)
        print(f"  {src:20s}: {avg:.4f}  ({len(scores)} problems, avg token len: {avg_tl:.1f})")

    overall_avg = sum(all_scores) / len(all_scores)
    print(f"  {'Overall':20s}: {overall_avg:.4f}  ({len(all_scores)} problems, avg token len: {avg_token_len:.1f})")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
