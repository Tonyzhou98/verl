"""
Reorder coding training data by data_source (each source = one topic).

Groups same-source prompts together for EBPO-topic training with shuffle=False.

Usage:
    python reorder_by_source.py

Input:
    /fsx/zyhang/verl/recipe/ebpo/prime-rl/train_coding.parquet

Output:
    /fsx/zyhang/verl/recipe/ebpo/prime-rl/train_coding_by_source.parquet
"""

import pyarrow.parquet as pq
import pyarrow as pa

# ─── Config ───────────────────────────────────────────────────────────
INPUT_PATH = "/fsx/zyhang/verl/recipe/ebpo/prime-rl/train_coding.parquet"
OUTPUT_PATH = "/fsx/zyhang/verl/recipe/ebpo/prime-rl/train_coding_by_source.parquet"

TARGET_SCHEMA = pa.schema([
    ("data_source", pa.string()),
    ("prompt", pa.list_(pa.struct([
        ("content", pa.string()),
        ("role", pa.string()),
    ]))),
    ("ability", pa.string()),
    ("reward_model", pa.struct([
        ("ground_truth", pa.string()),
        ("style", pa.string()),
    ])),
    ("extra_info", pa.struct([
        ("index", pa.string()),
    ])),
])

# ─── Load ─────────────────────────────────────────────────────────────
print("Reading parquet row-group by row-group ...")
pf = pq.ParquetFile(INPUT_PATH)
all_rows = []
for rg_idx in range(pf.metadata.num_row_groups):
    rg = pf.read_row_group(rg_idx)
    for j in range(len(rg)):
        row = {}
        for col_name in rg.schema.names:
            row[col_name] = rg.column(col_name)[j].as_py()
        all_rows.append(row)
print(f"Loaded {len(all_rows)} rows")

# ─── Sort by data_source ──────────────────────────────────────────────
sorted_rows = sorted(all_rows, key=lambda r: r["data_source"])

# Print distribution
counts = {}
for r in sorted_rows:
    s = r["data_source"]
    counts[s] = counts.get(s, 0) + 1
print(f"\nData source distribution (= topics):")
for s, c in sorted(counts.items()):
    print(f"  {s}: {c}")

# ─── Build table and save ─────────────────────────────────────────────
sorted_table = pa.table({
    "data_source": pa.array([r["data_source"] for r in sorted_rows], type=pa.string()),
    "prompt": pa.array([r["prompt"] for r in sorted_rows], type=TARGET_SCHEMA.field("prompt").type),
    "ability": pa.array([r["ability"] for r in sorted_rows], type=pa.string()),
    "reward_model": pa.array([r["reward_model"] for r in sorted_rows], type=TARGET_SCHEMA.field("reward_model").type),
    "extra_info": pa.array([r["extra_info"] for r in sorted_rows], type=TARGET_SCHEMA.field("extra_info").type),
}, schema=TARGET_SCHEMA)

pq.write_table(sorted_table, OUTPUT_PATH, row_group_size=1000)

print(f"\nSaved to {OUTPUT_PATH}")
print(f"Total rows: {len(sorted_table)}")
