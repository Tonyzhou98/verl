"""
Extract coding data from PRIME-RL dataset.

1. Train: filter for Apps, CodeContests, Taco, Codeforces
2. Validation: filter for same sources, sample 100 per source

Reads row-group by row-group to avoid pyarrow chunked-array bug,
then writes as a single row group matching the math parquet schema.

Usage:
    python extract_coding_data.py
"""

import pyarrow as pa
import pyarrow.parquet as pq
import random

INPUT_DIR = "/fsx/zyhang/verl/recipe/ebpo/prime-rl"
CODING_SOURCES = {"apps", "codecontests", "taco", "codeforces"}
SAMPLE_PER_SOURCE = 100
RANDOM_SEED = 42

# Target schema matching dapo-math-17k-unique.parquet
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


def read_and_filter(path, sources):
    """Read parquet row-group by row-group, filter, return list of dicts."""
    pf = pq.ParquetFile(path)
    rows = []
    for i in range(pf.metadata.num_row_groups):
        rg = pf.read_row_group(i)
        ds_col = rg.column("data_source")
        for j in range(len(rg)):
            if ds_col[j].as_py() in sources:
                row = {}
                for col_name in rg.schema.names:
                    row[col_name] = rg.column(col_name)[j].as_py()
                rows.append(row)
        if (i + 1) % 50 == 0:
            print(f"  Row group {i+1}/{pf.metadata.num_row_groups}, {len(rows)} coding rows so far")
    print(f"  Done: {pf.metadata.num_row_groups} row groups, {len(rows)} coding rows")
    return rows


def rows_to_table(rows):
    """Convert list of dicts to a pyarrow table with the target schema."""
    data_source = []
    prompt = []
    ability = []
    reward_model = []
    extra_info = []

    for row in rows:
        data_source.append(row["data_source"])
        prompt.append(row["prompt"])
        ability.append(row["ability"])

        # Ensure ground_truth is string
        rm = row["reward_model"]
        reward_model.append({
            "ground_truth": str(rm["ground_truth"]) if rm.get("ground_truth") is not None else "",
            "style": str(rm["style"]) if rm.get("style") is not None else "",
        })

        # Convert index to string
        ei = row["extra_info"]
        extra_info.append({
            "index": str(ei["index"]) if ei.get("index") is not None else "",
        })

    return pa.table({
        "data_source": pa.array(data_source, type=pa.string()),
        "prompt": pa.array(prompt, type=TARGET_SCHEMA.field("prompt").type),
        "ability": pa.array(ability, type=pa.string()),
        "reward_model": pa.array(reward_model, type=TARGET_SCHEMA.field("reward_model").type),
        "extra_info": pa.array(extra_info, type=TARGET_SCHEMA.field("extra_info").type),
    }, schema=TARGET_SCHEMA)


# --- Train ---
print("Reading and filtering train.parquet ...")
train_rows = read_and_filter(f"{INPUT_DIR}/train.parquet", CODING_SOURCES)

counts = {}
for row in train_rows:
    s = row["data_source"]
    counts[s] = counts.get(s, 0) + 1
print(f"Train coding: {len(train_rows)} rows")
for s, c in sorted(counts.items()):
    print(f"  {s}: {c}")

train_table = rows_to_table(train_rows)
# Write with small row group size to avoid pyarrow chunked-array bug on read
pq.write_table(train_table, f"{INPUT_DIR}/train_coding.parquet", row_group_size=1000)
print(f"Wrote {INPUT_DIR}/train_coding.parquet")

# --- Validation ---
print("\nReading and filtering validation.parquet ...")
valid_rows = read_and_filter(f"{INPUT_DIR}/validation.parquet", CODING_SOURCES)

# Group by data_source
source_rows = {}
for i, row in enumerate(valid_rows):
    source_rows.setdefault(row["data_source"], []).append(i)

# Sample 100 per source
random.seed(RANDOM_SEED)
sampled_rows = []
for src in sorted(CODING_SOURCES):
    indices = source_rows.get(src, [])
    n = min(SAMPLE_PER_SOURCE, len(indices))
    for idx in sorted(random.sample(indices, n)):
        sampled_rows.append(valid_rows[idx])
    print(f"  Sampled {n} from {src} (available: {len(indices)})")

valid_table = rows_to_table(sampled_rows)
pq.write_table(valid_table, f"{INPUT_DIR}/valid_coding.parquet", row_group_size=1000)

print(f"\nValidation coding: {len(sampled_rows)} rows")
print(f"\nSaved:")
print(f"  {INPUT_DIR}/train_coding.parquet  ({len(train_rows)} rows)")
print(f"  {INPUT_DIR}/valid_coding.parquet  ({len(sampled_rows)} rows)")

# Verify
print("\nVerifying train_coding.parquet ...")
pf = pq.ParquetFile(f"{INPUT_DIR}/train_coding.parquet")
print(f"  Schema:\n{pf.schema_arrow}")
print(f"  Num row groups: {pf.metadata.num_row_groups}")
total = sum(pf.metadata.row_group(i).num_rows for i in range(pf.metadata.num_row_groups))
print(f"  Total rows: {total}")
# Read first row group to verify
t = pf.read_row_group(0)
print(f"  Read row group 0 OK: {len(t)} rows")
print(f"  prompt[0]: {t.column('prompt')[0].as_py()[0]['content'][:100]}...")
