"""
Cluster coding training data into 10 topics using BERTopic,
then sort by topic for EBPO-topic training with shuffle=False.

Usage:
    pip install bertopic sentence-transformers
    python cluster_coding_bertopic.py

Input:
    /fsx/zyhang/verl/recipe/ebpo/prime-rl/train_coding.parquet
    (must be created by extract_coding_data.py first)

Output:
    /fsx/zyhang/verl/recipe/ebpo/prime-rl/train_coding_bertopic.parquet
"""

import pyarrow.parquet as pq
import pyarrow as pa
import numpy as np
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from umap import UMAP

# ─── Config ───────────────────────────────────────────────────────────
INPUT_PATH = "/fsx/zyhang/verl/recipe/ebpo/prime-rl/train_coding.parquet"
OUTPUT_PATH = "/fsx/zyhang/verl/recipe/ebpo/prime-rl/train_coding_bertopic.parquet"
NR_TOPICS = 10
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
RANDOM_SEED = 42

# Target schema
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

# ─── Load data as list of Python dicts ────────────────────────────────
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

# Extract prompt text
texts = [row["prompt"][0]["content"] for row in all_rows]
print(f"Extracted {len(texts)} prompt texts")

# ─── Embed ────────────────────────────────────────────────────────────
print(f"Embedding with {EMBEDDING_MODEL} ...")
sentence_model = SentenceTransformer(EMBEDDING_MODEL)
embeddings = sentence_model.encode(texts, show_progress_bar=True, batch_size=256)

# ─── BERTopic with fixed number of topics ─────────────────────────────
umap_model = UMAP(
    n_neighbors=15,
    n_components=5,
    min_dist=0.0,
    metric="cosine",
    random_state=RANDOM_SEED,
)
cluster_model = KMeans(n_clusters=NR_TOPICS, random_state=RANDOM_SEED, n_init=10)

topic_model = BERTopic(
    embedding_model=sentence_model,
    umap_model=umap_model,
    hdbscan_model=cluster_model,
    nr_topics=NR_TOPICS,
    verbose=True,
)

topics, probs = topic_model.fit_transform(texts, embeddings=embeddings)

# ─── Print topic info ─────────────────────────────────────────────────
topic_info = topic_model.get_topic_info()
print("\nTopic distribution:")
print(topic_info[["Topic", "Count", "Name"]].to_string(index=False))

# ─── Sort by topic and rebuild table ──────────────────────────────────
sort_order = np.argsort(topics).tolist()
sorted_rows = [all_rows[i] for i in sort_order]

# Build columns
data_source = [r["data_source"] for r in sorted_rows]
prompt = [r["prompt"] for r in sorted_rows]
ability = [r["ability"] for r in sorted_rows]
reward_model = [r["reward_model"] for r in sorted_rows]
extra_info = [r["extra_info"] for r in sorted_rows]

sorted_table = pa.table({
    "data_source": pa.array(data_source, type=pa.string()),
    "prompt": pa.array(prompt, type=TARGET_SCHEMA.field("prompt").type),
    "ability": pa.array(ability, type=pa.string()),
    "reward_model": pa.array(reward_model, type=TARGET_SCHEMA.field("reward_model").type),
    "extra_info": pa.array(extra_info, type=TARGET_SCHEMA.field("extra_info").type),
}, schema=TARGET_SCHEMA)

pq.write_table(sorted_table, OUTPUT_PATH, row_group_size=1000)

topic_counts = {}
for t in topics:
    topic_counts[t] = topic_counts.get(t, 0) + 1

print(f"\nSaved clustered data to {OUTPUT_PATH}")
print(f"Total rows: {len(sorted_table)}")
print(f"Topic counts: {dict(sorted(topic_counts.items()))}")
