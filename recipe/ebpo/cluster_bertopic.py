"""
Cluster dapo-math-17k prompts into 10 topics using BERTopic.

Produces a new parquet file with the same schema, but with rows sorted by
topic assignment so that same-topic prompts are contiguous.  The topic id
is stored in extra_info['topic'].

Usage:
    pip install bertopic sentence-transformers
    python cluster_bertopic.py

Output:
    recipe/ebpo/dapo-math-17k-unique-bertopic.parquet
"""

import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from umap import UMAP

# ─── Config ───────────────────────────────────────────────────────────
INPUT_PATH = "/fsx/zyhang/verl/recipe/ebpo/dapo-math-17k-unique.parquet"
OUTPUT_PATH = "/fsx/zyhang/verl/recipe/ebpo/dapo-math-17k-unique-bertopic.parquet"
NR_TOPICS = 10
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
RANDOM_SEED = 42

# ─── Load data ────────────────────────────────────────────────────────
df = pd.read_parquet(INPUT_PATH)
texts = [row[0]["content"] for row in df["prompt"]]
print(f"Loaded {len(texts)} prompts")

# ─── Embed ────────────────────────────────────────────────────────────
print(f"Embedding with {EMBEDDING_MODEL} ...")
sentence_model = SentenceTransformer(EMBEDDING_MODEL)
embeddings = sentence_model.encode(texts, show_progress_bar=True, batch_size=256)

# ─── BERTopic with fixed number of topics ─────────────────────────────
# Use KMeans as the clustering model to get exactly NR_TOPICS clusters.
# This avoids HDBSCAN's outlier class (-1) and gives a clean partition.
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

# ─── Store topic in extra_info and sort by topic ──────────────────────
df = df.copy()
df["_topic"] = topics

# Add topic to extra_info dict
def add_topic(row):
    info = dict(row["extra_info"])
    info["topic"] = int(row["_topic"])
    return info

df["extra_info"] = df.apply(add_topic, axis=1)

# Sort by topic so same-topic prompts are contiguous
df = df.sort_values("_topic").reset_index(drop=True)
df = df.drop(columns=["_topic"])

# ─── Save ─────────────────────────────────────────────────────────────
df.to_parquet(OUTPUT_PATH, index=False)
print(f"\nSaved clustered data to {OUTPUT_PATH}")
print(f"Topic counts: {pd.Series(topics).value_counts().sort_index().to_dict()}")
