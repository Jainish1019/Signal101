#!/usr/bin/env python3
"""Step 4: Build FAISS vector index from embeddings."""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import faiss
import numpy as np
import pandas as pd
from tqdm import tqdm
from config.settings import PROCESSED_DIR, VECTORDB_DIR
from src.nlp.embedder import embed_batch


def run():
    print("=" * 60)
    print("STEP 4: BUILDING FAISS INDEX")
    print("=" * 60)

    feat_path = PROCESSED_DIR / "features.parquet"
    if not feat_path.exists():
        print("Error: features.parquet not found. Run step 03 first.")
        return

    df = pd.read_parquet(feat_path)
    texts = df["clean_text"].tolist()

    print(f"Embedding {len(texts)} chunks...")
    embeddings = embed_batch(texts, batch_size=64).astype("float32")

    # Normalize for cosine similarity via inner product
    faiss.normalize_L2(embeddings)

    # Build index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    # Save
    faiss.write_index(index, str(VECTORDB_DIR / "faiss.index"))

    # ID map
    id_map = df[["chunk_id", "ticker", "filed_at", "item_type"]].copy()
    id_map.to_parquet(VECTORDB_DIR / "id_map.parquet", index=False)

    print(f"[+] FAISS index built with {index.ntotal} vectors ({dim}-dim)")


if __name__ == "__main__":
    run()
