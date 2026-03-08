"""
RAG retriever: FAISS-backed similarity search.
"""
import faiss
import numpy as np
import pandas as pd
from pathlib import Path
from config.settings import VECTORDB_DIR
from src.nlp.embedder import embed_text


def load_faiss_index():
    """Load FAISS index and ID mapping."""
    index_path = VECTORDB_DIR / "faiss.index"
    id_map_path = VECTORDB_DIR / "id_map.parquet"

    if not index_path.exists() or not id_map_path.exists():
        return None, None

    index = faiss.read_index(str(index_path))
    id_map = pd.read_parquet(id_map_path)
    return index, id_map


def search_similar(query_text: str, n: int = 5,
                   signals_df: pd.DataFrame = None) -> list[dict]:
    """
    Embed query and search FAISS for top-N similar chunks.
    Returns list of dicts with chunk metadata.
    """
    index, id_map = load_faiss_index()
    if index is None:
        return []

    # Embed query
    query_vec = embed_text(query_text).astype("float32").reshape(1, -1)
    faiss.normalize_L2(query_vec)

    # Search
    distances, indices = index.search(query_vec, n)

    results = []
    for i, idx in enumerate(indices[0]):
        if idx < 0 or idx >= len(id_map):
            continue

        row = id_map.iloc[idx]
        result = {
            "chunk_id": row.get("chunk_id", ""),
            "ticker": row.get("ticker", ""),
            "filed_at": row.get("filed_at", ""),
            "item_type": row.get("item_type", ""),
            "similarity": round(float(distances[0][i]), 4),
        }

        # Enrich with signal data if available
        if signals_df is not None:
            match = signals_df[signals_df["chunk_id"] == result["chunk_id"]]
            if not match.empty:
                s = match.iloc[0]
                result["headline"] = str(s.get("clean_text", ""))[:200]
                result["composite_score"] = s.get("composite_score", 0)
                result["decision"] = s.get("decision", "ARCHIVE")
                result["signal_score"] = s.get("composite_score", 0)

        results.append(result)

    return results
