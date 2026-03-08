#!/usr/bin/env python3
"""Step 6: Score all chunks and apply decision rule."""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from tqdm import tqdm
from config.settings import PROCESSED_DIR, MODELS_DIR
from src.models.classifier import EventClassifier
from src.models.drift_detector import DriftDetector
from src.models.explainer import build_explanation
from src.signal.scorer import compute_composite, decide, predict_direction
from src.nlp.embedder import embed_text


def run():
    print("=" * 60)
    print("STEP 6: SCORING AND DECIDING")
    print("=" * 60)

    feat_path = PROCESSED_DIR / "features.parquet"
    if not feat_path.exists():
        print("Error: features.parquet not found.")
        return

    df = pd.read_parquet(feat_path)

    # Load models
    clf = EventClassifier()
    model_a_path = MODELS_DIR / "model_a.pkl"
    if model_a_path.exists():
        clf.load()

    drift = DriftDetector()
    model_b_path = MODELS_DIR / "model_b_centroids.pkl"
    if model_b_path.exists():
        drift.load()

    # Score each chunk
    print(f"Scoring {len(df)} chunks...")

    # Model A: batch predict
    proba = clf.predict_proba(df["clean_text"].tolist())
    df["score_a"] = np.round(proba * 100, 2)

    # Model B + C: per-row
    score_b_list = []
    score_c_list = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Drift + Entity"):
        emb = embed_text(row["clean_text"])
        sb = drift.drift_score(row["ticker"], emb)
        sc = min(100.0, 20.0 * row.get("entity_richness", 0))
        score_b_list.append(sb)
        score_c_list.append(sc)

    df["score_b"] = score_b_list
    df["score_c"] = score_c_list

    # Composite
    df["composite_score"] = df.apply(
        lambda r: compute_composite(r["score_a"], r["score_b"], r["score_c"]),
        axis=1
    )

    # Decision
    df["decision"] = df["composite_score"].apply(decide)
    df["direction"] = df["vader_compound"].apply(predict_direction)

    # Stats
    alerts = (df["decision"] == "ALERT").sum()
    print(f"\nALERT: {alerts} | ARCHIVE: {len(df) - alerts}")

    out_path = PROCESSED_DIR / "signals.parquet"
    df.to_parquet(out_path, index=False)
    print(f"[+] Saved signals to {out_path}")


if __name__ == "__main__":
    run()
