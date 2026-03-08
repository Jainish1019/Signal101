#!/usr/bin/env python3
"""Step 5: Train walk-forward models + download prices."""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from config.settings import PROCESSED_DIR, PRICES_DIR, MODELS_DIR
from src.ingestion.price_fetcher import download_all_prices
from src.models.classifier import EventClassifier
from src.models.drift_detector import DriftDetector
from src.models.walk_forward import merge_with_returns
from src.nlp.embedder import embed_batch


def run():
    print("=" * 60)
    print("STEP 5: TRAINING MODELS (WALK-FORWARD)")
    print("=" * 60)

    feat_path = PROCESSED_DIR / "features.parquet"
    if not feat_path.exists():
        print("Error: features.parquet not found. Run step 03 first.")
        return

    # Download prices if not present
    prices_path = PRICES_DIR / "daily_prices.parquet"
    if not prices_path.exists():
        download_all_prices()

    features_df = pd.read_parquet(feat_path)
    prices_df = pd.read_parquet(prices_path) if prices_path.exists() else pd.DataFrame()
    spy_path = PRICES_DIR / "spy_prices.parquet"
    spy_df = pd.read_parquet(spy_path) if spy_path.exists() else pd.DataFrame()

    # Merge features with returns
    if not prices_df.empty and not spy_df.empty:
        merged = merge_with_returns(features_df, prices_df, spy_df)
    else:
        print("[WARN] No price data. Using synthetic labels for training.")
        merged = features_df.copy()
        merged["significant_move"] = (merged["keyword_score"] > 0.3).astype(int)
        merged["abnormal_return_1d"] = 0.0

    print(f"Training dataset: {len(merged)} rows")

    # Model A: Classifier
    print("\nTraining Model A (calibrated classifier)...")
    clf = EventClassifier()
    texts = merged["clean_text"].tolist()
    labels = merged["significant_move"].values.astype(int)

    if len(set(labels)) > 1:
        clf.fit(texts, labels)
        clf.save()
        print("  Model A saved.")
    else:
        print("  [WARN] Only one label class present; skipping classifier training.")

    # Model B: Drift detector (batch embed instead of per-row)
    print("\nBuilding Model B (drift centroids)...")
    drift = DriftDetector()
    embeddings = embed_batch(merged["clean_text"].tolist(), batch_size=64)
    for i, (_, row) in enumerate(merged.iterrows()):
        drift.update(row["ticker"], embeddings[i])
    drift.save()
    print("  Drift centroids saved.")

    print("\n[+] Model training complete.")


if __name__ == "__main__":
    run()
