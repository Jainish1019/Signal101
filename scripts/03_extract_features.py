#!/usr/bin/env python3
"""Step 3: Extract NLP features from parsed chunks."""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd
from config.settings import PROCESSED_DIR
from src.nlp.feature_pipeline import process_all_chunks


def run():
    print("=" * 60)
    print("STEP 3: EXTRACTING NLP FEATURES")
    print("=" * 60)

    chunks_path = PROCESSED_DIR / "chunks.parquet"
    if not chunks_path.exists():
        print("Error: chunks.parquet not found. Run step 02 first.")
        return

    df = pd.read_parquet(chunks_path)
    features_df = process_all_chunks(df)

    out_path = PROCESSED_DIR / "features.parquet"
    features_df.to_parquet(out_path, index=False)
    print(f"[+] Saved {len(features_df)} feature rows to {out_path}")


if __name__ == "__main__":
    run()
