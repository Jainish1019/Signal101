#!/usr/bin/env python3
"""Step 2: Parse downloaded filings, chunk by Item headers, detect boilerplate."""
import sys
import json
from pathlib import Path
import pandas as pd
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent))

from config.settings import RAW_DIR, PROCESSED_DIR
from src.ingestion.filing_parser import parse_filing


def run():
    index_path = RAW_DIR / "filing_index.jsonl"
    if not index_path.exists():
        print("Error: filing_index.jsonl not found. Run step 01 first.")
        return

    print("=" * 60)
    print("STEP 2: PARSING AND CHUNKING FILINGS")
    print("=" * 60)

    all_chunks = []
    with open(index_path) as f:
        filings = [json.loads(line) for line in f]

    for filing in tqdm(filings, desc="Parsing filings"):
        chunks = parse_filing(
            raw_path=filing["raw_path"],
            accession=filing["accession"],
            cik=filing["cik"],
            ticker=filing["ticker"],
            filed_at=filing["filed_at"],
        )
        all_chunks.extend(chunks)

    if not all_chunks:
        print("[WARN] No chunks produced!")
        return

    df = pd.DataFrame(all_chunks)

    # Stats
    total = len(df)
    boilerplate = df["is_boilerplate"].sum()
    print(f"\nTotal chunks: {total}")
    print(f"Boilerplate: {boilerplate} ({100*boilerplate/total:.1f}%)")
    print(f"Unique tickers: {df['ticker'].nunique()}")
    print(f"Date range: {df['filed_at'].min()} to {df['filed_at'].max()}")

    out_path = PROCESSED_DIR / "chunks.parquet"
    df.to_parquet(out_path, index=False)
    print(f"\n[+] Saved {len(df)} chunks to {out_path}")


if __name__ == "__main__":
    run()
