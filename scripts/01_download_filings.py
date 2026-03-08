#!/usr/bin/env python3
"""Step 1: Download EDGAR 8-K filing index and raw documents."""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.ingestion.edgar_client import build_filing_index

if __name__ == "__main__":
    # Use max_filings=500 for dev; remove cap for full run
    build_filing_index(max_filings=500)
