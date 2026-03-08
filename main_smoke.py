"""
Signal-X: Smoke Detector CLI Orchestrator
Usage: python main_smoke.py TSLA
"""
import sys
from src.ingestion.smoke_ingest import run_smoke_ingest
from src.ingestion.completeness_tracker import generate_completeness_report
from src.ingestion.filing_parser import parse_filing
from src.signal.smoke_trainer import run_smoke_training
from src.evaluation.smoke_impact import calculate_abnormal_returns
from config.settings import SMOKE_RAW_DIR, SMOKE_PROCESSED_DIR
import pandas as pd

def main():
    query = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    ticker = query.upper()
    
    print(f"--- Starting Smoke Detector Diagnostic for {ticker} ---")
    
    # 1. Ingest
    results_csv = run_smoke_ingest(query)
    if not results_csv: return
    
    # 2. Audit
    generate_completeness_report(ticker, results_csv)
    
    # 3. Parse
    print("Parsing documents...")
    all_chunks = []
    for htm_path in SMOKE_RAW_DIR.glob(f"{ticker}_*.htm"):
        parts = htm_path.stem.split("_")
        if len(parts) >= 4:
            date, form, acc = parts[1], parts[2], parts[3]
            chunks = parse_filing(str(htm_path), acc, 0, ticker, f"{date[:4]}-{date[4:6]}-{date[6:]}")
            all_chunks.extend(chunks)
            
    if not all_chunks:
        print("No narrative chunks extracted.")
        return
        
    chunks_df = pd.DataFrame(all_chunks)
    chunks_path = SMOKE_PROCESSED_DIR / f"{ticker}_total_chunks.parquet"
    chunks_df.to_parquet(chunks_path)
    
    # 4. Train
    scored_path = run_smoke_training(ticker, chunks_path)
    
    # 5. Impact
    calculate_abnormal_returns(ticker, scored_path)
    
    print(f"--- Diagnostic Complete. Artifacts at data/smoke/proof_pack/{ticker} ---")

if __name__ == "__main__":
    main()
