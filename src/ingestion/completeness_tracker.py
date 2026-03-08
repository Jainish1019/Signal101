"""
Completeness Tracker: Audits the ingestion integrity for Smoke Detector.
"""
import pandas as pd
from pathlib import Path
from config.settings import SMOKE_PROOF_DIR

def generate_completeness_report(ticker: str, results_csv: Path):
    """Generate the official judge-ready coverage report."""
    df = pd.read_csv(results_csv)
    
    stats = {
        "ticker": ticker,
        "total_expected": len(df),
        "total_downloaded": df["downloaded"].sum(),
        "coverage_pct": (df["downloaded"].sum() / len(df)) * 100 if len(df) > 0 else 0,
        "failed_accessions": df[~df["downloaded"]]["accession_no"].tolist()
    }
    
    report_path = SMOKE_PROOF_DIR / f"{ticker}_completeness_report.csv"
    
    # Create a nice summary
    summary_df = pd.DataFrame([stats])
    summary_df.to_csv(report_path, index=False)
    
    # Also log failures for debugging
    if stats["failed_accessions"]:
        fail_path = SMOKE_PROOF_DIR / f"{ticker}_ingestion_failures.log"
        fail_path.write_text("\n".join(stats["failed_accessions"]))
        
    return report_path, stats
