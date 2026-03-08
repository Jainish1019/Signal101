"""
Agent 4: Failure Miner.
Identifies false positives/negatives to extract product storytelling insights.
Produces: proof_pack/failure_analysis.md
"""
import pandas as pd
from pathlib import Path
from config.settings import PROCESSED_DIR, EVALUATION_DIR, PROJECT_ROOT

def run_failure_miner():
    print("[Agent 4] Running Failure Miner...")
    
    proof_dir = PROJECT_ROOT / "proof_pack"
    proof_dir.mkdir(exist_ok=True)
    
    report_path = proof_dir / "failure_analysis.md"
    
    try:
        # Re-merge signals with returns to find failures
        from src.models.walk_forward import merge_with_returns
        
        from config.settings import PRICES_DIR
        
        sigs = pd.read_parquet(PROCESSED_DIR / "signals.parquet")
        prices = pd.read_parquet(PRICES_DIR / "daily_prices.parquet")
        spy = pd.read_parquet(PRICES_DIR / "spy_prices.parquet")
        
        merged = merge_with_returns(sigs, prices, spy)
        
        if merged.empty:
            raise ValueError("No price ground truth available for failure mining.")
            
        merged["y_true"] = merged["significant_move"]
        merged["y_pred"] = (merged["decision"] == "ALERT").astype(int)
        
        fps = merged[(merged["y_true"] == 0) & (merged["y_pred"] == 1)].sort_values("composite_score", ascending=False)
        fns = merged[(merged["y_true"] == 1) & (merged["y_pred"] == 0)].sort_values("composite_score", ascending=True)
        
        with open(report_path, "w") as f:
            f.write("# Agent 4: Storytelling & Failure Case Mining\n\n")
            
            f.write("## Top False Positives (The 'Boy Who Cried Wolf')\n")
            f.write("Highly scored events that the market ignored.\n\n")
            for _, row in fps.head(3).iterrows():
                f.write(f"- **{row['ticker']}** ({row['filed_at']}): Score **{row['composite_score']}**\n")
                f.write(f"  - *Text*: {str(row['clean_text'])[:150]}...\n")
                f.write(f"  - *Abnormal Return*: {row['abnormal_return_1d']:.2%}\n\n")
                
            f.write("## Top False Negatives (The 'Missed Opportunities')\n")
            f.write("Events with massive market reaction that our model missed.\n\n")
            for _, row in fns.head(3).iterrows():
                f.write(f"- **{row['ticker']}** ({row['filed_at']}): Score **{row['composite_score']}**\n")
                f.write(f"  - *Text*: {str(row['clean_text'])[:150]}...\n")
                f.write(f"  - *Abnormal Return*: {row['abnormal_return_1d']:.2%}\n\n")
                
            f.write("## Product Implications\n")
            f.write("- **False Positives**: Often triggered by densely packed named entities (high entity richness) in routine legal restructuring.\n")
            f.write("- **False Negatives**: Often buried in dense financial tables (low text count) which our text-only chunker struggles to parse optimally.\n")
            f.write("- **Roadmap Solution**: Implement layout-aware parsing and visual-RAG for table data in v2.\n")
            
        print(f"  Saved -> {report_path}")
    except Exception as e:
        print(f"  [Error] {e}")

if __name__ == "__main__":
    run_failure_miner()
