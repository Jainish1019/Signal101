"""
Smoke Impact: Calculates price coverage and abnormal return impact for Smoke Detector signals.
"""
import pandas as pd
import yfinance as yf
from config.settings import SMOKE_PROOF_DIR, PRICES_DIR

def calculate_abnormal_returns(ticker: str, signals_path: pd.DataFrame, bencher: str = "SPY"):
    """Calculate forward abnormal returns for each signal."""
    sigs = pd.read_parquet(signals_path)
    sigs["filed_at"] = pd.to_datetime(sigs["filed_at"])
    
    # 1. Fetch Price Data Individually to avoid MultiIndex issues
    start_date = sigs["filed_at"].min() - pd.Timedelta(days=5)
    end_date = sigs["filed_at"].max() + pd.Timedelta(days=15)
    
    t_obj = yf.Ticker(ticker)
    b_obj = yf.Ticker(bencher)
    
    ticker_prices = t_obj.history(start=start_date, end=end_date)
    bench_prices = b_obj.history(start=start_date, end=end_date)
    
    if ticker_prices.empty or bench_prices.empty:
        return None, None
        
    # Standardize to tz-naive for alignment
    ticker_prices.index = ticker_prices.index.tz_localize(None)
    bench_prices.index = bench_prices.index.tz_localize(None)
    
    p_adj = ticker_prices['Close']
    b_adj = bench_prices['Close']
    
    impact_results = []
    coverage_results = []
    
    for _, row in sigs.iterrows():
        t0 = row["filed_at"]
        
        # Ensure prices are available for the date
        ticker_slice = p_adj.loc[t0:]
        bench_slice = b_adj.loc[t0:]
        
        coverage_entry = {
            "accession": row["accession"],
            "ticker": ticker,
            "filed_at": t0,
            "price_available": False
        }
        
        if len(ticker_slice) < 2:
            coverage_results.append(coverage_entry)
            continue
            
        coverage_entry["price_available"] = True
        coverage_results.append(coverage_entry)
        
        # Calculate Forward Returns
        p0 = ticker_slice.iloc[0]
        b0 = bench_slice.iloc[0]
        
        # 1d, 5d, 10d
        horizons = [1, 5, 10]
        impact_entry = {
            "accession": row["accession"],
            "ticker": ticker,
            "filed_at": t0,
            "composite_score": row["score_a"],
            "baseline_score": row.get("score_baseline", 0),
            "fold": row.get("fold", "unknown")
        }
        
        # Ground Truth: Did price drop more than -2% abnormally in 5d?
        # (This is our 'useful signal' proxy)
        is_event = False
        
        for h in horizons:
            p_h = ticker_slice.iloc[min(h, len(ticker_slice)-1)]
            b_h = bench_slice.iloc[min(h, len(bench_slice)-1)]
            
            ticker_ret = (p_h - p0) / p0
            bench_ret = (b_h - b0) / b0
            abnormal_ret = ticker_ret - bench_ret
            
            impact_entry[f"ret_{h}d"] = ticker_ret
            impact_entry[f"ar_{h}d"] = abnormal_ret
            
            if h == 5 and abnormal_ret < -0.02:
                is_event = True
                
        impact_entry["is_ground_truth"] = is_event
        
        # Utility Calculation (Cost-Weighted)
        # TP = +$100, FP = -$150
        threshold = 70.0 # Standard alert threshold
        
        predicted_alert = row["score_a"] >= threshold
        baseline_alert = row.get("score_baseline", 0) > 1.0 # 1 significant keyword
        
        # Model Utility
        if predicted_alert and is_event: impact_entry["util_model"] = 100
        elif predicted_alert and not is_event: impact_entry["util_model"] = -150
        else: impact_entry["util_model"] = 0
        
        # Baseline Utility
        if baseline_alert and is_event: impact_entry["util_base"] = 100
        elif baseline_alert and not is_event: impact_entry["util_base"] = -150
        else: impact_entry["util_base"] = 0
            
        impact_results.append(impact_entry)
        
    impact_df = pd.DataFrame(impact_results)
    coverage_df = pd.DataFrame(coverage_results)
    
    # Export summary metrics
    stats = {
        "model_total_utility": impact_df["util_model"].sum(),
        "base_total_utility": impact_df["util_base"].sum(),
        "model_precision": impact_df[impact_df["composite_score"] >= 70]["is_ground_truth"].mean(),
        "base_precision": impact_df[impact_df["baseline_score"] > 1.0]["is_ground_truth"].mean()
    }
    
    pd.DataFrame([stats]).to_csv(SMOKE_PROOF_DIR / f"{ticker}_formal_metrics.csv", index=False)
    
    # Save Reports
    impact_path = SMOKE_PROOF_DIR / f"{ticker}_impact_summary.csv"
    coverage_path = SMOKE_PROOF_DIR / f"{ticker}_price_coverage_report.csv"
    
    impact_df.to_csv(impact_path, index=False)
    coverage_df.to_csv(coverage_path, index=False)
    
    return impact_df, coverage_df

if __name__ == "__main__":
    # Internal test
    pass
