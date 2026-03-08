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
    
    # 1. Fetch Price Data (Stooq proxy via yfinance for convenience if Stooq file not present)
    # Using SPY as benchmark
    prices = yf.download([ticker, bencher], start=sigs["filed_at"].min() - pd.Timedelta(days=5), 
                         end=sigs["filed_at"].max() + pd.Timedelta(days=15))
    
    if prices.empty:
        return None, None
        
    adj_close = prices['Adj Close']
    
    impact_results = []
    coverage_results = []
    
    for _, row in sigs.iterrows():
        t0 = row["filed_at"]
        ticker_data = adj_close[ticker].loc[t0:]
        bench_data = adj_close[bencher].loc[t0:]
        
        coverage_entry = {
            "accession": row["accession"],
            "ticker": ticker,
            "filed_at": t0,
            "price_available": False
        }
        
        if len(ticker_data) < 2:
            coverage_results.append(coverage_entry)
            continue
            
        coverage_entry["price_available"] = True
        coverage_results.append(coverage_entry)
        
        # Calculate Forward Returns
        p0 = ticker_data.iloc[0]
        b0 = bench_data.iloc[0]
        
        # 1d, 5d, 10d
        horizons = [1, 5, 10]
        impact_entry = {
            "accession": row["accession"],
            "composite_score": row["score_a"],
            "ticker": ticker,
            "filed_at": t0
        }
        
        for h in horizons:
            p_h = ticker_data.iloc[min(h, len(ticker_data)-1)]
            b_h = bench_data.iloc[min(h, len(bench_data)-1)]
            
            ticker_ret = (p_h - p0) / p0
            bench_ret = (b_h - b0) / b0
            abnormal_ret = ticker_ret - bench_ret
            
            impact_entry[f"ret_{h}d"] = ticker_ret
            impact_entry[f"ar_{h}d"] = abnormal_ret
            
        impact_results.append(impact_entry)
        
    impact_df = pd.DataFrame(impact_results)
    coverage_df = pd.DataFrame(coverage_results)
    
    # Save Reports
    impact_path = SMOKE_PROOF_DIR / f"{ticker}_impact_summary.csv"
    coverage_path = SMOKE_PROOF_DIR / f"{ticker}_price_coverage_report.csv"
    
    impact_df.to_csv(impact_path, index=False)
    coverage_df.to_csv(coverage_path, index=False)
    
    return impact_df, coverage_df

if __name__ == "__main__":
    # Internal test
    pass
