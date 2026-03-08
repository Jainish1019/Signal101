"""
Signal decay analysis: correlation across time horizons.
"""
import pandas as pd
import numpy as np
from scipy.stats import pearsonr


def analyze_decay(signals_df: pd.DataFrame, prices_df: pd.DataFrame,
                  spy_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Pearson correlation between composite score and
    absolute abnormal return at T+1d, T+2d, T+5d.
    """
    results = []
    horizons = {"T+1d": 1, "T+2d": 2, "T+5d": 5}

    for label, offset in horizons.items():
        correlations = []

        for _, row in signals_df.iterrows():
            ticker = row["ticker"]
            filed_date = pd.to_datetime(row["filed_at"]).date()

            tk = prices_df[prices_df["ticker"] == ticker].copy()
            if tk.empty:
                continue
            tk["date_only"] = pd.to_datetime(tk["date"]).dt.date
            tk = tk.sort_values("date_only")

            # Find T+offset trading day
            future = tk[tk["date_only"] > filed_date]
            if len(future) < offset:
                continue

            t0_row = tk[tk["date_only"] <= filed_date]
            if t0_row.empty:
                continue

            t0_close = t0_row.iloc[-1]["close"]
            tn_close = future.iloc[offset - 1]["close"]
            stock_ret = (tn_close / t0_close) - 1

            correlations.append({
                "composite_score": row.get("composite_score", 0),
                "abs_return": abs(stock_ret),
            })

        if len(correlations) > 10:
            cor_df = pd.DataFrame(correlations)
            corr, p_val = pearsonr(cor_df["composite_score"], cor_df["abs_return"])
            results.append({
                "horizon": label,
                "correlation": round(corr, 4),
                "p_value": round(p_val, 4),
                "n_samples": len(correlations),
            })

    return pd.DataFrame(results)
