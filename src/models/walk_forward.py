"""
Walk-forward evaluation framework.
"""
import pandas as pd
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta


def create_walk_forward_windows(start_date: str, end_date: str,
                                 min_train_months: int = 6) -> list[dict]:
    """
    Generate monthly walk-forward windows.
    Each window has a train_end and test_start/test_end (one month).
    """
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)

    # First test month starts min_train_months after start
    first_test = start + relativedelta(months=min_train_months)

    windows = []
    current_test_start = first_test

    while current_test_start < end:
        test_end = current_test_start + relativedelta(months=1)
        if test_end > end:
            test_end = end

        windows.append({
            "train_start": start.isoformat(),
            "train_end": current_test_start.isoformat(),
            "test_start": current_test_start.isoformat(),
            "test_end": test_end.isoformat(),
        })

        current_test_start = test_end

    return windows


def merge_with_returns(signals_df: pd.DataFrame,
                       prices_df: pd.DataFrame,
                       spy_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge signals with next-day abnormal returns.
    abnormal_return = stock_return_1d - spy_return_1d
    """
    # Ensure date columns
    signals_df = signals_df.copy()
    signals_df["filed_date"] = pd.to_datetime(signals_df["filed_at"]).dt.date

    merged_rows = []
    for _, row in signals_df.iterrows():
        ticker = row["ticker"]
        filed_date = row["filed_date"]

        # Get ticker prices
        tk_prices = prices_df[prices_df["ticker"] == ticker].copy()
        if tk_prices.empty:
            continue

        tk_prices["date_only"] = pd.to_datetime(tk_prices["date"]).dt.date
        tk_prices = tk_prices.sort_values("date_only")

        # Find next trading day
        future = tk_prices[tk_prices["date_only"] > filed_date]
        if future.empty:
            continue

        # Current day (or closest prior)
        current = tk_prices[tk_prices["date_only"] <= filed_date]
        if current.empty:
            continue

        t0_close = current.iloc[-1]["close"]
        t1_close = future.iloc[0]["close"]
        stock_return = (t1_close / t0_close) - 1

        # SPY return for same dates
        spy = spy_df.copy()
        spy["date_only"] = pd.to_datetime(spy["date"]).dt.date
        spy = spy.sort_values("date_only")

        spy_current = spy[spy["date_only"] <= filed_date]
        spy_future = spy[spy["date_only"] > filed_date]

        if spy_current.empty or spy_future.empty:
            abnormal_return = stock_return
        else:
            spy_t0 = spy_current.iloc[-1]["close"]
            spy_t1 = spy_future.iloc[0]["close"]
            spy_return = (spy_t1 / spy_t0) - 1
            abnormal_return = stock_return - spy_return

        new_row = row.to_dict()
        new_row["stock_return_1d"] = round(stock_return, 6)
        new_row["abnormal_return_1d"] = round(abnormal_return, 6)
        new_row["significant_move"] = 1 if abs(abnormal_return) > 0.02 else 0
        merged_rows.append(new_row)

    return pd.DataFrame(merged_rows)
