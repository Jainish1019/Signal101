"""
Price fetcher: downloads daily OHLCV from Stooq (primary) or yfinance (fallback).
Also fetches SPY benchmark data for abnormal return calculation.
"""
import time
import pandas as pd
import yfinance as yf
from io import StringIO
import requests
from pathlib import Path

from config.settings import (
    STOOQ_URL_TPL, PRICE_FETCH_SLEEP, BENCHMARK_TICKER,
    PRICES_DIR, TICKERS, EDGAR_USER_AGENT,
)


def fetch_stooq(ticker: str) -> pd.DataFrame:
    """Download daily OHLCV from Stooq (free, no key)."""
    url = STOOQ_URL_TPL.format(ticker=ticker.lower())
    try:
        resp = requests.get(url, timeout=30)
        if resp.status_code != 200 or "No data" in resp.text:
            return pd.DataFrame()
        df = pd.read_csv(StringIO(resp.text))
        df.columns = [c.strip().lower() for c in df.columns]
        if "date" not in df.columns:
            return pd.DataFrame()
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        df["ticker"] = ticker.upper()
        return df
    except Exception as e:
        print(f"  [WARN] Stooq failed for {ticker}: {e}")
        return pd.DataFrame()


def fetch_yfinance(ticker: str, start: str = "2018-12-01",
                   end: str = "2024-02-01") -> pd.DataFrame:
    """Fallback: download from yfinance."""
    try:
        df = yf.download(ticker, start=start, end=end,
                         interval="1d", progress=False)
        if df.empty:
            return pd.DataFrame()
        df = df.reset_index()
        df.columns = [c.lower().replace(" ", "_") for c in df.columns]
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        df["ticker"] = ticker.upper()
        return df
    except Exception as e:
        print(f"  [WARN] yfinance failed for {ticker}: {e}")
        return pd.DataFrame()


def fetch_prices_for_ticker(ticker: str) -> pd.DataFrame:
    """Try Stooq first, fall back to yfinance."""
    df = fetch_stooq(ticker)
    if df.empty:
        print(f"  Stooq empty for {ticker}, trying yfinance...")
        df = fetch_yfinance(ticker)
    return df


def download_all_prices(tickers: list = None) -> None:
    """
    Download daily prices for all tickers + SPY benchmark.
    Saves to data/prices/daily_prices.parquet and spy_prices.parquet.
    """
    if tickers is None:
        tickers = TICKERS

    print("=" * 60)
    print("DOWNLOADING DAILY PRICE DATA")
    print("=" * 60)

    all_frames = []

    # Ticker prices
    for ticker in tickers:
        print(f"  Fetching {ticker}...")
        df = fetch_prices_for_ticker(ticker)
        if not df.empty:
            all_frames.append(df)
            print(f"    Got {len(df)} rows")
        else:
            print(f"    [SKIP] No data for {ticker}")
        time.sleep(PRICE_FETCH_SLEEP)

    if all_frames:
        prices_df = pd.concat(all_frames, ignore_index=True)
        out_path = PRICES_DIR / "daily_prices.parquet"
        prices_df.to_parquet(out_path, index=False)
        print(f"\n[+] Saved {len(prices_df)} price rows to {out_path}")
    else:
        print("\n[WARN] No price data downloaded!")

    # SPY benchmark
    print(f"\n  Fetching benchmark {BENCHMARK_TICKER}...")
    spy_df = fetch_prices_for_ticker(BENCHMARK_TICKER)
    if not spy_df.empty:
        spy_path = PRICES_DIR / "spy_prices.parquet"
        spy_df.to_parquet(spy_path, index=False)
        print(f"  [+] Saved {len(spy_df)} SPY rows to {spy_path}")
    else:
        print("  [WARN] Could not fetch SPY data!")
