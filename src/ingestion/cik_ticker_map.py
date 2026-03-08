"""
CIK-to-ticker mapper using SEC's public company_tickers.json.
"""
import json
import requests
from pathlib import Path
from config.settings import CIK_TICKER_URL, RAW_DIR, EDGAR_USER_AGENT

_CACHE_PATH = RAW_DIR / "cik_ticker_map.json"
_MAP = None


def _download_map() -> dict:
    """Download and parse the SEC CIK-ticker mapping."""
    headers = {"User-Agent": EDGAR_USER_AGENT}
    resp = requests.get(CIK_TICKER_URL, headers=headers, timeout=30)
    resp.raise_for_status()
    raw = resp.json()
    # raw is {index: {cik_str, ticker, title}}
    mapping = {}
    for entry in raw.values():
        cik = int(entry["cik_str"])
        mapping[cik] = entry["ticker"].upper()
    # Cache locally
    _CACHE_PATH.write_text(json.dumps(mapping))
    return mapping


def get_map() -> dict:
    """Return CIK->ticker dict, downloading if needed."""
    global _MAP
    if _MAP is not None:
        return _MAP
    if _CACHE_PATH.exists():
        _MAP = {int(k): v for k, v in json.loads(_CACHE_PATH.read_text()).items()}
    else:
        _MAP = _download_map()
    return _MAP


def cik_to_ticker(cik: int) -> str:
    """Resolve a CIK to its ticker symbol. Returns '' if unknown."""
    return get_map().get(cik, "")


def ticker_to_cik(ticker: str) -> int:
    """Resolve a ticker symbol to its CIK. Returns None if unknown."""
    ticker = ticker.upper()
    mapping = get_map()
    rev_map = {v: k for k, v in mapping.items()}
    return rev_map.get(ticker)


def resolve_company(query: str) -> tuple[int, str]:
    """
    Given a ticker, CIK, or name, resolve to (CIK, Ticker).
    Returns (None, '') if unresolved.
    """
    query = query.strip()
    
    # 1. Check if CIK (all digits)
    if query.isdigit():
        cik = int(query)
        return (cik, cik_to_ticker(cik))
    
    # 2. Check if Ticker
    cik = ticker_to_cik(query)
    if cik:
        return (cik, query.upper())
    
    # 3. Fuzzy match on title (download full map for this)
    headers = {"User-Agent": EDGAR_USER_AGENT}
    resp = requests.get(CIK_TICKER_URL, headers=headers, timeout=30)
    if resp.ok:
        raw = resp.json()
        query_l = query.lower()
        for entry in raw.values():
            if query_l in entry["title"].lower():
                return (int(entry["cik_str"]), entry["ticker"].upper())
                
    return (None, "")
