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
