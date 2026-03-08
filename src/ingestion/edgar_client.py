"""
EDGAR client: downloads 8-K filing index and raw HTML documents.
Respects SEC fair-access rate limits.
"""
import re
import time
import json
import requests
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

from config.settings import (
    EDGAR_BASE, EDGAR_FULL_IDX, EDGAR_USER_AGENT,
    EDGAR_RATE_LIMIT, EDGAR_START_YEAR, EDGAR_END_YEAR,
    RAW_DIR, FILINGS_DIR, TICKERS,
)
from src.ingestion.cik_ticker_map import cik_to_ticker, get_map

HEADERS = {"User-Agent": EDGAR_USER_AGENT}


def _index_url(year: int, qtr: int) -> str:
    return f"{EDGAR_FULL_IDX}/{year}/QTR{qtr}/company.idx"


def download_index_file(year: int, qtr: int) -> list[dict]:
    """Download one EDGAR quarterly index and filter 8-K filings."""
    url = _index_url(year, qtr)
    time.sleep(EDGAR_RATE_LIMIT)
    resp = requests.get(url, headers=HEADERS, timeout=60)
    if resp.status_code != 200:
        print(f"  [WARN] Could not fetch {url} ({resp.status_code})")
        return []

    lines = resp.text.splitlines()
    filings = []
    in_data = False

    for line in lines:
        if line.startswith("---"):
            in_data = True
            continue
        if not in_data:
            continue

        # Fixed-width: Company Name | Form Type | CIK | Date Filed | Filename
        parts = re.split(r"\s{2,}", line.strip())
        if len(parts) < 5:
            continue

        form_type = parts[1].strip()
        if form_type != "8-K":
            continue

        try:
            cik = int(parts[2].strip())
        except ValueError:
            continue

        date_filed = parts[3].strip()
        rel_path = parts[4].strip()

        ticker = cik_to_ticker(cik)

        filings.append({
            "cik": cik,
            "ticker": ticker,
            "form_type": form_type,
            "date_filed": date_filed,
            "index_path": rel_path,
        })

    return filings


def download_filing_document(index_path: str) -> tuple[str, str]:
    """
    Given an EDGAR index path like 'edgar/data/1234/000.../filing.txt',
    download the filing index page, find the primary document, and download it.
    Returns (accession_number, local_file_path).
    """
    # Parse accession from path
    parts = index_path.replace("edgar/data/", "").split("/")
    if len(parts) < 2:
        return "", ""

    cik_str = parts[0]
    accession_raw = parts[1] if len(parts) >= 2 else ""
    accession = accession_raw.replace("-", "")

    # Build filing index URL
    idx_url = f"https://www.sec.gov/Archives/{index_path}"
    if not idx_url.endswith(".txt"):
        idx_url = idx_url.rsplit("/", 1)[0] + "/"

    time.sleep(EDGAR_RATE_LIMIT)
    try:
        resp = requests.get(idx_url, headers=HEADERS, timeout=30)
        if resp.status_code != 200:
            return accession_raw, ""
    except Exception:
        return accession_raw, ""

    content = resp.text

    # If the URL was the actual document (ends in .txt or .htm), save directly
    if index_path.endswith((".txt", ".htm", ".html")):
        local_path = FILINGS_DIR / f"{accession_raw}.html"
        local_path.write_text(content, encoding="utf-8")
        return accession_raw, str(local_path)

    # Otherwise, try to find the primary document link from the index page
    # Look for .htm links that are likely the filing
    htm_links = re.findall(r'href="([^"]+\.htm[l]?)"', content, re.IGNORECASE)
    if not htm_links:
        # Save whatever we got
        local_path = FILINGS_DIR / f"{accession_raw}.html"
        local_path.write_text(content, encoding="utf-8")
        return accession_raw, str(local_path)

    # Download the first .htm document
    doc_url = htm_links[0]
    if not doc_url.startswith("http"):
        base = idx_url.rsplit("/", 1)[0]
        doc_url = f"{base}/{doc_url}"

    time.sleep(EDGAR_RATE_LIMIT)
    try:
        doc_resp = requests.get(doc_url, headers=HEADERS, timeout=30)
        local_path = FILINGS_DIR / f"{accession_raw}.html"
        local_path.write_text(doc_resp.text, encoding="utf-8")
        return accession_raw, str(local_path)
    except Exception:
        return accession_raw, ""


def build_filing_index(max_filings: int = None, ticker_filter: list = None) -> str:
    """
    Download EDGAR quarterly indices, filter 8-K filings, download documents,
    and produce a sorted JSONL file.

    Args:
        max_filings: cap total filings (for dev/testing)
        ticker_filter: only keep filings for these tickers (None = all)

    Returns: path to the output JSONL file
    """
    if ticker_filter is None:
        ticker_filter = TICKERS

    ticker_set = set(t.upper() for t in ticker_filter) if ticker_filter else None

    output_path = RAW_DIR / "filing_index.jsonl"
    all_filings = []

    print("=" * 60)
    print("DOWNLOADING EDGAR 8-K FILING INDEX")
    print("=" * 60)

    for year in range(EDGAR_START_YEAR, EDGAR_END_YEAR + 1):
        for qtr in range(1, 5):
            print(f"\n  Fetching {year} Q{qtr}...")
            entries = download_index_file(year, qtr)

            # Filter to our ticker universe
            if ticker_set:
                entries = [e for e in entries if e["ticker"] in ticker_set]

            print(f"    Found {len(entries)} 8-K filings for tracked tickers")
            all_filings.extend(entries)

            if max_filings and len(all_filings) >= max_filings:
                all_filings = all_filings[:max_filings]
                break
        if max_filings and len(all_filings) >= max_filings:
            break

    print(f"\nTotal filings to process: {len(all_filings)}")
    print("Downloading filing documents...")

    records = []
    for entry in tqdm(all_filings, desc="Downloading filings"):
        accession, local_path = download_filing_document(entry["index_path"])
        if not local_path:
            continue

        records.append({
            "accession": accession,
            "cik": entry["cik"],
            "ticker": entry["ticker"],
            "filed_at": entry["date_filed"] + "T16:00:00",
            "items": [],   # will be parsed in step 2
            "raw_path": local_path,
        })

    # Sort by filed_at (timestamp ordering is critical)
    records.sort(key=lambda r: r["filed_at"])

    # Write JSONL
    with open(output_path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    print(f"\n[+] Wrote {len(records)} filings to {output_path}")
    return str(output_path)
