"""
Smoke Ingestor: Exhaustive filing download for a specific company using Submissions API.
"""
import requests
import time
import pandas as pd
from pathlib import Path
from datetime import datetime
from config.settings import EDGAR_USER_AGENT, SMOKE_RAW_DIR
from src.ingestion.cik_ticker_map import resolve_company

def format_cik(cik: int) -> str:
    return str(cik).zfill(10)

def fetch_submissions(cik: int) -> dict:
    """Fetch the master submissions JSON for a CIK."""
    cik_10 = format_cik(cik)
    url = f"https://data.sec.gov/submissions/CIK{cik_10}.json"
    headers = {"User-Agent": EDGAR_USER_AGENT}
    resp = requests.get(url, headers=headers)
    resp.raise_for_status()
    return resp.json()

def get_primary_doc_url(cik: int, accession_no: str) -> str:
    """Find the primary document URL from the index.json."""
    acc_clean = accession_no.replace("-", "")
    url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc_clean}/index.json"
    headers = {"User-Agent": EDGAR_USER_AGENT}
    
    # Rate limit respect
    time.sleep(0.1)
    
    resp = requests.get(url, headers=headers)
    if not resp.ok:
        return None
    
    data = resp.json()
    # Looking for primary document
    directory = data.get("directory", {})
    item_list = directory.get("item", [])
    
    # Heuristic: the primary doc is often the first .htm or .txt after some metadata
    # better yet, index.json sometimes has a specific structure
    # Let's look for files matching the accession format or just the largest HTM
    candidate = None
    for item in item_list:
        name = item.get("name", "")
        if name.endswith((".htm", ".html", ".txt")):
            # Avoid small utility files
            if "index" in name or "header" in name:
                continue
            candidate = name
            break
            
    if candidate:
        return f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc_clean}/{candidate}"
    return None

def download_filing(url: str, save_path: Path) -> bool:
    """Download a single filing document."""
    headers = {"User-Agent": EDGAR_USER_AGENT}
    resp = requests.get(url, headers=headers)
    if resp.ok:
        save_path.write_bytes(resp.content)
        return True
    return False

def run_smoke_ingest(query: str, start_year: int = 2019, end_year: int = 2023, forms: list = None):
    """Execution entry point for smoke ingestion."""
    if forms is None:
        forms = ["8-K", "10-Q", "10-K"]
        
    cik, ticker = resolve_company(query)
    if not cik:
        print(f"Error: Could not resolve company '{query}'")
        return None
    
    print(f"Resolved {query} to CIK {cik} ({ticker})")
    
    # 1. Fetch Submissions
    subs = fetch_submissions(cik)
    filings = subs.get("filings", {}).get("recent", {})
    
    df = pd.DataFrame(filings)
    df["reportDate"] = pd.to_datetime(df["reportDate"])
    
    # 2. Filter by forms and date
    mask = (df["form"].isin(forms)) & (df["reportDate"].dt.year >= start_year) & (df["reportDate"].dt.year <= end_year)
    targets = df[mask].copy()
    
    print(f"Found {len(targets)} filings matching criteria.")
    
    results = []
    
    for _, row in targets.iterrows():
        acc = row["accessionNumber"]
        date = row["reportDate"].strftime("%Y%m%d")
        form = row["form"].replace("/", "-")
        
        save_name = f"{ticker}_{date}_{form}_{acc}.htm"
        save_path = SMOKE_RAW_DIR / save_name
        
        entry = {
            "accession_no": acc,
            "report_date": date,
            "form": row["form"],
            "expected": True,
            "downloaded": False,
            "url": None
        }
        
        if save_path.exists():
            entry["downloaded"] = True
            results.append(entry)
            continue
            
        # Find URL
        url = get_primary_doc_url(cik, acc)
        if url:
            entry["url"] = url
            if download_filing(url, save_path):
                entry["downloaded"] = True
                print(f"Downloaded: {save_name}")
            else:
                print(f"Failed download: {url}")
        else:
            print(f"Could not find primary doc for {acc}")
            
        results.append(entry)
        
    # 3. Report Completeness
    report_df = pd.DataFrame(results)
    report_path = SMOKE_RAW_DIR / f"{ticker}_completeness.csv"
    report_df.to_csv(report_path, index=False)
    
    coverage = (report_df["downloaded"].sum() / len(report_df)) * 100
    print(f"Ingestion Complete. Coverage: {coverage:.1f}%")
    
    return report_path

if __name__ == "__main__":
    import sys
    q = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    run_smoke_ingest(q)
