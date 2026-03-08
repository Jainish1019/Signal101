"""
Central configuration for the EDGAR 8-K Signal-in-Noise pipeline.
All paths, thresholds, and API settings live here.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Paths ───────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR     = PROJECT_ROOT / "data"
RAW_DIR      = DATA_DIR / "raw"
FILINGS_DIR  = RAW_DIR / "filings"
PROCESSED_DIR = DATA_DIR / "processed"
PRICES_DIR   = DATA_DIR / "prices"
VECTORDB_DIR = DATA_DIR / "vectordb"
MODELS_DIR   = DATA_DIR / "models"
EVALUATION_DIR = DATA_DIR / "evaluation"
PROOF_PACK_DIR = PROJECT_ROOT / "proof_pack"
SMOKE_DIR = DATA_DIR / "smoke"
SMOKE_RAW_DIR = SMOKE_DIR / "raw"
SMOKE_PROCESSED_DIR = SMOKE_DIR / "processed"
SMOKE_MODELS_DIR = SMOKE_DIR / "models"
SMOKE_PROOF_DIR = SMOKE_DIR / "proof_pack"

# Ensure dirs exist
for d in [DATA_DIR, RAW_DIR, FILINGS_DIR, PROCESSED_DIR, PRICES_DIR, VECTORDB_DIR, MODELS_DIR, EVALUATION_DIR, PROOF_PACK_DIR,
          SMOKE_DIR, SMOKE_RAW_DIR, SMOKE_PROCESSED_DIR, SMOKE_MODELS_DIR, SMOKE_PROOF_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── EDGAR settings ──────────────────────────────────────────────────
EDGAR_BASE      = "https://www.sec.gov/Archives/edgar"
EDGAR_FULL_IDX  = EDGAR_BASE + "/full-index"
EDGAR_SEARCH    = "https://efts.sec.gov/LATEST/search-index"
EDGAR_USER_AGENT = "SignalInNoise research@example.com"  # SEC fair-access
EDGAR_RATE_LIMIT = 0.12   # seconds between requests (≈8 req/s, under 10)
EDGAR_START_YEAR = 2019
EDGAR_END_YEAR   = 2023

# ── Ticker universe (top liquid names for price impact) ─────────────
TICKERS = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "NVDA", "JPM",
    "JNJ", "V", "PG", "UNH", "HD", "BAC", "MA", "DIS", "ADBE",
    "CRM", "NFLX", "PFE", "KO", "PEP", "MRK", "TMO", "CSCO",
    "ABT", "AVGO", "ACN", "COST", "WMT", "XOM", "CVX", "LLY",
]

# CIK-ticker mapping URL
CIK_TICKER_URL = "https://www.sec.gov/files/company_tickers.json"

# ── Price settings ──────────────────────────────────────────────────
STOOQ_URL_TPL   = "https://stooq.com/q/d/l/?s={ticker}.us&i=d"
PRICE_FETCH_SLEEP = 0.5   # seconds between Stooq requests
BENCHMARK_TICKER  = "SPY"

# ── NLP settings ────────────────────────────────────────────────────
EMBEDDING_MODEL  = "all-MiniLM-L6-v2"
EMBEDDING_DIM    = 384
SPACY_MODEL      = "en_core_web_sm"
TFIDF_MAX_FEATURES = 10000
ROLLING_WINDOW_DAYS = 90   # for centroid / TF-IDF fitting

# ── Signal thresholds ──────────────────────────────────────────────
WEIGHT_CLASSIFIER = 0.50
WEIGHT_DRIFT      = 0.35
WEIGHT_ENTITY     = 0.15
ALERT_THRESHOLD   = 60      # composite >= this → ALERT
SIGNIFICANT_MOVE  = 0.02    # 2% abnormal return = "event"

# ── Evaluation ──────────────────────────────────────────────────────
TP_REWARD  =  100   # dollars
FP_PENALTY = -150   # dollars
WALK_FORWARD_MIN_MONTHS = 6

# ── LLM (optional) ─────────────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL   = "gemini-1.5-flash"
