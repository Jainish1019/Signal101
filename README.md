# Signal in Noise: SEC 8-K Event Detection

An end-to-end NLP pipeline that ingests SEC EDGAR 8-K filings, scores them through
a calibrated ensemble (classifier + semantic drift detector + entity extractor),
and outputs actionable ALERT/ARCHIVE decisions validated against actual stock returns.

Built for the RIQE "Finding Signal in Noise: Applied NLP" hackathon track.

## Quick Start

```bash
# 1. Create virtual environment
python3 -m venv .venv && source .venv/bin/activate

# 2. Install dependencies
python main.py install

# 3. Run the full pipeline (downloads data, processes, trains, evaluates)
python main.py pipeline

# 4. Launch the interactive dashboard
python main.py dashboard
```

Or use the Makefile: `make all`

## Architecture

| Layer | Description |
|-------|-------------|
| Ingestion | SEC EDGAR 8-K filings (2019-2023), chunked by Item header |
| NLP | VADER sentiment, TF-IDF novelty, sentence-transformer embeddings, spaCy NER |
| Models | A) Calibrated LinearSVC classifier, B) Semantic drift detector, C) Deterministic explainer |
| Signal | composite = 0.50*A + 0.35*B + 0.15*C; ALERT if >= 60 |
| Evaluation | Walk-forward, 3 baselines, calibration, cost-weighted utility |
| Dashboard | 5-page Streamlit app with FAISS search and Gemini RAG |

## Free-Only Stack

No paid APIs required. All data comes from SEC EDGAR (free) and Stooq/yfinance (free).
Gemini API key is optional; a deterministic fallback handles LLM features when unavailable.

## Configuration

Copy `.env.example` to `.env` and optionally set:
```
GEMINI_API_KEY=your_key_here  # Optional, for richer AI explanations
```

## Project Structure

```
signal-in-noise/
  main.py                   # Entrypoint: pipeline | dashboard | install
  Makefile                  # make all / make pipeline / make dashboard
  config/                   # Settings and keyword lexicon
  scripts/                  # 7 numbered pipeline steps
  src/ingestion/            # EDGAR client, filing parser, price fetcher
  src/nlp/                  # Preprocessor, sentiment, TF-IDF, embedder, NER
  src/models/               # Classifier, drift detector, explainer
  src/signal/               # Composite scorer and decision engine
  src/evaluation/           # Metrics, baselines, decay analysis
  src/rag/                  # FAISS retriever, prompt builder, Gemini client
  dashboard/                # 5-page Streamlit app
  data/                     # Pipeline checkpoints (Parquet/JSONL)
  proof_pack/               # Antigravity agent artifacts
```
