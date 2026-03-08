import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

import streamlit as st

st.set_page_config(page_title="Signal in Noise", page_icon="📊", layout="wide")

st.title("📊 Signal in Noise: SEC 8-K Event Detection")
st.markdown("---")

st.markdown("""
### Autonomous Financial Filing Analysis Pipeline

This system ingests SEC EDGAR 8-K filings, extracts NLP features, scores each filing
through a calibrated ensemble (classifier + drift detector + entity extractor), and
outputs actionable ALERT/ARCHIVE decisions validated against real stock returns.

**Navigate using the sidebar to explore:**

1. **Alert Feed** -- Browse flagged filings ranked by signal strength
2. **Signal Timeline** -- Visualize signals overlaid on price charts
3. **Evaluation** -- Walk-forward metrics, baselines, calibration curves
4. **Similar Events** -- FAISS-powered semantic search across filings
5. **Ask Analyst** -- Natural language Q&A powered by Gemini RAG

**Architecture highlights:**
- 100% free stack (no paid APIs required)
- Walk-forward evaluation with no data leakage
- Deterministic fallbacks for every LLM-dependent feature
- Cost-weighted utility measurement ($100 TP / -$150 FP)
""")

st.info("👈 Select a page from the sidebar to begin.")
