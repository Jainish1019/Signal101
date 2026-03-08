# dashboard/pages/4_similar_events.py
import streamlit as st
import pandas as pd
from pathlib import Path
from config.settings import PROCESSED_DIR
from src.rag.retriever import search_similar

# Setup paths
DASHBOARD_DIR = Path(__file__).parent.parent
STYLE_CSS = DASHBOARD_DIR / "style.css"

if STYLE_CSS.exists():
    with open(STYLE_CSS) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.sidebar.markdown('<h1 class="header-gradient" style="font-size: 1.5rem !important;">SIGNAL-X</h1>', unsafe_allow_html=True)
st.sidebar.markdown("---")

st.markdown('<h1 class="header-gradient" style="font-size: 2.5rem !important;">🔍 Similar Events explorer</h1>', unsafe_allow_html=True)

@st.cache_data
def load_signals():
    path = PROCESSED_DIR / "signals.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame()

signals_df = load_signals()

st.markdown("""
<div class="glass-card">
    <p>Describe a corporate event or paste a filing snippet to find semantically similar historical filings 
    across the Signal-X knowledge base.</p>
</div>
""", unsafe_allow_html=True)

query = st.text_area("Event Description / Snippet", placeholder="e.g. CEO resignation due to health reasons...", height=100)
top_k = st.slider("Results", 1, 20, 5)

if st.button("Search Intelligence DB", key="search"):
    if not query:
        st.warning("Please enter a query.")
    else:
        with st.spinner("Searching vector index..."):
            results = search_similar(query, n=top_k, signals_df=signals_df)
        
        if not results:
            st.info("No similar events found in the index.")
        
        for i, res in enumerate(results):
            # Determine badge
            decision = res.get("decision", "ARCHIVE")
            badge_class = "badge-bearish" if decision == "ALERT" else "badge-neutral"
            
            st.markdown(f"""
<div class="glass-card">
    <div style="display: flex; justify-content: space-between;">
        <span style="font-weight: 700;">{i+1}. {res['ticker']}</span>
        <span class="{badge_class}">{decision}</span>
    </div>
    <div style="color: #90a4ae; margin-bottom: 10px;">Similarity: {res['similarity']:.3f} | Score: {res.get('composite_score', 0):.1f} | Date: {res.get('filed_at', 'N/A')}</div>
    <p style="font-size: 0.9rem; color: #cfd8dc;">{res.get('headline', 'No content summary available...')[:400]}</p>
</div>
            """, unsafe_allow_html=True)
