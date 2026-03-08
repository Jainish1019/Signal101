# dashboard/pages/4_similar_events.py
import streamlit as st
import pandas as pd
from pathlib import Path
from config.settings import PROCESSED_DIR
from src.rag.retriever import FilingRetriever

# Setup paths
DASHBOARD_DIR = Path(__file__).parent.parent
STYLE_CSS = DASHBOARD_DIR / "style.css"

if STYLE_CSS.exists():
    with open(STYLE_CSS) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.sidebar.markdown('<h1 class="header-gradient" style="font-size: 1.5rem !important;">SIGNAL-X</h1>', unsafe_allow_html=True)
st.sidebar.markdown("---")

st.markdown('<h1 class="header-gradient" style="font-size: 2.5rem !important;">🔍 Similar Events explorer</h1>', unsafe_allow_html=True)

@st.cache_resource
def get_retriever():
    r = FilingRetriever()
    r.load()
    return r

retriever = get_retriever()

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
        results = retriever.search(query, k=top_k)
        
        for i, res in enumerate(results):
            # Determine badge
            is_alert = res.get("decision") == "ALERT"
            badge_class = "badge-bearish" if is_alert else "badge-neutral"
            
            st.markdown(f"""
<div class="glass-card">
    <div style="display: flex; justify-content: space-between;">
        <span style="font-weight: 700;">{i+1}. {res['ticker']}</span>
        <span class="{badge_class}">{res['decision']}</span>
    </div>
    <div style="color: #90a4ae; margin-bottom: 10px;">Similarity: {res['score']:.3f} | Score: {res['composite_score']:.1f}</div>
    <p style="font-size: 0.9rem; color: #cfd8dc;">{res['text'][:400]}...</p>
</div>
            """, unsafe_allow_html=True)
