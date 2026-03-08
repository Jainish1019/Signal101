import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import streamlit as st
import pandas as pd
from config.settings import PROCESSED_DIR
from src.rag.retriever import search_similar

st.set_page_config(page_title="Similar Events", layout="wide")

@st.cache_data
def load_signals():
    path = PROCESSED_DIR / "signals.parquet"
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()

st.title("🔍 Similar Events Explorer")
st.markdown("Enter a filing excerpt or event description to find semantically similar filings.")

signals_df = load_signals()

query = st.text_area("Paste a filing excerpt or describe an event:", height=120,
                     placeholder="e.g., Company announces acquisition of subsidiary for $2.5 billion")

n_results = st.slider("Number of results", 3, 20, 10)

if st.button("Search", type="primary") and query:
    with st.spinner("Embedding query and searching FAISS..."):
        results = search_similar(query, n=n_results, signals_df=signals_df)

    if not results:
        st.warning("No results found. Make sure the FAISS index has been built.")
    else:
        for i, r in enumerate(results, 1):
            score = r.get("composite_score", r.get("signal_score", 0))
            decision = r.get("decision", "ARCHIVE")
            color = "#FF4B4B" if decision == "ALERT" else "#00CC96"

            st.markdown(f"### {i}. [{r.get('ticker', '??')}] {r.get('filed_at', 'Unknown date')}")
            st.markdown(f"Similarity: **{r.get('similarity', 0):.3f}** | "
                        f"Score: **{score}** | "
                        f'<span style="background:{color};color:white;padding:2px 8px;'
                        f'border-radius:8px;">{decision}</span>',
                        unsafe_allow_html=True)
            st.write(r.get("headline", "No text available"))
            st.markdown("---")
