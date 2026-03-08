import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import streamlit as st
import pandas as pd
from config.settings import PROCESSED_DIR

st.set_page_config(page_title="Alert Feed", layout="wide")

@st.cache_data
def load_signals():
    path = PROCESSED_DIR / "signals.parquet"
    if path.exists():
        df = pd.read_parquet(path)
        df["filed_at"] = pd.to_datetime(df["filed_at"])
        return df.sort_values("composite_score", ascending=False)
    return pd.DataFrame()

st.title("🚨 Alert Feed")
st.markdown("SEC 8-K filings flagged by the NLP scoring pipeline, ranked by composite score.")

df = load_signals()

if df.empty:
    st.warning("No signals found. Please run the pipeline first: `python main.py pipeline`")
    st.stop()

# Sidebar filters
st.sidebar.header("Filters")
tickers = st.sidebar.multiselect("Tickers", sorted(df["ticker"].unique()),
                                  default=sorted(df["ticker"].unique())[:10])
min_score = st.sidebar.slider("Minimum Score", 0, 100, 0)
decision_filter = st.sidebar.radio("Decision", ["ALL", "ALERT", "ARCHIVE"])

filtered = df[df["ticker"].isin(tickers) & (df["composite_score"] >= min_score)]
if decision_filter != "ALL":
    filtered = filtered[filtered["decision"] == decision_filter]

st.metric("Visible Filings", len(filtered))
st.markdown("---")

# Render cards
for _, row in filtered.head(50).iterrows():
    score = row["composite_score"]
    color = "#FF4B4B" if score >= 80 else "#FF8C00" if score >= 60 else "#00CC96"
    decision = row["decision"]

    with st.container():
        c1, c2 = st.columns([1, 4])
        with c1:
            st.markdown(f"**{row['ticker']}**")
            st.caption(str(row["filed_at"].date()))
            st.markdown(f'<span style="background:{color};color:white;padding:2px 10px;'
                        f'border-radius:10px;font-weight:bold;font-size:13px;">'
                        f'{decision} ({score:.0f})</span>', unsafe_allow_html=True)
            direction = row.get("direction", "NEUTRAL")
            arrow = "🟢" if direction == "BULLISH" else "🔴" if direction == "BEARISH" else "⚪"
            st.markdown(f"{arrow} {direction}")

        with c2:
            headline = str(row["clean_text"])[:200]
            with st.expander(headline, expanded=(decision == "ALERT")):
                st.write(row["clean_text"][:2000])
                st.markdown("---")
                mc1, mc2, mc3 = st.columns(3)
                mc1.metric("Classifier (A)", f"{row.get('score_a', 0):.1f}")
                mc2.metric("Drift (B)", f"{row.get('score_b', 0):.1f}")
                mc3.metric("Entity (C)", f"{row.get('score_c', 0):.1f}")

                kws = row.get("matched_keywords", "")
                if kws:
                    st.markdown(f"**Keywords:** {kws}")

                quotes = row.get("evidence_quotes", "")
                if quotes:
                    st.markdown("**Evidence Quotes:**")
                    for q in str(quotes).split(" ||| ")[:3]:
                        st.markdown(f'> "{q.strip()}"')
    st.markdown("---")
