# dashboard/pages/1_alert_feed.py
import streamlit as st
import pandas as pd
from pathlib import Path
from config.settings import PROCESSED_DIR

# Setup paths
DASHBOARD_DIR = Path(__file__).parent.parent
STYLE_CSS = DASHBOARD_DIR / "style.css"

# Inject Custom CSS
if STYLE_CSS.exists():
    with open(STYLE_CSS) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.sidebar.markdown('<h1 class="header-gradient" style="font-size: 1.5rem !important;">SIGNAL-X</h1>', unsafe_allow_html=True)
st.sidebar.markdown("---")

st.markdown('<h1 class="header-gradient" style="font-size: 2.5rem !important;">⚡ Alert Feed</h1>', unsafe_allow_html=True)

@st.cache_data
def load_signals():
    path = PROCESSED_DIR / "signals.parquet"
    if path.exists():
        df = pd.read_parquet(path)
        df["filed_at"] = pd.to_datetime(df["filed_at"])
        return df
    return pd.DataFrame()

df = load_signals()

if df.empty:
    st.warning("No signals found. Please run the pipeline first.")
else:
    # Sidebar Filters
    st.sidebar.header("Filters")
    tickers = st.sidebar.multiselect("Tickers", options=sorted(df["ticker"].unique()), default=df["ticker"].unique()[:5])
    min_score = st.sidebar.slider("Minimum Composite Score", 0, 100, 40)
    decision_filter = st.sidebar.radio("Decision", ["ALL", "ALERT", "ARCHIVE"])

    # Apply filters
    filt_df = df[df["ticker"].isin(tickers)]
    filt_df = filt_df[filt_df["composite_score"] >= min_score]
    if decision_filter != "ALL":
        filt_df = filt_df[filt_df["decision"] == decision_filter]

    st.write(f"Showing {len(filt_df)} events")

    for _, row in filt_df.sort_values("filed_at", ascending=False).iterrows():
        # Determine direction badge
        dir_class = "badge-neutral"
        dir_label = "NEUTRAL"
        if row.get("vader_compound", 0) > 0.1:
            dir_class = "badge-bullish"
            dir_label = "BULLISH"
        elif row.get("vader_compound", 0) < -0.1:
            dir_class = "badge-bearish"
            dir_label = "BEARISH"

        # Signal Header
        title_html = f"""
        <div class="glass-card">
            <div style="display: flex; justify-content: space-between; align-items: start;">
                <div>
                    <span style="font-size: 1.5rem; font-weight: 700; color: #fff;">{row['ticker']}</span>
                    <span style="margin-left: 10px; color: #90a4ae;">{row['filed_at'].strftime('%Y-%m-%d %H:%M')}</span>
                    <div style="margin-top: 5px;">
                        <span class="badge-neutral" style="background-color: #311b92;">{row['item_type']}</span>
                        <span class="{dir_class}" style="margin-left: 5px;">{dir_label}</span>
                    </div>
                </div>
                <div style="text-align: right;">
                    <div style="font-size: 0.8rem; color: #90a4ae;">COMPOSITE SCORE</div>
                    <div style="font-size: 2rem; font-weight: 800; color: #ff00cc;">{row['composite_score']:.0f}</div>
                </div>
            </div>
            <hr style="border-top: 1px solid rgba(255,255,255,0.1); margin: 15px 0;">
            <div style="display: flex; justify-content: space-around; text-align: center; margin-bottom: 20px;">
                <div>
                    <div class="metric-label">Classifier (A)</div>
                    <div class="metric-value" style="color: #64ffda;">{row.get('score_a', 0):.0f}</div>
                </div>
                <div>
                    <div class="metric-label">Drift (B)</div>
                    <div class="metric-value" style="color: #448aff;">{row.get('score_b', 0):.0f}</div>
                </div>
                <div>
                    <div class="metric-label">Entity (C)</div>
                    <div class="metric-value" style="color: #f48fb1;">{row.get('score_c', 0):.0f}</div>
                </div>
            </div>
            <p style="color: #cfd8dc; font-style: italic; font-size: 0.95rem;">"... {row['clean_text'][:300]} ..."</p>
            <div style="margin-top: 10px;">
                <code style="background: rgba(0,0,0,0.3); padding: 5px 10px; border-radius: 5px; color: #ffd54f;">Keywords: {row['matched_keywords']}</code>
            </div>
        </div>
        """
        st.markdown(title_html, unsafe_allow_html=True)
