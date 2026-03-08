# dashboard/pages/2_signal_timeline.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
from config.settings import PROCESSED_DIR, PRICES_DIR

# Setup paths
DASHBOARD_DIR = Path(__file__).parent.parent
STYLE_CSS = DASHBOARD_DIR / "style.css"

if STYLE_CSS.exists():
    with open(STYLE_CSS) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.sidebar.markdown('<h1 class="header-gradient" style="font-size: 1.5rem !important;">SIGNAL-X</h1>', unsafe_allow_html=True)
st.sidebar.markdown("---")

st.markdown('<h1 class="header-gradient" style="font-size: 2.5rem !important;">📈 Signal Timeline</h1>', unsafe_allow_html=True)

@st.cache_data
def load_data():
    sigs = pd.read_parquet(PROCESSED_DIR / "signals.parquet")
    prices = pd.read_parquet(PRICES_DIR / "daily_prices.parquet")
    return sigs, prices

sigs_df, prices_df = load_data()

tickers = sorted(sigs_df["ticker"].unique())
selected_ticker = st.sidebar.selectbox("Select Ticker", tickers)

t_sigs = sigs_df[sigs_df["ticker"] == selected_ticker].sort_values("filed_at")
t_prices = prices_df[prices_df["ticker"] == selected_ticker].sort_values("date")

st.markdown(f"""
<div class="glass-card">
    <h4>{selected_ticker} Performance vs NLP Signal</h4>
    <p style="color: #90a4ae;">Overlaying autonomous anomaly scores on historical price action.</p>
</div>
""", unsafe_allow_html=True)

# Create Plotly Chart
fig = go.Figure()

# Price Line
fig.add_trace(go.Scatter(
    x=t_prices["date"], y=t_prices["close"],
    mode='lines', name='Price',
    line=dict(color='#00ff88', width=3),
    yaxis='y2'
))

# Signal Dots
alerts = t_sigs[t_sigs["decision"] == "ALERT"]
archives = t_sigs[t_sigs["decision"] == "ARCHIVE"]

fig.add_trace(go.Scatter(
    x=alerts["filed_at"], y=alerts["composite_score"],
    mode='markers', name='ALERT',
    marker=dict(size=14, color='#00d2ff', symbol='diamond', line=dict(width=2, color='white')),
    yaxis='y1'
))

fig.add_trace(go.Scatter(
    x=archives["filed_at"], y=archives["composite_score"],
    mode='markers', name='ARCHIVE',
    marker=dict(size=8, color='rgba(255,255,255,0.1)', symbol='circle', line=dict(width=1, color='#8892b0')),
    yaxis='y1'
))

fig.update_layout(
    template="plotly_dark",
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    margin=dict(l=20, r=20, t=20, b=20),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    xaxis=dict(showgrid=False),
    yaxis=dict(title="Composite Score", side="left", range=[0, 105], showgrid=True, gridcolor='rgba(255,255,255,0.05)'),
    yaxis2=dict(title="Stock Price ($)", side="right", overlaying='y', showgrid=False)
)

st.plotly_chart(fig, use_container_width=True)

with st.expander("View Underlying Data"):
    st.dataframe(t_sigs[["filed_at", "item_type", "composite_score", "decision", "vader_compound", "matched_keywords"]])
