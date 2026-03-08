import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from config.settings import PROCESSED_DIR, PRICES_DIR

st.set_page_config(page_title="Signal Timeline", layout="wide")

@st.cache_data
def load_data():
    sig_path = PROCESSED_DIR / "signals.parquet"
    price_path = PRICES_DIR / "daily_prices.parquet"
    sigs = pd.read_parquet(sig_path) if sig_path.exists() else pd.DataFrame()
    prices = pd.read_parquet(price_path) if price_path.exists() else pd.DataFrame()
    if not sigs.empty:
        sigs["filed_at"] = pd.to_datetime(sigs["filed_at"])
    if not prices.empty:
        prices["date"] = pd.to_datetime(prices["date"])
    return sigs, prices

st.title("📈 Signal vs Price Timeline")

sigs, prices = load_data()

if sigs.empty:
    st.warning("No data available.")
    st.stop()

ticker = st.selectbox("Select Ticker", sorted(sigs["ticker"].unique()))
tk_sigs = sigs[sigs["ticker"] == ticker].sort_values("filed_at")
tk_prices = prices[prices["ticker"] == ticker].sort_values("date") if not prices.empty else pd.DataFrame()

fig = make_subplots(specs=[[{"secondary_y": True}]])

# Price line
if not tk_prices.empty:
    fig.add_trace(go.Scatter(
        x=tk_prices["date"], y=tk_prices["close"],
        name="Price", line=dict(color="rgba(150,150,150,0.5)", width=2),
    ), secondary_y=True)

# Signal scatter
colors = {"ALERT": "red", "ARCHIVE": "green"}
for dec in ["ALERT", "ARCHIVE"]:
    sub = tk_sigs[tk_sigs["decision"] == dec]
    if not sub.empty:
        fig.add_trace(go.Scatter(
            x=sub["filed_at"], y=sub["composite_score"],
            mode="markers", name=dec,
            marker=dict(color=colors[dec], size=10 if dec == "ALERT" else 6),
            text=sub["clean_text"].str[:100],
            hovertemplate="<b>%{text}</b><br>Score: %{y:.1f}<extra></extra>",
        ), secondary_y=False)

fig.update_layout(title=f"Signal Timeline: {ticker}", height=500,
                  plot_bgcolor="rgba(0,0,0,0)", hovermode="closest")
fig.update_yaxes(title_text="Composite Score", range=[0, 105], secondary_y=False)
fig.update_yaxes(title_text="Price ($)", secondary_y=True)

st.plotly_chart(fig, use_container_width=True)

with st.expander("View Raw Data"):
    cols = ["filed_at", "decision", "composite_score", "direction", "score_a", "score_b", "score_c"]
    display_cols = [c for c in cols if c in tk_sigs.columns]
    st.dataframe(tk_sigs[display_cols], use_container_width=True)
