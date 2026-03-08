import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import json
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from config.settings import EVALUATION_DIR

st.set_page_config(page_title="Evaluation", layout="wide")

st.title("📊 Evaluation Dashboard")

# Load metrics
metrics_path = EVALUATION_DIR / "eval_metrics.json"
if not metrics_path.exists():
    st.warning("No evaluation metrics. Run the pipeline first.")
    st.stop()

with open(metrics_path) as f:
    all_metrics = json.load(f)

adv = all_metrics.get("advanced", {})

# Top metrics
c1, c2, c3, c4 = st.columns(4)
c1.metric("Precision", f"{adv.get('precision', 0):.3f}")
c2.metric("Recall", f"{adv.get('recall', 0):.3f}")
c3.metric("F1 Score", f"{adv.get('f1', 0):.3f}")
c4.metric("Utility", f"${adv.get('utility', 0):,.0f}")

st.markdown("---")
col1, col2 = st.columns(2)

# Confusion matrix
with col1:
    st.subheader("Confusion Matrix")
    cm = pd.DataFrame(
        [[adv.get("tp", 0), adv.get("fn", 0)],
         [adv.get("fp", 0), adv.get("tn", 0)]],
        index=["Actual Event", "Actual Noise"],
        columns=["Pred ALERT", "Pred ARCHIVE"]
    )
    fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues")
    st.plotly_chart(fig, use_container_width=True)

# Baseline comparison
with col2:
    st.subheader("Baseline Comparison")
    comp_path = EVALUATION_DIR / "baseline_comparison.csv"
    if comp_path.exists():
        comp = pd.read_csv(comp_path)
        fig = px.bar(comp, x="model", y=["precision", "recall", "f1"],
                     barmode="group", title="Model Comparison")
        st.plotly_chart(fig, use_container_width=True)

# Calibration curve
st.markdown("---")
cal_path = EVALUATION_DIR / "calibration.csv"
if cal_path.exists():
    st.subheader("Calibration Curve")
    cal = pd.read_csv(cal_path)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=cal["predicted"], y=cal["observed"],
                             mode="markers+lines", name="Model"))
    fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines",
                             name="Perfect", line=dict(dash="dash")))
    fig.update_layout(xaxis_title="Predicted Probability",
                      yaxis_title="Observed Frequency", height=400)
    st.plotly_chart(fig, use_container_width=True)

# Decay
decay_path = EVALUATION_DIR / "decay_analysis.csv"
if decay_path.exists():
    st.subheader("Signal Decay")
    decay = pd.read_csv(decay_path)
    fig = px.bar(decay, x="horizon", y="correlation", title="Score-Return Correlation by Horizon")
    st.plotly_chart(fig, use_container_width=True)
