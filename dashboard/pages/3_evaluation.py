# dashboard/pages/3_evaluation.py
import streamlit as st
import pandas as pd
import json
from pathlib import Path
from config.settings import EVALUATION_DIR

# Setup paths
DASHBOARD_DIR = Path(__file__).parent.parent
STYLE_CSS = DASHBOARD_DIR / "style.css"

if STYLE_CSS.exists():
    with open(STYLE_CSS) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.sidebar.markdown('<h1 class="header-gradient" style="font-size: 1.5rem !important;">SIGNAL-X</h1>', unsafe_allow_html=True)
st.sidebar.markdown("---")

st.markdown('<h1 class="header-gradient" style="font-size: 2.5rem !important;">📊 Evaluation Dashboard</h1>', unsafe_allow_html=True)

@st.cache_data
def load_eval():
    metrics_path = EVALUATION_DIR / "eval_metrics.json"
    if metrics_path.exists():
        with open(metrics_path, "r") as f:
            return json.load(f)
    return {}

metrics = load_eval()

if not metrics:
    st.warning("No evaluation metrics found. Please run the pipeline first.")
else:
    adv = metrics.get("advanced", {})
    
    # Hero Metrics
    st.markdown("""
<div class="glass-card">
    <div style="display: flex; justify-content: space-around; text-align: center;">
        <div>
            <div class="metric-label">PRECISION</div>
            <div class="metric-value" style="color: #64ffda;">{pre:.3f}</div>
        </div>
        <div>
            <div class="metric-label">RECALL</div>
            <div class="metric-value" style="color: #448aff;">{rec:.3f}</div>
        </div>
        <div>
            <div class="metric-label">F1 SCORE</div>
            <div class="metric-value" style="color: #f48fb1;">{f1:.3f}</div>
        </div>
        <div>
            <div class="metric-label">UTILITY</div>
            <div class="metric-value" style="color: #ffd54f;">${util:,.0f}</div>
        </div>
    </div>
</div>
    """.format(
        pre=adv.get("precision", 0),
        rec=adv.get("recall", 0),
        f1=adv.get("f1", 0),
        util=adv.get("utility", 0)
    ), unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Confusion Matrix")
        # Load confusion matrix from file if we had one as image, but we render as table/plotly for now
        cm = adv.get("confusion_matrix", [[0,0],[0,0]])
        cm_df = pd.DataFrame(cm, index=["Actual Noise", "Actual Event"], columns=["Pred ARCHIVE", "Pred ALERT"])
        st.table(cm_df)

    with col2:
        st.markdown("### Baseline vs Advanced")
        kw = metrics.get("keyword_baseline", {})
        baseline_df = pd.DataFrame({
            "Metric": ["Precision", "Recall", "F1", "Utility"],
            "Advanced": [adv.get("precision"), adv.get("recall"), adv.get("f1"), adv.get("utility")],
            "Keyword": [kw.get("precision"), kw.get("recall"), kw.get("f1"), kw.get("utility")]
        })
        st.dataframe(baseline_df)

    st.markdown("---")
    st.markdown("### Signal Decay Analysis")
    decay_path = EVALUATION_DIR / "decay_analysis.csv"
    if decay_path.exists():
        decay_df = pd.read_parquet(decay_path) if str(decay_path).endswith(".parquet") else pd.read_csv(decay_path)
        st.line_chart(decay_df.set_index("horizon")["correlation"])
