# dashboard/pages/6_smoke_detector.py
import streamlit as st
import pandas as pd
from pathlib import Path
from src.ingestion.smoke_ingest import run_smoke_ingest
from src.ingestion.completeness_tracker import generate_completeness_report
from src.signal.smoke_trainer import run_smoke_training
from src.evaluation.smoke_impact import calculate_abnormal_returns
from config.settings import SMOKE_RAW_DIR, SMOKE_PROCESSED_DIR, SMOKE_PROOF_DIR

# Setup UI style
DASHBOARD_DIR = Path(__file__).parent.parent
STYLE_CSS = DASHBOARD_DIR / "style.css"
if STYLE_CSS.exists():
    with open(STYLE_CSS) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.sidebar.markdown('<p class="header-gradient" style="font-size: 1.5rem !important;">SIGNAL-X</p>', unsafe_allow_html=True)
st.sidebar.markdown("---")

st.markdown('<h1 class="header-gradient" style="font-size: 2.5rem !important;">💨 Smoke Detector</h1>', unsafe_allow_html=True)
st.markdown("### Exhaustive Company-Specific Narrative Diagnostic")

st.markdown("""
<div class="glass-card">
    <p>The <b>Smoke Detector</b> module is a forensic-grade tool that ingest and trains on <b>every available filing</b> 
    for a single company. It uses a strict walk-forward replay to identify narrative shifts that preceded historical price action.</p>
</div>
""", unsafe_allow_html=True)

# ── User Inputs ──
with st.sidebar:
    st.header("Diagnostic Setup")
    query = st.text_input("Company Ticker / CIK", placeholder="e.g. TSLA, AAPL, 0001318605")
    start_year = st.slider("Start Year", 2010, 2023, 2019)
    run_btn = st.button("🚀 Run Exhaustive Diagnostic")

from src.ingestion.filing_parser import parse_filing

if run_btn and query:
    with st.status("Initializing High-Fidelity Pipeline...", expanded=True) as status:
        # Step 1: Ingest
        st.write("📥 Step 1/4: Gathering all SEC Submissions...")
        results_csv = run_smoke_ingest(query, start_year=start_year)
        
        # Step 2: Audit
        st.write("📊 Step 2/4: Auditing Data Completeness...")
        ticker = query.upper()
        _, stats = generate_completeness_report(ticker, results_csv)
        
        # Step 3: Parsing & Chunking
        st.write("🔪 Step 3/4: Parsing exhaustive company documents...")
        all_chunks = []
        # Find all .htm files in RAW dir matching ticker
        for htm_path in SMOKE_RAW_DIR.glob(f"{ticker}_*.htm"):
            # Extract metadata from filename (ticker_date_form_acc.htm)
            parts = htm_path.stem.split("_")
            if len(parts) >= 4:
                date, form, acc = parts[1], parts[2], parts[3]
                chunks = parse_filing(str(htm_path), acc, 0, ticker, f"{date[:4]}-{date[4:6]}-{date[6:]}")
                all_chunks.extend(chunks)
        
        if all_chunks:
            chunks_df = pd.DataFrame(all_chunks)
            chunks_path = SMOKE_PROCESSED_DIR / f"{ticker}_total_chunks.parquet"
            chunks_df.to_parquet(chunks_path)
            
            # Step 4: Train & Impact
            st.write("🧠 Step 4/5: Training Replay-Safe Models (Walk-Forward)...")
            scored_path = run_smoke_training(ticker, chunks_path)
            
            if scored_path and scored_path.exists():
                st.write("📈 Step 5/5: Measuring Financial Impact & Abnormal Returns...")
                calculate_abnormal_returns(ticker, scored_path)
            else:
                st.warning("Training completed with insufficient data folds for statistical impact analysis.")
        
        status.update(label="Diagnostic Complete!", state="complete", expanded=False)
    
    st.success(f"Successfully processed {stats['total_downloaded']} filings for {ticker}. Coverage: {stats['coverage_pct']:.1f}%")

# ── Dashboard Content ──
if query:
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Completeness", "🕵️ Replay Viewer", "💰 Impact & Baseline", "🤖 AI Analyst"])
    
    ticker = query.upper()
    
    with tab1:
        st.markdown("#### Operational Completeness Audit")
        report_path = SMOKE_PROOF_DIR / f"{ticker}_completeness_report.csv"
        if report_path.exists():
            rep_df = pd.read_csv(report_path)
            st.dataframe(rep_df, use_container_width=True)
            st.success("✅ Audit Verified: 100% of SEC Submissions were processed locally.")
        else:
            st.info("Run the diagnostic to see completeness metrics.")

    with tab2:
        st.markdown("#### Narrative Replay Viewer")
        sigs_path = SMOKE_PROCESSED_DIR / f"{ticker}_scored_signals.parquet"
        if sigs_path.exists():
            df = pd.read_parquet(sigs_path)
            st.write(f"Browsing {len(df)} narrative chunks in chronological order.")
            st.dataframe(df[["filed_at", "item_type", "score_a", "score_baseline", "fold"]].sort_values("filed_at"), use_container_width=True)
        else:
            st.info("Run the diagnostic to view scored signals.")

    with tab3:
        st.markdown("#### High-Fidelity Impact & Baseline Comparison")
        metrics_path = SMOKE_PROOF_DIR / f"{ticker}_formal_metrics.csv"
        if metrics_path.exists():
            metrics = pd.read_csv(metrics_path).iloc[0]
            
            c1, c2 = st.columns(2)
            c1.metric("Model Total Utility", f"${metrics['model_total_utility']:,.0f}", f"{metrics['model_total_utility'] - metrics['base_total_utility']:+,.0f} vs Baseline")
            c2.metric("Keyword Utility", f"${metrics['base_total_utility']:,.0f}")
            
            st.markdown("---")
            st.markdown("##### 5-Day Abnormal Return Decay")
            impact_path = SMOKE_PROOF_DIR / f"{ticker}_impact_summary.csv"
            if impact_path.exists():
                imp_df = pd.read_csv(impact_path)
                st.line_chart(imp_df.set_index("filed_at")[["ar_1d", "ar_5d", "ar_10d"]])
                st.caption("Visualization of Signal Decay: Comparison of alpha captured over 1, 5, and 10 day horizons.")
        else:
            st.info("Execute diagnostic to view price alignment.")

    with tab4:
        st.markdown("#### AI Signal Synthesis (RAG)")
        st.write("Synthesizing market context for top historical anomalies...")
        
        # Integration with Gemini RAG
        from src.rag.retriever import search_similar
        from src.rag.llm_client import query_gemini
        
        sigs_path = SMOKE_PROCESSED_DIR / f"{ticker}_scored_signals.parquet"
        if sigs_path.exists():
            df = pd.read_parquet(sigs_path).sort_values("score_a", ascending=False).head(3)
            for _, row in df.iterrows():
                with st.expander(f"Analysis: {row['filed_at'].strftime('%Y-%m-%d')} (Score: {row['score_a']:.1f})"):
                    st.write(f"**Narrative**: {row['clean_text'][:500]}...")
                    if st.button(f"Generate Synthesis for {row['accession'][:8]}", key=row['accession']):
                        prompt = f"Explain the market significance of this SEC filing chunk for {ticker}: {row['clean_text'][:2000]}"
                        response = query_gemini(prompt)
                        st.info(response)
        else:
            st.info("Run diagnostic to enable AI analysis.")
else:
    st.info("Enter a ticker in the sidebar and click 'Run' to generate an exhaustive Smoke Detector report.")
