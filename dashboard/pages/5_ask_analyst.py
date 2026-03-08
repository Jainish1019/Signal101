# dashboard/pages/5_ask_analyst.py
import streamlit as st
from pathlib import Path
from src.rag.llm_client import answer_query, explain_signal

# Setup paths
DASHBOARD_DIR = Path(__file__).parent.parent
STYLE_CSS = DASHBOARD_DIR / "style.css"

if STYLE_CSS.exists():
    with open(STYLE_CSS) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.sidebar.markdown('<h1 class="header-gradient" style="font-size: 1.5rem !important;">SIGNAL-X</h1>', unsafe_allow_html=True)
st.sidebar.markdown("---")

st.markdown('<h1 class="header-gradient" style="font-size: 2.5rem !important;">🤖 Ask the Analyst</h1>', unsafe_allow_html=True)

st.markdown("""
<div class="glass-card">
    <p>Natural Language Q&A powered by <b>Gemini Pro RAG</b>. 
    Ask complex cross-filing questions to synthesize narrative signals.</p>
</div>
""", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Quick Queries
cols = st.columns(3)
if cols[0].button("Compare CEO resignations"):
    prompt = "Which companies had CEO resignations and what was the general sentiment compared to MSFT?"
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.rerun()

if cols[1].button("Find M&A activity"):
    prompt = "Show me any filings mentioning acquisitions or mergers in 2020."
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.rerun()

if prompt := st.chat_input("Ask Signal-X Analyst..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Synthesizing market signals..."):
            response, sources = answer_query(prompt)
            st.markdown(response)
            if sources:
                with st.expander("System Sources"):
                    for s in sources:
                        st.write(f"- {s['ticker']} ({s['date']}): {s['text'][:200]}...")
        
    st.session_state.messages.append({"role": "assistant", "content": response})
