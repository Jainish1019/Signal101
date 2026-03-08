import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import streamlit as st
from src.rag.llm_client import answer_query
from config.settings import GEMINI_API_KEY

st.set_page_config(page_title="Ask Analyst", layout="wide")

st.title("🤖 Ask the Analyst")
st.markdown("Natural language Q&A powered by FAISS retrieval + Google Gemini (with deterministic fallback).")

if not GEMINI_API_KEY:
    st.info("No Gemini API key configured. Using deterministic template responses. "
            "Set GEMINI_API_KEY in .env for richer answers.")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Suggested queries
with st.sidebar:
    st.subheader("Try These")
    examples = [
        "What types of 8-K filings get the highest scores?",
        "Show me the most novel filings from Tesla",
        "What usually happens after executive departure filings?",
        "Which sectors had the most ALERT filings?",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": ex})
            with st.spinner("Searching..."):
                res = answer_query(ex)
            st.session_state.messages.append({
                "role": "assistant", "content": res["answer"],
                "sources": res.get("sources", [])
            })
            st.rerun()

# Render history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander(f"📚 {len(msg['sources'])} sources"):
                for s in msg["sources"]:
                    st.markdown(f"- [{s.get('ticker')}] {s.get('filed_at')} "
                                f"(sim: {s.get('similarity', 0):.3f})")

# Chat input
if prompt := st.chat_input("Ask about the filing data..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving context and generating answer..."):
            res = answer_query(prompt)
        st.markdown(res["answer"])
        if res.get("sources"):
            with st.expander(f"📚 {len(res['sources'])} sources"):
                for s in res["sources"]:
                    st.markdown(f"- [{s.get('ticker')}] {s.get('filed_at')}")

        st.session_state.messages.append({
            "role": "assistant", "content": res["answer"],
            "sources": res.get("sources", [])
        })
