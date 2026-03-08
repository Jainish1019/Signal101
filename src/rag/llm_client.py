"""
LLM client: Google Gemini (free tier) with deterministic fallback.
"""
import os
from src.rag.retriever import search_similar, load_faiss_index
from src.rag.prompt_builder import build_system_prompt, build_context_block, build_explain_prompt
from src.models.explainer import build_explanation
from config.settings import GEMINI_API_KEY, GEMINI_MODEL, PROCESSED_DIR

# Try to import Gemini
_gemini_available = False
try:
    import google.generativeai as genai
    if GEMINI_API_KEY:
        genai.configure(api_key=GEMINI_API_KEY)
        _model = genai.GenerativeModel(
            model_name=GEMINI_MODEL,
            system_instruction=build_system_prompt(),
        )
        _gemini_available = True
except Exception:
    pass


def _load_signals():
    """Load signals dataframe for context enrichment."""
    import pandas as pd
    path = PROCESSED_DIR / "signals.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame()


def query_gemini(user_message: str, context: str = "General market analysis") -> str:
    """Call Gemini API with context."""
    if not _gemini_available:
        return _deterministic_fallback(user_message, context)

    try:
        prompt = f"Context:\n{context}\n\nUser Question:\n{user_message}"
        response = _model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Gemini API error: {e}. Falling back to deterministic response.\n\n{_deterministic_fallback(user_message, context)}"


def _deterministic_fallback(user_message: str, context: str) -> str:
    """Template-based fallback when Gemini is unavailable."""
    return (
        f"[Deterministic Analysis - Gemini unavailable]\n\n"
        f"Query: {user_message}\n\n"
        f"Based on the retrieved context:\n{context}\n\n"
        f"Note: For richer natural language explanations, configure a valid "
        f"GEMINI_API_KEY in your .env file."
    )


def answer_query(query: str, ticker: str = None) -> dict:
    """Full RAG pipeline: search FAISS, build context, generate response."""
    signals_df = _load_signals()
    context_articles = search_similar(query, n=5, signals_df=signals_df)
    context_block = build_context_block(context_articles)
    answer = query_gemini(query, context_block)

    return {
        "answer": answer,
        "sources": context_articles,
        "num_sources": len(context_articles),
    }


def explain_signal(article: dict) -> dict:
    """Explain a specific signal using RAG + Gemini or deterministic fallback."""
    signals_df = _load_signals()

    # Get similar events
    text = article.get("clean_text", "")
    similar = search_similar(text, n=3, signals_df=signals_df) if text else []

    if _gemini_available:
        prompt = build_explain_prompt(article, similar)
        answer = query_gemini(prompt, "Context injected into prompt.")
    else:
        answer = build_explanation(article)

    return {
        "answer": answer,
        "sources": similar,
        "num_sources": len(similar),
    }
