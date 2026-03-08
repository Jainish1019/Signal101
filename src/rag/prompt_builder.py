"""
Prompt builder for RAG context construction.
"""


def build_system_prompt() -> str:
    return (
        "You are a financial filing analyst specializing in SEC 8-K filings. "
        "You answer questions about corporate events and their market impact "
        "using ONLY the provided context from actual filings. "
        "Always cite specific filings by ticker and filing date. "
        "If the context does not contain enough information, say so explicitly. "
        "Never speculate beyond the provided data."
    )


def build_context_block(articles: list[dict]) -> str:
    if not articles:
        return "No relevant filings found in the database."

    lines = []
    for i, art in enumerate(articles, 1):
        ts = art.get("filed_at", "Unknown")
        tkr = art.get("ticker", "??")
        score = art.get("composite_score", art.get("signal_score", 0))
        decision = art.get("decision", "ARCHIVE")
        headline = art.get("headline", art.get("clean_text", ""))[:300]

        line = f"Filing {i}:\n"
        line += f"[{ts}] [{tkr}] [Score: {score} ({decision})]\n"
        line += f"{headline}\n"
        lines.append(line)

    return "\n".join(lines)


def build_explain_prompt(article: dict, similar: list[dict]) -> str:
    base = build_context_block([article])
    sim_block = build_context_block(similar) if similar else ""

    question = (
        "Explain why this filing triggered the signal level it received. "
        "Reference its sentiment, keywords, novelty, and compare to historical matches. "
        "Be concise and cite exact quotes where possible."
    )

    prompt = f"Target Filing:\n{base}\n\n"
    if sim_block:
        prompt += f"Similar Historical Events:\n{sim_block}\n\n"
    prompt += f"Question: {question}"
    return prompt
