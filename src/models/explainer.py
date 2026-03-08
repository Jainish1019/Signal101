"""
Model C: Deterministic explainer using NER + evidence quotes.
No LLM required. Extracts structured explanation from source text.
"""


def build_explanation(row: dict) -> str:
    """
    Build a deterministic explanation for why a filing was flagged.
    Uses extracted entities, keywords, and evidence quotes.
    """
    parts = []

    # Decision summary
    decision = row.get("decision", "ARCHIVE")
    score = row.get("composite_score", 0)
    parts.append(f"Decision: {decision} (composite score: {score:.1f}/100)")

    # Direction
    direction = row.get("direction", "NEUTRAL")
    parts.append(f"Predicted direction: {direction}")

    # Score breakdown
    parts.append(f"Score breakdown:")
    parts.append(f"  - Classifier confidence (Model A): {row.get('score_a', 0):.1f}")
    parts.append(f"  - Semantic drift (Model B): {row.get('score_b', 0):.1f}")
    parts.append(f"  - Entity richness (Model C): {row.get('score_c', 0):.1f}")

    # Keywords
    kws = row.get("matched_keywords", "")
    if kws:
        parts.append(f"Trigger keywords: {kws}")

    # Entities
    orgs = row.get("entities_org", "")
    persons = row.get("entities_person", "")
    money = row.get("entities_money", "")
    if orgs:
        parts.append(f"Organizations mentioned: {orgs}")
    if persons:
        parts.append(f"People mentioned: {persons}")
    if money:
        parts.append(f"Monetary amounts: {money}")

    # Evidence quotes
    quotes = row.get("evidence_quotes", "")
    if quotes:
        quote_list = quotes.split(" ||| ")
        parts.append("Key evidence quotes:")
        for i, q in enumerate(quote_list[:3], 1):
            parts.append(f'  {i}. "{q.strip()}"')

    return "\n".join(parts)
