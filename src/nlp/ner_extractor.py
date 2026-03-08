"""
Named Entity Recognition + keyword pattern extraction.
"""
import re
from config.keywords import KEYWORD_LEXICON, URGENCY_MARKERS, MATERIAL_ITEMS
from src.nlp.preprocessor import extract_entities


def match_keywords(text: str) -> dict:
    """
    Match text against the financial keyword lexicon.
    Returns matched keywords, categories, weighted score, and urgency flag.
    """
    text_lower = text.lower()
    matched = []
    categories = set()
    total_weight = 0.0

    for category, data in KEYWORD_LEXICON.items():
        for kw in data["keywords"]:
            if kw in text_lower:
                matched.append(kw)
                categories.add(category)
                total_weight += data["weight"]

    # Urgency detection
    has_urgency = any(u in text_lower for u in URGENCY_MARKERS)

    # Normalize score to 0-1 range (cap at 5 hits)
    keyword_score = min(1.0, total_weight / 5.0)

    return {
        "matched_keywords": matched,
        "keyword_categories": list(categories),
        "keyword_score": round(keyword_score, 4),
        "has_urgency": has_urgency,
    }


def check_material_items(item_type: str) -> bool:
    """Check if the Item code is typically material."""
    return item_type in MATERIAL_ITEMS


def extract_evidence_quotes(text: str, sentences: list[str], top_n: int = 3) -> list[str]:
    """
    Select the most informative sentences as evidence quotes.
    Uses absolute VADER compound score as a proxy for informativeness.
    """
    from src.nlp.sentiment import analyze_sentiment

    scored = []
    for sent in sentences:
        if len(sent.split()) < 5:
            continue
        s = analyze_sentiment(sent)
        scored.append((abs(s["vader_compound"]), sent))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [s[1] for s in scored[:top_n]]


def get_entity_richness(entities: dict) -> int:
    """Count total distinct named entities across all types."""
    total = 0
    for ent_list in entities.values():
        total += len(ent_list)
    return total
