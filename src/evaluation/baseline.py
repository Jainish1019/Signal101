"""
Baseline comparisons: keyword-only, sentiment-only, and random.
"""
import numpy as np
from config.keywords import KEYWORD_LEXICON


def keyword_baseline(texts: list[str]) -> np.ndarray:
    """ALERT if any keyword from the lexicon appears, else ARCHIVE."""
    all_keywords = []
    for cat in KEYWORD_LEXICON.values():
        all_keywords.extend(cat["keywords"])

    predictions = []
    for text in texts:
        text_lower = text.lower()
        hit = any(kw in text_lower for kw in all_keywords)
        predictions.append(1 if hit else 0)

    return np.array(predictions)


def sentiment_baseline(vader_compounds: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """ALERT if abs(VADER compound) > threshold."""
    return (np.abs(vader_compounds) > threshold).astype(int)


def random_baseline(n: int, base_rate: float = 0.1) -> np.ndarray:
    """Random predictions with given base rate."""
    return (np.random.random(n) < base_rate).astype(int)
