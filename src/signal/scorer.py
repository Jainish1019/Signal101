"""
Composite signal scorer and decision engine.
"""
from config.settings import (
    WEIGHT_CLASSIFIER, WEIGHT_DRIFT, WEIGHT_ENTITY,
    ALERT_THRESHOLD,
)


def compute_composite(score_a: float, score_b: float, score_c: float) -> float:
    """
    Composite = 0.50 * score_a + 0.35 * score_b + 0.15 * score_c
    All inputs should be on 0-100 scale.
    """
    return round(
        WEIGHT_CLASSIFIER * score_a +
        WEIGHT_DRIFT * score_b +
        WEIGHT_ENTITY * score_c,
        2
    )


def decide(composite: float) -> str:
    """Apply decision rule: ALERT if >= threshold, else ARCHIVE."""
    return "ALERT" if composite >= ALERT_THRESHOLD else "ARCHIVE"


def predict_direction(vader_compound: float) -> str:
    """Determine predicted direction from sentiment."""
    if vader_compound > 0.05:
        return "BULLISH"
    elif vader_compound < -0.05:
        return "BEARISH"
    return "NEUTRAL"
