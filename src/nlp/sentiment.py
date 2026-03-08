"""
VADER sentiment analysis wrapper.
"""
import nltk
try:
    nltk.data.find("sentiment/vader_lexicon.zip")
except LookupError:
    nltk.download("vader_lexicon", quiet=True)

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_analyzer = SentimentIntensityAnalyzer()


def analyze_sentiment(text: str) -> dict:
    """Return VADER polarity scores for text."""
    scores = _analyzer.polarity_scores(text)
    return {
        "vader_compound": scores["compound"],
        "vader_pos": scores["pos"],
        "vader_neg": scores["neg"],
        "vader_neu": scores["neu"],
    }


def sentence_sentiments(sentences: list[str]) -> list[dict]:
    """Score each sentence individually."""
    return [analyze_sentiment(s) for s in sentences]
