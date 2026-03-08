"""
TF-IDF engine with rolling window support to prevent data leakage.
"""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from config.settings import TFIDF_MAX_FEATURES


class RollingTfidf:
    """
    Maintains a rolling corpus for TF-IDF fitting.
    Only fits on documents within the lookback window (by index).
    """

    def __init__(self, max_features: int = TFIDF_MAX_FEATURES):
        self.max_features = max_features
        self._corpus: list[str] = []
        self._vectorizer = None
        self._matrix = None

    def add_document(self, text: str) -> None:
        """Add a document to the rolling corpus."""
        self._corpus.append(text)

    def fit_window(self, window_size: int = 500) -> None:
        """Fit TF-IDF on the most recent window_size documents."""
        corpus = self._corpus[-window_size:]
        if len(corpus) < 2:
            self._vectorizer = None
            self._matrix = None
            return

        self._vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            stop_words="english",
            ngram_range=(1, 2),
        )
        self._matrix = self._vectorizer.fit_transform(corpus)

    def transform(self, text: str) -> np.ndarray:
        """Transform a single document using the fitted vectorizer."""
        if self._vectorizer is None:
            return np.zeros(self.max_features)
        return self._vectorizer.transform([text]).toarray()[0]

    def novelty_score(self, text: str) -> float:
        """
        Compute novelty as 1 - max cosine similarity to the fitted window.
        High novelty = dissimilar to recent documents = potentially newsworthy.
        """
        if self._vectorizer is None or self._matrix is None:
            return 1.0  # No history → maximally novel

        vec = self._vectorizer.transform([text])
        sims = cosine_similarity(vec, self._matrix)[0]
        max_sim = float(np.max(sims))
        return round(1.0 - max_sim, 4)
