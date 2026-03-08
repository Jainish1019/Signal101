"""
Model B: Drift detector using rolling semantic centroids.
"""
import numpy as np
import pickle
from collections import defaultdict
from config.settings import MODELS_DIR, ROLLING_WINDOW_DAYS


class DriftDetector:
    """
    Tracks a rolling centroid per ticker using 384-dim embeddings.
    Drift = cosine distance to centroid, scaled to 0-100.
    """

    def __init__(self, window_days: int = ROLLING_WINDOW_DAYS):
        self.window_days = window_days
        self._centroids: dict[str, np.ndarray] = {}
        self._histories: dict[str, list[np.ndarray]] = defaultdict(list)

    def update(self, ticker: str, embedding: np.ndarray) -> None:
        """Add an embedding to the ticker's history and recompute centroid."""
        self._histories[ticker].append(embedding)
        # Keep only last N entries (proxy for window_days)
        max_entries = max(50, self.window_days)
        self._histories[ticker] = self._histories[ticker][-max_entries:]
        # Recompute centroid
        self._centroids[ticker] = np.mean(self._histories[ticker], axis=0)

    def drift_score(self, ticker: str, embedding: np.ndarray) -> float:
        """
        Compute drift as cosine distance to centroid, scaled 0-100.
        Higher = more unusual = bigger potential signal.
        """
        if ticker not in self._centroids:
            return 50.0  # No history, return neutral

        centroid = self._centroids[ticker]

        # Cosine distance
        dot = np.dot(embedding, centroid)
        norm_a = np.linalg.norm(embedding)
        norm_b = np.linalg.norm(centroid)

        if norm_a == 0 or norm_b == 0:
            return 50.0

        cos_sim = dot / (norm_a * norm_b)
        cos_dist = 1.0 - cos_sim  # 0 = identical, 2 = opposite

        # Scale to 0-100 (typical drift range is 0 to 0.5)
        score = min(100.0, cos_dist * 200)
        return round(score, 2)

    def save(self, path=None):
        if path is None:
            path = MODELS_DIR / "model_b_centroids.pkl"
        with open(path, "wb") as f:
            pickle.dump({"centroids": self._centroids, "histories": dict(self._histories)}, f)

    def load(self, path=None):
        if path is None:
            path = MODELS_DIR / "model_b_centroids.pkl"
        with open(path, "rb") as f:
            data = pickle.load(f)
        self._centroids = data["centroids"]
        self._histories = defaultdict(list, data["histories"])
