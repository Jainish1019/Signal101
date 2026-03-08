"""
Model A: Calibrated linear classifier for event prediction.
"""
import pickle
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from config.settings import MODELS_DIR


class EventClassifier:
    """
    Calibrated LinearSVC trained on TF-IDF features.
    Outputs p(significant_move) in [0, 1].
    """

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=8000,
            stop_words="english",
            ngram_range=(1, 2),
        )
        base_svc = LinearSVC(C=1.0, max_iter=5000, class_weight="balanced")
        self.model = CalibratedClassifierCV(base_svc, cv=3)
        self._fitted = False

    def fit(self, texts: list[str], labels: np.ndarray) -> None:
        """Train on texts and binary labels."""
        X = self.vectorizer.fit_transform(texts)
        self.model.fit(X, labels)
        self._fitted = True

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        """Return calibrated probability of positive class."""
        if not self._fitted:
            return np.full(len(texts), 0.5)
        X = self.vectorizer.transform(texts)
        return self.model.predict_proba(X)[:, 1]

    def save(self, path=None):
        if path is None:
            path = MODELS_DIR / "model_a.pkl"
        with open(path, "wb") as f:
            pickle.dump({"vectorizer": self.vectorizer, "model": self.model}, f)

    def load(self, path=None):
        if path is None:
            path = MODELS_DIR / "model_a.pkl"
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.vectorizer = data["vectorizer"]
        self.model = data["model"]
        self._fitted = True
