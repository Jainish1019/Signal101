"""
Sentence-transformer embedding wrapper.
"""
from sentence_transformers import SentenceTransformer
import numpy as np
from config.settings import EMBEDDING_MODEL

_model = None


def _get_model():
    global _model
    if _model is None:
        print(f"  Loading embedding model: {EMBEDDING_MODEL}")
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def embed_text(text: str) -> np.ndarray:
    """Embed a single text string. Returns 384-dim vector."""
    model = _get_model()
    return model.encode(text, show_progress_bar=False)


def embed_batch(texts: list[str], batch_size: int = 64) -> np.ndarray:
    """Embed a batch of texts. Returns (N, 384) array."""
    model = _get_model()
    return model.encode(texts, batch_size=batch_size, show_progress_bar=True)
