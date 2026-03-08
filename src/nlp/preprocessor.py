"""
Text preprocessor: clean, tokenize, and extract sentences from filing text.
"""
import re
import nltk
import spacy

# Ensure required data is available
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords", quiet=True)

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

_STOP_WORDS = set(stopwords.words("english"))

# Lazy-load spaCy
_nlp = None

def _get_nlp():
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer"])
    return _nlp


def clean_text(text: str) -> str:
    """Remove URLs, special chars, excess whitespace."""
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"[^\w\s.,;:!?'\"-]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def get_sentences(text: str) -> list[str]:
    """Split text into sentences."""
    return sent_tokenize(text)


def get_tokens(text: str, remove_stopwords: bool = True) -> list[str]:
    """Tokenize and optionally remove stopwords."""
    tokens = word_tokenize(text.lower())
    tokens = [t for t in tokens if t.isalpha()]
    if remove_stopwords:
        tokens = [t for t in tokens if t not in _STOP_WORDS]
    return tokens


def extract_entities(text: str) -> dict:
    """Extract named entities using spaCy."""
    nlp = _get_nlp()
    doc = nlp(text[:100000])  # Cap length for performance

    entities = {
        "ORG": [],
        "PERSON": [],
        "MONEY": [],
        "DATE": [],
        "GPE": [],
    }

    seen = set()
    for ent in doc.ents:
        if ent.label_ in entities and ent.text not in seen:
            entities[ent.label_].append(ent.text)
            seen.add(ent.text)

    return entities
