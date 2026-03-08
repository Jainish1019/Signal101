"""
Feature pipeline: orchestrates all NLP steps for each chunk.
Optimized for speed with text truncation and batched embeddings.
"""
import pandas as pd
import numpy as np
from tqdm import tqdm

from src.nlp.preprocessor import clean_text, get_sentences, extract_entities
from src.nlp.sentiment import analyze_sentiment
from src.nlp.tfidf_engine import RollingTfidf
from src.nlp.embedder import embed_batch
from src.nlp.ner_extractor import (
    match_keywords, check_material_items,
    extract_evidence_quotes, get_entity_richness,
)

# Cap text length to avoid VADER/spaCy slowdowns on huge filings
MAX_TEXT_CHARS = 5000


def process_all_chunks(chunks_df: pd.DataFrame) -> pd.DataFrame:
    """
    Process all non-boilerplate chunks through the NLP pipeline.
    Maintains strict timestamp ordering for rolling window calculations.
    """
    # Filter out boilerplate
    df = chunks_df[~chunks_df["is_boilerplate"]].copy()
    df = df.sort_values("filed_at").reset_index(drop=True)

    print(f"Processing {len(df)} non-boilerplate chunks through NLP pipeline...")

    tfidf = RollingTfidf()

    # Phase 1: text cleaning + truncation
    print("  Phase 1/4: Cleaning text...")
    cleaned_texts = []
    for text in df["clean_text"]:
        c = clean_text(text)[:MAX_TEXT_CHARS]
        cleaned_texts.append(c)
    df["_cleaned"] = cleaned_texts

    # Phase 2: batch embedding (much faster than per-row)
    print("  Phase 2/4: Batch embedding...")
    embeddings = embed_batch(cleaned_texts, batch_size=64)

    # Phase 3: sentiment + keywords + entities (per-row, on truncated text)
    print("  Phase 3/4: Sentiment, keywords, NER...")
    results = []
    for idx in tqdm(range(len(df)), desc="NLP features"):
        row = df.iloc[idx]
        cleaned = cleaned_texts[idx]

        # Sentiment (fast on truncated text)
        sentiment = analyze_sentiment(cleaned)

        # TF-IDF novelty (rolling, no leakage)
        tfidf.add_document(cleaned)
        if idx > 0 and idx % 200 == 0:
            tfidf.fit_window(window_size=500)
        novelty = tfidf.novelty_score(cleaned)

        # NER (fast on truncated text)
        entities = extract_entities(cleaned)
        entity_richness = get_entity_richness(entities)

        # Keywords
        kw = match_keywords(cleaned)

        # Evidence quotes
        sentences = get_sentences(cleaned)
        quotes = extract_evidence_quotes(cleaned, sentences, top_n=3)

        # Material item check
        is_material_item = check_material_items(row.get("item_type", ""))

        feat = {
            "chunk_id": row["chunk_id"],
            "accession": row["accession"],
            "cik": row["cik"],
            "ticker": row["ticker"],
            "filed_at": row["filed_at"],
            "item_type": row.get("item_type", "unknown"),
            "clean_text": cleaned,
            "char_count": len(cleaned),
            # Sentiment
            **sentiment,
            # Novelty
            "novelty_score": novelty,
            # Keywords
            "keyword_score": kw["keyword_score"],
            "matched_keywords": ",".join(kw["matched_keywords"]),
            "keyword_categories": ",".join(kw["keyword_categories"]),
            "has_urgency": kw["has_urgency"],
            # Entities
            "entities_org": ",".join(entities.get("ORG", [])),
            "entities_person": ",".join(entities.get("PERSON", [])),
            "entities_money": ",".join(entities.get("MONEY", [])),
            "entity_richness": entity_richness,
            # Material
            "is_material_item": is_material_item,
            # Evidence
            "evidence_quotes": " ||| ".join(quotes),
        }
        results.append(feat)

    result_df = pd.DataFrame(results)
    print(f"\n[+] Feature extraction complete: {len(result_df)} rows")
    return result_df
