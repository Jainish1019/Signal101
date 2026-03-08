# Agent 1: Integrity & No-Leakage Report

## 1. Timestamp Ordering Validation
> **PASS**: The signal dataset is strictly monotonically increasing by `filed_at` timestamp. This confirms the time-locked ingestion engine functioned correctly.

## 2. Walk-Forward Constraints
- **TF-IDF Novelty**: Computed using a rolling 500-document window (`fit_window`), ensuring no future vocabulary leakage.
- **Semantic Drift**: Drift centroids are updated sequentially per ticker. Evaluation uses T-1 state for T0 signals.
- **Pricing Ground Truth**: Abnormal returns strictly calculate T+1 close relative to T0 close strictly bounded by `filed_at` dates.

## 3. Reproducibility
- Random seeds fixed for baseline models.
- Deterministic NLP extraction (spaCy, VADER, TF-IDF).
- Free-only stack validated: no paid APIs leveraged in core scoring pipeline.
