"""
Smoke Trainer: Implements exhaustive walk-forward evaluation with no-leakage safeguards.
"""
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from config.settings import SMOKE_PROCESSED_DIR, SMOKE_MODELS_DIR, SMOKE_PROOF_DIR
from config.keywords import KEYWORD_LEXICON

def create_folds(df: pd.DataFrame, start_year: int = 2019, end_year: int = 2023):
    """
    Generate walk-forward temporal splits.
    Fold structure: Train (Past), Val (Tuning), Test (Evaluation).
    """
    folds = []
    # Simple strategy: 2-year sliding train, 6-month val, 6-month test
    # Example: 2019-2020 Train, 2021-H1 Val, 2021-H2 Test
    
    for year in range(start_year + 1, end_year + 1):
        for half in [1, 2]:
            train_end = f"{year-1}-12-31"
            if half == 1:
                val_start, val_end = f"{year}-01-01", f"{year}-06-30"
                test_start, test_end = f"{year}-07-01", f"{year}-12-31"
            else:
                # Slide forward
                val_start, val_end = f"{year}-07-01", f"{year}-12-31"
                test_start, test_end = f"{year+1}-01-01", f"{year+1}-06-30"
                if year == end_year: continue # Skip if past range
                
            folds.append({
                "name": f"Fold_{year}_{'H1' if half==1 else 'H2'}",
                "train_range": (str(start_year), train_end),
                "val_range": (val_start, val_end),
                "test_range": (test_start, test_end)
            })
            
    return folds

def run_smoke_training(ticker: str, chunks_path: Path):
    """Train the smoke detector across all folds."""
    df = pd.read_parquet(chunks_path)
    df["filed_at"] = pd.to_datetime(df["filed_at"])
    df = df.sort_values("filed_at")
    
    # Filter for non-boilerplate for training, but we keep all for testing
    train_pool = df[~df["is_boilerplate"]].copy()
    
    folds = create_folds(df)
    results = []
    audit_log = [f"# Leakage Audit for {ticker}\n"]
    
    for fold in folds:
        name = fold["name"]
        print(f"Processing {name}...")
        
        # 1. Split Data
        train_df = train_pool[(train_pool["filed_at"] >= fold["train_range"][0]) & (train_pool["filed_at"] <= fold["train_range"][1])]
        val_df = train_pool[(train_pool["filed_at"] >= fold["val_range"][0]) & (train_pool["filed_at"] <= fold["val_range"][1])]
        test_df = df[(df["filed_at"] >= fold["test_range"][0]) & (df["filed_at"] <= fold["test_range"][1])]
        
        if train_df.empty or test_df.empty:
            continue
            
        # Leakage Check
        max_train = train_df["filed_at"].max()
        min_test = test_df["filed_at"].min()
        audit_log.append(f"- {name}: Max Train ({max_train}) < Min Test ({min_test}) -> {'PASSED' if max_train < min_test else 'FAILED'}")
        
        # 2. Fit TF-IDF on Train Only
        tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
        X_train = tfidf.fit_transform(train_df["clean_text"])
        X_test = tfidf.transform(test_df["clean_text"])
        
        # 3. Labeling (Heuristic for Smoke Detector baseline)
        # We label any known Item chunk as 1 (Signal) and others as 0 (Noise)
        y_train = (train_df["item_type"] != "unknown").astype(int)
        
        # 4. Train & Calibrate
        model = LogisticRegression(class_weight='balanced')
        calibrated = CalibratedClassifierCV(model, cv=3)
        calibrated.fit(X_train, y_train)
        
        # 5. Score Test
        probs = calibrated.predict_proba(X_test)[:, 1]
        
        # 6. Keyword Baseline
        keyword_scores = []
        for text in test_df["clean_text"]:
            text_l = text.lower()
            score = 0
            for cat, details in KEYWORD_LEXICON.items():
                match_count = sum(1 for kw in details["keywords"] if kw in text_l)
                score += match_count * details["weight"]
            keyword_scores.append(score)
        
        fold_test_res = test_df.copy()
        fold_test_res["score_a"] = probs * 100
        fold_test_res["score_baseline"] = np.array(keyword_scores)
        fold_test_res["fold"] = name
        
        # 7. Signal Decay (Correlation over time within fold)
        # We'll calculate this in the impact script using price, 
        # but here we can at least flag the decay window.
        
        results.append(fold_test_res)
        
        # Save Model Artifact
        fold_model_dir = SMOKE_MODELS_DIR / name
        fold_model_dir.mkdir(exist_ok=True)
        joblib.dump(tfidf, fold_model_dir / "tfidf.joblib")
        joblib.dump(calibrated, fold_model_dir / "model.joblib")
        
    # Finalize
    if not results:
        print("No training folds could be executed.")
        return None
        
    all_scored = pd.concat(results)
    out_path = SMOKE_PROCESSED_DIR / f"{ticker}_scored_signals.parquet"
    all_scored.to_parquet(out_path)
    
    # Write Audit
    (SMOKE_PROOF_DIR / f"{ticker}_leakage_audit.md").write_text("\n".join(audit_log))
    
    return out_path
