#!/usr/bin/env python3
"""Step 7: Evaluate signals against actual returns."""
import sys, json
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from config.settings import PROCESSED_DIR, PRICES_DIR, EVALUATION_DIR
from src.evaluation.metrics import (
    compute_classification_metrics, compute_utility, compute_calibration
)
from src.evaluation.baseline import keyword_baseline, sentiment_baseline, random_baseline
from src.evaluation.decay import analyze_decay
from src.models.walk_forward import merge_with_returns


def run():
    print("=" * 60)
    print("STEP 7: EVALUATING SIGNALS")
    print("=" * 60)

    sig_path = PROCESSED_DIR / "signals.parquet"
    if not sig_path.exists():
        print("Error: signals.parquet not found.")
        return

    signals_df = pd.read_parquet(sig_path)
    prices_path = PRICES_DIR / "daily_prices.parquet"
    spy_path = PRICES_DIR / "spy_prices.parquet"

    prices_df = pd.read_parquet(prices_path) if prices_path.exists() else pd.DataFrame()
    spy_df = pd.read_parquet(spy_path) if spy_path.exists() else pd.DataFrame()

    # Merge with returns
    if not prices_df.empty and not spy_df.empty:
        merged = merge_with_returns(signals_df, prices_df, spy_df)
    else:
        print("[WARN] No price data. Using synthetic outcome labels.")
        merged = signals_df.copy()
        merged["significant_move"] = (merged["keyword_score"] > 0.3).astype(int)

    if merged.empty:
        print("[WARN] No evaluable signals after merging.")
        return

    # Ground truth and predictions
    y_true = merged["significant_move"].values
    y_pred = (merged["decision"] == "ALERT").astype(int).values
    y_prob = merged["composite_score"].values / 100.0

    # 1. Advanced model metrics
    adv_metrics = compute_classification_metrics(y_true, y_pred, y_prob)
    adv_metrics["utility"] = compute_utility(y_true, y_pred)
    adv_metrics["total_evaluated"] = len(merged)

    # 2. Baselines
    kw_pred = keyword_baseline(merged["clean_text"].tolist())
    sent_pred = sentiment_baseline(merged["vader_compound"].values)
    rand_pred = random_baseline(len(merged))

    kw_metrics = compute_classification_metrics(y_true, kw_pred)
    kw_metrics["utility"] = compute_utility(y_true, kw_pred)

    sent_metrics = compute_classification_metrics(y_true, sent_pred)
    sent_metrics["utility"] = compute_utility(y_true, sent_pred)

    rand_metrics = compute_classification_metrics(y_true, rand_pred)
    rand_metrics["utility"] = compute_utility(y_true, rand_pred)

    # Save metrics
    all_metrics = {
        "advanced": adv_metrics,
        "keyword_baseline": kw_metrics,
        "sentiment_baseline": sent_metrics,
        "random_baseline": rand_metrics,
    }

    with open(EVALUATION_DIR / "eval_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2, default=str)
    print("[+] Saved eval_metrics.json")

    # 3. Baseline comparison table
    comparison = pd.DataFrame({
        "model": ["Advanced", "Keyword-only", "Sentiment-only", "Random"],
        "precision": [adv_metrics["precision"], kw_metrics["precision"],
                     sent_metrics["precision"], rand_metrics["precision"]],
        "recall": [adv_metrics["recall"], kw_metrics["recall"],
                  sent_metrics["recall"], rand_metrics["recall"]],
        "f1": [adv_metrics["f1"], kw_metrics["f1"],
              sent_metrics["f1"], rand_metrics["f1"]],
        "utility": [adv_metrics["utility"], kw_metrics["utility"],
                   sent_metrics["utility"], rand_metrics["utility"]],
    })
    comparison.to_csv(EVALUATION_DIR / "baseline_comparison.csv", index=False)
    print("[+] Saved baseline_comparison.csv")

    # 4. Calibration
    cal_df = compute_calibration(y_true, y_prob)
    cal_df.to_csv(EVALUATION_DIR / "calibration.csv", index=False)
    print("[+] Saved calibration.csv")

    # 5. Decay analysis
    if not prices_df.empty:
        decay_df = analyze_decay(signals_df, prices_df, spy_df)
        decay_df.to_csv(EVALUATION_DIR / "decay_analysis.csv", index=False)
        print("[+] Saved decay_analysis.csv")

    print(f"\n--- Results Summary ---")
    print(f"Advanced: prec={adv_metrics['precision']:.3f} rec={adv_metrics['recall']:.3f} "
          f"f1={adv_metrics['f1']:.3f} utility=${adv_metrics['utility']:.0f}")
    print(f"Keyword:  prec={kw_metrics['precision']:.3f} rec={kw_metrics['recall']:.3f}")

    print("\n[+] Evaluation complete.")


if __name__ == "__main__":
    run()
