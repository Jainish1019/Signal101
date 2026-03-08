"""
Evaluation metrics: precision, recall, F1, AUC, calibration, utility.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    accuracy_score, roc_auc_score,
)
from config.settings import TP_REWARD, FP_PENALTY


def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                                    y_prob: np.ndarray = None) -> dict:
    """Compute standard classification metrics."""
    metrics = {
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
    }

    if y_prob is not None and len(np.unique(y_true)) > 1:
        metrics["auc"] = round(roc_auc_score(y_true, y_prob), 4)
    else:
        metrics["auc"] = 0.0

    # Confusion matrix elements
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    metrics.update({"tp": tp, "fp": fp, "tn": tn, "fn": fn})

    return metrics


def compute_utility(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Cost-weighted utility: TP * reward + FP * penalty."""
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    return float(tp * TP_REWARD + fp * FP_PENALTY)


def compute_calibration(y_true: np.ndarray, y_prob: np.ndarray,
                         n_bins: int = 10) -> pd.DataFrame:
    """Compute calibration curve: predicted probability vs observed frequency."""
    bins = np.linspace(0, 1, n_bins + 1)
    rows = []

    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        if mask.sum() == 0:
            continue
        predicted = y_prob[mask].mean()
        observed = y_true[mask].mean()
        rows.append({
            "bin_center": round((bins[i] + bins[i + 1]) / 2, 2),
            "predicted": round(predicted, 4),
            "observed": round(observed, 4),
            "count": int(mask.sum()),
        })

    return pd.DataFrame(rows)
