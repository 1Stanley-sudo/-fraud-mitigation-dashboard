"""
Metrics, all computed on the held-out test partition.

1. Classification quality: precision, recall, F1 and ROC-AUC against the
   is_fraud label. Accuracy is not reported, since at a 0.3% fraud rate a
   model that flags nothing scores 99.7%.

2. Financial impact: the share of fraudulent value intercepted, reported
   with the legitimate value wrongly held as its cost.

3. Intercept latency: per-transaction scoring time in milliseconds. This
   is the decision step only, not full pipeline latency.
"""

import time

import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score,
)


def classification_metrics(y_true, flagged, risk_score) -> dict:
    """Precision, recall, F1, ROC-AUC and the confusion matrix."""
    tn, fp, fn, tp = confusion_matrix(y_true, flagged, labels=[0, 1]).ravel()
    return {
        "precision": precision_score(y_true, flagged, zero_division=0),
        "recall": recall_score(y_true, flagged, zero_division=0),
        "f1": f1_score(y_true, flagged, zero_division=0),
        "roc_auc": roc_auc_score(y_true, risk_score),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_negatives": int(tn),
    }


def flv_metrics(y_true, flagged, amounts) -> dict:
    """Financial impact of the model's decisions.

    protected_value  value of fraud intercepted
    missed_value     value of fraud that got through
    blocked_legit    value of legitimate transactions wrongly held
    protection_ratio protected_value / total fraud value
    friction_ratio   blocked_legit / total legitimate value
    """
    y_true = np.asarray(y_true)
    flagged = np.asarray(flagged)
    amounts = np.asarray(amounts, dtype=float)

    fraud_total = amounts[y_true == 1].sum()
    legit_total = amounts[y_true == 0].sum()
    protected = amounts[(y_true == 1) & (flagged == 1)].sum()
    missed = amounts[(y_true == 1) & (flagged == 0)].sum()
    blocked_legit = amounts[(y_true == 0) & (flagged == 1)].sum()

    return {
        "fraud_value_total": fraud_total,
        "protected_value": protected,
        "missed_value": missed,
        "blocked_legit_value": blocked_legit,
        "protection_ratio": protected / fraud_total if fraud_total else 0.0,
        "friction_ratio": blocked_legit / legit_total if legit_total else 0.0,
    }


def intercept_latency(score_fn, X: pd.DataFrame, n: int = 200) -> dict:
    """Per-transaction scoring time.

    Scores n transactions one at a time, as a real-time stream would,
    and reports mean and 95th percentile in milliseconds.
    """
    n = min(n, len(X))
    sample = X.iloc[:n]
    timings = []
    for i in range(n):
        row = sample.iloc[[i]]
        t0 = time.perf_counter()
        score_fn(row)
        timings.append((time.perf_counter() - t0) * 1000.0)
    timings = np.array(timings)
    return {
        "mean_ms": float(timings.mean()),
        "p95_ms": float(np.percentile(timings, 95)),
        "n_sampled": n,
    }
