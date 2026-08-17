"""
Scoring for externally supplied transaction files.

Lets a user submit a file the models have never seen and have it scored by
the already-fitted models, which is how the generalisation claim is
demonstrated.

This is a file input, not a streaming API. A deployed system would receive
the same fields over a message queue and the scoring below would be
unchanged.
"""

import numpy as np
import pandas as pd

from . import data_pipeline as dp

# Fields required for scoring.
REQUIRED = set(dp.NUMERIC) | set(dp.CATEGORICAL)
# Optional. If is_fraud is present the file can also be scored for accuracy.
OPTIONAL = {dp.LABEL, "transaction_id", "customer_id", "timestamp"}


class SchemaError(ValueError):
    """Raised when an uploaded file is missing required fields."""


def validate(df: pd.DataFrame) -> dict:
    """Check an uploaded frame against the expected schema.

    Raises SchemaError only for missing required fields. Everything else is
    returned as a warning for the UI to show.
    """
    cols = set(df.columns)
    missing = sorted(REQUIRED - cols)
    if missing:
        raise SchemaError(
            f"{len(missing)} required column(s) missing: {', '.join(missing[:8])}"
            + ("..." if len(missing) > 8 else "")
        )

    unknown_cats = {}
    for c in dp.CATEGORICAL:
        seen = set(df[c].dropna().unique())
        known = set(_KNOWN_CATEGORIES.get(c, seen))
        novel = seen - known
        if novel:
            unknown_cats[c] = sorted(novel)[:5]

    return {
        "n_rows": len(df),
        "has_labels": dp.LABEL in cols,
        "extra_columns": sorted(cols - REQUIRED - OPTIONAL),
        "unknown_categories": unknown_cats,
        "null_counts": {c: int(df[c].isna().sum())
                        for c in sorted(REQUIRED) if df[c].isna().any()},
    }


# Category values seen in training. Unknown values are not fatal, since the
# one-hot columns are reindexed, but they are reported because they suggest
# the uploaded data differs from what the models were fitted on.
_KNOWN_CATEGORIES = {
    "channel": ["ATM", "ECOM", "IB", "Mobile", "POS", "Web"],
    "location": ["Abuja", "Lagos", "Ogun", "Other", "Oyo", "Rivers"],
    "age_group": ["<20", "20-29", "30-39", "40+"],
}


def prepare(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Build the feature matrix in the same columns and order the fitted
    models expect."""
    work = df.copy()
    for c in ("is_weekend", "is_peak_hour"):
        if work[c].dtype == object:
            work[c] = work[c].astype(str).str.lower().map(
                {"true": 1, "false": 0}).fillna(0)
        work[c] = work[c].astype(int)
    X_num = work[dp.NUMERIC].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    X_cat = pd.get_dummies(work[dp.CATEGORICAL], prefix=dp.CATEGORICAL,
                           dtype=float)
    X = pd.concat([X_num, X_cat], axis=1)
    # Reindex to the training columns: unknown categories are dropped and
    # missing ones zero-filled.
    return X.reindex(columns=feature_cols, fill_value=0.0)


def score_file(df: pd.DataFrame, bundle: dict, rf_threshold: float = 0.5) -> pd.DataFrame:
    """Score an uploaded frame.

    Returns the original rows plus risk scores, flags and the action tier
    assigned to each transaction.
    """
    from . import actions, models

    X = prepare(df, bundle["feature_cols"])
    out = df.copy().reset_index(drop=True)

    bds_risk, bds_flag = bundle["bds"].score(df.reset_index(drop=True))
    rf_prob, rf_flag = models.score_random_forest(
        bundle["rf"], X, threshold=rf_threshold)
    iso_risk, iso_flag = models.score_isolation_forest(bundle["iso"], X)

    out["deviation_risk"] = bds_risk
    out["deviation_flag"] = bds_flag
    out["rf_fraud_probability"] = rf_prob
    out["rf_flag"] = rf_flag
    out["iso_risk"] = iso_risk
    out["iso_flag"] = iso_flag

    # Action tier, based on rank within the flagged set.
    out["action_tier"] = "-"
    flagged = out.index[out["deviation_flag"] == 1]
    if len(flagged):
        pct = out.loc[flagged, "deviation_risk"].rank(pct=True)
        out.loc[flagged, "action_tier"] = [
            actions.assign_action(float(p)).tier for p in pct]
    return out
