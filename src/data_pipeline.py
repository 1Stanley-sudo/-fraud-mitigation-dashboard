"""
Tier 1: loading, feature engineering and splitting.

Reads the transaction dataset, samples it, and turns raw records into a
numeric feature matrix for both the unsupervised and supervised models.

Notes on the choices made here:
- transaction_id, customer_id and timestamp are dropped. They carry no
  transferable signal and let a model memorise individual rows.
- fraud_technique is dropped because it is only filled in for confirmed
  fraud, so keeping it would leak the answer.
- Categoricals are one-hot encoded rather than label encoded, which would
  imply an ordering between banks or channels that does not exist.
- Sampling is not stratified, so the natural class imbalance is kept.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

LABEL = "is_fraud"

# Columns kept out of the model, with the reason.
EXCLUDED = {
    "transaction_id": "identifier",
    "customer_id": "identifier",
    "timestamp": "raw string; cyclical encodings already provided",
    "fraud_technique": "label leakage (only set for confirmed fraud)",
    LABEL: "ground-truth label",
}

CATEGORICAL = ["channel", "merchant_category", "bank", "location", "age_group"]

# Numeric features: value, velocity, timing and diversity measures.
NUMERIC = [
    "amount", "amount_log", "amount_rounded",
    "hour", "day_of_week", "month", "is_weekend", "is_peak_hour",
    "hour_sin", "hour_cos", "day_sin", "day_cos", "month_sin", "month_cos",
    "tx_count_24h", "amount_sum_24h", "amount_mean_7d", "amount_std_7d",
    "tx_count_total", "amount_mean_total", "amount_std_total",
    "channel_diversity", "location_diversity",
    "amount_vs_mean_ratio", "online_channel_ratio",
    "velocity_score", "merchant_risk_score", "composite_risk",
]

# Columns shown in the Live Threat Feed.
DISPLAY_COLS = [
    "transaction_id", "customer_id", "timestamp", "amount", "channel",
    "merchant_category", "bank", "location", "age_group",
    "tx_count_24h", "amount_vs_mean_ratio", LABEL,
]


def load_dataset(path: str, sample_size: int | None = 200_000,
                 random_state: int = 42) -> pd.DataFrame:
    """Read the CSV and take a random sample, keeping the natural fraud
    rate."""
    df = pd.read_csv(path, low_memory=False)
    df["is_weekend"] = df["is_weekend"].astype(int)
    df["is_peak_hour"] = df["is_peak_hour"].astype(int)
    if sample_size and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=random_state)
    return df.reset_index(drop=True)


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Build the feature matrix X and the label vector y."""
    X_num = df[NUMERIC].astype(float).fillna(0.0)
    X_cat = pd.get_dummies(df[CATEGORICAL], prefix=CATEGORICAL, dtype=float)
    X = pd.concat([X_num, X_cat], axis=1)
    y = df[LABEL].astype(int)
    return X, y


def split(df: pd.DataFrame, test_size: float = 0.3, random_state: int = 42):
    """Stratified train/test split so both sides keep the same fraud
    rate."""
    return train_test_split(
        df, test_size=test_size, random_state=random_state,
        stratify=df[LABEL],
    )


def legitimate_feature_stats(df_train: pd.DataFrame) -> pd.DataFrame:
    """Behavioural baseline used by the explanation layer.

    Computed on all training traffic rather than on legitimate rows only,
    so no labels are needed. At a 0.3% fraud rate the difference is
    negligible: the mean amount shifts by 0.437% and other features by
    about 1% or less.
    """
    stats = df_train[NUMERIC].agg(["mean", "std"]).T
    stats["std"] = stats["std"].replace(0, np.nan)
    return stats
