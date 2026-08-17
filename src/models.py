"""
Tier 2: the three models.

1. Isolation Forest. The first label-free attempt. It fails on this data
   (AUC around 0.54) because fraud here is not an outlier: it sits at
   elevated value inside the normal range. Kept as a negative result.

2. Behavioural Deviation Scorer. The label-free model actually used. It
   scores upward deviation of transaction value from the baseline. A firm
   with no fraud history can fit and run it.

3. Random Forest. Supervised benchmark, trained with labels to show the
   ceiling. The gap between (2) and (3) is what labels are worth.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier

# Channel risk priors, kept as a sensitivity setting rather than a fixed
# weighting. Published sources disagree: NIBSS ranks e-commerce and internet
# banking as most affected, while the CBN attributes digital losses mainly to
# unregulated virtual asset platforms. Neither matches this dataset, where
# Web, Mobile and POS carry the highest fraud rates.
#
# Tuning the priors to fit the dataset's own fraud distribution would leak
# the labels, so the default is uniform and the alternatives are kept only
# so the sensitivity analysis can be reproduced.
CHANNELS = ["Mobile", "Web", "POS", "ECOM", "IB", "ATM"]

PRIOR_SETS = {
    # Default: no channel weighting.
    "uniform": {c: 1.00 for c in CHANNELS},
    # Ranking published by NIBSS.
    "nibss_channel_rank": {
        "ECOM": 1.25, "IB": 1.20, "POS": 1.05,
        "Mobile": 1.00, "Web": 0.95, "ATM": 0.60,
    },
    # Mobile-first ranking implied by CBN commentary.
    "mobile_first": {
        "Mobile": 1.25, "Web": 1.15, "POS": 1.00,
        "ECOM": 1.00, "IB": 0.85, "ATM": 0.60,
    },
}

# Backwards-compatible alias.
CHANNEL_PRIORS = PRIOR_SETS["uniform"]


def train_isolation_forest(X_train: pd.DataFrame,
                           contamination: float = 0.003,
                           random_state: int = 42) -> IsolationForest:
    """Fit the Isolation Forest.

    contamination is set to the observed fraud rate (0.3%) so the number of
    flags matches the amount of fraud actually present. No labels are used.
    """
    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train)
    return model


def train_random_forest(X_train: pd.DataFrame, y_train: pd.Series,
                        random_state: int = 42) -> RandomForestClassifier:
    """Fit the Random Forest benchmark.

    class_weight='balanced_subsample' stops the model collapsing into
    predicting 'legitimate' for everything at a 0.3% positive rate.
    """
    model = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced_subsample",
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def score_isolation_forest(model: IsolationForest, X: pd.DataFrame):
    """Return (risk_score, flagged).

    decision_function is high for normal points and low for anomalies, so
    it is negated here to keep higher = riskier across all models.
    """
    risk = -model.decision_function(X)
    flagged = (model.predict(X) == -1).astype(int)
    return risk, flagged


def score_random_forest(model: RandomForestClassifier, X: pd.DataFrame,
                        threshold: float = 0.5):
    """Return (fraud_probability, flagged)."""
    proba = model.predict_proba(X)[:, 1]
    flagged = (proba >= threshold).astype(int)
    return proba, flagged


class BehaviouralDeviationScorer:
    """Label-free scorer used in place of the Isolation Forest.

    risk = max(0, z) * channel_prior, where z is the standard score of
    log-amount against the training baseline. The clip is one-sided:
    only upward deviations raise risk, since an unusually small
    transaction is not suspicious.

    No labels are needed. The baseline uses all training traffic and the
    threshold is a percentile of that same traffic, so a firm with no
    fraud history can fit this on day one.
    """

    def __init__(self, flag_rate: float = 0.01, priors: str = "uniform"):
        # flag_rate: share of traffic flagged for intervention.
        # priors: key into PRIOR_SETS. Default 'uniform' applies no channel
        #         weighting.
        self.flag_rate = flag_rate
        self.priors = priors
        self.prior_map = PRIOR_SETS.get(priors, PRIOR_SETS["uniform"])
        self.mu_ = None
        self.sd_ = None
        self.threshold_ = None

    def fit(self, df_train: pd.DataFrame) -> "BehaviouralDeviationScorer":
        self.mu_ = float(df_train["amount_log"].mean())
        self.sd_ = float(df_train["amount_log"].std()) or 1.0
        # Threshold set so roughly flag_rate of training traffic exceeds it.
        train_risk = self._risk(df_train)
        self.threshold_ = float(np.percentile(train_risk, 100 * (1 - self.flag_rate)))
        return self

    def _risk(self, df: pd.DataFrame) -> np.ndarray:
        z = np.clip((df["amount_log"].values - self.mu_) / self.sd_, 0, None)
        priors = df["channel"].map(self.prior_map).fillna(1.0).values
        return z * priors

    def score(self, df: pd.DataFrame):
        """Score raw transaction records."""
        risk = self._risk(df)
        flagged = (risk >= self.threshold_).astype(int)
        return risk, flagged
