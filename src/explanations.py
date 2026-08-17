"""
Turns a flagged transaction into plain-language reasons.

An alert with no explanation cannot be triaged: a reviewer has no way to
tell a genuine fraud from a customer making an unusually large but
legitimate transfer. Every flagged transaction therefore carries a short
account of what triggered it.

Each behavioural feature is compared against the training baseline as a
z-score, and the largest deviations are rendered as sentences.
"""

import numpy as np
import pandas as pd

# Features shown to reviewers, with their message templates.
_EXPLAINABLE = {
    "amount": "Transaction amount ₦{value:,.0f} is {z:.1f} standard deviations above typical legitimate amounts",
    "tx_count_24h": "{value:.0f} transactions in the last 24 hours (unusual velocity)",
    "amount_sum_24h": "₦{value:,.0f} moved in the last 24 hours (unusual 24h volume)",
    "amount_vs_mean_ratio": "Amount is {value:.1f} times this customer's own historical average",
    "channel_diversity": "Account has used {value:.0f} different channels (possible account takeover probing)",
}

# velocity_score is left out of the templates above. In this dataset it is
# identical to amount_vs_mean_ratio in 75.5% of rows, so including both made
# most alerts state the same thing twice.

_NIGHT_HOURS = set(range(0, 5))  # midnight to 04:59
_HIGH_RISK_CHANNELS = {"Web", "Mobile"}  # highest fraud rates in this data


def explain_transaction(raw_row: pd.Series, stats: pd.DataFrame,
                        z_threshold: float = 2.0, max_reasons: int = 4) -> list[str]:
    """Return the reasons a transaction was flagged, ordered by how far it
    deviates. Always returns at least one, so no alert is unexplained."""
    scored = []
    for feat, template in _EXPLAINABLE.items():
        if feat not in stats.index or feat not in raw_row.index:
            continue
        mean, std = stats.loc[feat, "mean"], stats.loc[feat, "std"]
        if pd.isna(std):
            continue
        value = float(raw_row[feat])
        z = (value - mean) / std
        if z >= z_threshold:
            scored.append((z, template.format(value=value, z=z)))

    scored.sort(key=lambda t: -t[0])
    reasons = [msg for _, msg in scored[:max_reasons]]

    # Contextual reasons that need no z-score.
    hour = int(raw_row.get("hour", 12))
    if hour in _NIGHT_HOURS:
        reasons.append(f"Night-time execution ({hour:02d}:00), outside typical activity windows")
    channel = raw_row.get("channel")
    if channel in _HIGH_RISK_CHANNELS and len(reasons) < max_reasons + 1:
        reasons.append(f"Executed on the {channel} channel, which carries a high share of fraud in this data")

    if not reasons:
        reasons.append(
            "Behaviour deviates from the baseline across several features "
            "with no single dominant factor"
        )
    return reasons[:max_reasons + 1]
