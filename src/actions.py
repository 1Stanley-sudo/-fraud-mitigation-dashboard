"""
Maps a transaction's risk rank within the flagged batch to an intervention.

Response is graduated rather than all-or-nothing because hard-blocking
every anomaly would create a support burden a small fintech cannot carry,
so friction rises with risk instead.

All actions are simulated. Nothing is blocked or locked for real.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Action:
    tier: str
    name: str
    icon: str
    description: str


_TIERS = [
    # (minimum risk percentile within the flagged set, action)
    (0.95, Action(
        "CRITICAL", "Session Isolation + Account Lock", "🔴",
        "Terminate the active session, lock the account pending manual review, "
        "and send immediate SMS/email notification to the customer and the "
        "fraud desk.")),
    (0.75, Action(
        "HIGH", "Transaction Hold", "🟠",
        "Suspend settlement before funds move. The transaction is queued for "
        "review and auto-released if cleared within the SLA window.")),
    (0.40, Action(
        "ELEVATED", "Step-up Authentication", "🟡",
        "Challenge the user with an additional verification factor (OTP or "
        "biometric) before the transaction is allowed to proceed.")),
    (0.0, Action(
        "WATCH", "Enhanced Monitoring", "🟢",
        "Allow the transaction but tag the account for tightened velocity "
        "limits and priority scoring on subsequent activity.")),
]


def assign_action(risk_percentile: float) -> Action:
    """Return the action for a flagged transaction, given its risk
    percentile within the batch (0.0 to 1.0)."""
    for threshold, action in _TIERS:
        if risk_percentile >= threshold:
            return action
    return _TIERS[-1][1]
