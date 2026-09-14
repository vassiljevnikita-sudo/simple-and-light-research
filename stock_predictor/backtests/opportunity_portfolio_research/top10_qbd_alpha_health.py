from __future__ import annotations
from datetime import date
from .top10_qbd_health_metrics import AlphaHealth, alpha_from_outcomes
from .top10_qbd_shadow_ledger import MaturedOutcome

def update_alpha_health(previous: AlphaHealth, matured: tuple[MaturedOutcome, ...], policy, evaluated_at: date) -> AlphaHealth:
    return alpha_from_outcomes(previous, matured, evaluated_at, policy.alpha_discount)
