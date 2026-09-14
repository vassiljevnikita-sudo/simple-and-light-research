"""Causal shadow decision construction; no realized outcome enters this API."""
from __future__ import annotations
from datetime import date
from typing import Mapping
from .top10_qbd_shadow_ledger import ShadowDecision, PendingOutcome

def make_shadow_decision(*, decision_date: date, expert_id: str, eligible_opportunity: bool,
                         prediction: float | None, threshold: float | None,
                         holding_days: int, market_regime: str | None = None,
                         market_state_features: Mapping[str, float] | None = None,
                         prediction_uncertainty: float | None = None,
                         session_dates: tuple[date, ...] | None = None) -> tuple[ShadowDecision, PendingOutcome | None]:
    enter = bool(eligible_opportunity and prediction is not None and (threshold is None or prediction >= threshold))
    outcome_date = None
    pending = None
    if enter:
        if session_dates is not None:
            try:
                i = session_dates.index(decision_date)
                outcome_date = session_dates[min(len(session_dates) - 1, i + max(1, holding_days) + 1)]
            except ValueError:
                outcome_date = None
        else:
            # Fallback retained for small synthetic callers.  Production
            # providers must pass the exchange-session calendar.
            from datetime import timedelta
            outcome_date = decision_date + timedelta(days=max(1, holding_days) + 1)
        if outcome_date is None:
            pending = None
        else:
            pending = PendingOutcome(decision_date, expert_id, outcome_date, f"{decision_date.isoformat()}::{expert_id}")
    return ShadowDecision(decision_date, expert_id, eligible_opportunity, prediction, threshold, enter, False,
                          market_regime, dict(market_state_features or {}), prediction_uncertainty, outcome_date), pending
