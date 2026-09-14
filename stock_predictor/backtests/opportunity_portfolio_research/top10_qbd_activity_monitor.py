from __future__ import annotations
from dataclasses import replace
from datetime import date
from .top10_qbd_router_contracts import ActivityLevel
from .top10_qbd_health_metrics import ActivityPosterior, ActivityHealth

def p_zero(a: float, b: float, exposure: float) -> float:
    return (b / (b + max(0.0, exposure))) ** a

def update_activity(previous: ActivityPosterior, *, eligible_opportunity: bool, traded: bool, current_date: date,
                    watch: float = .25, anomalous: float = .05, critical: float = .01) -> ActivityHealth:
    if not eligible_opportunity:
        nxt = previous
    else:
        nxt = replace(previous, a=previous.a + int(traded), b=previous.b + 1.0,
                      eligible_opportunities=previous.eligible_opportunities + 1,
                      actual_trades=previous.actual_trades + int(traded),
                      opportunity_time_since_last_trade=0 if traded else previous.opportunity_time_since_last_trade + 1,
                      last_trade_date=current_date if traded else previous.last_trade_date)
    # The anomaly question is conditional on the opportunity-time streak,
    # not on the age of the dataset or total historical exposure.
    prob = p_zero(nxt.a, nxt.b, nxt.opportunity_time_since_last_trade)
    level = ActivityLevel.NORMAL if prob >= watch else ActivityLevel.WATCH if prob >= anomalous else ActivityLevel.ANOMALOUS if prob >= critical else ActivityLevel.CRITICAL
    return ActivityHealth(level, prob, nxt, current_date)
