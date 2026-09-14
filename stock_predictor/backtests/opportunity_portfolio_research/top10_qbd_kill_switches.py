from __future__ import annotations
from dataclasses import dataclass
from datetime import date
from .top10_qbd_router_contracts import HealthLevel, ActivityLevel
from .top10_qbd_health_metrics import ExpertHealthSnapshot

@dataclass(frozen=True)
class KillDecision:
    triggered: bool; reason_code: str | None; evidence_date: date; policy_rule_id: str
@dataclass(frozen=True)
class KillVector:
    structural: KillDecision; activity: KillDecision; alpha: KillDecision; regime: KillDecision

def evaluate_kills(health: ExpertHealthSnapshot, policy, at: date) -> KillVector:
    d = lambda on, reason, rule: KillDecision(on, reason if on else None, at, rule)
    return KillVector(d(health.structural.level == HealthLevel.FAIL, "STRUCTURAL_FAILURE", "KILL_STRUCTURAL"),
                      d(health.activity.level == ActivityLevel.CRITICAL, "ACTIVITY_ANOMALY", "KILL_ACTIVITY"),
                      d(health.alpha.lcb_net_excess is not None and health.alpha.lcb_net_excess < policy.suspension_lcb_margin, "NEGATIVE_NET_ALPHA", "KILL_ALPHA"),
                      d(health.regime.shrunk_score < policy.suspension_lcb_margin, "REGIME_DECAY", "KILL_REGIME"))
