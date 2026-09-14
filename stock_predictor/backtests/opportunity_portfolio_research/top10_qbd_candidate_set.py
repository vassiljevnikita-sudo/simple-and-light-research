from __future__ import annotations
from dataclasses import dataclass
from datetime import date
from .top10_qbd_health_metrics import ExpertHealthSnapshot
from .top10_qbd_kill_switches import KillVector

@dataclass(frozen=True)
class CandidateAssessment:
    expert_id: str; eligible: bool; score: float; lcb: float | None; uncertainty: float; exclusion_reason: str | None
@dataclass(frozen=True)
class CandidateSet:
    as_of: date; members: tuple[str, ...]; assessments: tuple[CandidateAssessment, ...]

def build_candidate_set(health, kills, policy, at: date) -> CandidateSet:
    rows=[]
    for eid, h in sorted(health.items()):
        k = kills.get(eid)
        sufficient = h.alpha.effective_n >= policy.min_matured_observations and h.activity.posterior.eligible_opportunities >= policy.min_eligible_opportunities
        killed = k is not None and any((k.structural.triggered, k.alpha.triggered, k.activity.triggered))
        cp_block = h.changepoint is not None and h.changepoint.change_probability >= float(policy.changepoint_config.get("alert", .65))
        eligible = eid.startswith("R") and sufficient and not killed and not cp_block
        reason = None if eligible else ("CHANGE_POINT_PRESSURE" if cp_block else "INSUFFICIENT_MATURED_EVIDENCE" if not sufficient else "KILL_SWITCH_OR_NOT_STOCK")
        rows.append(CandidateAssessment(eid, eligible, h.alpha.discounted_net_excess, h.alpha.lcb_net_excess, h.uncertainty_score, reason))
    eligible_rows = [x for x in rows if x.eligible and (x.lcb is None or x.lcb > policy.suspension_lcb_margin)]
    best = max((x.lcb if x.lcb is not None else x.score for x in eligible_rows), default=None)
    members = tuple(x.expert_id for x in eligible_rows if best is None or (x.lcb if x.lcb is not None else x.score) >= best - max(policy.promotion_lcb_margin, .0005))
    return CandidateSet(at, members, tuple(rows))
