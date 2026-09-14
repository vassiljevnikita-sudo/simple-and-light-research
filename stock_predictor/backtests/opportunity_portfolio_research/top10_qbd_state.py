from __future__ import annotations
from dataclasses import dataclass
from datetime import date
from .top10_qbd_router_contracts import LifecycleState
from .top10_qbd_health_metrics import ExpertHealthSnapshot
from .top10_qbd_candidate_set import CandidateSet
from .top10_qbd_kill_switches import KillVector

@dataclass(frozen=True)
class ExpertLifecycle:
    expert_id: str; state: LifecycleState; entered_state_at: date; prior_state: LifecycleState | None; reason_code: str; evidence_date: date; sessions_in_state: int = 0
@dataclass(frozen=True)
class RouterStateV1:
    schema_version: str; as_of: date; policy_hash: str; registry_hash: str; champion_id: str | None
    candidate_members: tuple[str, ...]; current_weights: dict[str, float]; lifecycle: dict[str, ExpertLifecycle]
    health: dict[str, ExpertHealthSnapshot]; last_switch_date: date | None; matured_evidence_cursor: str
    persisted_component_states: dict[str, object]

def initial_lifecycle(expert_id: str, at: date) -> ExpertLifecycle:
    return ExpertLifecycle(expert_id, LifecycleState.SHADOW, at, None, "INITIAL_SHADOW", at)

def transition_expert(previous, health, kills, candidates, policy, current_date):
    if kills.structural.triggered:
        return ExpertLifecycle(previous.expert_id, LifecycleState.SUSPENDED, current_date, previous.state, kills.structural.reason_code or "STRUCTURAL", kills.structural.evidence_date)
    statistical_kill = kills.activity.triggered or kills.alpha.triggered or kills.regime.triggered
    if statistical_kill and previous.state == LifecycleState.PROBATION:
        next_state = LifecycleState.SUSPENDED
    elif statistical_kill and previous.state in (LifecycleState.CHAMPION, LifecycleState.ENSEMBLE):
        next_state = LifecycleState.PROBATION
    elif previous.expert_id not in candidates.members:
        next_state = LifecycleState.PROBATION if previous.state in (LifecycleState.CHAMPION, LifecycleState.ENSEMBLE) else previous.state
    elif previous.state == LifecycleState.SUSPENDED:
        next_state = LifecycleState.CANDIDATE
    elif previous.state == LifecycleState.SHADOW:
        next_state = LifecycleState.CANDIDATE
    elif previous.state == LifecycleState.CANDIDATE and health.alpha.lcb_net_excess is not None and health.alpha.lcb_net_excess > policy.promotion_lcb_margin:
        next_state = LifecycleState.CHAMPION
    else: next_state = previous.state
    same = next_state == previous.state
    return ExpertLifecycle(previous.expert_id, next_state, previous.entered_state_at if same else current_date, previous.state if not same else previous.prior_state, "CANDIDATE_SET" if same else "CAUSAL_HEALTH_UPDATE", current_date, previous.sessions_in_state + 1 if same else 0)
