from __future__ import annotations
from dataclasses import dataclass
from .top10_qbd_router_contracts import LifecycleState
from .top10_qbd_policy import capabilities_for
from dataclasses import replace
from .top10_qbd_allocation import allocation_weights, validate_weights
from .top10_qbd_candidate_set import build_candidate_set
from .top10_qbd_kill_switches import evaluate_kills
from .top10_qbd_state import RouterStateV1, initial_lifecycle, transition_expert
from .top10_qbd_promotion import assess_promotion
from .top10_qbd_policy import RouterArm
from .top10_qbd_v4_adapter import FrozenV4ControllerAdapter

@dataclass(frozen=True)
class AllocationDecision:
    decision_date: object; weights: dict[str, float]; champion_id: str | None; candidate_members: tuple[str, ...]
    fallback_used: str | None; reason_code: str; policy_hash: str; registry_hash: str; matured_evidence_cursor: str

class QbdRouterV1:
    def __init__(self, policy, registry, registry_hash):
        self.policy, self.registry, self.registry_hash, self.capabilities = policy, registry, registry_hash, capabilities_for(policy.router_arm)
        self.v4_controller = FrozenV4ControllerAdapter()
    def decide(self, *, current_date, previous_state, matured_outcomes, shadow_decisions):
        health = dict(previous_state.health)
        v4_gate = self.v4_controller.causal_gate(tuple(float(x.net_excess) for x in matured_outcomes))
        kills = {eid: evaluate_kills(h, self.policy, current_date) for eid, h in health.items()}
        if not self.capabilities.opportunity_activity:
            kills = {eid: replace(k, activity=k.activity.__class__(False, None, current_date, "DISABLED_BY_ARM")) for eid,k in kills.items()}
        if not self.capabilities.regime_shrinkage:
            kills = {eid: replace(k, regime=k.regime.__class__(False, None, current_date, "DISABLED_BY_ARM")) for eid,k in kills.items()}
        candidates = build_candidate_set(health, kills, self.policy, current_date)
        lifecycle = {eid: transition_expert(previous_state.lifecycle.get(eid, initial_lifecycle(eid, current_date)), health[eid], kills[eid], candidates, self.policy, current_date) for eid in health}
        members = tuple(x for x in candidates.members if lifecycle[x].state != LifecycleState.SUSPENDED)
        scores = {}
        for x in members:
            h=health[x]; score=h.alpha.discounted_net_excess
            if self.capabilities.regime_shrinkage: score=h.regime.shrunk_score
            if self.capabilities.changepoint and h.changepoint is not None: score*=h.changepoint.forgetting_multiplier
            if self.capabilities.uncertainty_lcb: score=(h.alpha.lcb_net_excess if h.alpha.lcb_net_excess is not None else score) - h.uncertainty_score
            scores[x]=score * (v4_gate if matured_outcomes else 1.0)
        incumbent_id = previous_state.champion_id if previous_state.champion_id in members else None
        challenger_id = max(members, key=lambda x: (health[x].alpha.lcb_net_excess if health[x].alpha.lcb_net_excess is not None else scores[x], x), default=None)
        incumbent_health = health.get(incumbent_id) if incumbent_id else None
        challenger_health = health.get(challenger_id) if challenger_id else None
        duration_met = incumbent_id is None or previous_state.lifecycle[incumbent_id].sessions_in_state >= self.policy.min_champion_sessions
        paired_ch, paired_inc = [], []
        if challenger_id and incumbent_id:
            for outcome in matured_outcomes:
                if outcome.expert_id == challenger_id: paired_ch.append(outcome.net_excess)
                elif outcome.expert_id == incumbent_id: paired_inc.append(outcome.net_excess)
        promotion = assess_promotion(challenger_health, incumbent_health, self.policy, duration_met,
                                     challenger_returns=paired_ch if paired_ch and len(paired_ch)==len(paired_inc) else None,
                                     incumbent_returns=paired_inc if paired_ch and len(paired_ch)==len(paired_inc) else None) if challenger_health else None
        if self.policy.router_arm == RouterArm.A_STATIC_BEST_DEV:
            chosen = incumbent_id or challenger_id
        elif promotion and promotion.promotion_allowed:
            chosen = challenger_id
        else:
            chosen = incumbent_id or challenger_id
        hard = self.policy.router_arm in (RouterArm.B_HARD_CHAMPION_3M, RouterArm.C_HARD_CHAMPION_6M)
        mode = "equal" if self.capabilities.equal_weight else "softmax"
        fallback = None
        if hard and chosen:
            weights = {chosen: 1.0}
        elif chosen and self.policy.router_arm == RouterArm.A_STATIC_BEST_DEV:
            weights = {chosen: 1.0}
        else:
            weights = allocation_weights(members, scores, mode=mode, eta=self.policy.softmax_eta, fallback=self.policy.fallback_order[0])
        if not members: fallback = self.policy.fallback_order[0]
        if chosen and chosen in weights:
            lifecycle[chosen] = lifecycle[chosen].__class__(chosen, LifecycleState.CHAMPION, lifecycle[chosen].entered_state_at, lifecycle[chosen].prior_state, "PROMOTION_HURDLE" if promotion and promotion.promotion_allowed else "CHAMPION_HOLD", current_date, lifecycle[chosen].sessions_in_state)
        if previous_state.champion_id and previous_state.champion_id != chosen and previous_state.champion_id in lifecycle:
            old = lifecycle[previous_state.champion_id]
            lifecycle[previous_state.champion_id] = old.__class__(old.expert_id, LifecycleState.PROBATION, old.entered_state_at, old.state, "CHALLENGER_PROMOTED", current_date, old.sessions_in_state)
        validate_weights(weights, [x for x, v in lifecycle.items() if v.state == LifecycleState.SUSPENDED])
        champion = chosen
        switched = previous_state.champion_id != champion and champion is not None
        switch_date = current_date if switched else previous_state.last_switch_date
        state = RouterStateV1("QBD_ROUTER_STATE_V1", current_date, self.policy.policy_hash, self.registry_hash, champion, members, weights, lifecycle, health, switch_date, previous_state.matured_evidence_cursor, {"kills": kills, "v4_gate": v4_gate})
        return state, AllocationDecision(current_date, weights, champion, members, fallback, f"CAUSAL_MATURED_EVIDENCE_V4_GATE_{v4_gate:.6f}", self.policy.policy_hash, self.registry_hash, previous_state.matured_evidence_cursor)
