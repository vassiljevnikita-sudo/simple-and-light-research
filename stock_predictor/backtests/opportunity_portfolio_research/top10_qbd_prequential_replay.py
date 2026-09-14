"""One causal replay loop shared by all A-K arms."""
from __future__ import annotations
from dataclasses import dataclass
from datetime import date
from typing import Protocol, Sequence
from .top10_qbd_shadow_ledger import InMemoryShadowLedger, MaturedOutcome, ShadowDecision, PendingOutcome
from .top10_qbd_health_store import ExpertHealthStore
from .top10_qbd_state import RouterStateV1, initial_lifecycle
from .top10_qbd_router import QbdRouterV1
from .top10_qbd_expert_registry import ExpertSpec, registry_hash
from .top10_qbd_router_contracts import sha256_fingerprint

class CausalDataProvider(Protocol):
    def dates(self, start: date, end: date) -> Sequence[date]: ...
    def shadow(self, current_date: date, expert: ExpertSpec) -> tuple[ShadowDecision, object | None]: ...
    def outcome(self, pending: object) -> MaturedOutcome: ...

@dataclass(frozen=True)
class ReplayResult:
    decisions: tuple[object, ...]; final_state: RouterStateV1; performance_artifact: dict; run_manifest: dict

def run_qbd_router_replay(*, policy, registry, start_date, end_date, state_store=None, data_provider=None):
    if data_provider is None: raise ValueError('a causal data provider is required')
    rh=registry_hash(registry); health_store=ExpertHealthStore([x.expert_id for x in registry], policy); ledger=InMemoryShadowLedger(data_provider.outcome)
    state=RouterStateV1('QBD_ROUTER_STATE_V1', start_date, policy.policy_hash, rh, None, (), {}, {x.expert_id: initial_lifecycle(x.expert_id,start_date) for x in registry}, dict(health_store.values), None, '', {})
    loaded = state_store.load() if state_store is not None else None
    if loaded is not None:
        if loaded.policy_hash != policy.policy_hash or loaded.registry_hash != rh: raise ValueError('resume fingerprint mismatch')
        state = loaded; health_store.values = dict(loaded.health)
        for raw in loaded.persisted_component_states.get('pending', ()):
            item = PendingOutcome(date.fromisoformat(raw['decision_date']), raw['expert_id'], date.fromisoformat(raw['outcome_available_at']), raw['opaque_outcome_key'], raw.get('regime_id'))
            ledger.pending[(item.decision_date, item.expert_id)] = item
        for raw in loaded.persisted_component_states.get('outcomes', ()):
            item = MaturedOutcome(date.fromisoformat(raw['decision_date']), raw['expert_id'], date.fromisoformat(raw['outcome_available_at']), raw['realized_return'], raw['benchmark_return'], raw['transaction_cost'], raw['net_excess'], raw.get('regime_id'))
            ledger.outcomes[(item.decision_date, item.expert_id)] = item
    router=QbdRouterV1(policy, registry, rh); decisions=[]
    for current in data_provider.dates(start_date,end_date):
        if loaded is not None and current <= loaded.as_of:
            continue
        matured=ledger.mature(current); cursor=ledger.cursor(); health=health_store.update(matured,current,cursor)
        state=RouterStateV1(state.schema_version,current,state.policy_hash,state.registry_hash,state.champion_id,state.candidate_members,state.current_weights,state.lifecycle,health,state.last_switch_date,cursor,state.persisted_component_states)
        shadows=[]
        for expert in registry:
            decision,pending=data_provider.shadow(current,expert); ledger.append_decision(decision); shadows.append(decision)
            if pending is not None: ledger.append_pending(pending)
        health=health_store.observe_activity(shadows,current)
        state=RouterStateV1(state.schema_version,current,state.policy_hash,state.registry_hash,state.champion_id,state.candidate_members,state.current_weights,state.lifecycle,health,state.last_switch_date,cursor,state.persisted_component_states)
        state,allocation=router.decide(current_date=current,previous_state=state,matured_outcomes=matured,shadow_decisions=shadows); decisions.append(allocation)
        if state_store is not None:
            state = RouterStateV1(state.schema_version,state.as_of,state.policy_hash,state.registry_hash,state.champion_id,state.candidate_members,state.current_weights,state.lifecycle,state.health,state.last_switch_date,state.matured_evidence_cursor,{'kills': state.persisted_component_states.get('kills', {}), 'pending': tuple(ledger.pending.values()), 'outcomes': tuple(ledger.outcomes.values())})
            state_store.save_atomic(state)
    manifest={'contract_id':policy.contract_id,'policy_hash':policy.policy_hash,'registry_hash':rh,'data_fingerprint':sha256_fingerprint((start_date,end_date)),'router_arm':policy.router_arm.value,'start_date':start_date,'end_date':end_date,'validation_status':'PSEUDO_OOS_DEVELOPMENT'}
    return ReplayResult(tuple(decisions),state,{'decision_count':len(decisions),'ledger_cursor':ledger.cursor()},manifest)
