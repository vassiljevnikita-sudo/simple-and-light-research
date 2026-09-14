from __future__ import annotations
import json, os, time
from dataclasses import asdict
from pathlib import Path
from .top10_qbd_router_contracts import canonical_json, LifecycleState, HealthLevel, ActivityLevel
from .top10_qbd_state import RouterStateV1, ExpertLifecycle
from .top10_qbd_health_metrics import (ExpertHealthSnapshot, StructuralHealth, ActivityHealth, ActivityPosterior,
    AlphaHealth, RegimeScore, ChangePointState)
from datetime import date

def _d(value): return date.fromisoformat(value) if value else None
def _health(raw):
    return ExpertHealthSnapshot(raw['expert_id'], _d(raw['as_of']),
        StructuralHealth(HealthLevel(raw['structural']['level']), tuple(raw['structural']['reason_codes']), _d(raw['structural']['checked_at'])),
        ActivityHealth(ActivityLevel(raw['activity']['level']), raw['activity']['zero_trade_probability'],
            ActivityPosterior(**{**raw['activity']['posterior'], 'last_trade_date': _d(raw['activity']['posterior']['last_trade_date'])}), _d(raw['activity']['evaluated_at'])),
        AlphaHealth(**{**raw['alpha'], 'evaluated_at': _d(raw['alpha']['evaluated_at'])}),
        RegimeScore(**raw['regime']),
        raw['uncertainty_score'],
        ChangePointState(**{**raw['changepoint'], 'evaluated_at': _d(raw['changepoint']['evaluated_at'])}) if raw.get('changepoint') else None,
        raw['matured_evidence_cursor'])

class JsonRouterStateStore:
    def __init__(self, path, *, policy_hash, registry_hash):
        self.path=Path(path); self.policy_hash=policy_hash; self.registry_hash=registry_hash
    def save_atomic(self, state):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp=self.path.with_name(self.path.name + f'.{os.getpid()}.tmp')
        temp.write_text(canonical_json(asdict(state)), encoding='utf-8')
        last=None
        for _ in range(20):
            try:
                os.replace(temp, self.path); return
            except PermissionError as exc:
                last=exc; time.sleep(.10)
        raise last
    def load(self):
        if not self.path.exists(): return None
        raw=json.loads(self.path.read_text(encoding='utf-8'))
        if raw.get('policy_hash') != self.policy_hash or raw.get('registry_hash') != self.registry_hash or raw.get('schema_version') != 'QBD_ROUTER_STATE_V1': raise ValueError('incompatible router state')
        lifecycle={eid: ExpertLifecycle(eid, LifecycleState(v['state']), _d(v['entered_state_at']), LifecycleState(v['prior_state']) if v.get('prior_state') else None, v['reason_code'], _d(v['evidence_date']), int(v.get('sessions_in_state',0))) for eid,v in raw['lifecycle'].items()}
        health={eid: _health(v) for eid,v in raw['health'].items()}
        return RouterStateV1(raw['schema_version'], _d(raw['as_of']), raw['policy_hash'], raw['registry_hash'], raw.get('champion_id'), tuple(raw['candidate_members']), dict(raw['current_weights']), lifecycle, health, _d(raw.get('last_switch_date')), raw['matured_evidence_cursor'], dict(raw.get('persisted_component_states') or {}))
