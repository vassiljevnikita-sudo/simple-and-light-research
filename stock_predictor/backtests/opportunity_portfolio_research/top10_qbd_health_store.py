from __future__ import annotations
from datetime import date
from dataclasses import replace
from collections import defaultdict
from .top10_qbd_router_contracts import ExpertType, HealthLevel, ActivityLevel
from .top10_qbd_health_metrics import *
from .top10_qbd_shadow_ledger import MaturedOutcome
from .top10_qbd_activity_monitor import update_activity
from .top10_qbd_regime_health import shrink_regime
from .top10_qbd_changepoint import update_changepoint

class ExpertHealthStore:
    def __init__(self, expert_ids, policy):
        self.policy = policy; self.values = {}
        for eid in expert_ids:
            today = date.min
            self.values[eid] = ExpertHealthSnapshot(eid, today,
                StructuralHealth(HealthLevel.PASS, (), today),
                ActivityHealth(ActivityLevel.NORMAL, 1.0, ActivityPosterior(policy.activity_prior_a, policy.activity_prior_b, 0, 0, 0, None), today),
                AlphaHealth(0.0, None, None, 0.0, 0.0, None, today),
                RegimeScore("GLOBAL", None, 0.0, 0.0, 0.0, 0.0), 1.0, None, "")
        self._seen = set()
    def observe_activity(self, shadows, at: date) -> dict[str, ExpertHealthSnapshot]:
        """Record opportunity/trade observations without touching alpha evidence."""
        for shadow in shadows:
            old = self.values[shadow.expert_id]
            activity = update_activity(old.activity.posterior, eligible_opportunity=shadow.eligible_opportunity,
                                       traded=shadow.would_enter, current_date=at,
                                       watch=self.policy.activity_watch_zero_prob,
                                       anomalous=self.policy.activity_anomalous_zero_prob,
                                       critical=self.policy.activity_critical_zero_prob)
            self.values[shadow.expert_id] = replace(old, as_of=at, activity=activity)
            if shadow.prediction_uncertainty is not None:
                self.values[shadow.expert_id] = replace(self.values[shadow.expert_id], uncertainty_score=float(shadow.prediction_uncertainty))
        return dict(self.values)
    def update(self, outcomes, at: date, cursor: str) -> dict[str, ExpertHealthSnapshot]:
        grouped = defaultdict(list)
        for outcome in outcomes:
            key = (outcome.decision_date, outcome.expert_id)
            if key in self._seen: continue
            self._seen.add(key); grouped[outcome.expert_id].append(outcome)
        for eid, vals in grouped.items():
            old = self.values[eid]
            cp_discount = old.changepoint.forgetting_multiplier if old.changepoint else 1.0
            alpha = alpha_from_outcomes(old.alpha, tuple(vals), at, self.policy.alpha_discount * cp_discount)
            cp = old.changepoint
            for x in vals:
                cp = update_changepoint(cp or ChangePointState(0.0, 0.0, 1.0, at, {}), x.net_excess, self.policy.changepoint_config, at)
            self.values[eid] = replace(old, as_of=at, alpha=alpha,
                regime=shrink_regime(regime_id="GLOBAL", local_score=None, global_score=alpha.discounted_net_excess, effective_local_n=0.0, k=self.policy.regime_shrinkage_k),
                uncertainty_score=1.0 / max(1.0, alpha.effective_n) ** .5, changepoint=cp, matured_evidence_cursor=cursor)
        return dict(self.values)
