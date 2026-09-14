"""Fidelity runtime for the frozen R01-R10 entry policies."""
from __future__ import annotations
from dataclasses import dataclass
from datetime import date
from typing import Any, Protocol, Sequence
import math
from .top10_qbd_expert_registry import ExpertSpec

@dataclass(frozen=True)
class CandidateDecision:
    ticker: str
    score: float
    horizon: int

@dataclass(frozen=True)
class ShadowPositionIntent:
    ticker: str
    score: float
    allocation_fraction: float
    entry_policy_id: str
    exit_policy_id: str

@dataclass(frozen=True)
class ExpertShadowDecision:
    expert_id: str
    decision_date: date
    candidates: tuple[CandidateDecision, ...]
    selected_positions: tuple[ShadowPositionIntent, ...]
    eligible_opportunity: bool
    allocation_fraction: float
    market_regime: str
    prediction_uncertainty: float | None = None

class FrozenExpertRuntime(Protocol):
    def decision(self, *, decision_date: date, available_predictions: Sequence[Any], market_state: Any = None, v4_state: Any = None, open_tickers: Sequence[str] = ()) -> ExpertShadowDecision: ...

class FrozenExpertRuntimeImpl:
    def __init__(self, spec: ExpertSpec):
        if spec.expert_type.value != "STOCK" or spec.entry_policy is None: raise ValueError(f"missing frozen entry policy: {spec.expert_id}")
        self.spec=spec
    def decision(self, *, decision_date: date, available_predictions: Sequence[Any], market_state: Any = None, v4_state: Any = None, open_tickers: Sequence[str] = ()) -> ExpertShadowDecision:
        policy=self.spec.entry_policy; rows=[]
        for row in available_predictions:
            ticker=str(row["ticker"] if isinstance(row, dict) else getattr(row,"ticker"))
            score=float(row["predicted_net_excess_return"] if isinstance(row, dict) else getattr(row,"predicted_net_excess_return"))
            horizon=int(row["horizon_sessions"] if isinstance(row, dict) else getattr(row,"horizon_sessions"))
            if horizon == self.spec.horizon and score >= policy.resolved_threshold:
                rows.append(CandidateDecision(ticker,score,horizon))
        rows=sorted(rows,key=lambda x:(-x.score,x.ticker))
        # The frozen policy means ceil(group_size * top_fraction), followed
        # by the threshold and concurrent max-name gates.  `group_size` is
        # the complete horizon/date group, not a pre-truncated top-N view.
        group_size = max(1, len(available_predictions))
        fraction_limit = max(1, int(math.ceil(group_size * policy.top_fraction)))
        passing = rows[:fraction_limit]
        held = {str(x) for x in open_tickers}
        free_slots = max(0, int(policy.max_names) - len(held))
        selected=tuple(x for x in passing if x.ticker not in held)[:free_slots]
        fraction=1.0/len(selected) if selected else 0.0
        intents=tuple(ShadowPositionIntent(x.ticker,x.score,fraction,self.spec.entry_policy.policy_id,self.spec.exit_policy_id) for x in selected)
        return ExpertShadowDecision(self.spec.expert_id,decision_date,tuple(rows),intents,bool(selected),1.0 if selected else 0.0,getattr(market_state,"regime_id","GLOBAL"))

def runtime_for_registry(registry):
    return {spec.expert_id: FrozenExpertRuntimeImpl(spec) for spec in registry if spec.expert_type.value == "STOCK"}
