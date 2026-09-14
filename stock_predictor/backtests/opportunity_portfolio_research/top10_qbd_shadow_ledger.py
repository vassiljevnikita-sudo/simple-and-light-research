"""In-memory causal ledger with exact maturity and duplicate protection."""
from __future__ import annotations
from dataclasses import dataclass
from datetime import date
from typing import Callable, Sequence
from .top10_qbd_router_contracts import sha256_fingerprint

@dataclass(frozen=True)
class ShadowDecision:
    decision_date: date; expert_id: str; eligible_opportunity: bool; prediction: float | None
    threshold: float | None; would_enter: bool; would_exit: bool; market_regime: str | None
    market_state_features: dict[str, float]; prediction_uncertainty: float | None; outcome_available_at: date | None
    candidates: tuple[object, ...] = (); selected_positions: tuple[object, ...] = (); allocation_fraction: float = 0.0

@dataclass(frozen=True)
class PendingOutcome:
    decision_date: date; expert_id: str; outcome_available_at: date; opaque_outcome_key: str
    regime_id: str | None = None

@dataclass(frozen=True)
class MaturedOutcome:
    decision_date: date; expert_id: str; outcome_available_at: date; realized_return: float
    benchmark_return: float; transaction_cost: float; net_excess: float; regime_id: str | None = None

class InMemoryShadowLedger:
    def __init__(self, resolver: Callable[[PendingOutcome], MaturedOutcome] | None = None) -> None:
        self.decisions: dict[tuple[date, str], ShadowDecision] = {}
        self.pending: dict[tuple[date, str], PendingOutcome] = {}
        self.outcomes: dict[tuple[date, str], MaturedOutcome] = {}
        self._resolver = resolver
    def append_decision(self, decision: ShadowDecision) -> None:
        key = (decision.decision_date, decision.expert_id)
        if key in self.decisions and self.decisions[key] != decision: raise ValueError("duplicate decision mismatch")
        self.decisions[key] = decision
    def append_pending(self, pending: PendingOutcome) -> None:
        key = (pending.decision_date, pending.expert_id)
        if key in self.pending and self.pending[key] != pending: raise ValueError("duplicate pending mismatch")
        self.pending[key] = pending
    def mature(self, current_date: date) -> tuple[MaturedOutcome, ...]:
        fresh = []
        for key, pending in sorted(self.pending.items()):
            if pending.outcome_available_at <= current_date and key not in self.outcomes:
                if self._resolver is None: continue
                outcome = self._resolver(pending)
                if outcome is None:
                    continue
                if outcome.outcome_available_at != pending.outcome_available_at: raise ValueError("maturity mismatch")
                self.outcomes[key] = outcome; fresh.append(outcome)
        return tuple(fresh)
    def matured_since(self, cursor: str | None) -> tuple[MaturedOutcome, ...]:
        values = tuple(self.outcomes[k] for k in sorted(self.outcomes))
        if cursor is None: return values
        return tuple(x for x in values if (x.decision_date.isoformat() + "::" + x.expert_id) > cursor)
    def cursor(self) -> str:
        return max((x.decision_date.isoformat() + "::" + x.expert_id for x in self.outcomes.values()), default="")
    def replay_fingerprint(self) -> str:
        return sha256_fingerprint(tuple(self.decisions.values()))
