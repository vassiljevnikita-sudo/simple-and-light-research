"""Causal fixed and learned-exit adapters for expert shadow positions."""
from __future__ import annotations
from dataclasses import dataclass
from datetime import date
from typing import Protocol

@dataclass(frozen=True)
class ExitDecision:
    should_exit: bool
    execution_date: date | None
    reason_code: str
    remaining_sessions: int

class ExpertExitRuntime(Protocol):
    def evaluate(self, *, ticker: str, current_date: date, remaining_sessions: int) -> ExitDecision: ...

class FixedHorizonExitRuntime:
    def evaluate(self, *, ticker: str, current_date: date, remaining_sessions: int) -> ExitDecision:
        return ExitDecision(remaining_sessions <= 0, current_date if remaining_sessions <= 0 else None, "FIXED_HORIZON", remaining_sessions)

class LearnedExitRuntime:
    def __init__(self, provider): self.provider=provider
    def evaluate(self, *, ticker: str, current_date: date, remaining_sessions: int) -> ExitDecision:
        prediction=self.provider.prediction(ticker,current_date,remaining_sessions)
        if prediction is not None and prediction <= 0:
            return ExitDecision(True,None,"LEARNED_CONTINUATION_NONPOSITIVE_NEXT_OPEN",remaining_sessions)
        return ExitDecision(remaining_sessions <= 0,current_date if remaining_sessions <= 0 else None,"LEARNED_CONTINUATION_POSITIVE",remaining_sessions)
