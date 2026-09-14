"""Horizon-specific, session-aware maturity barrier."""
from __future__ import annotations

from bisect import bisect_right
from datetime import date
from typing import Iterable

import pandas as pd


class HorizonMaturityResolver:
    def __init__(self, sessions: Iterable[object]):
        values = tuple(sorted({pd.Timestamp(x).date() for x in sessions}))
        if not values:
            raise ValueError("MATURITY_RESOLVER_REQUIRES_SESSIONS")
        self.sessions = values
        self._index = {value: i for i, value in enumerate(values)}

    def terminal_date(self, decision_date: object, horizon_sessions: int) -> date:
        decision = pd.Timestamp(decision_date).date()
        if decision not in self._index:
            raise ValueError(f"DECISION_DATE_NOT_A_SESSION:{decision}")
        target = self._index[decision] + int(horizon_sessions)
        if target >= len(self.sessions):
            raise ValueError(f"OUTCOME_NOT_YET_TERMINAL:{decision}:H{horizon_sessions}")
        return self.sessions[target]

    def latest_matured_decision(self, information_cutoff: object, horizon_sessions: int) -> date | None:
        cutoff = pd.Timestamp(information_cutoff).date()
        terminal_index = bisect_right(self.sessions, cutoff) - 1
        decision_index = terminal_index - int(horizon_sessions)
        return self.sessions[decision_index] if decision_index >= 0 else None

    def filter_matured(self, frame: pd.DataFrame, information_cutoff: object, *, terminal_column: str = "terminal_date") -> pd.DataFrame:
        cutoff = pd.Timestamp(information_cutoff)
        if terminal_column not in frame:
            raise ValueError(f"MISSING_TERMINAL_COLUMN:{terminal_column}")
        terminal = pd.to_datetime(frame[terminal_column])
        return frame.loc[terminal.le(cutoff)].copy()

    @staticmethod
    def assert_matured(frame: pd.DataFrame, information_cutoff: object, *, terminal_column: str = "terminal_date") -> None:
        cutoff = pd.Timestamp(information_cutoff)
        if terminal_column not in frame or pd.to_datetime(frame[terminal_column]).gt(cutoff).any():
            raise AssertionError("UNMATURED_OUTCOME_CONSUMPTION")
