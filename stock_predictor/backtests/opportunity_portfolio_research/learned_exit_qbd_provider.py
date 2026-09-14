from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import pandas as pd

REQUIRED = {
    "decision_date", "ticker", "exit_horizon_sessions",
    "predicted_continuation_excess", "holdout_locked",
}

@dataclass(frozen=True)
class LearnedExitAudit:
    rows: int
    horizons: tuple[int, ...]
    locked_rows: int
    duplicate_keys: int
    min_date: str
    max_date: str

class LearnedExitProvider:
    def __init__(self, path: Path):
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(path)
        frame = pd.read_parquet(path)
        missing = sorted(REQUIRED - set(frame.columns))
        if missing:
            raise ValueError(f"LEARNED_EXIT_COLUMNS_MISSING:{missing}")
        generation_aware = "exit_generation_id" in frame.columns
        selected_columns = list(REQUIRED) + (["exit_generation_id"] if generation_aware else [])
        frame = frame[selected_columns].copy()
        frame["decision_date"] = pd.to_datetime(frame["decision_date"])
        frame["ticker"] = frame["ticker"].astype(str)
        frame["exit_horizon_sessions"] = frame["exit_horizon_sessions"].astype(int)
        locked = int(frame["holdout_locked"].fillna(False).astype(bool).sum())
        if locked:
            raise AssertionError(f"FROZEN_VALIDATION_EXIT_ROWS_FORBIDDEN:{locked}")
        bad_h = sorted(set(frame["exit_horizon_sessions"]) - set(range(1,31)))
        if bad_h:
            raise ValueError(f"EXIT_HORIZON_OUT_OF_CONTRACT:{bad_h}")
        key_cols = ["ticker","decision_date","exit_horizon_sessions"] + (["exit_generation_id"] if generation_aware else [])
        dup = int(frame.duplicated(key_cols, keep=False).sum())
        if dup:
            raise ValueError(f"LEARNED_EXIT_DUPLICATE_KEYS:{dup}")
        self.generation_aware = generation_aware
        self._lookup = {
            ((str(r.exit_generation_id),) if generation_aware else ()) +
            (r.ticker, pd.Timestamp(r.decision_date), int(r.exit_horizon_sessions)): float(r.predicted_continuation_excess)
            for r in frame.itertuples(index=False)
            if math.isfinite(float(r.predicted_continuation_excess))
        }
        self.audit = LearnedExitAudit(
            rows=len(frame), horizons=tuple(sorted(frame.exit_horizon_sessions.unique().tolist())),
            locked_rows=locked, duplicate_keys=dup,
            min_date=str(frame.decision_date.min().date()), max_date=str(frame.decision_date.max().date()),
        )

    def prediction(self, ticker: str, decision_date, remaining_sessions: int, exit_generation_id: str | None = None) -> float | None:
        if remaining_sessions <= 0:
            return None
        prefix = (str(exit_generation_id),) if self.generation_aware and exit_generation_id is not None else ()
        if self.generation_aware and not prefix:
            return None
        return self._lookup.get(prefix + (str(ticker), pd.Timestamp(decision_date), int(remaining_sessions)))
