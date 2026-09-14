"""Production-faithful family shadow replay through existing accounting engines."""
from __future__ import annotations

import pandas as pd

from .cost_contracts import CostModel
from .portfolio_policy_contracts import Policy
from .tax_contracts import TaxConfig
from .next_open_portfolio_replay import replay as fixed_replay
from .learned_exit_qbd_replay import replay as learned_replay


def _threshold_series(schedule: pd.DataFrame, dates) -> dict:
    schedule = schedule.copy().sort_values("activation_date")
    schedule["activation_date"] = pd.to_datetime(schedule["activation_date"])
    result = {}
    for d in pd.to_datetime(tuple(dates)):
        active = schedule.loc[schedule["activation_date"].le(d)]
        if not active.empty:
            result[pd.Timestamp(d).normalize()] = float(active.iloc[-1]["resolved_threshold"])
    return result


def replay_family(
    *, signals: pd.DataFrame, prices: pd.DataFrame, policy: Policy, generation_schedule: pd.DataFrame,
    cost: CostModel, tax: TaxConfig, start=None, end=None, initial: float = 10000.0,
    resume_state: dict | None = None,
    distributions: pd.DataFrame | None = None,
    authoritative_signals: pd.DataFrame | None = None,
) -> dict:
    if generation_schedule.empty:
        raise ValueError("FAMILY_REPLAY_REQUIRES_VALID_GENERATION")
    schedule = generation_schedule.copy()
    required = {"activation_date", "family_id", "generation_id", "resolved_threshold", "entry_policy_id", "exit_policy_id"}
    missing = required - set(schedule)
    if missing:
        raise ValueError(f"GENERATION_SCHEDULE_COLUMNS_MISSING:{sorted(missing)}")
    signals = (authoritative_signals if authoritative_signals is not None else
               _generation_authoritative_signals(signals, schedule))
    if policy.exit_family == "LEARNED_EXIT":
        if distributions is not None and not distributions.empty:
            raise ValueError("LEARNED_EXIT_DISTRIBUTION_LEDGER_NOT_IMPLEMENTED")
        dates = prices.loc[prices["ticker"].eq("URTH"), "date"].sort_values().unique()
        result = learned_replay(signals, prices, policy, cost, tax, start=start, end=end, initial=initial,
                                resolved_threshold_by_date=_threshold_series(schedule, dates), generation_schedule=schedule,
                                resume_state=resume_state)
        return result
    return fixed_replay(signals, prices, policy, cost, tax, start=start, end=end, initial=initial,
                        generation_schedule=schedule,resume_state=resume_state,distributions=distributions)


def _generation_authoritative_signals(signals: pd.DataFrame, schedule: pd.DataFrame) -> pd.DataFrame:
    """Stitch only scores produced by the model generation active that day."""
    required = {"decision_date", "ticker", "score", "model_artifact_id"}
    missing = required - set(signals)
    if missing:
        raise ValueError(f"GENERATION_TAGGED_SIGNAL_COLUMNS_MISSING:{sorted(missing)}")
    source = signals.copy()
    source["decision_date"] = pd.to_datetime(source["decision_date"]).dt.normalize().astype("datetime64[ns]")
    source["model_artifact_id"] = source["model_artifact_id"].astype(str)
    if source.empty:
        return source[["decision_date","ticker","score"]].copy()
    timeline = schedule[["activation_date", "model_artifact_id"]].copy()
    timeline["activation_date"] = pd.to_datetime(timeline["activation_date"]).dt.normalize().astype("datetime64[ns]")
    timeline["model_artifact_id"] = timeline["model_artifact_id"].astype(str)
    timeline = timeline.sort_values("activation_date").drop_duplicates("activation_date", keep="last")
    decisions = pd.DataFrame({"decision_date": sorted(source["decision_date"].unique())})
    authority = pd.merge_asof(decisions, timeline, left_on="decision_date", right_on="activation_date",
                              direction="backward").dropna(subset=["model_artifact_id"])
    stitched = source.merge(authority[["decision_date", "model_artifact_id"]],
                            on=["decision_date", "model_artifact_id"], how="inner")
    if stitched.duplicated(["decision_date", "ticker"]).any():
        raise ValueError("MULTIPLE_AUTHORITATIVE_GENERATION_SCORES")
    return stitched[["decision_date", "ticker", "score"]].sort_values(
        ["decision_date", "ticker"]
    ).reset_index(drop=True)


def generation_authoritative_signals(signals: pd.DataFrame, schedule: pd.DataFrame) -> pd.DataFrame:
    """Public wrapper for bounded replay batching across equivalent schedules."""
    return _generation_authoritative_signals(signals, schedule)


def replay_abc(*, signals, prices, policy, schedules, cost, tax, start=None, end=None, initial=10000.0):
    results = {}
    for arm, schedule in schedules.items():
        results[str(arm)] = replay_family(signals=signals, prices=prices, policy=policy,
                                          generation_schedule=schedule, cost=cost, tax=tax,
                                          start=start, end=end, initial=initial)
    return results
