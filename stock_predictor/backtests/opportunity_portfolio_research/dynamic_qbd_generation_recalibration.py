"""Generation-specific causal threshold calibration."""
from __future__ import annotations

from dataclasses import asdict
from datetime import date

import numpy as np
import pandas as pd

from .contract_fingerprints import stable_hash
from .dynamic_qbd_generation_contracts import CalibrationRecord, ExpertFamilySpec
from .dynamic_qbd_maturity import HorizonMaturityResolver


def _entry_candidates(family: ExpertFamilySpec, *, optimize_policy: bool = False) -> tuple[tuple[float, float], ...]:
    entry = family.entry_policy_rule
    if optimize_policy:
        design = entry.get("optional_policy_optimization_design", {})
        quantiles = design.get("score_quantiles", (entry.get("score_quantile", family.threshold_rule.get("score_quantile", 0.975)),))
        fractions = design.get("top_fractions", (entry.get("top_fraction", 0.005),))
    else:
        quantiles = (entry.get("score_quantile", family.threshold_rule.get("score_quantile", 0.975)),)
        fractions = (entry.get("top_fraction", 0.005),)
    return tuple((float(q), float(top)) for q in quantiles for top in fractions)


def _select_entry_policy(family: ExpertFamilySpec, frame: pd.DataFrame, *, optimize_policy: bool = False) -> dict:
    candidates = _entry_candidates(family, optimize_policy=optimize_policy)
    if len(candidates) == 1:
        return {"score_quantile": candidates[0][0], "top_fraction": candidates[0][1],
                "active_dates": 0, "month_count": 0, "positive_month_fraction": 0.0,
                "median_month_excess": 0.0, "q25_month_excess": 0.0}
    required = {"ticker", "observed_excess"}
    missing = required - set(frame)
    if missing:
        raise ValueError(f"CAUSAL_QBD_SELECTION_COLUMNS_MISSING:{sorted(missing)}")
    daily_top = frame.groupby(frame["decision_date"].dt.normalize())["score"].max().dropna()
    rows = []
    for quantile, top_fraction in candidates:
        threshold = float(np.quantile(daily_top.to_numpy(float), quantile))
        selected_daily = []
        active_dates = 0
        for decision_date, day in frame.groupby(frame["decision_date"].dt.normalize(), sort=True):
            eligible = day.loc[day["score"].ge(threshold)].sort_values(["score", "ticker"], ascending=[False, True])
            take = min(family.max_names, max(1, int(np.ceil(len(day) * top_fraction))))
            chosen = eligible.head(take)
            value = float(chosen["observed_excess"].mean()) if not chosen.empty else 0.0
            active_dates += int(not chosen.empty)
            selected_daily.append((pd.Timestamp(decision_date), value))
        monthly = pd.DataFrame(selected_daily, columns=["date", "excess"])
        monthly["month"] = monthly["date"].dt.to_period("M")
        month_values = monthly.groupby("month")["excess"].mean().to_numpy(float)
        rows.append({
            "score_quantile": quantile, "top_fraction": top_fraction, "threshold": threshold,
            "active_dates": active_dates, "month_count": int(len(month_values)),
            "positive_month_fraction": float(np.mean(month_values > 0)) if len(month_values) else 0.0,
            "median_month_excess": float(np.median(month_values)) if len(month_values) else -np.inf,
            "q25_month_excess": float(np.quantile(month_values, .25)) if len(month_values) else -np.inf,
            "worst_month_excess": float(np.min(month_values)) if len(month_values) else -np.inf,
        })
    configured_minimum = int(family.entry_policy_rule.get("minimum_active_dates", 8))
    history_scaled_minimum = max(3, int(np.ceil(len(daily_top) * .10)))
    minimum_active = min(configured_minimum, history_scaled_minimum)
    eligible = [x for x in rows if x["active_dates"] >= minimum_active]
    if not eligible:
        raise ValueError(f"NO_QBD_ENTRY_POLICY_WITH_MINIMUM_ACTIVE_DATES:{minimum_active}")
    return max(eligible, key=lambda x: (x["median_month_excess"], x["q25_month_excess"],
                                    x["positive_month_fraction"], x["worst_month_excess"],
                                    -x["active_dates"], x["score_quantile"], -x["top_fraction"]))


def recalibrate_generation(
    family: ExpertFamilySpec,
    generation_id: str,
    predictions: pd.DataFrame,
    *,
    information_cutoff: date,
    maturity: HorizonMaturityResolver,
    optimize_policy: bool = False,
) -> CalibrationRecord:
    required = {"decision_date", "score", "terminal_date"}
    missing = required - set(predictions)
    if missing:
        raise ValueError(f"CALIBRATION_COLUMNS_MISSING:{sorted(missing)}")
    frame = predictions.copy()
    frame["decision_date"] = pd.to_datetime(frame["decision_date"])
    frame["terminal_date"] = pd.to_datetime(frame["terminal_date"])
    frame = maturity.filter_matured(frame, information_cutoff)
    frame = frame.loc[frame["decision_date"].le(pd.Timestamp(information_cutoff))]
    if frame.empty:
        raise ValueError("NO_MATURED_CALIBRATION_OBSERVATIONS")
    assessment_dates = sorted(frame["decision_date"].dt.normalize().unique())
    keep = set(assessment_dates[-int(family.calibration_window_sessions):])
    frame = frame.loc[frame["decision_date"].dt.normalize().isin(keep)]
    maturity.assert_matured(frame, information_cutoff)
    daily_top = frame.groupby(frame["decision_date"].dt.normalize())["score"].max().dropna()
    if daily_top.empty:
        raise ValueError("NO_FINITE_DAILY_TOP_SCORES")
    selected_policy = _select_entry_policy(family, frame, optimize_policy=optimize_policy)
    quantile = float(selected_policy["score_quantile"])
    threshold = float(np.quantile(daily_top.to_numpy(float), quantile))
    calibration_start = pd.Timestamp(frame["decision_date"].min()).date()
    calibration_end = pd.Timestamp(frame["decision_date"].max()).date()
    terminal = pd.Timestamp(frame["terminal_date"].max()).date()
    identity = {
        "family_hash": family.family_hash,
        "generation_id": generation_id,
        "information_cutoff": information_cutoff,
        "calibration_start": calibration_start,
        "calibration_end": calibration_end,
        "latest_terminal_date_used": terminal,
        "score_quantile": quantile,
        "resolved_threshold": threshold,
        "daily_top_scores": tuple(float(x) for x in daily_top),
        "resolved_top_fraction": float(selected_policy["top_fraction"]),
        "entry_selection_statistics": selected_policy,
        "policy_optimization_mode": "ROLLING_Q_TOP" if optimize_policy else "FROZEN_Q_TOP",
    }
    return CalibrationRecord(
        family.family_id, generation_id, information_cutoff, calibration_start, calibration_end,
        terminal, quantile, threshold, int(len(daily_top)), stable_hash(identity),
        float(selected_policy["top_fraction"]), int(selected_policy["active_dates"]),
        int(selected_policy["month_count"]), float(selected_policy["positive_month_fraction"]),
        float(selected_policy["median_month_excess"]), float(selected_policy["q25_month_excess"]),
    )


def calibration_payload(record: CalibrationRecord) -> dict:
    return asdict(record)
