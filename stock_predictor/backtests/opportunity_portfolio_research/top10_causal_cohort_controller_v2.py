"""Causal cohort-attached monthly controller for frozen Top-10 entry policies.

Contract: TOP10_CAUSAL_COHORT_CONTROLLER_V2

This is a DEVELOPMENT replay, not a fresh independent OOS test: V2 was designed
after inspecting V1 results from 2023-2026. The final frozen holdout remains closed.

Warm-up: 2020-09-01 through 2023-08-10 using historical WF_001..WF_007 OOS
predictions. Evaluation: 2023-08-11 through 2026-07-24 using the already completed
causal-expanding entry/exit prediction artifacts (~17k checkpoints).

Unlike V1, weights are attached to fixed 21-session cohorts. A cohort keeps its
weight while aging from M1 to M12. Quality uses magnitude-aware robust z-scores
and evidence confidence. No model fit, prediction generation, threshold search,
weight-grid search, or final-holdout access occurs.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from .cost_contracts import CostModel
from .tax_contracts import TaxConfig
from .portfolio_research_inputs import load_price_panel
from .learned_exit_qbd_profit import ProfitTaxConfig
from .learned_exit_qbd_provider import LearnedExitProvider
from .learned_exit_qbd_replay import configure_replay, replay
from .next_open_portfolio_replay import prepare_prices
from .top10_causal_expanding_portfolio import LIVE_END, LIVE_START, _policy, _thresholds
from .top10_entry_activation_alpha_diagnostic import _causal_entry, _contracts, _markdown, _write_json
from .top10_causal_monthly_weight_controller import (
    WARMUP_START,
    MONTH_SESSIONS,
    EXPERT_COUNT,
    _historical_entry,
    _combined_entry,
    _historical_pressure,
    _prior_suite_gates,
    _assessment_dates,
    _prepare_model_rows,
)

CONTRACT_ID = "TOP10_CAUSAL_COHORT_CONTROLLER_V2"
INITIAL_WEIGHT = 1.0 / EXPERT_COUNT
WEIGHT_FLOOR = 0.01
WEIGHT_CEILING = 0.35
MAX_ABS_WEIGHT_CHANGE_PER_ASSESSMENT = 0.05
HEDGE_ETA = 1.0
ROBUST_Z_CLIP = 2.0
QUALITY_RETURN_CLIP = 0.50
MIN_CANDIDATES_PER_COHORT = 15
CONFIDENCE_TARGET_CAP = 30
DEVELOPMENT_REUSE_NOTICE = (
    "2023-2026 was already inspected while designing V2; results are development evidence, "
    "not a fresh independent OOS claim."
)


def _validate_contract() -> None:
    assert EXPERT_COUNT == 12
    assert MONTH_SESSIONS == 21
    assert abs(INITIAL_WEIGHT * EXPERT_COUNT - 1.0) < 1e-12
    assert 0 <= WEIGHT_FLOOR < INITIAL_WEIGHT < WEIGHT_CEILING <= 1
    assert 0 < MAX_ABS_WEIGHT_CHANGE_PER_ASSESSMENT <= 0.05
    assert WEIGHT_FLOOR * EXPERT_COUNT < 1.0 < WEIGHT_CEILING * EXPERT_COUNT
    assert WARMUP_START < LIVE_START <= LIVE_END


def _v1_gate(summary_path: Path, results_path: Path) -> pd.DataFrame:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    if (
        payload.get("status") != "COMPLETE"
        or payload.get("no_model_training") is not True
        or payload.get("no_prediction_generation") is not True
        or payload.get("no_threshold_optimization") is not True
        or payload.get("no_weight_optimization") is not True
        or payload.get("final_holdout_opened")
    ):
        raise RuntimeError("COHORT_V2_V1_REFERENCE_GATE_FAILED")
    frame = pd.read_csv(results_path)
    ref = frame.loc[frame["arm"].eq("CONTROLLER_12M")].copy()
    if len(ref) != 10:
        raise RuntimeError(f"COHORT_V2_V1_REFERENCE_MODEL_COUNT:{len(ref)}")
    return ref


def _fixed_cohorts(combined: pd.DataFrame) -> list[dict]:
    dates = [
        pd.Timestamp(x)
        for x in sorted(
            combined.loc[
                combined["decision_date"].between(WARMUP_START, LIVE_END), "decision_date"
            ].drop_duplicates()
        )
    ]
    cohorts = []
    cohort_no = 1
    for start in range(0, len(dates) - MONTH_SESSIONS + 1, MONTH_SESSIONS):
        block = dates[start : start + MONTH_SESSIONS]
        if len(block) != MONTH_SESSIONS:
            continue
        cohorts.append(
            {
                "cohort_id": f"C{cohort_no:03d}_{block[0].date()}_{block[-1].date()}",
                "cohort_no": cohort_no,
                "dates": tuple(block),
                "window_start": block[0],
                "window_end": block[-1],
            }
        )
        cohort_no += 1
    if len(cohorts) < EXPERT_COUNT:
        raise RuntimeError(f"COHORT_V2_INSUFFICIENT_FIXED_COHORTS:{len(cohorts)}")
    return cohorts


def _one_cohort_evidence(
    prepared: pd.DataFrame,
    cohort: dict,
    assessment_date: pd.Timestamp,
    *,
    max_names: int,
) -> dict:
    dates = set(cohort["dates"])
    window = prepared.loc[prepared["decision_date"].isin(dates)].copy()
    decision_days = int(window["decision_date"].nunique())
    candidates_all = window.loc[window["relative_candidate"]].copy()

    horizon_mature = (
        decision_days == MONTH_SESSIONS
        and len(window)
        and window["terminal_date"].notna().all()
        and pd.Timestamp(window["terminal_date"].max()) <= assessment_date
    )
    matured = candidates_all.loc[
        candidates_all["realized_price_available"]
        & candidates_all["terminal_date"].le(assessment_date)
        & candidates_all["realized_net_excess_20bps"].notna()
    ].copy()
    clipped = matured["realized_net_excess_20bps"].clip(
        lower=-QUALITY_RETURN_CLIP, upper=QUALITY_RETURN_CLIP
    )
    daily_p99 = window.groupby("decision_date")["score"].quantile(0.99)
    p99 = float(daily_p99.median()) if len(daily_p99) else np.nan
    target_candidates = max(
        1, min(CONFIDENCE_TARGET_CAP, MONTH_SESSIONS * max(1, int(max_names)))
    )
    confidence = (
        min(1.0, float(np.sqrt(len(matured) / target_candidates))) if len(matured) else 0.0
    )
    available = bool(
        horizon_mature
        and len(matured) >= MIN_CANDIDATES_PER_COHORT
        and np.isfinite(p99)
        and p99 > 0
    )
    return {
        "cohort_id": cohort["cohort_id"],
        "cohort_no": int(cohort["cohort_no"]),
        "window_start": cohort["window_start"],
        "window_end": cohort["window_end"],
        "available": available,
        "decision_days": decision_days,
        "candidate_count_total": int(len(candidates_all)),
        "candidate_count_realized": int(len(matured)),
        "quality_mean_excess_clipped": float(clipped.mean()) if len(clipped) else np.nan,
        "raw_mean_excess": (
            float(matured["realized_net_excess_20bps"].mean()) if len(matured) else np.nan
        ),
        "median_excess": (
            float(matured["realized_net_excess_20bps"].median()) if len(matured) else np.nan
        ),
        "hit_rate": (
            float(matured["realized_net_excess_20bps"].gt(0).mean()) if len(matured) else np.nan
        ),
        "confidence": float(confidence),
        "confidence_target_candidates": int(target_candidates),
        "median_p99": p99,
        "latest_terminal_date_used": (
            pd.Timestamp(matured["terminal_date"].max()) if len(matured) else pd.NaT
        ),
    }


def _current_cohorts(
    prepared: pd.DataFrame,
    cohorts: list[dict],
    assessment_date: pd.Timestamp,
    *,
    max_names: int,
) -> list[dict]:
    evidence = [
        _one_cohort_evidence(prepared, cohort, assessment_date, max_names=max_names)
        for cohort in cohorts
        if cohort["window_end"] < assessment_date
    ]
    available = [row for row in evidence if row["available"]]
    available.sort(key=lambda row: int(row["cohort_no"]))
    return available[-EXPERT_COUNT:]


def _robust_signals(snapshot: list[dict]) -> tuple[np.ndarray, dict]:
    if len(snapshot) != EXPERT_COUNT:
        raise RuntimeError("COHORT_V2_SNAPSHOT_LENGTH")
    qualities = np.array(
        [float(row["quality_mean_excess_clipped"]) for row in snapshot], dtype=float
    )
    if not np.isfinite(qualities).all():
        raise RuntimeError("COHORT_V2_QUALITY_NAN")
    median = float(np.median(qualities))
    mad = float(np.median(np.abs(qualities - median)))
    scale = 1.4826 * mad
    fallback = False
    if not np.isfinite(scale) or scale < 1e-8:
        scale = float(np.std(qualities))
        fallback = True
    if not np.isfinite(scale) or scale < 1e-8:
        z = np.zeros(EXPERT_COUNT, dtype=float)
        scale = 0.0
    else:
        z = np.clip((qualities - median) / scale, -ROBUST_Z_CLIP, ROBUST_Z_CLIP)
    confidence = np.array([float(row["confidence"]) for row in snapshot], dtype=float)
    signal = z * confidence
    return signal, {
        "quality_median": median,
        "quality_scale": scale,
        "quality_scale_fallback_std": fallback,
        "robust_z": z,
        "confidence": confidence,
        "signal": signal,
    }


def _project_simplex_with_bounds(current: np.ndarray, target: np.ndarray) -> np.ndarray:
    current = np.asarray(current, dtype=float)
    target = np.asarray(target, dtype=float)
    if len(current) != EXPERT_COUNT or len(target) != EXPERT_COUNT:
        raise RuntimeError("COHORT_V2_WEIGHT_VECTOR_LENGTH")
    if not np.isclose(current.sum(), 1.0, atol=1e-10):
        raise RuntimeError(f"COHORT_V2_CURRENT_WEIGHT_SUM:{current.sum()}")
    target = target / target.sum()

    lower = np.maximum(WEIGHT_FLOOR, current - MAX_ABS_WEIGHT_CHANGE_PER_ASSESSMENT)
    upper = np.minimum(WEIGHT_CEILING, current + MAX_ABS_WEIGHT_CHANGE_PER_ASSESSMENT)
    if lower.sum() > 1.0 + 1e-12 or upper.sum() < 1.0 - 1e-12:
        raise RuntimeError("COHORT_V2_BOUNDED_SIMPLEX_INFEASIBLE")

    lo, hi = -2.0, 2.0
    for _ in range(120):
        mid = (lo + hi) / 2.0
        candidate = np.clip(target + mid, lower, upper)
        if candidate.sum() > 1.0:
            hi = mid
        else:
            lo = mid
    result = np.clip(target + (lo + hi) / 2.0, lower, upper)
    result = result / result.sum()

    if np.max(np.abs(result - current)) > MAX_ABS_WEIGHT_CHANGE_PER_ASSESSMENT + 1e-9:
        raise RuntimeError("COHORT_V2_MAX_REWEIGHT_BREACH")
    if result.min() < WEIGHT_FLOOR - 1e-9 or result.max() > WEIGHT_CEILING + 1e-9:
        raise RuntimeError("COHORT_V2_WEIGHT_BOUND_BREACH")
    if not np.isclose(result.sum(), 1.0, atol=1e-10):
        raise RuntimeError("COHORT_V2_WEIGHT_SUM_BREACH")
    return result


def _roll_weights(
    previous: dict[str, float] | None,
    current_ids: list[str],
) -> tuple[np.ndarray, dict]:
    if len(current_ids) != EXPERT_COUNT:
        raise RuntimeError("COHORT_V2_CURRENT_ID_COUNT")
    if previous is None:
        return np.full(EXPERT_COUNT, INITIAL_WEIGHT, dtype=float), {
            "retained": 0,
            "entered": EXPERT_COUNT,
            "departed": 0,
            "roll_reason": "INITIAL_UNIFORM_12_COHORTS",
        }

    current_set = set(current_ids)
    retained = [cid for cid in current_ids if cid in previous]
    entered = [cid for cid in current_ids if cid not in previous]
    departed = [cid for cid in previous if cid not in current_set]

    weights = {cid: float(previous[cid]) for cid in retained}
    departed_mass = float(sum(previous[cid] for cid in departed))
    if entered:
        if departed_mass <= 0:
            raise RuntimeError("COHORT_V2_ENTERED_WITHOUT_DEPARTED_MASS")
        for cid in entered:
            weights[cid] = departed_mass / len(entered)
    elif departed:
        raise RuntimeError("COHORT_V2_DEPARTED_WITHOUT_ENTERED")

    vector = np.array([weights[cid] for cid in current_ids], dtype=float)
    if not np.isclose(vector.sum(), 1.0, atol=1e-10):
        raise RuntimeError(
            f"COHORT_V2_ROLLED_WEIGHT_SUM:{vector.sum()}:retained={len(retained)}:"
            f"entered={len(entered)}:departed={len(departed)}"
        )
    if vector.min() < WEIGHT_FLOOR - 1e-9 or vector.max() > WEIGHT_CEILING + 1e-9:
        raise RuntimeError("COHORT_V2_ROLLED_WEIGHT_BOUND_BREACH")
    return vector, {
        "retained": len(retained),
        "entered": len(entered),
        "departed": len(departed),
        "departed_mass": departed_mass,
        "roll_reason": "COHORT_WEIGHT_AGED_WITH_DATA_BLOCK",
    }


def _next_weights(
    rolled: np.ndarray,
    snapshot: list[dict],
) -> tuple[np.ndarray, dict]:
    signal, detail = _robust_signals(snapshot)
    target_raw = rolled * np.exp(HEDGE_ETA * signal)
    target = target_raw / target_raw.sum()
    updated = _project_simplex_with_bounds(rolled, target)
    detail.update(
        {
            "updated": True,
            "reason": "COHORT_MAGNITUDE_CONFIDENCE_UPDATE",
            "target": target,
            "max_abs_change": float(np.max(np.abs(updated - rolled))),
        }
    )
    return updated, detail


def _build_schedule(
    prepared: pd.DataFrame,
    cohorts: list[dict],
    assessment_dates: list[pd.Timestamp],
    *,
    model_id: str,
    contract_id: str,
    max_names: int,
    historical_pressure: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    previous_weights: dict[str, float] | None = None
    states: list[dict] = []
    evidence_rows: list[dict] = []

    for assessment_date in assessment_dates:
        snapshot = _current_cohorts(
            prepared, cohorts, assessment_date, max_names=max_names
        )
        if len(snapshot) < EXPERT_COUNT:
            states.append(
                {
                    "model_id": model_id,
                    "contract_id": contract_id,
                    "assessment_date": assessment_date,
                    "phase": "WARMUP" if assessment_date < LIVE_START else "LIVE_EVALUATION",
                    "updated": False,
                    "update_reason": "WAIT_FOR_12_FIXED_MATURED_COHORTS",
                    "available_cohorts": len(snapshot),
                }
            )
            continue

        snapshot = sorted(snapshot, key=lambda row: int(row["cohort_no"]), reverse=True)
        current_ids = [str(row["cohort_id"]) for row in snapshot]
        rolled, roll = _roll_weights(previous_weights, current_ids)
        if previous_weights is not None and roll["entered"] == 0 and roll["departed"] == 0:
            signal, detail = _robust_signals(snapshot)
            updated = rolled.copy()
            update = {
                **detail,
                "updated": False,
                "reason": "NO_NEW_MATURED_COHORT_NO_RELEARN",
                "target": rolled.copy(),
                "max_abs_change": 0.0,
            }
        else:
            updated, update = _next_weights(rolled, snapshot)
        p99 = np.array([float(row["median_p99"]) for row in snapshot], dtype=float)
        controller_p99 = float(np.dot(updated, p99))
        uniform_p99 = float(np.mean(p99))
        controller_replacement = float(historical_pressure * controller_p99)
        uniform_replacement = float(historical_pressure * uniform_p99)

        state = {
            "model_id": model_id,
            "contract_id": contract_id,
            "assessment_date": assessment_date,
            "phase": "WARMUP" if assessment_date < LIVE_START else "LIVE_EVALUATION",
            "updated": bool(update["updated"]),
            "update_reason": update["reason"],
            "available_cohorts": len(snapshot),
            "retained_cohorts": roll["retained"],
            "entered_cohorts": roll["entered"],
            "departed_cohorts": roll["departed"],
            "historical_threshold_over_p99": historical_pressure,
            "controller_weighted_p99": controller_p99,
            "uniform_p99": uniform_p99,
            "controller_replacement_threshold": controller_replacement,
            "uniform_replacement_threshold": uniform_replacement,
            "effective_memory_months": float(
                sum((i + 1) * updated[i] for i in range(EXPERT_COUNT))
            ),
            "weight_entropy": float(-sum(w * np.log(w) for w in updated if w > 0)),
            "max_weight": float(updated.max()),
            "max_abs_weight_change": float(np.max(np.abs(updated - rolled))),
            "quality_median": float(update["quality_median"]),
            "quality_scale": float(update["quality_scale"]),
        }
        for idx, (row, w_before, w_after) in enumerate(
            zip(snapshot, rolled, updated), start=1
        ):
            state[f"cohort_m{idx}"] = row["cohort_id"]
            state[f"w_m{idx}"] = float(w_after)
            evidence_rows.append(
                {
                    "model_id": model_id,
                    "contract_id": contract_id,
                    "assessment_date": assessment_date,
                    "phase": state["phase"],
                    "age_slot": f"M{idx}",
                    "weight_before_update": float(w_before),
                    "weight_after_update": float(w_after),
                    "robust_z": float(update["robust_z"][idx - 1]),
                    "confidence": float(update["confidence"][idx - 1]),
                    "controller_signal": float(update["signal"][idx - 1]),
                    "target_weight_unbounded": float(update["target"][idx - 1]),
                    **row,
                }
            )
        states.append(state)
        previous_weights = {
            cid: float(weight) for cid, weight in zip(current_ids, updated)
        }

    return pd.DataFrame(states), pd.DataFrame(evidence_rows)


def _live_thresholds(
    selected: pd.DataFrame,
    raw: pd.Series,
    schedule: pd.DataFrame,
) -> pd.DataFrame:
    dates = [
        pd.Timestamp(x)
        for x in sorted(
            selected.loc[
                selected["decision_date"].between(LIVE_START, LIVE_END), "decision_date"
            ].drop_duplicates()
        )
    ]
    valid_states = schedule.loc[schedule["updated"].eq(True)].sort_values("assessment_date")
    states = list(valid_states.itertuples(index=False))
    rows = []
    for decision_date in dates:
        raw_threshold = float(raw.loc[decision_date])
        prior = [row for row in states if pd.Timestamp(row.assessment_date) < decision_date]
        latest = prior[-1] if prior else None
        if latest is None:
            controller_replacement = uniform_replacement = np.nan
            source_date = pd.NaT
            effective_memory = np.nan
        else:
            controller_replacement = float(latest.controller_replacement_threshold)
            uniform_replacement = float(latest.uniform_replacement_threshold)
            source_date = pd.Timestamp(latest.assessment_date)
            effective_memory = float(latest.effective_memory_months)

        controller_effective = (
            min(raw_threshold, controller_replacement)
            if np.isfinite(controller_replacement)
            else raw_threshold
        )
        uniform_effective = (
            min(raw_threshold, uniform_replacement)
            if np.isfinite(uniform_replacement)
            else raw_threshold
        )
        rows.append(
            {
                "decision_date": decision_date,
                "raw_threshold": raw_threshold,
                "cohort_controller_replacement_threshold": controller_replacement,
                "cohort_uniform_replacement_threshold": uniform_replacement,
                "cohort_controller_effective_threshold": controller_effective,
                "cohort_uniform_effective_threshold": uniform_effective,
                "controller_adapted": bool(controller_effective < raw_threshold - 1e-15),
                "uniform_adapted": bool(uniform_effective < raw_threshold - 1e-15),
                "source_assessment_date": source_date,
                "effective_memory_months": effective_memory,
            }
        )
    frame = pd.DataFrame(rows)
    active = frame["source_assessment_date"].notna()
    if active.any() and frame.loc[active, "source_assessment_date"].ge(
        frame.loc[active, "decision_date"]
    ).any():
        raise RuntimeError("COHORT_V2_NONCAUSAL_THRESHOLD_SOURCE")
    if (
        frame["cohort_controller_effective_threshold"]
        > frame["raw_threshold"] + 1e-12
    ).any():
        raise RuntimeError("COHORT_V2_TIGHTENED_THRESHOLD")
    return frame


def _replay_arm(
    *,
    arm: str,
    model_id: str,
    contract_id: str,
    signals: pd.DataFrame,
    prices: pd.DataFrame,
    policy,
    threshold_by_date: pd.Series,
    exit_provider: LearnedExitProvider,
    prepared_prices,
    initial_capital: float,
) -> tuple[dict, list[dict], pd.DataFrame]:
    configure_replay(exit_provider, ProfitTaxConfig())
    result = replay(
        signals,
        prices,
        policy,
        CostModel(20),
        TaxConfig(False),
        start=LIVE_START,
        end=LIVE_END,
        initial=float(initial_capital),
        resolved_threshold_by_date=threshold_by_date,
        prepared_prices=prepared_prices,
    )
    metrics = result["metrics"]
    if int(metrics.get("learned_exit_missing_prediction_count", 0)):
        raise RuntimeError(f"COHORT_V2_LEARNED_EXIT_MISSING:{model_id}:{arm}")
    if float(metrics.get("exit_decision_coverage", 0.0)) != 1.0:
        raise RuntimeError(f"COHORT_V2_EXIT_COVERAGE:{model_id}:{arm}")
    trades = [
        {"model_id": model_id, "contract_id": contract_id, "arm": arm, **trade}
        for trade in result["trades"]
    ]
    curve = result["curve"].copy()
    curve.insert(0, "model_id", model_id)
    curve.insert(1, "contract_id", contract_id)
    curve.insert(2, "arm", arm)
    return metrics, trades, curve


def run(args: argparse.Namespace) -> dict:
    _validate_contract()
    output = Path(args.output_root)
    output.mkdir(parents=True, exist_ok=True)
    _prior_suite_gates(Path(args.retrospective_summary), Path(args.entry_diagnostic_summary))
    v1_reference = _v1_gate(Path(args.v1_summary), Path(args.v1_results))

    historical, historical_audit = _historical_entry(Path(args.historical_entry_predictions))
    causal = _causal_entry(Path(args.causal_entry_predictions))
    combined = _combined_entry(historical, causal)
    cohorts = _fixed_cohorts(combined)
    contracts, contracts_by_model = _contracts(Path(args.frozen_policy_manifest))
    pressure = _historical_pressure(Path(args.entry_diagnostic_summary))

    tickers = set(combined["ticker"].astype(str))
    prices, price_audit = load_price_panel(Path(args.daily_store_root), tickers)
    if pd.Timestamp(prices["date"].max()).normalize() < LIVE_END:
        raise RuntimeError("COHORT_V2_PRICES_END_BEFORE_LIVE_END")
    prepared_prices = prepare_prices(prices)
    assessment_dates = _assessment_dates(combined)

    manifest = json.loads(Path(args.frozen_policy_manifest).read_text(encoding="utf-8"))
    models = manifest.get("models", [])
    if len(models) != 10:
        raise RuntimeError(f"COHORT_V2_TOP10_COUNT:{len(models)}")

    exit_provider = LearnedExitProvider(Path(args.causal_exit_predictions))
    if (
        exit_provider.audit.min_date != str(LIVE_START.date())
        or exit_provider.audit.max_date != str(LIVE_END.date())
    ):
        raise RuntimeError(f"COHORT_V2_CAUSAL_EXIT_COVERAGE:{asdict(exit_provider.audit)}")

    all_states, all_evidence, all_daily = [], [], []
    results, trades, curves = [], [], []

    for model in models:
        model_id = str(model["model_id"])
        policy = _policy(model["frozen_entry_policy"])
        spec = contracts_by_model[model_id]
        match = contracts.loc[
            contracts["horizon"].eq(int(spec["horizon"]))
            & contracts["score_quantile"].eq(float(spec["score_quantile"]))
            & contracts["top_fraction"].eq(float(spec["top_fraction"])),
            "contract_id",
        ]
        if len(match) != 1:
            raise RuntimeError(f"COHORT_V2_MODEL_CONTRACT_MAPPING:{model_id}:{len(match)}")
        contract_id = str(match.iloc[0])
        if contract_id not in pressure:
            raise RuntimeError(f"COHORT_V2_PRESSURE_MISSING:{contract_id}")

        model_rows = _prepare_model_rows(
            combined,
            prices,
            horizon=int(policy.horizon),
            top_fraction=float(policy.top_fraction),
            max_names=int(policy.max_names),
        )
        schedule, evidence = _build_schedule(
            model_rows,
            cohorts,
            assessment_dates,
            model_id=model_id,
            contract_id=contract_id,
            max_names=int(policy.max_names),
            historical_pressure=float(pressure[contract_id]),
        )
        all_states.append(schedule)
        all_evidence.append(evidence)

        selected = causal.loc[causal["horizon"].eq(int(policy.horizon))].copy()
        raw = _thresholds(selected, float(policy.score_quantile))
        daily = _live_thresholds(selected, raw, schedule)
        daily.insert(0, "model_id", model_id)
        daily.insert(1, "contract_id", contract_id)
        all_daily.append(daily)

        signals = selected[["decision_date", "ticker", "score"]]
        arm_thresholds = {
            "RAW": daily.set_index("decision_date")["raw_threshold"],
            "COHORT_UNIFORM_12M": daily.set_index("decision_date")[
                "cohort_uniform_effective_threshold"
            ],
            "COHORT_CONTROLLER_V2": daily.set_index("decision_date")[
                "cohort_controller_effective_threshold"
            ],
        }
        for arm, threshold_series in arm_thresholds.items():
            metrics, arm_trades, curve = _replay_arm(
                arm=arm,
                model_id=model_id,
                contract_id=contract_id,
                signals=signals,
                prices=prices,
                policy=policy,
                threshold_by_date=threshold_series,
                exit_provider=exit_provider,
                prepared_prices=prepared_prices,
                initial_capital=args.initial_capital,
            )
            results.append(
                {
                    "model_id": model_id,
                    "contract_id": contract_id,
                    "arm": arm,
                    "h": policy.horizon,
                    "d": policy.holding_days,
                    "n": policy.max_names,
                    "mode": policy.exit_family,
                    "score_quantile": policy.score_quantile,
                    "top_fraction": policy.top_fraction,
                    **metrics,
                }
            )
            trades.extend(arm_trades)
            curves.append(curve)
            print(
                f"COHORT_CONTROLLER_V2_REPLAY_COMPLETE {model_id} {arm} "
                f"trades={metrics['trade_count']}",
                flush=True,
            )

    state_frame = pd.concat(all_states, ignore_index=True)
    evidence_frame = pd.concat(all_evidence, ignore_index=True)
    daily_frame = pd.concat(all_daily, ignore_index=True)
    result_frame = pd.DataFrame(results).sort_values(["arm", "model_id"]).reset_index(drop=True)

    state_frame.to_csv(output / "cohort_controller_state.csv", index=False)
    state_frame.to_parquet(output / "cohort_controller_state.parquet", index=False)
    evidence_frame.to_parquet(output / "cohort_controller_evidence.parquet", index=False)
    daily_frame.to_csv(output / "cohort_controller_live_thresholds.csv", index=False)
    daily_frame.to_parquet(output / "cohort_controller_live_thresholds.parquet", index=False)
    result_frame.to_csv(output / "cohort_controller_results.csv", index=False)
    pd.DataFrame(trades).to_csv(output / "cohort_controller_trades.csv", index=False)
    pd.concat(curves, ignore_index=True).to_parquet(
        output / "cohort_controller_curves.parquet", index=False
    )

    pre_live = state_frame.loc[
        state_frame["updated"].eq(True)
        & pd.to_datetime(state_frame["assessment_date"]).lt(LIVE_START)
    ].copy()
    live_start_rows = (
        pre_live.sort_values("assessment_date").groupby("model_id", as_index=False).tail(1)
    )
    live_start_rows.to_csv(output / "cohort_weights_at_live_start.csv", index=False)

    arm_summary = []
    for arm, group in result_frame.groupby("arm", sort=True):
        arm_summary.append(
            {
                "arm": arm,
                "models": int(len(group)),
                "total_trades": int(group["trade_count"].sum()),
                "positive_cagr_excess_models": int(group["cagr_excess"].gt(0).sum()),
                "median_cagr_excess": float(group["cagr_excess"].median()),
                "mean_cagr_excess": float(group["cagr_excess"].mean()),
                "selection_allowed": False,
            }
        )

    v1_cmp = v1_reference[
        ["model_id", "cagr_excess", "terminal_wealth_excess_eur", "trade_count"]
    ].copy()
    v1_cmp["reference_arm"] = "CONTROLLER_V1_REFERENCE"
    v2_cmp = result_frame.loc[
        result_frame["arm"].eq("COHORT_CONTROLLER_V2"),
        ["model_id", "cagr_excess", "terminal_wealth_excess_eur", "trade_count"],
    ].copy()
    comparison = v2_cmp.merge(
        v1_cmp, on="model_id", suffixes=("_v2", "_v1"), validate="one_to_one"
    )
    comparison["cagr_excess_delta_v2_minus_v1"] = (
        comparison["cagr_excess_v2"] - comparison["cagr_excess_v1"]
    )
    comparison["wealth_excess_delta_v2_minus_v1"] = (
        comparison["terminal_wealth_excess_eur_v2"]
        - comparison["terminal_wealth_excess_eur_v1"]
    )
    comparison.to_csv(output / "cohort_controller_v1_comparison.csv", index=False)

    summary = {
        "contract_id": CONTRACT_ID,
        "status": "COMPLETE",
        "evaluation_role": "DEVELOPMENT_REUSE_NOT_INDEPENDENT_OOS",
        "development_reuse_notice": DEVELOPMENT_REUSE_NOTICE,
        "no_model_training": True,
        "no_prediction_generation": True,
        "no_threshold_optimization": True,
        "no_weight_grid_search": True,
        "uses_existing_17000_causal_artifacts": True,
        "final_holdout_opened": False,
        "warmup": {
            "start": str(WARMUP_START.date()),
            "end": "2023-08-10",
            "source": "historical WF_001..WF_007 OOS predictions",
            "performance_interpretation_allowed": False,
        },
        "evaluation": {
            "start": str(LIVE_START.date()),
            "end": str(LIVE_END.date()),
            "source": "existing causal-expanding entry/exit predictions",
        },
        "controller_contract": {
            "fixed_cohort_sessions": MONTH_SESSIONS,
            "cohorts_in_memory": EXPERT_COUNT,
            "initial_weight_each": INITIAL_WEIGHT,
            "weight_attached_to_concrete_cohort": True,
            "cohort_ages_m1_to_m12": True,
            "quality": "mean realized net excess with individual returns clipped +/-50%",
            "magnitude_signal": "robust z via median/MAD, clipped +/-2",
            "confidence": "min(1, sqrt(realized_candidate_count/target_count))",
            "hedge_eta": HEDGE_ETA,
            "weight_floor": WEIGHT_FLOOR,
            "weight_ceiling": WEIGHT_CEILING,
            "max_abs_weight_change_per_assessment": MAX_ABS_WEIGHT_CHANGE_PER_ASSESSMENT,
            "outcome_visibility": "cohort terminal_date <= assessment_date",
            "update_effective": "strictly after assessment_date",
            "no_repeat_update_without_new_matured_cohort": True,
            "threshold_scale": "historical_threshold_over_p99 * cohort-weighted median_p99",
            "effective_threshold": "min(raw_threshold, replacement_threshold)",
        },
        "controller_updates": int(state_frame["updated"].fillna(False).sum()),
        "arm_summary": arm_summary,
        "v1_reference": {
            "source": str(args.v1_results),
            "replayed_in_v2": False,
            "median_cagr_excess": float(v1_reference["cagr_excess"].median()),
            "positive_cagr_excess_models": int(v1_reference["cagr_excess"].gt(0).sum()),
        },
        "v2_vs_v1": {
            "models_v2_better_cagr_excess": int(
                comparison["cagr_excess_delta_v2_minus_v1"].gt(0).sum()
            ),
            "median_cagr_excess_delta": float(
                comparison["cagr_excess_delta_v2_minus_v1"].median()
            ),
        },
        "historical_prediction_audit": historical_audit,
        "exit_provider_audit": asdict(exit_provider.audit),
        "price_audit": price_audit,
    }
    _write_json(output / "cohort_controller_summary.json", summary)

    report_cols = [
        "model_id",
        "arm",
        "trade_count",
        "cagr_excess",
        "terminal_wealth_excess_eur",
    ]
    report = result_frame[report_cols].copy()
    report["cagr_excess"] = report["cagr_excess"].map(lambda x: f"{float(x):.4%}")
    comparison_report = comparison[
        [
            "model_id",
            "cagr_excess_v1",
            "cagr_excess_v2",
            "cagr_excess_delta_v2_minus_v1",
        ]
    ].copy()
    for col in ("cagr_excess_v1", "cagr_excess_v2", "cagr_excess_delta_v2_minus_v1"):
        comparison_report[col] = comparison_report[col].map(
            lambda x: f"{float(x):.4%}"
        )
    text = [
        "# Top-10 Causal Cohort Controller V2",
        "",
        "Status: **COMPLETE**",
        "",
        "**Development-reuse warning:** " + DEVELOPMENT_REUSE_NOTICE,
        "",
        "V2 attaches weights to concrete 21-session cohorts, carries those weights as each "
        "cohort ages from M1 to M12, and uses magnitude-aware robust quality with confidence.",
        "",
        "Hard controller limits: floor 1%, ceiling 35%, max +/-5 percentage points per assessment.",
        "",
        "## V2 replay arms",
        "",
        _markdown(report),
        "",
        "## V2 controller versus frozen V1 reference",
        "",
        _markdown(comparison_report),
        "",
        "No model was trained, no prediction was regenerated, no threshold/weight grid was searched, "
        "and the final holdout remained closed.",
    ]
    (output / "REPORT.md").write_text("\n".join(text) + "\n", encoding="utf-8")
    return summary


def self_test() -> None:
    _validate_contract()

    previous_ids = [f"C{i:02d}" for i in range(1, 13)]
    previous = {cid: INITIAL_WEIGHT for cid in previous_ids}
    current_ids = [f"C{i:02d}" for i in range(2, 14)]
    rolled, audit = _roll_weights(previous, current_ids)
    assert audit["retained"] == 11 and audit["entered"] == 1 and audit["departed"] == 1
    assert np.allclose(rolled, INITIAL_WEIGHT)

    snapshot = []
    for i in range(EXPERT_COUNT):
        snapshot.append(
            {
                "quality_mean_excess_clipped": -0.05 + i * 0.01,
                "confidence": 1.0,
            }
        )
    updated, detail = _next_weights(rolled, snapshot)
    assert np.isclose(updated.sum(), 1.0)
    assert updated[-1] > updated[0]
    assert detail["robust_z"][-1] > detail["robust_z"][0]
    assert np.max(np.abs(updated - rolled)) <= MAX_ABS_WEIGHT_CHANGE_PER_ASSESSMENT + 1e-9
    assert updated.min() >= WEIGHT_FLOOR - 1e-9
    assert updated.max() <= WEIGHT_CEILING + 1e-9

    low_conf_snapshot = []
    for i in range(EXPERT_COUNT):
        low_conf_snapshot.append(
            {
                "quality_mean_excess_clipped": -0.05 + i * 0.01,
                "confidence": 0.1 if i == EXPERT_COUNT - 1 else 1.0,
            }
        )
    _, low_detail = _next_weights(rolled, low_conf_snapshot)
    assert abs(low_detail["signal"][-1]) < abs(detail["signal"][-1])

    print("TOP10_CAUSAL_COHORT_CONTROLLER_V2_SELF_TEST_OK")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--historical-entry-predictions")
    parser.add_argument("--causal-entry-predictions")
    parser.add_argument("--causal-exit-predictions")
    parser.add_argument("--frozen-policy-manifest")
    parser.add_argument("--entry-diagnostic-summary")
    parser.add_argument("--retrospective-summary")
    parser.add_argument("--v1-summary")
    parser.add_argument("--v1-results")
    parser.add_argument("--daily-store-root")
    parser.add_argument(
        "--output-root", default="artifacts/top10-causal-cohort-controller-v2"
    )
    parser.add_argument("--initial-capital", type=float, default=10000.0)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return

    required = (
        "historical_entry_predictions",
        "causal_entry_predictions",
        "causal_exit_predictions",
        "frozen_policy_manifest",
        "entry_diagnostic_summary",
        "retrospective_summary",
        "v1_summary",
        "v1_results",
        "daily_store_root",
    )
    missing = [name for name in required if not getattr(args, name)]
    if missing:
        parser.error(
            "required arguments missing: "
            + ", ".join("--" + x.replace("_", "-") for x in missing)
        )
    print(json.dumps(run(args), indent=2, default=str))


if __name__ == "__main__":
    main()
