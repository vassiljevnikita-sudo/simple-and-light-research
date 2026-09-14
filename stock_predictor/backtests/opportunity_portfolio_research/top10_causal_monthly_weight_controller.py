"""Causal 12-month online weight controller for frozen Top-10 entry policies.

Contract: TOP10_CAUSAL_MONTHLY_WEIGHT_CONTROLLER_V1

Development warm-up: 2020-09-01 through 2023-08-10 using historical WF_001..WF_007
OOS predictions. Evaluation: 2023-08-11 through 2026-07-24 using the already
completed causal-expanding prediction artifacts.

The controller starts uniform at 1/12 across M1..M12. Monthly expert quality is
computed only from fully matured prior outcomes. Weights are updated online with a
bounded multiplicative-weights target and a hard maximum change per assessment.
No model fitting, prediction generation, weight search, threshold search, or final
holdout access occurs in this suite.
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from .cost_contracts import CostModel
from .tax_contracts import TaxConfig
from .portfolio_research_inputs import load_predictions, load_price_panel
from .learned_exit_qbd_profit import ProfitTaxConfig
from .learned_exit_qbd_provider import LearnedExitProvider
from .learned_exit_qbd_replay import configure_replay, replay
from .next_open_portfolio_replay import prepare_prices
from .top10_causal_expanding_portfolio import LIVE_END, LIVE_START, _policy, _thresholds
from .top10_entry_activation_alpha_diagnostic import (
    _attach_realized,
    _causal_entry,
    _contracts,
    _markdown,
    _price_views,
    _write_json,
)

CONTRACT_ID = "TOP10_CAUSAL_MONTHLY_WEIGHT_CONTROLLER_V1"
WARMUP_START = pd.Timestamp("2020-09-01")
MONTH_SESSIONS = 21
EXPERT_COUNT = 12
ASSESS_EVERY_SESSIONS = 21
INITIAL_WEIGHT = 1.0 / EXPERT_COUNT
WEIGHT_FLOOR = 0.02
WEIGHT_CEILING = 0.25
MAX_ABS_WEIGHT_CHANGE_PER_ASSESSMENT = 0.02
HEDGE_ETA = 1.0
QUALITY_RETURN_CLIP = 0.50
MIN_CANDIDATES_PER_EXPERT = 15
HISTORICAL_WF_PATTERN = re.compile(r"WF_00[1-7]")


def _validate_contract() -> None:
    assert EXPERT_COUNT == 12
    assert abs(INITIAL_WEIGHT * EXPERT_COUNT - 1.0) < 1e-12
    assert 0 <= WEIGHT_FLOOR < INITIAL_WEIGHT < WEIGHT_CEILING <= 1
    assert 0 < MAX_ABS_WEIGHT_CHANGE_PER_ASSESSMENT < 0.1
    assert WEIGHT_FLOOR * EXPERT_COUNT < 1.0 < WEIGHT_CEILING * EXPERT_COUNT
    assert WARMUP_START < LIVE_START <= LIVE_END


def _historical_entry(path: Path) -> tuple[pd.DataFrame, dict]:
    frame, audit = load_predictions(path)
    frame = frame.loc[
        frame["horizon"].isin([11, 24, 28])
        & frame["fold_id"].str.contains(HISTORICAL_WF_PATTERN)
        & frame["decision_date"].ge(WARMUP_START)
        & frame["decision_date"].lt(LIVE_START)
    ].copy()
    if frame.empty:
        raise RuntimeError("MONTHLY_CONTROLLER_HISTORICAL_ENTRY_EMPTY")
    if frame["decision_date"].min() > WARMUP_START + pd.Timedelta(days=10):
        raise RuntimeError(
            f"MONTHLY_CONTROLLER_WARMUP_START_TOO_LATE:{frame['decision_date'].min().date()}"
        )
    if frame["decision_date"].max() != pd.Timestamp("2023-08-10"):
        raise RuntimeError(
            f"MONTHLY_CONTROLLER_HISTORICAL_ENTRY_END_MISMATCH:{frame['decision_date'].max().date()}"
        )
    if frame.duplicated(["decision_date", "ticker", "horizon"]).any():
        raise RuntimeError("MONTHLY_CONTROLLER_HISTORICAL_ENTRY_DUPLICATE_KEYS")
    audit = {
        **audit,
        "controller_rows": int(len(frame)),
        "controller_min_date": str(frame["decision_date"].min().date()),
        "controller_max_date": str(frame["decision_date"].max().date()),
        "wf_000_excluded": True,
    }
    return frame[["decision_date", "ticker", "horizon", "score"]], audit


def _combined_entry(historical: pd.DataFrame, causal: pd.DataFrame) -> pd.DataFrame:
    h = historical.copy()
    h["prediction_source"] = "HISTORICAL_WF_WARMUP"
    c = causal[["decision_date", "ticker", "horizon", "score"]].copy()
    c["prediction_source"] = "CAUSAL_EXPANDING_LIVE"
    frame = pd.concat([h, c], ignore_index=True)
    frame = frame.loc[
        frame["decision_date"].ge(WARMUP_START) & frame["decision_date"].le(LIVE_END)
    ].copy()
    if frame.duplicated(["decision_date", "ticker", "horizon"]).any():
        dup = frame.loc[
            frame.duplicated(["decision_date", "ticker", "horizon"], keep=False),
            ["decision_date", "ticker", "horizon", "prediction_source"],
        ].head(10)
        raise RuntimeError(f"MONTHLY_CONTROLLER_COMBINED_DUPLICATES:{dup.to_dict(orient='records')}")
    return frame.sort_values(["decision_date", "horizon", "ticker"]).reset_index(drop=True)


def _historical_pressure(path: Path) -> dict[str, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "COMPLETE" or payload.get("final_holdout_opened"):
        raise RuntimeError("MONTHLY_CONTROLLER_ENTRY_DIAGNOSTIC_GATE_FAILED")
    result = {}
    for row in payload.get("activation_summary", []):
        if row.get("source") != "HISTORICAL_WF":
            continue
        value = float(row["median_threshold_over_p99"])
        if not np.isfinite(value) or value <= 0:
            raise RuntimeError(f"MONTHLY_CONTROLLER_HISTORICAL_PRESSURE_INVALID:{row.get('contract_id')}")
        result[str(row["contract_id"])] = value
    if not result:
        raise RuntimeError("MONTHLY_CONTROLLER_HISTORICAL_PRESSURE_MISSING")
    return result


def _prior_suite_gates(retrospective_path: Path, diagnostic_path: Path) -> None:
    retro = json.loads(retrospective_path.read_text(encoding="utf-8"))
    if (
        retro.get("status") != "COMPLETE"
        or retro.get("no_forward_outcome_use") is not True
        or retro.get("no_threshold_optimization") is not True
        or retro.get("final_holdout_opened")
    ):
        raise RuntimeError("MONTHLY_CONTROLLER_RETROSPECTIVE_GATE_FAILED")
    diag = json.loads(diagnostic_path.read_text(encoding="utf-8"))
    if (
        diag.get("status") != "COMPLETE"
        or diag.get("no_model_training") is not True
        or diag.get("no_prediction_generation") is not True
        or diag.get("final_holdout_opened")
    ):
        raise RuntimeError("MONTHLY_CONTROLLER_DIAGNOSTIC_GATE_FAILED")


def _assessment_dates(combined: pd.DataFrame) -> list[pd.Timestamp]:
    dates = [
        pd.Timestamp(x)
        for x in sorted(
            combined.loc[
                combined["decision_date"].between(WARMUP_START, LIVE_END), "decision_date"
            ].drop_duplicates()
        )
    ]
    if not dates:
        raise RuntimeError("MONTHLY_CONTROLLER_NO_ASSESSMENT_DATES")
    anchors = dates[ASSESS_EVERY_SESSIONS - 1 :: ASSESS_EVERY_SESSIONS]
    pre_live = max((d for d in dates if d < LIVE_START), default=None)
    for required in (pre_live, dates[-1]):
        if required is not None and required not in anchors:
            anchors.append(required)
    return sorted(set(anchors))


def _prepare_model_rows(
    combined: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    horizon: int,
    top_fraction: float,
    max_names: int,
) -> pd.DataFrame:
    base = combined.loc[combined["horizon"].eq(int(horizon))].copy()
    if base.empty:
        raise RuntimeError(f"MONTHLY_CONTROLLER_MODEL_ROWS_EMPTY:H{horizon}")
    base["daily_rank"] = base.groupby("decision_date")["score"].rank(
        method="first", ascending=False
    )
    universe = base.groupby("decision_date")["ticker"].transform("size")
    top_fraction_limit = np.maximum(1, np.ceil(universe * float(top_fraction))).astype(int)
    base["candidate_limit"] = np.minimum(top_fraction_limit, int(max_names))
    base["relative_candidate"] = base["daily_rank"].le(base["candidate_limit"])

    stock, urth, sessions, session_index = _price_views(prices)
    realized = _attach_realized(base, stock, urth, sessions, session_index, int(horizon))
    flags = base[
        [
            "decision_date",
            "ticker",
            "prediction_source",
            "daily_rank",
            "candidate_limit",
            "relative_candidate",
        ]
    ]
    realized = realized.merge(
        flags, on=["decision_date", "ticker"], how="left", validate="one_to_one"
    )
    if realized["relative_candidate"].isna().any():
        raise RuntimeError("MONTHLY_CONTROLLER_REALIZED_FLAG_MERGE_FAILED")
    return realized


def _expert_snapshot(prepared: pd.DataFrame, assessment_date: pd.Timestamp) -> list[dict]:
    matured = prepared.loc[
        prepared["realized_price_available"]
        & prepared["terminal_date"].le(assessment_date)
        & prepared["decision_date"].lt(assessment_date)
    ].copy()
    matured_dates = [
        pd.Timestamp(x) for x in sorted(matured["decision_date"].drop_duplicates())
    ]
    rows = []
    for expert_index in range(1, EXPERT_COUNT + 1):
        end = len(matured_dates) - (expert_index - 1) * MONTH_SESSIONS
        start = end - MONTH_SESSIONS
        if start < 0 or end <= 0:
            rows.append(
                {
                    "expert": f"M{expert_index}",
                    "expert_index": expert_index,
                    "available": False,
                    "decision_days": 0,
                    "candidate_count": 0,
                    "quality_mean_excess_clipped": np.nan,
                    "raw_mean_excess": np.nan,
                    "median_excess": np.nan,
                    "hit_rate": np.nan,
                    "median_p99": np.nan,
                    "window_start": pd.NaT,
                    "window_end": pd.NaT,
                    "latest_terminal_date_used": pd.NaT,
                }
            )
            continue
        dates = matured_dates[start:end]
        window = matured.loc[matured["decision_date"].isin(dates)].copy()
        candidates = window.loc[window["relative_candidate"]].copy()
        clipped = candidates["realized_net_excess_20bps"].clip(
            lower=-QUALITY_RETURN_CLIP, upper=QUALITY_RETURN_CLIP
        )
        daily_p99 = window.groupby("decision_date")["score"].quantile(0.99)
        p99 = float(daily_p99.median()) if len(daily_p99) else np.nan
        available = (
            len(dates) == MONTH_SESSIONS
            and len(candidates) >= MIN_CANDIDATES_PER_EXPERT
            and np.isfinite(p99)
            and p99 > 0
        )
        rows.append(
            {
                "expert": f"M{expert_index}",
                "expert_index": expert_index,
                "available": bool(available),
                "decision_days": int(len(dates)),
                "candidate_count": int(len(candidates)),
                "quality_mean_excess_clipped": (
                    float(clipped.mean()) if len(clipped) else np.nan
                ),
                "raw_mean_excess": (
                    float(candidates["realized_net_excess_20bps"].mean())
                    if len(candidates)
                    else np.nan
                ),
                "median_excess": (
                    float(candidates["realized_net_excess_20bps"].median())
                    if len(candidates)
                    else np.nan
                ),
                "hit_rate": (
                    float(candidates["realized_net_excess_20bps"].gt(0).mean())
                    if len(candidates)
                    else np.nan
                ),
                "median_p99": p99,
                "window_start": min(dates) if dates else pd.NaT,
                "window_end": max(dates) if dates else pd.NaT,
                "latest_terminal_date_used": (
                    pd.Timestamp(window["terminal_date"].max()) if len(window) else pd.NaT
                ),
            }
        )
    return rows


def _rank_signal(qualities: np.ndarray) -> np.ndarray:
    series = pd.Series(qualities, dtype=float)
    if series.isna().any():
        raise RuntimeError("MONTHLY_CONTROLLER_QUALITY_NAN")
    ranks = series.rank(method="average", pct=True).to_numpy(dtype=float)
    return 2.0 * (ranks - 0.5)


def _bounded_simplex_toward_target(
    current: np.ndarray, target: np.ndarray
) -> np.ndarray:
    current = np.asarray(current, dtype=float)
    target = np.asarray(target, dtype=float)
    if len(current) != EXPERT_COUNT or len(target) != EXPERT_COUNT:
        raise RuntimeError("MONTHLY_CONTROLLER_WEIGHT_VECTOR_LENGTH")
    if not np.isclose(current.sum(), 1.0, atol=1e-12):
        raise RuntimeError("MONTHLY_CONTROLLER_CURRENT_WEIGHT_SUM")
    target = target / target.sum()

    lower = np.maximum(WEIGHT_FLOOR, current - MAX_ABS_WEIGHT_CHANGE_PER_ASSESSMENT)
    upper = np.minimum(WEIGHT_CEILING, current + MAX_ABS_WEIGHT_CHANGE_PER_ASSESSMENT)
    if lower.sum() > 1.0 + 1e-12 or upper.sum() < 1.0 - 1e-12:
        raise RuntimeError("MONTHLY_CONTROLLER_BOUNDED_SIMPLEX_INFEASIBLE")

    lo, hi = -2.0, 2.0
    for _ in range(100):
        mid = (lo + hi) / 2.0
        candidate = np.clip(target + mid, lower, upper)
        if candidate.sum() > 1.0:
            hi = mid
        else:
            lo = mid
    result = np.clip(target + (lo + hi) / 2.0, lower, upper)
    result = result / result.sum()

    if np.max(np.abs(result - current)) > MAX_ABS_WEIGHT_CHANGE_PER_ASSESSMENT + 1e-10:
        raise RuntimeError("MONTHLY_CONTROLLER_MAX_REWEIGHT_BREACH")
    if result.min() < WEIGHT_FLOOR - 1e-10 or result.max() > WEIGHT_CEILING + 1e-10:
        raise RuntimeError("MONTHLY_CONTROLLER_WEIGHT_BOUND_BREACH")
    return result


def _next_weights(current: np.ndarray, snapshot: list[dict]) -> tuple[np.ndarray, dict]:
    if len(snapshot) != EXPERT_COUNT:
        raise RuntimeError("MONTHLY_CONTROLLER_SNAPSHOT_LENGTH")
    all_available = all(bool(row["available"]) for row in snapshot)
    if not all_available:
        return current.copy(), {
            "updated": False,
            "reason": "WAIT_FOR_ALL_12_MATURED_MONTHS",
            "rank_signal": [np.nan] * EXPERT_COUNT,
        }
    qualities = np.array(
        [float(row["quality_mean_excess_clipped"]) for row in snapshot], dtype=float
    )
    signal = _rank_signal(qualities)
    target_raw = current * np.exp(HEDGE_ETA * signal)
    target = target_raw / target_raw.sum()
    updated = _bounded_simplex_toward_target(current, target)
    return updated, {
        "updated": True,
        "reason": "ONLINE_PAST_PERFORMANCE_UPDATE",
        "rank_signal": signal.tolist(),
    }


def _build_controller_schedule(
    prepared: pd.DataFrame,
    assessment_dates: list[pd.Timestamp],
    *,
    model_id: str,
    contract_id: str,
    historical_pressure: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    weights = np.full(EXPERT_COUNT, INITIAL_WEIGHT, dtype=float)
    states = []
    expert_rows = []

    for assessment_date in assessment_dates:
        snapshot = _expert_snapshot(prepared, assessment_date)
        next_weights, update = _next_weights(weights, snapshot)
        p99_values = np.array(
            [float(row["median_p99"]) if row["available"] else np.nan for row in snapshot],
            dtype=float,
        )
        all_available = np.isfinite(p99_values).all()
        controller_p99 = (
            float(np.dot(next_weights, p99_values)) if all_available else np.nan
        )
        uniform_p99 = float(np.mean(p99_values)) if all_available else np.nan
        controller_replacement = (
            float(historical_pressure * controller_p99)
            if np.isfinite(controller_p99) and controller_p99 > 0
            else np.nan
        )
        uniform_replacement = (
            float(historical_pressure * uniform_p99)
            if np.isfinite(uniform_p99) and uniform_p99 > 0
            else np.nan
        )
        state = {
            "model_id": model_id,
            "contract_id": contract_id,
            "assessment_date": assessment_date,
            "phase": "WARMUP" if assessment_date < LIVE_START else "LIVE_EVALUATION",
            "updated": bool(update["updated"]),
            "update_reason": update["reason"],
            "historical_threshold_over_p99": historical_pressure,
            "controller_weighted_p99": controller_p99,
            "uniform_p99": uniform_p99,
            "controller_replacement_threshold": controller_replacement,
            "uniform_replacement_threshold": uniform_replacement,
            "effective_memory_months": float(
                sum((i + 1) * next_weights[i] for i in range(EXPERT_COUNT))
            ),
            "weight_entropy": float(
                -sum(w * np.log(w) for w in next_weights if w > 0)
            ),
            "max_weight": float(next_weights.max()),
            "max_abs_weight_change": float(np.max(np.abs(next_weights - weights))),
        }
        for i, weight in enumerate(next_weights, 1):
            state[f"w_m{i}"] = float(weight)
        states.append(state)

        for i, row in enumerate(snapshot):
            expert_rows.append(
                {
                    "model_id": model_id,
                    "contract_id": contract_id,
                    "assessment_date": assessment_date,
                    "phase": state["phase"],
                    "weight_before": float(weights[i]),
                    "weight_after": float(next_weights[i]),
                    "rank_signal": update["rank_signal"][i],
                    **row,
                }
            )
        weights = next_weights

    return pd.DataFrame(states), pd.DataFrame(expert_rows)


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
    states = list(schedule.sort_values("assessment_date").itertuples(index=False))
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
                "controller_replacement_threshold": controller_replacement,
                "uniform_replacement_threshold": uniform_replacement,
                "controller_effective_threshold": controller_effective,
                "uniform_effective_threshold": uniform_effective,
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
        raise RuntimeError("MONTHLY_CONTROLLER_NONCAUSAL_THRESHOLD_SOURCE")
    if (frame["controller_effective_threshold"] > frame["raw_threshold"] + 1e-12).any():
        raise RuntimeError("MONTHLY_CONTROLLER_TIGHTENED_THRESHOLD")
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
        raise RuntimeError(f"MONTHLY_CONTROLLER_LEARNED_EXIT_MISSING:{model_id}:{arm}")
    if float(metrics.get("exit_decision_coverage", 0.0)) != 1.0:
        raise RuntimeError(f"MONTHLY_CONTROLLER_EXIT_COVERAGE:{model_id}:{arm}")
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

    historical, historical_audit = _historical_entry(Path(args.historical_entry_predictions))
    causal = _causal_entry(Path(args.causal_entry_predictions))
    combined = _combined_entry(historical, causal)
    contracts, contracts_by_model = _contracts(Path(args.frozen_policy_manifest))
    pressure = _historical_pressure(Path(args.entry_diagnostic_summary))

    tickers = set(combined["ticker"].astype(str))
    prices, price_audit = load_price_panel(Path(args.daily_store_root), tickers)
    if pd.Timestamp(prices["date"].max()).normalize() < LIVE_END:
        raise RuntimeError("MONTHLY_CONTROLLER_PRICES_END_BEFORE_LIVE_END")
    prepared_prices = prepare_prices(prices)
    assessment_dates = _assessment_dates(combined)

    manifest = json.loads(Path(args.frozen_policy_manifest).read_text(encoding="utf-8"))
    models = manifest.get("models", [])
    if len(models) != 10:
        raise RuntimeError(f"MONTHLY_CONTROLLER_TOP10_COUNT:{len(models)}")

    exit_provider = LearnedExitProvider(Path(args.causal_exit_predictions))
    if exit_provider.audit.min_date != str(LIVE_START.date()) or exit_provider.audit.max_date != str(LIVE_END.date()):
        raise RuntimeError(f"MONTHLY_CONTROLLER_CAUSAL_EXIT_COVERAGE:{asdict(exit_provider.audit)}")

    all_states, all_experts, all_daily = [], [], []
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
            raise RuntimeError(f"MONTHLY_CONTROLLER_MODEL_CONTRACT_MAPPING:{model_id}:{len(match)}")
        contract_id = str(match.iloc[0])
        if contract_id not in pressure:
            raise RuntimeError(f"MONTHLY_CONTROLLER_PRESSURE_MISSING:{contract_id}")

        model_rows = _prepare_model_rows(
            combined,
            prices,
            horizon=int(policy.horizon),
            top_fraction=float(policy.top_fraction),
            max_names=int(policy.max_names),
        )
        schedule, expert = _build_controller_schedule(
            model_rows,
            assessment_dates,
            model_id=model_id,
            contract_id=contract_id,
            historical_pressure=pressure[contract_id],
        )
        all_states.append(schedule)
        all_experts.append(expert)

        selected = causal.loc[causal["horizon"].eq(int(policy.horizon))].copy()
        raw = _thresholds(selected, float(policy.score_quantile))
        daily = _live_thresholds(selected, raw, schedule)
        daily.insert(0, "model_id", model_id)
        daily.insert(1, "contract_id", contract_id)
        all_daily.append(daily)

        signals = selected[["decision_date", "ticker", "score"]]
        arm_thresholds = {
            "RAW": daily.set_index("decision_date")["raw_threshold"],
            "UNIFORM_12M": daily.set_index("decision_date")["uniform_effective_threshold"],
            "CONTROLLER_12M": daily.set_index("decision_date")["controller_effective_threshold"],
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
                f"MONTHLY_WEIGHT_CONTROLLER_REPLAY_COMPLETE {model_id} {arm} trades={metrics['trade_count']}",
                flush=True,
            )

    state_frame = pd.concat(all_states, ignore_index=True)
    expert_frame = pd.concat(all_experts, ignore_index=True)
    daily_frame = pd.concat(all_daily, ignore_index=True)
    result_frame = pd.DataFrame(results).sort_values(["arm", "model_id"]).reset_index(drop=True)

    state_frame.to_csv(output / "monthly_controller_state.csv", index=False)
    state_frame.to_parquet(output / "monthly_controller_state.parquet", index=False)
    expert_frame.to_parquet(output / "monthly_controller_expert_evidence.parquet", index=False)
    daily_frame.to_csv(output / "monthly_controller_live_thresholds.csv", index=False)
    daily_frame.to_parquet(output / "monthly_controller_live_thresholds.parquet", index=False)
    result_frame.to_csv(output / "monthly_controller_results.csv", index=False)
    pd.DataFrame(trades).to_csv(output / "monthly_controller_trades.csv", index=False)
    pd.concat(curves, ignore_index=True).to_parquet(output / "monthly_controller_curves.parquet", index=False)

    boundary = (
        state_frame.loc[state_frame["assessment_date"].lt(LIVE_START)]
        .sort_values("assessment_date")
        .groupby("model_id", as_index=False)
        .tail(1)
        .sort_values("model_id")
    )
    boundary.to_csv(output / "controller_weights_at_live_start.csv", index=False)

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

    controller_states = state_frame.loc[state_frame["updated"]]
    summary = {
        "contract_id": CONTRACT_ID,
        "status": "COMPLETE",
        "no_model_training": True,
        "no_prediction_generation": True,
        "uses_existing_17000_causal_artifacts": True,
        "no_threshold_optimization": True,
        "no_weight_optimization": True,
        "final_holdout_opened": False,
        "warmup": {
            "start": str(WARMUP_START.date()),
            "end": "2023-08-10",
            "source": "historical WF_001..WF_007 OOS predictions",
            "performance_interpretation_allowed": False,
            "purpose": "controller_state_calibration_only",
        },
        "evaluation": {
            "start": str(LIVE_START.date()),
            "end": str(LIVE_END.date()),
            "source": "existing causal-expanding entry/exit predictions",
        },
        "controller_contract": {
            "experts": [f"M{i}" for i in range(1, EXPERT_COUNT + 1)],
            "sessions_per_expert": MONTH_SESSIONS,
            "initial_weight_each": INITIAL_WEIGHT,
            "weight_floor": WEIGHT_FLOOR,
            "weight_ceiling": WEIGHT_CEILING,
            "max_abs_weight_change_per_assessment": MAX_ABS_WEIGHT_CHANGE_PER_ASSESSMENT,
            "assessment_every_sessions": ASSESS_EVERY_SESSIONS,
            "first_update_requires_all_12_matured_months": True,
            "quality": "mean realized net excess of model-relative candidates, candidate returns clipped to +/-50%",
            "quality_to_target": "cross-expert percentile rank -> [-1,1], multiplicative exp(HEDGE_ETA * rank_signal)",
            "hedge_eta": HEDGE_ETA,
            "threshold_scale": "historical_threshold_over_p99 * weighted expert median_p99",
            "effective_threshold": "min(raw_threshold, replacement_threshold)",
            "controller_update_effective": "strictly after assessment date",
            "outcome_visibility": "terminal_date <= assessment_date",
        },
        "arm_summary": arm_summary,
        "controller_updates": int(len(controller_states)),
        "historical_prediction_audit": historical_audit,
        "exit_provider_audit": asdict(exit_provider.audit),
        "price_audit": price_audit,
        "arm_selection_from_evaluation_performance_allowed": False,
    }
    _write_json(output / "monthly_controller_summary.json", summary)

    report = result_frame[
        ["model_id", "arm", "trade_count", "cagr_excess", "terminal_wealth_excess_eur"]
    ].copy()
    report["cagr_excess"] = report["cagr_excess"].map(lambda x: f"{float(x):.4%}")
    boundary_cols = ["model_id", "assessment_date", "effective_memory_months", "max_weight"]
    for i in range(1, EXPERT_COUNT + 1):
        boundary_cols.append(f"w_m{i}")
    report_text = [
        "# Top-10 Causal Monthly Weight Controller",
        "",
        "Status: **COMPLETE**",
        "",
        "The controller is initialized at 1/12 for M1..M12 on 2020-09-01.",
        "September 2020 through 2023-08-10 is development warm-up only; no performance claim is made for that interval.",
        "Live evaluation remains 2023-08-11 through 2026-07-24 and reuses the already completed causal prediction artifacts.",
        "",
        "Weights may change by at most 2 percentage points per monthly assessment, remain within 2%-25%, and do not move at all until all 12 fully matured monthly experts exist.",
        "",
        "## Weights entering live evaluation",
        "",
        _markdown(boundary[boundary_cols]),
        "",
        "## Live evaluation",
        "",
        _markdown(report),
        "",
        "RAW, UNIFORM_12M, and CONTROLLER_12M are parallel predeclared arms. No winner may be promoted from this evaluation period.",
        "No models were trained, no predictions were regenerated, no weight or threshold was optimized, and the final holdout remained closed.",
    ]
    (output / "REPORT.md").write_text("\n".join(report_text) + "\n", encoding="utf-8")
    return summary


def self_test() -> None:
    _validate_contract()
    current = np.full(EXPERT_COUNT, INITIAL_WEIGHT)
    target = np.zeros(EXPERT_COUNT)
    target[0] = 1.0
    projected = _bounded_simplex_toward_target(current, target)
    assert np.isclose(projected.sum(), 1.0)
    assert np.max(np.abs(projected - current)) <= MAX_ABS_WEIGHT_CHANGE_PER_ASSESSMENT + 1e-10
    assert projected.min() >= WEIGHT_FLOOR - 1e-10
    assert projected.max() <= WEIGHT_CEILING + 1e-10

    unavailable = [
        {"available": False, "quality_mean_excess_clipped": np.nan}
        for _ in range(EXPERT_COUNT)
    ]
    unchanged, meta = _next_weights(current, unavailable)
    assert np.allclose(unchanged, current)
    assert meta["updated"] is False

    snapshot = []
    for i in range(EXPERT_COUNT):
        snapshot.append(
            {
                "available": True,
                "quality_mean_excess_clipped": float(EXPERT_COUNT - i),
            }
        )
    changed, meta = _next_weights(current, snapshot)
    assert meta["updated"] is True
    assert changed[0] > current[0]
    assert changed[-1] < current[-1]
    assert np.max(np.abs(changed - current)) <= MAX_ABS_WEIGHT_CHANGE_PER_ASSESSMENT + 1e-10
    print("TOP10_CAUSAL_MONTHLY_WEIGHT_CONTROLLER_SELF_TEST_OK")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--historical-entry-predictions")
    parser.add_argument("--causal-entry-predictions")
    parser.add_argument("--causal-exit-predictions")
    parser.add_argument("--frozen-policy-manifest")
    parser.add_argument("--entry-diagnostic-summary")
    parser.add_argument("--retrospective-summary")
    parser.add_argument("--daily-store-root")
    parser.add_argument(
        "--output-root", default="artifacts/top10-causal-monthly-weight-controller"
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
