"""Dynamic-QBD fold-clock refit decomposition.

Development-only matched experiment:

* A: initial model and initial calibration are frozen forever.
* F: the frozen recipe is refit and recalibrated only when fully matured OOS
  fold content expands; model and calibration are frozen between those events.
* M: the same frozen recipe is freshly fit and calibrated at every assessment.

The primary contrasts are F - A and M - F. No recipe switching, performance
trigger, parameter search, promotion, capital authority or holdout access is
allowed.
"""
from __future__ import annotations

import argparse
from datetime import date
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from .contract_fingerprints import stable_hash
from .cost_contracts import CostModel
from .portfolio_policy_contracts import Policy
from .tax_contracts import TaxConfig
from .dynamic_qbd_abc_recalibration_experiment import (
    ARM_A,
    _frozen_a_schedule,
    _parallel_fit_required_sources,
)
from .dynamic_qbd_conditional_refit_experiment import _bind_sources
from .dynamic_qbd_h1_30_adapter import ParquetH130DatasetMaterializer
from .dynamic_qbd_maturity import HorizonMaturityResolver
from .dynamic_qbd_model_combination_experiment import comparison_markdown
from .dynamic_qbd_development_pipeline import materialize_daily_store_prices
from .dynamic_qbd_portfolio_replay import replay_family
from .dynamic_qbd_generation_recalibration import recalibrate_generation
from .dynamic_qbd_recipe_hysteresis import recipe_key
from .dynamic_qbd_recipe_hysteresis_experiment import (
    _causal_recipe_assessments,
    _choice_for_key,
    _choice_key_text,
)
from .dynamic_qbd_runtime_resources import active_cpu_contract, configure_cpu_peak
from .dynamic_qbd_family_surface import build_family_specs
from .dynamic_qbd_wealth_metrics import wealth_path_metrics


AUTHORITY = "SHADOW_ONLY_NO_PROMOTION_NO_HOLDOUT"
CLOSED_HOLDOUT_START = date(2026, 7, 25)
SCHEMA_VERSION = "DYNAMIC_QBD_FOLD_CLOCK_REFIT_V1"

ARM_F = "F_FOLD_CLOCK_REFIT_FROZEN_BETWEEN_FOLDS"
ARM_M = "M_MONTHLY_REFIT_FROZEN_RECIPE"
ARMS = (ARM_A, ARM_F, ARM_M)


def _fold_ids(choice: dict[str, Any]) -> tuple[str, ...]:
    values = choice.get("fold_ids", ())
    result = tuple(sorted(str(value) for value in values))
    if int(choice.get("fold_count", len(result))) != len(result):
        raise AssertionError("FOLD_CLOCK_FOLD_COUNT_FOLD_IDS_MISMATCH")
    if len(set(result)) != len(result):
        raise AssertionError("FOLD_CLOCK_DUPLICATE_FOLD_ID")
    return result


def _fold_evidence_fingerprint(fold_ids: tuple[str, ...]) -> str:
    """Fingerprint only the matured fold content; calendar dates are excluded."""
    canonical_fold_ids = tuple(sorted(str(value) for value in fold_ids))
    return stable_hash({
        "fold_count": len(canonical_fold_ids),
        "fold_ids": list(canonical_fold_ids),
    })


def _build_plans(
    assessments: list[dict],
) -> tuple[dict[str, list[dict]], list[dict]]:
    if not assessments:
        raise ValueError("FOLD_CLOCK_ASSESSMENTS_EMPTY")
    frozen_key = recipe_key(assessments[0]["selector_winner"])
    plans = {ARM_A: [], ARM_F: [], ARM_M: []}
    audit: list[dict] = []
    previous_fold_ids: tuple[str, ...] | None = None

    for index, assessment in enumerate(assessments):
        choice = _choice_for_key(assessment["eligible_choices"], frozen_key)
        fold_ids = _fold_ids(choice)
        fingerprint = _fold_evidence_fingerprint(fold_ids)
        if previous_fold_ids is None:
            evidence_expanded = True
        elif fold_ids == previous_fold_ids:
            evidence_expanded = False
        elif set(previous_fold_ids).issubset(set(fold_ids)) and len(fold_ids) > len(previous_fold_ids):
            evidence_expanded = True
        else:
            raise AssertionError("FOLD_CLOCK_NON_MONOTONIC_FOLD_EVIDENCE")
        initial = index == 0
        f_refit = bool(initial or evidence_expanded)
        common = {
            "activation_cutoff": assessment["assessment_date"],
            "latest_matured_evidence_date": assessment["latest_matured_evidence_date"],
            "evidence_fingerprint": fingerprint,
            "evidence_expanded": evidence_expanded,
            "fold_count": len(fold_ids),
            "fold_ids": fold_ids,
            "choice": dict(choice),
        }
        plans[ARM_A].append({
            **common,
            "refit_required": initial,
            "refit_reason": "INITIAL_FIT" if initial else "FROZEN_MODEL_FROZEN_CALIBRATION",
        })
        plans[ARM_F].append({
            **common,
            "refit_required": f_refit,
            "refit_reason": (
                "INITIAL_FIT" if initial else
                "NEW_MATURED_OOS_FOLD" if evidence_expanded else
                "FROZEN_LAST_FIT_CALIBRATION_REUSED"
            ),
        })
        plans[ARM_M].append({
            **common,
            "refit_required": True,
            "refit_reason": "MONTHLY_FRESH_REFIT_CONTROL",
        })
        audit.append({
            "assessment_date": assessment["assessment_date"].isoformat(),
            "latest_matured_evidence_date": assessment["latest_matured_evidence_date"].isoformat(),
            "fold_count": len(fold_ids),
            "fold_ids": json.dumps(list(fold_ids), sort_keys=True),
            "evidence_fingerprint": fingerprint,
            "evidence_expanded": evidence_expanded,
            "f_refit": f_refit,
            "f_refit_reason": plans[ARM_F][-1]["refit_reason"],
            "m_refit": True,
            "frozen_recipe": _choice_key_text(choice),
        })
        previous_fold_ids = fold_ids
    return plans, audit


def _fold_clock_signals_schedule(
    *,
    arm: str,
    plan: list[dict],
    sources: dict,
    sessions: tuple[date, ...],
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    maturity = HorizonMaturityResolver(sessions)
    signals_by_model: dict[str, pd.DataFrame] = {}
    schedule_rows: list[dict] = []
    calibration_audit: list[dict] = []
    active_generation: dict[str, Any] | None = None
    active_resolved: Any | None = None
    active_source: dict[str, Any] | None = None

    for row in plan:
        source = sources[row["source_key"]]
        cutoff = row["activation_cutoff"]
        if row["refit_required"]:
            family = source["family"]
            builder = source["builder"]
            build = source["build"]
            frame = builder.calibration_predictions(
                family=family,
                build=build,
                information_cutoff=cutoff,
            )
            generation_id = stable_hash({
                "schema_version": SCHEMA_VERSION,
                "arm": arm,
                "model_artifact_id": build["model_artifact_id"],
                "model_fit_cutoff": source["fit_cutoff"],
                "calibration_contract": "ONE_CALIBRATION_PER_FOLD_CLOCK_FIT",
            })[:24]
            active_resolved = recalibrate_generation(
                family,
                generation_id,
                frame,
                information_cutoff=cutoff,
                maturity=maturity,
            )
            active_generation = {
                "activation_date": cutoff,
                "family_id": arm,
                "generation_id": generation_id,
                "model_artifact_id": str(build["model_artifact_id"]),
                "resolved_threshold": float(active_resolved.resolved_threshold),
                "resolved_top_fraction": float(active_resolved.resolved_top_fraction),
                "entry_policy_id": f"{arm}_ENTRY_{cutoff}",
                "exit_policy_id": "FIXED_D2",
            }
            active_source = source
            schedule_rows.append(dict(active_generation))
        if active_generation is None or active_resolved is None or active_source is None:
            raise AssertionError(f"FOLD_CLOCK_ACTIVE_GENERATION_MISSING:{arm}")

        model_artifact_id = str(active_source["build"]["model_artifact_id"])
        signals_by_model.setdefault(model_artifact_id, active_source["predictions"])
        refit = bool(row["refit_required"])
        calibration_audit.append({
            "arm": arm,
            "assessment_date": cutoff.isoformat(),
            "model_fit_cutoff": active_source["fit_cutoff"].isoformat(),
            "model_age_calendar_days": int((cutoff - active_source["fit_cutoff"]).days),
            "active_recipe": _choice_key_text(active_source["choice"]),
            "model_artifact_id": model_artifact_id,
            "calibration_source": (
                "FOLD_CLOCK_FIT_GENERATION_CALIBRATION" if refit
                else "FROZEN_LAST_FIT_CALIBRATION_REUSED"
            ),
            "resolved_threshold": float(active_resolved.resolved_threshold),
            "resolved_top_fraction": float(active_resolved.resolved_top_fraction),
            "matured_observation_count": int(active_resolved.observations),
            "refit_this_assessment": refit,
            "calibration_resolution_performed": refit,
            "refit_reason": str(row["refit_reason"]),
            "evidence_fingerprint": row["evidence_fingerprint"],
            "evidence_expanded": bool(row.get("evidence_expanded", False)),
            "fold_count": int(row["fold_count"]),
            "fold_ids": json.dumps(list(row["fold_ids"]), sort_keys=True),
        })

    signals = pd.concat(list(signals_by_model.values()), ignore_index=True)
    if signals.duplicated(["decision_date", "ticker", "model_artifact_id"]).any():
        raise ValueError(f"FOLD_CLOCK_DUPLICATE_MODEL_SIGNAL:{arm}")
    return signals, pd.DataFrame(schedule_rows), calibration_audit


def _distinct_model_count(schedule: pd.DataFrame) -> int:
    return int(schedule["model_artifact_id"].astype(str).nunique())


def _contract_audit(
    *,
    plans: dict[str, list[dict]],
    sources: dict,
    schedules: dict[str, pd.DataFrame],
    audits: dict[str, list[dict]],
    assessment_count: int,
) -> dict:
    initial_keys = {arm: plans[arm][0]["source_key"] for arm in ARMS}
    initial_ids = {
        arm: str(sources[initial_keys[arm]]["build"]["model_artifact_id"])
        for arm in ARMS
    }
    recipe_keys = {
        arm: {recipe_key(row["choice"]) for row in plans[arm]}
        for arm in ARMS
    }
    initial_thresholds = {
        arm: float(schedules[arm].iloc[0]["resolved_threshold"])
        for arm in ARMS
    }
    initial_top_fractions = {
        arm: float(schedules[arm].iloc[0]["resolved_top_fraction"])
        for arm in ARMS
    }
    same_initial_model = len(set(initial_ids.values())) == 1
    same_recipe = all(len(value) == 1 for value in recipe_keys.values()) and len({next(iter(value)) for value in recipe_keys.values()}) == 1
    same_initial_threshold = all(
        math.isclose(initial_thresholds[ARM_A], initial_thresholds[arm], rel_tol=1e-12, abs_tol=1e-12)
        for arm in ARMS
    )
    same_initial_top_fraction = all(
        math.isclose(initial_top_fractions[ARM_A], initial_top_fractions[arm], rel_tol=1e-12, abs_tol=1e-12)
        for arm in ARMS
    )
    f_audit = audits[ARM_F]
    f_expansions = [bool(row["evidence_expanded"]) for row in f_audit]
    f_refits = [bool(row["refit_this_assessment"]) for row in f_audit]
    f_refit_matches_clock = all(refit == (index == 0 or expanded) for index, (refit, expanded) in enumerate(zip(f_refits, f_expansions)))
    f_refit_count = sum(f_refits)
    f_calibration_count = sum(bool(row["calibration_resolution_performed"]) for row in f_audit)
    f_between_refits_frozen = True
    f_new_artifact_at_each_refit = True
    schedule_numeric_finite = True
    previous = None
    for schedule in schedules.values():
        for column in ("resolved_threshold", "resolved_top_fraction"):
            schedule_numeric_finite = schedule_numeric_finite and all(
                math.isfinite(float(value)) for value in schedule[column].tolist()
            )
    for row in f_audit:
        state = (row["model_artifact_id"], float(row["resolved_threshold"]), float(row["resolved_top_fraction"]))
        if not bool(row["refit_this_assessment"]) and previous is not None and state != previous:
            f_between_refits_frozen = False
        if bool(row["refit_this_assessment"]):
            if previous is not None and state[0] == previous[0]:
                f_new_artifact_at_each_refit = False
            previous = state
    all_dates_equal = len({tuple(row["assessment_date"] for row in audits[arm]) for arm in ARMS}) == 1
    checks = {
        "final_holdout_opened_false": True,
        "promotion_allowed_false": True,
        "same_initial_recipe_a_f_m": same_recipe,
        "same_initial_model_artifact_a_f_m": same_initial_model,
        "same_initial_threshold_a_f_m": same_initial_threshold,
        "same_initial_top_fraction_a_f_m": same_initial_top_fraction,
        "recipe_identity_constant_all_arms": same_recipe,
        "a_fresh_fit_count_one": sum(bool(row["refit_required"]) for row in plans[ARM_A]) == 1,
        "a_calibration_resolution_count_one": sum(bool(row["calibration_resolution_performed"]) for row in audits[ARM_A]) == 1,
        "a_model_artifact_count_one": _distinct_model_count(schedules[ARM_A]) == 1,
        "f_refits_only_on_evidence_expansion": f_refit_matches_clock,
        "f_refits_every_fold_expansion_once": f_refit_count == 1 + sum(f_expansions[1:]),
        "f_new_model_artifact_at_each_refit": f_new_artifact_at_each_refit,
        "f_between_refits_model_and_calibration_frozen": f_between_refits_frozen,
        "f_calibration_count_equals_fresh_fit_count": f_calibration_count == f_refit_count,
        "m_fresh_fit_count_equals_assessments": sum(bool(row["refit_required"]) for row in plans[ARM_M]) == assessment_count,
        "m_calibration_resolution_count_equals_assessments": sum(bool(row["calibration_resolution_performed"]) for row in audits[ARM_M]) == assessment_count,
        "m_distinct_model_artifact_count_equals_assessments": _distinct_model_count(schedules[ARM_M]) == assessment_count,
        "all_arms_identical_evaluation_dates": all_dates_equal,
        "identical_portfolio_cost_benchmark_contract": True,
        "all_schedule_calibration_values_finite": schedule_numeric_finite,
    }
    failed = sorted(name for name, value in checks.items() if not value)
    if failed:
        raise AssertionError(f"FOLD_CLOCK_CONTRACT_FAILED:{failed}")
    return {
        "checks": checks,
        "initial_model_artifact_ids": initial_ids,
        "initial_thresholds": initial_thresholds,
        "initial_top_fractions": initial_top_fractions,
        "assessment_count": assessment_count,
        "f_refit_count": f_refit_count,
        "f_evidence_expansion_count": sum(f_expansions),
    }


def _result_row(arm: str, result: dict, plan: list[dict], schedule: pd.DataFrame) -> dict:
    metrics = result["metrics"]
    risk = wealth_path_metrics(result["curve"])
    return {
        "strategy": arm,
        "terminal_value": float(metrics["terminal_value"]),
        "urth_terminal_value": float(metrics["urth_terminal_value"]),
        "terminal_excess_eur": float(metrics["terminal_value"] - metrics["urth_terminal_value"]),
        "terminal_relative_return": float(metrics["terminal_value"] / metrics["urth_terminal_value"] - 1.0),
        "cagr": float(metrics.get("cagr", float("nan"))),
        "urth_cagr": float(metrics.get("urth_cagr", float("nan"))),
        "cagr_excess": float(metrics.get("cagr_excess", float("nan"))),
        "trade_count": int(metrics["trade_count"]),
        "total_cost_eur": float(metrics["total_cost_eur"]),
        "relative_max_drawdown": float(risk["relative_max_drawdown"]),
        "fresh_fit_count": int(sum(bool(row["refit_required"]) for row in plan)),
        "calibration_resolution_count": int(len(schedule)),
        "distinct_model_artifact_count": _distinct_model_count(schedule),
        "recipe_change_count": 0,
        "model_family_switch_count": 0,
    }


def _replay(signals: pd.DataFrame, schedule: pd.DataFrame, prices: pd.DataFrame, *, arm: str, start: date, end: date, horizon: int, holding_days: int, max_names: int, top_fraction: float, initial: float) -> dict:
    return replay_family(
        signals=signals,
        prices=prices,
        policy=Policy(horizon=horizon, score_quantile=.5, top_fraction=top_fraction, max_names=max_names, holding_days=holding_days, sleeve=.5),
        generation_schedule=schedule,
        cost=CostModel(20.0),
        tax=TaxConfig(False),
        start=pd.Timestamp(start),
        end=pd.Timestamp(end),
        initial=initial,
    )


def _contrast(candidate: dict, baseline: dict) -> dict:
    return {
        "candidate": candidate["strategy"],
        "baseline": baseline["strategy"],
        "terminal_value_delta_eur": float(candidate["terminal_value"] - baseline["terminal_value"]),
        "terminal_relative_return_delta": float(candidate["terminal_relative_return"] - baseline["terminal_relative_return"]),
        "cagr_delta": float(candidate["cagr"] - baseline["cagr"]),
        "cagr_excess_delta": float(candidate["cagr_excess"] - baseline["cagr_excess"]),
        "relative_max_drawdown_delta": float(candidate["relative_max_drawdown"] - baseline["relative_max_drawdown"]),
        "trade_count_delta": int(candidate["trade_count"] - baseline["trade_count"]),
        "total_cost_delta_eur": float(candidate["total_cost_eur"] - baseline["total_cost_eur"]),
    }


def _pair_diagnosis(candidate: dict, baseline: dict, label: str) -> str:
    if (
        math.isclose(candidate["terminal_value"], baseline["terminal_value"], rel_tol=1e-12, abs_tol=1e-9)
        and math.isclose(candidate["relative_max_drawdown"], baseline["relative_max_drawdown"], rel_tol=1e-12, abs_tol=1e-9)
    ):
        return f"{label}_MATCH"
    return_better = candidate["terminal_value"] > baseline["terminal_value"]
    risk_better = candidate["relative_max_drawdown"] > baseline["relative_max_drawdown"]
    return_worse = candidate["terminal_value"] < baseline["terminal_value"]
    risk_worse = candidate["relative_max_drawdown"] < baseline["relative_max_drawdown"]
    if return_better and risk_better:
        return f"{label}_PARETO_IMPROVES"
    if return_worse and risk_worse:
        return f"{label}_PARETO_WORSENS"
    return f"{label}_TRADEOFF"


def run_experiment(
    *, signal_panel: str | Path, candidate_metrics: str | Path, daily_store_root: str | Path, output_root: str | Path,
    benchmark_daily_path: str | Path | None = None, direct_daily_stock_root: str | Path | None = None,
    start: date = date(2020, 8, 31), end: date = date(2023, 12, 29), horizon: int = 3,
    holding_days: int = 2, max_names: int = 1, score_quantile: float = .75, top_fraction: float = .01,
    initial: float = 10000.0, code_commit: str = "UNSPECIFIED_LOCAL_WORKTREE",
) -> dict:
    if end >= CLOSED_HOLDOUT_START:
        raise ValueError("FOLD_CLOCK_FINAL_HOLDOUT_MUST_REMAIN_CLOSED")
    if start > end:
        raise ValueError("FOLD_CLOCK_START_AFTER_END")
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    signal_panel = Path(signal_panel)
    metrics_path = Path(candidate_metrics)
    metric_rows = json.loads(metrics_path.read_text(encoding="utf-8"))
    materializer = ParquetH130DatasetMaterializer(signal_panel, metrics_path, end)
    base = next(
        x for x in build_family_specs(feature_schema_sha256=materializer.feature_schema_fingerprint, score_quantile=score_quantile, top_fraction=top_fraction)
        if x.horizon_sessions == horizon and x.holding_days == holding_days and x.max_names == max_names and x.exit_policy["family"] == "FIXED"
    )
    sessions = tuple(sorted(pd.to_datetime(pd.read_parquet(signal_panel, columns=["decision_date"])["decision_date"]).dt.date.unique()))
    assessments = _causal_recipe_assessments(
        family=base, materializer=materializer, metrics_path=metrics_path, sessions=sessions, end=end, root=output_root / "assessment-audit"
    )
    assessments = [row for row in assessments if row["assessment_date"] <= end]
    plans, plan_audit = _build_plans(assessments)
    pd.DataFrame(plan_audit).to_csv(output_root / "plan-audit.csv", index=False)
    sources, fit_audit, worker_telemetry = _parallel_fit_required_sources(
        plans=plans, base=base, metric_rows=metric_rows, signal_panel=signal_panel, sessions=sessions,
        output_root=output_root, end=end, horizon=horizon,
    )
    _bind_sources(plans, sources)
    (output_root / "fit-audit.json").write_text(json.dumps(fit_audit, default=str, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    a_signals, a_schedule, a_audit, _ = _frozen_a_schedule(plan=plans[ARM_A], sources=sources, sessions=sessions)
    f_signals, f_schedule, f_audit = _fold_clock_signals_schedule(arm=ARM_F, plan=plans[ARM_F], sources=sources, sessions=sessions)
    m_signals, m_schedule, m_audit = _fold_clock_signals_schedule(arm=ARM_M, plan=plans[ARM_M], sources=sources, sessions=sessions)
    signals = {ARM_A: a_signals, ARM_F: f_signals, ARM_M: m_signals}
    schedules = {ARM_A: a_schedule, ARM_F: f_schedule, ARM_M: m_schedule}
    audits = {ARM_A: a_audit, ARM_F: f_audit, ARM_M: m_audit}
    contract = _contract_audit(plans=plans, sources=sources, schedules=schedules, audits=audits, assessment_count=len(assessments))
    (output_root / "contract-audit.json").write_text(json.dumps(contract, default=str, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    fold_clock_events = pd.DataFrame([
        {
            "assessment_date": row["assessment_date"],
            "latest_matured_evidence_date": row["latest_matured_evidence_date"],
            "fold_count": row["fold_count"],
            "fold_ids": row["fold_ids"],
            "evidence_fingerprint": row["evidence_fingerprint"],
            "evidence_expanded": row["evidence_expanded"],
            "f_refit": row["f_refit"],
            "f_refit_reason": row["f_refit_reason"],
            "active_model_artifact_id": audit_row["model_artifact_id"],
            "model_fit_cutoff": audit_row["model_fit_cutoff"],
            "model_age_days": audit_row["model_age_calendar_days"],
            "resolved_threshold": audit_row["resolved_threshold"],
            "resolved_top_fraction": audit_row["resolved_top_fraction"],
        }
        for row, audit_row in zip(plan_audit, f_audit)
    ])
    fold_clock_events.to_csv(output_root / "fold-clock-events.csv", index=False)

    first_activation = plans[ARM_A][0]["activation_cutoff"]
    evaluation_start = max(start, first_activation)
    prices_path = materialize_daily_store_prices(
        daily_store_root=daily_store_root, signal_panel=signal_panel, start=first_activation, end=end,
        output_path=output_root / "inputs" / "prices.parquet", benchmark_daily_path=benchmark_daily_path, direct_daily_stock_root=direct_daily_stock_root,
    )
    prices = pd.read_parquet(prices_path)
    rows: list[dict] = []
    for arm in ARMS:
        result = _replay(signals[arm], schedules[arm], prices, arm=arm, start=evaluation_start, end=end, horizon=horizon, holding_days=holding_days, max_names=max_names, top_fraction=top_fraction, initial=initial)
        rows.append(_result_row(arm, result, plans[arm], schedules[arm]))
        schedules[arm].to_csv(output_root / f"{arm}-schedule.csv", index=False)
        pd.DataFrame(result["trades"]).to_parquet(output_root / f"{arm}-trades.parquet", index=False)
        curve = result["curve"].copy()
        curve["strategy"] = arm
        curve.to_parquet(output_root / f"{arm}-nav.parquet", index=False)

    result_numeric_fields = (
        "terminal_value", "urth_terminal_value", "terminal_excess_eur", "terminal_relative_return",
        "cagr", "urth_cagr", "cagr_excess", "relative_max_drawdown", "total_cost_eur",
    )
    results_finite = all(
        math.isfinite(float(row[field]))
        for row in rows
        for field in result_numeric_fields
    )
    if not results_finite:
        raise AssertionError("FOLD_CLOCK_NONFINITE_RESULT_METRIC")
    contract["checks"]["all_result_metrics_finite"] = results_finite
    (output_root / "contract-audit.json").write_text(json.dumps(contract, default=str, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    pd.DataFrame([row for arm in ARMS for row in audits[arm]]).to_csv(output_root / "calibration-audit.csv", index=False)
    comparison = pd.DataFrame(rows).sort_values("terminal_value", ascending=False)
    comparison.to_csv(output_root / "portfolio-value-comparison.csv", index=False)
    by_name = {row["strategy"]: row for row in rows}
    contrasts = {
        "F_MINUS_A_SPARSE_FOLD_CLOCK_REFIT_EFFECT": _contrast(by_name[ARM_F], by_name[ARM_A]),
        "M_MINUS_F_MONTHLY_VS_FOLD_CLOCK_REFIT_FREQUENCY_EFFECT": _contrast(by_name[ARM_M], by_name[ARM_F]),
    }
    (output_root / "causal-contrasts.json").write_text(json.dumps(contrasts, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    diagnosis = {
        "fold_clock_vs_frozen": _pair_diagnosis(by_name[ARM_F], by_name[ARM_A], "FOLD_CLOCK_REFIT"),
        "monthly_vs_fold_clock": _pair_diagnosis(by_name[ARM_M], by_name[ARM_F], "MONTHLY_REFIT_VS_FOLD_CLOCK"),
    }
    summary = {
        "schema_version": SCHEMA_VERSION, "status": "COMPLETE", "authority": AUTHORITY,
        "code_commit": code_commit, "promotion_allowed": False, "final_holdout_opened": False,
        "recipe_parameter_grid_searched": False, "experiment": {
            "horizon": horizon, "holding_days": holding_days, "max_names": max_names,
            "score_quantile": score_quantile, "top_fraction": top_fraction, "sleeve": .5,
            "round_trip_cost_bps": 20, "benchmark": "URTH", "initial_value_eur": initial,
            "evaluation_start": evaluation_start.isoformat(), "evaluation_end": end.isoformat(), "exit": "FIXED_D2",
        },
        "arm_contract": {
            ARM_A: "INITIAL_MODEL_AND_INITIAL_CALIBRATION_FROZEN_FOREVER",
            ARM_F: "REFIT_AND_RECALIBRATE_ONLY_ON_NEW_FULLY_MATURED_OOS_FOLD_CONTENT",
            ARM_M: "SAME_FROZEN_RECIPE_FRESHLY_FIT_AND_CALIBRATED_EVERY_ASSESSMENT",
        },
        "primary_contrasts": {"F_MINUS_A": "SPARSE_FOLD_CLOCK_REFIT_EFFECT", "M_MINUS_F": "MONTHLY_REFIT_FREQUENCY_EFFECT"},
        "assessment_count": len(assessments), "contract_audit": contract, "diagnosis": diagnosis,
        "causal_contrasts": contrasts, "portfolio_value_comparison": comparison.to_dict(orient="records"),
        "parallel_execution": worker_telemetry, "execution_resources": active_cpu_contract(),
    }
    (output_root / "summary.json").write_text(json.dumps(summary, default=str, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "code_commit": code_commit,
        "authority": AUTHORITY,
        "final_holdout_opened": False,
        "promotion_allowed": False,
        "generated_files": [
            "summary.json", "REPORT.md", "portfolio-value-comparison.csv", "causal-contrasts.json",
            "contract-audit.json", "fold-clock-events.csv", "fit-audit.json", "calibration-audit.csv",
            "plan-audit.csv", "A_FROZEN_MODEL_FROZEN_CALIBRATION-schedule.csv",
            "F_FOLD_CLOCK_REFIT_FROZEN_BETWEEN_FOLDS-schedule.csv",
            "M_MONTHLY_REFIT_FROZEN_RECIPE-schedule.csv",
            "A_FROZEN_MODEL_FROZEN_CALIBRATION-nav.parquet",
            "A_FROZEN_MODEL_FROZEN_CALIBRATION-trades.parquet",
            "F_FOLD_CLOCK_REFIT_FROZEN_BETWEEN_FOLDS-nav.parquet",
            "F_FOLD_CLOCK_REFIT_FROZEN_BETWEEN_FOLDS-trades.parquet",
            "M_MONTHLY_REFIT_FROZEN_RECIPE-nav.parquet",
            "M_MONTHLY_REFIT_FROZEN_RECIPE-trades.parquet",
            "worker-telemetry.jsonl",
        ],
        "parallel_execution": worker_telemetry,
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, default=str, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = [
        "# Dynamic-QBD Fold-Clock Refit Suite", "",
        f"Fold-clock vs frozen diagnosis: **{diagnosis['fold_clock_vs_frozen']}**",
        f"Monthly vs fold-clock diagnosis: **{diagnosis['monthly_vs_fold_clock']}**", "",
        f"Authority: `{AUTHORITY}`. The final holdout remains closed.", "",
        "## Scientific contract", "",
        f"- `{ARM_A}` freezes its first causal model and calibration for the complete replay.",
        f"- `{ARM_F}` uses the same frozen recipe and refits only when matured fold content expands; calibration is frozen between those events.",
        f"- `{ARM_M}` uses the same frozen recipe but freshly fits and calibrates every assessment.",
        "- Fold evidence is fingerprinted from sorted fold IDs and fold count only; calendar progress alone is not evidence.",
        "- No recipe switching, performance trigger, parameter search, promotion, capital authority or holdout access.", "",
        "## Portfolio comparison", "", comparison_markdown(comparison), "",
        "## Causal contrasts", "",
        f"- F-A terminal delta: {contrasts['F_MINUS_A_SPARSE_FOLD_CLOCK_REFIT_EFFECT']['terminal_value_delta_eur']:+.2f} EUR; CAGR delta: {contrasts['F_MINUS_A_SPARSE_FOLD_CLOCK_REFIT_EFFECT']['cagr_delta']:+.4%}; relative-MaxDD delta: {contrasts['F_MINUS_A_SPARSE_FOLD_CLOCK_REFIT_EFFECT']['relative_max_drawdown_delta']:+.4%}.",
        f"- M-F terminal delta: {contrasts['M_MINUS_F_MONTHLY_VS_FOLD_CLOCK_REFIT_FREQUENCY_EFFECT']['terminal_value_delta_eur']:+.2f} EUR; CAGR delta: {contrasts['M_MINUS_F_MONTHLY_VS_FOLD_CLOCK_REFIT_FREQUENCY_EFFECT']['cagr_delta']:+.4%}; relative-MaxDD delta: {contrasts['M_MINUS_F_MONTHLY_VS_FOLD_CLOCK_REFIT_FREQUENCY_EFFECT']['relative_max_drawdown_delta']:+.4%}.",
        "", "The fold-clock event ledger and calibration audit explicitly show frozen months between matured-fold events.",
        f"Parallel fit execution: {worker_telemetry['active_workers']} active workers, queue capacity {worker_telemetry['queue_capacity']}, {worker_telemetry['jobs_completed']} completed fit tasks; see `worker-telemetry.jsonl`.",
        "This is Development/Research evidence only.",
    ]
    (output_root / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return summary


def _parse_date(value: str) -> date:
    return date.fromisoformat(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--signal-panel", required=True)
    parser.add_argument("--candidate-metrics", required=True)
    parser.add_argument("--daily-store-root", required=True)
    parser.add_argument("--benchmark-daily-path")
    parser.add_argument("--direct-daily-stock-root")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--start", type=_parse_date, default=date(2020, 8, 31))
    parser.add_argument("--end", type=_parse_date, default=date(2023, 12, 29))
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--holding-days", type=int, default=2)
    parser.add_argument("--max-names", type=int, default=1)
    parser.add_argument("--score-quantile", type=float, default=.75)
    parser.add_argument("--top-fraction", type=float, default=.01)
    parser.add_argument("--initial", type=float, default=10000.0)
    parser.add_argument("--code-commit", default="UNSPECIFIED_LOCAL_WORKTREE")
    args = parser.parse_args()
    configure_cpu_peak(process_workers=32, native_threads_per_worker=1)
    result = run_experiment(
        signal_panel=args.signal_panel, candidate_metrics=args.candidate_metrics, daily_store_root=args.daily_store_root,
        output_root=args.output_root, benchmark_daily_path=args.benchmark_daily_path, direct_daily_stock_root=args.direct_daily_stock_root,
        start=args.start, end=args.end, horizon=args.horizon, holding_days=args.holding_days, max_names=args.max_names,
        score_quantile=args.score_quantile, top_fraction=args.top_fraction, initial=args.initial, code_commit=args.code_commit,
    )
    print(json.dumps(result["portfolio_value_comparison"], indent=2))


if __name__ == "__main__":
    main()
