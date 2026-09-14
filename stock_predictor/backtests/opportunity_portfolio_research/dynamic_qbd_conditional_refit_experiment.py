"""Shadow-only Dynamic-QBD conditional-refit experiment.

The experiment separates model-refresh timing from monthly recalibration:

B  initial model fit once, then rolling recalibration only;
C  same frozen recipe freshly refit every month (pure refit control);
D0 refit only when a real new OOS-fold evidence expansion changes recipe;
D1 refit only after two independent OOS-fold evidence expansions confirm the
   same exact challenger with a strict robust-score edge.

No parameter grid, holdout access, promotion or capital authority is permitted.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import date
import json
import math
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from .contract_fingerprints import stable_hash
from .cost_contracts import CostModel
from .portfolio_policy_contracts import Policy
from .tax_contracts import TaxConfig
from .dynamic_qbd_h1_30_adapter import H130ProductionGenerationBuilder, ParquetH130DatasetMaterializer
from .dynamic_qbd_maturity import HorizonMaturityResolver
from .dynamic_qbd_model_combination_experiment import comparison_markdown
from .dynamic_qbd_development_pipeline import materialize_daily_store_prices
from .dynamic_qbd_portfolio_replay import replay_family
from .dynamic_qbd_generation_recalibration import recalibrate_generation
from .dynamic_qbd_recipe_hysteresis import (
    HysteresisConfig,
    RecipeHysteresisState,
    decision_to_dict,
    recipe_key,
    state_to_dict,
    update_recipe_hysteresis,
)
from .dynamic_qbd_recipe_hysteresis_experiment import (
    _causal_recipe_assessments,
    _choice_for_key,
    _choice_key_text,
    _family_for_choice,
    _materializer_for_choice,
)
from .dynamic_qbd_runtime_resources import active_cpu_contract, configure_cpu_peak
from .dynamic_qbd_family_surface import build_family_specs
from .dynamic_qbd_wealth_metrics import wealth_path_metrics


AUTHORITY = "SHADOW_ONLY_NO_PROMOTION_NO_HOLDOUT"
CLOSED_HOLDOUT_START = date(2026, 7, 25)
SCHEMA_VERSION = "DYNAMIC_QBD_CONDITIONAL_REFIT_HYSTERESIS_V2"

ARM_B = "B_FROZEN_MODEL_ROLLING_RECALIBRATION"
ARM_C = "C_MONTHLY_REFIT_FROZEN_RECIPE"
ARM_D0 = "D0_NEW_EVIDENCE_RECIPE_CHANGE_REFIT"
ARM_D1 = "D1_HYSTERESIS_GATED_CONDITIONAL_REFIT"
ARMS = (ARM_B, ARM_C, ARM_D0, ARM_D1)


def _build_arm_plans(
    assessments: list[dict], *, config: HysteresisConfig
) -> tuple[dict[str, list[dict]], list[dict], list[dict], RecipeHysteresisState]:
    if not assessments:
        raise ValueError("CONDITIONAL_REFIT_ASSESSMENTS_EMPTY")

    plans = {arm: [] for arm in ARMS}
    state = RecipeHysteresisState()
    frozen_key = None
    d0_key = None
    previous_fingerprint = None
    hysteresis_rows: list[dict] = []
    audit_rows: list[dict] = []

    for assessment in assessments:
        choices = assessment["eligible_choices"]
        winner = assessment["selector_winner"]
        winner_key = recipe_key(winner)
        fingerprint = str(assessment["evidence_fingerprint"])
        evidence_expanded = previous_fingerprint is None or fingerprint != previous_fingerprint

        state, hysteresis_decision = update_recipe_hysteresis(
            state=state,
            assessment_date=assessment["assessment_date"],
            evidence_date=assessment["latest_matured_evidence_date"],
            evidence_fingerprint=fingerprint,
            selector_winner=winner,
            eligible_choices=choices,
            config=config,
        )

        initial = frozen_key is None
        previous_d0_key = d0_key
        if initial:
            frozen_key = winner_key
            d0_key = winner_key
        elif evidence_expanded and winner_key != d0_key:
            d0_key = winner_key

        frozen = _choice_for_key(choices, frozen_key)
        d0_choice = _choice_for_key(choices, d0_key)
        d1_choice = _choice_for_key(choices, state.incumbent)
        d0_refit = bool(initial or (evidence_expanded and previous_d0_key is not None and d0_key != previous_d0_key))
        d1_refit = bool(initial or hysteresis_decision.switch_allowed)

        common = {
            "activation_cutoff": assessment["assessment_date"],
            "latest_matured_evidence_date": assessment["latest_matured_evidence_date"],
            "evidence_fingerprint": fingerprint,
            "evidence_expanded": bool(evidence_expanded),
            "selector_winner": dict(winner),
        }
        plans[ARM_B].append({
            **common,
            "choice": frozen,
            "refit_required": bool(initial),
            "refit_reason": "INITIAL_FIT" if initial else "ROLLING_RECALIBRATION_ONLY",
        })
        plans[ARM_C].append({
            **common,
            "choice": frozen,
            "refit_required": True,
            "refit_reason": "MONTHLY_FROZEN_RECIPE_REFIT_CONTROL",
        })
        plans[ARM_D0].append({
            **common,
            "choice": d0_choice,
            "refit_required": d0_refit,
            "refit_reason": (
                "INITIAL_FIT" if initial else
                "NO_NEW_RECIPE_EVIDENCE" if not evidence_expanded else
                "NEW_FOLD_EVIDENCE_RECIPE_CHANGE" if d0_refit else
                "NEW_FOLD_EVIDENCE_RECIPE_UNCHANGED"
            ),
        })
        plans[ARM_D1].append({
            **common,
            "choice": d1_choice,
            "refit_required": d1_refit,
            "refit_reason": (
                "INITIAL_FIT" if initial else
                "HYSTERESIS_SWITCH_CONFIRMED" if hysteresis_decision.switch_allowed else
                "ROLLING_RECALIBRATION_ONLY"
            ),
        })

        hrow = decision_to_dict(hysteresis_decision)
        hrow.update({
            "evidence_expanded": bool(evidence_expanded),
            "selector_winner_text": _choice_key_text(winner),
            "active_d0_recipe_text": _choice_key_text(d0_choice),
            "active_d1_recipe_text": _choice_key_text(d1_choice),
            "frozen_recipe_text": _choice_key_text(frozen),
            "eligible_recipe_count": len(choices),
        })
        hysteresis_rows.append(hrow)
        audit_rows.append({
            "assessment_date": assessment["assessment_date"].isoformat(),
            "latest_matured_evidence_date": assessment["latest_matured_evidence_date"].isoformat(),
            "evidence_fingerprint": fingerprint,
            "evidence_expanded": bool(evidence_expanded),
            "selector_winner": _choice_key_text(winner),
            "selector_winner_fold_count": int(winner["fold_count"]),
            "selector_winner_robust_score": float(winner["robust_score"]),
            "b_active_recipe": _choice_key_text(frozen),
            "c_active_recipe": _choice_key_text(frozen),
            "d0_active_recipe": _choice_key_text(d0_choice),
            "d0_refit": d0_refit,
            "d0_reason": plans[ARM_D0][-1]["refit_reason"],
            "d1_active_recipe": _choice_key_text(d1_choice),
            "d1_refit": d1_refit,
            "d1_reason": plans[ARM_D1][-1]["refit_reason"],
            "d1_confirmations_after": int(hysteresis_decision.confirmations_after),
            "d1_switch_allowed": bool(hysteresis_decision.switch_allowed),
            "c_refit": True,
        })
        previous_fingerprint = fingerprint

    return plans, hysteresis_rows, audit_rows, state


def _fit_source(
    *,
    index: int,
    cutoff: date,
    choice: Mapping[str, Any],
    base,
    metric_rows: list[dict],
    signal_panel: Path,
    sessions: tuple[date, ...],
    output_root: Path,
    end: date,
    horizon: int,
    materializer_cache: dict,
) -> dict:
    maturity = HorizonMaturityResolver(sessions)
    latest = maturity.latest_matured_decision(cutoff, horizon)
    if latest is None:
        raise ValueError("CONDITIONAL_REFIT_WITHOUT_MATURED_LABEL_CUTOFF")
    key = recipe_key(choice)
    materializer = _materializer_for_choice(
        choice=choice,
        horizon=horizon,
        metric_rows=metric_rows,
        signal_panel=signal_panel,
        output_root=output_root,
        end=end,
        cache=materializer_cache,
    )
    family = _family_for_choice(base, choice, stable_hash(key)[:10])
    name = f"FIT_{index:03d}_{cutoff.isoformat()}_{str(choice['family'])}"
    builder = H130ProductionGenerationBuilder(
        materializer=materializer,
        root=output_root / "fresh-models" / name,
        code_commit="DYNAMIC_QBD_CONDITIONAL_REFIT_V2",
    )
    build = builder.build(
        family=family,
        information_cutoff=cutoff,
        latest_matured_label_cutoff=latest,
    )
    selected = build["selection_evidence"]
    if recipe_key(selected) != key:
        raise AssertionError(
            f"CONDITIONAL_REFIT_EXACT_RECIPE_FIT_MISMATCH:{key!r}!={recipe_key(selected)!r}"
        )
    generation_id = stable_hash((SCHEMA_VERSION, name, family.family_hash, cutoff, build["model_artifact_id"]))[:24]
    finalized = builder.finalize_generation(family=family, build=build, generation_id=generation_id)
    predictions = pd.read_parquet(finalized["prediction_artifact_path"])
    predictions["model_artifact_id"] = str(build["model_artifact_id"])
    return {
        "source_key": (cutoff, key),
        "fit_cutoff": cutoff,
        "choice": dict(choice),
        "family": family,
        "builder": builder,
        "build": build,
        "predictions": predictions[["decision_date", "ticker", "score", "model_artifact_id"]],
    }


def _fit_required_sources(
    *,
    plans: dict[str, list[dict]],
    base,
    metric_rows: list[dict],
    signal_panel: Path,
    sessions: tuple[date, ...],
    output_root: Path,
    end: date,
    horizon: int,
) -> tuple[dict, list[dict]]:
    required: dict[tuple, dict] = {}
    requested_by: dict[tuple, list[str]] = {}
    for arm, plan in plans.items():
        for row in plan:
            if not row["refit_required"]:
                continue
            source_key = (row["activation_cutoff"], recipe_key(row["choice"]))
            required[source_key] = dict(row["choice"])
            requested_by.setdefault(source_key, []).append(arm)

    sources = {}
    audit = []
    materializer_cache: dict = {}
    for index, (source_key, choice) in enumerate(
        sorted(required.items(), key=lambda item: (item[0][0], item[0][1]))
    ):
        cutoff, _ = source_key
        source = _fit_source(
            index=index,
            cutoff=cutoff,
            choice=choice,
            base=base,
            metric_rows=metric_rows,
            signal_panel=signal_panel,
            sessions=sessions,
            output_root=output_root,
            end=end,
            horizon=horizon,
            materializer_cache=materializer_cache,
        )
        sources[source_key] = source
        audit.append({
            "fit_cutoff": cutoff.isoformat(),
            "recipe": _choice_key_text(choice),
            "model_family": str(choice["family"]),
            "candidate_id": str(choice["candidate_id"]),
            "parameters": dict(choice.get("parameters", {})),
            "fold_count": int(choice["fold_count"]),
            "fold_ids": list(choice.get("fold_ids", ())),
            "robust_score": float(choice["robust_score"]),
            "requested_by_arms": sorted(set(requested_by[source_key])),
            "model_artifact_id": str(source["build"]["model_artifact_id"]),
            "model_artifact_sha256": str(source["build"]["model_artifact_sha256"]),
            "fresh_fit": True,
        })
    return sources, audit


def _bind_sources(plans: dict[str, list[dict]], sources: dict) -> None:
    for arm, plan in plans.items():
        active_source_key = None
        for row in plan:
            if row["refit_required"]:
                active_source_key = (row["activation_cutoff"], recipe_key(row["choice"]))
                if active_source_key not in sources:
                    raise KeyError(f"CONDITIONAL_REFIT_SOURCE_MISSING:{arm}:{active_source_key!r}")
            if active_source_key is None:
                raise AssertionError(f"CONDITIONAL_REFIT_ARM_WITHOUT_INITIAL_SOURCE:{arm}")
            source = sources[active_source_key]
            if recipe_key(source["choice"]) != recipe_key(row["choice"]):
                raise AssertionError(f"CONDITIONAL_REFIT_ACTIVE_RECIPE_SOURCE_MISMATCH:{arm}")
            row["source_key"] = active_source_key


def _arm_signals_schedule(
    *, arm: str, plan: list[dict], sources: dict, sessions: tuple[date, ...]
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    maturity = HorizonMaturityResolver(sessions)
    signals_by_model: dict[str, pd.DataFrame] = {}
    schedule_rows = []
    calibration_audit = []

    for row in plan:
        source = sources[row["source_key"]]
        cutoff = row["activation_cutoff"]
        latest = row["latest_matured_evidence_date"]
        family = source["family"]
        builder = source["builder"]
        build = source["build"]
        if cutoff == source["fit_cutoff"]:
            frame = builder.calibration_predictions(family=family, build=build, information_cutoff=cutoff)
            calibration_source = "FIT_GENERATION_CALIBRATION"
        else:
            frame = builder.recalibration_predictions(
                family=family,
                model_information_cutoff=source["fit_cutoff"],
                information_cutoff=cutoff,
                latest_matured_label_cutoff=latest,
                cache_result=False,
            )
            calibration_source = "ROLLING_RECALIBRATION_FROZEN_MODEL"

        generation_id = stable_hash({
            "schema_version": SCHEMA_VERSION,
            "arm": arm,
            "model_artifact_id": build["model_artifact_id"],
            "activation_cutoff": cutoff,
            "calibration_source": calibration_source,
        })[:24]
        resolved = recalibrate_generation(
            family,
            generation_id,
            frame,
            information_cutoff=cutoff,
            maturity=maturity,
        )
        model_artifact_id = str(build["model_artifact_id"])
        signals_by_model.setdefault(model_artifact_id, source["predictions"])
        schedule_rows.append({
            "activation_date": cutoff,
            "family_id": arm,
            "generation_id": generation_id,
            "model_artifact_id": model_artifact_id,
            "resolved_threshold": float(resolved.resolved_threshold),
            "resolved_top_fraction": float(resolved.resolved_top_fraction),
            "entry_policy_id": f"{arm}_ENTRY_{cutoff}",
            "exit_policy_id": "FIXED_D2",
        })
        calibration_audit.append({
            "arm": arm,
            "assessment_date": cutoff.isoformat(),
            "model_fit_cutoff": source["fit_cutoff"].isoformat(),
            "model_age_calendar_days": int((cutoff - source["fit_cutoff"]).days),
            "active_recipe": _choice_key_text(source["choice"]),
            "model_artifact_id": model_artifact_id,
            "calibration_source": calibration_source,
            "resolved_threshold": float(resolved.resolved_threshold),
            "resolved_top_fraction": float(resolved.resolved_top_fraction),
            "matured_observation_count": int(resolved.observations),
            "refit_this_assessment": bool(row["refit_required"]),
            "refit_reason": str(row["refit_reason"]),
        })

    signals = pd.concat(list(signals_by_model.values()), ignore_index=True)
    if signals.duplicated(["decision_date", "ticker", "model_artifact_id"]).any():
        raise ValueError(f"CONDITIONAL_REFIT_DUPLICATE_MODEL_SIGNAL:{arm}")
    return signals, pd.DataFrame(schedule_rows), calibration_audit


def _change_counts(plan: list[dict]) -> tuple[int, int]:
    exact = 0
    family = 0
    previous_key = None
    previous_family = None
    for row in plan:
        key = recipe_key(row["choice"])
        model_family = str(row["choice"]["family"])
        if previous_key is not None and key != previous_key:
            exact += 1
        if previous_family is not None and model_family != previous_family:
            family += 1
        previous_key = key
        previous_family = model_family
    return exact, family


def _replay_arm(
    *,
    arm: str,
    plan: list[dict],
    sources: dict,
    sessions: tuple[date, ...],
    prices: pd.DataFrame,
    start: date,
    end: date,
    horizon: int,
    holding_days: int,
    max_names: int,
    top_fraction: float,
    initial: float,
) -> tuple[dict, list[dict]]:
    signals, schedule, calibration_audit = _arm_signals_schedule(
        arm=arm, plan=plan, sources=sources, sessions=sessions
    )
    result = replay_family(
        signals=signals,
        prices=prices,
        policy=Policy(
            horizon=horizon,
            score_quantile=.5,
            top_fraction=top_fraction,
            max_names=max_names,
            holding_days=holding_days,
            sleeve=.5,
        ),
        generation_schedule=schedule,
        cost=CostModel(20.0),
        tax=TaxConfig(False),
        start=pd.Timestamp(start),
        end=pd.Timestamp(end),
        initial=initial,
    )
    risk = wealth_path_metrics(result["curve"])
    metrics = result["metrics"]
    exact_changes, family_changes = _change_counts(plan)
    row = {
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
        "recipe_change_count": exact_changes,
        "model_family_switch_count": family_changes,
        "fresh_fit_count": int(sum(bool(x["refit_required"]) for x in plan)),
        "monthly_recalibration_count": int(len(plan)),
    }
    return {"row": row, "result": result, "schedule": schedule}, calibration_audit


def _diagnosis(rows: list[dict]) -> str:
    by_name = {row["strategy"]: row for row in rows}
    d1 = by_name[ARM_D1]
    b = by_name[ARM_B]
    c = by_name[ARM_C]
    d1_matches_b = (
        math.isclose(d1["terminal_value"], b["terminal_value"], rel_tol=1e-12, abs_tol=1e-9)
        and math.isclose(d1["relative_max_drawdown"], b["relative_max_drawdown"], rel_tol=1e-12, abs_tol=1e-9)
    )
    if d1_matches_b:
        return "D1_MATCHES_FROZEN_MODEL_B"
    if (
        d1["terminal_value"] >= b["terminal_value"]
        and d1["relative_max_drawdown"] >= b["relative_max_drawdown"]
        and d1["terminal_value"] >= c["terminal_value"]
        and d1["relative_max_drawdown"] >= c["relative_max_drawdown"]
    ):
        return "D1_PARETO_DOMINATES_B_AND_MONTHLY_REFIT_C"
    if d1["terminal_value"] >= b["terminal_value"] and d1["relative_max_drawdown"] >= b["relative_max_drawdown"]:
        return "D1_PARETO_DOMINATES_FROZEN_MODEL_B"
    if d1["terminal_value"] <= b["terminal_value"] and d1["relative_max_drawdown"] <= b["relative_max_drawdown"]:
        return "D1_INFERIOR_TO_FROZEN_MODEL_B"
    return "D1_VS_B_TRADEOFF"


def run_experiment(
    *,
    signal_panel: str | Path,
    candidate_metrics: str | Path,
    daily_store_root: str | Path,
    output_root: str | Path,
    benchmark_daily_path: str | Path | None = None,
    direct_daily_stock_root: str | Path | None = None,
    start: date = date(2020, 8, 31),
    end: date = date(2023, 12, 29),
    horizon: int = 3,
    holding_days: int = 2,
    max_names: int = 1,
    score_quantile: float = 0.75,
    top_fraction: float = 0.01,
    initial: float = 10000.0,
) -> dict:
    if end >= CLOSED_HOLDOUT_START:
        raise ValueError("CONDITIONAL_REFIT_FINAL_HOLDOUT_MUST_REMAIN_CLOSED")
    if start > end:
        raise ValueError("CONDITIONAL_REFIT_START_AFTER_END")

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    signal_panel = Path(signal_panel)
    metrics_path = Path(candidate_metrics)
    metric_rows = json.loads(metrics_path.read_text(encoding="utf-8"))

    base_materializer = ParquetH130DatasetMaterializer(signal_panel, metrics_path, end)
    base = next(
        x for x in build_family_specs(
            feature_schema_sha256=base_materializer.feature_schema_fingerprint,
            score_quantile=score_quantile,
            top_fraction=top_fraction,
        )
        if x.horizon_sessions == horizon
        and x.holding_days == holding_days
        and x.max_names == max_names
        and x.exit_policy["family"] == "FIXED"
    )
    sessions_frame = pd.read_parquet(signal_panel, columns=["decision_date"]).drop_duplicates()
    sessions = tuple(sorted(pd.to_datetime(sessions_frame["decision_date"]).dt.date.unique()))

    assessments = _causal_recipe_assessments(
        family=base,
        materializer=base_materializer,
        metrics_path=metrics_path,
        sessions=sessions,
        end=end,
        root=output_root / "assessment-audit",
    )
    assessments = [row for row in assessments if row["assessment_date"] <= end]
    if not assessments:
        raise ValueError("NO_CONDITIONAL_REFIT_ASSESSMENTS")

    config = HysteresisConfig(
        confirmations_required=2,
        minimum_oos_folds=max(2, int(base.hyperparameter_rule.get("minimum_oos_folds", 2))),
        require_strict_robust_score_improvement=True,
    )
    plans, hysteresis_rows, decision_audit, final_state = _build_arm_plans(assessments, config=config)
    pd.DataFrame(hysteresis_rows).to_csv(output_root / "hysteresis-decisions.csv", index=False)
    pd.DataFrame(decision_audit).to_csv(output_root / "conditional-refit-decisions.csv", index=False)

    sources, fit_audit = _fit_required_sources(
        plans=plans,
        base=base,
        metric_rows=metric_rows,
        signal_panel=signal_panel,
        sessions=sessions,
        output_root=output_root,
        end=end,
        horizon=horizon,
    )
    _bind_sources(plans, sources)
    (output_root / "fit-audit.json").write_text(
        json.dumps(fit_audit, default=str, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    first_activation = min(row["activation_cutoff"] for row in plans[ARM_B])
    evaluation_start = max(start, first_activation)
    prices_path = materialize_daily_store_prices(
        daily_store_root=daily_store_root,
        signal_panel=signal_panel,
        start=first_activation,
        end=end,
        output_path=output_root / "inputs" / "prices.parquet",
        benchmark_daily_path=benchmark_daily_path,
        direct_daily_stock_root=direct_daily_stock_root,
    )
    prices = pd.read_parquet(prices_path)

    result_rows = []
    all_calibration_audit = []
    for arm in ARMS:
        replay, calibration_audit = _replay_arm(
            arm=arm,
            plan=plans[arm],
            sources=sources,
            sessions=sessions,
            prices=prices,
            start=evaluation_start,
            end=end,
            horizon=horizon,
            holding_days=holding_days,
            max_names=max_names,
            top_fraction=top_fraction,
            initial=initial,
        )
        result_rows.append(replay["row"])
        all_calibration_audit.extend(calibration_audit)
        curve = replay["result"]["curve"].copy()
        curve["strategy"] = arm
        curve.to_parquet(output_root / f"{arm}-nav.parquet", index=False)
        pd.DataFrame(replay["result"]["trades"]).to_parquet(
            output_root / f"{arm}-trades.parquet", index=False
        )
        replay["schedule"].to_csv(output_root / f"{arm}-schedule.csv", index=False)

    pd.DataFrame(all_calibration_audit).to_csv(output_root / "calibration-audit.csv", index=False)
    comparison = pd.DataFrame(result_rows).sort_values("terminal_value", ascending=False)
    comparison.to_csv(output_root / "portfolio-value-comparison.csv", index=False)
    rows = comparison.to_dict(orient="records")
    diagnosis = _diagnosis(rows)
    unique_evidence_expansions = len({str(x["evidence_fingerprint"]) for x in assessments})

    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": "COMPLETE",
        "authority": AUTHORITY,
        "promotion_allowed": False,
        "final_holdout_opened": False,
        "recipe_parameter_grid_searched": False,
        "experiment": {
            "horizon": horizon,
            "holding_days": holding_days,
            "max_names": max_names,
            "score_quantile": score_quantile,
            "top_fraction": top_fraction,
            "initial_value_eur": initial,
            "evaluation_start": evaluation_start.isoformat(),
            "evaluation_end": end.isoformat(),
        },
        "arm_contract": {
            ARM_B: "INITIAL_MODEL_FIT_ONCE_THEN_MONTHLY_RECALIBRATION_ONLY",
            ARM_C: "FROZEN_INITIAL_RECIPE_FRESHLY_REFIT_EVERY_MONTH_THEN_RECALIBRATED",
            ARM_D0: "REFIT_ONLY_WHEN_NEW_FOLD_EVIDENCE_CHANGES_EXACT_CAUSAL_RECIPE",
            ARM_D1: "REFIT_ONLY_AFTER_TWO_INDEPENDENT_FOLD_EVIDENCE_EXPANSIONS_CONFIRM_SAME_CHALLENGER",
        },
        "hysteresis_contract": {
            **asdict(config),
            "confirmation_unit": "DISTINCT_FOLD_CONTENT_FINGERPRINT_NOT_CALENDAR_DATE",
            "calendar_progress_without_fold_content_change_is_evidence": False,
            "challenger_identity": "EXACT_CANDIDATE_FAMILY_AND_PARAMETERS",
        },
        "assessment_count": len(assessments),
        "unique_recipe_evidence_expansion_count": unique_evidence_expansions,
        "final_hysteresis_state": state_to_dict(final_state),
        "execution_resources": active_cpu_contract(),
        "diagnosis": diagnosis,
        "primary_contrasts": [
            f"{ARM_C}_MINUS_{ARM_B}",
            f"{ARM_D0}_MINUS_{ARM_B}",
            f"{ARM_D1}_MINUS_{ARM_B}",
        ],
        "portfolio_value_comparison": rows,
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, default=str, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    report = [
        "# Dynamic-QBD Conditional Refit / Recipe Hysteresis",
        "",
        f"Status: **{diagnosis}**",
        "",
        "Authority: `SHADOW_ONLY_NO_PROMOTION_NO_HOLDOUT`.",
        "The final holdout remains closed.",
        "",
        "## Scientific contract",
        "",
        f"- `{ARM_B}` fits the initial model once and recalibrates it monthly.",
        f"- `{ARM_C}` uses the same initial frozen recipe but freshly refits it every month; C-B isolates unconditional refitting.",
        f"- `{ARM_D0}` refits only when a real new OOS-fold evidence expansion changes the exact causal recipe.",
        f"- `{ARM_D1}` refits only after two independent OOS-fold evidence expansions support the same exact challenger with a strict robust-score edge.",
        "- Calendar progress alone cannot increment hysteresis confirmation.",
        "- All arms use the same monthly recalibration dates, portfolio policy, costs and benchmark.",
        "- No grid search, evaluation-window tuning, promotion or holdout access.",
        "",
        "## Portfolio comparison",
        "",
        comparison_markdown(comparison),
        "",
        "## Interpretation",
        "",
        "The decomposition is C-B (pure unconditional refit value), D0-B (immediate recipe-change conditional-refit value), and D1-B (hysteresis-gated conditional-refit value).",
        "",
        f"Observed monthly assessments: {len(assessments)}; distinct fold-content evidence states: {unique_evidence_expansions}.",
        "",
        "Exact fit provenance is in `fit-audit.json`; monthly model age, thresholds and refit flags are in `calibration-audit.csv`; authority decisions are in `conditional-refit-decisions.csv` and `hysteresis-decisions.csv`.",
        "",
        "This remains Development/Research evidence only.",
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
    parser.add_argument("--score-quantile", type=float, default=0.75)
    parser.add_argument("--top-fraction", type=float, default=0.01)
    parser.add_argument("--initial", type=float, default=10000.0)
    args = parser.parse_args()
    configure_cpu_peak(process_workers=1)
    result = run_experiment(
        signal_panel=args.signal_panel,
        candidate_metrics=args.candidate_metrics,
        daily_store_root=args.daily_store_root,
        benchmark_daily_path=args.benchmark_daily_path,
        direct_daily_stock_root=args.direct_daily_stock_root,
        output_root=args.output_root,
        start=args.start,
        end=args.end,
        horizon=args.horizon,
        holding_days=args.holding_days,
        max_names=args.max_names,
        score_quantile=args.score_quantile,
        top_fraction=args.top_fraction,
        initial=args.initial,
    )
    print(json.dumps(result["portfolio_value_comparison"], indent=2))


if __name__ == "__main__":
    main()
