"""Shadow-only Dynamic-QBD recipe-switch hysteresis experiment.

The experiment keeps refit/recalibration dates identical across three arms:

1. frozen first causal recipe, freshly refit every assessment;
2. current rolling causal recipe winner, freshly refit every assessment;
3. hysteresis-gated recipe, freshly refit every assessment.

Only the recipe identity differs. Hysteresis requires two consecutive *distinct
matured-evidence expansions* supporting the same challenger and a strictly
positive robust-score edge over the incumbent. No parameter grid is searched.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import date
import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from .contract_fingerprints import stable_hash
from .dynamic_qbd_factory import monthly_refit_dates
from .dynamic_qbd_h1_30_adapter import (
    H130ProductionGenerationBuilder,
    ParquetH130DatasetMaterializer,
)
from .dynamic_qbd_maturity import HorizonMaturityResolver
from .dynamic_qbd_model_combination_experiment import (
    _fit_component,
    _scheduled_replay,
    comparison_markdown,
)
from .dynamic_qbd_development_pipeline import materialize_daily_store_prices
from .dynamic_qbd_recipe_hysteresis import (
    HysteresisConfig,
    RecipeHysteresisState,
    decision_to_dict,
    recipe_key,
    stable_evidence_fingerprint,
    state_to_dict,
    update_recipe_hysteresis,
)
from .dynamic_qbd_runtime_resources import active_cpu_contract, configure_cpu_peak
from .dynamic_qbd_family_surface import build_family_specs


AUTHORITY = "SHADOW_ONLY_NO_PROMOTION_NO_HOLDOUT"
CLOSED_HOLDOUT_START = date(2026, 7, 25)
SCHEMA_VERSION = "DYNAMIC_QBD_RECIPE_SWITCH_HYSTERESIS_V1"


def _choice_key_text(choice: Mapping[str, Any]) -> str:
    key = recipe_key(choice)
    return f"{key[1]}::{key[0]}::{key[2][:12]}"


def _choice_for_key(choices: tuple[dict, ...], key) -> dict:
    match = next((dict(x) for x in choices if recipe_key(x) == key), None)
    if match is None:
        raise ValueError(f"RECIPE_KEY_NOT_CURRENTLY_ELIGIBLE:{key!r}")
    return match


def _causal_recipe_assessments(
    *,
    family,
    materializer: ParquetH130DatasetMaterializer,
    metrics_path: Path,
    sessions: tuple[date, ...],
    end: date,
    root: Path,
) -> list[dict]:
    maturity = HorizonMaturityResolver(sessions)
    selector = H130ProductionGenerationBuilder(
        materializer=materializer,
        root=root,
        code_commit="DYNAMIC_QBD_RECIPE_HYSTERESIS_ASSESSMENT_V1",
    )
    assessments: list[dict] = []
    for cutoff in monthly_refit_dates(tuple(x for x in sessions if x <= end)):
        latest = maturity.latest_matured_decision(cutoff, family.horizon_sessions)
        if latest is None:
            continue
        try:
            choices = selector.candidate_choices(family, metrics_path, latest)
            winner = selector._select_candidate(family, metrics_path, latest)
        except ValueError as exc:
            if str(exc).startswith(("NO_CAUSAL_", "INSUFFICIENT_CAUSAL_")):
                continue
            raise
        fingerprint = stable_evidence_fingerprint(
            evidence_date=latest,
            choices=choices,
        )
        assessments.append(
            {
                "assessment_date": cutoff,
                "latest_matured_evidence_date": latest,
                "evidence_fingerprint": fingerprint,
                "eligible_choices": tuple(dict(x) for x in choices),
                "selector_winner": dict(winner),
            }
        )
    if not assessments:
        raise ValueError("NO_CAUSAL_RECIPE_HYSTERESIS_ASSESSMENTS")
    return assessments


def _build_arm_plans(
    assessments: list[dict],
    *,
    config: HysteresisConfig,
) -> tuple[dict[str, list[dict]], list[dict], RecipeHysteresisState]:
    state = RecipeHysteresisState()
    frozen_key = None
    plans = {
        "FROZEN_FIRST_CAUSAL_RECIPE_MONTHLY_REFIT": [],
        "ROLLING_CAUSAL_RECIPE_MONTHLY_REFIT": [],
        "HYSTERESIS_CAUSAL_RECIPE_MONTHLY_REFIT": [],
    }
    decision_rows: list[dict] = []

    for assessment in assessments:
        choices = assessment["eligible_choices"]
        winner = assessment["selector_winner"]
        state, decision = update_recipe_hysteresis(
            state=state,
            assessment_date=assessment["assessment_date"],
            evidence_date=assessment["latest_matured_evidence_date"],
            evidence_fingerprint=assessment["evidence_fingerprint"],
            selector_winner=winner,
            eligible_choices=choices,
            config=config,
        )
        if frozen_key is None:
            frozen_key = recipe_key(winner)
        frozen = _choice_for_key(choices, frozen_key)
        hysteresis = _choice_for_key(choices, state.incumbent)

        common = {
            "activation_cutoff": assessment["assessment_date"],
            "latest_matured_evidence_date": assessment["latest_matured_evidence_date"],
            "evidence_fingerprint": assessment["evidence_fingerprint"],
        }
        plans["FROZEN_FIRST_CAUSAL_RECIPE_MONTHLY_REFIT"].append(
            {**common, "choice": frozen}
        )
        plans["ROLLING_CAUSAL_RECIPE_MONTHLY_REFIT"].append(
            {**common, "choice": dict(winner)}
        )
        plans["HYSTERESIS_CAUSAL_RECIPE_MONTHLY_REFIT"].append(
            {**common, "choice": hysteresis}
        )

        row = decision_to_dict(decision)
        row.update(
            {
                "selector_winner_text": _choice_key_text(winner),
                "active_hysteresis_recipe_text": _choice_key_text(hysteresis),
                "frozen_recipe_text": _choice_key_text(frozen),
                "eligible_recipe_count": len(choices),
            }
        )
        decision_rows.append(row)

    return plans, decision_rows, state


def _change_counts(plan: list[dict]) -> tuple[int, int]:
    exact = 0
    family = 0
    prior_key = None
    prior_family = None
    for row in plan:
        key = recipe_key(row["choice"])
        model_family = str(row["choice"]["family"])
        if prior_key is not None and key != prior_key:
            exact += 1
        if prior_family is not None and model_family != prior_family:
            family += 1
        prior_key = key
        prior_family = model_family
    return exact, family


def _matching_metric_rows(
    rows: list[dict],
    *,
    horizon: int,
    choice: Mapping[str, Any],
) -> list[dict]:
    expected_parameters = dict(choice.get("parameters", {}))
    matched = [
        row
        for row in rows
        if int(row.get("horizon_sessions", -1)) == int(horizon)
        and str(row.get("candidate_id")) == str(choice["candidate_id"])
        and str(row.get("family")) == str(choice["family"])
        and dict(row.get("parameters", {})) == expected_parameters
    ]
    if not matched:
        raise ValueError(f"RECIPE_METRIC_ROWS_MISSING:{_choice_key_text(choice)}")
    return matched


def _materializer_for_choice(
    *,
    choice: Mapping[str, Any],
    horizon: int,
    metric_rows: list[dict],
    signal_panel: Path,
    output_root: Path,
    end: date,
    cache: dict,
) -> ParquetH130DatasetMaterializer:
    key = recipe_key(choice)
    if key in cache:
        return cache[key]
    selected = _matching_metric_rows(metric_rows, horizon=horizon, choice=choice)
    target = output_root / "recipe-metrics" / f"{stable_hash(key)[:20]}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(selected, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    materializer = ParquetH130DatasetMaterializer(signal_panel, target, end)
    cache[key] = materializer
    return materializer


def _family_for_choice(base, choice: Mapping[str, Any], key_suffix: str):
    return replace(
        base,
        family_id=f"{base.family_id}_HYST_{key_suffix}",
        model_family=str(choice["family"]),
        hyperparameter_rule={
            **base.hyperparameter_rule,
            "minimum_oos_folds": max(
                2, int(base.hyperparameter_rule.get("minimum_oos_folds", 2))
            ),
            "recipe_selection_contract": "CAUSAL_RESELECT_AT_EACH_REFIT",
        },
    )


def _fit_required_components(
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
    maturity = HorizonMaturityResolver(sessions)
    materializer_cache: dict = {}
    fitted: dict = {}
    audit: list[dict] = []

    required = {}
    for plan in plans.values():
        for row in plan:
            cutoff = row["activation_cutoff"]
            choice = row["choice"]
            required[(cutoff, recipe_key(choice))] = choice

    for index, ((cutoff, key), choice) in enumerate(
        sorted(required.items(), key=lambda item: (item[0][0], item[0][1]))
    ):
        latest = maturity.latest_matured_decision(cutoff, horizon)
        if latest is None:
            raise ValueError("HYSTERESIS_REFIT_WITHOUT_MATURED_LABEL_CUTOFF")
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
        component = _fit_component(
            name=f"REFIT_{index:03d}_{cutoff.isoformat()}_{str(choice['family'])}",
            family=family,
            materializer=materializer,
            root=output_root / "fresh-models",
            cutoff=cutoff,
            latest_matured=latest,
            sessions=sessions,
        )
        selected = component["build"]["selection_evidence"]
        if recipe_key(selected) != key:
            raise AssertionError(
                f"HYSTERESIS_EXACT_RECIPE_FIT_MISMATCH:{key!r}!={recipe_key(selected)!r}"
            )
        component["activation_cutoff"] = cutoff
        fitted[(cutoff, key)] = component
        audit.append(
            {
                "activation_cutoff": cutoff.isoformat(),
                "latest_matured_evidence_date": latest.isoformat(),
                "recipe": _choice_key_text(choice),
                "model_family": str(choice["family"]),
                "candidate_id": str(choice["candidate_id"]),
                "parameters": dict(choice["parameters"]),
                "fold_count": int(choice["fold_count"]),
                "fold_ids": tuple(choice.get("fold_ids", ())),
                "model_artifact_id": component["build"]["model_artifact_id"],
                "model_artifact_sha256": component["build"]["model_artifact_sha256"],
                "fresh_fit": True,
            }
        )
    return fitted, audit


def _components_for_plan(plan: list[dict], fitted: dict) -> list[dict]:
    return [
        fitted[(row["activation_cutoff"], recipe_key(row["choice"]))]
        for row in plan
    ]


def _write_replay_artifacts(root: Path, replay: dict) -> dict:
    result = replay["result"]
    curve = result["curve"].copy()
    curve["strategy"] = replay["strategy"]
    curve.to_parquet(root / f"{replay['strategy']}-nav.parquet", index=False)
    pd.DataFrame(result["trades"]).to_parquet(
        root / f"{replay['strategy']}-trades.parquet", index=False
    )
    metrics = result["metrics"]
    return {
        "strategy": replay["strategy"],
        "terminal_value": float(metrics["terminal_value"]),
        "urth_terminal_value": float(metrics["urth_terminal_value"]),
        "terminal_excess_eur": float(
            metrics["terminal_value"] - metrics["urth_terminal_value"]
        ),
        "terminal_relative_return": float(
            metrics["terminal_value"] / metrics["urth_terminal_value"] - 1.0
        ),
        "trade_count": int(metrics["trade_count"]),
        "total_cost_eur": float(metrics["total_cost_eur"]),
        "relative_max_drawdown": float(replay["risk"]["relative_max_drawdown"]),
    }


def _diagnosis(rows: list[dict]) -> str:
    by_name = {row["strategy"]: row for row in rows}
    rolling = by_name["ROLLING_CAUSAL_RECIPE_MONTHLY_REFIT"]
    hysteresis = by_name["HYSTERESIS_CAUSAL_RECIPE_MONTHLY_REFIT"]
    better_terminal = hysteresis["terminal_value"] >= rolling["terminal_value"]
    better_drawdown = (
        hysteresis["relative_max_drawdown"] >= rolling["relative_max_drawdown"]
    )
    no_more_switches = (
        hysteresis["recipe_change_count"] <= rolling["recipe_change_count"]
    )
    if better_terminal and better_drawdown and no_more_switches:
        return "HYSTERESIS_PARETO_DOMINATES_ROLLING"
    if (
        hysteresis["terminal_value"] <= rolling["terminal_value"]
        and hysteresis["relative_max_drawdown"] <= rolling["relative_max_drawdown"]
    ):
        return "HYSTERESIS_INFERIOR_TO_ROLLING"
    return "HYSTERESIS_ROLLING_TRADEOFF"


def run_experiment(
    *,
    signal_panel: str | Path,
    candidate_metrics: str | Path,
    daily_store_root: str | Path,
    benchmark_daily_path: str | Path | None = None,
    direct_daily_stock_root: str | Path | None = None,
    output_root: str | Path,
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
        raise ValueError("RECIPE_HYSTERESIS_FINAL_HOLDOUT_MUST_REMAIN_CLOSED")
    if start > end:
        raise ValueError("RECIPE_HYSTERESIS_START_AFTER_END")
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    signal_panel = Path(signal_panel)
    metrics_path = Path(candidate_metrics)
    metric_rows = json.loads(metrics_path.read_text(encoding="utf-8"))

    base_materializer = ParquetH130DatasetMaterializer(
        signal_panel, metrics_path, end
    )
    base = next(
        x
        for x in build_family_specs(
            feature_schema_sha256=base_materializer.feature_schema_fingerprint,
            score_quantile=score_quantile,
            top_fraction=top_fraction,
        )
        if x.horizon_sessions == horizon
        and x.holding_days == holding_days
        and x.max_names == max_names
        and x.exit_policy["family"] == "FIXED"
    )

    sessions_frame = pd.read_parquet(
        signal_panel, columns=["decision_date"]
    ).drop_duplicates()
    sessions = tuple(
        sorted(pd.to_datetime(sessions_frame["decision_date"]).dt.date.unique())
    )
    assessments = _causal_recipe_assessments(
        family=base,
        materializer=base_materializer,
        metrics_path=metrics_path,
        sessions=sessions,
        end=end,
        root=output_root / "assessment-audit",
    )
    assessments = [
        row for row in assessments if row["assessment_date"] <= end
    ]
    if not assessments:
        raise ValueError("NO_HYSTERESIS_ASSESSMENTS_BEFORE_END")

    config = HysteresisConfig(
        confirmations_required=2,
        minimum_oos_folds=max(2, int(base.hyperparameter_rule.get("minimum_oos_folds", 2))),
        require_strict_robust_score_improvement=True,
    )
    plans, decision_rows, final_state = _build_arm_plans(
        assessments, config=config
    )

    assessment_rows = []
    for row in assessments:
        winner = row["selector_winner"]
        assessment_rows.append(
            {
                "assessment_date": row["assessment_date"].isoformat(),
                "latest_matured_evidence_date": row[
                    "latest_matured_evidence_date"
                ].isoformat(),
                "evidence_fingerprint": row["evidence_fingerprint"],
                "selector_winner": _choice_key_text(winner),
                "selector_winner_family": str(winner["family"]),
                "selector_winner_candidate_id": str(winner["candidate_id"]),
                "selector_winner_robust_score": float(winner["robust_score"]),
                "selector_winner_fold_count": int(winner["fold_count"]),
                "eligible_recipe_count": len(row["eligible_choices"]),
            }
        )
    pd.DataFrame(assessment_rows).to_csv(
        output_root / "recipe-assessments.csv", index=False
    )
    pd.DataFrame(decision_rows).to_csv(
        output_root / "hysteresis-decisions.csv", index=False
    )

    fitted, fit_audit = _fit_required_components(
        plans=plans,
        base=base,
        metric_rows=metric_rows,
        signal_panel=signal_panel,
        sessions=sessions,
        output_root=output_root,
        end=end,
        horizon=horizon,
    )
    (output_root / "recipe-fit-audit.json").write_text(
        json.dumps(fit_audit, default=str, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    first_activation = min(
        row["activation_cutoff"]
        for plan in plans.values()
        for row in plan
    )
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

    replay_rows: list[dict] = []
    for strategy, plan in plans.items():
        replay = _scheduled_replay(
            strategy,
            _components_for_plan(plan, fitted),
            prices,
            start=evaluation_start,
            end=end,
            horizon=horizon,
            holding=holding_days,
            max_names=max_names,
            initial=initial,
        )
        row = _write_replay_artifacts(output_root, replay)
        exact_changes, family_changes = _change_counts(plan)
        row["recipe_change_count"] = exact_changes
        row["model_family_switch_count"] = family_changes
        row["monthly_refit_count"] = len(plan)
        replay_rows.append(row)

    comparison = pd.DataFrame(replay_rows).sort_values(
        "terminal_value", ascending=False
    )
    comparison.to_csv(output_root / "portfolio-value-comparison.csv", index=False)
    result_rows = comparison.to_dict(orient="records")
    diagnosis = _diagnosis(result_rows)

    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": "COMPLETE",
        "authority": AUTHORITY,
        "final_holdout_opened": False,
        "promotion_allowed": False,
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
        "hysteresis_contract": {
            **asdict(config),
            "confirmation_unit": "DISTINCT_LATEST_MATURED_EVIDENCE_DATE",
            "challenger_identity": "EXACT_CANDIDATE_FAMILY_AND_PARAMETERS",
            "switch_rule": (
                "TWO_CONSECUTIVE_DISTINCT_MATURED_EVIDENCE_EXPANSIONS"
                "_WITH_STRICT_POSITIVE_ROBUST_SCORE_EDGE"
            ),
            "duplicate_evidence_can_advance_confirmation": False,
            "tie_break_only_switch_allowed": False,
            "all_arms_refit_on_identical_monthly_assessment_dates": True,
        },
        "execution_resources": active_cpu_contract(),
        "assessment_count": len(assessments),
        "final_hysteresis_state": state_to_dict(final_state),
        "diagnosis": diagnosis,
        "primary_contrast": (
            "HYSTERESIS_CAUSAL_RECIPE_MONTHLY_REFIT"
            "_MINUS_ROLLING_CAUSAL_RECIPE_MONTHLY_REFIT"
        ),
        "matched_control": (
            "FROZEN_FIRST_CAUSAL_RECIPE_MONTHLY_REFIT"
            "_WITH_IDENTICAL_REFIT_DATES"
        ),
        "portfolio_value_comparison": result_rows,
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, default=str, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    report = [
        "# Dynamic-QBD causal recipe-switch hysteresis",
        "",
        f"Status: **{diagnosis}**",
        "",
        "Authority: `SHADOW_ONLY_NO_PROMOTION_NO_HOLDOUT`.",
        "The final holdout remains closed.",
        "",
        "## Fixed scientific contract",
        "",
        "- Same monthly refit/recalibration dates in all three arms.",
        "- Frozen arm keeps the first causal recipe but refits it every assessment.",
        "- Rolling arm uses the existing causal selector winner every assessment.",
        "- Hysteresis arm requires two consecutive distinct matured-evidence expansions supporting the same exact challenger.",
        "- Challenger robust score must be strictly greater than the incumbent robust score; a selector tie-break alone cannot switch.",
        "- Duplicate matured evidence cannot add a confirmation.",
        "- Minimum two completed causal OOS folds for incumbent and challenger.",
        "- No hysteresis parameter grid, no evaluation-period tuning, no final holdout.",
        "",
        "## Portfolio comparison",
        "",
        comparison_markdown(comparison),
        "",
        "Recipe-change and model-family-switch counts are persisted in `portfolio-value-comparison.csv`; exact assessment decisions are in `hysteresis-decisions.csv`.",
        "",
        "## Interpretation",
        "",
        "The diagnosis uses strict Pareto logic only: hysteresis dominates rolling only if terminal value is no lower, relative MaxDD is no worse, and recipe changes do not increase. Otherwise the result is a trade-off or inferiority; no tolerance was tuned on this evaluation window.",
        "",
        "This experiment is Development/Research evidence only and cannot promote a Family, recipe, router, or capital allocation.",
    ]
    (output_root / "REPORT.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    return summary


def _parse_date(value: str) -> date:
    return date.fromisoformat(value)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the Dynamic-QBD causal recipe-switch hysteresis shadow test."
    )
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
    run_experiment(
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


if __name__ == "__main__":
    main()
