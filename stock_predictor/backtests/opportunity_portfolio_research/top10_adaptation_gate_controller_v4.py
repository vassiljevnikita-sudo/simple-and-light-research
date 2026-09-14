"""Causal adaptation-strength gate over the model-specific V3 memory controller.

Contract: TOP10_ADAPTATION_GATE_CONTROLLER_V4

Development reuse only. V4 keeps the V3 model-specific SHORT/MID/LONG priors and
shadow adaptive learner, but introduces a causal gate lambda_t in [0, 1]. The gate
measures whether the *previous* adaptive weights forecast the newly matured monthly
expert-quality snapshot better than the static prior. Only then may adaptation gain
production-facing influence.

No model fitting, prediction generation, threshold search, gate-parameter search,
or final holdout access occurs in this suite.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from .portfolio_research_inputs import load_price_panel
from .learned_exit_qbd_provider import LearnedExitProvider
from .next_open_portfolio_replay import prepare_prices
from .top10_causal_expanding_portfolio import LIVE_END, LIVE_START, _policy, _thresholds
from .top10_causal_monthly_weight_controller import (
    WARMUP_START,
    MONTH_SESSIONS,
    EXPERT_COUNT,
    ASSESS_EVERY_SESSIONS,
    _historical_entry,
    _combined_entry,
    _historical_pressure,
    _prior_suite_gates,
    _assessment_dates,
    _prepare_model_rows,
    _expert_snapshot,
    _replay_arm,
)
from .top10_entry_activation_alpha_diagnostic import (
    _causal_entry,
    _contracts,
    _markdown,
    _write_json,
)
from .top10_model_specific_memory_controller_v3 import (
    MODEL_PRIOR,
    CLASS_PROFILE,
    _prior,
    _profile_gate,
    _next,
    _validate as _validate_v3,
)

CONTRACT_ID = "TOP10_ADAPTATION_GATE_CONTROLLER_V4"
VALIDATION_STATUS = "DEVELOPMENT_REUSE_NOT_INDEPENDENT_OOS"

GATE_LOOKBACK_ASSESSMENTS = 6
MIN_GATE_OBSERVATIONS = 3
INITIAL_LAMBDA = 0.0
MAX_LAMBDA_CHANGE_PER_ASSESSMENT = 0.20
ADVANTAGE_Z_CLIP = 2.0
LAMBDA_TARGET_Z_FULL = 1.0


def _validate_contract() -> None:
    _validate_v3()
    if WARMUP_START != pd.Timestamp("2020-09-01"):
        raise RuntimeError("V4_WARMUP_CONTRACT_CHANGED")
    if MONTH_SESSIONS != 21 or EXPERT_COUNT != 12 or ASSESS_EVERY_SESSIONS != 21:
        raise RuntimeError("V4_PARENT_MONTH_CONTRACT_CHANGED")
    if not (0.0 <= INITIAL_LAMBDA <= 1.0):
        raise RuntimeError("V4_INITIAL_LAMBDA_INVALID")
    if not (0.0 < MAX_LAMBDA_CHANGE_PER_ASSESSMENT <= 1.0):
        raise RuntimeError("V4_LAMBDA_STEP_INVALID")
    if MIN_GATE_OBSERVATIONS < 1 or GATE_LOOKBACK_ASSESSMENTS < MIN_GATE_OBSERVATIONS:
        raise RuntimeError("V4_GATE_LOOKBACK_INVALID")
    if ADVANTAGE_Z_CLIP <= 0 or LAMBDA_TARGET_Z_FULL <= 0:
        raise RuntimeError("V4_GATE_SCALE_INVALID")


def _v3_reference_gate(summary_path: Path, results_path: Path) -> pd.DataFrame:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if (
        summary.get("status") != "COMPLETE"
        or summary.get("validation_status") != VALIDATION_STATUS
        or summary.get("no_model_training") is not True
        or summary.get("no_prediction_generation") is not True
        or summary.get("uses_existing_17000_causal_artifacts") is not True
        or summary.get("final_holdout_opened")
        or summary.get("model_system_selection_from_development_performance_allowed")
    ):
        raise RuntimeError("V4_V3_REFERENCE_SUMMARY_GATE")
    frame = pd.read_csv(results_path)
    adaptive = frame.loc[frame["arm"].astype(str).eq("ADAPTIVE_MODEL_CONTROLLER")].copy()
    if len(adaptive) != 10 or set(adaptive["model_id"].astype(str)) != set(MODEL_PRIOR):
        raise RuntimeError("V4_V3_REFERENCE_RESULTS_GATE")
    return adaptive[["model_id", "cagr_excess", "trade_count"]].rename(
        columns={
            "cagr_excess": "v3_frozen_cagr_excess",
            "trade_count": "v3_frozen_trade_count",
        }
    )


def _quality_scale(qualities: np.ndarray) -> tuple[float, float]:
    q = np.asarray(qualities, dtype=float)
    if len(q) != EXPERT_COUNT or not np.isfinite(q).all():
        raise RuntimeError("V4_GATE_QUALITY_INVALID")
    center = float(np.median(q))
    mad = float(np.median(np.abs(q - center)))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = float(np.std(q, ddof=0))
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = 1.0
    return center, scale


def _gate_observation(
    prior: np.ndarray,
    adaptive_before: np.ndarray,
    snapshot: list[dict],
) -> dict:
    if len(snapshot) != EXPERT_COUNT:
        raise RuntimeError("V4_GATE_SNAPSHOT_LENGTH")
    if not all(bool(row["available"]) for row in snapshot):
        return {
            "available": False,
            "prior_quality_score": np.nan,
            "adaptive_quality_score": np.nan,
            "advantage_raw": np.nan,
            "advantage_z": np.nan,
            "quality_center": np.nan,
            "quality_scale": np.nan,
        }
    qualities = np.array(
        [float(row["quality_mean_excess_clipped"]) for row in snapshot], dtype=float
    )
    center, scale = _quality_scale(qualities)
    prior_score = float(np.dot(prior, qualities))
    adaptive_score = float(np.dot(adaptive_before, qualities))
    raw = adaptive_score - prior_score
    z = float(np.clip(raw / scale, -ADVANTAGE_Z_CLIP, ADVANTAGE_Z_CLIP))
    return {
        "available": True,
        "prior_quality_score": prior_score,
        "adaptive_quality_score": adaptive_score,
        "advantage_raw": raw,
        "advantage_z": z,
        "quality_center": center,
        "quality_scale": scale,
    }


def _lambda_step(current_lambda: float, gate_history: list[float]) -> tuple[float, dict]:
    current = float(current_lambda)
    if not 0.0 <= current <= 1.0:
        raise RuntimeError("V4_LAMBDA_STATE_OUT_OF_RANGE")
    usable = [float(x) for x in gate_history if np.isfinite(x)]
    recent = usable[-GATE_LOOKBACK_ASSESSMENTS:]
    if len(recent) < MIN_GATE_OBSERVATIONS:
        return current, {
            "gate_observations": len(recent),
            "rolling_advantage_z": np.nan,
            "lambda_target": current,
            "lambda_updated": False,
            "lambda_reason": "WAIT_FOR_GATE_EVIDENCE",
        }
    rolling = float(np.mean(recent))
    target = float(np.clip(max(0.0, rolling) / LAMBDA_TARGET_Z_FULL, 0.0, 1.0))
    delta = float(
        np.clip(
            target - current,
            -MAX_LAMBDA_CHANGE_PER_ASSESSMENT,
            MAX_LAMBDA_CHANGE_PER_ASSESSMENT,
        )
    )
    updated = float(np.clip(current + delta, 0.0, 1.0))
    if abs(updated - current) > MAX_LAMBDA_CHANGE_PER_ASSESSMENT + 1e-12:
        raise RuntimeError("V4_LAMBDA_STEP_BREACH")
    return updated, {
        "gate_observations": len(recent),
        "rolling_advantage_z": rolling,
        "lambda_target": target,
        "lambda_updated": not np.isclose(updated, current, atol=1e-15),
        "lambda_reason": "ONE_STEP_ADAPTIVE_ADVANTAGE_GATE",
    }


def _schedule(
    prepared: pd.DataFrame,
    dates: list[pd.Timestamp],
    *,
    model_id: str,
    contract_id: str,
    prior_class: str,
    historical_pressure: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    prior = _prior(prior_class)
    adaptive_weights = prior.copy()
    gate_lambda = INITIAL_LAMBDA
    gate_history: list[float] = []
    states: list[dict] = []
    evidence: list[dict] = []

    for assessment_date in dates:
        snapshot = _expert_snapshot(prepared, assessment_date)

        # Crucial ordering: score the PREVIOUS adaptive state on the newly matured
        # snapshot before using that snapshot to compute the next adaptive state.
        gate_obs = _gate_observation(prior, adaptive_weights, snapshot)
        if gate_obs["available"]:
            gate_history.append(float(gate_obs["advantage_z"]))
        next_lambda, gate_update = _lambda_step(gate_lambda, gate_history)

        next_adaptive, adaptive_updated, rank_signal, adaptive_target = _next(
            adaptive_weights, prior, snapshot
        )
        gated_weights = (1.0 - next_lambda) * prior + next_lambda * next_adaptive
        if not np.isclose(gated_weights.sum(), 1.0, atol=1e-12):
            raise RuntimeError("V4_GATED_WEIGHT_SUM")
        if (gated_weights < -1e-12).any() or (gated_weights > 1.0 + 1e-12).any():
            raise RuntimeError("V4_GATED_WEIGHT_RANGE")

        p99 = np.array(
            [float(row["median_p99"]) if row["available"] else np.nan for row in snapshot],
            dtype=float,
        )
        all_p99 = np.isfinite(p99).all()
        static_p99 = float(np.dot(prior, p99)) if all_p99 else np.nan
        adaptive_p99 = float(np.dot(next_adaptive, p99)) if all_p99 else np.nan
        gated_p99 = float(np.dot(gated_weights, p99)) if all_p99 else np.nan

        def replacement(value: float) -> float:
            if np.isfinite(value) and value > 0:
                return float(historical_pressure * value)
            return np.nan

        state = {
            "model_id": model_id,
            "contract_id": contract_id,
            "prior_class": prior_class,
            "assessment_date": assessment_date,
            "phase": "WARMUP" if assessment_date < LIVE_START else "DEVELOPMENT_EVALUATION",
            "adaptive_updated": bool(adaptive_updated),
            "gate_observation_available": bool(gate_obs["available"]),
            "historical_threshold_over_p99": float(historical_pressure),
            "static_prior_replacement_threshold": replacement(static_p99),
            "adaptive_v3_replacement_threshold": replacement(adaptive_p99),
            "gated_v4_replacement_threshold": replacement(gated_p99),
            "prior_quality_score": gate_obs["prior_quality_score"],
            "adaptive_before_quality_score": gate_obs["adaptive_quality_score"],
            "gate_advantage_raw": gate_obs["advantage_raw"],
            "gate_advantage_z": gate_obs["advantage_z"],
            "gate_quality_center": gate_obs["quality_center"],
            "gate_quality_scale": gate_obs["quality_scale"],
            "gate_observations": gate_update["gate_observations"],
            "rolling_advantage_z": gate_update["rolling_advantage_z"],
            "lambda_before": float(gate_lambda),
            "lambda_target": float(gate_update["lambda_target"]),
            "lambda_after": float(next_lambda),
            "lambda_updated": bool(gate_update["lambda_updated"]),
            "lambda_reason": gate_update["lambda_reason"],
            "prior_effective_memory_months": float(np.dot(prior, np.arange(1, 13))),
            "adaptive_effective_memory_months": float(
                np.dot(next_adaptive, np.arange(1, 13))
            ),
            "gated_effective_memory_months": float(
                np.dot(gated_weights, np.arange(1, 13))
            ),
            "adaptive_l1_distance_from_prior": float(np.abs(next_adaptive - prior).sum()),
            "gated_l1_distance_from_prior": float(np.abs(gated_weights - prior).sum()),
        }
        for i in range(EXPERT_COUNT):
            state[f"prior_w_m{i+1}"] = float(prior[i])
            state[f"adaptive_w_m{i+1}"] = float(next_adaptive[i])
            state[f"gated_w_m{i+1}"] = float(gated_weights[i])
        states.append(state)

        for i, row in enumerate(snapshot):
            evidence.append(
                {
                    "model_id": model_id,
                    "contract_id": contract_id,
                    "prior_class": prior_class,
                    "assessment_date": assessment_date,
                    "phase": state["phase"],
                    "age_slot": f"M{i+1}",
                    "prior_weight": float(prior[i]),
                    "adaptive_weight_before": float(adaptive_weights[i]),
                    "adaptive_target_weight": float(adaptive_target[i]),
                    "adaptive_weight_after": float(next_adaptive[i]),
                    "gated_weight_after": float(gated_weights[i]),
                    "rank_signal": float(rank_signal[i]) if np.isfinite(rank_signal[i]) else np.nan,
                    "gate_lambda": float(next_lambda),
                    **row,
                }
            )

        adaptive_weights = next_adaptive
        gate_lambda = next_lambda

    state_frame = pd.DataFrame(states)
    if len(state_frame):
        if state_frame["lambda_after"].lt(-1e-12).any() or state_frame["lambda_after"].gt(1.0 + 1e-12).any():
            raise RuntimeError("V4_LAMBDA_RANGE_BREACH")
        if (
            (state_frame["lambda_after"] - state_frame["lambda_before"]).abs()
            > MAX_LAMBDA_CHANGE_PER_ASSESSMENT + 1e-12
        ).any():
            raise RuntimeError("V4_LAMBDA_SCHEDULE_STEP_BREACH")
    return state_frame, pd.DataFrame(evidence)


def _daily_thresholds(
    selected: pd.DataFrame,
    raw: pd.Series,
    schedule: pd.DataFrame,
) -> pd.DataFrame:
    states = list(schedule.sort_values("assessment_date").itertuples(index=False))
    rows = []
    decision_dates = sorted(
        selected.loc[
            selected["decision_date"].between(LIVE_START, LIVE_END), "decision_date"
        ].unique()
    )
    for value in decision_dates:
        decision_date = pd.Timestamp(value)
        raw_threshold = float(raw.loc[decision_date])
        past = [s for s in states if pd.Timestamp(s.assessment_date) < decision_date]
        latest = past[-1] if past else None
        if latest is None:
            static = adaptive = gated = np.nan
            source = pd.NaT
            lam = np.nan
        else:
            static = float(latest.static_prior_replacement_threshold)
            adaptive = float(latest.adaptive_v3_replacement_threshold)
            gated = float(latest.gated_v4_replacement_threshold)
            source = pd.Timestamp(latest.assessment_date)
            lam = float(latest.lambda_after)

        def effective(replacement_value: float) -> float:
            return (
                min(raw_threshold, replacement_value)
                if np.isfinite(replacement_value)
                else raw_threshold
            )

        rows.append(
            {
                "decision_date": decision_date,
                "raw_threshold": raw_threshold,
                "static_prior_effective_threshold": effective(static),
                "adaptive_v3_effective_threshold": effective(adaptive),
                "gated_v4_effective_threshold": effective(gated),
                "source_assessment_date": source,
                "gate_lambda": lam,
            }
        )

    frame = pd.DataFrame(rows)
    active = frame["source_assessment_date"].notna()
    if active.any() and frame.loc[active, "source_assessment_date"].ge(
        frame.loc[active, "decision_date"]
    ).any():
        raise RuntimeError("V4_NONCAUSAL_THRESHOLD_SOURCE")
    for column in (
        "static_prior_effective_threshold",
        "adaptive_v3_effective_threshold",
        "gated_v4_effective_threshold",
    ):
        if (frame[column] > frame["raw_threshold"] + 1e-12).any():
            raise RuntimeError(f"V4_THRESHOLD_TIGHTENED:{column}")
    return frame


def run(args: argparse.Namespace) -> dict:
    _validate_contract()
    output = Path(args.output_root)
    output.mkdir(parents=True, exist_ok=True)

    _prior_suite_gates(
        Path(args.retrospective_summary), Path(args.entry_diagnostic_summary)
    )
    prior_manifest = _profile_gate(Path(args.profile_summary), Path(args.profile_results))
    v3_reference = _v3_reference_gate(Path(args.v3_summary), Path(args.v3_results))

    historical, historical_audit = _historical_entry(Path(args.historical_entry_predictions))
    causal = _causal_entry(Path(args.causal_entry_predictions))
    combined = _combined_entry(historical, causal)
    contracts, contracts_by_model = _contracts(Path(args.frozen_policy_manifest))
    pressure = _historical_pressure(Path(args.entry_diagnostic_summary))

    prices, price_audit = load_price_panel(
        Path(args.daily_store_root), set(combined["ticker"].astype(str))
    )
    prepared_prices = prepare_prices(prices)
    dates = _assessment_dates(combined)

    manifest = json.loads(Path(args.frozen_policy_manifest).read_text(encoding="utf-8"))
    models = manifest.get("models", [])
    if len(models) != 10 or {str(m["model_id"]) for m in models} != set(MODEL_PRIOR):
        raise RuntimeError("V4_MANIFEST_MODELS")

    exits = LearnedExitProvider(Path(args.causal_exit_predictions))
    if exits.audit.min_date != str(LIVE_START.date()) or exits.audit.max_date != str(
        LIVE_END.date()
    ):
        raise RuntimeError("V4_EXIT_COVERAGE")

    all_states = []
    all_evidence = []
    all_daily = []
    results = []
    trades = []
    curves = []

    prior_lookup = prior_manifest.set_index("model_id")

    for model in models:
        model_id = str(model["model_id"])
        prior_class = MODEL_PRIOR[model_id]
        if str(prior_lookup.loc[model_id, "prior_class"]) != prior_class:
            raise RuntimeError(f"V4_PRIOR_MANIFEST_MISMATCH:{model_id}")

        policy = _policy(model["frozen_entry_policy"])
        spec = contracts_by_model[model_id]
        match = contracts.loc[
            contracts["horizon"].eq(int(spec["horizon"]))
            & contracts["score_quantile"].eq(float(spec["score_quantile"]))
            & contracts["top_fraction"].eq(float(spec["top_fraction"])),
            "contract_id",
        ]
        if len(match) != 1:
            raise RuntimeError(f"V4_CONTRACT_MAP:{model_id}:{len(match)}")
        contract_id = str(match.iloc[0])

        prepared = _prepare_model_rows(
            combined,
            prices,
            horizon=int(policy.horizon),
            top_fraction=float(policy.top_fraction),
            max_names=int(policy.max_names),
        )
        schedule, evidence = _schedule(
            prepared,
            dates,
            model_id=model_id,
            contract_id=contract_id,
            prior_class=prior_class,
            historical_pressure=pressure[contract_id],
        )
        all_states.append(schedule)
        all_evidence.append(evidence)

        selected = causal.loc[causal["horizon"].eq(int(policy.horizon))].copy()
        raw = _thresholds(selected, float(policy.score_quantile))
        daily = _daily_thresholds(selected, raw, schedule)
        daily.insert(0, "model_id", model_id)
        daily.insert(1, "prior_class", prior_class)
        all_daily.append(daily)

        signals = selected[["decision_date", "ticker", "score"]]
        arm_thresholds = {
            "RAW": daily.set_index("decision_date")["raw_threshold"],
            "STATIC_MODEL_PRIOR": daily.set_index("decision_date")[
                "static_prior_effective_threshold"
            ],
            "ADAPTIVE_V3_REFERENCE": daily.set_index("decision_date")[
                "adaptive_v3_effective_threshold"
            ],
            "GATED_ADAPTIVE_V4": daily.set_index("decision_date")[
                "gated_v4_effective_threshold"
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
                exit_provider=exits,
                prepared_prices=prepared_prices,
                initial_capital=args.initial_capital,
            )
            results.append(
                {
                    "model_id": model_id,
                    "prior_class": prior_class,
                    "source_profile": CLASS_PROFILE[prior_class],
                    "arm": arm,
                    "h": policy.horizon,
                    "d": policy.holding_days,
                    "n": policy.max_names,
                    "mode": policy.exit_family,
                    **metrics,
                }
            )
            trades.extend(arm_trades)
            curves.append(curve)
            print(
                f"V4_REPLAY_COMPLETE {model_id} {arm} trades={metrics['trade_count']}",
                flush=True,
            )

    state_frame = pd.concat(all_states, ignore_index=True)
    evidence_frame = pd.concat(all_evidence, ignore_index=True)
    daily_frame = pd.concat(all_daily, ignore_index=True)
    result_frame = pd.DataFrame(results).sort_values(["arm", "model_id"])

    state_frame.to_csv(output / "adaptation_gate_state.csv", index=False)
    state_frame.to_parquet(output / "adaptation_gate_state.parquet", index=False)
    evidence_frame.to_parquet(output / "adaptation_gate_expert_evidence.parquet", index=False)
    daily_frame.to_csv(output / "adaptation_gate_live_thresholds.csv", index=False)
    daily_frame.to_parquet(output / "adaptation_gate_live_thresholds.parquet", index=False)
    result_frame.to_csv(output / "adaptation_gate_results.csv", index=False)
    pd.DataFrame(trades).to_csv(output / "adaptation_gate_trades.csv", index=False)
    pd.concat(curves, ignore_index=True).to_parquet(
        output / "adaptation_gate_curves.parquet", index=False
    )
    prior_manifest.to_csv(output / "adaptation_gate_prior_manifest.csv", index=False)

    live_start_state = (
        state_frame.loc[state_frame["assessment_date"].lt(LIVE_START)]
        .sort_values("assessment_date")
        .groupby("model_id", as_index=False)
        .tail(1)
        .sort_values("model_id")
    )
    if len(live_start_state) != 10:
        raise RuntimeError(f"V4_LIVE_START_STATE_COUNT:{len(live_start_state)}")
    live_start_state.to_csv(output / "adaptation_gate_state_at_live_start.csv", index=False)

    # Reproduce frozen V3 exactly before interpreting any V4 delta.
    v3_replay = result_frame.loc[
        result_frame["arm"].eq("ADAPTIVE_V3_REFERENCE"),
        ["model_id", "cagr_excess", "trade_count"],
    ].rename(
        columns={
            "cagr_excess": "v3_replayed_cagr_excess",
            "trade_count": "v3_replayed_trade_count",
        }
    )
    comparison = v3_reference.merge(v3_replay, on="model_id", validate="one_to_one")
    comparison["v3_cagr_abs_error"] = (
        comparison["v3_replayed_cagr_excess"] - comparison["v3_frozen_cagr_excess"]
    ).abs()
    comparison["v3_trade_count_match"] = (
        comparison["v3_replayed_trade_count"].astype(int)
        == comparison["v3_frozen_trade_count"].astype(int)
    )
    if comparison["v3_cagr_abs_error"].max() > 1e-12 or not comparison[
        "v3_trade_count_match"
    ].all():
        raise RuntimeError("V4_V3_REFERENCE_REPRODUCTION_FAILED")

    gated = result_frame.loc[
        result_frame["arm"].eq("GATED_ADAPTIVE_V4"),
        ["model_id", "cagr_excess", "trade_count", "terminal_wealth_excess_eur"],
    ].rename(
        columns={
            "cagr_excess": "v4_gated_cagr_excess",
            "trade_count": "v4_gated_trade_count",
            "terminal_wealth_excess_eur": "v4_gated_terminal_wealth_excess_eur",
        }
    )
    comparison = comparison.merge(gated, on="model_id", validate="one_to_one")
    comparison["v4_minus_v3_cagr_excess"] = (
        comparison["v4_gated_cagr_excess"] - comparison["v3_frozen_cagr_excess"]
    )
    comparison["selection_allowed"] = False
    comparison.to_csv(output / "adaptation_gate_v4_vs_v3.csv", index=False)

    arm_summary = []
    for arm, group in result_frame.groupby("arm"):
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

    summary = {
        "contract_id": CONTRACT_ID,
        "status": "COMPLETE",
        "validation_status": VALIDATION_STATUS,
        "no_model_training": True,
        "no_prediction_generation": True,
        "uses_existing_17000_causal_artifacts": True,
        "no_threshold_optimization": True,
        "no_gate_parameter_grid_search": True,
        "no_online_prior_class_switching": True,
        "gate_uses_only_one_step_matured_evidence": True,
        "v3_reference_reproduced": True,
        "final_holdout_opened": False,
        "development_reuse": {
            "prior_assignment_uses_seen_2023_2026_fixed_profile_results": True,
            "v4_design_uses_seen_v3_development_results": True,
            "evaluation_is_not_independent_oos": True,
        },
        "warmup": {
            "start": str(WARMUP_START.date()),
            "end": "2023-08-10",
            "source": "historical WF_001..WF_007 OOS predictions",
        },
        "evaluation": {
            "start": str(LIVE_START.date()),
            "end": str(LIVE_END.date()),
            "selection_allowed": False,
        },
        "gate_contract": {
            "initial_lambda": INITIAL_LAMBDA,
            "lookback_assessments": GATE_LOOKBACK_ASSESSMENTS,
            "min_gate_observations": MIN_GATE_OBSERVATIONS,
            "max_lambda_change_per_assessment": MAX_LAMBDA_CHANGE_PER_ASSESSMENT,
            "advantage_z_clip": ADVANTAGE_Z_CLIP,
            "lambda_target": "clip(max(0, rolling_mean_one_step_advantage_z) / 1.0, 0, 1)",
            "evaluation_order": "score previous adaptive weights on newly matured snapshot before updating adaptive weights",
            "blend": "gated_weights=(1-lambda)*static_prior + lambda*v3_adaptive_weights",
            "effective_threshold": "min(raw_threshold, historical_pressure * gated_weighted_p99)",
            "update_effective": "strictly after assessment date",
            "outcome_visibility": "terminal_date <= assessment_date",
        },
        "arms": [
            "RAW",
            "STATIC_MODEL_PRIOR",
            "ADAPTIVE_V3_REFERENCE",
            "GATED_ADAPTIVE_V4",
        ],
        "arm_summary": arm_summary,
        "gate_lambda_updates": int(state_frame["lambda_updated"].sum()),
        "mean_live_lambda": float(
            state_frame.loc[
                state_frame["phase"].eq("DEVELOPMENT_EVALUATION"), "lambda_after"
            ].mean()
        ),
        "models_with_nonzero_lambda_at_live_start": int(
            live_start_state["lambda_after"].gt(0).sum()
        ),
        "v4_vs_v3": {
            "median_cagr_excess_delta": float(
                comparison["v4_minus_v3_cagr_excess"].median()
            ),
            "models_v4_better_cagr_excess": int(
                comparison["v4_minus_v3_cagr_excess"].gt(0).sum()
            ),
        },
        "historical_prediction_audit": historical_audit,
        "exit_provider_audit": asdict(exits.audit),
        "price_audit": price_audit,
        "selection_from_development_performance_allowed": False,
    }
    _write_json(output / "adaptation_gate_summary.json", summary)

    report_table = result_frame[
        [
            "model_id",
            "prior_class",
            "arm",
            "trade_count",
            "cagr_excess",
            "terminal_wealth_excess_eur",
        ]
    ].copy()
    report_table["cagr_excess"] = report_table["cagr_excess"].map(
        lambda x: f"{float(x):.4%}"
    )
    compare_table = comparison[
        [
            "model_id",
            "v3_frozen_cagr_excess",
            "v4_gated_cagr_excess",
            "v4_minus_v3_cagr_excess",
        ]
    ].copy()
    for column in (
        "v3_frozen_cagr_excess",
        "v4_gated_cagr_excess",
        "v4_minus_v3_cagr_excess",
    ):
        compare_table[column] = compare_table[column].map(lambda x: f"{float(x):.4%}")

    report = (
        "# Top-10 Adaptation Gate Controller V4\n\n"
        f"Status: **COMPLETE** / **{VALIDATION_STATUS}**\n\n"
        "V4 keeps the V3 adaptive learner in shadow mode and uses a causal lambda gate "
        "to decide how much of that adaptive state may override each model's static prior.\n\n"
        "The gate scores the previous adaptive weights against the newly matured monthly "
        "quality snapshot before that snapshot is used for the next adaptive update.\n\n"
        "## Development replay\n\n"
        + _markdown(report_table)
        + "\n\n## Frozen V3 reproduction and V4 delta\n\n"
        + _markdown(compare_table)
        + "\n\nNo model was trained, no prediction was regenerated, no threshold or gate grid was "
        "searched, and the Final Holdout remained closed. Selection from this development "
        "replay is prohibited.\n"
    )
    (output / "REPORT.md").write_text(report, encoding="utf-8")
    return summary


def _self_test() -> None:
    _validate_contract()
    prior = np.full(EXPERT_COUNT, 1.0 / EXPERT_COUNT)
    qualities = np.linspace(-0.06, 0.10, EXPERT_COUNT)
    adaptive = np.linspace(0.02, 0.15, EXPERT_COUNT)
    adaptive = adaptive / adaptive.sum()
    snapshot = [
        {
            "available": True,
            "quality_mean_excess_clipped": float(qualities[i]),
        }
        for i in range(EXPERT_COUNT)
    ]
    obs = _gate_observation(prior, adaptive, snapshot)
    if not obs["available"] or obs["advantage_z"] <= 0:
        raise AssertionError("V4_SELF_TEST_POSITIVE_ADVANTAGE")
    lam, detail = _lambda_step(0.0, [0.8, 0.8, 0.8])
    if not np.isclose(lam, MAX_LAMBDA_CHANGE_PER_ASSESSMENT):
        raise AssertionError("V4_SELF_TEST_LAMBDA_STEP")
    lam2, _ = _lambda_step(lam, [-1.0, -1.0, -1.0])
    if lam2 >= lam:
        raise AssertionError("V4_SELF_TEST_NEGATIVE_GATE")
    v = np.arange(1, EXPERT_COUNT + 1, dtype=float)
    v = v / v.sum()
    if not np.allclose((1.0 - 0.0) * prior + 0.0 * v, prior):
        raise AssertionError("V4_SELF_TEST_BLEND_ZERO")
    if not np.allclose((1.0 - 1.0) * prior + 1.0 * v, v):
        raise AssertionError("V4_SELF_TEST_BLEND_ONE")
    print("TOP10_ADAPTATION_GATE_CONTROLLER_V4_SELF_TEST_OK")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--historical-entry-predictions")
    parser.add_argument("--causal-entry-predictions")
    parser.add_argument("--causal-exit-predictions")
    parser.add_argument("--frozen-policy-manifest")
    parser.add_argument("--entry-diagnostic-summary")
    parser.add_argument("--retrospective-summary")
    parser.add_argument("--profile-summary")
    parser.add_argument("--profile-results")
    parser.add_argument("--v3-summary")
    parser.add_argument("--v3-results")
    parser.add_argument("--daily-store-root")
    parser.add_argument("--output-root")
    parser.add_argument("--initial-capital", type=float, default=10000.0)
    parser.add_argument("--self-test", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.self_test:
        _self_test()
        return
    required = [
        "historical_entry_predictions",
        "causal_entry_predictions",
        "causal_exit_predictions",
        "frozen_policy_manifest",
        "entry_diagnostic_summary",
        "retrospective_summary",
        "profile_summary",
        "profile_results",
        "v3_summary",
        "v3_results",
        "daily_store_root",
        "output_root",
    ]
    missing = [name for name in required if not getattr(args, name)]
    if missing:
        raise SystemExit("V4_REQUIRED_ARGS_MISSING:" + ",".join(missing))
    run(args)


if __name__ == "__main__":
    main()
