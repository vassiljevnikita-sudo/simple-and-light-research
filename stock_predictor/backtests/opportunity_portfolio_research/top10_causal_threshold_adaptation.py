"""Causal, maturity-aware threshold adaptation for the frozen Top-10 policies.

Contract: TOP10_CAUSAL_THRESHOLD_ADAPTATION_V1

Reuses existing causal entry/exit predictions. No model fitting, prediction generation,
threshold optimization, weight optimization, or final-holdout access.

Three weighting profiles are predeclared and evaluated in parallel for 1M/3M/6M
retrospective evidence signals:
- W_RECENT: 0.50 / 0.30 / 0.20
- W_MID:    0.20 / 0.50 / 0.30
- W_LONG:   0.20 / 0.30 / 0.50

At each 21-session assessment date, outcome evidence may use only rows whose terminal
return is already known. Any adaptation becomes effective strictly after the assessment.
The replacement threshold is a score-scale correction only and can only relax the raw
threshold, never tighten it.
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
from .top10_retrospective_threshold_validation import _historical_crossing_baseline, _prepare_contract_rows

CONTRACT_ID = "TOP10_CAUSAL_THRESHOLD_ADAPTATION_V1"
ASSESS_EVERY_SESSIONS = 21
LOOKBACKS = {"1M": 21, "3M": 63, "6M": 126}
WEIGHT_PROFILES = {
    "W_RECENT": {"1M": 0.50, "3M": 0.30, "6M": 0.20},
    "W_MID": {"1M": 0.20, "3M": 0.50, "6M": 0.30},
    "W_LONG": {"1M": 0.20, "3M": 0.30, "6M": 0.50},
}
ADAPT_TRIGGER = 0.50
MIN_MATURED_FRACTION = 2.0 / 3.0
MIN_SHADOW_CANDIDATES = 30


def _validate_weights() -> None:
    for name, weights in WEIGHT_PROFILES.items():
        if set(weights) != set(LOOKBACKS):
            raise RuntimeError(f"WEIGHT_PROFILE_WINDOWS_INVALID:{name}")
        if abs(sum(weights.values()) - 1.0) > 1e-12:
            raise RuntimeError(f"WEIGHT_PROFILE_NOT_NORMALIZED:{name}")
        if any(value < 0 for value in weights.values()):
            raise RuntimeError(f"WEIGHT_PROFILE_NEGATIVE:{name}")


def _prior_suite_gates(retrospective_path: Path, diagnostic_path: Path) -> None:
    retrospective = json.loads(retrospective_path.read_text(encoding="utf-8"))
    if retrospective.get("status") != "COMPLETE":
        raise RuntimeError("RETROSPECTIVE_THRESHOLD_VALIDATION_NOT_COMPLETE")
    if retrospective.get("no_forward_outcome_use") is not True:
        raise RuntimeError("RETROSPECTIVE_CAUSALITY_GATE_FAILED")
    if retrospective.get("no_threshold_optimization") is not True:
        raise RuntimeError("RETROSPECTIVE_THRESHOLD_OPTIMIZATION_GATE_FAILED")
    if retrospective.get("final_holdout_opened"):
        raise RuntimeError("FINAL_HOLDOUT_MUST_REMAIN_CLOSED")

    diagnostic = json.loads(diagnostic_path.read_text(encoding="utf-8"))
    if diagnostic.get("status") != "COMPLETE":
        raise RuntimeError("ENTRY_DIAGNOSTIC_NOT_COMPLETE")
    if diagnostic.get("no_model_training") is not True or diagnostic.get("no_prediction_generation") is not True:
        raise RuntimeError("ENTRY_DIAGNOSTIC_MUTATION_GATE_FAILED")
    if diagnostic.get("final_holdout_opened"):
        raise RuntimeError("FINAL_HOLDOUT_MUST_REMAIN_CLOSED")


def _historical_pressure(path: Path) -> dict[str, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    result = {}
    for row in payload.get("activation_summary", []):
        if row.get("source") != "HISTORICAL_WF":
            continue
        value = float(row["median_threshold_over_p99"])
        if not np.isfinite(value) or value <= 0:
            raise RuntimeError(f"HISTORICAL_PRESSURE_INVALID:{row.get('contract_id')}:{value}")
        result[str(row["contract_id"])] = value
    if not result:
        raise RuntimeError("HISTORICAL_PRESSURE_BASELINE_MISSING")
    return result


def _assessment_dates(causal: pd.DataFrame) -> list[pd.Timestamp]:
    dates = [pd.Timestamp(x) for x in sorted(causal["decision_date"].drop_duplicates())]
    if not dates or dates[0] != LIVE_START or dates[-1] != LIVE_END:
        raise RuntimeError("CAUSAL_DATE_COVERAGE_MISMATCH")
    anchors = dates[ASSESS_EVERY_SESSIONS - 1 :: ASSESS_EVERY_SESSIONS]
    if dates[-1] not in anchors:
        anchors.append(dates[-1])
    return anchors


def _window_signal(prepared: pd.DataFrame, assessment_date: pd.Timestamp, lookback: int,
                   historical_crossing_rate: float) -> dict:
    matured = prepared.loc[
        prepared["realized_price_available"]
        & prepared["terminal_date"].le(assessment_date)
        & prepared["decision_date"].lt(assessment_date)
    ].copy()
    matured_dates = [pd.Timestamp(x) for x in sorted(matured["decision_date"].drop_duplicates())]
    window_dates = matured_dates[-int(lookback):]
    window = matured.loc[matured["decision_date"].isin(window_dates)].copy()
    min_days = max(1, int(np.ceil(lookback * MIN_MATURED_FRACTION)))

    if not window_dates:
        return {"available": False, "signal": np.nan, "matured_days": 0, "shadow_count": 0,
                "recent_crossing_rate": np.nan, "shadow_mean_excess": np.nan, "shadow_hit_rate": np.nan,
                "latest_terminal_date_used": pd.NaT}

    daily = window.groupby("decision_date")["accepted_by_threshold"].max()
    shadow = window.loc[window["shadow_rejected_candidate"]]
    crossing = float(daily.mean()) if len(daily) else np.nan
    shadow_mean = float(shadow["realized_net_excess_20bps"].mean()) if len(shadow) else np.nan
    shadow_hit = float(shadow["realized_net_excess_20bps"].gt(0).mean()) if len(shadow) else np.nan
    available = len(window_dates) >= min_days and len(shadow) >= MIN_SHADOW_CANDIDATES
    collapse = (
        available
        and historical_crossing_rate >= 0.01
        and crossing < max(0.005, 0.25 * historical_crossing_rate)
    )
    too_restrictive = bool(collapse and np.isfinite(shadow_mean) and shadow_mean > 0.0)
    return {
        "available": bool(available),
        "signal": 1.0 if too_restrictive else (0.0 if available else np.nan),
        "matured_days": int(len(window_dates)),
        "shadow_count": int(len(shadow)),
        "recent_crossing_rate": crossing,
        "shadow_mean_excess": shadow_mean,
        "shadow_hit_rate": shadow_hit,
        "latest_terminal_date_used": pd.Timestamp(window["terminal_date"].max()) if len(window) else pd.NaT,
    }


def _score_anchor(contract_rows: pd.DataFrame, assessment_date: pd.Timestamp, lookback: int) -> dict:
    scores = contract_rows.loc[
        contract_rows["decision_date"].lt(assessment_date), ["decision_date", "score"]
    ].copy()
    dates = [pd.Timestamp(x) for x in sorted(scores["decision_date"].drop_duplicates())][-int(lookback):]
    if not dates:
        return {"available": False, "median_p99": np.nan, "decision_days": 0}
    p99 = scores.loc[scores["decision_date"].isin(dates)].groupby("decision_date")["score"].quantile(0.99)
    value = float(p99.median()) if len(p99) else np.nan
    return {"available": bool(np.isfinite(value) and value > 0), "median_p99": value, "decision_days": len(dates)}


def _weighted(values: dict[str, float], available: dict[str, bool], weights: dict[str, float]) -> tuple[float, dict[str, float]]:
    live_weights = {name: weights[name] for name in LOOKBACKS if available.get(name) and np.isfinite(values.get(name, np.nan))}
    total = float(sum(live_weights.values()))
    if total <= 0:
        return np.nan, {name: 0.0 for name in LOOKBACKS}
    effective = {name: float(live_weights.get(name, 0.0) / total) for name in LOOKBACKS}
    result = float(sum(effective[name] * values[name] for name in LOOKBACKS if effective[name] > 0))
    return result, effective


def _build_schedule(causal: pd.DataFrame, prepared: pd.DataFrame, contract_id: str,
                    horizon: int, score_quantile: float, top_fraction: float,
                    historical_crossing_rate: float, historical_pressure: float,
                    anchors: list[pd.Timestamp]) -> pd.DataFrame:
    contract_rows = causal.loc[causal["horizon"].eq(horizon)].copy()
    rows = []
    for assessment_date in anchors:
        signals = {name: _window_signal(prepared, assessment_date, sessions, historical_crossing_rate)
                   for name, sessions in LOOKBACKS.items()}
        score_anchors = {name: _score_anchor(contract_rows, assessment_date, sessions)
                         for name, sessions in LOOKBACKS.items()}
        for profile, weights in WEIGHT_PROFILES.items():
            evidence_values = {name: float(signals[name]["signal"]) for name in LOOKBACKS}
            evidence_available = {name: bool(signals[name]["available"]) for name in LOOKBACKS}
            weighted_evidence, evidence_weights = _weighted(evidence_values, evidence_available, weights)

            p99_values = {name: float(score_anchors[name]["median_p99"]) for name in LOOKBACKS}
            p99_available = {name: bool(score_anchors[name]["available"]) for name in LOOKBACKS}
            weighted_p99, p99_weights = _weighted(p99_values, p99_available, weights)

            adapt = bool(np.isfinite(weighted_evidence) and weighted_evidence >= ADAPT_TRIGGER
                         and np.isfinite(weighted_p99) and weighted_p99 > 0)
            replacement = float(historical_pressure * weighted_p99) if adapt else np.nan
            row = {
                "contract_id": contract_id, "horizon": horizon, "score_quantile": score_quantile,
                "top_fraction": top_fraction, "profile": profile, "assessment_date": assessment_date,
                "historical_crossing_day_rate": historical_crossing_rate,
                "historical_threshold_over_p99": historical_pressure,
                "weighted_evidence": weighted_evidence, "weighted_p99_anchor": weighted_p99,
                "adapt_active": adapt, "replacement_threshold": replacement,
            }
            for name in LOOKBACKS:
                key = name.lower()
                row[f"{key}_base_weight"] = weights[name]
                row[f"{key}_effective_evidence_weight"] = evidence_weights[name]
                row[f"{key}_effective_p99_weight"] = p99_weights[name]
                row[f"{key}_signal"] = signals[name]["signal"]
                row[f"{key}_matured_days"] = signals[name]["matured_days"]
                row[f"{key}_shadow_count"] = signals[name]["shadow_count"]
                row[f"{key}_recent_crossing_rate"] = signals[name]["recent_crossing_rate"]
                row[f"{key}_shadow_mean_excess"] = signals[name]["shadow_mean_excess"]
                row[f"{key}_shadow_hit_rate"] = signals[name]["shadow_hit_rate"]
                row[f"{key}_latest_terminal_date_used"] = signals[name]["latest_terminal_date_used"]
                row[f"{key}_median_p99"] = score_anchors[name]["median_p99"]
            rows.append(row)
    return pd.DataFrame(rows)


def _effective_thresholds(selected: pd.DataFrame, raw: pd.Series, schedule: pd.DataFrame,
                          profile: str) -> tuple[pd.Series, pd.DataFrame]:
    dates = [pd.Timestamp(x) for x in sorted(selected["decision_date"].drop_duplicates())]
    scheduled = list(schedule.loc[schedule["profile"].eq(profile)].sort_values("assessment_date").itertuples(index=False))
    rows = []
    for decision_date in dates:
        raw_threshold = float(raw.loc[decision_date])
        prior = [row for row in scheduled if pd.Timestamp(row.assessment_date) < decision_date]
        latest = prior[-1] if prior else None
        if latest is None or not bool(latest.adapt_active) or not np.isfinite(float(latest.replacement_threshold)):
            replacement = np.nan
            effective = raw_threshold
            source_date = pd.NaT
            evidence = np.nan
        else:
            replacement = float(latest.replacement_threshold)
            effective = min(raw_threshold, replacement)
            source_date = pd.Timestamp(latest.assessment_date)
            evidence = float(latest.weighted_evidence)
        rows.append({
            "decision_date": decision_date, "raw_threshold": raw_threshold,
            "replacement_threshold": replacement, "effective_threshold": effective,
            "adapted": bool(effective < raw_threshold - 1e-15),
            "source_assessment_date": source_date, "weighted_evidence": evidence,
        })
    daily = pd.DataFrame(rows)
    if (daily["effective_threshold"] > daily["raw_threshold"] + 1e-12).any():
        raise RuntimeError("ADAPTATION_MADE_THRESHOLD_MORE_RESTRICTIVE")
    active = daily["source_assessment_date"].notna()
    if active.any() and daily.loc[active, "source_assessment_date"].ge(daily.loc[active, "decision_date"]).any():
        raise RuntimeError("ADAPTATION_ASSESSMENT_NOT_STRICTLY_PRIOR")
    return daily.set_index("decision_date")["effective_threshold"], daily


def run(args: argparse.Namespace) -> dict:
    _validate_weights()
    output = Path(args.output_root)
    output.mkdir(parents=True, exist_ok=True)
    _prior_suite_gates(Path(args.retrospective_summary), Path(args.entry_diagnostic_summary))

    causal = _causal_entry(Path(args.causal_entry_predictions))
    contracts, contracts_by_model = _contracts(Path(args.frozen_policy_manifest))
    crossing_baseline = _historical_crossing_baseline(Path(args.entry_diagnostic_summary))
    pressure_baseline = _historical_pressure(Path(args.entry_diagnostic_summary))
    anchors = _assessment_dates(causal)

    tickers = set(causal["ticker"].astype(str))
    prices, price_audit = load_price_panel(Path(args.daily_store_root), tickers)
    if pd.Timestamp(prices["date"].max()).normalize() < LIVE_END:
        raise RuntimeError("DAILY_PRICES_END_BEFORE_LIVE_END")
    prepared_prices = prepare_prices(prices)

    schedules = {}
    all_schedule = []
    for contract in contracts.itertuples(index=False):
        if contract.contract_id not in crossing_baseline or contract.contract_id not in pressure_baseline:
            raise RuntimeError(f"HISTORICAL_BASELINE_MISSING:{contract.contract_id}")
        prepared = _prepare_contract_rows(causal, prices, int(contract.horizon),
                                          float(contract.score_quantile), float(contract.top_fraction))
        schedule = _build_schedule(
            causal, prepared, contract.contract_id, int(contract.horizon), float(contract.score_quantile),
            float(contract.top_fraction), crossing_baseline[contract.contract_id],
            pressure_baseline[contract.contract_id], anchors,
        )
        schedules[contract.contract_id] = schedule
        all_schedule.append(schedule)
    schedule_frame = pd.concat(all_schedule, ignore_index=True)
    schedule_frame.to_csv(output / "causal_threshold_adaptation_schedule.csv", index=False)
    schedule_frame.to_parquet(output / "causal_threshold_adaptation_schedule.parquet", index=False)

    exit_provider = LearnedExitProvider(Path(args.causal_exit_predictions))
    if exit_provider.audit.min_date != str(LIVE_START.date()) or exit_provider.audit.max_date != str(LIVE_END.date()):
        raise RuntimeError(f"CAUSAL_EXIT_DATE_COVERAGE_MISMATCH:{asdict(exit_provider.audit)}")

    manifest = json.loads(Path(args.frozen_policy_manifest).read_text(encoding="utf-8"))
    models = manifest.get("models", [])
    if len(models) != 10:
        raise RuntimeError(f"FROZEN_TOP10_COUNT_MISMATCH:{len(models)}")

    results, trades, curves, daily_thresholds = [], [], [], []
    for model in models:
        model_id = str(model["model_id"])
        policy = _policy(model["frozen_entry_policy"])
        spec = contracts_by_model[model_id]
        match = contracts.loc[
            contracts["horizon"].eq(int(spec["horizon"]))
            & contracts["score_quantile"].eq(float(spec["score_quantile"]))
            & contracts["top_fraction"].eq(float(spec["top_fraction"])), "contract_id"
        ]
        if len(match) != 1:
            raise RuntimeError(f"MODEL_CONTRACT_MAPPING_INVALID:{model_id}:{len(match)}")
        contract_id = str(match.iloc[0])
        selected = causal.loc[causal["horizon"].eq(policy.horizon)].copy()
        raw = _thresholds(selected, policy.score_quantile)
        signals = selected[["decision_date", "ticker", "score"]]

        for profile in WEIGHT_PROFILES:
            effective, daily = _effective_thresholds(selected, raw, schedules[contract_id], profile)
            daily.insert(0, "model_id", model_id)
            daily.insert(1, "contract_id", contract_id)
            daily.insert(2, "profile", profile)
            daily_thresholds.append(daily)

            configure_replay(exit_provider, ProfitTaxConfig())
            result = replay(
                signals, prices, policy, CostModel(20), TaxConfig(False),
                start=LIVE_START, end=LIVE_END, initial=float(args.initial_capital),
                resolved_threshold_by_date=effective, prepared_prices=prepared_prices,
            )
            metrics = result["metrics"]
            if int(metrics.get("learned_exit_missing_prediction_count", 0)):
                raise RuntimeError(f"LEARNED_EXIT_MISSING:{model_id}:{profile}")
            if float(metrics.get("exit_decision_coverage", 0.0)) != 1.0:
                raise RuntimeError(f"LEARNED_EXIT_COVERAGE_INCOMPLETE:{model_id}:{profile}")
            results.append({
                "model_id": model_id, "contract_id": contract_id, "profile": profile,
                "w_1m": WEIGHT_PROFILES[profile]["1M"], "w_3m": WEIGHT_PROFILES[profile]["3M"],
                "w_6m": WEIGHT_PROFILES[profile]["6M"], "h": policy.horizon,
                "d": policy.holding_days, "n": policy.max_names, "mode": policy.exit_family,
                "score_quantile": policy.score_quantile, "top_fraction": policy.top_fraction,
                **metrics,
            })
            for trade in result["trades"]:
                trades.append({"model_id": model_id, "contract_id": contract_id, "profile": profile, **trade})
            curve = result["curve"].copy()
            curve.insert(0, "model_id", model_id)
            curve.insert(1, "contract_id", contract_id)
            curve.insert(2, "profile", profile)
            curves.append(curve)
            print(f"CAUSAL_THRESHOLD_ADAPTATION_COMPLETE {model_id} {profile} trades={metrics['trade_count']}", flush=True)

    daily_frame = pd.concat(daily_thresholds, ignore_index=True)
    daily_frame.to_csv(output / "causal_threshold_daily_thresholds.csv", index=False)
    daily_frame.to_parquet(output / "causal_threshold_daily_thresholds.parquet", index=False)
    ranking = pd.DataFrame(results).sort_values(["profile", "model_id"]).reset_index(drop=True)
    ranking.to_csv(output / "causal_threshold_adaptation_results.csv", index=False)
    pd.DataFrame(trades).to_csv(output / "causal_threshold_adaptation_trades.csv", index=False)
    pd.concat(curves, ignore_index=True).to_parquet(output / "causal_threshold_adaptation_curves.parquet", index=False)

    profile_summary = []
    for profile, group in ranking.groupby("profile", sort=True):
        profile_summary.append({
            "profile": profile, "weights": WEIGHT_PROFILES[profile], "models": len(group),
            "total_trades": int(group["trade_count"].sum()),
            "positive_cagr_excess_models": int(group["cagr_excess"].gt(0).sum()),
            "median_cagr_excess": float(group["cagr_excess"].median()),
            "mean_cagr_excess": float(group["cagr_excess"].mean()),
            "selection_allowed": False,
        })

    summary = {
        "contract_id": CONTRACT_ID, "status": "COMPLETE",
        "no_model_training": True, "no_prediction_generation": True,
        "no_threshold_optimization": True, "no_weight_optimization": True,
        "weight_profiles_predeclared": WEIGHT_PROFILES,
        "profile_selection_from_forward_performance_allowed": False,
        "final_holdout_opened": False,
        "causality": {
            "assessment_every_sessions": ASSESS_EVERY_SESSIONS,
            "evidence_lookbacks_sessions": LOOKBACKS,
            "outcome_visibility_rule": "terminal_date <= assessment_date",
            "adaptation_effective_rule": "strictly after assessment_date",
            "adapt_trigger": ADAPT_TRIGGER,
            "replacement_threshold_rule": "historical_threshold_over_p99 * weighted_recent_p99_anchor",
            "effective_threshold_rule": "min(raw_threshold, replacement_threshold)",
        },
        "profile_summary": profile_summary,
        "exit_provider_audit": asdict(exit_provider.audit), "price_audit": price_audit,
    }
    _write_json(output / "causal_threshold_adaptation_summary.json", summary)

    report = ranking[["model_id", "profile", "w_1m", "w_3m", "w_6m", "trade_count",
                      "cagr_excess", "terminal_wealth_excess_eur"]].copy()
    report["cagr_excess"] = report["cagr_excess"].map(lambda x: f"{float(x):.4%}")
    text = [
        "# Top-10 Causal Threshold Adaptation", "", "Status: **COMPLETE**", "",
        "Three weighting profiles were predeclared and run in parallel. This suite does not promote a winner from forward performance.", "",
        "- W_RECENT = 0.50 / 0.30 / 0.20 for 1M / 3M / 6M",
        "- W_MID = 0.20 / 0.50 / 0.30",
        "- W_LONG = 0.20 / 0.30 / 0.50", "",
        "Outcome evidence uses only fully matured rows at each assessment. Any change becomes effective on the next decision session.",
        "The replacement threshold can only relax, never tighten, the original daily threshold.", "",
        "## Results", "", _markdown(report), "",
        "No models were trained, no predictions were regenerated, no threshold or weight profile was optimized, and the final holdout remained closed.",
    ]
    (output / "REPORT.md").write_text("\n".join(text) + "\n", encoding="utf-8")
    return summary


def self_test() -> None:
    _validate_weights()
    assert WEIGHT_PROFILES["W_RECENT"] == {"1M": 0.50, "3M": 0.30, "6M": 0.20}
    assert WEIGHT_PROFILES["W_MID"] == {"1M": 0.20, "3M": 0.50, "6M": 0.30}
    assert WEIGHT_PROFILES["W_LONG"] == {"1M": 0.20, "3M": 0.30, "6M": 0.50}
    available = {"1M": True, "3M": True, "6M": True}
    one, _ = _weighted({"1M": 1.0, "3M": 0.0, "6M": 0.0}, available, WEIGHT_PROFILES["W_RECENT"])
    three, _ = _weighted({"1M": 0.0, "3M": 1.0, "6M": 0.0}, available, WEIGHT_PROFILES["W_MID"])
    six, _ = _weighted({"1M": 0.0, "3M": 0.0, "6M": 1.0}, available, WEIGHT_PROFILES["W_LONG"])
    assert one == ADAPT_TRIGGER and three == ADAPT_TRIGGER and six == ADAPT_TRIGGER
    print("TOP10_CAUSAL_THRESHOLD_ADAPTATION_SELF_TEST_OK")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--causal-entry-predictions")
    parser.add_argument("--causal-exit-predictions")
    parser.add_argument("--frozen-policy-manifest")
    parser.add_argument("--entry-diagnostic-summary")
    parser.add_argument("--retrospective-summary")
    parser.add_argument("--daily-store-root")
    parser.add_argument("--output-root", default="artifacts/top10-causal-threshold-adaptation")
    parser.add_argument("--initial-capital", type=float, default=10000.0)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    required = ("causal_entry_predictions", "causal_exit_predictions", "frozen_policy_manifest",
                "entry_diagnostic_summary", "retrospective_summary", "daily_store_root")
    missing = [name for name in required if not getattr(args, name)]
    if missing:
        parser.error("required arguments missing: " + ", ".join("--" + x.replace("_", "-") for x in missing))
    print(json.dumps(run(args), indent=2, default=str))


if __name__ == "__main__":
    main()
