"""Compare the frozen Top-10 known-OOS evidence with causal expanding-live replays."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from .cost_contracts import CostModel
from .portfolio_policy_contracts import Policy
from .tax_contracts import TaxConfig
from .portfolio_research_inputs import load_predictions, load_price_panel
from .learned_exit_qbd_profit import ProfitTaxConfig
from .learned_exit_qbd_provider import LearnedExitProvider
from .learned_exit_qbd_replay import configure_replay, replay
from .next_open_portfolio_replay import prepare_prices


CONTRACT_ID = "TOP10_HISTORICAL_REPRODUCTION_AND_CAUSAL_EXPANDING_LIVE_V1"
LIVE_START = pd.Timestamp("2023-08-11")
LIVE_END = pd.Timestamp("2026-07-24")


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    temporary.replace(path)


def _markdown_table(frame: pd.DataFrame) -> str:
    """Dependency-free, deterministic Markdown table for the final report."""
    columns = [str(column) for column in frame.columns]
    rows = [[str(value) for value in row] for row in frame.itertuples(index=False, name=None)]
    return "\n".join([
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
        *("| " + " | ".join(row) + " |" for row in rows),
    ])


def _historical_gate(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    historical = payload.get("historical_reproduction", {})
    feature = payload.get("feature_coverage", {})
    if historical.get("status") != "PASSED":
        raise RuntimeError("HISTORICAL_REPRODUCTION_GATE_FAILED")
    if feature.get("status") != "PASSED" or feature.get("observed_common_feature_end") != str(LIVE_END.date()):
        raise RuntimeError("FEATURE_COVERAGE_GATE_FAILED")
    entry_rows = historical.get("entry_rows", [])
    exit_rows = historical.get("exit_rows", [])
    if len(entry_rows) != 21 or len(exit_rows) != 140:
        raise RuntimeError(f"WF_REPRODUCTION_MODEL_COUNT_MISMATCH:{len(entry_rows)}:{len(exit_rows)}")
    if any(row.get("status") != "EXACT_REPRODUCTION" for row in [*entry_rows, *exit_rows]):
        raise RuntimeError("WF_REPRODUCTION_NOT_EXACT")
    threshold_mismatches = sum(
        int(item.get("crossing_mismatches", 0))
        for row in entry_rows for item in row.get("threshold_crossings", {}).values()
    )
    if threshold_mismatches:
        raise RuntimeError(f"WF_THRESHOLD_CROSSING_MISMATCH:{threshold_mismatches}")
    return {
        "status": "PASSED",
        "entry_models": len(entry_rows),
        "exit_models": len(exit_rows),
        "threshold_crossing_mismatches": threshold_mismatches,
        "max_entry_score_delta": max(float(row["max_abs_score_delta"]) for row in entry_rows),
        "max_exit_score_delta": max(float(row["max_abs_score_delta"]) for row in exit_rows),
        "feature_end": feature["observed_common_feature_end"],
        "source": str(path),
    }


def _entry_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    required = {
        "decision_date", "ticker", "horizon_sessions", "predicted_net_excess_return",
        "model_training_cutoff",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"CAUSAL_ENTRY_COLUMNS_MISSING:{missing}")
    frame["decision_date"] = pd.to_datetime(frame["decision_date"]).dt.normalize()
    frame["ticker"] = frame["ticker"].astype(str)
    frame["horizon"] = frame["horizon_sessions"].astype(int)
    frame["score"] = pd.to_numeric(frame["predicted_net_excess_return"], errors="coerce")
    frame["model_training_cutoff"] = pd.to_datetime(frame["model_training_cutoff"]).dt.normalize()
    if (frame["model_training_cutoff"] >= frame["decision_date"]).any():
        raise RuntimeError("CAUSAL_ENTRY_TRAINING_CUTOFF_NOT_PRIOR")
    if frame["decision_date"].min() != LIVE_START or frame["decision_date"].max() != LIVE_END:
        raise RuntimeError("CAUSAL_ENTRY_DATE_COVERAGE_MISMATCH")
    return frame


def _thresholds(frame: pd.DataFrame, quantile: float) -> pd.Series:
    column = f"resolved_threshold_q{str(quantile).replace('.', '_')}"
    if column not in frame:
        raise RuntimeError(f"CAUSAL_THRESHOLD_COLUMN_MISSING:{column}")
    uniqueness = frame.groupby("decision_date")[column].nunique(dropna=False)
    if not uniqueness.eq(1).all():
        raise RuntimeError(f"CAUSAL_THRESHOLD_NOT_MODEL_DATE_SPECIFIC:{column}")
    result = frame.groupby("decision_date")[column].first()
    if result.isna().any():
        raise RuntimeError(f"CAUSAL_THRESHOLD_NAN:{column}")
    return result


def _policy(value: dict) -> Policy:
    fields = set(Policy.__dataclass_fields__)
    return Policy(**{key: value[key] for key in fields})


def _historical_outer_replays(
    models: list[dict], predictions: pd.DataFrame, prices: pd.DataFrame, exit_path: Path,
    prepared_price_data: tuple | None = None,
    *, legacy_historical_mode: bool,
) -> dict:
    """Replay frozen outer rows under the requested historical replay contract."""
    configure_replay(LearnedExitProvider(exit_path), ProfitTaxConfig())
    delisting_rules = {
        "EXE": {
            "security_id": "CHK_PRE_2021", "ticker_at_entry": "CHK",
            "valid_entry_through": "2020-06-26", "series_end": "2020-06-26",
            "exit_reason": "DELISTING_ZERO_RECOVERY_ASSUMPTION", "terminal_value": 0.0,
            "terminal_value_source": "FROZEN_ZERO_RECOVERY_NO_VERIFIED_DISTRIBUTION",
        }
    }
    comparisons = []
    cache = {}
    prepared_price_data = prepared_price_data or prepare_prices(prices)
    metric_fields = ("terminal_value", "urth_terminal_value", "total_return", "cagr_excess", "turnover")
    for model in models:
        source = Path(model["source_path"])
        payload = json.loads(source.read_text(encoding="utf-8"))
        cache_key = str(source.resolve())
        if cache_key in cache:
            comparisons.extend({**row, "model_id": model["model_id"]} for row in cache[cache_key])
            continue
        source_rows = []
        for expected in payload.get("outer_rows", []):
            fold_id = str(expected["fold_id"])
            horizon = int(expected["horizon"])
            fold_predictions = predictions.loc[
                predictions["horizon"].eq(horizon) & predictions["fold_id"].eq(fold_id)
            ]
            policy = _policy(expected)
            actual = replay(
                fold_predictions, prices, policy, CostModel(20), TaxConfig(False),
                start=pd.Timestamp(expected["fold_start"]), end=pd.Timestamp(expected["fold_end"]),
                initial=10000.0, resolved_threshold=float(expected["threshold"]),
                delisting_rules=None if legacy_historical_mode else delisting_rules,
                prepared_prices=prepared_price_data,
                legacy_historical_mode=legacy_historical_mode,
            )
            metrics = actual["metrics"]
            deltas = {key: abs(float(metrics[key]) - float(expected[key])) for key in metric_fields}
            expected_coverage = expected.get("exit_decision_coverage")
            row = {
                "model_id": model["model_id"], "fold_id": fold_id,
                "expected_trade_count": int(expected["trade_count"]),
                "actual_trade_count": int(metrics["trade_count"]),
                "max_metric_abs_delta": max(deltas.values()), "metric_deltas": deltas,
                "expected_exit_decision_coverage": None if expected_coverage is None else float(expected_coverage),
                "actual_exit_decision_coverage": float(metrics.get("exit_decision_coverage", 1.0)),
                "expected_metrics": {key: expected.get(key) for key in metric_fields},
                "actual_metrics": {key: metrics.get(key) for key in metric_fields},
                "policy": asdict(policy), "threshold": float(expected["threshold"]),
            }
            source_rows.append(row)
            comparisons.append(row)
        cache[cache_key] = source_rows
    mismatches = []
    if legacy_historical_mode:
        mismatches = [
            row for row in comparisons
            if row["expected_trade_count"] != row["actual_trade_count"]
            or row["max_metric_abs_delta"] > 1e-8
            or (
                row["expected_exit_decision_coverage"] is not None
                and abs(row["actual_exit_decision_coverage"] - row["expected_exit_decision_coverage"]) > 1e-12
            )
        ]
    return {
        "status": "PASSED" if not mismatches else "FAILED",
        "replay_contract": "LEGACY_EXACT_OUTER_ROW_V1" if legacy_historical_mode else "CURRENT_NORMALIZED_HISTORICAL_V1",
        "legacy_results_reproduction_evidence_only": legacy_historical_mode,
        "outer_replays": len(comparisons),
        "unique_cell_outer_replays": sum(len(rows) for rows in cache.values()),
        "trade_count_mismatches": sum(row["expected_trade_count"] != row["actual_trade_count"] for row in comparisons),
        "metric_mismatches": sum(row["max_metric_abs_delta"] > 1e-8 for row in comparisons),
        "coverage_mismatches": sum(
            row["expected_exit_decision_coverage"] is not None
            and abs(row["actual_exit_decision_coverage"] - row["expected_exit_decision_coverage"]) > 1e-12
            for row in comparisons
        ),
        "max_metric_abs_delta": max((row["max_metric_abs_delta"] for row in comparisons), default=0.0),
        "rows": comparisons,
        "mismatches": mismatches,
        "security_id_delisting_contract_used": not legacy_historical_mode,
    }


def _reproduce_historical_trades(
    models: list[dict], predictions: pd.DataFrame, prices: pd.DataFrame, exit_path: Path,
    prepared_price_data: tuple | None = None,
) -> dict:
    return _historical_outer_replays(
        models, predictions, prices, exit_path, prepared_price_data,
        legacy_historical_mode=True,
    )


def run(args: argparse.Namespace) -> dict:
    output = Path(args.output_root)
    output.mkdir(parents=True, exist_ok=True)
    historical = _historical_gate(Path(args.historical_reproduction_summary))
    entry = _entry_frame(Path(args.causal_entry_predictions))
    exit_provider = LearnedExitProvider(Path(args.causal_exit_predictions))
    if exit_provider.audit.min_date != str(LIVE_START.date()) or exit_provider.audit.max_date != str(LIVE_END.date()):
        raise RuntimeError(f"CAUSAL_EXIT_DATE_COVERAGE_MISMATCH:{asdict(exit_provider.audit)}")
    configure_replay(exit_provider, ProfitTaxConfig())

    manifest = json.loads(Path(args.frozen_policy_manifest).read_text(encoding="utf-8"))
    models = manifest.get("models", [])
    if len(models) != 10:
        raise RuntimeError(f"FROZEN_TOP10_COUNT_MISMATCH:{len(models)}")
    historical_predictions, historical_prediction_audit = load_predictions(Path(args.historical_entry_predictions))
    tickers = set(entry["ticker"].astype(str)) | set(historical_predictions["ticker"].astype(str))
    prices, price_audit = load_price_panel(Path(args.daily_store_root), tickers)
    if pd.Timestamp(prices["date"].max()).normalize() < LIVE_END:
        raise RuntimeError("DAILY_PRICES_END_BEFORE_LIVE_END")
    prepared_price_data = prepare_prices(prices)

    legacy_reproduction = _reproduce_historical_trades(
        models, historical_predictions, prices, Path(args.historical_exit_predictions), prepared_price_data
    )
    if legacy_reproduction["status"] != "PASSED":
        first = legacy_reproduction["mismatches"][0]
        raise RuntimeError(
            "LEGACY_HISTORICAL_REPRODUCTION_FAILED:"
            f"model={first['model_id']}:fold={first['fold_id']}:"
            f"expected_trades={first['expected_trade_count']}:actual_trades={first['actual_trade_count']}:"
            f"expected_coverage={first['expected_exit_decision_coverage']}:"
            f"actual_coverage={first['actual_exit_decision_coverage']}:"
            f"metric_delta={first['max_metric_abs_delta']}"
        )
    current_historical = _historical_outer_replays(
        models, historical_predictions, prices, Path(args.historical_exit_predictions), prepared_price_data,
        legacy_historical_mode=False,
    )
    historical["legacy_exact_reproduction"] = legacy_reproduction
    historical["current_contract_historical_baseline"] = current_historical
    historical["prediction_audit"] = historical_prediction_audit
    configure_replay(exit_provider, ProfitTaxConfig())

    rows = []
    all_trades = []
    curves = []
    for index, model in enumerate(models, 1):
        policy = _policy(model["frozen_entry_policy"])
        selected = entry.loc[entry["horizon"].eq(policy.horizon)].copy()
        threshold_by_date = _thresholds(selected, policy.score_quantile)
        signals = selected[["decision_date", "ticker", "score"]]
        result = replay(
            signals, prices, policy, CostModel(20), TaxConfig(False),
            start=LIVE_START, end=LIVE_END, initial=float(args.initial_capital),
            resolved_threshold_by_date=threshold_by_date,
            prepared_prices=prepared_price_data,
        )
        metrics = result["metrics"]
        if metrics.get("learned_exit_missing_prediction_count", 0):
            raise RuntimeError(f"LEARNED_EXIT_MISSING:{model['model_id']}:{metrics['learned_exit_missing_prediction_count']}")
        if float(metrics.get("exit_decision_coverage", 0.0)) != 1.0:
            raise RuntimeError(f"LEARNED_EXIT_COVERAGE_INCOMPLETE:{model['model_id']}")
        row = {
            "rank": index, "model_id": model["model_id"], "h": policy.horizon,
            "d": policy.holding_days, "n": policy.max_names, "mode": policy.exit_family,
            "score_quantile": policy.score_quantile, "top_fraction": policy.top_fraction,
            "known_median_active_cagr_excess": model.get("median_active_cagr_excess"),
            "known_outer_folds": model.get("outer_folds"), "known_active_folds": model.get("active_folds"),
            "expanding_live_trade_count": int(metrics["trade_count"]),
            **{f"expanding_live_{key}": value for key, value in metrics.items()},
        }
        rows.append(row)
        for trade in result["trades"]:
            all_trades.append({"model_id": model["model_id"], "segment": "CAUSAL_EXPANDING_LIVE", **trade})
        curve = result["curve"].copy()
        curve.insert(0, "model_id", model["model_id"])
        curves.append(curve)
        print(f"CAUSAL_EXPANDING_PORTFOLIO_COMPLETE {index}/10 {model['model_id']} trades={metrics['trade_count']}", flush=True)

    ranking = pd.DataFrame(rows).sort_values(
        ["expanding_live_cagr_excess", "expanding_live_terminal_wealth_excess_eur"], ascending=False
    )
    ranking.to_csv(output / "causal_expanding_live_ranking.csv", index=False)
    pd.DataFrame(all_trades).to_csv(output / "causal_expanding_live_trades.csv", index=False)
    pd.concat(curves, ignore_index=True).to_parquet(output / "causal_expanding_live_curves.parquet", index=False)
    summary = {
        "contract_id": CONTRACT_ID,
        "status": "DUAL_VALIDATION_COMPLETE",
        "historical_reproduction": historical,
        "causal_expanding_live": {
            "status": "COMPLETE", "start": str(LIVE_START.date()), "end": str(LIVE_END.date()),
            "models": len(ranking), "total_trades": int(ranking["expanding_live_trade_count"].sum()),
            "positive_cagr_excess_models": int(ranking["expanding_live_cagr_excess"].gt(0).sum()),
            "best_model": ranking.iloc[0].to_dict(),
            "exit_provider_audit": asdict(exit_provider.audit),
        },
        "shared_contract": {
            "portfolio_engine": "learned_exit_qbd_replay.replay",
            "execution": "DECISION_CLOSE_ENTRY_NEXT_OPEN_EXIT_NEXT_OPEN",
            "roundtrip_cost_bps": 20,
            "tax": asdict(ProfitTaxConfig()),
            "allocation": "EQUAL_ACTIVE", "replacement": "IGNORE_NEW", "sleeve": 0.5,
            "security_series": {
                "CHK_PRE_2021": {"ticker": "CHK", "valid_through": "2020-06-26", "series_break": "2020-06-29"},
                "CHK_POST_2021_EXE": {"new_security_from": "2021-02-10", "ticker_change_to_EXE": "2024-10-02"},
                "synthetic_alias_across_bankruptcy": False, "otc_CHKAQ_implicit_alias": False,
            },
            "final_holdout_opened": False,
            "legacy_replay_never_used_for_causal_expanding_live": True,
        },
        "price_audit": price_audit,
    }
    _write_json(output / "dual_validation_summary.json", summary)
    report = [
        "# Top-10 Historical Reproduction and Causal Expanding Live",
        "", f"Status: **{summary['status']}**", "",
        "Legacy exact reproduction passed before the expanding-live results were interpreted.",
        "Current-contract historical replay is the comparison baseline; legacy metrics are reproduction evidence only.", "",
        f"Expanding-live window: {LIVE_START.date()} through {LIVE_END.date()}.",
        f"Total expanding-live trades: {summary['causal_expanding_live']['total_trades']}.",
        f"Models with positive CAGR excess: {summary['causal_expanding_live']['positive_cagr_excess_models']}/10.", "",
        _markdown_table(ranking[["model_id", "expanding_live_cagr_excess", "expanding_live_terminal_wealth_excess_eur", "expanding_live_trade_count"]]),
        "", "Final holdout remained closed.",
    ]
    (output / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--historical-reproduction-summary")
    parser.add_argument("--historical-entry-predictions")
    parser.add_argument("--historical-exit-predictions")
    parser.add_argument("--frozen-policy-manifest")
    parser.add_argument("--causal-entry-predictions")
    parser.add_argument("--causal-exit-predictions")
    parser.add_argument("--daily-store-root")
    parser.add_argument("--output-root", default="artifacts/top10-causal-expanding-portfolio")
    parser.add_argument("--initial-capital", type=float, default=10000.0)
    parser.add_argument("--historical-only", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--finalize-existing", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        from .learned_exit_qbd_replay import configure_replay
        class _Provider:
            def prediction(self, ticker, decision_date, remaining_sessions):
                return -1.0
        dates = pd.bdate_range("2024-01-02", periods=6)
        prices = pd.concat([
            pd.DataFrame({"date": dates, "ticker": ticker, "open": 100.0, "close": 100.0})
            for ticker in ("URTH", "AAA")
        ], ignore_index=True)
        signals = pd.DataFrame({"decision_date": [dates[-3]], "ticker": ["AAA"], "score": [1.0]})
        policy = Policy(28, .9, 1.0, 1, 4, "LEARNED_EXIT")
        configure_replay(_Provider(), ProfitTaxConfig(enabled=False))
        legacy = replay(signals, prices, policy, CostModel(0), TaxConfig(False),
                        resolved_threshold=0.0, legacy_historical_mode=True)
        current = replay(signals, prices, policy, CostModel(0), TaxConfig(False),
                         resolved_threshold=0.0, legacy_historical_mode=False)
        assert legacy["metrics"]["learned_exit_fallback_positions"] == 1
        assert current["metrics"]["learned_exit_fallback_positions"] == 0
        assert legacy["metrics"]["learned_exit_horizon_boundary_count"] == 0
        assert current["metrics"]["learned_exit_horizon_boundary_count"] > 0
        assert not legacy["trades"]
        assert current["trades"]
        assert current["trades"][0]["exit_reason"] == "LEARNED_EXIT"
        print("TOP10_DUAL_LEGACY_CURRENT_REPLAY_SELF_TEST_OK")
        return
    required = ("historical_reproduction_summary", "historical_entry_predictions",
                "historical_exit_predictions", "frozen_policy_manifest", "daily_store_root")
    missing = [name for name in required if not getattr(args, name)]
    if missing:
        parser.error("required arguments missing: " + ", ".join("--" + name.replace("_", "-") for name in missing))
    if args.finalize_existing:
        output = Path(args.output_root)
        ranking_path = output / "causal_expanding_live_ranking.csv"
        legacy_path = output / "historical_legacy_reproduction.json"
        current_path = output / "historical_current_contract_replay.json"
        if not ranking_path.is_file() or not legacy_path.is_file() or not current_path.is_file():
            raise RuntimeError("FINALIZE_EXISTING_REQUIRED_OUTPUT_MISSING")
        ranking = pd.read_csv(ranking_path)
        legacy_payload = json.loads(legacy_path.read_text(encoding="utf-8"))
        current_payload = json.loads(current_path.read_text(encoding="utf-8"))
        if legacy_payload["trade_and_portfolio_reproduction"]["status"] != "PASSED":
            raise RuntimeError("FINALIZE_EXISTING_LEGACY_GATE_FAILED")
        summary = {
            "contract_id": CONTRACT_ID, "status": "DUAL_VALIDATION_COMPLETE",
            "historical_reproduction": {
                **legacy_payload["gate"],
                "legacy_exact_reproduction": legacy_payload["trade_and_portfolio_reproduction"],
                "current_contract_historical_baseline": current_payload,
            },
            "causal_expanding_live": {
                "status": "COMPLETE", "start": str(LIVE_START.date()), "end": str(LIVE_END.date()),
                "models": len(ranking), "total_trades": int(ranking["expanding_live_trade_count"].sum()),
                "positive_cagr_excess_models": int(ranking["expanding_live_cagr_excess"].gt(0).sum()),
                "best_model": ranking.iloc[0].to_dict(),
                "from_completed_replay_artifacts": True,
            },
            "shared_contract": {"portfolio_engine": "learned_exit_qbd_replay.replay", "legacy_replay_never_used_for_causal_expanding_live": True, "final_holdout_opened": False},
        }
        _write_json(output / "dual_validation_summary.json", summary)
        report = ["# Top-10 Historical Reproduction and Causal Expanding Live", "",
                  "Status: **DUAL_VALIDATION_COMPLETE**", "",
                  "Legacy exact reproduction passed. Current-contract history is the comparison baseline.", "",
                  _markdown_table(ranking[["model_id", "expanding_live_cagr_excess", "expanding_live_terminal_wealth_excess_eur", "expanding_live_trade_count"]]),
                  "", "Final holdout remained closed."]
        (output / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
        print(json.dumps(summary, indent=2, default=str))
        return
    if args.historical_only:
        gate = _historical_gate(Path(args.historical_reproduction_summary))
        manifest = json.loads(Path(args.frozen_policy_manifest).read_text(encoding="utf-8"))
        predictions, audit = load_predictions(Path(args.historical_entry_predictions))
        prices, price_audit = load_price_panel(Path(args.daily_store_root), set(predictions["ticker"]))
        result = _reproduce_historical_trades(
            manifest["models"], predictions, prices, Path(args.historical_exit_predictions)
        )
        if result["status"] != "PASSED":
            first = result["mismatches"][0]
            raise RuntimeError(f"LEGACY_HISTORICAL_REPRODUCTION_FAILED:{first}")
        current = _historical_outer_replays(
            manifest["models"], predictions, prices, Path(args.historical_exit_predictions),
            legacy_historical_mode=False,
        )
        payload = {"gate": gate, "trade_and_portfolio_reproduction": result,
                   "current_contract_historical_baseline": current,
                   "prediction_audit": audit, "price_audit": price_audit}
        _write_json(Path(args.output_root) / "historical_legacy_reproduction.json", payload)
        _write_json(Path(args.output_root) / "historical_current_contract_replay.json", current)
        print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))
        return
    if not args.causal_entry_predictions or not args.causal_exit_predictions:
        parser.error("causal prediction paths are required unless --historical-only is used")
    print(json.dumps(run(args), indent=2, default=str))


if __name__ == "__main__":
    main()
