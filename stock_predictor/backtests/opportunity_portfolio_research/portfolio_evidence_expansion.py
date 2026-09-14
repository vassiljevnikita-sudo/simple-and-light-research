from __future__ import annotations

from collections import defaultdict
from concurrent.futures import as_completed
from math import comb, sqrt
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from .portfolio_policy_contracts import policy_dict
from .cost_contracts import CostModel
from .tax_contracts import TaxConfig
from .portfolio_research_inputs import load_predictions, load_price_panel
from . import portfolio_replay_process_backend as backend
from . import portfolio_resilient_process_pool as resilient
from .affinity_coordinator_pool import AffinityCoordinatorPool


EVIDENCE_FILES = (
    "outer_replay_consistency.csv",
    "outer_oos_cost_tax_stress.csv",
    "outer_oos_stress_summary.csv",
    "outer_subwindow_results.csv",
    "outer_subwindow_summary.csv",
    "outer_activation_diagnostics.csv",
    "outer_statistical_evidence.csv",
    "outer_leave_one_fold_out.csv",
    "outer_oos_concentration.csv",
    "outer_oos_trade_log.csv",
    "outer_oos_open_positions.csv",
    "expanded_evidence_summary.json",
)


def _json(value):
    return json.dumps(value, sort_keys=True, default=str, separators=(",", ":"))


class EvidenceCheckpoint:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.completed: dict[str, dict] = {}
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("complete") and row.get("job_key"):
                    self.completed[str(row["job_key"])] = row.get("result", {})

    def get(self, key: str):
        return self.completed.get(str(key))

    def append(self, key: str, result: dict) -> None:
        record = {
            "job_key": str(key),
            "complete": True,
            "completed_at": time.time(),
            "result": result,
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(_json(record) + "\n")
            handle.flush()
        self.completed[str(key)] = result


def _policy_row(row: dict) -> dict:
    keys = (
        "horizon", "score_quantile", "top_fraction", "max_names", "holding_days",
        "exit_family", "exit_value", "replacement", "allocation", "sleeve",
    )
    value = {key: row[key] for key in keys if key in row}
    value["horizon"] = int(value["horizon"])
    value["max_names"] = int(value["max_names"])
    value["holding_days"] = int(value["holding_days"])
    for key in ("score_quantile", "top_fraction", "exit_value", "sleeve"):
        value[key] = float(value[key])
    return value


def _outer_rows(path: Path) -> list[dict]:
    frame = pd.read_csv(path)
    required = {"horizon", "fold_id", "fold_start", "fold_end", "threshold", "policy_id"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"OUTER_RESULTS_MISSING_COLUMNS:{sorted(missing)}")
    return frame.to_dict("records")


def _key(job: dict) -> str:
    stable = {key: job[key] for key in (
        "horizon", "fold_id", "policy_id", "threshold", "start", "end",
        "cost_bps", "tax_world", "diagnostic_type",
    )}
    return _json(stable)


def _base_job(row: dict, cost_bps: float, tax_world: str, start=None, end=None, diagnostic_type="outer") -> dict:
    job = {
        "horizon": int(row["horizon"]),
        "fold_id": str(row["fold_id"]),
        "policy_id": str(row["policy_id"]),
        "policy": _policy_row(row),
        "threshold": float(row["threshold"]),
        "start": str(start or row["fold_start"]),
        "end": str(end or row["fold_end"]),
        "cost_bps": float(cost_bps),
        "tax_world": str(tax_world),
        "tax_enabled": str(tax_world) == "DE_RETAIL_TAX_AWARE",
        "initial": 10000.0,
        "diagnostic_type": diagnostic_type,
    }
    job["job_key"] = _key(job)
    return job


def _write_csv(path: Path, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def _run_jobs(jobs: list[dict], label: str, horizon: int, checkpoint: EvidenceCheckpoint) -> list[dict]:
    if not jobs:
        return []
    cached = []
    misses = []
    for job in jobs:
        value = checkpoint.get(job["job_key"])
        needs_open_positions = label == "evidence_outer_baseline" and (
            value is None or "open_positions" not in value
        )
        if value is None or needs_open_positions:
            misses.append(job)
        else:
            cached.append(value)
    results = list(cached)
    if misses:
        pool = backend._ensure_pool(backend._CONTEXT_SIGNALS, backend._CONTEXT_PRICES, 12)
        indexed = list(enumerate(misses))

        def commit(_index: int, value: dict) -> None:
            clean = dict(value)
            checkpoint.append(str(clean["job_key"]), clean)
            results.append(clean)

        backend._run_process_futures(
            pool,
            indexed,
            backend._worker_evidence_replay,
            label,
            commit,
            jobs_reused=len(cached),
            horizon=int(horizon),
        )
    return results


def _metric(result: dict, name: str, default=0.0):
    return float((result.get("metrics") or {}).get(name, default) or default)


def _baseline_rows(rows: list[dict], results: dict[str, dict]) -> tuple[list[dict], list[dict]]:
    consistency = []
    outer = []
    for row in rows:
        result = results[_base_job(row, 20, "PRE_TAX")["job_key"]]
        metrics = result.get("metrics", {})
        checks = {
            "trade_count": int(metrics.get("trade_count", 0)) == int(row.get("trade_count", 0)),
            "terminal_value": abs(_metric(result, "terminal_value") - float(row.get("terminal_value", 0.0))) <= 1e-8,
            "urth_terminal_value": abs(_metric(result, "urth_terminal_value") - float(row.get("urth_terminal_value", 0.0))) <= 1e-8,
            "cagr_excess": abs(_metric(result, "cagr_excess") - float(row.get("cagr_excess", 0.0))) <= 1e-10,
            "total_return": abs(_metric(result, "total_return") - float(row.get("total_return", 0.0))) <= 1e-10,
        }
        consistency.append({
            "horizon": int(row["horizon"]), "fold_id": row["fold_id"],
            "expected_trade_count": int(row.get("trade_count", 0)),
            "actual_trade_count": int(metrics.get("trade_count", 0)),
            "expected_cagr_excess": float(row.get("cagr_excess", 0.0)),
            "actual_cagr_excess": _metric(result, "cagr_excess"),
            "delta": _metric(result, "cagr_excess") - float(row.get("cagr_excess", 0.0)),
            "consistent": bool(all(checks.values())),
            "check_trade_count": checks["trade_count"],
            "check_terminal_value": checks["terminal_value"],
            "check_urth_terminal_value": checks["urth_terminal_value"],
            "check_cagr_excess": checks["cagr_excess"],
            "check_total_return": checks["total_return"],
        })
        outer.append({**row, "evidence_result": result})
    return consistency, outer


def _stress_rows(rows: list[dict], results: dict[str, dict]) -> list[dict]:
    output = []
    for row in rows:
        for cost in (20, 30, 50):
            for tax in ("PRE_TAX", "DE_RETAIL_TAX_AWARE"):
                result = results[_base_job(row, cost, tax)["job_key"]]
                metrics = result.get("metrics", {})
                output.append({
                    "horizon": int(row["horizon"]), "fold_id": row["fold_id"],
                    "cost_bps": cost, "tax_world": tax,
                    "trade_count": int(metrics.get("trade_count", 0)),
                    "total_return": _metric(result, "total_return"),
                    "urth_total_return": _metric(result, "urth_total_return"),
                    "cagr": _metric(result, "cagr"), "urth_cagr": _metric(result, "urth_cagr"),
                    "cagr_excess": _metric(result, "cagr_excess"),
                    "terminal_wealth_excess_eur": _metric(result, "terminal_wealth_excess_eur"),
                    "tax_paid": _metric(result, "tax_paid"), "total_cost_eur": _metric(result, "total_cost_eur"),
                })
    return output


def _subwindows(row: dict, split_count: int, prices: pd.DataFrame) -> list[dict]:
    dates = sorted(pd.to_datetime(prices.loc[
        (prices["date"] >= pd.Timestamp(row["fold_start"]))
        & (prices["date"] <= pd.Timestamp(row["fold_end"])), "date"
    ]).dt.normalize().unique())
    chunks = np.array_split(dates, split_count)
    jobs = []
    for index, chunk in enumerate(chunks, 1):
        if len(chunk) < 3:
            continue
        jobs.append(_base_job(
            row, 20, "PRE_TAX", pd.Timestamp(chunk[0]), pd.Timestamp(chunk[-1]), f"subwindow_{split_count}"
        ) | {"split_count": split_count, "subwindow_index": index})
        jobs[-1]["job_key"] = _key(jobs[-1] | {"diagnostic_type": f"subwindow_{split_count}_{index}"})
    return jobs


def _subwindow_outputs(jobs: list[dict], results: dict[str, dict]) -> list[dict]:
    output = []
    for job in jobs:
        result = results[job["job_key"]]
        metrics = result.get("metrics", {})
        output.append({
            "parent_fold_id": job["fold_id"], "horizon": job["horizon"],
            "split_count": job["split_count"], "subwindow_index": job["subwindow_index"],
            "subwindow_start": job["start"], "subwindow_end": job["end"],
            "trade_count": int(metrics.get("trade_count", 0)),
            "cagr_excess": _metric(result, "cagr_excess"),
            "total_return_excess": _metric(result, "total_return_excess"),
            "dependent_subwindow": True, "independent_outer_fold": False, "diagnostic_only": True,
        })
    return output


def _activation(rows: list[dict], predictions: pd.DataFrame, baseline: dict[str, dict]) -> list[dict]:
    output = []
    for row in rows:
        sig = predictions.loc[
            (predictions["horizon"] == int(row["horizon"]))
            & (predictions["fold_id"].astype(str) == str(row["fold_id"]))
        ]
        above = sig.loc[sig["score"] >= float(row["threshold"])]
        trade_count = int((baseline[_base_job(row, 20, "PRE_TAX")["job_key"]].get("metrics") or {}).get("trade_count", 0))
        reason = "ACTIVE_FOLD"
        if trade_count == 0:
            if above.empty:
                reason = "NO_SCORE_ABOVE_THRESHOLD"
            else:
                next_open = predictions  # actual price availability is checked in run() below
                reason = "OTHER"
        output.append({
            "horizon": int(row["horizon"]), "fold_id": row["fold_id"],
            "prediction_days": int(sig["decision_date"].nunique()),
            "days_with_score_above_threshold": int(above["decision_date"].nunique()),
            "activation_day_fraction": float(above["decision_date"].nunique() / max(1, sig["decision_date"].nunique())),
            "candidate_tickers_above_threshold": int(above["ticker"].nunique()),
            "actual_entry_count": trade_count, "roundtrip_count": trade_count,
            "inactive_reason": reason,
        })
    return output


def _wilson(successes: int, total: int) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 0.0
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = z * sqrt((p * (1 - p) + z * z / (4 * total)) / total) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def _sign_pvalue(successes: int, total: int) -> float:
    if total <= 0:
        return 1.0
    return sum(comb(total, k) for k in range(successes, total + 1)) / (2 ** total)


def _statistical(rows: list[dict], baseline: dict[str, dict]) -> list[dict]:
    by_h = defaultdict(list)
    for row in rows:
        result = baseline[_base_job(row, 20, "PRE_TAX")["job_key"]]
        metrics = result.get("metrics", {})
        if int(metrics.get("trade_count", 0)) > 0:
            by_h[int(row["horizon"])].append(_metric(result, "cagr_excess"))
    output = []
    for horizon, values in sorted(by_h.items()):
        positive = sum(value > 0 for value in values)
        low, high = _wilson(positive, len(values))
        output.append({
            "horizon": horizon, "total_outer_folds": sum(int(r["horizon"]) == horizon for r in rows),
            "active_outer_folds": len(values), "positive_active_outer_folds": positive,
            "positive_active_fraction": positive / len(values) if values else 0.0,
            "sign_test_pvalue": _sign_pvalue(positive, len(values)),
            "wilson_low_95": low, "wilson_high_95": high,
            "independent_unit": "OUTER_FOLD", "trade_count_not_independent_sample": True,
        })
    return output


def _leave_one_out(rows: list[dict], baseline: dict[str, dict]) -> list[dict]:
    output = []
    for horizon in sorted({int(row["horizon"]) for row in rows}):
        hrows = [row for row in rows if int(row["horizon"]) == horizon]
        for removed in hrows:
            kept = [row for row in hrows if row["fold_id"] != removed["fold_id"]]
            active = []
            strategy_factor = urth_factor = 1.0
            for row in kept:
                result = baseline[_base_job(row, 20, "PRE_TAX")["job_key"]]
                metrics = result.get("metrics", {})
                strategy_factor *= 1 + _metric(result, "total_return")
                urth_factor *= 1 + _metric(result, "urth_total_return")
                if int(metrics.get("trade_count", 0)) > 0:
                    active.append(_metric(result, "cagr_excess"))
            years = max(1.0, sum(int((pd.Timestamp(r["fold_end"]) - pd.Timestamp(r["fold_start"])).days) for r in kept) / 365.25)
            output.append({
                "horizon": horizon, "removed_fold_id": removed["fold_id"],
                "active_fold_count": len(active),
                "positive_active_fraction": sum(x > 0 for x in active) / len(active) if active else 0.0,
                "median_active_cagr_excess": float(np.median(active)) if active else 0.0,
                "q25_active_cagr_excess": float(np.quantile(active, .25)) if active else 0.0,
                "worst_active_cagr_excess": float(min(active)) if active else 0.0,
                "chained_oos_cagr_excess": float(strategy_factor ** (1 / years) - urth_factor ** (1 / years)),
            })
    return output


def _concentration(outer: list[dict]) -> tuple[list[dict], list[dict]]:
    trades = []
    for row in outer:
        result = row["evidence_result"]
        for index, trade in enumerate(result.get("trades", []), 1):
            trade = dict(trade)
            pnl = float(trade.get("excess_return", 0.0)) * float(trade.get("buy_notional", 0.0))
            trades.append({
                "horizon": int(row["horizon"]), "fold_id": row["fold_id"], "trade_index": index,
                "ticker": trade.get("ticker"), "entry_date": trade.get("entry_date"),
                "exit_date": trade.get("exit_date"), "excess_pnl_eur": pnl,
                "excess_return": float(trade.get("excess_return", 0.0)),
                "buy_notional": float(trade.get("buy_notional", 0.0)),
            })
    output = []
    for horizon in sorted({int(x["horizon"]) for x in trades}):
        values = [x for x in trades if int(x["horizon"]) == horizon]
        positive = sorted([max(0.0, x["excess_pnl_eur"]) for x in values], reverse=True)
        ticker_counts = pd.Series([x["ticker"] for x in values]).value_counts() if values else pd.Series(dtype=int)
        fold_positive = defaultdict(float)
        for value in values:
            fold_positive[value["fold_id"]] += max(0.0, value["excess_pnl_eur"])
        total_positive = sum(positive)
        output.append({
            "horizon": horizon, "oos_trade_count": len(values), "unique_tickers": len(ticker_counts),
            "top_ticker_trade_share": float(ticker_counts.iloc[0] / len(values)) if values else 0.0,
            "top3_ticker_trade_share": float(ticker_counts.iloc[:3].sum() / len(values)) if values else 0.0,
            "top1_positive_pnl_share": positive[0] / total_positive if positive and total_positive else 0.0,
            "top5_positive_pnl_share": sum(positive[:5]) / total_positive if total_positive else 0.0,
            "top10_positive_pnl_share": sum(positive[:10]) / total_positive if total_positive else 0.0,
            "largest_fold_positive_pnl_share": max(fold_positive.values(), default=0.0) / total_positive if total_positive else 0.0,
            "trade_count_by_fold": _json(dict(pd.Series([x["fold_id"] for x in values]).value_counts())) if values else "{}",
            "positive_pnl_by_fold": _json(dict(fold_positive)),
        })
    return trades, output


def _open_position_rows(outer: list[dict]) -> list[dict]:
    rows = []
    for row in outer:
        result = row["evidence_result"]
        for position in result.get("open_positions", []):
            rows.append({
                "horizon": int(row["horizon"]), "fold_id": str(row["fold_id"]),
                "ticker": str(position["ticker"]), "entry_date": position["entry_date"],
                "mark_date": row["fold_end"], "entry_price": float(position["entry_price"]),
                "qty": float(position["qty"]), "buy_notional": float(position["buy_notional"]),
                "buy_fee": float(position["buy_fee"]), "tax_basis": float(position["tax_basis"]),
            })
    return rows


def run_evidence(predictions_path: Path, daily_store_root: Path, output_root: Path, max_workers: int = 12, coordinator_threads: int = 4) -> dict:
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    rows = _outer_rows(output_root / "outer_fold_results.csv")
    predictions, prediction_audit = load_predictions(Path(predictions_path))
    prices, price_audit = load_price_panel(Path(daily_store_root), set(predictions["ticker"].unique()))
    checkpoint = EvidenceCheckpoint(output_root / "expanded_oos_replay_checkpoint.jsonl")
    telemetry_path = output_root / "multicore_telemetry.jsonl"
    import os
    os.environ["OPPORTUNITY_TELEMETRY_PATH"] = str(telemetry_path)
    backend.install_multicore_backend(max_workers)
    resilient.install_resilient_process_pool()
    backend._set_context(predictions, prices)
    all_results: dict[str, dict] = {}
    try:
        with AffinityCoordinatorPool(max_workers=coordinator_threads, thread_name_prefix="evidence-coordinator") as coordinators:
            # Preparation is intentionally coordinated in four light threads; replay remains process-bound.
            prepared = list(coordinators.map(lambda x: x, rows))
        for horizon in sorted({int(row["horizon"]) for row in prepared}):
            hrows = [row for row in prepared if int(row["horizon"]) == horizon]
            jobs = [_base_job(row, 20, "PRE_TAX") for row in hrows]
            all_results.update({job["job_key"]: checkpoint.get(job["job_key"]) for job in jobs if checkpoint.get(job["job_key"]) is not None})
            for result in _run_jobs(jobs, "evidence_outer_baseline", horizon, checkpoint):
                all_results[result["job_key"]] = result

            stress_jobs = [_base_job(row, cost, tax) for row in hrows for cost in (20, 30, 50) for tax in ("PRE_TAX", "DE_RETAIL_TAX_AWARE")]
            for result in _run_jobs(stress_jobs, "evidence_outer_cost_tax_stress", horizon, checkpoint):
                all_results[result["job_key"]] = result

            for split in (2, 3):
                sub_jobs = [job for row in hrows for job in _subwindows(row, split, prices)]
                for result in _run_jobs(sub_jobs, f"evidence_subwindow_{split}", horizon, checkpoint):
                    all_results[result["job_key"]] = result
    finally:
        resilient.restore_process_pool_runner()
        backend.shutdown_multicore_backend()

    consistency, outer = _baseline_rows(rows, all_results)
    stress = _stress_rows(rows, all_results)
    sub_jobs_all = [job for row in rows for split in (2, 3) for job in _subwindows(row, split, prices)]
    sub_results = _subwindow_outputs(sub_jobs_all, all_results)
    activation = _activation(rows, predictions, all_results)
    stats = _statistical(rows, all_results)
    loo = _leave_one_out(rows, all_results)
    trades, concentration = _concentration(outer)
    open_positions = _open_position_rows(outer)

    stress_summary = []
    if stress:
        frame = pd.DataFrame(stress)
        for (horizon, cost, tax), group in frame.groupby(["horizon", "cost_bps", "tax_world"]):
            stress_summary.append({
                "horizon": int(horizon), "cost_bps": int(cost), "tax_world": tax,
                "outer_folds": int(len(group)), "trade_count": int(group["trade_count"].sum()),
                "mean_cagr_excess": float(group["cagr_excess"].mean()),
                "median_cagr_excess": float(group["cagr_excess"].median()),
                "positive_fold_fraction": float((group["cagr_excess"] > 0).mean()),
            })
    sub_summary = []
    if sub_results:
        frame = pd.DataFrame(sub_results)
        for (horizon, split), group in frame.groupby(["horizon", "split_count"]):
            sub_summary.append({"horizon": int(horizon), "split_count": int(split), "rows": int(len(group)), "trade_count": int(group["trade_count"].sum()), "mean_cagr_excess": float(group["cagr_excess"].mean()), "diagnostic_only": True, "independent_outer_fold": False})

    _write_csv(output_root / "outer_replay_consistency.csv", consistency)
    _write_csv(output_root / "outer_oos_cost_tax_stress.csv", stress)
    _write_csv(output_root / "outer_oos_stress_summary.csv", stress_summary)
    _write_csv(output_root / "outer_subwindow_results.csv", sub_results)
    _write_csv(output_root / "outer_subwindow_summary.csv", sub_summary)
    _write_csv(output_root / "outer_activation_diagnostics.csv", activation)
    _write_csv(output_root / "outer_statistical_evidence.csv", stats)
    _write_csv(output_root / "outer_leave_one_fold_out.csv", loo)
    _write_csv(output_root / "outer_oos_trade_log.csv", trades)
    _write_csv(output_root / "outer_oos_open_positions.csv", open_positions)
    _write_csv(output_root / "outer_oos_concentration.csv", concentration)

    summary = {
        "status": "EXPANDED_OOS_EVIDENCE_COMPLETE" if all(x["consistent"] for x in consistency) else "FAIL_CLOSED_BASELINE_INCONSISTENCY",
        "baseline_consistency_pass": bool(consistency and all(x["consistent"] for x in consistency)),
        "search_py_changed": False, "policy_selection_changed": False, "threshold_selection_changed": False,
        "final_holdout": "CLOSED", "independent_unit": "OUTER_FOLD",
        "subwindows_dependent": True, "subwindows_independent_outer_fold": False,
        "prediction_audit": prediction_audit, "price_audit": price_audit,
        "checkpoint_path": str(checkpoint.path), "outer_folds": len(rows),
        "evidence_files": list(EVIDENCE_FILES),
        "horizons": sorted({int(row["horizon"]) for row in rows}),
    }
    (output_root / "expanded_evidence_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return summary


# Compatibility name used by the process entrypoint.  Keeping this alias avoids
# changing the research runner while allowing the standalone evidence CLI and
# the integrated post-search stage to share the same implementation.
run_evidence_expansion = run_evidence
