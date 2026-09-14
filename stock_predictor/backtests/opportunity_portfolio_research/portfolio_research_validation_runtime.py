from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
from statistics import median

import numpy as np
import pandas as pd

from . import portfolio_policy_walk_forward as multicore_walk_forward
from . import portfolio_research_cli as runner

_ORIGINAL_OUTER_SUMMARY = runner._outer_fold_summary
_ORIGINAL_READINESS = runner._readiness_gates
_ORIGINAL_PLATEAU = runner._parameter_plateau
_ORIGINAL_CONCENTRATION = runner._concentration_diagnostics
_ORIGINAL_STORE_GET = None
_ORIGINAL_STORE_PUT = None
_INSTALLED = False

MIN_ACTIVE_OUTER_FOLDS = 4
MIN_ACTIVE_FOLD_FRACTION = 0.50
MIN_POSITIVE_ACTIVE_FOLD_FRACTION = 0.60
MIN_OUTER_TRADES = 20
MAX_MEDIAN_NEIGHBOR_DELTA_CAGR = 0.05
MAX_SINGLE_NEIGHBOR_DELTA_CAGR = 0.15
MAX_TOP5_POSITIVE_PNL_SHARE = 0.80

_EXECUTION_FAILURE_PATTERNS = (
    "brokenprocesspool",
    "broken process pool",
    "process pool is not usable",
    "child process terminated abruptly",
    "a child process terminated abruptly",
    "execution_failure:",
    "pool_recovery_context_missing",
    "pool_recovery_exhausted",
)


def is_execution_failure_status(value) -> bool:
    text = str(value or "").strip().lower()
    return bool(text) and any(pattern in text for pattern in _EXECUTION_FAILURE_PATTERNS)


def _payload_has_execution_failure(payload) -> bool:
    if not isinstance(payload, dict):
        return False
    if not payload.get("final_policy"):
        return True
    for row in payload.get("history", []) or []:
        if isinstance(row, dict) and is_execution_failure_status(row.get("status")):
            return True
    for row in payload.get("search_coverage", []) or []:
        if isinstance(row, dict) and is_execution_failure_status(row.get("status")):
            return True
    return False


def _guard_fragment_store() -> None:
    """Never let a partial/failed horizon masquerade as a complete resume checkpoint."""
    global _ORIGINAL_STORE_GET, _ORIGINAL_STORE_PUT
    cls = multicore_walk_forward._SanitizingFragmentStore
    if _ORIGINAL_STORE_GET is None:
        _ORIGINAL_STORE_GET = cls.get
    if _ORIGINAL_STORE_PUT is None:
        _ORIGINAL_STORE_PUT = cls.put

    def guarded_get(self, kind: str, cache_key: str):
        found, value = _ORIGINAL_STORE_GET(self, kind, cache_key)
        if found and kind == "horizon_checkpoint" and _payload_has_execution_failure(value):
            print(
                "[portfolio] REJECT incomplete horizon checkpoint; resuming from window/replay fragments",
                flush=True,
            )
            return False, None
        return found, value

    def guarded_put(self, kind: str, cache_key: str, value) -> None:
        if kind == "horizon_checkpoint" and _payload_has_execution_failure(value):
            print(
                "[portfolio] SKIP incomplete horizon checkpoint after execution failure",
                flush=True,
            )
            return
        _ORIGINAL_STORE_PUT(self, kind, cache_key, value)

    cls.get = guarded_get
    cls.put = guarded_put


def _growth_factor(row: dict, strategy: bool) -> float:
    if strategy:
        if row.get("total_return") is not None:
            return max(0.0, 1.0 + float(row.get("total_return", 0.0)))
        initial = float(row.get("initial_value", 0.0))
        terminal = float(row.get("terminal_value", initial))
    else:
        if row.get("urth_total_return") is not None:
            return max(0.0, 1.0 + float(row.get("urth_total_return", 0.0)))
        initial = float(row.get("initial_value", 0.0))
        terminal = float(row.get("urth_terminal_value", initial))
    return terminal / initial if initial > 0 else 1.0


def sparse_outer_fold_summary(rows: list[dict]) -> list[dict]:
    """Treat zero-trade folds as inactive evidence, not negative evidence."""
    result = []
    by_h = defaultdict(list)
    for row in rows:
        by_h[int(row["horizon"])].append(row)

    for horizon, values in sorted(by_h.items()):
        values = sorted(values, key=lambda x: (str(x.get("fold_start", "")), str(x.get("fold_id", ""))))
        all_excess = [float(x.get("cagr_excess", 0.0)) for x in values]
        active = [x for x in values if int(x.get("trade_count", 0)) > 0]
        active_excess = [float(x.get("cagr_excess", 0.0)) for x in active]
        strategy_factor = 1.0
        urth_factor = 1.0
        for row in values:
            strategy_factor *= _growth_factor(row, True)
            urth_factor *= _growth_factor(row, False)

        calendar_days = 0
        if values and values[0].get("fold_start") and values[-1].get("fold_end"):
            start = pd.Timestamp(values[0]["fold_start"])
            end = pd.Timestamp(values[-1]["fold_end"])
            calendar_days = max(1, int((end - start).days))
        years = max(calendar_days / 365.25, 1.0 / 365.25)
        strategy_cagr = strategy_factor ** (1.0 / years) - 1.0 if strategy_factor > 0 else -1.0
        urth_cagr = urth_factor ** (1.0 / years) - 1.0 if urth_factor > 0 else -1.0

        result.append({
            "horizon": int(horizon),
            "folds": len(values),
            "active_folds": len(active),
            "inactive_folds": len(values) - len(active),
            "active_fold_fraction": len(active) / max(1, len(values)),
            "positive_folds": sum(x > 0 for x in all_excess),
            "positive_fold_fraction": sum(x > 0 for x in all_excess) / max(1, len(all_excess)),
            "positive_active_folds": sum(x > 0 for x in active_excess),
            "positive_active_fold_fraction": (
                sum(x > 0 for x in active_excess) / len(active_excess) if active_excess else 0.0
            ),
            "median_cagr_excess": float(median(all_excess)) if all_excess else 0.0,
            "q25_cagr_excess": float(np.quantile(all_excess, 0.25)) if all_excess else 0.0,
            "worst_cagr_excess": float(min(all_excess)) if all_excess else 0.0,
            "median_active_cagr_excess": float(median(active_excess)) if active_excess else 0.0,
            "q25_active_cagr_excess": float(np.quantile(active_excess, 0.25)) if active_excess else 0.0,
            "worst_active_cagr_excess": float(min(active_excess)) if active_excess else 0.0,
            "trade_count": int(sum(int(x.get("trade_count", 0)) for x in values)),
            "chained_oos_total_return": float(strategy_factor - 1.0),
            "chained_urth_total_return": float(urth_factor - 1.0),
            "chained_oos_total_return_excess": float(strategy_factor - urth_factor),
            "chained_oos_cagr": float(strategy_cagr),
            "chained_urth_cagr": float(urth_cagr),
            "chained_oos_cagr_excess": float(strategy_cagr - urth_cagr),
            "chained_oos_diagnostic_only": True,
        })
    return result


def strict_parameter_plateau(*args, **kwargs):
    rows, gates = _ORIGINAL_PLATEAU(*args, **kwargs)
    final_policies = args[2] if len(args) > 2 else kwargs.get("final_policies", {})
    baseline_metrics = args[4] if len(args) > 4 else kwargs.get("baseline_metrics", {})

    for horizon, gate in gates.items():
        hrows = [row for row in rows if int(row.get("horizon", -1)) == int(horizon)]
        abs_deltas = [abs(float(row.get("delta_cagr_excess", 0.0))) for row in hrows]
        median_abs_delta = float(median(abs_deltas)) if abs_deltas else float("inf")
        max_abs_delta = float(max(abs_deltas)) if abs_deltas else float("inf")
        sensitivity_pass = bool(
            abs_deltas
            and median_abs_delta <= MAX_MEDIAN_NEIGHBOR_DELTA_CAGR
            and max_abs_delta <= MAX_SINGLE_NEIGHBOR_DELTA_CAGR
        )

        center = final_policies.get(horizon)
        center_excess = float(hrows[0].get("center_cagr_excess", 0.0)) if hrows else 0.0
        center_trades = int(baseline_metrics.get(horizon, {}).get("trade_count", 0))
        behaviorally_active_dimensions = []
        behaviorally_inactive_dimensions = []
        for dimension in sorted({str(row.get("changed_dimension")) for row in hrows}):
            dim_rows = [row for row in hrows if str(row.get("changed_dimension")) == dimension]
            changed = any(
                abs(float(row.get("neighbor_cagr_excess", 0.0)) - center_excess) > 1e-10
                or int(row.get("trade_count", 0)) != center_trades
                for row in dim_rows
            )
            (behaviorally_active_dimensions if changed else behaviorally_inactive_dimensions).append(dimension)

        max_positions_observed = int(baseline_metrics.get(horizon, {}).get("max_positions", 0))
        max_names_exercised = bool(
            center is not None and max_positions_observed >= min(int(center.max_names), 2)
        )
        gate.update({
            "median_abs_neighbor_delta_cagr": median_abs_delta,
            "max_abs_neighbor_delta_cagr": max_abs_delta,
            "sensitivity_pass": sensitivity_pass,
            "behaviorally_active_dimensions": behaviorally_active_dimensions,
            "behaviorally_inactive_dimensions": behaviorally_inactive_dimensions,
            "max_positions_observed": max_positions_observed,
            "max_names_parameter_exercised": max_names_exercised,
            "parameter_identifiability_pass": bool(behaviorally_active_dimensions),
            "plateau_pass": bool(gate.get("plateau_pass") and sensitivity_pass),
        })
    return rows, gates


def refined_concentration_diagnostics(*args, **kwargs):
    rows, gates = _ORIGINAL_CONCENTRATION(*args, **kwargs)
    by_h = {int(row["horizon"]): row for row in rows}
    for horizon, gate in gates.items():
        row = by_h.get(int(horizon), gate)
        trades = int(row.get("trade_count", 0))
        top_ticker_share = float(row.get("top_ticker_trade_share", 0.0))
        top1_share = float(row.get("top1_positive_contribution_share", 0.0))
        top5_share = float(row.get("top5_positive_contribution_share", 0.0))
        exclude_ticker = float(row.get("exclude_top_ticker_cagr_excess", 0.0))
        exclude_one = float(row.get("exclude_best_1_trade_cagr_excess", 0.0))
        exclude_five = float(row.get("exclude_best_5_trade_cagr_excess", 0.0))

        ticker_dependence = bool(trades and top_ticker_share > 0.50 and exclude_ticker <= 0.0)
        single_winner_dependence = bool(trades and top1_share > 0.50 and exclude_one <= 0.0)
        winner_cluster_dependence = bool(
            trades >= 5
            and top5_share > MAX_TOP5_POSITIVE_PNL_SHARE
            and exclude_five <= 0.0
        )
        gate.update({
            "legacy_extreme_winner_or_ticker_dependence": bool(
                gate.get("extreme_winner_or_ticker_dependence", False)
            ),
            "ticker_dependence_confirmed_by_exclusion": ticker_dependence,
            "single_winner_dependence_confirmed_by_exclusion": single_winner_dependence,
            "top5_winner_cluster_dependence": winner_cluster_dependence,
            "extreme_winner_or_ticker_dependence": bool(
                ticker_dependence or single_winner_dependence or winner_cluster_dependence
            ),
            "concentration_pass": not (
                ticker_dependence or single_winner_dependence or winner_cluster_dependence
            ),
        })
        row.update(gate)
    return rows, gates


def _expected_horizons(search_meta: dict) -> list[int]:
    coverage = search_meta.get("search_coverage", {}) or {}
    horizons = sorted({int(h) for h in coverage.keys()})
    if horizons:
        return horizons
    return [5, 10, 20]


def sparse_readiness_gates(
    outer_summary: list[dict],
    portfolio_rows: list[dict],
    search_meta: dict,
    plateau_gates: dict,
    concentration_gates: dict,
    cross_regime_rows: list[dict],
) -> dict:
    base = _ORIGINAL_READINESS(
        outer_summary,
        portfolio_rows,
        search_meta,
        plateau_gates,
        concentration_gates,
        cross_regime_rows,
    )

    expected_horizons = _expected_horizons(search_meta)
    expected_outer_folds = max(0, len(search_meta.get("folds", []) or []) - 2)
    summary_by_h = {int(row["horizon"]): row for row in outer_summary}
    coverage_map = search_meta.get("search_coverage", {}) or {}
    final_policies = search_meta.get("final_policies", {}) or {}
    final_thresholds = search_meta.get("final_thresholds", {}) or {}

    horizon_execution = {}
    nominal_coverage_ok = True
    dynamic_stage_ok = True
    execution_complete = True
    evidence_coverage_ok = True
    outer_robust_ok = True

    for horizon in expected_horizons:
        entries = list(coverage_map.get(horizon, coverage_map.get(str(horizon), [])) or [])
        final_entries = [row for row in entries if isinstance(row, dict) and row.get("final_fit")]
        execution_errors = [
            str(row.get("status"))
            for row in entries
            if isinstance(row, dict) and is_execution_failure_status(row.get("status"))
        ]
        summary = summary_by_h.get(horizon, {})
        actual_outer = int(summary.get("folds", 0))
        final_policy_present = horizon in final_policies or str(horizon) in final_policies
        threshold_present = horizon in final_thresholds or str(horizon) in final_thresholds
        outer_complete = actual_outer >= expected_outer_folds if expected_outer_folds else actual_outer > 0
        h_execution_complete = bool(
            final_policy_present and threshold_present and outer_complete and not execution_errors
        )
        execution_complete = execution_complete and h_execution_complete

        h_nominal = bool(final_entries) and all(bool(row.get("coverage_complete")) for row in final_entries)
        h_dynamic = bool(final_entries) and all(bool(row.get("dynamic_stage_complete")) for row in final_entries)
        nominal_coverage_ok = nominal_coverage_ok and h_nominal
        dynamic_stage_ok = dynamic_stage_ok and h_dynamic

        active_folds = int(summary.get("active_folds", 0))
        active_fraction = float(summary.get("active_fold_fraction", 0.0))
        positive_active_fraction = float(summary.get("positive_active_fold_fraction", 0.0))
        median_active = float(summary.get("median_active_cagr_excess", 0.0))
        trade_count = int(summary.get("trade_count", 0))
        evidence_ok = bool(
            active_folds >= MIN_ACTIVE_OUTER_FOLDS
            and active_fraction >= MIN_ACTIVE_FOLD_FRACTION
            and trade_count >= MIN_OUTER_TRADES
        )
        robust_ok = bool(
            h_execution_complete
            and actual_outer >= max(5, expected_outer_folds)
            and evidence_ok
            and positive_active_fraction >= MIN_POSITIVE_ACTIVE_FOLD_FRACTION
            and median_active > 0.0
        )
        evidence_coverage_ok = evidence_coverage_ok and evidence_ok
        outer_robust_ok = outer_robust_ok and robust_ok

        horizon_execution[horizon] = {
            "expected_outer_folds": int(expected_outer_folds),
            "actual_outer_folds": actual_outer,
            "final_policy_present": bool(final_policy_present),
            "frozen_threshold_present": bool(threshold_present),
            "execution_failure_detected": bool(execution_errors),
            "execution_failure_messages": execution_errors,
            "execution_complete": h_execution_complete,
            "nominal_grid_coverage_complete": h_nominal,
            "dynamic_stage_complete": h_dynamic,
            "active_folds": active_folds,
            "inactive_folds": int(summary.get("inactive_folds", 0)),
            "active_fold_fraction": active_fraction,
            "positive_active_fold_fraction": positive_active_fraction,
            "oos_trade_count": trade_count,
            "evidence_coverage_complete": evidence_ok,
            "sparse_outer_robustness_pass": robust_ok,
        }

    regime_horizons = {int(row["horizon"]) for row in cross_regime_rows if row.get("horizon") is not None}
    regime_ok = bool(expected_horizons) and set(expected_horizons).issubset(regime_horizons)
    plateau_ok = bool(plateau_gates) and all(
        bool(gate.get("plateau_pass")) for horizon, gate in plateau_gates.items()
        if int(horizon) in expected_horizons
    ) and set(expected_horizons).issubset({int(h) for h in plateau_gates.keys()})
    concentration_ok = bool(concentration_gates) and all(
        bool(gate.get("concentration_pass")) for horizon, gate in concentration_gates.items()
        if int(horizon) in expected_horizons
    ) and set(expected_horizons).issubset({int(h) for h in concentration_gates.keys()})

    base.update({
        "development_execution_complete": bool(execution_complete),
        "horizon_execution_coverage": horizon_execution,
        "nominal_grid_coverage_complete": bool(nominal_coverage_ok),
        "search_coverage_complete": bool(execution_complete and nominal_coverage_ok),
        "dynamic_exit_stage_complete": bool(execution_complete and dynamic_stage_ok),
        "sparse_evidence_coverage_complete": bool(execution_complete and evidence_coverage_ok),
        "outer_fold_robustness_pass": bool(execution_complete and outer_robust_ok),
        "parameter_plateau_pass": bool(execution_complete and plateau_ok),
        "concentration_pass": bool(execution_complete and concentration_ok),
        "cross_regime_context_matrix_complete": bool(execution_complete and regime_ok),
        "inactive_outer_folds_are_negative_evidence": False,
        "minimum_active_outer_folds": MIN_ACTIVE_OUTER_FOLDS,
        "minimum_active_fold_fraction": MIN_ACTIVE_FOLD_FRACTION,
        "minimum_oos_roundtrips": MIN_OUTER_TRADES,
    })
    required = (
        "accounting_invariants_pass",
        "exposure_invariants_pass",
        "development_execution_complete",
        "search_coverage_complete",
        "dynamic_exit_stage_complete",
        "sparse_evidence_coverage_complete",
        "outer_fold_robustness_pass",
        "parameter_plateau_pass",
        "concentration_pass",
        "cost_tax_stress_pass",
        "cross_regime_context_matrix_complete",
    )
    base["all_readiness_gates_pass"] = all(bool(base.get(key)) for key in required)
    return base


def _horizon_research_status(summary: dict, execution: dict) -> str:
    if not execution.get("execution_complete"):
        if execution.get("execution_failure_detected") or not execution.get("final_policy_present"):
            return "NOT_TESTED_DUE_EXECUTION_FAILURE"
        return "INCOMPLETE_RESEARCH_PATH"
    if execution.get("sparse_outer_robustness_pass"):
        return "ROBUST_OOS_EVIDENCE"
    if (
        float(summary.get("chained_oos_cagr_excess", 0.0)) > 0.0
        or float(summary.get("median_active_cagr_excess", 0.0)) > 0.0
    ):
        return "INTERESTING_BUT_SPARSE_AND_UNSTABLE"
    return "NO_ROBUST_AFTER_COST_ALPHA"


def _repair_yearly_roundtrip_counts(output_root: Path) -> None:
    yearly_path = output_root / "yearly_results.csv"
    trade_path = output_root / "trade_log.csv"
    if not yearly_path.exists() or not trade_path.exists():
        return
    yearly = pd.read_csv(yearly_path)
    trades = pd.read_csv(trade_path)
    if yearly.empty:
        return
    if trades.empty:
        yearly["trades"] = 0
        yearly.to_csv(yearly_path, index=False)
        return
    date_col = "entry_date" if "entry_date" in trades.columns else "exit_date"
    trades[date_col] = pd.to_datetime(trades[date_col], errors="coerce")
    trades["year"] = trades[date_col].dt.year
    group_cols = [column for column in ("horizon", "tax_world", "year") if column in trades.columns]
    counts = trades.dropna(subset=["year"]).groupby(group_cols).size().to_dict()

    def count_row(row):
        key = tuple(row[column] for column in group_cols)
        return int(counts.get(key, 0))

    yearly["trades"] = yearly.apply(count_row, axis=1)
    yearly.to_csv(yearly_path, index=False)


def postprocess_research_artifacts(output_root: str | Path) -> None:
    output_root = Path(output_root)
    _repair_yearly_roundtrip_counts(output_root)
    summary_path = output_root / "summary.json"
    frozen_path = output_root / "frozen_portfolio_policy.json"
    if not summary_path.exists():
        return

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    gates = dict(summary.get("readiness_gates", {}))
    execution = gates.get("horizon_execution_coverage", {}) or {}
    outer_summary = summary.get("outer_fold_summary", []) or []
    summary_by_h = {int(row["horizon"]): row for row in outer_summary}
    horizon_status = {}
    for horizon_key, info in execution.items():
        horizon = int(horizon_key)
        horizon_status[str(horizon)] = _horizon_research_status(
            summary_by_h.get(horizon, {}), info
        )

    if not gates.get("development_execution_complete", False):
        status = "INCOMPLETE_DEVELOPMENT_SEARCH"
        decision = "INCOMPLETE_DEVELOPMENT_SEARCH"
    elif gates.get("all_readiness_gates_pass", False):
        status = "DEVELOPMENT_SUITE_COMPLETE"
        decision = "FROZEN_POLICY_READY_FOR_FINAL_HOLDOUT"
    elif any(value in ("ROBUST_OOS_EVIDENCE", "INTERESTING_BUT_SPARSE_AND_UNSTABLE") for value in horizon_status.values()):
        status = "DEVELOPMENT_SUITE_COMPLETE"
        decision = "MIXED_PORTFOLIO_EVIDENCE_NOT_READY_FOR_FINAL_HOLDOUT"
    else:
        status = "DEVELOPMENT_SUITE_COMPLETE"
        decision = "NO_ROBUST_AFTER_COST_ALPHA_NOT_READY_FOR_FINAL_HOLDOUT"

    summary.update({
        "status": status,
        "decision": decision,
        "ready_for_final_holdout": bool(gates.get("all_readiness_gates_pass", False)),
        "horizon_research_status": horizon_status,
        "yearly_trade_count_definition": "completed_roundtrips_by_entry_year",
        "coverage_contract": {
            "nominal_grid": "all configured parameter levels represented in final-fit search",
            "execution": "all expected H5/H10/H20 outer folds plus final fit completed without execution failure",
            "sparse_evidence": "inactive no-trade folds are neutral; require >=4 active folds, >=50% activity and >=20 OOS roundtrips",
        },
        "next_action": (
            "resume missing Development horizons from fragment cache; keep final holdout closed"
            if not gates.get("development_execution_complete", False)
            else "keep final holdout closed until every readiness gate passes"
        ),
    })
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )

    if frozen_path.exists():
        frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
        frozen.update({
            "status": status,
            "decision": decision,
            "ready_for_final_holdout": bool(gates.get("all_readiness_gates_pass", False)),
            "horizon_research_status": horizon_status,
        })
        frozen_path.write_text(
            json.dumps(frozen, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )

    report_path = output_root / "REPORT.md"
    if report_path.exists():
        lines = report_path.read_text(encoding="utf-8").splitlines()
        for index, line in enumerate(lines):
            if line.startswith("Decision:"):
                lines[index] = f"Decision: `{status}` / `{decision}`"
                break
        marker = "## Sparse-opportunity execution/evidence coverage"
        if marker not in lines:
            lines.extend([
                "",
                marker,
                "",
                "- BrokenProcessPool/child-process failures are execution failures, not negative alpha evidence.",
                "- Zero-trade outer folds are inactive evidence, not negative folds.",
                "- Readiness requires all expected horizons to finish, >=4 active folds and >=20 OOS roundtrips per horizon.",
                "- Chained OOS fold return is diagnostic only and is not a readiness gate.",
                "- Yearly trade counts are completed roundtrips by entry year.",
                "",
                json.dumps(horizon_status, indent=2, sort_keys=True),
            ])
        report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def install_research_runtime_v2() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    runner._outer_fold_summary = sparse_outer_fold_summary
    runner._readiness_gates = sparse_readiness_gates
    runner._parameter_plateau = strict_parameter_plateau
    runner._concentration_diagnostics = refined_concentration_diagnostics
    _guard_fragment_store()
    _INSTALLED = True
