from __future__ import annotations

import json
from pathlib import Path
import tempfile

import pandas as pd

from .portfolio_research_validation_runtime import (
    _payload_has_execution_failure,
    postprocess_research_artifacts,
    sparse_outer_fold_summary,
    sparse_readiness_gates,
)


def _h5_rows() -> list[dict]:
    excess = (-0.2706, 0.0275, 0.0, 0.2694, 0.0, 0.0)
    trades = (10, 8, 0, 1, 0, 0)
    strategy_returns = (-0.05, 0.10, 0.0, 0.25, 0.0, 0.0)
    urth_returns = (0.02, 0.03, 0.03, 0.05, 0.02, 0.01)
    rows = []
    starts = pd.date_range("2020-07-01", periods=6, freq="180D")
    for i in range(6):
        rows.append({
            "horizon": 5,
            "fold_id": f"F{i}",
            "fold_start": str(starts[i].date()),
            "fold_end": str((starts[i] + pd.Timedelta(days=170)).date()),
            "cagr_excess": excess[i],
            "trade_count": trades[i],
            "total_return": strategy_returns[i],
            "urth_total_return": urth_returns[i],
            "initial_value": 10000.0,
            "terminal_value": 10000.0 * (1.0 + strategy_returns[i]),
            "urth_terminal_value": 10000.0 * (1.0 + urth_returns[i]),
        })
    return rows


def main() -> int:
    summary = sparse_outer_fold_summary(_h5_rows())[0]
    assert summary["folds"] == 6
    assert summary["active_folds"] == 3
    assert summary["inactive_folds"] == 3
    assert abs(summary["positive_fold_fraction"] - (2.0 / 6.0)) < 1e-12
    assert abs(summary["positive_active_fold_fraction"] - (2.0 / 3.0)) < 1e-12
    assert summary["trade_count"] == 19
    assert summary["chained_oos_cagr_excess"] > 0.0

    search_meta = {
        "folds": [(f"F{i}", None, None) for i in range(8)],
        "search_coverage": {
            5: [{"final_fit": True, "coverage_complete": True, "dynamic_stage_complete": True}],
            10: [{"status": "A child process terminated abruptly, the process pool is not usable anymore"}],
            20: [{"status": "A child process terminated abruptly, the process pool is not usable anymore"}],
        },
        "final_policies": {5: object()},
        "final_thresholds": {5: 1.0},
    }
    gates = sparse_readiness_gates(
        [summary],
        [],
        search_meta,
        {},
        {},
        [],
    )
    assert not gates["development_execution_complete"]
    assert not gates["sparse_evidence_coverage_complete"]
    assert gates["horizon_execution_coverage"][5]["execution_complete"]
    assert gates["horizon_execution_coverage"][5]["active_folds"] == 3
    assert not gates["horizon_execution_coverage"][5]["evidence_coverage_complete"]
    assert gates["horizon_execution_coverage"][10]["execution_failure_detected"]
    assert not gates["horizon_execution_coverage"][10]["execution_complete"]

    assert _payload_has_execution_failure({
        "final_policy": {"x": 1},
        "history": [{"status": "A child process terminated abruptly, the process pool is not usable anymore"}],
    })

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        pd.DataFrame([
            {"horizon": 5, "tax_world": "PRE_TAX", "year": 2020, "trades": 10},
            {"horizon": 5, "tax_world": "PRE_TAX", "year": 2021, "trades": 14},
            {"horizon": 5, "tax_world": "PRE_TAX", "year": 2022, "trades": 4},
        ]).to_csv(root / "yearly_results.csv", index=False)
        trade_rows = []
        for year, count in ((2020, 5), (2021, 7), (2022, 2)):
            for i in range(count):
                trade_rows.append({
                    "horizon": 5,
                    "tax_world": "PRE_TAX",
                    "entry_date": f"{year}-01-{i + 1:02d}",
                })
        pd.DataFrame(trade_rows).to_csv(root / "trade_log.csv", index=False)
        (root / "summary.json").write_text(json.dumps({
            "status": "DEVELOPMENT_SUITE_COMPLETE",
            "decision": "INSUFFICIENT_SEARCH_COVERAGE",
            "outer_fold_summary": [summary],
            "readiness_gates": gates,
        }), encoding="utf-8")
        (root / "frozen_portfolio_policy.json").write_text(json.dumps({
            "status": "DEVELOPMENT_SUITE_COMPLETE",
            "decision": "INSUFFICIENT_SEARCH_COVERAGE",
        }), encoding="utf-8")
        (root / "REPORT.md").write_text(
            "Decision: `DEVELOPMENT_SUITE_COMPLETE` / `INSUFFICIENT_SEARCH_COVERAGE`\n",
            encoding="utf-8",
        )
        postprocess_research_artifacts(root)
        repaired = pd.read_csv(root / "yearly_results.csv")
        assert repaired["trades"].tolist() == [5, 7, 2]
        repaired_summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
        assert repaired_summary["decision"] == "INCOMPLETE_DEVELOPMENT_SEARCH"
        assert repaired_summary["status"] == "INCOMPLETE_DEVELOPMENT_SEARCH"
        assert repaired_summary["horizon_research_status"]["5"] == "INTERESTING_BUT_SPARSE_AND_UNSTABLE"
        assert repaired_summary["horizon_research_status"]["10"] == "NOT_TESTED_DUE_EXECUTION_FAILURE"
        assert repaired_summary["horizon_research_status"]["20"] == "NOT_TESTED_DUE_EXECUTION_FAILURE"

    print("RESEARCH_RUNTIME_V2_SELF_TEST_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
