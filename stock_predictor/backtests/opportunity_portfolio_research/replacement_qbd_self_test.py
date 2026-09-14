from __future__ import annotations

"""Deterministic contract, replay and evaluator checks for Phase-3 Replacement QbD."""

import json
from pathlib import Path
import tempfile

import pandas as pd

from . import next_open_portfolio_replay as portfolio, portfolio_policy_search as search
from .portfolio_policy_contracts import Policy
from .cost_contracts import CostModel
from .tax_contracts import TaxConfig
from .replacement_qbd_contract import (
    ALLOCATION_FIXED,
    BASELINE,
    CONTRACT_ID,
    DEFAULT_TREATMENTS,
    EXIT_FAMILY_FIXED,
    EXIT_VALUE_FIXED,
    SLEEVE_FIXED,
    parse_replacement,
)
from .replacement_qbd_evaluate import (
    add_paired_baseline,
    evaluate_surface,
    load_phase2_allocation_lock,
    summarize_treatments,
    validate_surface_coverage,
)
from .replacement_qbd_surface import (
    ENTRY_GRID_SIZE,
    _load_complete,
    _write_json,
    replacement_grid,
    replacement_search_contract,
)
from .allocation_qbd_evaluate import load_primary_phase1_design_space

EXPECTED_PRIMARY_PLATEAU = {
    (23, 3),
    (23, 4),
    (23, 6),
    (24, 3),
    (24, 4),
    (24, 5),
    (24, 6),
    (24, 7),
    (25, 4),
    (25, 6),
    (25, 7),
    (25, 8),
    (26, 7),
    (26, 8),
}


def _assert_raises(expected_exception, fn, *args, **kwargs) -> None:
    try:
        fn(*args, **kwargs)
    except expected_exception:
        return
    raise AssertionError(
        f"expected {expected_exception.__name__} from {getattr(fn, '__name__', fn)}"
    )


def _parser_check() -> None:
    assert parse_replacement("ignore_new") == BASELINE
    assert parse_replacement("REPLACE_WEAKEST") == "REPLACE_WEAKEST"
    _assert_raises(ValueError, parse_replacement, "REPLACE_RANDOM")
    _assert_raises(ValueError, parse_replacement, "")


def _phase1_provenance_check() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    path = repo_root / "artifacts" / "prediction-hold-qbd-surface" / "qbd_design_space.csv"
    assert path.is_file(), f"committed Phase-1 design space missing: {path}"
    actual = set(load_primary_phase1_design_space(path))
    assert actual == EXPECTED_PRIMARY_PLATEAU


def _write_phase2_fixture(root: Path, *, winner: str = ALLOCATION_FIXED) -> tuple[Path, Path]:
    summary = {
        "contract_id": "ENTRY_ALLOCATION_QBD_V1",
        "qbd_complete": True,
        "final_holdout_opened": False,
        "interpolation_used": False,
        "v45_exit_overlay_used": False,
        "best_nonbaseline_treatment": None,
    }
    summary_path = root / "allocation_qbd_summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    treatment_path = root / "allocation_qbd_treatment_summary.csv"
    pd.DataFrame(
        [
            {
                "allocation": winner,
                "allocation_qbd_pass": True,
                "baseline_reference": winner == ALLOCATION_FIXED,
            },
            {
                "allocation": "RANK_POWER:1.0",
                "allocation_qbd_pass": False,
                "baseline_reference": False,
            },
        ]
    ).to_csv(treatment_path, index=False)
    return summary_path, treatment_path


def _phase2_lock_check() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        summary, treatment = _write_phase2_fixture(root)
        assert load_phase2_allocation_lock(summary, treatment) == ALLOCATION_FIXED

        bad_summary = json.loads(summary.read_text(encoding="utf-8"))
        bad_summary["final_holdout_opened"] = True
        summary.write_text(json.dumps(bad_summary), encoding="utf-8")
        _assert_raises(
            ValueError,
            load_phase2_allocation_lock,
            summary,
            treatment,
        )

        summary, treatment = _write_phase2_fixture(root, winner="RANK_POWER:1.0")
        _assert_raises(
            ValueError,
            load_phase2_allocation_lock,
            summary,
            treatment,
        )

    repo_root = Path(__file__).resolve().parents[3]
    summary = repo_root / "artifacts" / "allocation-qbd" / "allocation_qbd_summary.json"
    treatment = (
        repo_root
        / "artifacts"
        / "allocation-qbd"
        / "allocation_qbd_treatment_summary.csv"
    )
    assert summary.is_file(), f"committed Phase-2 summary missing: {summary}"
    assert treatment.is_file(), f"committed Phase-2 treatment summary missing: {treatment}"
    assert load_phase2_allocation_lock(summary, treatment) == ALLOCATION_FIXED


def _search_contract_check() -> None:
    grid = replacement_grid(24, 5, "REPLACE_WEAKEST")
    assert len(grid) == ENTRY_GRID_SIZE == 48
    assert len({p.policy_id for p in grid}) == ENTRY_GRID_SIZE
    assert {p.horizon for p in grid} == {24}
    assert {p.holding_days for p in grid} == {5}
    assert {p.replacement for p in grid} == {"REPLACE_WEAKEST"}
    assert {p.allocation for p in grid} == {ALLOCATION_FIXED}
    assert {p.sleeve for p in grid} == {SLEEVE_FIXED}
    assert {p.exit_family for p in grid} == {EXIT_FAMILY_FIXED}
    assert {p.exit_value for p in grid} == {EXIT_VALUE_FIXED}

    baseline_grid = replacement_grid(24, 5, BASELINE)
    assert baseline_grid[0].policy_id != grid[0].policy_id
    _assert_raises(ValueError, replacement_grid, 24, 5, BASELINE, 0.75)

    original = search.grid
    with replacement_search_contract(24, 5, BASELINE):
        key_a = search._horizon_checkpoint_key(24, 48)
        active = search.grid(24)
        assert {p.holding_days for p in active} == {5}
        assert {p.replacement for p in active} == {BASELINE}
        assert {p.allocation for p in active} == {ALLOCATION_FIXED}
        _assert_raises(ValueError, search.grid, 24, 0.75)
    assert search.grid is original

    with replacement_search_contract(24, 5, "REPLACE_WEAKEST"):
        key_b = search._horizon_checkpoint_key(24, 48)
    with replacement_search_contract(24, 6, BASELINE):
        key_c = search._horizon_checkpoint_key(24, 48)
    assert len({key_a, key_b, key_c}) == 3


def _synthetic_replay(replacement: str | None) -> dict:
    dates = pd.bdate_range("2024-01-02", periods=7)
    price_rows = []
    for date in dates:
        for ticker in ("URTH", "AAA", "BBB"):
            price_rows.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "open": 100.0,
                    "close": 100.0,
                }
            )
    prices = pd.DataFrame(price_rows)
    signals = pd.DataFrame(
        {
            "decision_date": [dates[0], dates[1]],
            "ticker": ["AAA", "BBB"],
            "score": [5.0, 10.0],
        }
    )
    kwargs = dict(
        horizon=24,
        score_quantile=0.50,
        top_fraction=1.0,
        max_names=1,
        holding_days=4,
        allocation=ALLOCATION_FIXED,
        sleeve=SLEEVE_FIXED,
    )
    policy = Policy(**kwargs) if replacement is None else Policy(**kwargs, replacement=replacement)
    return portfolio.replay(
        signals,
        prices,
        policy,
        CostModel(roundtrip_bps=0.0),
        TaxConfig(enabled=False),
        initial=10000.0,
        resolved_threshold=0.0,
    )


def _replay_integration_check() -> None:
    implicit_baseline = _synthetic_replay(None)
    baseline = _synthetic_replay(BASELINE)
    replacement = _synthetic_replay("REPLACE_WEAKEST")

    assert implicit_baseline["trades"] == baseline["trades"]
    for metric in (
        "terminal_value",
        "trade_count",
        "turnover",
        "max_positions",
        "max_total_exposure",
    ):
        assert implicit_baseline["metrics"][metric] == baseline["metrics"][metric]

    baseline_tickers = [str(row["ticker"]) for row in baseline["trades"]]
    replacement_tickers = [str(row["ticker"]) for row in replacement["trades"]]
    assert baseline_tickers == ["AAA"]
    assert replacement_tickers == ["AAA", "BBB"]
    assert int(baseline["trades"][0]["holding_days"]) == 4
    assert int(replacement["trades"][0]["holding_days"]) == 1
    assert int(replacement["metrics"]["trade_count"]) == 2
    assert float(replacement["metrics"]["turnover"]) > float(
        baseline["metrics"]["turnover"]
    )

    for result in (baseline, replacement):
        metrics = result["metrics"]
        assert int(metrics["max_positions"]) <= 1
        assert float(metrics["max_total_exposure"]) <= 1.0 + 5e-6
        assert float(metrics["max_abs_accounting_error_eur"]) <= 1e-7


def _resume_guard_check() -> None:
    valid = {
        "contract_id": CONTRACT_ID,
        "status": "COMPLETE",
        "prediction_horizon": 24,
        "holding_days": 5,
        "replacement": "REPLACE_WEAKEST",
        "phase1_holding_and_prediction_frozen": True,
        "phase2_allocation_frozen": True,
        "allocation": ALLOCATION_FIXED,
        "sleeve": SLEEVE_FIXED,
        "exit_family": EXIT_FAMILY_FIXED,
        "exit_value": EXIT_VALUE_FIXED,
        "v45_exit_overlay_used": False,
        "final_holdout_opened": False,
    }
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "cell.json"
        _write_json(path, valid)
        assert _load_complete(path, 24, 5, "REPLACE_WEAKEST") is not None

        for field, value in (
            ("allocation", "RANK_POWER:1.0"),
            ("sleeve", 0.75),
            ("final_holdout_opened", True),
            ("phase2_allocation_frozen", False),
            ("replacement", BASELINE),
        ):
            bad = dict(valid)
            bad[field] = value
            _write_json(path, bad)
            assert _load_complete(path, 24, 5, "REPLACE_WEAKEST") is None


def _surface_evaluator_check() -> None:
    design = [(23, 3), (24, 3)]
    treatments = (BASELINE, "REPLACE_WEAKEST")
    outer_rows = []
    status_rows = []
    for h, d in design:
        for replacement in treatments:
            status_rows.append(
                {
                    "prediction_horizon": h,
                    "holding_days": d,
                    "replacement": replacement,
                    "status": "COMPLETE",
                }
            )
            values = (
                (0.08, 0.06, 0.04, -0.01)
                if replacement == BASELINE
                else (0.11, 0.09, 0.07, 0.00)
            )
            for fold, excess in enumerate(values, 1):
                outer_rows.append(
                    {
                        "prediction_horizon": h,
                        "holding_days": d,
                        "replacement": replacement,
                        "fold_id": f"F{fold}",
                        "cagr_excess": excess,
                        "trade_count": 3,
                        "turnover": 1.0 if replacement == BASELINE else 1.2,
                        "worst_relative_drawdown": -0.10,
                    }
                )

    cells, treatment, summary = evaluate_surface(
        pd.DataFrame(outer_rows),
        pd.DataFrame(status_rows),
        design,
        treatments,
    )
    assert summary["qbd_complete"] is True
    assert summary["phase2_allocation_fixed"] == ALLOCATION_FIXED
    assert len(cells) == len(design) * len(treatments)
    row = treatment.loc[
        treatment["replacement"].eq("REPLACE_WEAKEST")
    ].iloc[0]
    assert bool(row["replacement_qbd_pass"])
    assert float(row["beat_ignore_new_cell_fraction"]) == 1.0
    assert float(row["median_delta_vs_ignore_new"]) > 0.0
    assert float(row["median_turnover_delta_vs_ignore_new"]) > 0.0

    broken = pd.DataFrame(status_rows)
    broken.loc[0, "status"] = "FAILED"
    _, _, broken_summary = evaluate_surface(
        pd.DataFrame(outer_rows), broken, design, treatments
    )
    assert broken_summary["qbd_complete"] is False

    wanted = [
        (h, d, replacement)
        for h, d in design
        for replacement in treatments
    ]
    coverage = validate_surface_coverage(broken, wanted)
    assert len(coverage["failed_or_incomplete_cells"]) == 1

    duplicate = pd.concat(
        [pd.DataFrame(status_rows), pd.DataFrame(status_rows).iloc[[0]]]
    )
    coverage = validate_surface_coverage(duplicate, wanted)
    assert coverage["surface_complete"] is False
    assert coverage["duplicate_cells"]


def _broad_stability_selection_check() -> None:
    rows = []
    design = sorted(EXPECTED_PRIMARY_PLATEAU)
    for h, d in design:
        rows.append(
            {
                "prediction_horizon": h,
                "holding_days": d,
                "replacement": BASELINE,
                "median_active_cagr_excess": 0.10,
                "q25_active_cagr_excess": 0.04,
                "worst_active_cagr_excess": -0.02,
                "robust_gate_pass": True,
                "trade_count": 20,
                "median_turnover": 1.0,
            }
        )
        rows.append(
            {
                "prediction_horizon": h,
                "holding_days": d,
                "replacement": "REPLACE_WEAKEST",
                "median_active_cagr_excess": 0.12,
                "q25_active_cagr_excess": 0.05,
                "worst_active_cagr_excess": -0.01,
                "robust_gate_pass": True,
                "trade_count": 24,
                "median_turnover": 1.2,
            }
        )
        spike = 0.50 if (h, d) == design[0] else 0.09
        rows.append(
            {
                "prediction_horizon": h,
                "holding_days": d,
                "replacement": "SPIKE_ONLY",
                "median_active_cagr_excess": spike,
                "q25_active_cagr_excess": 0.03,
                "worst_active_cagr_excess": -0.03,
                "robust_gate_pass": True,
                "trade_count": 24,
                "median_turnover": 1.5,
            }
        )
    cells = add_paired_baseline(pd.DataFrame(rows))
    treatment = summarize_treatments(cells, len(design))
    stable = treatment.loc[
        treatment["replacement"].eq("REPLACE_WEAKEST")
    ].iloc[0]
    spike = treatment.loc[treatment["replacement"].eq("SPIKE_ONLY")].iloc[0]
    assert bool(stable["replacement_qbd_pass"])
    assert not bool(spike["replacement_qbd_pass"])
    assert float(stable["beat_ignore_new_cell_fraction"]) == 1.0
    assert float(stable["q25_delta_vs_ignore_new"]) >= 0.0


def main() -> int:
    _parser_check()
    _phase1_provenance_check()
    _phase2_lock_check()
    _search_contract_check()
    _replay_integration_check()
    _resume_guard_check()
    _surface_evaluator_check()
    _broad_stability_selection_check()

    assert len(EXPECTED_PRIMARY_PLATEAU) == 14
    assert len(DEFAULT_TREATMENTS) == 2
    assert len(EXPECTED_PRIMARY_PLATEAU) * len(DEFAULT_TREATMENTS) == 28
    print("REPLACEMENT_QBD_END_TO_END_SELF_TEST_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
