from __future__ import annotations

"""Deterministic contract and replay checks for Phase-2 Allocation QbD."""

from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from . import next_open_portfolio_replay as portfolio, portfolio_policy_search as search
from .allocation_qbd_evaluate import (
    BASELINE,
    add_paired_baseline,
    evaluate_surface,
    load_primary_phase1_design_space,
    summarize_treatments,
    validate_surface_coverage,
)
from .allocation_qbd_surface import (
    ENTRY_GRID_SIZE,
    EXIT_FAMILY_FIXED,
    REPLACEMENT_FIXED,
    SLEEVE_FIXED,
    _load_complete,
    _write_json,
    allocation_grid,
    allocation_search_contract,
)
from .portfolio_allocation_weights import (
    DEFAULT_TREATMENTS,
    allocation_weights,
    parse_allocation,
)
from .portfolio_policy_contracts import Policy
from .cost_contracts import CostModel
from .tax_contracts import TaxConfig

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


def _weight_checks() -> None:
    scores = [4.0, 3.0, 2.0, 1.0]
    equal = allocation_weights("EQUAL_ACTIVE", scores, threshold=0.0)
    assert np.array_equal(equal, np.full(4, 0.25))

    for treatment in DEFAULT_TREATMENTS:
        weights = allocation_weights(treatment, scores, threshold=0.5)
        assert len(weights) == 4
        assert np.isfinite(weights).all()
        assert (weights >= 0.0).all()
        assert abs(float(weights.sum()) - 1.0) < 1e-12
        if treatment != BASELINE:
            assert all(
                weights[i] >= weights[i + 1] - 1e-15
                for i in range(len(weights) - 1)
            )

    assert np.allclose(
        allocation_weights(
            "SCORE_SOFTMAX:2.0",
            [1.0, 1.0, 1.0],
            threshold=0.5,
        ),
        [1 / 3, 1 / 3, 1 / 3],
    )
    assert np.array_equal(
        allocation_weights("RANK_POWER:2.0", [7.0], threshold=0.5),
        np.ones(1),
    )
    _assert_raises(ValueError, parse_allocation, "RANK_POWER")
    _assert_raises(ValueError, parse_allocation, "EQUAL_ACTIVE:1.0")
    _assert_raises(ValueError, parse_allocation, "RANK_POWER:0")
    _assert_raises(ValueError, parse_allocation, "FUTURE_RETURN_WEIGHTED:2.0")
    _assert_raises(
        ValueError,
        allocation_weights,
        "SCORE_EXCESS_POWER:1.0",
        scores,
    )


def _phase1_loader_check() -> None:
    rows = []
    for h, d in sorted(EXPECTED_PRIMARY_PLATEAU):
        rows.append(
            {
                "prediction_horizon": h,
                "holding_days": d,
                "plateau_id": 5,
                "plateau_size": 14,
                "plateau_rank": 1,
                "design_space_pass": True,
            }
        )
    for h, d in [(10, 1), (10, 2), (11, 1), (11, 2)]:
        rows.append(
            {
                "prediction_horizon": h,
                "holding_days": d,
                "plateau_id": 3,
                "plateau_size": 4,
                "plateau_rank": 2,
                "design_space_pass": True,
            }
        )
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "qbd_design_space.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        actual = set(load_primary_phase1_design_space(path))
    assert actual == EXPECTED_PRIMARY_PLATEAU


def _committed_phase1_design_space_check() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    path = repo_root / "artifacts" / "prediction-hold-qbd-surface" / "qbd_design_space.csv"
    assert path.is_file(), f"committed Phase-1 design space missing: {path}"
    actual = set(load_primary_phase1_design_space(path))
    assert actual == EXPECTED_PRIMARY_PLATEAU


def _search_contract_check() -> None:
    grid = allocation_grid(24, 5, "RANK_POWER:1.5")
    assert len(grid) == ENTRY_GRID_SIZE == 48
    assert len({p.policy_id for p in grid}) == ENTRY_GRID_SIZE
    assert {p.horizon for p in grid} == {24}
    assert {p.holding_days for p in grid} == {5}
    assert {p.allocation for p in grid} == {"RANK_POWER:1.5"}
    assert {p.replacement for p in grid} == {REPLACEMENT_FIXED}
    assert {p.sleeve for p in grid} == {SLEEVE_FIXED}
    assert {p.exit_family for p in grid} == {EXIT_FAMILY_FIXED}
    assert {p.exit_value for p in grid} == {0.0}

    equal_grid = allocation_grid(24, 5, BASELINE)
    assert equal_grid[0].policy_id != grid[0].policy_id
    _assert_raises(ValueError, allocation_grid, 24, 5, BASELINE, 0.75)

    original = search.grid
    with allocation_search_contract(24, 5, "RANK_POWER:1.0"):
        key_a = search._horizon_checkpoint_key(24, 48)
        assert {p.holding_days for p in search.grid(24)} == {5}
        assert {p.sleeve for p in search.grid(24)} == {SLEEVE_FIXED}
        _assert_raises(ValueError, search.grid, 24, 0.75)
    assert search.grid is original

    with allocation_search_contract(24, 5, "RANK_POWER:2.0"):
        key_b = search._horizon_checkpoint_key(24, 48)
    with allocation_search_contract(24, 6, "RANK_POWER:1.0"):
        key_c = search._horizon_checkpoint_key(24, 48)
    assert len({key_a, key_b, key_c}) == 3


def _synthetic_replay(allocation: str) -> dict:
    dates = pd.bdate_range("2024-01-02", periods=5)
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
            "decision_date": [dates[0], dates[0]],
            "ticker": ["AAA", "BBB"],
            "score": [4.0, 2.0],
        }
    )
    policy = Policy(
        horizon=24,
        score_quantile=0.50,
        top_fraction=1.0,
        max_names=2,
        holding_days=2,
        replacement=REPLACEMENT_FIXED,
        allocation=allocation,
        sleeve=SLEEVE_FIXED,
    )
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
    original_requested = portfolio.requested_notionals

    def _must_not_run(*_args, **_kwargs):
        raise AssertionError("EQUAL_ACTIVE unexpectedly entered Phase-2 weighting helper")

    portfolio.requested_notionals = _must_not_run
    try:
        baseline = _synthetic_replay(BASELINE)
    finally:
        portfolio.requested_notionals = original_requested

    baseline_buys = {
        str(row["ticker"]): float(row["buy_notional"])
        for row in baseline["trades"]
    }
    assert set(baseline_buys) == {"AAA", "BBB"}
    assert abs(baseline_buys["AAA"] - 2500.0) < 1e-9
    assert abs(baseline_buys["BBB"] - 2500.0) < 1e-9

    rank = _synthetic_replay("RANK_POWER:1.0")
    rank_buys = {
        str(row["ticker"]): float(row["buy_notional"])
        for row in rank["trades"]
    }
    assert abs(sum(rank_buys.values()) - 5000.0) < 1e-9
    assert rank_buys["AAA"] > rank_buys["BBB"]
    assert abs(rank_buys["AAA"] / rank_buys["BBB"] - 2.0) < 1e-9

    softmax = _synthetic_replay("SCORE_SOFTMAX:2.0")
    softmax_buys = {
        str(row["ticker"]): float(row["buy_notional"])
        for row in softmax["trades"]
    }
    assert abs(sum(softmax_buys.values()) - 5000.0) < 1e-9
    assert softmax_buys["AAA"] > softmax_buys["BBB"]

    for replay_result in (baseline, rank, softmax):
        metrics = replay_result["metrics"]
        assert int(metrics["trade_count"]) == 2
        assert float(metrics["max_total_exposure"]) <= 1.0 + 5e-6
        assert float(metrics["max_abs_accounting_error_eur"]) <= 1e-7


def _resume_guard_check() -> None:
    valid = {
        "contract_id": "ENTRY_ALLOCATION_QBD_V1",
        "status": "COMPLETE",
        "prediction_horizon": 24,
        "holding_days": 5,
        "allocation": "RANK_POWER:1.5",
        "phase1_holding_and_prediction_frozen": True,
        "replacement": REPLACEMENT_FIXED,
        "sleeve": SLEEVE_FIXED,
        "v45_exit_overlay_used": False,
        "final_holdout_opened": False,
    }
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "cell.json"
        _write_json(path, valid)
        assert _load_complete(path, 24, 5, "RANK_POWER:1.5") is not None

        bad = dict(valid)
        bad["sleeve"] = 0.75
        _write_json(path, bad)
        assert _load_complete(path, 24, 5, "RANK_POWER:1.5") is None

        bad = dict(valid)
        bad["final_holdout_opened"] = True
        _write_json(path, bad)
        assert _load_complete(path, 24, 5, "RANK_POWER:1.5") is None


def _surface_evaluator_check() -> None:
    design = [(23, 3), (24, 3)]
    treatments = (BASELINE, "RANK_POWER:1.0")
    outer_rows = []
    status_rows = []
    for h, d in design:
        for allocation in treatments:
            status_rows.append(
                {
                    "prediction_horizon": h,
                    "holding_days": d,
                    "allocation": allocation,
                    "status": "COMPLETE",
                }
            )
            values = (
                (0.08, 0.06, 0.04, -0.01)
                if allocation == BASELINE
                else (0.10, 0.08, 0.06, -0.005)
            )
            for fold, excess in enumerate(values, 1):
                outer_rows.append(
                    {
                        "prediction_horizon": h,
                        "holding_days": d,
                        "allocation": allocation,
                        "fold_id": f"F{fold}",
                        "cagr_excess": excess,
                        "trade_count": 3,
                        "turnover": 1.0,
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
    assert len(cells) == len(design) * len(treatments)
    rank = treatment.loc[treatment["allocation"].eq("RANK_POWER:1.0")].iloc[0]
    assert bool(rank["allocation_qbd_pass"])
    assert float(rank["beat_equal_active_cell_fraction"]) == 1.0
    assert float(rank["median_delta_vs_equal_active"]) > 0.0

    broken = pd.DataFrame(status_rows)
    broken.loc[0, "status"] = "FAILED"
    _, _, broken_summary = evaluate_surface(
        pd.DataFrame(outer_rows),
        broken,
        design,
        treatments,
    )
    assert broken_summary["qbd_complete"] is False
    coverage = validate_surface_coverage(
        broken,
        [(h, d, allocation) for h, d in design for allocation in treatments],
    )
    assert len(coverage["failed_or_incomplete_cells"]) == 1

    duplicate = pd.concat([pd.DataFrame(status_rows), pd.DataFrame(status_rows).iloc[[0]]])
    coverage = validate_surface_coverage(
        duplicate,
        [(h, d, allocation) for h, d in design for allocation in treatments],
    )
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
                "allocation": BASELINE,
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
                "allocation": "RANK_POWER:1.5",
                "median_active_cagr_excess": 0.12,
                "q25_active_cagr_excess": 0.05,
                "worst_active_cagr_excess": -0.01,
                "robust_gate_pass": True,
                "trade_count": 20,
                "median_turnover": 1.0,
            }
        )
        spike = 0.50 if (h, d) == design[0] else 0.09
        rows.append(
            {
                "prediction_horizon": h,
                "holding_days": d,
                "allocation": "SCORE_SOFTMAX:2.0",
                "median_active_cagr_excess": spike,
                "q25_active_cagr_excess": 0.03,
                "worst_active_cagr_excess": -0.03,
                "robust_gate_pass": True,
                "trade_count": 20,
                "median_turnover": 1.0,
            }
        )
    cells = add_paired_baseline(pd.DataFrame(rows))
    treatment = summarize_treatments(cells, len(design))
    stable = treatment.loc[treatment["allocation"].eq("RANK_POWER:1.5")].iloc[0]
    spike = treatment.loc[treatment["allocation"].eq("SCORE_SOFTMAX:2.0")].iloc[0]
    assert bool(stable["allocation_qbd_pass"])
    assert not bool(spike["allocation_qbd_pass"])
    assert float(stable["beat_equal_active_cell_fraction"]) == 1.0


def main() -> int:
    _weight_checks()
    _phase1_loader_check()
    _committed_phase1_design_space_check()
    _search_contract_check()
    _replay_integration_check()
    _resume_guard_check()
    _surface_evaluator_check()
    _broad_stability_selection_check()

    assert len(EXPECTED_PRIMARY_PLATEAU) == 14
    assert len(DEFAULT_TREATMENTS) == 10
    assert len(EXPECTED_PRIMARY_PLATEAU) * len(DEFAULT_TREATMENTS) == 140
    print("ALLOCATION_QBD_FULL_SELF_TEST_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
