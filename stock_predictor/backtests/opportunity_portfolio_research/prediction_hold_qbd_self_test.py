from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import os

import pandas as pd

from . import portfolio_replay_process_backend as multicore_backend, portfolio_policy_search as search
from .prediction_hold_qbd_evaluate import (
    add_local_stability,
    aggregate_outer_results,
    assign_design_space,
    evaluate_surface,
    expected_cells,
    validate_surface_coverage,
)
from .prediction_hold_qbd_surface import ENTRY_GRID_SIZE, fixed_hold_grid, fixed_hold_search_contract
from .prediction_hold_qbd_process_pool_readiness import _scope_key, _wait_for_process_pool_readiness, qbd_process_pool_readiness_contract


def _outer_fixture() -> pd.DataFrame:
    rows = []
    for horizon, hold in expected_cells(1, 3):
        for fold, excess in enumerate((.04, .02, .01, -.005), 1):
            rows.append({
                "prediction_horizon": horizon,
                "holding_days": hold,
                "fold_id": f"F{fold}",
                "cagr_excess": excess + horizon * .001 - hold * .0005,
                "trade_count": 3,
                "turnover": 1.0,
                "worst_relative_drawdown": -.08,
            })
    return pd.DataFrame(rows)


def main() -> int:
    cells = expected_cells()
    assert len(cells) == 465 and len(set(cells)) == 465
    assert all(1 <= hold <= horizon <= 30 for horizon, hold in cells)

    grid = fixed_hold_grid(10, 5)
    assert len(grid) == ENTRY_GRID_SIZE == 48
    assert {p.horizon for p in grid} == {10}
    assert {p.holding_days for p in grid} == {5}
    assert len({p.policy_id for p in grid}) == 48

    original_grid = search.grid
    with fixed_hold_search_contract(10, 5):
        assert {p.holding_days for p in search.grid(10)} == {5}
        key5 = search._horizon_checkpoint_key(10, 48)
    assert search.grid is original_grid
    with fixed_hold_search_contract(10, 6):
        key6 = search._horizon_checkpoint_key(10, 48)
    assert key5 != key6

    # QbD pool identity must be stable across the DataFrame copies created by
    # run_walk_forward so one initialized worker pool serves the complete horizon.
    queue_prices = pd.DataFrame({"date": pd.to_datetime(["2024-01-02"]), "ticker": ["AAA"]})
    queue_signals = pd.DataFrame({
        "horizon": [10, 10],
        "decision_date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
    })
    assert _scope_key(queue_signals, queue_prices, 12) == _scope_key(
        queue_signals.copy(), queue_prices, 12
    )
    assert _scope_key(queue_signals, queue_prices, 12) != _scope_key(
        queue_signals.assign(horizon=11), queue_prices, 12
    )
    original_ensure_pool = multicore_backend._ensure_pool
    with qbd_process_pool_readiness_contract():
        assert multicore_backend._ensure_pool is not original_ensure_pool
    assert multicore_backend._ensure_pool is original_ensure_pool

    # Integration-level scheduler smoke test: both spawned workers must take one
    # blocking readiness probe before the barrier releases them.
    old_timeout = os.environ.get("QBD_WORKER_READY_TIMEOUT_SECONDS")
    os.environ["QBD_WORKER_READY_TIMEOUT_SECONDS"] = "30"
    try:
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=2, mp_context=ctx) as pool:
            _wait_for_process_pool_readiness(pool, 2)
    finally:
        if old_timeout is None:
            os.environ.pop("QBD_WORKER_READY_TIMEOUT_SECONDS", None)
        else:
            os.environ["QBD_WORKER_READY_TIMEOUT_SECONDS"] = old_timeout

    expected = expected_cells(1, 3)
    outer = _outer_fixture()
    aggregated = aggregate_outer_results(outer)
    assert len(aggregated) == 6 and aggregated["robust_gate_pass"].all()
    stable = add_local_stability(aggregated, expected)
    assert stable["local_stability_pass"].all()
    designed, plateaus = assign_design_space(stable, 3)
    assert len(plateaus) == 1 and int(plateaus.iloc[0]["cells"]) == 6
    assert designed["design_space_pass"].all()

    status = pd.DataFrame([{"prediction_horizon": h, "holding_days": d, "status": "COMPLETE"} for h, d in expected])
    evaluated, plateau, summary = evaluate_surface(outer, status, prediction_min=1, prediction_max=3)
    assert summary["qbd_complete"] is True and summary["design_space_cells"] == 6
    assert len(evaluated) == 6 and len(plateau) == 1

    broken = status.copy(); broken.loc[0, "status"] = "FAILED"
    coverage = validate_surface_coverage(broken, expected)
    assert not coverage["surface_complete"] and len(coverage["failed_or_incomplete_cells"]) == 1

    duplicate = pd.concat([status, status.iloc[[0]]], ignore_index=True)
    coverage = validate_surface_coverage(duplicate, expected)
    assert not coverage["surface_complete"] and coverage["duplicate_cells"]

    print("PREDICTION_HOLD_QBD_END_TO_END_SELF_TEST_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
