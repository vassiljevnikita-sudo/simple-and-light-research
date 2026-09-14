from __future__ import annotations

import multiprocessing as mp

import numpy as np
import pandas as pd

from . import portfolio_policy_search as search
from .portfolio_replay_process_backend import backend_stats, install_multicore_backend, shutdown_multicore_backend


def _fixture() -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.date_range("2021-01-04", periods=36, freq="B")
    price_frames = [
        pd.DataFrame({
            "date": dates,
            "ticker": "URTH",
            "open": 100.0 + np.arange(len(dates)) * .10,
            "close": 100.05 + np.arange(len(dates)) * .10,
        })
    ]
    for offset, ticker in enumerate(("AAA", "BBB", "CCC")):
        base = 50.0 + offset * 5.0
        price_frames.append(pd.DataFrame({
            "date": dates,
            "ticker": ticker,
            "open": base + np.arange(len(dates)) * (.15 + offset * .02),
            "close": base + .05 + np.arange(len(dates)) * (.15 + offset * .02),
        }))
    rows = []
    for i, d in enumerate(dates[:-1]):
        for rank, ticker in enumerate(("AAA", "BBB", "CCC")):
            rows.append({
                "decision_date": d,
                "ticker": ticker,
                "fold_id": f"WF_{i // 6:03d}",
                "horizon": 10,
                "score": 2.0 + i * .01 - rank * .2,
            })
    return pd.concat(price_frames, ignore_index=True), pd.DataFrame(rows)


def _outer_signature(rows: list[dict]) -> list[tuple]:
    return [
        (
            str(r["fold_id"]),
            str(r["policy_id"]),
            float(r["threshold"]),
            int(r.get("trade_count", 0)),
            round(float(r.get("cagr_excess", 0.0)), 12),
            round(float(r.get("terminal_value", 0.0)), 8),
        )
        for r in rows
    ]


def main() -> int:
    prices, signals = _fixture()
    original_parallel_map = search._parallel_map

    # Baseline: current research implementation with its original thread executor.
    thread_outer, _thread_history, thread_meta = search.run_walk_forward(
        signals, prices, horizons=(10,), budget=8, max_workers=2
    )

    try:
        install_multicore_backend(2)
        process_outer, _process_history, process_meta = search.run_walk_forward(
            signals, prices, horizons=(10,), budget=8, max_workers=2
        )
        stats = backend_stats()
    finally:
        shutdown_multicore_backend()
        search._parallel_map = original_parallel_map

    assert _outer_signature(process_outer) == _outer_signature(thread_outer)
    assert process_meta["final_policies"][10].policy_id == thread_meta["final_policies"][10].policy_id
    assert process_meta["final_thresholds"] == thread_meta["final_thresholds"]
    assert stats["pool_starts"] >= 1
    assert stats["history_jobs_computed"] >= 1
    assert stats["fold_jobs_computed"] >= 1

    print("MULTICORE_SELF_TEST_PASS")
    return 0


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())
