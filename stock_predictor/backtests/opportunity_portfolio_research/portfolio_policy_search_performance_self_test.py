from __future__ import annotations

import pandas as pd
import numpy as np

from .portfolio_policy_contracts import Policy
from .cost_contracts import CostModel
from .tax_contracts import TaxConfig
from .next_open_portfolio_replay import prepare_prices, prepare_signals, replay
from .portfolio_policy_search import _balanced_budget, _resolved_threshold, grid, run_walk_forward


def _fixture():
    dates = pd.date_range("2021-01-04", periods=30, freq="B")
    price_frames = [
        pd.DataFrame({"date": dates, "ticker": "URTH", "open": 100 + np.arange(len(dates)) * .10, "close": 100.05 + np.arange(len(dates)) * .10})
    ]
    for offset, ticker in enumerate(("AAA", "BBB", "CCC")):
        base = 50.0 + offset * 5.0
        price_frames.append(
            pd.DataFrame({
                "date": dates,
                "ticker": ticker,
                "open": base + np.arange(len(dates)) * (.15 + offset * .02),
                "close": base + .05 + np.arange(len(dates)) * (.15 + offset * .02),
            })
        )
    prices = pd.concat(price_frames, ignore_index=True)

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
    signals = pd.DataFrame(rows)
    return dates, prices, signals


def main() -> int:
    dates, prices, signals = _fixture()
    policy = Policy(10, .95, .50, 2, 5, sleeve=.50)
    threshold = _resolved_threshold(signals, policy)

    price_prepared_1 = prepare_prices(prices)
    price_prepared_2 = prepare_prices(prices)
    assert price_prepared_1 is price_prepared_2

    signal_prepared_1 = prepare_signals(signals)
    signal_prepared_2 = prepare_signals(signals)
    assert signal_prepared_1 is signal_prepared_2

    raw = replay(
        signals, prices, policy, CostModel(20), TaxConfig(False),
        start=dates[0], end=dates[-1], initial=10000.0,
        resolved_threshold=threshold,
    )
    prepared = replay(
        signals, prices, policy, CostModel(20), TaxConfig(False),
        start=dates[0], end=dates[-1], initial=10000.0,
        resolved_threshold=threshold, prepared_signals=signal_prepared_1,
    )

    assert abs(float(raw["metrics"]["terminal_value"]) - float(prepared["metrics"]["terminal_value"])) < 1e-9
    assert abs(float(raw["metrics"]["cagr_excess"]) - float(prepared["metrics"]["cagr_excess"])) < 1e-12
    assert raw["trades"] == prepared["trades"]

    # Daily-top threshold contract: the three-ticker cross-section must resolve from
    # each day's maximum score, never from all ticker rows.
    daily_tops = signals.groupby("decision_date")["score"].max().to_numpy(dtype=float)
    expected = float(np.quantile(daily_tops, policy.score_quantile))
    assert abs(threshold - expected) < 1e-12

    selected, meta = _balanced_budget(grid(10), 48)
    assert len(selected) == 48
    assert meta["coverage_complete"] is True
    assert {5, 7, 10}.issubset(set(meta["coverage"]["holding_days"]))

    # End-to-end smoke test of the optimized search path. Budget=8 is sufficient for
    # marginal coverage on the tiny fixture and keeps this test fast.
    outer, history, search_meta = run_walk_forward(
        signals, prices, horizons=(10,), budget=8, max_workers=2
    )
    assert len(outer) >= 1
    assert history
    perf = search_meta.get("performance", {})
    assert perf.get("price_panel_prepared_once") is True
    assert perf.get("horizon_signal_frame_prepared_once") is True
    assert perf.get("historical_views_reuse_prepared_horizon") is True
    assert perf.get("robustness_parallelized") is True

    print("PERFORMANCE_SELF_TEST_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
