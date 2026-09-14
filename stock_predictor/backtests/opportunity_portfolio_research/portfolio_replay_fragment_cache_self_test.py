from __future__ import annotations

from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from .portfolio_policy_search import run_walk_forward


def _fixture():
    dates = pd.date_range("2021-01-04", periods=30, freq="B")
    prices = [pd.DataFrame({
        "date": dates,
        "ticker": "URTH",
        "open": 100 + np.arange(len(dates)) * .10,
        "close": 100.05 + np.arange(len(dates)) * .10,
    })]
    for offset, ticker in enumerate(("AAA", "BBB", "CCC")):
        base = 50.0 + offset * 5.0
        prices.append(pd.DataFrame({
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
    return pd.concat(prices, ignore_index=True), pd.DataFrame(rows)


def main() -> int:
    prices, signals = _fixture()
    with tempfile.TemporaryDirectory() as tmp:
        cache = Path(tmp) / "fragments.sqlite3"
        outer1, history1, meta1 = run_walk_forward(
            signals, prices, horizons=(10,), budget=8, max_workers=2,
            fragment_cache_path=cache,
        )
        assert outer1 and history1
        first = meta1["performance"]["persistent_fragment_cache"]
        assert first["writes"] > 0

        outer2, history2, meta2 = run_walk_forward(
            signals, prices, horizons=(10,), budget=8, max_workers=2,
            fragment_cache_path=cache,
        )
        second = meta2["performance"]["persistent_fragment_cache"]
        assert second["entries_loaded_at_start"] > 0
        assert second["hits"] >= 1
        assert len(outer2) == len(outer1)
        assert len(history2) == len(history1)
        assert meta2["final_policies"][10].policy_id == meta1["final_policies"][10].policy_id
        assert meta2["final_thresholds"] == meta1["final_thresholds"]

    print("FRAGMENT_CACHE_SELF_TEST_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
