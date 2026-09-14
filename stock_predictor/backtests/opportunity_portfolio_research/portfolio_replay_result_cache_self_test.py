from __future__ import annotations

from pathlib import Path
import tempfile

import pandas as pd

from .portfolio_replay_result_cache import ReplayResultStore


def main() -> int:
    result = {
        "metrics": {
            "terminal_value": 12345.67,
            "cagr_excess": 0.1234,
            "trade_count": 2,
        },
        "curve": pd.DataFrame({
            "date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "strategy_value": [10000.0, 10010.0],
            "urth_value": [10000.0, 10005.0],
        }),
        "trades": [
            {"ticker": "AAA", "entry_date": pd.Timestamp("2024-01-02"), "exit_date": pd.Timestamp("2024-01-03")}
        ],
        "cost_audit": {"transaction_cost_eur": 1.25},
        "tax_audit": {"tax_paid": 0.0},
    }

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "final_replay_cache.sqlite3"
        namespace = "synthetic-research-namespace"
        key = "deterministic-replay-key"

        first = ReplayResultStore(path, namespace)
        found, _ = first.get(key)
        assert found is False
        first.put(key, result)
        stats1 = first.stats()
        assert stats1["writes"] == 1
        assert stats1["entries"] == 1
        first.close()

        second = ReplayResultStore(path, namespace)
        found, restored = second.get(key)
        assert found is True
        assert restored["metrics"] == result["metrics"]
        assert restored["trades"] == result["trades"]
        pd.testing.assert_frame_equal(restored["curve"], result["curve"])
        assert second.stats()["entries_loaded_at_start"] == 1
        second.close()

        # A different research namespace must never see the old replay.
        third = ReplayResultStore(path, namespace + "-changed")
        found, _ = third.get(key)
        assert found is False
        third.close()

    print("REPLAY_RESULT_CACHE_SELF_TEST_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
