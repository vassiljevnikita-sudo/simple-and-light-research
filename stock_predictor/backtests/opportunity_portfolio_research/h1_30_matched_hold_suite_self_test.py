from __future__ import annotations

import math
import pandas as pd

from . import portfolio_policy_search as search
from .h1_30_matched_hold_search_contract import (
    HORIZONS,
    mark_clean_search_meta,
    matched_balanced_budget,
    matched_grid,
)
from .h1_30_matched_hold_entrypoint import _matched_parameter_neighbors, _normalize_clean_search_meta
from .v45_exit_overlay import (
    StopSpec,
    V45_CANDIDATES,
    _overlay_replay,
    _schedule_from_clean_trade_log,
    _stable_job_key,
    _stop_threshold,
)


def _flat_ohlc(dates, ticker_rows):
    result = {
        "URTH": {
            "dates": tuple(dates),
            "bars": {d: (100.0, 100.0, 100.0, 100.0) for d in dates},
            "atr14": {d: 0.0 for d in dates},
        }
    }
    result["AAA"] = {
        "dates": tuple(dates),
        "bars": {d: tuple(map(float, ticker_rows[i])) for i, d in enumerate(dates)},
        "atr14": {d: 1.0 for d in dates},
    }
    return result


def _schedule(dates, exit_index):
    return {
        "dates": tuple(dates),
        "events": {
            dates[1]: [{"kind": "buy", "ticker": "AAA", "target_gross": 5000.0}],
            dates[exit_index]: [{"kind": "sell", "ticker": "AAA", "reason": "FIXED_HORIZON"}],
        },
    }


def main() -> None:
    # H1-H30 is a complete, horizon-matched clean grid: 4 quantiles x 3 top
    # fractions x 4 max-name choices = 48 policies per horizon.
    for h in HORIZONS:
        items = matched_grid(h)
        assert len(items) == 48
        assert all(p.horizon == h and p.holding_days == h for p in items)
        selected, meta = matched_balanced_budget(items, 48)
        assert len(selected) == 48 and meta["coverage_complete"] is True
        assert all(p.exit_family == "FIXED" and p.replacement == "IGNORE_NEW" for p in selected)

        # Reuse the standard parameter-plateau machinery but never allow its
        # holding-day neighbors to violate the clean Hn -> holding_days=n contract.
        plateau_neighbors = _matched_parameter_neighbors(items[0])
        assert plateau_neighbors
        assert all(dimension != "holding_days" for dimension, _ in plateau_neighbors)
        assert all(neighbor.holding_days == h for _, neighbor in plateau_neighbors)

    # The standard suite historically expects an in-arm dynamic-exit stage. For the
    # paired H1-H30 design that stage is deliberately externalized to V4.5, therefore
    # it is complete-by-contract (N/A), not an execution failure.
    clean_meta = mark_clean_search_meta({"dynamic_stage_complete": False, "dynamic_paths_evaluated": 9})
    assert clean_meta["dynamic_stage_complete"] is True
    assert clean_meta["dynamic_paths_evaluated"] == 0
    assert clean_meta["dynamic_stage_status"] == "NOT_APPLICABLE_CLEAN_FIXED_EXIT_SEPARATE_V45_ARM"
    assert clean_meta["clean_fixed_exit_only"] is True

    # Existing scientifically compatible horizon checkpoints remain reusable. Their
    # old dynamic-stage metadata is upgraded after loading rather than forcing replay.
    resumed = _normalize_clean_search_meta({
        "search_coverage": {
            10: [{
                "final_fit": True,
                "coverage_complete": True,
                "dynamic_stage_complete": False,
                "dynamic_paths_evaluated": 7,
            }]
        }
    })
    resumed_row = resumed["search_coverage"][10][0]
    assert resumed_row["dynamic_stage_complete"] is True
    assert resumed_row["dynamic_paths_evaluated"] == 0
    assert resumed["research_contract"] == "H1_30_MATCHED_HOLD_V1"

    ids = {x.candidate_id for x in V45_CANDIDATES}
    assert ids == {
        "NONE", "HARD_10_HOLD2_CD5", "HARD_12_HOLD5_CD10",
        "TRAIL_12_HOLD2_CD5", "TRAIL_15_HOLD5_CD10",
        "ATR_3_HOLD2_CD5", "ATR_4_HOLD5_CD10",
        "COMBINED_12_15_ATR3_HOLD5",
    }

    # Direct stop contract: minimum hold is enforced.
    pos = {
        "entry_price": 100.0, "holding_days": 1,
        "prior_known_high": 120.0, "prior_known_atr": 5.0,
    }
    hard = StopSpec("T", "HARD", hard_loss_fraction=0.10, minimum_holding_days=2)
    assert _stop_threshold(pos, hard) is None
    pos["holding_days"] = 2
    assert abs(float(_stop_threshold(pos, hard)) - 90.0) < 1e-12

    dates = list(pd.bdate_range("2024-01-02", periods=6))
    flat_bars = [(100, 101, 99, 100)] * 6
    flat_ohlc = _flat_ohlc(dates, flat_bars)

    # Production V4.5 must consume the already-realized clean Evidence trade log,
    # not rerun candidate selection. 20-bps clean buy_notional=4995 maps back to
    # the exact original 5000 gross target (10 bps per side).
    outer_row = {
        "horizon": 5,
        "fold_id": "F0",
        "fold_start": str(dates[0].date()),
        "fold_end": str(dates[-1].date()),
        "trade_count": 1,
        "policy_id": "clean-policy",
        "threshold": 0.1,
        "terminal_value": 10000.0,
    }
    clean_trade_log = pd.DataFrame([{
        "horizon": 5,
        "fold_id": "F0",
        "trade_index": 0,
        "ticker": "AAA",
        "entry_date": dates[1],
        "exit_date": dates[4],
        "buy_notional": 4995.0,
    }])
    reused_schedule = _schedule_from_clean_trade_log(outer_row, clean_trade_log, flat_ohlc)
    buy_event = reused_schedule["events"][dates[1]][0]
    sell_event = reused_schedule["events"][dates[4]][0]
    assert reused_schedule["source"] == "EXISTING_CLEAN_OUTER_OOS_TRADE_LOG"
    assert buy_event["source"] == "outer_oos_trade_log.csv"
    assert abs(float(buy_event["target_gross"]) - 5000.0) < 1e-9
    assert sell_event["kind"] == "sell" and sell_event["reason"] == "FIXED_HORIZON"

    # Reuse is input-safe: prediction/daily-store fingerprint and clean trade-log
    # hash are both part of the V4.5 horizon checkpoint key.
    key_a = _stable_job_key(5, [outer_row], "source-a", "trades-a")
    assert key_a != _stable_job_key(5, [outer_row], "source-b", "trades-a")
    assert key_a != _stable_job_key(5, [outer_row], "source-a", "trades-b")

    # NONE must honor the fixed-H fallback and never create an early exit.
    none = _overlay_replay(_schedule(dates, 4), flat_ohlc, StopSpec("NONE"), 20, False)
    assert len(none["trades"]) == 1
    assert pd.Timestamp(none["trades"][0]["exit_date"]) == dates[4]
    assert none["trades"][0]["exit_reason"] == "FIXED_HORIZON"

    # Hard stop: an adverse move before minimum hold must not stop; once holding=2,
    # a low through 90 executes exactly at 90 (touch semantics).
    touch_bars = [
        (100, 101, 99, 100),
        (100, 101, 99, 100),
        (100, 101, 80, 100),
        (100, 101, 89, 95),
        (95, 96, 94, 95),
        (95, 96, 94, 95),
    ]
    touch = _overlay_replay(_schedule(dates, 4), _flat_ohlc(dates, touch_bars), hard, 20, False)
    assert touch["trades"][0]["exit_reason"] == "STOP_TOUCH"
    assert pd.Timestamp(touch["trades"][0]["exit_date"]) == dates[3]
    assert abs(float(touch["trades"][0]["sell_notional"]) / float(touch["trades"][0]["buy_notional"]) * 100.0 - 90.0) < 1.0

    # Gap below the threshold executes at the open, not at the unreachable stop.
    gap_bars = list(touch_bars)
    gap_bars[3] = (85, 90, 80, 88)
    gap = _overlay_replay(_schedule(dates, 4), _flat_ohlc(dates, gap_bars), hard, 20, False)
    assert gap["trades"][0]["exit_reason"] == "STOP_GAP"
    assert pd.Timestamp(gap["trades"][0]["exit_date"]) == dates[3]

    # Trailing stop must not use today's high for today's stop. Day 2 can print 200
    # without retroactively raising its own threshold; that high becomes known only
    # for the following session, where 12% trailing gives 176.
    trailing = StopSpec("TR", "TRAILING", trailing_loss_fraction=0.12, minimum_holding_days=1)
    trail_bars = [
        (100, 100, 100, 100),
        (100, 100, 100, 100),
        (100, 200, 95, 190),
        (190, 195, 170, 180),
        (180, 185, 175, 180),
        (180, 185, 175, 180),
    ]
    tr = _overlay_replay(_schedule(dates, 5), _flat_ohlc(dates, trail_bars), trailing, 20, False)
    assert pd.Timestamp(tr["trades"][0]["exit_date"]) == dates[3]
    assert tr["trades"][0]["exit_reason"] == "STOP_TOUCH"

    # H1 cannot be stopped by a V4.5 candidate whose minimum holding period is 2;
    # fixed fallback owns the next-session open.
    h1 = _overlay_replay(_schedule(dates, 2), _flat_ohlc(dates, gap_bars), hard, 20, False)
    assert pd.Timestamp(h1["trades"][0]["exit_date"]) == dates[2]
    assert h1["trades"][0]["exit_reason"] == "FIXED_HORIZON"

    # Performance/telemetry search globals are only inspected here, not rewritten.
    assert search.SEARCH_QUANTILES == (0.90, 0.95, 0.975, 0.99)
    print("H1_30_SUITE_SELF_TEST_PASS")


if __name__ == "__main__":
    main()
