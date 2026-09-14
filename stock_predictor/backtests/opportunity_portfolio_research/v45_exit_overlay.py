from __future__ import annotations

"""Paired V4.5 exit overlay for the H1-H30 clean Development suite.

The clean portfolio policy is selected by the established Opportunity Portfolio
suite. Production overlay replays consume its already-realized OOS trade log as the
authoritative entry/fixed-exit schedule. Each V4.5 candidate receives those exact
clean entry events and may only exit an existing position earlier. No replacement
entry is created by an early stop.

This module is adaptive Development research. It never opens or evaluates the
final holdout and never changes the core 12+4 execution/telemetry implementation.
"""

from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import argparse
import hashlib
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import time
from typing import Any, Iterable

import numpy as np
import pandas as pd

from . import portfolio_replay_process_backend as backend
from . import portfolio_resilient_process_pool as resilient
from .portfolio_policy_contracts import Policy
from .tax_contracts import TaxConfig
from .affinity_coordinator_pool import AffinityCoordinatorPool
from .cpu_topology import set_current_process_logical_affinity
from .portfolio_research_inputs import _manifest_paths
from .portfolio_replay_fragment_cache import fingerprint_data_sources
from .next_open_portfolio_replay import _metrics, prepare_signals
from .german_retail_tax_engine import TaxLedger, benchmark_tax_approximate

OVERLAY_CONTRACT_ID = "H1_30_V45_PAIRED_EXIT_ONLY_V1"
V45_SOURCE_BRANCH = "research/v45-n25-multifactor-search"
V45_SOURCE_CONFIG_SHA = "767d004ae83f441083b2a96209a7c25ddc0fbe04"
REQUIRED_HORIZONS = tuple(range(1, 31))
PRIMARY_SCENARIO = "PRE_TAX_20_BPS"
CLEAN_BASELINE_ROUNDTRIP_BPS = 20.0
SCENARIOS = (
    ("PRE_TAX_20_BPS", 20.0, False),
    ("PRE_TAX_30_BPS", 30.0, False),
    ("PRE_TAX_50_BPS", 50.0, False),
    ("DE_TAX_20_BPS", 20.0, True),
    ("DE_TAX_30_BPS", 30.0, True),
    ("DE_TAX_50_BPS", 50.0, True),
)


@dataclass(frozen=True)
class StopSpec:
    candidate_id: str
    family: str = "NONE"
    hard_loss_fraction: float = 0.0
    trailing_loss_fraction: float = 0.0
    atr_multiple: float = 0.0
    minimum_holding_days: int = 0
    cooldown_days: int = 0
    replacement_rank_limit: int = 38


V45_CANDIDATES = (
    StopSpec("NONE"),
    StopSpec("HARD_10_HOLD2_CD5", "HARD", hard_loss_fraction=0.10, minimum_holding_days=2, cooldown_days=5),
    StopSpec("HARD_12_HOLD5_CD10", "HARD", hard_loss_fraction=0.12, minimum_holding_days=5, cooldown_days=10),
    StopSpec("TRAIL_12_HOLD2_CD5", "TRAILING", trailing_loss_fraction=0.12, minimum_holding_days=2, cooldown_days=5),
    StopSpec("TRAIL_15_HOLD5_CD10", "TRAILING", trailing_loss_fraction=0.15, minimum_holding_days=5, cooldown_days=10),
    StopSpec("ATR_3_HOLD2_CD5", "ATR", atr_multiple=3.0, minimum_holding_days=2, cooldown_days=5),
    StopSpec("ATR_4_HOLD5_CD10", "ATR", atr_multiple=4.0, minimum_holding_days=5, cooldown_days=10),
    StopSpec(
        "COMBINED_12_15_ATR3_HOLD5",
        "COMBINED",
        hard_loss_fraction=0.12,
        trailing_loss_fraction=0.15,
        atr_multiple=3.0,
        minimum_holding_days=5,
        cooldown_days=10,
    ),
)

_WORKER_PREDICTIONS: Path | None = None
_WORKER_DAILY_ROOT: Path | None = None
_WORKER_PATHS: dict[str, Path] = {}
_WORKER_CLEAN_TRADES: pd.DataFrame | None = None
_WORKER_CLEAN_OPEN_POSITIONS: pd.DataFrame | None = None
_WORKER_INDEX = -1
_WORKER_LAST_FINISH = 0.0


def _policy_from_row(row: dict[str, Any]) -> Policy:
    policy = Policy(
        int(row["horizon"]),
        float(row["score_quantile"]),
        float(row["top_fraction"]),
        int(float(row["max_names"])),
        int(float(row["holding_days"])),
        str(row.get("exit_family", "FIXED")),
        float(row.get("exit_value", 0.0)),
        str(row.get("replacement", "IGNORE_NEW")),
        str(row.get("allocation", "EQUAL_ACTIVE")),
        float(row.get("sleeve", 0.50)),
    )
    if policy.horizon != policy.holding_days:
        raise RuntimeError(f"PAIRED_EXIT_REQUIRES_MATCHED_HOLD:H{policy.horizon}:hold={policy.holding_days}")
    if policy.exit_family != "FIXED":
        raise RuntimeError(f"PAIRED_EXIT_REQUIRES_CLEAN_FIXED_EXIT:H{policy.horizon}:{policy.exit_family}")
    if policy.replacement != "IGNORE_NEW":
        raise RuntimeError(f"PAIRED_EXIT_REQUIRES_NO_REPLACEMENT:{policy.replacement}")
    return policy


def _load_horizon_predictions(path: Path, horizon: int) -> pd.DataFrame:
    """Legacy/reference loader retained for parity tests; production uses clean trades."""
    columns = ["decision_date", "ticker", "fold_id", "horizon_sessions", "predicted_net_excess_return", "family"]
    try:
        frame = pd.read_parquet(path, columns=columns, filters=[("horizon_sessions", "==", int(horizon))])
    except Exception:
        frame = pd.read_parquet(path, columns=columns)
        frame = frame.loc[frame["horizon_sessions"].astype(int).eq(int(horizon))].copy()
    frame = frame.loc[~frame["family"].astype(str).str.upper().eq("V2_RANKING")].copy()
    frame["decision_date"] = pd.to_datetime(frame["decision_date"]).dt.normalize()
    frame["horizon"] = frame["horizon_sessions"].astype(int)
    frame["score"] = pd.to_numeric(frame["predicted_net_excess_return"], errors="coerce")
    frame["ticker"] = frame["ticker"].astype(str)
    frame["fold_id"] = frame["fold_id"].astype(str)
    return frame.dropna(subset=["decision_date", "ticker", "fold_id", "score"])[
        ["decision_date", "ticker", "fold_id", "horizon", "score"]
    ]


def _load_ohlc(tickers: Iterable[str]) -> dict[str, dict[str, Any]]:
    if _WORKER_DAILY_ROOT is None:
        raise RuntimeError("V45_WORKER_DAILY_ROOT_NOT_INITIALIZED")
    result: dict[str, dict[str, Any]] = {}
    for ticker in sorted({str(x).upper() for x in tickers} | {"URTH"}):
        path = _WORKER_PATHS.get(ticker)
        if path is None or not path.is_file():
            continue
        frame = pd.read_parquet(path, columns=["session_date", "open", "high", "low", "close"])
        frame["date"] = pd.to_datetime(frame.pop("session_date")).dt.normalize()
        frame = frame.dropna(subset=["date", "open", "high", "low", "close"])
        frame = frame.loc[
            (frame["open"] > 0) & (frame["high"] > 0) & (frame["low"] > 0) & (frame["close"] > 0)
        ].sort_values("date")
        dates = tuple(pd.Timestamp(x) for x in frame["date"])
        rows = {
            pd.Timestamp(r.date): (float(r.open), float(r.high), float(r.low), float(r.close))
            for r in frame.itertuples(index=False)
        }
        previous_close = frame["close"].shift(1)
        tr = np.maximum.reduce([
            (frame["high"] - frame["low"]).to_numpy(dtype=float),
            (frame["high"] - previous_close).abs().fillna(0.0).to_numpy(dtype=float),
            (frame["low"] - previous_close).abs().fillna(0.0).to_numpy(dtype=float),
        ])
        atr = pd.Series(tr, index=frame.index).rolling(14, min_periods=14).mean().fillna(0.0)
        atr_map = {pd.Timestamp(d): float(a) for d, a in zip(frame["date"], atr)}
        result[ticker] = {"dates": dates, "bars": rows, "atr14": atr_map}
    if "URTH" not in result:
        raise RuntimeError("V45_OVERLAY_URTH_MISSING")
    return result


def _position_value(position: dict, bar: tuple[float, float, float, float], use_open: bool) -> float:
    return float(position["qty"]) * float(bar[0 if use_open else 3])


def _curve_metrics(curve: list[dict], trades: list[dict], costs: dict, tax: TaxLedger, tax_config: TaxConfig, initial: float) -> dict:
    frame = pd.DataFrame(curve)
    if frame.empty:
        return {"initial_value": initial, "terminal_value": initial, "trade_count": 0, "cagr_excess": 0.0}
    benchmark_terminal = float(frame["urth_value"].iloc[-1])
    after_tax, benchmark_tax = benchmark_tax_approximate(initial, benchmark_terminal, tax_config)
    return _metrics(
        frame, initial, trades, costs, tax, initial,
        benchmark_terminal_after_tax=after_tax if tax_config.enabled else None,
        benchmark_tax=benchmark_tax,
    )


def _build_clean_schedule(
    fold_signals: pd.DataFrame,
    ohlc: dict[str, dict[str, Any]],
    policy: Policy,
    threshold: float,
    fold_start: pd.Timestamp,
    fold_end: pd.Timestamp,
    initial: float = 10000.0,
) -> dict:
    """Reference reconstruction retained for synthetic parity/regression tests."""
    urth = ohlc["URTH"]["bars"]
    dates = tuple(d for d in ohlc["URTH"]["dates"] if fold_start <= d <= fold_end)
    if len(dates) < 3:
        raise RuntimeError("CLEAN_SCHEDULE_PERIOD_TOO_SHORT")
    idx = {d: i for i, d in enumerate(dates)}
    prepared = prepare_signals(fold_signals)
    by_date = prepared["by_date"]
    pending: dict[pd.Timestamp, list[dict]] = defaultdict(list)
    positions: dict[str, dict] = {}
    events: dict[pd.Timestamp, list[dict]] = defaultdict(list)
    cash = 0.0
    urth_units = initial / urth[dates[0]][3]
    side = 10.0 / 10000.0
    costs = {"transaction_cost_eur": 0.0}
    trades: list[dict] = []
    curve: list[dict] = []

    def components(d: pd.Timestamp, use_open: bool = False):
        u = urth[d][0 if use_open else 3]
        urth_value = urth_units * u
        stock_value = 0.0
        for ticker, position in positions.items():
            bar = ohlc.get(ticker, {}).get("bars", {}).get(d)
            if bar is not None:
                stock_value += _position_value(position, bar, use_open)
        return cash + urth_value + stock_value, cash, urth_value, stock_value

    def buy_urth(notional: float, d: pd.Timestamp) -> None:
        nonlocal cash, urth_units
        take = min(max(0.0, notional), max(0.0, cash))
        if take:
            urth_units += take / urth[d][0]
            cash -= take

    def sell_urth(notional: float, d: pd.Timestamp) -> None:
        nonlocal cash, urth_units
        take = min(max(0.0, notional), max(0.0, urth_units * urth[d][0]))
        if take:
            urth_units -= take / urth[d][0]
            cash += take

    def sell_stock(ticker: str, d: pd.Timestamp) -> None:
        nonlocal cash
        position = positions.pop(ticker, None)
        bar = ohlc.get(ticker, {}).get("bars", {}).get(d)
        if position is None or bar is None:
            return
        px = bar[0]
        gross = float(position["qty"]) * px
        fee = gross * side
        cash += gross - fee
        costs["transaction_cost_eur"] += fee
        excess = px / position["entry_price"] / (urth[d][0] / position["entry_urth"]) - 1.0
        trades.append({
            "ticker": ticker,
            "entry_date": position["entry_date"],
            "exit_date": d,
            "holding_days": idx[d] - position["entry_index"],
            "stock_return": px / position["entry_price"] - 1.0,
            "excess_return": excess,
            "buy_notional": position["buy_notional"],
            "sell_notional": gross,
            "cost_eur": position["buy_fee"] + fee,
            "tax_eur": 0.0,
            "exit_reason": "FIXED_HORIZON",
        })
        events[d].append({"kind": "sell", "ticker": ticker, "reason": "FIXED_HORIZON"})
        buy_urth(max(0.0, cash), d)

    for d in dates:
        actions = pending.pop(d, [])
        for action in actions:
            if action["kind"] == "sell":
                sell_stock(str(action["ticker"]), d)

        buy_actions = [
            action for action in actions
            if action["kind"] == "buy"
            and str(action["ticker"]) not in positions
            and d in ohlc.get(str(action["ticker"]), {}).get("bars", {})
        ]
        if buy_actions:
            equity_open, _, _, stock_open = components(d, True)
            sleeve_capacity = max(0.0, equity_open * policy.sleeve - stock_open)
            requested = (
                [sleeve_capacity / len(buy_actions)] * len(buy_actions)
                if policy.allocation == "EQUAL_ACTIVE"
                else [equity_open * policy.sleeve / max(policy.max_names, 1)] * len(buy_actions)
            )
            for action, requested_notional in zip(buy_actions, requested):
                equity_open, _, _, stock_open = components(d, True)
                remaining = max(0.0, equity_open * policy.sleeve - stock_open)
                target = min(max(0.0, float(requested_notional)), remaining)
                if target <= 0:
                    continue
                sell_urth(target, d)
                ticker = str(action["ticker"])
                px = ohlc[ticker]["bars"][d][0]
                buy_fee = target * side
                buy_notional = max(0.0, target - buy_fee)
                if buy_notional <= 0:
                    buy_urth(max(0.0, cash), d)
                    continue
                qty = buy_notional / px
                cash -= target
                if cash < -1e-7:
                    raise AssertionError(f"CLEAN_NEGATIVE_CASH:{cash}")
                if abs(cash) <= 1e-7:
                    cash = 0.0
                positions[ticker] = {
                    "qty": qty, "entry_price": px, "entry_urth": urth[d][0],
                    "entry_date": d, "entry_index": idx[d], "buy_notional": buy_notional,
                    "buy_fee": buy_fee,
                }
                events[d].append({
                    "kind": "buy", "ticker": ticker, "target_gross": target,
                    "buy_notional_clean_20bps": buy_notional,
                    "score": float(action.get("score", 0.0)),
                })

        current = by_date.get(d)
        valid: list[dict] = []
        if current is not None:
            tickers, scores, neg_scores, group_size = current
            limit = max(1, math.ceil(group_size * policy.top_fraction))
            passing = int(np.searchsorted(neg_scores, -float(threshold), side="right")) if len(neg_scores) else 0
            take = min(limit, passing)
            valid = [{"ticker": str(tickers[i]), "score": float(scores[i])} for i in range(take)]

        for ticker, position in list(positions.items()):
            if d not in ohlc.get(ticker, {}).get("bars", {}):
                continue
            held_close_sessions = idx[d] - int(position["entry_index"]) + 1
            if held_close_sessions >= policy.holding_days and idx[d] + 1 < len(dates):
                pending[dates[idx[d] + 1]].append({"kind": "sell", "ticker": ticker})

        chosen = [row for row in valid if row["ticker"] not in positions][: policy.max_names]
        next_date = dates[idx[d] + 1] if idx[d] + 1 < len(dates) else None
        scheduled_sells = (
            {str(action["ticker"]) for action in pending.get(next_date, []) if action.get("kind") == "sell"}
            if next_date is not None else set()
        )
        effective_positions = max(0, len(positions) - len(scheduled_sells))
        available = max(0, policy.max_names - effective_positions)
        chosen = chosen[:available]
        if next_date is not None:
            for row in chosen:
                pending[next_date].append({"kind": "buy", "ticker": row["ticker"], "score": row["score"]})

        total, cash_value, urth_value, stock_value = components(d, False)
        if total <= 0:
            raise AssertionError(f"CLEAN_NON_POSITIVE_NAV:{d}:{total}")
        curve.append({
            "date": d, "strategy_value": total,
            "urth_value": initial * urth[d][3] / urth[dates[0]][3],
            "positions": len(positions), "stock_exposure": stock_value / total,
            "urth_exposure": urth_value / total, "cash_exposure": cash_value / total,
            "accounting_error_eur": total - (cash_value + urth_value + stock_value), "regime": None,
        })

    metrics = _curve_metrics(curve, trades, costs, TaxLedger(TaxConfig(False)), TaxConfig(False), initial)
    return {"dates": dates, "events": {d: [dict(e) for e in rows] for d, rows in events.items()}, "curve": curve, "trades": trades, "metrics": metrics}


def _schedule_from_clean_trade_log(
    outer_row: dict[str, Any],
    fold_trades: pd.DataFrame,
    fold_open_positions: pd.DataFrame | dict[str, dict[str, Any]],
    ohlc: dict[str, dict[str, Any]] | None = None,
) -> dict:
    """Consume exact realized clean entries/exits instead of rerunning selection."""
    # Keep the historical three-argument helper contract used by the self-test
    # and downstream diagnostics: (outer_row, fold_trades, ohlc).
    if ohlc is None:
        ohlc = fold_open_positions  # type: ignore[assignment]
        fold_open_positions = pd.DataFrame()
    fold_start = pd.Timestamp(outer_row["fold_start"])
    fold_end = pd.Timestamp(outer_row["fold_end"])
    dates = tuple(d for d in ohlc["URTH"]["dates"] if fold_start <= d <= fold_end)
    if len(dates) < 3:
        raise RuntimeError("CLEAN_TRADE_SCHEDULE_PERIOD_TOO_SHORT")
    date_set = set(dates)
    expected = int(outer_row.get("trade_count", 0))
    if len(fold_trades) != expected:
        raise RuntimeError(
            f"CLEAN_TRADE_LOG_COUNT_MISMATCH:H{int(outer_row['horizon'])}:{outer_row['fold_id']}:"
            f"expected={expected}:actual={len(fold_trades)}"
        )
    side = CLEAN_BASELINE_ROUNDTRIP_BPS / 2.0 / 10000.0
    events: dict[pd.Timestamp, list[dict]] = defaultdict(list)
    for row in fold_trades.sort_values(["entry_date", "trade_index"], kind="stable").itertuples(index=False):
        ticker = str(row.ticker)
        entry = pd.Timestamp(row.entry_date)
        exit_date = pd.Timestamp(row.exit_date)
        if entry not in date_set or exit_date not in date_set:
            raise RuntimeError(f"CLEAN_TRADE_EVENT_OUTSIDE_FOLD:{ticker}:{entry}:{exit_date}")
        if entry not in ohlc.get(ticker, {}).get("bars", {}) or exit_date not in ohlc.get(ticker, {}).get("bars", {}):
            raise RuntimeError(f"CLEAN_TRADE_OHLC_MISSING:{ticker}:{entry}:{exit_date}")
        buy_notional = float(row.buy_notional)
        if buy_notional <= 0:
            raise RuntimeError(f"CLEAN_TRADE_NON_POSITIVE_BUY_NOTIONAL:{ticker}:{buy_notional}")
        target_gross = buy_notional / (1.0 - side)
        events[entry].append({
            "kind": "buy",
            "ticker": ticker,
            "target_gross": target_gross,
            "buy_notional_clean_20bps": buy_notional,
            "source": "outer_oos_trade_log.csv",
        })
        events[exit_date].append({
            "kind": "sell",
            "ticker": ticker,
            "reason": "FIXED_HORIZON",
            "source": "outer_oos_trade_log.csv",
        })
    if not fold_open_positions.empty:
        for row in fold_open_positions.sort_values(["entry_date", "ticker"], kind="stable").itertuples(index=False):
            ticker = str(row.ticker)
            entry = pd.Timestamp(row.entry_date)
            if entry not in date_set:
                raise RuntimeError(f"CLEAN_OPEN_POSITION_OUTSIDE_FOLD:{ticker}:{entry}")
            if entry not in ohlc.get(ticker, {}).get("bars", {}):
                raise RuntimeError(f"CLEAN_OPEN_POSITION_OHLC_MISSING:{ticker}:{entry}")
            buy_notional = float(row.buy_notional)
            buy_fee = float(row.buy_fee)
            if buy_notional <= 0 or buy_fee < 0:
                raise RuntimeError(f"CLEAN_OPEN_POSITION_INVALID_NOTIONAL:{ticker}:{buy_notional}:{buy_fee}")
            events[entry].append({
                "kind": "buy", "ticker": ticker,
                "target_gross": buy_notional + buy_fee,
                "buy_notional_clean_20bps": buy_notional,
                "source": "outer_oos_open_positions.csv",
            })
    source = "EXISTING_CLEAN_OUTER_OOS_TRADE_LOG_PLUS_OPEN_POSITIONS" if not fold_open_positions.empty else "EXISTING_CLEAN_OUTER_OOS_TRADE_LOG"
    return {
        "dates": dates,
        "events": {d: [dict(e) for e in rows] for d, rows in events.items()},
        "source_trade_count": int(len(fold_trades)),
        "source": source,
    }


def _stop_threshold(position: dict, spec: StopSpec) -> float | None:
    if spec.family == "NONE" or int(position["holding_days"]) < spec.minimum_holding_days:
        return None
    levels = []
    if spec.family in {"HARD", "COMBINED"}:
        levels.append(float(position["entry_price"]) * (1.0 - spec.hard_loss_fraction))
    if spec.family in {"TRAILING", "COMBINED"}:
        levels.append(float(position["prior_known_high"]) * (1.0 - spec.trailing_loss_fraction))
    if spec.family in {"ATR", "COMBINED"}:
        levels.append(float(position["prior_known_high"]) - spec.atr_multiple * float(position["prior_known_atr"]))
    return max(levels) if levels else None


def _overlay_replay(schedule: dict, ohlc: dict[str, dict[str, Any]], spec: StopSpec, roundtrip_bps: float, tax_enabled: bool, initial: float = 10000.0) -> dict:
    dates: tuple[pd.Timestamp, ...] = schedule["dates"]
    events: dict[pd.Timestamp, list[dict]] = schedule["events"]
    urth = ohlc["URTH"]["bars"]
    idx = {d: i for i, d in enumerate(dates)}
    side = float(roundtrip_bps) / 2.0 / 10000.0
    urth_units = initial / urth[dates[0]][3]
    cash = 0.0
    pending_cash: list[dict] = []
    positions: dict[str, dict] = {}
    tax_config = TaxConfig(bool(tax_enabled))
    tax = TaxLedger(tax_config)
    costs = {"transaction_cost_eur": 0.0}
    trades: list[dict] = []
    curve: list[dict] = []
    early_stops = 0
    funding_shortfalls = 0

    def components(d: pd.Timestamp, use_open: bool = False):
        pending_value = sum(float(item["cash"]) for item in pending_cash)
        u = urth[d][0 if use_open else 3]
        urth_value = urth_units * u
        stock_value = 0.0
        for ticker, position in positions.items():
            bar = ohlc.get(ticker, {}).get("bars", {}).get(d)
            if bar is not None:
                stock_value += _position_value(position, bar, use_open)
        total_cash = cash + pending_value
        return total_cash + urth_value + stock_value, total_cash, urth_value, stock_value

    def buy_urth_all(d: pd.Timestamp) -> None:
        nonlocal cash, urth_units
        if cash > 0:
            urth_units += cash / urth[d][0]
            cash = 0.0

    def sell_urth_for(amount: float, d: pd.Timestamp) -> float:
        nonlocal cash, urth_units
        take = min(max(0.0, amount), max(0.0, urth_units * urth[d][0]))
        if take:
            urth_units -= take / urth[d][0]
            cash += take
        return take

    def close_position(ticker: str, d: pd.Timestamp, price: float, reason: str, delayed_reinvestment: bool) -> None:
        nonlocal cash, early_stops
        position = positions.pop(ticker, None)
        if position is None:
            return
        gross = float(position["qty"]) * float(price)
        fee = gross * side
        costs["transaction_cost_eur"] += fee
        tax_paid = tax.realize_stock_trade(d.date(), gross, float(position["tax_basis"]), fee)
        proceeds = gross - fee - tax_paid
        excess = float(price) / float(position["entry_price"]) / (urth[d][0] / float(position["entry_urth"])) - 1.0
        trades.append({
            "ticker": ticker, "entry_date": position["entry_date"], "exit_date": d,
            "holding_days": idx[d] - int(position["entry_index"]),
            "stock_return": float(price) / float(position["entry_price"]) - 1.0,
            "excess_return": excess, "buy_notional": float(position["buy_notional"]),
            "sell_notional": gross, "cost_eur": float(position["buy_fee"]) + fee,
            "tax_eur": tax_paid, "exit_reason": reason,
            "early_exit": bool(reason.startswith("STOP_")), "v45_candidate_id": spec.candidate_id,
        })
        if delayed_reinvestment:
            pending_cash.append({"cash": proceeds, "earliest_index": idx[d] + 1})
            early_stops += 1
        else:
            cash += proceeds
            buy_urth_all(d)

    for d in dates:
        session_index = idx[d]
        matured = [item for item in pending_cash if int(item["earliest_index"]) <= session_index]
        if matured:
            cash += sum(float(item["cash"]) for item in matured)
            pending_cash[:] = [item for item in pending_cash if int(item["earliest_index"]) > session_index]
            buy_urth_all(d)

        day_events = list(events.get(d, []))
        fixed_sells = {str(e["ticker"]) for e in day_events if e.get("kind") == "sell"}
        for ticker in list(fixed_sells):
            if ticker in positions:
                close_position(ticker, d, ohlc[ticker]["bars"][d][0], "FIXED_HORIZON", False)

        for ticker in list(positions):
            if ticker in fixed_sells:
                continue
            bar = ohlc.get(ticker, {}).get("bars", {}).get(d)
            if bar is None:
                continue
            position = positions[ticker]
            level = _stop_threshold(position, spec)
            if level is None:
                continue
            execution = None
            reason = None
            if bar[0] <= level:
                execution, reason = bar[0], "STOP_GAP"
            elif bar[2] <= level:
                execution, reason = level, "STOP_TOUCH"
            if execution is not None:
                close_position(ticker, d, execution, str(reason), True)

        for event in day_events:
            if event.get("kind") != "buy":
                continue
            ticker = str(event["ticker"])
            if ticker in positions:
                raise RuntimeError(f"PAIRED_DUPLICATE_LIVE_ENTRY:{d}:{ticker}")
            bar = ohlc.get(ticker, {}).get("bars", {}).get(d)
            if bar is None:
                raise RuntimeError(f"PAIRED_ENTRY_BAR_MISSING:{d}:{ticker}")
            target = float(event["target_gross"])
            liquid = cash + urth_units * urth[d][0]
            effective_target = min(target, max(0.0, liquid))
            if effective_target + 1e-7 < target:
                funding_shortfalls += 1
            if cash < effective_target:
                sell_urth_for(effective_target - cash, d)
            fee = effective_target * side
            buy_notional = max(0.0, effective_target - fee)
            if buy_notional <= 0:
                continue
            px = bar[0]
            qty = buy_notional / px
            cash -= effective_target
            if cash < -1e-7:
                raise AssertionError(f"V45_NEGATIVE_CASH_AFTER_BUY:{cash}")
            if abs(cash) <= 1e-7:
                cash = 0.0
            costs["transaction_cost_eur"] += fee
            positions[ticker] = {
                "qty": qty, "entry_price": px, "entry_urth": urth[d][0],
                "entry_date": d, "entry_index": session_index, "holding_days": 0,
                "prior_known_high": px, "prior_known_atr": 0.0,
                "buy_notional": buy_notional, "buy_fee": fee, "tax_basis": buy_notional + fee,
            }

        for ticker, position in positions.items():
            bar = ohlc[ticker]["bars"].get(d)
            if bar is None:
                continue
            position["holding_days"] = int(position["holding_days"]) + 1
            position["prior_known_high"] = max(float(position["prior_known_high"]), float(bar[1]))
            position["prior_known_atr"] = float(ohlc[ticker]["atr14"].get(d, 0.0))

        total, cash_value, urth_value, stock_value = components(d, False)
        if total <= 0:
            raise AssertionError(f"V45_NON_POSITIVE_NAV:{d}:{total}")
        curve.append({
            "date": d, "strategy_value": total,
            "urth_value": initial * urth[d][3] / urth[dates[0]][3],
            "positions": len(positions), "stock_exposure": stock_value / total,
            "urth_exposure": urth_value / total, "cash_exposure": cash_value / total,
            "accounting_error_eur": total - (cash_value + urth_value + stock_value), "regime": None,
        })

    metrics = _curve_metrics(curve, trades, costs, tax, tax_config, initial)
    metrics.update({
        "tax_world": "DE_RETAIL_TAX_AWARE" if tax_enabled else "PRE_TAX",
        "v45_candidate_id": spec.candidate_id, "v45_family": spec.family,
        "early_stop_count": int(early_stops), "funding_shortfall_count": int(funding_shortfalls),
        "paired_entries_no_stop_replacements": True,
    })
    return {"metrics": metrics, "trades": trades, "curve": curve}


def _official_consistency(clean_metrics: dict, outer_row: dict[str, Any]) -> dict:
    terminal_error = abs(float(clean_metrics.get("terminal_value", 0.0)) - float(outer_row.get("terminal_value", 0.0)))
    cagr_error = abs(float(clean_metrics.get("cagr_excess", 0.0)) - float(outer_row.get("cagr_excess", 0.0)))
    trade_match = int(clean_metrics.get("trade_count", 0)) == int(outer_row.get("trade_count", 0))
    return {
        "terminal_value_abs_error": terminal_error,
        "cagr_excess_abs_error": cagr_error,
        "trade_count_match": trade_match,
        "pass": bool(terminal_error <= 1e-5 and cagr_error <= 1e-9 and trade_match),
    }


def _worker_init(predictions_path: str, daily_root: str, clean_trade_log_path: str, clean_open_positions_path: str, worker_map: list[dict], slot_counter) -> None:
    global _WORKER_PREDICTIONS, _WORKER_DAILY_ROOT, _WORKER_PATHS, _WORKER_CLEAN_TRADES, _WORKER_CLEAN_OPEN_POSITIONS, _WORKER_INDEX, _WORKER_LAST_FINISH
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS"):
        os.environ[name] = "1"
    if worker_map and slot_counter is not None:
        with slot_counter.get_lock():
            slot = int(slot_counter.value)
            slot_counter.value += 1
        _WORKER_INDEX = slot % len(worker_map)
        row = worker_map[_WORKER_INDEX]
        ok = set_current_process_logical_affinity(int(row["logical_processor"]))
        print(
            "[v45-exit-worker] "
            f"pid={os.getpid()} slot={slot} core={row['core_index']} logical_cpu={row['logical_processor']} "
            f"affinity={'OK' if ok else 'FALLBACK'}", flush=True,
        )
    _WORKER_PREDICTIONS = Path(predictions_path)
    _WORKER_DAILY_ROOT = Path(daily_root)
    _WORKER_PATHS = _manifest_paths(_WORKER_DAILY_ROOT)
    try:
        clean_trades = pd.read_csv(clean_trade_log_path)
    except pd.errors.EmptyDataError:
        clean_trades = pd.DataFrame(columns=[
            "horizon", "fold_id", "trade_index", "ticker", "entry_date", "exit_date", "buy_notional"
        ])
    required = {"horizon", "fold_id", "trade_index", "ticker", "entry_date", "exit_date", "buy_notional"}
    missing = required - set(clean_trades.columns)
    if missing:
        raise RuntimeError(f"CLEAN_TRADE_LOG_COLUMNS_MISSING:{sorted(missing)}")
    clean_trades["horizon"] = clean_trades["horizon"].astype(int)
    clean_trades["fold_id"] = clean_trades["fold_id"].astype(str)
    clean_trades["ticker"] = clean_trades["ticker"].astype(str)
    clean_trades["entry_date"] = pd.to_datetime(clean_trades["entry_date"]).dt.normalize()
    clean_trades["exit_date"] = pd.to_datetime(clean_trades["exit_date"]).dt.normalize()
    _WORKER_CLEAN_TRADES = clean_trades
    try:
        _WORKER_CLEAN_OPEN_POSITIONS = pd.read_csv(clean_open_positions_path)
    except FileNotFoundError as exc:
        raise RuntimeError(
            "V45_CLEAN_INPUT_MISSING: outer_oos_open_positions.csv is required "
            f"to reproduce terminal holdings. Expected={clean_open_positions_path}"
        ) from exc
    _WORKER_LAST_FINISH = time.perf_counter()


def _horizon_worker(job: dict) -> dict:
    global _WORKER_LAST_FINISH
    idle_started = _WORKER_LAST_FINISH or time.perf_counter()
    started = time.perf_counter()
    horizon = int(job["horizon"])
    if _WORKER_CLEAN_TRADES is None:
        raise RuntimeError("V45_WORKER_CLEAN_TRADES_NOT_INITIALIZED")
    if _WORKER_CLEAN_OPEN_POSITIONS is None:
        raise RuntimeError("V45_WORKER_CLEAN_OPEN_POSITIONS_NOT_INITIALIZED")
    horizon_trades = _WORKER_CLEAN_TRADES.loc[_WORKER_CLEAN_TRADES["horizon"].eq(horizon)].copy()
    ohlc = _load_ohlc(horizon_trades["ticker"].unique() if not horizon_trades.empty else [])
    fold_results: list[dict] = []
    trade_results: list[dict] = []
    consistency: list[dict] = []

    for outer_row in job["outer_rows"]:
        fold_id = str(outer_row["fold_id"])
        _policy_from_row(outer_row)  # fail closed on clean fixed-H / no-replacement contract
        fold_trades = horizon_trades.loc[horizon_trades["fold_id"].eq(fold_id)].copy()
        horizon_open_positions = _WORKER_CLEAN_OPEN_POSITIONS.loc[
            _WORKER_CLEAN_OPEN_POSITIONS["horizon"].astype(int).eq(horizon)
            & _WORKER_CLEAN_OPEN_POSITIONS["fold_id"].astype(str).eq(fold_id)
        ].copy()
        schedule = _schedule_from_clean_trade_log(outer_row, fold_trades, horizon_open_positions, ohlc)

        for scenario, bps, tax_enabled in SCENARIOS:
            scenario_rows = []
            for spec in V45_CANDIDATES:
                replayed = _overlay_replay(schedule, ohlc, spec, bps, tax_enabled)
                metrics = dict(replayed["metrics"])
                row = {
                    "horizon": horizon, "fold_id": fold_id,
                    "fold_start": str(outer_row["fold_start"]), "fold_end": str(outer_row["fold_end"]),
                    "policy_id": str(outer_row["policy_id"]), "scenario": scenario,
                    "roundtrip_bps": bps, "tax_enabled": tax_enabled,
                    "candidate_id": spec.candidate_id, "family": spec.family, **metrics,
                }
                scenario_rows.append(row)
                for trade in replayed["trades"]:
                    trade_results.append({
                        **trade, "horizon": horizon, "fold_id": fold_id,
                        "scenario": scenario, "candidate_id": spec.candidate_id, "family": spec.family,
                    })
            none_row = next(row for row in scenario_rows if row["candidate_id"] == "NONE")
            for row in scenario_rows:
                row["cagr_excess_delta_vs_none"] = float(row["cagr_excess"]) - float(none_row["cagr_excess"])
                row["terminal_value_delta_vs_none"] = float(row["terminal_value"]) - float(none_row["terminal_value"])
                row["max_drawdown_improvement_vs_none"] = float(row["max_drawdown"]) - float(none_row["max_drawdown"])
            if scenario == PRIMARY_SCENARIO:
                official = _official_consistency(none_row, outer_row)
                consistency.append({
                    "horizon": horizon,
                    "fold_id": fold_id,
                    "clean_schedule_source": schedule["source"],
                    **official,
                })
                if not official["pass"]:
                    raise RuntimeError(
                        f"V45_NONE_EXISTING_CLEAN_REPLAY_MISMATCH:H{horizon}:{fold_id}:"
                        f"terminal={official['terminal_value_abs_error']}:cagr={official['cagr_excess_abs_error']}:"
                        f"trade_match={official['trade_count_match']}"
                    )
            fold_results.extend(scenario_rows)

    finished = time.perf_counter()
    result = {
        "job_key": str(job["job_key"]), "horizon": horizon,
        "fold_results": fold_results, "trade_results": trade_results, "consistency": consistency,
        "_worker_telemetry": {
            "worker_index": int(_WORKER_INDEX),
            "compute_seconds": max(0.0, finished - started),
            "idle_seconds_before_job": max(0.0, started - idle_started),
        },
    }
    _WORKER_LAST_FINISH = finished
    return result


def _stable_job_key(
    horizon: int,
    outer_rows: list[dict],
    source_fingerprint: str,
    clean_trade_log_sha256: str,
) -> str:
    payload = {
        "contract": OVERLAY_CONTRACT_ID,
        "v45_config_sha": V45_SOURCE_CONFIG_SHA,
        "source_fingerprint": str(source_fingerprint),
        "clean_trade_log_sha256": str(clean_trade_log_sha256),
        "horizon": int(horizon),
        "outer": [{
            "fold_id": str(row["fold_id"]), "policy_id": str(row["policy_id"]),
            "threshold": float(row["threshold"]), "start": str(row["fold_start"]), "end": str(row["fold_end"]),
            "terminal_value": float(row.get("terminal_value", 0.0)), "trade_count": int(row.get("trade_count", 0)),
        } for row in outer_rows],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _load_checkpoint(path: Path) -> dict[str, dict]:
    out = {}
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = str(row.get("job_key", ""))
            if key:
                out[key] = row
    return out


def _append_checkpoint(path: Path, payload: dict) -> None:
    clean = dict(payload)
    clean.pop("_worker_telemetry", None)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(clean, sort_keys=True, default=str) + "\n")
        handle.flush()


def _coordinator_job(item: tuple[int, list[dict], str, str]) -> dict:
    horizon, rows, source_fingerprint, clean_trade_log_sha256 = item
    return {
        "horizon": int(horizon),
        "outer_rows": rows,
        "job_key": _stable_job_key(horizon, rows, source_fingerprint, clean_trade_log_sha256),
    }


def _summarize(fold_frame: pd.DataFrame, trade_frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if fold_frame.empty:
        return pd.DataFrame(), pd.DataFrame()
    primary = fold_frame.loc[fold_frame["scenario"].eq(PRIMARY_SCENARIO)].copy()
    summaries = []
    for (horizon, candidate), group in primary.groupby(["horizon", "candidate_id"], sort=True):
        deltas = group["cagr_excess_delta_vs_none"].astype(float).to_numpy()
        summaries.append({
            "horizon": int(horizon), "candidate_id": str(candidate), "family": str(group["family"].iloc[0]),
            "outer_folds": int(len(group)),
            "median_cagr_excess": float(np.median(group["cagr_excess"].astype(float))),
            "q25_cagr_excess": float(np.quantile(group["cagr_excess"].astype(float), 0.25)),
            "worst_cagr_excess": float(group["cagr_excess"].astype(float).min()),
            "median_cagr_excess_delta_vs_none": float(np.median(deltas)),
            "q25_cagr_excess_delta_vs_none": float(np.quantile(deltas, 0.25)),
            "worst_cagr_excess_delta_vs_none": float(np.min(deltas)),
            "positive_delta_fold_fraction": float((deltas > 0).mean()),
            "median_drawdown_improvement_vs_none": float(np.median(group["max_drawdown_improvement_vs_none"].astype(float))),
            "early_stop_count": int(group["early_stop_count"].sum()),
            "funding_shortfall_count": int(group["funding_shortfall_count"].sum()),
            "adaptive_development_only": True,
        })
    summary = pd.DataFrame(summaries)
    ranked = []
    for horizon, group in summary.groupby("horizon", sort=True):
        non_none = group.loc[~group["candidate_id"].eq("NONE")].copy()
        pool = non_none if not non_none.empty else group.copy()
        pool = pool.sort_values(
            ["median_cagr_excess_delta_vs_none", "q25_cagr_excess_delta_vs_none", "worst_cagr_excess_delta_vs_none", "median_drawdown_improvement_vs_none"],
            ascending=[False, False, False, False],
        )
        best = pool.iloc[0].to_dict()
        best["selection_status"] = (
            "V45_EXIT_CANDIDATE_POSITIVE_DEVELOPMENT_DELTA"
            if float(best["median_cagr_excess_delta_vs_none"]) > 0 else "NO_POSITIVE_V45_EXIT_DELTA"
        )
        ranked.append(best)
    best_frame = pd.DataFrame(ranked)

    if not trade_frame.empty and not best_frame.empty:
        primary_trades = trade_frame.loc[trade_frame["scenario"].eq(PRIMARY_SCENARIO)].copy()
        stop_trades = primary_trades.loc[primary_trades["early_exit"].astype(bool)]
        timing = []
        for (horizon, candidate), group in stop_trades.groupby(["horizon", "candidate_id"], sort=True):
            timing.append({
                "horizon": int(horizon), "candidate_id": str(candidate), "stop_exits": int(len(group)),
                "median_exit_session": float(np.median(group["holding_days"].astype(float))),
                "mean_exit_session": float(group["holding_days"].astype(float).mean()),
                "gap_stop_fraction": float(group["exit_reason"].eq("STOP_GAP").mean()),
                "touch_stop_fraction": float(group["exit_reason"].eq("STOP_TOUCH").mean()),
            })
        timing_frame = pd.DataFrame(timing)
        if not timing_frame.empty:
            best_frame = best_frame.merge(timing_frame, on=["horizon", "candidate_id"], how="left")
    return summary, best_frame


def run_overlay(
    *, predictions_path: Path, daily_store_root: Path, clean_output_root: Path, output_root: Path,
    process_workers: int = 12, coordinator_threads: int = 4, telemetry_path: Path | None = None,
) -> dict:
    outer_path = clean_output_root / "outer_fold_results.csv"
    consistency_path = clean_output_root / "outer_replay_consistency.csv"
    clean_trade_log_path = clean_output_root / "outer_oos_trade_log.csv"
    clean_open_positions_path = clean_output_root / "outer_oos_open_positions.csv"
    if not outer_path.exists():
        raise RuntimeError(
            "V45_CLEAN_INPUT_MISSING: outer_fold_results.csv not found. "
            f"Expected={outer_path}; CleanRootExists={clean_output_root.exists()}"
        )
    if not clean_trade_log_path.exists():
        available = sorted(path.name for path in clean_output_root.glob("*.csv"))
        raise RuntimeError(
            "V45_CLEAN_INPUT_MISSING: outer_oos_trade_log.csv was not created by "
            "the required post-selection evidence expansion. "
            f"Expected={clean_trade_log_path}; AvailableCsv={available}; "
            "Run evidence_expansion before starting the V4.5 overlay."
        )
    if not clean_open_positions_path.exists():
        raise RuntimeError(
            "V45_CLEAN_INPUT_MISSING: outer_oos_open_positions.csv is required to "
            f"reproduce terminal holdings. Expected={clean_open_positions_path}"
        )
    if consistency_path.exists():
        previous = pd.read_csv(consistency_path)
        consistency_column = "consistent" if "consistent" in previous.columns else "pass" if "pass" in previous.columns else None
        if consistency_column is None:
            raise RuntimeError(
                "V45_CLEAN_INPUT_SCHEMA_MISMATCH: outer_replay_consistency.csv must contain "
                f"'consistent' (actual_columns={list(previous.columns)})"
            )
        if not previous.empty and not previous[consistency_column].astype(bool).all():
            raise RuntimeError(
                "CLEAN_EVIDENCE_BASELINE_CONSISTENCY_FAILED: "
                f"column={consistency_column}"
            )
    outer = pd.read_csv(outer_path)
    if outer.empty:
        raise RuntimeError("H1_30_CLEAN_OUTER_RESULTS_EMPTY")
    horizons = sorted({int(x) for x in outer["horizon"]})
    if horizons != list(REQUIRED_HORIZONS):
        raise RuntimeError(
            f"H1_30_OUTER_HORIZON_COVERAGE_INCOMPLETE:expected={list(REQUIRED_HORIZONS)}:actual={horizons}"
        )

    source_audit = fingerprint_data_sources(predictions_path, daily_store_root)
    source_fingerprint = str(source_audit["fingerprint"])
    clean_trade_log_sha256 = hashlib.sha256(clean_trade_log_path.read_bytes()).hexdigest()
    clean_open_positions_sha256 = hashlib.sha256(clean_open_positions_path.read_bytes()).hexdigest()
    source_audit = dict(source_audit) | {
        "clean_trade_log": str(clean_trade_log_path),
        "clean_trade_log_sha256": clean_trade_log_sha256,
        "clean_open_positions": str(clean_open_positions_path),
        "clean_open_positions_sha256": clean_open_positions_sha256,
    }

    output_root.mkdir(parents=True, exist_ok=True)
    if telemetry_path is None:
        telemetry_path = clean_output_root / "multicore_telemetry.jsonl"
    os.environ["OPPORTUNITY_TELEMETRY_PATH"] = str(telemetry_path)
    process_workers = max(1, min(int(process_workers), 12))
    coordinator_threads = max(1, min(int(coordinator_threads), 4))
    backend.install_multicore_backend(process_workers)
    affinity = dict(backend.backend_stats().get("affinity") or {})
    worker_map = list(affinity.get("worker_map") or [])[:process_workers]

    grouped = [
        (
            int(h),
            g.sort_values(["fold_start", "fold_id"]).to_dict("records"),
            source_fingerprint,
            f"{clean_trade_log_sha256}:{clean_open_positions_sha256}",
        )
        for h, g in outer.groupby("horizon", sort=True)
    ]
    with AffinityCoordinatorPool(max_workers=coordinator_threads, thread_name_prefix="v45-exit-coordinator") as coordinators:
        jobs = list(coordinators.map(_coordinator_job, grouped))

    checkpoint_path = output_root / "v45_exit_checkpoint.jsonl"
    existing = _load_checkpoint(checkpoint_path)
    reused = [existing[job["job_key"]] for job in jobs if job["job_key"] in existing]
    pending = [job for job in jobs if job["job_key"] not in existing]
    results = list(reused)

    try:
        if pending:
            ctx = mp.get_context("spawn")
            slot_counter = ctx.Value("i", 0, lock=True)

            def pool_factory(workers: int):
                current_worker_map = list(worker_map)[: max(1, int(workers))]
                return ProcessPoolExecutor(
                    max_workers=max(1, int(workers)),
                    mp_context=ctx,
                    initializer=_worker_init,
                    initargs=(
                        str(predictions_path),
                        str(daily_store_root),
                        str(clean_trade_log_path),
                        str(clean_open_positions_path),
                        current_worker_map,
                        slot_counter,
                    ),
                )

            def commit(_index: int, payload: dict) -> None:
                _append_checkpoint(checkpoint_path, payload)
                results.append(dict(payload))

            # Same retry/degradation/checkpoint semantics as clean and Evidence:
            # first retry keeps 12, repeated crashes degrade 12 -> 10 -> 8.
            resilient._RECOVERY_COUNT = 0
            resilient.resilient_run_external_process_futures(
                pool_factory,
                process_workers,
                list(enumerate(pending)),
                _horizon_worker,
                "H1_30_V45_EXIT_OVERLAY",
                commit,
                jobs_reused=len(reused),
                telemetry_extra={"research_contract": OVERLAY_CONTRACT_ID},
            )
        else:
            backend._write_telemetry({
                "batch": "H1_30_V45_EXIT_OVERLAY",
                "jobs_submitted": 0,
                "jobs_reused": int(len(reused)),
                "jobs_computed": 0,
                "pool_recoveries": 0,
                "research_contract": OVERLAY_CONTRACT_ID,
            })
    finally:
        backend.shutdown_multicore_backend()

    result_map = {int(row["horizon"]): row for row in results}
    ordered = [result_map[h] for h in sorted(result_map)]
    fold_rows = [row for result in ordered for row in result.get("fold_results", [])]
    trade_rows = [row for result in ordered for row in result.get("trade_results", [])]
    consistency_rows = [row for result in ordered for row in result.get("consistency", [])]
    fold_frame = pd.DataFrame(fold_rows)
    trade_frame = pd.DataFrame(trade_rows)
    consistency_frame = pd.DataFrame(consistency_rows)
    summary_frame, best_frame = _summarize(fold_frame, trade_frame)

    fold_frame.to_csv(output_root / "v45_exit_fold_results.csv", index=False)
    trade_frame.to_csv(output_root / "v45_exit_trade_results.csv", index=False)
    consistency_frame.to_csv(output_root / "v45_exit_clean_reconstruction.csv", index=False)
    summary_frame.to_csv(output_root / "v45_exit_horizon_summary.csv", index=False)
    best_frame.to_csv(output_root / "v45_exit_best_development_by_horizon.csv", index=False)
    surface = best_frame.copy()
    if not surface.empty:
        surface["clean_control"] = "NONE"
        surface["paired_entry_policy"] = "EXACT_EXISTING_CLEAN_OOS_TRADE_EVENTS"
        surface["fixed_fallback_exit"] = "HORIZON_H"
        surface["v45_may_exit_early_only"] = True
        surface["cooldown_replacement_disabled"] = True
        surface["adaptive_development_only"] = True
    surface.to_csv(output_root / "h1_30_clean_vs_v45_exit_surface.csv", index=False)

    consistency_pass = not consistency_frame.empty and consistency_frame["pass"].astype(bool).all()
    payload = {
        "status": "H1_30_V45_EXIT_OVERLAY_COMPLETE" if sorted(result_map) == list(REQUIRED_HORIZONS) and consistency_pass else "H1_30_V45_EXIT_OVERLAY_INCOMPLETE",
        "contract": OVERLAY_CONTRACT_ID, "adaptive_development_only": True, "final_holdout": "CLOSED",
        "v45_source_branch": V45_SOURCE_BRANCH, "v45_source_config_sha": V45_SOURCE_CONFIG_SHA,
        "historical_v45_active_control": "NONE", "candidate_count_including_none": len(V45_CANDIDATES),
        "horizons_requested": list(REQUIRED_HORIZONS), "horizons_completed": sorted(result_map),
        "clean_reconstruction_pass": bool(consistency_pass),
        "clean_schedule_source": "EXISTING_CLEAN_OUTER_OOS_TRADE_LOG_PLUS_OPEN_POSITIONS",
        "source_fingerprint": source_audit,
        "process_workers": process_workers, "coordinator_threads": coordinator_threads, "native_threads_per_worker": 1,
        "telemetry_path": str(telemetry_path),
        "paired_design": {
            "entries": "EXACT_EXISTING_CLEAN_OOS_ENTRY_EVENTS_AND_TARGET_GROSS_NOTIONAL",
            "clean_exit": "EXISTING_CLEAN_FIXED_EXIT_HORIZON_H_NEXT_OPEN",
            "v45_exit": "EARLY_ONLY", "replacement_after_stop": False,
            "stop_proceeds_reinvested_in_urth": "T_PLUS_1_OPEN",
            "cooldown_and_replacement_rank": "NOT_APPLIED_BECAUSE_ENTRY_SCHEDULE_IS_FROZEN",
            "current_day_high_used_for_same_day_stop": False,
        },
        "artifacts": [
            "v45_exit_fold_results.csv", "v45_exit_trade_results.csv", "v45_exit_clean_reconstruction.csv",
            "v45_exit_horizon_summary.csv", "v45_exit_best_development_by_horizon.csv", "h1_30_clean_vs_v45_exit_surface.csv",
        ],
    }
    (output_root / "v45_exit_summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v5-predictions", required=True)
    parser.add_argument("--daily-store-root", required=True)
    parser.add_argument("--clean-output-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--max-workers", type=int, default=12)
    parser.add_argument("--coordinator-threads", type=int, default=4)
    parser.add_argument("--telemetry-path", default="")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    summary = run_overlay(
        predictions_path=Path(args.v5_predictions), daily_store_root=Path(args.daily_store_root),
        clean_output_root=Path(args.clean_output_root), output_root=Path(args.output_root),
        process_workers=args.max_workers, coordinator_threads=args.coordinator_threads,
        telemetry_path=Path(args.telemetry_path) if args.telemetry_path else None,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["status"] == "H1_30_V45_EXIT_OVERLAY_COMPLETE" else 2


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())
