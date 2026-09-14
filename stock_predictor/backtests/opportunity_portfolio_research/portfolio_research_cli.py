from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import asdict, replace
from datetime import date
from pathlib import Path
from statistics import median
import pandas as pd
import numpy as np

from .contract_fingerprints import stable_hash
from .portfolio_policy_contracts import CAPITALS, Policy
from .cost_contracts import CostModel
from .tax_contracts import TaxConfig
from .market_regime_contracts import RegimeContract
from .portfolio_research_inputs import load_predictions, load_price_panel
from .next_open_portfolio_replay import replay
from .urth_market_regime_classifier import classify_urth
from .portfolio_policy_search import (
    SEARCH_HOLDING_DAYS,
    SEARCH_MAX_NAMES,
    SEARCH_QUANTILES,
    SEARCH_TOP_FRACTIONS,
    _balanced_budget,
    _resolved_threshold,
    grid,
    run_walk_forward,
)
from .german_retail_tax_engine import TaxLedger

DEFAULT_MAX_WORKERS = 2
HARD_MAX_WORKERS = 4


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def write_csv(path: Path, rows) -> None:
    if isinstance(rows, pd.DataFrame):
        rows.to_csv(path, index=False)
    else:
        pd.DataFrame(rows).to_csv(path, index=False)


def policy_from(row: dict) -> Policy:
    return Policy(
        int(row["horizon"]),
        float(row["score_quantile"]),
        float(row["top_fraction"]),
        int(row["max_names"]),
        int(row["holding_days"]),
        str(row.get("exit_family", "FIXED")),
        float(row.get("exit_value", 0.0)),
        str(row.get("replacement", "IGNORE_NEW")),
        str(row.get("allocation", "EQUAL_ACTIVE")),
        float(row.get("sleeve", .5)),
    )


def _normalize_workers(requested: int) -> int:
    return min(HARD_MAX_WORKERS, max(1, int(requested)))


def _flat_fixture(periods: int = 12, tickers=("AAA",)) -> tuple[pd.DatetimeIndex, pd.DataFrame, pd.DataFrame]:
    dates = pd.date_range("2020-01-01", periods=periods, freq="B")
    frames = [pd.DataFrame({"date": dates, "ticker": "URTH", "open": 100.0, "close": 100.0})]
    for ticker in tickers:
        frames.append(pd.DataFrame({"date": dates, "ticker": ticker, "open": 100.0, "close": 100.0}))
    prices = pd.concat(frames, ignore_index=True)
    signal_rows = []
    for d in dates[:-1]:
        for rank, ticker in enumerate(tickers):
            signal_rows.append({"decision_date": d, "ticker": ticker, "fold_id": "WF", "horizon": 5, "score": 10.0 - rank})
    return dates, prices, pd.DataFrame(signal_rows)


def _outer_fold_summary(rows: list[dict]) -> list[dict]:
    result = []
    by_h = defaultdict(list)
    for row in rows:
        by_h[int(row["horizon"])].append(row)
    for h, vals in sorted(by_h.items()):
        excess = [float(x.get("cagr_excess", 0.0)) for x in vals]
        result.append({
            "horizon": h,
            "folds": len(vals),
            "positive_folds": sum(x > 0 for x in excess),
            "positive_fold_fraction": sum(x > 0 for x in excess) / max(1, len(excess)),
            "median_cagr_excess": float(median(excess)) if excess else 0.0,
            "q25_cagr_excess": float(np.quantile(excess, .25)) if excess else 0.0,
            "worst_cagr_excess": float(min(excess)) if excess else 0.0,
            "trade_count": int(sum(int(x.get("trade_count", 0)) for x in vals)),
        })
    return result


def _regime_attribution(curve: pd.DataFrame, horizon: int, tax_world: str) -> list[dict]:
    if curve.empty:
        return []
    c = curve.sort_values("date").copy()
    c["strategy_daily_return"] = c["strategy_value"].pct_change()
    c["urth_daily_return"] = c["urth_value"].pct_change()
    c = c.dropna(subset=["strategy_daily_return", "urth_daily_return"])
    rows = []
    for regime_name, g in c.groupby("regime", dropna=False):
        sr = g["strategy_daily_return"].astype(float)
        ur = g["urth_daily_return"].astype(float)
        strategy_compounded = float((1.0 + sr).prod() - 1.0)
        urth_compounded = float((1.0 + ur).prod() - 1.0)
        rows.append({
            "horizon": int(horizon),
            "tax_world": tax_world,
            "regime": str(regime_name),
            "observations": int(len(g)),
            "active_days": int((g["positions"] > 0).sum()),
            "strategy_compounded_return_on_regime_days": strategy_compounded,
            "urth_compounded_return_on_regime_days": urth_compounded,
            "compounded_return_excess": strategy_compounded - urth_compounded,
            "mean_daily_excess": float((sr - ur).mean()) if len(g) else 0.0,
        })
    return rows


def _adjacent(values: tuple, current) -> list:
    vals = list(values)
    if current not in vals:
        return []
    i = vals.index(current)
    out = []
    if i > 0:
        out.append(vals[i - 1])
    if i + 1 < len(vals):
        out.append(vals[i + 1])
    return out


def _policy_neighbors(policy: Policy) -> list[tuple[str, Policy]]:
    neighbors = []
    for v in _adjacent(SEARCH_QUANTILES, policy.score_quantile):
        neighbors.append(("score_quantile", replace(policy, score_quantile=v)))
    for v in _adjacent(SEARCH_TOP_FRACTIONS, policy.top_fraction):
        neighbors.append(("top_fraction", replace(policy, top_fraction=v)))
    for v in _adjacent(SEARCH_MAX_NAMES, policy.max_names):
        neighbors.append(("max_names", replace(policy, max_names=v)))
    for v in _adjacent(SEARCH_HOLDING_DAYS, policy.holding_days):
        neighbors.append(("holding_days", replace(policy, holding_days=v)))
    unique = {}
    for dim, p in neighbors:
        unique[p.policy_id] = (dim, p)
    return list(unique.values())


def _parameter_plateau(
    predictions: pd.DataFrame,
    prices: pd.DataFrame,
    final_policies: dict,
    final_thresholds: dict,
    baseline_metrics: dict,
    full_start: pd.Timestamp,
    full_end: pd.Timestamp,
) -> tuple[list[dict], dict]:
    rows = []
    gates = {}
    for h, center in final_policies.items():
        sig = predictions.loc[predictions["horizon"].eq(h)]
        center_excess = float(baseline_metrics.get(h, {}).get("cagr_excess", 0.0))
        center_threshold = float(final_thresholds[h])
        for changed_dimension, neighbor in _policy_neighbors(center):
            threshold = center_threshold if neighbor.score_quantile == center.score_quantile else _resolved_threshold(sig, neighbor)
            r = replay(
                sig,
                prices,
                neighbor,
                CostModel(20),
                TaxConfig(False),
                start=full_start,
                end=full_end,
                initial=10000.0,
                resolved_threshold=threshold,
            )
            m = r["metrics"]
            rows.append({
                "horizon": h,
                "center_policy_id": center.policy_id,
                "neighbor_policy_id": neighbor.policy_id,
                "changed_dimension": changed_dimension,
                "score_quantile": neighbor.score_quantile,
                "top_fraction": neighbor.top_fraction,
                "max_names": neighbor.max_names,
                "holding_days": neighbor.holding_days,
                "resolved_threshold": threshold,
                "center_cagr_excess": center_excess,
                "neighbor_cagr_excess": float(m.get("cagr_excess", 0.0)),
                "delta_cagr_excess": float(m.get("cagr_excess", 0.0)) - center_excess,
                "trade_count": int(m.get("trade_count", 0)),
            })
        hrows = [x for x in rows if x["horizon"] == h]
        positive_fraction = sum(x["neighbor_cagr_excess"] > 0 for x in hrows) / max(1, len(hrows))
        gates[h] = {
            "neighbors": len(hrows),
            "positive_neighbor_fraction": positive_fraction,
            "median_neighbor_cagr_excess": float(median([x["neighbor_cagr_excess"] for x in hrows])) if hrows else 0.0,
            "plateau_pass": bool(center_excess > 0 and len(hrows) >= 3 and positive_fraction >= .50),
        }
    return rows, gates


def _previous_session_map(prices: pd.DataFrame) -> dict[pd.Timestamp, pd.Timestamp]:
    dates = sorted(pd.to_datetime(prices.loc[prices["ticker"].eq("URTH"), "date"]).unique())
    return {pd.Timestamp(dates[i]): pd.Timestamp(dates[i - 1]) for i in range(1, len(dates))}


def _concentration_diagnostics(
    predictions: pd.DataFrame,
    prices: pd.DataFrame,
    final_policies: dict,
    final_thresholds: dict,
    baseline_results: dict,
    full_start: pd.Timestamp,
    full_end: pd.Timestamp,
) -> tuple[list[dict], dict]:
    rows = []
    gates = {}
    prev_session = _previous_session_map(prices)
    for h, p in final_policies.items():
        base = baseline_results.get(h)
        trades = list(base.get("trades", [])) if base else []
        vals = sorted([float(x.get("excess_return", 0.0)) for x in trades], reverse=True)
        positive_total = sum(max(v, 0.0) for v in vals)
        counts = Counter(str(x.get("ticker")) for x in trades)
        top_ticker, top_ticker_count = counts.most_common(1)[0] if counts else ("", 0)
        top3_count = sum(v for _, v in counts.most_common(3))
        ticker_positive = defaultdict(float)
        for t in trades:
            ticker_positive[str(t.get("ticker"))] += max(float(t.get("excess_return", 0.0)), 0.0)
        top_ticker_positive_share = max(ticker_positive.values(), default=0.0) / positive_total if positive_total else 0.0

        sig = predictions.loc[predictions["horizon"].eq(h)].copy()
        threshold = float(final_thresholds[h])
        exclusion_metrics = {}
        if top_ticker:
            r = replay(
                sig.loc[~sig["ticker"].eq(top_ticker)], prices, p, CostModel(20), TaxConfig(False),
                start=full_start, end=full_end, initial=10000.0, resolved_threshold=threshold,
            )
            exclusion_metrics["exclude_top_ticker_cagr_excess"] = float(r["metrics"].get("cagr_excess", 0.0))

        ranked_trades = sorted(trades, key=lambda x: float(x.get("excess_return", 0.0)), reverse=True)
        for n in (1, 5):
            blocked = set()
            for t in ranked_trades[:n]:
                entry = pd.Timestamp(t.get("entry_date"))
                decision = prev_session.get(entry)
                if decision is not None:
                    blocked.add((str(t.get("ticker")), decision))
            if blocked:
                mask = pd.Series(True, index=sig.index)
                for ticker, decision in blocked:
                    mask &= ~(sig["ticker"].eq(ticker) & sig["decision_date"].eq(decision))
                r = replay(
                    sig.loc[mask], prices, p, CostModel(20), TaxConfig(False),
                    start=full_start, end=full_end, initial=10000.0, resolved_threshold=threshold,
                )
                exclusion_metrics[f"exclude_best_{n}_trade_cagr_excess"] = float(r["metrics"].get("cagr_excess", 0.0))
            else:
                exclusion_metrics[f"exclude_best_{n}_trade_cagr_excess"] = 0.0

        extreme = bool(
            (len(trades) and top_ticker_count / len(trades) > .50)
            or top_ticker_positive_share > .50
            or (positive_total and max(vals, default=0.0) / positive_total > .50)
        )
        row = {
            "horizon": h,
            "trade_count": len(trades),
            "top1_positive_contribution_share": max(vals[0], 0.0) / positive_total if vals and positive_total else 0.0,
            "top5_positive_contribution_share": sum(max(x, 0.0) for x in vals[:5]) / positive_total if positive_total else 0.0,
            "top10_positive_contribution_share": sum(max(x, 0.0) for x in vals[:10]) / positive_total if positive_total else 0.0,
            "top_ticker": top_ticker,
            "top_ticker_trade_share": top_ticker_count / max(1, len(trades)),
            "top3_ticker_trade_share": top3_count / max(1, len(trades)),
            "top_ticker_positive_contribution_share": top_ticker_positive_share,
            "extreme_winner_or_ticker_dependence": extreme,
            **exclusion_metrics,
        }
        rows.append(row)
        gates[h] = {"concentration_pass": not extreme, **row}
    return rows, gates


def _cross_regime_context_matrix(
    outer_rows: list[dict],
    predictions: pd.DataFrame,
    prices: pd.DataFrame,
    regime: pd.DataFrame,
) -> list[dict]:
    # This is a genuine chronological train-context -> test-regime matrix. Policies
    # are still general policies, not separately fitted regime specialists.
    out = []
    rmap = regime.set_index("date")["regime"]
    for row in outer_rows:
        h = int(row["horizon"])
        p = policy_from(row)
        fold = str(row["fold_id"])
        test = predictions.loc[(predictions["horizon"] == h) & (predictions["fold_id"] == fold)]
        if test.empty:
            continue
        start, end = test["decision_date"].min(), test["decision_date"].max()
        calibration_end = pd.Timestamp(row.get("calibration_end", start - pd.Timedelta(days=1)))
        train_labels = regime.loc[regime["date"] <= calibration_end, "regime"].dropna().astype(str)
        train_context = train_labels.mode().iloc[0] if not train_labels.empty else "UNKNOWN"
        rr = replay(
            test, prices, p, CostModel(20), TaxConfig(False), start=start, end=end,
            initial=10000.0, regime=regime, resolved_threshold=float(row["threshold"]),
        )
        for a in _regime_attribution(rr["curve"], h, "PRE_TAX"):
            out.append({
                "horizon": h,
                "fold_id": fold,
                "calibration_end": str(calibration_end.date()),
                "train_context_dominant_regime": train_context,
                "test_regime": a["regime"],
                "observations": a["observations"],
                "active_days": a["active_days"],
                "strategy_compounded_return_on_test_regime_days": a["strategy_compounded_return_on_regime_days"],
                "urth_compounded_return_on_test_regime_days": a["urth_compounded_return_on_regime_days"],
                "compounded_return_excess": a["compounded_return_excess"],
                "mean_daily_excess": a["mean_daily_excess"],
            })
    return out


def _readiness_gates(
    outer_summary: list[dict],
    portfolio_rows: list[dict],
    search_meta: dict,
    plateau_gates: dict,
    concentration_gates: dict,
    cross_regime_rows: list[dict],
) -> dict:
    accounting_ok = all(float(r.get("max_abs_accounting_error_eur", 0.0)) < 1e-5 for r in portfolio_rows)
    exposure_ok = all(float(r.get("max_total_exposure", 1.0)) <= 1.00001 for r in portfolio_rows)

    final_search_meta = []
    for _, entries in search_meta.get("search_coverage", {}).items():
        final_search_meta.extend([x for x in entries if x.get("final_fit")])
    coverage_ok = bool(final_search_meta) and all(bool(x.get("coverage_complete")) for x in final_search_meta)
    dynamic_ok = bool(final_search_meta) and all(bool(x.get("dynamic_stage_complete")) for x in final_search_meta)

    outer_ok = bool(outer_summary) and all(
        x["folds"] >= 5
        and x["positive_fold_fraction"] >= .60
        and x["median_cagr_excess"] > 0
        and x["trade_count"] >= 20
        for x in outer_summary
    )
    plateau_ok = bool(plateau_gates) and all(bool(x.get("plateau_pass")) for x in plateau_gates.values())
    concentration_ok = bool(concentration_gates) and all(bool(x.get("concentration_pass")) for x in concentration_gates.values())

    # Stress gate: final frozen policy must remain positive at 50 bps pre-tax and
    # 20/50 bps in the tax-aware world. These are Development stress diagnostics,
    # not substitutes for outer-fold evidence.
    by_h = defaultdict(list)
    for r in portfolio_rows:
        if float(r.get("capital", 0.0)) == 10000.0:
            by_h[int(r["horizon"])].append(r)
    stress_ok = bool(by_h)
    for _, vals in by_h.items():
        required = [
            r for r in vals
            if r.get("cost_world") in ("BASELINE_20_BPS_RT", "50_BPS_RT")
            and r.get("tax_world") in ("PRE_TAX", "DE_RETAIL_TAX_AWARE")
        ]
        if len(required) < 4 or any(float(r.get("cagr_excess", -1e9)) <= 0 for r in required):
            stress_ok = False

    regime_ok = bool(cross_regime_rows)
    gates = {
        "accounting_invariants_pass": accounting_ok,
        "exposure_invariants_pass": exposure_ok,
        "search_coverage_complete": coverage_ok,
        "dynamic_exit_stage_complete": dynamic_ok,
        "outer_fold_robustness_pass": outer_ok,
        "parameter_plateau_pass": plateau_ok,
        "concentration_pass": concentration_ok,
        "cost_tax_stress_pass": stress_ok,
        "cross_regime_context_matrix_complete": regime_ok,
    }
    gates["all_readiness_gates_pass"] = all(gates.values())
    return gates


def self_test() -> None:
    initial = 10000.0
    dates, flat_prices, flat_signals = _flat_fixture()

    p1 = Policy(5, .75, 1.0, 1, 1, sleeve=1.0)
    zero = replay(flat_signals, flat_prices, p1, CostModel(0), TaxConfig(False), initial=initial)
    assert abs(float(zero["curve"].iloc[0]["strategy_value"]) - initial) < 1e-8
    assert abs(float(zero["metrics"]["terminal_value"]) - initial) < 1e-6
    assert float(zero["metrics"]["max_abs_accounting_error_eur"]) < 1e-8
    assert float(zero["metrics"]["max_total_exposure"]) <= 1.000005
    assert zero["metrics"]["max_positions"] <= p1.max_names
    assert zero["trades"] and all(int(t["holding_days"]) == 1 for t in zero["trades"])

    costed = replay(flat_signals, flat_prices, p1, CostModel(20), TaxConfig(False), initial=initial)
    nav_loss = initial - float(costed["metrics"]["terminal_value"])
    total_cost = float(costed["metrics"]["total_cost_eur"])
    assert total_cost > 0 and abs(nav_loss - total_cost) < 1e-5
    assert float(costed["curve"]["cash_exposure"].min()) >= -1e-8

    _, multi_prices, multi_signals = _flat_fixture(tickers=("AAA", "BBB", "CCC"))
    p_multi = Policy(5, .0, 1.0, 3, 3, allocation="EQUAL_ACTIVE", sleeve=.50)
    multi = replay(multi_signals, multi_prices, p_multi, CostModel(0), TaxConfig(False), initial=initial, resolved_threshold=-1e9)
    exposure_sum = multi["curve"]["stock_exposure"] + multi["curve"]["urth_exposure"] + multi["curve"]["cash_exposure"]
    assert float(multi["curve"]["stock_exposure"].max()) <= .500005
    assert float(exposure_sum.max()) <= 1.000005
    assert int(multi["curve"]["positions"].max()) <= 3

    rising_prices = pd.concat([
        pd.DataFrame({"date": dates, "ticker": "URTH", "open": 100.0, "close": 100.0}),
        pd.DataFrame({"date": dates, "ticker": "AAA", "open": [100.0 + i for i in range(len(dates))], "close": [100.5 + i for i in range(len(dates))]}),
    ], ignore_index=True)
    one_signal = pd.DataFrame({"decision_date": [dates[0]], "ticker": ["AAA"], "fold_id": ["WF"], "horizon": [5], "score": [1.0]})
    h1 = replay(one_signal, rising_prices, p1, CostModel(0), TaxConfig(False), initial=initial, resolved_threshold=0.0)
    assert len(h1["trades"]) == 1
    assert pd.Timestamp(h1["trades"][0]["entry_date"]) == dates[1]
    assert pd.Timestamp(h1["trades"][0]["exit_date"]) == dates[2]
    p3 = Policy(5, .75, 1.0, 1, 3, sleeve=1.0)
    h3 = replay(one_signal, rising_prices, p3, CostModel(0), TaxConfig(False), initial=initial, resolved_threshold=0.0)
    assert pd.Timestamp(h3["trades"][0]["exit_date"]) == dates[4]

    ledger = TaxLedger(TaxConfig(True))
    assert ledger.realize_stock_trade(date(2020, 6, 1), 900.0, 1000.0, 0.0) == 0.0
    assert abs(ledger.loss_carryforward - 100.0) < 1e-8
    assert ledger.realize_stock_trade(date(2020, 7, 1), 1200.0, 1000.0, 0.0) == 0.0
    assert abs(ledger.loss_carryforward) < 1e-8
    assert ledger.realize_stock_trade(date(2021, 1, 4), 1100.0, 1000.0, 0.0) == 0.0
    assert ledger.tax_paid >= 0.0

    # Threshold contract: median of daily maxima [100, 10] is 55, not the median
    # across all four stock rows.
    threshold_fixture = pd.DataFrame([
        {"decision_date": dates[0], "ticker": "AAA", "score": 100.0},
        {"decision_date": dates[0], "ticker": "BBB", "score": 0.0},
        {"decision_date": dates[1], "ticker": "AAA", "score": 10.0},
        {"decision_date": dates[1], "ticker": "BBB", "score": 9.0},
    ])
    assert abs(_resolved_threshold(threshold_fixture, Policy(5, .50, .01, 1, 1)) - 55.0) < 1e-8

    sampled, smeta = _balanced_budget(grid(10), 4)
    assert smeta["budget_auto_raised"] and smeta["coverage_complete"]
    assert set(p.holding_days for p in sampled) >= set(SEARCH_HOLDING_DAYS)
    assert {5, 7, 10}.issubset(set(p.holding_days for p in sampled))
    assert set(p.max_names for p in sampled) >= set(SEARCH_MAX_NAMES)

    # Exact median aggregation regression: +5.32% is not the median of this H10-like set.
    fs = _outer_fold_summary([
        {"horizon": 10, "cagr_excess": x, "trade_count": 10}
        for x in (.3455, -.0606, .0532, -.2877, -.6050, .1642)
    ])[0]
    assert abs(fs["median_cagr_excess"] - (-.0037)) < 5e-4

    regime_fixture = pd.DataFrame({
        "date": dates[:5], "strategy_value": [100, 101, 100, 102, 103], "urth_value": [100, 100.5, 101, 101.5, 102],
        "positions": [0, 1, 1, 0, 1], "regime": ["A", "A", "B", "A", "B"],
    })
    ra = _regime_attribution(regime_fixture, 10, "PRE_TAX")
    assert sum(x["observations"] for x in ra) == 4

    assert _normalize_workers(0) == 1
    assert _normalize_workers(2) == 2
    assert _normalize_workers(999) == HARD_MAX_WORKERS
    assert p1.horizon == 5 and "family" not in asdict(p1)
    print("SELF_TEST_PASS")


def main(args: argparse.Namespace) -> int:
    if args.self_test:
        self_test()
        return 0
    if args.open_final_holdout:
        raise RuntimeError("FINAL_HOLDOUT_LOCKED: --open-final-holdout is intentionally not implemented in this development run")
    if not args.v5_predictions or not args.daily_store_root:
        raise ValueError("--v5-predictions and --daily-store-root are required for a Development 2016-2025 run")

    max_workers = _normalize_workers(args.max_workers)
    out = Path(args.output_root)
    out.mkdir(parents=True, exist_ok=True)
    predictions, prediction_audit = load_predictions(Path(args.v5_predictions))
    tickers = set(predictions["ticker"].unique())
    prices, price_audit = load_price_panel(Path(args.daily_store_root), tickers)
    urth = prices.loc[prices["ticker"].eq("URTH"), ["date", "close"]]
    regime = classify_urth(urth, RegimeContract())
    prices = prices.merge(regime[["date", "regime"]], on="date", how="left")

    outer, history, search_meta = run_walk_forward(
        predictions, prices, budget=args.stage_a_budget, max_workers=max_workers
    )
    final_policies = search_meta["final_policies"]
    final_thresholds = search_meta.get("final_thresholds", {})
    full_start, full_end = predictions["decision_date"].min(), predictions["decision_date"].max()

    portfolio_rows = []
    curves = []
    trades = []
    yearly = []
    regimes = []
    cost_audit = []
    tax_audit = []
    baseline_results = {}
    baseline_metrics = {}

    for h, p in final_policies.items():
        if h not in final_thresholds:
            raise RuntimeError(f"MISSING_FROZEN_THRESHOLD:H{h}")
        threshold = float(final_thresholds[h])
        sig = predictions.loc[predictions["horizon"].eq(h)]
        for capital in CAPITALS:
            for tax_enabled in (False, True):
                tag = "DE_RETAIL_TAX_AWARE" if tax_enabled else "PRE_TAX"
                r = replay(
                    sig, prices, p, CostModel(20), TaxConfig(tax_enabled), start=full_start, end=full_end,
                    initial=capital, regime=regime, resolved_threshold=threshold,
                )
                m = r["metrics"] | {
                    "horizon": h, "capital": capital, "tax_world": tag,
                    "cost_world": "BASELINE_20_BPS_RT", "policy_id": p.policy_id,
                    "resolved_threshold": threshold, "development_curve_only": True,
                }
                portfolio_rows.append(m)
                cost_audit.append({"horizon": h, "capital": capital, "tax_world": tag, **r["cost_audit"]})
                tax_audit.append({"horizon": h, "capital": capital, "tax_world": tag, **r["tax_audit"]})
                if capital == 10000.0:
                    c = r["curve"].copy()
                    c["horizon"] = h
                    c["capital"] = capital
                    c["tax_world"] = tag
                    curves.append(c)
                    trades.extend([{**t, "horizon": h, "capital": capital, "tax_world": tag} for t in r["trades"]])
                    regimes.extend(_regime_attribution(c, h, tag))
                    if not tax_enabled:
                        baseline_results[h] = r
                        baseline_metrics[h] = m
                    if not c.empty:
                        yc = c.assign(year=c.date.dt.year).groupby("year").agg(
                            starting_capital=("strategy_value", "first"),
                            ending_capital=("strategy_value", "last"),
                            strategy_return=("strategy_value", lambda x: x.iloc[-1] / x.iloc[0] - 1),
                            urth_return=("urth_value", lambda x: x.iloc[-1] / x.iloc[0] - 1),
                            trades=("positions", lambda x: int((x.diff().abs() > 0).sum())),
                            max_drawdown=("strategy_value", lambda x: float((x / x.cummax() - 1).min())),
                        ).reset_index()
                        yc["horizon"] = h
                        yc["tax_world"] = tag
                        yearly.append(yc)

                for rt in (30, 50):
                    stress = replay(
                        sig, prices, p, CostModel(rt), TaxConfig(tax_enabled), start=full_start, end=full_end,
                        initial=10000.0, regime=regime, resolved_threshold=threshold,
                    )
                    portfolio_rows.append(stress["metrics"] | {
                        "horizon": h, "capital": 10000.0, "tax_world": tag,
                        "cost_world": f"{rt}_BPS_RT", "policy_id": p.policy_id,
                        "resolved_threshold": threshold, "development_curve_only": True,
                    })

    outer_tax_rows = []
    for row in outer:
        p = policy_from(row)
        fold = row["fold_id"]
        h = int(row["horizon"])
        dates_h = predictions.loc[(predictions["horizon"] == h) & (predictions["fold_id"] == fold), "decision_date"]
        r = replay(
            predictions.loc[(predictions["horizon"] == h) & (predictions["fold_id"] == fold)],
            prices, p, CostModel(20), TaxConfig(True), start=dates_h.min(), end=dates_h.max(), initial=10000.0,
            regime=regime, resolved_threshold=float(row.get("threshold", float("inf"))),
        )
        outer_tax_rows.append(row | {
            "after_tax_cagr_excess": r["metrics"].get("cagr_excess", 0.0),
            "after_tax_tax_paid": r["metrics"].get("tax_paid", 0.0),
        })

    plateau_rows, plateau_gates = _parameter_plateau(
        predictions, prices, final_policies, final_thresholds, baseline_metrics, full_start, full_end
    )
    concentration_rows, concentration_gates = _concentration_diagnostics(
        predictions, prices, final_policies, final_thresholds, baseline_results, full_start, full_end
    )
    cross_regime_rows = _cross_regime_context_matrix(outer, predictions, prices, regime)
    outer_summary = _outer_fold_summary(outer)
    gates = _readiness_gates(
        outer_summary, portfolio_rows, search_meta, plateau_gates, concentration_gates, cross_regime_rows
    )

    if not gates["accounting_invariants_pass"] or not gates["exposure_invariants_pass"]:
        decision = "DEVELOPMENT_ACCOUNTING_GATE_FAILED"
    elif not gates["search_coverage_complete"]:
        decision = "INSUFFICIENT_SEARCH_COVERAGE"
    elif gates["all_readiness_gates_pass"]:
        decision = "FROZEN_POLICY_READY_FOR_FINAL_HOLDOUT"
    elif any(x["median_cagr_excess"] > 0 for x in outer_summary):
        decision = "MIXED_PORTFOLIO_EVIDENCE_NOT_READY_FOR_FINAL_HOLDOUT"
    else:
        decision = "NO_ROBUST_AFTER_COST_ALPHA_NOT_READY_FOR_FINAL_HOLDOUT"

    selected = []
    for h, p in sorted(final_policies.items()):
        selected.append({
            "horizon": h, **asdict(p), "policy_id": p.policy_id,
            "resolved_threshold": float(final_thresholds[h]),
        })
    frozen = {
        "status": "DEVELOPMENT_SUITE_COMPLETE",
        "decision": decision,
        "final_holdout_locked": True,
        "final_holdout_opened": False,
        "ready_for_final_holdout": bool(gates["all_readiness_gates_pass"]),
        "policy_contract": "V5_SELECTED:H5/H10/H20; daily-top-score threshold; family/candidate provenance only",
        "policies": selected,
        "policy_hash": stable_hash(selected),
        "readiness_gates": gates,
        "usage": "FROZEN_DEVELOPMENT_INPUT_NOT_INDEPENDENT_OOS_RESULT",
    }

    write_json(out / "frozen_portfolio_policy.json", frozen)
    write_json(out / "regime_contract.json", asdict(RegimeContract()))
    write_csv(out / "portfolio_results.csv", portfolio_rows)
    write_csv(out / "portfolio_curves.csv", pd.concat(curves, ignore_index=True) if curves else [])
    write_csv(out / "yearly_results.csv", pd.concat(yearly, ignore_index=True) if yearly else [])
    write_csv(out / "regime_results.csv", regimes)
    write_csv(out / "cross_regime_matrix.csv", cross_regime_rows)
    write_csv(out / "holdout_fold_results.csv", outer_tax_rows)
    write_csv(out / "outer_fold_results.csv", outer_tax_rows)
    write_csv(out / "outer_fold_summary.csv", outer_summary)
    write_csv(out / "trade_log.csv", trades)
    write_csv(out / "cost_audit.csv", cost_audit)
    write_csv(out / "tax_audit.csv", tax_audit)
    write_csv(out / "search_leaderboard.csv", history)
    write_csv(out / "parameter_plateau.csv", plateau_rows)
    write_csv(out / "concentration_audit.csv", concentration_rows)

    summary = {
        "status": "DEVELOPMENT_SUITE_COMPLETE",
        "decision": decision,
        "paper_trading_only": True,
        "final_holdout_opened": False,
        "final_holdout_locked": True,
        "ready_for_final_holdout": bool(gates["all_readiness_gates_pass"]),
        "prediction_audit": prediction_audit,
        "price_audit": price_audit,
        "regime_contract": asdict(RegimeContract()),
        "signal_contract": "V5_SELECTED:H5/H10/H20; daily-top-score threshold; V2 excluded",
        "outer_fold_count": len(outer_tax_rows),
        "outer_fold_summary": outer_summary,
        "policy_selection_history_rows": len(history),
        "final_policy_hash": frozen["policy_hash"],
        "search_budget_per_horizon_requested": args.stage_a_budget,
        "search_meta": search_meta,
        "max_workers_requested": int(args.max_workers),
        "max_workers_effective": max_workers,
        "max_workers_hard_cap": HARD_MAX_WORKERS,
        "tax_benchmark_approximate": True,
        "holdout_used_for_optimization": False,
        "readiness_gates": gates,
        "parameter_plateau_gates": plateau_gates,
        "concentration_gates": concentration_gates,
        "next_action": "rerun Development after any research-contract change; keep final holdout closed until every readiness gate passes",
    }
    write_json(out / "summary.json", summary)

    report = [
        "# Chronological V5 Opportunity Portfolio Research",
        "",
        f"Decision: `DEVELOPMENT_SUITE_COMPLETE` / `{decision}`",
        "",
        "The locked final holdout was not opened.",
        "",
        "## Corrected research contract",
        "",
        "- Score thresholds are calibrated on historical daily top scores, matching Opportunity-Core.",
        "- The numeric resolved threshold is persisted and reused in every Development/cost/tax replay.",
        "- Stage-A sampling is balanced across score quantile, top fraction, max_names and holding_days.",
        "- Policy ranking is lexicographic across chronological prior-fold robustness, not a single historical CAGR.",
        f"- CPU concurrency is bounded: requested {args.max_workers}, effective {max_workers}, hard cap {HARD_MAX_WORKERS}.",
        "- Regime results compound daily returns attributed to each regime; they do not use last/first NAV across disjoint dates.",
        "- Concentration diagnostics include ticker dependence and explicit exclude-top-trade/top-ticker replays.",
        "- parameter_plateau.csv contains actual adjacent-policy replays.",
        "",
        "## Outer-fold summary",
        "",
        json.dumps(outer_summary, indent=2, sort_keys=True),
        "",
        "## Readiness gates",
        "",
        json.dumps(gates, indent=2, sort_keys=True),
        "",
        "## Frozen Development policy",
        "",
        json.dumps(frozen, indent=2, sort_keys=True),
        "",
        "Any previous artifacts generated under the all-stock threshold or incomplete Stage-A coverage contract are superseded and must not be used as evidence.",
    ]
    (out / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True, default=str))
    return 0


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--v5-predictions", default="")
    p.add_argument("--daily-store-root", default="")
    p.add_argument("--output-root", default="artifacts/opportunity-portfolio-research")
    p.add_argument("--stage-a-budget", type=int, default=48)
    p.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS)
    p.add_argument("--open-final-holdout", action="store_true")
    p.add_argument("--self-test", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    raise SystemExit(main(parse_args()))
