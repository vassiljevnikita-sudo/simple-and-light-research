from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
from statistics import median
from threading import Lock

import numpy as np
import pandas as pd

from .portfolio_policy_contracts import Policy, policy_dict
from .cost_contracts import CostModel
from .tax_contracts import TaxConfig
from .portfolio_replay_fragment_cache import FragmentStore, build_fragment_namespace
from .next_open_portfolio_replay import prepare_prices, prepare_signals, replay

SEARCH_QUANTILES = (0.90, 0.95, 0.975, 0.99)
SEARCH_TOP_FRACTIONS = (0.005, 0.01, 0.025)
SEARCH_MAX_NAMES = (1, 2, 3, 5)
SEARCH_HOLDING_DAYS = (1, 2, 3, 5, 7, 10, 15, 20)
MIN_POSITIVE_FOLD_FRACTION = 0.60
MIN_ROBUST_FOLDS = 3
MIN_TOTAL_TRADES = 8

_FOLD_RESULT_CACHE: dict[tuple, dict | None] = {}
_FOLD_RESULT_CACHE_LOCK = Lock()
_FOLD_CACHE_HITS = 0
_FOLD_CACHE_MISSES = 0
_FRAGMENT_STORE: FragmentStore | None = None


def _cache_key(*parts) -> str:
    raw = json.dumps(parts, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _research_input_fingerprint(pred: pd.DataFrame, prices: pd.DataFrame) -> str:
    """Exact content fingerprint of columns that can change search/replay results."""
    h = hashlib.sha256()
    specs = (
        ("predictions", pred, ("decision_date", "ticker", "fold_id", "horizon", "score")),
        ("prices", prices, ("date", "ticker", "open", "close")),
    )
    for label, frame, wanted in specs:
        cols = [c for c in wanted if c in frame.columns]
        h.update(label.encode("utf-8"))
        h.update(str(tuple(cols)).encode("utf-8"))
        h.update(str(frame.shape).encode("utf-8"))
        if cols:
            hashes = pd.util.hash_pandas_object(frame[cols], index=False, categorize=True).to_numpy(dtype=np.uint64)
            h.update(hashes.tobytes())
    return h.hexdigest()


def _path_hash(result: dict) -> str:
    path = [
        (str(x.get("ticker")), str(x.get("entry_date")), str(x.get("exit_date")))
        for x in result.get("trades", [])
    ]
    return hashlib.sha256(json.dumps(path, separators=(",", ":")).encode("utf-8")).hexdigest()[:20]


def _policy_from_dict(row: dict) -> Policy:
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
        float(row.get("sleeve", .50)),
    )


def grid(horizon: int, sleeve: float = .50) -> list[Policy]:
    return [
        Policy(horizon, q, top, n, hold, sleeve=sleeve)
        for q in SEARCH_QUANTILES
        for top in SEARCH_TOP_FRACTIONS
        for n in SEARCH_MAX_NAMES
        for hold in SEARCH_HOLDING_DAYS
    ]


def minimum_coverage_budget() -> int:
    return max(len(SEARCH_QUANTILES), len(SEARCH_TOP_FRACTIONS), len(SEARCH_MAX_NAMES), len(SEARCH_HOLDING_DAYS))


def _coverage(items: list[Policy]) -> dict:
    return {
        "score_quantiles": sorted({float(p.score_quantile) for p in items}),
        "top_fractions": sorted({float(p.top_fraction) for p in items}),
        "max_names": sorted({int(p.max_names) for p in items}),
        "holding_days": sorted({int(p.holding_days) for p in items}),
    }


def _coverage_complete(items: list[Policy]) -> bool:
    c = _coverage(items)
    return (
        set(c["score_quantiles"]) >= set(SEARCH_QUANTILES)
        and set(c["top_fractions"]) >= set(SEARCH_TOP_FRACTIONS)
        and set(c["max_names"]) >= set(SEARCH_MAX_NAMES)
        and set(c["holding_days"]) >= set(SEARCH_HOLDING_DAYS)
    )


def _balanced_budget(items: list[Policy], budget: int) -> tuple[list[Policy], dict]:
    minimum = minimum_coverage_budget()
    effective_budget = max(int(budget), minimum)
    lookup = {(p.score_quantile, p.top_fraction, p.max_names, p.holding_days): p for p in items}
    if effective_budget >= len(items):
        selected = list(items)
    else:
        selected = []
        seen = set()
        for i, hold in enumerate(SEARCH_HOLDING_DAYS):
            key = (
                SEARCH_QUANTILES[i % len(SEARCH_QUANTILES)],
                SEARCH_TOP_FRACTIONS[i % len(SEARCH_TOP_FRACTIONS)],
                SEARCH_MAX_NAMES[i % len(SEARCH_MAX_NAMES)],
                hold,
            )
            p = lookup[key]
            selected.append(p)
            seen.add(p.policy_id)
        i = 0
        while len(selected) < effective_budget and i < len(items) * 8:
            key = (
                SEARCH_QUANTILES[(i * 3 + i // 5) % len(SEARCH_QUANTILES)],
                SEARCH_TOP_FRACTIONS[(i * 5 + i // 7) % len(SEARCH_TOP_FRACTIONS)],
                SEARCH_MAX_NAMES[(i * 7 + i // 3) % len(SEARCH_MAX_NAMES)],
                SEARCH_HOLDING_DAYS[(i * 11 + i // 2) % len(SEARCH_HOLDING_DAYS)],
            )
            p = lookup[key]
            if p.policy_id not in seen:
                seen.add(p.policy_id)
                selected.append(p)
            i += 1
        if len(selected) < effective_budget:
            for p in items:
                if p.policy_id in seen:
                    continue
                selected.append(p)
                seen.add(p.policy_id)
                if len(selected) >= effective_budget:
                    break
    meta = {
        "requested_budget": int(budget),
        "effective_budget": int(len(selected)),
        "minimum_coverage_budget": int(minimum),
        "budget_auto_raised": bool(int(budget) < minimum),
        "coverage": _coverage(selected),
        "coverage_complete": _coverage_complete(selected),
    }
    return selected, meta


def _top_scores_from_prepared(prepared: dict, history_end: pd.Timestamp | None = None) -> np.ndarray:
    values = prepared.get("top_score_by_date", {})
    if history_end is None:
        return np.asarray(tuple(values.values()), dtype=float)
    cutoff = pd.Timestamp(history_end)
    return np.asarray([score for d, score in values.items() if d <= cutoff], dtype=float)


def _thresholds_from_prepared(prepared: dict, history_end: pd.Timestamp | None = None) -> dict[float, float]:
    tops = _top_scores_from_prepared(prepared, history_end)
    if len(tops) == 0:
        return {q: float("inf") for q in SEARCH_QUANTILES}
    return {q: float(np.quantile(tops, q)) for q in SEARCH_QUANTILES}


def _resolved_threshold(hist: pd.DataFrame, policy: Policy) -> float:
    tops = _top_scores_from_prepared(prepare_signals(hist))
    return float(np.quantile(tops, policy.score_quantile)) if len(tops) else float("inf")


def _fold_ranges(pred: pd.DataFrame) -> list[tuple[str, pd.Timestamp, pd.Timestamp]]:
    x = pred.groupby("fold_id")["decision_date"].agg(["min", "max"]).sort_values(["min", "max"])
    return [(str(i), pd.Timestamp(r["min"]), pd.Timestamp(r["max"])) for i, r in x.iterrows()]


def _prepare_fold_data(pred: pd.DataFrame) -> list[dict]:
    folds = []
    for fold_id, g in pred.groupby("fold_id", sort=False):
        frame = g.copy()
        start = pd.Timestamp(frame["decision_date"].min())
        end = pd.Timestamp(frame["decision_date"].max())
        folds.append({
            "fold_id": str(fold_id),
            "start": start,
            "end": end,
            "signals": frame,
            "prepared": prepare_signals(frame),
        })
    folds.sort(key=lambda x: (x["start"], x["end"], x["fold_id"]))
    return folds


def _single_replay_rank(result: dict) -> tuple:
    m = result["metrics"]
    return (m.get("cagr_excess", -1e9), m.get("terminal_wealth_excess_eur", -1e9), -m.get("turnover", 1e9))


def _dedupe(results: list[tuple[Policy, dict]]) -> list[tuple[Policy, dict]]:
    best = {}
    for p, r in results:
        h = _path_hash(r)
        if h not in best or _single_replay_rank(r) > _single_replay_rank(best[h][1]):
            best[h] = (p, r)
    return list(best.values())


def _parallel_map(func, args: list, max_workers: int, label: str) -> list:
    total = len(args)
    if total == 0:
        return []
    print(f"[{label}] start: {total} jobs, workers={max_workers}", flush=True)
    if max_workers <= 1 or total == 1:
        out = []
        step = max(1, total // 8)
        for i, arg in enumerate(args, 1):
            out.append(func(arg))
            if i == total or i % step == 0:
                print(f"[{label}] {i}/{total}", flush=True)
        return out
    results = [None] * total
    step = max(1, total // 8)
    with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix=label.replace(" ", "-")) as pool:
        future_to_index = {pool.submit(func, arg): i for i, arg in enumerate(args)}
        completed = 0
        for future in as_completed(future_to_index):
            results[future_to_index[future]] = future.result()
            completed += 1
            if completed == total or completed % step == 0:
                print(f"[{label}] {completed}/{total}", flush=True)
    return results


def _compact_search_result(result: dict, threshold: float) -> dict:
    trades = []
    for trade in result.get("trades", []):
        row = dict(trade)
        if row.get("entry_date") is not None:
            row["entry_date"] = str(row["entry_date"])
        if row.get("exit_date") is not None:
            row["exit_date"] = str(row["exit_date"])
        trades.append(row)
    return {"metrics": dict(result.get("metrics", {})), "trades": trades, "resolved_threshold": float(threshold)}


def _evaluate_policy(args):
    signals, prices, prepared_signals, policy, history_end, threshold = args
    cache_key = _cache_key(
        "history_replay", policy.horizon, policy.policy_id, str(pd.Timestamp(history_end)),
        float(threshold).hex(), 20.0, False, 10000.0,
    )
    if _FRAGMENT_STORE is not None:
        found, payload = _FRAGMENT_STORE.get("history_replay", cache_key)
        if found:
            return policy, dict(payload)
    result = replay(
        signals, prices, policy, CostModel(20), TaxConfig(False), end=history_end,
        initial=10000.0, resolved_threshold=threshold, prepared_signals=prepared_signals,
    )
    compact = _compact_search_result(result, threshold)
    if _FRAGMENT_STORE is not None:
        _FRAGMENT_STORE.put("history_replay", cache_key, compact)
    return policy, compact


def _fold_metric_cached(prices: pd.DataFrame, fold: dict, policy: Policy, threshold: float) -> dict | None:
    global _FOLD_CACHE_HITS, _FOLD_CACHE_MISSES
    stable_key = (
        policy.horizon, fold["fold_id"], str(fold["start"]), str(fold["end"]),
        policy.policy_id, float(threshold).hex(),
    )
    with _FOLD_RESULT_CACHE_LOCK:
        if stable_key in _FOLD_RESULT_CACHE:
            _FOLD_CACHE_HITS += 1
            return _FOLD_RESULT_CACHE[stable_key]
    disk_key = _cache_key("fold_metric", *stable_key, 20.0, False, 10000.0)
    if _FRAGMENT_STORE is not None:
        found, payload = _FRAGMENT_STORE.get("fold_metric", disk_key)
        if found:
            with _FOLD_RESULT_CACHE_LOCK:
                _FOLD_CACHE_HITS += 1
                _FOLD_RESULT_CACHE[stable_key] = payload
            return payload
    with _FOLD_RESULT_CACHE_LOCK:
        _FOLD_CACHE_MISSES += 1
    r = replay(
        fold["signals"], prices, policy, CostModel(20), TaxConfig(False),
        start=fold["start"], end=fold["end"], initial=10000.0,
        resolved_threshold=threshold, prepared_signals=fold["prepared"],
    )
    m = r["metrics"]
    value = None
    if m.get("trade_count", 0) > 0:
        value = {
            "fold_id": fold["fold_id"],
            "cagr_excess": float(m.get("cagr_excess", 0.0)),
            "worst_relative_drawdown": float(m.get("worst_relative_drawdown", 0.0)),
            "turnover": float(m.get("turnover", 0.0)),
            "trade_count": int(m.get("trade_count", 0)),
        }
    with _FOLD_RESULT_CACHE_LOCK:
        _FOLD_RESULT_CACHE[stable_key] = value
    if _FRAGMENT_STORE is not None:
        _FRAGMENT_STORE.put("fold_metric", disk_key, value)
    return value


def _historical_fold_metrics_from_data(
    fold_data: list[dict], prices: pd.DataFrame, policy: Policy, threshold: float, history_end: pd.Timestamp,
) -> dict:
    folds = []
    for fold in fold_data:
        if fold["end"] > history_end:
            continue
        value = _fold_metric_cached(prices, fold, policy, threshold)
        if value is not None:
            folds.append(value)
    vals = [x["cagr_excess"] for x in folds]
    return {
        "folds": folds,
        "fold_count": len(folds),
        "median_cagr_excess": float(median(vals)) if vals else -1e9,
        "q25_cagr_excess": float(np.quantile(vals, .25)) if vals else -1e9,
        "positive_fold_fraction": float(sum(v > 0 for v in vals) / len(vals)) if vals else 0.0,
        "worst_cagr_excess": float(min(vals)) if vals else -1e9,
        "worst_relative_drawdown": float(min((x["worst_relative_drawdown"] for x in folds), default=-1e9)),
        "turnover": float(np.mean([x["turnover"] for x in folds])) if folds else 1e9,
        "trade_count": int(sum(x["trade_count"] for x in folds)),
    }


def _evaluate_robustness(args):
    fold_data, prices, policy, result, history_end = args
    stats = _historical_fold_metrics_from_data(
        fold_data, prices, policy, float(result["resolved_threshold"]), history_end
    )
    return policy, result, stats


def _robust_rank(stats: dict) -> tuple:
    return (
        stats["median_cagr_excess"], stats["q25_cagr_excess"], stats["positive_fold_fraction"],
        stats["worst_cagr_excess"], stats["worst_relative_drawdown"], -stats["turnover"],
    )


def _eligible_robust(stats: dict) -> bool:
    return (
        stats["fold_count"] >= MIN_ROBUST_FOLDS
        and stats["positive_fold_fraction"] >= MIN_POSITIVE_FOLD_FRACTION
        and stats["trade_count"] >= MIN_TOTAL_TRADES
    )


def _dynamic_neighbors(base: Policy) -> list[Policy]:
    return [
        Policy(base.horizon, base.score_quantile, base.top_fraction, base.max_names, base.holding_days, "SIGNAL_DECAY", 0.0, base.replacement, base.allocation, base.sleeve),
        Policy(base.horizon, base.score_quantile, base.top_fraction, base.max_names, base.holding_days, "RELATIVE_STOP", -.02, base.replacement, base.allocation, base.sleeve),
        Policy(base.horizon, base.score_quantile, base.top_fraction, base.max_names, base.holding_days, "RELATIVE_STOP", -.04, base.replacement, base.allocation, base.sleeve),
        Policy(base.horizon, base.score_quantile, base.top_fraction, base.max_names, base.holding_days, "TRAILING_RELATIVE_STOP", .03, base.replacement, base.allocation, base.sleeve),
        Policy(base.horizon, base.score_quantile, base.top_fraction, base.max_names, base.holding_days, "TAKE_PROFIT_RELATIVE", .05, base.replacement, base.allocation, base.sleeve),
    ]


def choose_policy(
    pred: pd.DataFrame, prices: pd.DataFrame, horizon: int, history_end: pd.Timestamp,
    budget: int = 48, max_workers: int = 4, fold_data: list[dict] | None = None,
    progress_prefix: str = "", prepared_pred: dict | None = None,
) -> tuple[Policy, dict, list[dict], dict]:
    if prepared_pred is None:
        hist = pred.loc[pred["decision_date"] <= history_end].copy()
        if "horizon" in hist.columns:
            hist = hist.loc[hist["horizon"] == horizon]
        if hist.empty:
            raise ValueError(f"no historical predictions for H{horizon}")
        replay_signals = hist
        prepared_hist = prepare_signals(hist)
        threshold_end = None
    else:
        if pred.empty:
            raise ValueError(f"no historical predictions for H{horizon}")
        replay_signals = pred
        prepared_hist = prepared_pred
        threshold_end = history_end
    prepare_prices(prices)
    thresholds = _thresholds_from_prepared(prepared_hist, threshold_end)
    if all(not np.isfinite(v) for v in thresholds.values()):
        raise ValueError(f"no historical top scores for H{horizon}")
    if fold_data is None:
        fold_data = _prepare_fold_data(pred.loc[pred["horizon"] == horizon] if "horizon" in pred.columns else pred)
    candidates, search_meta = _balanced_budget(grid(horizon), budget)
    max_workers = max(1, int(max_workers))
    prefix = progress_prefix or f"H{horizon}"
    coarse_args = [(replay_signals, prices, prepared_hist, p, history_end, thresholds[p.score_quantile]) for p in candidates]
    evaluated = _parallel_map(_evaluate_policy, coarse_args, max_workers, f"{prefix} coarse")
    evaluated = [(p, r) for p, r in evaluated if r["metrics"].get("trade_count", 0) >= 1]
    evaluated = _dedupe(evaluated)
    if not evaluated:
        raise RuntimeError(f"NO_HISTORICAL_PORTFOLIO_PATH:H{horizon}")
    robust_args = [(fold_data, prices, p, r, history_end) for p, r in evaluated]
    robust = _parallel_map(_evaluate_robustness, robust_args, max_workers, f"{prefix} robustness")
    robust.sort(key=lambda x: _robust_rank(x[2]), reverse=True)
    robust_bases = [x for x in robust if _eligible_robust(x[2])][:5]
    dynamic_evaluated = []
    if robust_bases:
        dyn_policies = []
        seen = set()
        for base, _, _ in robust_bases:
            for p in _dynamic_neighbors(base):
                if p.policy_id not in seen:
                    seen.add(p.policy_id)
                    dyn_policies.append(p)
        dyn_args = [(replay_signals, prices, prepared_hist, p, history_end, thresholds[p.score_quantile]) for p in dyn_policies]
        dyn_raw = _parallel_map(_evaluate_policy, dyn_args, max_workers, f"{prefix} dynamic")
        dyn_raw = _dedupe([(p, r) for p, r in dyn_raw if r["metrics"].get("trade_count", 0) >= 1])
        dyn_robust_args = [(fold_data, prices, p, r, history_end) for p, r in dyn_raw]
        dynamic_evaluated = _parallel_map(_evaluate_robustness, dyn_robust_args, max_workers, f"{prefix} dynamic-robustness")
    finalists = robust + dynamic_evaluated
    finalists.sort(key=lambda x: _robust_rank(x[2]), reverse=True)
    chosen, chosen_result, chosen_stats = finalists[0]
    search_meta = dict(search_meta)
    search_meta.update({
        "max_workers": max_workers,
        "prepared_history_rows": int(prepared_hist.get("row_count", 0)),
        "history_view_reuses_prepared_horizon": bool(prepared_pred is not None),
        "prepared_fold_count": int(len(fold_data)),
        "coarse_paths_evaluated": len(robust),
        "dynamic_paths_evaluated": len(dynamic_evaluated),
        "dynamic_stage_complete": bool(robust_bases and dynamic_evaluated),
        "chosen_robust_gate_pass": _eligible_robust(chosen_stats),
        "chosen_fold_stats": {k: v for k, v in chosen_stats.items() if k != "folds"},
    })
    leaderboard = []
    for p, r, stats in finalists[:50]:
        leaderboard.append(policy_dict(p) | {
            "path_hash": _path_hash(r),
            "resolved_threshold": float(r["resolved_threshold"]),
            "history_median_fold_cagr_excess": stats["median_cagr_excess"],
            "history_q25_fold_cagr_excess": stats["q25_cagr_excess"],
            "history_positive_fold_fraction": stats["positive_fold_fraction"],
            "history_worst_fold_cagr_excess": stats["worst_cagr_excess"],
            "history_worst_relative_drawdown": stats["worst_relative_drawdown"],
            "history_fold_count": stats["fold_count"],
            "history_trade_count": stats["trade_count"],
            "history_turnover": stats["turnover"],
            "robust_gate_pass": _eligible_robust(stats),
        })
    chosen_result["robust_stats"] = chosen_stats
    return chosen, chosen_result, leaderboard, search_meta


def _window_checkpoint_key(horizon: int, fold: str, history_end: pd.Timestamp, budget: int) -> str:
    return _cache_key("outer_window", int(horizon), str(fold), str(pd.Timestamp(history_end)), int(budget))


def _final_fit_checkpoint_key(horizon: int, history_end: pd.Timestamp, budget: int) -> str:
    return _cache_key("final_fit", int(horizon), str(pd.Timestamp(history_end)), int(budget))


def _horizon_checkpoint_key(horizon: int, budget: int) -> str:
    return _cache_key("horizon", int(horizon), int(budget))


def run_walk_forward(
    pred: pd.DataFrame, prices: pd.DataFrame, horizons=(5, 10, 20), budget=48,
    max_workers: int = 4, fragment_cache_path: str | Path | None = None,
    fragment_namespace: str | None = None,
) -> tuple[list[dict], list[dict], dict]:
    global _FOLD_CACHE_HITS, _FOLD_CACHE_MISSES, _FRAGMENT_STORE
    with _FOLD_RESULT_CACHE_LOCK:
        _FOLD_RESULT_CACHE.clear()
        _FOLD_CACHE_HITS = 0
        _FOLD_CACHE_MISSES = 0

    if fragment_cache_path is None:
        configured = os.environ.get("OPPORTUNITY_FRAGMENT_CACHE_PATH", "").strip()
        if configured:
            fragment_cache_path = configured
    if fragment_cache_path is not None and not fragment_namespace:
        input_fingerprint = _research_input_fingerprint(pred, prices)
        fragment_namespace = build_fragment_namespace(input_fingerprint)
        print(f"[portfolio] fragment input fingerprint: {input_fingerprint[:16]}...", flush=True)

    store = None
    if fragment_cache_path is not None and fragment_namespace:
        store = FragmentStore(Path(fragment_cache_path), str(fragment_namespace))
        _FRAGMENT_STORE = store
        s = store.stats()
        print(f"[portfolio] fragment cache: {s['entries_loaded_at_start']} reusable fragments loaded from {s['path']}", flush=True)
    else:
        _FRAGMENT_STORE = None

    prepare_prices(prices)
    outer: list[dict] = []
    history: list[dict] = []
    leaderboard: dict = {}
    final_policies: dict = {}
    final_thresholds: dict = {}
    coverage: dict = {}
    folds = _fold_ranges(pred)
    try:
        for horizon in horizons:
            horizon_key = _horizon_checkpoint_key(horizon, budget)
            if store is not None:
                found, payload = store.get("horizon_checkpoint", horizon_key)
                if found:
                    payload = dict(payload)
                    h_outer = list(payload.get("outer", []))
                    h_history = list(payload.get("history", []))
                    h_board = list(payload.get("leaderboard", []))
                    hmeta = list(payload.get("search_coverage", []))
                    outer.extend(h_outer)
                    history.extend(h_history)
                    leaderboard[horizon] = h_board
                    coverage[horizon] = hmeta
                    if payload.get("final_policy"):
                        final_policies[horizon] = _policy_from_dict(payload["final_policy"])
                        final_thresholds[horizon] = float(payload["final_threshold"])
                    print(f"[portfolio] H{horizon}: RESUME complete horizon checkpoint ({len(h_outer)} outer rows)", flush=True)
                    continue

            hpred = pred.loc[pred["horizon"] == horizon].copy()
            prepared_hpred = prepare_signals(hpred)
            fold_data = _prepare_fold_data(hpred)
            hfolds = [(x["fold_id"], x["start"], x["end"]) for x in fold_data]
            h_outer: list[dict] = []
            h_history: list[dict] = []
            h_board_all: list[dict] = []
            hmeta: list[dict] = []
            print(f"[portfolio] H{horizon}: {len(hfolds)} folds prepared, {prepared_hpred.get('row_count', 0)} signal rows; workers={max_workers}", flush=True)

            for i, (fold, start, end) in enumerate(hfolds):
                if i < 2:
                    continue
                hist_end = hfolds[i - 1][2]
                checkpoint_key = _window_checkpoint_key(horizon, fold, hist_end, budget)
                if store is not None:
                    found, payload = store.get("outer_window_checkpoint", checkpoint_key)
                    if found:
                        payload = dict(payload)
                        h_outer.append(dict(payload["outer_row"]))
                        rows = list(payload.get("history_rows", []))
                        board_rows = list(payload.get("leaderboard_rows", []))
                        h_history.extend(rows)
                        h_board_all.extend(board_rows)
                        hmeta.append(dict(payload.get("search_meta", {})))
                        print(f"[portfolio] H{horizon} fold {fold}: RESUME outer-window checkpoint", flush=True)
                        continue

                progress_prefix = f"H{horizon} outer {i - 1}/{max(1, len(hfolds) - 2)}"
                try:
                    policy, hist_result, board, smeta = choose_policy(
                        hpred, prices, horizon, hist_end, budget, max_workers,
                        fold_data=fold_data, progress_prefix=progress_prefix, prepared_pred=prepared_hpred,
                    )
                except (ValueError, RuntimeError) as exc:
                    h_history.append({"horizon": horizon, "test_fold": fold, "status": str(exc)})
                    continue
                test_fold = fold_data[i]
                threshold = float(hist_result.get("resolved_threshold", float("inf")))
                test_result = replay(
                    test_fold["signals"], prices, policy, CostModel(20), TaxConfig(False),
                    start=start, end=end, initial=10000.0, resolved_threshold=threshold,
                    prepared_signals=test_fold["prepared"],
                )
                m = test_result["metrics"]
                outer_row = {
                    "horizon": horizon, "fold_id": fold, "fold_start": str(start.date()),
                    "fold_end": str(end.date()), "calibration_end": str(hist_end.date()),
                    "policy_id": policy.policy_id, **policy_dict(policy), "threshold": threshold, **m,
                }
                history_rows = [
                    {**row, "test_fold": fold, "calibration_end": str(hist_end.date()), "selected": row.get("policy_id") == policy.policy_id}
                    for row in board
                ]
                search_meta_row = {"test_fold": fold, **smeta}
                h_outer.append(outer_row)
                h_history.extend(history_rows)
                h_board_all.extend(board)
                hmeta.append(search_meta_row)
                if store is not None:
                    store.put("outer_window_checkpoint", checkpoint_key, {
                        "outer_row": outer_row, "history_rows": history_rows,
                        "leaderboard_rows": board, "search_meta": search_meta_row,
                    })
                    store.flush()

            if hfolds:
                final_hist_end = hfolds[-1][2]
                final_key = _final_fit_checkpoint_key(horizon, final_hist_end, budget)
                resumed_final = False
                if store is not None:
                    found, payload = store.get("final_fit_checkpoint", final_key)
                    if found:
                        payload = dict(payload)
                        p = _policy_from_dict(payload["policy"])
                        final_policies[horizon] = p
                        final_thresholds[horizon] = float(payload["threshold"])
                        hmeta.append(dict(payload["search_meta"]))
                        resumed_final = True
                        print(f"[portfolio] H{horizon}: RESUME final-fit checkpoint", flush=True)
                if not resumed_final:
                    try:
                        p, r, _, smeta = choose_policy(
                            hpred, prices, horizon, final_hist_end, budget, max_workers,
                            fold_data=fold_data, progress_prefix=f"H{horizon} final-fit", prepared_pred=prepared_hpred,
                        )
                        final_policies[horizon] = p
                        final_thresholds[horizon] = float(r["resolved_threshold"])
                        final_meta = {"final_fit": True, **smeta}
                        hmeta.append(final_meta)
                        if store is not None:
                            store.put("final_fit_checkpoint", final_key, {
                                "policy": policy_dict(p), "threshold": float(r["resolved_threshold"]), "search_meta": final_meta,
                            })
                            store.flush()
                    except Exception as exc:
                        hmeta.append({"final_fit": True, "status": str(exc)})

            outer.extend(h_outer)
            history.extend(h_history)
            leaderboard[horizon] = h_board_all
            coverage[horizon] = hmeta
            if store is not None and horizon in final_policies:
                store.put("horizon_checkpoint", horizon_key, {
                    "outer": h_outer, "history": h_history, "leaderboard": h_board_all,
                    "search_coverage": hmeta, "final_policy": policy_dict(final_policies[horizon]),
                    "final_threshold": float(final_thresholds[horizon]),
                })
                store.flush()
                print(f"[portfolio] H{horizon}: horizon checkpoint committed", flush=True)

        with _FOLD_RESULT_CACHE_LOCK:
            memory_cache_stats = {
                "fold_replay_cache_entries": int(len(_FOLD_RESULT_CACHE)),
                "fold_replay_cache_hits": int(_FOLD_CACHE_HITS),
                "fold_replay_cache_misses": int(_FOLD_CACHE_MISSES),
            }
        fragment_stats = store.stats() if store is not None else {"enabled": False}
        print(f"[portfolio] search complete; memory fold cache {memory_cache_stats}; persistent fragments {fragment_stats}", flush=True)
        return outer, history, {
            "folds": folds, "final_policies": final_policies, "final_thresholds": final_thresholds,
            "leaderboard": leaderboard, "search_coverage": coverage, "max_workers": max(1, int(max_workers)),
            "performance": memory_cache_stats | {
                "price_panel_prepared_once": True,
                "horizon_signal_frame_prepared_once": True,
                "fold_signal_frames_prepared_once_per_horizon": True,
                "historical_views_reuse_prepared_horizon": True,
                "robustness_parallelized": True,
                "persistent_fragment_reuse": bool(store is not None),
                "persistent_fragment_cache": fragment_stats,
            },
        }
    finally:
        if store is not None:
            store.close()
        _FRAGMENT_STORE = None
