from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from pathlib import Path

import pandas as pd

from . import portfolio_policy_search as search
from .portfolio_policy_contracts import policy_dict
from .cost_contracts import CostModel
from .tax_contracts import TaxConfig
from .portfolio_replay_fragment_cache import FragmentStore
from .next_open_portfolio_replay import prepare_prices, prepare_signals, replay


class _SanitizingFragmentStore:
    """Execution-layer proxy that keeps worker telemetry out of research fragments."""

    def __init__(self, inner: FragmentStore) -> None:
        self._inner = inner
        self.path = inner.path
        self.namespace = inner.namespace

    @staticmethod
    def _clean(value):
        if isinstance(value, dict) and "_worker_telemetry" in value:
            value = dict(value)
            value.pop("_worker_telemetry", None)
        return value

    def get(self, kind: str, cache_key: str):
        found, value = self._inner.get(kind, cache_key)
        return found, self._clean(value)

    def put(self, kind: str, cache_key: str, value) -> None:
        self._inner.put(kind, cache_key, self._clean(value))

    def flush(self) -> None:
        self._inner.flush()

    def stats(self) -> dict:
        return self._inner.stats()

    def close(self) -> None:
        self._inner.close()


def _pipeline_workers(task_count: int, process_workers: int) -> int:
    if task_count <= 0:
        return 0
    raw = os.environ.get("OPPORTUNITY_WINDOW_PIPELINE", "4").strip()
    try:
        requested = int(raw)
    except ValueError:
        requested = 4
    return max(1, min(requested, int(process_workers), int(task_count), 8))


def _outer_window_task(
    *,
    horizon: int,
    i: int,
    hfolds: list[tuple[str, pd.Timestamp, pd.Timestamp]],
    hpred: pd.DataFrame,
    prices: pd.DataFrame,
    fold_data: list[dict],
    prepared_hpred: dict,
    budget: int,
    max_workers: int,
    store,
) -> tuple[int, dict]:
    fold, start, end = hfolds[i]
    hist_end = hfolds[i - 1][2]
    checkpoint_key = search._window_checkpoint_key(horizon, fold, hist_end, budget)
    progress_prefix = f"H{horizon} outer {i - 1}/{max(1, len(hfolds) - 2)}"
    try:
        policy, hist_result, board, smeta = search.choose_policy(
            hpred,
            prices,
            horizon,
            hist_end,
            budget,
            max_workers,
            fold_data=fold_data,
            progress_prefix=progress_prefix,
            prepared_pred=prepared_hpred,
        )
    except (ValueError, RuntimeError) as exc:
        return i, {
            "error_row": {"horizon": horizon, "test_fold": fold, "status": str(exc)}
        }

    test_fold = fold_data[i]
    threshold = float(hist_result.get("resolved_threshold", float("inf")))
    test_result = replay(
        test_fold["signals"],
        prices,
        policy,
        CostModel(20),
        TaxConfig(False),
        start=start,
        end=end,
        initial=10000.0,
        resolved_threshold=threshold,
        prepared_signals=test_fold["prepared"],
    )
    metrics = test_result["metrics"]
    outer_row = {
        "horizon": horizon,
        "fold_id": fold,
        "fold_start": str(start.date()),
        "fold_end": str(end.date()),
        "calibration_end": str(hist_end.date()),
        "policy_id": policy.policy_id,
        **policy_dict(policy),
        "threshold": threshold,
        **metrics,
    }
    history_rows = [
        {
            **row,
            "test_fold": fold,
            "calibration_end": str(hist_end.date()),
            "selected": row.get("policy_id") == policy.policy_id,
        }
        for row in board
    ]
    search_meta_row = {"test_fold": fold, **smeta}
    payload = {
        "outer_row": outer_row,
        "history_rows": history_rows,
        "leaderboard_rows": board,
        "search_meta": search_meta_row,
    }
    if store is not None:
        store.put("outer_window_checkpoint", checkpoint_key, payload)
        store.flush()
    return i, payload


def _final_fit_task(
    *,
    horizon: int,
    final_hist_end: pd.Timestamp,
    hpred: pd.DataFrame,
    prices: pd.DataFrame,
    fold_data: list[dict],
    prepared_hpred: dict,
    budget: int,
    max_workers: int,
    store,
) -> dict:
    final_key = search._final_fit_checkpoint_key(horizon, final_hist_end, budget)
    try:
        policy, result, _, smeta = search.choose_policy(
            hpred,
            prices,
            horizon,
            final_hist_end,
            budget,
            max_workers,
            fold_data=fold_data,
            progress_prefix=f"H{horizon} final-fit",
            prepared_pred=prepared_hpred,
        )
    except Exception as exc:
        return {"error_meta": {"final_fit": True, "status": str(exc)}}

    threshold = float(result["resolved_threshold"])
    final_meta = {"final_fit": True, **smeta}
    if store is not None:
        store.put(
            "final_fit_checkpoint",
            final_key,
            {
                "policy": policy_dict(policy),
                "threshold": threshold,
                "search_meta": final_meta,
            },
        )
        store.flush()
    return {"policy": policy, "threshold": threshold, "search_meta": final_meta}


def run_walk_forward(
    pred: pd.DataFrame,
    prices: pd.DataFrame,
    horizons=(5, 10, 20),
    budget=48,
    max_workers: int = 4,
    fragment_cache_path: str | Path | None = None,
    fragment_namespace: str | None = None,
) -> tuple[list[dict], list[dict], dict]:
    """Execution-equivalent walk-forward with concurrent independent windows.

    The research contract remains in search.py. This execution layer only overlaps
    independent outer windows (and final-fit) so their replay batches feed the same
    long-lived ProcessPool concurrently. Results and checkpoints are reassembled in
    the original chronological order.
    """

    with search._FOLD_RESULT_CACHE_LOCK:
        search._FOLD_RESULT_CACHE.clear()
        search._FOLD_CACHE_HITS = 0
        search._FOLD_CACHE_MISSES = 0

    if fragment_cache_path is None:
        configured = os.environ.get("OPPORTUNITY_FRAGMENT_CACHE_PATH", "").strip()
        if configured:
            fragment_cache_path = configured
    if fragment_cache_path is not None and not fragment_namespace:
        input_fingerprint = search._research_input_fingerprint(pred, prices)
        fragment_namespace = search.build_fragment_namespace(input_fingerprint)
        print(
            f"[portfolio] fragment input fingerprint: {input_fingerprint[:16]}...",
            flush=True,
        )

    store = None
    if fragment_cache_path is not None and fragment_namespace:
        raw_store = FragmentStore(Path(fragment_cache_path), str(fragment_namespace))
        store = _SanitizingFragmentStore(raw_store)
        search._FRAGMENT_STORE = store
        stats = store.stats()
        print(
            f"[portfolio] fragment cache: {stats['entries_loaded_at_start']} reusable fragments loaded from {stats['path']}",
            flush=True,
        )
    else:
        search._FRAGMENT_STORE = None

    prepare_prices(prices)
    outer: list[dict] = []
    history: list[dict] = []
    leaderboard: dict = {}
    final_policies: dict = {}
    final_thresholds: dict = {}
    coverage: dict = {}
    pipeline_meta: dict[int, dict] = {}
    folds = search._fold_ranges(pred)

    try:
        for horizon in horizons:
            horizon_key = search._horizon_checkpoint_key(horizon, budget)
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
                        final_policies[horizon] = search._policy_from_dict(payload["final_policy"])
                        final_thresholds[horizon] = float(payload["final_threshold"])
                    pipeline_meta[horizon] = {
                        "resumed_horizon": True,
                        "coordinator_workers": 0,
                        "outer_windows_computed": 0,
                    }
                    print(
                        f"[portfolio] H{horizon}: RESUME complete horizon checkpoint ({len(h_outer)} outer rows)",
                        flush=True,
                    )
                    continue

            hpred = pred.loc[pred["horizon"] == horizon].copy()
            prepared_hpred = prepare_signals(hpred)
            fold_data = search._prepare_fold_data(hpred)
            hfolds = [
                (x["fold_id"], x["start"], x["end"])
                for x in fold_data
            ]
            h_outer: list[dict] = []
            h_history: list[dict] = []
            h_board_all: list[dict] = []
            hmeta: list[dict] = []
            print(
                f"[portfolio] H{horizon}: {len(hfolds)} folds prepared, "
                f"{prepared_hpred.get('row_count', 0)} signal rows; workers={max_workers}",
                flush=True,
            )

            window_results: dict[int, dict] = {}
            pending_windows: list[int] = []
            for i, (fold, _start, _end) in enumerate(hfolds):
                if i < 2:
                    continue
                hist_end = hfolds[i - 1][2]
                checkpoint_key = search._window_checkpoint_key(
                    horizon, fold, hist_end, budget
                )
                if store is not None:
                    found, payload = store.get(
                        "outer_window_checkpoint", checkpoint_key
                    )
                    if found:
                        window_results[i] = dict(payload)
                        print(
                            f"[portfolio] H{horizon} fold {fold}: RESUME outer-window checkpoint",
                            flush=True,
                        )
                        continue
                pending_windows.append(i)

            final_payload = None
            final_hist_end = hfolds[-1][2] if hfolds else None
            final_missing = False
            if final_hist_end is not None:
                final_key = search._final_fit_checkpoint_key(
                    horizon, final_hist_end, budget
                )
                if store is not None:
                    found, payload = store.get("final_fit_checkpoint", final_key)
                    if found:
                        payload = dict(payload)
                        final_payload = {
                            "policy": search._policy_from_dict(payload["policy"]),
                            "threshold": float(payload["threshold"]),
                            "search_meta": dict(payload["search_meta"]),
                        }
                        print(
                            f"[portfolio] H{horizon}: RESUME final-fit checkpoint",
                            flush=True,
                        )
                    else:
                        final_missing = True
                else:
                    final_missing = True

            task_count = len(pending_windows) + int(final_missing)
            coordinator_workers = _pipeline_workers(task_count, max_workers)
            pipeline_meta[horizon] = {
                "resumed_horizon": False,
                "coordinator_workers": int(coordinator_workers),
                "outer_windows_pending": int(len(pending_windows)),
                "outer_windows_resumed": int(max(0, len(hfolds) - 2 - len(pending_windows))),
                "final_fit_overlapped": bool(
                    final_missing and pending_windows and coordinator_workers > 1
                ),
            }
            if task_count:
                print(
                    f"[multicore] H{horizon} window pipeline: "
                    f"outer={len(pending_windows)}, final_fit={int(final_missing)}, "
                    f"coordinators={coordinator_workers}",
                    flush=True,
                )

            if coordinator_workers <= 1:
                for i in pending_windows:
                    result_i, payload = _outer_window_task(
                        horizon=horizon,
                        i=i,
                        hfolds=hfolds,
                        hpred=hpred,
                        prices=prices,
                        fold_data=fold_data,
                        prepared_hpred=prepared_hpred,
                        budget=budget,
                        max_workers=max_workers,
                        store=store,
                    )
                    window_results[result_i] = payload
                if final_missing and final_hist_end is not None:
                    final_payload = _final_fit_task(
                        horizon=horizon,
                        final_hist_end=final_hist_end,
                        hpred=hpred,
                        prices=prices,
                        fold_data=fold_data,
                        prepared_hpred=prepared_hpred,
                        budget=budget,
                        max_workers=max_workers,
                        store=store,
                    )
            elif coordinator_workers > 1:
                with ThreadPoolExecutor(
                    max_workers=coordinator_workers,
                    thread_name_prefix=f"H{horizon}-window",
                ) as coordinator:
                    future_meta = {}
                    for i in pending_windows:
                        future = coordinator.submit(
                            _outer_window_task,
                            horizon=horizon,
                            i=i,
                            hfolds=hfolds,
                            hpred=hpred,
                            prices=prices,
                            fold_data=fold_data,
                            prepared_hpred=prepared_hpred,
                            budget=budget,
                            max_workers=max_workers,
                            store=store,
                        )
                        future_meta[future] = ("outer", i)
                    if final_missing and final_hist_end is not None:
                        future = coordinator.submit(
                            _final_fit_task,
                            horizon=horizon,
                            final_hist_end=final_hist_end,
                            hpred=hpred,
                            prices=prices,
                            fold_data=fold_data,
                            prepared_hpred=prepared_hpred,
                            budget=budget,
                            max_workers=max_workers,
                            store=store,
                        )
                        future_meta[future] = ("final", None)

                    for future in as_completed(future_meta):
                        kind, _index = future_meta[future]
                        value = future.result()
                        if kind == "outer":
                            result_i, payload = value
                            window_results[result_i] = payload
                        else:
                            final_payload = value

            # Reassemble exactly as the original chronological loop did.
            for i in range(2, len(hfolds)):
                payload = window_results.get(i)
                if payload is None:
                    raise RuntimeError(
                        f"MULTICORE_WINDOW_RESULT_MISSING:H{horizon}:index={i}"
                    )
                if payload.get("error_row") is not None:
                    h_history.append(dict(payload["error_row"]))
                    continue
                h_outer.append(dict(payload["outer_row"]))
                h_history.extend(list(payload.get("history_rows", [])))
                h_board_all.extend(list(payload.get("leaderboard_rows", [])))
                hmeta.append(dict(payload.get("search_meta", {})))

            if final_payload is not None:
                if final_payload.get("error_meta") is not None:
                    hmeta.append(dict(final_payload["error_meta"]))
                else:
                    policy = final_payload["policy"]
                    final_policies[horizon] = policy
                    final_thresholds[horizon] = float(final_payload["threshold"])
                    hmeta.append(dict(final_payload["search_meta"]))

            outer.extend(h_outer)
            history.extend(h_history)
            leaderboard[horizon] = h_board_all
            coverage[horizon] = hmeta
            if store is not None and horizon in final_policies:
                store.put(
                    "horizon_checkpoint",
                    horizon_key,
                    {
                        "outer": h_outer,
                        "history": h_history,
                        "leaderboard": h_board_all,
                        "search_coverage": hmeta,
                        "final_policy": policy_dict(final_policies[horizon]),
                        "final_threshold": float(final_thresholds[horizon]),
                    },
                )
                store.flush()
                print(
                    f"[portfolio] H{horizon}: horizon checkpoint committed",
                    flush=True,
                )

        with search._FOLD_RESULT_CACHE_LOCK:
            memory_cache_stats = {
                "fold_replay_cache_entries": int(len(search._FOLD_RESULT_CACHE)),
                "fold_replay_cache_hits": int(search._FOLD_CACHE_HITS),
                "fold_replay_cache_misses": int(search._FOLD_CACHE_MISSES),
            }
        fragment_stats = store.stats() if store is not None else {"enabled": False}
        print(
            f"[portfolio] search complete; memory fold cache {memory_cache_stats}; "
            f"persistent fragments {fragment_stats}",
            flush=True,
        )
        return outer, history, {
            "folds": folds,
            "final_policies": final_policies,
            "final_thresholds": final_thresholds,
            "leaderboard": leaderboard,
            "search_coverage": coverage,
            "max_workers": max(1, int(max_workers)),
            "performance": memory_cache_stats
            | {
                "price_panel_prepared_once": True,
                "horizon_signal_frame_prepared_once": True,
                "fold_signal_frames_prepared_once_per_horizon": True,
                "historical_views_reuse_prepared_horizon": True,
                "robustness_parallelized": True,
                "outer_window_pipeline": True,
                "outer_window_pipeline_by_horizon": pipeline_meta,
                "persistent_fragment_reuse": bool(store is not None),
                "persistent_fragment_cache": fragment_stats,
            },
        }
    finally:
        if store is not None:
            store.close()
        search._FRAGMENT_STORE = None
