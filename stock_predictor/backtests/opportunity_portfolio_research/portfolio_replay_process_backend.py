from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing as mp
import os
from pathlib import Path
import time
from threading import Lock
from typing import Callable

import pandas as pd

from .portfolio_policy_contracts import Policy
from .cost_contracts import CostModel
from .tax_contracts import TaxConfig
from .cpu_topology import physical_core_affinity_plan, set_current_process_logical_affinity
from .next_open_portfolio_replay import prepare_prices, prepare_signals, replay
from . import portfolio_policy_search as search

# Worker-local immutable research data. Under Windows spawn these frames are copied
# exactly once when a horizon pool is created; individual jobs only send small policy
# descriptors, dates and thresholds.
_WORKER_SIGNALS: pd.DataFrame | None = None
_WORKER_PRICES: pd.DataFrame | None = None
_WORKER_PREPARED_SIGNALS: dict | None = None
_WORKER_FOLDS: dict[str, dict] = {}

_POOL: ProcessPoolExecutor | None = None
_POOL_SIGNATURE: tuple | None = None
_POOL_SLOT_COUNTER = None
_CONTEXT_SIGNALS: pd.DataFrame | None = None
_CONTEXT_PRICES: pd.DataFrame | None = None
_ORIGINAL_PARALLEL_MAP = None
_BACKEND_LOCK = Lock()
_TELEMETRY_LOCK = Lock()
# The backend is deliberately bounded below the host's logical-CPU count.  On
# the current 32-logical / 24-physical-core workstation this gives one process
# per physical core first; the topology planner keeps SMT siblings out of the
# normal 24-worker allocation.
MAX_PROCESS_WORKERS = 24
DEFAULT_PROCESS_WORKERS = 24
_PROCESS_WORKERS = DEFAULT_PROCESS_WORKERS
_AFFINITY_PLAN: dict = {}
_TELEMETRY_PATH: Path | None = None
_TELEMETRY_SUITE_STARTED = 0.0
_TELEMETRY_TOTALS = {
    "batches": 0,
    "jobs_submitted": 0,
    "jobs_reused": 0,
    "jobs_computed": 0,
    "pool_recoveries": 0,
    "main_cache_seconds": 0.0,
    "main_aggregation_seconds": 0.0,
    "worker_job_counts": {},
    "worker_compute_seconds": {},
    "worker_idle_seconds": {},
}
_WORKER_INDEX = -1
_WORKER_LAST_FINISH = 0.0
_STATS = {
    "backend": "process",
    "pool_starts": 0,
    "pool_restarts": 0,
    "history_jobs_reused": 0,
    "history_jobs_computed": 0,
    "fold_jobs_reused": 0,
    "fold_jobs_computed": 0,
}


def _worker_init(
    signals: pd.DataFrame,
    prices: pd.DataFrame,
    worker_map: list[dict],
    slot_counter,
) -> None:
    global _WORKER_SIGNALS, _WORKER_PRICES, _WORKER_PREPARED_SIGNALS, _WORKER_FOLDS, _WORKER_INDEX, _WORKER_LAST_FINISH
    for name in (
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS",
    ):
        os.environ[name] = "1"

    affinity_row = None
    affinity_ok = False
    if worker_map and slot_counter is not None:
        with slot_counter.get_lock():
            slot = int(slot_counter.value)
            slot_counter.value += 1
        _WORKER_INDEX = slot % len(worker_map)
        affinity_row = worker_map[_WORKER_INDEX]
        affinity_ok = set_current_process_logical_affinity(
            int(affinity_row["logical_processor"])
        )
        print(
            "[multicore-worker] "
            f"pid={os.getpid()} slot={slot} role={affinity_row.get('role', 'worker')} "
            f"core={affinity_row['core_index']} logical_cpu={affinity_row['logical_processor']} "
            f"smt_siblings={affinity_row.get('smt_siblings', [])} "
            f"affinity={'OK' if affinity_ok else 'FALLBACK'}",
            flush=True,
        )

    _WORKER_SIGNALS = signals
    _WORKER_PRICES = prices
    prepare_prices(prices)
    _WORKER_PREPARED_SIGNALS = prepare_signals(signals)
    _WORKER_FOLDS = {x["fold_id"]: x for x in search._prepare_fold_data(signals)}
    _WORKER_LAST_FINISH = time.perf_counter()


def _worker_result(result: dict | None, started: float, idle_started: float) -> dict:
    global _WORKER_LAST_FINISH
    finished = time.perf_counter()
    payload = dict(result) if result is not None else {"_worker_empty_result": True}
    payload["_worker_telemetry"] = {
        "worker_index": int(_WORKER_INDEX),
        "worker_pid": int(os.getpid()),
        "compute_seconds": max(0.0, finished - started),
        "idle_seconds_before_job": max(0.0, started - idle_started),
    }
    _WORKER_LAST_FINISH = finished
    return payload


def _clean_worker_result(value) -> tuple[object, dict]:
    """Remove execution-only metadata while preserving telemetry for empty results."""
    if not isinstance(value, dict):
        return value, {}
    payload = dict(value)
    telemetry = dict(payload.pop("_worker_telemetry", {}) or {})
    empty = bool(payload.pop("_worker_empty_result", False))
    return (None if empty else payload), telemetry


def _worker_history_replay(job: tuple[Policy, pd.Timestamp, float]) -> dict:
    idle_started = _WORKER_LAST_FINISH or time.perf_counter()
    started = time.perf_counter()
    policy, history_end, threshold = job
    if _WORKER_SIGNALS is None or _WORKER_PRICES is None or _WORKER_PREPARED_SIGNALS is None:
        raise RuntimeError("MULTICORE_WORKER_NOT_INITIALIZED")
    result = replay(
        _WORKER_SIGNALS,
        _WORKER_PRICES,
        policy,
        CostModel(20),
        TaxConfig(False),
        end=pd.Timestamp(history_end),
        initial=10000.0,
        resolved_threshold=float(threshold),
        prepared_signals=_WORKER_PREPARED_SIGNALS,
    )
    return _worker_result(search._compact_search_result(result, float(threshold)), started, idle_started)


def _worker_fold_metric(job: tuple[str, Policy, float]) -> dict | None:
    idle_started = _WORKER_LAST_FINISH or time.perf_counter()
    started = time.perf_counter()
    fold_id, policy, threshold = job
    if _WORKER_PRICES is None:
        raise RuntimeError("MULTICORE_WORKER_NOT_INITIALIZED")
    fold = _WORKER_FOLDS.get(str(fold_id))
    if fold is None:
        raise RuntimeError(f"MULTICORE_FOLD_NOT_FOUND:{fold_id}")
    result = replay(
        fold["signals"],
        _WORKER_PRICES,
        policy,
        CostModel(20),
        TaxConfig(False),
        start=fold["start"],
        end=fold["end"],
        initial=10000.0,
        resolved_threshold=float(threshold),
        prepared_signals=fold["prepared"],
    )
    metrics = result["metrics"]
    if metrics.get("trade_count", 0) <= 0:
        return _worker_result(None, started, idle_started)
    return _worker_result({
        "fold_id": str(fold_id),
        "cagr_excess": float(metrics.get("cagr_excess", 0.0)),
        "worst_relative_drawdown": float(metrics.get("worst_relative_drawdown", 0.0)),
        "turnover": float(metrics.get("turnover", 0.0)),
        "trade_count": int(metrics.get("trade_count", 0)),
    }, started, idle_started)


def _worker_evidence_replay(job: dict) -> dict:
    """Replay one already-selected outer-fold policy for Evidence Expansion."""
    idle_started = _WORKER_LAST_FINISH or time.perf_counter()
    started = time.perf_counter()
    if _WORKER_SIGNALS is None or _WORKER_PRICES is None:
        raise RuntimeError("MULTICORE_WORKER_NOT_INITIALIZED")
    horizon = int(job["horizon"])
    fold_id = str(job["fold_id"])
    signals = _WORKER_SIGNALS.loc[
        (_WORKER_SIGNALS["horizon"] == horizon)
        & (_WORKER_SIGNALS["fold_id"].astype(str) == fold_id)
    ].copy()
    policy = search._policy_from_dict(dict(job["policy"]))
    result = replay(
        signals,
        _WORKER_PRICES,
        policy,
        CostModel(float(job["cost_bps"])),
        TaxConfig(bool(job["tax_enabled"])),
        start=pd.Timestamp(job["start"]),
        end=pd.Timestamp(job["end"]),
        initial=float(job.get("initial", 10000.0)),
        resolved_threshold=float(job["threshold"]),
        prepared_signals=prepare_signals(signals),
    )
    payload = {
        "job_key": str(job["job_key"]),
        "horizon": horizon,
        "fold_id": fold_id,
        "cost_bps": float(job["cost_bps"]),
        "tax_world": "DE_RETAIL_TAX_AWARE" if job["tax_enabled"] else "PRE_TAX",
        "metrics": dict(result.get("metrics", {})),
        "trades": [dict(trade) for trade in result.get("trades", [])],
        "open_positions": [dict(position) for position in result.get("open_positions", [])],
    }
    return _worker_result(payload, started, idle_started)


def _write_telemetry(record: dict) -> None:
    if _TELEMETRY_PATH is None:
        return
    with _TELEMETRY_LOCK:
        _TELEMETRY_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _TELEMETRY_PATH.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True, default=str) + "\n")


def _write_cumulative_telemetry(batch_record: dict) -> None:
    """Append suite-wide worker totals after every completed batch.

    Batch records remain available for diagnosis, while these records provide one
    continuously updated view of every worker across all batches.  Idle time is
    the measured time before a worker starts its next job; utilization is compute
    divided by measured compute plus measured idle time.
    """
    global _TELEMETRY_TOTALS
    totals = _TELEMETRY_TOTALS
    totals["batches"] += 1
    totals["jobs_submitted"] += int(batch_record.get("jobs_submitted", 0)) + int(batch_record.get("jobs_reused", 0))
    totals["jobs_reused"] += int(batch_record.get("jobs_reused", 0))
    totals["jobs_computed"] += int(batch_record.get("jobs_computed", 0))
    totals["pool_recoveries"] += int(batch_record.get("pool_recoveries", 0))
    totals["main_cache_seconds"] += float(batch_record.get("main_cache_seconds", 0.0) or 0.0)
    totals["main_aggregation_seconds"] += float(batch_record.get("main_aggregation_seconds", 0.0) or 0.0)
    for worker, count in dict(batch_record.get("worker_job_counts", {})).items():
        totals["worker_job_counts"][str(worker)] = totals["worker_job_counts"].get(str(worker), 0) + int(count)
    for field in ("worker_compute_seconds", "worker_idle_seconds"):
        source = "worker_compute_time" if field == "worker_compute_seconds" else "worker_idle_time"
        for worker, seconds in dict(batch_record.get(source, {})).items():
            key = str(worker)
            totals[field][key] = totals[field].get(key, 0.0) + float(seconds)

    workers = set(str(i) for i in range(max(1, int(_PROCESS_WORKERS))))
    workers.update(totals["worker_job_counts"])
    workers.update(totals["worker_compute_seconds"])
    workers.update(totals["worker_idle_seconds"])
    compute = {worker: float(totals["worker_compute_seconds"].get(worker, 0.0)) for worker in sorted(workers, key=lambda x: (x == "unknown", int(x) if x.isdigit() else 10**9))}
    idle = {worker: float(totals["worker_idle_seconds"].get(worker, 0.0)) for worker in compute}
    utilization = {
        worker: (compute[worker] / (compute[worker] + idle[worker]) if compute[worker] + idle[worker] > 0 else 0.0)
        for worker in compute
    }
    record = {
        "event": "cumulative_telemetry",
        "scope": "entire_suite_to_date",
        "suite_elapsed_seconds": max(0.0, time.perf_counter() - _TELEMETRY_SUITE_STARTED) if _TELEMETRY_SUITE_STARTED else None,
        "batches_completed": int(totals["batches"]),
        "jobs_submitted": int(totals["jobs_submitted"]),
        "jobs_reused": int(totals["jobs_reused"]),
        "jobs_computed": int(totals["jobs_computed"]),
        "pool_recoveries": int(totals["pool_recoveries"]),
        "main_cache_seconds": float(totals["main_cache_seconds"]),
        "main_aggregation_seconds": float(totals["main_aggregation_seconds"]),
        "worker_job_counts": {worker: int(totals["worker_job_counts"].get(worker, 0)) for worker in compute},
        "worker_compute_seconds": compute,
        "worker_idle_seconds": idle,
        "worker_utilization": utilization,
        "worker_measured_busy_idle_seconds": {
            worker: compute[worker] + idle[worker] for worker in compute
        },
    }
    _write_telemetry(record)


def _expected_worker_slots() -> list[str]:
    """Return the complete configured worker slot set for every telemetry record.

    A slot that has not received a job in a batch must still be represented as
    zero rather than disappearing from the record.  This is especially
    important for the SMT-overflow slots 8-11: omission was previously
    indistinguishable from broken worker attribution in downstream reports.
    """
    return [str(i) for i in range(max(1, int(_PROCESS_WORKERS)))]


def _set_context(signals: pd.DataFrame, prices: pd.DataFrame) -> None:
    global _CONTEXT_SIGNALS, _CONTEXT_PRICES
    _CONTEXT_SIGNALS = signals
    _CONTEXT_PRICES = prices


def _effective_workers(requested: int) -> int:
    limit = max(
        1,
        min(
            int(requested),
            int(_PROCESS_WORKERS),
            int(os.cpu_count() or 1),
            MAX_PROCESS_WORKERS,
        ),
    )
    if _AFFINITY_PLAN.get("enabled"):
        limit = min(limit, int(_AFFINITY_PLAN.get("workers", limit)))
    return max(1, limit)


def _ensure_pool(signals: pd.DataFrame, prices: pd.DataFrame, workers: int) -> ProcessPoolExecutor:
    global _POOL, _POOL_SIGNATURE, _POOL_SLOT_COUNTER
    workers = _effective_workers(workers)
    worker_map = list(_AFFINITY_PLAN.get("worker_map") or [])[:workers]
    affinity_signature = tuple(
        (int(x["core_index"]), int(x["logical_processor"])) for x in worker_map
    )
    signature = (id(signals), id(prices), workers, affinity_signature)
    with _BACKEND_LOCK:
        if _POOL is not None and _POOL_SIGNATURE == signature:
            return _POOL
        if _POOL is not None:
            _POOL.shutdown(wait=True, cancel_futures=False)
            _STATS["pool_restarts"] += 1
        ctx = mp.get_context("spawn")
        _POOL_SLOT_COUNTER = ctx.Value("i", 0, lock=True)
        _POOL = ProcessPoolExecutor(
            max_workers=workers,
            mp_context=ctx,
            initializer=_worker_init,
            initargs=(signals, prices, worker_map, _POOL_SLOT_COUNTER),
        )
        _POOL_SIGNATURE = signature
        _STATS["pool_starts"] += 1
        if worker_map:
            mapping = ", ".join(
                f"{x.get('role', 'worker')}:core{x['core_index']}->cpu{x['logical_processor']}"
                for x in worker_map
            )
            print(f"[multicore] worker affinity: {mapping}", flush=True)
            print(
                f"[multicore] coordinator reserve logical CPUs: "
                f"{_AFFINITY_PLAN.get('reserved_logical_processors', [])}",
                flush=True,
            )
        else:
            print("[multicore] affinity unavailable; Windows scheduler fallback", flush=True)
        print(
            f"[multicore] process pool started: workers={workers}, "
            f"horizon_rows={len(signals):,}, price_rows={len(prices):,}",
            flush=True,
        )
        return _POOL


def _history_cache_key(policy: Policy, history_end: pd.Timestamp, threshold: float) -> str:
    return search._cache_key(
        "history_replay",
        policy.horizon,
        policy.policy_id,
        str(pd.Timestamp(history_end)),
        float(threshold).hex(),
        20.0,
        False,
        10000.0,
    )


def _fold_keys(fold: dict, policy: Policy, threshold: float) -> tuple[tuple, str]:
    stable = (
        policy.horizon,
        fold["fold_id"],
        str(fold["start"]),
        str(fold["end"]),
        policy.policy_id,
        float(threshold).hex(),
    )
    disk = search._cache_key("fold_metric", *stable, 20.0, False, 10000.0)
    return stable, disk


def _run_process_futures(
    pool: ProcessPoolExecutor,
    jobs: list[tuple[int, object]],
    worker_func,
    label: str,
    on_result: Callable[[int, object], None],
    jobs_reused: int = 0,
    main_cache_seconds: float = 0.0,
    horizon: int | None = None,
) -> None:
    if not jobs:
        return
    futures = {pool.submit(worker_func, payload): index for index, payload in jobs}
    started = time.perf_counter()
    first_finished = None
    worker_counts: dict[str, int] = {worker: 0 for worker in _expected_worker_slots()}
    worker_compute: dict[str, float] = {worker: 0.0 for worker in _expected_worker_slots()}
    worker_idle: dict[str, float] = {worker: 0.0 for worker in _expected_worker_slots()}
    worker_pids: dict[str, int] = {}
    main_aggregation_seconds = 0.0
    total = len(futures)
    step = max(1, total // 8)
    completed = 0
    for future in as_completed(futures):
        index = futures[future]
        value = future.result()
        finished = time.perf_counter()
        if first_finished is None:
            first_finished = finished
        value, telemetry = _clean_worker_result(value)
        worker = str(telemetry.get("worker_index", "unknown"))
        worker_counts[worker] = worker_counts.get(worker, 0) + 1
        worker_compute[worker] = worker_compute.get(worker, 0.0) + float(telemetry.get("compute_seconds", 0.0))
        worker_idle[worker] = worker_idle.get(worker, 0.0) + float(telemetry.get("idle_seconds_before_job", 0.0))
        if telemetry.get("worker_pid") is not None and worker != "unknown":
            worker_pids[worker] = int(telemetry["worker_pid"])
        aggregation_started = time.perf_counter()
        on_result(index, value)
        main_aggregation_seconds += time.perf_counter() - aggregation_started
        completed += 1
        if completed == total or completed % step == 0:
            print(f"[{label}] computed+checkpointed {completed}/{total}", flush=True)
    batch_record = {
        "batch": label,
        "horizon": horizon,
        "jobs_submitted": total,
        "jobs_reused": int(jobs_reused),
        "jobs_computed": total,
        "time_first_job_finished_seconds": (first_finished - started) if first_finished is not None else None,
        "time_last_job_finished_seconds": time.perf_counter() - started,
        "worker_job_counts": worker_counts,
        "worker_compute_time": worker_compute,
        "worker_idle_time": worker_idle,
        "worker_pids": worker_pids,
        "worker_slots_expected": _expected_worker_slots(),
        "worker_slots_with_jobs": sorted(
            [worker for worker, count in worker_counts.items() if count > 0],
            key=lambda value: int(value) if value.isdigit() else 10**9,
        ),
        "main_cache_seconds": float(main_cache_seconds),
        "main_aggregation_seconds": float(main_aggregation_seconds),
    }
    _write_telemetry(batch_record)
    _write_cumulative_telemetry(batch_record)


def _parallel_history(args: list, max_workers: int, label: str) -> list:
    if not args:
        return []
    signals, prices = args[0][0], args[0][1]
    _set_context(signals, prices)
    results: list[object | None] = [None] * len(args)
    misses: list[tuple[int, tuple[Policy, pd.Timestamp, float]]] = []
    store = search._FRAGMENT_STORE
    cache_started = time.perf_counter()

    for index, arg in enumerate(args):
        _, _, _, policy, history_end, threshold = arg
        key = _history_cache_key(policy, history_end, threshold)
        if store is not None:
            found, payload = store.get("history_replay", key)
            if found:
                results[index] = (policy, dict(payload))
                _STATS["history_jobs_reused"] += 1
                continue
        misses.append((index, (policy, pd.Timestamp(history_end), float(threshold))))

    workers = _effective_workers(max_workers)
    print(
        f"[{label}] process backend: reused={len(args) - len(misses)}, "
        f"compute={len(misses)}, workers={workers}",
        flush=True,
    )
    if misses:
        pool = _ensure_pool(signals, prices, workers)

        def commit(index: int, compact: object) -> None:
            _, _, _, policy, history_end, threshold = args[index]
            results[index] = (policy, compact)
            if store is not None:
                store.put(
                    "history_replay",
                    _history_cache_key(policy, history_end, threshold),
                    compact,
                )
            _STATS["history_jobs_computed"] += 1

        _run_process_futures(
            pool, misses, _worker_history_replay, label, commit,
            jobs_reused=len(args) - len(misses),
            main_cache_seconds=time.perf_counter() - cache_started,
        )
    return list(results)


def _parallel_robustness(func, args: list, max_workers: int, label: str) -> list:
    if not args:
        return []
    store = search._FRAGMENT_STORE
    missing: dict[tuple, tuple[str, Policy, float, str]] = {}
    reused = 0
    cache_started = time.perf_counter()

    for fold_data, _prices, policy, result, history_end in args:
        threshold = float(result["resolved_threshold"])
        for fold in fold_data:
            if fold["end"] > history_end:
                continue
            stable, disk = _fold_keys(fold, policy, threshold)
            with search._FOLD_RESULT_CACHE_LOCK:
                if stable in search._FOLD_RESULT_CACHE:
                    reused += 1
                    continue
            if store is not None:
                found, payload = store.get("fold_metric", disk)
                if found:
                    with search._FOLD_RESULT_CACHE_LOCK:
                        search._FOLD_RESULT_CACHE[stable] = payload
                    reused += 1
                    continue
            if stable not in missing:
                with search._FOLD_RESULT_CACHE_LOCK:
                    search._FOLD_CACHE_MISSES += 1
                missing[stable] = (str(fold["fold_id"]), policy, threshold, disk)

    workers = _effective_workers(max_workers)
    print(
        f"[{label}] process backend: reused_fold_fragments={reused}, "
        f"compute_fold_fragments={len(missing)}, workers={workers}",
        flush=True,
    )

    if missing:
        signals = _CONTEXT_SIGNALS
        prices = _CONTEXT_PRICES
        if signals is None or prices is None:
            fold_data = args[0][0]
            prices = args[0][1]
            signals = pd.concat([x["signals"] for x in fold_data], ignore_index=True)
            _set_context(signals, prices)
        pool = _ensure_pool(signals, prices, workers)
        ordered = list(missing.items())
        jobs = [
            (i, (fold_id, policy, threshold))
            for i, (_stable, (fold_id, policy, threshold, _disk)) in enumerate(ordered)
        ]

        def commit(i: int, value: object) -> None:
            stable, (_fold_id, _policy, _threshold, disk) = ordered[i]
            with search._FOLD_RESULT_CACHE_LOCK:
                search._FOLD_RESULT_CACHE[stable] = value
            if store is not None:
                store.put("fold_metric", disk, value)
            _STATS["fold_jobs_computed"] += 1

        _run_process_futures(
            pool, jobs, _worker_fold_metric, label, commit,
            jobs_reused=reused,
            main_cache_seconds=time.perf_counter() - cache_started,
        )
    _STATS["fold_jobs_reused"] += reused
    return [func(arg) for arg in args]


def multicore_parallel_map(func, args: list, max_workers: int, label: str) -> list:
    workers = _effective_workers(max_workers)
    if workers <= 1:
        return _ORIGINAL_PARALLEL_MAP(func, args, 1, label)
    if func is search._evaluate_policy or getattr(func, "__name__", "") == "_evaluate_policy":
        return _parallel_history(args, workers, label)
    if func is search._evaluate_robustness or getattr(func, "__name__", "") == "_evaluate_robustness":
        return _parallel_robustness(func, args, workers, label)
    return _ORIGINAL_PARALLEL_MAP(func, args, workers, label)


def install_multicore_backend(process_workers: int = DEFAULT_PROCESS_WORKERS) -> None:
    global _ORIGINAL_PARALLEL_MAP, _PROCESS_WORKERS, _AFFINITY_PLAN, _TELEMETRY_PATH
    global _TELEMETRY_SUITE_STARTED, _TELEMETRY_TOTALS
    if _ORIGINAL_PARALLEL_MAP is None:
        _ORIGINAL_PARALLEL_MAP = search._parallel_map
    _PROCESS_WORKERS = max(
        1,
        min(int(process_workers), int(os.cpu_count() or 1), MAX_PROCESS_WORKERS),
    )
    _AFFINITY_PLAN = physical_core_affinity_plan(
        _PROCESS_WORKERS,
        reserve_logical_processors=4,
    )
    telemetry = os.environ.get("OPPORTUNITY_TELEMETRY_PATH", "").strip()
    _TELEMETRY_PATH = Path(telemetry) if telemetry else None
    _TELEMETRY_SUITE_STARTED = time.perf_counter()
    _TELEMETRY_TOTALS = {
        "batches": 0,
        "jobs_submitted": 0,
        "jobs_reused": 0,
        "jobs_computed": 0,
        "pool_recoveries": 0,
        "main_cache_seconds": 0.0,
        "main_aggregation_seconds": 0.0,
        "worker_job_counts": {},
        "worker_compute_seconds": {},
        "worker_idle_seconds": {},
    }
    if _AFFINITY_PLAN.get("enabled"):
        _PROCESS_WORKERS = min(_PROCESS_WORKERS, int(_AFFINITY_PLAN["workers"]))
    search._parallel_map = multicore_parallel_map
    for key in (
        "pool_starts", "pool_restarts", "history_jobs_reused", "history_jobs_computed",
        "fold_jobs_reused", "fold_jobs_computed",
    ):
        _STATS[key] = 0
    _STATS["backend"] = "process"
    topo = _AFFINITY_PLAN.get("topology", {})
    print(
        f"[multicore] backend installed; process_workers={_PROCESS_WORKERS}; "
        f"max_process_workers={MAX_PROCESS_WORKERS}; "
        f"physical_cores={topo.get('physical_cores')}; "
        f"logical_processors={topo.get('logical_processors')}; "
        f"overflow_workers={len(_AFFINITY_PLAN.get('overflow_logical_processors', []))}; "
        f"coordinator_reserve={_AFFINITY_PLAN.get('reserved_logical_processors', [])}; "
        f"affinity={'ON' if _AFFINITY_PLAN.get('enabled') else 'scheduler'}",
        flush=True,
    )


def backend_stats() -> dict:
    return {
        **_STATS,
        "configured_process_workers": int(_PROCESS_WORKERS),
        "max_process_workers": int(MAX_PROCESS_WORKERS),
        "scheduler_version": "MULTICORE_PROCESS_POOL_V2_24_WORKER_PCORE_FIRST",
        "active_pool": bool(_POOL is not None),
        "pool_signature": list(_POOL_SIGNATURE) if _POOL_SIGNATURE is not None else None,
        "affinity": _AFFINITY_PLAN,
    }


def shutdown_multicore_backend() -> None:
    global _POOL, _POOL_SIGNATURE, _POOL_SLOT_COUNTER, _CONTEXT_SIGNALS, _CONTEXT_PRICES
    with _BACKEND_LOCK:
        if _POOL is not None:
            _POOL.shutdown(wait=True, cancel_futures=False)
            _POOL = None
            _POOL_SIGNATURE = None
            _POOL_SLOT_COUNTER = None
    _CONTEXT_SIGNALS = None
    _CONTEXT_PRICES = None
    print(f"[multicore] shutdown; stats={backend_stats()}", flush=True)
