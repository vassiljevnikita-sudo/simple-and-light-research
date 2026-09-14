from __future__ import annotations

from concurrent.futures import as_completed
from concurrent.futures.process import BrokenProcessPool
import time

from . import portfolio_replay_process_backend as backend

_ORIGINAL_RUN_PROCESS_FUTURES = None
_RECOVERY_COUNT = 0
_MAX_RECOVERIES_PER_BATCH = 3


def _clean_worker_value(value):
    """Separate execution telemetry from the research payload before checkpointing."""
    return backend._clean_worker_result(value)


def _next_worker_count(current_workers: int, recovery_number: int) -> int:
    """Keep 12 workers on first retry, then degrade only if the pool keeps dying."""
    current = max(1, int(current_workers))
    if recovery_number <= 1:
        return current
    if recovery_number == 2:
        return min(current, 10) if current > 8 else current
    return min(current, 8) if current > 6 else current


def _recovery_telemetry(
    *, label: str, recovery_number: int, exc: BaseException,
    pending_jobs: int, current_workers: int, target_workers: int,
) -> None:
    backend._write_telemetry({
        "event": "pool_recovery",
        "batch": label,
        "recovery_number": int(recovery_number),
        "failure_type": type(exc).__name__,
        "failure": str(exc),
        "pending_jobs": int(pending_jobs),
        "from_workers": int(current_workers),
        "to_workers": int(target_workers),
    })
    print(
        f"[{label}] BrokenProcessPool recovery {recovery_number}: "
        f"pending={pending_jobs}, workers={current_workers}->{target_workers}",
        flush=True,
    )


def _recover_pool(failed_pool, label: str, pending_jobs: int, exc: BaseException):
    global _RECOVERY_COUNT
    current_workers = int(getattr(failed_pool, "_max_workers", backend._PROCESS_WORKERS))

    with backend._BACKEND_LOCK:
        # Another coordinator may already have rebuilt the shared pool after the same
        # crash. Reuse that replacement instead of triggering a second restart.
        if backend._POOL is not None and backend._POOL is not failed_pool:
            return backend._POOL

        _RECOVERY_COUNT += 1
        recovery_number = _RECOVERY_COUNT
        target_workers = _next_worker_count(current_workers, recovery_number)
        try:
            failed_pool.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass
        backend._POOL = None
        backend._POOL_SIGNATURE = None
        backend._POOL_SLOT_COUNTER = None
        backend._STATS["pool_restarts"] += 1

    _recovery_telemetry(
        label=label,
        recovery_number=recovery_number,
        exc=exc,
        pending_jobs=pending_jobs,
        current_workers=current_workers,
        target_workers=target_workers,
    )

    signals = backend._CONTEXT_SIGNALS
    prices = backend._CONTEXT_PRICES
    if signals is None or prices is None:
        raise RuntimeError(
            "EXECUTION_FAILURE:POOL_RECOVERY_CONTEXT_MISSING"
        ) from exc
    return backend._ensure_pool(signals, prices, target_workers)


def _recover_external_pool(
    failed_pool,
    label: str,
    pending_jobs: int,
    exc: BaseException,
    pool_factory,
):
    """Apply the established recovery policy to a pool with a custom initializer.

    V4.5 needs daily high/low/ATR worker state rather than the standard signal/price
    frames, so it cannot be rebuilt by backend._ensure_pool. The retry policy,
    degradation schedule and telemetry remain identical; only pool construction is
    supplied by the caller.
    """
    global _RECOVERY_COUNT
    current_workers = int(getattr(failed_pool, "_max_workers", backend._PROCESS_WORKERS))
    _RECOVERY_COUNT += 1
    recovery_number = _RECOVERY_COUNT
    target_workers = _next_worker_count(current_workers, recovery_number)
    try:
        failed_pool.shutdown(wait=False, cancel_futures=True)
    except Exception:
        pass
    backend._STATS["pool_restarts"] += 1
    _recovery_telemetry(
        label=label,
        recovery_number=recovery_number,
        exc=exc,
        pending_jobs=pending_jobs,
        current_workers=current_workers,
        target_workers=target_workers,
    )
    return pool_factory(target_workers)


def _run_resilient_loop(
    pool,
    jobs: list[tuple[int, object]],
    worker_func,
    label: str,
    on_result,
    jobs_reused: int = 0,
    main_cache_seconds: float = 0.0,
    horizon: int | None = None,
    recover_pool=None,
    shutdown_on_complete: bool = False,
    telemetry_extra: dict | None = None,
) -> None:
    if not jobs:
        return

    pending = {int(index): payload for index, payload in jobs}
    total = len(pending)
    current_pool = pool
    started = time.perf_counter()
    first_finished = None
    worker_counts: dict[str, int] = {
        worker: 0 for worker in backend._expected_worker_slots()
    }
    worker_compute: dict[str, float] = {
        worker: 0.0 for worker in backend._expected_worker_slots()
    }
    worker_idle: dict[str, float] = {
        worker: 0.0 for worker in backend._expected_worker_slots()
    }
    worker_pids: dict[str, int] = {}
    main_aggregation_seconds = 0.0
    completed = 0
    recoveries_this_batch = 0
    step = max(1, total // 8)
    recover = recover_pool or _recover_pool

    try:
        while pending:
            futures = {}
            submit_failure = None
            for index, payload in list(pending.items()):
                try:
                    futures[current_pool.submit(worker_func, payload)] = index
                except BrokenProcessPool as exc:
                    submit_failure = exc
                    break

            if submit_failure is not None:
                recoveries_this_batch += 1
                if recoveries_this_batch > _MAX_RECOVERIES_PER_BATCH:
                    backend._write_telemetry({
                        "event": "pool_recovery_exhausted",
                        "batch": label,
                        "pending_jobs": len(pending),
                        "recoveries": recoveries_this_batch - 1,
                    })
                    raise RuntimeError(
                        f"EXECUTION_FAILURE:BROKEN_PROCESS_POOL_RETRIES_EXHAUSTED:{label}"
                    ) from submit_failure
                current_pool = recover(
                    current_pool, label, len(pending), submit_failure
                )
                continue

            pool_failure = None
            for future in as_completed(futures):
                index = futures[future]
                try:
                    raw_value = future.result()
                except BrokenProcessPool as exc:
                    pool_failure = exc
                    break

                finished = time.perf_counter()
                if first_finished is None:
                    first_finished = finished
                value, telemetry = _clean_worker_value(raw_value)
                worker = str(telemetry.get("worker_index", "unknown"))
                worker_counts[worker] = worker_counts.get(worker, 0) + 1
                worker_compute[worker] = worker_compute.get(worker, 0.0) + float(
                    telemetry.get("compute_seconds", 0.0)
                )
                worker_idle[worker] = worker_idle.get(worker, 0.0) + float(
                    telemetry.get("idle_seconds_before_job", 0.0)
                )
                if telemetry.get("worker_pid") is not None and worker != "unknown":
                    worker_pids[worker] = int(telemetry["worker_pid"])

                aggregation_started = time.perf_counter()
                on_result(index, value)
                main_aggregation_seconds += time.perf_counter() - aggregation_started
                pending.pop(index, None)
                completed += 1
                if completed == total or completed % step == 0:
                    print(
                        f"[{label}] computed+checkpointed {completed}/{total}",
                        flush=True,
                    )

            if pool_failure is not None:
                recoveries_this_batch += 1
                if recoveries_this_batch > _MAX_RECOVERIES_PER_BATCH:
                    backend._write_telemetry({
                        "event": "pool_recovery_exhausted",
                        "batch": label,
                        "pending_jobs": len(pending),
                        "recoveries": recoveries_this_batch - 1,
                    })
                    raise RuntimeError(
                        f"EXECUTION_FAILURE:BROKEN_PROCESS_POOL_RETRIES_EXHAUSTED:{label}"
                    ) from pool_failure
                current_pool = recover(
                    current_pool, label, len(pending), pool_failure
                )
                continue

            # No pool failure and every submitted future was consumed.
            break

        record = {
            "batch": label,
            "horizon": horizon,
            "jobs_submitted": total,
            "jobs_reused": int(jobs_reused),
            "jobs_computed": total,
            "pool_recoveries": int(recoveries_this_batch),
            "time_first_job_finished_seconds": (
                first_finished - started if first_finished is not None else None
            ),
            "time_last_job_finished_seconds": time.perf_counter() - started,
            "worker_job_counts": worker_counts,
            "worker_compute_time": worker_compute,
            "worker_idle_time": worker_idle,
            "worker_pids": worker_pids,
            "worker_slots_expected": backend._expected_worker_slots(),
            "worker_slots_with_jobs": sorted(
                [worker for worker, count in worker_counts.items() if count > 0],
                key=lambda value: int(value) if value.isdigit() else 10**9,
            ),
            "main_cache_seconds": float(main_cache_seconds),
            "main_aggregation_seconds": float(main_aggregation_seconds),
        }
        if telemetry_extra:
            record.update(dict(telemetry_extra))
        backend._write_telemetry(record)
        backend._write_cumulative_telemetry(record)
    finally:
        if shutdown_on_complete:
            try:
                current_pool.shutdown(wait=True, cancel_futures=False)
            except Exception:
                pass


def resilient_run_process_futures(
    pool,
    jobs: list[tuple[int, object]],
    worker_func,
    label: str,
    on_result,
    jobs_reused: int = 0,
    main_cache_seconds: float = 0.0,
    horizon: int | None = None,
) -> None:
    """Checkpoint completed jobs and rebuild a dead shared ProcessPool around misses."""
    return _run_resilient_loop(
        pool,
        jobs,
        worker_func,
        label,
        on_result,
        jobs_reused=jobs_reused,
        main_cache_seconds=main_cache_seconds,
        horizon=horizon,
        recover_pool=_recover_pool,
        shutdown_on_complete=False,
    )


def resilient_run_external_process_futures(
    pool_factory,
    workers: int,
    jobs: list[tuple[int, object]],
    worker_func,
    label: str,
    on_result,
    jobs_reused: int = 0,
    main_cache_seconds: float = 0.0,
    horizon: int | None = None,
    telemetry_extra: dict | None = None,
) -> None:
    """Run custom-initialized process work with the standard recovery semantics."""
    if not jobs:
        return
    initial_pool = pool_factory(max(1, int(workers)))

    def recover(pool, batch, pending_jobs, exc):
        return _recover_external_pool(
            pool, batch, pending_jobs, exc, pool_factory
        )

    return _run_resilient_loop(
        initial_pool,
        jobs,
        worker_func,
        label,
        on_result,
        jobs_reused=jobs_reused,
        main_cache_seconds=main_cache_seconds,
        horizon=horizon,
        recover_pool=recover,
        shutdown_on_complete=True,
        telemetry_extra=telemetry_extra,
    )


def install_resilient_process_pool() -> None:
    global _ORIGINAL_RUN_PROCESS_FUTURES, _RECOVERY_COUNT
    if backend._run_process_futures is resilient_run_process_futures:
        return
    if _ORIGINAL_RUN_PROCESS_FUTURES is None:
        _ORIGINAL_RUN_PROCESS_FUTURES = backend._run_process_futures
    _RECOVERY_COUNT = 0
    backend._run_process_futures = resilient_run_process_futures


def restore_process_pool_runner() -> None:
    if _ORIGINAL_RUN_PROCESS_FUTURES is not None:
        backend._run_process_futures = _ORIGINAL_RUN_PROCESS_FUTURES
