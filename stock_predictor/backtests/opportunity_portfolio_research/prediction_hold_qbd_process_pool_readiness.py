from __future__ import annotations

"""QbD process-pool readiness contract.

The replay ProcessPool uses expensive Windows-spawn initializers. Without a
readiness gate, the first workers that finish prepare_prices()/prepare_signals()
can drain short replay jobs while later SMT workers are still initializing.

For the Prediction x Hold QbD runner we keep one persistent replay pool per
prediction horizon and prewarm it with one blocking probe per worker. A probe
cannot finish until every configured worker has taken exactly one probe, so all
workers are fully initialized before any research replay job enters the queue.
Subsequent coarse/robustness/window work shares that same horizon pool. The
research/search contract is unchanged; this module changes execution only.
"""

from contextlib import contextmanager
import os
from pathlib import Path
import tempfile
from threading import Lock
import time
from typing import Iterator

import pandas as pd

from . import portfolio_replay_process_backend as backend

_ORIGINAL_ENSURE_POOL = backend._ensure_pool
_ACTIVE_SCOPE: tuple | None = None
_ACTIVE_POOL_ID: int | None = None
_POOL_READY_LOCK = Lock()


def _worker_ready_probe(directory: str) -> int:
    """Occupy one initialized worker until the parent releases the full pool."""
    root = Path(directory)
    pid = os.getpid()
    ready = root / f"{pid}.ready"
    release = root / "RELEASE"
    ready.write_text(str(pid), encoding="ascii")
    while not release.exists():
        time.sleep(0.01)
    return pid


def _scope_key(signals: pd.DataFrame, prices: pd.DataFrame, workers: int) -> tuple:
    horizons: tuple[int, ...] = ()
    if "horizon" in signals.columns and not signals.empty:
        horizons = tuple(sorted(int(x) for x in signals["horizon"].dropna().unique()))
    if "decision_date" in signals.columns and not signals.empty:
        first_date = str(pd.Timestamp(signals["decision_date"].min()))
        last_date = str(pd.Timestamp(signals["decision_date"].max()))
    else:
        first_date = ""
        last_date = ""
    return (
        int(workers),
        id(prices),
        horizons,
        int(len(signals)),
        first_date,
        last_date,
    )


def _wait_for_process_pool_readiness(pool, workers: int) -> None:
    workers = max(1, int(workers))
    timeout = max(
        30.0,
        float(os.environ.get("QBD_WORKER_READY_TIMEOUT_SECONDS", "300")),
    )
    with tempfile.TemporaryDirectory(prefix="qbd-worker-ready-") as directory:
        root = Path(directory)
        futures = [
            pool.submit(_worker_ready_probe, directory)
            for _ in range(workers)
        ]
        deadline = time.monotonic() + timeout
        unique_ready: set[int] = set()
        try:
            while len(unique_ready) < workers:
                unique_ready = {
                    int(path.stem)
                    for path in root.glob("*.ready")
                    if path.stem.isdigit()
                }
                for future in futures:
                    if future.done():
                        exc = future.exception()
                        if exc is not None:
                            raise RuntimeError(
                                "QBD_WORKER_READY_PROBE_FAILED"
                            ) from exc
                if len(unique_ready) >= workers:
                    break
                if time.monotonic() >= deadline:
                    raise RuntimeError(
                        "QBD_PROCESS_POOL_READY_TIMEOUT:"
                        f"ready={len(unique_ready)}:workers={workers}"
                    )
                time.sleep(0.05)
        finally:
            # Always release probes so a diagnostic failure cannot strand workers.
            (root / "RELEASE").write_text("release\n", encoding="ascii")

        pids = [int(future.result(timeout=30.0)) for future in futures]
        unique_pids = set(pids)
        if len(unique_pids) != workers:
            raise RuntimeError(
                "QBD_PROCESS_POOL_NOT_DISTRIBUTED:"
                f"unique_workers={len(unique_pids)}:workers={workers}"
            )
        print(
            "[qbd-queue] replay process pool ready: "
            f"workers={workers}, unique_pids={len(unique_pids)}; "
            "cell stages now share one persistent queue",
            flush=True,
        )


def _qbd_ensure_pool(signals: pd.DataFrame, prices: pd.DataFrame, workers: int):
    """Reuse equivalent horizon pools and gate every new/recovered pool once."""
    global _ACTIVE_SCOPE, _ACTIVE_POOL_ID
    effective = backend._effective_workers(workers)
    scope = _scope_key(signals, prices, effective)

    # Several outer-window coordinator threads can request the same pool at once.
    # Only creation/prewarm is serialized; replay submission remains fully parallel.
    with _POOL_READY_LOCK:
        current = backend._POOL
        if (
            current is not None
            and _ACTIVE_SCOPE == scope
            and _ACTIVE_POOL_ID == id(current)
        ):
            return current

        pool = _ORIGINAL_ENSURE_POOL(signals, prices, effective)
        pool_changed = _ACTIVE_POOL_ID != id(pool)
        scope_changed = _ACTIVE_SCOPE != scope
        if pool_changed or scope_changed:
            _wait_for_process_pool_readiness(pool, effective)
            _ACTIVE_SCOPE = scope
            _ACTIVE_POOL_ID = id(pool)
        return pool


@contextmanager
def qbd_process_pool_readiness_contract() -> Iterator[None]:
    """Install the QbD scheduler overlay without changing research semantics."""
    global _ACTIVE_SCOPE, _ACTIVE_POOL_ID
    previous = backend._ensure_pool
    _ACTIVE_SCOPE = None
    _ACTIVE_POOL_ID = None
    backend._ensure_pool = _qbd_ensure_pool
    try:
        yield
    finally:
        backend._ensure_pool = previous
        _ACTIVE_SCOPE = None
        _ACTIVE_POOL_ID = None
