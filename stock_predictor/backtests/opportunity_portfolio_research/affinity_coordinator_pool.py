from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import ctypes
from ctypes import wintypes
import os
from queue import Empty, SimpleQueue

from .portfolio_replay_process_backend import backend_stats


def _set_current_thread_logical_affinity(logical_processor: int) -> bool:
    """Pin only the current coordinator thread, not the whole Python process."""
    if os.name != "nt":
        return False
    logical = int(logical_processor)
    if logical < 0 or logical >= 64:
        return False
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    get_current = kernel32.GetCurrentThread
    get_current.argtypes = []
    get_current.restype = wintypes.HANDLE
    set_mask = kernel32.SetThreadAffinityMask
    set_mask.argtypes = [wintypes.HANDLE, ctypes.c_size_t]
    set_mask.restype = ctypes.c_size_t
    previous = set_mask(get_current(), ctypes.c_size_t(1 << logical))
    return bool(previous)


def _coordinator_thread_init(cpu_queue: SimpleQueue) -> None:
    try:
        logical = int(cpu_queue.get_nowait())
    except Empty:
        return
    ok = _set_current_thread_logical_affinity(logical)
    try:
        print(
            f"[multicore-coordinator] logical_cpu={logical} "
            f"affinity={'OK' if ok else 'FALLBACK'}",
            flush=True,
        )
    except OSError as exc:
        # Detached/redirected Windows workers can expose an invalid stdout
        # handle. Affinity setup succeeded independently; do not turn a
        # diagnostic print failure into a misleading BrokenThreadPool error.
        if getattr(exc, "errno", None) not in (22,):
            raise RuntimeError(
                f"MULTICORE_COORDINATOR_OUTPUT_INIT_FAILED:logical_cpu={logical}:"
                f"errno={getattr(exc, 'errno', None)}"
            ) from exc


class AffinityCoordinatorPool(ThreadPoolExecutor):
    """ThreadPoolExecutor mapped onto logical CPUs reserved from replay workers.

    On the Ryzen 7 5800X target the normal layout is 12 replay processes plus four
    coordinator threads. The coordinator threads prepare/cache/aggregate independent
    outer windows and feed the shared replay ProcessPool. They intentionally remain
    lightweight; their four hardware threads are also available to Windows/SQLite/IPC
    whenever a coordinator thread is waiting on replay futures.
    """

    def __init__(self, max_workers: int | None = None, thread_name_prefix: str = "", **kwargs) -> None:
        requested = max(1, int(max_workers or 1))
        affinity = dict(backend_stats().get("affinity") or {})
        reserved = [int(x) for x in affinity.get("reserved_logical_processors", [])]
        if reserved:
            effective = min(requested, len(reserved))
            cpu_queue: SimpleQueue = SimpleQueue()
            for logical in reserved[:effective]:
                cpu_queue.put(logical)
            self._coordinator_cpu_queue = cpu_queue
            super().__init__(
                max_workers=effective,
                thread_name_prefix=thread_name_prefix,
                initializer=_coordinator_thread_init,
                initargs=(cpu_queue,),
                **kwargs,
            )
        else:
            self._coordinator_cpu_queue = None
            super().__init__(
                max_workers=requested,
                thread_name_prefix=thread_name_prefix,
                **kwargs,
            )
