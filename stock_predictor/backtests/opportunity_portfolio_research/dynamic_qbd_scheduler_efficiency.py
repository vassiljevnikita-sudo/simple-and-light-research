"""Execution-only scheduler efficiency layer for Dynamic-QBD Step-9.

This layer fixes scheduler/pool/SQLite inefficiencies without changing the
scientific DAG, causal visibility, RAM thresholds, numerical GPU kernels or
holdout authority.

Active Step-9 contract:
- 32 shared CPU lanes across SHORT/PRIMARY/LONG;
- persistent CPU/GPU lane pools across scheduler waves;
- four GPU staging workers plus two parent queue slots per physical GPU;
- one unsafe kernel section per physical device remains enforced by the
  existing device-section broker;
- bounded cached dependency-ready frontiers;
- lightweight worker-side ManifestedJobStore opens;
- watchdog liveness uses EXISTS rather than full ready counts.

The installer is split around the existing runtime-hardening installer:
``install_scheduler_efficiency`` must run before runtime hardening captures the
coordinator functions; ``install_scheduler_efficiency_post_hardening`` runs
afterwards to bind the hardened watchdog plus batched/runner hooks.
"""
from __future__ import annotations

import atexit
from collections import deque
from concurrent.futures.process import BrokenProcessPool
from contextlib import contextmanager
from functools import wraps
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
import re
import sqlite3
import sys
import textwrap
import threading
import time
from typing import Any, Iterable, Mapping

from . import dynamic_qbd_manifested_job_coordinator as coordinator
from .dynamic_qbd_attempt_fenced_store import AttemptFencedManifestedJobStore


_ACTIVE_ENV = "DQBD_SCHEDULER_EFFICIENCY_V41"
_CPU_LANES = 32
_GPU_WORKERS_PER_DEVICE = 4
_GPU_QUEUE_AHEAD = 2
_GPU_GLOBAL_CAPACITY = _GPU_WORKERS_PER_DEVICE + _GPU_QUEUE_AHEAD
_READY_SCAN_FLOOR = 4096
_READY_SCAN_CEILING = 8192
_READY_CACHE_TTL_SECONDS = 2.0

_PRE_INSTALLED = False
_POST_INSTALLED = False

_POOL_LOCK = threading.RLock()
_PERSISTENT_POOLS: dict[tuple[str, str, int], Any] = {}
_PERSISTENT_CONTROLLERS: dict[tuple[int, int], Any] = {}

_READY_LOCK = threading.RLock()
_READY_EPOCH: dict[str, int] = {}
_READY_CACHE: dict[tuple[str, tuple[str, ...] | None], dict[str, Any]] = {}
_READY_STATS = {
    "db_refreshes": 0,
    "cache_hits": 0,
    "claim_discards": 0,
    "invalidations": 0,
}

_STORE_LOCK = threading.RLock()
_STORE_PATH_LOCKS: dict[str, threading.Lock] = {}
_STORE_BOOTSTRAPPED: set[str] = set()


def _norm_path(value: str | Path) -> str:
    return os.path.normcase(os.path.abspath(os.fspath(Path(value))))


def _runtime_root_from_telemetry(telemetry_root: str | Path | None) -> Path:
    if telemetry_root is None:
        return Path.cwd()
    telemetry = Path(telemetry_root)
    owner_root = telemetry.parent
    if owner_root.name.lower() in {"short", "primary", "long"}:
        return owner_root.parent
    return owner_root


def _shared_telemetry_root(telemetry_root: str | Path | None) -> Path:
    return (
        _runtime_root_from_telemetry(telemetry_root)
        / "_shared-compute"
        / "scheduler-telemetry"
    )


def _execution_efficiency_enabled() -> bool:
    return os.environ.get(_ACTIVE_ENV) == "1"


def _recompile_function(module, fn, source: str):
    namespace: dict[str, Any] = {}
    filename = inspect.getsourcefile(fn) or f"<{module.__name__}>"
    exec(compile(source, filename, "exec"), module.__dict__, namespace)
    return namespace[fn.__name__]


def _patch_gpu_topology() -> None:
    """Restore v40.0.4.3's additive 4+2 Step-9 staging topology."""
    for name in ("execute_candidate_oos_jobs", "execute_ready_jobs"):
        fn = getattr(coordinator, name)
        if getattr(fn, "_dqbd_gpu4_topology", False):
            continue
        source = textwrap.dedent(inspect.getsource(fn))
        if "desired_gpu_workers_per_device = 1" not in source:
            raise RuntimeError(f"DQBD_GPU_TOPOLOGY_PATCH_ANCHOR_INVALID:{name}")
        source = source.replace(
            "desired_gpu_workers_per_device = 1",
            "desired_gpu_workers_per_device = (\n"
            "            4 if os.environ.get('DQBD_SCHEDULER_EFFICIENCY_V41') == '1'\n"
            "            else 1)",
            1,
        )
        pattern = re.compile(
            r"gpu_pool_count = min\(\s*"
            r"gpu_device_count,\s*"
            r"max\(\s*0,\s*"
            r"\(pool_workers - 1\)\s*//\s*"
            r"desired_gpu_workers_per_device\)\s*\)"
        )
        replacement = (
            "gpu_pool_count = (\n"
            "            gpu_device_count\n"
            "            if (os.environ.get('DQBD_SCHEDULER_EFFICIENCY_V41') == '1'\n"
            "                and pool_workers > 1)\n"
            "            else min(\n"
            "                gpu_device_count,\n"
            "                max(0, (pool_workers - 1) // desired_gpu_workers_per_device)))"
        )
        source, count = pattern.subn(replacement, source, count=1)
        if count != 1:
            raise RuntimeError(f"DQBD_GPU_POOL_COUNT_PATCH_ANCHOR_INVALID:{name}")
        source = source.replace(
            "len(gpu_pending[index]) < gpu_queue_capacity",
            "(len(gpu_pending[index]) < gpu_queue_capacity "
            "and getattr(gpu_pools[index], "
            "'dqbd_available_submission_slots', "
            "lambda: gpu_queue_capacity)() > 0)",
        )
        patched = _recompile_function(coordinator, fn, source)
        patched._dqbd_gpu4_topology = True
        setattr(coordinator, name, patched)


def _store_path_lock(identity: str) -> threading.Lock:
    with _STORE_LOCK:
        lock = _STORE_PATH_LOCKS.get(identity)
        if lock is None:
            lock = threading.Lock()
            _STORE_PATH_LOCKS[identity] = lock
        return lock


def _patch_store_open_and_frontier() -> None:
    cls = coordinator.ManifestedJobStore
    if getattr(cls.__init__, "_dqbd_lightweight_open", False):
        return

    original_init = cls.__init__
    original_connect = cls._connect
    original_ready = cls.ready_jobs
    original_claim = cls.claim_ready

    @wraps(original_init)
    def init(self, root: str | Path):
        self.root = Path(root)
        self.jobs_path = self.root / "run-state" / "jobs.json"
        self.db_path = self.root / "run-state" / "jobs.sqlite3"
        identity = _norm_path(self.db_path)

        if os.environ.get("DQBD_MANIFEST_STORE_WORKER_OPEN") == "1":
            if not self.db_path.is_file():
                raise RuntimeError(
                    "DQBD_WORKER_MANIFEST_STORE_DATABASE_MISSING:"
                    f"{self.db_path}"
                )
            return

        lock = _store_path_lock(identity)
        with lock:
            with _STORE_LOCK:
                already = identity in _STORE_BOOTSTRAPPED
            if already and self.db_path.is_file():
                return
            self._dqbd_bootstrap_connection = True
            try:
                original_init(self, root)
            finally:
                self._dqbd_bootstrap_connection = False
            with _STORE_LOCK:
                _STORE_BOOTSTRAPPED.add(identity)

    @contextmanager
    def connect(self):
        if getattr(self, "_dqbd_bootstrap_connection", False):
            with original_connect(self) as db:
                yield db
            return
        db = sqlite3.connect(self.db_path, timeout=60, isolation_level=None)
        db.execute("PRAGMA busy_timeout=60000")
        db.execute("PRAGMA synchronous=NORMAL")
        try:
            yield db
        finally:
            db.close()

    init._dqbd_lightweight_open = True
    connect._dqbd_lightweight_open = True
    cls.__init__ = init
    cls._connect = connect

    original_initializer = coordinator._manifested_process_initializer

    @wraps(original_initializer)
    def process_initializer(*args, **kwargs):
        os.environ["DQBD_MANIFEST_STORE_WORKER_OPEN"] = "1"
        return original_initializer(*args, **kwargs)

    process_initializer._dqbd_worker_lightweight_open = True
    coordinator._manifested_process_initializer = process_initializer

    @wraps(original_ready)
    def ready_jobs(
        self, *, kinds: Iterable[str] | None = None, limit: int = 256,
        exclude_job_ids: Iterable[str] | None = None,
    ) -> list[dict]:
        requested = max(1, int(limit))
        kinds_tuple = tuple(str(value) for value in kinds) if kinds is not None else None
        root = _norm_path(self.root)
        key = (root, kinds_tuple)
        excluded = {str(value) for value in (exclude_job_ids or ())}
        now = time.monotonic()

        with _READY_LOCK:
            epoch = int(_READY_EPOCH.get(root, 0))
            entry = _READY_CACHE.get(key)
            usable = (
                entry is not None
                and int(entry["epoch"]) == epoch
                and now - float(entry["refreshed_at"]) <= _READY_CACHE_TTL_SECONDS
                and int(entry["scan_limit"]) >= requested
            )
            if usable:
                rows = [
                    dict(row) for row in entry["rows"]
                    if str(row.get("job_id", "")) not in excluded
                ]
                if len(rows) >= requested or bool(entry["exhausted"]):
                    _READY_STATS["cache_hits"] += 1
                    return rows[:requested]
                scan_limit = min(
                    _READY_SCAN_CEILING,
                    max(_READY_SCAN_FLOOR, int(entry["scan_limit"]) * 2, requested * 4),
                )
            else:
                scan_limit = min(
                    _READY_SCAN_CEILING,
                    max(_READY_SCAN_FLOOR, requested * 4),
                )

        rows = original_ready(
            self,
            kinds=kinds_tuple,
            limit=scan_limit,
            exclude_job_ids=None,
        )
        with _READY_LOCK:
            current_epoch = int(_READY_EPOCH.get(root, 0))
            if current_epoch == epoch:
                _READY_CACHE[key] = {
                    "epoch": epoch,
                    "scan_limit": scan_limit,
                    "rows": [dict(row) for row in rows],
                    "exhausted": len(rows) < scan_limit,
                    "refreshed_at": time.monotonic(),
                }
            _READY_STATS["db_refreshes"] += 1
        return [
            dict(row) for row in rows
            if str(row.get("job_id", "")) not in excluded
        ][:requested]

    @wraps(original_claim)
    def claim_ready(self, *args, **kwargs):
        result = original_claim(self, *args, **kwargs)
        if result is not None:
            _discard_ready_ids(self.root, (str(result["job_id"]),))
        elif kwargs.get("job_ids") is not None:
            _discard_ready_ids(self.root, kwargs["job_ids"])
        return result

    def ready_job_exists(self) -> bool:
        with self._connect() as db:
            row = db.execute(
                "SELECT 1 FROM jobs j WHERE j.state='PENDING' "
                "AND NOT EXISTS (SELECT 1 FROM job_dependencies d "
                "LEFT JOIN jobs p ON p.job_id=d.depends_on "
                "WHERE d.job_id=j.job_id "
                "AND (p.job_id IS NULL OR p.state!='COMPLETE')) LIMIT 1"
            ).fetchone()
        return row is not None

    def ready_job_count_bounded(self, limit: int = 4096) -> int:
        cap = max(1, int(limit))
        with self._connect() as db:
            row = db.execute(
                "SELECT COUNT(*) FROM ("
                "SELECT 1 FROM jobs j WHERE j.state='PENDING' "
                "AND NOT EXISTS (SELECT 1 FROM job_dependencies d "
                "LEFT JOIN jobs p ON p.job_id=d.depends_on "
                "WHERE d.job_id=j.job_id "
                "AND (p.job_id IS NULL OR p.state!='COMPLETE')) "
                "LIMIT ?)",
                (cap,),
            ).fetchone()
        return int(row[0] or 0) if row else 0

    def pending_cutoff_frontier(self, *, kind: str, limit: int = 256) -> list[dict]:
        cap = max(1, int(limit))
        readiness = (
            " NOT EXISTS (SELECT 1 FROM job_dependencies d "
            " LEFT JOIN jobs p ON p.job_id=d.depends_on "
            " WHERE d.job_id=j.job_id "
            " AND (p.job_id IS NULL OR p.state!='COMPLETE'))"
        )
        with self._connect() as db:
            row = db.execute(
                "SELECT MIN(json_extract(j.payload,'$.cutoff')) FROM jobs j "
                "WHERE j.state='PENDING' AND j.kind=? AND" + readiness,
                (str(kind),),
            ).fetchone()
            cutoff = row[0] if row else None
            if cutoff is None:
                return []
            rows = db.execute(
                "SELECT j.payload,j.state,j.attempt,j.lease_owner,j.lease_until,"
                "j.heartbeat,j.last_error,j.started_at,j.finished_at "
                "FROM jobs j WHERE j.state='PENDING' AND j.kind=? "
                "AND json_extract(j.payload,'$.cutoff')=? AND" + readiness +
                " ORDER BY j.job_id LIMIT ?",
                (str(kind), cutoff, cap),
            ).fetchall()
        return [
            self._overlay_sql_state(json.loads(row[0]), row[1:])
            for row in rows
        ]

    def scheduler_ready_capacity(self, limit: int = _CPU_LANES) -> int:
        """Count physical scheduler units, not logical Replay fan-out rows."""
        cap = max(1, int(limit))
        readiness = (
            " NOT EXISTS (SELECT 1 FROM job_dependencies d "
            " LEFT JOIN jobs p ON p.job_id=d.depends_on "
            " WHERE d.job_id=j.job_id "
            " AND (p.job_id IS NULL OR p.state!='COMPLETE'))"
        )
        total = 0
        with self._connect() as db:
            row = db.execute(
                "SELECT COUNT(*) FROM (SELECT 1 FROM jobs j "
                "WHERE j.state='PENDING' "
                "AND j.kind NOT IN ('replay','evidence','candidate_oos_fold') "
                "AND" + readiness + " LIMIT ?)",
                (cap,),
            ).fetchone()
            total = min(cap, int(row[0] or 0) if row else 0)
            for kind in ("replay", "evidence"):
                if total >= cap:
                    break
                cutoff_row = db.execute(
                    "SELECT MIN(json_extract(j.payload,'$.cutoff')) FROM jobs j "
                    "WHERE j.state='PENDING' AND j.kind=? AND" + readiness,
                    (kind,),
                ).fetchone()
                cutoff = cutoff_row[0] if cutoff_row else None
                if cutoff is None:
                    continue
                remaining = cap - total
                group_row = db.execute(
                    "SELECT COUNT(*) FROM ("
                    "SELECT DISTINCT substr(json_extract(j.payload,'$.portfolio_family_key'),2,2) "
                    "FROM jobs j WHERE j.state='PENDING' AND j.kind=? "
                    "AND json_extract(j.payload,'$.cutoff')=? AND" + readiness +
                    " LIMIT ?)",
                    (kind, cutoff, remaining),
                ).fetchone()
                total += int(group_row[0] or 0) if group_row else 0
        return min(cap, total)

    ready_jobs._dqbd_cached_frontier = True
    cls.ready_jobs = ready_jobs
    cls.claim_ready = claim_ready
    cls.ready_job_exists = ready_job_exists
    cls.ready_job_count_bounded = ready_job_count_bounded
    cls.pending_cutoff_frontier = pending_cutoff_frontier
    cls.scheduler_ready_capacity = scheduler_ready_capacity

    for name in (
        "set_job", "seed_jobs", "requeue_descendants", "release_claim",
        "requeue_reclaimed_job", "requeue_execution_failure", "finish_job",
        "requeue_interrupted_jobs", "ensure_execution_backend_contract",
    ):
        method = getattr(cls, name, None)
        if method is None or getattr(method, "_dqbd_ready_invalidator", False):
            continue

        def make_wrapper(original):
            @wraps(original)
            def wrapper(self, *args, **kwargs):
                result = original(self, *args, **kwargs)
                _invalidate_ready_root(self.root)
                return result
            wrapper._dqbd_ready_invalidator = True
            return wrapper

        setattr(cls, name, make_wrapper(method))

    facade = AttemptFencedManifestedJobStore
    for name in (
        "finish_job", "release_claim", "requeue_reclaimed_job",
        "requeue_execution_failure", "requeue_interrupted_jobs",
    ):
        method = getattr(facade, name, None)
        if method is None or getattr(method, "_dqbd_ready_invalidator", False):
            continue

        def make_facade_wrapper(original):
            @wraps(original)
            def wrapper(self, *args, **kwargs):
                result = original(self, *args, **kwargs)
                _invalidate_ready_root(self._store.root)
                return result
            wrapper._dqbd_ready_invalidator = True
            return wrapper

        setattr(facade, name, make_facade_wrapper(method))


def _invalidate_ready_root(root: str | Path) -> None:
    identity = _norm_path(root)
    with _READY_LOCK:
        _READY_EPOCH[identity] = int(_READY_EPOCH.get(identity, 0)) + 1
        _READY_STATS["invalidations"] += 1


def _discard_ready_ids(root: str | Path, job_ids: Iterable[str]) -> None:
    identity = _norm_path(root)
    ids = {str(value) for value in job_ids}
    if not ids:
        return
    with _READY_LOCK:
        for key, entry in _READY_CACHE.items():
            if key[0] != identity:
                continue
            entry["rows"] = [
                row for row in entry["rows"]
                if str(row.get("job_id", "")) not in ids
            ]
        _READY_STATS["claim_discards"] += len(ids)


def _build_worker_map(
    desired: int, provided: Iterable[Mapping[str, Any]] | None
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[int] = set()

    def add(values) -> None:
        for raw in values or ():
            row = dict(raw)
            logical = row.get("logical_processor")
            if logical is None:
                continue
            logical = int(logical)
            if logical in seen:
                continue
            row["logical_processor"] = logical
            row.setdefault("core_index", logical)
            row.setdefault("role", "shared")
            rows.append(row)
            seen.add(logical)
            if len(rows) >= desired:
                return

    add(coordinator.active_cpu_contract().get("worker_map") or ())
    add(provided)
    logical_count = max(1, int(os.cpu_count() or 1))
    add(
        {
            "logical_processor": index,
            "core_index": index,
            "role": "shared-fallback",
        }
        for index in range(logical_count)
    )
    if not rows:
        rows.append({
            "logical_processor": 0,
            "core_index": 0,
            "role": "shared-fallback",
        })
    seed_rows = [dict(row) for row in rows]
    cursor = 0
    while len(rows) < desired:
        rows.append(dict(seed_rows[cursor % len(seed_rows)]))
        cursor += 1
    return rows[:desired]


def _shared_task_metadata(args: tuple[Any, ...]) -> dict[str, Any]:
    payload = args[0] if args and isinstance(args[0], Mapping) else {}
    job = payload.get("job", {}) if isinstance(payload, Mapping) else {}
    manifest_job_id = str(job.get("job_id", ""))
    store_root = (
        _norm_path(str(payload.get("output_root")))
        if payload.get("output_root") else ""
    )
    if store_root and manifest_job_id:
        scope = hashlib.sha1(store_root.encode("utf-8")).hexdigest()[:12]
        lane_job_id = f"{scope}:{manifest_job_id}"
    else:
        lane_job_id = manifest_job_id
    return {
        "job_id": lane_job_id,
        "manifest_job_id": manifest_job_id,
        "store_root": store_root,
        "kind": str(job.get("kind", "")),
        "job_class": str(payload.get("_ram_job_class", "")),
        "gpu_device_index": payload.get("gpu_device_index"),
        "attempt": int(job.get("attempt", 0) or 0),
        "ram_reclaim_count": int(job.get("ram_reclaim_count", 0) or 0),
        "submitted_at": time.time(),
    }


def _patch_persistent_pools() -> None:
    original_pool_factory = coordinator._manifested_pool
    original_managed = coordinator._managed_pool_with_ram_pause
    lane_cls = coordinator._ReclaimableLanePool

    if not getattr(lane_cls._task_metadata, "_dqbd_store_scoped", False):
        _shared_task_metadata._dqbd_store_scoped = True
        lane_cls._task_metadata = staticmethod(_shared_task_metadata)

    @wraps(original_pool_factory)
    def pool_factory(
        *, workers: int, role: str, telemetry_root: Path | None,
        worker_map_override: list[dict] | None = None,
        max_tasks_per_child: int | None = None,
        max_pending: int | None = None,
    ):
        if not _execution_efficiency_enabled():
            return original_pool_factory(
                workers=workers,
                role=role,
                telemetry_root=telemetry_root,
                worker_map_override=worker_map_override,
                max_tasks_per_child=max_tasks_per_child,
                max_pending=max_pending,
            )

        role_text = str(role)
        gpu_match = re.fullmatch(r"(?:candidate-oos|causal)-gpu-(\d+)", role_text)
        is_cpu = role_text in {"candidate-oos-cpu", "causal-cpu", "portfolio-batch"}
        if not is_cpu and gpu_match is None:
            return original_pool_factory(
                workers=workers,
                role=role,
                telemetry_root=telemetry_root,
                worker_map_override=worker_map_override,
                max_tasks_per_child=max_tasks_per_child,
                max_pending=max_pending,
            )

        runtime_root = _runtime_root_from_telemetry(telemetry_root)
        runtime_identity = _norm_path(runtime_root)
        device_index = int(gpu_match.group(1)) if gpu_match else -1
        kind = "gpu" if gpu_match else "cpu"
        key = (runtime_identity, kind, device_index)
        desired = (
            _GPU_WORKERS_PER_DEVICE if gpu_match else min(
                _CPU_LANES,
                int(coordinator.MAX_RUNTIME_PROCESS_LANES),
                max(1, int(os.cpu_count() or 1)),
            )
        )
        caller_capacity = max(1, min(int(workers), desired))

        with _POOL_LOCK:
            existing = _PERSISTENT_POOLS.get(key)
            if existing is not None and not getattr(existing, "_closed", False):
                return existing, caller_capacity

            worker_map = _build_worker_map(desired, worker_map_override)
            pool, _ = original_pool_factory(
                workers=desired,
                role=(f"shared-gpu-{device_index}" if gpu_match else "shared-cpu"),
                telemetry_root=_shared_telemetry_root(telemetry_root),
                worker_map_override=worker_map,
                max_tasks_per_child=max_tasks_per_child,
                max_pending=(
                    _GPU_GLOBAL_CAPACITY if gpu_match else max(
                        desired * 3, int(max_pending or 0)
                    )
                ),
            )
            pool._dqbd_persistent_shared = True
            pool._dqbd_shared_kind = kind
            pool._dqbd_runtime_root = runtime_identity
            pool._dqbd_device_index = device_index if gpu_match else None
            _PERSISTENT_POOLS[key] = pool
            return pool, caller_capacity

    def controller_for(pool, scheduler, telemetry_root):
        if scheduler is None:
            return None
        key = (id(pool), id(scheduler))
        with _POOL_LOCK:
            controller = _PERSISTENT_CONTROLLERS.get(key)
            if controller is None:
                controller = coordinator.RamWorkerPauseController(
                    scheduler, pool, _shared_telemetry_root(telemetry_root)
                )
                _PERSISTENT_CONTROLLERS[key] = controller
            return controller

    def cancel_scope(pool, store_root: str, reason: str) -> None:
        if not store_root:
            return
        failure = BrokenProcessPool(reason)
        active_identities: list[str] = []
        with pool._lock:
            kept = deque()
            for task in tuple(pool._queued):
                metadata = task[4]
                if str(metadata.get("store_root", "")) != store_root:
                    kept.append(task)
                    continue
                outer = task[0]
                if not outer.done():
                    outer.set_exception(failure)
            pool._queued = kept
            for lane in pool._lanes:
                metadata = dict(lane.get("metadata") or {})
                outer = lane.get("outer")
                if (
                    outer is not None and not outer.done()
                    and str(metadata.get("store_root", "")) == store_root
                    and metadata.get("job_id")
                ):
                    active_identities.append(str(metadata["job_id"]))
        for identity in active_identities:
            try:
                pool.reclaim_job(
                    identity,
                    reason="DQBD_SHARED_POOL_SCOPE_UNWIND",
                    released_gib=0.0,
                    failure=failure,
                )
            except BaseException:
                pass

    @contextmanager
    def managed(pool, scheduler, telemetry_root: Path):
        if not getattr(pool, "_dqbd_persistent_shared", False):
            with original_managed(pool, scheduler, telemetry_root) as controller:
                yield controller
            return
        controller = controller_for(pool, scheduler, telemetry_root)
        store_root = _norm_path(Path(telemetry_root).parent)
        try:
            yield controller
        except BaseException:
            cancel_scope(pool, store_root, "DQBD_SHARED_POOL_SCOPE_UNWIND")
            raise

    pool_factory._dqbd_persistent_shared = True
    managed._dqbd_persistent_shared = True
    coordinator._manifested_pool = pool_factory
    coordinator._managed_pool_with_ram_pause = managed

    scheduler_cls = coordinator.RamAdmissionScheduler
    original_close = scheduler_cls.close
    if not getattr(original_close, "_dqbd_closes_shared_pools", False):
        @wraps(original_close)
        def close(self, *args, **kwargs):
            close_persistent_scheduler_pools()
            return original_close(self, *args, **kwargs)
        close._dqbd_closes_shared_pools = True
        scheduler_cls.close = close


def close_persistent_scheduler_pools() -> None:
    with _POOL_LOCK:
        controllers = list(_PERSISTENT_CONTROLLERS.values())
        pools = list(_PERSISTENT_POOLS.values())
        _PERSISTENT_CONTROLLERS.clear()
        _PERSISTENT_POOLS.clear()
    for controller in controllers:
        try:
            controller.resume_all()
        except BaseException:
            pass
        try:
            controller.scheduler.unregister_reclaim_controller(controller)
        except BaseException:
            pass
    for pool in pools:
        try:
            pool.shutdown(wait=True, cancel_futures=True)
        except TypeError:
            try:
                pool.shutdown(wait=True)
            except BaseException:
                pass
        except BaseException:
            pass


def _patch_post_hardening_pool_methods() -> None:
    cls = coordinator._ReclaimableLanePool

    if not hasattr(cls, "dqbd_submission_depth"):
        def submission_depth(self) -> int:
            with self._lock:
                active = sum(
                    1 for lane in self._lanes
                    if (
                        lane.get("outer") is not None and not lane["outer"].done()
                    ) or bool(lane.get("replacing"))
                )
                return active + len(self._queued)

        def available_submission_slots(self) -> int:
            return max(0, int(self.max_pending) - int(self.dqbd_submission_depth()))

        cls.dqbd_submission_depth = submission_depth
        cls.dqbd_available_submission_slots = available_submission_slots

    current_submit = cls.submit
    if not getattr(current_submit, "_dqbd_gpu_backpressure", False):
        @wraps(current_submit)
        def submit(self, fn, /, *args, **kwargs):
            if not (
                getattr(self, "_dqbd_persistent_shared", False)
                and getattr(self, "_dqbd_shared_kind", "") == "gpu"
            ):
                return current_submit(self, fn, *args, **kwargs)
            while True:
                try:
                    return current_submit(self, fn, *args, **kwargs)
                except RuntimeError as exc:
                    if "DQBD_RECLAIMABLE_LANE_POOL_FULL" not in str(exc):
                        raise
                    if getattr(self, "_closed", False):
                        raise
                    time.sleep(.005)

        submit._dqbd_gpu_backpressure = True
        cls.submit = submit


def _patch_lane_watchdog() -> None:
    from . import dynamic_qbd_runtime_hardening as hardening

    cls = coordinator._LaneHealthWatchdog
    if getattr(cls._tick, "_dqbd_exists_shared_scope", False):
        return

    def tick(self) -> None:
        stale_activity_age = None
        if self.activity_heartbeat_path is not None:
            try:
                activity = json.loads(
                    self.activity_heartbeat_path.read_text(encoding="utf-8")
                )
                if (
                    activity.get("state") == "ACTIVE"
                    and str(activity.get("owner", "")) == self.owner
                ):
                    stale_activity_age = (
                        time.time() - float(activity.get("updated_at_epoch", 0.0))
                    )
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                stale_activity_age = None

        stale_dispatcher = False
        if (
            stale_activity_age is not None
            and stale_activity_age > self.stale_after_seconds
        ):
            stale_dispatcher = bool(self.store.ready_job_exists())

        target_root = _norm_path(self.store.root)
        for pool in self.pools:
            if pool is None:
                continue
            try:
                if hasattr(pool, "repair_offline_lanes"):
                    repaired = pool.repair_offline_lanes()
                    for lane_index in repaired:
                        self._record({
                            "event": "OFFLINE_LANE_REPAIRED",
                            "lane_index": int(lane_index),
                            "role": str(getattr(pool, "role", "")),
                        })

                active = pool.active_lanes()
                for lane in active:
                    lane_root = str(lane.get("store_root", ""))
                    if lane_root and _norm_path(lane_root) != target_root:
                        continue
                    lane_job_id = str(lane.get("job_id", ""))
                    manifest_job_id = str(lane.get("manifest_job_id") or lane_job_id)
                    if not lane_job_id or not manifest_job_id:
                        continue

                    if stale_dispatcher and hasattr(pool, "reclaim_job"):
                        failure = BrokenProcessPool(
                            "DQBD_STALE_DISPATCHER_ACTIVITY:"
                            f"age={stale_activity_age:.3f}"
                        )
                        reclaimed = pool.reclaim_job(
                            lane_job_id,
                            reason="DQBD_STALE_DISPATCHER_ACTIVITY",
                            released_gib=0.0,
                            failure=failure,
                        )
                        if reclaimed is not None:
                            self._record({
                                "event": (
                                    "STALE_DISPATCHER_LANE_OFFLINE"
                                    if reclaimed.get("lane_state") == hardening._LANE_OFFLINE
                                    else "STALE_DISPATCHER_LANE_RECLAIMED"
                                ),
                                "job_id": manifest_job_id,
                                "lane_job_id": lane_job_id,
                                "activity_age_seconds": float(stale_activity_age),
                                **dict(reclaimed),
                            })
                        else:
                            self.store.heartbeat(
                                manifest_job_id,
                                self.owner,
                                lease_seconds=coordinator.JOB_LEASE_SECONDS,
                            )
                            self._record({
                                "event": (
                                    "STALE_DISPATCHER_TERMINATION_DEFERRED_"
                                    "HEARTBEAT_PRESERVED"
                                ),
                                "job_id": manifest_job_id,
                                "lane_job_id": lane_job_id,
                                "activity_age_seconds": float(stale_activity_age),
                            })
                        continue

                    self.store.heartbeat(
                        manifest_job_id,
                        self.owner,
                        lease_seconds=coordinator.JOB_LEASE_SECONDS,
                    )

                reclaimed = pool.reclaim_unhealthy_lanes(
                    startup_grace_seconds=self.startup_grace_seconds
                )
                for row in reclaimed:
                    self._record({
                        "event": "ORPHANED_LANE_RECLAIMED",
                        "role": str(getattr(pool, "role", "")),
                        **row,
                    })
            except BaseException as exc:
                self._record({
                    "event": "WATCHDOG_TICK_ERROR",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                })

    tick._dqbd_exists_shared_scope = True
    cls._tick = tick


def _patch_batched_runtime() -> None:
    module = importlib.import_module(
        ".dynamic_qbd_batched_portfolio_runtime", package=__package__
    )

    process_fn = module._portfolio_batch_process
    if not getattr(process_fn, "_dqbd_single_store_open", False):
        process_source = textwrap.dedent(inspect.getsource(process_fn))
        old = (
            "handlers = build_manifested_job_handlers(\n"
            "        store=ManifestedJobStore(output_root),\n"
            "        inputs=inputs,\n"
            "        output_root=output_root,\n"
            "        evidence_snapshot_root=payload.get(\"evidence_snapshot_root\"),\n"
            "        recipe_selection_root=payload.get(\"recipe_selection_root\"),\n"
            "    )\n"
            "    store = ManifestedJobStore(output_root)"
        )
        new = (
            "store = ManifestedJobStore(output_root)\n"
            "    handlers = build_manifested_job_handlers(\n"
            "        store=store,\n"
            "        inputs=inputs,\n"
            "        output_root=output_root,\n"
            "        evidence_snapshot_root=payload.get(\"evidence_snapshot_root\"),\n"
            "        recipe_selection_root=payload.get(\"recipe_selection_root\"),\n"
            "    )"
        )
        if old not in process_source:
            raise RuntimeError("DQBD_PORTFOLIO_SINGLE_STORE_PATCH_ANCHOR_INVALID")
        process_source = process_source.replace(old, new, 1)
        patched_process = _recompile_function(module, process_fn, process_source)
        patched_process._dqbd_single_store_open = True
        module._portfolio_batch_process = patched_process

    fn = module.execute_manifested_development_jobs
    if getattr(fn, "_dqbd_scheduler_efficiency", False):
        return
    source = textwrap.dedent(inspect.getsource(fn))

    old_batches = (
        "max_physical_batches=(\n"
        "                    None if portfolio_budget is None else min(\n"
        "                        1, portfolio_budget)),"
    )
    new_batches = (
        "max_physical_batches=(\n"
        "                    None if portfolio_budget is None else min(\n"
        "                        int(workers), portfolio_budget)),"
    )
    if old_batches not in source:
        raise RuntimeError("DQBD_PORTFOLIO_BATCH_WIDTH_PATCH_ANCHOR_INVALID")
    source = source.replace(old_batches, new_batches, 1)
    source = source.replace(
        '"DQBD_CANDIDATE_MAX_INFLIGHT", "26"',
        '"DQBD_CANDIDATE_MAX_INFLIGHT", str(workers)',
    )
    source = source.replace(
        "pending_coverage = store.jobs_by_kind_state(\n"
        "            kind=\"candidate_evidence_coverage\", state=\"PENDING\")",
        "pending_coverage = store.pending_cutoff_frontier(\n"
        "            kind=\"candidate_evidence_coverage\", limit=256)",
        1,
    )

    patched = _recompile_function(module, fn, source)
    patched._dqbd_scheduler_efficiency = True
    module.execute_manifested_development_jobs = patched


def _capacity_allocate_seed_lanes(
    *, worker_map: list[dict], worker_count: int,
    ready_by_seed: Mapping[str, int], seed_names: Iterable[str],
):
    names = tuple(str(name) for name in seed_names)
    active = [name for name in names if int(ready_by_seed.get(name, 0)) > 0]
    counts = {name: 0 for name in names}
    if not active:
        return counts, {name: (0, []) for name in names}

    total_workers = max(1, int(worker_count))
    capacities = {
        name: max(2, min(total_workers, int(ready_by_seed.get(name, 0))))
        for name in active
    }
    remaining = total_workers
    while remaining > 0:
        progressed = False
        for name in active:
            if remaining <= 0:
                break
            if counts[name] >= capacities[name]:
                continue
            counts[name] += 1
            remaining -= 1
            progressed = True
        if not progressed:
            break

    allocations: dict[str, tuple[int, list[dict]]] = {}
    cursor = 0
    for name in names:
        count = counts[name]
        allocations[name] = (count, worker_map[cursor:cursor + count])
        cursor += count
    return counts, allocations


def _patch_runner_watchdog(module) -> None:
    cls = module._SeedBankWatchdog
    if getattr(cls._tick, "_dqbd_exists_watchdog", False):
        return
    source = textwrap.dedent(inspect.getsource(cls._tick))
    source = source.replace(
        "store.ready_job_count() <= 0",
        "not store.ready_job_exists()",
    )
    source = source.replace(
        "ready_job_count=store.ready_job_count(),",
        "ready_job_count=store.ready_job_count_bounded(limit=4096),",
    )
    source = source.replace(
        "store.ready_job_count() > 0",
        "store.ready_job_exists()",
    )
    patched = _recompile_function(module, cls._tick, source)
    patched._dqbd_exists_watchdog = True
    cls._tick = patched


def _patch_runner_module(module) -> None:
    if getattr(module, "_dqbd_scheduler_efficiency_v41", False):
        return

    _patch_runner_watchdog(module)
    module._allocate_seed_lanes = _capacity_allocate_seed_lanes

    run_fn = module.run
    source = textwrap.dedent(inspect.getsource(run_fn))
    source = source.replace(
        "checkpoint_root: Path | None = None, worker_count: int = 26,",
        "checkpoint_root: Path | None = None, worker_count: int = 32,",
        1,
    )
    source = source.replace(
        "    repo_root = Path(__file__).resolve().parents[3]\n",
        "    os.environ['DQBD_SCHEDULER_EFFICIENCY_V41'] = '1'\n"
        "    repo_root = Path(__file__).resolve().parents[3]\n",
        1,
    )
    source = source.replace(
        "    worker_count = int(worker_count)\n",
        "    worker_count = max(\n"
        "        int(worker_count),\n"
        "        min(32, logical_processors, MAX_RUNTIME_PROCESS_LANES),\n"
        "    )\n",
        1,
    )
    source = source.replace(
        "v40 defaults to 26 logical lanes. A later 32-lane host test is allowed\n"
        "    # explicitly; RAM admission, not hidden oversubscription, remains the gate.",
        "Step-9 targets 32 shared CPU lanes. RAM admission, not hidden\n"
        "    # oversubscription, remains the authority for executable concurrency.",
        1,
    )
    source = source.replace(
        '"schema_version": "DQBD_STEP9_FIXED_CAUSAL_MODEL_STORE_RUN_V40_1_BATCHED_PORTFOLIO_VALIDATED_RESUME",',
        '"schema_version": "DQBD_STEP9_FIXED_CAUSAL_MODEL_STORE_RUN_V40_1_SCHEDULER_EFFICIENCY",',
        1,
    )
    source = source.replace(
        "return max(\n            len(SEEDS),\n"
        "            min(requested, logical_processors, MAX_RUNTIME_PROCESS_LANES))",
        "return max(\n            current,\n            len(SEEDS),\n"
        "            min(requested, logical_processors, MAX_RUNTIME_PROCESS_LANES))",
        1,
    )
    source = source.replace('"gpu_workers_per_device": 1', '"gpu_workers_per_device": 4')
    source = source.replace(
        'seed_contexts[name]["store"].ready_job_count()',
        'seed_contexts[name]["store"].scheduler_ready_capacity(limit=32)',
    )

    old_wave = (
        "selected_seed, next_seed_index = _next_ready_seed_wave(\n"
        "                seed_names=SEEDS, remaining=remaining,\n"
        "                ready_by_seed=ready_by_seed, next_index=next_seed_index)\n"
        "            if selected_seed is None:\n"
        "                raise RuntimeError(\n"
        "                    f\"DQBD_STEP9_SEED_FRONTIER_STARVED:{ready_by_seed}\")\n"
        "            # Only one seed bank owns pools in a wave to prevent multiplying\n"
        "            # 32 CPU lanes plus two GPU lanes across all three banks. The\n"
        "            # bounded causal quantum above rotates ownership, so a bank can\n"
        "            # use the complete pool without starving its siblings.\n"
        "            active_names = (selected_seed,)\n"
        "            allocation_ready_by_seed = {\n"
        "                name: ready_by_seed.get(name, 0)\n"
        "                if name in active_names else 0\n"
        "                for name in SEEDS\n"
        "            }"
    )
    new_wave = (
        "active_names = tuple(\n"
        "                name for name in SEEDS\n"
        "                if name in remaining and ready_by_seed.get(name, 0) > 0\n"
        "            )\n"
        "            if not active_names:\n"
        "                raise RuntimeError(\n"
        "                    f\"DQBD_STEP9_SEED_FRONTIER_STARVED:{ready_by_seed}\")\n"
        "            # CPU/GPU processes are persistent shared physical pools. READY\n"
        "            # banks may therefore execute concurrently without multiplying\n"
        "            # the 32 CPU lanes or per-device 4+2 GPU staging capacity.\n"
        "            allocation_ready_by_seed = {\n"
        "                name: ready_by_seed.get(name, 0)\n"
        "                if name in active_names else 0\n"
        "                for name in SEEDS\n"
        "            }"
    )
    if old_wave not in source:
        raise RuntimeError("DQBD_CROSS_SEED_WAVE_PATCH_ANCHOR_INVALID")
    source = source.replace(old_wave, new_wave, 1)
    source = source.replace(
        "seed_pool = ThreadPoolExecutor(\n                max_workers=1,",
        "seed_pool = ThreadPoolExecutor(\n                max_workers=max(1, len(active_names)),",
        1,
    )

    patched_run = _recompile_function(module, run_fn, source)
    patched_run._dqbd_scheduler_efficiency = True
    module.run = patched_run

    main_fn = module.main
    main_source = textwrap.dedent(inspect.getsource(main_fn))
    main_source = main_source.replace(
        'parser.add_argument(\n        "--workers", type=int, default=26,',
        'parser.add_argument(\n        "--workers", type=int, default=32,',
        1,
    )
    main_source = main_source.replace(
        'help="Logical process-lane capacity. Keep 26 for the v40.0.4.3 execution baseline; 32 remains an explicit later host test.")',
        'help="Logical process-lane capacity. Step-9 targets 32 shared CPU lanes; RAM admission remains authoritative.")',
        1,
    )
    patched_main = _recompile_function(module, main_fn, main_source)
    patched_main._dqbd_scheduler_efficiency = True
    module.main = patched_main

    evaluate = getattr(module, "_evaluate_seed", None)
    if evaluate is not None:
        evaluate_source = textwrap.dedent(inspect.getsource(evaluate))
        evaluate_source = evaluate_source.replace(
            '"requested_worker_count": 26, "queue_capacity": 52,',
            '"requested_worker_count": 32, "queue_capacity": 64,',
        )
        module._evaluate_seed = _recompile_function(module, evaluate, evaluate_source)

    module._dqbd_scheduler_efficiency_v41 = True


def _bind_runner_patch() -> None:
    from . import dynamic_qbd_runtime_hardening_extensions as ext

    current = ext._patch_runner_module
    if getattr(current, "_dqbd_scheduler_efficiency", False):
        existing = sys.modules.get(ext._RUNNER_MODULE)
        if existing is not None:
            current(existing)
        return

    @wraps(current)
    def combined(module):
        current(module)
        _patch_runner_module(module)

    combined._dqbd_scheduler_efficiency = True
    ext._patch_runner_module = combined

    existing = sys.modules.get(ext._RUNNER_MODULE)
    if existing is not None:
        combined(existing)


def scheduler_efficiency_state() -> dict[str, Any]:
    with _POOL_LOCK:
        pools = [
            {
                "runtime_root": key[0],
                "kind": key[1],
                "device_index": key[2],
                "workers": len(getattr(pool, "worker_map", ())),
                "max_pending": int(getattr(pool, "max_pending", 0)),
                "submission_depth": (
                    pool.dqbd_submission_depth()
                    if hasattr(pool, "dqbd_submission_depth") else None
                ),
            }
            for key, pool in _PERSISTENT_POOLS.items()
        ]
    with _READY_LOCK:
        frontier = dict(_READY_STATS)
        frontier["cache_entries"] = len(_READY_CACHE)
    return {
        "schema_version": "DQBD_SCHEDULER_EFFICIENCY_V41",
        "enabled": _execution_efficiency_enabled(),
        "cpu_lanes": _CPU_LANES,
        "gpu_workers_per_device": _GPU_WORKERS_PER_DEVICE,
        "gpu_queue_ahead": _GPU_QUEUE_AHEAD,
        "gpu_global_capacity": _GPU_GLOBAL_CAPACITY,
        "persistent_pools": pools,
        "ready_frontier": frontier,
    }


def install_scheduler_efficiency() -> None:
    global _PRE_INSTALLED
    if _PRE_INSTALLED:
        return
    _patch_gpu_topology()
    _patch_store_open_and_frontier()
    _patch_persistent_pools()
    atexit.register(close_persistent_scheduler_pools)
    _PRE_INSTALLED = True


def install_scheduler_efficiency_post_hardening() -> None:
    global _POST_INSTALLED
    if _POST_INSTALLED:
        return
    _patch_post_hardening_pool_methods()
    _patch_lane_watchdog()
    _patch_batched_runtime()
    _bind_runner_patch()
    _POST_INSTALLED = True
