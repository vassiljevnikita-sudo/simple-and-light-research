"""Execution-only hardening for the Dynamic-QBD manifested runtime.

This module does not change scientific payloads, DAG dependencies, model
selection, RAM thresholds, GPU kernel policy or holdout authority.  It patches
only execution mechanics that can otherwise turn a recoverable local runtime
incident into a stalled or prematurely failed Step-9 bank.

The installer is idempotent and is activated from the package bootstrap before
runtime call-sites import coordinator functions.
"""
from __future__ import annotations

from collections import deque
from concurrent.futures import Future
from concurrent.futures.process import BrokenProcessPool
from functools import wraps
import json
import os
from pathlib import Path
import threading
import time
from typing import Any, Mapping

from . import dynamic_qbd_manifested_job_coordinator as coordinator
from .dynamic_qbd_attempt_fenced_store import (
    AttemptFencedManifestedJobStore,
)


_INSTALLED = False
_LANE_ONLINE = "ONLINE"
_LANE_REPAIRING = "REPAIRING"
_LANE_OFFLINE = "OFFLINE"
_EXECUTION_RECOVERY_MARKERS = (
    "DQBD_STALE_DISPATCHER_ACTIVITY",
    "DQBD_LANE_REPLACEMENT_FAILED",
)


def _safe_atomic_text(path: Path, text: str) -> None:
    """Atomic UTF-8 publication with thread-unique temp identity.

    Publication semantics are unchanged: after bounded sharing-violation
    retries, authoritative callers still receive the OSError.  The unique temp
    name removes same-process thread collisions that PID-only names allowed.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(
        target.name
        + f".{os.getpid()}.{threading.get_ident()}.{time.time_ns()}.tmp"
    )
    temporary.write_text(text, encoding="utf-8")
    for attempt in range(8):
        try:
            os.replace(temporary, target)
            return
        except OSError:
            if attempt == 7:
                try:
                    temporary.unlink()
                except OSError:
                    pass
                raise
            time.sleep(.02 * (attempt + 1))


def _safe_write_json(path: Path, payload: Any) -> None:
    _safe_atomic_text(
        Path(path),
        json.dumps(payload, default=str, indent=2, sort_keys=True) + "\n",
    )


def _patch_manifest_json_writer() -> None:
    def materialize_json_manifest(self) -> None:
        with self._connect() as db:
            target = self.root / "run-state" / "jobs.json"
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(
                target.name
                + f".{os.getpid()}.{threading.get_ident()}."
                + f"{time.time_ns()}.tmp"
            )
            try:
                with temporary.open("w", encoding="utf-8") as handle:
                    handle.write("{\n")
                    first = True
                    for payload, in db.execute(
                        "SELECT payload FROM jobs ORDER BY job_id"
                    ):
                        job = json.loads(payload)
                        if not first:
                            handle.write(",\n")
                        first = False
                        handle.write(json.dumps(str(job["job_id"])))
                        handle.write(":")
                        handle.write(
                            json.dumps(job, default=str, sort_keys=True)
                        )
                    handle.write("\n}\n")
                for attempt in range(8):
                    try:
                        os.replace(temporary, target)
                        return
                    except OSError:
                        if attempt == 7:
                            raise
                        time.sleep(.02 * (attempt + 1))
            finally:
                try:
                    temporary.unlink()
                except OSError:
                    pass

    coordinator._write_json = _safe_write_json
    coordinator.ManifestedJobStore.materialize_json_manifest = (
        materialize_json_manifest
    )


def _patch_attempt_fenced_dispatch() -> None:
    original_candidate = coordinator.execute_candidate_oos_jobs
    original_ready = coordinator.execute_ready_jobs

    @wraps(original_candidate)
    def candidate_wrapper(*args, **kwargs):
        if args:
            raise TypeError(
                "DQBD_ATTEMPT_FENCE_CANDIDATE_REQUIRES_KEYWORD_ARGUMENTS"
            )
        store = AttemptFencedManifestedJobStore.ensure(kwargs["store"])
        owner = str(kwargs.get("owner", "candidate-oos-worker"))
        call = dict(kwargs)
        call["store"] = store
        consecutive_execution_recoveries = 0
        while True:
            try:
                return original_candidate(**call)
            except BrokenProcessPool as exc:
                message = str(exc)
                if not any(
                    marker in message
                    for marker in _EXECUTION_RECOVERY_MARKERS
                ):
                    raise
                # Candidate-OOS historically evaluated worker_failure_count
                # before checking whether the incident was dispatcher-only.
                # The fenced store has already prevented terminal FAILED
                # publication for these markers.  Clear the bank-local claims
                # and restart this scheduler invocation from SQLite.
                store.requeue_interrupted_jobs(owner=owner)
                consecutive_execution_recoveries += 1
                if (
                    consecutive_execution_recoveries
                    > coordinator.WORKER_FAILURE_RETRY_LIMIT * 4
                ):
                    raise
                time.sleep(.05)

    @wraps(original_ready)
    def ready_wrapper(*args, **kwargs):
        if args:
            values = list(args)
            values[0] = AttemptFencedManifestedJobStore.ensure(values[0])
            return original_ready(*values, **kwargs)
        call = dict(kwargs)
        call["store"] = AttemptFencedManifestedJobStore.ensure(call["store"])
        return original_ready(**call)

    coordinator.execute_candidate_oos_jobs = candidate_wrapper
    coordinator.execute_ready_jobs = ready_wrapper


def _patch_lane_pool() -> None:
    cls = coordinator._ReclaimableLanePool
    original_init = cls.__init__
    original_reclaim_job = cls.reclaim_job
    original_reclaim_unhealthy = cls.reclaim_unhealthy_lanes

    @wraps(original_init)
    def init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        with self._lock:
            for lane in self._lanes:
                lane["state"] = _LANE_ONLINE
                lane["repair_attempts"] = 0
                lane["last_repair_error"] = None
                lane["offline_since"] = None

    def lane_states(self) -> list[dict[str, Any]]:
        with self._lock:
            return [
                {
                    "lane_index": int(lane["index"]),
                    "state": str(lane.get("state", _LANE_ONLINE)),
                    "repair_attempts": int(
                        lane.get("repair_attempts", 0) or 0
                    ),
                    "last_repair_error": lane.get("last_repair_error"),
                    "offline_since": lane.get("offline_since"),
                }
                for lane in self._lanes
            ]

    def _mark_offline_locked(
        self, lane: dict[str, Any], exc: BaseException,
    ) -> None:
        lane["executor"] = None
        lane["replacing"] = False
        lane["state"] = _LANE_OFFLINE
        lane["repair_attempts"] = int(
            lane.get("repair_attempts", 0) or 0
        ) + 1
        lane["last_repair_error"] = f"{type(exc).__name__}:{exc}"
        lane["offline_since"] = time.time()

    def repair_offline_lanes(self) -> list[int]:
        repaired: list[int] = []
        while True:
            with self._lock:
                if self._closed:
                    return repaired
                lane = next(
                    (
                        value
                        for value in self._lanes
                        if value.get("state") == _LANE_OFFLINE
                        and value.get("outer") is None
                        and not value.get("replacing")
                    ),
                    None,
                )
                if lane is None:
                    break
                lane["state"] = _LANE_REPAIRING
                lane["replacing"] = True
                lane_index = int(lane["index"])
                row = dict(lane["row"])
            try:
                replacement = self._new_executor(lane_index, row)
            except BaseException as exc:
                with self._lock:
                    current = self._lanes[lane_index]
                    _mark_offline_locked(self, current, exc)
                break
            with self._lock:
                current = self._lanes[lane_index]
                if self._closed:
                    try:
                        replacement.shutdown(
                            wait=False, cancel_futures=True
                        )
                    except TypeError:
                        replacement.shutdown(wait=False)
                    return repaired
                current["executor"] = replacement
                current["replacing"] = False
                current["state"] = _LANE_ONLINE
                current["last_repair_error"] = None
                current["offline_since"] = None
                repaired.append(lane_index)
        if repaired:
            self._dispatch_queued()
        return repaired

    def replace_broken_lane(
        self, *, lane_index: int, generation: int,
    ) -> None:
        with self._lock:
            lane = self._lanes[int(lane_index)]
            if (
                int(lane["generation"]) != int(generation)
                or lane["replacing"]
            ):
                return
            lane["replacing"] = True
            lane["state"] = _LANE_REPAIRING
            executor = lane.get("executor")
            row = dict(lane["row"])
        try:
            if executor is not None:
                try:
                    executor.shutdown(wait=False, cancel_futures=True)
                except TypeError:
                    executor.shutdown(wait=False)
            replacement = self._new_executor(int(lane_index), row)
        except BaseException as exc:
            with self._lock:
                current = self._lanes[int(lane_index)]
                if int(current["generation"]) == int(generation):
                    _mark_offline_locked(self, current, exc)
            return
        with self._lock:
            current = self._lanes[int(lane_index)]
            if (
                int(current["generation"]) == int(generation)
                and current["replacing"]
            ):
                current["executor"] = replacement
                current["replacing"] = False
                current["state"] = _LANE_ONLINE
                current["last_repair_error"] = None
                current["offline_since"] = None
            else:
                try:
                    replacement.shutdown(
                        wait=False, cancel_futures=True
                    )
                except TypeError:
                    replacement.shutdown(wait=False)
        self._dispatch_queued()

    def dispatch_queued(self) -> None:
        attachments: list[tuple[int, int, Future, Any]] = []
        with self._lock:
            if self._closed:
                return
            while self._queued:
                selected = next(
                    (
                        lane
                        for lane in self._lanes
                        if lane["outer"] is None
                        and not lane["replacing"]
                        and lane.get("state", _LANE_ONLINE)
                        == _LANE_ONLINE
                    ),
                    None,
                )
                if selected is None:
                    break
                task = self._queued.popleft()
                try:
                    attachment = self._start_task_locked(selected, task)
                except BaseException as exc:
                    outer = task[0]
                    if not outer.done():
                        outer.set_exception(exc)
                    continue
                if attachment is not None:
                    attachments.append(attachment)
        for lane_index, generation, outer, inner in attachments:
            self._attach_relay(
                lane_index=lane_index,
                generation=generation,
                outer=outer,
                inner=inner,
            )

    def submit(self, fn, /, *args, **kwargs) -> Future:
        # Opportunistic repair means an OFFLINE lane can recover even for a
        # pool without a LaneHealthWatchdog (for example one portfolio batch).
        self.repair_offline_lanes()
        with self._lock:
            if self._closed:
                raise RuntimeError("DQBD_RECLAIMABLE_LANE_POOL_CLOSED")
            count = len(self._lanes)
            outer = Future()
            task = (
                outer,
                fn,
                args,
                kwargs,
                self._task_metadata(args),
            )
            selected = None
            for offset in range(count):
                index = (self._cursor + offset) % count
                lane = self._lanes[index]
                if (
                    lane["outer"] is None
                    and not lane["replacing"]
                    and lane.get("state", _LANE_ONLINE)
                    == _LANE_ONLINE
                    and not self._queued
                ):
                    selected = lane
                    self._cursor = (index + 1) % count
                    break
            if selected is None:
                active = sum(
                    1
                    for lane in self._lanes
                    if lane["outer"] is not None
                    or lane["replacing"]
                    or lane.get("state", _LANE_ONLINE)
                    != _LANE_ONLINE
                )
                if active + len(self._queued) >= self.max_pending:
                    raise RuntimeError("DQBD_RECLAIMABLE_LANE_POOL_FULL")
                self._queued.append(task)
                return outer
            attachment = self._start_task_locked(selected, task)
            if attachment is None:
                return outer
            lane_index, generation, outer, inner = attachment
        self._attach_relay(
            lane_index=lane_index,
            generation=generation,
            outer=outer,
            inner=inner,
        )
        return outer

    @wraps(original_reclaim_job)
    def reclaim_job(
        self, job_id: str, *, reason: str, released_gib: float,
        failure: BaseException | None = None,
    ):
        identity = str(job_id)
        try:
            result = original_reclaim_job(
                self,
                identity,
                reason=reason,
                released_gib=released_gib,
                failure=failure,
            )
            if result is not None:
                with self._lock:
                    lane_index = int(result["lane_index"])
                    lane = self._lanes[lane_index]
                    lane["state"] = _LANE_ONLINE
                    lane["last_repair_error"] = None
                    lane["offline_since"] = None
            return result
        except BaseException as repair_exc:
            with self._lock:
                lane = next(
                    (
                        value
                        for value in self._lanes
                        if value.get("replacing")
                        and str(
                            (value.get("metadata") or {}).get(
                                "job_id", ""
                            )
                        )
                        == identity
                    ),
                    None,
                )
                if lane is None:
                    raise
                lane_index = int(lane["index"])
                outer = lane.get("outer")
                metadata = dict(lane.get("metadata") or {})
                pid = None
                executor = lane.get("executor")
                for process in tuple(
                    getattr(executor, "_processes", {}).values()
                    if executor is not None
                    else ()
                ):
                    value = getattr(process, "pid", None)
                    if value is not None:
                        pid = int(value)
                        break
                lane["inner"] = None
                lane["outer"] = None
                lane["metadata"] = None
                _mark_offline_locked(self, lane, repair_exc)
            message = (
                "DQBD_LANE_REPLACEMENT_FAILED:"
                f"lane={lane_index}:job={identity}:"
                f"repair={type(repair_exc).__name__}:{repair_exc}"
            )
            if failure is not None:
                message += (
                    f":trigger={type(failure).__name__}:{failure}"
                )
            transient = BrokenProcessPool(message)
            if outer is not None and not outer.done():
                outer.set_exception(transient)
            self._dispatch_queued()
            return {
                **metadata,
                "lane_index": lane_index,
                "pid": pid,
                "released_gib": float(released_gib),
                "terminated": True,
                "reason": str(reason),
                "lane_state": _LANE_OFFLINE,
                "replacement_error": (
                    f"{type(repair_exc).__name__}:{repair_exc}"
                ),
            }

    @wraps(original_reclaim_unhealthy)
    def reclaim_unhealthy_lanes(self, *args, **kwargs):
        self.repair_offline_lanes()
        return original_reclaim_unhealthy(self, *args, **kwargs)

    cls.__init__ = init
    cls.lane_states = lane_states
    cls.repair_offline_lanes = repair_offline_lanes
    cls._replace_broken_lane = replace_broken_lane
    cls._dispatch_queued = dispatch_queued
    cls.submit = submit
    cls.reclaim_job = reclaim_job
    cls.reclaim_unhealthy_lanes = reclaim_unhealthy_lanes


def _scheduler_health_init(self) -> None:
    self._runtime_health_lock = threading.Lock()
    self._sample_error_count = 0
    self._consecutive_sample_errors = 0
    self._last_sample_error = None
    self._last_sample_success_epoch = None
    self._telemetry_write_error_count = 0
    self._last_telemetry_write_error = None
    self._last_telemetry_write_success_epoch = None
    self._profile_write_error_count = 0
    self._last_profile_write_error = None
    self._last_profile_write_success_epoch = None


def _scheduler_health_error(self, category: str, exc: BaseException) -> None:
    lock = getattr(self, "_runtime_health_lock", None)
    if lock is None:
        return
    with lock:
        if category == "sample":
            self._sample_error_count += 1
            self._consecutive_sample_errors += 1
            self._last_sample_error = f"{type(exc).__name__}:{exc}"
        elif category == "telemetry":
            self._telemetry_write_error_count += 1
            self._last_telemetry_write_error = (
                f"{type(exc).__name__}:{exc}"
            )
        elif category == "profile":
            self._profile_write_error_count += 1
            self._last_profile_write_error = (
                f"{type(exc).__name__}:{exc}"
            )


def _patch_ram_scheduler() -> None:
    cls = coordinator.RamAdmissionScheduler
    original_init = cls.__init__
    original_persist = cls._persist_profiles
    original_telemetry = cls.telemetry

    @wraps(original_init)
    def init(self, *args, **kwargs):
        _scheduler_health_init(self)
        original_init(self, *args, **kwargs)

    def controller_loop(self) -> None:
        while not self._stop_controller.wait(.01):
            try:
                total, available = self._system_memory()
                used = max(0, total - available)
                live = self._live_board.snapshot()
            except Exception as exc:
                _scheduler_health_error(self, "sample", exc)
                continue
            now = time.monotonic()
            with self._runtime_health_lock:
                self._consecutive_sample_errors = 0
                self._last_sample_success_epoch = time.time()
            with self._lock:
                self._system_samples.append((now, int(used)))
                self._live_snapshot = live
                self._peak = max(self._peak, int(used))
            if (
                self._telemetry_path is not None
                and now - self._last_telemetry_rollup
                >= self._telemetry_rollup_seconds
            ):
                try:
                    self._write_runtime_rollup(now)
                except (OSError, TypeError, ValueError) as exc:
                    _scheduler_health_error(self, "telemetry", exc)
                else:
                    with self._runtime_health_lock:
                        self._last_telemetry_write_success_epoch = time.time()
                finally:
                    self._last_telemetry_rollup = now
            with self._condition:
                self._condition.notify_all()

    @wraps(original_persist)
    def persist_profiles(self) -> None:
        try:
            original_persist(self)
        except (OSError, TypeError, ValueError) as exc:
            _scheduler_health_error(self, "profile", exc)
            # Profiles are learned execution telemetry.  The in-memory
            # observation already updated scheduling estimates; persistence
            # failure must not retroactively fail a valid worker result.
            return
        with self._runtime_health_lock:
            self._last_profile_write_success_epoch = time.time()

    @wraps(original_telemetry)
    def telemetry(self) -> dict[str, Any]:
        payload = dict(original_telemetry(self))
        with self._runtime_health_lock:
            payload.update({
                "ram_scheduler_controller_thread_alive": bool(
                    self._controller_thread.is_alive()
                ),
                "ram_scheduler_last_sample_success_epoch":
                    self._last_sample_success_epoch,
                "ram_scheduler_sample_error_count":
                    int(self._sample_error_count),
                "ram_scheduler_consecutive_sample_errors":
                    int(self._consecutive_sample_errors),
                "ram_scheduler_last_sample_error": self._last_sample_error,
                "ram_scheduler_telemetry_write_error_count":
                    int(self._telemetry_write_error_count),
                "ram_scheduler_last_telemetry_write_error":
                    self._last_telemetry_write_error,
                "ram_scheduler_last_telemetry_write_success_epoch":
                    self._last_telemetry_write_success_epoch,
                "ram_scheduler_profile_write_error_count":
                    int(self._profile_write_error_count),
                "ram_scheduler_last_profile_write_error":
                    self._last_profile_write_error,
                "ram_scheduler_last_profile_write_success_epoch":
                    self._last_profile_write_success_epoch,
            })
        return payload

    cls.__init__ = init
    cls._controller_loop = controller_loop
    cls._persist_profiles = persist_profiles
    cls.telemetry = telemetry

    controller_cls = coordinator.RamWorkerPauseController
    original_record = controller_cls._record
    original_spill = controller_cls._spill

    @wraps(original_record)
    def record(self, payload: dict[str, Any]) -> None:
        try:
            original_record(self, payload)
        except OSError as exc:
            _scheduler_health_error(self.scheduler, "telemetry", exc)

    @wraps(original_spill)
    def spill(self, *, event: str, usage: float) -> None:
        try:
            original_spill(self, event=event, usage=usage)
        except OSError as exc:
            _scheduler_health_error(self.scheduler, "telemetry", exc)

    controller_cls._record = record
    controller_cls._spill = spill


def _patch_lane_watchdog() -> None:
    cls = coordinator._LaneHealthWatchdog

    def tick(self) -> None:
        stale_activity_age = None
        if self.activity_heartbeat_path is not None:
            try:
                activity = json.loads(
                    self.activity_heartbeat_path.read_text(
                        encoding="utf-8"
                    )
                )
                if (
                    activity.get("state") == "ACTIVE"
                    and str(activity.get("owner", "")) == self.owner
                ):
                    stale_activity_age = (
                        time.time()
                        - float(activity.get("updated_at_epoch", 0.0))
                    )
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                stale_activity_age = None
        stale_dispatcher = False
        if (
            stale_activity_age is not None
            and stale_activity_age > self.stale_after_seconds
        ):
            stale_dispatcher = self.store.ready_job_count() > 0

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
                    job_id = str(lane.get("job_id", ""))
                    if not job_id:
                        continue
                    if stale_dispatcher and hasattr(pool, "reclaim_job"):
                        failure = BrokenProcessPool(
                            "DQBD_STALE_DISPATCHER_ACTIVITY:"
                            f"age={stale_activity_age:.3f}"
                        )
                        reclaimed = pool.reclaim_job(
                            job_id,
                            reason="DQBD_STALE_DISPATCHER_ACTIVITY",
                            released_gib=0.0,
                            failure=failure,
                        )
                        if reclaimed is not None:
                            self._record({
                                "event": (
                                    "STALE_DISPATCHER_LANE_OFFLINE"
                                    if reclaimed.get("lane_state")
                                    == _LANE_OFFLINE
                                    else "STALE_DISPATCHER_LANE_RECLAIMED"
                                ),
                                "job_id": job_id,
                                "activity_age_seconds": float(
                                    stale_activity_age
                                ),
                                **dict(reclaimed),
                            })
                        else:
                            # Termination was explicitly deferred because the
                            # old child could not be proven dead.  Preserve
                            # this exact attempt's lease until either the child
                            # settles or a later tick can reclaim it.  Dropping
                            # the heartbeat here recreates the ABA surface.
                            self.store.heartbeat(
                                job_id,
                                self.owner,
                                lease_seconds=coordinator.JOB_LEASE_SECONDS,
                            )
                            self._record({
                                "event": (
                                    "STALE_DISPATCHER_TERMINATION_DEFERRED_"
                                    "HEARTBEAT_PRESERVED"
                                ),
                                "job_id": job_id,
                                "activity_age_seconds": float(
                                    stale_activity_age
                                ),
                            })
                        continue
                    self.store.heartbeat(
                        job_id,
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

    cls._tick = tick


def install_runtime_hardening() -> None:
    """Install the v40.1 execution hardening exactly once per interpreter."""
    global _INSTALLED
    if _INSTALLED:
        return
    _patch_manifest_json_writer()
    _patch_lane_pool()
    _patch_ram_scheduler()
    _patch_lane_watchdog()
    _patch_attempt_fenced_dispatch()
    _INSTALLED = True
