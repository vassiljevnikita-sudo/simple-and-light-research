"""Focused regression checks for v40.1 execution hardening.

No model fit, portfolio replay, prospective holdout read or heavy Step-9 work is
performed.  The test covers only scheduler/checkpoint recovery mechanics.
"""
from __future__ import annotations

from concurrent.futures import Future
import json
from pathlib import Path
import tempfile
import threading
import time

from .dynamic_qbd_attempt_fenced_store import (
    AttemptFencedManifestedJobStore,
)
from .dynamic_qbd_manifested_job_coordinator import (
    WORKER_FAILURE_RETRY_LIMIT,
    ManifestedJobStore,
    RamAdmissionScheduler,
    _LaneHealthWatchdog,
    _ReclaimableLanePool,
)
from .dynamic_qbd_runtime_hardening import (
    _LANE_OFFLINE,
    _LANE_ONLINE,
    _safe_write_json,
)


class _FakeExecutor:
    def __init__(self) -> None:
        self._processes = {}
        self.shutdown_calls = 0

    def shutdown(self, wait=False, cancel_futures=False) -> None:
        self.shutdown_calls += 1


class _DeferredPool:
    role = "test-deferred"

    def __init__(self, job_id: str) -> None:
        self.job_id = str(job_id)
        self.reclaim_calls = 0

    def repair_offline_lanes(self):
        return []

    def active_lanes(self):
        return [{"job_id": self.job_id, "lane_index": 0, "pid": 123}]

    def reclaim_job(self, job_id, **kwargs):
        assert str(job_id) == self.job_id
        self.reclaim_calls += 1
        return None

    def reclaim_unhealthy_lanes(self, **kwargs):
        return []


class _HeartbeatStore:
    def __init__(self) -> None:
        self.heartbeats = []

    def ready_job_count(self) -> int:
        return 1

    def heartbeat(self, job_id, owner, lease_seconds=900) -> None:
        self.heartbeats.append((str(job_id), str(owner), int(lease_seconds)))


def _pending_job(
    store: ManifestedJobStore, job_id: str, **fields,
) -> None:
    store.set_job(
        job_id,
        "PENDING",
        kind="recipe_selection",
        cutoff="2020-08-31",
        recipe_id="H09_RIDGE_HGB_FROZEN_RULE",
        **fields,
    )


def _test_attempt_aba(root: Path) -> None:
    raw = ManifestedJobStore(root / "attempt-aba")
    fenced = AttemptFencedManifestedJobStore(raw)
    owner = "attempt-owner"
    job_id = "recipe_selection:2020-08-31:H09_RIDGE_HGB_FROZEN_RULE"
    _pending_job(raw, job_id)
    first = fenced.claim_ready(
        owner,
        kinds=("recipe_selection",),
        job_ids=(job_id,),
    )
    assert first is not None and int(first["attempt"]) == 1

    # External recovery wins before the old Future settles, then the same
    # owner claims a newer attempt.  The old facade still remembers attempt 1.
    assert raw.requeue_interrupted_jobs(owner=owner) == 1
    second = raw.claim_ready(
        owner,
        kinds=("recipe_selection",),
        job_ids=(job_id,),
    )
    assert second is not None and int(second["attempt"]) == 2
    try:
        fenced.finish_job(job_id, owner, "COMPLETE", result={"bad": True})
    except RuntimeError as exc:
        assert str(exc) == "MANIFESTED_JOB_JOB_OWNERSHIP_OR_LEASE_INVALID"
    else:
        raise AssertionError("stale attempt unexpectedly published")
    current = raw.job(job_id)
    assert current is not None
    assert current["state"] == "RUNNING"
    assert int(current["attempt"]) == 2
    assert current.get("result") is None


def _test_dispatcher_accounting(root: Path) -> None:
    raw = ManifestedJobStore(root / "dispatcher-accounting")
    fenced = AttemptFencedManifestedJobStore(raw)
    owner = "dispatcher-owner"
    job_id = "recipe_selection:2020-08-31:H10_RIDGE_HGB_FROZEN_RULE"
    _pending_job(
        raw,
        job_id,
        worker_failure_count=WORKER_FAILURE_RETRY_LIMIT,
    )
    claimed = fenced.claim_ready(
        owner,
        kinds=("recipe_selection",),
        job_ids=(job_id,),
    )
    assert claimed is not None
    fenced.finish_job(
        job_id,
        owner,
        "FAILED",
        last_error=(
            "BrokenProcessPool:DQBD_STALE_DISPATCHER_ACTIVITY:age=130"
        ),
    )
    current = raw.job(job_id)
    assert current is not None and current["state"] == "PENDING"
    assert int(current.get("worker_failure_count", 0)) == (
        WORKER_FAILURE_RETRY_LIMIT
    )
    assert int(current.get("dispatcher_recovery_count", 0)) == 1


def _test_offline_lane_settles_future() -> None:
    pool = object.__new__(_ReclaimableLanePool)
    pool.worker_map = [{"logical_processor": 0}]
    pool.role = "offline-test"
    pool.telemetry_root = None
    pool.max_tasks_per_child = None
    pool.max_pending = 1
    pool._lock = threading.RLock()
    pool._closed = False
    pool._cursor = 0
    pool._queued = __import__("collections").deque()
    outer = Future()
    inner = Future()
    fake_executor = _FakeExecutor()
    pool._lanes = [{
        "index": 0,
        "row": {"logical_processor": 0},
        "executor": fake_executor,
        "outer": outer,
        "inner": inner,
        "metadata": {
            "job_id": "model:test",
            "kind": "model",
            "submitted_at": time.time(),
        },
        "generation": 1,
        "replacing": False,
        "state": _LANE_ONLINE,
        "repair_attempts": 0,
        "last_repair_error": None,
        "offline_since": None,
    }]

    def fail_new_executor(*args, **kwargs):
        raise OSError("synthetic replacement construction failure")

    pool._new_executor = fail_new_executor
    result = pool.reclaim_job(
        "model:test",
        reason="RAM_HARD_RECLAIM_ONE",
        released_gib=1.0,
    )
    assert result is not None
    assert result["lane_state"] == _LANE_OFFLINE
    assert outer.done()
    exc = outer.exception()
    assert exc is not None
    assert "DQBD_LANE_REPLACEMENT_FAILED" in str(exc)
    assert pool.lane_states()[0]["state"] == _LANE_OFFLINE

    repaired_executor = _FakeExecutor()
    pool._new_executor = lambda *args, **kwargs: repaired_executor
    repaired = pool.repair_offline_lanes()
    assert repaired == [0]
    assert pool.lane_states()[0]["state"] == _LANE_ONLINE
    assert pool._lanes[0]["executor"] is repaired_executor


def _test_deferred_termination_heartbeat(root: Path) -> None:
    owner = "watchdog-owner"
    job_id = "model:watchdog-test"
    activity = root / "watchdog-activity.json"
    activity.write_text(
        json.dumps({
            "state": "ACTIVE",
            "owner": owner,
            "updated_at_epoch": time.time() - 1000.0,
        }) + "\n",
        encoding="utf-8",
    )
    store = _HeartbeatStore()
    pool = _DeferredPool(job_id)
    watchdog = _LaneHealthWatchdog(
        store=store,
        owner=owner,
        pools=(pool,),
        telemetry_root=root / "watchdog-telemetry",
        activity_heartbeat_path=activity,
        stale_after_seconds=30.0,
        interval_seconds=10.0,
    )
    watchdog._tick()
    assert pool.reclaim_calls == 1
    assert store.heartbeats == [(job_id, owner, 900)]


def _test_ram_telemetry_is_advisory(root: Path) -> None:
    invalid_profile_target = root / "profile-is-directory"
    invalid_profile_target.mkdir(parents=True)
    scheduler = RamAdmissionScheduler(
        max_slots=1,
        profile_path=invalid_profile_target,
        telemetry_path=None,
    )
    try:
        scheduler.observe(
            "causal:replay:CPU",
            peak_process_gib=1.0,
            incremental_peak_gib=.5,
            descriptor={"horizon": 9, "prepared": True},
        )
        telemetry = scheduler.telemetry()
        assert telemetry["ram_scheduler_profile_write_error_count"] >= 1
        assert telemetry["ram_scheduler_controller_thread_alive"] is True
        assert scheduler.incremental_estimate_for(
            "causal:replay:CPU",
            {"horizon": 9, "prepared": True},
        ) > 0.0
    finally:
        scheduler.close()


def _test_runner_writer_hook(root: Path) -> None:
    from . import dynamic_qbd_causal_model_store_run as runner

    assert runner._json is _safe_write_json
    target = root / "runner-json.json"
    runner._json(target, {"ok": True})
    assert json.loads(target.read_text(encoding="utf-8")) == {"ok": True}


def run_self_test() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _test_attempt_aba(root)
        _test_dispatcher_accounting(root)
        _test_offline_lane_settles_future()
        _test_deferred_termination_heartbeat(root)
        _test_ram_telemetry_is_advisory(root)
        _test_runner_writer_hook(root)


if __name__ == "__main__":
    run_self_test()
    print("PASS dynamic_qbd_runtime_hardening_self_test")
