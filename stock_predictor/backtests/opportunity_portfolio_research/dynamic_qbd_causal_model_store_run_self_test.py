"""Focused self-test for independent seed-bank recovery."""
from __future__ import annotations

import json
from pathlib import Path
import sqlite3
import tempfile
import threading
import time

from .dynamic_qbd_causal_model_store_run import (
    _SeedBankWatchdog, _allocate_seed_lanes, _next_ready_seed_wave,
)
from .dynamic_qbd_manifested_job_coordinator import (
    ManifestedJobStore, _LaneHealthWatchdog, _causal_ready_kind_priority,
)


class _FailOnceRecoveryStore:
    """Fault-injection proxy: first recovery write fails, next succeeds."""

    def __init__(self, store: ManifestedJobStore):
        self._store = store
        self.recovery_attempts = 0

    def __getattr__(self, name):
        return getattr(self._store, name)

    def requeue_interrupted_jobs(self, **kwargs):
        self.recovery_attempts += 1
        if self.recovery_attempts == 1:
            raise sqlite3.OperationalError("injected transient recovery failure")
        return self._store.requeue_interrupted_jobs(**kwargs)


class _BlockingHeartbeatStore:
    """Fault-injection proxy: one bank blocks its liveness read."""

    def __init__(self, store: ManifestedJobStore, started: threading.Event):
        self._store = store
        self._started = started

    def __getattr__(self, name):
        return getattr(self._store, name)

    def running_job_heartbeat_stats(self, owner):
        self._started.set()
        time.sleep(3.0)
        return self._store.running_job_heartbeat_stats(owner)


class _LaneWatchdogStore:
    """Minimal store double for dispatcher-stall recovery tests."""

    def __init__(self, ready_jobs: int):
        self.ready_jobs = int(ready_jobs)
        self.heartbeats = []

    def ready_job_count(self):
        return self.ready_jobs

    def heartbeat(self, job_id, owner, lease_seconds):
        self.heartbeats.append((job_id, owner, lease_seconds))


class _LaneWatchdogPool:
    """Minimal independently-replaceable lane pool double."""

    def __init__(self, job_id="stale-job"):
        self.job_id = job_id
        self.reclaimed = []

    def active_lanes(self):
        return [{"job_id": self.job_id, "lane_index": 0}]

    def reclaim_job(self, job_id, **kwargs):
        self.reclaimed.append((job_id, kwargs))
        return {"lane_index": 0, "pid": 1234, "reason": kwargs["reason"]}

    def reclaim_unhealthy_lanes(self, **kwargs):
        return []


def main() -> int:
    root = Path(tempfile.mkdtemp(prefix="dqbd-seed-bank-recovery-self-test-"))
    stores = {
        name: ManifestedJobStore(root / name.lower())
        for name in ("SHORT", "PRIMARY", "LONG")
    }
    owners = {name: f"self-test-{name.lower()}" for name in stores}
    now = time.time()
    for name, store in stores.items():
        stalled = name == "PRIMARY"
        store.set_job(
            f"{name.lower()}-stalled", "RUNNING", kind="model",
            depends_on=[], lease_owner=owners[name],
            lease_until=(now - 1.0 if stalled else now + 300.0),
            heartbeat=(now - 301.0 if stalled else now),
        )
        if stalled:
            with sqlite3.connect(store.db_path) as db:
                db.execute(
                    "UPDATE jobs SET lease_until=?, heartbeat=? WHERE job_id=?",
                    (now - 1.0, now - 301.0, f"{name.lower()}-stalled"),
                )
        store.set_job(
            f"{name.lower()}-complete", "COMPLETE", kind="model",
            depends_on=[], result={"sentinel": name},
        )

    watchdog = _SeedBankWatchdog(
        stores=stores, owners=owners,
        activity_paths={
            name: root / f"{name.lower()}-activity.json"
            for name in stores
        },
        telemetry_path=root / "watchdog-events.jsonl",
    )
    # Inject a stall only into PRIMARY. A single tick must recover that bank
    # and leave SHORT/LONG and every COMPLETE checkpoint untouched.
    watchdog._tick()
    assert stores["PRIMARY"].job("primary-stalled")["state"] == "PENDING"
    assert stores["SHORT"].job("short-stalled")["state"] == "RUNNING"
    assert stores["LONG"].job("long-stalled")["state"] == "RUNNING"
    assert stores["PRIMARY"].job("primary-complete")["state"] == "COMPLETE"
    assert watchdog.failure is None
    assert watchdog.recovery_counts == {"PRIMARY": 1}

    recovered = stores["PRIMARY"].claim_ready(
        "primary-retry", lease_seconds=60, kinds=("model",),
        job_ids=("primary-stalled",))
    assert recovered is not None
    events = [
        json.loads(line) for line in
        (root / "watchdog-events.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert any(
        event.get("event") == "SEED_BANK_SELF_HEALED"
        and event.get("seed_name") == "PRIMARY"
        and event.get("requeued_jobs") == 1
        for event in events
    )

    # Fault injection 2: SHORT has READY work but no running owner and its
    # activity heartbeat is stale. The watchdog must record a local
    # starvation recovery; it must not stop the whole run.
    stores["SHORT"].requeue_interrupted_jobs(owner=owners["SHORT"])
    (root / "short-activity.json").write_text(json.dumps({
        "schema_version": "DQBD_SEED_BANK_ACTIVITY_V1",
        "state": "ACTIVE", "owner": owners["SHORT"],
        "updated_at_epoch": now - 301.0,
    }), encoding="utf-8")
    watchdog._tick()
    events = [
        json.loads(line) for line in
        (root / "watchdog-events.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert any(
        event.get("event") == "SEED_BANK_QUEUE_STARVATION_SELF_HEALED"
        and event.get("seed_name") == "SHORT"
        for event in events
    )

    # Fault injection 3: the recovery write fails once.  The watchdog must
    # remain usable, record the error, retry on the next tick, and leave all
    # ready work claimable rather than stranded in RUNNING/PENDING limbo.
    fault_root = root / "fault-injection"
    fault_store = ManifestedJobStore(fault_root)
    fault_owner = "self-test-fault-owner"
    stale_now = time.time()
    fault_store.set_job(
        "fault-stalled", "RUNNING", kind="model", depends_on=[],
        lease_owner=fault_owner, lease_until=stale_now - 1.0,
        heartbeat=stale_now - 301.0,
    )
    # set_job stamps a fresh heartbeat for every RUNNING transition; overwrite
    # it through the same persisted state channel used by a crashed worker.
    with sqlite3.connect(fault_store.db_path) as db:
        db.execute(
            "UPDATE jobs SET lease_until=?, heartbeat=? WHERE job_id=?",
            (stale_now - 1.0, stale_now - 301.0, "fault-stalled"),
        )
    for index in range(8):
        fault_store.set_job(
            f"fault-ready-{index}", "PENDING", kind="model",
            depends_on=[],
        )
    fault_proxy = _FailOnceRecoveryStore(fault_store)
    fault_watchdog = _SeedBankWatchdog(
        stores={"FAULT": fault_proxy}, owners={"FAULT": fault_owner},
        telemetry_path=root / "fault-watchdog-events.jsonl",
        interval_seconds=1.0,
    )
    # Exercise the real daemon loop, not only a direct private-method call.
    # The first injected failure must not terminate the watchdog thread.
    fault_watchdog.start()
    deadline = time.time() + 8.0
    while time.time() < deadline:
        if (fault_proxy.recovery_attempts >= 2 and
                fault_store.job("fault-stalled")["state"] == "PENDING"):
            break
        time.sleep(0.05)
    fault_watchdog.close()
    assert fault_proxy.recovery_attempts >= 2
    assert fault_store.job("fault-stalled")["state"] == "PENDING"
    claimed = []
    for index in range(8):
        job = fault_store.claim_ready(
            "fault-retry", lease_seconds=60, kinds=("model",),
            job_ids=(f"fault-ready-{index}",),
        )
        assert job is not None
        claimed.append(job["job_id"])
    fault_events = [
        json.loads(line) for line in
        (root / "fault-watchdog-events.jsonl").read_text(
            encoding="utf-8").splitlines() if line.strip()
    ]
    assert any(
        event.get("event") == "SEED_BANK_WATCHDOG_RECOVERY_ERROR"
        and event.get("seed_name") == "FAULT"
        for event in fault_events
    )
    assert len(claimed) == 8

    # Fault injection 4: the worker lease is still valid, but the bank
    # dispatcher heartbeat is stale.  Ready work must not wait for the long
    # lease timeout; the bank activity signal is the stronger liveness gate.
    lease_root = root / "lease-valid-stall"
    lease_store = ManifestedJobStore(lease_root)
    lease_owner = "self-test-lease-owner"
    lease_now = time.time()
    lease_store.set_job(
        "lease-stalled", "RUNNING", kind="model", depends_on=[],
        lease_owner=lease_owner, lease_until=lease_now + 900.0,
    )
    lease_store.set_job(
        "lease-ready", "PENDING", kind="model", depends_on=[],
    )
    with sqlite3.connect(lease_store.db_path) as db:
        db.execute(
            "UPDATE jobs SET lease_until=?, heartbeat=? WHERE job_id=?",
            (lease_now + 900.0, lease_now - 301.0, "lease-stalled"),
        )
    lease_activity = lease_root / "activity.json"
    lease_activity.write_text(json.dumps({
        "schema_version": "DQBD_SEED_BANK_ACTIVITY_V1",
        "state": "ACTIVE", "owner": lease_owner,
        "updated_at_epoch": lease_now - 301.0,
    }), encoding="utf-8")
    lease_watchdog = _SeedBankWatchdog(
        stores={"LEASE": lease_store}, owners={"LEASE": lease_owner},
        activity_paths={"LEASE": lease_activity},
        telemetry_path=root / "lease-watchdog-events.jsonl",
    )
    lease_watchdog._tick()
    assert lease_store.job("lease-stalled")["state"] == "PENDING"
    lease_events = [
        json.loads(line) for line in
        (root / "lease-watchdog-events.jsonl").read_text(
            encoding="utf-8").splitlines() if line.strip()
    ]
    assert any(
        event.get("event") == "SEED_BANK_ACTIVITY_STALE_SELF_HEALED"
        and event.get("seed_name") == "LEASE"
        for event in lease_events
    )

    # Fault injection 5: one bank's liveness read blocks.  Independent bank
    # watchdog loops must still recover a second stale bank during that block.
    isolation_root = root / "watchdog-isolation"
    blocked_store = ManifestedJobStore(isolation_root / "blocked")
    recover_store = ManifestedJobStore(isolation_root / "recover")
    blocked_started = threading.Event()
    blocked_owner = "self-test-blocked-owner"
    recover_owner = "self-test-recover-owner"
    isolation_now = time.time()
    for store, owner, prefix in (
        (blocked_store, blocked_owner, "blocked"),
        (recover_store, recover_owner, "recover"),
    ):
        store.set_job(
            f"{prefix}-stalled", "RUNNING", kind="model", depends_on=[],
            lease_owner=owner, lease_until=isolation_now - 1.0,
            heartbeat=isolation_now - 301.0,
        )
        with sqlite3.connect(store.db_path) as db:
            db.execute(
                "UPDATE jobs SET lease_until=?, heartbeat=? WHERE job_id=?",
                (isolation_now - 1.0, isolation_now - 301.0,
                 f"{prefix}-stalled"),
            )
    isolation_watchdog = _SeedBankWatchdog(
        stores={
            "BLOCKED": _BlockingHeartbeatStore(
                blocked_store, blocked_started),
            "RECOVER": recover_store,
        },
        owners={"BLOCKED": blocked_owner, "RECOVER": recover_owner},
        telemetry_path=isolation_root / "watchdog-events.jsonl",
        interval_seconds=0.1,
    )
    isolation_watchdog.start()
    deadline = time.time() + 2.0
    while time.time() < deadline:
        if recover_store.job("recover-stalled")["state"] == "PENDING":
            break
        time.sleep(0.02)
    isolation_watchdog.close()
    assert blocked_started.is_set()
    assert recover_store.job("recover-stalled")["state"] == "PENDING"

    # Fault injection 6: an alive/stuck CPU lane must be recycled when the
    # dispatcher activity heartbeat is stale. Requeueing SQL rows alone is
    # insufficient because the old Future would otherwise keep the bank
    # permanently occupied. A healthy dispatcher still renews its lease.
    lane_root = root / "lane-watchdog"
    lane_store = _LaneWatchdogStore(ready_jobs=1)
    lane_pool = _LaneWatchdogPool()
    lane_activity = lane_root / "activity.json"
    lane_activity.parent.mkdir(parents=True, exist_ok=True)
    lane_activity.write_text(json.dumps({
        "schema_version": "DQBD_SEED_BANK_ACTIVITY_V1",
        "state": "ACTIVE", "owner": "lane-owner",
        "updated_at_epoch": time.time() - 301.0,
    }), encoding="utf-8")
    lane_watchdog = _LaneHealthWatchdog(
        store=lane_store, owner="lane-owner", pools=[lane_pool],
        telemetry_root=lane_root / "telemetry",
        activity_heartbeat_path=lane_activity,
    )
    lane_watchdog._tick()
    assert len(lane_pool.reclaimed) == 1
    assert lane_pool.reclaimed[0][0] == "stale-job"
    assert lane_pool.reclaimed[0][1]["reason"] == (
        "DQBD_STALE_DISPATCHER_ACTIVITY")
    assert lane_store.heartbeats == []
    lane_events = [
        json.loads(line) for line in
        (lane_root / "telemetry" / "lane-watchdog-events.jsonl").read_text(
            encoding="utf-8").splitlines() if line.strip()
    ]
    assert any(
        event.get("event") == "STALE_DISPATCHER_LANE_RECLAIMED"
        for event in lane_events
    )
    lane_activity.write_text(json.dumps({
        "schema_version": "DQBD_SEED_BANK_ACTIVITY_V1",
        "state": "ACTIVE", "owner": "lane-owner",
        "updated_at_epoch": time.time(),
    }), encoding="utf-8")
    lane_pool.reclaimed.clear()
    lane_watchdog._tick()
    assert lane_pool.reclaimed == []
    assert len(lane_store.heartbeats) == 1

    # Dispatcher recovery must not consume the finite worker-crash budget.
    # This is what keeps repeated stale-bank healing resumable.
    recovery_store = ManifestedJobStore(lane_root / "recovery-store")
    recovery_store.set_job(
        "dispatcher-recovered", "RUNNING", kind="model", depends_on=[],
        lease_owner="lane-owner", lease_until=time.time() + 60.0,
        worker_failure_count=3,
    )
    assert recovery_store.requeue_execution_failure(
        "dispatcher-recovered", "lane-owner",
        reason="BrokenProcessPool:DQBD_STALE_DISPATCHER_ACTIVITY:age=301",
        count_worker_failure=False,
    )
    recovered_job = recovery_store.job("dispatcher-recovered")
    assert recovered_job["state"] == "PENDING"
    assert recovered_job["worker_failure_count"] == 3
    assert recovered_job["dispatcher_recovery_count"] == 1

    # Scheduling E2E guard: a large ready coverage backlog must not hide the
    # recipe-selection frontier that unlocks GPU-capable model jobs.
    ready_frontier = [
        {"job_id": "candidate_evidence_coverage:2021-09-30:H01" ,
         "kind": "candidate_evidence_coverage"},
        {"job_id": "recipe_selection:2021-09-30:H01",
         "kind": "recipe_selection"},
    ]
    ranked = sorted(
        ready_frontier,
        key=lambda job: (
            _causal_ready_kind_priority(job["kind"]), job["job_id"]),
    )
    assert ranked[0]["kind"] == "recipe_selection"
    assert (
        _causal_ready_kind_priority("recipe_selection")
        < _causal_ready_kind_priority("candidate_evidence_coverage")
    )

    counts, allocations = _allocate_seed_lanes(
        worker_map=[{"logical_processor": i} for i in range(32)],
        worker_count=32,
        ready_by_seed={"SHORT": 1, "PRIMARY": 1, "LONG": 0},
    )
    assert counts == {"SHORT": 16, "PRIMARY": 16, "LONG": 0}
    assert allocations["LONG"] == (0, [])
    assert sum(counts.values()) == 32

    # Fair-wave guard: the full 32+2 pool may belong to one bank per wave,
    # but READY siblings must get their turn before the owner can continue.
    remaining = {"SHORT", "PRIMARY", "LONG"}
    ready = {"SHORT": 1, "PRIMARY": 1, "LONG": 1}
    selected = []
    cursor = 0
    for _ in range(3):
        name, cursor = _next_ready_seed_wave(
            seed_names=("SHORT", "PRIMARY", "LONG"),
            remaining=remaining, ready_by_seed=ready, next_index=cursor)
        selected.append(name)
    assert selected == ["SHORT", "PRIMARY", "LONG"]
    print("DQBD_CAUSAL_MODEL_STORE_RUN_SELF_TEST_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
