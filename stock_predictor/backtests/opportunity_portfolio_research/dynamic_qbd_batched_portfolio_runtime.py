"""Batch the logical Replay/Evidence surface without changing its science.

The legacy manifested DAG deliberately keeps one immutable Replay and Evidence
row per H×D×N×cutoff.  Those rows are useful checkpoint identities, but using
all of them as physical scheduler tasks caused the Step-9 run to contain more
than 600k jobs and repeatedly reload the same H×cutoff segment in different
worker processes.

This runtime keeps the logical rows and their immutable artifacts intact while
changing the physical execution unit to one horizon×cutoff batch:

* the normal mixed causal scheduler still owns Coverage, Selection, Model,
  Calibration, Prediction and Generation-Ready;
* Replay/Evidence rows are excluded from that per-row scheduler frontier;
* ready Replay rows at the earliest causal cutoff are grouped by horizon;
* one reclaimable CPU lane processes all remaining families in that group,
  reusing process-local signal/price/distribution caches;
* each family Replay is committed immediately and its Evidence leaf follows in
  the same worker context, so a reclaimed batch loses only unfinished logical
  rows;
* a batch is a normal v40.0.4 reclaimable lane with the existing RAM admission,
  10-ms live sampling, single-lane reclaim and learned Replay memory profile;
* existing COMPLETE Replay/Evidence rows are never re-executed.

The prospective holdout, seed-local causal visibility, immutable artifacts and
all v40.0.4.3 minimum resource contracts remain unchanged.
"""
from __future__ import annotations

from collections import deque
from concurrent.futures import FIRST_COMPLETED, wait
from concurrent.futures.process import BrokenProcessPool
from datetime import date
import json
import os
from pathlib import Path
import threading
import time
from typing import Any, Iterable, Mapping

from .dynamic_qbd_manifested_job_coordinator import (
    JOB_LEASE_SECONDS,
    WORKER_FAILURE_RETRY_LIMIT,
    ManifestedJobInputs,
    ManifestedJobStore,
    RamAdmissionScheduler,
    RamJobReclaimed,
    _JobMemorySampler,
    _configured_lane_recycle_limit,
    _managed_pool_with_ram_pause,
    _manifested_pool,
    _HotReloadRequested,
    HotReloadController,
    build_manifested_job_handlers,
    execute_candidate_oos_jobs,
    execute_ready_jobs,
)


PORTFOLIO_BATCH_SCHEMA = "DQBD_CAUSAL_PORTFOLIO_BATCH_V40_1"
PORTFOLIO_BATCH_POLICY = "HORIZON_CUTOFF_REPLAY_EVIDENCE_BATCH_V1"
_BATCH_JOB_CLASS = "causal:replay:CPU"
_MANIFEST_LEASE_RACE_ERRORS = {
    "MANIFESTED_JOB_JOB_OWNERSHIP_OR_LEASE_INVALID",
    "MANIFESTED_JOB_JOB_FINISH_RACE",
}


def _is_manifest_lease_race(exc: BaseException) -> bool:
    """Return True for stale-attempt completion races, not scientific errors."""
    return isinstance(exc, RuntimeError) and str(exc) in _MANIFEST_LEASE_RACE_ERRORS


def _write_seed_activity_heartbeat(
    path: str | Path | None, *, owner: str, state: str = "ACTIVE",
) -> None:
    """Publish advisory seed activity without becoming execution authority.

    Portfolio batches can spend materially longer inside one worker than the
    mixed causal dispatch loop.  Keep the same seed-bank heartbeat fresh while
    such a batch is alive so Hotstart can distinguish legitimate long work from
    a PID-alive logical stall.  SQLite remains authoritative; a sharing/I/O
    failure here must never fail the batch.
    """
    if path is None:
        return
    target = Path(path)
    payload = {
        "schema_version": "DQBD_SEED_BANK_ACTIVITY_V1",
        "state": str(state),
        "owner": str(owner),
        "pid": os.getpid(),
        "updated_at_epoch": time.time(),
        "activity_source": "PORTFOLIO_BATCH_PARENT",
    }
    temporary = target.with_name(
        target.name
        + f".{os.getpid()}.{threading.get_ident()}.{time.time_ns()}.tmp")
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, target)
    except OSError:
        try:
            temporary.unlink()
        except OSError:
            pass


def _recover_broken_process_pool_state(
    store: ManifestedJobStore, *, owner: str, exc: BrokenProcessPool,
) -> dict[str, Any]:
    """Recover one pool failure without turning execution health into science.

    A seed-level watchdog may return RUNNING rows to PENDING before the old
    Future delivers its BrokenProcessPool exception.  That external requeue is
    already a successful recovery and must not be converted into a terminal
    FAILED node.  Conversely, a real repeated child crash keeps the historical
    bounded worker-failure budget.

    The pool that raised has already unwound before this helper is called, so
    every still-RUNNING row owned by this bank can safely return to PENDING.
    FAILED rows are revived only when their failure is explicitly a
    BrokenProcessPool execution incident from this same seed-bank owner.
    """
    now = time.time()
    reason = f"{type(exc).__name__}:{exc}"
    dispatcher_recovery = "DQBD_STALE_DISPATCHER_ACTIVITY" in str(exc)
    running_requeued = 0
    failed_requeued = 0
    terminal_failed = 0
    dispatcher_rows = 0
    with store._connect() as db:
        db.execute("BEGIN IMMEDIATE")
        running = db.execute(
            "SELECT job_id,payload FROM jobs "
            "WHERE state='RUNNING' AND lease_owner=?",
            (str(owner),),
        ).fetchall()
        for job_id, raw_payload in running:
            payload = json.loads(raw_payload)
            payload["state"] = "PENDING"
            payload["pool_recovery_count"] = (
                int(payload.get("pool_recovery_count", 0) or 0) + 1)
            payload["last_pool_recovery_at"] = now
            payload["last_pool_recovery"] = reason
            payload.pop("lease_owner", None)
            payload.pop("lease_until", None)
            payload.pop("heartbeat", None)
            payload.pop("started_at", None)
            db.execute(
                "UPDATE jobs SET state='PENDING',payload=?,"
                "lease_owner=NULL,lease_until=NULL,heartbeat=NULL,"
                "last_error=NULL,started_at=NULL,finished_at=NULL "
                "WHERE job_id=? AND state='RUNNING' AND lease_owner=?",
                (json.dumps(payload, default=str), str(job_id), str(owner)),
            )
            running_requeued += int(db.execute(
                "SELECT changes()"
            ).fetchone()[0])

        failed = db.execute(
            "SELECT job_id,payload,last_error FROM jobs "
            "WHERE state='FAILED' AND last_error LIKE 'BrokenProcessPool:%' "
            "AND json_extract(payload,'$.lease_owner')=?",
            (str(owner),),
        ).fetchall()
        for job_id, raw_payload, last_error in failed:
            payload = json.loads(raw_payload)
            failure_reason = str(last_error or reason)
            is_dispatcher = (
                "DQBD_STALE_DISPATCHER_ACTIVITY" in failure_reason)
            if is_dispatcher:
                payload["dispatcher_recovery_count"] = (
                    int(payload.get("dispatcher_recovery_count", 0) or 0) + 1)
                payload["last_dispatcher_recovery_at"] = now
                payload["last_dispatcher_recovery"] = failure_reason
                dispatcher_rows += 1
            else:
                failure_count = (
                    int(payload.get("worker_failure_count", 0) or 0) + 1)
                payload["worker_failure_count"] = failure_count
                payload["last_worker_failure_at"] = now
                payload["last_worker_failure"] = failure_reason
                if failure_count > WORKER_FAILURE_RETRY_LIMIT:
                    db.execute(
                        "UPDATE jobs SET payload=? WHERE job_id=? "
                        "AND state='FAILED' "
                        "AND json_extract(payload,'$.lease_owner')=?",
                        (
                            json.dumps(payload, default=str),
                            str(job_id), str(owner),
                        ),
                    )
                    terminal_failed += 1
                    continue
            payload["state"] = "PENDING"
            payload.pop("lease_owner", None)
            payload.pop("lease_until", None)
            payload.pop("heartbeat", None)
            payload.pop("started_at", None)
            payload.pop("finished_at", None)
            db.execute(
                "UPDATE jobs SET state='PENDING',payload=?,"
                "lease_owner=NULL,lease_until=NULL,heartbeat=NULL,"
                "last_error=NULL,started_at=NULL,finished_at=NULL "
                "WHERE job_id=? AND state='FAILED' "
                "AND last_error LIKE 'BrokenProcessPool:%' "
                "AND json_extract(payload,'$.lease_owner')=?",
                (
                    json.dumps(payload, default=str),
                    str(job_id), str(owner),
                ),
            )
            failed_requeued += int(db.execute(
                "SELECT changes()"
            ).fetchone()[0])
        db.execute("COMMIT")
    try:
        store.materialize_progress()
    except OSError:
        # SQLite remains authoritative. A transient status-file sharing
        # violation must not undo an otherwise complete recovery transaction.
        pass
    return {
        "dispatcher_recovery": bool(dispatcher_recovery),
        "running_requeued": int(running_requeued),
        "failed_requeued": int(failed_requeued),
        "dispatcher_failed_rows_requeued": int(dispatcher_rows),
        "terminal_failed": int(terminal_failed),
    }


def _ready_logical_rows(
    store: ManifestedJobStore, *, kind: str, limit: int = 8192,
) -> list[dict[str, Any]]:
    """Return dependency-ready rows without materializing the full DAG."""
    # Use the store-owned connection context: sqlite3.Connection.__exit__
    # commits/rolls back but does not close the connection.  On Windows that
    # leaves jobs.sqlite3 locked until GC, which can make an otherwise passing
    # batch self-test fail during TemporaryDirectory cleanup.
    with store._connect() as db:
        rows = db.execute(
            "SELECT j.payload FROM jobs j "
            "WHERE j.state='PENDING' AND j.kind=? "
            "AND NOT EXISTS ("
            " SELECT 1 FROM job_dependencies d "
            " LEFT JOIN jobs p ON p.job_id=d.depends_on "
            " WHERE d.job_id=j.job_id "
            " AND (p.job_id IS NULL OR p.state!='COMPLETE')) "
            "ORDER BY json_extract(j.payload,'$.cutoff'),j.job_id LIMIT ?",
            (str(kind), max(1, int(limit))),
        ).fetchall()
    result = []
    for (payload,) in rows:
        job = json.loads(payload)
        job["state"] = "PENDING"
        result.append(job)
    return result


def _earliest_cutoff_groups(
    store: ManifestedJobStore, *, kind: str,
) -> list[dict[str, Any]]:
    rows = _ready_logical_rows(store, kind=kind)
    if not rows:
        return []
    cutoff = min(str(row.get("cutoff", "")) for row in rows)
    rows = [row for row in rows if str(row.get("cutoff", "")) == cutoff]
    groups: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        family_key = str(row.get("portfolio_family_key", ""))
        if not family_key.startswith("H") or len(family_key) < 3:
            raise RuntimeError(
                f"DQBD_PORTFOLIO_BATCH_FAMILY_KEY_INVALID:{family_key}")
        try:
            horizon = int(family_key[1:3])
        except ValueError as exc:
            raise RuntimeError(
                f"DQBD_PORTFOLIO_BATCH_HORIZON_INVALID:{family_key}") from exc
        groups.setdefault(horizon, []).append(row)
    return [
        {
            "kind": str(kind),
            "cutoff": cutoff,
            "horizon": horizon,
            "jobs": sorted(
                values,
                key=lambda row: str(row.get("portfolio_family_key", ""))),
        }
        for horizon, values in sorted(groups.items())
    ]


def _claim_exact_batch(
    store: ManifestedJobStore, *, job_ids: Iterable[str], kind: str,
    owner: str, lease_seconds: int = JOB_LEASE_SECONDS,
) -> list[dict[str, Any]]:
    """Claim many already-ready logical rows in one SQLite transaction."""
    ids = tuple(sorted({str(value) for value in job_ids}))
    if not ids:
        return []
    placeholders = ",".join("?" for _ in ids)
    now = time.time()
    with store._connect() as db:
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("PRAGMA synchronous=NORMAL")
        db.execute("BEGIN IMMEDIATE")
        rows = db.execute(
            "SELECT j.job_id,j.payload,j.attempt FROM jobs j "
            f"WHERE j.job_id IN ({placeholders}) "
            "AND j.kind=? AND j.state='PENDING' "
            "AND NOT EXISTS ("
            " SELECT 1 FROM job_dependencies d "
            " LEFT JOIN jobs p ON p.job_id=d.depends_on "
            " WHERE d.job_id=j.job_id "
            " AND (p.job_id IS NULL OR p.state!='COMPLETE')) "
            "ORDER BY j.job_id",
            (*ids, str(kind)),
        ).fetchall()
        claimed = []
        for job_id, raw_payload, attempt in rows:
            payload = json.loads(raw_payload)
            started = float(payload.get("started_at") or now)
            payload.update({
                "state": "RUNNING",
                "lease_owner": str(owner),
                "lease_until": now + int(lease_seconds),
                "heartbeat": now,
                "started_at": started,
                "attempt": int(attempt) + 1,
            })
            db.execute(
                "UPDATE jobs SET state='RUNNING',payload=?,attempt=?,"
                "lease_owner=?,lease_until=?,heartbeat=?,"
                "started_at=COALESCE(started_at,?) "
                "WHERE job_id=? AND state='PENDING'",
                (
                    json.dumps(payload, default=str),
                    int(attempt) + 1,
                    str(owner),
                    now + int(lease_seconds),
                    now,
                    started,
                    str(job_id),
                ),
            )
            if db.execute("SELECT changes()").fetchone()[0] != 1:
                continue
            claimed.append(payload)
        db.execute("COMMIT")
    return claimed


def _requeue_batch_owner(store: ManifestedJobStore, owner: str) -> int:
    """Recover unfinished logical rows after one physical batch lane dies."""
    with store._connect() as db:
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("BEGIN IMMEDIATE")
        rows = db.execute(
            "SELECT job_id,payload FROM jobs "
            "WHERE state='RUNNING' AND lease_owner=?",
            (str(owner),),
        ).fetchall()
        for job_id, raw_payload in rows:
            payload = json.loads(raw_payload)
            payload["state"] = "PENDING"
            payload.pop("result", None)
            payload.pop("lease_owner", None)
            payload.pop("lease_until", None)
            payload.pop("heartbeat", None)
            payload.pop("started_at", None)
            db.execute(
                "UPDATE jobs SET state='PENDING',payload=?,"
                "lease_owner=NULL,lease_until=NULL,heartbeat=NULL,"
                "last_error=NULL,started_at=NULL,finished_at=NULL "
                "WHERE job_id=? AND state='RUNNING' AND lease_owner=?",
                (json.dumps(payload, default=str), str(job_id), str(owner)),
            )
        db.execute("COMMIT")
    return len(rows)


def _cleanup_family_temps(root: Path, family_key: str, cutoff: str) -> None:
    """Remove only orphan temp dirs for the exact family/cutoff being retried."""
    import shutil

    for base in ("portfolio-replays", "portfolio-evidence"):
        parent = root / base / family_key
        if not parent.is_dir():
            continue
        for candidate in parent.glob(f".{cutoff}.tmp.*"):
            if candidate.is_dir():
                shutil.rmtree(candidate, ignore_errors=True)


def _portfolio_batch_process(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Execute one H×cutoff physical batch inside one reclaimable CPU lane."""
    output_root = Path(str(payload["output_root"]))
    inputs: ManifestedJobInputs = payload["inputs"]
    owner = str(payload["owner"])
    logical_kind = str(payload["logical_kind"])
    job_ids = tuple(str(value) for value in payload.get("job_ids", ()))
    handlers = build_manifested_job_handlers(
        store=ManifestedJobStore(output_root),
        inputs=inputs,
        output_root=output_root,
        evidence_snapshot_root=payload.get("evidence_snapshot_root"),
        recipe_selection_root=payload.get("recipe_selection_root"),
    )
    store = ManifestedJobStore(output_root)
    processed_replay = 0
    processed_evidence = 0
    execution_started = time.monotonic()
    with _JobMemorySampler(
        interval_seconds=.01,
        live_config=payload.get("_live_ram"),
    ) as sampler:
        claimed = _claim_exact_batch(
            store,
            job_ids=job_ids,
            kind=logical_kind,
            owner=owner,
            lease_seconds=24 * 60 * 60,
        )
        for job in claimed:
            job_id = str(job["job_id"])
            family_key = str(job.get("portfolio_family_key", ""))
            cutoff = str(job.get("cutoff", ""))
            _cleanup_family_temps(output_root, family_key, cutoff)
            try:
                if logical_kind == "replay":
                    replay_result = handlers["replay"](job)
                    store.finish_job(
                        job_id, owner, "COMPLETE", result=replay_result)
                    processed_replay += 1

                    evidence_id = f"evidence:{cutoff}:{family_key}"
                    evidence_job = store.claim_ready(
                        owner,
                        lease_seconds=24 * 60 * 60,
                        kinds=("evidence",),
                        job_ids=(evidence_id,),
                    )
                    if evidence_job is not None:
                        evidence_result = handlers["evidence"](evidence_job)
                        store.finish_job(
                            evidence_id,
                            owner,
                            "COMPLETE",
                            result=evidence_result,
                        )
                        processed_evidence += 1
                elif logical_kind == "evidence":
                    evidence_result = handlers["evidence"](job)
                    store.finish_job(
                        job_id, owner, "COMPLETE", result=evidence_result)
                    processed_evidence += 1
                else:
                    raise RuntimeError(
                        f"DQBD_PORTFOLIO_BATCH_KIND_INVALID:{logical_kind}")
            except Exception as exc:
                try:
                    current = store.job(job_id)
                    if (
                        current is not None
                        and current.get("state") == "RUNNING"
                        and current.get("lease_owner") == owner
                    ):
                        store.finish_job(
                            job_id,
                            owner,
                            "FAILED",
                            last_error=f"{type(exc).__name__}:{exc}",
                        )
                except RuntimeError:
                    pass
                raise
    execution_seconds = max(.001, time.monotonic() - execution_started)
    return {
        "schema_version": PORTFOLIO_BATCH_SCHEMA,
        "logical_kind": logical_kind,
        "cutoff": str(payload["cutoff"]),
        "horizon": int(payload["horizon"]),
        "requested_logical_jobs": len(job_ids),
        "replay_jobs_completed": processed_replay,
        "evidence_jobs_completed": processed_evidence,
        "logical_jobs_completed": processed_replay + processed_evidence,
        "_runtime_execution_seconds": execution_seconds,
        "_runtime_memory": sampler.telemetry(_BATCH_JOB_CLASS),
    }


def _run_physical_groups(
    *, store: ManifestedJobStore,
    inputs: ManifestedJobInputs,
    groups: list[dict[str, Any]],
    workers: int,
    worker_map_override: list[dict] | None,
    ram_scheduler: RamAdmissionScheduler | None,
    evidence_snapshot_root: str | Path | None,
    recipe_selection_root: str | Path | None,
    activity_heartbeat_path: str | Path | None = None,
    activity_owner: str | None = None,
    hot_reload_controller: HotReloadController | None = None,
    max_batches: int | None = None,
) -> dict[str, Any]:
    if max_batches is not None:
        groups = list(groups[:max(0, int(max_batches))])
    if not groups:
        return {
            "physical_batches": 0,
            "logical_jobs_completed": 0,
            "reclaimed_batches": 0,
            "requeued_logical_jobs": 0,
        }
    selected_workers = max(1, min(int(workers), len(groups)))
    pool, actual_workers = _manifested_pool(
        workers=selected_workers,
        role="portfolio-batch",
        telemetry_root=store.root / "telemetry",
        worker_map_override=worker_map_override,
        max_pending=selected_workers,
        max_tasks_per_child=_configured_lane_recycle_limit(),
    )
    queue = deque(groups)
    pending: dict[Any, dict[str, Any]] = {}
    completed_batches = 0
    logical_completed = 0
    reclaimed_batches = 0
    requeued_logical_jobs = 0
    owner_prefix = (
        f"portfolio-batch-{os.getpid()}-{threading.get_ident()}-")
    heartbeat_owner = str(activity_owner or owner_prefix.rstrip("-"))
    last_activity_touch = 0.0

    def touch_activity(*, force: bool = False) -> None:
        nonlocal last_activity_touch
        now = time.time()
        if not force and now - last_activity_touch < 5.0:
            return
        _write_seed_activity_heartbeat(
            activity_heartbeat_path,
            owner=heartbeat_owner,
            state="ACTIVE",
        )
        last_activity_touch = now

    touch_activity(force=True)
    try:
        with _managed_pool_with_ram_pause(
            pool, ram_scheduler, store.root / "telemetry",
        ) as pause:
            while queue or pending:
                touch_activity()
                if hot_reload_controller is not None and hot_reload_controller.changed():
                    raise _HotReloadRequested()
                if pause is not None:
                    pause.reconcile()
                admission_budget = (
                    ram_scheduler.admission_budget()
                    if ram_scheduler is not None
                    else actual_workers)
                admitted = 0
                while (
                    queue
                    and len(pending) < actual_workers
                    and admitted < admission_budget
                ):
                    group = queue[0]
                    horizon = int(group["horizon"])
                    cutoff = str(group["cutoff"])
                    jobs = list(group["jobs"])
                    first = jobs[0]
                    segment_sessions = 0
                    if first.get("segment_start") and first.get("segment_end"):
                        try:
                            segment_sessions = max(
                                1,
                                (
                                    date.fromisoformat(str(first["segment_end"]))
                                    - date.fromisoformat(str(first["segment_start"]))
                                ).days,
                            )
                        except (TypeError, ValueError):
                            segment_sessions = 0
                    batch_id = f"portfolio_batch:{cutoff}:H{horizon:02d}:{group['kind']}"
                    descriptor = {
                        "job_id": batch_id,
                        "kind": "portfolio_batch",
                        "horizon": horizon,
                        "workload": "CPU",
                        "prepared": True,
                        "device_index": None,
                        "segment_sessions": segment_sessions,
                        "logical_job_count": len(jobs),
                        "ram_reclaim_count": int(group.get("reclaim_count", 0)),
                    }
                    reservation = (
                        ram_scheduler.try_acquire(
                            1.0,
                            wait_callback=(pause.reconcile if pause is not None else None),
                            job_class=_BATCH_JOB_CLASS,
                            descriptor=descriptor,
                        )
                        if ram_scheduler is not None
                        else 0.0
                    )
                    if reservation is None:
                        break
                    queue.popleft()
                    owner = owner_prefix + batch_id.replace(":", "-")
                    live_config = (
                        ram_scheduler.open_live_job(
                            _BATCH_JOB_CLASS, descriptor)
                        if ram_scheduler is not None
                        else None
                    )
                    task_payload = {
                        "schema_version": PORTFOLIO_BATCH_SCHEMA,
                        "output_root": str(store.root),
                        "inputs": inputs,
                        "owner": owner,
                        "logical_kind": str(group["kind"]),
                        "job_ids": [str(job["job_id"]) for job in jobs],
                        "cutoff": cutoff,
                        "horizon": horizon,
                        "job": {
                            "job_id": batch_id,
                            "kind": "portfolio_batch",
                            "horizon": horizon,
                            "ram_reclaim_count": int(group.get("reclaim_count", 0)),
                        },
                        "_ram_job_class": _BATCH_JOB_CLASS,
                        "_live_ram": live_config,
                    }
                    if evidence_snapshot_root is not None:
                        task_payload["evidence_snapshot_root"] = str(
                            evidence_snapshot_root)
                    if recipe_selection_root is not None:
                        task_payload["recipe_selection_root"] = str(
                            recipe_selection_root)
                    try:
                        future = pool.submit(
                            _portfolio_batch_process, task_payload)
                    except Exception:
                        if ram_scheduler is not None:
                            ram_scheduler.close_live_job(live_config)
                            ram_scheduler.release(
                                reservation, job_class=_BATCH_JOB_CLASS)
                        queue.appendleft(group)
                        raise
                    if ram_scheduler is not None:
                        def release_runtime(
                            _future,
                            amount=reservation,
                            live=live_config,
                        ) -> None:
                            ram_scheduler.close_live_job(live)
                            ram_scheduler.release(
                                amount, job_class=_BATCH_JOB_CLASS)
                        future.add_done_callback(release_runtime)
                    pending[future] = {
                        "group": group,
                        "owner": owner,
                        "descriptor": descriptor,
                    }
                    admitted += 1

                if not pending:
                    if queue:
                        time.sleep(.01)
                        continue
                    break
                done, _ = wait(
                    tuple(pending), timeout=.01,
                    return_when=FIRST_COMPLETED)
                touch_activity()
                if hot_reload_controller is not None and hot_reload_controller.changed():
                    raise _HotReloadRequested()
                if pause is not None:
                    pause.reconcile()
                for future in done:
                    meta = pending.pop(future)
                    group = meta["group"]
                    owner = str(meta["owner"])
                    try:
                        result = future.result()
                        completed_batches += 1
                        logical_completed += int(
                            result.get("logical_jobs_completed", 0))
                        if ram_scheduler is not None:
                            memory = result.get("_runtime_memory") or {}
                            try:
                                ram_scheduler.observe(
                                    _BATCH_JOB_CLASS,
                                    memory.get("rss_peak_gib"),
                                    memory.get("incremental_peak_gib"),
                                    descriptor=meta["descriptor"],
                                )
                            except OSError:
                                # The batch already committed its logical rows.
                                # Profile persistence is execution telemetry and
                                # cannot retroactively fail scientific work.
                                pass
                    except RamJobReclaimed:
                        reclaimed_batches += 1
                        requeued_logical_jobs += _requeue_batch_owner(
                            store, owner)
                        retry = dict(group)
                        retry["reclaim_count"] = int(
                            group.get("reclaim_count", 0)) + 1
                        queue.appendleft(retry)
                    except Exception:
                        requeued_logical_jobs += _requeue_batch_owner(
                            store, owner)
                        raise

    except _HotReloadRequested:
        # Pool cleanup runs before this handler, so no batch worker can
        # commit after its logical rows are returned to PENDING.
        for meta in pending.values():
            requeued_logical_jobs += _requeue_batch_owner(
                store, str(meta["owner"]))
        raise
    finally:
        touch_activity(force=True)
    try:
        store.materialize_progress()
    except OSError:
        # SQLite is authoritative; transient operator-telemetry write failure
        # must not stop a batch that has already committed its logical rows.
        pass
    return {
        "physical_batches": completed_batches,
        "logical_jobs_completed": logical_completed,
        "reclaimed_batches": reclaimed_batches,
        "requeued_logical_jobs": requeued_logical_jobs,
        "actual_batch_workers": actual_workers,
    }


def execute_portfolio_batch_frontier(
    *, store: ManifestedJobStore,
    inputs: ManifestedJobInputs,
    workers: int,
    worker_map_override: list[dict] | None = None,
    ram_scheduler: RamAdmissionScheduler | None = None,
    evidence_snapshot_root: str | Path | None = None,
    recipe_selection_root: str | Path | None = None,
    activity_heartbeat_path: str | Path | None = None,
    activity_owner: str | None = None,
    hot_reload_controller: HotReloadController | None = None,
    max_physical_batches: int | None = None,
) -> dict[str, Any]:
    """Process the earliest ready Replay/Evidence frontier in physical batches."""
    replay_groups = _earliest_cutoff_groups(store, kind="replay")
    replay_result = _run_physical_groups(
        store=store,
        inputs=inputs,
        groups=replay_groups,
        workers=workers,
        worker_map_override=worker_map_override,
        ram_scheduler=ram_scheduler,
        evidence_snapshot_root=evidence_snapshot_root,
        recipe_selection_root=recipe_selection_root,
        activity_heartbeat_path=activity_heartbeat_path,
        activity_owner=activity_owner,
        hot_reload_controller=hot_reload_controller,
        max_batches=max_physical_batches,
    )
    # A resumed checkpoint may already have COMPLETE Replay leaves whose
    # Evidence children are still pending. Process that residual frontier only
    # after Replay batches settle so the two paths cannot race for one leaf.
    evidence_groups = _earliest_cutoff_groups(store, kind="evidence")
    evidence_result = _run_physical_groups(
        store=store,
        inputs=inputs,
        groups=evidence_groups,
        workers=workers,
        worker_map_override=worker_map_override,
        ram_scheduler=ram_scheduler,
        evidence_snapshot_root=evidence_snapshot_root,
        recipe_selection_root=recipe_selection_root,
        activity_heartbeat_path=activity_heartbeat_path,
        activity_owner=activity_owner,
        hot_reload_controller=hot_reload_controller,
        max_batches=(
            None if max_physical_batches is None else max(
                0, int(max_physical_batches)
                - int(replay_result["physical_batches"]))),
    )
    return {
        "schema_version": PORTFOLIO_BATCH_SCHEMA,
        "policy": PORTFOLIO_BATCH_POLICY,
        "logical_rows_preserved": True,
        "replay": replay_result,
        "evidence_residual": evidence_result,
        "physical_batches": (
            int(replay_result["physical_batches"])
            + int(evidence_result["physical_batches"])),
        "logical_jobs_completed": (
            int(replay_result["logical_jobs_completed"])
            + int(evidence_result["logical_jobs_completed"])),
    }


def execute_manifested_development_jobs(
    *, store: ManifestedJobStore,
    inputs: ManifestedJobInputs,
    output_root: str | Path,
    owner: str = "local-worker",
    candidate_jobs: int | None = None,
    causal_jobs: int | None = None,
    workers: int = 1,
    queue_ahead: int = 2,
    worker_map_override: list[dict] | None = None,
    ram_scheduler: RamAdmissionScheduler | None = None,
    evidence_snapshot_root: str | Path | None = None,
    recipe_selection_root: str | Path | None = None,
    activity_heartbeat_path: str | Path | None = None,
    hot_reload_controller: HotReloadController | None = None,
    causal_job_budget: int | None = None,
) -> dict[str, Any]:
    """Run Step-9 with logical per-family leaves and batched physical replay."""
    summary_path = Path(output_root) / "summary.json"
    if summary_path.is_file() and not inputs.allow_dirty_development_fixture:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if "DIRTY_WORKTREE" in str(summary.get("status", "")):
            raise RuntimeError(
                "MANIFESTED_JOB_DIRTY_WORKTREE_EXECUTION_BLOCKED")
    handlers = build_manifested_job_handlers(
        store=store,
        inputs=inputs,
        output_root=output_root,
        evidence_snapshot_root=evidence_snapshot_root,
        recipe_selection_root=recipe_selection_root,
    )
    candidate: dict[str, Any] = {}
    causal: dict[str, Any] = {}
    causal_processed = 0
    batch_totals = {
        "physical_batches": 0,
        "logical_jobs_completed": 0,
    }
    broken_pool_recoveries = 0
    unattributed_broken_pool_recoveries = 0
    manifest_lease_race_recoveries = 0
    consecutive_manifest_lease_race_recoveries = 0
    while True:
        pending_coverage = store.jobs_by_kind_state(
            kind="candidate_evidence_coverage", state="PENDING")
        pending_coverage.sort(
            key=lambda row: (
                str(row.get("cutoff", "")),
                str(row.get("job_id", "")),
            ))
        allowed = None
        if pending_coverage:
            frontier_cutoff = str(pending_coverage[0].get("cutoff", ""))
            frontier = [
                job for job in pending_coverage
                if str(job.get("cutoff", "")) == frontier_cutoff
            ]
            allowed_ids: set[str] = set()
            for coverage_job in frontier:
                dependencies = tuple(
                    str(value)
                    for value in coverage_job.get("depends_on", ()))
                candidate_dependencies = [
                    value for value in dependencies
                    if value.startswith("candidate_oos_fold:")]
                if candidate_dependencies:
                    allowed_ids.update(candidate_dependencies)
                    continue
                marker = str(coverage_job.get("job_id", ""))
                horizon_text = next(
                    (
                        value[1:3] for value in marker.split(":")
                        if value.startswith("H") and len(value) >= 3
                    ),
                    None,
                )
                if horizon_text and horizon_text.isdigit():
                    allowed_ids.update(store.job_ids_by_kind_prefix(
                        kind="candidate_oos_fold",
                        prefix=(
                            f"candidate_oos_fold:{int(horizon_text)}:"),
                    ))
            allowed = sorted(allowed_ids)

        before = store.progress().copy()
        try:
            candidate = execute_candidate_oos_jobs(
                store=store,
                inputs=inputs,
                output_root=output_root,
                owner=owner,
                max_jobs=candidate_jobs,
                allowed_job_ids=allowed,
                workers=workers,
                queue_ahead=queue_ahead,
                max_inflight=min(
                    workers,
                    int(os.environ.get(
                        "DQBD_CANDIDATE_MAX_INFLIGHT", "26")),
                ),
                worker_map_override=worker_map_override,
                ram_scheduler=ram_scheduler,
                estimated_task_gib=4.5,
                activity_heartbeat_path=activity_heartbeat_path,
                hot_reload_controller=hot_reload_controller,
            )
            remaining_causal_budget = (
                None if causal_job_budget is None else max(
                    0, int(causal_job_budget) - causal_processed))
            if remaining_causal_budget == 0:
                break
            mixed = execute_ready_jobs(
                store,
                handlers,
                owner=owner,
                max_jobs=(
                    remaining_causal_budget
                    if causal_jobs is None else min(
                        int(causal_jobs), remaining_causal_budget)),
                kinds=(
                    "candidate_evidence_coverage",
                    "recipe_selection",
                    "model",
                    "calibration",
                    "prediction",
                    "generation_ready",
                ),
                workers=workers,
                queue_ahead=queue_ahead,
                inputs=inputs,
                max_inflight=min(
                    workers,
                    int(os.environ.get(
                        "DQBD_CAUSAL_MAX_INFLIGHT", str(workers))),
                ),
                worker_map_override=worker_map_override,
                ram_scheduler=ram_scheduler,
                estimated_task_gib=1.0,
                evidence_snapshot_root=evidence_snapshot_root,
                recipe_selection_root=recipe_selection_root,
                activity_heartbeat_path=activity_heartbeat_path,
                hot_reload_controller=hot_reload_controller,
            )
            causal_processed += int(mixed.get("processed_this_call", 0))
            portfolio_budget = (
                None if causal_job_budget is None else max(
                    0, int(causal_job_budget) - causal_processed))
            portfolio = execute_portfolio_batch_frontier(
                store=store,
                inputs=inputs,
                workers=workers,
                worker_map_override=worker_map_override,
                ram_scheduler=ram_scheduler,
                evidence_snapshot_root=evidence_snapshot_root,
                recipe_selection_root=recipe_selection_root,
                activity_heartbeat_path=activity_heartbeat_path,
                activity_owner=owner,
                hot_reload_controller=hot_reload_controller,
                # A single physical batch keeps portfolio progress alive,
                # while preventing a large Hxcutoff batch from monopolising
                # the fair seed quantum after mixed causal work.
                max_physical_batches=(
                    None if portfolio_budget is None else min(
                        1, portfolio_budget)),
            )
        except _HotReloadRequested:
            # The SQLite checkpoint remains the source of truth. Requeue only
            # this coordinator's claims, load the changed code, and continue
            # from the same frontier without rebuilding the manifested graph.
            store.requeue_interrupted_jobs(owner=owner)
            if hot_reload_controller is None:
                raise
            hot_reload_controller.reload_coordinator()
            handlers = build_manifested_job_handlers(
                store=store,
                inputs=inputs,
                output_root=output_root,
                evidence_snapshot_root=evidence_snapshot_root,
                recipe_selection_root=recipe_selection_root,
            )
            continue
        except BrokenProcessPool as exc:
            recovery = _recover_broken_process_pool_state(
                store, owner=owner, exc=exc)
            broken_pool_recoveries += 1
            if recovery["terminal_failed"]:
                raise
            attributed = (
                int(recovery["running_requeued"])
                + int(recovery["failed_requeued"])
            )
            if recovery["dispatcher_recovery"]:
                # Dispatcher recycle is orchestration repair. It deliberately
                # does not consume the worker-crash budget; the seed watchdog
                # may already have returned the exact job to PENDING.
                unattributed_broken_pool_recoveries = 0
            elif attributed:
                unattributed_broken_pool_recoveries = 0
            else:
                # A synchronous submit can fail before a manifested row is
                # RUNNING. Restart with a fresh pool a few times, but fail
                # closed instead of spinning forever on a deterministically
                # broken local process environment.
                unattributed_broken_pool_recoveries += 1
                if (
                    unattributed_broken_pool_recoveries
                    > WORKER_FAILURE_RETRY_LIMIT
                ):
                    raise
            time.sleep(.05)
            continue
        except RuntimeError as exc:
            if not _is_manifest_lease_race(exc):
                raise
            # The old Future lost its lease because a watchdog or expiry path
            # already recovered the row. The just-unwound process pool can no
            # longer publish anything, so discard all remaining claims from
            # this owner and resume from SQLite rather than terminating the
            # seed bank.
            store.requeue_interrupted_jobs(owner=owner)
            manifest_lease_race_recoveries += 1
            consecutive_manifest_lease_race_recoveries += 1
            if (
                consecutive_manifest_lease_race_recoveries
                > WORKER_FAILURE_RETRY_LIMIT * 4
            ):
                # Escalate only a persistent consecutive ownership loop to
                # the outer Hotstart supervisor. Sparse lease races separated
                # by successful scheduler progress are independent incidents
                # and must not eventually exhaust a lifetime counter.
                raise
            time.sleep(.05)
            continue

        # A completed scheduler quantum proves that the process environment
        # and manifested ownership path recovered. Retry limits are therefore
        # consecutive-incident guards, not lifetime failure counters. The
        # cumulative metrics remain available above for telemetry/audit.
        unattributed_broken_pool_recoveries = 0
        consecutive_manifest_lease_race_recoveries = 0
        batch_totals["physical_batches"] += int(
            portfolio["physical_batches"])
        batch_totals["logical_jobs_completed"] += int(
            portfolio["logical_jobs_completed"])
        causal = {
            "mixed_frontier": mixed,
            "portfolio_batch_frontier": portfolio,
            "heavy_and_light_merged": False,
            "portfolio_replay_batched": True,
            "ram_targeted_reclaims_observed": (
                ram_scheduler.telemetry().get(
                    "ram_scheduler_targeted_reclaim_count", 0)
                if ram_scheduler is not None else 0
            ),
            "broken_process_pool_recoveries": broken_pool_recoveries,
            "manifest_lease_race_recoveries": manifest_lease_race_recoveries,
            "progress": store.progress(),
            "processed_this_call": (
                causal_processed
                + int(portfolio["logical_jobs_completed"])),
            "causal_job_budget": causal_job_budget,
        }
        causal_processed += int(portfolio["logical_jobs_completed"])
        after = store.progress()
        if (causal_job_budget is not None
                and causal_processed >= int(causal_job_budget)):
            break
        if (
            after.get("COMPLETE", 0) == before.get("COMPLETE", 0)
            and after.get("FAILED", 0) == before.get("FAILED", 0)
        ):
            break
        if not pending_coverage and after.get("PENDING", 0) == 0:
            break
    return {
        "candidate_oos": candidate,
        "causal": causal,
        "portfolio_batching": {
            "schema_version": PORTFOLIO_BATCH_SCHEMA,
            "policy": PORTFOLIO_BATCH_POLICY,
            "logical_rows_preserved": True,
            **batch_totals,
        },
        "broken_process_pool_recoveries": broken_pool_recoveries,
        "manifest_lease_race_recoveries": manifest_lease_race_recoveries,
        "ram_targeted_reclaims_observed": (
            ram_scheduler.telemetry().get(
                "ram_scheduler_targeted_reclaim_count", 0)
            if ram_scheduler is not None else 0
        ),
        "progress": store.progress(),
        "portfolio_replay_authority": (
            "BLOCKED_UNTIL_STOCK_INPUTS"
            if inputs.stock_execution_prices is None
            or inputs.stock_distributions is None
            else "PRODUCTIVE_BATCHED_DAG_HANDLER"
        ),
    }
