"""Execution-only attempt fencing for manifested Dynamic-QBD job stores.

The persisted manifested DAG remains owned by ``ManifestedJobStore``.  This
facade adds one runtime invariant for concurrent worker execution: a Future may
only mutate the exact claim attempt that created it.  It also hides locally
in-flight job IDs from subsequent ready/claim scans, preventing an external
watchdog requeue from creating a same-owner ABA claim while the old Future is
still alive.

No scientific payload identity, dependency, artifact or causal boundary is
changed by this layer.
"""
from __future__ import annotations

import json
import threading
import time
from typing import Any, Iterable, Mapping

from .dynamic_qbd_manifested_job_coordinator import ManifestedJobStore


_EXECUTION_RECOVERY_ERRORS = (
    "DQBD_STALE_DISPATCHER_ACTIVITY",
    "DQBD_LANE_REPLACEMENT_FAILED",
)


class AttemptFencedManifestedJobStore:
    """Duck-typed ManifestedJobStore facade with atomic attempt predicates."""

    def __init__(self, store: ManifestedJobStore) -> None:
        self._store = store
        self._claim_lock = threading.RLock()
        self._claims: dict[tuple[str, str], int] = {}

    @classmethod
    def ensure(cls, store):
        return store if isinstance(store, cls) else cls(store)

    def __getattr__(self, name: str):
        return getattr(self._store, name)

    def _key(self, job_id: str, owner: str) -> tuple[str, str]:
        return str(owner), str(job_id)

    def _remember(self, job: Mapping[str, Any], owner: str) -> None:
        key = self._key(str(job["job_id"]), owner)
        with self._claim_lock:
            if key in self._claims:
                raise RuntimeError(
                    "MANIFESTED_JOB_DUPLICATE_LOCAL_INFLIGHT_CLAIM:"
                    f"{job['job_id']}:{owner}")
            self._claims[key] = int(job.get("attempt", 0) or 0)

    def expected_attempt(self, job_id: str, owner: str) -> int | None:
        with self._claim_lock:
            value = self._claims.get(self._key(job_id, owner))
        return int(value) if value is not None else None

    def active_job_ids(self) -> tuple[str, ...]:
        with self._claim_lock:
            return tuple(sorted({job_id for _, job_id in self._claims}))

    def _forget(self, job_id: str, owner: str) -> None:
        with self._claim_lock:
            self._claims.pop(self._key(job_id, owner), None)

    def forget_owner_claims(self, owner: str) -> int:
        owner = str(owner)
        with self._claim_lock:
            keys = [key for key in self._claims if key[0] == owner]
            for key in keys:
                self._claims.pop(key, None)
        return len(keys)

    def clear_claims(self) -> int:
        with self._claim_lock:
            count = len(self._claims)
            self._claims.clear()
        return count

    def ready_jobs(
        self, *, kinds: Iterable[str] | None = None, limit: int = 256,
        exclude_job_ids: Iterable[str] | None = None,
    ) -> list[dict]:
        excluded = set(str(value) for value in (exclude_job_ids or ()))
        excluded.update(self.active_job_ids())
        return self._store.ready_jobs(
            kinds=kinds,
            limit=limit,
            exclude_job_ids=tuple(sorted(excluded)),
        )

    def claim_ready(
        self, owner: str, lease_seconds: int = 900,
        kinds: Iterable[str] | None = None,
        job_ids: Iterable[str] | None = None,
        exclude_job_ids: Iterable[str] | None = None,
    ) -> dict | None:
        excluded = set(str(value) for value in (exclude_job_ids or ()))
        excluded.update(self.active_job_ids())
        job = self._store.claim_ready(
            owner,
            lease_seconds=lease_seconds,
            kinds=kinds,
            job_ids=job_ids,
            exclude_job_ids=tuple(sorted(excluded)),
        )
        if job is None:
            return None
        self._remember(job, owner)
        return job

    def _require_attempt(self, job_id: str, owner: str) -> int:
        attempt = self.expected_attempt(job_id, owner)
        if attempt is None:
            raise RuntimeError(
                "MANIFESTED_JOB_ATTEMPT_FENCE_MISSING:"
                f"{job_id}:{owner}")
        return int(attempt)

    def finish_job(
        self, job_id: str, owner: str, state: str, **fields: Any,
    ) -> None:
        if state not in {"COMPLETE", "FAILED", "BLOCKED"}:
            raise ValueError("MANIFESTED_JOB_INVALID_FINISH_STATE")

        # Dispatcher recycle and lane-repair construction failure are
        # orchestration incidents.  Candidate-OOS historically computed the
        # worker-failure budget before it noticed the dispatcher marker.  At a
        # pre-existing count of three that could therefore fall through to the
        # generic FAILED path even though this incident must not consume the
        # worker-crash budget.  Intercept that terminalization at the fenced
        # store boundary and return the exact attempt to PENDING instead.
        last_error = str(fields.get("last_error", "") or "")
        if state == "FAILED" and any(
            marker in last_error for marker in _EXECUTION_RECOVERY_ERRORS
        ):
            requeued = self.requeue_execution_failure(
                job_id,
                owner,
                reason=last_error,
                count_worker_failure=False,
            )
            if requeued:
                return
            current = self._store.job(job_id)
            if current is not None and current.get("state") == "PENDING":
                self._forget(job_id, owner)
                return
            raise RuntimeError(
                "MANIFESTED_JOB_EXECUTION_RECOVERY_REQUEUE_INVALID")

        expected_attempt = self._require_attempt(job_id, owner)
        now = time.time()
        with self._store._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT payload,state,lease_owner,lease_until,attempt "
                "FROM jobs WHERE job_id=?",
                (str(job_id),),
            ).fetchone()
            if (
                row is None
                or row[1] != "RUNNING"
                or row[2] != str(owner)
                or row[3] is None
                or float(row[3]) < now
                or int(row[4]) != expected_attempt
            ):
                db.execute("ROLLBACK")
                raise RuntimeError(
                    "MANIFESTED_JOB_JOB_OWNERSHIP_OR_LEASE_INVALID")
            payload = json.loads(row[0])
            payload.update(fields)
            payload["state"] = state
            payload["finished_at"] = now
            db.execute(
                "UPDATE jobs SET state=?,payload=?,lease_owner=NULL,"
                "lease_until=NULL,last_error=?,finished_at=? "
                "WHERE job_id=? AND state='RUNNING' AND lease_owner=? "
                "AND attempt=?",
                (
                    state,
                    json.dumps(payload, default=str),
                    payload.get("last_error"),
                    now,
                    str(job_id),
                    str(owner),
                    expected_attempt,
                ),
            )
            if int(db.execute("SELECT changes()").fetchone()[0]) != 1:
                db.execute("ROLLBACK")
                raise RuntimeError("MANIFESTED_JOB_JOB_FINISH_RACE")
            db.execute("COMMIT")
        self._forget(job_id, owner)

    def heartbeat(
        self, job_id: str, owner: str, lease_seconds: int = 900,
    ) -> None:
        expected_attempt = self.expected_attempt(job_id, owner)
        if expected_attempt is None:
            return
        now = time.time()
        with self._store._connect() as db:
            db.execute(
                "UPDATE jobs SET heartbeat=?,lease_until=? "
                "WHERE job_id=? AND lease_owner=? AND state='RUNNING' "
                "AND attempt=?",
                (
                    now,
                    now + int(lease_seconds),
                    str(job_id),
                    str(owner),
                    int(expected_attempt),
                ),
            )

    def release_claim(self, job_id: str, owner: str) -> None:
        expected_attempt = self.expected_attempt(job_id, owner)
        if expected_attempt is None:
            return
        try:
            with self._store._connect() as db:
                db.execute("BEGIN IMMEDIATE")
                row = db.execute(
                    "SELECT payload,state,lease_owner,attempt FROM jobs "
                    "WHERE job_id=?",
                    (str(job_id),),
                ).fetchone()
                if (
                    row is None
                    or row[1] != "RUNNING"
                    or row[2] != str(owner)
                    or int(row[3]) != int(expected_attempt)
                ):
                    db.execute("ROLLBACK")
                    return
                payload = json.loads(row[0])
                payload["state"] = "PENDING"
                payload.pop("lease_owner", None)
                payload.pop("lease_until", None)
                payload.pop("heartbeat", None)
                payload.pop("started_at", None)
                db.execute(
                    "UPDATE jobs SET state='PENDING',payload=?,"
                    "lease_owner=NULL,lease_until=NULL,heartbeat=NULL,"
                    "started_at=NULL WHERE job_id=? AND state='RUNNING' "
                    "AND lease_owner=? AND attempt=?",
                    (
                        json.dumps(payload, default=str),
                        str(job_id),
                        str(owner),
                        int(expected_attempt),
                    ),
                )
                db.execute("COMMIT")
        finally:
            self._forget(job_id, owner)

    def requeue_reclaimed_job(
        self, job_id: str, owner: str, *, reason: str,
        released_gib: float = 0.0,
        preferred_executor: str | None = None,
    ) -> bool:
        expected_attempt = self.expected_attempt(job_id, owner)
        if expected_attempt is None:
            return False
        now = time.time()
        try:
            with self._store._connect() as db:
                db.execute("BEGIN IMMEDIATE")
                row = db.execute(
                    "SELECT payload,state,lease_owner,attempt FROM jobs "
                    "WHERE job_id=?",
                    (str(job_id),),
                ).fetchone()
                if (
                    row is None
                    or row[1] != "RUNNING"
                    or row[2] != str(owner)
                    or int(row[3]) != int(expected_attempt)
                ):
                    db.execute("ROLLBACK")
                    return False
                payload = json.loads(row[0])
                payload["state"] = "PENDING"
                payload["ram_reclaim_count"] = (
                    int(payload.get("ram_reclaim_count", 0) or 0) + 1)
                payload["ram_last_reclaim_at"] = now
                payload["ram_last_reclaim_reason"] = str(reason)
                payload["ram_last_released_gib"] = float(released_gib)
                if preferred_executor is not None:
                    payload["execution_preference"] = str(
                        preferred_executor).upper()
                payload.pop("lease_owner", None)
                payload.pop("lease_until", None)
                payload.pop("started_at", None)
                db.execute(
                    "UPDATE jobs SET state='PENDING',payload=?,"
                    "lease_owner=NULL,lease_until=NULL,heartbeat=NULL,"
                    "started_at=NULL,last_error=NULL,finished_at=NULL "
                    "WHERE job_id=? AND state='RUNNING' AND lease_owner=? "
                    "AND attempt=?",
                    (
                        json.dumps(payload, default=str),
                        str(job_id), str(owner), int(expected_attempt),
                    ),
                )
                updated = int(db.execute(
                    "SELECT changes()"
                ).fetchone()[0]) == 1
                if not updated:
                    db.execute("ROLLBACK")
                    return False
                db.execute("COMMIT")
                return True
        finally:
            # The Future that requested reclaim is already settled. Even when
            # an external watchdog won the requeue race, the old attempt must
            # no longer hide this job from a future scheduler pass.
            self._forget(job_id, owner)

    def requeue_execution_failure(
        self, job_id: str, owner: str, *, reason: str,
        preferred_executor: str | None = None,
        count_worker_failure: bool = True,
    ) -> bool:
        expected_attempt = self.expected_attempt(job_id, owner)
        if expected_attempt is None:
            return False
        now = time.time()
        try:
            with self._store._connect() as db:
                db.execute("BEGIN IMMEDIATE")
                row = db.execute(
                    "SELECT payload,state,lease_owner,attempt FROM jobs "
                    "WHERE job_id=?",
                    (str(job_id),),
                ).fetchone()
                if (
                    row is None
                    or row[1] != "RUNNING"
                    or row[2] != str(owner)
                    or int(row[3]) != int(expected_attempt)
                ):
                    db.execute("ROLLBACK")
                    return False
                payload = json.loads(row[0])
                payload["state"] = "PENDING"
                if count_worker_failure:
                    payload["worker_failure_count"] = (
                        int(payload.get("worker_failure_count", 0) or 0) + 1)
                    payload["last_worker_failure_at"] = now
                    payload["last_worker_failure"] = str(reason)
                else:
                    payload["dispatcher_recovery_count"] = (
                        int(payload.get("dispatcher_recovery_count", 0) or 0)
                        + 1)
                    payload["last_dispatcher_recovery_at"] = now
                    payload["last_dispatcher_recovery"] = str(reason)
                if preferred_executor is not None:
                    payload["execution_preference"] = str(
                        preferred_executor).upper()
                payload.pop("lease_owner", None)
                payload.pop("lease_until", None)
                payload.pop("started_at", None)
                db.execute(
                    "UPDATE jobs SET state='PENDING',payload=?,"
                    "lease_owner=NULL,lease_until=NULL,heartbeat=NULL,"
                    "started_at=NULL,finished_at=NULL,last_error=? "
                    "WHERE job_id=? AND state='RUNNING' AND lease_owner=? "
                    "AND attempt=?",
                    (
                        json.dumps(payload, default=str), str(reason),
                        str(job_id), str(owner), int(expected_attempt),
                    ),
                )
                changed = int(db.execute(
                    "SELECT changes()"
                ).fetchone()[0]) == 1
                if not changed:
                    db.execute("ROLLBACK")
                    return False
                db.execute("COMMIT")
                return True
        finally:
            self._forget(job_id, owner)

    def requeue_interrupted_jobs(
        self, *, retry_failed: bool = False, owner: str | None = None,
    ) -> int:
        count = self._store.requeue_interrupted_jobs(
            retry_failed=retry_failed,
            owner=owner,
        )
        if owner is not None:
            self.forget_owner_claims(owner)
        else:
            self.clear_claims()
        return count
