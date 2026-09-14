"""Focused regression tests for Step-9 BrokenProcessPool recovery.

These tests exercise only checkpoint state transitions. They deliberately do
not run the scientific model factory or open the prospective holdout.
"""
from __future__ import annotations

from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
import tempfile

from .dynamic_qbd_batched_portfolio_runtime import (
    _is_manifest_lease_race,
    _recover_broken_process_pool_state,
)
from .dynamic_qbd_manifested_job_coordinator import (
    WORKER_FAILURE_RETRY_LIMIT,
    ManifestedJobStore,
)


def _pending_recipe(store: ManifestedJobStore, job_id: str) -> None:
    store.set_job(
        job_id,
        "PENDING",
        kind="recipe_selection",
        cutoff="2020-08-31",
        recipe_id="H09_RIDGE_HGB_FROZEN_RULE",
    )


def _claim(store: ManifestedJobStore, owner: str, job_id: str) -> dict:
    job = store.claim_ready(
        owner,
        kinds=("recipe_selection",),
        job_ids=(job_id,),
    )
    if job is None:
        raise AssertionError(f"job did not claim: {job_id}")
    return job


def _fail_broken_pool(
    store: ManifestedJobStore, owner: str, job_id: str,
    message: str = "worker exited unexpectedly",
) -> None:
    _claim(store, owner, job_id)
    store.finish_job(
        job_id,
        owner,
        "FAILED",
        last_error=f"BrokenProcessPool:{message}",
    )


def run_self_test() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        store = ManifestedJobStore(Path(tmp) / "seed")
        owner = "seed-bank-test"

        # Incident regression: the seed watchdog requeues the row before the
        # reclaimed Future reports BrokenProcessPool. The later exception is
        # already recovered and must not recreate a terminal FAILED row.
        stale_id = (
            "recipe_selection:2020-08-31:H09_RIDGE_HGB_FROZEN_RULE")
        _pending_recipe(store, stale_id)
        _claim(store, owner, stale_id)
        requeued = store.requeue_interrupted_jobs(owner=owner)
        assert requeued >= 1
        stale = _recover_broken_process_pool_state(
            store,
            owner=owner,
            exc=BrokenProcessPool(
                "DQBD_STALE_DISPATCHER_ACTIVITY:age=126.102"),
        )
        assert stale["dispatcher_recovery"] is True
        assert stale["terminal_failed"] == 0
        assert store.job(stale_id)["state"] == "PENDING"

        # A real process crash that was already marked FAILED is retryable and
        # consumes the worker-failure budget rather than becoming scientific
        # evidence or killing the whole seed bank immediately.
        crash_id = (
            "recipe_selection:2020-08-31:H10_RIDGE_HGB_FROZEN_RULE")
        _pending_recipe(store, crash_id)
        _fail_broken_pool(store, owner, crash_id)
        crash = _recover_broken_process_pool_state(
            store,
            owner=owner,
            exc=BrokenProcessPool("worker exited unexpectedly"),
        )
        recovered = store.job(crash_id)
        assert crash["failed_requeued"] == 1
        assert crash["terminal_failed"] == 0
        assert recovered["state"] == "PENDING"
        assert int(recovered.get("worker_failure_count", 0)) == 1

        # Recovery is seed-local. A BrokenProcessPool row from another owner
        # must remain FAILED when this bank repairs its own pool.
        foreign_owner = "other-seed-bank"
        foreign_id = (
            "recipe_selection:2020-08-31:H11_RIDGE_HGB_FROZEN_RULE")
        _pending_recipe(store, foreign_id)
        _fail_broken_pool(store, foreign_owner, foreign_id)
        _recover_broken_process_pool_state(
            store,
            owner=owner,
            exc=BrokenProcessPool(
                "DQBD_STALE_DISPATCHER_ACTIVITY:age=130.0"),
        )
        assert store.job(foreign_id)["state"] == "FAILED"

        # Genuine repeated child crashes remain bounded. Dispatcher recycling
        # is separate and does not consume this worker-crash budget.
        for expected_count in range(2, WORKER_FAILURE_RETRY_LIMIT + 1):
            _fail_broken_pool(store, owner, crash_id)
            retry = _recover_broken_process_pool_state(
                store,
                owner=owner,
                exc=BrokenProcessPool("worker exited unexpectedly"),
            )
            assert retry["terminal_failed"] == 0
            current = store.job(crash_id)
            assert current["state"] == "PENDING"
            assert int(current.get("worker_failure_count", 0)) == expected_count
        _fail_broken_pool(store, owner, crash_id)
        terminal = _recover_broken_process_pool_state(
            store,
            owner=owner,
            exc=BrokenProcessPool("worker exited unexpectedly"),
        )
        current = store.job(crash_id)
        assert terminal["terminal_failed"] == 1
        assert current["state"] == "FAILED"
        assert int(current.get("worker_failure_count", 0)) == (
            WORKER_FAILURE_RETRY_LIMIT + 1)

        # Lease loss caused by an external requeue is an execution race, while
        # unrelated RuntimeErrors remain fail-closed.
        assert _is_manifest_lease_race(RuntimeError(
            "MANIFESTED_JOB_JOB_OWNERSHIP_OR_LEASE_INVALID"))
        assert not _is_manifest_lease_race(RuntimeError(
            "MANIFESTED_JOB_EVIDENCE_SNAPSHOT_IMMUTABLE_CONFLICT"))


if __name__ == "__main__":
    run_self_test()
    print("PASS dynamic_qbd_broken_process_pool_recovery_self_test")
