"""Focused source-level regression checks for v40.1 scheduler efficiency.

This self-test is intentionally light: it creates a tiny manifested SQLite DAG
but never starts ProcessPool workers, never opens the prospective holdout and
never launches Step-9.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

from . import dynamic_qbd_manifested_job_coordinator as coordinator
from . import dynamic_qbd_scheduler_efficiency as efficiency
from . import dynamic_qbd_causal_model_store_run as runner
from . import dynamic_qbd_batched_portfolio_runtime as batched


def _marker_in_wrapped_chain(fn, marker: str) -> bool:
    seen = set()
    current = fn
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if bool(getattr(current, marker, False)):
            return True
        current = getattr(current, "__wrapped__", None)
    return False


def _job(job_id: str, kind: str, **fields):
    return {
        "job_id": job_id,
        "kind": kind,
        "state": "PENDING",
        "depends_on": [],
        **fields,
    }


def main() -> int:
    state = efficiency.scheduler_efficiency_state()
    assert state["cpu_lanes"] == 32
    assert state["gpu_workers_per_device"] == 4
    assert state["gpu_queue_ahead"] == 2
    assert state["gpu_global_capacity"] == 6

    assert coordinator.MAX_RUNTIME_PROCESS_LANES == 32
    assert _marker_in_wrapped_chain(
        coordinator.execute_candidate_oos_jobs, "_dqbd_gpu4_topology"
    )
    assert _marker_in_wrapped_chain(
        coordinator.execute_ready_jobs, "_dqbd_gpu4_topology"
    )
    assert getattr(coordinator.ManifestedJobStore.__init__,
                   "_dqbd_lightweight_open", False)
    assert getattr(coordinator.ManifestedJobStore.ready_jobs,
                   "_dqbd_cached_frontier", False)
    assert hasattr(coordinator.ManifestedJobStore, "ready_job_exists")
    assert hasattr(coordinator.ManifestedJobStore, "ready_job_count_bounded")
    assert hasattr(coordinator.ManifestedJobStore, "pending_cutoff_frontier")
    assert hasattr(coordinator.ManifestedJobStore, "scheduler_ready_capacity")
    assert getattr(coordinator._manifested_pool,
                   "_dqbd_persistent_shared", False)
    assert getattr(coordinator._managed_pool_with_ram_pause,
                   "_dqbd_persistent_shared", False)
    assert getattr(coordinator._LaneHealthWatchdog._tick,
                   "_dqbd_exists_shared_scope", False)

    assert getattr(batched.execute_manifested_development_jobs,
                   "_dqbd_scheduler_efficiency", False)
    assert getattr(batched._portfolio_batch_process,
                   "_dqbd_single_store_open", False)
    assert getattr(runner.run, "_dqbd_scheduler_efficiency", False)
    assert getattr(runner.main, "_dqbd_scheduler_efficiency", False)
    assert getattr(runner._SeedBankWatchdog._tick,
                   "_dqbd_exists_watchdog", False)

    # Cross-seed allocation must consume one shared 32-lane budget rather than
    # constructing 32 lanes per seed. Capacity is bounded by physically ready
    # units, so a narrow seed cannot park a third of the machine.
    worker_map = [
        {"logical_processor": i, "core_index": i, "role": "selftest"}
        for i in range(32)
    ]
    counts, allocations = runner._allocate_seed_lanes(
        worker_map=worker_map,
        worker_count=32,
        ready_by_seed={"SHORT": 2, "PRIMARY": 16, "LONG": 32},
        seed_names=("SHORT", "PRIMARY", "LONG"),
    )
    assert sum(counts.values()) == 32
    assert counts["SHORT"] == 2
    assert counts["PRIMARY"] <= 16
    assert counts["LONG"] <= 32
    assert sum(len(value[1]) for value in allocations.values()) == 32

    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        store = coordinator.ManifestedJobStore(root)
        store.seed_jobs({
            "coverage:2020-01:H03": _job(
                "coverage:2020-01:H03",
                "candidate_evidence_coverage",
                cutoff="2020-01-31",
            ),
            "coverage:2020-02:H03": _job(
                "coverage:2020-02:H03",
                "candidate_evidence_coverage",
                cutoff="2020-02-28",
            ),
            "recipe:2020-01:H03": _job(
                "recipe:2020-01:H03",
                "recipe_selection",
                cutoff="2020-01-31",
            ),
            "replay:2020-01:H03:D01:N01": _job(
                "replay:2020-01:H03:D01:N01",
                "replay",
                cutoff="2020-01-31",
                portfolio_family_key="H03_D01_N01_FIXED",
            ),
            "replay:2020-01:H03:D02:N01": _job(
                "replay:2020-01:H03:D02:N01",
                "replay",
                cutoff="2020-01-31",
                portfolio_family_key="H03_D02_N01_FIXED",
            ),
            "replay:2020-01:H05:D01:N01": _job(
                "replay:2020-01:H05:D01:N01",
                "replay",
                cutoff="2020-01-31",
                portfolio_family_key="H05_D01_N01_FIXED",
            ),
        })

        assert store.ready_job_exists()
        assert 1 <= store.ready_job_count_bounded(limit=2) <= 2

        frontier = store.pending_cutoff_frontier(
            kind="candidate_evidence_coverage", limit=32
        )
        assert frontier
        assert {str(row["cutoff"]) for row in frontier} == {"2020-01-31"}

        # The three Replay logical rows represent only two physical H×cutoff
        # batches (H03 and H05). One recipe node is another physical CPU unit,
        # and both ready coverage rows remain distinct causal CPU units.
        capacity = store.scheduler_ready_capacity(limit=32)
        assert capacity == 5, capacity

        before = efficiency.scheduler_efficiency_state()["ready_frontier"].copy()
        first = store.ready_jobs(kinds=("recipe_selection",), limit=8)
        second = store.ready_jobs(kinds=("recipe_selection",), limit=8)
        after = efficiency.scheduler_efficiency_state()["ready_frontier"].copy()
        assert [row["job_id"] for row in first] == [row["job_id"] for row in second]
        assert after["cache_hits"] >= before["cache_hits"] + 1

        # Worker-side open must attach to an existing SQLite store without
        # triggering schema/bootstrap creation.
        prior = os.environ.get("DQBD_MANIFEST_STORE_WORKER_OPEN")
        os.environ["DQBD_MANIFEST_STORE_WORKER_OPEN"] = "1"
        try:
            worker_store = coordinator.ManifestedJobStore(root)
            assert worker_store.db_path == store.db_path
            assert worker_store.ready_job_exists()
        finally:
            if prior is None:
                os.environ.pop("DQBD_MANIFEST_STORE_WORKER_OPEN", None)
            else:
                os.environ["DQBD_MANIFEST_STORE_WORKER_OPEN"] = prior

    print("DYNAMIC_QBD_SCHEDULER_EFFICIENCY_SELF_TEST:PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
