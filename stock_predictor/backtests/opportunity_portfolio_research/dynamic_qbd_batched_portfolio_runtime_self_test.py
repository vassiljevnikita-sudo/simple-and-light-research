"""Focused scheduler-only self-test for the batched portfolio runtime."""
from __future__ import annotations

from pathlib import Path
import tempfile

from .dynamic_qbd_batched_portfolio_runtime import (
    _claim_exact_batch,
    _earliest_cutoff_groups,
    _requeue_batch_owner,
)
from .dynamic_qbd_manifested_job_coordinator import ManifestedJobStore


def _replay(
    *, cutoff: str, family: str, ready: str, previous: str | None = None,
) -> dict:
    job_id = f"replay:{cutoff}:{family}"
    depends_on = [ready]
    if previous is not None:
        depends_on.append(previous)
    return {
        "job_id": job_id,
        "kind": "replay",
        "cutoff": cutoff,
        "portfolio_family_key": family,
        "generation_ready_job_id": ready,
        "segment_start": cutoff,
        "segment_end": "2025-03-31",
        "state": "PENDING",
        "depends_on": depends_on,
    }


def _evidence(*, cutoff: str, family: str) -> dict:
    replay_id = f"replay:{cutoff}:{family}"
    return {
        "job_id": f"evidence:{cutoff}:{family}",
        "kind": "evidence",
        "cutoff": cutoff,
        "portfolio_family_key": family,
        "state": "PENDING",
        "depends_on": [replay_id],
    }


def main() -> int:
    with tempfile.TemporaryDirectory(
        prefix="dqbd-portfolio-batch-selftest-"
    ) as temporary:
        root = Path(temporary)
        store = ManifestedJobStore(root)
        cutoff_a = "2025-01-31"
        cutoff_b = "2025-02-28"
        ready_h1 = f"generation_ready:{cutoff_a}:H01_RIDGE_HGB_FROZEN_RULE"
        ready_h2 = f"generation_ready:{cutoff_a}:H02_RIDGE_HGB_FROZEN_RULE"
        families_h1 = (
            "H01_D01_N01_FIXED",
            "H01_D01_N02_FIXED",
        )
        families_h2 = ("H02_D01_N01_FIXED",)
        jobs: dict[str, dict] = {
            "source": {
                "job_id": "source",
                "kind": "input",
                "state": "COMPLETE",
                "depends_on": [],
            },
            ready_h1: {
                "job_id": ready_h1,
                "kind": "generation_ready",
                "state": "COMPLETE",
                "cutoff": cutoff_a,
                "model_family_key": "H01_RIDGE_HGB_FROZEN_RULE",
                "depends_on": ["source"],
                "result": {"generation_id": "G-H1"},
            },
            ready_h2: {
                "job_id": ready_h2,
                "kind": "generation_ready",
                "state": "COMPLETE",
                "cutoff": cutoff_a,
                "model_family_key": "H02_RIDGE_HGB_FROZEN_RULE",
                "depends_on": ["source"],
                "result": {"generation_id": "G-H2"},
            },
            "independent": {
                "job_id": "independent",
                "kind": "input",
                "state": "COMPLETE",
                "depends_on": [],
            },
        }
        first_replays = []
        for family in (*families_h1, *families_h2):
            ready = ready_h1 if family.startswith("H01") else ready_h2
            replay = _replay(
                cutoff=cutoff_a,
                family=family,
                ready=ready,
            )
            jobs[replay["job_id"]] = replay
            jobs[f"evidence:{cutoff_a}:{family}"] = _evidence(
                cutoff=cutoff_a,
                family=family,
            )
            first_replays.append(replay["job_id"])

        # A later family replay must remain closed behind its own previous
        # Replay row and must never appear in the current physical batch.
        later_family = families_h1[0]
        later_replay = _replay(
            cutoff=cutoff_b,
            family=later_family,
            ready=ready_h1,
            previous=f"replay:{cutoff_a}:{later_family}",
        )
        jobs[later_replay["job_id"]] = later_replay
        jobs[f"evidence:{cutoff_b}:{later_family}"] = _evidence(
            cutoff=cutoff_b,
            family=later_family,
        )
        store.seed_jobs(jobs)

        groups = _earliest_cutoff_groups(store, kind="replay")
        assert len(groups) == 2
        assert [group["horizon"] for group in groups] == [1, 2]
        assert all(group["cutoff"] == cutoff_a for group in groups)
        assert [len(group["jobs"]) for group in groups] == [2, 1]
        assert later_replay["job_id"] not in {
            job["job_id"] for group in groups for job in group["jobs"]
        }

        owner = "self-test-batch"
        h1_group = groups[0]
        claimed = _claim_exact_batch(
            store,
            job_ids=[job["job_id"] for job in h1_group["jobs"]],
            kind="replay",
            owner=owner,
            lease_seconds=3600,
        )
        assert len(claimed) == 2
        assert all(store.job(job["job_id"])["state"] == "RUNNING"
                   for job in claimed)
        assert all(store.job(job["job_id"])["lease_owner"] == owner
                   for job in claimed)

        # Simulate one family finishing before its physical lane is reclaimed.
        first = claimed[0]
        store.finish_job(
            first["job_id"], owner, "COMPLETE",
            result={"synthetic": True},
        )
        requeued = _requeue_batch_owner(store, owner)
        assert requeued == 1
        assert store.job(first["job_id"])["state"] == "COMPLETE"
        unfinished = claimed[1]
        assert store.job(unfinished["job_id"])["state"] == "PENDING"
        assert store.job("independent")["state"] == "COMPLETE"
        assert store.job(later_replay["job_id"])["state"] == "PENDING"

        # Logical scientific rows remain unchanged; only the physical
        # scheduling dimension collapsed from per-family rows to H×cutoff.
        progress = store.progress()
        assert progress["total"] == len(jobs)
        assert len(first_replays) == 3
        assert len(groups) == 2

    print("dynamic-qbd batched portfolio runtime self-test: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
