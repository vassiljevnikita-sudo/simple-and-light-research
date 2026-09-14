"""Focused self-test for Dynamic-QBD COMPLETE-result resume authority."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile

from .dynamic_qbd_complete_result_validation import (
    CompleteResultValidationError,
    _accepted_run_contract_hashes,
    _scientific_contract_hash,
    audit_complete_results,
    complete_result_validation_authority_path,
    complete_result_validation_authority_sha256,
)
from .dynamic_qbd_manifested_job_coordinator import ManifestedJobStore


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _contract(*, git_sha: str, run_hash: str) -> dict:
    return {
        "schema_version": "SELF_TEST_CONTRACT_V1",
        "git_sha": git_sha,
        "source_tree_sha256": f"tree-{git_sha}",
        "worktree_status": "CLEAN",
        "dirty_patch_sha256": None,
        "run_contract_hash": run_hash,
        "semantic_marker": "SAME_SCIENCE",
        "fold_policy_hash": "fold-v1",
        "target_contract_hash": "target-v1",
        "recipe_selection_policy_hash": "selection-v1",
    }


def _seed_metadata(root: Path, contract: dict) -> ManifestedJobStore:
    store = ManifestedJobStore(root)
    _write_json(root / "manifested-job-contract.json", contract)
    _write_json(root / "family-registry.json", {"portfolio_families": []})
    return store


def _selection_payload(path: Path, *, cutoff: str = "2025-01-31") -> dict:
    return {
        "selected_recipe_sha256": "recipe-sha",
        "selected_candidate_id": "candidate-a",
        "horizon": 1,
        "selection_cutoff": cutoff,
        "selection_policy_hash": "selection-v1",
    }


def _selection_job(
    *, job_id: str, path: Path, job_cutoff: str,
    artifact_cutoff: str = "2025-01-31",
    depends_on: list[str] | None = None,
) -> dict:
    payload = _selection_payload(path, cutoff=artifact_cutoff)
    _write_json(path, payload)
    return {
        "job_id": job_id,
        "kind": "recipe_selection",
        "state": "COMPLETE",
        "cutoff": job_cutoff,
        "model_family_key": "H01_RIDGE_HGB_FROZEN_RULE",
        "recipe_selection_policy_hash": "selection-v1",
        "depends_on": list(depends_on or []),
        "result": {
            "path": str(path),
            "selected_recipe_sha256": payload["selected_recipe_sha256"],
            "selected_candidate_id": payload["selected_candidate_id"],
            "horizon": payload["horizon"],
            "selection_cutoff": payload["selection_cutoff"],
        },
    }


def main() -> int:
    # Scheduler/code provenance must not invalidate scientifically identical
    # immutable results merely because the full run-contract hash changes.
    prior = _contract(git_sha="a" * 40, run_hash="old-run")
    current = _contract(git_sha="b" * 40, run_hash="new-run")
    assert _scientific_contract_hash(prior) == _scientific_contract_hash(current)
    accepted = _accepted_run_contract_hashes(current, prior, None)
    assert accepted == {"old-run", "new-run"}

    with tempfile.TemporaryDirectory(
        prefix="dqbd-complete-validation-"
    ) as temporary:
        root = Path(temporary)

        # Fast authority is valid only for the exact audited job-store state.
        cache_root = root / "authority-cache"
        store = _seed_metadata(cache_root, current)
        store.seed_jobs({
            "input-a": {
                "job_id": "input-a",
                "kind": "input",
                "state": "COMPLETE",
                "depends_on": [],
            },
        })
        first = audit_complete_results(store=store, prior_contract=prior)
        assert first["status"] == "VALIDATED_COMPLETE_FRONTIER"
        assert first["complete_job_count"] == 1
        assert first["full_artifact_validations"] == 1
        first_token = complete_result_validation_authority_sha256(cache_root)
        assert first_token

        second = audit_complete_results(store=store, prior_contract=prior)
        assert second["complete_job_count"] == 1
        assert second["validation_cache_hits"] == 1
        assert second["full_artifact_validations"] == 0
        second_token = complete_result_validation_authority_sha256(cache_root)
        assert second_token

        store.seed_jobs({
            "input-a": {
                "job_id": "input-a",
                "kind": "input",
                "state": "COMPLETE",
                "depends_on": [],
            },
            "input-b": {
                "job_id": "input-b",
                "kind": "input",
                "state": "COMPLETE",
                "depends_on": [],
            },
        })
        stale_token = complete_result_validation_authority_sha256(cache_root)
        assert stale_token != second_token
        third = audit_complete_results(store=store, prior_contract=prior)
        assert third["complete_job_count"] == 2
        assert third["validation_cache_hits"] >= 1
        assert third["full_artifact_validations"] >= 1

        # A self-consistent old artifact with an obsolete causal identity is
        # not corruption. The validator requeues that root and only derived
        # descendants while retaining unrelated COMPLETE checkpoints.
        stale_root = root / "stale-derived-state"
        stale_store = _seed_metadata(stale_root, current)
        stale_selection_path = (
            stale_root / "recipe-selections" / "stale-selection.json")
        stale_store.seed_jobs({
            "source": {
                "job_id": "source",
                "kind": "input",
                "state": "COMPLETE",
                "depends_on": [],
            },
            "stale-selection": _selection_job(
                job_id="stale-selection",
                path=stale_selection_path,
                job_cutoff="2025-02-28",
                artifact_cutoff="2025-01-31",
                depends_on=["source"],
            ),
            "derived-child": {
                "job_id": "derived-child",
                "kind": "input",
                "state": "COMPLETE",
                "depends_on": ["stale-selection"],
            },
            "independent": {
                "job_id": "independent",
                "kind": "input",
                "state": "COMPLETE",
                "depends_on": [],
            },
        })
        repaired = audit_complete_results(
            store=stale_store, prior_contract=prior)
        assert repaired["status"] == (
            "VALIDATED_COMPLETE_FRONTIER_STALE_REQUEUED")
        assert repaired["stale_roots_requeued"] == 1
        assert repaired["requeued_complete_jobs"] == 2
        assert stale_store.job("stale-selection")["state"] == "PENDING"
        assert stale_store.job("derived-child")["state"] == "PENDING"
        assert stale_store.job("source")["state"] == "COMPLETE"
        assert stale_store.job("independent")["state"] == "COMPLETE"
        assert complete_result_validation_authority_sha256(stale_root)

        # A changed backing artifact is different: disk and persisted COMPLETE
        # result disagree, so the run remains fail-closed.
        corrupt_root = root / "corrupt-result"
        corrupt_store = _seed_metadata(corrupt_root, current)
        selection_path = (
            corrupt_root / "recipe-selections" / "selection.json")
        selection_job = _selection_job(
            job_id="recipe-selection",
            path=selection_path,
            job_cutoff="2025-01-31",
        )
        corrupt_store.seed_jobs({
            "source": {
                "job_id": "source",
                "kind": "input",
                "state": "COMPLETE",
                "depends_on": [],
            },
            "recipe-selection": selection_job,
        })
        clean = audit_complete_results(
            store=corrupt_store, prior_contract=prior)
        assert clean["complete_job_count"] == 2
        assert clean["corrupt_complete_jobs"] == 0

        tampered = _selection_payload(selection_path)
        tampered["selected_candidate_id"] = "tampered-candidate"
        _write_json(selection_path, tampered)
        try:
            audit_complete_results(
                store=corrupt_store, prior_contract=prior)
        except CompleteResultValidationError as exc:
            assert exc.job_id == "recipe-selection"
            assert "SELECTION_RESULT_MISMATCH" in exc.reason
        else:
            raise AssertionError("CORRUPT_COMPLETE_RESULT_WAS_ACCEPTED")
        blocked = json.loads(
            complete_result_validation_authority_path(corrupt_root).read_text(
                encoding="utf-8"))
        assert blocked["status"] == "CORRUPT_COMPLETE_RESULT_BLOCKED"
        assert blocked["corrupt_job_id"] == "recipe-selection"
        assert complete_result_validation_authority_sha256(corrupt_root) is None
        assert blocked["evaluation_opened"] is False
        assert blocked["holdout_opened"] is False

        # Graph reconciliation remains the first line of defense. Payload
        # drift requeues the changed node and its descendants before artifact
        # validation; an unrelated COMPLETE sibling survives.
        drift_root = root / "payload-drift"
        drift_store = _seed_metadata(drift_root, current)
        v1 = {
            "parent": {
                "job_id": "parent",
                "kind": "input",
                "state": "COMPLETE",
                "semantic_value": "v1",
                "depends_on": [],
            },
            "child": {
                "job_id": "child",
                "kind": "input",
                "state": "COMPLETE",
                "depends_on": ["parent"],
            },
            "sibling": {
                "job_id": "sibling",
                "kind": "input",
                "state": "COMPLETE",
                "depends_on": [],
            },
        }
        drift_store.seed_jobs(v1)
        v2 = {key: dict(value) for key, value in v1.items()}
        v2["parent"] = dict(v2["parent"], semantic_value="v2")
        reconciliation = drift_store.seed_jobs(
            v2, invalidate_descendants=True)
        assert reconciliation["updated_jobs"] == 1
        assert drift_store.job("parent")["state"] == "PENDING"
        assert drift_store.job("child")["state"] == "PENDING"
        assert drift_store.job("sibling")["state"] == "COMPLETE"
        drift_audit = audit_complete_results(
            store=drift_store, prior_contract=prior)
        assert drift_audit["complete_job_count"] == 1
        assert drift_audit["complete_jobs_by_kind"] == {"input": 1}

    print("dynamic-qbd complete-result validation self-test: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
