"""Central COMPLETE-result validation authority for Dynamic-QBD resumes.

A scheduler row is not scientific evidence. Fast resume therefore receives
authority only after the current manifested DAG has been reconciled and every
surviving COMPLETE result has been checked against its immutable artifacts.

V2 makes one distinction explicit:

* corrupt/partial/hash-invalid artifacts fail closed;
* self-consistent but scientifically stale artifacts are not fatal. Their
  owning job and only its transitive derived descendants are returned to
  PENDING and recomputed by the normal causal scheduler;
* unrelated COMPLETE checkpoints survive;
* scheduler/code-only provenance changes may retain scientifically compatible
  immutable results even when the full run-contract hash changes;
* validation is cached by semantic payload, result, artifact-stat and
  scientific-contract identity and is streamed directly from SQLite;
* the fast-resume authority token also binds the current SQLite/WAL state, so
  work completed after an audit automatically invalidates the old authority.

The validator never promotes work to COMPLETE, never opens the prospective
holdout and never weakens seed-local causal visibility.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sqlite3
import time
from typing import Any, Iterable, Mapping

import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .candidate_oos import CandidateOosStore, canonical_sha256
from .contract_fingerprints import stable_hash
from .dynamic_qbd_manifested_job_coordinator import (
    ManifestedJobStore,
    _compatible_resume_contract_view,
    sha256_file,
)


COMPLETE_RESULT_VALIDATION_SCHEMA = (
    "DQBD_COMPLETE_RESULT_VALIDATION_V2_STALE_REQUEUE")
COMPLETE_RESULT_VALIDATION_CACHE_SCHEMA = (
    "DQBD_COMPLETE_RESULT_VALIDATION_CACHE_V2")
AUTHORITY_FILENAME = "complete-result-validation-authority-v2.json"
CACHE_FILENAME = "complete-result-validation-v2.sqlite3"

_RUNTIME_PAYLOAD_KEYS = {
    "state", "result", "last_error", "started_at", "finished_at",
    "lease_owner", "lease_until", "heartbeat", "attempt", "payload_hash",
    "ram_reclaim_count", "ram_last_reclaim_at", "ram_last_reclaim_reason",
    "ram_last_released_gib", "worker_failure_count", "last_worker_failure_at",
    "last_worker_failure", "execution_preference",
}


class CompleteResultValidationError(RuntimeError):
    """A persisted COMPLETE artifact is missing, corrupt or unverifiable."""

    def __init__(self, job_id: str, reason: str) -> None:
        self.job_id = str(job_id)
        self.reason = str(reason)
        super().__init__(
            f"DQBD_COMPLETE_RESULT_CORRUPT:{self.job_id}:{self.reason}")


class CompleteResultStaleError(RuntimeError):
    """A valid old artifact no longer belongs to the current causal job."""

    def __init__(self, job_id: str, reason: str) -> None:
        self.job_id = str(job_id)
        self.reason = str(reason)
        super().__init__(
            f"DQBD_COMPLETE_RESULT_STALE_REQUEUE:{self.job_id}:{self.reason}")


@dataclass(frozen=True)
class _ValidationContext:
    store: ManifestedJobStore
    contract: Mapping[str, Any]
    registry: Mapping[str, Any]
    scientific_contract_hash: str
    accepted_run_contract_hashes: frozenset[str]


def _json_hash(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, default=str, separators=(",", ":"),
        allow_nan=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{time.time_ns()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _scientific_contract_hash(contract: Mapping[str, Any]) -> str:
    return stable_hash(_compatible_resume_contract_view(contract))


def complete_result_validation_authority_path(root: str | Path) -> Path:
    return Path(root) / "run-state" / AUTHORITY_FILENAME


def _job_store_state_fingerprint(root: str | Path) -> str:
    """Cheap token for the persisted scheduler state in SQLite/WAL mode."""
    run_state = Path(root) / "run-state"
    rows = []
    for name in ("jobs.sqlite3", "jobs.sqlite3-wal"):
        path = run_state / name
        try:
            stat = path.stat()
            rows.append({
                "name": name,
                "exists": True,
                "size": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
            })
        except OSError:
            rows.append({"name": name, "exists": False})
    return _json_hash(rows)


def complete_result_validation_authority_sha256(
    root: str | Path,
) -> str | None:
    """Return fast-resume authority only for a successful V2 audit."""
    path = complete_result_validation_authority_path(root)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    if not str(payload.get("status", "")).startswith(
        "VALIDATED_COMPLETE_FRONTIER"
    ):
        return None
    if payload.get("schema_version") != COMPLETE_RESULT_VALIDATION_SCHEMA:
        return None
    return stable_hash({
        "authority_file_sha256": sha256_file(path),
        "job_store_state_fingerprint": _job_store_state_fingerprint(root),
    })


def _normalized_path(root: Path, value: Any) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else root / path


def _load_json(path: Path, *, job_id: str, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise CompleteResultValidationError(
            job_id, f"{label}_MISSING:{path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as exc:
        raise CompleteResultValidationError(
            job_id, f"{label}_INVALID_JSON:{type(exc).__name__}") from exc
    if not isinstance(value, dict):
        raise CompleteResultValidationError(job_id, f"{label}_NOT_OBJECT")
    return value


def _require_hash(
    path: Path, expected: Any, *, job_id: str, label: str,
) -> None:
    # Missing legacy hash metadata means the old artifact cannot acquire
    # current authority, but it is not evidence of corruption. Recompute it.
    if not expected:
        raise CompleteResultStaleError(
            job_id, f"{label}_EXPECTED_HASH_MISSING")
    if not path.is_file():
        raise CompleteResultValidationError(
            job_id, f"{label}_MISSING:{path}")
    actual = sha256_file(path)
    if actual != str(expected):
        raise CompleteResultValidationError(
            job_id, f"{label}_HASH_MISMATCH:{actual}:{expected}")


def _require_contract_compatibility(
    payload: Mapping[str, Any], ctx: _ValidationContext, *, job_id: str,
) -> None:
    scientific = payload.get("scientific_contract_hash")
    if scientific is not None:
        if str(scientific) != ctx.scientific_contract_hash:
            raise CompleteResultStaleError(
                job_id, "SCIENTIFIC_CONTRACT_HASH_MISMATCH")
        return
    run_hash = payload.get("run_contract_hash")
    if run_hash is None:
        raise CompleteResultStaleError(
            job_id, "ARTIFACT_CONTRACT_IDENTITY_MISSING")
    if str(run_hash) not in ctx.accepted_run_contract_hashes:
        raise CompleteResultStaleError(
            job_id, "LEGACY_RUN_CONTRACT_NOT_COMPATIBLE")


def _generation_manifest(
    result: Mapping[str, Any], *, job_id: str, root: Path,
) -> dict[str, Any]:
    embedded = (
        result.get("artifact_manifest") or result.get("generation_manifest"))
    manifest_path = result.get("manifest_path")
    if manifest_path:
        disk = _load_json(
            _normalized_path(root, manifest_path),
            job_id=job_id,
            label="GENERATION_MANIFEST",
        )
        if isinstance(embedded, Mapping):
            embedded_sha = embedded.get("manifest_sha256")
            disk_sha = disk.get("manifest_sha256")
            if (
                embedded_sha and disk_sha
                and str(embedded_sha) != str(disk_sha)
            ):
                raise CompleteResultValidationError(
                    job_id, "GENERATION_EMBEDDED_MANIFEST_MISMATCH")
        return disk
    if not isinstance(embedded, Mapping):
        raise CompleteResultStaleError(
            job_id, "GENERATION_MANIFEST_MISSING")
    return dict(embedded)


def _validate_generation_files(
    result: Mapping[str, Any], ctx: _ValidationContext, *, job_id: str,
) -> dict[str, Any]:
    manifest = _generation_manifest(
        result, job_id=job_id, root=ctx.store.root)
    generation_id = str(manifest.get("generation_id", ""))
    if not generation_id:
        raise CompleteResultStaleError(job_id, "GENERATION_ID_MISSING")
    if (
        result.get("generation_id")
        and str(result["generation_id"]) != generation_id
    ):
        raise CompleteResultValidationError(
            job_id, "GENERATION_ID_RESULT_MISMATCH")
    for result_key, manifest_key, label in (
        ("model_path", "model_artifact_sha256", "GENERATION_MODEL"),
        ("calibration_path", "calibration_sha256", "GENERATION_CALIBRATION"),
        ("prediction_path", "prediction_sha256", "GENERATION_PREDICTION"),
    ):
        expected = manifest.get(manifest_key)
        path_value = result.get(result_key) or manifest.get(result_key)
        if not expected and not path_value:
            continue
        if not path_value:
            raise CompleteResultStaleError(
                job_id, f"{label}_PATH_MISSING")
        _require_hash(
            _normalized_path(ctx.store.root, path_value),
            expected,
            job_id=job_id,
            label=label,
        )
    return manifest


def _validate_candidate(
    job: Mapping[str, Any], result: Mapping[str, Any], ctx: _ValidationContext,
) -> dict[str, Any]:
    job_id = str(job["job_id"])
    path_value = result.get("path")
    if not path_value:
        raise CompleteResultStaleError(job_id, "CANDIDATE_PATH_MISSING")
    path = _normalized_path(ctx.store.root, path_value)
    try:
        manifest = CandidateOosStore(
            ctx.store.root / "candidate-oos").verify_partition_metadata(path)
    except Exception as exc:
        reason = str(exc)
        if "PARTITION_SCHEMA_MISSING" in reason:
            raise CompleteResultStaleError(
                job_id, f"CANDIDATE_PARTITION_LEGACY_SCHEMA:{reason}") from exc
        raise CompleteResultValidationError(
            job_id,
            f"CANDIDATE_PARTITION_INVALID:{type(exc).__name__}:{reason}",
        ) from exc
    for key, expected in (
        ("horizon", int(job["horizon"])),
        ("fold_id", str(job["fold_id"])),
        ("candidate_id", str(job["candidate_id"])),
    ):
        if str(manifest.get(key)) != str(expected):
            raise CompleteResultStaleError(
                job_id, f"CANDIDATE_IDENTITY_MISMATCH:{key}")
    if (
        result.get("manifest_sha256")
        and manifest.get("manifest_sha256")
        and str(result["manifest_sha256"])
            != str(manifest["manifest_sha256"])
    ):
        raise CompleteResultValidationError(
            job_id, "CANDIDATE_MANIFEST_RESULT_MISMATCH")
    return {"manifest_sha256": manifest.get("manifest_sha256")}


def _verify_snapshot_metadata(
    path: Path, *, job_id: str,
) -> dict[str, Any]:
    manifest_path = path.with_name("manifest.json")
    if not path.is_file() or not manifest_path.is_file():
        raise CompleteResultValidationError(
            job_id, "SNAPSHOT_ARTIFACT_OR_MANIFEST_MISSING")
    manifest = _load_json(
        manifest_path, job_id=job_id, label="SNAPSHOT_MANIFEST")
    if canonical_sha256({
        key: value for key, value in manifest.items()
        if key != "manifest_sha256"
    }) != manifest.get("manifest_sha256"):
        raise CompleteResultValidationError(
            job_id, "SNAPSHOT_MANIFEST_HASH_MISMATCH")
    _require_hash(
        path, manifest.get("snapshot_sha256"),
        job_id=job_id, label="SNAPSHOT_FILE")
    selection_cutoff = manifest.get("selection_cutoff")
    if not selection_cutoff:
        raise CompleteResultStaleError(
            job_id, "SNAPSHOT_SELECTION_CUTOFF_MISSING")
    try:
        cutoff_timestamp = pd.Timestamp(selection_cutoff)
    except (TypeError, ValueError) as exc:
        raise CompleteResultStaleError(
            job_id, "SNAPSHOT_SELECTION_CUTOFF_INVALID") from exc
    try:
        parquet = pq.ParquetFile(path)
    except Exception as exc:
        raise CompleteResultValidationError(
            job_id,
            f"SNAPSHOT_PARQUET_INVALID:{type(exc).__name__}:{exc}",
        ) from exc
    names = set(parquet.schema_arrow.names)
    required = {
        "horizon", "fold_id", "candidate_id", "information_available_at"}
    missing = sorted(required - names)
    if missing:
        raise CompleteResultStaleError(
            job_id, f"SNAPSHOT_LEGACY_SCHEMA_MISSING:{missing}")
    observation_count = manifest.get("observation_count")
    if observation_count is None:
        raise CompleteResultStaleError(
            job_id, "SNAPSHOT_OBSERVATION_COUNT_MISSING")
    try:
        expected_rows = int(observation_count)
    except (TypeError, ValueError) as exc:
        raise CompleteResultStaleError(
            job_id, "SNAPSHOT_OBSERVATION_COUNT_INVALID") from exc
    if int(parquet.metadata.num_rows) != expected_rows:
        raise CompleteResultValidationError(
            job_id, "SNAPSHOT_ROW_COUNT_MISMATCH")
    # Read only the causal timestamp column. This preserves the historical
    # future-information guard without materializing the complete snapshot.
    try:
        timestamp_table = pq.read_table(
            path, columns=["information_available_at"])
        maximum = pc.max(
            timestamp_table["information_available_at"]).as_py()
    except Exception as exc:
        raise CompleteResultValidationError(
            job_id,
            f"SNAPSHOT_TIMESTAMP_COLUMN_INVALID:{type(exc).__name__}:{exc}",
        ) from exc
    if maximum is not None:
        try:
            maximum_timestamp = pd.Timestamp(maximum)
        except (TypeError, ValueError) as exc:
            raise CompleteResultValidationError(
                job_id, "SNAPSHOT_TIMESTAMP_VALUE_INVALID") from exc
        if maximum_timestamp > cutoff_timestamp:
            raise CompleteResultValidationError(
                job_id, "SNAPSHOT_FUTURE_INFORMATION")
    return manifest


def _validate_coverage(
    job: Mapping[str, Any], result: Mapping[str, Any], ctx: _ValidationContext,
) -> dict[str, Any]:
    job_id = str(job["job_id"])
    snapshot_path = result.get("snapshot_path")
    if not snapshot_path:
        raise CompleteResultStaleError(job_id, "SNAPSHOT_PATH_MISSING")
    manifest = _verify_snapshot_metadata(
        _normalized_path(ctx.store.root, snapshot_path), job_id=job_id)
    horizon = int(str(job["model_family_key"])[1:3])
    if int(manifest.get("horizon", -1)) != horizon:
        raise CompleteResultStaleError(job_id, "SNAPSHOT_HORIZON_MISMATCH")
    if str(manifest.get("selection_cutoff")) != str(job.get("cutoff")):
        raise CompleteResultStaleError(job_id, "SNAPSHOT_CUTOFF_MISMATCH")
    registry_hash = ctx.registry.get("candidate_registry", {}).get(
        "candidate_registry_sha256")
    manifest_registry = (
        manifest.get("candidate_registry_hash")
        or manifest.get("candidate_registry_sha256"))
    if registry_hash and str(manifest_registry) != str(registry_hash):
        raise CompleteResultStaleError(
            job_id, "SNAPSHOT_CANDIDATE_REGISTRY_MISMATCH")
    rows = int(manifest.get("observation_count", 0))
    if result.get("rows") is not None and int(result["rows"]) != rows:
        raise CompleteResultStaleError(
            job_id, "SNAPSHOT_RESULT_ROW_COUNT_STALE")
    return {
        "evidence_snapshot_hash": manifest.get("evidence_snapshot_hash"),
        "rows": rows,
    }


def _validate_selection(
    job: Mapping[str, Any], result: Mapping[str, Any], ctx: _ValidationContext,
) -> dict[str, Any]:
    job_id = str(job["job_id"])
    path_value = result.get("path")
    if not path_value:
        raise CompleteResultStaleError(job_id, "SELECTION_PATH_MISSING")
    disk = _load_json(
        _normalized_path(ctx.store.root, path_value),
        job_id=job_id,
        label="SELECTION",
    )
    for key in (
        "selected_recipe_sha256", "selected_candidate_id", "horizon",
        "selection_cutoff",
    ):
        if (
            result.get(key) is not None
            and str(disk.get(key)) != str(result.get(key))
        ):
            # Disk/result disagreement means a published COMPLETE artifact was
            # changed after the scheduler recorded it. That remains corrupt.
            raise CompleteResultValidationError(
                job_id, f"SELECTION_RESULT_MISMATCH:{key}")
    if str(disk.get("selection_cutoff")) != str(job.get("cutoff")):
        raise CompleteResultStaleError(job_id, "SELECTION_CUTOFF_MISMATCH")
    if (
        job.get("recipe_selection_policy_hash")
        and disk.get("selection_policy_hash")
        and str(job["recipe_selection_policy_hash"])
            != str(disk["selection_policy_hash"])
    ):
        raise CompleteResultStaleError(job_id, "SELECTION_POLICY_MISMATCH")
    return {"selected_recipe_sha256": disk.get("selected_recipe_sha256")}


def _validate_model_or_generation_child(
    job: Mapping[str, Any], result: Mapping[str, Any], ctx: _ValidationContext,
) -> dict[str, Any]:
    job_id = str(job["job_id"])
    manifest = _validate_generation_files(result, ctx, job_id=job_id)
    if (
        job.get("cutoff")
        and str(manifest.get("information_cutoff")) != str(job["cutoff"])
    ):
        raise CompleteResultStaleError(
            job_id, "GENERATION_INFORMATION_CUTOFF_MISMATCH")
    return {"generation_id": manifest.get("generation_id")}


def _family_by_key(ctx: _ValidationContext, family_key: str) -> dict[str, Any]:
    for family in ctx.registry.get("portfolio_families", ()):
        if str(family.get("portfolio_family_key")) == str(family_key):
            return dict(family)
    raise CompleteResultStaleError(
        family_key, "PORTFOLIO_FAMILY_NOT_REGISTERED")


def _without_hash(
    manifest: Mapping[str, Any], hash_key: str,
) -> tuple[dict[str, Any], Any]:
    body = dict(manifest)
    expected = body.pop(hash_key, None)
    return body, expected


def _validate_replay(
    job: Mapping[str, Any], result: Mapping[str, Any], ctx: _ValidationContext,
) -> dict[str, Any]:
    job_id = str(job["job_id"])
    manifest_value = result.get("replay_manifest_path")
    if not manifest_value:
        raise CompleteResultStaleError(
            job_id, "REPLAY_MANIFEST_PATH_MISSING")
    raw = _load_json(
        _normalized_path(ctx.store.root, manifest_value),
        job_id=job_id,
        label="REPLAY_MANIFEST",
    )
    manifest, artifact_hash = _without_hash(raw, "artifact_hash")
    if not artifact_hash or str(artifact_hash) != str(stable_hash(manifest)):
        raise CompleteResultValidationError(
            job_id, "REPLAY_MANIFEST_SELF_HASH_MISMATCH")
    _require_contract_compatibility(manifest, ctx, job_id=job_id)

    family_key = str(job["portfolio_family_key"])
    try:
        family = _family_by_key(ctx, family_key)
    except CompleteResultStaleError as exc:
        raise CompleteResultStaleError(
            job_id, exc.reason) from exc
    expected_identity = {
        "family_id": str(family["family_id"]),
        "segment_start": str(job.get("segment_start")),
        "segment_end": str(job.get("segment_end")),
        "fold_policy_hash": str(ctx.contract.get("fold_policy_hash")),
        "target_contract_hash": str(ctx.contract.get("target_contract_hash")),
        "cost_contract_hash": str(stable_hash(family["cost_contract"])),
        "tax_contract_hash": str(stable_hash(family["tax_contract"])),
    }
    for key, expected in expected_identity.items():
        if str(manifest.get(key)) != expected:
            raise CompleteResultStaleError(
                job_id, f"REPLAY_IDENTITY_MISMATCH:{key}")

    for path_key, hash_key, label in (
        ("nav_path", "nav_sha256", "REPLAY_NAV"),
        ("trades_path", "trades_sha256", "REPLAY_TRADES"),
        ("state_path", "state_sha256", "REPLAY_STATE"),
    ):
        path_value = manifest.get(path_key) or result.get(path_key)
        if not path_value:
            raise CompleteResultStaleError(job_id, f"{label}_PATH_MISSING")
        _require_hash(
            _normalized_path(ctx.store.root, path_value),
            manifest.get(hash_key),
            job_id=job_id,
            label=label,
        )

    state = _load_json(
        _normalized_path(ctx.store.root, manifest["state_path"]),
        job_id=job_id,
        label="REPLAY_STATE",
    )
    state_hash = stable_hash(state)
    if (
        str(manifest.get("terminal_state_hash")) != str(state_hash)
        or str(manifest.get("state_hash")) != str(state_hash)
    ):
        raise CompleteResultValidationError(
            job_id, "REPLAY_TERMINAL_STATE_HASH_MISMATCH")
    if result.get("state_hash") and str(result["state_hash"]) != str(state_hash):
        raise CompleteResultValidationError(
            job_id, "REPLAY_RESULT_STATE_HASH_MISMATCH")
    if (
        result.get("replay_state") is not None
        and stable_hash(result["replay_state"]) != state_hash
    ):
        raise CompleteResultValidationError(
            job_id, "REPLAY_RESULT_STATE_CONTENT_MISMATCH")

    previous_ids = [
        str(value) for value in job.get("depends_on", ())
        if str(value).startswith("replay:")]
    expected_previous = None
    if previous_ids:
        previous = ctx.store.job(previous_ids[-1])
        if previous is None or previous.get("state") != "COMPLETE":
            raise CompleteResultStaleError(
                job_id, "REPLAY_PREVIOUS_JOB_NOT_COMPLETE")
        expected_previous = (previous.get("result") or {}).get("state_hash")
    if str(manifest.get("previous_state_hash")) != str(expected_previous):
        raise CompleteResultStaleError(
            job_id, "REPLAY_PREVIOUS_STATE_HASH_MISMATCH")

    ready_id = str(job.get("generation_ready_job_id", ""))
    ready = ctx.store.job(ready_id) if ready_id else None
    if ready is None or ready.get("state") != "COMPLETE":
        raise CompleteResultStaleError(
            job_id, "REPLAY_GENERATION_READY_NOT_COMPLETE")
    generation_id = str((ready.get("result") or {}).get("generation_id", ""))
    if not generation_id or str(manifest.get("generation_id")) != generation_id:
        raise CompleteResultStaleError(
            job_id, "REPLAY_GENERATION_ID_MISMATCH")
    generation_manifest = (
        (ready.get("result") or {}).get("generation_manifest") or {})
    if (
        generation_manifest.get("manifest_sha256")
        and str(manifest.get("generation_manifest_sha256"))
            != str(generation_manifest.get("manifest_sha256"))
    ):
        raise CompleteResultStaleError(
            job_id, "REPLAY_GENERATION_MANIFEST_MISMATCH")
    return {"artifact_hash": artifact_hash, "state_hash": state_hash}


def _validate_evidence(
    job: Mapping[str, Any], result: Mapping[str, Any], ctx: _ValidationContext,
) -> dict[str, Any]:
    job_id = str(job["job_id"])
    manifest_value = result.get("manifest_path")
    evidence_value = result.get("evidence_path")
    if not manifest_value or not evidence_value:
        raise CompleteResultStaleError(job_id, "EVIDENCE_PATHS_MISSING")
    raw = _load_json(
        _normalized_path(ctx.store.root, manifest_value),
        job_id=job_id,
        label="EVIDENCE_MANIFEST",
    )
    manifest, manifest_hash = _without_hash(raw, "manifest_sha256")
    if not manifest_hash or str(manifest_hash) != str(stable_hash(manifest)):
        raise CompleteResultValidationError(
            job_id, "EVIDENCE_MANIFEST_SELF_HASH_MISMATCH")
    _require_contract_compatibility(manifest, ctx, job_id=job_id)
    if str(result.get("manifest_sha256")) != str(manifest_hash):
        raise CompleteResultValidationError(
            job_id, "EVIDENCE_RESULT_MANIFEST_MISMATCH")
    _require_hash(
        _normalized_path(ctx.store.root, evidence_value),
        manifest.get("evidence_sha256"),
        job_id=job_id,
        label="EVIDENCE_FILE",
    )
    if str(manifest.get("family_key")) != str(job.get("portfolio_family_key")):
        raise CompleteResultStaleError(job_id, "EVIDENCE_FAMILY_MISMATCH")
    if str(manifest.get("assessment_date")) != str(job.get("cutoff")):
        raise CompleteResultStaleError(job_id, "EVIDENCE_CUTOFF_MISMATCH")
    for manifest_key, contract_key in (
        ("target_contract_hash", "target_contract_hash"),
        ("fold_policy_hash", "fold_policy_hash"),
        ("recipe_selection_policy_hash", "recipe_selection_policy_hash"),
    ):
        if str(manifest.get(manifest_key)) != str(ctx.contract.get(contract_key)):
            raise CompleteResultStaleError(
                job_id, f"EVIDENCE_POLICY_CONTRACT_MISMATCH:{manifest_key}")

    replay_ids = [
        str(value) for value in job.get("depends_on", ())
        if str(value).startswith("replay:")]
    if not replay_ids:
        raise CompleteResultStaleError(
            job_id, "EVIDENCE_REPLAY_DEPENDENCY_MISSING")
    replay = ctx.store.job(replay_ids[0])
    if replay is None or replay.get("state") != "COMPLETE":
        raise CompleteResultStaleError(job_id, "EVIDENCE_REPLAY_NOT_COMPLETE")
    replay_manifest_value = (
        (replay.get("result") or {}).get("replay_manifest_path"))
    if not replay_manifest_value:
        raise CompleteResultStaleError(
            job_id, "EVIDENCE_REPLAY_MANIFEST_MISSING")
    replay_raw = _load_json(
        _normalized_path(ctx.store.root, replay_manifest_value),
        job_id=job_id,
        label="EVIDENCE_REPLAY_MANIFEST",
    )
    replay_body, replay_hash = _without_hash(replay_raw, "artifact_hash")
    if not replay_hash or str(replay_hash) != str(stable_hash(replay_body)):
        raise CompleteResultValidationError(
            job_id, "EVIDENCE_REPLAY_SELF_HASH_MISMATCH")
    if str(manifest.get("replay_manifest_hash")) != str(stable_hash(replay_body)):
        raise CompleteResultStaleError(
            job_id, "EVIDENCE_REPLAY_HASH_MISMATCH")
    return {"manifest_sha256": manifest_hash}


def _validate_complete_job(
    job: Mapping[str, Any], ctx: _ValidationContext,
) -> dict[str, Any]:
    job_id = str(job["job_id"])
    kind = str(job.get("kind", ""))
    result = job.get("result")
    if kind == "input":
        return {"kind": kind, "input_complete": True}
    if not isinstance(result, Mapping):
        raise CompleteResultStaleError(job_id, "COMPLETE_RESULT_MISSING")
    result = dict(result)
    if kind == "candidate_oos_fold":
        return _validate_candidate(job, result, ctx)
    if kind == "candidate_evidence_coverage":
        return _validate_coverage(job, result, ctx)
    if kind == "recipe_selection":
        return _validate_selection(job, result, ctx)
    if kind in {"model", "calibration", "prediction", "generation_ready"}:
        return _validate_model_or_generation_child(job, result, ctx)
    if kind == "replay":
        return _validate_replay(job, result, ctx)
    if kind == "evidence":
        return _validate_evidence(job, result, ctx)
    raise CompleteResultValidationError(
        job_id, f"VALIDATOR_UNSUPPORTED_COMPLETE_KIND:{kind}")


def _artifact_paths(job: Mapping[str, Any], root: Path) -> list[Path]:
    result = job.get("result") or {}
    if not isinstance(result, Mapping):
        return []
    paths: set[Path] = set()
    for key in (
        "path", "snapshot_path", "manifest_path", "model_path",
        "calibration_path", "prediction_path", "nav_path", "trades_path",
        "replay_manifest_path", "evidence_path",
    ):
        value = result.get(key)
        if value:
            paths.add(_normalized_path(root, value))
    snapshot_value = result.get("snapshot_path")
    if snapshot_value:
        paths.add(
            _normalized_path(root, snapshot_value).with_name("manifest.json"))
    for key in ("manifest_path", "replay_manifest_path"):
        value = result.get(key)
        if not value:
            continue
        path = _normalized_path(root, value)
        if not path.is_file():
            continue
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            continue
        if isinstance(manifest, Mapping):
            for manifest_key, item in manifest.items():
                if manifest_key.endswith("_path") and item:
                    paths.add(_normalized_path(root, item))
    if str(job.get("kind")) in {"candidate_oos_fold", "replay", "evidence"}:
        for path in tuple(paths):
            parent = path.parent
            if not parent.is_dir():
                continue
            try:
                paths.update(
                    child for child in parent.iterdir() if child.is_file())
            except OSError:
                pass
    return sorted(paths, key=lambda value: str(value))


def _artifact_stat_hash(job: Mapping[str, Any], root: Path) -> str:
    rows = []
    for path in _artifact_paths(job, root):
        try:
            stat = path.stat()
            rows.append({
                "path": str(path),
                "exists": True,
                "size": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
            })
        except OSError:
            rows.append({"path": str(path), "exists": False})
    return _json_hash(rows)


def _semantic_payload_hash(job: Mapping[str, Any]) -> str:
    stored = job.get("payload_hash")
    if stored:
        return str(stored)
    semantic = {
        key: value for key, value in job.items()
        if key not in _RUNTIME_PAYLOAD_KEYS}
    return stable_hash(semantic)


def _cache_connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path, timeout=30)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA synchronous=NORMAL")
    db.execute(
        "CREATE TABLE IF NOT EXISTS validations ("
        "job_id TEXT PRIMARY KEY, kind TEXT NOT NULL, "
        "payload_hash TEXT NOT NULL, result_hash TEXT NOT NULL, "
        "artifact_stat_hash TEXT NOT NULL, "
        "scientific_contract_hash TEXT NOT NULL, "
        "validator_schema TEXT NOT NULL, details TEXT NOT NULL, "
        "validated_at REAL NOT NULL)")
    db.commit()
    return db


def _iter_complete_jobs(store: ManifestedJobStore) -> Iterable[dict[str, Any]]:
    """Stream COMPLETE rows directly from SQLite with bounded Python memory."""
    # ManifestedJobStore closes its connection in __exit__.  sqlite3's native
    # context manager only commits/rolls back and leaves the Windows file
    # handle open, which blocks checkpoint cleanup after validation.
    with store._connect() as db:
        cursor = db.execute(
            "SELECT job_id,kind,payload FROM jobs "
            "WHERE state='COMPLETE' ORDER BY kind,job_id")
        for job_id, kind, payload in cursor:
            job = json.loads(payload)
            job["job_id"] = str(job_id)
            job["kind"] = str(kind)
            job["state"] = "COMPLETE"
            yield job


def _accepted_run_contract_hashes(
    current: Mapping[str, Any], prior_contract: Mapping[str, Any] | None,
    prior_authority: Mapping[str, Any] | None,
) -> set[str]:
    scientific = _scientific_contract_hash(current)
    accepted = {str(current.get("run_contract_hash", ""))}
    if prior_contract and _scientific_contract_hash(prior_contract) == scientific:
        value = prior_contract.get("run_contract_hash")
        if value:
            accepted.add(str(value))
    if (
        prior_authority
        and str(prior_authority.get("scientific_contract_hash")) == scientific
    ):
        accepted.update(
            str(value)
            for value in prior_authority.get("accepted_run_contract_hashes", ())
            if value)
    accepted.discard("")
    return accepted


def _requeue_stale_root(
    store: ManifestedJobStore, *, job_id: str,
) -> dict[str, int]:
    """Requeue one stale root and only its transitive derived descendants."""
    with store._connect() as db:
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("PRAGMA synchronous=NORMAL")
        db.execute("BEGIN IMMEDIATE")
        db.execute(
            "CREATE TEMP TABLE IF NOT EXISTS "
            "validation_stale_root(job_id TEXT PRIMARY KEY)")
        db.execute("DELETE FROM validation_stale_root")
        db.execute(
            "INSERT OR IGNORE INTO validation_stale_root(job_id) VALUES(?)",
            (str(job_id),),
        )
        affected_cte = (
            "WITH RECURSIVE affected(job_id) AS ("
            " SELECT job_id FROM validation_stale_root"
            " UNION "
            " SELECT d.job_id FROM job_dependencies d "
            " JOIN affected a ON d.depends_on=a.job_id"
            ") ")
        affected_total = int(db.execute(
            affected_cte
            + "SELECT COUNT(*) FROM jobs WHERE job_id IN "
              "(SELECT job_id FROM affected)"
        ).fetchone()[0])
        affected_complete = int(db.execute(
            affected_cte
            + "SELECT COUNT(*) FROM jobs WHERE state='COMPLETE' AND job_id IN "
              "(SELECT job_id FROM affected)"
        ).fetchone()[0])
        db.execute(
            affected_cte
            + "UPDATE jobs SET state='PENDING',"
              "payload=json_remove("
              " json_set(payload,'$.state','PENDING'),"
              " '$.result','$.last_error','$.started_at',"
              " '$.finished_at','$.lease_owner','$.lease_until',"
              " '$.heartbeat'),"
              "lease_owner=NULL,lease_until=NULL,heartbeat=NULL,"
              "last_error=NULL,started_at=NULL,finished_at=NULL "
              "WHERE job_id IN (SELECT job_id FROM affected)"
        )
        db.execute("DELETE FROM validation_stale_root")
        db.execute("COMMIT")
    return {
        "affected_jobs": affected_total,
        "requeued_complete_jobs": affected_complete,
    }


def _purge_noncomplete_validation_cache(
    cache: sqlite3.Connection, store: ManifestedJobStore,
) -> int:
    """Drop cache authority for every job no longer COMPLETE after repair."""
    cache.commit()
    attached = False
    try:
        cache.execute("ATTACH DATABASE ? AS dqbd_jobstate", (str(store.db_path),))
        attached = True
        before = int(cache.total_changes)
        cache.execute(
            "DELETE FROM validations WHERE job_id NOT IN "
            "(SELECT job_id FROM dqbd_jobstate.jobs WHERE state='COMPLETE')")
        removed = int(cache.total_changes) - before
        cache.commit()
        return max(0, removed)
    finally:
        if attached:
            try:
                cache.execute("DETACH DATABASE dqbd_jobstate")
            except sqlite3.Error:
                pass


def _complete_stats(store: ManifestedJobStore) -> tuple[int, dict[str, int]]:
    with store._connect() as db:
        rows = db.execute(
            "SELECT kind,COUNT(*) FROM jobs WHERE state='COMPLETE' "
            "GROUP BY kind ORDER BY kind").fetchall()
    by_kind = {str(kind): int(count) for kind, count in rows}
    return sum(by_kind.values()), by_kind


def audit_complete_results(
    *, store: ManifestedJobStore,
    prior_contract: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate COMPLETE state, automatically repairing stale old checkpoints.

    Graph reconciliation must run first. If validation finds an additional
    self-consistent derived-identity mismatch, that job plus only its causal
    descendants are requeued and the audit restarts. Cache hits make repeated
    passes cheap; corrupt immutable content still aborts immediately.
    """
    contract_path = store.root / "manifested-job-contract.json"
    registry_path = store.root / "family-registry.json"
    contract = _load_json(
        contract_path, job_id="<seed>", label="MANIFESTED_JOB_CONTRACT")
    registry = _load_json(
        registry_path, job_id="<seed>", label="FAMILY_REGISTRY")
    authority_path = complete_result_validation_authority_path(store.root)
    prior_authority = None
    if authority_path.is_file():
        try:
            candidate = json.loads(authority_path.read_text(encoding="utf-8"))
            if isinstance(candidate, dict):
                prior_authority = candidate
        except (OSError, ValueError, TypeError):
            prior_authority = None

    scientific = _scientific_contract_hash(contract)
    accepted_run_hashes = _accepted_run_contract_hashes(
        contract, prior_contract, prior_authority)
    ctx = _ValidationContext(
        store=store,
        contract=contract,
        registry=registry,
        scientific_contract_hash=scientific,
        accepted_run_contract_hashes=frozenset(accepted_run_hashes),
    )
    cache = _cache_connect(store.root / "run-state" / CACHE_FILENAME)
    cache_hits = 0
    full_validations = 0
    scan_attempts = 0
    stale_events: list[dict[str, Any]] = []
    requeued_complete_jobs = 0
    requeued_total_jobs = 0
    purged_cache_rows = 0
    started = time.monotonic()
    try:
        while True:
            stale: CompleteResultStaleError | None = None
            try:
                for job in _iter_complete_jobs(store):
                    scan_attempts += 1
                    job_id = str(job["job_id"])
                    payload_hash = _semantic_payload_hash(job)
                    result_hash = _json_hash(job.get("result"))
                    stat_hash = _artifact_stat_hash(job, store.root)
                    row = cache.execute(
                        "SELECT payload_hash,result_hash,artifact_stat_hash,"
                        "scientific_contract_hash,validator_schema "
                        "FROM validations WHERE job_id=?",
                        (job_id,),
                    ).fetchone()
                    if row == (
                        payload_hash,
                        result_hash,
                        stat_hash,
                        scientific,
                        COMPLETE_RESULT_VALIDATION_SCHEMA,
                    ):
                        cache_hits += 1
                        continue
                    try:
                        details = _validate_complete_job(job, ctx)
                    except CompleteResultStaleError as exc:
                        stale = exc
                        cache.execute(
                            "DELETE FROM validations WHERE job_id=?", (job_id,))
                        break
                    full_validations += 1
                    cache.execute(
                        "INSERT INTO validations("
                        "job_id,kind,payload_hash,result_hash,artifact_stat_hash,"
                        "scientific_contract_hash,validator_schema,details,"
                        "validated_at) VALUES(?,?,?,?,?,?,?,?,?) "
                        "ON CONFLICT(job_id) DO UPDATE SET "
                        "kind=excluded.kind,payload_hash=excluded.payload_hash,"
                        "result_hash=excluded.result_hash,"
                        "artifact_stat_hash=excluded.artifact_stat_hash,"
                        "scientific_contract_hash="
                        "excluded.scientific_contract_hash,"
                        "validator_schema=excluded.validator_schema,"
                        "details=excluded.details,"
                        "validated_at=excluded.validated_at",
                        (
                            job_id,
                            str(job.get("kind", "")),
                            payload_hash,
                            result_hash,
                            stat_hash,
                            scientific,
                            COMPLETE_RESULT_VALIDATION_SCHEMA,
                            json.dumps(
                                details, sort_keys=True, default=str),
                            time.time(),
                        ),
                    )
                    if full_validations % 128 == 0:
                        cache.commit()
                cache.commit()
            except CompleteResultValidationError:
                raise

            if stale is None:
                break
            repair = _requeue_stale_root(store, job_id=stale.job_id)
            purged_cache_rows += _purge_noncomplete_validation_cache(
                cache, store)
            requeued_complete_jobs += int(
                repair["requeued_complete_jobs"])
            requeued_total_jobs += int(repair["affected_jobs"])
            stale_events.append({
                "job_id": stale.job_id,
                "reason": stale.reason,
                **repair,
            })
            # The next pass sees the post-requeue COMPLETE frontier. Previously
            # validated independent rows become cache hits; descendants are no
            # longer visible to validation and cannot fail merely because their
            # parent was already classified stale.
    except CompleteResultValidationError as exc:
        cache.rollback()
        blocked = {
            "schema_version": COMPLETE_RESULT_VALIDATION_SCHEMA,
            "cache_schema_version": COMPLETE_RESULT_VALIDATION_CACHE_SCHEMA,
            "status": "CORRUPT_COMPLETE_RESULT_BLOCKED",
            "scientific_contract_hash": scientific,
            "manifested_job_contract_sha256": sha256_file(contract_path),
            "job_store_state_fingerprint": _job_store_state_fingerprint(
                store.root),
            "git_sha": contract.get("git_sha"),
            "input_stat_fingerprint": contract.get("input_stat_fingerprint"),
            "accepted_run_contract_hashes": sorted(accepted_run_hashes),
            "scan_attempts_before_failure": scan_attempts,
            "validation_cache_hits": cache_hits,
            "full_artifact_validations": full_validations,
            "stale_roots_requeued_before_failure": len(stale_events),
            "stale_requeues": stale_events,
            "purged_stale_validation_cache_rows": purged_cache_rows,
            "corrupt_job_id": exc.job_id,
            "corrupt_reason": exc.reason,
            "evaluation_opened": False,
            "holdout_opened": False,
        }
        _write_json(authority_path, blocked)
        raise
    finally:
        cache.close()

    if stale_events:
        # SQLite is authoritative. Materialize the human-inspection mirror only
        # once after all stale roots have been repaired, not once per root.
        store.materialize_json_manifest()
        store.materialize_progress()
    complete_count, kinds = _complete_stats(store)
    authority = {
        "schema_version": COMPLETE_RESULT_VALIDATION_SCHEMA,
        "cache_schema_version": COMPLETE_RESULT_VALIDATION_CACHE_SCHEMA,
        "status": (
            "VALIDATED_COMPLETE_FRONTIER_STALE_REQUEUED"
            if stale_events else "VALIDATED_COMPLETE_FRONTIER"),
        "scientific_contract_hash": scientific,
        "manifested_job_contract_sha256": sha256_file(contract_path),
        "job_store_state_fingerprint": _job_store_state_fingerprint(store.root),
        "git_sha": contract.get("git_sha"),
        "input_stat_fingerprint": contract.get("input_stat_fingerprint"),
        "accepted_run_contract_hashes": sorted(accepted_run_hashes),
        "complete_job_count": complete_count,
        "complete_jobs_by_kind": kinds,
        "scan_attempts": scan_attempts,
        "validation_cache_hits": cache_hits,
        "full_artifact_validations": full_validations,
        "stale_roots_requeued": len(stale_events),
        "requeued_complete_jobs": requeued_complete_jobs,
        "requeued_total_jobs": requeued_total_jobs,
        "purged_stale_validation_cache_rows": purged_cache_rows,
        "stale_requeues": stale_events,
        "validation_seconds": time.monotonic() - started,
        "corrupt_complete_jobs": 0,
        "evaluation_opened": False,
        "holdout_opened": False,
    }
    authority["authority_hash"] = stable_hash(authority)
    _write_json(authority_path, authority)
    return authority
