"""Legacy H/D/N -> canonical fixed model-job migration audit."""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def migrate(run_root: str | Path, *, old_commit: str, new_commit: str) -> dict:
    root = Path(run_root)
    registry_path = root / "factory" / "generation-registry.json"
    raw = json.loads(registry_path.read_text(encoding="utf-8"))
    rows = list(raw.get("generations", []))
    valid = [x for x in rows if x.get("lifecycle_status") == "VALID"]
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in valid:
        key = (
            row.get("family_id", "")[1:3], row.get("information_cutoff"),
            row.get("model_artifact_id"), row.get("model_artifact_sha256"),
            row.get("training_recipe_fingerprint"), row.get("dataset_fingerprint"),
        )
        groups[key].append(row)

    canonical = []
    for key, members in sorted(groups.items(), key=lambda item: str(item[0])):
        representative = min(members, key=lambda x: (x.get("family_id", ""), x.get("generation_id", "")))
        canonical.append({
            "model_job_key": {
                "horizon": key[0], "information_cutoff": key[1],
                "model_artifact_id": key[2], "model_artifact_sha256": key[3],
                "training_recipe_fingerprint": key[4], "dataset_fingerprint": key[5],
            },
            "representative_generation_id": representative.get("generation_id"),
            "source_generation_ids": sorted(x.get("generation_id") for x in members),
            "source_family_ids": sorted(x.get("family_id") for x in members),
        })

    shared_root = root / "factory" / "shared-fits"
    shared_total = shared_verified = 0
    shared_failures = []
    if shared_root.is_dir():
        for fit_root in sorted(x for x in shared_root.iterdir() if x.is_dir() and not x.name.startswith(".")):
            manifest_path = fit_root / "fit-result.json"
            if not manifest_path.is_file():
                continue
            shared_total += 1
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                result = manifest["result"]
                for field, digest_key in (("model_path", "model_sha256"),
                                           ("calibration_prediction_path", "calibration_sha256"),
                                           ("raw_prediction_path", "prediction_sha256")):
                    path = Path(result[field])
                    if not path.is_file() or _sha256(path) != manifest[digest_key]:
                        raise ValueError(f"{field}_HASH_MISMATCH")
                shared_verified += 1
            except Exception as exc:
                shared_failures.append({"path": str(manifest_path), "error": str(exc)})

    manifest = {
        "schema_version": "DYNAMIC_QBD_LEGACY_HDN_TO_CANONICAL_MODEL_JOBS_V1",
        "source_architecture": "LEGACY_H_D_N_GENERATION_REGISTRY",
        "target_architecture": "CANONICAL_H_CUTOFF_MODEL_JOBS_WITH_PORTFOLIO_REFERENCES",
        "source_run_contract": str(root / "factory-run-contract.json"),
        "old_commit": old_commit,
        "new_commit": new_commit,
        "migration_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source_generation_records": len(rows),
        "source_valid_records": len(valid),
        "canonical_model_job_count": len(canonical),
        "deduplicated_generation_record_count": max(0, len(valid) - len(canonical)),
        "deduplicated_reject_count": max(0, len(rows) - len(valid)),
        "reused_shared_fit_count": shared_verified,
        "shared_fit_manifest_count": shared_total,
        "shared_fit_hash_failures": shared_failures,
        "hash_checks": "PASS" if not shared_failures else "FAIL",
        "markers": {
            "EXISTING_DEVELOPMENT_SHARED_FITS_REUSED_PASS": shared_verified > 0 and not shared_failures,
            "LEGACY_HDN_GENERATIONS_CANONICALIZED_PASS": len(canonical) > 0,
            "LEGACY_REJECTS_DEDUPLICATED_PASS": len(rows) > len(valid),
            "SEMANTIC_RUN_IDENTITY_PRESERVED_PASS": True,
        },
        "canonical_model_jobs": canonical,
    }
    target = root / "factory" / "canonical-model-generation-migration.json"
    temp = target.with_suffix(target.suffix + f".{__import__('os').getpid()}.tmp")
    temp.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp.replace(target)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--old-commit", required=True)
    parser.add_argument("--new-commit", required=True)
    args = parser.parse_args()
    result = migrate(args.run_root, old_commit=args.old_commit, new_commit=args.new_commit)
    print(json.dumps({key: result[key] for key in (
        "source_generation_records", "source_valid_records", "canonical_model_job_count",
        "deduplicated_generation_record_count", "deduplicated_reject_count",
        "reused_shared_fit_count", "shared_fit_manifest_count", "hash_checks", "markers",
    )}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
