"""Algorithm-level final-holdout freeze manifest."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from .contract_fingerprints import stable_hash


REQUIRED_ALGORITHM_KEYS = {
    "code_commit", "family_registry_hash", "candidate_design_space", "feature_schema",
    "training_recipe", "hyperparameter_rule", "training_window_rule", "refit_cadence",
    "horizon_maturity_rule", "calibration_window_rule", "recalibration_algorithm", "threshold_rule",
    "exit_runtime_rule", "benchmark_contract", "cost_contract", "tax_contract", "evidence_panel_schema",
    "gate_rules", "final_holdout_boundaries", "initial_pre_holdout_state_hash",
}


def build_algorithm_manifest(**algorithm) -> dict:
    missing = REQUIRED_ALGORITHM_KEYS - set(algorithm)
    if missing:
        raise ValueError(f"DYNAMIC_FREEZE_FIELDS_MISSING:{sorted(missing)}")
    if "concrete_future_generation_hashes" in algorithm:
        raise ValueError("FUTURE_GENERATIONS_MUST_NOT_BE_FROZEN")
    payload = {"schema_version": "DYNAMIC_QBD_ALGORITHM_FREEZE_V1", "algorithm": algorithm,
               "future_generations": "CAUSALLY_CREATED_APPEND_ONLY_INSIDE_HOLDOUT"}
    payload["manifest_hash"] = stable_hash(payload)
    return payload


def write_algorithm_manifest(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, default=str, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def validate_algorithm_manifest(payload: dict) -> None:
    supplied = payload.get("manifest_hash")
    body = dict(payload)
    body.pop("manifest_hash", None)
    if supplied != stable_hash(body):
        raise ValueError("DYNAMIC_FREEZE_HASH_MISMATCH")
    missing = REQUIRED_ALGORITHM_KEYS - set(payload.get("algorithm", {}))
    if missing:
        raise ValueError("DYNAMIC_FREEZE_INCOMPLETE")
