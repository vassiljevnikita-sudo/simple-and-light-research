"""Canonical model-job identities for the Dynamic-QBD factory.

Fixed-exit model fitting is keyed by causal model inputs only. Portfolio
holding period and max-name capacity remain downstream replay dimensions.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Iterable

from .contract_fingerprints import stable_hash


@dataclass(frozen=True, order=True)
class ModelJobKey:
    horizon_sessions: int
    information_cutoff: date
    feature_schema_sha256: str
    recipe_fingerprint: str
    training_window_sessions: int
    calibration_window_sessions: int
    purge_sessions: int
    random_seed: int
    data_identity: str

    @property
    def job_id(self) -> str:
        return stable_hash(self)[:24]


def fixed_model_job_key(family, information_cutoff: date, *, data_identity: str) -> ModelJobKey:
    recipe = {
        "model_family": family.model_family,
        "training_recipe": dict(family.training_recipe),
        "hyperparameter_rule": dict(family.hyperparameter_rule),
    }
    return ModelJobKey(
        horizon_sessions=int(family.horizon_sessions),
        information_cutoff=information_cutoff,
        feature_schema_sha256=str(family.feature_schema_sha256),
        recipe_fingerprint=stable_hash(recipe),
        training_window_sessions=int(family.training_window_sessions),
        calibration_window_sessions=int(family.calibration_window_sessions),
        purge_sessions=int(family.training_recipe.get("purge_sessions", 30)),
        random_seed=int(family.random_seed),
        data_identity=str(data_identity),
    )


def fixed_portfolio_groups(families: Iterable) -> dict[ModelJobKey, tuple]:
    """Group fixed-exit portfolio specs by their model-job identity.

    ``information_cutoff`` is intentionally omitted here because the caller
    supplies it when constructing a queue. D and N therefore cannot split a
    fixed model job.
    """
    groups: dict[tuple, list] = {}
    for family in families:
        if str(family.exit_policy.get("family", "FIXED")).upper() != "FIXED":
            continue
        key = (
            int(family.horizon_sessions), str(family.feature_schema_sha256),
            str(family.model_family), stable_hash(dict(family.training_recipe)),
            stable_hash(dict(family.hyperparameter_rule)), int(family.training_window_sessions),
            int(family.calibration_window_sessions), int(family.random_seed),
        )
        groups.setdefault(key, []).append(family)
    return {key: tuple(value) for key, value in groups.items()}


def unique_fixed_model_job_count(families: Iterable, cutoffs_by_horizon: dict[int, Iterable[date]]) -> int:
    groups = fixed_portfolio_groups(families)
    return sum(len(tuple(cutoffs_by_horizon.get(int(key[0]), ()))) for key in groups)


def canonical_queue_sort_key(payload: dict) -> tuple:
    """Stable queue order independent of family/D/N input ordering."""
    family = payload["family"]
    return (
        int(family.horizon_sessions),
        str(payload.get("cutoff", "")),
        str(family.feature_schema_sha256),
        str(family.model_family),
        stable_hash(dict(family.training_recipe)),
        stable_hash(dict(family.hyperparameter_rule)),
    )
