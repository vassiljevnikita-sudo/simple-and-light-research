"""Structural recipe registry and seed-time eligibility for Dynamic-QBD V1."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Iterable, Mapping

from .contract_fingerprints import stable_hash
from .dynamic_qbd_orchestrator_contract import DEVELOPMENT_END


@dataclass(frozen=True)
class RecipeRecord:
    recipe_id: str
    family: str
    hyperparameters: Mapping[str, Any]
    horizons: tuple[int, ...] = tuple(range(1, 31))

    def __post_init__(self) -> None:
        if not self.recipe_id or self.family not in {"RIDGE_LOGISTIC", "HIST_GRADIENT_BOOSTING"}:
            raise ValueError("DQBD_RECIPE_RECORD_INVALID")
        if not self.horizons or any(not 1 <= int(horizon) <= 30 for horizon in self.horizons):
            raise ValueError("DQBD_RECIPE_HORIZONS_INVALID")

    @property
    def fingerprint(self) -> str:
        return stable_hash({
            "recipe_id": self.recipe_id,
            "family": self.family,
            "hyperparameters": dict(self.hyperparameters),
            "horizons": list(self.horizons),
        })


@dataclass(frozen=True)
class RecipeSeedEligibility:
    recipe_id: str
    seed: date
    eligible: bool
    reason: str
    required_features_available: bool
    minimum_training_sessions_met: bool
    minimum_matured_target_sessions_met: bool
    required_folds_available: bool
    calibration_feasible: bool
    finite_model_fit: bool
    identity_and_provenance_complete: bool

    def __post_init__(self) -> None:
        point = self.seed if isinstance(self.seed, date) else date.fromisoformat(str(self.seed))
        object.__setattr__(self, "seed", point)
        if point > DEVELOPMENT_END:
            raise ValueError("DQBD_RECIPE_ELIGIBILITY_PROSPECTIVE_HOLDOUT_CLOSED")
        if self.eligible != all((self.required_features_available,
                                 self.minimum_training_sessions_met,
                                 self.minimum_matured_target_sessions_met,
                                 self.required_folds_available,
                                 self.calibration_feasible,
                                 self.finite_model_fit,
                                 self.identity_and_provenance_complete)):
            raise ValueError("DQBD_RECIPE_ELIGIBILITY_FLAG_MISMATCH")

    @property
    def fingerprint(self) -> str:
        return stable_hash(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "recipe_id": self.recipe_id,
            "seed": self.seed.isoformat(),
            "eligible": self.eligible,
            "reason": self.reason,
            "required_features_available": self.required_features_available,
            "minimum_training_sessions_met": self.minimum_training_sessions_met,
            "minimum_matured_target_sessions_met": self.minimum_matured_target_sessions_met,
            "required_folds_available": self.required_folds_available,
            "calibration_feasible": self.calibration_feasible,
            "finite_model_fit": self.finite_model_fit,
            "identity_and_provenance_complete": self.identity_and_provenance_complete,
        }


@dataclass(frozen=True)
class RecipeSeedInputs:
    """Only structural facts allowed into seed eligibility; no performance fields."""

    required_features_available: bool
    matured_training_sessions: int
    matured_target_sessions: int
    matured_fold_count: int
    calibration_feasible: bool
    finite_model_fit: bool
    identity_and_provenance_complete: bool


def recipe_seed_eligibility(recipe: RecipeRecord, seed: date, inputs: RecipeSeedInputs) -> RecipeSeedEligibility:
    checks = {
        "required_features_available": bool(inputs.required_features_available),
        "minimum_training_sessions_met": int(inputs.matured_training_sessions) >= 504,
        "minimum_matured_target_sessions_met": int(inputs.matured_target_sessions) >= 252,
        "required_folds_available": int(inputs.matured_fold_count) >= 2,
        "calibration_feasible": bool(inputs.calibration_feasible),
        "finite_model_fit": bool(inputs.finite_model_fit),
        "identity_and_provenance_complete": bool(inputs.identity_and_provenance_complete),
    }
    eligible = all(checks.values())
    reason = "STRUCTURAL_ELIGIBILITY_PASS" if eligible else "TECHNICAL_OR_CAUSAL_UNTRAINABILITY"
    return RecipeSeedEligibility(recipe.recipe_id, seed, eligible, reason, **checks)


class RecipeStore:
    """Append-only recipe definitions with explicit, seed-scoped eligibility."""

    def __init__(self, recipes: Iterable[RecipeRecord] = ()) -> None:
        self._recipes: dict[str, RecipeRecord] = {}
        self._eligibility: dict[tuple[str, date], RecipeSeedEligibility] = {}
        for recipe in recipes:
            self.add_recipe(recipe)

    def add_recipe(self, recipe: RecipeRecord) -> RecipeRecord:
        existing = self._recipes.get(recipe.recipe_id)
        if existing is not None and existing != recipe:
            raise ValueError("DQBD_RECIPE_IMMUTABLE")
        self._recipes[recipe.recipe_id] = recipe
        return recipe

    def get_recipe(self, recipe_id: str) -> RecipeRecord:
        try:
            return self._recipes[str(recipe_id)]
        except KeyError as exc:
            raise KeyError(f"DQBD_RECIPE_NOT_FOUND:{recipe_id}") from exc

    def record_eligibility(self, result: RecipeSeedEligibility) -> RecipeSeedEligibility:
        key = (result.recipe_id, result.seed)
        if result.recipe_id not in self._recipes:
            raise KeyError(f"DQBD_RECIPE_NOT_FOUND:{result.recipe_id}")
        existing = self._eligibility.get(key)
        if existing is not None and existing != result:
            raise ValueError("DQBD_RECIPE_ELIGIBILITY_IMMUTABLE")
        self._eligibility[key] = result
        return result

    def eligible_as_of(self, seed: date | str) -> tuple[RecipeRecord, ...]:
        point = seed if isinstance(seed, date) else date.fromisoformat(str(seed))
        if point > DEVELOPMENT_END:
            raise ValueError("DQBD_RECIPE_STORE_PROSPECTIVE_HOLDOUT_CLOSED")
        return tuple(self._recipes[recipe_id] for (recipe_id, eligibility_seed), result in sorted(self._eligibility.items())
                     if eligibility_seed == point and result.eligible)

    def eligible_for_horizon_as_of(self, seed: date | str, horizon: int) -> tuple[RecipeRecord, ...]:
        return tuple(recipe for recipe in self.eligible_as_of(seed) if int(horizon) in recipe.horizons)

    def eligibility_as_of(self, seed: date | str) -> tuple[RecipeSeedEligibility, ...]:
        point = seed if isinstance(seed, date) else date.fromisoformat(str(seed))
        return tuple(result for (recipe_id, eligibility_seed), result in sorted(self._eligibility.items())
                     if eligibility_seed == point)

    def __len__(self) -> int:
        return len(self._recipes)
