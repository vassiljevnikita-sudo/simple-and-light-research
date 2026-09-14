"""Build a bounded historical ModelStore/EvidenceCursor pair for Development."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Iterable

from .dynamic_qbd_evidence_store_cursor import EvidenceCursor, EvidenceRecord
from .dynamic_qbd_model_store import ModelStore
from .dynamic_qbd_orchestrator_contract import ModelGenerationRecord
from .dynamic_qbd_recipe_store import RecipeStore


@dataclass(frozen=True)
class HistoricalStoreBundle:
    model_store: ModelStore
    evidence_cursor: EvidenceCursor


@dataclass(frozen=True)
class InitialStorePlan:
    """Causal G0 slots; fitting/artifact publication is supplied by a provider."""

    seed: date
    recipe_ids: tuple[str, ...]
    horizons: tuple[int, ...]

    @property
    def slots(self) -> tuple[tuple[str, int], ...]:
        return tuple((recipe_id, horizon) for recipe_id in self.recipe_ids for horizon in self.horizons)


def plan_initial_store(*, recipe_store: RecipeStore, seed: date | str,
                       horizons: Iterable[int] = range(1, 31)) -> InitialStorePlan:
    point = seed if isinstance(seed, date) else date.fromisoformat(str(seed))
    selected_horizons = tuple(sorted({int(horizon) for horizon in horizons}))
    if not selected_horizons or any(not 1 <= horizon <= 30 for horizon in selected_horizons):
        raise ValueError("DQBD_INITIAL_STORE_HORIZONS_INVALID")
    recipes = tuple(recipe.recipe_id for recipe in recipe_store.eligible_as_of(point)
                    if set(selected_horizons) & set(recipe.horizons))
    return InitialStorePlan(point, recipes, selected_horizons)


def build_historical_store(*, generations: Iterable[ModelGenerationRecord], evidence: Iterable[EvidenceRecord]) -> HistoricalStoreBundle:
    """Materialize only the closed Development history; no prospective rows are accepted."""
    return HistoricalStoreBundle(ModelStore(generations), EvidenceCursor(evidence))
