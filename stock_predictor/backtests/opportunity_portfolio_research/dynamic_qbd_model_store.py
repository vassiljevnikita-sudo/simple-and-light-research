"""Append-only, development-only store for immutable Dynamic-QBD generations."""
from __future__ import annotations

from datetime import date
import json
from pathlib import Path
from typing import Iterable

from .dynamic_qbd_orchestrator_contract import DEVELOPMENT_END, ModelGenerationRecord


class ModelStore:
    """Causal generation visibility boundary.

    The store is intentionally policy-free: it records generations and answers
    point-in-time visibility queries.  Selection remains a separate layer.
    """

    schema_version = "DQBD_MODEL_STORE_V1"

    def __init__(self, generations: Iterable[ModelGenerationRecord] = ()) -> None:
        self._generations: dict[str, ModelGenerationRecord] = {}
        for generation in generations:
            self.add_generation(generation)

    def add_generation(self, generation: ModelGenerationRecord) -> ModelGenerationRecord:
        if not isinstance(generation, ModelGenerationRecord):
            raise TypeError("MODEL_STORE_REQUIRES_MODEL_GENERATION_RECORD")
        existing = self._generations.get(generation.generation_id)
        if existing is not None and existing != generation:
            raise ValueError("MODEL_GENERATION_IMMUTABLE")
        self._generations[generation.generation_id] = generation
        return generation

    def get_generation(self, generation_id: str) -> ModelGenerationRecord:
        try:
            return self._generations[str(generation_id)]
        except KeyError as exc:
            raise KeyError(f"MODEL_GENERATION_NOT_FOUND:{generation_id}") from exc

    @staticmethod
    def _as_of(value: date | str) -> date:
        point = value if isinstance(value, date) else date.fromisoformat(str(value))
        if point > DEVELOPMENT_END:
            raise ValueError("MODEL_STORE_PROSPECTIVE_HOLDOUT_CLOSED")
        return point

    def generations_as_of(self, t: date | str) -> tuple[ModelGenerationRecord, ...]:
        point = self._as_of(t)
        return tuple(sorted(
            (record for record in self._generations.values() if record.created_at <= point),
            key=lambda record: (record.created_at, record.generation_id),
        ))

    def generations_for_recipe(self, recipe_id: str, as_of: date | str) -> tuple[ModelGenerationRecord, ...]:
        return tuple(record for record in self.generations_as_of(as_of) if record.recipe_id == str(recipe_id))

    def generations_for_recipe_and_horizon(self, recipe_id: str, horizon: int,
                                           as_of: date | str) -> tuple[ModelGenerationRecord, ...]:
        return tuple(record for record in self.generations_for_recipe(recipe_id, as_of)
                     if record.horizon == int(horizon))

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "generations": [record.to_dict() for record in self.generations_as_of(DEVELOPMENT_END)],
        }

    def write(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(destination)

    @classmethod
    def read(cls, path: str | Path) -> "ModelStore":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("schema_version") != cls.schema_version:
            raise ValueError("MODEL_STORE_SCHEMA_MISMATCH")
        return cls(ModelGenerationRecord.from_dict(value) for value in payload.get("generations", ()))

    def __len__(self) -> int:
        return len(self._generations)
