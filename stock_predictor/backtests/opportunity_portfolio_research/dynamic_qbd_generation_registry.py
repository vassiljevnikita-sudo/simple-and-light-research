"""Append-only family/generation registry with fail-safe activation."""
from __future__ import annotations

import json
import os
from dataclasses import asdict
from datetime import date
from pathlib import Path
import time

from .contract_fingerprints import stable_hash
from .dynamic_qbd_generation_contracts import ExpertFamilySpec, GenerationStatus, ModelGeneration


class GenerationRegistry:
    def __init__(self, path: str | Path, families: tuple[ExpertFamilySpec, ...], *, autosave: bool = True):
        self.path = Path(path)
        self.autosave = bool(autosave)
        self.families = {x.family_id: x for x in families}
        if len(self.families) != len(families):
            raise ValueError("DUPLICATE_FAMILY_ID")
        self.records: dict[str, ModelGeneration] = {}
        self.current: dict[str, str] = {}
        self._family_cutoff_index: dict[tuple[str, date], str] = {}
        if self.path.exists():
            self._load()

    @property
    def family_registry_hash(self) -> str:
        return stable_hash(tuple(asdict(self.families[x]) for x in sorted(self.families)))

    def _load(self) -> None:
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        if raw["family_registry_hash"] != self.family_registry_hash:
            raise ValueError("FAMILY_REGISTRY_HASH_MISMATCH")
        self.records.clear()
        self._family_cutoff_index.clear()
        for item in raw.get("generations", []):
            item = dict(item)
            from datetime import date, datetime
            item["refit_timestamp"] = datetime.fromisoformat(item["refit_timestamp"])
            for key in ("information_cutoff", "latest_matured_label_cutoff", "train_start", "train_end", "calibration_start", "calibration_end", "activation_date"):
                item[key] = date.fromisoformat(item[key]) if item.get(key) else None
            item["selection_oos_fold_ids"] = tuple(item.get("selection_oos_fold_ids", ()))
            item["lifecycle_status"] = GenerationStatus(item["lifecycle_status"])
            generation = ModelGeneration(**item)
            self.records[generation.generation_id] = generation
            self._family_cutoff_index[(generation.family_id, generation.information_cutoff)] = generation.generation_id
        self.current = dict(raw.get("current_generation_by_family", {}))
        self._validate_current()

    def _validate_current(self) -> None:
        for family_id, generation_id in self.current.items():
            generation = self.records.get(generation_id)
            if generation is None or generation.family_id != family_id or generation.lifecycle_status != GenerationStatus.VALID:
                raise ValueError("INVALID_CURRENT_GENERATION")

    def register(self, generation: ModelGeneration) -> None:
        if not self.autosave:
            self._register_in_memory(generation)
            return
        lock = self.path.with_name(self.path.name + ".lock")
        acquired = False
        for _ in range(1200):
            try:
                lock.mkdir(parents=False, exist_ok=False)
                acquired = True
                break
            except FileExistsError:
                time.sleep(.05)
        if not acquired:
            raise TimeoutError("GENERATION_REGISTRY_LOCK_TIMEOUT")
        try:
            # Reload under the lock so concurrent horizon lanes merge instead
            # of overwriting each other's append-only records.
            if self.path.exists():
                self._load()
            self._register_in_memory(generation)
            self._validate_current()
            self.save()
        finally:
            try:
                lock.rmdir()
            except FileNotFoundError:
                pass

    def _register_in_memory(self, generation: ModelGeneration) -> None:
        if generation.family_id not in self.families:
            raise ValueError("UNKNOWN_GENERATION_FAMILY")
        family = self.families[generation.family_id]
        if generation.feature_schema_sha256 != family.feature_schema_sha256 or generation.random_seed != family.random_seed:
            raise ValueError("GENERATION_DOES_NOT_MATCH_FAMILY_RECIPE")
        if generation.activation_date is not None and generation.activation_date < generation.information_cutoff:
            raise ValueError("GENERATION_ACTIVATED_BEFORE_INFORMATION_CUTOFF")
        prior = self.records.get(generation.generation_id)
        if prior is not None and prior != generation:
            raise ValueError("GENERATION_ID_COLLISION")
        if prior is None:
            self.records[generation.generation_id] = generation
            self._family_cutoff_index[(generation.family_id, generation.information_cutoff)] = generation.generation_id
        if generation.lifecycle_status == GenerationStatus.VALID:
            self.current[generation.family_id] = generation.generation_id

    def current_generation(self, family_id: str) -> ModelGeneration | None:
        generation_id = self.current.get(family_id)
        return self.records.get(generation_id) if generation_id else None

    def generation_for_family_cutoff(self, family_id: str, information_cutoff: date) -> ModelGeneration | None:
        generation_id = self._family_cutoff_index.get((family_id, information_cutoff))
        return self.records.get(generation_id) if generation_id else None

    def checkpoint(self) -> None:
        """Persist a single-writer shard without reloading it from disk."""
        self._validate_current()
        self.save()

    def save(self) -> None:
        payload = {
            "schema_version": "DYNAMIC_QBD_GENERATION_REGISTRY_V1",
            "family_registry_hash": self.family_registry_hash,
            "families": [asdict(self.families[x]) for x in sorted(self.families)],
            "generations": [asdict(self.records[x]) for x in sorted(self.records)],
            "current_generation_by_family": dict(sorted(self.current.items())),
        }
        payload["registry_hash"] = stable_hash(payload)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.path.with_name(self.path.name + f".{os.getpid()}.tmp")
        temp.write_text(json.dumps(payload, default=str, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        # Windows readers (including the live progress tracker) can briefly
        # hold the destination open.  Keep the atomic replace, but tolerate
        # that short read-side sharing window instead of killing a lane.
        last_error = None
        for attempt in range(80):
            try:
                os.replace(temp, self.path)
                last_error = None
                break
            except PermissionError as exc:
                last_error = exc
                time.sleep(min(0.05 * (attempt + 1), 0.5))
        if last_error is not None:
            raise last_error
