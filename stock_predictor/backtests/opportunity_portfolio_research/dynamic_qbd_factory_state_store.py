"""Atomic restart state for the Dynamic-QBD factory and replay."""
from __future__ import annotations

import json
import os
from dataclasses import asdict
from datetime import date
from pathlib import Path

from .contract_fingerprints import stable_hash
from .dynamic_qbd_generation_contracts import DynamicFactoryState, PositionLineage


class DynamicFactoryStateStore:
    def __init__(self, path: str | Path, *, family_registry_hash: str):
        self.path = Path(path)
        self.family_registry_hash = family_registry_hash

    def save_atomic(self, state: DynamicFactoryState) -> None:
        if state.family_registry_hash != self.family_registry_hash:
            raise ValueError("FACTORY_STATE_FAMILY_HASH_MISMATCH")
        payload = asdict(state)
        payload["state_hash"] = stable_hash(payload)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.path.with_name(self.path.name + f".{os.getpid()}.tmp")
        temp.write_text(json.dumps(payload, default=str, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        os.replace(temp, self.path)

    def load(self) -> DynamicFactoryState | None:
        if not self.path.exists():
            return None
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        supplied = raw.pop("state_hash")
        if supplied != stable_hash(raw):
            raise ValueError("FACTORY_STATE_HASH_MISMATCH")
        if raw["family_registry_hash"] != self.family_registry_hash:
            raise ValueError("FACTORY_STATE_FAMILY_HASH_MISMATCH")
        raw["as_of"] = date.fromisoformat(raw["as_of"])
        raw["open_position_lineage"] = {key: PositionLineage(**value) for key, value in raw.get("open_position_lineage", {}).items()}
        return DynamicFactoryState(**raw)
