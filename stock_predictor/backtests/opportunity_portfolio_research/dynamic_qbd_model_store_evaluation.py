"""Non-economic visibility diagnostics for the causal model-store experiment."""
from __future__ import annotations

from datetime import date
from typing import Iterable

from .dynamic_qbd_evidence_store_cursor import EvidenceCursor
from .dynamic_qbd_model_store import ModelStore


def evaluate_visibility(*, model_store: ModelStore, evidence: EvidenceCursor,
                        decision_times: Iterable[date | str]) -> tuple[dict, ...]:
    """Return deterministic as-of diagnostics without fitting or replaying portfolios."""
    rows = []
    for raw_time in decision_times:
        point = raw_time if isinstance(raw_time, date) else date.fromisoformat(str(raw_time))
        generations = model_store.generations_as_of(point)
        rows.append({
            "decision_time": point.isoformat(),
            "generation_ids": [record.generation_id for record in generations],
            "evidence_fingerprint": evidence.fingerprint_as_of(point),
        })
    return tuple(rows)
