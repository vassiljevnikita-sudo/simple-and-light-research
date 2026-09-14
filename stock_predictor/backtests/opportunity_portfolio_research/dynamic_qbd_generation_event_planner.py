"""Causal generation-event planning without fitting or selecting models."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Iterable

from .contract_fingerprints import stable_hash
from .dynamic_qbd_evidence_store_cursor import EvidenceCursor
from .dynamic_qbd_model_store import ModelStore
from .dynamic_qbd_orchestrator_contract import DEVELOPMENT_END, GenerationTrigger


@dataclass(frozen=True)
class GenerationEvent:
    event_id: str
    event_time: date
    generation_id: str
    recipe_id: str
    trigger: GenerationTrigger
    evidence_fingerprint: str

    def to_dict(self) -> dict[str, str]:
        return {
            "event_id": self.event_id,
            "event_time": self.event_time.isoformat(),
            "generation_id": self.generation_id,
            "recipe_id": self.recipe_id,
            "trigger": self.trigger,
            "evidence_fingerprint": self.evidence_fingerprint,
        }


class GenerationEventPlanner:
    """Plan visible append-only events; it has no economic decision authority."""

    def __init__(self, model_store: ModelStore, evidence: EvidenceCursor) -> None:
        self.model_store = model_store
        self.evidence = evidence

    def plan(self, decision_times: Iterable[date | str]) -> tuple[GenerationEvent, ...]:
        events: dict[str, GenerationEvent] = {}
        for raw_time in sorted({
            raw_time if isinstance(raw_time, date) else date.fromisoformat(str(raw_time))
            for raw_time in decision_times
        }):
            point = raw_time if isinstance(raw_time, date) else date.fromisoformat(str(raw_time))
            if point > DEVELOPMENT_END:
                raise ValueError("GENERATION_EVENT_PLANNER_PROSPECTIVE_HOLDOUT_CLOSED")
            for transition_time in self.evidence.transition_times_as_of(point):
                transition_ids = set(self.evidence.ids_as_of(transition_time))
                prior_ids = set(self.evidence.ids_as_of(transition_time.fromordinal(transition_time.toordinal() - 1)))
                if not transition_ids - prior_ids:
                    continue
                for generation in self.model_store.generations_as_of(transition_time):
                    if generation.created_at != transition_time:
                        continue
                    if not self.evidence.contains_snapshot(transition_time, generation.evidence_fingerprint_at_creation):
                        raise ValueError(f"GENERATION_EVIDENCE_SNAPSHOT_MISMATCH:{generation.generation_id}")
                    event_payload = {
                        "event_time": generation.created_at.isoformat(),
                        "generation_id": generation.generation_id,
                        "recipe_id": generation.recipe_id,
                        "trigger": "NEW_MATURED_FOLD",
                        "evidence_fingerprint": generation.evidence_fingerprint_at_creation,
                    }
                    event_id = stable_hash(event_payload)
                    events[event_id] = GenerationEvent(
                        event_id=event_id,
                        event_time=generation.created_at,
                        generation_id=generation.generation_id,
                        recipe_id=generation.recipe_id,
                        trigger="NEW_MATURED_FOLD",
                        evidence_fingerprint=generation.evidence_fingerprint_at_creation,
                    )
        return tuple(sorted(events.values(), key=lambda event: (event.event_time, event.generation_id)))
