"""Selector-free pseudo-live trace over the causal ModelStore boundary."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Iterable

from .dynamic_qbd_evidence_store_cursor import EvidenceCursor
from .dynamic_qbd_model_store import ModelStore
from .dynamic_qbd_orchestrator_contract import DEVELOPMENT_END, OrchestratorDecision


@dataclass(frozen=True)
class PseudoLiveSnapshot:
    decision_time: date
    visible_generation_ids: tuple[str, ...]
    evidence_fingerprint: str


class PseudoLiveCoordinator:
    """Record causal inputs and externally supplied decisions, never choose one."""

    def __init__(self, *, model_store: ModelStore, evidence: EvidenceCursor,
                 expected_decision_contract_hash: str) -> None:
        self.model_store = model_store
        self.evidence = evidence
        if not expected_decision_contract_hash:
            raise ValueError("PSEUDOLIVE_DECISION_CONTRACT_HASH_REQUIRED")
        self.expected_decision_contract_hash = str(expected_decision_contract_hash)
        self._decisions: list[OrchestratorDecision] = []

    def snapshot(self, decision_time: date | str) -> PseudoLiveSnapshot:
        point = decision_time if isinstance(decision_time, date) else date.fromisoformat(str(decision_time))
        if point > DEVELOPMENT_END:
            raise ValueError("PSEUDOLIVE_PROSPECTIVE_HOLDOUT_CLOSED")
        visible = self.model_store.generations_as_of(point)
        return PseudoLiveSnapshot(
            decision_time=point,
            visible_generation_ids=tuple(record.generation_id for record in visible),
            evidence_fingerprint=self.evidence.fingerprint_as_of(point),
        )

    def record_decision(self, decision: OrchestratorDecision) -> OrchestratorDecision:
        snapshot = self.snapshot(decision.decision_time)
        visible = set(snapshot.visible_generation_ids)
        if decision.incumbent_generation_id not in visible:
            raise ValueError("PSEUDOLIVE_INCUMBENT_NOT_VISIBLE")
        if not set(decision.candidate_generation_ids) <= visible:
            raise ValueError("PSEUDOLIVE_CANDIDATE_NOT_VISIBLE")
        if decision.evidence_fingerprint != snapshot.evidence_fingerprint:
            raise ValueError("PSEUDOLIVE_EVIDENCE_CURSOR_MISMATCH")
        if decision.decision_contract_hash != self.expected_decision_contract_hash:
            raise ValueError("PSEUDOLIVE_DECISION_CONTRACT_HASH_MISMATCH")
        if any(existing.decision_time == decision.decision_time for existing in self._decisions):
            raise ValueError("PSEUDOLIVE_DECISION_TIME_ALREADY_RECORDED")
        self._decisions.append(decision)
        self._decisions.sort(key=lambda item: item.decision_time)
        return decision

    def decisions(self) -> tuple[OrchestratorDecision, ...]:
        return tuple(self._decisions)


def replay_decision_log(coordinator: PseudoLiveCoordinator, decisions: Iterable[OrchestratorDecision]) -> tuple[OrchestratorDecision, ...]:
    """Re-validate a decision log against the same point-in-time stores."""
    for decision in decisions:
        coordinator.record_decision(decision)
    return coordinator.decisions()
