"""Point-in-time cursor over matured Dynamic-QBD evidence."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from types import MappingProxyType
from typing import Any, Iterable, Mapping

from .contract_fingerprints import stable_hash
from .dynamic_qbd_orchestrator_contract import DEVELOPMENT_END


@dataclass(frozen=True)
class EvidenceRecord:
    evidence_id: str
    matured_at: date
    evidence_fingerprint: str
    payload: Mapping[str, Any] = None

    def __post_init__(self) -> None:
        if not self.evidence_id or not self.evidence_fingerprint:
            raise ValueError("EVIDENCE_RECORD_ID_AND_FINGERPRINT_REQUIRED")
        point = self.matured_at if isinstance(self.matured_at, date) else date.fromisoformat(str(self.matured_at))
        object.__setattr__(self, "matured_at", point)
        if point > DEVELOPMENT_END:
            raise ValueError("EVIDENCE_PROSPECTIVE_HOLDOUT_CLOSED")
        object.__setattr__(self, "payload", MappingProxyType(dict(self.payload or {})))

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "matured_at": self.matured_at.isoformat(),
            "evidence_fingerprint": self.evidence_fingerprint,
            "payload": dict(self.payload),
        }


class EvidenceCursor:
    """Evidence access is only through ``as_of(t)``; no unbounded collection API."""

    def __init__(self, records: Iterable[EvidenceRecord] = ()) -> None:
        self._records: dict[str, EvidenceRecord] = {}
        for record in records:
            self.add(record)

    def add(self, record: EvidenceRecord) -> EvidenceRecord:
        if not isinstance(record, EvidenceRecord):
            raise TypeError("EVIDENCE_CURSOR_REQUIRES_EVIDENCE_RECORD")
        existing = self._records.get(record.evidence_id)
        if existing is not None and existing != record:
            raise ValueError("EVIDENCE_RECORD_IMMUTABLE")
        self._records[record.evidence_id] = record
        return record

    @staticmethod
    def _as_of(value: date | str) -> date:
        point = value if isinstance(value, date) else date.fromisoformat(str(value))
        if point > DEVELOPMENT_END:
            raise ValueError("EVIDENCE_CURSOR_PROSPECTIVE_HOLDOUT_CLOSED")
        return point

    def as_of(self, t: date | str) -> tuple[EvidenceRecord, ...]:
        point = self._as_of(t)
        return tuple(sorted(
            (record for record in self._records.values() if record.matured_at <= point),
            key=lambda record: (record.matured_at, record.evidence_id),
        ))

    def ids_as_of(self, t: date | str) -> tuple[str, ...]:
        """Return the point-in-time fold/evidence identity set."""
        return tuple(record.evidence_id for record in self.as_of(t))

    def transition_times_as_of(self, t: date | str) -> tuple[date, ...]:
        """Return dates on which the visible evidence identity grew."""
        records = self.as_of(t)
        return tuple(sorted({record.matured_at for record in records}))

    def fingerprint_as_of(self, t: date | str) -> str:
        return stable_hash([record.to_dict() for record in self.as_of(t)])

    def contains_snapshot(self, t: date | str, fingerprint: str) -> bool:
        return self.fingerprint_as_of(t) == str(fingerprint)

    def __len__(self) -> int:
        return len(self._records)
