"""Strict adapters from the existing Dynamic-QBD generation authority."""
from __future__ import annotations

from dataclasses import asdict
from datetime import date
import re
from typing import Any

from .contract_fingerprints import stable_hash
from .dynamic_qbd_generation_contracts import GenerationStatus, ModelGeneration
from .dynamic_qbd_orchestrator_contract import ModelGenerationRecord


_FAMILY_HORIZON = re.compile(r"^H(?P<horizon>\d{2})_")


def _horizon_from_family(family_id: str) -> int:
    match = _FAMILY_HORIZON.match(str(family_id))
    if match is None:
        raise ValueError(f"GENERATION_FAMILY_HORIZON_UNPARSEABLE:{family_id}")
    return int(match.group("horizon"))


def adapt_model_generation(source: ModelGeneration, *, recipe_id: str,
                           evidence_fingerprint: str) -> ModelGenerationRecord:
    """Adapt one valid existing generation without dropping provenance.

    The evidence fingerprint is deliberately supplied by the point-in-time
    EvidenceCursor.  The adapter never invents a visibility timestamp.
    """
    if not isinstance(source, ModelGeneration):
        raise TypeError("MODEL_STORE_ADAPTER_REQUIRES_MODEL_GENERATION")
    if source.lifecycle_status != GenerationStatus.VALID:
        raise ValueError(f"MODEL_STORE_ADAPTER_REQUIRES_VALID_GENERATION:{source.generation_id}")
    if source.activation_date is None or source.activation_date != source.information_cutoff:
        raise ValueError(f"MODEL_STORE_ADAPTER_ACTIVATION_CUTOFF_MISMATCH:{source.generation_id}")
    created_at = source.refit_timestamp.date()
    if created_at != source.information_cutoff:
        raise ValueError(f"MODEL_STORE_ADAPTER_CREATION_CUTOFF_MISMATCH:{source.generation_id}")
    if not evidence_fingerprint:
        raise ValueError("MODEL_STORE_ADAPTER_EVIDENCE_FINGERPRINT_REQUIRED")
    provenance: dict[str, Any] = asdict(source)
    source_fingerprint = source.generation_fingerprint
    if stable_hash(provenance) == "":  # pragma: no cover - defensive contract guard
        raise AssertionError("MODEL_STORE_ADAPTER_PROVENANCE_HASH_EMPTY")
    return ModelGenerationRecord(
        generation_id=source.generation_id,
        recipe_id=str(recipe_id),
        horizon=_horizon_from_family(source.family_id),
        created_at=created_at,
        training_start=source.train_start,
        training_end=source.train_end,
        target_maturity_cutoff=source.latest_matured_label_cutoff,
        model_artifact_id=source.model_artifact_id,
        model_hash=source.model_artifact_sha256,
        calibration_id=f"{source.generation_id}:{source.calibration_fingerprint}",
        evidence_fingerprint_at_creation=str(evidence_fingerprint),
        source_generation_id=source.generation_id,
        source_generation_fingerprint=source_fingerprint,
        source_generation_provenance=provenance,
    )


def adapt_model_generations(sources: list[ModelGeneration] | tuple[ModelGeneration, ...], *,
                            recipe_id: str, evidence_fingerprint: str) -> tuple[ModelGenerationRecord, ...]:
    return tuple(adapt_model_generation(source, recipe_id=recipe_id,
                                        evidence_fingerprint=evidence_fingerprint)
                 for source in sources)
