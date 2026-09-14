"""Deterministic contracts for the causal Dynamic-QBD model-store layer."""
from __future__ import annotations

from datetime import date
from pathlib import Path
import tempfile

from .contract_fingerprints import stable_hash
from .dynamic_qbd_evidence_store_cursor import EvidenceCursor, EvidenceRecord
from .dynamic_qbd_generation_event_planner import GenerationEventPlanner
from .dynamic_qbd_model_store import ModelStore
from .dynamic_qbd_historical_store_builder import build_historical_store
from .dynamic_qbd_model_store_evaluation import evaluate_visibility
from .dynamic_qbd_orchestrator_contract import (
    DEVELOPMENT_END, HOLDOUT_BOUNDARY, ModelGenerationRecord, OrchestratorDecision, RunContract,
    PseudoLiveExperimentContract, default_pseudolive_experiment_contract,
)
from .dynamic_qbd_pseudolive_replay import PseudoLiveCoordinator


def _generation(generation_id: str, created_at: date, evidence_fingerprint: str) -> ModelGenerationRecord:
    return ModelGenerationRecord(
        generation_id=generation_id,
        recipe_id="R0",
        horizon=3,
        created_at=created_at,
        training_start=date(2021, 1, 1),
        training_end=date(2025, 1, 31),
        target_maturity_cutoff=created_at,
        model_artifact_id=f"artifact-{generation_id}",
        model_hash=stable_hash([generation_id, "model"]),
        calibration_id=f"calibration-{generation_id}",
        evidence_fingerprint_at_creation=evidence_fingerprint,
    )


def main() -> int:
    evidence = EvidenceCursor((
        EvidenceRecord("fold-0", date(2025, 2, 28), "fold-fingerprint-0"),
        EvidenceRecord("fold-1", date(2025, 6, 30), "fold-fingerprint-1"),
    ))
    evidence_at_g0 = evidence.fingerprint_as_of(date(2025, 2, 28))
    evidence_at_g1 = evidence.fingerprint_as_of(date(2025, 6, 30))
    g0 = _generation("G0", date(2025, 2, 28), evidence_at_g0)
    g1 = _generation("G1", date(2025, 6, 30), evidence_at_g1)
    late_without_transition = _generation("G-LATE", date(2025, 7, 1), evidence_at_g1)
    store = ModelStore((g0, g1, late_without_transition))
    assert store.get_generation("G0") == g0
    assert tuple(x.generation_id for x in store.generations_as_of(date(2025, 3, 1))) == ("G0",)
    assert tuple(x.generation_id for x in store.generations_for_recipe("R0", date(2025, 12, 31))) == ("G0", "G1", "G-LATE")
    store.add_generation(g0)  # idempotent publication is allowed
    try:
        store.add_generation(_generation("G0", date(2025, 2, 28), stable_hash("different")))
    except ValueError as exc:
        assert str(exc) == "MODEL_GENERATION_IMMUTABLE"
    else:
        raise AssertionError("generation overwrite was accepted")
    assert len(evidence.as_of(date(2025, 3, 1))) == 1
    assert len(evidence.as_of(date(2025, 12, 31))) == 2
    events = GenerationEventPlanner(store, evidence).plan((date(2025, 3, 1), date(2025, 12, 31)))
    assert tuple(x.generation_id for x in events) == ("G0", "G1")
    assert "G-LATE" not in {x.generation_id for x in events}
    historical = build_historical_store(generations=(g0, g1), evidence=evidence.as_of(DEVELOPMENT_END))
    assert evaluate_visibility(model_store=historical.model_store, evidence=historical.evidence_cursor,
                               decision_times=(date(2025, 3, 1),))[-1]["generation_ids"] == ["G0"]
    contract = RunContract(
        baseline_commit="e6b72df4ecb0e595293df62099860781436181dc",
        recipe_universe="PRIMARY_CANDIDATE_REGISTRY_V1",
        seed={"primary": "2021-02-26"},
    )
    assert contract.development_end == DEVELOPMENT_END
    assert contract.holdout_boundary == HOLDOUT_BOUNDARY
    assert len(contract.run_contract_hash) == 64
    with tempfile.TemporaryDirectory() as folder:
        contract_path = Path(folder) / "run-contract.json"
        assert contract.write(contract_path) == contract.run_contract_hash
        assert contract.write(contract_path) == contract.run_contract_hash
        try:
            RunContract(baseline_commit=contract.baseline_commit, recipe_universe="different",
                        seed=contract.seed).write(contract_path)
        except ValueError as exc:
            assert str(exc) == "DQBD_RUN_CONTRACT_IMMUTABLE"
        else:
            raise AssertionError("run contract overwrite was accepted")
    frozen_contract = default_pseudolive_experiment_contract(
        baseline_commit="e6b72df4ecb0e595293df62099860781436181dc")
    frozen_path = Path(__file__).resolve().parents[3] / "research" / "DQBD_CAUSAL_MODEL_STORE_V1.json"
    frozen_payload = __import__("json").loads(frozen_path.read_text(encoding="utf-8"))
    assert PseudoLiveExperimentContract.from_dict(frozen_payload).contract_hash == frozen_payload["contract_hash"]
    assert frozen_contract.contract_hash == frozen_payload["contract_hash"]
    assert dict(frozen_contract.expected_fold_states) == {"short": 2, "primary": 3, "long": 4}
    decision = OrchestratorDecision(
        decision_time=date(2025, 6, 30),
        incumbent_generation_id="G0",
        candidate_generation_ids=("G1",),
        selected_generation_id="G1",
        action="ACTIVATE_NEW",
        evidence_fingerprint=evidence_at_g1,
        decision_contract_hash=contract.run_contract_hash,
    )
    assert decision.fingerprint != contract.run_contract_hash
    coordinator = PseudoLiveCoordinator(model_store=store, evidence=evidence,
                                         expected_decision_contract_hash=contract.run_contract_hash)
    assert coordinator.record_decision(decision) == decision
    assert coordinator.decisions() == (decision,)
    for call in (
        lambda: store.generations_as_of(HOLDOUT_BOUNDARY),
        lambda: evidence.as_of(HOLDOUT_BOUNDARY),
        lambda: GenerationEventPlanner(store, evidence).plan((HOLDOUT_BOUNDARY,)),
    ):
        try:
            call()
        except ValueError as exc:
            assert "HOLDOUT_CLOSED" in str(exc)
        else:
            raise AssertionError("prospective holdout was opened")
    print("DYNAMIC_QBD_MODEL_STORE_CONTRACT_SELF_TEST_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
