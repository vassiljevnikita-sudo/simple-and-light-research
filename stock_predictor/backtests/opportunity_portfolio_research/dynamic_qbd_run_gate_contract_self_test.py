"""Self-test for the frozen pre-run selection and evaluation contracts."""
from __future__ import annotations

from dataclasses import replace
from datetime import date
from pathlib import Path
import json

from .dynamic_qbd_baseline_arms import S0InitialStatic, S2bGenerationAndRecipe, S3EqualWeightPool
from .dynamic_qbd_evidence_store_cursor import EvidenceCursor, EvidenceRecord
from .dynamic_qbd_model_store import ModelStore
from .dynamic_qbd_orchestrator_contract import (ModelGenerationRecord, OrchestratorDecision,
                                                 default_pseudolive_experiment_contract)
from .dynamic_qbd_pseudolive_replay import PseudoLiveCoordinator
from .dynamic_qbd_run_gate_contract import (EvaluationContract, FrozenOrchestratorPolicy,
                                             FrozenS3PoolRule, GenerationOption,
                                             InitialSelectionRule, OrchestratorInputAsOf,
                                             RunGateContract, default_run_gate_contract)


BASELINE = "e6b72df4ecb0e595293df62099860781436181dc"
SEED = date(2021, 2, 26)
EVENT = date(2021, 8, 31)


def _generation(generation_id: str, recipe_id: str, created_at: date, evidence_fingerprint: str) -> ModelGenerationRecord:
    return ModelGenerationRecord(
        generation_id=generation_id, recipe_id=recipe_id, horizon=3,
        created_at=created_at, training_start=date(2018, 1, 1),
        training_end=date(2020, 12, 31), target_maturity_cutoff=created_at,
        model_artifact_id=f"artifact-{generation_id}", model_hash=f"hash-{generation_id}",
        calibration_id=f"calibration-{generation_id}",
        evidence_fingerprint_at_creation=evidence_fingerprint,
        source_generation_id=generation_id,
        source_generation_fingerprint=f"fingerprint-{generation_id}",
        source_generation_provenance={"recipe_id": recipe_id, "generation_id": generation_id},
    )


def main() -> int:
    base = default_pseudolive_experiment_contract(baseline_commit=BASELINE)
    gate = default_run_gate_contract(base)
    artifact = Path(__file__).resolve().parents[3] / "research" / "DQBD_CAUSAL_MODEL_STORE_V1_RUN_GATES.json"
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    loaded = RunGateContract.from_dict(payload)
    assert loaded.contract_hash == payload["contract_hash"] == gate.contract_hash
    assert payload["decision_contract_hash"] == gate.decision_contract_hash
    assert payload["s3_pool_rule_hash"] == gate.s3_pool_rule.rule_hash
    assert gate.s3_pool_rule.rule_hash == loaded.s3_pool_rule.rule_hash
    assert loaded.evaluation.contract_hash == gate.evaluation.contract_hash

    selection = {
        "selection_cutoff": SEED.isoformat(),
        "selection_algorithm_version": "QBD_RECIPE_SELECTION_POLICY_V1",
        "selected_candidate_id": "R0",
        "winner_evidence": {"fold_count": 2},
    }
    assert gate.initial_selection.validate_selection_artifact(selection, seed=SEED) == "R0"
    try:
        gate.initial_selection.validate_selection_artifact(selection | {"selection_cutoff": EVENT.isoformat()}, seed=SEED)
    except ValueError as exc:
        assert str(exc) == "INITIAL_SELECTION_ARTIFACT_CUTOFF_MISMATCH"
    else:
        raise AssertionError("future selection artifact was accepted")

    evidence = EvidenceCursor((EvidenceRecord("fold-seed", SEED, "seed"),
                               EvidenceRecord("fold-event", EVENT, "event")))
    seed_fp = evidence.fingerprint_as_of(SEED)
    event_fp = evidence.fingerprint_as_of(EVENT)
    g0 = _generation("G0", "R0", SEED, seed_fp)
    g1 = _generation("G1", "R0", EVENT, event_fp)
    g_other = _generation("G-OTHER", "R1", EVENT, event_fp)
    store = ModelStore((g0, g1, g_other))
    initial = S0InitialStatic.from_selection_artifact(
        store=store, seed=SEED, horizon=3,
        selection_rule=gate.initial_selection, selection_artifact=selection,
    )
    assert initial.state_as_of(EVENT, "matched").active_generation_ids == ("G0",)

    s2 = S2bGenerationAndRecipe(store=store, horizon=3)
    inputs = s2.input_as_of(decision_time=EVENT, incumbent_generation_id="G0", evidence=evidence)
    assert isinstance(inputs, OrchestratorInputAsOf)
    assert gate.orchestrator_policy.select(inputs) == "G1"
    assert s2.state_as_of_policy(inputs=inputs, portfolio_state_fingerprint="matched",
                                 policy=gate.orchestrator_policy).active_generation_ids == ("G1",)
    assert gate.s3_pool_rule.select_pool(inputs.candidate_options) == ("G-OTHER", "G1")
    pool = S3EqualWeightPool(store=store, horizon=3).state_as_of_rule(
        decision_time=EVENT, portfolio_state_fingerprint="matched", rule=gate.s3_pool_rule)
    assert pool.active_generation_ids == ("G-OTHER", "G1")
    assert pool.active_generation_weights == (("G-OTHER", .5), ("G1", .5))
    repeated = S3EqualWeightPool(store=store, horizon=3).state_as_of_rule(
        decision_time=EVENT, portfolio_state_fingerprint="matched", rule=gate.s3_pool_rule)
    assert pool.pool_fingerprint == repeated.pool_fingerprint

    seed_pool = S3EqualWeightPool(store=store, horizon=3).state_as_of_rule(
        decision_time=SEED, portfolio_state_fingerprint="matched", rule=gate.s3_pool_rule)
    assert seed_pool.active_generation_ids == ("G0",)
    assert seed_pool.active_generation_weights == (("G0", 1.0),)
    assert "G1" not in seed_pool.active_generation_ids

    duplicate_options = inputs.candidate_options + (inputs.candidate_options[-1],)
    assert gate.s3_pool_rule.select_pool(duplicate_options) == ("G-OTHER", "G1")
    try:
        gate.s3_pool_rule.select_pool(duplicate_options[:-1] +
                                      (GenerationOption("G1", "R2", EVENT),))
    except ValueError as exc:
        assert str(exc) == "S3_POOL_DUPLICATE_GENERATION_ID_CONFLICT"
    else:
        raise AssertionError("conflicting duplicate generation was accepted")

    try:
        S3EqualWeightPool(store=store, horizon=None).state_as_of_rule(
            decision_time=EVENT, portfolio_state_fingerprint="matched", rule=gate.s3_pool_rule)
    except ValueError as exc:
        assert str(exc) == "S3_HORIZON_REQUIRED"
    else:
        raise AssertionError("unbound S3 horizon was accepted")
    try:
        S3EqualWeightPool(store=store, horizon=4).state_as_of_rule(
            decision_time=EVENT, portfolio_state_fingerprint="matched", rule=gate.s3_pool_rule)
    except ValueError as exc:
        assert str(exc) == "S3_POOL_NO_VISIBLE_GENERATION"
    else:
        raise AssertionError("cross-horizon/no-visible S3 pool was accepted")
    try:
        S3EqualWeightPool(store=ModelStore((replace(g0, source_generation_id=""),)), horizon=3).state_as_of_rule(
            decision_time=SEED, portfolio_state_fingerprint="matched", rule=gate.s3_pool_rule)
    except ValueError as exc:
        assert str(exc) == "S3_POOL_INVALID_GENERATION_PROVENANCE"
    else:
        raise AssertionError("invalid generation provenance was accepted")

    decision = OrchestratorDecision(
        decision_time=EVENT, incumbent_generation_id="G0",
        candidate_generation_ids=("G1", "G-OTHER"), selected_generation_id="G1",
        action="ACTIVATE_NEW", evidence_fingerprint=event_fp,
        decision_contract_hash=gate.decision_contract_hash,
    )
    coordinator = PseudoLiveCoordinator(
        model_store=store, evidence=evidence,
        expected_decision_contract_hash=gate.decision_contract_hash,
    )
    assert coordinator.record_decision(decision) == decision

    older_only = OrchestratorInputAsOf(EVENT, "G1", inputs.candidate_options, event_fp)
    assert gate.orchestrator_policy.select(older_only) == "G1"
    assert gate.evaluation.primary_arm_contrast == "S2_MINUS_S0"
    assert gate.evaluation.bootstrap_method == "MOVING_CALENDAR_BLOCK_BOOTSTRAP"
    print("DQBD_RUN_GATE_CONTRACT_SELF_TEST_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
