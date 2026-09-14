"""Small chronology-only gate for H3/H11/H23 before any performance run."""
from __future__ import annotations

from datetime import date

from .contract_fingerprints import stable_hash
from .dynamic_qbd_baseline_arms import OracleAvailability, S0InitialStatic, S1AutoRefitSameRecipe, S2aGenerationOnly, S2bGenerationAndRecipe, S3EqualWeightPool
from .dynamic_qbd_evidence_store_cursor import EvidenceCursor, EvidenceRecord
from .dynamic_qbd_generation_event_planner import GenerationEventPlanner
from .dynamic_qbd_historical_store_builder import plan_initial_store
from .dynamic_qbd_model_store import ModelStore
from .dynamic_qbd_orchestrator_contract import ModelGenerationRecord
from .dynamic_qbd_recipe_store import RecipeRecord, RecipeSeedInputs, RecipeStore, recipe_seed_eligibility


HORIZONS = (3, 11, 23)
SEED = date(2021, 2, 26)
EVENT_1 = date(2021, 8, 31)
EVENT_2 = date(2022, 2, 28)


def _generation(generation_id: str, horizon: int, created_at: date, evidence_fingerprint: str) -> ModelGenerationRecord:
    return ModelGenerationRecord(
        generation_id=generation_id,
        recipe_id="R0",
        horizon=horizon,
        created_at=created_at,
        training_start=date(2018, 1, 1),
        training_end=date(2020, 12, 31),
        target_maturity_cutoff=created_at,
        model_artifact_id=f"artifact-{generation_id}",
        model_hash=stable_hash([generation_id, "model"]),
        calibration_id=f"calibration-{generation_id}",
        evidence_fingerprint_at_creation=evidence_fingerprint,
    )


def run_chronology_smoke() -> dict:
    recipes = RecipeStore((RecipeRecord("R0", "RIDGE_LOGISTIC", {"alpha": 1.0}),))
    recipes.record_eligibility(recipe_seed_eligibility(
        recipes.get_recipe("R0"), SEED,
        RecipeSeedInputs(True, 504, 252, 2, True, True, True),
    ))
    initial_plan = plan_initial_store(recipe_store=recipes, seed=SEED, horizons=HORIZONS)
    assert initial_plan.slots == tuple(("R0", horizon) for horizon in HORIZONS)
    evidence = EvidenceCursor((
        EvidenceRecord("fold-seed", SEED, "fold-seed"),
        EvidenceRecord("fold-event-1", EVENT_1, "fold-event-1"),
        EvidenceRecord("fold-event-2", EVENT_2, "fold-event-2"),
    ))
    records = []
    for horizon in HORIZONS:
        records.extend((
            _generation(f"H{horizon:02d}-R0-G0", horizon, SEED, evidence.fingerprint_as_of(SEED)),
            _generation(f"H{horizon:02d}-R0-G1", horizon, EVENT_1, evidence.fingerprint_as_of(EVENT_1)),
            _generation(f"H{horizon:02d}-R0-G2", horizon, EVENT_2, evidence.fingerprint_as_of(EVENT_2)),
        ))
    store = ModelStore(records)
    planner = GenerationEventPlanner(store, evidence)
    assert all(x.generation_id.endswith("-G0") for x in store.generations_as_of(SEED))
    assert not any(x.generation_id.endswith("-G1") for x in store.generations_as_of(date(2021, 8, 30)))
    assert all(x.generation_id.endswith(("-G0", "-G1")) for x in store.generations_as_of(EVENT_1))
    assert len(planner.plan((SEED, EVENT_1, EVENT_2))) == len(records)
    state_fingerprint = stable_hash({"portfolio_state": "MATCHED_PRE_EVENT_STATE"})
    for horizon in HORIZONS:
        g0 = f"H{horizon:02d}-R0-G0"
        g1 = f"H{horizon:02d}-R0-G1"
        s0 = S0InitialStatic(store=store, seed=SEED, generation_id=g0)
        s1 = S1AutoRefitSameRecipe(store=store, recipe_id="R0", horizon=horizon)
        s2a = S2aGenerationOnly(store=store, recipe_id="R0", horizon=horizon)
        s2b = S2bGenerationAndRecipe(store=store, horizon=horizon)
        s3 = S3EqualWeightPool(store=store, horizon=horizon)
        oracle = OracleAvailability(store=store, horizon=horizon)
        assert s0.state_as_of(EVENT_1, state_fingerprint).active_generation_ids == (g0,)
        assert s1.state_as_of(EVENT_1, state_fingerprint).active_generation_ids == (g1,)
        assert s2a.state_as_of(EVENT_1, state_fingerprint, g0).active_generation_ids == (g0,)
        assert s2b.state_as_of(EVENT_1, state_fingerprint, g1, lambda candidates: g1).active_generation_ids == (g1,)
        assert s3.state_as_of(EVENT_1, state_fingerprint, (g0, g1)).active_generation_ids == (g0, g1)
        assert g1 in oracle.available_generation_ids(EVENT_1)
        assert all(s0.state_as_of(EVENT_1, state_fingerprint).matched_portfolio_state_fingerprint == state_fingerprint
                   for s0 in (S0InitialStatic(store=store, seed=SEED, generation_id=g0),))
    return {
        "horizons": list(HORIZONS),
        "seed": SEED.isoformat(),
        "events": [EVENT_1.isoformat(), EVENT_2.isoformat()],
        "checks": [
            "G0_AT_SEED", "G1_HIDDEN_BEFORE_EVENT", "EVIDENCE_MATURITY_CAUSAL",
            "G1_CREATED_AT_EVENT", "G0_IMMUTABLE", "S0_FIXED", "S1_AUTO_REFIT",
            "S2A_GENERATION_ONLY", "S2B_INTERFACE_ONLY", "S3_EQUAL_WEIGHT_POOL",
            "ORACLE_AVAILABILITY_ONLY", "MATCHED_PORTFOLIO_STATE", "HOLDOUT_CLOSED",
        ],
    }


def main() -> int:
    result = run_chronology_smoke()
    print("DQBD_CHRONOLOGY_SMOKE_PASS", result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
