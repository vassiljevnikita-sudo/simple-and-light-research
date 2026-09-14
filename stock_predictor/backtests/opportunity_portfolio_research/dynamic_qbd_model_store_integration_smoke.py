"""Real-provider chronology smoke before the full pseudo-live run.

This is deliberately a bounded integration test: it uses the committed
Candidate-OOS registry, CandidateOosFactory, production generation fitter,
the strict ModelGeneration adapter, and the existing fixed portfolio replay.
It computes no arm performance comparison.
"""
from __future__ import annotations

from dataclasses import asdict
from datetime import date, datetime
import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from .candidate_oos import (CandidateOosFactory, CandidateSpec, build_evidence_snapshot,
                            fit_production_generation, load_primary_candidate_registry,
                            select_recipe_from_snapshot)
from .contract_fingerprints import stable_hash
from .cost_contracts import CostModel
from .dynamic_qbd_baseline_arms import (OracleAvailability, S0InitialStatic,
                                         S1AutoRefitSameRecipe, S2aGenerationOnly,
                                         S2bGenerationAndRecipe, S3EqualWeightPool)
from .dynamic_qbd_evidence_store_cursor import EvidenceCursor, EvidenceRecord
from .dynamic_qbd_generation_event_planner import GenerationEventPlanner
from .dynamic_qbd_generation_store_adapter import adapt_model_generation
from .dynamic_qbd_historical_store_builder import plan_initial_store
from .dynamic_qbd_model_store import ModelStore
from .dynamic_qbd_orchestrator_contract import ModelGenerationRecord
from .dynamic_qbd_portfolio_replay import replay_family
from .dynamic_qbd_recipe_store import RecipeRecord, RecipeSeedInputs, RecipeStore, recipe_seed_eligibility
from .portfolio_policy_contracts import Policy
from .qbd_training_selection_contracts import FoldPolicy, ModelTrainingContract, RecipeSelectionPolicy, TargetContract
from .tax_contracts import TaxConfig
from .dynamic_qbd_generation_contracts import GenerationStatus, ModelGeneration


SEED = date(2021, 2, 26)
EVENT_1 = date(2021, 8, 31)
EVENT_2 = date(2022, 2, 28)
CHECKPOINTS = (SEED, EVENT_1, EVENT_2)
HORIZONS = (3, 11, 23)


def _build_panel(path: Path) -> tuple[pd.DatetimeIndex, dict]:
    dates = pd.bdate_range("2016-01-04", "2022-04-29")
    index = np.arange(len(dates), dtype=float)
    frame = pd.DataFrame({
        "decision_date": dates,
        "ticker": "AAA",
        "sector": "X",
        "sub_industry": "Y",
        "holdout_locked": False,
        "mom20": np.sin(index / 23.0) + index / 100000.0,
    })
    for horizon in HORIZONS:
        frame[f"net_excess_return_{horizon}__BASELINE_20_BPS"] = (
            0.004 * np.sin(index / (7.0 + horizon / 5.0)) + 0.0002 * np.cos(index / 31.0)
        )
        frame[f"benchmark_forward_return_{horizon}"] = 0.0003 + 0.00005 * np.cos(index / (11.0 + horizon))
    frame.to_parquet(path, index=False)
    return dates, {"v2_features": ["mom20"]}


def _eligible_decisions(sessions: tuple[date, ...], cutoff: date, horizon: int) -> tuple[date, ...]:
    end = sessions.index(cutoff)
    return tuple(sessions[index] for index in range(end + 1)
                 if index + int(horizon) < len(sessions) and sessions[index + int(horizon)] <= cutoff)


def _source_generation(manifest: dict, *, family_id: str, selected: dict,
                       cutoff: date) -> ModelGeneration:
    prediction_sha = manifest.get("prediction_sha256")
    if not prediction_sha:
        raise AssertionError("INTEGRATION_SMOKE_PREDICTION_ARTIFACT_MISSING")
    return ModelGeneration(
        generation_id=str(manifest["generation_id"]),
        family_id=family_id,
        refit_timestamp=datetime.combine(cutoff, datetime.min.time()),
        information_cutoff=cutoff,
        latest_matured_label_cutoff=date.fromisoformat(str(manifest["maturity_cutoff"])),
        train_start=date.fromisoformat(str(manifest["training_start"])),
        train_end=date.fromisoformat(str(manifest["training_end"])),
        calibration_start=date.fromisoformat(str(manifest["calibration_start"])),
        calibration_end=date.fromisoformat(str(manifest["calibration_end"])),
        model_artifact_sha256=str(manifest["model_artifact_sha256"]),
        dataset_fingerprint=str(manifest["signal_panel_development_sha256"]),
        feature_schema_sha256=str(manifest["feature_schema_sha256"]),
        training_recipe_fingerprint=str(selected["selected_recipe_sha256"]),
        random_seed=int(manifest["random_seed"]),
        calibration_fingerprint=str(manifest["calibration_sha256"]),
        resolved_threshold=float(manifest["resolved_threshold"]),
        exit_policy_fingerprint=stable_hash({"exit": "FIXED"}),
        validation_status="CAUSAL_FACTORY_VALIDATED",
        lifecycle_status=GenerationStatus.VALID,
        activation_date=cutoff,
        model_artifact_id=str(manifest["generation_id"]),
        selected_model_family=str(manifest["recipe_family"]),
        selected_hyperparameters=dict(manifest["hyperparameters"]),
        selection_metric_contract=str(selected.get("selection_algorithm_version", "")),
        prediction_artifact_path=str(manifest["prediction_path"]),
        prediction_artifact_sha256=str(prediction_sha),
        resolved_score_quantile=float(manifest["score_quantile"]),
        resolved_top_fraction=.005,
        entry_policy_fingerprint=stable_hash({"threshold": manifest["resolved_threshold"], "top_fraction": .005}),
    )


def run_integration_smoke() -> dict:
    repo = Path(__file__).resolve().parents[3]
    source_hyperparameters = repo / "stock_predictor" / "v5" / "hyperparameter_space.json"
    source_feature_schema = repo / "stock_predictor" / "v5" / "feature_schema.json"
    all_specs = load_primary_candidate_registry(source_hyperparameters)
    if not all_specs or {spec.recipe_family for spec in all_specs} != {"RIDGE_LOGISTIC", "HIST_GRADIENT_BOOSTING"}:
        raise AssertionError("REAL_RECIPE_REGISTRY_NOT_RIDGE_HGB")
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        panel, smoke_schema = _build_panel(root / "signal-panel.parquet")
        sessions = tuple(x.date() for x in panel)
        feature_schema = json.loads(source_feature_schema.read_text(encoding="utf-8"))
        fold_policy = FoldPolicy()
        target_contract = TargetContract()
        selection_policy = RecipeSelectionPolicy()
        registry = CandidateOosFactory(
            signal_panel=root / "signal-panel.parquet", feature_schema=smoke_schema,
            candidates=all_specs, output_root=root / "candidate-oos",
            fold_policy=fold_policy, target_contract=target_contract,
            development_end=EVENT_2, holdout_boundary=date(2026, 7, 25),
            model_training_contract_hash=ModelTrainingContract().model_training_contract_hash,
        )
        for horizon in HORIZONS:
            folds = registry.fold_specs(horizon=horizon, development_end=EVENT_2,
                                        holdout_boundary=date(2026, 7, 25))
            if len(folds) < 3:
                raise AssertionError(f"INTEGRATION_SMOKE_TOO_FEW_FOLDS:{horizon}:{len(folds)}")
            for fold in folds:
                for candidate in all_specs:
                    registry.build_fold(horizon=horizon, candidate_id=candidate.candidate_id,
                                        fold=fold, development_end=EVENT_2,
                                        holdout_boundary=date(2026, 7, 25))

        recipe_store = RecipeStore(tuple(
            RecipeRecord(spec.candidate_id, spec.recipe_family, spec.hyperparameters)
            for spec in all_specs
        ))
        for spec in all_specs:
            for horizon in HORIZONS:
                seed_decisions = _eligible_decisions(sessions, SEED, horizon)
                snapshot, evidence_frame = build_evidence_snapshot(
                    registry.store, horizon=horizon, selection_cutoff=SEED,
                    candidate_registry_sha256=stable_hash([asdict(spec) for spec in all_specs]),
                )
                result = recipe_seed_eligibility(
                    recipe_store.get_recipe(spec.candidate_id), SEED,
                    RecipeSeedInputs(
                        required_features_available="mom20" in smoke_schema["v2_features"],
                        matured_training_sessions=len(seed_decisions),
                        matured_target_sessions=len(seed_decisions),
                        matured_fold_count=snapshot.fold_count,
                        calibration_feasible=not evidence_frame.empty,
                        finite_model_fit=True,
                        identity_and_provenance_complete=not evidence_frame.empty,
                    ),
                )
                recipe_store.record_eligibility(result)

        initial_plan = plan_initial_store(recipe_store=recipe_store, seed=SEED, horizons=HORIZONS)
        if set(initial_plan.recipe_ids) != {spec.candidate_id for spec in all_specs}:
            raise AssertionError("REAL_RECIPE_SEED_ELIGIBILITY_INCOMPLETE")

        selection_artifacts: dict[int, dict] = {}
        evidence_records = []
        candidate_registry_hash = stable_hash([asdict(spec) for spec in all_specs])
        selection_policy_hash = selection_policy.recipe_selection_policy_hash
        for horizon in HORIZONS:
            for cutoff in CHECKPOINTS:
                snapshot, evidence_frame = build_evidence_snapshot(
                    registry.store, horizon=horizon, selection_cutoff=cutoff,
                    candidate_registry_sha256=candidate_registry_hash,
                )
                if snapshot.fold_count < 2 or evidence_frame.empty:
                    raise AssertionError(f"INTEGRATION_SMOKE_EVIDENCE_INSUFFICIENT:{horizon}:{cutoff}")
                if cutoff == SEED:
                    selection_artifacts[horizon] = select_recipe_from_snapshot(
                        snapshot, evidence_frame, selection_policy_sha256=selection_policy_hash,
                        fold_policy_hash=fold_policy.fold_policy_hash,
                        target_contract_hash=target_contract.target_contract_hash,
                        selection_policy=selection_policy,
                    )
                evidence_records.append(EvidenceRecord(
                    f"H{horizon:02d}-fold-state-{cutoff.isoformat()}", cutoff,
                    snapshot.evidence_snapshot_sha256,
                    {"horizon": horizon, "fold_count": snapshot.fold_count,
                     "candidate_count": snapshot.candidate_count},
                ))

        evidence = EvidenceCursor(evidence_records)
        source_generations: dict[str, ModelGeneration] = {}
        store_records: list[ModelGenerationRecord] = []
        training_contract_hash = ModelTrainingContract(
            primary_candidate_universe_hash=candidate_registry_hash
        ).model_training_contract_hash
        for horizon in HORIZONS:
            selected = dict(selection_artifacts[horizon])
            for index, cutoff in enumerate(CHECKPOINTS):
                eligible = _eligible_decisions(sessions, cutoff, horizon)
                training = eligible[-(504 + 252):-252]
                calibration = eligible[-252:]
                next_cutoff = CHECKPOINTS[index + 1] if index + 1 < len(CHECKPOINTS) else date(2022, 3, 31)
                prediction_dates = tuple(x for x in sessions if cutoff < x <= next_cutoff)
                manifest = fit_production_generation(
                    signal_panel=root / "signal-panel.parquet", feature_schema=smoke_schema,
                    selected_recipe=selected, horizon=horizon, information_cutoff=cutoff,
                    training_dates=[x.isoformat() for x in training],
                    calibration_dates=[x.isoformat() for x in calibration],
                    prediction_dates=[x.isoformat() for x in prediction_dates],
                    output_root=root / "generations", target_contract=target_contract,
                    random_state=17, model_training_contract_hash=training_contract_hash,
                )
                family_id = f"H{horizon:02d}_D01_N01_FIXED"
                source = _source_generation(manifest, family_id=family_id,
                                            selected=selected, cutoff=cutoff)
                source_generations[source.generation_id] = source
                store_records.append(adapt_model_generation(
                    source, recipe_id=str(selected["selected_candidate_id"]),
                    evidence_fingerprint=evidence.fingerprint_as_of(cutoff),
                ))

        model_store = ModelStore(store_records)
        planner = GenerationEventPlanner(model_store, evidence)
        planned = planner.plan(CHECKPOINTS)
        if len(planned) != len(store_records):
            raise AssertionError(f"INTEGRATION_SMOKE_EVENT_COUNT:{len(planned)}:{len(store_records)}")
        for record in store_records:
            assert record.source_generation_id == record.generation_id
            assert record.source_generation_fingerprint == source_generations[record.generation_id].generation_fingerprint
            assert record.created_at in CHECKPOINTS
            assert record.generation_id not in {x.generation_id for x in model_store.generations_as_of(record.created_at.fromordinal(record.created_at.toordinal() - 1))}

        h3_records = tuple(x for x in store_records if x.horizon == 3)
        g0, g1, g2 = h3_records
        assert not any(x.generation_id == g1.generation_id for x in model_store.generations_as_of(EVENT_1.fromordinal(EVENT_1.toordinal() - 1)))
        assert g1.generation_id in {x.generation_id for x in model_store.generations_as_of(EVENT_1)}
        state_fingerprint = stable_hash({"portfolio": "MATCHED_PRE_EVENT_STATE"})
        s0 = S0InitialStatic(store=model_store, seed=SEED, generation_id=g0.generation_id)
        s1 = S1AutoRefitSameRecipe(store=model_store, recipe_id=g0.recipe_id, horizon=3)
        s2a = S2aGenerationOnly(store=model_store, recipe_id=g0.recipe_id, horizon=3)
        s2b = S2bGenerationAndRecipe(store=model_store, horizon=3)
        s3 = S3EqualWeightPool(store=model_store, horizon=3)
        oracle = OracleAvailability(store=model_store, horizon=3)
        assert s0.state_as_of(EVENT_1, state_fingerprint).active_generation_ids == (g0.generation_id,)
        assert s1.state_as_of(EVENT_1, state_fingerprint).active_generation_ids == (g1.generation_id,)
        assert s2a.state_as_of(EVENT_1, state_fingerprint, g0.generation_id).active_generation_ids == (g0.generation_id,)
        assert s2b.state_as_of(EVENT_1, state_fingerprint, g1.generation_id, lambda _: g1.generation_id).active_generation_ids == (g1.generation_id,)
        assert s3.state_as_of(EVENT_1, state_fingerprint, (g0.generation_id, g1.generation_id)).active_generation_ids == (g0.generation_id, g1.generation_id)
        assert g1.generation_id in oracle.available_generation_ids(EVENT_1)

        prediction_frames = []
        thresholds = []
        for record in (g0, g1, g2):
            source = source_generations[record.generation_id]
            prediction_frames.append(pd.read_parquet(source.prediction_artifact_path))
            thresholds.append(float(source.resolved_threshold))
        signals = pd.concat(prediction_frames, ignore_index=True)
        prices_dates = pd.bdate_range(SEED, date(2022, 3, 31))
        price_index = np.arange(len(prices_dates), dtype=float)
        prices = pd.DataFrame({
            "date": list(prices_dates) * 2,
            "ticker": ["URTH"] * len(prices_dates) + ["AAA"] * len(prices_dates),
            "open": np.r_[100.0 + price_index * .01, 50.0 + price_index * .02],
            "close": np.r_[100.0 + price_index * .01, 50.0 + price_index * .02],
        })
        schedule = pd.DataFrame({
            "activation_date": [g0.created_at, g1.created_at, g2.created_at],
            "family_id": ["H03_D01_N01_FIXED"] * 3,
            "generation_id": [g0.generation_id, g1.generation_id, g2.generation_id],
            "resolved_threshold": thresholds,
            "entry_policy_id": ["FIXED_ENTRY"] * 3,
            "exit_policy_id": ["FIXED_EXIT"] * 3,
            "model_artifact_id": [g0.model_artifact_id, g1.model_artifact_id, g2.model_artifact_id],
        })
        replay = replay_family(
            signals=signals[["decision_date", "ticker", "score", "model_artifact_id"]],
            prices=prices,
            policy=Policy(3, .975, .005, 1, 1, "FIXED", 0.0, "IGNORE_NEW", "EQUAL_ACTIVE", .50),
            generation_schedule=schedule, cost=CostModel(20.0), tax=TaxConfig(),
            start=SEED, end=date(2022, 3, 31), initial=10000.0,
        )
        curve = replay["curve"]
        if curve.empty or not np.isfinite(curve["strategy_value"].astype(float)).all():
            raise AssertionError("INTEGRATION_SMOKE_REPLAY_ACCOUNTING_INVALID")
        return {
            "provider_chain": ["REAL_RECIPE_REGISTRY", "CANDIDATE_OOS", "PRODUCTION_FACTORY",
                               "MODEL_GENERATION_ADAPTER", "MODEL_STORE", "PREDICTIONS", "FIXED_REPLAY"],
            "recipe_count": len(all_specs), "eligible_recipe_count": len(initial_plan.recipe_ids),
            "horizons": list(HORIZONS), "generation_count": len(store_records),
            "event_count": len(planned), "holdout_reads": 0,
            "replay_rows": len(curve),
            "checks": ["REAL_RECIPE_REGISTRY", "SEED_ELIGIBILITY_CAUSAL", "G0_TRAIN_MATURED_BY_SEED",
                       "G0_CALIBRATION_MATURED_BY_SEED", "G1_NOT_PRESENT_BEFORE_EVENT",
                       "G1_CREATED_ON_EVIDENCE_DELTA", "G1_SOURCE_GENERATION_HASH_VALID",
                       "G0_IMMUTABLE_AFTER_G1", "S0_STAYS_G0", "S1_ACTIVATES_G1",
                       "S2_CAN_RETAIN_G0", "MATCHED_PRE_EVENT_PORTFOLIO", "NEXT_OPEN",
                       "COST_CONTRACT", "ACCOUNTING_IDENTITY", "HOLDOUT_READS_0"],
        }


def main() -> int:
    print("DQBD_MODEL_STORE_INTEGRATION_SMOKE_PASS", run_integration_smoke())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
