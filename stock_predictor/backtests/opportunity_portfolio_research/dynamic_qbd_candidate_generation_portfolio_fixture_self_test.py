"""Small deterministic Candidate-OOS -> generation -> portfolio fixture."""
from __future__ import annotations

from datetime import date
from pathlib import Path
import tempfile
import json
from dataclasses import asdict

import numpy as np
import pandas as pd

from .candidate_oos import (CandidateOosFactory, CandidateOosStore, CandidateSpec,
                            build_evidence_snapshot, select_recipe_from_snapshot,
                            fit_production_generation, load_primary_candidate_registry)
from .portfolio_policy_contracts import Policy
from .cost_contracts import CostModel
from .tax_contracts import TaxConfig
from .qbd_training_selection_contracts import FoldPolicy, TargetContract, RecipeSelectionPolicy
from .dynamic_qbd_portfolio_replay import replay_family
from .dynamic_qbd_evidence import build_monthly_family_evidence
from .dynamic_qbd_manifested_job_coordinator import (ManifestedJobInputs, ManifestedJobStore, _job_graph,
                                           execute_manifested_development_jobs, sha256_file)
from .dynamic_qbd_family_surface import build_family_specs
from .contract_fingerprints import stable_hash
from .candidate_oos import candidate_registry_document


def main() -> int:
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        dates = pd.bdate_range("2010-01-01", periods=900)
        i = np.arange(len(dates), dtype=float)
        panel = root / "panel.parquet"
        pd.DataFrame({"decision_date": dates, "ticker": ["AAA"] * len(dates), "sector": ["X"] * len(dates),
                      "sub_industry": ["Y"] * len(dates), "mom20": np.sin(i / 17.0),
                      "net_excess_return_1__BASELINE_20_BPS": 0.004 * np.sin(i / 9.0),
                      "benchmark_forward_return_1": 0.0005 + 0.0001 * np.cos(i / 13.0)}).to_parquet(panel, index=False)
        candidate = CandidateSpec.create("RIDGE_LOGISTIC", {"alpha": 1.0, "positive_C": 1.0, "downside_C": 1.0})
        factory = CandidateOosFactory(signal_panel=panel, feature_schema={"v2_features": ["mom20"]},
                                      candidates=(candidate,), output_root=root / "candidate-oos",
                                      fold_policy=FoldPolicy(), target_contract=TargetContract(),
                                      development_end=date(2013, 6, 30), holdout_boundary=date(2026, 7, 25))
        folds = factory.fold_specs(horizon=1, development_end=date(2013, 6, 30), holdout_boundary=date(2026, 7, 25))
        factory.build_fold(horizon=1, candidate_id=candidate.candidate_id, fold=folds[0],
                           development_end=date(2013, 6, 30), holdout_boundary=date(2026, 7, 25))
        cutoff = dates[820].date()
        snapshot, evidence = build_evidence_snapshot(CandidateOosStore(root / "candidate-oos"), horizon=1,
                                                      selection_cutoff=cutoff, candidate_registry_sha256="r" * 64)
        selected = select_recipe_from_snapshot(snapshot, evidence, selection_policy_sha256="s" * 64,
                                               selection_policy=RecipeSelectionPolicy(minimum_candidate_folds=1))
        selected.update(recipe_family=candidate.recipe_family, hyperparameters=candidate.hyperparameters)
        authorized = [x.date() for x in dates if x.date() <= cutoff]
        generation = fit_production_generation(signal_panel=panel, feature_schema={"v2_features": ["mom20"]},
                                               selected_recipe=selected, horizon=1, information_cutoff=cutoff,
                                               training_dates=[str(x) for x in authorized[-504 - 30 - 252:-30 - 252]],
                                               calibration_dates=[str(x) for x in authorized[-252:]],
                                               target_contract=TargetContract(),
                                               prediction_dates=[str(x.date()) for x in dates if x.date() > cutoff][:30],
                                               output_root=root / "generations")
        predictions = pd.read_parquet(generation["prediction_path"])
        threshold = float(pd.read_parquet(generation["calibration_path"])["score"].quantile(.75))
        prices = pd.DataFrame({"date": list(dates) * 2, "ticker": ["URTH"] * len(dates) + ["AAA"] * len(dates),
                               "open": np.concatenate([100 + i * .02, 50 + i * .03]),
                               "close": np.concatenate([100 + i * .02, 50 + i * .03])})
        schedule = pd.DataFrame({"activation_date": [predictions["decision_date"].min()], "family_id": ["H01_D01_N01_FIXED"],
                                 "generation_id": [generation["generation_id"]], "resolved_threshold": [threshold],
                                 "entry_policy_id": ["ENTRY"], "exit_policy_id": ["EXIT"], "model_artifact_id": [generation["generation_id"]],
                                 "lifecycle_status": ["VALID"]})
        result = replay_family(signals=predictions[["decision_date", "ticker", "score", "model_artifact_id"]], prices=prices,
                               policy=Policy(1, .75, .50, 1, 1, "FIXED", 0.0, "IGNORE_NEW", "EQUAL_ACTIVE", .50),
                               generation_schedule=schedule, cost=CostModel(0.0), tax=TaxConfig(), initial=10000.0)
        curve = result["curve"].copy()
        curve["family_id"] = "H01_D01_N01_FIXED"
        evidence_monthly = build_monthly_family_evidence(
            curve, schedule, trades=pd.DataFrame(result.get("trades", [])))
        assert not predictions.empty and Path(generation["manifest_path"]).is_file()
        assert not evidence_monthly.empty and {"family_id", "assessment_date", "generation_id"} <= set(evidence_monthly)

    # Productive DAG fixture: the same manifested coordinator store, handlers and artifact
    # contracts are used; only the registry/data horizon is deliberately tiny.
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        repo = Path(__file__).resolve().parents[3]
        dates = pd.bdate_range("2010-01-01", periods=1300)
        i = np.arange(len(dates), dtype=float)
        panel = root / "dag-panel.parquet"
        panel_frame = pd.DataFrame({"decision_date": dates, "ticker": ["AAA"] * len(dates),
                                    "mom20": np.sin(i / 19.0),
                                    "net_excess_return_1__BASELINE_20_BPS": .003 * np.sin(i / 13.0),
                                    "net_excess_return_3__BASELINE_20_BPS": .005 * np.sin(i / 17.0),
                                    "benchmark_forward_return_1": .0002 + i * 0.0,
                                    "benchmark_forward_return_3": .0006 + i * 0.0})
        panel_frame.to_parquet(panel, index=False)
        schema = root / "schema.json"; schema.write_text(json.dumps({"v2_features": ["mom20"]}), encoding="utf-8")
        hp = root / "hyperparameters.json"
        hp.write_text(json.dumps({"families": {
            "RIDGE_LOGISTIC": [{"alpha": 1.0, "positive_C": 1.0, "downside_C": 1.0}],
            "HIST_GRADIENT_BOOSTING": [{"learning_rate": .05, "max_iter": 10, "max_leaf_nodes": 5, "l2_regularization": 1.0}]}}), encoding="utf-8")
        candidates = load_primary_candidate_registry(hp)
        candidate_registry = candidate_registry_document(candidates, source_sha256="QBD_PRIMARY_RIDGE_HGB_V1")
        families = [asdict(x) for x in build_family_specs(feature_schema_sha256=sha256_file(schema),
                    training_window_sessions=504, calibration_window_sessions=252,
                    cost_contract={"roundtrip_bps": 0.0}, tax_contract={"mode": "POST_TAX", "enabled": True,
                    "engine": "DE_RETAIL_APPROX", "capital_gains_rate": .25, "solidarity_surcharge": .055,
                    "allowance_eur": 1000.0, "church_tax_rate": 0.0})
                    if (x.horizon_sessions, x.holding_days, x.max_names) in {(1, 1, 1), (3, 1, 1), (3, 1, 2)}]
        for family in families:
            family["model_family_key"] = f"H{int(family['horizon_sessions']):02d}_RIDGE_HGB_FROZEN_RULE"
            family["portfolio_family_key"] = family["family_id"]
        registry = {"candidate_registry": candidate_registry, "portfolio_families": families,
                    "model_family_count": 2, "portfolio_family_count": len(families), "model_families": []}
        registry["family_registry_hash"] = stable_hash(registry)
        benchmark = root / "benchmark.parquet"
        prices = pd.DataFrame({"date": list(dates) * 2, "ticker": ["URTH"] * len(dates) + ["AAA"] * len(dates),
                               "open": np.r_[100 + i * .01, 50 + i * .02], "close": np.r_[100 + i * .01, 50 + i * .02]})
        prices.to_parquet(benchmark, index=False)
        distributions = root / "distributions.parquet"
        pd.DataFrame({"ticker": pd.Series(dtype=str), "ex_date": pd.Series(dtype=str),
                      "payable_date": pd.Series(dtype=str), "cash_amount": pd.Series(dtype=float)}).to_parquet(distributions, index=False)
        inputs = ManifestedJobInputs(repo_root=repo, signal_panel=panel, candidate_metrics=None,
                               feature_schema=schema, benchmark_prices=benchmark,
                               benchmark_distributions=distributions, stock_execution_prices=benchmark,
                               stock_distributions=distributions, development_start=dates[0].date(),
                               development_end=dates[-1].date(), hyperparameter_space=hp,
                               training_window_sessions=504, calibration_window_sessions=252,
                               allow_missing_stock_inputs=False)
        store_root = root / "dag-run"; store = ManifestedJobStore(store_root)
        sessions = tuple(x.date() for x in dates)
        folds = {}
        from stock_predictor.v5.walk_forward import expanding_folds
        for horizon in (1, 3):
            folds[horizon] = tuple(expanding_folds([x.isoformat() for x in sessions], minimum_train_dates=504,
                                                   validation_dates=126, step_dates=126, purge_dates=30, embargo_dates=0))[:2]
        cutoffs = (dates[1000].date(), dates[1150].date())
        jobs = _job_graph(registry, {1: cutoffs, 3: cutoffs}, folds, trading_sessions=sessions,
                          fold_policy=inputs.resolved_fold_policy(), target_contract=inputs.resolved_target_contract(),
                          recipe_selection_policy=inputs.resolved_contracts().recipe_selection_policy,
                          model_training_contract=inputs.resolved_contracts(
                              primary_candidate_universe_hash=candidate_registry["candidate_registry_sha256"]
                          ).model_training_contract)
        store.seed_jobs(jobs)
        from .development_slice_hash import parquet_development_slice_sha256
        signal_hash = parquet_development_slice_sha256(panel, date_column="decision_date",
                                                        development_end=inputs.development_end,
                                                        holdout_boundary=inputs.holdout_boundary)
        store.write("manifested-job-contract.json", {"signal_panel_development_sha256": signal_hash,
                    "candidate_registry_sha256": candidate_registry["candidate_registry_sha256"],
                    "fold_policy_hash": inputs.resolved_fold_policy().fold_policy_hash,
                    "target_contract_hash": inputs.resolved_target_contract().target_contract_hash,
                    "recipe_selection_policy_hash": inputs.resolved_contracts().recipe_selection_policy.recipe_selection_policy_hash,
                    "model_training_contract_hash": inputs.resolved_contracts(
                        primary_candidate_universe_hash=candidate_registry["candidate_registry_sha256"]
                    ).model_training_contract.model_training_contract_hash})
        store.write("family-registry.json", registry)
        result = {}
        for _ in range(8):
            result = execute_manifested_development_jobs(store=store, inputs=inputs, output_root=store_root)
            pending_h3 = [x["job_id"] for x in store.all_jobs()
                          if x["kind"] == "candidate_oos_fold" and int(x.get("horizon", -1)) == 3 and x["state"] == "PENDING"]
            if pending_h3:
                # Explicit resumable frontier advancement, still through the
                # production Candidate-OOS executor (never a direct fit).
                from .dynamic_qbd_manifested_job_coordinator import execute_candidate_oos_jobs
                execute_candidate_oos_jobs(store=store, inputs=inputs, output_root=store_root,
                                           allowed_job_ids=pending_h3)
            if store.progress().get("pending", 0) == 0:
                break
        # Drain all now-unblocked production nodes after the resumable
        # candidate frontier has settled.
        from .dynamic_qbd_manifested_job_coordinator import build_manifested_job_handlers, execute_ready_jobs
        execute_ready_jobs(store, build_manifested_job_handlers(store=store, inputs=inputs, output_root=store_root),
                           kinds=("candidate_evidence_coverage", "recipe_selection", "model", "calibration",
                                  "prediction", "generation_ready", "replay", "evidence"))
        final_jobs = store.all_jobs()
        assert any(x["kind"] == "evidence" and x["state"] == "COMPLETE" for x in final_jobs), (result, [(x["job_id"], x.get("last_error"), x.get("state")) for x in final_jobs if x["state"] == "FAILED"])
        assert all(x["kind"] != "replay" or x["state"] == "COMPLETE" for x in final_jobs), ([(x["job_id"], x["state"], x.get("depends_on")) for x in final_jobs if x["kind"] == "replay"], [(x["job_id"], x["state"], x.get("last_error")) for x in final_jobs if x["state"] == "FAILED"])
    print("QBD_E2E_FIXTURE_SELF_TEST_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
