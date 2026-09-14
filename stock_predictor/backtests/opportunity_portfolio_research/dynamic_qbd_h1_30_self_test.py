"""Fast integration regression for the real H1-H30 generation adapter."""
from __future__ import annotations

from datetime import date
import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from .dynamic_qbd_h1_30_adapter import FEATURE_COLUMNS, H130ProductionGenerationBuilder, ParquetH130DatasetMaterializer
from .dynamic_qbd_maturity import HorizonMaturityResolver
from .dynamic_qbd_development_pipeline import run_pipeline
from .dynamic_qbd_portfolio_replay import replay_family
from .dynamic_qbd_family_replay_store import FamilyReplayCheckpointStore, run_checkpointed_family_replay
from .dynamic_qbd_family_surface import build_family_specs
from .cost_contracts import CostModel
from .portfolio_policy_contracts import Policy
from .tax_contracts import TaxConfig
from .learned_exit_qbd_profit import ProfitTaxConfig
from .learned_exit_qbd_provider import LearnedExitProvider
from .learned_exit_qbd_replay import configure_replay
from .dynamic_qbd_runtime_resources import cpu_budget


def main() -> int:
    assert cpu_budget(.90,16)=={"logical_processors":16,"target_fraction":.90,
        "target_logical_processors":16,"reserve_logical_processors":0,
        "available_capacity_fraction":1.0,"capacity_fraction_enforced":False}
    rng = np.random.default_rng(17)
    sessions = pd.bdate_range("2018-01-01", periods=820)
    tickers = ("A", "B", "C")
    rows = []
    for day_index, session in enumerate(sessions):
        for ticker_index, ticker in enumerate(tickers):
            features = {name: float(rng.normal()) for name in FEATURE_COLUMNS}
            target = .002 * features["mom20"] + .0001 * ticker_index + float(rng.normal(0, .001))
            rows.append({"decision_date": session, "ticker": ticker, "sector": "S", "sub_industry": "I",
                         "holdout_locked": False, **features,
                         "net_excess_return_3__BASELINE_20_BPS": target,
                         "gross_excess_return_1":target*.5})
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        panel_frame=pd.DataFrame(rows)
        panel = root/"signal-panel.parquet"; panel_frame.to_parquet(panel, index=False)
        metrics = []
        for fold, end in enumerate(("2019-06-28", "2019-12-31", "2020-06-30")):
            for identifier, family, spearman, parameters in (
                ("ridge", "RIDGE", .20+.01*fold, {"alpha":1.0}),
                ("hgb", "HIST_GRADIENT_BOOSTING", .90 if fold==2 else -.10, {"learning_rate":.05,"max_iter":20,"max_leaf_nodes":7,"l2_regularization":1.0}),
            ):
                metrics.append({"candidate_id":identifier,"family":family,"fold_id":f"WF_{fold:03d}_2019-01-01_{end}",
                                "horizon_sessions":3,"selection_only":False,"parameters":parameters,
                                "metrics":{"spearman":spearman,"mae":.01,"rmse":.02,"r2":0,"sign_accuracy":.5}})
        metric_path=root/"metrics.json"; metric_path.write_text(json.dumps(metrics),encoding="utf-8")
        exit_metrics=[{**row,"horizon_sessions":1,"target":"gross_excess_return_1"} for row in metrics]
        exit_metric_path=root/"exit-metrics.json"; exit_metric_path.write_text(json.dumps(exit_metrics),encoding="utf-8")
        materializer=ParquetH130DatasetMaterializer(panel,metric_path,sessions[-1].date())
        family = next(x for x in build_family_specs(feature_schema_sha256=materializer.feature_schema_fingerprint,
                          training_window_sessions=252,calibration_window_sessions=63)
                      if x.horizon_sessions==3 and x.holding_days==2 and x.max_names==1)
        maturity=HorizonMaturityResolver(sessions)
        cutoff=sessions[530].date(); latest=maturity.latest_matured_decision(cutoff,3)
        builder=H130ProductionGenerationBuilder(materializer=materializer,
                                                root=root/"factory",code_commit="TEST")
        build=builder.build(family=family,information_cutoff=cutoff,latest_matured_label_cutoff=latest)
        assert build["selected_model_family"]=="RIDGE" and build["selected_hyperparameters"]=={"alpha":1.0}
        selection=build["selection_evidence"]
        assert selection["model_combination_contract"]=="SINGLE_SELECTED_RECIPE_NO_ENSEMBLE_NO_MODEL_INTERSECTION"
        assert "R2_DIAGNOSTIC_ONLY" in selection["selection_metric_contract"]
        assert selection["fold_count"]==len(selection["fold_ids"])>=2
        assert all(pd.Timestamp(fold_id.rsplit("_",1)[-1]).date()<=latest for fold_id in selection["fold_ids"])
        r2_mutated=[]
        for row in metrics:
            changed=json.loads(json.dumps(row)); changed["metrics"]["r2"]=999.0 if changed["family"]=="HIST_GRADIENT_BOOSTING" else -999.0
            r2_mutated.append(changed)
        r2_path=root/"r2-mutated-metrics.json"; r2_path.write_text(json.dumps(r2_mutated),encoding="utf-8")
        r2_builder=H130ProductionGenerationBuilder(
            materializer=ParquetH130DatasetMaterializer(panel,r2_path,sessions[-1].date()),
            root=root/"r2-mutated-factory",code_commit="TEST")
        r2_choice=r2_builder._select_candidate(family,r2_path,latest)
        assert r2_choice["candidate_id"]==selection["candidate_id"], "R2 changed the model-selection winner"
        calibration=builder.calibration_predictions(family=family,build=build,information_cutoff=cutoff)
        assert {"ticker","observed_excess","score","terminal_date"}<=set(calibration)
        assert pd.to_datetime(calibration["terminal_date"]).le(pd.Timestamp(cutoff)).all()
        finalized=builder.finalize_generation(family=family,build=build,generation_id="GEN_A")
        predictions=pd.read_parquet(finalized["prediction_artifact_path"])
        assert set(predictions["model_artifact_id"])=={build["model_artifact_id"]} and predictions["decision_date"].min()>pd.Timestamp(cutoff)
        mutation_date=sessions[750]
        mutated=panel_frame.copy(); future=pd.to_datetime(mutated["decision_date"]).gt(mutation_date)
        mutated.loc[future,"mom20"]+=999.0
        mutated.loc[future,"net_excess_return_3__BASELINE_20_BPS"]-=999.0
        mutated_panel=root/"mutated-panel.parquet"; mutated.to_parquet(mutated_panel,index=False)
        mutated_materializer=ParquetH130DatasetMaterializer(mutated_panel,metric_path,sessions[-1].date())
        mutated_builder=H130ProductionGenerationBuilder(materializer=mutated_materializer,root=root/"mutated-factory",code_commit="TEST")
        mutated_build=mutated_builder.build(family=family,information_cutoff=cutoff,latest_matured_label_cutoff=latest)
        assert build["dataset_fingerprint"]==mutated_build["dataset_fingerprint"]
        assert build["model_artifact_id"]==mutated_build["model_artifact_id"]
        assert build["model_artifact_sha256"]==mutated_build["model_artifact_sha256"]
        mutated_final=mutated_builder.finalize_generation(family=family,build=mutated_build,generation_id="GEN_A")
        mutated_predictions=pd.read_parquet(mutated_final["prediction_artifact_path"])
        prefix_columns=["decision_date","ticker","score","model_artifact_id"]
        left=predictions.loc[pd.to_datetime(predictions["decision_date"]).le(mutation_date),prefix_columns].reset_index(drop=True)
        right=mutated_predictions.loc[pd.to_datetime(mutated_predictions["decision_date"]).le(mutation_date),prefix_columns].reset_index(drop=True)
        pd.testing.assert_frame_equal(left,right)
        learned_materializer=ParquetH130DatasetMaterializer(panel,metric_path,sessions[-1].date(),exit_metric_path)
        learned_family=next(x for x in build_family_specs(feature_schema_sha256=learned_materializer.feature_schema_fingerprint,
                            training_window_sessions=252,calibration_window_sessions=63,include_learned_exit=True)
                            if x.horizon_sessions==3 and x.holding_days==2 and x.max_names==1 and x.exit_policy["family"]=="LEARNED_EXIT")
        learned_builder=H130ProductionGenerationBuilder(materializer=learned_materializer,root=root/"exit-factory",code_commit="TEST")
        learned_build=learned_builder.build(family=learned_family,information_cutoff=cutoff,
                                            latest_matured_label_cutoff=latest)
        learned_final=learned_builder.finalize_generation(family=learned_family,build=learned_build,generation_id="GEN_EXIT")
        exit_predictions=pd.read_parquet(learned_final["exit_prediction_artifact_path"])
        assert set(exit_predictions["exit_generation_id"])=={learned_final["exit_generation_id"]}
        assert learned_final["exit_model_artifact_sha256"] and learned_final["exit_calibration_fingerprint"]
        price_rows=[]
        for ticker_index,ticker in enumerate(("URTH",*tickers)):
            for day_index,session in enumerate(sessions):
                value=100+day_index*(.02 if ticker=="URTH" else .03+ticker_index*.001)
                price_rows.append({"date":session,"ticker":ticker,"open":value,"close":value+.01})
        price_frame=pd.DataFrame(price_rows)
        price_path=root/"prices.parquet"; price_frame.to_parquet(price_path,index=False)
        learned_signals=pd.read_parquet(learned_final["prediction_artifact_path"])
        learned_schedule=pd.DataFrame([{"activation_date":cutoff,"family_id":learned_family.family_id,
            "generation_id":"GEN_EXIT","model_artifact_id":learned_build["model_artifact_id"],
            "resolved_threshold":float(learned_build.get("resolved_threshold",-1e9)),
            "resolved_top_fraction":1.0,"entry_policy_id":"ENTRY_EXIT",
            "exit_policy_id":learned_final["exit_generation_id"],
            "exit_generation_id":learned_final["exit_generation_id"]}])
        configure_replay(LearnedExitProvider(learned_final["exit_prediction_artifact_path"]),ProfitTaxConfig(enabled=False))
        learned_policy=Policy(horizon=3,score_quantile=.5,top_fraction=1.0,max_names=1,holding_days=2,
                              exit_family="LEARNED_EXIT")
        learned_kwargs=dict(signals=learned_signals,prices=price_frame,policy=learned_policy,
                            generation_schedule=learned_schedule,cost=CostModel(0),tax=TaxConfig(False),
                            start=pd.Timestamp(cutoff))
        continuous_learned=replay_family(end=sessions[760],**learned_kwargs)
        checkpoint=FamilyReplayCheckpointStore(root/"learned-replay-checkpoint.json")
        run_checkpointed_family_replay(checkpoint_store=checkpoint,end=sessions[650],**learned_kwargs)
        resumed_learned=run_checkpointed_family_replay(checkpoint_store=checkpoint,end=sessions[760],**learned_kwargs)
        pd.testing.assert_frame_equal(resumed_learned["curve"],continuous_learned["curve"],check_dtype=False,
                                      rtol=0,atol=1e-10)
        assert resumed_learned["trades"]==continuous_learned["trades"]
        checkpoint_payload=checkpoint.load()
        assert checkpoint_payload["schema_version"]=="DYNAMIC_QBD_REPLAY_CHECKPOINT_V2"
        assert {"cash","urth_units","positions","pending_orders","tax_ledger","session_cursor"} <= set(checkpoint_payload["replay_state"])
        pipeline=run_pipeline(signal_panel=panel,candidate_metrics=metric_path,prices=price_path,
                              output_root=root/"pipeline",family_ids=[family.family_id],
                              start=sessions[500].date(),end=sessions[760].date(),
                              holdout_contract="PROSPECTIVE_FROM_2026_07_25",feature_schema_sha256="AUTO",
                              training_window_sessions=252,calibration_window_sessions=63,
                              include_policy_optimization_arms=True)
        assert pipeline["result"]["status"]=="DYNAMIC_QBD_DEVELOPMENT_COMPLETE"
        assert (root/"pipeline"/"model_predictions.parquet").is_file()
        assert (root/"pipeline"/"manifest.json").is_file() and (root/"pipeline"/"REPORT.md").is_file()
        for directory in ("generations","predictions","calibration","portfolio_paths","abc","evidence",
                          "gate1","gate1b","gate2","gate3","freeze"):
            assert (root/"pipeline"/directory).is_dir(), f"missing structured artifact directory: {directory}"
        schedule=pd.read_parquet(root/"pipeline"/"abc_generation_schedule.parquet")
        assert {"B2_FROZEN_MODEL_ROLLING_POLICY_AND_RECALIBRATION",
                "C2_ROLLING_REFIT_ROLLING_POLICY_AND_RECALIBRATION"} <= set(schedule["arm"])
        pipeline_generations=pd.read_parquet(root/"pipeline"/"valid_generations.parquet")
        assert set(pipeline_generations["selected_model_family"])=={"RIDGE"}, "monthly refit changed frozen model recipe"
        resumed=run_pipeline(signal_panel=panel,candidate_metrics=metric_path,prices=price_path,
                             output_root=root/"pipeline",family_ids=[family.family_id],
                             start=sessions[500].date(),end=sessions[760].date(),
                             holdout_contract="PROSPECTIVE_FROM_2026_07_25",feature_schema_sha256="AUTO",
                             training_window_sessions=252,calibration_window_sessions=63,
                             include_policy_optimization_arms=True)
        assert resumed["pipeline_fingerprint"]==pipeline["pipeline_fingerprint"]
        assert resumed["completed_run_reused"] and resumed["verified_artifact_count"]>0
    print("DYNAMIC_QBD_H1_30_ADAPTER_SELF_TEST_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
