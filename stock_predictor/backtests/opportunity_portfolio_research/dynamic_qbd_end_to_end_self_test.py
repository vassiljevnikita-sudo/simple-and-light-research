"""Fast end-to-end contract tests for the Dynamic-QBD factory."""
from __future__ import annotations

from dataclasses import asdict, replace
from datetime import date
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from .cost_contracts import CostModel
from .portfolio_policy_contracts import Policy
from .tax_contracts import TaxConfig
from .contract_fingerprints import stable_hash
from .dynamic_qbd_abc_schedules import compare_abc
from .dynamic_qbd_generation_contracts import DynamicFactoryState, FactoryArm, GenerationStatus, PositionLineage, PRIMARY_FACTORY_ARMS
from .dynamic_qbd_evidence import build_monthly_family_evidence, DOWNSIDE_FEATURES, GENERATION_HEALTH_FEATURES
from .dynamic_qbd_factory import refit_family
from .dynamic_qbd_algorithm_freeze import build_algorithm_manifest, validate_algorithm_manifest
from .dynamic_qbd_gate1 import run_gate1
from .dynamic_qbd_incremental_gates import run_gate2
from .dynamic_qbd_maturity import HorizonMaturityResolver
from .dynamic_qbd_portfolio_replay import replay_family
from .dynamic_qbd_generation_registry import GenerationRegistry
from .dynamic_qbd_family_replay_store import FamilyReplayCheckpointStore, run_checkpointed_family_replay
from .dynamic_qbd_development_evaluation import run_development
from .dynamic_qbd_factory_state_store import DynamicFactoryStateStore
from .dynamic_qbd_family_surface import assert_surface_contract, build_family_specs
from .dynamic_qbd_wealth_metrics import wealth_path_metrics, wealth_path_metrics_at_indices


class Builder:
    def __init__(self, sessions, fail=False, future_mutation=0.0):
        self.sessions, self.fail, self.future_mutation = sessions, fail, future_mutation

    def build(self, *, family, information_cutoff, latest_matured_label_cutoff):
        if self.fail:
            raise RuntimeError("synthetic refit failure")
        usable = tuple(x for x in self.sessions if x <= latest_matured_label_cutoff)
        return {
            "train_start": usable[max(0, len(usable)-family.training_window_sessions)],
            "train_end": usable[-1],
            "dataset_fingerprint": stable_hash((family.family_id, usable)),
            "model_artifact_sha256": stable_hash((family.family_id, usable, family.random_seed)),
            "model_artifact_id": stable_hash(("model", family.family_id, usable)),
            "selected_model_family": "RIDGE",
            "selected_hyperparameters": {"alpha": 1.0},
            "training_recipe_fingerprint": stable_hash(family.training_recipe),
        }

    def calibration_predictions(self, *, family, build, information_cutoff):
        usable = [x for x in self.sessions if x <= information_cutoff]
        rows = []
        for i, decision in enumerate(usable[:-family.horizon_sessions]):
            terminal = usable[i+family.horizon_sessions]
            if terminal <= information_cutoff:
                rows.append({"decision_date": decision, "terminal_date": terminal, "ticker": "A",
                             "observed_excess": .001 + i/1_000_000, "score": .1 + i/10000})
        # A post-cutoff mutation exists but is causally excluded.
        rows.append({"decision_date": pd.Timestamp(information_cutoff)+pd.Timedelta(days=10),
                     "terminal_date": pd.Timestamp(information_cutoff)+pd.Timedelta(days=20), "ticker": "A",
                     "observed_excess": self.future_mutation, "score": self.future_mutation})
        return pd.DataFrame(rows)

    def finalize_generation(self, *, family, build, generation_id):
        return {"prediction_artifact_path": f"{generation_id}.parquet",
                "prediction_artifact_sha256": stable_hash((generation_id, "predictions"))}


def _family(feature="f"*64, horizon=3, holding=2, names=1):
    return next(x for x in build_family_specs(feature_schema_sha256=feature) if
                x.horizon_sessions == horizon and x.holding_days == holding and x.max_names == names)


def _market():
    dates = pd.bdate_range("2020-01-01", periods=45)
    prices = []
    for ticker_index, ticker in enumerate(("URTH", "A", "B", "C", "D", "E", "F")):
        for i, d in enumerate(dates):
            base = 100 + i * (0.1 if ticker == "URTH" else .2 + ticker_index*.01)
            prices.append({"date": d, "ticker": ticker, "open": base, "close": base+.05})
    signals = pd.DataFrame({"decision_date": np.repeat(dates[:-4], 6), "ticker": list("ABCDEF")*(len(dates)-4),
                            "score": np.tile(np.linspace(.9,.4,6), len(dates)-4)})
    return dates, pd.DataFrame(prices), signals


def main() -> int:
    assert_surface_contract()
    sessions = tuple(pd.bdate_range("2018-01-01", periods=800).date)
    maturity = HorizonMaturityResolver(sessions)
    cutoff = sessions[650]
    assert maturity.latest_matured_decision(cutoff, 1) != maturity.latest_matured_decision(cutoff, 30)
    bad = pd.DataFrame({"terminal_date": [cutoff, sessions[651]]})
    try:
        maturity.assert_matured(bad, cutoff)
        raise AssertionError("unmatured row accepted")
    except AssertionError as exc:
        assert "UNMATURED" in str(exc)

    family = _family()
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "generations.json"
        registry = GenerationRegistry(path, (family,))
        first = refit_family(family=family, information_cutoff=cutoff, maturity=maturity,
                             builder=Builder(sessions, future_mutation=999), registry=registry)
        mirror = GenerationRegistry(Path(folder)/"mirror.json", (family,))
        second = refit_family(family=family, information_cutoff=cutoff, maturity=maturity,
                              builder=Builder(sessions, future_mutation=-999), registry=mirror)
        assert first.lifecycle_status == GenerationStatus.VALID
        assert first.generation_id == second.generation_id and first.resolved_threshold == second.resolved_threshold
        assert first.generation_fingerprint == second.generation_fingerprint
        failed = refit_family(family=family, information_cutoff=sessions[670], maturity=maturity,
                              builder=Builder(sessions, fail=True), registry=registry)
        assert failed.lifecycle_status == GenerationStatus.FAILED
        assert registry.current_generation(family.family_id) == first

        state = DynamicFactoryState(as_of=cutoff, family_registry_hash=registry.family_registry_hash,
                                    current_generation_by_family={family.family_id:first.generation_id},
                                    open_position_lineage={"A":PositionLineage(family.family_id,first.generation_id,"ENTRY","FIXED")})
        store = DynamicFactoryStateStore(Path(folder)/"state.json", family_registry_hash=registry.family_registry_hash)
        store.save_atomic(state)
        assert store.load() == state

        continuous=GenerationRegistry(Path(folder)/"continuous.json",(family,))
        for refit_date in (sessions[620],sessions[640],sessions[660]):
            refit_family(family=family,information_cutoff=refit_date,maturity=maturity,builder=Builder(sessions),registry=continuous)
        restarted=GenerationRegistry(Path(folder)/"restarted.json",(family,))
        for refit_date in (sessions[620],sessions[640]):
            refit_family(family=family,information_cutoff=refit_date,maturity=maturity,builder=Builder(sessions),registry=restarted)
        restarted=GenerationRegistry(Path(folder)/"restarted.json",(family,))
        refit_family(family=family,information_cutoff=sessions[660],maturity=maturity,builder=Builder(sessions),registry=restarted)
        assert continuous.records==restarted.records and continuous.current==restarted.current

    dates, prices, signals = _market()
    signals["model_artifact_id"] = first.model_artifact_id
    schedule = pd.DataFrame([{"activation_date":dates[0], "family_id":family.family_id, "generation_id":first.generation_id,
                              "model_artifact_id":first.model_artifact_id,"resolved_threshold":.1,
                              "entry_policy_id":"ENTRY", "exit_policy_id":"FIXED"}])
    common = dict(horizon=3, score_quantile=.5, top_fraction=1.0, holding_days=2, sleeve=.5)
    n1 = replay_family(signals=signals, prices=prices, policy=Policy(max_names=1, **common), generation_schedule=schedule,
                       cost=CostModel(0), tax=TaxConfig(False))
    n6 = replay_family(signals=signals, prices=prices, policy=Policy(max_names=6, **common), generation_schedule=schedule,
                       cost=CostModel(0), tax=TaxConfig(False))
    assert n1["metrics"]["max_positions"] == 1 and n6["metrics"]["max_positions"] == 6
    assert n1["curve"]["accounting_error_eur"].abs().max() < 1e-7
    assert all(t["generation_id"] == first.generation_id and t["family_id"] == family.family_id for t in n1["trades"])
    assert all(x["generation_id"] == first.generation_id for x in n1["open_positions"])
    invalid_prices=prices.assign(open_quality_ok=True,close_quality_ok=True)
    invalid_prices.loc[
        invalid_prices["ticker"].eq("URTH") & invalid_prices["date"].eq(dates[0]),"close_quality_ok"
    ]=False
    try:
        replay_family(signals=signals,prices=invalid_prices,policy=Policy(max_names=1,**common),
                      generation_schedule=schedule,cost=CostModel(0),tax=TaxConfig(False))
        raise AssertionError("invalid benchmark close was accepted")
    except ValueError as exc:
        assert str(exc).startswith("REPLAY_CLOSE_PRICE_BOUNDARY_INVALID:URTH:")
    dividend_dates=pd.bdate_range("2021-01-04",periods=4)
    dividend_prices=pd.DataFrame({"date":dividend_dates,"ticker":"URTH","open":10.0,"close":10.0,
                                  "open_quality_ok":True,"close_quality_ok":True})
    dividend_signals=pd.DataFrame(columns=["decision_date","ticker","score","model_artifact_id"])
    dividend_schedule=schedule.iloc[:1].copy(); dividend_schedule["activation_date"]=dividend_dates[0]
    distributions=pd.DataFrame([{"ticker":"URTH","ex_date":dividend_dates[1],
                                 "payable_date":dividend_dates[2],"cash_amount":1.0}])
    dividend_policy=Policy(horizon=3,score_quantile=.5,top_fraction=1.0,max_names=1,holding_days=2)
    total_return=replay_family(signals=dividend_signals,prices=dividend_prices,policy=dividend_policy,
        generation_schedule=dividend_schedule,cost=CostModel(0),tax=TaxConfig(False),distributions=distributions,
        initial=100.0)
    assert abs(total_return["metrics"]["terminal_value"]-110.0)<1e-10
    assert abs(total_return["metrics"]["urth_terminal_value"]-110.0)<1e-10
    assert abs(total_return["metrics"]["gross_distribution_income_eur"]-10.0)<1e-10
    # Ex-date price drops must be offset immediately by the dividend
    # receivable; payment and reinvestment must not create a later NAV jump.
    ex_drop_prices=dividend_prices.copy()
    ex_drop_prices.loc[ex_drop_prices["date"].ge(dividend_dates[1]),["open","close"]]=9.0
    ex_drop=replay_family(signals=dividend_signals,prices=ex_drop_prices,policy=dividend_policy,
        generation_schedule=dividend_schedule,cost=CostModel(0),tax=TaxConfig(False),distributions=distributions,
        initial=100.0)
    ex_curve=ex_drop["curve"].set_index("date")
    assert abs(ex_curve.loc[dividend_dates[1],"strategy_value"]-100.0)<1e-10
    assert abs(ex_curve.loc[dividend_dates[1],"urth_value"]-100.0)<1e-10
    assert abs(ex_curve.loc[dividend_dates[1],"distribution_receivable_value"]-10.0)<1e-10
    assert abs(ex_curve.loc[dividend_dates[2],"distribution_receivable_value"])<1e-10
    assert abs(ex_curve.loc[dividend_dates[2],"strategy_value"]-100.0)<1e-10
    assert abs(ex_curve.loc[dividend_dates[2],"urth_value"]-100.0)<1e-10
    post_ex=replay_family(signals=dividend_signals,prices=dividend_prices,policy=dividend_policy,
        generation_schedule=dividend_schedule,cost=CostModel(0),tax=TaxConfig(False),
        distributions=distributions,start=dividend_dates[1],initial=100.0)
    assert abs(post_ex["metrics"]["terminal_value"]-100.0)<1e-10,"start-on-ex-date received past entitlement"
    switched_schedule=pd.concat([schedule,pd.DataFrame([{"activation_date":dates[5],"family_id":family.family_id,"generation_id":"GENERATION_2","model_artifact_id":"MODEL_2","resolved_threshold":.1,"entry_policy_id":"ENTRY_2","exit_policy_id":"FIXED_2"}])],ignore_index=True)
    model2=signals.assign(model_artifact_id="MODEL_2",score=1.3-signals["score"])
    switched_signals=pd.concat([signals,model2],ignore_index=True)
    lineage_replay=replay_family(signals=switched_signals,prices=prices,policy=Policy(horizon=10,score_quantile=.5,top_fraction=1.0,max_names=1,holding_days=10),generation_schedule=switched_schedule,cost=CostModel(0),tax=TaxConfig(False))
    spanning=[x for x in lineage_replay["trades"] if pd.Timestamp(x["entry_date"])<dates[5]<pd.Timestamp(x["exit_date"])]
    assert spanning and all(x["generation_id"]==first.generation_id and x["exit_policy_id"]=="FIXED" for x in spanning)
    assert any(x["generation_id"]=="GENERATION_2" and x["ticker"]=="F" for x in lineage_replay["trades"])
    mutation_cutoff=dates[25]
    mutated_signals=signals.copy(); mutated_signals.loc[pd.to_datetime(mutated_signals["decision_date"]).gt(mutation_cutoff),"score"]=-999.0
    original_prefix=replay_family(signals=signals,prices=prices,policy=Policy(max_names=1,**common),generation_schedule=schedule,cost=CostModel(0),tax=TaxConfig(False),end=mutation_cutoff)
    mutated_prefix=replay_family(signals=mutated_signals,prices=prices,policy=Policy(max_names=1,**common),generation_schedule=schedule,cost=CostModel(0),tax=TaxConfig(False),end=mutation_cutoff)
    assert original_prefix["curve"].equals(mutated_prefix["curve"])
    assert original_prefix["trades"] == mutated_prefix["trades"]
    with tempfile.TemporaryDirectory() as folder:
        replay_store=FamilyReplayCheckpointStore(Path(folder)/"replay.json")
        replay_kwargs=dict(signals=signals,prices=prices,policy=Policy(max_names=1,**common),generation_schedule=schedule,cost=CostModel(0),tax=TaxConfig(False))
        run_checkpointed_family_replay(checkpoint_store=replay_store,end=dates[20],**replay_kwargs)
        resumed=run_checkpointed_family_replay(checkpoint_store=replay_store,end=dates[-1],**replay_kwargs)
        pd.testing.assert_frame_equal(resumed["curve"],n1["curve"],check_dtype=False,rtol=0,atol=1e-10)
        checkpoint=replay_store.load()
        assert checkpoint["schema_version"]=="DYNAMIC_QBD_REPLAY_CHECKPOINT_V2"
        assert {"cash","urth_units","positions","pending_orders","tax_ledger","session_cursor"} <= set(checkpoint["replay_state"])
    risk = wealth_path_metrics(n1["curve"])
    assert {"cdar_95", "expected_shortfall_95", "capital_impairment_area"} <= set(risk)
    prefix_risk = wealth_path_metrics_at_indices(n1["curve"], [len(n1["curve"]) - 1])[len(n1["curve"]) - 1]
    for key, value in risk.items():
        assert np.isclose(prefix_risk[key], value, equal_nan=True, rtol=1e-11, atol=1e-12), key

    nav_parts = []
    schedules = []
    for fid, scale in (("F1",1.0),("F2",1.002),("F3",.999)):
        curve = n1["curve"].copy()
        curve["family_id"] = fid
        curve["strategy_value"] *= np.power(scale, np.arange(len(curve)))
        nav_parts.append(curve)
        schedules.append({"family_id":fid,"generation_id":f"{fid}_G1","activation_date":dates[0],"resolved_threshold":.1})
    evidence_schedule=pd.DataFrame(schedules+[{"family_id":"F1","generation_id":"F1_G2","activation_date":dates[30],"resolved_threshold":.2}])
    panel = build_monthly_family_evidence(pd.concat(nav_parts), evidence_schedule)
    assert panel.groupby("family_id").size().min() >= 2
    # Family rows persist through a generation change and do not copy another
    # generation's prediction-health values.
    assert panel["relative_wealth"].notna().all() and panel["rank_ic"].isna().all()
    f1=panel.loc[panel["family_id"].eq("F1")]
    assert set(f1["generation_id"])=={"F1_G1","F1_G2"} and f1["nav"].iloc[-1]!=f1["nav"].iloc[0]
    future_nav=pd.concat(nav_parts).copy(); future_nav.loc[pd.to_datetime(future_nav["date"]).gt(mutation_cutoff),"strategy_value"]*=100
    original_evidence=build_monthly_family_evidence(pd.concat(nav_parts).loc[lambda x:pd.to_datetime(x["date"]).le(mutation_cutoff)],pd.DataFrame(schedules))
    mutated_evidence=build_monthly_family_evidence(future_nav.loc[lambda x:pd.to_datetime(x["date"]).le(mutation_cutoff)],pd.DataFrame(schedules))
    assert original_evidence.equals(mutated_evidence)

    abc_rows=[]
    for month_index, assessment in enumerate(pd.date_range("2020-01-31", periods=18, freq="ME")):
        for fid in ("H03_D02_N01_FIXED","H03_D02_N02_FIXED","H05_D03_N01_FIXED"):
            for arm, value in ((FactoryArm.A_FROZEN_MODEL_FROZEN_CALIBRATION.value,.001),
                               (FactoryArm.B_FROZEN_MODEL_ROLLING_RECALIBRATION.value,.002),
                               (FactoryArm.C_ROLLING_REFIT_ROLLING_RECALIBRATION.value,.003)):
                abc_rows.append({"assessment_date":assessment,"family_id":fid,"arm":arm,"relative_return":value})
    abc = compare_abc(pd.DataFrame(abc_rows))
    assert abc["rolling_recalibration_promoted"] and abc["rolling_refit_promoted"]
    gate_rows=[]
    for month_index,assessment in enumerate(pd.date_range("2018-01-31",periods=36,freq="ME")):
        for family_index,fid in enumerate(("F1","F2","F3")):
            edge=(family_index+1)*.002
            gate_rows.append({"assessment_date":assessment,"family_id":fid,"relative_wealth":1+(month_index+1)*edge,
                              "excess_1m":edge,"excess_3m":edge*3,"excess_6m":edge*6,"excess_12m":edge*12,
                              "ewma_excess":edge,"recent_minus_long_run_excess":edge/2,"positive_period_fraction":.6+family_index*.1})
    gate=run_gate1(pd.DataFrame(gate_rows),minimum_train_months=12)
    assert {"static_best_development_excess","previous_incumbent_held_excess","trailing_6m_winner_excess","oracle_regret","selector_excess_30bps"}<=set(gate.baseline_comparison)
    economic_3m=gate.bootstrap["economic_summary"]["3"]
    baseline_3m=gate.baseline_comparison.loc[gate.baseline_comparison["target_months"].eq(3)]
    assert economic_3m["economic_path_contract"]=="NON_OVERLAPPING_3M_COHORTS"
    assert economic_3m["economic_observations"]==(len(baseline_3m)+2)//3
    gate2_panel=pd.DataFrame(gate_rows)
    for column in DOWNSIDE_FEATURES+GENERATION_HEALTH_FEATURES:
        gate2_panel[column]=np.linspace(.01,.02,len(gate2_panel))
    gate2=run_gate2(gate2_panel,include_downside={1:True,3:False},minimum_train_months=12)
    feature_contract=gate2.bootstrap["feature_contract"]
    assert set(DOWNSIDE_FEATURES)&set(feature_contract["1"]["base"])
    assert not (set(DOWNSIDE_FEATURES)&set(feature_contract["3"]["base"]))
    assert "matured_prediction_count" not in set(feature_contract["1"]["incremental"])
    assert "top_vs_median_realised_excess" in set(feature_contract["1"]["incremental"])
    gate_cutoff=pd.Timestamp("2020-06-30"); mutated_gate_rows=pd.DataFrame(gate_rows)
    mutated_gate_rows.loc[pd.to_datetime(mutated_gate_rows["assessment_date"]).gt(gate_cutoff),"relative_wealth"]*=50
    mutated_gate=run_gate1(mutated_gate_rows,minimum_train_months=12)
    causal_columns=["assessment_date","family_id","target_months","predicted_excess"]
    left=gate.predictions.loc[pd.to_datetime(gate.predictions["assessment_date"]).le(gate_cutoff),causal_columns].reset_index(drop=True)
    right=mutated_gate.predictions.loc[pd.to_datetime(mutated_gate.predictions["assessment_date"]).le(gate_cutoff),causal_columns].reset_index(drop=True)
    assert left.equals(right), "future mutation changed pre-cutoff selector predictions"

    with tempfile.TemporaryDirectory() as folder:
        arm_schedule=[]
        for arm in PRIMARY_FACTORY_ARMS:
            arm_schedule.append({"family_id":family.family_id,"arm":arm.value,"generation_id":first.generation_id,
                                 "model_artifact_id":first.model_artifact_id,"activation_date":dates[0],
                                 "resolved_threshold":.1,"entry_policy_id":"ENTRY","exit_policy_id":"FIXED"})
        research_predictions=signals.copy()
        research=run_development(families=(family,),predictions=research_predictions,prices=prices,
                                 generation_schedule=pd.DataFrame(arm_schedule),output_root=folder,
                                 holdout_contract="PROSPECTIVE_FROM_2026_07_25")
        assert research["status"]=="DYNAMIC_QBD_DEVELOPMENT_COMPLETE"
        assert (Path(folder)/"monthly_family_evidence.parquet").is_file()
        try:
            run_development(families=(family,),predictions=research_predictions.assign(decision_date=date(2026,7,25)),prices=prices,
                            generation_schedule=pd.DataFrame(arm_schedule),output_root=Path(folder)/"forbidden",
                            holdout_contract="PROSPECTIVE_FROM_2026_07_25")
            raise AssertionError("final holdout opened")
        except PermissionError as exc:
            assert "FINAL_HOLDOUT_LOCKED" in str(exc)

    manifest = build_algorithm_manifest(code_commit="TEST", family_registry_hash="x", candidate_design_space="H1-H30,D1-H,N1-N6",
        feature_schema="f", training_recipe="r", hyperparameter_rule="h", training_window_rule="504 sessions",
        refit_cadence="MONTH_END", horizon_maturity_rule="terminal_date<=cutoff", calibration_window_rule="252 sessions",
        recalibration_algorithm="daily-top quantile", threshold_rule="q=.975", exit_runtime_rule="next-open",
        benchmark_contract="URTH", cost_contract="20bps", tax_contract="DE", evidence_panel_schema="V1",
        gate_rules="GATE1_ONLY_BEFORE_SELECTOR", final_holdout_boundaries=("LOCKED","LOCKED"), initial_pre_holdout_state_hash="state")
    validate_algorithm_manifest(manifest)
    print("DYNAMIC_QBD_FACTORY_SELF_TEST_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
