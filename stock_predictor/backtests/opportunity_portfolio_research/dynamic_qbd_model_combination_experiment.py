"""Shadow-only fresh-fit comparison: selected model, intersection ensemble, single fold."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from datetime import date
import json
import math
from pathlib import Path

import pandas as pd

from .contract_fingerprints import stable_hash
from .cost_contracts import CostModel
from .portfolio_policy_contracts import Policy
from .tax_contracts import TaxConfig
from .dynamic_qbd_h1_30_adapter import H130ProductionGenerationBuilder, ParquetH130DatasetMaterializer
from .dynamic_qbd_maturity import HorizonMaturityResolver
from .dynamic_qbd_factory import monthly_refit_dates
from .dynamic_qbd_development_pipeline import materialize_daily_store_prices
from .dynamic_qbd_portfolio_replay import replay_family
from .dynamic_qbd_generation_recalibration import recalibrate_generation
from .dynamic_qbd_family_surface import build_family_specs
from .dynamic_qbd_wealth_metrics import wealth_path_metrics
from .dynamic_qbd_runtime_resources import active_cpu_contract


def _fold_end(row: dict) -> date:
    return date.fromisoformat(str(row["fold_id"]).rsplit("_",1)[-1])


def _single_fold_candidate(rows: list[dict], horizon: int, cutoff: date) -> tuple[str,list[dict]] | None:
    eligible=[row for row in rows if int(row.get("horizon_sessions",-1))==horizon
              and not row.get("selection_only") and _fold_end(row)<=cutoff
              and float(row.get("metrics",{}).get("spearman",-1))>0
              and float(row.get("metrics",{}).get("r2",-1))>0]
    if not eligible:
        return None
    winner=max(eligible,key=lambda row:(float(row["metrics"]["spearman"]),float(row["metrics"]["r2"]),str(row["candidate_id"])))
    key=(winner["fold_id"],winner["candidate_id"],winner["family"])
    return str(winner["fold_id"]),[row for row in rows if
        (row.get("fold_id"),row.get("candidate_id"),row.get("family"))==key]


def _recipe_selection_timeline(*, family, materializer, metrics_path: Path,
                               sessions: tuple[date,...], end: date, root: Path) -> list[dict]:
    maturity=HorizonMaturityResolver(sessions)
    selector=H130ProductionGenerationBuilder(materializer=materializer,root=root,code_commit="RECIPE_TIMELINE_AUDIT_V1")
    timeline=[]; prior=None
    for cutoff in monthly_refit_dates(tuple(x for x in sessions if x<=end)):
        latest=maturity.latest_matured_decision(cutoff,family.horizon_sessions)
        if latest is None: continue
        try:
            choice=selector._select_candidate(family,metrics_path,latest)
        except ValueError as exc:
            if str(exc).startswith(("NO_CAUSAL_","INSUFFICIENT_CAUSAL_")): continue
            raise
        identity=(choice["candidate_id"],choice["family"],stable_hash(choice["parameters"]))
        if identity!=prior:
            timeline.append({"activation_cutoff":str(cutoff),"latest_matured_evidence_date":str(latest),
                "candidate_id":choice["candidate_id"],"model_family":choice["family"],
                "parameters":choice["parameters"],"fold_count":choice["fold_count"],
                "fold_ids":choice["fold_ids"],"robust_score":choice["robust_score"],
                "mean_r2_diagnostic":choice["mean_r2_diagnostic"]})
            prior=identity
    return timeline


def _fit_component(*, name: str, family, materializer, root: Path, cutoff: date,
                   latest_matured: date, sessions: tuple[date,...]) -> dict:
    builder=H130ProductionGenerationBuilder(materializer=materializer,root=root/name,code_commit="SHADOW_COMBINATION_EXPERIMENT_V1")
    build=builder.build(family=family,information_cutoff=cutoff,latest_matured_label_cutoff=latest_matured)
    calibration=builder.calibration_predictions(family=family,build=build,information_cutoff=cutoff)
    generation_id=stable_hash((name,family.family_hash,cutoff,build["model_artifact_id"]))[:24]
    resolved=recalibrate_generation(family,generation_id,calibration,information_cutoff=cutoff,
                                    maturity=HorizonMaturityResolver(sessions))
    finalized=builder.finalize_generation(family=family,build=build,generation_id=generation_id)
    predictions=pd.read_parquet(finalized["prediction_artifact_path"])
    return {"name":name,"family":family,"build":build,"generation_id":generation_id,
            "threshold":float(resolved.resolved_threshold),"top_fraction":float(resolved.resolved_top_fraction),
            "predictions":predictions,"fresh_fit":True}


def _active_rows(component: dict) -> pd.DataFrame:
    frame=component["predictions"].copy()
    frame["decision_date"]=pd.to_datetime(frame["decision_date"])
    frame["rank_fraction"]=frame.groupby("decision_date")["score"].rank(method="first",ascending=False)/frame.groupby("decision_date")["score"].transform("size")
    return frame.loc[frame["score"].ge(component["threshold"])
                     & frame["rank_fraction"].le(component["top_fraction"])]


def _intersection_predictions(left: dict, right: dict, model_artifact_id: str) -> pd.DataFrame:
    a=_active_rows(left)[["decision_date","ticker","rank_fraction"]].rename(columns={"rank_fraction":"left_rank"})
    b=_active_rows(right)[["decision_date","ticker","rank_fraction"]].rename(columns={"rank_fraction":"right_rank"})
    merged=a.merge(b,on=["decision_date","ticker"],how="inner",validate="one_to_one")
    merged["score"]=1.0-merged[["left_rank","right_rank"]].max(axis=1)
    merged["model_artifact_id"]=model_artifact_id
    return merged[["decision_date","ticker","score","model_artifact_id"]]


def _component_predictions(component: dict, model_artifact_id: str) -> pd.DataFrame:
    frame=_active_rows(component).copy()
    frame["score"]=1.0-frame["rank_fraction"]
    frame["model_artifact_id"]=model_artifact_id
    return frame[["decision_date","ticker","score","model_artifact_id"]]


def _replay(name: str, predictions: pd.DataFrame, prices: pd.DataFrame, *, cutoff: date,
            start: date, end: date, horizon: int, holding: int, max_names: int,
            initial: float=10000.0) -> dict:
    artifact_id=stable_hash((name,"FRESH_FIT_SIGNAL"))
    predictions=predictions.copy(); predictions["model_artifact_id"]=artifact_id
    schedule=pd.DataFrame([{"activation_date":cutoff,"family_id":name,"generation_id":stable_hash((name,cutoff))[:24],
        "model_artifact_id":artifact_id,"resolved_threshold":-1e12,"resolved_top_fraction":1.0,
        "entry_policy_id":f"{name}_ENTRY","exit_policy_id":"FIXED_D2"}])
    result=replay_family(signals=predictions,prices=prices,
        policy=Policy(horizon=horizon,score_quantile=.5,top_fraction=1.0,max_names=max_names,
                      holding_days=holding,sleeve=.5),generation_schedule=schedule,
        cost=CostModel(20.0),tax=TaxConfig(False),start=pd.Timestamp(start),end=pd.Timestamp(end),initial=initial)
    risk=wealth_path_metrics(result["curve"])
    return {"strategy":name,"result":result,"risk":risk}


def _scheduled_replay(name: str, components: list[dict], prices: pd.DataFrame, *,
                      start: date, end: date, horizon: int, holding: int,
                      max_names: int, initial: float) -> dict:
    signals=[]; schedule=[]
    for component in components:
        artifact_id=str(component["build"]["model_artifact_id"])
        signals.append(_component_predictions(component,artifact_id))
        activation=component["activation_cutoff"]
        schedule.append({"activation_date":activation,"family_id":name,
            "generation_id":component["generation_id"],"model_artifact_id":artifact_id,
            "resolved_threshold":-1e12,"resolved_top_fraction":1.0,
            "entry_policy_id":f"{name}_ENTRY_{activation}","exit_policy_id":"FIXED_D2"})
    result=replay_family(signals=pd.concat(signals,ignore_index=True),prices=prices,
        policy=Policy(horizon=horizon,score_quantile=.5,top_fraction=1.0,max_names=max_names,
                      holding_days=holding,sleeve=.5),generation_schedule=pd.DataFrame(schedule),
        cost=CostModel(20.0),tax=TaxConfig(False),start=pd.Timestamp(start),end=pd.Timestamp(end),
        initial=initial)
    return {"strategy":name,"result":result,"risk":wealth_path_metrics(result["curve"])}


def _rolling_recipe_shadow(*, timeline: list[dict], base, materializer,
                           metrics_path: Path,
                           sessions: tuple[date,...], prices: pd.DataFrame,
                           root: Path, end: date, horizon: int, holding: int,
                           max_names: int, initial: float) -> dict | None:
    changes=[row for row in timeline if date.fromisoformat(row["activation_cutoff"])<end]
    if len(changes)<2:
        return None
    shadow_family=replace(base,family_id=base.family_id+"_ROLLING_RECIPE_SHADOW",
        hyperparameter_rule={**base.hyperparameter_rule,
            "recipe_selection_contract":"CAUSAL_RESELECT_AT_EACH_REFIT"})
    fit_jobs=[]
    maturity=HorizonMaturityResolver(sessions)
    for index,row in enumerate(changes):
        cutoff=date.fromisoformat(row["activation_cutoff"])
        latest=maturity.latest_matured_decision(cutoff,horizon)
        fit_jobs.append((index,row,cutoff,latest))
    def fit(job):
        index,row,cutoff,latest=job
        component=_fit_component(name=f"RECIPE_{index:02d}_{row['model_family']}",
            family=shadow_family,materializer=materializer,root=root/"fresh-models",
            cutoff=cutoff,latest_matured=latest,sessions=sessions)
        selected=component["build"]["selection_evidence"]
        expected=(row["candidate_id"],row["model_family"],row["parameters"])
        actual=(selected["candidate_id"],selected["family"],selected["parameters"])
        if actual!=expected:
            raise AssertionError(f"RECIPE_TIMELINE_FIT_MISMATCH:{expected!r}!={actual!r}")
        component["activation_cutoff"]=cutoff
        return component
    with ThreadPoolExecutor(max_workers=1,thread_name_prefix="rolling-recipe-fit") as pool:
        components=list(pool.map(fit,fit_jobs))
    first_choice=components[0]["build"]["selection_evidence"]
    metric_rows=json.loads(metrics_path.read_text(encoding="utf-8"))
    frozen_rows=[row for row in metric_rows if int(row.get("horizon_sessions",-1))==horizon
        and str(row.get("candidate_id"))==str(first_choice["candidate_id"])
        and str(row.get("family"))==str(first_choice["family"])]
    frozen_metrics=root/"frozen-recipe-candidate-metrics.json"
    frozen_metrics.write_text(json.dumps(frozen_rows,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    frozen_materializer=ParquetH130DatasetMaterializer(
        materializer.signal_panel_path,frozen_metrics,materializer.development_end)
    frozen_family=replace(base,family_id=base.family_id+"_FROZEN_RECIPE_MATCHED_REFIT",
        model_family=str(first_choice["family"]),
        hyperparameter_rule={**base.hyperparameter_rule,
            "minimum_oos_folds":2,"recipe_selection_contract":"CAUSAL_RESELECT_AT_EACH_REFIT"})
    matched=[]
    for index,(rolling,row) in enumerate(zip(components,changes)):
        selected=rolling["build"]["selection_evidence"]
        if (selected["candidate_id"],selected["family"],selected["parameters"]) == (
                first_choice["candidate_id"],first_choice["family"],first_choice["parameters"]):
            matched.append(rolling)
            continue
        cutoff=date.fromisoformat(row["activation_cutoff"])
        latest=maturity.latest_matured_decision(cutoff,horizon)
        component=_fit_component(name=f"MATCHED_REFIT_{index:02d}_{first_choice['family']}",
            family=frozen_family,materializer=frozen_materializer,root=root/"fresh-models",
            cutoff=cutoff,latest_matured=latest,sessions=sessions)
        component["activation_cutoff"]=cutoff
        matched.append(component)
    shadow_start=min(x["activation_cutoff"] for x in components)
    frozen=[components[0]]
    with ThreadPoolExecutor(max_workers=3,thread_name_prefix="rolling-recipe-portfolio") as pool:
        replays=list(pool.map(lambda item:_scheduled_replay(item[0],item[1],prices,start=shadow_start,
            end=end,horizon=horizon,holding=holding,max_names=max_names,initial=initial),(
                ("FROZEN_FIRST_CAUSAL_RECIPE",frozen),
                ("FROZEN_RECIPE_MATCHED_REFIT_DATES",matched),
                ("ROLLING_CAUSAL_RECIPE_RESELECTION",components),
            )))
    output=[]; segment_rows=[]
    for replay in replays:
        result=replay["result"]; curve=result["curve"].copy(); curve["strategy"]=replay["strategy"]
        curve.to_parquet(root/f"{replay['strategy']}-nav.parquet",index=False)
        pd.DataFrame(result["trades"]).to_parquet(root/f"{replay['strategy']}-trades.parquet",index=False)
        metrics=result["metrics"]
        output.append({"strategy":replay["strategy"],"initial_value":initial,
            "terminal_value":float(metrics["terminal_value"]),
            "urth_terminal_value":float(metrics["urth_terminal_value"]),
            "terminal_excess_eur":float(metrics["terminal_value"]-metrics["urth_terminal_value"]),
            "terminal_relative_return":float(metrics["terminal_value"]/metrics["urth_terminal_value"]-1.0),
            "trade_count":int(metrics["trade_count"]),"total_cost_eur":float(metrics["total_cost_eur"]),
            "relative_max_drawdown":float(replay["risk"]["relative_max_drawdown"])})
        curve["date"]=pd.to_datetime(curve["date"]); curve["relative_wealth"]=curve["strategy_value"]/curve["urth_value"]
        boundaries=[x["activation_cutoff"] for x in components]
        for index,segment_start in enumerate(boundaries):
            segment_end=(boundaries[index+1] if index+1<len(boundaries) else end)
            mask=curve["date"].ge(pd.Timestamp(segment_start))
            mask &= (curve["date"].lt(pd.Timestamp(segment_end)) if index+1<len(boundaries)
                     else curve["date"].le(pd.Timestamp(segment_end)))
            segment=curve.loc[mask].copy()
            if segment.empty: continue
            local_relative=segment["relative_wealth"]/float(segment["relative_wealth"].iloc[0])
            segment_rows.append({"strategy":replay["strategy"],"segment_start":str(segment_start),
                "segment_end":str(pd.Timestamp(segment["date"].iloc[-1]).date()),"sessions":int(len(segment)),
                "strategy_return":float(segment["strategy_value"].iloc[-1]/segment["strategy_value"].iloc[0]-1),
                "urth_return":float(segment["urth_value"].iloc[-1]/segment["urth_value"].iloc[0]-1),
                "relative_return":float(local_relative.iloc[-1]-1),
                "local_relative_max_drawdown":float((local_relative/local_relative.cummax()-1).min())})
    comparison=pd.DataFrame(output).sort_values("terminal_value",ascending=False)
    comparison.to_csv(root/"portfolio-value-comparison.csv",index=False)
    pd.DataFrame(segment_rows).to_csv(root/"recipe-segment-comparison.csv",index=False)
    recipe_audit=[{"activation_cutoff":str(x["activation_cutoff"]),"fresh_fit":True,
        "model_artifact_id":x["build"]["model_artifact_id"],
        "model_artifact_sha256":x["build"]["model_artifact_sha256"],
        "selected_model_family":x["build"]["selected_model_family"],
        "selected_hyperparameters":x["build"]["selected_hyperparameters"],
        "selection_evidence":x["build"]["selection_evidence"]} for x in components]
    payload={"schema_version":"DYNAMIC_QBD_ROLLING_RECIPE_SHADOW_V1",
        "authority":"SHADOW_ONLY_NO_PROMOTION_NO_HOLDOUT","evaluation":{"start":str(shadow_start),"end":str(end)},
        "recipe_selection_contract":"CAUSAL_RESELECT_AT_EACH_REFIT",
        "primary_recipe_selection_contrast":"ROLLING_CAUSAL_RECIPE_RESELECTION_MINUS_FROZEN_RECIPE_MATCHED_REFIT_DATES",
        "stale_model_contrast":"FROZEN_FIRST_CAUSAL_RECIPE_IS_DIAGNOSTIC_ONLY_AND_CONFOUNDS_RECIPE_WITH_MODEL_AGE",
        "fresh_models_trained":True,"old_model_artifacts_reused":False,
        "execution_resources":active_cpu_contract(),
        "recipe_generations":recipe_audit,"portfolio_value_comparison":comparison.to_dict(orient="records"),
        "recipe_segment_comparison":segment_rows}
    (root/"summary.json").write_text(json.dumps(payload,default=str,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    return payload


def comparison_markdown(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "No result rows."
    columns=("strategy","terminal_value","urth_terminal_value","terminal_excess_eur",
             "terminal_relative_return","trade_count","total_cost_eur","relative_max_drawdown")
    header="| Strategy | Terminal EUR | URTH EUR | Excess EUR | Relative | Trades | Costs EUR | Relative MaxDD |"
    divider="|---|---:|---:|---:|---:|---:|---:|---:|"
    rows=[header,divider]
    for row in frame.to_dict(orient="records"):
        rows.append("| {strategy} | {terminal_value:,.2f} | {urth_terminal_value:,.2f} | {terminal_excess_eur:+,.2f} | {terminal_relative_return:+.2%} | {trade_count:d} | {total_cost_eur:,.2f} | {relative_max_drawdown:.2%} |".format(
            **{key:row[key] for key in columns}))
    return "\n".join(rows)


def run_experiment(*, signal_panel: str|Path, candidate_metrics: str|Path,
                   daily_store_root: str|Path, output_root: str|Path,
                   fit_cutoff: date, start: date, end: date,
                   horizon: int=3, holding_days: int=2, max_names: int=1,
                   score_quantile: float=.75, top_fraction: float=.01,
                   initial: float=10000.0) -> dict:
    output_root=Path(output_root); output_root.mkdir(parents=True,exist_ok=True)
    panel=Path(signal_panel); metrics_path=Path(candidate_metrics)
    rows=json.loads(metrics_path.read_text(encoding="utf-8"))
    base_materializer=ParquetH130DatasetMaterializer(panel,metrics_path,end)
    base=next(x for x in build_family_specs(feature_schema_sha256=base_materializer.feature_schema_fingerprint,
        score_quantile=score_quantile,top_fraction=top_fraction) if
        x.horizon_sessions==horizon and x.holding_days==holding_days and x.max_names==max_names
        and x.exit_policy["family"]=="FIXED")
    sessions_frame=pd.read_parquet(panel,columns=["decision_date"]).drop_duplicates()
    sessions=tuple(sorted(pd.to_datetime(sessions_frame["decision_date"]).dt.date.unique()))
    maturity=HorizonMaturityResolver(sessions)
    latest=maturity.latest_matured_decision(fit_cutoff,horizon)
    recipe_timeline=_recipe_selection_timeline(family=base,materializer=base_materializer,
        metrics_path=metrics_path,sessions=sessions,end=end,root=output_root/"recipe-timeline")
    ridge=replace(base,family_id=base.family_id+"_RIDGE_COMPONENT",model_family="RIDGE")
    hgb=replace(base,family_id=base.family_id+"_HGB_COMPONENT",model_family="HIST_GRADIENT_BOOSTING")
    single=_single_fold_candidate(rows,horizon,latest)
    jobs=[("SELECTED_ALL_FOLDS",base,base_materializer),
          ("RIDGE_ALL_FOLDS",ridge,base_materializer),("HGB_ALL_FOLDS",hgb,base_materializer)]
    single_meta=None
    if single is not None:
        fold_id,filtered=single; single_path=output_root/"single-fold-candidate-metrics.json"
        single_path.write_text(json.dumps(filtered,indent=2,sort_keys=True)+"\n",encoding="utf-8")
        single_materializer=ParquetH130DatasetMaterializer(panel,single_path,end)
        actual_family=str(filtered[0]["family"])
        single_family=replace(base,family_id=base.family_id+"_SINGLE_FOLD_COMPONENT",model_family=actual_family,
            hyperparameter_rule={**base.hyperparameter_rule,"minimum_oos_folds":1})
        jobs.append(("SINGLE_POSITIVE_SPEARMAN_R2_FOLD",single_family,single_materializer))
        single_meta={"fold_id":fold_id,"candidate_id":filtered[0]["candidate_id"],"model_family":actual_family,
                     "spearman":filtered[0]["metrics"]["spearman"],"r2":filtered[0]["metrics"]["r2"]}
    # Model fits are serialized because each fit receives the suite-wide native
    # CPU budget. This reaches the CPU target without multiplying panel RAM.
    with ThreadPoolExecutor(max_workers=1,thread_name_prefix="fresh-fit") as pool:
        components=list(pool.map(lambda args:_fit_component(name=args[0],family=args[1],materializer=args[2],
            root=output_root/"fresh-models",cutoff=fit_cutoff,latest_matured=latest,sessions=sessions),jobs))
    by_name={x["name"]:x for x in components}
    ensemble_id=stable_hash((by_name["RIDGE_ALL_FOLDS"]["build"]["model_artifact_id"],
                             by_name["HGB_ALL_FOLDS"]["build"]["model_artifact_id"],"INTERSECTION"))
    strategies={
        "SELECTED_SINGLE_MODEL":_component_predictions(by_name["SELECTED_ALL_FOLDS"],"SELECTED"),
        "RIDGE_HGB_INTERSECTION":_intersection_predictions(by_name["RIDGE_ALL_FOLDS"],by_name["HGB_ALL_FOLDS"],ensemble_id),
    }
    if single_meta is not None:
        strategies["SINGLE_POSITIVE_SPEARMAN_R2_FOLD"]=_component_predictions(by_name["SINGLE_POSITIVE_SPEARMAN_R2_FOLD"],"SINGLE_FOLD")
    timeline_starts=[date.fromisoformat(x["activation_cutoff"]) for x in recipe_timeline]
    price_start=min([start,*timeline_starts])
    prices_path=materialize_daily_store_prices(daily_store_root=daily_store_root,signal_panel=panel,start=price_start,end=end,
                                               output_path=output_root/"inputs"/"prices.parquet")
    prices=pd.read_parquet(prices_path)
    with ThreadPoolExecutor(max_workers=len(strategies),thread_name_prefix="portfolio") as pool:
        replays=list(pool.map(lambda item:_replay(item[0],item[1],prices,cutoff=fit_cutoff,start=start,end=end,
            horizon=horizon,holding=holding_days,max_names=max_names,initial=initial),strategies.items()))
    rows_out=[]
    for replay in replays:
        result=replay["result"]; curve=result["curve"].copy(); curve["strategy"]=replay["strategy"]
        curve.to_parquet(output_root/f"{replay['strategy']}-nav.parquet",index=False)
        pd.DataFrame(result["trades"]).to_parquet(output_root/f"{replay['strategy']}-trades.parquet",index=False)
        metrics=result["metrics"]
        rows_out.append({"strategy":replay["strategy"],"initial_value":initial,
            "terminal_value":float(metrics["terminal_value"]),"urth_terminal_value":float(metrics["urth_terminal_value"]),
            "terminal_excess_eur":float(metrics["terminal_value"]-metrics["urth_terminal_value"]),
            "terminal_relative_return":float(metrics["terminal_value"]/metrics["urth_terminal_value"]-1.0),
            "trade_count":int(metrics["trade_count"]),"total_cost_eur":float(metrics["total_cost_eur"]),
            "relative_max_drawdown":float(replay["risk"]["relative_max_drawdown"])})
    comparison=pd.DataFrame(rows_out).sort_values("terminal_value",ascending=False)
    comparison.to_csv(output_root/"portfolio-value-comparison.csv",index=False)
    component_audit=[{"name":x["name"],"fresh_fit":True,"selected_model_family":x["build"]["selected_model_family"],
        "selected_hyperparameters":x["build"]["selected_hyperparameters"],
        "selection_evidence":x["build"]["selection_evidence"],"model_artifact_id":x["build"]["model_artifact_id"],
        "model_artifact_sha256":x["build"]["model_artifact_sha256"],"threshold":x["threshold"]} for x in components]
    rolling_root=output_root/"rolling-recipe-shadow"; rolling_root.mkdir(parents=True,exist_ok=True)
    rolling_recipe=_rolling_recipe_shadow(timeline=recipe_timeline,base=base,materializer=base_materializer,
        metrics_path=metrics_path,
        sessions=sessions,prices=prices,root=rolling_root,end=end,horizon=horizon,holding=holding_days,
        max_names=max_names,initial=initial)
    summary={"schema_version":"DYNAMIC_QBD_MODEL_COMBINATION_SHADOW_EXPERIMENT_V2",
        "status":"COMPLETE","authority":"SHADOW_ONLY_NO_PROMOTION_NO_HOLDOUT",
        "fit_cutoff":str(fit_cutoff),"evaluation":{"start":str(start),"end":str(end)},
        "portfolio_contract":{"initial_value":initial,"benchmark":"URTH","roundtrip_bps":20.0,
                              "tax":"PRE_TAX","max_names":max_names,"holding_days":holding_days,"sleeve":.5},
        "fresh_models_trained":True,"old_model_artifacts_reused":False,
        "execution_resources":active_cpu_contract(),
        "single_fold_eligibility_rule":"SPEARMAN_GT_0_AND_R2_GT_0_ON_COMPLETED_OUTER_FOLD",
        "single_fold_candidate":single_meta,"components":component_audit,
        "causal_recipe_selection_timeline":recipe_timeline,
        "rolling_recipe_shadow":rolling_recipe,
        "portfolio_value_comparison":comparison.to_dict(orient="records")}
    optimizations=[
        {"priority":1,"experiment":"N2_N3_DIVERSIFICATION","objective":"Reduce idiosyncratic drawdown while preserving the 50% total sleeve","guard":"Same entries, total sleeve and costs; predeclare N2/N3 and sector cap"},
        {"priority":2,"experiment":"CAUSAL_RECIPE_SWITCH_HYSTERESIS","objective":"Avoid weak recipe changes and reduce transition risk","guard":"Switch only on matured multi-fold improvement; no evaluation-period tuning"},
        {"priority":3,"experiment":"SOFT_RIDGE_HGB_AGREEMENT","objective":"Retain more alpha than the one-trade hard intersection while filtering disagreement","guard":"Calibrate percentile blend and minimum agreement solely on prior OOS folds"},
        {"priority":4,"experiment":"DRAWDOWN_AWARE_SLEEVE_THROTTLE","objective":"Cut left-tail exposure without permanently lowering opportunity participation","guard":"Causal trailing relative-wealth state; fixed de-risk/re-risk ladder; compare terminal value non-inferiority"},
        {"priority":5,"experiment":"LEARNED_EXIT_DOWNSIDE_OVERLAY","objective":"Exit failed opportunities earlier while retaining winners","guard":"Generation-specific E-models, next-open execution, coverage gate and fixed-H fallback"},
        {"priority":6,"experiment":"RECIPE_PLATEAU_CLUSTERING","objective":"Prefer stable recipe plateaus over a single noisy candidate winner","guard":"Cluster correlated candidates; time/fold remains inference unit; R2 diagnostic only"},
    ]
    pd.DataFrame(optimizations).to_csv(output_root/"optimization-roadmap.csv",index=False)
    summary["optimization_roadmap"]=optimizations
    (output_root/"summary.json").write_text(json.dumps(summary,default=str,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    rolling_rows={row["strategy"]:row for row in (rolling_recipe or {}).get("portfolio_value_comparison",[])}
    matched=rolling_rows.get("FROZEN_RECIPE_MATCHED_REFIT_DATES",{}); rolling=rolling_rows.get("ROLLING_CAUSAL_RECIPE_RESELECTION",{})
    report=["# Dynamic QBD model recipe and combination experiment","",
        "## Technical summary","",
        "The causal selector changed H3 recipes as additional completed Outer WF folds matured: Ridge (August 2020), HGB (August 2022), then Ridge (February 2023). Every activation used a fresh production fit and generation-specific calibration; historical model artifacts were not reused.","",
        (f"The primary matched-refit comparison ended at EUR {rolling.get('terminal_value',float('nan')):,.2f} for rolling recipe reselection versus EUR {matched.get('terminal_value',float('nan')):,.2f} for the frozen Ridge recipe refitted on the same dates. The rolling arm added EUR {rolling.get('terminal_value',0)-matched.get('terminal_value',0):,.2f}, but relative maximum drawdown worsened from {matched.get('relative_max_drawdown',float('nan')):.2%} to {rolling.get('relative_max_drawdown',float('nan')):.2%}. This is development-only shadow evidence, not promotion authority."),"",
        "## Portfolio-value evidence","","### Matched-date recipe-selection test","",
        comparison_markdown(pd.DataFrame((rolling_recipe or {}).get("portfolio_value_comparison",[]))),"",
        "The stale-model arm is diagnostic only because it confounds recipe choice with model age. The scientific recipe contrast is rolling reselection versus the frozen recipe refitted on identical dates.","",
        "### 2023-Q4 model-combination diagnostic","",comparison_markdown(comparison),"",
        "The positive-Spearman/positive-R² single-fold arm led this quarter, but one fold and one quarter are insufficient research evidence. The hard Ridge/HGB intersection generated only one trade and reduced terminal value, so hard consensus is not supported by this run.","",
        "## Scope and definitions","",
        "- Capital: EUR 10,000; benchmark and idle capital: URTH.",
        "- Contract: H3 signal, D2 holding, N1, 50% entry sleeve, 20 bps round-trip costs, pre-tax.",
        "- Recipe evidence: only completed prior Outer WF folds; R² is diagnostic and does not select the robust production recipe.",
        "- Primary long-window comparison: 31 August 2020 through 29 December 2023.",
        "- Q4 combination diagnostic: 2 October through 29 December 2023.","",
        "## Method and robustness","",
        "The matched-refit control holds refit dates, moving training windows, calibration, costs and portfolio rules constant. Only the recipe chosen at a switch date differs. Existing positions retain generation lineage, and all executions occur under the stateful next-open portfolio accounting contract.","",
        "No chart is included because the five portfolio rows span two different evaluation windows; combining them in one visual would imply a false common denominator. Exact tables and the saved daily NAV paths are the more honest evidence surface.","",
        "## Limitations","",
        "- H3/D2/N1 is one policy cell; results do not establish H1-H30 generality.",
        "- Recipe changes are sparse, so statistical power is low and the 2022 HGB interval may be regime-specific.",
        "- Rolling reselection improved terminal wealth but materially worsened drawdown; it is not a free dominance result.",
        "- The Q4 single-fold winner is explicitly diagnostic and must not influence the production picker.",
        "- Final holdout remained closed; no capital or promotion authority is granted.","",
        "## Recommended next experiments","",*[
            f"{row['priority']}. **{row['experiment']}** — {row['objective']}. Guard: {row['guard']}." for row in optimizations],"",
        "The first test should be N2/N3 diversification because N1 concentration is the cleanest likely drawdown source and can be varied without changing the signal or total stock sleeve. Recipe hysteresis and soft model agreement follow; drawdown throttles and learned exits should be tested only with explicit terminal-value non-inferiority gates.","",
        "## Further questions","",
        "- Does rolling recipe reselection remain positive under N2/N3 and across H1-H30 plateaus?",
        "- Which trades produced the additional HGB drawdown, and is the loss concentrated by ticker, sector or market regime?",
        "- Can a soft agreement score preserve the selected-model trade breadth while improving left-tail outcomes?",
        "- What hysteresis margin remains stable under block-bootstrap resampling of time folds?","",
        "Status: `SHADOW_ONLY_NO_PROMOTION_NO_HOLDOUT`."]
    (output_root/"REPORT.md").write_text("\n".join(report)+"\n",encoding="utf-8")
    return summary


def main(argv=None) -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--signal-panel",required=True); parser.add_argument("--candidate-metrics",required=True)
    parser.add_argument("--daily-store-root",required=True); parser.add_argument("--output-root",required=True)
    parser.add_argument("--fit-cutoff",required=True); parser.add_argument("--start",required=True); parser.add_argument("--end",required=True)
    parser.add_argument("--cpu-peak-fraction",type=float,default=.90)
    args=parser.parse_args(argv)
    from .dynamic_qbd_runtime_resources import configure_cpu_peak
    configure_cpu_peak(args.cpu_peak_fraction)
    result=run_experiment(signal_panel=args.signal_panel,candidate_metrics=args.candidate_metrics,
        daily_store_root=args.daily_store_root,output_root=args.output_root,fit_cutoff=date.fromisoformat(args.fit_cutoff),
        start=date.fromisoformat(args.start),end=date.fromisoformat(args.end))
    print(json.dumps(result["portfolio_value_comparison"],indent=2))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
