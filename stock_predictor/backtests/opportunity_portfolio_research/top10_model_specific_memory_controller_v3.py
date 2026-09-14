"""Model-specific monthly-memory controller V3.

Development reuse only: SHORT/MID/LONG priors are frozen from the already-inspected
W_RECENT/W_MID/W_LONG experiment. Reuses historical WF warm-up and existing causal
entry/exit artifacts; no fitting, no prediction generation, Final Holdout closed.
"""
from __future__ import annotations

import argparse, json
from dataclasses import asdict
from pathlib import Path
import numpy as np
import pandas as pd

from .portfolio_research_inputs import load_price_panel
from .learned_exit_qbd_provider import LearnedExitProvider
from .next_open_portfolio_replay import prepare_prices
from .top10_causal_expanding_portfolio import LIVE_END, LIVE_START, _policy, _thresholds
from .top10_entry_activation_alpha_diagnostic import _causal_entry, _contracts, _markdown, _write_json
from .top10_causal_monthly_weight_controller import (
    WARMUP_START, MONTH_SESSIONS, EXPERT_COUNT, ASSESS_EVERY_SESSIONS,
    WEIGHT_FLOOR, WEIGHT_CEILING, MAX_ABS_WEIGHT_CHANGE_PER_ASSESSMENT, HEDGE_ETA,
    _historical_entry, _combined_entry, _historical_pressure, _prior_suite_gates,
    _assessment_dates, _prepare_model_rows, _expert_snapshot, _rank_signal,
    _bounded_simplex_toward_target, _replay_arm,
)

CONTRACT_ID="TOP10_MODEL_SPECIFIC_MEMORY_CONTROLLER_V3"
VALIDATION_STATUS="DEVELOPMENT_REUSE_NOT_INDEPENDENT_OOS"
MODEL_PRIOR={
 "R01_L_H11_D03_N1":"SHORT","R02_F_H24_D05_N1":"SHORT","R03_L_H28_D21_N1":"MID",
 "R04_L_H28_D21_N5":"LONG","R05_L_H28_D21_N4":"LONG","R06_L_H28_D21_N6":"LONG",
 "R07_L_H28_D21_N2":"LONG","R08_L_H28_D21_N3":"LONG","R09_L_H24_D21_N5":"LONG",
 "R10_L_H24_D21_N6":"LONG",
}
CLASS_PROFILE={"SHORT":"W_RECENT","MID":"W_MID","LONG":"W_LONG"}
BAND_MASS={"SHORT":(.50,.30,.20),"MID":(.20,.50,.30),"LONG":(.20,.30,.50)}


def _prior(kind:str)->np.ndarray:
    a,b,c=BAND_MASS[kind]
    x=np.array([a/3]*3+[b/3]*3+[c/6]*6,float)
    assert len(x)==12 and np.isclose(x.sum(),1)
    if x.min()<WEIGHT_FLOOR-1e-12 or x.max()>WEIGHT_CEILING+1e-12: raise RuntimeError("V3_PRIOR_BOUNDS")
    return x


def _validate():
    if WARMUP_START!=pd.Timestamp("2020-09-01") or EXPERT_COUNT!=12 or MONTH_SESSIONS!=21 or ASSESS_EVERY_SESSIONS!=21: raise RuntimeError("V3_PARENT_CONTRACT_CHANGED")
    if len(MODEL_PRIOR)!=10 or set(MODEL_PRIOR.values())!={"SHORT","MID","LONG"}: raise RuntimeError("V3_PRIOR_MAP")
    for k in BAND_MASS:_prior(k)


def _profile_gate(summary_path:Path, results_path:Path)->pd.DataFrame:
    s=json.loads(summary_path.read_text(encoding="utf-8"))
    if s.get("status")!="COMPLETE" or s.get("final_holdout_opened") or s.get("profile_selection_from_forward_performance_allowed") is not False: raise RuntimeError("V3_PROFILE_SUMMARY_GATE")
    f=pd.read_csv(results_path)
    if len(f)!=30 or set(f.model_id.astype(str))!=set(MODEL_PRIOR): raise RuntimeError("V3_PROFILE_RESULTS_GATE")
    rows=[]
    for model,g in f.groupby("model_id",sort=True):
        best=g.sort_values(["cagr_excess","profile"],ascending=[False,True]).iloc[0]
        kind=MODEL_PRIOR[str(model)]; expected=CLASS_PROFILE[kind]
        if str(best.profile)!=expected: raise RuntimeError(f"V3_PRIOR_SOURCE_MISMATCH:{model}:{best.profile}:{expected}")
        p=_prior(kind); row={"model_id":str(model),"prior_class":kind,"source_profile":expected,"source_profile_cagr_excess":float(best.cagr_excess),"prior_effective_memory_months":float(p@np.arange(1,13)),"development_selected":True}
        row.update({f"prior_w_m{i+1}":float(p[i]) for i in range(12)});rows.append(row)
    return pd.DataFrame(rows)


def _next(current:np.ndarray, prior:np.ndarray, snapshot:list[dict]):
    if not all(bool(r["available"]) for r in snapshot): return current.copy(),False,[np.nan]*12,prior.copy()
    q=np.array([float(r["quality_mean_excess_clipped"]) for r in snapshot]); sig=_rank_signal(q)
    target=prior*np.exp(HEDGE_ETA*sig);target/=target.sum()
    return _bounded_simplex_toward_target(current,target),True,sig.tolist(),target


def _schedule(prepared, dates, model_id, contract_id, kind, pressure):
    prior=_prior(kind);w=prior.copy();states=[];evidence=[]
    for d in dates:
        snap=_expert_snapshot(prepared,d);nw,updated,sig,target=_next(w,prior,snap)
        p99=np.array([float(r["median_p99"]) if r["available"] else np.nan for r in snap]); ok=np.isfinite(p99).all()
        vals={"uniform":float(np.mean(p99)) if ok else np.nan,"static":float(prior@p99) if ok else np.nan,"adaptive":float(nw@p99) if ok else np.nan}
        rep={k:(float(pressure*v) if np.isfinite(v) and v>0 else np.nan) for k,v in vals.items()}
        st={"model_id":model_id,"contract_id":contract_id,"prior_class":kind,"assessment_date":d,"phase":"WARMUP" if d<LIVE_START else "DEVELOPMENT_EVALUATION","updated":updated,"historical_threshold_over_p99":pressure,"uniform_replacement_threshold":rep["uniform"],"static_prior_replacement_threshold":rep["static"],"adaptive_replacement_threshold":rep["adaptive"],"prior_effective_memory_months":float(prior@np.arange(1,13)),"adaptive_effective_memory_months":float(nw@np.arange(1,13)),"max_abs_weight_change":float(np.max(np.abs(nw-w))),"l1_distance_from_prior":float(np.abs(nw-prior).sum())}
        st.update({f"prior_w_m{i+1}":float(prior[i]) for i in range(12)});st.update({f"w_m{i+1}":float(nw[i]) for i in range(12)});states.append(st)
        for i,r in enumerate(snap): evidence.append({"model_id":model_id,"contract_id":contract_id,"prior_class":kind,"assessment_date":d,"prior_weight":float(prior[i]),"weight_before":float(w[i]),"target_weight":float(target[i]),"weight_after":float(nw[i]),"rank_signal":sig[i],**r})
        w=nw
    return pd.DataFrame(states),pd.DataFrame(evidence)


def _daily(selected, raw, schedule):
    states=list(schedule.sort_values("assessment_date").itertuples(index=False));out=[]
    for d in sorted(selected.loc[selected.decision_date.between(LIVE_START,LIVE_END),"decision_date"].unique()):
        d=pd.Timestamp(d);r=float(raw.loc[d]);past=[s for s in states if pd.Timestamp(s.assessment_date)<d];s=past[-1] if past else None
        u=st=ad=np.nan;src=pd.NaT;mem=np.nan
        if s is not None:u=float(s.uniform_replacement_threshold);st=float(s.static_prior_replacement_threshold);ad=float(s.adaptive_replacement_threshold);src=pd.Timestamp(s.assessment_date);mem=float(s.adaptive_effective_memory_months)
        eff=lambda x:min(r,x) if np.isfinite(x) else r
        out.append({"decision_date":d,"raw_threshold":r,"uniform_effective_threshold":eff(u),"static_prior_effective_threshold":eff(st),"adaptive_effective_threshold":eff(ad),"source_assessment_date":src,"adaptive_effective_memory_months":mem})
    f=pd.DataFrame(out);a=f.source_assessment_date.notna()
    if a.any() and f.loc[a,"source_assessment_date"].ge(f.loc[a,"decision_date"]).any():raise RuntimeError("V3_NONCAUSAL_SOURCE")
    for c in ["uniform_effective_threshold","static_prior_effective_threshold","adaptive_effective_threshold"]:
        if (f[c]>f.raw_threshold+1e-12).any():raise RuntimeError("V3_THRESHOLD_TIGHTENED")
    return f


def run(args):
    _validate();out=Path(args.output_root);out.mkdir(parents=True,exist_ok=True)
    _prior_suite_gates(Path(args.retrospective_summary),Path(args.entry_diagnostic_summary));prior_manifest=_profile_gate(Path(args.profile_summary),Path(args.profile_results))
    hist,hist_audit=_historical_entry(Path(args.historical_entry_predictions));causal=_causal_entry(Path(args.causal_entry_predictions));combined=_combined_entry(hist,causal)
    contracts,by_model=_contracts(Path(args.frozen_policy_manifest));pressure=_historical_pressure(Path(args.entry_diagnostic_summary))
    prices,price_audit=load_price_panel(Path(args.daily_store_root),set(combined.ticker.astype(str)));prepared_prices=prepare_prices(prices);dates=_assessment_dates(combined)
    manifest=json.loads(Path(args.frozen_policy_manifest).read_text(encoding="utf-8"));models=manifest.get("models",[])
    if len(models)!=10 or {str(m["model_id"]) for m in models}!=set(MODEL_PRIOR):raise RuntimeError("V3_MANIFEST_MODELS")
    exits=LearnedExitProvider(Path(args.causal_exit_predictions))
    if exits.audit.min_date!=str(LIVE_START.date()) or exits.audit.max_date!=str(LIVE_END.date()):raise RuntimeError("V3_EXIT_COVERAGE")
    states=[];evidence=[];dailies=[];results=[];trades=[];curves=[]
    for m in models:
        mid=str(m["model_id"]);kind=MODEL_PRIOR[mid];policy=_policy(m["frozen_entry_policy"]);spec=by_model[mid]
        hit=contracts.loc[contracts.horizon.eq(int(spec["horizon"]))&contracts.score_quantile.eq(float(spec["score_quantile"]))&contracts.top_fraction.eq(float(spec["top_fraction"])),"contract_id"]
        if len(hit)!=1:raise RuntimeError(f"V3_CONTRACT_MAP:{mid}")
        cid=str(hit.iloc[0]);prep=_prepare_model_rows(combined,prices,horizon=int(policy.horizon),top_fraction=float(policy.top_fraction),max_names=int(policy.max_names))
        sched,ev=_schedule(prep,dates,mid,cid,kind,pressure[cid]);states.append(sched);evidence.append(ev)
        sel=causal.loc[causal.horizon.eq(int(policy.horizon))].copy();raw=_thresholds(sel,float(policy.score_quantile));day=_daily(sel,raw,sched);day.insert(0,"model_id",mid);day.insert(1,"prior_class",kind);dailies.append(day)
        arms={"RAW":day.set_index("decision_date").raw_threshold,"UNIFORM_12M":day.set_index("decision_date").uniform_effective_threshold,"STATIC_MODEL_PRIOR":day.set_index("decision_date").static_prior_effective_threshold,"ADAPTIVE_MODEL_CONTROLLER":day.set_index("decision_date").adaptive_effective_threshold}
        sigs=sel[["decision_date","ticker","score"]]
        for arm,thr in arms.items():
            met,tr,cv=_replay_arm(arm=arm,model_id=mid,contract_id=cid,signals=sigs,prices=prices,policy=policy,threshold_by_date=thr,exit_provider=exits,prepared_prices=prepared_prices,initial_capital=args.initial_capital)
            results.append({"model_id":mid,"prior_class":kind,"source_profile":CLASS_PROFILE[kind],"arm":arm,"h":policy.horizon,"d":policy.holding_days,"n":policy.max_names,"mode":policy.exit_family,**met});trades.extend(tr);curves.append(cv);print(f"V3_REPLAY_COMPLETE {mid} {arm} trades={met['trade_count']}",flush=True)
    sf=pd.concat(states,ignore_index=True);ef=pd.concat(evidence,ignore_index=True);df=pd.concat(dailies,ignore_index=True);rf=pd.DataFrame(results).sort_values(["arm","model_id"])
    prior_manifest.to_csv(out/"model_specific_memory_prior_manifest.csv",index=False);sf.to_csv(out/"model_specific_controller_state.csv",index=False);sf.to_parquet(out/"model_specific_controller_state.parquet",index=False);ef.to_parquet(out/"model_specific_controller_expert_evidence.parquet",index=False);df.to_csv(out/"model_specific_live_thresholds.csv",index=False);df.to_parquet(out/"model_specific_live_thresholds.parquet",index=False);rf.to_csv(out/"model_specific_controller_results.csv",index=False);pd.DataFrame(trades).to_csv(out/"model_specific_controller_trades.csv",index=False);pd.concat(curves,ignore_index=True).to_parquet(out/"model_specific_controller_curves.parquet",index=False)
    sf.loc[sf.assessment_date.lt(LIVE_START)].sort_values("assessment_date").groupby("model_id",as_index=False).tail(1).sort_values("model_id").to_csv(out/"model_specific_weights_at_live_start.csv",index=False)
    a=rf.loc[rf.arm.eq("ADAPTIVE_MODEL_CONTROLLER")].copy();u=rf.loc[rf.arm.eq("UNIFORM_12M"),["model_id","cagr_excess"]].rename(columns={"cagr_excess":"uniform_cagr_excess"});st=rf.loc[rf.arm.eq("STATIC_MODEL_PRIOR"),["model_id","cagr_excess"]].rename(columns={"cagr_excess":"static_prior_cagr_excess"});sys=a.merge(u,on="model_id").merge(st,on="model_id");sys["adaptive_minus_uniform_cagr_excess"]=sys.cagr_excess-sys.uniform_cagr_excess;sys["adaptive_minus_static_cagr_excess"]=sys.cagr_excess-sys.static_prior_cagr_excess;sys["development_rank"]=sys.cagr_excess.rank(method="min",ascending=False).astype(int);sys["selection_allowed"]=False;sys.sort_values(["development_rank","model_id"]).to_csv(out/"development_model_system_comparison.csv",index=False)
    arm_summary=[{"arm":arm,"models":len(g),"total_trades":int(g.trade_count.sum()),"positive_cagr_excess_models":int(g.cagr_excess.gt(0).sum()),"median_cagr_excess":float(g.cagr_excess.median()),"mean_cagr_excess":float(g.cagr_excess.mean()),"selection_allowed":False} for arm,g in rf.groupby("arm")]
    summary={"contract_id":CONTRACT_ID,"status":"COMPLETE","validation_status":VALIDATION_STATUS,"no_model_training":True,"no_prediction_generation":True,"uses_existing_17000_causal_artifacts":True,"no_threshold_optimization":True,"no_online_prior_class_switching":True,"final_holdout_opened":False,"development_reuse":{"prior_assignment_uses_seen_2023_2026_fixed_profile_results":True,"evaluation_is_not_independent_oos":True},"warmup":{"start":str(WARMUP_START.date()),"end":"2023-08-10","source":"historical WF_001..WF_007 OOS predictions"},"evaluation":{"start":str(LIVE_START.date()),"end":str(LIVE_END.date()),"selection_allowed":False},"model_prior_assignment":prior_manifest.to_dict(orient="records"),"controller_contract":{"prior_age_bands":{"SHORT":"M1-M3","MID":"M4-M6","LONG":"M7-M12"},"prior_band_mass":{k:list(v) for k,v in BAND_MASS.items()},"weight_floor":WEIGHT_FLOOR,"weight_ceiling":WEIGHT_CEILING,"max_abs_weight_change_per_assessment":MAX_ABS_WEIGHT_CHANGE_PER_ASSESSMENT,"assessment_every_sessions":ASSESS_EVERY_SESSIONS,"prior_anchor_reapplied_each_update":True,"outcome_visibility":"terminal_date <= assessment_date","effective_threshold":"min(raw, replacement)"},"arms":["RAW","UNIFORM_12M","STATIC_MODEL_PRIOR","ADAPTIVE_MODEL_CONTROLLER"],"arm_summary":arm_summary,"adaptive_controller_updates":int(sf.updated.sum()),"historical_prediction_audit":hist_audit,"exit_provider_audit":asdict(exits.audit),"price_audit":price_audit,"model_system_selection_from_development_performance_allowed":False}
    _write_json(out/"model_specific_controller_summary.json",summary)
    rep=rf[["model_id","prior_class","arm","trade_count","cagr_excess","terminal_wealth_excess_eur"]].copy();rep.cagr_excess=rep.cagr_excess.map(lambda x:f"{float(x):.4%}")
    (out/"REPORT.md").write_text("\n".join(["# Top-10 Model-Specific Memory Controller V3","",f"Status: **COMPLETE** / **{VALIDATION_STATUS}**","","Each model has its own frozen SHORT/MID/LONG prior. Priors were selected from the already inspected fixed-profile development experiment, so this replay is not independent OOS.","","## Development comparison","",_markdown(rep),"","RAW, UNIFORM_12M, STATIC_MODEL_PRIOR, and ADAPTIVE_MODEL_CONTROLLER are compared per complete model system. Selection from this development replay is prohibited. Final Holdout remained closed."])+"\n",encoding="utf-8")
    return summary


def self_test():
    _validate();s,m,l=_prior("SHORT"),_prior("MID"),_prior("LONG");months=np.arange(1,13)
    assert s@months<m@months<l@months and s[:3].sum()>m[:3].sum() and m[3:6].sum()>l[3:6].sum() and l[6:].sum()>m[6:].sum()
    snap=[{"available":True,"quality_mean_excess_clipped":float(12-i)} for i in range(12)];changed,updated,_,_=_next(s,s,snap);assert updated and changed[0]>s[0] and changed[-1]<s[-1] and np.max(np.abs(changed-s))<=MAX_ABS_WEIGHT_CHANGE_PER_ASSESSMENT+1e-10
    print("TOP10_MODEL_SPECIFIC_MEMORY_CONTROLLER_V3_SELF_TEST_OK")


def main():
    p=argparse.ArgumentParser()
    for x in ["historical-entry-predictions","causal-entry-predictions","causal-exit-predictions","frozen-policy-manifest","entry-diagnostic-summary","retrospective-summary","profile-summary","profile-results","daily-store-root"]:p.add_argument("--"+x)
    p.add_argument("--output-root",default="artifacts/top10-model-specific-memory-controller-v3");p.add_argument("--initial-capital",type=float,default=10000.0);p.add_argument("--self-test",action="store_true");a=p.parse_args()
    if a.self_test:self_test();return
    miss=[x for x in vars(a) if x not in {"self_test","output_root","initial_capital"} and getattr(a,x) is None]
    if miss:p.error("missing: "+",".join(miss))
    print(json.dumps(run(a),indent=2,default=str))

if __name__=="__main__":main()
