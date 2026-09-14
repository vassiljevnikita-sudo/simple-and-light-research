from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .allocation_qbd_evaluate import load_primary_phase1_design_space
from .prediction_hold_qbd_evaluate import _aggregate_cell
from .concentration_replacement_qbd_contract import (
    ALLOCATION_FIXED, BASELINE_REPLACEMENT, CONTRACT_ID, DEFAULT_MAX_NAMES,
    DEFAULT_REPLACEMENTS, PHASE3_CONTRACT_ID,
)


def treatment_slug(value: str) -> str:
    return str(value).replace(":", "_").replace(".", "p").replace("/", "_")


def _bool(value) -> bool:
    return str(value).lower() in ("true", "1", "yes") if not isinstance(value, (bool, np.bool_)) else bool(value)


def load_phase3_provenance(summary_path: Path, treatment_summary_path: Path) -> dict:
    s = json.loads(Path(summary_path).read_text(encoding="utf-8"))
    checks = (
        (s.get("contract_id") == PHASE3_CONTRACT_ID, "CONTRACT_MISMATCH"),
        (s.get("qbd_complete") is True, "NOT_COMPLETE"),
        (s.get("phase2_allocation_fixed") == ALLOCATION_FIXED, "ALLOCATION_MISMATCH"),
        (s.get("final_holdout_opened") is False, "HOLDOUT_OPENED"),
        (s.get("interpolation_used") is False, "INTERPOLATION_USED"),
        (s.get("v45_exit_overlay_used") is False, "V45_USED"),
    )
    for ok, code in checks:
        if not ok: raise ValueError(f"CONCENTRATION_QBD_PHASE3_{code}")
    t = pd.read_csv(treatment_summary_path)
    required={"replacement","replacement_qbd_pass","baseline_reference"}
    if missing:=sorted(required-set(t.columns)): raise ValueError(f"CONCENTRATION_QBD_PHASE3_COLUMNS_MISSING:{missing}")
    if not set(DEFAULT_REPLACEMENTS).issubset(set(t.replacement.astype(str))): raise ValueError("CONCENTRATION_QBD_PHASE3_TREATMENTS_INCOMPLETE")
    base=t.loc[t.replacement.eq(BASELINE_REPLACEMENT)]
    if len(base)!=1 or not _bool(base.iloc[0].baseline_reference) or not _bool(base.iloc[0].replacement_qbd_pass):
        raise ValueError("CONCENTRATION_QBD_PHASE3_BASELINE_INVALID")
    return {"contract_id":PHASE3_CONTRACT_ID,"baseline_replacement":BASELINE_REPLACEMENT,"replacement_reopened_for_concentration_test":True}


def expected_cells(design_cells: Iterable[tuple[int,int]], max_names_values=DEFAULT_MAX_NAMES, replacements=DEFAULT_REPLACEMENTS):
    cells=sorted({(int(h),int(d)) for h,d in design_cells}); names=tuple(dict.fromkeys(map(int,max_names_values))); tx=tuple(dict.fromkeys(map(str,replacements)))
    if not cells or not names or BASELINE_REPLACEMENT not in tx: raise ValueError("CONCENTRATION_QBD_EXPECTED_CELLS_INVALID")
    return [(h,d,n,r) for h,d in cells for n in names for r in tx]


def validate_surface_coverage(status: pd.DataFrame, expected) -> dict:
    wanted=set(expected); keys=["prediction_horizon","holding_days","max_names","replacement"]
    if status.empty: observed=set(); complete=set(); duplicates=[]
    else:
        f=status.copy(); f[["prediction_horizon","holding_days","max_names"]]=f[["prediction_horizon","holding_days","max_names"]].astype(int); f["replacement"]=f.replacement.astype(str)
        dup=f.duplicated(keys,keep=False); duplicates=f.loc[dup,keys].drop_duplicates().to_dict("records")
        observed=set(map(tuple,f[keys].itertuples(index=False,name=None))); ok=f.status.astype(str).eq("COMPLETE"); complete=set(map(tuple,f.loc[ok,keys].itertuples(index=False,name=None)))
    def pack(rows): return [{"prediction_horizon":h,"holding_days":d,"max_names":n,"replacement":r} for h,d,n,r in sorted(rows)]
    missing=wanted-observed; failed=wanted-complete; unexpected=observed-wanted
    return {"expected_cells":len(wanted),"observed_cells":len(observed),"complete_cells":len(complete&wanted),"missing_cells":pack(missing),"failed_or_incomplete_cells":pack(failed),"unexpected_cells":pack(unexpected),"duplicate_cells":duplicates,"surface_complete":not missing and not failed and not unexpected and not duplicates}


def aggregate_outer_results(rows: pd.DataFrame) -> pd.DataFrame:
    required={"prediction_horizon","holding_days","max_names","replacement","cagr_excess","trade_count"}
    if missing:=sorted(required-set(rows.columns)): raise ValueError(f"CONCENTRATION_QBD_OUTER_COLUMNS_MISSING:{missing}")
    out=[]
    for (_h,_d,n,r),g in rows.groupby(["prediction_horizon","holding_days","max_names","replacement"],sort=True):
        row=_aggregate_cell(g); row["max_names"]=int(n); row["replacement"]=str(r); out.append(row)
    return pd.DataFrame(out).sort_values(["prediction_horizon","holding_days","max_names","replacement"]).reset_index(drop=True)


def _add_references(cells: pd.DataFrame) -> pd.DataFrame:
    x=cells.copy()
    b=x.loc[x.replacement.eq(BASELINE_REPLACEMENT),["prediction_horizon","holding_days","max_names","median_active_cagr_excess","q25_active_cagr_excess","median_turnover"]].rename(columns={"median_active_cagr_excess":"ignore_median","q25_active_cagr_excess":"ignore_q25","median_turnover":"ignore_turnover"})
    x=x.merge(b,on=["prediction_horizon","holding_days","max_names"],how="left")
    if x.ignore_median.isna().any(): raise ValueError("CONCENTRATION_QBD_IGNORE_NEW_CELL_MISSING")
    x["delta_median_vs_ignore_new"]=x.median_active_cagr_excess-x.ignore_median; x["delta_q25_vs_ignore_new"]=x.q25_active_cagr_excess-x.ignore_q25; x["delta_turnover_vs_ignore_new"]=x.median_turnover-x.ignore_turnover; x["beats_ignore_new"]=x.delta_median_vs_ignore_new>0
    x.loc[x.replacement.eq(BASELINE_REPLACEMENT),"beats_ignore_new"]=False
    n1=x.loc[x.max_names.eq(1),["prediction_horizon","holding_days","replacement","median_active_cagr_excess","q25_active_cagr_excess","median_turnover"]].rename(columns={"median_active_cagr_excess":"n1_median","q25_active_cagr_excess":"n1_q25","median_turnover":"n1_turnover"})
    x=x.merge(n1,on=["prediction_horizon","holding_days","replacement"],how="left")
    if x.n1_median.isna().any(): raise ValueError("CONCENTRATION_QBD_MAX_NAMES_1_REFERENCE_MISSING")
    x["delta_median_vs_max_names_1_same_replacement"]=x.median_active_cagr_excess-x.n1_median; x["delta_q25_vs_max_names_1_same_replacement"]=x.q25_active_cagr_excess-x.n1_q25; x["delta_turnover_vs_max_names_1_same_replacement"]=x.median_turnover-x.n1_turnover
    return x


def summarize_treatments(cells: pd.DataFrame, design_count: int) -> pd.DataFrame:
    rows=[]
    for (n,r),g in cells.groupby(["max_names","replacement"],sort=True):
        med=g.median_active_cagr_excess.astype(float); delta=g.delta_median_vs_ignore_new.astype(float); qdelta=g.delta_q25_vs_ignore_new.astype(float)
        robust=float(g.robust_gate_pass.astype(bool).mean()); positive=float((med>0).mean()); beat=0.0 if r==BASELINE_REPLACEMENT else float((delta>0).mean())
        base_pass=len(g)==design_count and robust>=.50 and positive>=.50
        rep_pass=r==BASELINE_REPLACEMENT or (beat>=.60 and float(delta.median())>0 and float(np.quantile(delta,.25))>=0)
        rows.append({"max_names":int(n),"replacement":str(r),"cells":len(g),"robust_cells":int(g.robust_gate_pass.astype(bool).sum()),"robust_cell_fraction":robust,"positive_cell_fraction":positive,"median_cell_median_active_cagr_excess":float(med.median()),"q25_cell_median_active_cagr_excess":float(np.quantile(med,.25)),"worst_cell_median_active_cagr_excess":float(med.min()),"beat_ignore_new_cell_fraction":beat,"median_delta_vs_ignore_new":float(delta.median()),"q25_delta_vs_ignore_new":float(np.quantile(delta,.25)),"worst_delta_vs_ignore_new":float(delta.min()),"median_turnover_delta_vs_ignore_new":float(g.delta_turnover_vs_ignore_new.median()),"median_delta_vs_max_names_1_same_replacement":float(g.delta_median_vs_max_names_1_same_replacement.median()),"total_trades":int(g.trade_count.sum()),"median_turnover":float(g.median_turnover.median()),"replacement_qbd_pass_within_max_names":bool(base_pass and rep_pass),"treatment_qbd_pass":bool(base_pass and rep_pass),"baseline_reference":r==BASELINE_REPLACEMENT})
    out=pd.DataFrame(rows); return out.sort_values(["max_names","replacement"]).reset_index(drop=True) if not out.empty else out


def summarize_concentration(treatment: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    for n,g in treatment.groupby("max_names",sort=True):
        a=g.loc[g.replacement.eq(BASELINE_REPLACEMENT)].iloc[0]; b=g.loc[g.replacement.eq("REPLACE_WEAKEST")].iloc[0]
        rows.append({"max_names":int(n),"ignore_new_robust_cell_fraction":float(a.robust_cell_fraction),"ignore_new_median_active_cagr_excess":float(a.median_cell_median_active_cagr_excess),"ignore_new_q25_active_cagr_excess":float(a.q25_cell_median_active_cagr_excess),"ignore_new_median_turnover":float(a.median_turnover),"replace_weakest_robust_cell_fraction":float(b.robust_cell_fraction),"replace_weakest_beat_ignore_new_fraction":float(b.beat_ignore_new_cell_fraction),"replace_weakest_median_delta_vs_ignore_new":float(b.median_delta_vs_ignore_new),"replace_weakest_q25_delta_vs_ignore_new":float(b.q25_delta_vs_ignore_new),"replace_weakest_turnover_delta_vs_ignore_new":float(b.median_turnover_delta_vs_ignore_new),"preferred_replacement_under_phase4_gate":"REPLACE_WEAKEST" if bool(b.replacement_qbd_pass_within_max_names) else BASELINE_REPLACEMENT})
    return pd.DataFrame(rows).sort_values("max_names").reset_index(drop=True)


def evaluate_surface(outer,status,design_cells,max_names_values,replacements):
    wanted=expected_cells(design_cells,max_names_values,replacements); coverage=validate_surface_coverage(status,wanted)
    cells=aggregate_outer_results(outer) if not outer.empty else pd.DataFrame(); treatment=pd.DataFrame(); concentration=pd.DataFrame()
    if not cells.empty: cells=_add_references(cells); treatment=summarize_treatments(cells,len(design_cells)); concentration=summarize_concentration(treatment)
    keys=["prediction_horizon","holding_days","max_names","replacement"]
    aggregates=set(map(tuple,cells[keys].itertuples(index=False,name=None))) if not cells.empty else set(); complete=set(map(tuple,status.loc[status.status.astype(str).eq("COMPLETE"),keys].itertuples(index=False,name=None))) if not status.empty else set(); parity=aggregates==complete
    summary={"contract_id":CONTRACT_ID,"phase1_design_cells":len(design_cells),"phase2_allocation_fixed":ALLOCATION_FIXED,"phase2_allocation_frozen":True,"phase3_replacement_reopened":True,"max_names_values":list(map(int,max_names_values)),"replacement_treatments":list(map(str,replacements)),"expected_surface_cells":len(wanted),"surface_coverage":coverage,"aggregate_complete_cell_parity":parity,"aggregated_cells":len(cells),"passing_treatments":int(treatment.treatment_qbd_pass.sum()) if not treatment.empty else 0,"selection_principle":"broad H/D stability; max_names fixed per cell; REPLACE_WEAKEST paired to IGNORE_NEW at identical max_names","roundtrip_cost_stress_in_phase4":False,"tax_stress_in_phase4":False,"cost_and_tax_stress_deferred":True,"final_holdout_opened":False,"final_holdout_locked":True,"interpolation_used":False,"v45_exit_overlay_used":False,"qbd_complete":bool(coverage["surface_complete"] and parity)}
    return cells,treatment,concentration,summary


def load_cell_artifacts(root: Path):
    status=[]; outer=[]; final=[]
    for path in sorted((Path(root)/"cells").glob("H*_D*__N*__*.json")):
        p=json.loads(path.read_text(encoding="utf-8")); h=int(p["prediction_horizon"]); d=int(p["holding_days"]); n=int(p["max_names"]); r=str(p["replacement"]); state=str(p.get("status","UNKNOWN")); status.append({"prediction_horizon":h,"holding_days":d,"max_names":n,"replacement":r,"status":state,"error":p.get("error"),"artifact":str(path)})
        if state!="COMPLETE": continue
        outer.extend({**row,"prediction_horizon":h,"holding_days":d,"max_names":n,"replacement":r} for row in p.get("outer_rows",[]))
        if p.get("final_policy"): final.append({"prediction_horizon":h,"holding_days":d,"max_names":n,"replacement":r,**p["final_policy"],"resolved_threshold":p.get("final_threshold")})
    return pd.DataFrame(status),pd.DataFrame(outer),pd.DataFrame(final)


def write_evaluation_artifacts(root: Path, *, phase1_design_space: Path, phase3_summary: Path, phase3_treatment_summary: Path, max_names_values=DEFAULT_MAX_NAMES, replacements=DEFAULT_REPLACEMENTS) -> dict:
    root=Path(root); root.mkdir(parents=True,exist_ok=True); design=load_primary_phase1_design_space(phase1_design_space); load_phase3_provenance(phase3_summary,phase3_treatment_summary)
    status,outer,final=load_cell_artifacts(root); cells,treatment,concentration,summary=evaluate_surface(outer,status,design,max_names_values,replacements)
    outputs=(("concentration_replacement_qbd_cell_status.csv",status),("concentration_replacement_qbd_outer_fold_results.csv",outer),("concentration_replacement_qbd_final_policies.csv",final),("concentration_replacement_qbd_surface_cells.csv",cells),("concentration_replacement_qbd_treatment_summary.csv",treatment),("concentration_replacement_qbd_concentration_summary.csv",concentration))
    for name,frame in outputs: frame.to_csv(root/name,index=False)
    paired=cells.loc[~cells.replacement.eq(BASELINE_REPLACEMENT)].copy() if not cells.empty else pd.DataFrame(); paired.to_csv(root/"concentration_replacement_qbd_paired_vs_ignore_new.csv",index=False)
    passing=treatment.loc[treatment.treatment_qbd_pass.astype(bool),["max_names","replacement"]] if not treatment.empty else pd.DataFrame(columns=["max_names","replacement"]); design_out=cells.merge(passing,on=["max_names","replacement"],how="inner") if not cells.empty and not passing.empty else pd.DataFrame(); design_out.to_csv(root/"concentration_replacement_qbd_design_space.csv",index=False)
    (root/"concentration_replacement_qbd_summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True,default=str)+"\n",encoding="utf-8")
    return summary
