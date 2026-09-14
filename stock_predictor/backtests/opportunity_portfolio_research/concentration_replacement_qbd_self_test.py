from __future__ import annotations

"""Deterministic contract, replay, resume and evaluator checks for Phase 4."""

import json
from pathlib import Path
import tempfile
import pandas as pd

from . import next_open_portfolio_replay as portfolio, portfolio_policy_search as search
from .portfolio_policy_contracts import Policy
from .cost_contracts import CostModel
from .tax_contracts import TaxConfig
from .allocation_qbd_evaluate import load_primary_phase1_design_space
from .replacement_qbd_evaluate import load_phase2_allocation_lock
from .concentration_replacement_qbd_contract import (
    ALLOCATION_FIXED, BASELINE_REPLACEMENT, CONTRACT_ID, DEFAULT_MAX_NAMES,
    DEFAULT_REPLACEMENTS, EXIT_FAMILY_FIXED, EXIT_VALUE_FIXED, SLEEVE_FIXED,
    parse_max_names, parse_replacement_treatment,
)
from .concentration_replacement_qbd_evaluate import evaluate_surface, load_phase3_provenance, summarize_treatments, validate_surface_coverage
from .concentration_replacement_qbd_surface import ENTRY_GRID_SIZE, _load_complete, _write_json, concentration_grid, concentration_search_contract

EXPECTED_PRIMARY_PLATEAU={(23,3),(23,4),(23,6),(24,3),(24,4),(24,5),(24,6),(24,7),(25,4),(25,6),(25,7),(25,8),(26,7),(26,8)}


def _raises(exc, fn, *args):
    try: fn(*args)
    except exc: return
    raise AssertionError(f"expected {exc.__name__}")


def _provenance_check():
    root=Path(__file__).resolve().parents[3]
    phase1=root/"artifacts/prediction-hold-qbd-surface/qbd_design_space.csv"; assert phase1.is_file(); assert set(load_primary_phase1_design_space(phase1))==EXPECTED_PRIMARY_PLATEAU
    p2s=root/"artifacts/allocation-qbd/allocation_qbd_summary.json"; p2t=root/"artifacts/allocation-qbd/allocation_qbd_treatment_summary.csv"; assert load_phase2_allocation_lock(p2s,p2t)==ALLOCATION_FIXED
    p3s=root/"artifacts/replacement-qbd/replacement_qbd_summary.json"; p3t=root/"artifacts/replacement-qbd/replacement_qbd_treatment_summary.csv"; assert load_phase3_provenance(p3s,p3t)["baseline_replacement"]==BASELINE_REPLACEMENT


def _contract_check():
    assert tuple(parse_max_names(x) for x in DEFAULT_MAX_NAMES)==(1,2,3,4,5); assert 4 not in search.SEARCH_MAX_NAMES
    _raises(ValueError,parse_max_names,0); _raises(ValueError,parse_max_names,6); _raises(ValueError,parse_max_names,"abc")
    assert parse_replacement_treatment("replace_weakest")=="REPLACE_WEAKEST"
    ids=set(); keys=set(); original_names=search.SEARCH_MAX_NAMES
    for n in DEFAULT_MAX_NAMES:
        for tx in DEFAULT_REPLACEMENTS:
            grid=concentration_grid(24,5,n,tx); assert len(grid)==ENTRY_GRID_SIZE==12; assert {p.max_names for p in grid}=={n}; assert {p.replacement for p in grid}=={tx}; assert {p.allocation for p in grid}=={ALLOCATION_FIXED}; assert {p.sleeve for p in grid}=={SLEEVE_FIXED}; assert {p.exit_family for p in grid}=={EXIT_FAMILY_FIXED}
            ids.update(p.policy_id for p in grid)
            with concentration_search_contract(24,5,n,tx):
                keys.add(search._horizon_checkpoint_key(24,ENTRY_GRID_SIZE)); selected,meta=search._balanced_budget(search.grid(24),1); assert len(selected)==ENTRY_GRID_SIZE; assert meta["coverage_complete"] is True
    assert len(ids)==ENTRY_GRID_SIZE*len(DEFAULT_MAX_NAMES)*len(DEFAULT_REPLACEMENTS); assert len(keys)==10; assert search.SEARCH_MAX_NAMES==original_names


def _synthetic(n: int, tx: str):
    dates=pd.bdate_range("2024-01-02",periods=14); tickers=["URTH","AAA","BBB","CCC","DDD","EEE","FFF"]
    prices=pd.DataFrame([{"date":d,"ticker":t,"open":100.0,"close":100.0} for d in dates for t in tickers])
    signal_rows=[]
    for i,t in enumerate(tickers[1:],1): signal_rows.append({"decision_date":dates[i-1],"ticker":t,"score":float(i*10)})
    p=Policy(horizon=24,score_quantile=.5,top_fraction=1.0,max_names=n,holding_days=10,replacement=tx,allocation=ALLOCATION_FIXED,sleeve=SLEEVE_FIXED)
    return portfolio.replay(pd.DataFrame(signal_rows),prices,p,CostModel(roundtrip_bps=0),TaxConfig(enabled=False),initial=10000,resolved_threshold=0)


def _replay_check():
    for n in DEFAULT_MAX_NAMES:
        ignore=_synthetic(n,BASELINE_REPLACEMENT); replace=_synthetic(n,"REPLACE_WEAKEST")
        for r in (ignore,replace):
            m=r["metrics"]; assert int(m["max_positions"])<=n; assert float(m["max_total_exposure"])<=1+5e-6; assert float(m["max_abs_accounting_error_eur"])<=1e-7
        assert float(replace["metrics"]["turnover"])>=float(ignore["metrics"]["turnover"])
    assert [t["ticker"] for t in _synthetic(1,BASELINE_REPLACEMENT)["trades"]]==["AAA"]
    assert len(_synthetic(1,"REPLACE_WEAKEST")["trades"])>1


def _resume_check():
    valid={"contract_id":CONTRACT_ID,"status":"COMPLETE","prediction_horizon":24,"holding_days":5,"max_names":4,"replacement":"REPLACE_WEAKEST","max_names_removed_from_inner_search":True,"phase3_replacement_reopened":True,"allocation":ALLOCATION_FIXED,"sleeve":SLEEVE_FIXED,"exit_family":EXIT_FAMILY_FIXED,"exit_value":EXIT_VALUE_FIXED,"v45_exit_overlay_used":False,"final_holdout_opened":False}
    with tempfile.TemporaryDirectory() as td:
        p=Path(td)/"cell.json"; _write_json(p,valid); assert _load_complete(p,24,5,4,"REPLACE_WEAKEST") is not None
        for field,value in (("max_names",3),("replacement",BASELINE_REPLACEMENT),("allocation","RANK_POWER:1.0"),("sleeve",.75),("max_names_removed_from_inner_search",False),("phase3_replacement_reopened",False),("final_holdout_opened",True)):
            bad=dict(valid); bad[field]=value; _write_json(p,bad); assert _load_complete(p,24,5,4,"REPLACE_WEAKEST") is None


def _fixture_cells(design, spike=False):
    outer=[]; status=[]
    for h,d in design:
        for n in DEFAULT_MAX_NAMES:
            for tx in DEFAULT_REPLACEMENTS:
                status.append({"prediction_horizon":h,"holding_days":d,"max_names":n,"replacement":tx,"status":"COMPLETE"})
                base=.08 + .002*n
                if tx==BASELINE_REPLACEMENT: vals=(base+.02,base+.01,base,-.01)
                else:
                    delta=.02 if n==3 else (.50 if spike and n==5 and (h,d)==design[0] else (-.01 if spike and n==5 else .01))
                    vals=tuple(v+delta for v in (base+.02,base+.01,base,-.01))
                for i,v in enumerate(vals,1): outer.append({"prediction_horizon":h,"holding_days":d,"max_names":n,"replacement":tx,"fold_id":f"F{i}","cagr_excess":v,"trade_count":3,"turnover":1+.1*n+(.2 if tx!="IGNORE_NEW" else 0),"worst_relative_drawdown":-.1})
    return pd.DataFrame(outer),pd.DataFrame(status)


def _evaluator_check():
    design=sorted(EXPECTED_PRIMARY_PLATEAU); outer,status=_fixture_cells(design)
    cells,treatment,concentration,summary=evaluate_surface(outer,status,design,DEFAULT_MAX_NAMES,DEFAULT_REPLACEMENTS)
    assert summary["qbd_complete"] is True; assert summary["expected_surface_cells"]==140; assert len(cells)==140; assert set(concentration.max_names.astype(int))==set(DEFAULT_MAX_NAMES)
    n3=treatment.loc[(treatment.max_names.eq(3)) & treatment.replacement.eq("REPLACE_WEAKEST")].iloc[0]; assert bool(n3.replacement_qbd_pass_within_max_names); assert float(n3.beat_ignore_new_cell_fraction)==1
    broken=status.copy(); broken.loc[0,"status"]="FAILED"; assert evaluate_surface(outer,broken,design,DEFAULT_MAX_NAMES,DEFAULT_REPLACEMENTS)[3]["qbd_complete"] is False
    cov=validate_surface_coverage(pd.concat([status,status.iloc[[0]]]),[(h,d,n,r) for h,d in design for n in DEFAULT_MAX_NAMES for r in DEFAULT_REPLACEMENTS]); assert cov["surface_complete"] is False and cov["duplicate_cells"]


def _broad_stability_check():
    design=sorted(EXPECTED_PRIMARY_PLATEAU); outer,status=_fixture_cells(design,spike=True); cells,treatment,_,_=evaluate_surface(outer,status,design,DEFAULT_MAX_NAMES,DEFAULT_REPLACEMENTS)
    stable=treatment.loc[(treatment.max_names.eq(3)) & treatment.replacement.eq("REPLACE_WEAKEST")].iloc[0]; spike=treatment.loc[(treatment.max_names.eq(5)) & treatment.replacement.eq("REPLACE_WEAKEST")].iloc[0]
    assert bool(stable.replacement_qbd_pass_within_max_names); assert not bool(spike.replacement_qbd_pass_within_max_names); assert float(spike.beat_ignore_new_cell_fraction)<.60


def main() -> int:
    _provenance_check(); _contract_check(); _replay_check(); _resume_check(); _evaluator_check(); _broad_stability_check()
    assert len(EXPECTED_PRIMARY_PLATEAU)==14 and len(DEFAULT_MAX_NAMES)*len(DEFAULT_REPLACEMENTS)*14==140 and ENTRY_GRID_SIZE==12
    print("CONCENTRATION_REPLACEMENT_QBD_END_TO_END_SELF_TEST_PASS"); return 0


if __name__=="__main__": raise SystemExit(main())
