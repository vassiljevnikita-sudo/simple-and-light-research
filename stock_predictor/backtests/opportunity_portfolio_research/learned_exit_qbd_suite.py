from __future__ import annotations

import argparse, json
from contextlib import contextmanager
import os
from pathlib import Path
import pandas as pd

from . import portfolio_policy_search as search
from .portfolio_policy_contracts import Policy
from .portfolio_research_inputs import load_predictions, load_price_panel, assert_no_final_frozen_holdout
from .learned_exit_qbd_profit import (
    CONTRACT_ID, LEARNED_EXIT_SOURCE_COMMIT, LEARNED_EXIT_METHODOLOGY, ALLOCATION,
    REPLACEMENT, SLEEVE, MAX_NAMES, FIXED_ISLAND, PRIMARY_ROUNDTRIP_BPS,
    ProfitTaxConfig, learned_cells, assert_contract,
)
from .learned_exit_qbd_provider import LearnedExitProvider
from .learned_exit_qbd_replay import configure_replay, replay as qbd_replay
from . import portfolio_replay_process_backend as backend
from . import portfolio_policy_walk_forward as multicore_walk_forward
from .prediction_hold_qbd_process_pool_readiness import qbd_process_pool_readiness_contract
from .prediction_hold_qbd_throughput_runtime import _install_qbd_multicore_backend, _qbd_effective_workers
from .portfolio_resilient_process_pool import install_resilient_process_pool, restore_process_pool_runner
from .affinity_coordinator_pool import AffinityCoordinatorPool

ENTRY_GRID_SIZE=len(search.SEARCH_QUANTILES)*len(search.SEARCH_TOP_FRACTIONS)

@contextmanager
def learned_qbd_runtime(max_workers: int):
    """Use the shared QbD pool/readiness runtime for the learned-exit surface."""
    old_effective = backend._effective_workers
    old_parallel_map = search._parallel_map
    old_coordinator_executor = multicore_walk_forward.ThreadPoolExecutor
    old_window_pipeline = os.environ.get("OPPORTUNITY_WINDOW_PIPELINE")
    backend._effective_workers = _qbd_effective_workers
    _install_qbd_multicore_backend(max_workers)
    install_resilient_process_pool()
    multicore_walk_forward.ThreadPoolExecutor = AffinityCoordinatorPool
    os.environ["OPPORTUNITY_WINDOW_PIPELINE"] = "8"
    try:
        with qbd_process_pool_readiness_contract():
            yield
    finally:
        multicore_walk_forward.ThreadPoolExecutor = old_coordinator_executor
        restore_process_pool_runner()
        backend.shutdown_multicore_backend()
        backend._effective_workers = old_effective
        search._parallel_map = old_parallel_map
        if old_window_pipeline is None:
            os.environ.pop("OPPORTUNITY_WINDOW_PIPELINE", None)
        else:
            os.environ["OPPORTUNITY_WINDOW_PIPELINE"] = old_window_pipeline

def _write_json(path:Path,value):
    path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix(path.suffix+'.tmp'); tmp.write_text(json.dumps(value,indent=2,sort_keys=True,default=str)+'\n',encoding='utf-8'); tmp.replace(path)

def _cell_path(root,mode,h,d,n): return root/'cells'/mode.lower()/f'H{h:02d}_D{d:02d}_N{n}.json'

@contextmanager
def cell_contract(h:int,d:int,n:int,mode:str):
    if mode not in {'FIXED','LEARNED_EXIT'}: raise ValueError(mode)
    old=(search.grid,search._dynamic_neighbors,search.replay,search.SEARCH_MAX_NAMES,search.SEARCH_HOLDING_DAYS,search._window_checkpoint_key,search._final_fit_checkpoint_key,search._horizon_checkpoint_key)
    search.SEARCH_MAX_NAMES=(n,); search.SEARCH_HOLDING_DAYS=(d,)
    search.grid=lambda horizon,sleeve=SLEEVE:[Policy(int(horizon),q,top,n,d,mode,0.0,REPLACEMENT,ALLOCATION,SLEEVE) for q in search.SEARCH_QUANTILES for top in search.SEARCH_TOP_FRACTIONS]
    search._dynamic_neighbors=lambda base:[]; search.replay=qbd_replay
    prefix=(CONTRACT_ID,mode,h,d,n,PRIMARY_ROUNDTRIP_BPS)
    search._window_checkpoint_key=lambda horizon,fold,history_end,budget:search._cache_key(*prefix,'outer',int(horizon),str(fold),str(pd.Timestamp(history_end)),int(budget))
    search._final_fit_checkpoint_key=lambda horizon,history_end,budget:search._cache_key(*prefix,'final',int(horizon),str(pd.Timestamp(history_end)),int(budget))
    search._horizon_checkpoint_key=lambda horizon,budget:search._cache_key(*prefix,'cell',int(horizon),int(budget))
    try: yield
    finally:
        search.grid,search._dynamic_neighbors,search.replay,search.SEARCH_MAX_NAMES,search.SEARCH_HOLDING_DAYS,search._window_checkpoint_key,search._final_fit_checkpoint_key,search._horizon_checkpoint_key=old

def _aggregate(outer,initial=10000.0):
    active=[r for r in outer if int(r.get('trade_count',0))>0]; vals=[float(r.get('cagr_excess',0)) for r in active]; sg=bg=1.0
    for r in outer: sg*=1+float(r.get('total_return',0)); bg*=1+float(r.get('urth_total_return',0))
    req=sum(int(r.get('required_exit_decisions',0)) for r in outer); avail=sum(int(r.get('available_exit_decisions',0)) for r in outer)
    return {'outer_folds':len(outer),'active_folds':len(active),'positive_active_fold_fraction':sum(v>0 for v in vals)/len(vals) if vals else 0.0,'median_active_cagr_excess':float(pd.Series(vals).median()) if vals else 0.0,'q25_active_cagr_excess':float(pd.Series(vals).quantile(.25)) if vals else 0.0,'worst_active_cagr_excess':min(vals) if vals else 0.0,'fold_compounded_terminal_value':initial*sg,'fold_compounded_urth_terminal_value':initial*bg,'fold_compounded_wealth_excess_eur':initial*(sg-bg),'total_trades':sum(int(r.get('trade_count',0)) for r in outer),'total_tax_paid_eur':sum(float(r.get('tax_paid',0)) for r in outer),'total_benchmark_tax_paid_eur':sum(float(r.get('benchmark_tax_paid',0)) for r in outer),'mean_turnover':float(pd.Series([float(r.get('turnover',0)) for r in outer]).mean()) if outer else 0.0,'learned_exit_count':sum(int(r.get('learned_exit_count',0)) for r in outer),'required_exit_decisions':req,'available_exit_decisions':avail,'exit_decision_coverage':1.0 if req==0 else avail/req}

def run_cell(hpred,prices,*,h,d,n,mode,budget,max_workers):
    with cell_contract(h,d,n,mode): outer,history,meta=search.run_walk_forward(hpred,prices,horizons=(h,),budget=budget,max_workers=max_workers)
    p=meta.get('final_policies',{}).get(h); final_policy=dict(p.__dict__) if p is not None else None
    return {'contract_id':CONTRACT_ID,'status':'COMPLETE','mode':mode,'prediction_horizon':h,'holding_days':d,'max_names':n,'allocation':ALLOCATION,'replacement':REPLACEMENT,'sleeve':SLEEVE,'entry_grid_size':ENTRY_GRID_SIZE,'max_names_removed_from_inner_search':True,'holding_days_removed_from_inner_search':True,'exit_removed_from_inner_search':True,'learned_exit_source_commit':LEARNED_EXIT_SOURCE_COMMIT,'learned_exit_methodology':LEARNED_EXIT_METHODOLOGY,'final_holdout_opened':False,'promotion_eligible':False,'outer_rows':outer,'final_policy':final_policy,'final_threshold':meta.get('final_thresholds',{}).get(h),'profit_summary':_aggregate(outer)}

def run_surface(a):
    assert_contract(); tax=ProfitTaxConfig(allowance_eur=a.tax_allowance_eur,church_tax_rate=a.church_tax_rate,benchmark_partial_exemption_rate=a.benchmark_partial_exemption)
    provider=LearnedExitProvider(Path(a.learned_exit_predictions)); configure_replay(provider,tax)
    predictions,pred_audit=load_predictions(Path(a.v5_predictions)); missing=sorted(set(range(1,31))-set(int(x) for x in predictions.horizon.unique()))
    if missing: raise RuntimeError(f'V5_HORIZONS_MISSING:{missing}')
    prices,price_audit=load_price_panel(Path(a.daily_store_root),set(predictions.loc[predictions.horizon.isin(range(1,31)),'ticker'].unique()))
    root=Path(a.output_root); root.mkdir(parents=True,exist_ok=True); names=tuple(dict.fromkeys(int(x) for x in a.max_names.split(',') if x.strip()))
    if not names or any(n not in MAX_NAMES for n in names): raise ValueError('max-names must be subset of 1..6')
    learned=list(learned_cells(names)); fixed=[(h,d,1) for h,d in FIXED_ISLAND]; requested=[('LEARNED_EXIT',*x) for x in learned]+[('FIXED',*x) for x in fixed]
    completed=reused=0; failed=[]
    with learned_qbd_runtime(a.max_workers):
      for i,(mode,h,d,n) in enumerate(requested,1):
          path=_cell_path(root,mode,h,d,n)
          if path.is_file() and not a.force:
              try:
                  old=json.loads(path.read_text(encoding='utf-8'))
                  if old.get('contract_id')==CONTRACT_ID and old.get('status')=='COMPLETE' and old.get('mode')==mode and int(old.get('prediction_horizon',-1))==h and int(old.get('holding_days',-1))==d and int(old.get('max_names',-1))==n and old.get('learned_exit_source_commit')==LEARNED_EXIT_SOURCE_COMMIT and old.get('learned_exit_methodology')==LEARNED_EXIT_METHODOLOGY and old.get('tax_fingerprint')==tax.fingerprint(): reused+=1; completed+=1; continue
              except Exception: pass
          print(f'[learned-exit-qbd] {i}/{len(requested)} {mode} H{h:02d}/D{d:02d}/N{n}',flush=True); hpred=predictions.loc[predictions.horizon.eq(h)].copy()
          try:
              payload=run_cell(hpred,prices,h=h,d=d,n=n,mode=mode,budget=a.entry_budget,max_workers=a.max_workers); payload['tax_config']=tax.__dict__; payload['tax_fingerprint']=tax.fingerprint(); _write_json(path,payload); completed+=1
          except Exception as exc:
              fail={'contract_id':CONTRACT_ID,'status':'FAILED','mode':mode,'prediction_horizon':h,'holding_days':d,'max_names':n,'error':f'{type(exc).__name__}:{exc}','final_holdout_opened':False}; _write_json(path,fail); failed.append(fail)
              if a.stop_on_error: raise
    result={'contract_id':CONTRACT_ID,'status':'COMPLETE' if not failed and completed==len(requested) else 'INCOMPLETE','requested_cells':len(requested),'learned_cells':len(learned),'fixed_reference_cells':len(fixed),'completed_cells':completed,'reused_cells':reused,'failed_cells':failed,'learned_exit_source_commit':LEARNED_EXIT_SOURCE_COMMIT,'learned_exit_methodology':LEARNED_EXIT_METHODOLOGY,'provider_audit':provider.audit.__dict__,'tax_config':tax.__dict__,'tax_fingerprint':tax.fingerprint(),'roundtrip_bps':20.0,'initial_capital':10000.0,'prediction_audit':pred_audit,'price_audit':price_audit,'final_holdout_opened':False,'final_holdout_locked':True,'promotion_eligible':False,'benchmark_tax_is_approximate':True,'benchmark_vorabpauschale_mode':tax.benchmark_vorabpauschale_mode,'fold_tax_ledgers_are_independent':True}; _write_json(root/'run_summary.json',result); return result

def _read_cells(root):
    rows=[]
    for p in (root/'cells').rglob('*.json'):
        x=json.loads(p.read_text(encoding='utf-8'))
        if x.get('contract_id')==CONTRACT_ID and x.get('status')=='COMPLETE': rows.append({'mode':x['mode'],'h':x['prediction_horizon'],'d':x['holding_days'],'n':x['max_names'],**x['profit_summary']})
    return pd.DataFrame(rows)

def _islands(df):
    out=[]; iid=0
    for n,g in df.loc[df["mode"].eq('LEARNED_EXIT')].groupby('n'):
        positive={(int(r.h),int(r.d)) for r in g.itertuples() if float(r.fold_compounded_wealth_excess_eur)>0}
        while positive:
            seed=positive.pop(); comp={seed}; stack=[seed]
            while stack:
                h,d=stack.pop()
                for nb in ((h-1,d),(h+1,d),(h,d-1),(h,d+1)):
                    if nb in positive: positive.remove(nb); comp.add(nb); stack.append(nb)
            iid+=1; sub=g.loc[g.apply(lambda r:(int(r.h),int(r.d)) in comp,axis=1)]; b=sub.sort_values('fold_compounded_wealth_excess_eur',ascending=False).iloc[0]
            out.append({'island_id':iid,'max_names':int(n),'cells':len(comp),'median_wealth_excess_eur':float(sub.fold_compounded_wealth_excess_eur.median()),'max_wealth_excess_eur':float(b.fold_compounded_wealth_excess_eur),'best_h':int(b.h),'best_d':int(b.d),'best_n':int(b.n)})
    return sorted(out,key=lambda x:(x['max_wealth_excess_eur'],x['median_wealth_excess_eur']),reverse=True)

def evaluate(root:Path):
    df=_read_cells(root)
    if df.empty: raise RuntimeError('NO_COMPLETE_CELLS')
    learned=df.loc[df["mode"].eq('LEARNED_EXIT')].copy(); fixed=df.loc[df["mode"].eq('FIXED')].copy(); observed={(int(r.h),int(r.d)) for r in fixed.itertuples()}
    if observed!=set(FIXED_ISLAND): raise RuntimeError('FIXED_REFERENCE_INCOMPLETE')
    islands=_islands(df); bl=learned.sort_values('fold_compounded_wealth_excess_eur',ascending=False).iloc[0].to_dict(); bf=fixed.sort_values('fold_compounded_wealth_excess_eur',ascending=False).iloc[0].to_dict(); champion=bl if bl['fold_compounded_wealth_excess_eur']>bf['fold_compounded_wealth_excess_eur'] else bf
    by_n=[]
    for n,g in learned.groupby('n'):
        b=g.sort_values('fold_compounded_wealth_excess_eur',ascending=False).iloc[0]; by_n.append({'max_names':int(n),'point_count':len(g),'median_wealth_excess_eur':float(g.fold_compounded_wealth_excess_eur.median()),'max_wealth_excess_eur':float(b.fold_compounded_wealth_excess_eur),'best_h':int(b.h),'best_d':int(b.d)})
    result={'contract_id':CONTRACT_ID,'status':'COMPLETE','selection_rule':'HIGHEST_OOS_FOLD_COMPOUNDED_AFTER_TAX_WEALTH_EXCESS_VS_AFTER_TAX_URTH','island_is_context_not_hard_gate':True,'learned_point_count':len(learned),'fixed_reference_point_count':len(fixed),'learned_median_wealth_excess_eur':float(learned.fold_compounded_wealth_excess_eur.median()),'learned_max_wealth_excess_eur':float(bl['fold_compounded_wealth_excess_eur']),'fixed_island_median_wealth_excess_eur':float(fixed.fold_compounded_wealth_excess_eur.median()),'fixed_island_max_wealth_excess_eur':float(bf['fold_compounded_wealth_excess_eur']),'best_learned_point':bl,'best_fixed_point':bf,'learned_islands':islands,'best_learned_island':islands[0] if islands else None,'learned_by_max_names':by_n,'champion':champion,'champion_locked_for_frozen_validation':True,'final_holdout_opened':False,'promotion_eligible':False}; (root/'point_ranking.csv').write_text(df.sort_values('fold_compounded_wealth_excess_eur',ascending=False).to_csv(index=False),encoding='utf-8'); _write_json(root/'islands.json',{'islands':islands}); _write_json(root/'evaluation_summary.json',result); return result

def self_test():
    assert_contract(); assert len(learned_cells())==2790 and ENTRY_GRID_SIZE==12
    assert_no_final_frozen_holdout(pd.DataFrame({'holdout_locked':[False], 'dataset_role':['DEVELOPMENT_OOS']}), source='self-test')
    try:
        assert_no_final_frozen_holdout(pd.DataFrame({'holdout_locked':[True]}), source='self-test')
    except ValueError as exc:
        assert str(exc).startswith('FINAL_FROZEN_HOLDOUT_ENTRY_PREDICTIONS_REJECTED:')
    else:
        raise AssertionError('holdout guard did not reject locked rows')
    try:
        assert_no_final_frozen_holdout(pd.DataFrame({'dataset_role':['FINAL_FROZEN_HOLDOUT']}), source='self-test')
    except ValueError as exc:
        assert str(exc).startswith('FINAL_FROZEN_HOLDOUT_ENTRY_PREDICTIONS_REJECTED:')
    else:
        raise AssertionError('holdout role guard did not reject final rows')
    with cell_contract(25,15,6,'LEARNED_EXIT'):
        g=search.grid(25); assert len(g)==12 and {p.max_names for p in g}=={6} and {p.holding_days for p in g}=={15} and {p.exit_family for p in g}=={'LEARNED_EXIT'} and search._dynamic_neighbors(g[0])==[]
    with cell_contract(25,7,1,'FIXED'): assert {p.exit_family for p in search.grid(25)}=={'FIXED'}
    print('LEARNED_EXIT_QBD_PROFIT_SELF_TEST_PASS'); return 0

def parse_args():
    p=argparse.ArgumentParser(description='Learned Exit HxDxN QbD profit suite'); p.add_argument('--v5-predictions'); p.add_argument('--daily-store-root'); p.add_argument('--learned-exit-predictions'); p.add_argument('--output-root',default='artifacts/learned-exit-qbd-profit'); p.add_argument('--max-names',default='1,2,3,4,5,6'); p.add_argument('--max-workers',type=int,default=8); p.add_argument('--entry-budget',type=int,default=ENTRY_GRID_SIZE); p.add_argument('--tax-allowance-eur',type=float,default=1000.0); p.add_argument('--church-tax-rate',type=float,default=0.0); p.add_argument('--benchmark-partial-exemption',type=float,default=.30); p.add_argument('--force',action='store_true'); p.add_argument('--stop-on-error',action='store_true'); p.add_argument('--self-test',action='store_true'); return p.parse_args()
def main():
    a=parse_args()
    if a.self_test: return self_test()
    if not a.v5_predictions or not a.daily_store_root or not a.learned_exit_predictions: raise ValueError('three data inputs are required')
    run=run_surface(a); ev=evaluate(Path(a.output_root)) if run['status']=='COMPLETE' else None; print(json.dumps({'run_status':run['status'],'evaluation_status':None if ev is None else ev['status'],'champion':None if ev is None else ev['champion'],'final_holdout_opened':False},indent=2,default=str)); return 0 if ev is not None else 2
if __name__=='__main__': raise SystemExit(main())
