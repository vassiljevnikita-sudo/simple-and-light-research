"""Run the QBD router against the locally available frozen prediction stream.

This is intentionally a live-preview harness: it exercises chronological
routing and state persistence, but never labels predictions as independent
holdout performance.  Realized outcomes must be supplied by a separately
validated provider before a performance conclusion is possible.
"""
from __future__ import annotations
import argparse, json
from datetime import date
from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq
from .top10_qbd_router_contracts import default_policy
from .top10_qbd_expert_registry import build_top10_qbd_registry
from .top10_qbd_shadow_engine import make_shadow_decision
from .top10_qbd_shadow_ledger import MaturedOutcome, ShadowDecision, PendingOutcome
from .top10_qbd_prequential_replay import run_qbd_router_replay
from .top10_qbd_state_store import JsonRouterStateStore
from .top10_qbd_freeze import build_manifest, write_manifest
from .top10_qbd_ablation import run_all_arms
from .top10_qbd_expert_runtime import runtime_for_registry
from .top10_qbd_evaluation import router_adaptation_metrics
from .learned_exit_qbd_provider import LearnedExitProvider
from .top10_qbd_market_state import market_state_as_of
from .top10_qbd_exit_runtime import LearnedExitRuntime

class FrozenPredictionPreviewProvider:
    def __init__(self, path, start, end, daily_store_root=None, stock_roundtrip_bps=20.0, learned_exit_path=None):
        self.daily_store_root=Path(daily_store_root) if daily_store_root else None; self.stock_roundtrip_bps=stock_roundtrip_bps; self._prices={}; self._paths={}; self.runtimes={}; self.learned_exit=LearnedExitProvider(learned_exit_path) if learned_exit_path else None
        if self.daily_store_root:
            for candidate in self.daily_store_root.rglob('*.parquet'):
                text=str(candidate); marker='ticker='
                if marker in text:
                    ticker=text.split(marker,1)[1].split('\\',1)[0].split('/',1)[0]; self._paths.setdefault(ticker,candidate)
        frame=pd.read_parquet(path, columns=['decision_date','ticker','predicted_net_excess_return','horizon_sessions','family'])
        frame['decision_date']=pd.to_datetime(frame['decision_date']).dt.date
        frame=frame[(frame.decision_date>=start)&(frame.decision_date<=end)]
        # Keep the complete horizon/date groups.  Frozen top_fraction is
        # defined against the original group size; truncating to a preview
        # top-N before runtime selection changes the frozen policy.
        frame=frame.sort_values(['horizon_sessions','decision_date','predicted_net_excess_return'],ascending=[True,True,False])
        self._frame_dates=tuple(sorted(frame.decision_date.unique()))
        self.values={}
        for (h,d), group in frame.groupby(['horizon_sessions','decision_date'],sort=False):
            self.values[(int(h),d)]=tuple({'ticker':str(row.ticker),'predicted_net_excess_return':float(row.predicted_net_excess_return),'horizon_sessions':int(row.horizon_sessions)} for row in group.itertuples())
        self.sessions=self._frame_dates
        self._open_until={}
        self._learned_exit_runtime=LearnedExitRuntime(self.learned_exit) if self.learned_exit else None
        self._benchmark_closes=None
    def dates(self,start,end): return tuple(sorted({d for _, d in self.values}))
    def shadow(self,current_date,expert):
        if expert.expert_type.value != 'STOCK':
            return make_shadow_decision(decision_date=current_date,expert_id=expert.expert_id,eligible_opportunity=False,prediction=None,threshold=None,holding_days=1)
        rows=self.values.get((int(expert.horizon),current_date),()); runtime=self.runtimes.get(expert.expert_id)
        open_now=tuple(t for t, until in self._open_until.get(expert.expert_id, {}).items() if until > current_date)
        market_state = self._market_state(current_date)
        shadow=runtime.decision(decision_date=current_date,available_predictions=rows,market_state=market_state,open_tickers=open_now) if runtime else None
        prediction=shadow.candidates[0].score if shadow and shadow.candidates else None; tickers=tuple(x.ticker for x in shadow.selected_positions) if shadow else ()
        # The expert identity selects its frozen horizon stream.  It is not
        # reassigned from a cross-expert daily rank; max_names remains part of
        # the registry identity and is applied by the downstream policy.
        decision,pending=make_shadow_decision(decision_date=current_date,expert_id=expert.expert_id,eligible_opportunity=prediction is not None,prediction=prediction,threshold=None,holding_days=int(expert.holding_days or 1))
        if pending is not None and tickers:
            max_index=None
            try:
                decision_index=self.sessions.index(current_date)
                entry_index=min(len(self.sessions)-1,decision_index+1)
                max_index=min(len(self.sessions)-1,entry_index+int(expert.holding_days or 1))
            except (ValueError,IndexError): pass
            if max_index is not None:
                exit_dates={}
                for ticker in tickers:
                    index=decision_index; selected_index=max_index
                    if self.learned_exit and self.runtimes[expert.expert_id].spec.exit_policy_id == 'LEARNED_EXIT':
                        # E_h is evaluated only after entry, at a session
                        # close.  Any close-derived exit is executed at the
                        # following session open; fixed D exits are also the
                        # open after D held sessions.
                        for probe in range(entry_index,max_index):
                            remaining=max_index-probe
                            exit_decision=self._learned_exit_runtime.evaluate(ticker=ticker,current_date=self.sessions[probe],remaining_sessions=remaining)
                            if exit_decision.should_exit:
                                selected_index=min(max_index,probe+1); break
                    exit_dates[ticker]=self.sessions[selected_index]
                exit_date=max(exit_dates.values())
            else:
                exit_dates={ticker: pending.outcome_available_at for ticker in tickers}; exit_date=pending.outcome_available_at
            decision=ShadowDecision(decision.decision_date,decision.expert_id,decision.eligible_opportunity,decision.prediction,decision.threshold,decision.would_enter,decision.would_exit,decision.market_regime,decision.market_state_features,decision.prediction_uncertainty,exit_date)
            decision=ShadowDecision(decision.decision_date,decision.expert_id,decision.eligible_opportunity,decision.prediction,decision.threshold,decision.would_enter,decision.would_exit,decision.market_regime,decision.market_state_features,decision.prediction_uncertainty,exit_date,shadow.candidates,shadow.selected_positions,shadow.allocation_fraction)
            entry_date=self.sessions[entry_index]
            pending=PendingOutcome(current_date,expert.expert_id,exit_date,f'{entry_date.isoformat()}::{expert.expert_id}::'+','.join(f'{t}@{exit_dates[t].isoformat()}' for t in tickers),getattr(market_state,'regime_id','GLOBAL'))
            self._open_until.setdefault(expert.expert_id,{}).update({t: exit_dates[t] for t in tickers})
        return decision,pending
    def outcome(self,pending):
        if not self.daily_store_root: return None
        parts=pending.opaque_outcome_key.split('::'); encoded=tuple(parts[2].split(',')) if len(parts)>2 else (); positions=tuple((x.split('@',1)[0],date.fromisoformat(x.split('@',1)[1])) for x in encoded if '@' in x); tickers=tuple(x[0] for x in positions)
        if not tickers or any(t not in self._paths for t in tickers): return None
        entry=date.fromisoformat(parts[0]); exit_date=pending.outcome_available_at
        def prices(symbol):
            if symbol not in self._prices:
                t=pq.read_table(self._paths[symbol],columns=['session_date','open','close']).to_pandas(); t['session_date']=pd.to_datetime(t['session_date']).dt.date; self._prices[symbol]=t.set_index('session_date')
            return self._prices[symbol]
        benchmark=prices('URTH') if 'URTH' in self._paths else None
        if benchmark is None or entry not in benchmark.index or exit_date not in benchmark.index: return None
        stock_returns=[]; benchmark_returns=[]
        for ticker,position_exit in positions:
            stock=prices(ticker)
            if entry not in stock.index or position_exit not in stock.index or position_exit not in benchmark.index: return None
            # Both legs use the contract's execution prices: next-session
            # open entry and next-session open exit.  No exit-day close is
            # visible to an order executed at that day's open.
            stock_returns.append(float(stock.loc[position_exit,'open']/stock.loc[entry,'open']-1)); benchmark_returns.append(float(benchmark.loc[position_exit,'open']/benchmark.loc[entry,'open']-1))
        stock_ret=sum(stock_returns)/len(stock_returns); bench_ret=sum(benchmark_returns)/len(benchmark_returns); cost=self.stock_roundtrip_bps/10000.0
        return MaturedOutcome(pending.decision_date,pending.expert_id,exit_date,stock_ret,bench_ret,cost,stock_ret-bench_ret-cost,pending.regime_id)

    def _market_state(self, current_date):
        if self._benchmark_closes is None:
            if 'URTH' not in self._paths:
                self._benchmark_closes=()
            else:
                t=pq.read_table(self._paths['URTH'],columns=['session_date','close']).to_pandas()
                t['session_date']=pd.to_datetime(t['session_date']).dt.date
                self._benchmark_closes=tuple((r.session_date,float(r.close)) for r in t.itertuples())
        return market_state_as_of(current_date,self._benchmark_closes)

def run_preview(predictions, output, start=date(2016,1,1), end=date(2026,7,24), all_arms=False, daily_store_root=None, learned_exit_path=None):
    output=Path(output); output.mkdir(parents=True,exist_ok=True); policy=default_policy(); registry=build_top10_qbd_registry()
    provider=FrozenPredictionPreviewProvider(predictions,start,end,daily_store_root=daily_store_root,stock_roundtrip_bps=policy.stock_roundtrip_bps,learned_exit_path=learned_exit_path); provider.runtimes=runtime_for_registry(registry); store=JsonRouterStateStore(output/'router_state.json',policy_hash=policy.policy_hash,registry_hash=__import__('stock_predictor.backtests.opportunity_portfolio_research.top10_qbd_expert_registry',fromlist=['registry_hash']).registry_hash(registry))
    result=run_qbd_router_replay(policy=policy,registry=registry,start_date=start,end_date=end,state_store=store,data_provider=provider)
    arm_results=run_all_arms(registry=registry,start_date=start,end_date=end,data_provider=provider) if all_arms else {}
    manifest=build_manifest(policy=policy,registry=registry,code_commit='LOCAL_PREVIEW',data_fingerprint=str(Path(predictions).stat().st_size),evaluation_contract_hash='TOP10_QBD_ROUTER_V1_TEST_CONTRACT')
    write_manifest(output/'QBD_ROUTER_V1_FROZEN_MANIFEST.json',manifest)
    validation='PSEUDO_OOS_DEVELOPMENT_WITH_REALIZED_DAILY_OUTCOMES' if daily_store_root else 'LIVE_PREVIEW_NO_REALIZED_OUTCOMES'
    available=provider.dates(start,end); observed_start=available[0] if available else start; observed_end=available[-1] if available else end
    arm_summary={k:{'decision_count':len(v.decisions),**router_adaptation_metrics(v.decisions)} for k,v in arm_results.items()}
    arm_summary.setdefault(policy.router_arm.value,{'decision_count':len(available),**router_adaptation_metrics(result.decisions)})
    (output/'summary.json').write_text(json.dumps({'status':'LIVE_PREVIEW_COMPLETE','validation_status':validation,'requested_start':str(start),'requested_end':str(end),'observed_start':str(observed_start),'observed_end':str(observed_end),'training_years':2,'decision_count':len(available),'policy_hash':policy.policy_hash,'ablation_arms':arm_summary},indent=2)+'\n')
    (output/'REPORT.md').write_text('# TOP10 QBD Router V1 live preview\n\nCompleted chronological preview over the frozen per-horizon prediction streams.\n\n- Requested window: %s through %s\n- Observed data window: %s through %s\n- Training context: 2 years requested\n- Decisions: %d\n- Status: `%s`\n' % (start,end,observed_start,observed_end,len(available),validation))
    print(json.dumps({'status':'LIVE_PREVIEW_COMPLETE','decisions':len(available),'observed_end':str(observed_end),'output':str(output)},indent=2))

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--predictions',default='artifacts/top10-frozen-live-replay/frozen-entry-predictions.parquet'); ap.add_argument('--output',default='artifacts/top10-qbd-router-v1'); ap.add_argument('--start',default='2016-01-01'); ap.add_argument('--end',default='2026-07-24'); ap.add_argument('--all-arms',action='store_true'); ap.add_argument('--daily-store-root'); ap.add_argument('--learned-exit-predictions'); a=ap.parse_args(); run_preview(a.predictions,a.output,date.fromisoformat(a.start),date.fromisoformat(a.end),a.all_arms,a.daily_store_root,a.learned_exit_predictions)
