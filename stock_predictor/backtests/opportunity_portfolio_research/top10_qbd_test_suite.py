"""Fast contract, causality, synthetic and restart tests for the QbD router."""
from __future__ import annotations
import argparse, tempfile
from datetime import date, timedelta
from dataclasses import replace
from .top10_qbd_router_contracts import default_policy, canonical_json
from .top10_qbd_policy import RouterArm
from .top10_qbd_expert_registry import build_top10_qbd_registry, validate_registry
from .top10_qbd_activity_monitor import update_activity
from .top10_qbd_health_metrics import ActivityPosterior
from .top10_qbd_evaluation import concentration
from .top10_qbd_freeze import build_manifest, write_manifest, validate_manifest
from .top10_qbd_shadow_ledger import InMemoryShadowLedger, PendingOutcome, MaturedOutcome
from .top10_qbd_allocation import allocation_weights, validate_weights
from .top10_qbd_ablation import arm_manifest
from .top10_qbd_promotion import assess_promotion
from .top10_qbd_health_metrics import AlphaHealth
from .top10_qbd_state_store import JsonRouterStateStore
from .top10_qbd_prequential_replay import run_qbd_router_replay
from .top10_qbd_shadow_engine import make_shadow_decision
from .top10_qbd_expert_runtime import FrozenExpertRuntimeImpl
from .top10_qbd_market_state import market_state_as_of
from .top10_qbd_promotion import relative_edge_lcb

def run_self_test():
    p=default_policy(); assert p.to_json()==p.to_json(); assert p.policy_hash != replace(p,softmax_eta=5).policy_hash
    r=build_top10_qbd_registry(); validate_registry(r); assert len(r)==12
    a=ActivityPosterior(1,9,0,0,0,None); h=update_activity(a,eligible_opportunity=False,traded=False,current_date=date(2020,1,1)); assert h.level.value=='NORMAL'
    for i in range(20):
        h=update_activity(h.posterior,eligible_opportunity=True,traded=True,current_date=date(2020,1,2+i))
    assert h.level.value == 'NORMAL' and h.posterior.opportunity_time_since_last_trade == 0
    for i in range(30):
        h=update_activity(h.posterior,eligible_opportunity=True,traded=False,current_date=date(2021,1,2+i))
    assert h.level.value in ('ANOMALOUS','CRITICAL')
    assert concentration([1,2,3])['top_1']==3/6 and concentration([-1,-2])['top_1']==0
    ledger=InMemoryShadowLedger(lambda p: MaturedOutcome(p.decision_date,p.expert_id,p.outcome_available_at,.1,.05,.01,.04))
    pending=PendingOutcome(date(2020,1,1),'R01',date(2020,1,20),'opaque'); ledger.append_pending(pending)
    assert ledger.mature(date(2020,1,19))==() and len(ledger.mature(date(2020,1,20)))==1 and ledger.mature(date(2020,1,20))==()
    weights=allocation_weights(('R01','R02'),{'R01':1,'R02':1},mode='equal'); validate_weights(weights); assert weights=={'R01':.5,'R02':.5}
    assert len({arm_manifest(arm)['policy_hash'] for arm in RouterArm})==len(tuple(RouterArm))
    runtime=FrozenExpertRuntimeImpl(r[3]); shadow=runtime.decision(decision_date=date(2020,1,1),available_predictions=[{'ticker':f'T{x:04d}','predicted_net_excess_return':.9,'horizon_sessions':28} for x in range(1000)])
    assert len(shadow.selected_positions)==5 and [x.ticker for x in shadow.selected_positions]==[f'T{x:04d}' for x in range(5)]
    held=runtime.decision(decision_date=date(2020,1,2),available_predictions=[{'ticker':f'T{x:04d}','predicted_net_excess_return':.9,'horizon_sessions':28} for x in range(1000)],open_tickers=[f'T{x:04d}' for x in range(5)])
    assert len(held.selected_positions)==0, 'concurrent max_names must cap open positions'
    assert market_state_as_of(date(2020,1,10),[(date(2020,1,i),100+i) for i in range(1,11)]).regime_id != ''
    assert relative_edge_lcb([.02,.03],[.01,.01],switch_cost=.001).lcb > 0
    sessions=(date(2020,1,2),date(2020,1,3),date(2020,1,6),date(2020,1,7),date(2020,1,8))
    _, timed_pending=make_shadow_decision(decision_date=sessions[0],expert_id='R01',eligible_opportunity=True,prediction=1.0,threshold=0.0,holding_days=3,session_dates=sessions)
    assert timed_pending is not None and timed_pending.outcome_available_at == sessions[4], 'fixed D3 must exit at next-open after three held sessions'
    incumbent=type('H',(),{'expert_id':'R01','alpha':AlphaHealth(0.010,.010,.010,0,50,.010,date(2020,1,1))})()
    challenger=type('H',(),{'expert_id':'R02','alpha':AlphaHealth(0.011,.011,.011,0,50,.011,date(2020,1,1))})()
    assert not assess_promotion(challenger,incumbent,p).promotion_allowed
    class Provider:
        def dates(self, start, end): return [start + timedelta(days=i) for i in range((end-start).days+1)]
        def shadow(self, current_date, expert):
            return make_shadow_decision(decision_date=current_date, expert_id=expert.expert_id, eligible_opportunity=expert.expert_type.value=='STOCK', prediction=1.0, threshold=0.0, holding_days=1)
        def outcome(self, pending):
            return MaturedOutcome(pending.decision_date,pending.expert_id,pending.outcome_available_at,.02,0,.001,.019)
    quick=replace(p,min_matured_observations=0,min_eligible_opportunities=0)
    registry=build_top10_qbd_registry()
    continuous=run_qbd_router_replay(policy=quick,registry=registry,start_date=date(2020,1,1),end_date=date(2020,1,5),data_provider=Provider())
    with tempfile.TemporaryDirectory() as d:
        store=JsonRouterStateStore(f'{d}/state.json',policy_hash=quick.policy_hash,registry_hash=__import__('stock_predictor.backtests.opportunity_portfolio_research.top10_qbd_expert_registry',fromlist=['registry_hash']).registry_hash(registry))
        run_qbd_router_replay(policy=quick,registry=registry,start_date=date(2020,1,1),end_date=date(2020,1,3),state_store=store,data_provider=Provider())
        resumed=run_qbd_router_replay(policy=quick,registry=registry,start_date=date(2020,1,1),end_date=date(2020,1,5),state_store=store,data_provider=Provider())
        assert resumed.final_state.health['R01'].alpha.effective_n == continuous.final_state.health['R01'].alpha.effective_n
    with tempfile.TemporaryDirectory() as d:
        m=build_manifest(policy=p,registry=r,code_commit='test',data_fingerprint='data',evaluation_contract_hash='contract')
        path=f'{d}/manifest.json'; write_manifest(path,m); validate_manifest(path,policy=p,registry=r,code_commit='test',evaluation_contract_hash='contract')
    print('QBD_ROUTER_V1_SELF_TEST_PASS')

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--self-test',action='store_true'); args=ap.parse_args(); run_self_test() if args.self_test else run_self_test()
