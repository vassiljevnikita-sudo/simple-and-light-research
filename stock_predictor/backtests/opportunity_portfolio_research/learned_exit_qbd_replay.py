from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections import defaultdict
from dataclasses import asdict
from copy import deepcopy
import math
import pandas as pd
import numpy as np

from .portfolio_allocation_weights import requested_notionals
from .portfolio_policy_contracts import Policy
from .cost_contracts import CostModel
from .tax_contracts import TaxConfig
from .next_open_portfolio_replay import prepare_prices, prepare_signals, _metrics
from .german_retail_tax_engine import TaxLedger
from .learned_exit_qbd_profit import EXIT_THRESHOLD, ProfitTaxConfig
from .learned_exit_qbd_provider import LearnedExitProvider

_ACCOUNTING_TOL = 1e-7
_PROVIDER: LearnedExitProvider | None = None
_TAX = ProfitTaxConfig()


def configure_replay(provider: LearnedExitProvider | None, tax: ProfitTaxConfig) -> None:
    global _PROVIDER, _TAX
    _PROVIDER, _TAX = provider, tax


def _tax_config() -> TaxConfig:
    return TaxConfig(enabled=_TAX.enabled, capital_gains_rate=_TAX.capital_gains_rate,
        solidarity_surcharge=_TAX.solidarity_surcharge, allowance_eur=_TAX.allowance_eur,
        church_tax_rate=_TAX.church_tax_rate)


def _benchmark_after_tax(initial: float, terminal: float) -> tuple[float,float]:
    if not _TAX.enabled: return terminal, 0.0
    gain=max(0.0,terminal-initial)*(1.0-_TAX.benchmark_partial_exemption_rate)
    taxable=max(0.0,gain-_TAX.allowance_eur)
    rate=_TAX.capital_gains_rate*(1.0+_TAX.solidarity_surcharge+_TAX.church_tax_rate)
    tax=taxable*rate
    return terminal-tax,tax


def replay(signals: pd.DataFrame, prices: pd.DataFrame, policy: Policy, cost: CostModel, tax_config: TaxConfig,
           start: pd.Timestamp|None=None, end: pd.Timestamp|None=None, initial: float=10000.0,
           regime: pd.DataFrame|None=None, resolved_threshold: float|None=None,
           prepared_signals: dict|None=None, delisting_rules: dict|None=None,
           resolved_threshold_by_date: dict|pd.Series|None=None,
           prepared_prices: tuple|None=None,
           legacy_historical_mode: bool=False,
           generation_schedule: pd.DataFrame|None=None,
           resume_state: dict|None=None) -> dict:
    if policy.exit_family=="LEARNED_EXIT" and _PROVIDER is None: raise RuntimeError("LEARNED_EXIT_PROVIDER_NOT_CONFIGURED")
    delisting_rules = delisting_rules or {}
    pmap,umap,all_dates=prepared_prices if prepared_prices is not None else prepare_prices(prices)
    original_lo=0 if start is None else bisect_left(all_dates,pd.Timestamp(start)); lo=original_lo
    if resume_state is not None: lo=bisect_right(all_dates,pd.Timestamp(resume_state["as_of"]))
    hi=len(all_dates) if end is None else bisect_right(all_dates,pd.Timestamp(end))
    dates_tuple=all_dates[lo:hi]
    if not dates_tuple and resume_state is None: return {"metrics":{"trade_count":0,"initial_value":initial,"terminal_value":initial},"curve":pd.DataFrame(),"trades":[],"cost_audit":{},"tax_audit":{}}
    dates=pd.DatetimeIndex(dates_tuple); idx={d:i for i,d in enumerate(all_dates)}
    prepared=prepared_signals if prepared_signals is not None else prepare_signals(signals); by_date=prepared["by_date"]; tops=prepared["top_score_by_date"]
    topvals=[tops[d] for d in dates if d in tops]
    threshold=(float(resume_state["threshold"]) if resume_state is not None and "threshold" in resume_state else
               (float(resolved_threshold) if resolved_threshold is not None else (float(np.quantile(topvals,policy.score_quantile)) if topvals else float("inf"))))
    if resolved_threshold_by_date is not None:
        raw_thresholds = resolved_threshold_by_date.to_dict() if isinstance(resolved_threshold_by_date, pd.Series) else dict(resolved_threshold_by_date)
        daily_thresholds = {pd.Timestamp(k).normalize(): float(v) for k, v in raw_thresholds.items() if pd.notna(v)}
        missing_threshold_dates = sorted(d for d in dates if d in by_date and d not in daily_thresholds)
        if missing_threshold_dates:
            raise RuntimeError(
                "DAILY_RESOLVED_THRESHOLD_MISSING:"
                f"count={len(missing_threshold_dates)}:first={missing_threshold_dates[0].date()}"
            )
    else:
        daily_thresholds = None
    lineage_schedule=[]
    if generation_schedule is not None and not generation_schedule.empty:
        required={"activation_date","family_id","generation_id","entry_policy_id","exit_policy_id"}
        missing=required-set(generation_schedule)
        if missing: raise ValueError(f"GENERATION_SCHEDULE_COLUMNS_MISSING:{sorted(missing)}")
        for row in generation_schedule.sort_values("activation_date").itertuples(index=False):
            lineage_schedule.append((pd.Timestamp(row.activation_date),{"family_id":str(row.family_id),"generation_id":str(row.generation_id),"model_artifact_id":str(getattr(row,"model_artifact_id","")),"entry_policy_id":str(row.entry_policy_id),"exit_policy_id":str(row.exit_policy_id),"exit_generation_id":str(getattr(row,"exit_generation_id",row.exit_policy_id)),"resolved_top_fraction":float(getattr(row,"resolved_top_fraction",policy.top_fraction))}))
    def lineage_at(d):
        active=None
        for activation,record in lineage_schedule:
            if activation>d: break
            active=record
        return active
    def threshold_at(d):
        return daily_thresholds[d] if daily_thresholds is not None else threshold
    def top_fraction_at(d):
        lineage=lineage_at(d)
        return float(lineage["resolved_top_fraction"]) if lineage else policy.top_fraction
    if resume_state is None:
        benchmark_start_date=all_dates[original_lo]; benchmark_initial_close=umap[benchmark_start_date][1]
        pending=defaultdict(list); positions={}; cash=0.0; urth_units=initial/benchmark_initial_close; urth_tax_basis=initial
        tax=TaxLedger(_tax_config()); trades=[]; costs={"transaction_cost_eur":0.0}; curve=[]; sleeve_breach_count=0; entry_scores={}
        required_exit_decisions=available_exit_decisions=learned_exit_count=0
        learned_exit_fallback_positions=learned_exit_missing_prediction_count=learned_exit_horizon_boundary_count=0
        delisting_exit_decisions=0
    else:
        if float(resume_state["initial_value"])!=float(initial): raise ValueError("REPLAY_RESUME_INITIAL_VALUE_MISMATCH")
        benchmark_start_date=pd.Timestamp(resume_state["benchmark_start_date"]); benchmark_initial_close=float(resume_state["benchmark_initial_close"])
        pending=defaultdict(list,{pd.Timestamp(key):list(value) for key,value in resume_state.get("pending_orders",{}).items()})
        positions={}
        for ticker,raw in resume_state.get("positions",{}).items():
            position=dict(raw); position["entry_date"]=pd.Timestamp(position["entry_date"])
            position["learned_exit_plan"]={pd.Timestamp(key):float(value) for key,value in position.get("learned_exit_plan",{}).items()}
            positions[str(ticker)]=position
        cash=float(resume_state["cash"]); urth_units=float(resume_state["urth_units"]); urth_tax_basis=float(resume_state["urth_tax_basis"])
        tax=TaxLedger.from_snapshot(_tax_config(),resume_state.get("tax_ledger")); trades=list(resume_state.get("trades",[]))
        for trade in trades: trade["entry_date"]=pd.Timestamp(trade["entry_date"]); trade["exit_date"]=pd.Timestamp(trade["exit_date"])
        costs={"transaction_cost_eur":float(resume_state.get("transaction_cost_eur",0.0))}; curve=list(resume_state.get("curve",[]))
        for point in curve: point["date"]=pd.Timestamp(point["date"])
        sleeve_breach_count=int(resume_state.get("sleeve_breach_count",0)); entry_scores={str(k):float(v) for k,v in resume_state.get("entry_scores",{}).items()}
        counters=resume_state.get("learned_exit_counters",{})
        required_exit_decisions=int(counters.get("required_exit_decisions",0)); available_exit_decisions=int(counters.get("available_exit_decisions",0))
        learned_exit_count=int(counters.get("learned_exit_count",0)); learned_exit_fallback_positions=int(counters.get("learned_exit_fallback_positions",0))
        learned_exit_missing_prediction_count=int(counters.get("learned_exit_missing_prediction_count",0)); learned_exit_horizon_boundary_count=int(counters.get("learned_exit_horizon_boundary_count",0))
        delisting_exit_decisions=int(counters.get("delisting_exit_decisions",0))
    def security_rule(ticker, entry_date=None):
        rule = delisting_rules.get(str(ticker))
        if rule is None:
            return None
        entry_date = pd.Timestamp(entry_date).normalize() if entry_date is not None else None
        if entry_date is not None and rule.get("valid_entry_from") is not None and entry_date < pd.Timestamp(rule["valid_entry_from"]):
            return None
        if entry_date is not None and rule.get("valid_entry_through") is not None and entry_date > pd.Timestamp(rule["valid_entry_through"]):
            return None
        return rule
    def components(d,px="close"):
        u=umap[d][1 if px=="close" else 0]; uv=urth_units*u; sv=sum(pos["qty"]*pmap[(d,t)][1 if px=="close" else 0] for t,pos in positions.items() if (d,t) in pmap)
        return cash+uv+sv,cash,uv,sv
    def buy_urth(notional,d):
        nonlocal urth_units,cash,urth_tax_basis
        take=min(max(0.0,notional),max(0.0,cash)); u=umap[d][0]; urth_units+=take/u; cash-=take
        urth_tax_basis += take
    def sell_urth(notional,d):
        nonlocal urth_units,cash,urth_tax_basis
        u=umap[d][0]; units_before=urth_units; take=min(max(0.0,notional),max(0.0,units_before*u)); sold_units=take/u
        basis=urth_tax_basis*(sold_units/units_before) if units_before else 0.0
        urth_units-=sold_units; urth_tax_basis=max(0.0,urth_tax_basis-basis); cash+=take
        if not legacy_historical_mode:
            cash-=tax.realize_stock_trade(d.date(),take,basis,0.0,taxable_fraction=0.70)
    def sell_stock(t,d,reason="FIXED", terminal_value=None):
        nonlocal cash
        pos=positions.pop(t,None)
        if not pos or (terminal_value is None and (d,t) not in pmap): return
        px=float(terminal_value) if terminal_value is not None else pmap[(d,t)][0]
        gross=pos["qty"]*px; fee=gross*cost.per_side_bps/10000; cash+=gross-fee; costs["transaction_cost_eur"]+=fee
        tax_paid=tax.realize_stock_trade(d.date(),gross,pos["tax_basis"],fee); cash-=tax_paid
        excess=px/pos["entry_price"]/(umap[d][0]/pos["entry_urth"])-1
        trade = {"ticker":t,"entry_date":pos["entry_date"],"exit_date":d,"holding_days":idx[d]-pos["entry_index"],"stock_return":px/pos["entry_price"]-1,"excess_return":excess,"buy_notional":pos["buy_notional"],"sell_notional":gross,"cost_eur":pos["buy_fee"]+fee,"tax_eur":tax_paid,"exit_reason":reason,"family_id":pos.get("family_id"),"generation_id":pos.get("generation_id"),"entry_family_id":pos.get("family_id"),"entry_generation_id":pos.get("generation_id"),"entry_model_artifact_id":pos.get("model_artifact_id"),"entry_policy_id":pos.get("entry_policy_id"),"exit_policy_id":pos.get("exit_policy_id"),"exit_generation_id":pos.get("exit_generation_id")}
        if reason.startswith("DELISTING_"):
            rule = security_rule(t, pos.get("entry_date")) or {}
            trade.update({
                "security_id": rule.get("security_id"),
                "ticker_at_entry": rule.get("ticker_at_entry", t),
                "last_valid_market_date": rule.get("series_end"),
                "series_end_reason": "SECURITY_SERIES_BREAK",
                "terminal_value_source": rule.get("terminal_value_source"),
                "terminal_value": float(rule.get("terminal_value", 0.0)),
            })
        trades.append(trade)
        buy_urth(max(0.0,cash),d)
    for d in dates:
        actions=pending.pop(d,[])
        for a in actions:
            if a["kind"]=="sell": sell_stock(a["ticker"],d,a.get("reason","FIXED"))
        buys=[a for a in actions if a["kind"]=="buy" and a["ticker"] not in positions and (d,a["ticker"]) in pmap and not (security_rule(a["ticker"], d) is not None and d > pd.Timestamp(security_rule(a["ticker"], d)["series_end"]))]
        if buys:
            equity_open,_,_,stock_open=components(d,"open"); cap=max(0.0,equity_open*policy.sleeve-stock_open)
            if policy.allocation=="EQUAL_ACTIVE": requested=[cap/len(buys)]*len(buys)
            elif policy.allocation.startswith(("RANK_POWER:","SCORE_EXCESS_POWER:","SCORE_SOFTMAX:")): requested=requested_notionals(policy.allocation,[float(a.get("score",0)) for a in buys],cap,threshold=threshold_at(d))
            else: requested=[equity_open*policy.sleeve/max(policy.max_names,1)]*len(buys)
            for a,want in zip(buys,requested):
                equity_open,_,_,stock_open=components(d,"open"); target=min(max(0.0,want),max(0.0,equity_open*policy.sleeve-stock_open))
                if target<=0: continue
                sell_urth(target,d); target=min(target,max(0.0,cash)); px=pmap[(d,a["ticker"])][0]; fee=target*cost.per_side_bps/10000; notion=max(0.0,target-fee)
                if notion<=0: buy_urth(max(0.0,cash),d); continue
                cash-=target; costs["transaction_cost_eur"]+=fee
                if abs(cash)<=_ACCOUNTING_TOL: cash=0.0
                exit_plan={}; fallback=False
                if policy.exit_family=="LEARNED_EXIT":
                    for off in range(max(0,policy.holding_days-1)):
                        if idx[d]+off>=len(all_dates):
                            if legacy_historical_mode:
                                # Reproduction-only semantics used by the
                                # published outer rows before right-censoring.
                                fallback=True
                                continue
                            # Right-censoring at the final available market date
                            # is not a missing model prediction.  Preserve every
                            # observed learned-exit decision and mark any open
                            # terminal position to market instead of converting
                            # the whole position to a fixed-exit fallback.
                            learned_exit_horizon_boundary_count+=1; break
                        decision_day=all_dates[idx[d]+off]; remaining=policy.holding_days-1-off
                        if remaining<=0: continue
                        rule = None if legacy_historical_mode else security_rule(a["ticker"], d)
                        if rule is not None and decision_day > pd.Timestamp(rule["series_end"]):
                            required_exit_decisions += 1
                            delisting_exit_decisions += 1
                            break
                        required_exit_decisions+=1
                        if _PROVIDER and getattr(_PROVIDER,"generation_aware",False):
                            pv=_PROVIDER.prediction(a["ticker"],decision_day,remaining,a.get("exit_policy_id"))
                        else:
                            pv=_PROVIDER.prediction(a["ticker"],decision_day,remaining) if _PROVIDER else None
                        if pv is None or not math.isfinite(float(pv)):
                            fallback=True; learned_exit_missing_prediction_count+=1
                        else: available_exit_decisions+=1; exit_plan[decision_day]=float(pv)
                if fallback: learned_exit_fallback_positions+=1
                positions[a["ticker"]]={"qty":notion/px,"entry_price":px,"entry_urth":umap[d][0],"entry_date":d,"entry_index":idx[d],"peak_excess":0.0,"buy_notional":notion,"buy_fee":fee,"tax_basis":notion+fee,"learned_exit_plan":exit_plan,"learned_exit_fallback":fallback,"family_id":a.get("family_id"),"generation_id":a.get("generation_id"),"model_artifact_id":a.get("model_artifact_id"),"entry_policy_id":a.get("entry_policy_id"),"exit_policy_id":a.get("exit_policy_id"),"exit_generation_id":a.get("exit_generation_id")}
                entry_scores[a["ticker"]]=a.get("score",0.0)
        current=by_date.get(d); valid=[]
        if current is not None:
            current_threshold=threshold_at(d)
            tickers,scores,neg_scores,group_size=current; limit=max(1,math.ceil(group_size*top_fraction_at(d))); passing=int(np.searchsorted(neg_scores,-current_threshold,side="right")) if len(neg_scores) else 0; take=min(limit,passing); valid=[{"ticker":tickers[i],"score":float(scores[i])} for i in range(take)]
        valid_tickers={str(x["ticker"]) for x in valid}; next_d=all_dates[idx[d]+1] if idx[d]+1<len(all_dates) else None
        for t,pos in list(positions.items()):
            rule = None if legacy_historical_mode else security_rule(t, pos.get("entry_date"))
            if rule is not None and d > pd.Timestamp(rule["series_end"]):
                sell_stock(t, d, str(rule["exit_reason"]), terminal_value=float(rule["terminal_value"]))
                continue
            if (d,t) not in pmap: continue
            stock_close=pmap[(d,t)][1]; urth_close=umap[d][1]; excess=stock_close/pos["entry_price"]/(urth_close/pos["entry_urth"])-1; pos["peak_excess"]=max(pos["peak_excess"],excess)
            held=idx[d]-pos["entry_index"]+1; exit_now=held>=policy.holding_days; reason="FIXED"
            if policy.exit_family=="LEARNED_EXIT" and held<policy.holding_days:
                pv=None if pos.get("learned_exit_fallback") else pos.get("learned_exit_plan",{}).get(d)
                if pv is not None and float(pv)<=EXIT_THRESHOLD: exit_now=True; reason="LEARNED_EXIT"; learned_exit_count+=1
            elif policy.exit_family=="SIGNAL_DECAY": exit_now|=t not in valid_tickers
            elif policy.exit_family=="RELATIVE_STOP": exit_now|=excess<=policy.exit_value
            elif policy.exit_family=="TRAILING_RELATIVE_STOP": exit_now|=excess<=pos["peak_excess"]-policy.exit_value
            elif policy.exit_family=="TAKE_PROFIT_RELATIVE": exit_now|=excess>=policy.exit_value
            if exit_now and next_d is not None: pending[next_d].append({"kind":"sell","ticker":t,"reason":reason})
        chosen=[x for x in valid if str(x["ticker"]) not in positions][:policy.max_names]
        scheduled={a["ticker"] for a in pending.get(next_d,[]) if a.get("kind")=="sell"} if next_d is not None else set(); available=max(0,policy.max_names-max(0,len(positions)-len(scheduled))); chosen=chosen[:available]
        if next_d is not None:
            for row in chosen:
                lineage=lineage_at(d) or {"family_id":None,"generation_id":None,"entry_policy_id":policy.policy_id,"exit_policy_id":policy.exit_family}
                pending[next_d].append({"kind":"buy","ticker":str(row["ticker"]),"score":float(row["score"]),**lineage})
        total,cashv,urthv,stockv=components(d,"close")
        if total<=0: raise AssertionError(f"NON_POSITIVE_NAV:{d}:{total}")
        se=stockv/total; ue=urthv/total; ce=cashv/total
        if se>policy.sleeve+5e-6: sleeve_breach_count+=1
        if len(positions)>policy.max_names: raise AssertionError("MAX_NAMES_EXCEEDED")
        curve.append({"date":d,"strategy_value":total,"urth_value":initial*umap[d][1]/benchmark_initial_close,"positions":len(positions),"stock_exposure":se,"urth_exposure":ue,"cash_exposure":ce,"accounting_error_eur":0.0,"regime":None})
    curve=pd.DataFrame(curve).reindex(columns=["date","strategy_value","urth_value","positions","stock_exposure",
        "urth_exposure","cash_exposure","accounting_error_eur","regime"])
    bench_pre=float(curve["urth_value"].iloc[-1]); bench_after,bench_tax=_benchmark_after_tax(initial,bench_pre)
    terminal_tax=0.0
    if _TAX.enabled and not legacy_historical_mode:
        terminal_ledger=deepcopy(tax); final_date=pd.Timestamp(curve["date"].iloc[-1]); final_total=float(curve["strategy_value"].iloc[-1])
        for ticker,pos in positions.items():
            quote=pmap.get((final_date,ticker))
            if quote is None: continue
            gross=pos["qty"]*quote[1]; fee=gross*cost.per_side_bps/10000
            terminal_ledger.realize_stock_trade(final_date.date(),gross,pos["tax_basis"],fee)
        urth_gross=urth_units*umap[final_date][1]
        terminal_ledger.realize_stock_trade(final_date.date(),urth_gross,urth_tax_basis,0.0,taxable_fraction=0.70)
        terminal_tax=max(0.0,terminal_ledger.tax_paid-tax.tax_paid)
        terminal_after_tax=final_total-terminal_tax
    else:
        terminal_after_tax=None
    metrics=_metrics(curve,initial,trades,costs,tax,initial,benchmark_terminal_after_tax=bench_after if _TAX.enabled else None,benchmark_tax=bench_tax,strategy_terminal_after_tax=terminal_after_tax)
    metrics.update({"tax_world":"DE_RETAIL_TAX_AWARE_APPROX" if _TAX.enabled else "PRE_TAX","benchmark_partial_exemption_rate":_TAX.benchmark_partial_exemption_rate,"benchmark_vorabpauschale_mode":_TAX.benchmark_vorabpauschale_mode,"terminal_liquidation_tax_eur":terminal_tax,"learned_exit_count":learned_exit_count,"required_exit_decisions":required_exit_decisions,"available_exit_decisions":available_exit_decisions,"delisting_exit_decisions":delisting_exit_decisions,"exit_decision_coverage":1.0 if required_exit_decisions==0 else (available_exit_decisions+delisting_exit_decisions)/required_exit_decisions,"learned_exit_fallback_positions":learned_exit_fallback_positions,"learned_exit_missing_prediction_count":learned_exit_missing_prediction_count,"learned_exit_horizon_boundary_count":learned_exit_horizon_boundary_count,"sleeve_mark_to_market_breach_days":sleeve_breach_count})
    open_positions=[{"ticker":str(t),"entry_date":str(p["entry_date"]),"family_id":p.get("family_id"),"generation_id":p.get("generation_id"),"entry_family_id":p.get("family_id"),"entry_generation_id":p.get("generation_id"),"entry_model_artifact_id":p.get("model_artifact_id"),"entry_policy_id":p.get("entry_policy_id"),"exit_policy_id":p.get("exit_policy_id"),"exit_generation_id":p.get("exit_generation_id")} for t,p in sorted(positions.items())]
    serialized_positions={}
    for ticker,position in positions.items():
        raw=dict(position); raw["entry_date"]=str(raw["entry_date"])
        raw["learned_exit_plan"]={str(key):value for key,value in raw.get("learned_exit_plan",{}).items()}
        serialized_positions[str(ticker)]=raw
    as_of=pd.Timestamp(curve["date"].iloc[-1])
    replay_state={"schema_version":"LEARNED_EXIT_REPLAY_STATE_V1","as_of":str(as_of),"session_cursor":int(idx[as_of]),
                  "initial_value":float(initial),"benchmark_start_date":str(benchmark_start_date),"benchmark_initial_close":float(benchmark_initial_close),
                  "cash":float(cash),"urth_units":float(urth_units),"urth_tax_basis":float(urth_tax_basis),"positions":serialized_positions,
                  "pending_orders":{str(key):value for key,value in pending.items()},"tax_ledger":tax.snapshot(),
                  "transaction_cost_eur":float(costs["transaction_cost_eur"]),"entry_scores":entry_scores,
                  "sleeve_breach_count":int(sleeve_breach_count),"curve":curve.to_dict(orient="records"),"trades":trades,"threshold":float(threshold),
                  "learned_exit_counters":{"required_exit_decisions":required_exit_decisions,"available_exit_decisions":available_exit_decisions,
                    "learned_exit_count":learned_exit_count,"learned_exit_fallback_positions":learned_exit_fallback_positions,
                    "learned_exit_missing_prediction_count":learned_exit_missing_prediction_count,"learned_exit_horizon_boundary_count":learned_exit_horizon_boundary_count,
                    "delisting_exit_decisions":delisting_exit_decisions}}
    return {"metrics":metrics,"curve":curve,"trades":trades,"open_positions":open_positions,"cost_audit":costs|{"roundtrip_bps":cost.roundtrip_bps,"per_side_bps":cost.per_side_bps},"tax_audit":tax.snapshot()|{"terminal_liquidation_tax_eur":terminal_tax},"threshold":threshold,"replay_state":replay_state,"daily_thresholds_used":daily_thresholds is not None,"generation_schedule_applied":bool(lineage_schedule),"legacy_historical_mode":legacy_historical_mode,"policy":asdict(policy)}
