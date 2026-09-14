"""Normalised independent shadow sleeve for one frozen expert."""
from __future__ import annotations
from dataclasses import dataclass, replace
from datetime import date

@dataclass(frozen=True)
class ShadowPosition:
    expert_id: str; ticker: str; entry_decision_date: date; entry_date: date; entry_price: float
    shares_or_weight: float; max_exit_date: date; current_age_sessions: int; exit_policy_id: str

@dataclass(frozen=True)
class ExpertShadowPortfolioState:
    expert_id: str; cash: float; positions: tuple[ShadowPosition, ...]; nav: float; last_date: date | None

def initial_portfolio(expert_id: str) -> ExpertShadowPortfolioState:
    return ExpertShadowPortfolioState(expert_id,1.0,(),1.0,None)

def enter_positions(state, positions: tuple[ShadowPosition, ...], *, current_date: date, cost_rate: float) -> ExpertShadowPortfolioState:
    existing={p.ticker: p for p in state.positions}
    new=[p for p in positions if p.ticker not in existing]
    notional=sum(max(0.0,float(p.shares_or_weight)) for p in new)
    fees=cost_rate*len(new)
    if notional+fees > state.cash + 1e-12:
        raise ValueError('shadow sleeve cannot fund entry notional and costs')
    existing.update({p.ticker:p for p in new})
    cash=state.cash-notional-fees
    return replace(state,cash=cash,positions=tuple(existing.values()),nav=cash+notional,last_date=current_date)

def mark_to_market(state, prices: dict[str,float], *, current_date: date, exits: tuple[str,...]=(), cost_rate: float=0.0) -> ExpertShadowPortfolioState:
    exit_set=set(exits)
    held=[p for p in state.positions if p.ticker not in exit_set]
    proceeds=sum((prices.get(p.ticker,p.entry_price)/p.entry_price)*p.shares_or_weight for p in state.positions if p.ticker in exit_set)
    exit_cost=cost_rate*len(exit_set)
    mark=sum((prices.get(p.ticker,p.entry_price)/p.entry_price)*p.shares_or_weight for p in held)
    nav=max(0.0,state.cash+proceeds+mark-exit_cost)
    cash=state.cash+proceeds-exit_cost
    return replace(state,cash=cash,nav=nav,positions=tuple(held),last_date=current_date)
