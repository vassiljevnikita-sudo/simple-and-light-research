"""Router-level NAV and turnover accounting, independent of selection logic."""
from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class RouterPortfolioState:
    nav: float; weights: dict[str,float]; gross_return: float; transaction_cost: float
    net_return: float; benchmark_return: float; net_excess: float; date: object

def apply_router_allocation(previous, new_weights, expert_returns, benchmark_return, *, cost_rate, current_date):
    universe=set(previous.weights)|set(new_weights); turnover=.5*sum(abs(new_weights.get(x,0)-previous.weights.get(x,0)) for x in universe)
    gross=sum(new_weights.get(x,0)*expert_returns.get(x,0) for x in universe); cost=turnover*cost_rate; net=gross-cost
    return RouterPortfolioState(previous.nav*(1+net),dict(new_weights),gross,cost,net,benchmark_return,net-benchmark_return,current_date)

def initial_router_portfolio(date, fallback='MSCI_WORLD'):
    return RouterPortfolioState(1.0,{fallback:1.0},0.0,0.0,0.0,0.0,0.0,date)
