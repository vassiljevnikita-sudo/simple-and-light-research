"""Deterministic, as-of-only market regime buckets."""
from __future__ import annotations
from dataclasses import dataclass
from statistics import pstdev

@dataclass(frozen=True)
class MarketState:
    regime_id: str; benchmark_mom20: float; benchmark_mom60: float; benchmark_mom120: float
    benchmark_vol20: float; trend100: float; breadth20: float; breadth120: float; stress: float

def market_state_as_of(decision_date, benchmark_closes, breadth20=0.5, breadth120=0.5):
    closes=tuple(float(x) for d,x in benchmark_closes if d <= decision_date)
    if not closes: return MarketState('GLOBAL',0,0,0,0,0,breadth20,breadth120,0)
    last=closes[-1]
    mom=lambda n: last/closes[-n-1]-1 if len(closes)>n else 0.0
    returns=tuple(closes[i]/closes[i-1]-1 for i in range(max(1,len(closes)-20),len(closes)))
    vol=pstdev(returns) if len(returns)>1 else 0.0; trend=last/(sum(closes[-min(100,len(closes)):])/min(100,len(closes)))-1
    stress=max(0.0,min(1.0,vol*10 + max(0.0,-mom(20))*2)); m20=mom(20); regime='STRESS' if stress>.7 else ('UP_HIGH_VOL' if m20>0 and vol>.02 else 'UP_LOW_VOL' if m20>0 else 'DOWN_HIGH_VOL' if vol>.02 else 'DOWN_LOW_VOL')
    return MarketState(regime,m20,mom(60),mom(120),vol,trend,breadth20,breadth120,stress)
