from __future__ import annotations
from dataclasses import dataclass
@dataclass(frozen=True)
class CostStressResult:
    roundtrip_bps: float; gross_excess: float; costs: float; net_excess: float
def cost_stress(gross_excess,trade_count,switch_count,costs=(0,10,20,30,50)):
    return tuple(CostStressResult(float(b),gross_excess,(trade_count+switch_count)*b/10000,gross_excess-(trade_count+switch_count)*b/10000) for b in costs)
def attempted_variants(inventory): return {'attempted_variant_count':len(tuple(inventory)),'inventory':tuple(inventory)}

def deflated_sharpe(sharpe, attempted_count, observations):
    penalty=(2.0*max(0.0,__import__('math').log(max(1,attempted_count))))**.5/max(1.0,observations)**.5
    return float(sharpe-penalty)

def block_bootstrap_mean(values, block_size=5, repetitions=1000, seed=17):
    import random
    rng=random.Random(seed); values=tuple(values)
    if not values:return {'mean':0.0,'q05':0.0,'q95':0.0,'seed':seed}
    samples=[]
    for _ in range(repetitions):
        draw=[]
        while len(draw)<len(values):
            start=rng.randrange(len(values)); draw.extend(values[start:start+block_size])
        samples.append(sum(draw[:len(values)])/len(values))
    samples.sort(); return {'mean':sum(samples)/len(samples),'q05':samples[int(.05*len(samples))],'q95':samples[int(.95*len(samples))-1],'seed':seed}

def pbo_like(sharpe_values):
    values=tuple(sharpe_values)
    if not values:return 0.0
    median=sorted(values)[len(values)//2]
    return sum(x < median for x in values)/len(values)
