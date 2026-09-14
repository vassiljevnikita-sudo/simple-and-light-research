from __future__ import annotations
def oracle_best_expert(returns): return max(returns,key=lambda k:sum(returns[k])) if returns else None
def false_discovery_rate(runs): return sum(bool(x) for x in runs)/len(runs) if runs else 0.0
