from __future__ import annotations
import math
from statistics import mean
def net_excess_cagr(values, dates):
    if not values or values[0]<=0 or values[-1]<=0 or len(dates)<2:return 0.0
    return (values[-1]/values[0])**(365.25/((dates[-1]-dates[0]).days))-1
def information_ratio(excess):
    if not excess:return 0.0
    m=mean(excess); sd=(sum((x-m)**2 for x in excess)/max(1,len(excess)-1))**.5
    return m/sd*math.sqrt(252) if sd else 0.0
def sortino(excess):
    if not excess:return 0.0
    downside=(sum(min(0,x)**2 for x in excess)/len(excess))**.5
    return mean(excess)/downside*math.sqrt(252) if downside else 0.0
def relative_max_drawdown(values):
    peak=values[0] if values else 1.0; worst=0.0
    for x in values: peak=max(peak,x); worst=min(worst,x/peak-1)
    return worst
def concentration(values, top=(1,5,10)):
    positive=sorted((max(0,x) for x in values),reverse=True); total=sum(positive)
    return {f'top_{n}':sum(positive[:n])/total if total else 0.0 for n in top}

def router_adaptation_metrics(decisions):
    switches=0; fallback=0; durations=[]; previous=None; current=0
    for decision in decisions:
        if decision.fallback_used: fallback += 1
        if decision.champion_id != previous:
            if previous is not None: durations.append(current)
            switches += int(previous is not None); previous=decision.champion_id; current=1
        else: current += 1
    if previous is not None: durations.append(current)
    return {'switch_count':switches,'benchmark_fallback_days':fallback,'champion_durations':durations,'mean_champion_duration':mean(durations) if durations else 0.0}

def router_report(decisions, excess, values, dates):
    return {'performance':{'net_excess_cagr':net_excess_cagr(values,dates),'information_ratio':information_ratio(excess),'sortino':sortino(excess),'relative_max_drawdown':relative_max_drawdown(values)},'adaptation':router_adaptation_metrics(decisions),'concentration':concentration(excess)}
