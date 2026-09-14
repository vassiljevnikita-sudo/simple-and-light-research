from __future__ import annotations
from dataclasses import dataclass
import math
@dataclass(frozen=True)
class PromotionAssessment:
    challenger_id: str; incumbent_id: str | None; point_edge: float; switch_cost: float; lcb_net_edge_after_switch_cost: float
    minimum_evidence_met: bool; minimum_duration_met: bool; promotion_allowed: bool; reason_code: str

@dataclass(frozen=True)
class RelativeEdgeEstimate:
    mean_edge: float; standard_error: float; lcb: float; observations: int

def relative_edge_lcb(challenger_returns, incumbent_returns, *, switch_cost, block_size=5):
    pairs=[float(a)-float(b)-float(switch_cost) for a,b in zip(challenger_returns,incumbent_returns)]
    if not pairs: return RelativeEdgeEstimate(0.0,float('inf'),float('-inf'),0)
    mean_edge=sum(pairs)/len(pairs); variance=sum((x-mean_edge)**2 for x in pairs)/max(1,len(pairs)-1)
    return RelativeEdgeEstimate(mean_edge,(variance/len(pairs))**.5,mean_edge-1.96*(variance/len(pairs))**.5,len(pairs))
def assess_promotion(challenger, incumbent, policy, duration_met=True, *, challenger_returns=None, incumbent_returns=None):
    inc = incumbent.alpha.lcb_net_excess if incumbent and incumbent.alpha.lcb_net_excess is not None else 0.0
    chal = challenger.alpha.lcb_net_excess if challenger.alpha.lcb_net_excess is not None else 0.0
    if challenger_returns is not None and incumbent_returns is not None:
        paired = relative_edge_lcb(challenger_returns, incumbent_returns, switch_cost=policy.switch_cost_bps / 10000)
        edge, net = paired.mean_edge + policy.switch_cost_bps / 10000, paired.lcb
    else:
        edge = chal - inc; net = edge - policy.switch_cost_bps / 10000
    enough = challenger.alpha.effective_n >= policy.min_matured_observations
    return PromotionAssessment(challenger.expert_id, incumbent.expert_id if incumbent else None, edge, policy.switch_cost_bps/10000, net, enough, duration_met, enough and duration_met and net > policy.promotion_lcb_margin, "PROMOTION_HURDLE" if enough and duration_met else "INSUFFICIENT_EVIDENCE")
