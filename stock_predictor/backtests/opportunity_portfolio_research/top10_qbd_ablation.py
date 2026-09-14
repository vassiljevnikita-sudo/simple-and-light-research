"""A-K arm composition over the exact same prequential engine."""
from __future__ import annotations
from dataclasses import asdict
from .top10_qbd_policy import RouterArm, capabilities_for
from .top10_qbd_router_contracts import default_policy
from .top10_qbd_prequential_replay import run_qbd_router_replay

def arm_manifest(arm: RouterArm, policy=None):
    p=policy or default_policy(arm)
    return {'arm':arm.value,'policy_hash':p.policy_hash,'capabilities':asdict(capabilities_for(arm))}

def run_all_arms(*, registry, start_date, end_date, data_provider):
    return {arm.value: run_qbd_router_replay(policy=default_policy(arm), registry=registry, start_date=start_date, end_date=end_date, data_provider=data_provider) for arm in RouterArm}
