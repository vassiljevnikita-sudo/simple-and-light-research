"""Policy enums and immutable capability resolution."""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum


class RouterArm(str, Enum):
    A_STATIC_BEST_DEV = "A_STATIC_BEST_DEV"
    B_HARD_CHAMPION_3M = "B_HARD_CHAMPION_3M"
    C_HARD_CHAMPION_6M = "C_HARD_CHAMPION_6M"
    D_EQUAL_WEIGHT = "D_EQUAL_WEIGHT"
    E_DISCOUNTED_SOFTMAX = "E_DISCOUNTED_SOFTMAX"
    F_SUPERIOR_SET_EQUAL = "F_SUPERIOR_SET_EQUAL"
    G_SUPERIOR_SET_DMA = "G_SUPERIOR_SET_DMA"
    H_REGIME_SHRINKAGE = "H_REGIME_SHRINKAGE"
    I_CHANGEPOINT = "I_CHANGEPOINT"
    J_OPPORTUNITY_TIME_ACTIVITY = "J_OPPORTUNITY_TIME_ACTIVITY"
    K_UNCERTAINTY_LCB = "K_UNCERTAINTY_LCB"


@dataclass(frozen=True)
class RouterCapabilities:
    hard_window_selector: bool = False
    equal_weight: bool = False
    discounted_softmax: bool = False
    superior_set: bool = False
    regime_shrinkage: bool = False
    changepoint: bool = False
    opportunity_activity: bool = False
    uncertainty_lcb: bool = False


def capabilities_for(arm: RouterArm) -> RouterCapabilities:
    if arm == RouterArm.A_STATIC_BEST_DEV:
        return RouterCapabilities(hard_window_selector=True)
    if arm in (RouterArm.B_HARD_CHAMPION_3M, RouterArm.C_HARD_CHAMPION_6M):
        return RouterCapabilities(hard_window_selector=True)
    if arm == RouterArm.D_EQUAL_WEIGHT:
        return RouterCapabilities(equal_weight=True)
    if arm == RouterArm.E_DISCOUNTED_SOFTMAX:
        return RouterCapabilities(discounted_softmax=True)
    if arm == RouterArm.F_SUPERIOR_SET_EQUAL:
        return RouterCapabilities(equal_weight=True, superior_set=True)
    if arm == RouterArm.G_SUPERIOR_SET_DMA:
        return RouterCapabilities(discounted_softmax=True, superior_set=True)
    return RouterCapabilities(
        discounted_softmax=True, superior_set=True,
        regime_shrinkage=arm in (RouterArm.H_REGIME_SHRINKAGE, RouterArm.I_CHANGEPOINT, RouterArm.J_OPPORTUNITY_TIME_ACTIVITY, RouterArm.K_UNCERTAINTY_LCB),
        changepoint=arm in (RouterArm.I_CHANGEPOINT, RouterArm.J_OPPORTUNITY_TIME_ACTIVITY, RouterArm.K_UNCERTAINTY_LCB),
        opportunity_activity=arm in (RouterArm.J_OPPORTUNITY_TIME_ACTIVITY, RouterArm.K_UNCERTAINTY_LCB),
        uncertainty_lcb=arm == RouterArm.K_UNCERTAINTY_LCB,
    )
