"""URTH market-regime classification contract."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RegimeContract:
    version: str = "URTH_TRAILING_REGIME_V1"
    bull_return_126: float = 0.12
    bear_return_126: float = -0.12
    trend_distance: float = 0.03
    transition_vol: float = 0.025
    classification_uses_future_data: bool = False
