"""Transaction-cost contracts for Opportunity-Portfolio research."""
from __future__ import annotations

from dataclasses import dataclass

ROUNDTRIP_BPS = (20, 30, 50)


@dataclass(frozen=True)
class CostModel:
    roundtrip_bps: float = 20.0

    @property
    def per_side_bps(self) -> float:
        return self.roundtrip_bps / 2.0
