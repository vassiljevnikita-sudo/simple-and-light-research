"""Tax configuration contracts for Opportunity-Portfolio research."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TaxConfig:
    enabled: bool = False
    capital_gains_rate: float = 0.25
    solidarity_surcharge: float = 0.055
    allowance_eur: float = 1000.0
    church_tax_rate: float = 0.0
