"""Portfolio policy identity and search-domain constants."""
from __future__ import annotations

from dataclasses import asdict, dataclass

from .contract_fingerprints import stable_hash

HORIZONS = (5, 10, 20)
QUANTILES = (0.75, 0.85, 0.90, 0.95, 0.975, 0.99)
TOP_FRACTIONS = (0.005, 0.01, 0.025, 0.05)
MAX_NAMES = (1, 2, 3, 5, 8)
HOLDING_DAYS = (1, 2, 3, 5, 7, 10, 15, 20)
SLEEVES = (0.25, 0.50, 0.75, 1.00)
CAPITALS = (5000.0, 10000.0, 15000.0)


@dataclass(frozen=True)
class Policy:
    horizon: int
    score_quantile: float
    top_fraction: float
    max_names: int
    holding_days: int
    exit_family: str = "FIXED"
    exit_value: float = 0.0
    replacement: str = "IGNORE_NEW"
    allocation: str = "EQUAL_ACTIVE"
    sleeve: float = 0.50

    @property
    def policy_id(self) -> str:
        return stable_hash(asdict(self))[:16]


def policy_dict(policy: Policy) -> dict:
    return asdict(policy) | {"policy_id": policy.policy_id}
