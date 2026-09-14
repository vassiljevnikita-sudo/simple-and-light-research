from __future__ import annotations

from dataclasses import dataclass, asdict
import hashlib, json

CONTRACT_ID = "LEARNED_EXIT_QBD_PROFIT_V1"
LEARNED_EXIT_SOURCE_COMMIT = "aa8f1b5be6c5c4947ac2446e06627c9c915bc7d9"
LEARNED_EXIT_METHODOLOGY = "V5_E1_30_CONTINUATION_VALUE_PREQUENTIAL_V1"
EXIT_THRESHOLD = 0.0
HORIZONS = tuple(range(1, 31))
MAX_NAMES = tuple(range(1, 7))
FIXED_ISLAND = (
    (23,3),(23,4),(23,6),(24,3),(24,4),(24,5),(24,6),(24,7),
    (25,4),(25,6),(25,7),(25,8),(26,7),(26,8),
)
ALLOCATION = "EQUAL_ACTIVE"
REPLACEMENT = "IGNORE_NEW"
SLEEVE = 0.50
PRIMARY_ROUNDTRIP_BPS = 20.0
DEFAULT_CAPITAL = 10000.0
DEFAULT_TAX_RATE = 0.25
DEFAULT_SOLI = 0.055
DEFAULT_ALLOWANCE = 1000.0
DEFAULT_BENCHMARK_PARTIAL_EXEMPTION = 0.30


def learned_cells(max_names_values=MAX_NAMES):
    return tuple((h,d,n) for h in HORIZONS for d in range(1,h+1) for n in max_names_values)


def stable_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",",":"), default=str).encode()).hexdigest()

@dataclass(frozen=True)
class ProfitTaxConfig:
    enabled: bool = True
    capital_gains_rate: float = DEFAULT_TAX_RATE
    solidarity_surcharge: float = DEFAULT_SOLI
    allowance_eur: float = DEFAULT_ALLOWANCE
    church_tax_rate: float = 0.0
    benchmark_partial_exemption_rate: float = DEFAULT_BENCHMARK_PARTIAL_EXEMPTION
    benchmark_vorabpauschale_mode: str = "NOT_MODELED"

    def fingerprint(self) -> str:
        return stable_hash(asdict(self))[:20]


def assert_contract() -> None:
    assert len(learned_cells()) == 2790
    assert len(FIXED_ISLAND) == 14
    assert all(1 <= d <= h <= 30 for h,d,_ in learned_cells())
    assert set(n for _,_,n in learned_cells()) == set(MAX_NAMES)
    assert EXIT_THRESHOLD == 0.0
    assert ALLOCATION == "EQUAL_ACTIVE" and REPLACEMENT == "IGNORE_NEW" and SLEEVE == 0.50
