from __future__ import annotations

"""Frozen contract values for Phase-3 Replacement QbD."""

CONTRACT_ID = "ENTRY_REPLACEMENT_QBD_V1"
PHASE2_CONTRACT_ID = "ENTRY_ALLOCATION_QBD_V1"
BASELINE = "IGNORE_NEW"
DEFAULT_TREATMENTS = (
    BASELINE,
    "REPLACE_WEAKEST",
)
ALLOCATION_FIXED = "EQUAL_ACTIVE"
SLEEVE_FIXED = 0.50
EXIT_FAMILY_FIXED = "FIXED"
EXIT_VALUE_FIXED = 0.0


def parse_replacement(value: str) -> str:
    treatment = str(value).strip().upper()
    if treatment not in DEFAULT_TREATMENTS:
        raise ValueError(f"REPLACEMENT_QBD_UNKNOWN_TREATMENT:{value}")
    return treatment
