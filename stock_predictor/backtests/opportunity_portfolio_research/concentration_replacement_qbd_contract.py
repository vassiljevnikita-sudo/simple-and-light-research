from __future__ import annotations

"""Frozen contract values for Phase-4 Concentration x Replacement QbD."""

from .replacement_qbd_contract import (
    ALLOCATION_FIXED,
    BASELINE as BASELINE_REPLACEMENT,
    EXIT_FAMILY_FIXED,
    EXIT_VALUE_FIXED,
    SLEEVE_FIXED,
    parse_replacement,
)

CONTRACT_ID = "CONCENTRATION_REPLACEMENT_QBD_V1"
PHASE3_CONTRACT_ID = "ENTRY_REPLACEMENT_QBD_V1"
DEFAULT_MAX_NAMES = (1, 2, 3, 4, 5)
DEFAULT_REPLACEMENTS = (
    BASELINE_REPLACEMENT,
    "REPLACE_WEAKEST",
)


def parse_max_names(value: int | str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"CONCENTRATION_QBD_INVALID_MAX_NAMES:{value}") from exc
    if parsed not in DEFAULT_MAX_NAMES:
        raise ValueError(f"CONCENTRATION_QBD_INVALID_MAX_NAMES:{value}")
    return parsed


def parse_replacement_treatment(value: str) -> str:
    treatment = parse_replacement(value)
    if treatment not in DEFAULT_REPLACEMENTS:
        raise ValueError(f"CONCENTRATION_QBD_INVALID_REPLACEMENT:{value}")
    return treatment
