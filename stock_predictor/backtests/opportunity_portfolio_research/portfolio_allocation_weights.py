from __future__ import annotations

"""Pure allocation-treatment functions for Phase-2 Allocation QbD.

Treatments are intentionally based only on scores available at the decision date.
No volatility, future return, or final-holdout information is used.
"""

from dataclasses import dataclass
import math
import re
from typing import Sequence

import numpy as np


CONTRACT_ID = "ENTRY_ALLOCATION_QBD_V1"

DEFAULT_TREATMENTS = (
    "EQUAL_ACTIVE",
    "RANK_POWER:1.0",
    "RANK_POWER:1.5",
    "RANK_POWER:2.0",
    "SCORE_EXCESS_POWER:1.0",
    "SCORE_EXCESS_POWER:1.5",
    "SCORE_EXCESS_POWER:2.0",
    "SCORE_SOFTMAX:0.5",
    "SCORE_SOFTMAX:1.0",
    "SCORE_SOFTMAX:2.0",
)

_TREATMENT_RE = re.compile(
    r"^(?P<family>EQUAL_ACTIVE|RANK_POWER|SCORE_EXCESS_POWER|SCORE_SOFTMAX)"
    r"(?::(?P<strength>[0-9]+(?:\.[0-9]+)?))?$"
)


@dataclass(frozen=True)
class AllocationSpec:
    raw: str
    family: str
    strength: float | None


def parse_allocation(value: str) -> AllocationSpec:
    raw = str(value).strip().upper()
    match = _TREATMENT_RE.fullmatch(raw)
    if not match:
        raise ValueError(f"ALLOCATION_QBD_UNKNOWN_TREATMENT:{value}")
    family = match.group("family")
    raw_strength = match.group("strength")
    if family == "EQUAL_ACTIVE":
        if raw_strength is not None:
            raise ValueError("EQUAL_ACTIVE_DOES_NOT_ACCEPT_STRENGTH")
        return AllocationSpec(raw=raw, family=family, strength=None)
    if raw_strength is None:
        raise ValueError(f"ALLOCATION_QBD_STRENGTH_REQUIRED:{family}")
    strength = float(raw_strength)
    if not math.isfinite(strength) or strength <= 0.0:
        raise ValueError(f"ALLOCATION_QBD_INVALID_STRENGTH:{value}")
    return AllocationSpec(raw=raw, family=family, strength=strength)


def _normalize(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    if x.ndim != 1:
        raise ValueError("ALLOCATION_WEIGHTS_REQUIRE_1D_VALUES")
    if len(x) == 0:
        return x
    if not np.isfinite(x).all() or (x < 0.0).any():
        raise ValueError("ALLOCATION_WEIGHTS_MUST_BE_FINITE_NONNEGATIVE")
    total = float(x.sum())
    if total <= 0.0:
        return np.full(len(x), 1.0 / len(x), dtype=float)
    result = x / total
    return result / float(result.sum())


def allocation_weights(
    treatment: str,
    scores: Sequence[float],
    *,
    threshold: float | None = None,
) -> np.ndarray:
    """Return normalized weights for one simultaneous entry batch.

    Scores must be ordered from strongest to weakest, matching the replay's
    deterministic signal ordering. The function never inspects future prices.
    """
    spec = parse_allocation(treatment)
    x = np.asarray(tuple(float(v) for v in scores), dtype=float)
    if x.ndim != 1 or not np.isfinite(x).all():
        raise ValueError("ALLOCATION_QBD_SCORES_MUST_BE_FINITE_1D")
    n = len(x)
    if n == 0:
        return np.empty(0, dtype=float)
    if spec.family == "EQUAL_ACTIVE":
        return np.full(n, 1.0 / n, dtype=float)

    strength = float(spec.strength)
    if spec.family == "RANK_POWER":
        ranks = np.arange(1, n + 1, dtype=float)
        return _normalize(np.power(ranks, -strength))

    if spec.family == "SCORE_EXCESS_POWER":
        if threshold is None or not math.isfinite(float(threshold)):
            raise ValueError("SCORE_EXCESS_POWER_REQUIRES_FINITE_THRESHOLD")
        excess = np.maximum(x - float(threshold), 0.0)
        positive = excess[excess > 0.0]
        epsilon = max(float(np.min(positive)) * 1e-6, 1e-12) if len(positive) else 1e-12
        return _normalize(np.power(excess + epsilon, strength))

    if spec.family == "SCORE_SOFTMAX":
        if n == 1:
            return np.ones(1, dtype=float)
        std = float(np.std(x))
        z = np.zeros(n, dtype=float) if std <= 1e-15 else (x - float(np.mean(x))) / std
        logits = strength * z
        logits -= float(np.max(logits))
        return _normalize(np.exp(logits))

    raise AssertionError(f"UNREACHABLE_ALLOCATION_FAMILY:{spec.family}")


def requested_notionals(
    treatment: str,
    scores: Sequence[float],
    sleeve_capacity: float,
    *,
    threshold: float | None = None,
) -> list[float]:
    capacity = max(0.0, float(sleeve_capacity))
    weights = allocation_weights(treatment, scores, threshold=threshold)
    return [capacity * float(w) for w in weights]


def concentration(weights: Sequence[float]) -> dict[str, float]:
    x = np.asarray(tuple(float(v) for v in weights), dtype=float)
    if len(x) == 0:
        return {"max_name_weight": 0.0, "hhi": 0.0, "effective_names": 0.0}
    x = _normalize(x)
    hhi = float(np.square(x).sum())
    return {
        "max_name_weight": float(np.max(x)),
        "hhi": hhi,
        "effective_names": float(1.0 / hhi) if hhi > 0.0 else 0.0,
    }
