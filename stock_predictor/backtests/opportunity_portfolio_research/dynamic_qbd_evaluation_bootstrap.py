"""Deterministic paired moving-calendar-block bootstrap for Step 8.

The bootstrap resamples complete paired daily-return paths.  It never uses a
mean daily-return contrast as a proxy for the economic metric: every draw is
compounded and evaluated with the same CAGR definition as the portfolio
replay.
"""
from __future__ import annotations

from math import ceil
from typing import Sequence

import numpy as np

from .contract_fingerprints import stable_hash


IMPLEMENTATION_ID = "DQBD_MOVING_CALENDAR_BLOCK_BOOTSTRAP_V1"
BLOCK_SIZE = 21
REPETITIONS = 2000
SEED = 212
CONFIDENCE_LEVEL = 0.95


def _canonical_cagr(initial: float, terminal: float, calendar_days: int) -> float:
    years = max(float(calendar_days) / 365.25, 1.0 / 365.25)
    return (terminal / initial) ** (1.0 / years) - 1.0 if initial > 0 and terminal > 0 else -1.0


def _path_cagr_excess(arm_returns: np.ndarray, benchmark_returns: np.ndarray,
                      calendar_days: int) -> float:
    arm_terminal = float(np.prod(1.0 + arm_returns))
    benchmark_terminal = float(np.prod(1.0 + benchmark_returns))
    return _canonical_cagr(1.0, arm_terminal, calendar_days) - _canonical_cagr(
        1.0, benchmark_terminal, calendar_days
    )


def moving_calendar_block_bootstrap(
    arm_a: Sequence[float], arm_b: Sequence[float], *, block_size: int = 21,
    benchmark: Sequence[float] | None = None, calendar_days: int | None = None,
    repetitions: int = 2000, seed: int = 212, confidence_level: float = .95,
) -> dict[str, object]:
    """Bootstrap paired daily returns using contiguous trading-session blocks.

    ``arm_a``, ``arm_b`` and ``benchmark`` are aligned daily total-return
    paths. The same sampled block indices are applied to all three paths,
    preserving pairing and the common benchmark path. Sampling is with
    replacement and the resampled path has the original length.
    """
    a, b = np.asarray(arm_a, dtype=float), np.asarray(arm_b, dtype=float)
    bench = np.zeros(len(a), dtype=float) if benchmark is None else np.asarray(benchmark, dtype=float)
    if (a.ndim != 1 or b.ndim != 1 or bench.ndim != 1 or len(a) != len(b)
            or len(a) != len(bench) or len(a) < BLOCK_SIZE):
        raise ValueError("DQBD_BOOTSTRAP_PAIRED_SERIES_INVALID")
    if (int(repetitions) != REPETITIONS or int(block_size) != BLOCK_SIZE
            or int(seed) != SEED or float(confidence_level) != CONFIDENCE_LEVEL):
        raise ValueError("DQBD_BOOTSTRAP_PARAMETERS_NOT_FROZEN")
    if not np.isfinite(a).all() or not np.isfinite(b).all() or not np.isfinite(bench).all():
        raise ValueError("DQBD_BOOTSTRAP_PAIRED_SERIES_INVALID")
    n = len(a)
    effective_calendar_days = int(calendar_days if calendar_days is not None else round(n * 365.25 / 252.0))
    if effective_calendar_days <= 0:
        raise ValueError("DQBD_BOOTSTRAP_CALENDAR_DAYS_INVALID")
    starts = np.arange(0, n - BLOCK_SIZE + 1, dtype=int)
    rng = np.random.default_rng(SEED)
    observed_a = _path_cagr_excess(a, bench, effective_calendar_days)
    observed_b = _path_cagr_excess(b, bench, effective_calendar_days)
    observed = observed_b - observed_a
    samples = np.empty(REPETITIONS, dtype=float)
    blocks_per_path = ceil(n / BLOCK_SIZE)
    for rep in range(REPETITIONS):
        sampled_starts = rng.choice(starts, size=blocks_per_path, replace=True)
        indices = np.concatenate([np.arange(start, start + BLOCK_SIZE) for start in sampled_starts])[:n]
        samples[rep] = (
            _path_cagr_excess(b[indices], bench[indices], effective_calendar_days)
            - _path_cagr_excess(a[indices], bench[indices], effective_calendar_days)
        )
    alpha = (1.0 - CONFIDENCE_LEVEL) / 2.0
    return {
        "implementation": IMPLEMENTATION_ID,
        "implementation_fingerprint": stable_hash({"implementation": IMPLEMENTATION_ID,
                                                     "metric": "cagr_excess",
                                                     "path_resampling": "COMPOUNDED_DAILY_TOTAL_RETURN",
                                                     "block_size": BLOCK_SIZE, "repetitions": REPETITIONS,
                                                     "seed": SEED, "confidence_level": CONFIDENCE_LEVEL}),
        "block_size": BLOCK_SIZE, "repetitions": REPETITIONS, "seed": SEED,
        "confidence_level": CONFIDENCE_LEVEL, "metric": "cagr_excess",
        "observed_arm_a_cagr_excess": observed_a,
        "observed_arm_b_cagr_excess": observed_b,
        "observed_contrast": observed,
        "confidence_interval": [float(np.quantile(samples, alpha)),
                                 float(np.quantile(samples, 1.0 - alpha))],
        "paired_path_length": n, "calendar_days": effective_calendar_days,
    }
