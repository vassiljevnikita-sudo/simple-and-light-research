from __future__ import annotations

"""Adaptive-development H1-H30 clean-search contract.

This module intentionally patches only research search-space functions at runtime.
It does not modify the multicore backend, CPU topology, coordinator pool, queueing,
or telemetry implementation.
"""

from itertools import product
from typing import Iterable

from .portfolio_policy_contracts import Policy
from . import portfolio_policy_search as search

CONTRACT_ID = "H1_30_MATCHED_HOLD_V1"
HORIZONS = tuple(range(1, 31))
_ORIGINAL_CHOOSE_POLICY = search.choose_policy


def matched_grid(horizon: int, sleeve: float = 0.50) -> list[Policy]:
    h = int(horizon)
    if h not in HORIZONS:
        raise ValueError(f"H1_30_CLEAN_HORIZON_OUT_OF_RANGE:{h}")
    return [
        Policy(h, q, top, n, h, sleeve=sleeve)
        for q, top, n in product(
            search.SEARCH_QUANTILES,
            search.SEARCH_TOP_FRACTIONS,
            search.SEARCH_MAX_NAMES,
        )
    ]


def matched_coverage(items: Iterable[Policy]) -> dict:
    rows = list(items)
    return {
        "score_quantiles": sorted({float(p.score_quantile) for p in rows}),
        "top_fractions": sorted({float(p.top_fraction) for p in rows}),
        "max_names": sorted({int(p.max_names) for p in rows}),
        "holding_days": sorted({int(p.holding_days) for p in rows}),
    }


def matched_coverage_complete(items: list[Policy]) -> bool:
    if not items:
        return False
    horizons = {int(p.horizon) for p in items}
    holds = {int(p.holding_days) for p in items}
    if len(horizons) != 1 or holds != horizons:
        return False
    coverage = matched_coverage(items)
    return (
        set(coverage["score_quantiles"]) >= set(search.SEARCH_QUANTILES)
        and set(coverage["top_fractions"]) >= set(search.SEARCH_TOP_FRACTIONS)
        and set(coverage["max_names"]) >= set(search.SEARCH_MAX_NAMES)
    )


def matched_balanced_budget(items: list[Policy], budget: int) -> tuple[list[Policy], dict]:
    """Evaluate the complete 4 x 3 x 4 clean grid for each horizon.

    The clean research arm intentionally has no holding-period dimension: Hn uses
    holding_days=n. Forty-eight policies is small enough to cover the whole grid,
    avoiding another sampling layer when the research question is the horizon itself.
    """
    selected = list(items)
    if not matched_coverage_complete(selected):
        raise RuntimeError("H1_30_CLEAN_GRID_INCOMPLETE")
    return selected, {
        "requested_budget": int(budget),
        "effective_budget": int(len(selected)),
        "minimum_coverage_budget": int(len(selected)),
        "budget_auto_raised": bool(int(budget) < len(selected)),
        "coverage": matched_coverage(selected),
        "coverage_complete": True,
        "research_contract": CONTRACT_ID,
        "holding_period_matched_to_prediction_horizon": True,
        "full_clean_grid_evaluated": True,
    }


def no_dynamic_neighbors(_base: Policy) -> list[Policy]:
    """Clean arm has a fixed horizon exit only; V4.5 is evaluated separately."""
    return []


def mark_clean_search_meta(meta: dict) -> dict:
    """Mark the dynamic-exit stage as intentionally not applicable for the clean arm.

    The standard suite expects a dynamic-exit stage because H5/H10/H20 historically
    searched exit families inside the same arm. H1-H30 clean deliberately does not:
    its only exit is fixed H, while V4.5 is evaluated as the paired second arm. This
    metadata translation prevents an intentional omission from being misclassified as
    an incomplete execution without changing any candidate ranking or replay result.
    """
    clean = dict(meta)
    clean.update({
        "dynamic_paths_evaluated": 0,
        "dynamic_stage_complete": True,
        "dynamic_stage_status": "NOT_APPLICABLE_CLEAN_FIXED_EXIT_SEPARATE_V45_ARM",
        "clean_fixed_exit_only": True,
    })
    return clean


def clean_choose_policy(*args, **kwargs):
    chosen, result, leaderboard, meta = _ORIGINAL_CHOOSE_POLICY(*args, **kwargs)
    return chosen, result, leaderboard, mark_clean_search_meta(meta)


def install_clean_search_contract() -> None:
    search.grid = matched_grid
    search.minimum_coverage_budget = lambda: 48
    search._coverage = matched_coverage
    search._coverage_complete = matched_coverage_complete
    search._balanced_budget = matched_balanced_budget
    search._dynamic_neighbors = no_dynamic_neighbors
    search.choose_policy = clean_choose_policy

    original_window = search._window_checkpoint_key
    original_final = search._final_fit_checkpoint_key
    original_horizon = search._horizon_checkpoint_key
    if not getattr(original_window, "_h1_30_wrapped", False):
        def window_key(horizon, fold, history_end, budget):
            return search._cache_key(CONTRACT_ID, "outer_window", int(horizon), str(fold), str(history_end), int(budget))
        def final_key(horizon, history_end, budget):
            return search._cache_key(CONTRACT_ID, "final_fit", int(horizon), str(history_end), int(budget))
        def horizon_key(horizon, budget):
            return search._cache_key(CONTRACT_ID, "horizon", int(horizon), int(budget))
        window_key._h1_30_wrapped = True
        search._window_checkpoint_key = window_key
        search._final_fit_checkpoint_key = final_key
        search._horizon_checkpoint_key = horizon_key


def validate_prediction_horizons(horizons: Iterable[int]) -> None:
    actual = tuple(sorted({int(h) for h in horizons}))
    if actual != HORIZONS:
        raise RuntimeError(f"H1_30_PREDICTION_CONTRACT_MISMATCH:{actual}")
