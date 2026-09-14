from __future__ import annotations

"""12+4 H1-H30 clean opportunity-research entrypoint.

Execution is delegated to the existing multicore entrypoint. Only the research
search contract is changed: horizons are H1..H30, each with holding_days=horizon,
and the clean arm has no dynamic exit family. The existing process pool, affinity,
coordinator threads, caches and telemetry code are untouched.
"""

import multiprocessing as mp

from . import portfolio_research_multicore_entrypoint as multicore_entrypoint
from . import portfolio_policy_walk_forward as multicore_walk_forward
from . import portfolio_research_cli as runner
from .h1_30_matched_hold_search_contract import (
    HORIZONS,
    install_clean_search_contract,
    mark_clean_search_meta,
    validate_prediction_horizons,
)

_ORIGINAL_WALK_FORWARD = multicore_walk_forward.run_walk_forward
_ORIGINAL_POLICY_NEIGHBORS = runner._policy_neighbors


def _normalize_clean_search_meta(meta: dict) -> dict:
    """Upgrade cached/fresh search metadata without invalidating valid replay work."""
    clean = dict(meta)
    coverage = {}
    for horizon, entries in (clean.get("search_coverage", {}) or {}).items():
        coverage[horizon] = [
            mark_clean_search_meta(entry) if isinstance(entry, dict) else entry
            for entry in (entries or [])
        ]
    clean["search_coverage"] = coverage
    clean["research_contract"] = "H1_30_MATCHED_HOLD_V1"
    clean["dynamic_exit_stage"] = "NOT_APPLICABLE_CLEAN_FIXED_EXIT_SEPARATE_V45_ARM"
    return clean


def _h1_30_walk_forward(pred, prices, *args, **kwargs):
    validate_prediction_horizons(pred["horizon"].unique())
    kwargs = dict(kwargs)
    kwargs["horizons"] = HORIZONS
    outer, history, meta = _ORIGINAL_WALK_FORWARD(pred, prices, *args, **kwargs)
    return outer, history, _normalize_clean_search_meta(meta)


def _matched_parameter_neighbors(policy):
    """Reuse the standard plateau test without changing the clean holding horizon."""
    return [
        (dimension, neighbor)
        for dimension, neighbor in _ORIGINAL_POLICY_NEIGHBORS(policy)
        if dimension != "holding_days"
        and int(neighbor.holding_days) == int(policy.holding_days)
    ]


def main() -> int:
    install_clean_search_contract()
    multicore_walk_forward.run_walk_forward = _h1_30_walk_forward
    runner._policy_neighbors = _matched_parameter_neighbors
    try:
        return multicore_entrypoint.main()
    finally:
        runner._policy_neighbors = _ORIGINAL_POLICY_NEIGHBORS
        multicore_walk_forward.run_walk_forward = _ORIGINAL_WALK_FORWARD


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())
