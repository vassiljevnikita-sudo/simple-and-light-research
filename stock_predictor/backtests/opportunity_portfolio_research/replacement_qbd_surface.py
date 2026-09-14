from __future__ import annotations

import argparse
from contextlib import contextmanager
from itertools import product
import json
import multiprocessing as mp
import os
from pathlib import Path
from typing import Iterator

import pandas as pd

from . import portfolio_policy_walk_forward as multicore_walk_forward, portfolio_policy_search as search
from .allocation_qbd_evaluate import load_primary_phase1_design_space
from .portfolio_policy_contracts import Policy, policy_dict
from .affinity_coordinator_pool import AffinityCoordinatorPool
from .portfolio_research_inputs import load_predictions, load_price_panel
from .portfolio_replay_fragment_cache import build_fragment_namespace
from .portfolio_replay_process_backend import (
    backend_stats,
    install_multicore_backend,
    shutdown_multicore_backend,
)
from .replacement_qbd_contract import (
    ALLOCATION_FIXED,
    BASELINE,
    CONTRACT_ID,
    DEFAULT_TREATMENTS,
    EXIT_FAMILY_FIXED,
    EXIT_VALUE_FIXED,
    SLEEVE_FIXED,
    parse_replacement,
)
from .replacement_qbd_evaluate import (
    load_phase2_allocation_lock,
    treatment_slug,
    write_evaluation_artifacts,
)
from .portfolio_resilient_process_pool import (
    install_resilient_process_pool,
    restore_process_pool_runner,
)

ENTRY_GRID_SIZE = (
    len(search.SEARCH_QUANTILES)
    * len(search.SEARCH_TOP_FRACTIONS)
    * len(search.SEARCH_MAX_NAMES)
)
_ORIGINAL_COORDINATOR_EXECUTOR = multicore_walk_forward.ThreadPoolExecutor


def replacement_grid(
    horizon: int,
    holding_days: int,
    replacement: str,
    sleeve: float = SLEEVE_FIXED,
) -> list[Policy]:
    h, hold = int(horizon), int(holding_days)
    treatment = parse_replacement(replacement)
    if h < 1 or hold < 1 or hold > h:
        raise ValueError(f"REPLACEMENT_QBD_INVALID_HOLD_CELL:H{h}:D{hold}")
    if abs(float(sleeve) - SLEEVE_FIXED) > 1e-12:
        raise ValueError(
            "REPLACEMENT_QBD_SLEEVE_MUST_REMAIN_FIXED:"
            f"expected={SLEEVE_FIXED}:actual={float(sleeve)}"
        )
    return [
        Policy(
            h,
            q,
            top,
            n,
            hold,
            exit_family=EXIT_FAMILY_FIXED,
            exit_value=EXIT_VALUE_FIXED,
            replacement=treatment,
            allocation=ALLOCATION_FIXED,
            sleeve=SLEEVE_FIXED,
        )
        for q, top, n in product(
            search.SEARCH_QUANTILES,
            search.SEARCH_TOP_FRACTIONS,
            search.SEARCH_MAX_NAMES,
        )
    ]


def _coverage(items: list[Policy]) -> dict:
    return {
        "score_quantiles": sorted({float(p.score_quantile) for p in items}),
        "top_fractions": sorted({float(p.top_fraction) for p in items}),
        "max_names": sorted({int(p.max_names) for p in items}),
        "holding_days": sorted({int(p.holding_days) for p in items}),
        "allocations": sorted({str(p.allocation) for p in items}),
        "replacements": sorted({str(p.replacement) for p in items}),
        "sleeves": sorted({float(p.sleeve) for p in items}),
        "exit_families": sorted({str(p.exit_family) for p in items}),
        "exit_values": sorted({float(p.exit_value) for p in items}),
    }


def _coverage_complete(
    items: list[Policy],
    horizon: int,
    hold: int,
    replacement: str,
) -> bool:
    c = _coverage(items)
    return bool(items) and (
        {p.horizon for p in items} == {horizon}
        and {p.holding_days for p in items} == {hold}
        and {p.replacement for p in items} == {replacement}
        and {p.allocation for p in items} == {ALLOCATION_FIXED}
        and {float(p.sleeve) for p in items} == {SLEEVE_FIXED}
        and {p.exit_family for p in items} == {EXIT_FAMILY_FIXED}
        and {float(p.exit_value) for p in items} == {EXIT_VALUE_FIXED}
        and set(c["score_quantiles"]) >= set(search.SEARCH_QUANTILES)
        and set(c["top_fractions"]) >= set(search.SEARCH_TOP_FRACTIONS)
        and set(c["max_names"]) >= set(search.SEARCH_MAX_NAMES)
    )


@contextmanager
def replacement_search_contract(
    horizon: int,
    holding_days: int,
    replacement: str,
) -> Iterator[None]:
    """Freeze H/D, Phase-2 allocation, replacement treatment, sleeve and exits."""
    h, hold = int(horizon), int(holding_days)
    treatment = parse_replacement(replacement)
    replacement_grid(h, hold, treatment)
    names = (
        "grid",
        "minimum_coverage_budget",
        "_coverage",
        "_coverage_complete",
        "_balanced_budget",
        "_dynamic_neighbors",
        "choose_policy",
        "_window_checkpoint_key",
        "_final_fit_checkpoint_key",
        "_horizon_checkpoint_key",
    )
    originals = {name: getattr(search, name) for name in names}
    original_choose = search.choose_policy

    def cell_grid(requested_horizon: int, sleeve: float = SLEEVE_FIXED):
        if int(requested_horizon) != h:
            raise RuntimeError(
                "REPLACEMENT_QBD_CELL_HORIZON_MISMATCH:"
                f"expected={h}:actual={requested_horizon}"
            )
        return replacement_grid(h, hold, treatment, sleeve)

    def cell_complete(items):
        return _coverage_complete(list(items), h, hold, treatment)

    def cell_budget(items, budget):
        selected = list(items)
        if len(selected) != ENTRY_GRID_SIZE or not cell_complete(selected):
            raise RuntimeError(
                "REPLACEMENT_QBD_ENTRY_GRID_INCOMPLETE:"
                f"H{h}:D{hold}:{treatment}:rows={len(selected)}"
            )
        return selected, {
            "requested_budget": int(budget),
            "effective_budget": len(selected),
            "minimum_coverage_budget": ENTRY_GRID_SIZE,
            "budget_auto_raised": int(budget) < ENTRY_GRID_SIZE,
            "coverage": _coverage(selected),
            "coverage_complete": True,
            "research_contract": CONTRACT_ID,
            "prediction_horizon_fixed": h,
            "holding_days_fixed": hold,
            "replacement_fixed": treatment,
            "allocation_fixed": ALLOCATION_FIXED,
            "sleeve_fixed": SLEEVE_FIXED,
            "exit_family_fixed": EXIT_FAMILY_FIXED,
            "full_entry_grid_evaluated": True,
        }

    def cell_choose(*args, **kwargs):
        chosen, result, leaderboard, meta = original_choose(*args, **kwargs)
        escaped = (
            chosen.horizon != h
            or chosen.holding_days != hold
            or chosen.replacement != treatment
            or chosen.allocation != ALLOCATION_FIXED
            or abs(float(chosen.sleeve) - SLEEVE_FIXED) > 1e-12
            or chosen.exit_family != EXIT_FAMILY_FIXED
            or abs(float(chosen.exit_value) - EXIT_VALUE_FIXED) > 1e-12
        )
        if escaped:
            raise RuntimeError(
                "REPLACEMENT_QBD_CHOSEN_POLICY_ESCAPED_CELL:"
                f"H{h}:D{hold}:{treatment}"
            )
        meta = dict(meta) | {
            "research_contract": CONTRACT_ID,
            "prediction_horizon_fixed": h,
            "holding_days_fixed": hold,
            "replacement_fixed": treatment,
            "allocation_fixed": ALLOCATION_FIXED,
            "sleeve_fixed": SLEEVE_FIXED,
            "exit_family_fixed": EXIT_FAMILY_FIXED,
            "dynamic_paths_evaluated": 0,
            "dynamic_stage_complete": True,
            "dynamic_stage_status": "NOT_APPLICABLE_REPLACEMENT_QBD",
            "phase1_holding_and_prediction_frozen": True,
            "phase2_allocation_frozen": True,
            "v45_exit_overlay_used": False,
            "final_holdout_opened": False,
        }
        return chosen, result, leaderboard, meta

    def window_key(requested_horizon, fold, history_end, budget):
        return search._cache_key(
            CONTRACT_ID,
            "outer_window",
            h,
            hold,
            treatment,
            ALLOCATION_FIXED,
            int(requested_horizon),
            str(fold),
            str(pd.Timestamp(history_end)),
            int(budget),
        )

    def final_key(requested_horizon, history_end, budget):
        return search._cache_key(
            CONTRACT_ID,
            "final_fit",
            h,
            hold,
            treatment,
            ALLOCATION_FIXED,
            int(requested_horizon),
            str(pd.Timestamp(history_end)),
            int(budget),
        )

    def horizon_key(requested_horizon, budget):
        return search._cache_key(
            CONTRACT_ID,
            "cell",
            h,
            hold,
            treatment,
            ALLOCATION_FIXED,
            int(requested_horizon),
            int(budget),
        )

    search.grid = cell_grid
    search.minimum_coverage_budget = lambda: ENTRY_GRID_SIZE
    search._coverage = _coverage
    search._coverage_complete = cell_complete
    search._balanced_budget = cell_budget
    search._dynamic_neighbors = lambda _base: []
    search.choose_policy = cell_choose
    search._window_checkpoint_key = window_key
    search._final_fit_checkpoint_key = final_key
    search._horizon_checkpoint_key = horizon_key
    try:
        yield
    finally:
        for name, value in originals.items():
            setattr(search, name, value)


def _cell_path(root: Path, horizon: int, hold: int, replacement: str) -> Path:
    return root / "cells" / (
        f"H{int(horizon):02d}_D{int(hold):02d}__{treatment_slug(replacement)}.json"
    )


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def _load_complete(
    path: Path,
    horizon: int,
    hold: int,
    replacement: str,
) -> dict | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    valid = (
        payload.get("contract_id") == CONTRACT_ID
        and payload.get("status") == "COMPLETE"
        and int(payload.get("prediction_horizon", -1)) == int(horizon)
        and int(payload.get("holding_days", -1)) == int(hold)
        and str(payload.get("replacement")) == str(replacement)
        and payload.get("phase1_holding_and_prediction_frozen") is True
        and payload.get("phase2_allocation_frozen") is True
        and str(payload.get("allocation")) == ALLOCATION_FIXED
        and abs(float(payload.get("sleeve", -1.0)) - SLEEVE_FIXED) <= 1e-12
        and str(payload.get("exit_family")) == EXIT_FAMILY_FIXED
        and abs(float(payload.get("exit_value", 99.0)) - EXIT_VALUE_FIXED) <= 1e-12
        and payload.get("v45_exit_overlay_used") is False
        and payload.get("final_holdout_opened") is False
    )
    return payload if valid else None


def _validate_cell(
    horizon: int,
    hold: int,
    replacement: str,
    outer: list[dict],
    meta: dict,
) -> tuple[dict, float]:
    if not outer:
        raise RuntimeError(
            "REPLACEMENT_QBD_CELL_HAS_NO_OUTER_RESULTS:"
            f"H{horizon}:D{hold}:{replacement}"
        )
    for row in outer:
        escaped = (
            int(row.get("horizon", -1)) != horizon
            or int(row.get("holding_days", -1)) != hold
            or str(row.get("replacement")) != replacement
            or str(row.get("allocation")) != ALLOCATION_FIXED
            or abs(float(row.get("sleeve", -1.0)) - SLEEVE_FIXED) > 1e-12
            or str(row.get("exit_family")) != EXIT_FAMILY_FIXED
            or abs(float(row.get("exit_value", 0.0)) - EXIT_VALUE_FIXED) > 1e-12
        )
        if escaped:
            raise RuntimeError(
                "REPLACEMENT_QBD_OUTER_POLICY_ESCAPED_CELL:"
                f"H{horizon}:D{hold}:{replacement}"
            )

    policies = meta.get("final_policies", {})
    thresholds = meta.get("final_thresholds", {})
    policy = policies.get(horizon, policies.get(str(horizon)))
    threshold = thresholds.get(horizon, thresholds.get(str(horizon)))
    if policy is None or threshold is None:
        raise RuntimeError(
            f"REPLACEMENT_QBD_FINAL_FIT_MISSING:H{horizon}:D{hold}:{replacement}"
        )
    escaped = (
        policy.horizon != horizon
        or policy.holding_days != hold
        or policy.replacement != replacement
        or policy.allocation != ALLOCATION_FIXED
        or abs(float(policy.sleeve) - SLEEVE_FIXED) > 1e-12
        or policy.exit_family != EXIT_FAMILY_FIXED
        or abs(float(policy.exit_value) - EXIT_VALUE_FIXED) > 1e-12
    )
    if escaped:
        raise RuntimeError(
            "REPLACEMENT_QBD_FINAL_POLICY_ESCAPED_CELL:"
            f"H{horizon}:D{hold}:{replacement}"
        )
    return policy_dict(policy), float(threshold)


def _run_cell(
    hpred: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    horizon: int,
    hold: int,
    replacement: str,
    budget: int,
    workers: int,
    cache_path: Path | None,
    namespace: str | None,
) -> dict:
    with replacement_search_contract(horizon, hold, replacement):
        outer, _history, meta = multicore_walk_forward.run_walk_forward(
            hpred,
            prices,
            horizons=(horizon,),
            budget=budget,
            max_workers=workers,
            fragment_cache_path=cache_path,
            fragment_namespace=namespace,
        )
    final_policy, final_threshold = _validate_cell(
        horizon, hold, replacement, outer, meta
    )
    return {
        "contract_id": CONTRACT_ID,
        "status": "COMPLETE",
        "prediction_horizon": horizon,
        "holding_days": hold,
        "replacement": replacement,
        "entry_grid_size": ENTRY_GRID_SIZE,
        "phase1_holding_and_prediction_frozen": True,
        "phase2_allocation_frozen": True,
        "allocation": ALLOCATION_FIXED,
        "sleeve": SLEEVE_FIXED,
        "exit_family": EXIT_FAMILY_FIXED,
        "exit_value": EXIT_VALUE_FIXED,
        "v45_exit_overlay_used": False,
        "final_holdout_opened": False,
        "entry_grid_dimensions": {
            "score_quantile": list(search.SEARCH_QUANTILES),
            "top_fraction": list(search.SEARCH_TOP_FRACTIONS),
            "max_names": list(search.SEARCH_MAX_NAMES),
        },
        "outer_rows": outer,
        "final_policy": final_policy,
        "final_threshold": final_threshold,
        "search_coverage": meta.get("search_coverage", {}).get(horizon, []),
        "performance": meta.get("performance", {}),
    }


def parse_treatments(raw: str | None) -> tuple[str, ...]:
    if raw is None or not raw.strip():
        return tuple(DEFAULT_TREATMENTS)
    values = tuple(
        parse_replacement(x.strip())
        for x in raw.split(",")
        if x.strip()
    )
    if BASELINE not in values:
        values = (BASELINE,) + values
    return tuple(dict.fromkeys(values))


def _self_test() -> None:
    grid = replacement_grid(24, 5, "REPLACE_WEAKEST")
    assert len(grid) == ENTRY_GRID_SIZE == 48
    assert {p.replacement for p in grid} == {"REPLACE_WEAKEST"}
    assert {p.allocation for p in grid} == {ALLOCATION_FIXED}
    assert {p.sleeve for p in grid} == {SLEEVE_FIXED}
    assert {p.exit_family for p in grid} == {EXIT_FAMILY_FIXED}
    original = search.grid
    with replacement_search_contract(24, 5, BASELINE):
        key1 = search._horizon_checkpoint_key(24, 48)
    assert search.grid is original
    with replacement_search_contract(24, 5, "REPLACE_WEAKEST"):
        key2 = search._horizon_checkpoint_key(24, 48)
    assert key1 != key2
    print("REPLACEMENT_QBD_SURFACE_SELF_TEST_PASS")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Phase-3 Replacement QbD over frozen Phase-1 H/D cells with "
            "Phase-2 EQUAL_ACTIVE fixed"
        )
    )
    p.add_argument("--v5-predictions", default="")
    p.add_argument("--daily-store-root", default="")
    p.add_argument(
        "--phase1-design-space",
        default="artifacts/prediction-hold-qbd-surface/qbd_design_space.csv",
    )
    p.add_argument(
        "--phase2-summary",
        default="artifacts/allocation-qbd/allocation_qbd_summary.json",
    )
    p.add_argument(
        "--phase2-treatment-summary",
        default="artifacts/allocation-qbd/allocation_qbd_treatment_summary.csv",
    )
    p.add_argument("--output-root", default="artifacts/replacement-qbd")
    p.add_argument("--treatments", default="")
    p.add_argument("--stage-a-budget", type=int, default=ENTRY_GRID_SIZE)
    p.add_argument("--max-workers", type=int, default=24)
    p.add_argument("--coordinator-threads", type=int, default=4)
    p.add_argument("--force", action="store_true")
    p.add_argument("--stop-on-error", action="store_true")
    p.add_argument("--fine-grained-fragment-cache", action="store_true")
    p.add_argument("--self-test", action="store_true")
    return p.parse_args()


def main(args: argparse.Namespace | None = None) -> int:
    args = args or parse_args()
    if args.self_test:
        _self_test()
        return 0
    if not args.v5_predictions or not args.daily_store_root:
        raise ValueError("--v5-predictions and --daily-store-root are required")
    if not 1 <= int(args.max_workers) <= 24:
        raise ValueError("max workers out of range")
    if not 1 <= int(args.coordinator_threads) <= 8:
        raise ValueError("coordinator threads out of range")

    allocation_lock = load_phase2_allocation_lock(
        Path(args.phase2_summary), Path(args.phase2_treatment_summary)
    )
    if allocation_lock != ALLOCATION_FIXED:
        raise RuntimeError("REPLACEMENT_QBD_PHASE2_ALLOCATION_NOT_EQUAL_ACTIVE")

    treatments = parse_treatments(args.treatments)
    design_cells = load_primary_phase1_design_space(Path(args.phase1_design_space))
    requested = [
        (h, d, treatment)
        for h, d in design_cells
        for treatment in treatments
    ]
    horizons = sorted({h for h, _, _ in requested})
    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=True)

    predictions, prediction_audit = load_predictions(Path(args.v5_predictions))
    missing = sorted(
        set(horizons) - set(int(x) for x in predictions.horizon.unique())
    )
    if missing:
        raise RuntimeError(f"REPLACEMENT_QBD_PREDICTION_HORIZONS_MISSING:{missing}")
    tickers = set(
        predictions.loc[predictions.horizon.isin(horizons), "ticker"].unique()
    )
    prices, price_audit = load_price_panel(Path(args.daily_store_root), tickers)

    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "BLIS_NUM_THREADS",
    ):
        os.environ[name] = "1"
    os.environ["OPPORTUNITY_WINDOW_PIPELINE"] = str(args.coordinator_threads)
    os.environ.setdefault(
        "OPPORTUNITY_TELEMETRY_PATH",
        str(root / "replacement_qbd_multicore_telemetry.jsonl"),
    )

    fine_cache = bool(args.fine_grained_fragment_cache)
    install_multicore_backend(args.max_workers)
    install_resilient_process_pool()
    multicore_walk_forward.ThreadPoolExecutor = AffinityCoordinatorPool
    failed: list[dict] = []
    reused = 0
    complete = 0
    try:
        for horizon in horizons:
            hpred = predictions.loc[predictions.horizon.eq(horizon)].copy()
            namespace_base = (
                build_fragment_namespace(
                    search._research_input_fingerprint(hpred, prices)
                )
                if fine_cache
                else None
            )
            cache_path = (
                root / "replacement_qbd_fragment_cache.sqlite3"
                if fine_cache
                else None
            )
            cells_for_horizon = [x for x in requested if x[0] == horizon]
            for h, hold, replacement in cells_for_horizon:
                path = _cell_path(root, h, hold, replacement)
                if (
                    not args.force
                    and _load_complete(path, h, hold, replacement) is not None
                ):
                    reused += 1
                    complete += 1
                    print(
                        f"[replacement-qbd] H{h:02d}/D{hold:02d}/{replacement}: RESUME",
                        flush=True,
                    )
                    continue

                print(
                    f"[replacement-qbd] H{h:02d}/D{hold:02d}/{replacement}: "
                    f"measure {complete + len(failed) + 1}/{len(requested)}; "
                    f"entry_grid={ENTRY_GRID_SIZE}",
                    flush=True,
                )
                namespace = (
                    f"{namespace_base}:{CONTRACT_ID}:{h}:{hold}:{replacement}"
                    if namespace_base
                    else None
                )
                try:
                    payload = _run_cell(
                        hpred,
                        prices,
                        horizon=h,
                        hold=hold,
                        replacement=replacement,
                        budget=args.stage_a_budget,
                        workers=args.max_workers,
                        cache_path=cache_path,
                        namespace=namespace,
                    )
                    _write_json(path, payload)
                    complete += 1
                except Exception as exc:
                    failure = {
                        "contract_id": CONTRACT_ID,
                        "status": "FAILED",
                        "prediction_horizon": h,
                        "holding_days": hold,
                        "replacement": replacement,
                        "error": f"{type(exc).__name__}:{exc}",
                        "phase1_holding_and_prediction_frozen": True,
                        "phase2_allocation_frozen": True,
                        "allocation": ALLOCATION_FIXED,
                        "sleeve": SLEEVE_FIXED,
                        "exit_family": EXIT_FAMILY_FIXED,
                        "exit_value": EXIT_VALUE_FIXED,
                        "v45_exit_overlay_used": False,
                        "final_holdout_opened": False,
                    }
                    _write_json(path, failure)
                    failed.append(failure)
                    print(
                        f"[replacement-qbd] H{h:02d}/D{hold:02d}/{replacement}: "
                        f"FAILED {failure['error']}",
                        flush=True,
                    )
                    if args.stop_on_error:
                        raise

        evaluation = write_evaluation_artifacts(
            root,
            phase1_design_space=Path(args.phase1_design_space),
            phase2_summary=Path(args.phase2_summary),
            phase2_treatment_summary=Path(args.phase2_treatment_summary),
            treatments=treatments,
        )
        summary = {
            "contract_id": CONTRACT_ID,
            "status": (
                "COMPLETE"
                if evaluation["qbd_complete"] and not failed
                else "INCOMPLETE"
            ),
            "phase1_design_cells": len(design_cells),
            "phase1_holding_and_prediction_frozen": True,
            "phase2_allocation_fixed": ALLOCATION_FIXED,
            "phase2_allocation_frozen": True,
            "treatments": list(treatments),
            "requested_surface_cells": len(requested),
            "completed_cells_this_or_prior_run": complete,
            "reused_complete_cells": reused,
            "failed_cells_this_run": failed,
            "entry_grid_size_per_cell": ENTRY_GRID_SIZE,
            "sleeve_fixed": SLEEVE_FIXED,
            "exit_family_fixed": EXIT_FAMILY_FIXED,
            "fine_grained_fragment_cache": fine_cache,
            "v45_exit_overlay_used": False,
            "old_v45_180_suite_invoked": False,
            "final_holdout_opened": False,
            "final_holdout_locked": True,
            "interpolation_used": False,
            "prediction_audit": prediction_audit,
            "price_audit": price_audit,
            "backend": backend_stats(),
            "evaluation": evaluation,
        }
        _write_json(root / "replacement_qbd_run_summary.json", summary)
        print(json.dumps(summary, indent=2, sort_keys=True, default=str))
        return 0 if summary["status"] == "COMPLETE" else 2
    finally:
        multicore_walk_forward.ThreadPoolExecutor = _ORIGINAL_COORDINATOR_EXECUTOR
        restore_process_pool_runner()
        shutdown_multicore_backend()


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())
