from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import multiprocessing as mp
import os
from itertools import product
from pathlib import Path
from typing import Iterator

import pandas as pd

from . import portfolio_policy_walk_forward as multicore_walk_forward, portfolio_policy_search as search
from .portfolio_policy_contracts import Policy, policy_dict
from .affinity_coordinator_pool import AffinityCoordinatorPool
from .portfolio_research_inputs import load_predictions, load_price_panel
from .portfolio_replay_fragment_cache import build_fragment_namespace
from .portfolio_replay_process_backend import backend_stats, install_multicore_backend, shutdown_multicore_backend
from .prediction_hold_qbd_evaluate import CONTRACT_ID, expected_cells, write_evaluation_artifacts
from .portfolio_resilient_process_pool import install_resilient_process_pool, restore_process_pool_runner

ENTRY_GRID_SIZE = len(search.SEARCH_QUANTILES) * len(search.SEARCH_TOP_FRACTIONS) * len(search.SEARCH_MAX_NAMES)
_ORIGINAL_COORDINATOR_EXECUTOR = multicore_walk_forward.ThreadPoolExecutor


def fixed_hold_grid(horizon: int, holding_days: int, sleeve: float = .50) -> list[Policy]:
    h, hold = int(horizon), int(holding_days)
    if h < 1 or hold < 1 or hold > h:
        raise ValueError(f"QBD_INVALID_PREDICTION_HOLD_CELL:H{h}:D{hold}")
    return [
        Policy(h, q, top, n, hold, sleeve=sleeve)
        for q, top, n in product(search.SEARCH_QUANTILES, search.SEARCH_TOP_FRACTIONS, search.SEARCH_MAX_NAMES)
    ]


def _coverage(items: list[Policy]) -> dict:
    return {
        "score_quantiles": sorted({float(p.score_quantile) for p in items}),
        "top_fractions": sorted({float(p.top_fraction) for p in items}),
        "max_names": sorted({int(p.max_names) for p in items}),
        "holding_days": sorted({int(p.holding_days) for p in items}),
    }


def _coverage_complete(items: list[Policy], horizon: int, hold: int) -> bool:
    c = _coverage(items)
    return bool(items) and {p.horizon for p in items} == {horizon} and {p.holding_days for p in items} == {hold} and (
        set(c["score_quantiles"]) >= set(search.SEARCH_QUANTILES)
        and set(c["top_fractions"]) >= set(search.SEARCH_TOP_FRACTIONS)
        and set(c["max_names"]) >= set(search.SEARCH_MAX_NAMES)
    )


@contextmanager
def fixed_hold_search_contract(horizon: int, holding_days: int) -> Iterator[None]:
    """Measure one cell while only the established entry dimensions remain searchable."""
    h, hold = int(horizon), int(holding_days)
    fixed_hold_grid(h, hold)
    names = (
        "grid", "minimum_coverage_budget", "_coverage", "_coverage_complete", "_balanced_budget",
        "_dynamic_neighbors", "choose_policy", "_window_checkpoint_key", "_final_fit_checkpoint_key", "_horizon_checkpoint_key",
    )
    originals = {name: getattr(search, name) for name in names}
    original_choose = search.choose_policy

    def cell_grid(requested_horizon: int, sleeve: float = .50):
        if int(requested_horizon) != h:
            raise RuntimeError(f"QBD_CELL_HORIZON_MISMATCH:expected={h}:actual={requested_horizon}")
        return fixed_hold_grid(h, hold, sleeve)

    def cell_complete(items):
        return _coverage_complete(list(items), h, hold)

    def cell_budget(items, budget):
        selected = list(items)
        if len(selected) != ENTRY_GRID_SIZE or not cell_complete(selected):
            raise RuntimeError(f"QBD_ENTRY_GRID_INCOMPLETE:H{h}:D{hold}:rows={len(selected)}")
        return selected, {
            "requested_budget": int(budget), "effective_budget": len(selected), "minimum_coverage_budget": ENTRY_GRID_SIZE,
            "budget_auto_raised": int(budget) < ENTRY_GRID_SIZE, "coverage": _coverage(selected), "coverage_complete": True,
            "research_contract": CONTRACT_ID, "prediction_horizon_fixed": h, "holding_days_fixed": hold,
            "prediction_and_hold_independent": True, "full_entry_grid_evaluated": True,
        }

    def cell_choose(*args, **kwargs):
        chosen, result, leaderboard, meta = original_choose(*args, **kwargs)
        if chosen.horizon != h or chosen.holding_days != hold:
            raise RuntimeError(f"QBD_CHOSEN_POLICY_ESCAPED_CELL:H{h}:D{hold}")
        meta = dict(meta) | {
            "research_contract": CONTRACT_ID, "prediction_horizon_fixed": h, "holding_days_fixed": hold,
            "prediction_and_hold_independent": True, "dynamic_paths_evaluated": 0, "dynamic_stage_complete": True,
            "dynamic_stage_status": "NOT_APPLICABLE_QBD_FIXED_HOLD_SURFACE", "v45_exit_overlay_used": False,
        }
        return chosen, result, leaderboard, meta

    def window_key(requested_horizon, fold, history_end, budget):
        return search._cache_key(CONTRACT_ID, "outer_window", h, hold, int(requested_horizon), str(fold), str(pd.Timestamp(history_end)), int(budget))

    def final_key(requested_horizon, history_end, budget):
        return search._cache_key(CONTRACT_ID, "final_fit", h, hold, int(requested_horizon), str(pd.Timestamp(history_end)), int(budget))

    def horizon_key(requested_horizon, budget):
        return search._cache_key(CONTRACT_ID, "cell", h, hold, int(requested_horizon), int(budget))

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


def _cell_path(root: Path, horizon: int, hold: int) -> Path:
    return root / "cells" / f"H{horizon:02d}_D{hold:02d}.json"


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    tmp.replace(path)


def _load_complete(path: Path, horizon: int, hold: int) -> dict | None:
    if not path.is_file():
        return None
    try:
        p = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if p.get("contract_id") == CONTRACT_ID and p.get("status") == "COMPLETE" and int(p.get("prediction_horizon", -1)) == horizon and int(p.get("holding_days", -1)) == hold:
        return p
    return None


def _validate_cell(horizon: int, hold: int, outer: list[dict], meta: dict) -> tuple[dict, float]:
    if not outer:
        raise RuntimeError(f"QBD_CELL_HAS_NO_OUTER_RESULTS:H{horizon}:D{hold}")
    if any(int(row.get("horizon", -1)) != horizon or int(row.get("holding_days", -1)) != hold for row in outer):
        raise RuntimeError(f"QBD_OUTER_POLICY_ESCAPED_CELL:H{horizon}:D{hold}")
    policies, thresholds = meta.get("final_policies", {}), meta.get("final_thresholds", {})
    policy = policies.get(horizon, policies.get(str(horizon)))
    threshold = thresholds.get(horizon, thresholds.get(str(horizon)))
    if policy is None or threshold is None:
        raise RuntimeError(f"QBD_FINAL_FIT_MISSING:H{horizon}:D{hold}")
    if policy.horizon != horizon or policy.holding_days != hold:
        raise RuntimeError(f"QBD_FINAL_POLICY_ESCAPED_CELL:H{horizon}:D{hold}")
    return policy_dict(policy), float(threshold)


def _run_cell(
    hpred: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    horizon: int,
    hold: int,
    budget: int,
    workers: int,
    cache_path: Path | None,
    namespace: str | None,
) -> dict:
    with fixed_hold_search_contract(horizon, hold):
        outer, _history, meta = multicore_walk_forward.run_walk_forward(
            hpred, prices, horizons=(horizon,), budget=budget, max_workers=workers,
            fragment_cache_path=cache_path, fragment_namespace=namespace,
        )
    final_policy, final_threshold = _validate_cell(horizon, hold, outer, meta)
    return {
        "contract_id": CONTRACT_ID, "status": "COMPLETE", "prediction_horizon": horizon, "holding_days": hold,
        "prediction_and_hold_independent": True, "entry_grid_size": ENTRY_GRID_SIZE, "v45_exit_overlay_used": False,
        "entry_grid_dimensions": {"score_quantile": list(search.SEARCH_QUANTILES), "top_fraction": list(search.SEARCH_TOP_FRACTIONS), "max_names": list(search.SEARCH_MAX_NAMES)},
        "outer_rows": outer, "final_policy": final_policy, "final_threshold": final_threshold,
        "search_coverage": meta.get("search_coverage", {}).get(horizon, []), "performance": meta.get("performance", {}),
    }


def _self_test() -> None:
    assert len(expected_cells()) == 465
    grid = fixed_hold_grid(6, 3)
    assert len(grid) == ENTRY_GRID_SIZE == 48 and {p.horizon for p in grid} == {6} and {p.holding_days for p in grid} == {3}
    original = search.grid
    with fixed_hold_search_contract(6, 2):
        key2 = search._horizon_checkpoint_key(6, 48)
        assert {p.holding_days for p in search.grid(6)} == {2}
    assert search.grid is original
    with fixed_hold_search_contract(6, 3):
        key3 = search._horizon_checkpoint_key(6, 48)
    assert key2 != key3
    print("PREDICTION_HOLD_QBD_SELF_TEST_PASS")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Measure the Prediction-Horizon x Holding-Days QbD surface")
    p.add_argument("--v5-predictions", default=""); p.add_argument("--daily-store-root", default="")
    p.add_argument("--output-root", default="artifacts/prediction-hold-qbd-surface")
    p.add_argument("--prediction-min", type=int, default=1); p.add_argument("--prediction-max", type=int, default=30); p.add_argument("--hold-max", type=int, default=None)
    p.add_argument("--stage-a-budget", type=int, default=ENTRY_GRID_SIZE); p.add_argument("--max-workers", type=int, default=24); p.add_argument("--coordinator-threads", type=int, default=4)
    p.add_argument("--minimum-plateau-cells", type=int, default=3); p.add_argument("--force", action="store_true"); p.add_argument("--stop-on-error", action="store_true")
    p.add_argument("--fine-grained-fragment-cache", action="store_true", help="Persist individual replay/fold fragments. QbD defaults to cell-level resume for throughput.")
    p.add_argument("--self-test", action="store_true")
    return p.parse_args()


def main(args: argparse.Namespace | None = None) -> int:
    args = args or parse_args()
    if args.self_test:
        _self_test(); return 0
    if not args.v5_predictions or not args.daily_store_root:
        raise ValueError("--v5-predictions and --daily-store-root are required")
    if not 1 <= int(args.max_workers) <= 24 or not 1 <= int(args.coordinator_threads) <= 8:
        raise ValueError("workers/coordinator threads out of range")

    requested = expected_cells(args.prediction_min, args.prediction_max, args.hold_max)
    horizons = sorted({h for h, _ in requested})
    root = Path(args.output_root); root.mkdir(parents=True, exist_ok=True)
    predictions, prediction_audit = load_predictions(Path(args.v5_predictions))
    missing = sorted(set(horizons) - set(int(x) for x in predictions.horizon.unique()))
    if missing:
        raise RuntimeError(f"QBD_PREDICTION_HORIZONS_MISSING:{missing}")
    tickers = set(predictions.loc[predictions.horizon.isin(horizons), "ticker"].unique())
    prices, price_audit = load_price_panel(Path(args.daily_store_root), tickers)

    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS"):
        os.environ[name] = "1"
    os.environ["OPPORTUNITY_WINDOW_PIPELINE"] = str(args.coordinator_threads)
    os.environ.setdefault("OPPORTUNITY_TELEMETRY_PATH", str(root / "qbd_multicore_telemetry.jsonl"))

    fine_cache = bool(args.fine_grained_fragment_cache)
    print(
        "[qbd-cache] fine-grained replay/fold fragment cache: "
        + ("ON (mid-cell fragment resume enabled)" if fine_cache else "OFF (cell-level resume; throughput mode)"),
        flush=True,
    )

    install_multicore_backend(args.max_workers)
    install_resilient_process_pool()
    multicore_walk_forward.ThreadPoolExecutor = AffinityCoordinatorPool
    failed, reused, complete = [], 0, 0
    try:
        for horizon in horizons:
            hpred = predictions.loc[predictions.horizon.eq(horizon)].copy()
            namespace = (
                build_fragment_namespace(search._research_input_fingerprint(hpred, prices))
                if fine_cache else None
            )
            cache_path = root / "qbd_fragment_cache.sqlite3" if fine_cache else None
            for hold in [d for h, d in requested if h == horizon]:
                path = _cell_path(root, horizon, hold)
                if not args.force and _load_complete(path, horizon, hold) is not None:
                    reused += 1; complete += 1
                    print(f"[qbd] H{horizon:02d}/D{hold:02d}: RESUME", flush=True)
                    continue
                print(f"[qbd] H{horizon:02d}/D{hold:02d}: measure cell {complete + len(failed) + 1}/{len(requested)}; entry_grid={ENTRY_GRID_SIZE}", flush=True)
                try:
                    payload = _run_cell(
                        hpred,
                        prices,
                        horizon=horizon,
                        hold=hold,
                        budget=args.stage_a_budget,
                        workers=args.max_workers,
                        cache_path=cache_path,
                        namespace=namespace,
                    )
                    _write_json(path, payload); complete += 1
                except Exception as exc:
                    failure = {"contract_id": CONTRACT_ID, "status": "FAILED", "prediction_horizon": horizon, "holding_days": hold, "error": f"{type(exc).__name__}:{exc}", "v45_exit_overlay_used": False}
                    _write_json(path, failure); failed.append(failure)
                    print(f"[qbd] H{horizon:02d}/D{hold:02d}: FAILED {failure['error']}", flush=True)
                    if args.stop_on_error:
                        raise

        evaluation = write_evaluation_artifacts(root, prediction_min=args.prediction_min, prediction_max=args.prediction_max, hold_max=args.hold_max, minimum_plateau_cells=args.minimum_plateau_cells)
        summary = {
            "contract_id": CONTRACT_ID, "status": "COMPLETE" if evaluation["qbd_complete"] and not failed else "INCOMPLETE",
            "requested_cells": len(requested), "completed_cells_this_or_prior_run": complete, "reused_complete_cells": reused, "failed_cells_this_run": failed,
            "entry_grid_size_per_cell": ENTRY_GRID_SIZE, "prediction_and_hold_independent": True,
            "fine_grained_fragment_cache": fine_cache,
            "resume_granularity": "FRAGMENT_AND_CELL" if fine_cache else "COMPLETE_CELL",
            "v45_exit_overlay_used": False, "old_v45_180_suite_invoked": False,
            "final_holdout_opened": False, "final_holdout_locked": True,
            "prediction_audit": prediction_audit, "price_audit": price_audit, "backend": backend_stats(), "evaluation": evaluation,
        }
        _write_json(root / "qbd_run_summary.json", summary)
        print(json.dumps(summary, indent=2, sort_keys=True, default=str))
        return 0 if summary["status"] == "COMPLETE" else 2
    finally:
        multicore_walk_forward.ThreadPoolExecutor = _ORIGINAL_COORDINATOR_EXECUTOR
        restore_process_pool_runner(); shutdown_multicore_backend()


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())
