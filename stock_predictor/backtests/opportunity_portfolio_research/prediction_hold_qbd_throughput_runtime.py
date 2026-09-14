from __future__ import annotations

"""QbD throughput runtime: deeper shared queue plus 24 replay workers.

This module changes execution only. The QbD research contract, entry grid,
threshold calibration, chronological folds, ranking and final holdout state remain
owned by prediction_hold_qbd_surface/search.py.

Stage 1: run up to eight independent outer-window/final-fit coordinators at once.
Each coordinator submits its complete 48-policy coarse stage to the same persistent
ProcessPool, so the pool can hold several 48-job waves concurrently. Robustness is
still dependency-correct: it is submitted only after that window's coarse results
exist.

Stage 2: allow up to 24 replay workers. The affinity plan fills one logical
processor per physical core before using SMT siblings; spare logical processors
remain available for coordinators/OS when the host has more capacity.
"""

import json
import os
from pathlib import Path

from . import portfolio_replay_process_backend as backend
from . import portfolio_policy_walk_forward as multicore_walk_forward, portfolio_policy_search as search
from .affinity_coordinator_pool import AffinityCoordinatorPool
from .cpu_topology import physical_core_affinity_plan
from .portfolio_research_inputs import load_predictions, load_price_panel
from .portfolio_replay_fragment_cache import build_fragment_namespace
from .prediction_hold_qbd_evaluate import CONTRACT_ID, expected_cells, write_evaluation_artifacts
from .portfolio_resilient_process_pool import install_resilient_process_pool, restore_process_pool_runner
from . import prediction_hold_qbd_surface as surface

MAX_QBD_WORKERS = backend.MAX_PROCESS_WORKERS
DEFAULT_QBD_WORKERS = backend.DEFAULT_PROCESS_WORKERS
DEFAULT_QBD_COORDINATORS = 8
MAX_QBD_COORDINATORS = 8


def coordinator_reserve(logical_processors: int, workers: int) -> int:
    """Keep the historical 4-thread reserve only when capacity is actually spare."""
    logical = max(1, int(logical_processors))
    worker_count = max(1, min(int(workers), logical))
    return max(0, min(4, logical - worker_count))


def coarse_queue_capacity(coordinator_threads: int) -> int:
    """Maximum simultaneously submitted coarse QbD jobs before dependencies bite."""
    return max(1, int(coordinator_threads)) * int(surface.ENTRY_GRID_SIZE)


def _qbd_effective_workers(requested: int) -> int:
    limit = max(
        1,
        min(
            int(requested),
            int(backend._PROCESS_WORKERS),
            int(os.cpu_count() or 1),
            MAX_QBD_WORKERS,
        ),
    )
    if backend._AFFINITY_PLAN.get("enabled"):
        limit = min(limit, int(backend._AFFINITY_PLAN.get("workers", limit)))
    return max(1, limit)


def _install_qbd_multicore_backend(process_workers: int) -> None:
    """Install the established backend with the shared 24-worker ceiling."""
    if backend._ORIGINAL_PARALLEL_MAP is None:
        backend._ORIGINAL_PARALLEL_MAP = search._parallel_map

    logical = int(os.cpu_count() or 1)
    configured = max(1, min(int(process_workers), logical, MAX_QBD_WORKERS))
    reserve = coordinator_reserve(logical, configured)
    backend._PROCESS_WORKERS = configured
    backend._AFFINITY_PLAN = physical_core_affinity_plan(
        configured,
        reserve_logical_processors=reserve,
    )
    if backend._AFFINITY_PLAN.get("enabled"):
        backend._PROCESS_WORKERS = min(
            backend._PROCESS_WORKERS,
            int(backend._AFFINITY_PLAN["workers"]),
        )

    telemetry = os.environ.get("OPPORTUNITY_TELEMETRY_PATH", "").strip()
    backend._TELEMETRY_PATH = Path(telemetry) if telemetry else None
    backend._TELEMETRY_SUITE_STARTED = backend.time.perf_counter()
    backend._TELEMETRY_TOTALS = {
        "batches": 0,
        "jobs_submitted": 0,
        "jobs_reused": 0,
        "jobs_computed": 0,
        "pool_recoveries": 0,
        "main_cache_seconds": 0.0,
        "main_aggregation_seconds": 0.0,
        "worker_job_counts": {},
        "worker_compute_seconds": {},
        "worker_idle_seconds": {},
    }
    search._parallel_map = backend.multicore_parallel_map
    for key in (
        "pool_starts",
        "pool_restarts",
        "history_jobs_reused",
        "history_jobs_computed",
        "fold_jobs_reused",
        "fold_jobs_computed",
    ):
        backend._STATS[key] = 0
    backend._STATS["backend"] = "process"

    topo = backend._AFFINITY_PLAN.get("topology", {})
    print(
        "[qbd-throughput] backend installed; "
        f"process_workers={backend._PROCESS_WORKERS}; "
        f"physical_cores={topo.get('physical_cores')}; "
        f"logical_processors={topo.get('logical_processors')}; "
        f"overflow_workers={len(backend._AFFINITY_PLAN.get('overflow_logical_processors', []))}; "
        f"coordinator_reserve={backend._AFFINITY_PLAN.get('reserved_logical_processors', [])}; "
        f"affinity={'ON' if backend._AFFINITY_PLAN.get('enabled') else 'scheduler'}",
        flush=True,
    )


def run_qbd_throughput(args) -> int:
    if args.self_test:
        surface._self_test()
        assert coordinator_reserve(16, 16) == 0
        assert coordinator_reserve(16, 12) == 4
        assert coarse_queue_capacity(8) == 384
        print("PREDICTION_HOLD_QBD_THROUGHPUT_SELF_TEST_PASS")
        return 0

    if not args.v5_predictions or not args.daily_store_root:
        raise ValueError("--v5-predictions and --daily-store-root are required")
    if not 1 <= int(args.max_workers) <= MAX_QBD_WORKERS:
        raise ValueError(f"max-workers out of range: 1-{MAX_QBD_WORKERS}")
    if not 1 <= int(args.coordinator_threads) <= MAX_QBD_COORDINATORS:
        raise ValueError(f"coordinator-threads out of range: 1-{MAX_QBD_COORDINATORS}")

    requested = expected_cells(args.prediction_min, args.prediction_max, args.hold_max)
    horizons = sorted({h for h, _ in requested})
    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=True)
    predictions, prediction_audit = load_predictions(Path(args.v5_predictions))
    missing = sorted(set(horizons) - set(int(x) for x in predictions.horizon.unique()))
    if missing:
        raise RuntimeError(f"QBD_PREDICTION_HORIZONS_MISSING:{missing}")
    tickers = set(predictions.loc[predictions.horizon.isin(horizons), "ticker"].unique())
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
        str(root / "qbd_multicore_telemetry.jsonl"),
    )

    fine_cache = bool(args.fine_grained_fragment_cache)
    print(
        "[qbd-cache] fine-grained replay/fold fragment cache: "
        + (
            "ON (mid-cell fragment resume enabled)"
            if fine_cache
            else "OFF (cell-level resume; throughput mode)"
        ),
        flush=True,
    )
    print(
        "[qbd-throughput] Stage 1 shared queue: "
        f"coordinators={args.coordinator_threads}, "
        f"entry_grid={surface.ENTRY_GRID_SIZE}, "
        f"coarse_queue_capacity_up_to={coarse_queue_capacity(args.coordinator_threads)} jobs",
        flush=True,
    )
    print(
        "[qbd-throughput] Stage 2 replay pool: "
        f"requested_workers={args.max_workers}, max_workers={MAX_QBD_WORKERS}",
        flush=True,
    )

    original_effective_workers = backend._effective_workers
    original_coordinator_executor = multicore_walk_forward.ThreadPoolExecutor
    backend._effective_workers = _qbd_effective_workers
    _install_qbd_multicore_backend(args.max_workers)
    install_resilient_process_pool()
    multicore_walk_forward.ThreadPoolExecutor = AffinityCoordinatorPool

    failed: list[dict] = []
    reused = 0
    complete = 0
    try:
        for horizon in horizons:
            hpred = predictions.loc[predictions.horizon.eq(horizon)].copy()
            namespace = (
                build_fragment_namespace(search._research_input_fingerprint(hpred, prices))
                if fine_cache
                else None
            )
            cache_path = root / "qbd_fragment_cache.sqlite3" if fine_cache else None
            for hold in [d for h, d in requested if h == horizon]:
                path = surface._cell_path(root, horizon, hold)
                if not args.force and surface._load_complete(path, horizon, hold) is not None:
                    reused += 1
                    complete += 1
                    print(f"[qbd] H{horizon:02d}/D{hold:02d}: RESUME", flush=True)
                    continue
                print(
                    f"[qbd] H{horizon:02d}/D{hold:02d}: measure cell "
                    f"{complete + len(failed) + 1}/{len(requested)}; "
                    f"entry_grid={surface.ENTRY_GRID_SIZE}",
                    flush=True,
                )
                try:
                    payload = surface._run_cell(
                        hpred,
                        prices,
                        horizon=horizon,
                        hold=hold,
                        budget=args.stage_a_budget,
                        workers=args.max_workers,
                        cache_path=cache_path,
                        namespace=namespace,
                    )
                    surface._write_json(path, payload)
                    complete += 1
                except Exception as exc:
                    failure = {
                        "contract_id": CONTRACT_ID,
                        "status": "FAILED",
                        "prediction_horizon": horizon,
                        "holding_days": hold,
                        "error": f"{type(exc).__name__}:{exc}",
                        "v45_exit_overlay_used": False,
                    }
                    surface._write_json(path, failure)
                    failed.append(failure)
                    print(
                        f"[qbd] H{horizon:02d}/D{hold:02d}: FAILED {failure['error']}",
                        flush=True,
                    )
                    if args.stop_on_error:
                        raise

        evaluation = write_evaluation_artifacts(
            root,
            prediction_min=args.prediction_min,
            prediction_max=args.prediction_max,
            hold_max=args.hold_max,
            minimum_plateau_cells=args.minimum_plateau_cells,
        )
        summary = {
            "contract_id": CONTRACT_ID,
            "status": "COMPLETE" if evaluation["qbd_complete"] and not failed else "INCOMPLETE",
            "requested_cells": len(requested),
            "completed_cells_this_or_prior_run": complete,
            "reused_complete_cells": reused,
            "failed_cells_this_run": failed,
            "entry_grid_size_per_cell": surface.ENTRY_GRID_SIZE,
            "prediction_and_hold_independent": True,
            "fine_grained_fragment_cache": fine_cache,
            "resume_granularity": "FRAGMENT_AND_CELL" if fine_cache else "COMPLETE_CELL",
            "execution_policy": {
                "id": "QBD_DEEP_QUEUE_16_WORKERS_V1",
                "stage_1": {
                    "shared_process_queue": True,
                    "coordinator_threads": int(args.coordinator_threads),
                    "coarse_queue_capacity_up_to": coarse_queue_capacity(args.coordinator_threads),
                    "dependency_order_preserved": True,
                },
                "stage_2": {
                    "replay_workers": int(backend._PROCESS_WORKERS),
                    "max_replay_workers": MAX_QBD_WORKERS,
                    "logical_cpu_reserve": list(
                        backend._AFFINITY_PLAN.get("reserved_logical_processors", [])
                    ),
                },
            },
            "v45_exit_overlay_used": False,
            "old_v45_180_suite_invoked": False,
            "final_holdout_opened": False,
            "final_holdout_locked": True,
            "prediction_audit": prediction_audit,
            "price_audit": price_audit,
            "backend": backend.backend_stats(),
            "evaluation": evaluation,
        }
        surface._write_json(root / "qbd_run_summary.json", summary)
        print(json.dumps(summary, indent=2, sort_keys=True, default=str))
        return 0 if summary["status"] == "COMPLETE" else 2
    finally:
        multicore_walk_forward.ThreadPoolExecutor = original_coordinator_executor
        restore_process_pool_runner()
        backend.shutdown_multicore_backend()
        backend._effective_workers = original_effective_workers
