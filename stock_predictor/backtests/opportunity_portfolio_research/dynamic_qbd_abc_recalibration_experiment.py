"""Matched Dynamic-QBD A/B/C recalibration decomposition.

This Development-only experiment isolates two adaptation mechanisms while keeping
recipe identity, portfolio policy, replay inputs, costs and evaluation dates fixed:

A  initial frozen model + initial frozen calibration;
B  same initial frozen model + rolling monthly recalibration;
C  same frozen recipe freshly refit each month + calibration at each fresh fit.

Primary causal contrasts:
    B - A = rolling recalibration effect
    C - B = fresh-refit effect conditional on monthly calibration

No recipe switching, parameter search, holdout access, promotion or capital
authority is permitted.
"""
from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from datetime import date
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import time
from typing import Any

import pandas as pd

from .contract_fingerprints import stable_hash
from .cost_contracts import CostModel
from .portfolio_policy_contracts import Policy
from .tax_contracts import TaxConfig
from .dynamic_qbd_conditional_refit_experiment import (
    _arm_signals_schedule,
    _bind_sources,
    _fit_source,
)
from .dynamic_qbd_h1_30_adapter import ParquetH130DatasetMaterializer
from .dynamic_qbd_maturity import HorizonMaturityResolver
from .dynamic_qbd_model_combination_experiment import comparison_markdown
from .dynamic_qbd_development_pipeline import materialize_daily_store_prices
from .dynamic_qbd_portfolio_replay import replay_family
from .dynamic_qbd_generation_recalibration import recalibrate_generation
from .dynamic_qbd_recipe_hysteresis import recipe_key
from .dynamic_qbd_recipe_hysteresis_experiment import (
    _causal_recipe_assessments,
    _choice_for_key,
    _choice_key_text,
    _materializer_for_choice,
)
from .dynamic_qbd_runtime_resources import active_cpu_contract, configure_cpu_peak
from .dynamic_qbd_family_surface import build_family_specs
from .dynamic_qbd_wealth_metrics import wealth_path_metrics
from .cpu_topology import physical_core_affinity_plan, set_current_process_logical_affinity


AUTHORITY = "SHADOW_ONLY_NO_PROMOTION_NO_HOLDOUT"
CLOSED_HOLDOUT_START = date(2026, 7, 25)
SCHEMA_VERSION = "DYNAMIC_QBD_MATCHED_ABC_RECALIBRATION_V1"

ARM_A = "A_FROZEN_MODEL_FROZEN_CALIBRATION"
ARM_B = "B_FROZEN_MODEL_ROLLING_RECALIBRATION"
ARM_C = "C_MONTHLY_REFIT_FROZEN_RECIPE"
ARMS = (ARM_A, ARM_B, ARM_C)

ABC_FIT_WORKERS = 24
ABC_FIT_QUEUE_MULTIPLIER = 2

_ABC_WORKER_CONTEXT: dict | None = None
_ABC_WORKER_INDEX = -1
_ABC_WORKER_LOGICAL_PROCESSOR: int | None = None
_ABC_WORKER_AFFINITY_PINNED = False
_ABC_WORKER_LAST_FINISH = 0.0


def _abc_process_initializer(context: dict, worker_map: list[dict], slot_counter) -> None:
    """Initialize one ABC fit process and bind it to one physical-core lane."""
    global _ABC_WORKER_CONTEXT, _ABC_WORKER_INDEX, _ABC_WORKER_LOGICAL_PROCESSOR
    global _ABC_WORKER_AFFINITY_PINNED, _ABC_WORKER_LAST_FINISH
    for name in (
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS",
    ):
        os.environ[name] = "1"
    if slot_counter is not None:
        with slot_counter.get_lock():
            slot = int(slot_counter.value)
            slot_counter.value += 1
    else:
        slot = 0
    _ABC_WORKER_INDEX = slot
    _ABC_WORKER_CONTEXT = context
    if worker_map:
        row = worker_map[slot % len(worker_map)]
        _ABC_WORKER_LOGICAL_PROCESSOR = int(row["logical_processor"])
        _ABC_WORKER_AFFINITY_PINNED = set_current_process_logical_affinity(
            _ABC_WORKER_LOGICAL_PROCESSOR
        )
    _ABC_WORKER_LAST_FINISH = time.perf_counter()


def _abc_fit_process_worker(payload: tuple[int, tuple, dict]) -> dict:
    """Fit one independent source in a spawned process and return its telemetry."""
    global _ABC_WORKER_LAST_FINISH
    if _ABC_WORKER_CONTEXT is None:
        raise RuntimeError("ABC_PROCESS_WORKER_NOT_INITIALIZED")
    item_index, source_key, choice = payload
    started = time.perf_counter()
    idle_seconds = max(0.0, started - _ABC_WORKER_LAST_FINISH)
    cutoff, recipe = source_key
    context = _ABC_WORKER_CONTEXT
    if _ABC_WORKER_LOGICAL_PROCESSOR is not None:
        # Re-assert the binding before each task.  This protects the contract
        # after external affinity changes and makes the task audit explicit.
        affinity_pinned = set_current_process_logical_affinity(
            _ABC_WORKER_LOGICAL_PROCESSOR
        )
    else:
        affinity_pinned = False
    try:
        source = _fit_source(
            index=item_index,
            cutoff=cutoff,
            choice=choice,
            base=context["base"],
            metric_rows=context["metric_rows"],
            signal_panel=context["signal_panel"],
            sessions=context["sessions"],
            output_root=context["output_root"],
            end=context["end"],
            horizon=context["horizon"],
            materializer_cache={recipe: context["materializer_cache"][recipe]},
        )
    except BaseException as exc:
        finished = time.perf_counter()
        _ABC_WORKER_LAST_FINISH = finished
        return {
            "status": "FAILED",
            "item_index": int(item_index),
            "error": f"{type(exc).__name__}:{exc}",
            "telemetry": {
                "worker_index": int(_ABC_WORKER_INDEX),
                "worker_pid": int(os.getpid()),
                "logical_processor": _ABC_WORKER_LOGICAL_PROCESSOR,
                "affinity_pinned": bool(affinity_pinned),
                "compute_seconds": max(0.0, finished - started),
                "idle_seconds_before_job": idle_seconds,
            },
        }
    finished = time.perf_counter()
    _ABC_WORKER_LAST_FINISH = finished
    return {
        "status": "COMPLETE",
        "item_index": int(item_index),
        "source": source,
        "telemetry": {
            "worker_index": int(_ABC_WORKER_INDEX),
            "worker_pid": int(os.getpid()),
            "logical_processor": _ABC_WORKER_LOGICAL_PROCESSOR,
            "affinity_pinned": bool(affinity_pinned),
            "compute_seconds": max(0.0, finished - started),
            "idle_seconds_before_job": idle_seconds,
        },
    }


def _parallel_fit_required_sources(
    *,
    plans: dict[str, list[dict]],
    base,
    metric_rows: list[dict],
    signal_panel: Path,
    sessions: tuple[date, ...],
    output_root: Path,
    end: date,
    horizon: int,
) -> tuple[dict, list[dict], dict]:
    """Fit independent monthly sources on a bounded spawned process pool.

    The initial source is a single deduplicated task shared by A/B/C. Every
    later task is an independent C fresh fit, so submission order cannot change
    the scientific contract. At most 2 * worker_count futures are in flight;
    the queue is intentionally bounded for predictable memory pressure. Each
    process owns one physical-core-first affinity lane and reports measured
    compute/idle time, process id and logical processor.
    """
    required: dict[tuple, dict] = {}
    requested_by: dict[tuple, list[str]] = {}
    for arm, plan in plans.items():
        for row in plan:
            if not row["refit_required"]:
                continue
            source_key = (row["activation_cutoff"], recipe_key(row["choice"]))
            required[source_key] = dict(row["choice"])
            requested_by.setdefault(source_key, []).append(arm)

    items = sorted(required.items(), key=lambda item: (item[0][0], item[0][1]))
    if not items:
        raise ValueError("ABC_RECALIBRATION_REQUIRED_SOURCES_EMPTY")

    workers = min(ABC_FIT_WORKERS, len(items))
    queue_capacity = workers * ABC_FIT_QUEUE_MULTIPLIER
    telemetry_path = output_root / "worker-telemetry.jsonl"
    telemetry_path.parent.mkdir(parents=True, exist_ok=True)
    telemetry_path.write_text("", encoding="utf-8")
    # Materialize the one frozen recipe once before concurrent work starts.
    # Workers receive the already-created object and therefore never race on
    # recipe-metrics file creation.
    materializer_cache: dict = {}
    for _, choice in items:
        _materializer_for_choice(
            choice=choice,
            horizon=horizon,
            metric_rows=metric_rows,
            signal_panel=signal_panel,
            output_root=output_root,
            end=end,
            cache=materializer_cache,
        )

    sources: dict = {}
    completed: dict[int, dict] = {}
    pending: dict = {}
    next_index = 0
    logical_processors = int(os.cpu_count() or 1)
    affinity_plan = physical_core_affinity_plan(
        workers,
        reserve_logical_processors=max(0, logical_processors - workers),
    )
    worker_map = list(affinity_plan.get("worker_map") or [])[:workers]
    context = {
        "base": base,
        "metric_rows": metric_rows,
        "signal_panel": signal_panel,
        "sessions": sessions,
        "output_root": output_root,
        "end": end,
        "horizon": horizon,
        "materializer_cache": materializer_cache,
    }
    spawn_context = mp.get_context("spawn")
    slot_counter = spawn_context.Value("i", 0, lock=True)
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=spawn_context,
        initializer=_abc_process_initializer,
        initargs=(context, worker_map, slot_counter),
    ) as pool:
        while next_index < len(items) and len(pending) < queue_capacity:
            source_key, choice = items[next_index]
            future = pool.submit(
                _abc_fit_process_worker,
                (next_index, source_key, choice),
            )
            pending[future] = next_index
            next_index += 1
        max_pending = len(pending)
        while pending:
            done, _ = wait(tuple(pending), return_when=FIRST_COMPLETED)
            for future in done:
                pending.pop(future)
                result = future.result()
                telemetry = dict(result.get("telemetry") or {})
                cutoff, _ = items[int(result["item_index"])][0]
                record = {
                    "event": "abc_fit_worker_task",
                    "status": str(result.get("status", "FAILED")),
                    "task_index": int(result["item_index"]),
                    "fit_cutoff": cutoff.isoformat(),
                    **telemetry,
                }
                with telemetry_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, sort_keys=True) + "\n")
                if result.get("status") != "COMPLETE":
                    raise RuntimeError(
                        f"ABC_PROCESS_FIT_FAILED:{result.get('error', 'UNKNOWN')}"
                    )
                item_index = int(result["item_index"])
                completed[item_index] = result["source"]
                while next_index < len(items) and len(pending) < queue_capacity:
                    source_key, choice = items[next_index]
                    replacement = pool.submit(
                        _abc_fit_process_worker,
                        (next_index, source_key, choice),
                    )
                    pending[replacement] = next_index
                    next_index += 1
                max_pending = max(max_pending, len(pending))

    for item_index, (source_key, choice) in enumerate(items):
        source = completed[item_index]
        sources[source_key] = source

    records = [json.loads(line) for line in telemetry_path.read_text(encoding="utf-8").splitlines()]
    compute: dict[str, float] = {}
    idle: dict[str, float] = {}
    jobs: dict[str, int] = {}
    worker_pids: dict[str, int] = {}
    worker_logical_processors: dict[str, int] = {}
    affinity_results: list[bool] = []
    for record in records:
        key = str(record["worker_index"])
        compute[key] = compute.get(key, 0.0) + float(record["compute_seconds"])
        idle[key] = idle.get(key, 0.0) + float(record["idle_seconds_before_job"])
        jobs[key] = jobs.get(key, 0) + 1
        if record.get("worker_pid") is not None:
            worker_pids[key] = int(record["worker_pid"])
        if record.get("logical_processor") is not None:
            worker_logical_processors[key] = int(record["logical_processor"])
        if "affinity_pinned" in record:
            affinity_results.append(bool(record["affinity_pinned"]))
    worker_telemetry = {
        "event": "abc_fit_worker_summary",
        "worker_kind": "PROCESS_POOL",
        "scheduler_version": "ABC_PROCESS_POOL_V1_BOUNDED_QUEUE_PCORE_FIRST",
        "configured_workers": ABC_FIT_WORKERS,
        "active_workers": workers,
        "queue_capacity": queue_capacity,
        "max_queue_depth": max_pending,
        "affinity_enabled": bool(affinity_plan.get("enabled")),
        "affinity_plan_logical_processors": {
            str(row.get("worker_slot")): int(row["logical_processor"])
            for row in worker_map
        },
        "jobs_submitted": len(items),
        "jobs_completed": len(records),
        "worker_job_counts": jobs,
        "worker_pids": worker_pids,
        "worker_logical_processors": worker_logical_processors,
        "affinity_pinned_fraction": (
            sum(affinity_results) / len(affinity_results)
            if affinity_results else 0.0
        ),
        "worker_compute_seconds": compute,
        "worker_idle_seconds": idle,
        "worker_utilization": {
            key: (compute[key] / (compute[key] + idle[key])
                  if compute[key] + idle[key] > 0 else 0.0)
            for key in compute
        },
    }
    with telemetry_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(worker_telemetry, sort_keys=True) + "\n")

    audit = []
    for source_key, choice in items:
        source = sources[source_key]
        cutoff, _ = source_key
        audit.append({
            "fit_cutoff": cutoff.isoformat(),
            "recipe": _choice_key_text(choice),
            "model_family": str(choice["family"]),
            "candidate_id": str(choice["candidate_id"]),
            "parameters": dict(choice.get("parameters", {})),
            "fold_count": int(choice["fold_count"]),
            "fold_ids": list(choice.get("fold_ids", ())),
            "robust_score": float(choice["robust_score"]),
            "requested_by_arms": sorted(set(requested_by[source_key])),
            "model_artifact_id": str(source["build"]["model_artifact_id"]),
            "model_artifact_sha256": str(source["build"]["model_artifact_sha256"]),
            "fresh_fit": True,
        })
    return sources, audit, worker_telemetry


def _build_plans(assessments: list[dict]) -> tuple[dict[str, list[dict]], list[dict]]:
    """Freeze the first exact causal recipe in every arm.

    A/B share one initial model source. C requests the same exact recipe at every
    monthly assessment, but freshly fits it. No later selector winner may alter
    recipe identity in this experiment.
    """
    if not assessments:
        raise ValueError("ABC_RECALIBRATION_ASSESSMENTS_EMPTY")

    frozen_key = recipe_key(assessments[0]["selector_winner"])
    plans = {arm: [] for arm in ARMS}
    audit: list[dict] = []

    for index, assessment in enumerate(assessments):
        choice = _choice_for_key(assessment["eligible_choices"], frozen_key)
        initial = index == 0
        common = {
            "activation_cutoff": assessment["assessment_date"],
            "latest_matured_evidence_date": assessment["latest_matured_evidence_date"],
            "evidence_fingerprint": str(assessment["evidence_fingerprint"]),
            "choice": choice,
        }
        plans[ARM_A].append({
            **common,
            "refit_required": initial,
            "refit_reason": "INITIAL_FIT" if initial else "FROZEN_MODEL_FROZEN_CALIBRATION",
        })
        plans[ARM_B].append({
            **common,
            "refit_required": initial,
            "refit_reason": "INITIAL_FIT" if initial else "ROLLING_RECALIBRATION_ONLY",
        })
        plans[ARM_C].append({
            **common,
            "refit_required": True,
            "refit_reason": "MONTHLY_FROZEN_RECIPE_REFIT_CONTROL",
        })
        audit.append({
            "assessment_date": assessment["assessment_date"].isoformat(),
            "latest_matured_evidence_date": assessment["latest_matured_evidence_date"].isoformat(),
            "frozen_recipe": _choice_key_text(choice),
            "selector_winner": _choice_key_text(assessment["selector_winner"]),
            "selector_winner_is_frozen_recipe": recipe_key(assessment["selector_winner"]) == frozen_key,
            "frozen_recipe_fold_count": int(choice["fold_count"]),
            "frozen_recipe_robust_score": float(choice["robust_score"]),
            "a_refit": initial,
            "b_refit": initial,
            "c_refit": True,
        })

    return plans, audit


def _frozen_a_schedule(
    *,
    plan: list[dict],
    sources: dict,
    sessions: tuple[date, ...],
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict], dict]:
    """Resolve A exactly once and reuse the same model and calibration forever."""
    if not plan:
        raise ValueError("ABC_RECALIBRATION_A_PLAN_EMPTY")
    first = plan[0]
    source = sources[first["source_key"]]
    cutoff = first["activation_cutoff"]
    family = source["family"]
    builder = source["builder"]
    build = source["build"]
    frame = builder.calibration_predictions(
        family=family,
        build=build,
        information_cutoff=cutoff,
    )
    generation_id = stable_hash({
        "schema_version": SCHEMA_VERSION,
        "arm": ARM_A,
        "model_artifact_id": build["model_artifact_id"],
        "activation_cutoff": cutoff,
        "calibration_contract": "INITIAL_CALIBRATION_FROZEN_FOREVER",
    })[:24]
    resolved = recalibrate_generation(
        family,
        generation_id,
        frame,
        information_cutoff=cutoff,
        maturity=HorizonMaturityResolver(sessions),
    )
    model_artifact_id = str(build["model_artifact_id"])
    schedule = pd.DataFrame([{
        "activation_date": cutoff,
        "family_id": ARM_A,
        "generation_id": generation_id,
        "model_artifact_id": model_artifact_id,
        "resolved_threshold": float(resolved.resolved_threshold),
        "resolved_top_fraction": float(resolved.resolved_top_fraction),
        "entry_policy_id": f"{ARM_A}_ENTRY_{cutoff}",
        "exit_policy_id": "FIXED_D2",
    }])
    signals = source["predictions"].copy()

    audit = []
    for index, row in enumerate(plan):
        assessment = row["activation_cutoff"]
        audit.append({
            "arm": ARM_A,
            "assessment_date": assessment.isoformat(),
            "model_fit_cutoff": source["fit_cutoff"].isoformat(),
            "model_age_calendar_days": int((assessment - source["fit_cutoff"]).days),
            "active_recipe": _choice_key_text(source["choice"]),
            "model_artifact_id": model_artifact_id,
            "calibration_source": (
                "FIT_GENERATION_CALIBRATION_FROZEN"
                if index == 0 else "FROZEN_INITIAL_CALIBRATION_REUSED"
            ),
            "resolved_threshold": float(resolved.resolved_threshold),
            "resolved_top_fraction": float(resolved.resolved_top_fraction),
            "matured_observation_count": int(resolved.observations),
            "calibration_resolution_performed": index == 0,
            "refit_this_assessment": index == 0,
            "refit_reason": row["refit_reason"],
        })

    identity = {
        "model_artifact_id": model_artifact_id,
        "generation_id": generation_id,
        "resolved_threshold": float(resolved.resolved_threshold),
        "resolved_top_fraction": float(resolved.resolved_top_fraction),
        "calibration_fingerprint": str(resolved.calibration_fingerprint),
        "calibration_observations": int(resolved.observations),
    }
    return signals, schedule, audit, identity


def _result_row(
    *,
    arm: str,
    result: dict,
    fresh_fit_count: int,
    calibration_resolution_count: int,
    rolling_recalibration_count: int,
    frozen_calibration_reuse_count: int,
) -> dict:
    metrics = result["metrics"]
    risk = wealth_path_metrics(result["curve"])
    return {
        "strategy": arm,
        "terminal_value": float(metrics["terminal_value"]),
        "urth_terminal_value": float(metrics["urth_terminal_value"]),
        "terminal_excess_eur": float(metrics["terminal_value"] - metrics["urth_terminal_value"]),
        "terminal_relative_return": float(metrics["terminal_value"] / metrics["urth_terminal_value"] - 1.0),
        "cagr": float(metrics.get("cagr", float("nan"))),
        "urth_cagr": float(metrics.get("urth_cagr", float("nan"))),
        "cagr_excess": float(metrics.get("cagr_excess", float("nan"))),
        "trade_count": int(metrics["trade_count"]),
        "total_cost_eur": float(metrics["total_cost_eur"]),
        "relative_max_drawdown": float(risk["relative_max_drawdown"]),
        "fresh_fit_count": int(fresh_fit_count),
        "calibration_resolution_count": int(calibration_resolution_count),
        "rolling_recalibration_count": int(rolling_recalibration_count),
        "frozen_calibration_reuse_count": int(frozen_calibration_reuse_count),
        "recipe_change_count": 0,
        "model_family_switch_count": 0,
    }


def _replay(
    *,
    arm: str,
    signals: pd.DataFrame,
    schedule: pd.DataFrame,
    prices: pd.DataFrame,
    start: date,
    end: date,
    horizon: int,
    holding_days: int,
    max_names: int,
    top_fraction: float,
    initial: float,
) -> dict:
    return replay_family(
        signals=signals,
        prices=prices,
        policy=Policy(
            horizon=horizon,
            score_quantile=.5,
            top_fraction=top_fraction,
            max_names=max_names,
            holding_days=holding_days,
            sleeve=.5,
        ),
        generation_schedule=schedule,
        cost=CostModel(20.0),
        tax=TaxConfig(False),
        start=pd.Timestamp(start),
        end=pd.Timestamp(end),
        initial=initial,
    )


def _contrast(candidate: dict, baseline: dict) -> dict:
    """Positive drawdown delta means the candidate's relative MaxDD is better."""
    return {
        "candidate": candidate["strategy"],
        "baseline": baseline["strategy"],
        "terminal_value_delta_eur": float(candidate["terminal_value"] - baseline["terminal_value"]),
        "terminal_relative_return_delta": float(
            candidate["terminal_relative_return"] - baseline["terminal_relative_return"]
        ),
        "cagr_delta": float(candidate["cagr"] - baseline["cagr"]),
        "cagr_excess_delta": float(candidate["cagr_excess"] - baseline["cagr_excess"]),
        "relative_max_drawdown_delta": float(
            candidate["relative_max_drawdown"] - baseline["relative_max_drawdown"]
        ),
        "trade_count_delta": int(candidate["trade_count"] - baseline["trade_count"]),
        "total_cost_delta_eur": float(candidate["total_cost_eur"] - baseline["total_cost_eur"]),
    }


def _pair_diagnosis(candidate: dict, baseline: dict, label: str) -> str:
    terminal_equal = math.isclose(
        candidate["terminal_value"], baseline["terminal_value"], rel_tol=1e-12, abs_tol=1e-9
    )
    risk_equal = math.isclose(
        candidate["relative_max_drawdown"], baseline["relative_max_drawdown"], rel_tol=1e-12, abs_tol=1e-9
    )
    if terminal_equal and risk_equal:
        return f"{label}_MATCH"
    return_better = candidate["terminal_value"] > baseline["terminal_value"]
    risk_better = candidate["relative_max_drawdown"] > baseline["relative_max_drawdown"]
    return_worse = candidate["terminal_value"] < baseline["terminal_value"]
    risk_worse = candidate["relative_max_drawdown"] < baseline["relative_max_drawdown"]
    if return_better and risk_better:
        return f"{label}_PARETO_IMPROVES"
    if return_worse and risk_worse:
        return f"{label}_PARETO_WORSENS"
    return f"{label}_TRADEOFF"


def _contract_audit(
    *,
    plans: dict[str, list[dict]],
    sources: dict,
    schedules: dict[str, pd.DataFrame],
    calibration_audits: dict[str, list[dict]],
    assessment_count: int,
) -> dict:
    first_keys = {arm: plans[arm][0]["source_key"] for arm in ARMS}
    initial_ids = {
        arm: str(sources[first_keys[arm]]["build"]["model_artifact_id"])
        for arm in ARMS
    }
    same_initial_model = len(set(initial_ids.values())) == 1
    frozen_recipe_keys = {
        arm: {recipe_key(row["choice"]) for row in plans[arm]}
        for arm in ARMS
    }
    one_recipe_per_arm = all(len(keys) == 1 for keys in frozen_recipe_keys.values())
    same_recipe_all_arms = len({next(iter(keys)) for keys in frozen_recipe_keys.values()}) == 1

    a_threshold = float(schedules[ARM_A].iloc[0]["resolved_threshold"])
    b_threshold = float(schedules[ARM_B].iloc[0]["resolved_threshold"])
    c_threshold = float(schedules[ARM_C].iloc[0]["resolved_threshold"])
    initial_thresholds_match = (
        math.isclose(a_threshold, b_threshold, rel_tol=1e-12, abs_tol=1e-12)
        and math.isclose(a_threshold, c_threshold, rel_tol=1e-12, abs_tol=1e-12)
    )

    checks = {
        "same_initial_model_artifact_all_arms": same_initial_model,
        "same_exact_recipe_all_months_and_arms": one_recipe_per_arm and same_recipe_all_arms,
        "initial_threshold_matches_all_arms": initial_thresholds_match,
        "a_schedule_has_one_frozen_generation": len(schedules[ARM_A]) == 1,
        "b_schedule_matches_assessment_count": len(schedules[ARM_B]) == assessment_count,
        "c_schedule_matches_assessment_count": len(schedules[ARM_C]) == assessment_count,
        "a_calibration_resolved_once": sum(
            bool(x["calibration_resolution_performed"])
            for x in calibration_audits[ARM_A]
        ) == 1,
        "b_uses_one_model_artifact": schedules[ARM_B]["model_artifact_id"].nunique() == 1,
        "c_fresh_model_each_assessment": schedules[ARM_C]["model_artifact_id"].nunique() == assessment_count,
    }
    failed = sorted(key for key, value in checks.items() if not value)
    if failed:
        raise AssertionError(f"ABC_RECALIBRATION_CONTRACT_FAILED:{failed}")
    return {
        "checks": checks,
        "initial_model_artifact_ids": initial_ids,
        "initial_thresholds": {
            ARM_A: a_threshold,
            ARM_B: b_threshold,
            ARM_C: c_threshold,
        },
        "assessment_count": assessment_count,
    }


def run_experiment(
    *,
    signal_panel: str | Path,
    candidate_metrics: str | Path,
    daily_store_root: str | Path,
    output_root: str | Path,
    benchmark_daily_path: str | Path | None = None,
    direct_daily_stock_root: str | Path | None = None,
    start: date = date(2020, 8, 31),
    end: date = date(2023, 12, 29),
    horizon: int = 3,
    holding_days: int = 2,
    max_names: int = 1,
    score_quantile: float = 0.75,
    top_fraction: float = 0.01,
    initial: float = 10000.0,
    code_commit: str = "UNSPECIFIED_LOCAL_WORKTREE",
) -> dict:
    if end >= CLOSED_HOLDOUT_START:
        raise ValueError("ABC_RECALIBRATION_FINAL_HOLDOUT_MUST_REMAIN_CLOSED")
    if start > end:
        raise ValueError("ABC_RECALIBRATION_START_AFTER_END")

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    signal_panel = Path(signal_panel)
    metrics_path = Path(candidate_metrics)
    metric_rows = json.loads(metrics_path.read_text(encoding="utf-8"))

    base_materializer = ParquetH130DatasetMaterializer(signal_panel, metrics_path, end)
    base = next(
        x for x in build_family_specs(
            feature_schema_sha256=base_materializer.feature_schema_fingerprint,
            score_quantile=score_quantile,
            top_fraction=top_fraction,
        )
        if x.horizon_sessions == horizon
        and x.holding_days == holding_days
        and x.max_names == max_names
        and x.exit_policy["family"] == "FIXED"
    )
    sessions_frame = pd.read_parquet(signal_panel, columns=["decision_date"]).drop_duplicates()
    sessions = tuple(sorted(pd.to_datetime(sessions_frame["decision_date"]).dt.date.unique()))

    assessments = _causal_recipe_assessments(
        family=base,
        materializer=base_materializer,
        metrics_path=metrics_path,
        sessions=sessions,
        end=end,
        root=output_root / "assessment-audit",
    )
    assessments = [row for row in assessments if row["assessment_date"] <= end]
    if not assessments:
        raise ValueError("NO_ABC_RECALIBRATION_ASSESSMENTS")

    plans, plan_audit = _build_plans(assessments)
    pd.DataFrame(plan_audit).to_csv(output_root / "abc-plan-audit.csv", index=False)

    sources, fit_audit, worker_telemetry = _parallel_fit_required_sources(
        plans=plans,
        base=base,
        metric_rows=metric_rows,
        signal_panel=signal_panel,
        sessions=sessions,
        output_root=output_root,
        end=end,
        horizon=horizon,
    )
    _bind_sources(plans, sources)
    (output_root / "fit-audit.json").write_text(
        json.dumps(fit_audit, default=str, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    a_signals, a_schedule, a_calibration_audit, a_identity = _frozen_a_schedule(
        plan=plans[ARM_A], sources=sources, sessions=sessions
    )
    b_signals, b_schedule, b_calibration_audit = _arm_signals_schedule(
        arm=ARM_B, plan=plans[ARM_B], sources=sources, sessions=sessions
    )
    c_signals, c_schedule, c_calibration_audit = _arm_signals_schedule(
        arm=ARM_C, plan=plans[ARM_C], sources=sources, sessions=sessions
    )
    schedules = {ARM_A: a_schedule, ARM_B: b_schedule, ARM_C: c_schedule}
    calibration_audits = {
        ARM_A: a_calibration_audit,
        ARM_B: b_calibration_audit,
        ARM_C: c_calibration_audit,
    }

    contract = _contract_audit(
        plans=plans,
        sources=sources,
        schedules=schedules,
        calibration_audits=calibration_audits,
        assessment_count=len(assessments),
    )
    contract["a_frozen_calibration_identity"] = a_identity
    (output_root / "contract-audit.json").write_text(
        json.dumps(contract, default=str, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    first_activation = plans[ARM_A][0]["activation_cutoff"]
    evaluation_start = max(start, first_activation)
    prices_path = materialize_daily_store_prices(
        daily_store_root=daily_store_root,
        signal_panel=signal_panel,
        start=first_activation,
        end=end,
        output_path=output_root / "inputs" / "prices.parquet",
        benchmark_daily_path=benchmark_daily_path,
        direct_daily_stock_root=direct_daily_stock_root,
    )
    prices = pd.read_parquet(prices_path)

    signals = {ARM_A: a_signals, ARM_B: b_signals, ARM_C: c_signals}
    rows: list[dict] = []
    replay_results: dict[str, dict] = {}
    for arm in ARMS:
        result = _replay(
            arm=arm,
            signals=signals[arm],
            schedule=schedules[arm],
            prices=prices,
            start=evaluation_start,
            end=end,
            horizon=horizon,
            holding_days=holding_days,
            max_names=max_names,
            top_fraction=top_fraction,
            initial=initial,
        )
        replay_results[arm] = result
        if arm == ARM_A:
            calibration_resolution_count = 1
            rolling_recalibration_count = 0
            frozen_reuse = len(assessments) - 1
        elif arm == ARM_B:
            calibration_resolution_count = len(assessments)
            rolling_recalibration_count = max(0, len(assessments) - 1)
            frozen_reuse = 0
        else:
            calibration_resolution_count = len(assessments)
            rolling_recalibration_count = 0
            frozen_reuse = 0
        rows.append(_result_row(
            arm=arm,
            result=result,
            fresh_fit_count=sum(bool(x["refit_required"]) for x in plans[arm]),
            calibration_resolution_count=calibration_resolution_count,
            rolling_recalibration_count=rolling_recalibration_count,
            frozen_calibration_reuse_count=frozen_reuse,
        ))

        curve = result["curve"].copy()
        curve["strategy"] = arm
        curve.to_parquet(output_root / f"{arm}-nav.parquet", index=False)
        pd.DataFrame(result["trades"]).to_parquet(
            output_root / f"{arm}-trades.parquet", index=False
        )
        schedules[arm].to_csv(output_root / f"{arm}-schedule.csv", index=False)

    calibration_frame = pd.concat(
        [pd.DataFrame(calibration_audits[arm]) for arm in ARMS],
        ignore_index=True,
    )
    calibration_frame.to_csv(output_root / "calibration-audit.csv", index=False)

    comparison = pd.DataFrame(rows).sort_values("terminal_value", ascending=False)
    comparison.to_csv(output_root / "portfolio-value-comparison.csv", index=False)
    by_name = {row["strategy"]: row for row in rows}
    b_minus_a = _contrast(by_name[ARM_B], by_name[ARM_A])
    c_minus_b = _contrast(by_name[ARM_C], by_name[ARM_B])
    contrasts = {
        "B_MINUS_A_ROLLING_RECALIBRATION_EFFECT": b_minus_a,
        "C_MINUS_B_FRESH_REFIT_EFFECT": c_minus_b,
    }
    (output_root / "causal-contrasts.json").write_text(
        json.dumps(contrasts, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    diagnosis = {
        "rolling_recalibration": _pair_diagnosis(
            by_name[ARM_B], by_name[ARM_A], "B_MINUS_A_ROLLING_RECALIBRATION"
        ),
        "fresh_refit_given_monthly_calibration": _pair_diagnosis(
            by_name[ARM_C], by_name[ARM_B], "C_MINUS_B_FRESH_REFIT"
        ),
    }
    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": "COMPLETE",
        "authority": AUTHORITY,
        "code_commit": code_commit,
        "promotion_allowed": False,
        "final_holdout_opened": False,
        "recipe_parameter_grid_searched": False,
        "experiment": {
            "horizon": horizon,
            "holding_days": holding_days,
            "max_names": max_names,
            "score_quantile": score_quantile,
            "top_fraction": top_fraction,
            "initial_value_eur": initial,
            "evaluation_start": evaluation_start.isoformat(),
            "evaluation_end": end.isoformat(),
        },
        "arm_contract": {
            ARM_A: "INITIAL_MODEL_AND_INITIAL_CALIBRATION_FROZEN_FOR_FULL_REPLAY",
            ARM_B: "SAME_INITIAL_MODEL_MONTHLY_ROLLING_RECALIBRATION_NO_REFIT",
            ARM_C: "SAME_EXACT_FROZEN_RECIPE_FRESHLY_REFIT_AND_CALIBRATED_EACH_MONTH",
        },
        "primary_contrasts": {
            "B_MINUS_A": "PURE_ROLLING_RECALIBRATION_EFFECT",
            "C_MINUS_B": "FRESH_REFIT_EFFECT_CONDITIONAL_ON_MONTHLY_CALIBRATION",
        },
        "assessment_count": len(assessments),
        "contract_audit": contract,
        "diagnosis": diagnosis,
        "causal_contrasts": contrasts,
        "portfolio_value_comparison": comparison.to_dict(orient="records"),
        "parallel_execution": worker_telemetry,
        "execution_resources": active_cpu_contract(),
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, default=str, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    report = [
        "# Dynamic-QBD matched A/B/C recalibration decomposition",
        "",
        f"Rolling recalibration diagnosis: **{diagnosis['rolling_recalibration']}**",
        f"Fresh-refit diagnosis: **{diagnosis['fresh_refit_given_monthly_calibration']}**",
        "",
        "Authority: `SHADOW_ONLY_NO_PROMOTION_NO_HOLDOUT`. The final holdout remains closed.",
        "",
        "## Scientific contract",
        "",
        f"- `{ARM_A}`: one initial model fit and one initial calibration; both remain frozen.",
        f"- `{ARM_B}`: exactly the same initial model, but threshold/top-fraction are recalibrated monthly.",
        f"- `{ARM_C}`: exactly the same frozen recipe is freshly fit and calibrated every monthly assessment.",
        "- Recipe identity is fixed across every arm and every assessment.",
        "- Initial model artifact and initial threshold must match across A/B/C; the runner fails closed otherwise.",
        "- Same replay inputs, portfolio policy, costs, benchmark and evaluation dates.",
        "- No recipe switching, grid search, evaluation-window tuning, promotion or holdout access.",
        "",
        "## Portfolio comparison",
        "",
        comparison_markdown(comparison),
        "",
        "## Causal contrasts",
        "",
        f"- B-A terminal delta: {b_minus_a['terminal_value_delta_eur']:+.2f} EUR; CAGR delta: {b_minus_a['cagr_delta']:+.4%}; relative-MaxDD delta: {b_minus_a['relative_max_drawdown_delta']:+.4%}.",
        f"- C-B terminal delta: {c_minus_b['terminal_value_delta_eur']:+.2f} EUR; CAGR delta: {c_minus_b['cagr_delta']:+.4%}; relative-MaxDD delta: {c_minus_b['relative_max_drawdown_delta']:+.4%}.",
        "",
        "Positive relative-MaxDD delta means the candidate has a less severe relative drawdown.",
        "",
        "Exact provenance is persisted in `contract-audit.json`, `fit-audit.json`, `calibration-audit.csv`, each arm schedule/NAV/trades, and `causal-contrasts.json`.",
        "",
        "This is Development/Research evidence only.",
    ]
    (output_root / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return summary


def _parse_date(value: str) -> date:
    return date.fromisoformat(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--signal-panel", required=True)
    parser.add_argument("--candidate-metrics", required=True)
    parser.add_argument("--daily-store-root", required=True)
    parser.add_argument("--benchmark-daily-path")
    parser.add_argument("--direct-daily-stock-root")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--start", type=_parse_date, default=date(2020, 8, 31))
    parser.add_argument("--end", type=_parse_date, default=date(2023, 12, 29))
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--holding-days", type=int, default=2)
    parser.add_argument("--max-names", type=int, default=1)
    parser.add_argument("--score-quantile", type=float, default=0.75)
    parser.add_argument("--top-fraction", type=float, default=0.01)
    parser.add_argument("--initial", type=float, default=10000.0)
    parser.add_argument("--code-commit", default="UNSPECIFIED_LOCAL_WORKTREE")
    args = parser.parse_args()

    configure_cpu_peak(
        process_workers=ABC_FIT_WORKERS,
        native_threads_per_worker=1,
    )
    result = run_experiment(
        signal_panel=args.signal_panel,
        candidate_metrics=args.candidate_metrics,
        daily_store_root=args.daily_store_root,
        output_root=args.output_root,
        benchmark_daily_path=args.benchmark_daily_path,
        direct_daily_stock_root=args.direct_daily_stock_root,
        start=args.start,
        end=args.end,
        horizon=args.horizon,
        holding_days=args.holding_days,
        max_names=args.max_names,
        score_quantile=args.score_quantile,
        top_fraction=args.top_fraction,
        initial=args.initial,
        code_commit=args.code_commit,
    )
    print(json.dumps(result["portfolio_value_comparison"], indent=2))


if __name__ == "__main__":
    main()
