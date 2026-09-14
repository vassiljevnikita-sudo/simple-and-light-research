"""Development 2016-2025 H/D/N/exit surface validation of Dynamic-QBD Fold-Clock refitting.

The experiment is deliberately narrower than the production Dynamic-QBD
factory. It asks whether one already-selected entry recipe should be refreshed
on independent OOS-fold evidence rather than on calendar month ends.

For every H1-H30 horizon, all D1-D_H, N1-N6 and FIXED/LEARNED_EXIT portfolio
families are replayed under three matched entry-model arms:

A -- first causal entry model and calibration frozen forever;
F -- same exact entry recipe refit/recalibrated only when its matured OOS fold
     content strictly expands; model and calibration remain frozen between;
M -- same exact entry recipe freshly refit/recalibrated every assessment.

Learned-exit predictors are causally fitted once at the horizon's first
activation and then matched/frozen across A/F/M. This isolates entry-refit
frequency while still testing whether the fold-clock effect survives both exit
modes. No recipe switching, performance trigger, grid search, promotion,
capital authority or final-holdout access is permitted.
"""
from __future__ import annotations

# The Dynamic-QBD surface installs the Windows Job Object at import time. These
# defaults therefore have to be present before importing any suite module.
import os
os.environ.setdefault("DYNAMIC_QBD_MEMORY_LIMIT_GB", "90")
os.environ.setdefault("DYNAMIC_QBD_MEMORY_SOFT_TARGET_GB", "84")

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import date
import hashlib
import json
import math
import multiprocessing as mp
from pathlib import Path
import re
import threading
import time
from typing import Any, Iterable, Mapping

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .contract_fingerprints import stable_hash
from .cost_contracts import CostModel
from .portfolio_policy_contracts import Policy
from .tax_contracts import TaxConfig
from .dynamic_qbd_factory import monthly_refit_dates
from .dynamic_qbd_h1_30_adapter import (
    FEATURE_COLUMNS,
    H130ProductionGenerationBuilder,
    ParquetH130DatasetMaterializer,
    _robust_statistics,
)
from .dynamic_qbd_maturity import HorizonMaturityResolver
from .dynamic_qbd_development_pipeline import materialize_daily_store_prices
from .dynamic_qbd_portfolio_replay import replay_family
from .dynamic_qbd_generation_recalibration import recalibrate_generation
from .dynamic_qbd_runtime_resources import active_cpu_contract, configure_cpu_peak, cpu_budget
from .dynamic_qbd_runtime_telemetry import NWInfoSampler
from .cpu_topology import physical_core_affinity_plan, set_current_process_affinity
from .dynamic_qbd_family_surface import (
    build_family_specs,
    structural_plateau_id_from_family_id,
)
from .dynamic_qbd_wealth_metrics import wealth_path_metrics
from .learned_exit_qbd_profit import ProfitTaxConfig
from .learned_exit_qbd_provider import LearnedExitProvider
from .learned_exit_qbd_replay import configure_replay


AUTHORITY = "SHADOW_ONLY_NO_PROMOTION_NO_HOLDOUT"
CLOSED_HOLDOUT_START = date(2026, 7, 25)
HOLDOUT_CONTRACT = "PROSPECTIVE_FROM_2026_07_25"
SCHEMA_VERSION = "DYNAMIC_QBD_FOLD_CLOCK_SURFACE_VALIDATION_2016_2025_V1"
LEGACY_SCHEMA_VERSIONS = frozenset({"DYNAMIC_QBD_FULL_SPACE_FOLD_CLOCK_VALIDATION_V1"})
ARM_A = "A_FROZEN_MODEL_FROZEN_CALIBRATION"
ARM_F = "F_FOLD_CLOCK_REFIT_FROZEN_BETWEEN_FOLDS"
ARM_M = "M_MONTHLY_REFIT_FROZEN_RECIPE"
ARMS = (ARM_A, ARM_F, ARM_M)

# Predeclared before observing Fold-Clock surface results. These are validation gates,
# never promotion/capital gates.
GATE_PLATEAU_DIRECTION_FRACTION = 0.60
GATE_TEMPORAL_DIRECTION_FRACTION = 0.60
GATE_BOOTSTRAP_DRAWS = 5000
GATE_BOOTSTRAP_SEED = 1701

_FOLD_END = re.compile(r"_(\d{4}-\d{2}-\d{2})$")
_TELEMETRY_LOCK = threading.Lock()
# The entry and replay workloads are Python-heavy. Isolated spawned processes
# provide independent CPU execution lanes; native numeric libraries stay at one
# thread per process. Twenty-four lanes fit the physical-core-first plan on the
# current workstation while leaving logical capacity for the coordinator/OS.
DEFAULT_HORIZON_PROCESS_WORKERS = 24
DEFAULT_ENTRY_PROCESS_WORKERS = 24

_ENTRY_WORKER_INDEX = -1
_ENTRY_WORKER_LOGICAL_PROCESSOR: int | None = None
_ENTRY_WORKER_LAST_FINISH = 0.0


def _entry_process_initializer(worker_map: list[dict], slot_counter) -> None:
    """Bind one entry-fit process to a physical-core-first execution lane."""
    global _ENTRY_WORKER_INDEX, _ENTRY_WORKER_LOGICAL_PROCESSOR, _ENTRY_WORKER_LAST_FINISH
    for name in (
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS",
    ):
        os.environ[name] = "1"
    with slot_counter.get_lock():
        slot = int(slot_counter.value)
        slot_counter.value += 1
    _ENTRY_WORKER_INDEX = slot
    if worker_map:
        row = worker_map[slot % len(worker_map)]
        _ENTRY_WORKER_LOGICAL_PROCESSOR = int(row["logical_processor"])
        set_current_process_affinity([_ENTRY_WORKER_LOGICAL_PROCESSOR])
    _ENTRY_WORKER_LAST_FINISH = time.perf_counter()


def _entry_process_worker(payload: dict[str, Any]) -> dict[str, Any]:
    """Run one complete horizon entry lane in a spawned process."""
    global _ENTRY_WORKER_LAST_FINISH
    started = time.perf_counter()
    idle_seconds = max(0.0, started - _ENTRY_WORKER_LAST_FINISH)
    affinity_pinned = False
    if _ENTRY_WORKER_LOGICAL_PROCESSOR is not None:
        affinity_pinned = bool(
            set_current_process_affinity([_ENTRY_WORKER_LOGICAL_PROCESSOR])
        )
    audit = _entry_fit_lane(**payload)
    finished = time.perf_counter()
    _ENTRY_WORKER_LAST_FINISH = finished
    return {
        "horizon": int(payload["horizon"]),
        "audit": audit,
        "telemetry": {
            "event": "entry_horizon_lane",
            "horizon": int(payload["horizon"]),
            "worker_index": int(_ENTRY_WORKER_INDEX),
            "worker_pid": int(os.getpid()),
            "logical_processor": _ENTRY_WORKER_LOGICAL_PROCESSOR,
            "affinity_pinned": affinity_pinned,
            "idle_seconds_before_job": idle_seconds,
            "compute_seconds": max(0.0, finished - started),
            "fit_count": len(audit),
            "status": "COMPLETE",
        },
    }


def _fold_evidence_fingerprint(fold_ids: Iterable[str]) -> str:
    """Hash only the sorted matured fold content, never calendar progress."""
    raw = tuple(str(value) for value in fold_ids)
    normalized = tuple(sorted(set(raw)))
    if len(normalized) != len(raw):
        raise AssertionError("FOLD_CLOCK_SURFACE_DUPLICATE_FOLD_IDS")
    return stable_hash({
        "schema_version": "FOLD_CLOCK_SURFACE_FOLD_CONTENT_FINGERPRINT_V1",
        "fold_count": len(normalized),
        "fold_ids": list(normalized),
    })


def _fold_evidence_expanded(
    previous_fold_ids: tuple[str, ...] | None,
    current_fold_ids: tuple[str, ...],
) -> bool:
    """Return true only for a strict superset of the prior fold content."""
    current = tuple(sorted(str(value) for value in current_fold_ids))
    if len(set(current)) != len(current):
        raise AssertionError("FOLD_CLOCK_SURFACE_DUPLICATE_FOLD_IDS")
    if previous_fold_ids is None:
        return True
    previous = tuple(sorted(str(value) for value in previous_fold_ids))
    if current == previous:
        return False
    if set(previous).issubset(set(current)) and len(current) > len(previous):
        return True
    raise AssertionError("FOLD_CLOCK_SURFACE_NON_MONOTONIC_FOLD_EVIDENCE")


def _holdout_marker_audit(
    sessions_frame: pd.DataFrame, *, start: date, end: date
) -> dict[str, Any]:
    """Audit legacy markers without treating pre-boundary rows as open holdout."""
    dates = pd.to_datetime(sessions_frame["decision_date"]).dt.date
    locked = sessions_frame["holdout_locked"].fillna(False).astype(bool)
    legacy = int((locked & dates.lt(CLOSED_HOLDOUT_START)).sum())
    prospective = int((locked & dates.ge(CLOSED_HOLDOUT_START)).sum())
    if end >= CLOSED_HOLDOUT_START:
        raise PermissionError("FOLD_CLOCK_SURFACE_FINAL_HOLDOUT_REQUESTED")
    if prospective:
        raise PermissionError("FOLD_CLOCK_SURFACE_PROSPECTIVE_HOLDOUT_MARKER_PRESENT")
    development_dates = dates.loc[dates.between(start, end)].drop_duplicates()
    return {
        "contract": HOLDOUT_CONTRACT,
        "boundary": CLOSED_HOLDOUT_START.isoformat(),
        "historical_pre_boundary_marker_count": legacy,
        "post_boundary_marker_count": prospective,
        "development_session_count": int(len(development_dates)),
        "final_holdout_opened": False,
        "promotion_allowed": False,
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, default=str, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _append_telemetry(path: Path, payload: dict[str, Any]) -> None:
    payload = {"timestamp": time.time(), **payload}
    with _TELEMETRY_LOCK:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, default=str, sort_keys=True) + "\n")


def _choice_identity(choice: Mapping[str, Any]) -> tuple[str, str, str]:
    return (
        str(choice["candidate_id"]),
        str(choice["family"]),
        stable_hash(dict(choice.get("parameters", {}))),
    )


def _candidate_allowed(base, row: Mapping[str, Any]) -> bool:
    requested = str(base.model_family).upper()
    actual = str(row.get("family", "")).upper()
    if requested not in {"RIDGE_HGB_FROZEN_RULE", "RIDGE_HGB"} and actual != requested:
        return False
    allowed = tuple(
        str(x).upper()
        for x in base.hyperparameter_rule.get("candidate_models", ("RIDGE", "HGB"))
    )
    normalized = "HGB" if actual == "HIST_GRADIENT_BOOSTING" else actual
    return normalized in allowed


def _choices_from_metric_rows(
    *,
    metric_rows: list[dict],
    base,
    horizon: int,
    latest_matured: date,
) -> tuple[dict, ...]:
    grouped: dict[tuple[str, str, str], list[dict]] = {}
    metadata: dict[tuple[str, str, str], dict] = {}
    for row in metric_rows:
        if int(row.get("horizon_sessions", -1)) != int(horizon):
            continue
        if row.get("selection_only") or not _candidate_allowed(base, row):
            continue
        match = _FOLD_END.search(str(row.get("fold_id", "")))
        if match is None or date.fromisoformat(match.group(1)) > latest_matured:
            continue
        key = (
            str(row["candidate_id"]),
            str(row["family"]),
            stable_hash(dict(row.get("parameters", {}))),
        )
        grouped.setdefault(key, []).append(row)
        metadata[key] = row
    minimum_folds = int(base.hyperparameter_rule.get("minimum_oos_folds", 2))
    choices: list[dict] = []
    for key, evidence in grouped.items():
        if len(evidence) < minimum_folds:
            continue
        row = metadata[key]
        choices.append({
            "candidate_id": str(row["candidate_id"]),
            "family": str(row["family"]),
            "parameters": dict(row["parameters"]),
            "fold_count": len(evidence),
            "fold_ids": tuple(sorted(str(x["fold_id"]) for x in evidence)),
            "selection_metric_contract": "ROBUST_FOLD_SPEARMAN_THEN_MAE_TIEBREAK_R2_DIAGNOSTIC_ONLY",
            "model_combination_contract": "SINGLE_SELECTED_RECIPE_NO_ENSEMBLE_NO_MODEL_INTERSECTION",
            **_robust_statistics(evidence),
        })
    return tuple(sorted(choices, key=lambda x: _choice_identity(x)))


def _select_recipe(choices: Iterable[dict]) -> dict:
    values = list(choices)
    if not values:
        raise ValueError("FOLD_CLOCK_SURFACE_NO_CAUSAL_RECIPE_CHOICES")
    return dict(max(
        values,
        key=lambda x: (
            float(x["robust_score"]),
            float(x["median_spearman"]),
            -float(x["mean_mae"]),
            str(x["candidate_id"]),
        ),
    ))


def _current_frozen_choice(choices: Iterable[dict], frozen_identity) -> dict:
    match = next((dict(x) for x in choices if _choice_identity(x) == frozen_identity), None)
    if match is None:
        raise ValueError(f"FOLD_CLOCK_SURFACE_FROZEN_RECIPE_NOT_CAUSALLY_AVAILABLE:{frozen_identity!r}")
    return match


def _build_horizon_plan(
    *,
    horizon: int,
    base,
    metric_rows: list[dict],
    sessions: tuple[date, ...],
    start: date,
    end: date,
) -> list[dict]:
    maturity = HorizonMaturityResolver(sessions)
    cutoffs = tuple(
        x for x in monthly_refit_dates(tuple(d for d in sessions if d <= end))
        if start <= x <= end
    )
    rows: list[dict] = []
    frozen_identity = None
    frozen_recipe = None
    previous_fold_ids: tuple[str, ...] | None = None
    for cutoff in cutoffs:
        latest = maturity.latest_matured_decision(cutoff, horizon)
        if latest is None:
            continue
        choices = _choices_from_metric_rows(
            metric_rows=metric_rows,
            base=base,
            horizon=horizon,
            latest_matured=latest,
        )
        if not choices:
            continue
        if frozen_identity is None:
            frozen_recipe = _select_recipe(choices)
            frozen_identity = _choice_identity(frozen_recipe)
        current = _current_frozen_choice(choices, frozen_identity)
        fold_ids = tuple(sorted(str(x) for x in current.get("fold_ids", ())))
        if int(current["fold_count"]) != len(fold_ids) or len(set(fold_ids)) != len(fold_ids):
            raise AssertionError(f"FOLD_CLOCK_SURFACE_FOLD_EVIDENCE_INVALID:H{horizon}")
        try:
            expanded = _fold_evidence_expanded(previous_fold_ids, fold_ids)
        except AssertionError as exc:
            raise AssertionError(f"{exc}:H{horizon}") from exc
        rows.append({
            "horizon": int(horizon),
            "assessment_date": cutoff,
            "latest_matured_evidence_date": latest,
            "choice": current,
            "frozen_recipe": dict(frozen_recipe),
            "fold_count": len(fold_ids),
            "fold_ids": fold_ids,
            "evidence_fingerprint": _fold_evidence_fingerprint(fold_ids),
            "evidence_expanded": bool(expanded),
            "f_refit": bool(not rows or expanded),
        })
        previous_fold_ids = fold_ids
    if not rows:
        raise ValueError(f"FOLD_CLOCK_SURFACE_NO_CAUSAL_ASSESSMENTS:H{horizon}")

    f_indices = [i for i, row in enumerate(rows) if row["f_refit"]]
    for index, row in enumerate(rows):
        if index == 0:
            exclusive = None  # A needs the first model through the full horizon replay.
        elif row["f_refit"]:
            later_f = next((j for j in f_indices if j > index), None)
            exclusive = rows[later_f]["assessment_date"] if later_f is not None else None
        else:
            exclusive = rows[index + 1]["assessment_date"] if index + 1 < len(rows) else None
        row["prediction_end_exclusive"] = exclusive
        row["prediction_end"] = end
    return rows


def _entry_fit_lane(
    *,
    horizon: int,
    base,
    plan: list[dict],
    signal_panel: Path,
    candidate_metrics: Path,
    output_root: Path,
    end: date,
    sessions: tuple[date, ...],
    code_commit: str,
) -> list[dict]:
    materializer = ParquetH130DatasetMaterializer(signal_panel, candidate_metrics, end)
    builder = H130ProductionGenerationBuilder(
        materializer=materializer,
        root=output_root / "entry-models" / f"H{horizon:02d}",
        code_commit=code_commit,
    )
    first = plan[0]
    frozen = dict(first["frozen_recipe"])
    frozen["recipe_frozen_at"] = first["assessment_date"].isoformat()
    frozen["selection_contract"] = "FIRST_CAUSAL_REFIT_THEN_FROZEN_WITHIN_FAMILY"
    builder._frozen_choice_by_family[base.family_id] = (first["assessment_date"], frozen)
    maturity = HorizonMaturityResolver(sessions)
    audit: list[dict] = []
    for index, row in enumerate(plan):
        cutoff = row["assessment_date"]
        kwargs: dict[str, Any] = {}
        if row["prediction_end_exclusive"] is not None:
            kwargs["prediction_end_exclusive"] = row["prediction_end_exclusive"]
        else:
            kwargs["prediction_end"] = end
        build = builder.build(
            family=base,
            information_cutoff=cutoff,
            latest_matured_label_cutoff=row["latest_matured_evidence_date"],
            **kwargs,
        )
        if _choice_identity(build["selection_evidence"]) != _choice_identity(frozen):
            raise AssertionError(f"FOLD_CLOCK_SURFACE_ENTRY_RECIPE_DRIFT:H{horizon}:{cutoff}")
        generation_id = stable_hash({
            "schema_version": SCHEMA_VERSION,
            "horizon": horizon,
            "cutoff": cutoff,
            "model_artifact_id": build["model_artifact_id"],
            "recipe": _choice_identity(frozen),
        })[:24]
        calibration_frame = builder.calibration_predictions(
            family=base, build=build, information_cutoff=cutoff
        )
        resolved = recalibrate_generation(
            base,
            generation_id,
            calibration_frame,
            information_cutoff=cutoff,
            maturity=maturity,
        )
        finalized = builder.finalize_generation(
            family=base, build=build, generation_id=generation_id
        )
        audit.append({
            "horizon": horizon,
            "assessment_index": index,
            "assessment_date": cutoff.isoformat(),
            "latest_matured_evidence_date": row["latest_matured_evidence_date"].isoformat(),
            "fold_count": int(row["fold_count"]),
            "fold_ids": json.dumps(list(row["fold_ids"])),
            "evidence_fingerprint": row["evidence_fingerprint"],
            "evidence_expanded": bool(row["evidence_expanded"]),
            "f_refit": bool(row["f_refit"]),
            "candidate_id": str(frozen["candidate_id"]),
            "model_family": str(frozen["family"]),
            "parameters_json": json.dumps(dict(frozen["parameters"]), sort_keys=True),
            "generation_id": generation_id,
            "model_artifact_id": str(build["model_artifact_id"]),
            "model_artifact_sha256": str(build["model_artifact_sha256"]),
            "prediction_artifact_path": str(finalized["prediction_artifact_path"]),
            "prediction_artifact_sha256": str(finalized["prediction_artifact_sha256"]),
            "prediction_end_exclusive": (
                row["prediction_end_exclusive"].isoformat()
                if row["prediction_end_exclusive"] is not None else None
            ),
            "resolved_threshold": float(resolved.resolved_threshold),
            "resolved_top_fraction": float(resolved.resolved_top_fraction),
            "calibration_fingerprint": str(resolved.calibration_fingerprint),
            "calibration_observations": int(resolved.observations),
            "train_start": str(build["train_start"]),
            "train_end": str(build["train_end"]),
        })
    return audit


def _entry_cache_valid(path: Path, meta_path: Path, run_hashes: set[str]) -> bool:
    if not path.is_file() or not meta_path.is_file():
        return False
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("semantic_run_contract_hash") not in run_hashes:
            return False
        frame = pd.read_parquet(path)
        if len(frame) != int(meta.get("row_count", -1)):
            return False
        return all(Path(value).is_file() for value in frame["prediction_artifact_path"].astype(str))
    except Exception:
        return False


def _fit_all_entry_sources(
    *,
    plans: dict[int, list[dict]],
    representatives: dict[int, Any],
    signal_panel: Path,
    candidate_metrics: Path,
    output_root: Path,
    end: date,
    sessions: tuple[date, ...],
    code_commit: str,
    workers: int,
    run_hashes: set[str],
    telemetry_path: Path,
) -> pd.DataFrame:
    cache_path = output_root / "entry-fit-audit.parquet"
    meta_path = output_root / "entry-fit-cache.json"
    if _entry_cache_valid(cache_path, meta_path, run_hashes):
        print("[fold-clock-surface] entry-fit cache hit", flush=True)
        return pd.read_parquet(cache_path)
    rows: list[dict] = []
    lane_workers = max(1, min(int(workers), DEFAULT_ENTRY_PROCESS_WORKERS, len(plans)))
    entry_affinity_plan = physical_core_affinity_plan(
        lane_workers,
        reserve_logical_processors=max(0, int(os.cpu_count() or 1) - lane_workers),
    )
    worker_map = list(entry_affinity_plan.get("worker_map") or [])[:lane_workers]
    spawn_context = mp.get_context("spawn")
    slot_counter = spawn_context.Value("i", 0, lock=True)
    with ProcessPoolExecutor(
        max_workers=lane_workers,
        mp_context=spawn_context,
        initializer=_entry_process_initializer,
        initargs=(worker_map, slot_counter),
    ) as pool:
        futures = {
            pool.submit(
                _entry_process_worker,
                {
                    "horizon": h,
                    "base": representatives[h],
                    "plan": plans[h],
                    "signal_panel": signal_panel,
                    "candidate_metrics": candidate_metrics,
                    "output_root": output_root,
                    "end": end,
                    "sessions": sessions,
                    "code_commit": code_commit,
                },
            ): h
            for h in sorted(plans)
        }
        for future in as_completed(futures):
            horizon = futures[future]
            result = future.result()
            lane = result["audit"]
            rows.extend(lane)
            _append_telemetry(telemetry_path, result["telemetry"])
            print(
                f"[fold-clock-surface] entry lane H{horizon:02d} complete "
                f"({len(lane)} monthly fits)",
                flush=True,
            )
    frame = pd.DataFrame(rows).sort_values(["horizon", "assessment_date"]).reset_index(drop=True)
    temporary = cache_path.with_name(cache_path.name + f".{os.getpid()}.tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, cache_path)
    _write_json(meta_path, {
        "schema_version": "FOLD_CLOCK_SURFACE_ENTRY_FIT_CACHE_V1",
        "semantic_run_contract_hash": sorted(run_hashes)[-1],
        "row_count": len(frame),
    })
    return frame


def _schedule_from_audit(horizon_audit: pd.DataFrame, arm: str) -> pd.DataFrame:
    source = horizon_audit.sort_values("assessment_date").copy()
    if arm == ARM_A:
        source = source.iloc[:1]
    elif arm == ARM_F:
        source = source.loc[source["f_refit"].astype(bool)]
    elif arm != ARM_M:
        raise ValueError(f"FOLD_CLOCK_SURFACE_UNKNOWN_ARM:{arm}")
    return pd.DataFrame({
        "activation_date": pd.to_datetime(source["assessment_date"]),
        "family_id": [f"H{int(horizon_audit.iloc[0]['horizon']):02d}_SHARED_{arm}"] * len(source),
        "generation_id": source["generation_id"].astype(str).to_numpy(),
        "model_artifact_id": source["model_artifact_id"].astype(str).to_numpy(),
        "resolved_threshold": source["resolved_threshold"].astype(float).to_numpy(),
        "resolved_top_fraction": source["resolved_top_fraction"].astype(float).to_numpy(),
        "entry_policy_id": [f"{arm}_ENTRY_{x}" for x in source["assessment_date"].astype(str)],
        "exit_policy_id": ["FIXED_SHARED"] * len(source),
    })


def _build_contract_audit(
    *, entry_audit: pd.DataFrame, plans: dict[int, list[dict]],
    holdout_audit: dict[str, Any],
) -> dict[str, Any]:
    """Validate the matched A/F/M fit contract before replay is accepted."""
    checks: dict[str, bool] = {
        "final_holdout_opened": holdout_audit["final_holdout_opened"] is False,
        "promotion_allowed": holdout_audit["promotion_allowed"] is False,
        "same_initial_recipe_across_arms": True,
        "same_initial_model_artifact_across_arms": True,
        "same_initial_threshold_across_arms": True,
        "recipe_identity_constant": True,
        "A_fresh_fit_count_is_one": True,
        "F_refit_only_on_evidence_expansion": True,
        "F_refit_on_every_evidence_expansion": True,
        "F_calibration_resolution_matches_refits": True,
        "M_fresh_fit_count_matches_assessments": True,
        "all_horizons_have_identical_evaluation_dates": True,
        "no_nonfinite_fit_values": True,
    }
    for horizon, plan in sorted(plans.items()):
        frame = entry_audit.loc[entry_audit["horizon"].eq(horizon)].sort_values("assessment_date")
        if len(frame) != len(plan) or frame.empty:
            checks["all_horizons_have_identical_evaluation_dates"] = False
            continue
        expected_dates = [row["assessment_date"].isoformat() for row in plan]
        if frame["assessment_date"].astype(str).tolist() != expected_dates:
            checks["all_horizons_have_identical_evaluation_dates"] = False
        first = frame.iloc[0]
        schedules = {arm: _schedule_from_audit(frame, arm) for arm in ARMS}
        initial_values = {
            arm: tuple(schedules[arm].iloc[0][key] for key in (
                "model_artifact_id", "resolved_threshold", "resolved_top_fraction"
            )) for arm in ARMS
        }
        if len({value[0] for value in initial_values.values()}) != 1:
            checks["same_initial_model_artifact_across_arms"] = False
        if len({value[1] for value in initial_values.values()}) != 1:
            checks["same_initial_threshold_across_arms"] = False
        if int(frame["f_refit"].astype(bool).sum()) != sum(bool(row["f_refit"]) for row in plan):
            checks["F_refit_on_every_evidence_expansion"] = False
        for index, row in frame.iterrows():
            plan_row = plan[int(row["assessment_index"])]
            if bool(row["f_refit"]) != bool(plan_row["f_refit"]):
                checks["F_refit_only_on_evidence_expansion"] = False
            if not math.isfinite(float(row["resolved_threshold"])) or not math.isfinite(float(row["resolved_top_fraction"])):
                checks["no_nonfinite_fit_values"] = False
        if int(frame["f_refit"].astype(bool).sum()) < 1:
            checks["A_fresh_fit_count_is_one"] = False
        if frame["candidate_id"].nunique() != 1 or frame["model_family"].nunique() != 1:
            checks["recipe_identity_constant"] = False
            checks["same_initial_recipe_across_arms"] = False
        # A uses the first source row, F uses only fold events, M uses every row.
        if len(schedules[ARM_A]) != 1:
            checks["A_fresh_fit_count_is_one"] = False
        if len(schedules[ARM_F]) != int(frame["f_refit"].astype(bool).sum()):
            checks["F_calibration_resolution_matches_refits"] = False
        if len(schedules[ARM_M]) != len(plan):
            checks["M_fresh_fit_count_matches_assessments"] = False
    status = "PASS" if all(checks.values()) else "FAIL"
    result = {
        "schema_version": "FOLD_CLOCK_SURFACE_CONTRACT_AUDIT_V1",
        "status": status,
        "checks": checks,
        "holdout": holdout_audit,
    }
    if status != "PASS":
        raise AssertionError("FOLD_CLOCK_SURFACE_CONTRACT_AUDIT_FAILED:" + json.dumps(checks, sort_keys=True))
    return result


def _bounded_predictions(path: Path, cutoff: pd.Timestamp, exclusive: pd.Timestamp | None, end: date) -> pd.DataFrame:
    frame = pd.read_parquet(path, columns=["decision_date", "ticker", "score"])
    frame["decision_date"] = pd.to_datetime(frame["decision_date"]).dt.normalize()
    mask = frame["decision_date"].gt(pd.Timestamp(cutoff))
    if exclusive is None:
        mask &= frame["decision_date"].le(pd.Timestamp(end))
    else:
        mask &= frame["decision_date"].lt(pd.Timestamp(exclusive))
    return frame.loc[mask, ["decision_date", "ticker", "score"]].copy()


def _authoritative_signals(
    horizon_audit: pd.DataFrame,
    *,
    end: date,
    output_root: Path,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    source = horizon_audit.sort_values("assessment_date").reset_index(drop=True)
    h = int(source.iloc[0]["horizon"])
    signal_root = output_root / "authoritative-signals"
    signal_root.mkdir(parents=True, exist_ok=True)
    schedules = {arm: _schedule_from_audit(source, arm) for arm in ARMS}
    cached_paths = {arm: signal_root / f"H{h:02d}-{arm}.parquet" for arm in ARMS}
    if all(path.is_file() for path in cached_paths.values()):
        return (
            {arm: pd.read_parquet(path) for arm, path in cached_paths.items()},
            schedules,
        )

    a_parts: list[pd.DataFrame] = []
    f_parts: list[pd.DataFrame] = []
    m_parts: list[pd.DataFrame] = []
    f_rows = source.index[source["f_refit"].astype(bool)].tolist()
    for index, row in source.iterrows():
        cutoff = pd.Timestamp(row["assessment_date"])
        path = Path(row["prediction_artifact_path"])
        if index == 0:
            full = _bounded_predictions(path, cutoff, None, end)
            a_parts.append(full)
        next_month = (
            pd.Timestamp(source.iloc[index + 1]["assessment_date"])
            if index + 1 < len(source) else None
        )
        m_parts.append(_bounded_predictions(path, cutoff, next_month, end))
        if bool(row["f_refit"]):
            later = next((j for j in f_rows if j > index), None)
            next_f = pd.Timestamp(source.iloc[later]["assessment_date"]) if later is not None else None
            f_parts.append(_bounded_predictions(path, cutoff, next_f, end))

    signals = {
        ARM_A: pd.concat(a_parts, ignore_index=True) if a_parts else pd.DataFrame(),
        ARM_F: pd.concat(f_parts, ignore_index=True) if f_parts else pd.DataFrame(),
        ARM_M: pd.concat(m_parts, ignore_index=True) if m_parts else pd.DataFrame(),
    }
    for arm, frame in signals.items():
        if frame.empty:
            raise ValueError(f"FOLD_CLOCK_SURFACE_AUTHORITATIVE_SIGNALS_EMPTY:H{h}:{arm}")
        if frame.duplicated(["decision_date", "ticker"]).any():
            raise ValueError(f"FOLD_CLOCK_SURFACE_DUPLICATE_AUTHORITATIVE_SIGNAL:H{h}:{arm}")
        frame.sort_values(["decision_date", "score"], ascending=[True, False], inplace=True)
        frame.reset_index(drop=True, inplace=True)
        temporary = cached_paths[arm].with_name(cached_paths[arm].name + f".{os.getpid()}.tmp")
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, cached_paths[arm])
    return signals, schedules


def _potential_entry_signals(
    signals: pd.DataFrame,
    schedule: pd.DataFrame,
    *,
    max_names: int = 6,
) -> set[tuple[pd.Timestamp, str]]:
    schedule = schedule.sort_values("activation_date").copy()
    activations = pd.to_datetime(schedule["activation_date"]).to_numpy(dtype="datetime64[ns]")
    thresholds = schedule["resolved_threshold"].to_numpy(float)
    fractions = schedule["resolved_top_fraction"].to_numpy(float)
    result: set[tuple[pd.Timestamp, str]] = set()
    for decision_date, day in signals.groupby("decision_date", sort=True):
        d = np.datetime64(pd.Timestamp(decision_date).to_datetime64())
        idx = int(np.searchsorted(activations, d, side="right") - 1)
        if idx < 0:
            continue
        threshold = float(thresholds[idx])
        top_fraction = float(fractions[idx])
        eligible = day.loc[day["score"].ge(threshold)].sort_values(
            ["score", "ticker"], ascending=[False, True]
        )
        take = min(int(max_names), max(1, int(math.ceil(len(day) * top_fraction))))
        for row in eligible.head(take).itertuples(index=False):
            result.add((pd.Timestamp(decision_date).normalize(), str(row.ticker)))
    return result


def _required_exit_keys(
    *,
    horizon: int,
    signals: dict[str, pd.DataFrame],
    schedules: dict[str, pd.DataFrame],
    market_dates: tuple[pd.Timestamp, ...],
) -> dict[int, set[tuple[pd.Timestamp, str]]]:
    if horizon <= 1:
        return {}
    index = {pd.Timestamp(value).normalize(): i for i, value in enumerate(market_dates)}
    potential: set[tuple[pd.Timestamp, str]] = set()
    for arm in ARMS:
        potential.update(_potential_entry_signals(signals[arm], schedules[arm]))
    required = {remaining: set() for remaining in range(1, horizon)}
    for signal_date, ticker in potential:
        position = index.get(pd.Timestamp(signal_date).normalize())
        if position is None or position + 1 >= len(market_dates):
            continue
        buy_index = position + 1
        for remaining in range(1, horizon):
            max_offset = horizon - 1 - remaining
            for offset in range(max_offset + 1):
                decision_index = buy_index + offset
                if decision_index >= len(market_dates):
                    break
                required[remaining].add((
                    pd.Timestamp(market_dates[decision_index]).normalize(), ticker
                ))
    return required


def _exit_recipe_choice(
    *, metric_rows: list[dict], base, exit_horizon: int, latest_matured: date
) -> dict:
    choices = _choices_from_metric_rows(
        metric_rows=metric_rows,
        base=base,
        horizon=exit_horizon,
        latest_matured=latest_matured,
    )
    return _select_recipe(choices)


def _build_exit_provider(
    *,
    horizon: int,
    base,
    first_plan_row: dict,
    required: dict[int, set[tuple[pd.Timestamp, str]]],
    signal_panel: Path,
    exit_metric_rows: list[dict],
    output_root: Path,
    workers: int,
    telemetry_path: Path,
) -> tuple[Path | None, str, dict]:
    if horizon <= 1:
        return None, "", {"horizon": horizon, "required": False}
    root = output_root / "learned-exit-providers" / f"H{horizon:02d}"
    root.mkdir(parents=True, exist_ok=True)
    provider_path = root / "provider.parquet"
    audit_path = root / "audit.json"
    if provider_path.is_file() and audit_path.is_file():
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        return provider_path, str(audit["exit_generation_id"]), audit

    latest = first_plan_row["latest_matured_evidence_date"]
    exit_horizons = tuple(x for x in range(1, horizon) if required.get(x))
    choices = {
        e: _exit_recipe_choice(
            metric_rows=exit_metric_rows,
            base=base,
            exit_horizon=e,
            latest_matured=latest,
        )
        for e in exit_horizons
    }
    exit_generation_id = stable_hash({
        "schema_version": "FOLD_CLOCK_SURFACE_MATCHED_FROZEN_EXIT_PROVIDER_V1",
        "entry_horizon": horizon,
        "fit_cutoff": first_plan_row["assessment_date"],
        "latest_matured_evidence_date": latest,
        "choices": {str(e): _choice_identity(choice) for e, choice in choices.items()},
    })[:24]

    target_columns = [f"gross_excess_return_{e}" for e in exit_horizons]
    available = set(pq.ParquetFile(signal_panel).schema_arrow.names)
    columns = ["decision_date", "ticker", *FEATURE_COLUMNS, *target_columns]
    missing = set(columns) - available
    if missing:
        raise ValueError(f"FOLD_CLOCK_SURFACE_EXIT_PANEL_COLUMNS_MISSING:H{horizon}:{sorted(missing)}")
    panel = pd.read_parquet(signal_panel, columns=columns)
    panel["decision_date"] = pd.to_datetime(panel["decision_date"]).dt.normalize()
    if panel.duplicated(["decision_date", "ticker"]).any():
        raise ValueError(f"FOLD_CLOCK_SURFACE_EXIT_PANEL_DUPLICATE_KEYS:H{horizon}")
    all_dates = np.asarray(sorted(panel["decision_date"].unique()))
    matured_dates = np.asarray([
        x for x in all_dates if pd.Timestamp(x) <= pd.Timestamp(latest)
    ])
    calibration_count = int(base.calibration_window_sessions)
    purge = max(int(base.horizon_sessions), int(base.training_recipe.get("purge_sessions", 30)))
    train_end_index = len(matured_dates) - calibration_count - purge
    if train_end_index <= 0:
        raise ValueError(f"FOLD_CLOCK_SURFACE_EXIT_INSUFFICIENT_HISTORY:H{horizon}")
    train_dates = matured_dates[
        max(0, train_end_index - int(base.training_window_sessions)):train_end_index
    ]
    if len(train_dates) < int(base.training_window_sessions):
        raise ValueError(f"FOLD_CLOCK_SURFACE_EXIT_TRAIN_WINDOW_SHORT:H{horizon}:{len(train_dates)}")
    train = panel.loc[panel["decision_date"].isin(train_dates)]
    feature_matrix = train[list(FEATURE_COLUMNS)].to_numpy(float)
    panel_index = pd.MultiIndex.from_frame(panel[["decision_date", "ticker"]])

    def fit_exit(exit_horizon: int) -> tuple[int, pd.DataFrame, dict]:
        started = time.perf_counter()
        choice = choices[exit_horizon]
        model = H130ProductionGenerationBuilder._model(
            choice, int(base.random_seed) + int(exit_horizon)
        )
        target = f"gross_excess_return_{exit_horizon}"
        model.fit(feature_matrix, train[target].to_numpy(float))
        model_path = root / "models" / f"E{exit_horizon:02d}.joblib"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_path, compress=3)
        ordered_keys = sorted(required[exit_horizon], key=lambda x: (x[0], x[1]))
        requested_index = pd.MultiIndex.from_tuples(
            ordered_keys, names=["decision_date", "ticker"]
        )
        positions = panel_index.get_indexer(requested_index)
        valid_mask = positions >= 0
        valid_positions = positions[valid_mask]
        valid_keys = [key for key, valid in zip(ordered_keys, valid_mask) if valid]
        scoring = panel.iloc[valid_positions] if len(valid_positions) else panel.iloc[:0]
        values = (
            model.predict(scoring[list(FEATURE_COLUMNS)].to_numpy(float))
            if len(scoring) else np.asarray([], dtype=float)
        )
        part = pd.DataFrame({
            "decision_date": [x[0] for x in valid_keys],
            "ticker": [x[1] for x in valid_keys],
            "exit_horizon_sessions": exit_horizon,
            "predicted_continuation_excess": values,
            "holdout_locked": False,
            "exit_generation_id": exit_generation_id,
        })
        row = {
            "exit_horizon": exit_horizon,
            "candidate_id": str(choice["candidate_id"]),
            "model_family": str(choice["family"]),
            "parameters": dict(choice["parameters"]),
            "fold_count": int(choice["fold_count"]),
            "fold_ids": list(choice["fold_ids"]),
            "required_keys": len(ordered_keys),
            "predicted_keys": len(part),
            "panel_missing_keys": int(len(ordered_keys) - len(part)),
            "model_path": str(model_path),
            "model_sha256": _sha256_file(model_path),
        }
        _append_telemetry(telemetry_path, {
            "event": "exit_model_fit",
            "horizon": horizon,
            "exit_horizon": exit_horizon,
            "thread_id": threading.get_ident(),
            "compute_seconds": time.perf_counter() - started,
            "status": "COMPLETE",
        })
        return exit_horizon, part, row

    parts: list[pd.DataFrame] = []
    audit_rows: list[dict] = []
    if exit_horizons:
        with ThreadPoolExecutor(
            max_workers=max(1, min(int(workers), len(exit_horizons))),
            thread_name_prefix=f"foldclock-exit-H{horizon:02d}",
        ) as pool:
            futures = [pool.submit(fit_exit, e) for e in exit_horizons]
            for future in as_completed(futures):
                _, part, row = future.result()
                parts.append(part)
                audit_rows.append(row)
    provider = (
        pd.concat(parts, ignore_index=True)
        if parts else
        pd.DataFrame(columns=[
            "decision_date", "ticker", "exit_horizon_sessions",
            "predicted_continuation_excess", "holdout_locked", "exit_generation_id",
        ])
    )
    if not provider.empty and provider.duplicated(
        ["ticker", "decision_date", "exit_horizon_sessions", "exit_generation_id"]
    ).any():
        raise ValueError(f"FOLD_CLOCK_SURFACE_EXIT_PROVIDER_DUPLICATE_KEYS:H{horizon}")
    provider.sort_values(
        ["exit_horizon_sessions", "decision_date", "ticker"], inplace=True
    )
    provider.to_parquet(provider_path, index=False)
    audit = {
        "schema_version": "FOLD_CLOCK_SURFACE_MATCHED_FROZEN_EXIT_PROVIDER_V1",
        "horizon": horizon,
        "fit_cutoff": first_plan_row["assessment_date"].isoformat(),
        "latest_matured_evidence_date": latest.isoformat(),
        "exit_generation_id": exit_generation_id,
        "row_count": len(provider),
        "model_count": len(audit_rows),
        "models": sorted(audit_rows, key=lambda x: x["exit_horizon"]),
        "contract": "CAUSAL_INITIAL_EXIT_MODELS_FROZEN_AND_MATCHED_ACROSS_A_F_M",
    }
    _write_json(audit_path, audit)
    return provider_path, exit_generation_id, audit


def _family_schedule(
    shared: pd.DataFrame,
    *,
    family,
    arm: str,
    exit_generation_id: str,
) -> pd.DataFrame:
    schedule = shared.copy()
    schedule["family_id"] = str(family.family_id)
    schedule["entry_policy_id"] = [
        f"{family.family_id}:{arm}:{pd.Timestamp(x).date()}"
        for x in schedule["activation_date"]
    ]
    if str(family.exit_policy.get("family", "FIXED")) == "LEARNED_EXIT":
        schedule["exit_policy_id"] = str(exit_generation_id)
        schedule["exit_generation_id"] = str(exit_generation_id)
    else:
        schedule["exit_policy_id"] = f"FIXED_D{int(family.holding_days):02d}"
        schedule["exit_generation_id"] = schedule["exit_policy_id"]
    return schedule


def _yearly_rows(curve: pd.DataFrame, *, family_id: str, arm: str) -> list[dict]:
    frame = curve[["date", "strategy_value", "urth_value"]].copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame["year"] = frame["date"].dt.year
    rows = []
    for year, group in frame.groupby("year", sort=True):
        group = group.sort_values("date")
        if len(group) < 2:
            continue
        rows.append({
            "family_id": family_id,
            "arm": arm,
            "year": int(year),
            "strategy_return": float(group.iloc[-1]["strategy_value"] / group.iloc[0]["strategy_value"] - 1.0),
            "urth_return": float(group.iloc[-1]["urth_value"] / group.iloc[0]["urth_value"] - 1.0),
            "excess_return": float(
                group.iloc[-1]["strategy_value"] / group.iloc[0]["strategy_value"]
                - group.iloc[-1]["urth_value"] / group.iloc[0]["urth_value"]
            ),
            "active_start": str(group.iloc[0]["date"].date()),
            "active_end": str(group.iloc[-1]["date"].date()),
        })
    return rows


def _replay_family_triplet(
    *,
    family,
    signals: dict[str, pd.DataFrame],
    schedules: dict[str, pd.DataFrame],
    prices: pd.DataFrame,
    evaluation_start: date,
    end: date,
    initial: float,
    exit_generation_id: str,
    telemetry_path: Path,
) -> tuple[list[dict], list[dict]]:
    started = time.perf_counter()
    results: list[dict] = []
    yearly: list[dict] = []
    policy = Policy(
        horizon=int(family.horizon_sessions),
        score_quantile=float(family.entry_policy_rule.get("score_quantile", .975)),
        top_fraction=float(family.entry_policy_rule.get("top_fraction", .005)),
        max_names=int(family.max_names),
        holding_days=int(family.holding_days),
        exit_family=str(family.exit_policy.get("family", "FIXED")),
        exit_value=float(family.exit_policy.get("value", 0.0)),
        replacement=str(family.exit_policy.get("replacement", "IGNORE_NEW")),
        allocation=str(family.entry_policy_rule.get("allocation", "EQUAL_ACTIVE")),
        sleeve=float(family.entry_policy_rule.get("sleeve", .50)),
    )
    for arm in ARMS:
        schedule = _family_schedule(
            schedules[arm], family=family, arm=arm,
            exit_generation_id=exit_generation_id,
        )
        result = replay_family(
            signals=signals[arm],
            authoritative_signals=signals[arm],
            prices=prices,
            policy=policy,
            generation_schedule=schedule,
            cost=CostModel(float(family.cost_contract.get("roundtrip_bps", 20.0))),
            tax=TaxConfig(enabled=False),
            start=pd.Timestamp(evaluation_start),
            end=pd.Timestamp(end),
            initial=initial,
        )
        metrics = result["metrics"]
        risk = wealth_path_metrics(result["curve"])
        row = {
            "family_id": str(family.family_id),
            "structural_plateau_id": structural_plateau_id_from_family_id(family.family_id),
            "horizon": int(family.horizon_sessions),
            "holding_days": int(family.holding_days),
            "max_names": int(family.max_names),
            "exit_mode": str(family.exit_policy.get("family", "FIXED")),
            "arm": arm,
            "evaluation_start": evaluation_start.isoformat(),
            "evaluation_end": end.isoformat(),
            "terminal_value": float(metrics["terminal_value"]),
            "urth_terminal_value": float(metrics["urth_terminal_value"]),
            "terminal_relative_return": float(metrics["terminal_value"] / metrics["urth_terminal_value"] - 1.0),
            "cagr": float(metrics.get("cagr", float("nan"))),
            "urth_cagr": float(metrics.get("urth_cagr", float("nan"))),
            "cagr_excess": float(metrics.get("cagr_excess", float("nan"))),
            "relative_max_drawdown": float(risk["relative_max_drawdown"]),
            "trade_count": int(metrics.get("trade_count", 0)),
            "total_cost_eur": float(metrics.get("total_cost_eur", metrics.get("transaction_cost_eur", 0.0))),
            "entry_fit_count": int(len(schedule)),
            "learned_exit_count": int(metrics.get("learned_exit_count", 0)),
            "exit_decision_coverage": float(metrics.get("exit_decision_coverage", 1.0)),
            "learned_exit_missing_prediction_count": int(metrics.get("learned_exit_missing_prediction_count", 0)),
        }
        numeric = (
            "terminal_value", "urth_terminal_value", "terminal_relative_return", "cagr",
            "urth_cagr", "cagr_excess", "relative_max_drawdown", "total_cost_eur",
            "exit_decision_coverage",
        )
        if not all(math.isfinite(float(row[name])) for name in numeric):
            raise AssertionError(f"FOLD_CLOCK_SURFACE_NONFINITE_FAMILY_METRIC:{family.family_id}:{arm}")
        results.append(row)
        yearly.extend(_yearly_rows(
            result["curve"], family_id=str(family.family_id), arm=arm
        ))
    _append_telemetry(telemetry_path, {
        "event": "family_triplet_replay",
        "family_id": str(family.family_id),
        "thread_id": threading.get_ident(),
        "compute_seconds": time.perf_counter() - started,
        "status": "COMPLETE",
    })
    return results, yearly


def _checkpoint_paths(root: Path, horizon: int) -> tuple[Path, Path, Path]:
    checkpoint = root / "checkpoints"
    return (
        checkpoint / f"H{horizon:02d}-family-results.parquet",
        checkpoint / f"H{horizon:02d}-yearly-results.parquet",
        checkpoint / f"H{horizon:02d}.json",
    )


def _load_checkpoint(root: Path, horizon: int, run_hashes: set[str]):
    result_path, yearly_path, meta_path = _checkpoint_paths(root, horizon)
    if not (result_path.is_file() and yearly_path.is_file() and meta_path.is_file()):
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("semantic_run_contract_hash") not in run_hashes or not meta.get("complete"):
            return None
        results = pd.read_parquet(result_path)
        yearly = pd.read_parquet(yearly_path)
        if len(results) != int(meta["result_rows"]):
            return None
        return results, yearly
    except Exception:
        return None


def _scheduler_only_contract_compatible(
    existing: Mapping[str, Any], current: Mapping[str, Any]
) -> bool:
    """Allow resume only when research semantics are identical.

    A scheduler transition may change process/thread topology and source
    revision, but never data, dates, families, costs, gates, or arm semantics.
    """
    left = dict(existing)
    right = dict(current)
    for payload in (left, right):
        payload.pop("semantic_run_contract_hash", None)
        payload.pop("execution_semantics", None)
        payload.pop("code_commit", None)
        payload.pop("checkpoint_compatibility", None)
    return stable_hash(left) == stable_hash(right)


def _save_checkpoint(
    root: Path, horizon: int, run_hash: str,
    results: pd.DataFrame, yearly: pd.DataFrame,
) -> None:
    result_path, yearly_path, meta_path = _checkpoint_paths(root, horizon)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_parquet(result_path, index=False)
    yearly.to_parquet(yearly_path, index=False)
    _write_json(meta_path, {
        "schema_version": "FOLD_CLOCK_SURFACE_HORIZON_CHECKPOINT_V1",
        "semantic_run_contract_hash": run_hash,
        "horizon": horizon,
        "complete": True,
        "result_rows": len(results),
        "yearly_rows": len(yearly),
    })


def _replay_horizon_process(
    *, horizon: int, horizon_audit: pd.DataFrame, plan: list[dict],
    horizon_families: tuple[Any, ...], representative: Any,
    signal_panel: Path, learned_exit_candidate_metrics: Path,
    prices_path: Path, output_root: Path, end: date, initial: float,
    market_dates: tuple[pd.Timestamp, ...], workers: int, run_hash: str,
    logical_processor: int | None = None,
) -> dict[str, Any]:
    """Replay one horizon in an isolated process and atomically checkpoint it."""
    started = time.perf_counter()
    affinity_pinned = False
    if logical_processor is not None:
        affinity_pinned = bool(set_current_process_affinity([int(logical_processor)]))
    horizon_telemetry = output_root / f"worker-telemetry-H{int(horizon):02d}.jsonl"
    horizon_telemetry.write_text("", encoding="utf-8")
    prices = pd.read_parquet(prices_path)
    signals, schedules = _authoritative_signals(
        horizon_audit, end=end, output_root=output_root
    )
    required_exit = _required_exit_keys(
        horizon=horizon, signals=signals, schedules=schedules,
        market_dates=market_dates,
    )
    exit_metric_rows = json.loads(
        learned_exit_candidate_metrics.read_text(encoding="utf-8")
    )
    provider_path, exit_generation_id, exit_audit = _build_exit_provider(
        horizon=horizon,
        base=representative,
        first_plan_row=plan[0],
        required=required_exit,
        signal_panel=signal_panel,
        exit_metric_rows=exit_metric_rows,
        output_root=output_root,
        workers=workers,
        telemetry_path=horizon_telemetry,
    )
    if provider_path is not None:
        configure_replay(
            LearnedExitProvider(provider_path), ProfitTaxConfig(enabled=False)
        )
    else:
        configure_replay(None, ProfitTaxConfig(enabled=False))
    evaluation_start = plan[0]["assessment_date"]
    family_rows: list[dict] = []
    yearly_rows: list[dict] = []
    with ThreadPoolExecutor(
        max_workers=max(1, min(int(workers), len(horizon_families))),
        thread_name_prefix=f"foldclock-replay-H{horizon:02d}",
    ) as pool:
        futures = [
            pool.submit(
                _replay_family_triplet,
                family=family,
                signals=signals,
                schedules=schedules,
                prices=prices,
                evaluation_start=evaluation_start,
                end=end,
                initial=initial,
                exit_generation_id=exit_generation_id,
                telemetry_path=horizon_telemetry,
            )
            for family in horizon_families
        ]
        for future in as_completed(futures):
            result_rows, year_rows = future.result()
            family_rows.extend(result_rows)
            yearly_rows.extend(year_rows)
    family_frame = pd.DataFrame(family_rows).sort_values(["family_id", "arm"])
    yearly_frame = pd.DataFrame(yearly_rows).sort_values(["family_id", "arm", "year"])
    expected_rows = len(horizon_families) * len(ARMS)
    if len(family_frame) != expected_rows:
        raise AssertionError(
            f"FOLD_CLOCK_SURFACE_HORIZON_REPLAY_ROWCOUNT:H{horizon}:{len(family_frame)}!={expected_rows}"
        )
    _save_checkpoint(output_root, horizon, run_hash, family_frame, yearly_frame)
    _append_telemetry(horizon_telemetry, {
        "event": "horizon_process_complete",
        "horizon": int(horizon),
        "family_count": len(horizon_families),
        "family_result_rows": len(family_frame),
        "worker_count": int(workers),
        "logical_processor": logical_processor,
        "affinity_pinned": affinity_pinned,
        "compute_seconds": time.perf_counter() - started,
        "status": "COMPLETE",
    })
    return {
        "horizon": int(horizon),
        "family_results": family_frame,
        "yearly_results": yearly_frame,
        "exit_audit": exit_audit,
        "telemetry_path": str(horizon_telemetry),
    }


def _contrast_rows(family_results: pd.DataFrame) -> pd.DataFrame:
    indexed = family_results.set_index(["family_id", "arm"])
    output: list[dict] = []
    pairs = (
        ("F_MINUS_A", ARM_F, ARM_A),
        ("M_MINUS_F", ARM_M, ARM_F),
    )
    metadata = family_results.drop_duplicates("family_id").set_index("family_id")
    for family_id in sorted(family_results["family_id"].unique()):
        meta = metadata.loc[family_id]
        for contrast, candidate_arm, baseline_arm in pairs:
            candidate = indexed.loc[(family_id, candidate_arm)]
            baseline = indexed.loc[(family_id, baseline_arm)]
            output.append({
                "family_id": family_id,
                "structural_plateau_id": meta["structural_plateau_id"],
                "horizon": int(meta["horizon"]),
                "holding_days": int(meta["holding_days"]),
                "max_names": int(meta["max_names"]),
                "exit_mode": str(meta["exit_mode"]),
                "contrast": contrast,
                "candidate_arm": candidate_arm,
                "baseline_arm": baseline_arm,
                "terminal_value_delta_eur": float(candidate["terminal_value"] - baseline["terminal_value"]),
                "terminal_relative_return_delta": float(candidate["terminal_relative_return"] - baseline["terminal_relative_return"]),
                "cagr_delta": float(candidate["cagr"] - baseline["cagr"]),
                "cagr_excess_delta": float(candidate["cagr_excess"] - baseline["cagr_excess"]),
                "relative_max_drawdown_delta": float(candidate["relative_max_drawdown"] - baseline["relative_max_drawdown"]),
                "trade_count_delta": int(candidate["trade_count"] - baseline["trade_count"]),
                "total_cost_delta_eur": float(candidate["total_cost_eur"] - baseline["total_cost_eur"]),
            })
    return pd.DataFrame(output)


def _yearly_contrasts(yearly_results: pd.DataFrame, family_results: pd.DataFrame) -> pd.DataFrame:
    meta = family_results.drop_duplicates("family_id").set_index("family_id")
    indexed = yearly_results.set_index(["family_id", "arm", "year"])
    rows: list[dict] = []
    for family_id in sorted(yearly_results["family_id"].unique()):
        years = sorted(yearly_results.loc[yearly_results["family_id"].eq(family_id), "year"].unique())
        for year in years:
            for contrast, candidate_arm, baseline_arm in (
                ("F_MINUS_A", ARM_F, ARM_A),
                ("M_MINUS_F", ARM_M, ARM_F),
            ):
                key_c = (family_id, candidate_arm, year)
                key_b = (family_id, baseline_arm, year)
                if key_c not in indexed.index or key_b not in indexed.index:
                    continue
                candidate = indexed.loc[key_c]
                baseline = indexed.loc[key_b]
                rows.append({
                    "family_id": family_id,
                    "structural_plateau_id": meta.loc[family_id, "structural_plateau_id"],
                    "exit_mode": meta.loc[family_id, "exit_mode"],
                    "year": int(year),
                    "contrast": contrast,
                    "strategy_return_delta": float(candidate["strategy_return"] - baseline["strategy_return"]),
                    "excess_return_delta": float(candidate["excess_return"] - baseline["excess_return"]),
                })
    return pd.DataFrame(rows)


def _plateau_contrasts(family_contrasts: pd.DataFrame) -> pd.DataFrame:
    numeric = [
        "terminal_value_delta_eur", "terminal_relative_return_delta", "cagr_delta",
        "cagr_excess_delta", "relative_max_drawdown_delta", "trade_count_delta",
        "total_cost_delta_eur",
    ]
    grouped = family_contrasts.groupby(
        ["structural_plateau_id", "exit_mode", "contrast"], as_index=False
    )[numeric].median()
    counts = family_contrasts.groupby(
        ["structural_plateau_id", "exit_mode", "contrast"], as_index=False
    )["family_id"].nunique().rename(columns={"family_id": "family_count"})
    return grouped.merge(counts, on=["structural_plateau_id", "exit_mode", "contrast"])


def _horizon_contrasts(family_contrasts: pd.DataFrame) -> pd.DataFrame:
    numeric = [
        "terminal_value_delta_eur", "cagr_delta", "cagr_excess_delta",
        "relative_max_drawdown_delta", "trade_count_delta", "total_cost_delta_eur",
    ]
    return family_contrasts.groupby(
        ["horizon", "exit_mode", "contrast"], as_index=False
    )[numeric].median()


def _bootstrap_median(values: np.ndarray, *, draws: int, seed: int) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"mean": float("nan"), "q05": float("nan"), "q50": float("nan"), "q95": float("nan")}
    rng = np.random.default_rng(seed)
    samples = np.empty(int(draws), dtype=float)
    for i in range(int(draws)):
        samples[i] = float(np.median(rng.choice(values, size=len(values), replace=True)))
    return {
        "mean": float(np.mean(samples)),
        "q05": float(np.quantile(samples, .05)),
        "q50": float(np.quantile(samples, .50)),
        "q95": float(np.quantile(samples, .95)),
    }


def _gate_for_stratum(
    *,
    contrast: str,
    family: pd.DataFrame,
    plateau: pd.DataFrame,
    yearly: pd.DataFrame,
    stratum: str,
) -> dict:
    if stratum == "ALL":
        f = family.loc[family["contrast"].eq(contrast)]
        p = plateau.loc[plateau["contrast"].eq(contrast)]
        y = yearly.loc[yearly["contrast"].eq(contrast)]
    else:
        f = family.loc[family["contrast"].eq(contrast) & family["exit_mode"].eq(stratum)]
        p = plateau.loc[plateau["contrast"].eq(contrast) & plateau["exit_mode"].eq(stratum)]
        y = yearly.loc[yearly["contrast"].eq(contrast) & yearly["exit_mode"].eq(stratum)]
    if f.empty or p.empty:
        return {"stratum": stratum, "contrast": contrast, "pass": False, "reason": "NO_EVIDENCE"}
    plateau_values = p["cagr_excess_delta"].to_numpy(float)
    bootstrap = _bootstrap_median(
        plateau_values, draws=GATE_BOOTSTRAP_DRAWS,
        seed=GATE_BOOTSTRAP_SEED + (0 if contrast == "F_MINUS_A" else 1),
    )
    plateau_year = y.groupby(["structural_plateau_id", "year"], as_index=False)["excess_return_delta"].median()
    year_medians = plateau_year.groupby("year")["excess_return_delta"].median() if not plateau_year.empty else pd.Series(dtype=float)
    if contrast == "F_MINUS_A":
        checks = {
            "median_family_cagr_excess_delta_positive": float(f["cagr_excess_delta"].median()) > 0,
            "median_plateau_cagr_excess_delta_positive": float(np.median(plateau_values)) > 0,
            "positive_plateau_fraction_at_least_60pct": float(np.mean(plateau_values > 0)) >= GATE_PLATEAU_DIRECTION_FRACTION,
            "bootstrap_q05_plateau_median_positive": float(bootstrap["q05"]) > 0,
            "median_plateau_relative_maxdd_not_worse": float(p["relative_max_drawdown_delta"].median()) >= 0,
            "positive_temporal_year_fraction_at_least_60pct": bool(len(year_medians)) and float(np.mean(year_medians.to_numpy(float) > 0)) >= GATE_TEMPORAL_DIRECTION_FRACTION,
        }
    else:
        checks = {
            "median_family_cagr_excess_delta_negative": float(f["cagr_excess_delta"].median()) < 0,
            "median_plateau_cagr_excess_delta_negative": float(np.median(plateau_values)) < 0,
            "negative_plateau_fraction_at_least_60pct": float(np.mean(plateau_values < 0)) >= GATE_PLATEAU_DIRECTION_FRACTION,
            "bootstrap_q95_plateau_median_negative": float(bootstrap["q95"]) < 0,
            "negative_temporal_year_fraction_at_least_60pct": bool(len(year_medians)) and float(np.mean(year_medians.to_numpy(float) < 0)) >= GATE_TEMPORAL_DIRECTION_FRACTION,
        }
    return {
        "stratum": stratum,
        "contrast": contrast,
        "pass": all(checks.values()),
        "checks": checks,
        "family_count": int(f["family_id"].nunique()),
        "plateau_count": int(len(p)),
        "year_count": int(len(year_medians)),
        "median_family_cagr_excess_delta": float(f["cagr_excess_delta"].median()),
        "median_plateau_cagr_excess_delta": float(p["cagr_excess_delta"].median()),
        "median_plateau_relative_maxdrawdown_delta": float(p["relative_max_drawdown_delta"].median()),
        "directional_plateau_fraction": float(np.mean(plateau_values > 0 if contrast == "F_MINUS_A" else plateau_values < 0)),
        "directional_year_fraction": float(np.mean(year_medians.to_numpy(float) > 0 if contrast == "F_MINUS_A" else year_medians.to_numpy(float) < 0)) if len(year_medians) else 0.0,
        "bootstrap_plateau_median": bootstrap,
    }


def _evaluate_gates(
    family: pd.DataFrame, plateau: pd.DataFrame, yearly: pd.DataFrame
) -> dict:
    strata = ("ALL", "FIXED", "LEARNED_EXIT")
    gates = {
        "F_MINUS_A": {
            stratum: _gate_for_stratum(
                contrast="F_MINUS_A", family=family, plateau=plateau,
                yearly=yearly, stratum=stratum,
            )
            for stratum in strata
        },
        "M_MINUS_F": {
            stratum: _gate_for_stratum(
                contrast="M_MINUS_F", family=family, plateau=plateau,
                yearly=yearly, stratum=stratum,
            )
            for stratum in strata
        },
    }
    gates["primary_validation_pass"] = bool(
        gates["F_MINUS_A"]["FIXED"]["pass"]
        and gates["F_MINUS_A"]["LEARNED_EXIT"]["pass"]
        and gates["M_MINUS_F"]["FIXED"]["pass"]
        and gates["M_MINUS_F"]["LEARNED_EXIT"]["pass"]
    )
    gates["authority"] = AUTHORITY
    gates["promotion_allowed"] = False
    gates["predeclared_contract"] = {
        "plateau_direction_fraction": GATE_PLATEAU_DIRECTION_FRACTION,
        "temporal_direction_fraction": GATE_TEMPORAL_DIRECTION_FRACTION,
        "bootstrap_draws": GATE_BOOTSTRAP_DRAWS,
        "bootstrap_seed": GATE_BOOTSTRAP_SEED,
        "plateaus": "EX_ANTE_3X3_H_D_NEIGHBORHOODS_SEPARATE_BY_EXIT_MODE",
    }
    return gates


def _compact_report(summary: dict, gates: dict) -> str:
    lines = [
        "# Dynamic-QBD Fold-Clock Surface Fold-Clock Validation",
        "",
        f"Status: **{summary['status']}**",
        f"Primary Development validation gate: **{'PASS' if gates['primary_validation_pass'] else 'FAIL'}**",
        f"Authority: `{AUTHORITY}`; final holdout remains closed.",
        "",
        "## Scope",
        "",
        f"- Families: {summary['family_count']} across H1-H30, D1-D_H, N1-N6, FIXED and LEARNED_EXIT.",
        f"- A: initial entry model + calibration frozen.",
        f"- F: same entry recipe refit only on real matured-fold expansion; frozen between events.",
        f"- M: same entry recipe refit monthly.",
        "- Learned-exit models are initial-causal and frozen/matched across A/F/M so the contrast isolates entry-refit frequency.",
        "",
        "## Predeclared gates",
        "",
    ]
    for contrast in ("F_MINUS_A", "M_MINUS_F"):
        for stratum in ("ALL", "FIXED", "LEARNED_EXIT"):
            row = gates[contrast][stratum]
            lines.append(
                f"- {contrast} / {stratum}: **{'PASS' if row.get('pass') else 'FAIL'}**, "
                f"median plateau CAGR-excess delta {row.get('median_plateau_cagr_excess_delta', float('nan')):+.4%}."
            )
    lines.extend([
        "",
        "## Execution",
        "",
        f"- Worker threads: {summary['execution_resources']['model_fit_processes']} with one native numerical thread each.",
        f"- Replay topology: {summary['execution_resources'].get('horizon_process_workers', 1)} isolated horizon processes, "
        f"{summary['execution_resources'].get('family_workers_per_horizon', summary['execution_resources']['model_fit_processes'])} family workers per horizon; next horizon is queued on completion.",
        f"- CPU capacity target: {summary['execution_resources']['target_fraction']:.0%}; enforced capacity fraction {summary['execution_resources']['available_capacity_fraction']:.2%}.",
        f"- Aggregate Windows memory ceiling: {summary['execution_resources']['memory_limit_gb']:.1f} GiB; soft target {summary['execution_resources']['memory_soft_target_gb']:.1f} GiB.",
        "- NWinfo telemetry is diagnostic-only and fail-open.",
        "",
        "No gate in this report grants promotion or capital authority.",
    ])
    return "\n".join(lines) + "\n"


def run_validation(
    *,
    signal_panel: str | Path,
    candidate_metrics: str | Path,
    learned_exit_candidate_metrics: str | Path,
    daily_store_root: str | Path,
    output_root: str | Path,
    benchmark_daily_path: str | Path | None = None,
    direct_daily_stock_root: str | Path | None = None,
    start: date = date(2016, 6, 24),
    end: date = date(2025, 12, 31),
    score_quantile: float = .975,
    top_fraction: float = .005,
    initial: float = 10000.0,
    code_commit: str = "",
    workers: int | None = None,
    horizon_workers: int = DEFAULT_HORIZON_PROCESS_WORKERS,
    nwinfo_interval_seconds: float = 60.0,
) -> dict:
    if not code_commit:
        raise ValueError("FOLD_CLOCK_SURFACE_CODE_COMMIT_REQUIRED")
    if end >= CLOSED_HOLDOUT_START:
        raise PermissionError("FOLD_CLOCK_SURFACE_FINAL_HOLDOUT_MUST_REMAIN_CLOSED")
    if start > end:
        raise ValueError("FOLD_CLOCK_SURFACE_START_AFTER_END")
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    signal_panel = Path(signal_panel)
    candidate_metrics = Path(candidate_metrics)
    learned_exit_candidate_metrics = Path(learned_exit_candidate_metrics)
    if not signal_panel.is_file() or not candidate_metrics.is_file() or not learned_exit_candidate_metrics.is_file():
        raise FileNotFoundError("FOLD_CLOCK_SURFACE_REQUIRED_INPUT_MISSING")

    capacity = cpu_budget(.80, enforce_capacity_fraction=True)
    target_workers = int(capacity["target_logical_processors"])
    worker_count = max(1, min(int(workers or target_workers), target_workers))
    horizon_process_count = max(1, min(int(horizon_workers), worker_count, 30))
    entry_process_count = max(1, min(worker_count, DEFAULT_ENTRY_PROCESS_WORKERS, 30))
    family_workers_per_horizon = max(1, worker_count // horizon_process_count)
    resources = configure_cpu_peak(
        target_fraction=.80,
        process_workers=worker_count,
        native_threads_per_worker=1,
        memory_limit_gb=float(os.environ.get("DYNAMIC_QBD_MEMORY_LIMIT_GB", "90")),
        memory_soft_target_gb=float(os.environ.get("DYNAMIC_QBD_MEMORY_SOFT_TARGET_GB", "84")),
        enforce_capacity_fraction=True,
    )
    selected_logical_processors = tuple(
        int(value) for value in resources.get("selected_logical_processors", [])
    )
    telemetry_path = output_root / "worker-telemetry.jsonl"
    telemetry_path.write_text("", encoding="utf-8")

    metric_rows = json.loads(candidate_metrics.read_text(encoding="utf-8"))
    exit_metric_rows = json.loads(learned_exit_candidate_metrics.read_text(encoding="utf-8"))
    materializer = ParquetH130DatasetMaterializer(signal_panel, candidate_metrics, end)
    sessions_frame = pd.read_parquet(signal_panel, columns=["decision_date", "holdout_locked"])
    sessions_frame["decision_date"] = pd.to_datetime(sessions_frame["decision_date"]).dt.date
    holdout_audit = _holdout_marker_audit(sessions_frame, start=start, end=end)
    sessions = tuple(sorted(x for x in sessions_frame["decision_date"].unique() if x < CLOSED_HOLDOUT_START))

    families = build_family_specs(
        feature_schema_sha256=materializer.feature_schema_fingerprint,
        include_learned_exit=True,
        score_quantile=score_quantile,
        top_fraction=top_fraction,
    )
    expected_family_count = sum(
        6 * (1 if d == 1 else 2)
        for h in range(1, 31) for d in range(1, h + 1)
    )
    if len(families) != expected_family_count:
        raise AssertionError(f"FOLD_CLOCK_SURFACE_FAMILY_COUNT_MISMATCH:{len(families)}!={expected_family_count}")
    family_by_h = {
        h: tuple(x for x in families if int(x.horizon_sessions) == h)
        for h in range(1, 31)
    }
    representatives = {
        h: next(
            x for x in family_by_h[h]
            if int(x.holding_days) == 1 and int(x.max_names) == 1
            and str(x.exit_policy.get("family")) == "FIXED"
        )
        for h in range(1, 31)
    }

    semantic_contract = {
        "schema_version": SCHEMA_VERSION,
        "authority": AUTHORITY,
        "final_holdout_opened": False,
        "promotion_allowed": False,
        "holdout_contract": HOLDOUT_CONTRACT,
        "holdout_marker_audit": holdout_audit,
        "code_commit": code_commit,
        "signal_panel_sha256": materializer.dataset_fingerprint,
        "candidate_metrics_sha256": stable_hash(metric_rows),
        "learned_exit_candidate_metrics_sha256": stable_hash(exit_metric_rows),
        "development_start": start,
        "development_end": end,
        "family_count": len(families),
        "score_quantile": score_quantile,
        "top_fraction": top_fraction,
        "initial": initial,
        "round_trip_cost_bps": 20.0,
        "sleeve": .50,
        "entry_adaptation_arms": list(ARMS),
        "learned_exit_contract": "INITIAL_CAUSAL_FROZEN_MATCHED_ACROSS_ENTRY_ARMS",
        "gate_contract": {
            "plateau_direction_fraction": GATE_PLATEAU_DIRECTION_FRACTION,
            "temporal_direction_fraction": GATE_TEMPORAL_DIRECTION_FRACTION,
            "bootstrap_draws": GATE_BOOTSTRAP_DRAWS,
            "bootstrap_seed": GATE_BOOTSTRAP_SEED,
        },
        "execution_semantics": {
            "worker_kind": "PROCESS_POOLS_FOR_ENTRY_AND_HORIZON_REPLAY",
            "scheduler_version": "ENTRY_AND_HORIZON_PROCESS_POOLS_V1_PCORE_FIRST",
            "entry_process_workers": entry_process_count,
            "horizon_process_workers": horizon_process_count,
            "family_workers_per_horizon": family_workers_per_horizon,
            "process_affinity_mode": "ONE_SELECTED_LOGICAL_PROCESSOR_PER_HORIZON_TASK",
            "cpu_target_fraction": .80,
            "native_threads_per_worker": 1,
            "memory_limit_gib": resources["memory_limit_gb"],
            "memory_soft_target_gib": resources["memory_soft_target_gb"],
        },
    }
    run_hash = stable_hash(semantic_contract)
    contract_path = output_root / "run-contract.json"
    checkpoint_hashes = {run_hash}
    if contract_path.is_file():
        existing = json.loads(contract_path.read_text(encoding="utf-8"))
        if existing.get("semantic_run_contract_hash") != run_hash:
            if not _scheduler_only_contract_compatible(existing, semantic_contract):
                raise ValueError("FOLD_CLOCK_SURFACE_RESTART_RUN_CONTRACT_MISMATCH")
            checkpoint_hashes.add(str(existing["semantic_run_contract_hash"]))
            prior = existing.get("checkpoint_compatibility", {})
            while isinstance(prior, Mapping) and prior.get("accepted_previous_hash"):
                previous_hash = str(prior["accepted_previous_hash"])
                if previous_hash in checkpoint_hashes:
                    break
                checkpoint_hashes.add(previous_hash)
                prior = prior.get("checkpoint_compatibility", {})
            _write_json(contract_path, {
                **semantic_contract,
                "semantic_run_contract_hash": run_hash,
                "checkpoint_compatibility": {
                    "mode": "SCHEDULER_ONLY_TRANSITION",
                    "accepted_previous_hash": existing["semantic_run_contract_hash"],
                    "research_contract_unchanged": True,
                },
            })
    else:
        _write_json(contract_path, {
            **semantic_contract,
            "semantic_run_contract_hash": run_hash,
        })

    plans = {
        h: _build_horizon_plan(
            horizon=h,
            base=representatives[h],
            metric_rows=metric_rows,
            sessions=sessions,
            start=start,
            end=end,
        )
        for h in range(1, 31)
    }
    fold_rows = []
    for h, plan in plans.items():
        for row in plan:
            fold_rows.append({
                "horizon": h,
                "assessment_date": row["assessment_date"],
                "latest_matured_evidence_date": row["latest_matured_evidence_date"],
                "fold_count": row["fold_count"],
                "fold_ids": json.dumps(list(row["fold_ids"])),
                "evidence_fingerprint": row["evidence_fingerprint"],
                "evidence_expanded": row["evidence_expanded"],
                "f_refit": row["f_refit"],
                "candidate_id": row["frozen_recipe"]["candidate_id"],
                "model_family": row["frozen_recipe"]["family"],
                "parameters_json": json.dumps(row["frozen_recipe"]["parameters"], sort_keys=True),
            })
    pd.DataFrame(fold_rows).to_csv(output_root / "fold-clock-events.csv", index=False)

    prices_path = materialize_daily_store_prices(
        daily_store_root=daily_store_root,
        signal_panel=signal_panel,
        start=start,
        end=end,
        output_path=output_root / "inputs" / "prices.parquet",
        benchmark_daily_path=benchmark_daily_path,
        direct_daily_stock_root=direct_daily_stock_root,
    )
    prices = pd.read_parquet(prices_path)
    market_dates = tuple(
        pd.to_datetime(prices.loc[prices["ticker"].eq("URTH"), "date"])
        .drop_duplicates().sort_values().tolist()
    )

    sampler = NWInfoSampler(output_root, interval_seconds=nwinfo_interval_seconds).start()
    try:
        entry_audit = _fit_all_entry_sources(
            plans=plans,
            representatives=representatives,
            signal_panel=signal_panel,
            candidate_metrics=candidate_metrics,
            output_root=output_root,
            end=end,
            sessions=sessions,
            code_commit=code_commit,
            workers=worker_count,
            run_hashes=checkpoint_hashes,
            telemetry_path=telemetry_path,
        )
        contract_audit = _build_contract_audit(
            entry_audit=entry_audit, plans=plans, holdout_audit=holdout_audit
        )
        _write_json(output_root / "contract-audit.json", contract_audit)
        all_family_results: list[pd.DataFrame] = []
        all_yearly_results: list[pd.DataFrame] = []
        exit_audits: list[dict] = []
        pending_horizons: list[int] = []
        for h in range(1, 31):
            checkpoint = _load_checkpoint(output_root, h, checkpoint_hashes)
            if checkpoint is not None:
                family_checkpoint, yearly_checkpoint = checkpoint
                all_family_results.append(family_checkpoint)
                all_yearly_results.append(yearly_checkpoint)
                audit_path = output_root / "learned-exit-providers" / f"H{h:02d}" / "audit.json"
                if audit_path.is_file():
                    exit_audits.append(json.loads(audit_path.read_text(encoding="utf-8")))
                print(f"[fold-clock-surface] replay checkpoint H{h:02d} hit", flush=True)
                continue
            pending_horizons.append(h)
        # Entry fits are complete and checkpointed before this stage. Each
        # horizon process owns its learned-exit global replay context, so
        # horizons can run concurrently without cross-horizon state races.
        del prices
        if pending_horizons:
            context = mp.get_context("spawn")
            with ProcessPoolExecutor(
                max_workers=horizon_process_count,
                mp_context=context,
            ) as pool:
                futures = {
                    pool.submit(
                        _replay_horizon_process,
                        horizon=h,
                        horizon_audit=entry_audit.loc[entry_audit["horizon"].eq(h)].copy(),
                        plan=plans[h],
                        horizon_families=family_by_h[h],
                        representative=representatives[h],
                        signal_panel=signal_panel,
                        learned_exit_candidate_metrics=learned_exit_candidate_metrics,
                        prices_path=prices_path,
                        output_root=output_root,
                        end=end,
                        initial=initial,
                        market_dates=market_dates,
                        workers=family_workers_per_horizon,
                        run_hash=run_hash,
                        logical_processor=(
                            selected_logical_processors[index % len(selected_logical_processors)]
                            if selected_logical_processors else None
                        ),
                    ): h for index, h in enumerate(pending_horizons)
                }
                for future in as_completed(futures):
                    result = future.result()
                    h = int(result["horizon"])
                    all_family_results.append(result["family_results"])
                    all_yearly_results.append(result["yearly_results"])
                    exit_audits.append(result["exit_audit"])
                    telemetry_file = Path(result["telemetry_path"])
                    if telemetry_file.is_file():
                        with telemetry_file.open("r", encoding="utf-8") as handle:
                            with telemetry_path.open("a", encoding="utf-8") as target:
                                target.write(handle.read())
                    print(
                        f"[fold-clock-surface] H{h:02d} replay complete families={len(family_by_h[h])}",
                        flush=True,
                    )
        family_results = pd.concat(all_family_results, ignore_index=True)
        yearly_results = pd.concat(all_yearly_results, ignore_index=True)
        if family_results["family_id"].nunique() != len(families):
            raise AssertionError("FOLD_CLOCK_SURFACE_INCOMPLETE_FAMILY_COVERAGE")
        if len(family_results) != len(families) * len(ARMS):
            raise AssertionError("FOLD_CLOCK_SURFACE_INCOMPLETE_ARM_COVERAGE")

        family_contrasts = _contrast_rows(family_results)
        yearly_contrasts = _yearly_contrasts(yearly_results, family_results)
        plateau_contrasts = _plateau_contrasts(family_contrasts)
        horizon_contrasts = _horizon_contrasts(family_contrasts)
        gates = _evaluate_gates(family_contrasts, plateau_contrasts, yearly_contrasts)

        family_results.to_parquet(output_root / "family-results.parquet", index=False)
        family_contrasts.to_parquet(output_root / "family-contrasts.parquet", index=False)
        plateau_contrasts.to_parquet(output_root / "plateau-contrasts.parquet", index=False)
        plateau_contrasts.to_csv(output_root / "plateau-contrasts.csv", index=False)
        horizon_contrasts.to_csv(output_root / "horizon-contrasts.csv", index=False)
        yearly_contrasts.to_parquet(output_root / "yearly-contrasts.parquet", index=False)
        yearly_contrasts.to_csv(output_root / "yearly-contrasts.csv", index=False)
        _write_json(output_root / "gates.json", gates)
        _write_json(output_root / "exit-provider-audit.json", exit_audits)
        entry_audit.to_csv(output_root / "calibration-audit.csv", index=False)
        _write_json(output_root / "fit-audit.json", {
            "schema_version": "FOLD_CLOCK_SURFACE_FIT_AUDIT_V1",
            "arms": {
                ARM_A: {"fresh_fit_count": 1, "calibration_resolution_count": 1},
                ARM_F: {
                    "fresh_fit_count": int(entry_audit["f_refit"].astype(bool).sum()),
                    "calibration_resolution_count": int(entry_audit["f_refit"].astype(bool).sum()),
                },
                ARM_M: {
                    "fresh_fit_count": int(len(entry_audit)),
                    "calibration_resolution_count": int(len(entry_audit)),
                },
            },
            "entry_fit_rows": int(len(entry_audit)),
        })
        pd.DataFrame([
            {"horizon": int(h), "assessment_count": int(len(plans[h])),
             "f_refit_count": int(sum(bool(row["f_refit"]) for row in plans[h])),
             "first_fold_count": int(plans[h][0]["fold_count"]),
             "last_fold_count": int(plans[h][-1]["fold_count"]),
             "recipe_identity": json.dumps(_choice_identity(plans[h][0]["frozen_recipe"]))}
            for h in sorted(plans)
        ]).to_csv(output_root / "plan-audit.csv", index=False)

        summary = {
            "schema_version": SCHEMA_VERSION,
            "status": "COMPLETE",
            "authority": AUTHORITY,
            "code_commit": code_commit,
            "semantic_run_contract_hash": run_hash,
            "promotion_allowed": False,
            "holdout_contract": HOLDOUT_CONTRACT,
            "final_holdout_opened": False,
            "recipe_parameter_grid_searched": False,
            "family_count": int(len(families)),
            "family_result_rows": int(len(family_results)),
            "family_contrast_rows": int(len(family_contrasts)),
            "plateau_count": int(plateau_contrasts["structural_plateau_id"].nunique()),
            "horizon_count": 30,
            "arms": list(ARMS),
            "development_start": start.isoformat(),
            "development_end": end.isoformat(),
            "score_quantile": score_quantile,
            "top_fraction": top_fraction,
            "initial_value_eur": initial,
            "entry_fit_count_total": int(len(entry_audit)),
            "fold_clock_fit_count_total": int(sum(len(_schedule_from_audit(entry_audit.loc[entry_audit['horizon'].eq(h)], ARM_F)) for h in range(1, 31))),
            "monthly_fit_count_total": int(len(entry_audit)),
            "horizon_process_workers": int(horizon_process_count),
            "family_workers_per_horizon": int(family_workers_per_horizon),
            "gates": gates,
            "execution_resources": {
                **active_cpu_contract(),
                "horizon_process_workers": int(horizon_process_count),
                "family_workers_per_horizon": int(family_workers_per_horizon),
            },
            "learned_exit_contract": "INITIAL_CAUSAL_FROZEN_MATCHED_ACROSS_A_F_M",
        }
        _write_json(output_root / "summary.json", summary)
        (output_root / "REPORT.md").write_text(
            _compact_report(summary, gates), encoding="utf-8"
        )
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "status": "COMPLETE",
            "authority": AUTHORITY,
            "code_commit": code_commit,
            "semantic_run_contract_hash": run_hash,
            "final_holdout_opened": False,
            "promotion_allowed": False,
            "compact_remote_results": [
                "summary.json", "REPORT.md", "gates.json", "fold-clock-events.csv",
                "horizon-contrasts.csv", "plateau-contrasts.csv", "yearly-contrasts.csv",
                "exit-provider-audit.json", "run-contract.json", "nwinfo-summary.json",
            ],
            "large_local_results": [
                "family-results.parquet", "family-contrasts.parquet",
                "plateau-contrasts.parquet", "yearly-contrasts.parquet",
                "entry-fit-audit.parquet", "authoritative-signals/",
                "entry-models/", "learned-exit-providers/", "inputs/", "checkpoints/",
                "nwinfo-sensors.jsonl", "worker-telemetry.jsonl",
            ],
        }
        _write_json(output_root / "manifest.json", manifest)
        return summary
    finally:
        sampler.stop()


def _parse_date(value: str) -> date:
    return date.fromisoformat(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--signal-panel", required=True)
    parser.add_argument("--candidate-metrics", required=True)
    parser.add_argument("--learned-exit-candidate-metrics", required=True)
    parser.add_argument("--daily-store-root", required=True)
    parser.add_argument("--benchmark-daily-path")
    parser.add_argument("--direct-daily-stock-root")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--start", type=_parse_date, default=date(2016, 6, 24))
    parser.add_argument("--end", type=_parse_date, default=date(2025, 12, 31))
    parser.add_argument("--score-quantile", type=float, default=.975)
    parser.add_argument("--top-fraction", type=float, default=.005)
    parser.add_argument("--initial", type=float, default=10000.0)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--horizon-workers", type=int, default=DEFAULT_HORIZON_PROCESS_WORKERS)
    parser.add_argument("--nwinfo-interval-seconds", type=float, default=60.0)
    args = parser.parse_args()
    summary = run_validation(
        signal_panel=args.signal_panel,
        candidate_metrics=args.candidate_metrics,
        learned_exit_candidate_metrics=args.learned_exit_candidate_metrics,
        daily_store_root=args.daily_store_root,
        output_root=args.output_root,
        benchmark_daily_path=args.benchmark_daily_path,
        direct_daily_stock_root=args.direct_daily_stock_root,
        start=args.start,
        end=args.end,
        score_quantile=args.score_quantile,
        top_fraction=args.top_fraction,
        initial=args.initial,
        code_commit=args.code_commit,
        workers=args.workers,
        horizon_workers=args.horizon_workers,
        nwinfo_interval_seconds=args.nwinfo_interval_seconds,
    )
    print(json.dumps({
        "status": summary["status"],
        "family_count": summary["family_count"],
        "primary_validation_pass": summary["gates"]["primary_validation_pass"],
        "execution_resources": summary["execution_resources"],
    }, indent=2, default=str))


if __name__ == "__main__":
    main()
