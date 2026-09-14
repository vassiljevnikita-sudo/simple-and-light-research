"""Legacy manifested H1--H30 Fixed-Exit job coordinator.

This is the older manifest/DAG orchestration path. It is not a Full Run and
does not invent candidate evidence: a run is blocked until candidate-level
OOS scores for every recipe and fold are available.  Model fitting is keyed by
H/recipe/training identity; D and N are portfolio fan-out dimensions only.
"""
from __future__ import annotations

import argparse
import atexit
from collections import OrderedDict, deque
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from contextlib import ExitStack, contextmanager
from dataclasses import asdict, dataclass
from datetime import date
import hashlib
import heapq
import json
import math
import multiprocessing as mp
from multiprocessing import shared_memory
import os
from pathlib import Path
import shutil
import sqlite3
import struct
import subprocess
import sys
import threading
import time
from typing import Any, Callable, Iterable, Mapping

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from .contract_fingerprints import stable_hash
from .dynamic_qbd_family_surface import build_family_specs, dynamic_qbd_cells, DYNAMIC_QBD_MAX_NAMES
from .candidate_oos import (CandidateOosFactory, CandidateSpec, CandidateOosStore,
                             load_primary_candidate_registry, candidate_registry_document,
                             build_evidence_snapshot, select_recipe_from_snapshot,
                             fit_production_generation, persist_evidence_snapshot,
                             read_evidence_snapshot_artifact, fold_information_available_at,
                             mature_decision_sessions)
from stock_predictor.v5.walk_forward import Fold, expanding_folds
from .qbd_training_selection_contracts import (FoldPolicy, TargetContract, RecipeSelectionPolicy,
                            ModelTrainingContract, ResolvedContracts)
from .development_slice_hash import (cleanup_legacy_slice_hash_scratch,
                                     cleanup_orphaned_slice_hash_scratch,
                                     cleanup_slice_hash_scratch_for_pid,
                                     parquet_development_slice_sha256)
from .cost_contracts import CostModel
from .portfolio_policy_contracts import Policy
from .dynamic_qbd_development_evaluation import tax_config_from_contract
from .dynamic_qbd_portfolio_replay import replay_family
from .dynamic_qbd_evidence import build_monthly_family_evidence
from .cpu_topology import set_current_process_logical_affinity
from .dynamic_qbd_runtime_resources import active_cpu_contract
from .dynamic_qbd_shared_compute_store import (
    compute_identity_key, compute_key_lock, materialize_immutable_tree,
    shared_store_contract,
)
from .dynamic_qbd_gpu_pretraining import (
    CPU_EXECUTION_BACKEND, GPU_EXECUTION_BACKEND, GPU_HGB_EXECUTION_BACKEND,
    GPUSectionQueueTimeout,
    expected_gpu_section_seconds, gpu_section_queue_snapshot,
    gpu_workload_allowed, hgb_kernel_parameters, opencl_fp64_devices,
)
from stock_predictor.v5.dataset_builder import canonical_sha256


MANIFESTED_JOB_SCHEMA = "DYNAMIC_QBD_MANIFESTED_JOB_COORDINATOR_V1"
HOLDOUT_BOUNDARY = date(2026, 7, 25)
REQUIRED_CANDIDATE_OOS_COLUMNS = {
    "fold_id", "horizon", "candidate_id", "recipe_family", "hyperparameters",
    "decision_date", "ticker", "oos_score", "terminal_date", "realized_excess",
    "information_available_at",
}
JOB_STATES = ("PENDING", "RUNNING", "COMPLETE", "FAILED", "BLOCKED", "STALE")
JOB_LEASE_SECONDS = 900
JOB_HEARTBEAT_SECONDS = 60
WORKER_FAILURE_RETRY_LIMIT = 3
# A worker process can materialize several GiB of feature state before the
# RAM admission loop sees its first job. Keep the process count bounded before
# admission, so the scheduler can always reclaim one lane instead of entering
# a system-wide paging spiral.
MAX_RUNTIME_PROCESS_LANES = 32


class RamJobReclaimed(RuntimeError):
    """One exact lane was reclaimed for RAM pressure and must be requeued."""

    def __init__(self, *, job_id: str, pid: int | None,
                 reason: str, released_gib: float) -> None:
        self.job_id = str(job_id)
        self.pid = int(pid) if pid is not None else None
        self.reason = str(reason)
        self.released_gib = float(released_gib)
        super().__init__(
            "DQBD_RAM_SINGLE_LANE_RECLAIM:"
            f"{self.job_id}:{self.pid}:{self.reason}:"
            f"{self.released_gib:.6f}")


_MANIFESTED_WORKER_SLOT = -1
_MANIFESTED_WORKER_CPU: int | None = None
_MANIFESTED_WORKER_AFFINITY = False
_MANIFESTED_WORKER_ROLE = "causal"
_CANDIDATE_WORKER_FACTORY: CandidateOosFactory | None = None
_CANDIDATE_WORKER_FACTORY_KEY: tuple[str, str] | None = None
_CAUSAL_WORKER_HANDLERS: dict[str, Callable[[dict], Any]] | None = None
_CAUSAL_WORKER_HANDLERS_KEY: str | None = None


class _HotReloadRequested(RuntimeError):
    """Interrupt one scheduler quantum so the current source can be loaded."""


class BoundaryStopRequested(RuntimeError):
    """Leave the scheduler cleanly so a new code version can take over."""


class HotReloadController:
    """Interrupt the scheduler at a safe boundary on external request.

    This used to reload the coordinator module in place.  That was unsound
    and was observed to abort the run: ``importlib.reload`` rebinds the
    module's classes, so live ``ManifestedJobInputs`` instances stop matching
    the class now registered under their qualified name and every subsequent
    worker submission dies with

        PicklingError: Can't pickle <class ...ManifestedJobInputs>:
        it's not the same object as ...ManifestedJobInputs

    (observed on ``model:2020-08-31:H13_RIDGE_HGB_FROZEN_RULE`` after five
    attempts, which then failed the whole seed).  Reload could not update the
    parent scheduler anyway, because ``dynamic_qbd_batched_portfolio_runtime``
    binds ``execute_ready_jobs`` and friends with ``from ... import`` at
    import time, and it invalidated live ``except`` clauses by replacing the
    ``_HotReloadRequested`` and ``RamJobReclaimed`` class objects.

    New code is therefore adopted by restarting the compute process from an
    immutable code snapshot; see ``ops/dqbd_hot_supervisor.py``.  This
    controller only reports that the supervisor asked for a boundary stop, so
    the scheduler can stop admitting work, requeue its own claims and exit
    while every COMPLETE checkpoint stays reusable.
    """

    def __init__(self, roots: Iterable[str | Path], *,
                 interval_seconds: float = 1.0,
                 boundary_flag: str | Path | None = None) -> None:
        self.roots = tuple(Path(root) for root in roots)
        self.interval_seconds = max(.1, float(interval_seconds))
        if boundary_flag is None:
            boundary_flag = os.environ.get("DQBD_BOUNDARY_STOP_FLAG") or None
        self.boundary_flag = (
            Path(boundary_flag) if boundary_flag is not None else None)
        self._lock = threading.RLock()
        self._last_scan = 0.0
        self._requested = False

    def changed(self) -> bool:
        """True once the supervisor has requested a boundary stop."""
        if self.boundary_flag is None:
            return False
        with self._lock:
            if self._requested:
                return True
            now = time.monotonic()
            if now - self._last_scan < self.interval_seconds:
                return False
            self._last_scan = now
            try:
                self._requested = self.boundary_flag.is_file()
            except OSError:
                self._requested = False
            return self._requested

    # Retained for call-site compatibility.  Adopting new source is the
    # supervisor's job; this process only stops.
    def reload_coordinator(self) -> bool:
        raise BoundaryStopRequested(
            "DQBD_BOUNDARY_STOP_REQUESTED_RESTART_FROM_CHECKPOINT")


_HOT_RELOAD_CONTROLLER: HotReloadController | None = None
_DEVELOPMENT_HASH_CACHE: dict[tuple[str, int, int, str, str], str] = {}
_REPLAY_BENCHMARK_CACHE: dict[tuple[str, int, int, str], pd.DataFrame] = {}
_REPLAY_DISTRIBUTION_CACHE: dict[tuple[str, int, int, str], pd.DataFrame] = {}
_REPLAY_SEGMENT_PRICE_CACHE: OrderedDict[tuple[str, int, int, str, str, str], pd.DataFrame] = OrderedDict()
_REPLAY_SEGMENT_PRICE_CACHE_SIZE = 8
_REPLAY_SIGNAL_CACHE: OrderedDict[tuple[str, int, int], pd.DataFrame] = OrderedDict()
_REPLAY_SIGNAL_CACHE_SIZE = 32
_REPLAY_GENERATION_READY_CACHE: OrderedDict[tuple[str, str], dict] = OrderedDict()
_REPLAY_GENERATION_READY_CACHE_SIZE = 64
_REPLAY_STATIC_CACHE_LOCK = threading.Lock()


def _execution_backend_for_family(recipe_family: str,
                                  override: str | None = None) -> str:
    if override:
        return str(override)
    forced = os.environ.get("DQBD_FORCE_EXECUTION_BACKEND", "").strip()
    if forced:
        return forced
    if recipe_family == "RIDGE_LOGISTIC" and os.environ.get("DQBD_ENABLE_GPU_PRETRAINING") == "1":
        return GPU_EXECUTION_BACKEND
    if recipe_family == "HIST_GRADIENT_BOOSTING" and os.environ.get("DQBD_ENABLE_GPU_HGB") == "1":
        return GPU_HGB_EXECUTION_BACKEND
    return CPU_EXECUTION_BACKEND



def _is_gpu_backend_failure(exc: BaseException) -> bool:
    """Return True only for device/backend failures that justify disabling a GPU."""
    if isinstance(exc, GPUSectionQueueTimeout):
        # A section wait expiring is a scheduler back-pressure signal. It is
        # intentionally handled by the transient requeue path and must never
        # be treated as evidence that the physical adapter is broken.
        return False
    if isinstance(exc, MemoryError):
        return False
    name = type(exc).__name__.upper()
    message = str(exc).upper()
    if "MEMORY" in name or "MEMORYERROR" in message or "ARRAYMEMORY" in message:
        return False
    tokens = (
        "OPENCL", "CL_", "GPU", "LIGHTGBM", "DEVICE",
        "KERNEL", "WATCHDOG", "TDR",
    )
    return any(token in name or token in message for token in tokens)


def _gpu_execution_fingerprint(workload: str, device_index: int | None) -> str:
    """Bind device-specific kernel policy into immutable execution identity."""
    if device_index is None:
        return stable_hash({"backend": CPU_EXECUTION_BACKEND, "device": "CPU"})
    devices = opencl_fp64_devices()
    if not (0 <= int(device_index) < len(devices)):
        raise IndexError("DQBD_GPU_DEVICE_INDEX_OUT_OF_RANGE")
    device = devices[int(device_index)]
    kernel = hgb_kernel_parameters(device) if workload == "HGB" else {
        "precision": "FP64", "kernel": "RIDGE_GRAM_RHS"}
    return stable_hash({
        "workload": str(workload),
        "device_index": int(device_index),
        "platform_index": int(device["platform_index"]),
        "opencl_device_index": int(device["device_index"]),
        "vendor": str(device["vendor"]),
        "name": str(device["name"]),
        "kernel": kernel,
    })


def _choose_fair_gpu_device(gpu_busy: list[bool], assignments: list[int],
                            cursor: int) -> tuple[int | None, int]:
    """Choose the least-assigned free GPU with round-robin tie breaking."""
    if not gpu_busy:
        return None, 0
    free = [index for index, busy in enumerate(gpu_busy) if not busy]
    if not free:
        return None, int(cursor) % len(gpu_busy)
    start = int(cursor) % len(gpu_busy)
    selected = min(free, key=lambda index: (int(assignments[index]),
                                             (index - start) % len(gpu_busy)))
    return selected, (selected + 1) % len(gpu_busy)


def _choose_gpu_feed_device(
    free_devices: Iterable[int], *,
    pending_depths: Mapping[int, int],
    predicted_finishes: Mapping[int, float],
) -> int | None:
    """Prefer physical-device breadth before stacking more staged Futures.

    Whole-job service time contains CPU/materialization intervals, while the
    actual GPU kernel is serialized per physical device. A faster GPU must not
    receive a second staged Future while another healthy physical GPU is still
    locally empty merely because its learned full-job P75 is lower.
    """
    free = [int(index) for index in free_devices]
    if not free:
        return None
    return min(
        free,
        key=lambda index: (
            int(pending_depths.get(index, 0)),
            float(predicted_finishes.get(index, float("inf"))),
            index,
        ),
    )


def _candidate_gpu_workload_priority(
    workload: str, *, prepared: bool,
) -> int:
    """Candidate GPU order: prepared HGB, prepared Ridge, otherwise CPU."""
    if not prepared:
        return 99
    if str(workload) == "HGB":
        return 0
    if str(workload) == "RIDGE":
        return 1
    return 99


class _CandidateOosReadyFrontier:
    """In-memory Candidate-OOS frontier built once per executor invocation.

    Candidate-OOS dependencies are intentionally static after initialization:
    every fold job depends on the already-complete candidate registry input.
    The old dispatcher nevertheless re-read and JSON-decoded the entire
    PENDING frontier for every GPU/prep/CPU scheduling decision.  At the full
    scale this turned dispatch into a CPU-bound table scan while worker lanes
    remained resident.

    This frontier keeps the initial manifest read as the only broad queue
    operation.  Claims and completions remove one ID; a prepared fold or a
    RAM-reclaimed job re-adds only the affected IDs.  Heaps are inspected only
    up to the small requested look-ahead, so a blocked route cannot trigger a
    second full frontier scan.
    """

    def __init__(
        self,
        jobs: Iterable[Mapping[str, Any]],
        workload_for_job: Callable[[Mapping[str, Any]], str],
        fold_prepared: Callable[[Mapping[str, Any]], bool],
    ) -> None:
        self._workload_for_job = workload_for_job
        self._fold_prepared = fold_prepared
        self._jobs: dict[str, dict[str, Any]] = {}
        self._active: set[str] = set()
        self._known_by_fold: dict[tuple[int, str], list[str]] = {}
        self._prepared_folds: set[tuple[int, str]] = set()
        self._versions: dict[str, int] = {}
        self._gpu_heap: list[tuple[tuple[Any, ...], str, int]] = []
        self._prepare_heap: list[tuple[tuple[Any, ...], str, int]] = []
        self._cpu_heap: list[tuple[tuple[Any, ...], str, int]] = []
        self._cpu_prepared_heap: list[
            tuple[tuple[Any, ...], str, int]
        ] = []
        self._cpu_other_heap: list[
            tuple[tuple[Any, ...], str, int]
        ] = []
        self._prepare_folds: set[tuple[int, str]] = set()
        self.initial_scan_count = 1
        self.frontier_refresh_count = 0
        self.claim_count = 0
        self.complete_count = 0

        for raw_job in sorted(
            jobs, key=lambda value: str(value["job_id"]),
        ):
            job = dict(raw_job)
            job_id = str(job["job_id"])
            self._jobs[job_id] = job
            self._active.add(job_id)
            key = self._fold_key(job)
            self._known_by_fold.setdefault(key, []).append(job_id)
            if self._fold_prepared(job):
                self._prepared_folds.add(key)
        for job_id in tuple(self._active):
            self._enqueue(job_id)

    @staticmethod
    def _fold_key(job: Mapping[str, Any]) -> tuple[int, str]:
        return int(job["horizon"]), str(job["fold_id"])

    def _next_version(self, job_id: str) -> int:
        version = int(self._versions.get(job_id, 0)) + 1
        self._versions[job_id] = version
        return version

    def _enqueue(self, job_id: str) -> None:
        if job_id not in self._active:
            return
        job = self._jobs[job_id]
        version = self._next_version(job_id)
        workload = self._workload_for_job(job)
        prepared = self._fold_key(job) in self._prepared_folds
        gpu_priority = _candidate_gpu_workload_priority(
            workload, prepared=prepared)
        if gpu_priority < 99:
            preference = (
                0 if str(job.get("execution_preference", "")).upper()
                == "GPU" else 1)
            heapq.heappush(
                self._gpu_heap,
                ((gpu_priority, preference, job_id), job_id, version),
            )
        cpu_priority = (
            -1 if int(job.get("ram_reclaim_count", 0) or 0) > 0 else 0,
            0 if workload not in {"HGB", "RIDGE"} else (
                1 if workload == "HGB" and prepared else (
                    2 if workload == "RIDGE" and prepared else 0)),
            job_id,
        )
        heapq.heappush(self._cpu_heap, (cpu_priority, job_id, version))
        if workload in {"HGB", "RIDGE"} and prepared:
            heapq.heappush(
                self._cpu_prepared_heap,
                (cpu_priority, job_id, version),
            )
        elif workload not in {"HGB", "RIDGE"}:
            heapq.heappush(
                self._cpu_other_heap,
                (cpu_priority, job_id, version),
            )

        if (
            workload in {"HGB", "RIDGE"}
            and not prepared
            and self._fold_key(job) not in self._prepare_folds
        ):
            fold = self._fold_key(job)
            heapq.heappush(
                self._prepare_heap,
                ((0 if workload == "RIDGE" else 1, fold,
                  job_id), job_id, version),
            )
            self._prepare_folds.add(fold)

    def _valid(
        self, job_id: str, version: int, *, require_prepared: bool = False,
    ) -> bool:
        if job_id not in self._active:
            return False
        if int(self._versions.get(job_id, -1)) != int(version):
            return False
        if require_prepared:
            return self._fold_key(self._jobs[job_id]) in self._prepared_folds
        return True

    def _peek_heap(
        self,
        heap: list[tuple[tuple[Any, ...], str, int]],
        limit: int,
        *,
        require_prepared: bool = False,
        blocked_folds: set[tuple[int, str]] | None = None,
    ) -> tuple[str, ...]:
        limit = max(1, int(limit))
        held: list[tuple[tuple[Any, ...], str, int]] = []
        result: list[str] = []
        while heap and len(result) < limit:
            entry = heapq.heappop(heap)
            _, job_id, version = entry
            if not self._valid(
                job_id, version, require_prepared=require_prepared,
            ):
                continue
            fold = self._fold_key(self._jobs[job_id])
            if blocked_folds and fold in blocked_folds:
                held.append(entry)
                continue
            held.append(entry)
            result.append(job_id)
        for entry in held:
            heapq.heappush(heap, entry)
        return tuple(result)

    def gpu_ids(self, limit: int = 128) -> tuple[str, ...]:
        return self._peek_heap(
            self._gpu_heap, limit, require_prepared=True)

    def prepare_ids(
        self, blocked_folds: set[tuple[int, str]], limit: int = 128,
    ) -> tuple[str, ...]:
        return self._peek_heap(
            self._prepare_heap, limit, blocked_folds=blocked_folds)

    def cpu_ids(
        self, limit: int = 128, *, prepared_gpu_only: bool = False,
    ) -> tuple[str, ...]:
        if not prepared_gpu_only:
            return self._peek_heap(self._cpu_heap, limit)
        limit = max(1, int(limit))
        prepared = self._peek_heap(
            self._cpu_prepared_heap, limit)
        other = self._peek_heap(self._cpu_other_heap, limit)
        return tuple((*prepared, *other)[:limit])

    def claim(self, job_id: str) -> None:
        job_id = str(job_id)
        self._active.discard(job_id)
        self.claim_count += 1

    def complete(self, job_id: str) -> None:
        self._active.discard(str(job_id))
        self.complete_count += 1

    def discard(self, job_id: str) -> None:
        """Drop an ID found non-PENDING by an external resume/publisher."""
        self._active.discard(str(job_id))

    def requeue(self, job: Mapping[str, Any]) -> None:
        job_id = str(job["job_id"])
        self._jobs[job_id] = dict(job)
        self._active.add(job_id)
        self.frontier_refresh_count += 1
        self._enqueue(job_id)

    def mark_prepared(self, fold: tuple[int, str]) -> None:
        fold = (int(fold[0]), str(fold[1]))
        if fold in self._prepared_folds:
            return
        self._prepared_folds.add(fold)
        for job_id in self._known_by_fold.get(fold, ()):
            if job_id in self._active:
                self._enqueue(job_id)

    def stats(self) -> dict[str, int]:
        return {
            "initial_scan_count": int(self.initial_scan_count),
            "frontier_refresh_count": int(self.frontier_refresh_count),
            "claim_count": int(self.claim_count),
            "complete_count": int(self.complete_count),
            "active_count": len(self._active),
            "known_count": len(self._jobs),
        }


def _gpu_job_seconds(
    device: Mapping[str, Any], workload: str,
) -> float:
    """Estimate total device work represented by one staged model job."""
    section = expected_gpu_section_seconds(device, workload)
    # HGB bundle = regressor + positive classifier + downside classifier.
    return section * (3.0 if str(workload) == "HGB" else 1.0)


class _CandidateExecutionDurationProfiles:
    """Learn full Candidate-OOS service times for CPU/GPU routing only."""

    SCHEMA = "DQBD_CANDIDATE_EXECUTION_DURATION_PROFILES_V1"

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.path = self.root / "candidate-execution-duration-profiles.json"
        self._lock = threading.RLock()
        self._profiles: dict[str, dict[str, Any]] = {}
        if self.path.is_file():
            try:
                payload = json.loads(self.path.read_text(encoding="utf-8"))
                if payload.get("schema_version") == self.SCHEMA:
                    self._profiles = dict(payload.get("profiles", {}))
            except (OSError, json.JSONDecodeError):
                self._profiles = {}

    @staticmethod
    def _key(
        workload: str, executor: str, device: Mapping[str, Any] | None,
    ) -> str:
        if executor == "CPU":
            return f"{str(workload)}:CPU"
        row = dict(device or {})
        identity = (
            f"{row.get('vendor', '')}|{row.get('name', '')}|"
            f"{row.get('platform_index', '')}|{row.get('device_index', '')}")
        return f"{str(workload)}:GPU:{identity}"

    @staticmethod
    def _cold_seconds(workload: str, executor: str) -> float:
        parts = {
            value.upper()
            for value in str(workload).split(":")
            if value}
        if "HGB" in parts:
            return 45.0 if executor == "CPU" else 14.0
        if "RIDGE" in parts:
            return 12.0 if executor == "CPU" else 5.0
        return 15.0

    def estimate(
        self, workload: str, executor: str,
        device: Mapping[str, Any] | None = None,
    ) -> float:
        key = self._key(workload, executor, device)
        with self._lock:
            row = dict(self._profiles.get(key, {}))
        observed = float(row.get("p75_seconds", 0.0) or 0.0)
        return max(
            .05,
            observed if observed > 0.0
            else self._cold_seconds(workload, executor))

    def observe(
        self, workload: str, executor: str, seconds: float,
        device: Mapping[str, Any] | None = None,
    ) -> None:
        value = max(.001, float(seconds))
        key = self._key(workload, executor, device)
        with compute_key_lock(
            self.root, "candidate-execution-duration-profile",
            {"schema_version": self.SCHEMA, "profile_key": "ALL"},
        ):
            with self._lock:
                if self.path.is_file():
                    try:
                        payload = json.loads(
                            self.path.read_text(encoding="utf-8"))
                        if payload.get("schema_version") == self.SCHEMA:
                            self._profiles = dict(
                                payload.get("profiles", {}))
                    except (OSError, json.JSONDecodeError):
                        pass
                prior = dict(self._profiles.get(key, {}))
                samples = [
                    float(x) for x in prior.get("recent_seconds", ())
                    if float(x) > 0.0]
                samples.append(value)
                samples = samples[-64:]
                ordered = sorted(samples)
                p75 = ordered[
                    min(
                        len(ordered) - 1,
                        max(0, int(math.ceil(.75 * len(ordered))) - 1))
                ]
                self._profiles[key] = {
                    "count": int(prior.get("count", 0)) + 1,
                    "recent_seconds": samples,
                    "p75_seconds": float(p75),
                    "mean_seconds": float(sum(samples) / len(samples)),
                    "last_seconds": value,
                    "updated_at": time.time(),
                }
                payload = {
                    "schema_version": self.SCHEMA,
                    "profiles": self._profiles,
                }
                self.path.parent.mkdir(parents=True, exist_ok=True)
                temporary = self.path.with_name(
                    self.path.name + f".{os.getpid()}.tmp")
                temporary.write_text(
                    json.dumps(payload, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")
                os.replace(temporary, self.path)



def _gpu_backlog_rows(
    root: Path, devices: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    snapshot = gpu_section_queue_snapshot(root, devices)
    rows = list(snapshot.get("devices", ()))
    while len(rows) < len(devices):
        rows.append({
            "device_index": len(rows),
            "ticket_count": 0,
            "backlog_seconds": 0.0,
            "backlog_seconds_by_workload": {},
        })
    return rows


def _gpu_global_backlog_seconds(
    root: Path, devices: list[dict[str, Any]], device_index: int,
) -> float:
    """Return shared physical-device ready-section backlog for routing."""
    index = int(device_index)
    if index < 0:
        return 0.0
    rows = _gpu_backlog_rows(root, devices)
    if index >= len(rows):
        return 0.0
    return max(
        0.0,
        float(rows[index].get("backlog_seconds", 0.0) or 0.0))


class GpuRuntimeMonitor:
    """Non-blocking parent-side monitor used by the GPU ready queue.

    Hardware probes must never sit on the scheduler's dispatch path.  The
    previous synchronous ``nvidia-smi`` call could consume the entire queue
    tick, while AMD had no live values at all.  A daemon sampler now owns the
    probe latency and the scheduler reads the latest snapshot immediately.
    """

    def __init__(
        self, telemetry_root: Path, *, queue_root: Path | None = None,
    ) -> None:
        self.telemetry_path = Path(telemetry_root) / "gpu-runtime.jsonl"
        self.queue_root = Path(queue_root) if queue_root is not None else None
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._devices: tuple[dict[str, Any], ...] = ()
        self._latest: dict[str, Any] | None = None
        self._thread: threading.Thread | None = None

    @staticmethod
    def _nvidia_smi() -> list[dict[str, Any]]:
        if shutil.which("nvidia-smi") is None:
            return []
        try:
            completed = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=False, timeout=3, check=False)
        except (OSError, subprocess.TimeoutExpired):
            return []
        rows = []
        stdout = (completed.stdout or b"").decode("utf-8", errors="replace")
        for line in stdout.splitlines():
            fields = [value.strip() for value in line.split(",")]
            if len(fields) != 6:
                continue
            try:
                rows.append({"index": int(fields[0]), "name": fields[1],
                             "utilization_percent": float(fields[2]),
                             "memory_used_mib": float(fields[3]),
                             "memory_total_mib": float(fields[4]),
                             "temperature_c": float(fields[5])})
            except ValueError:
                continue
        return rows

    @staticmethod
    def _libre_hardware_monitor() -> dict[str, Any]:
        local = os.environ.get("LOCALAPPDATA", "")
        candidates = []
        configured = os.environ.get("DQBD_LIBRE_HARDWARE_MONITOR_PATH", "").strip()
        if configured:
            candidates.append(Path(configured))
        if local:
            candidates.append(Path(local) / "Microsoft" / "WinGet" / "Packages" /
                              "LibreHardwareMonitor.LibreHardwareMonitor_Microsoft.Winget.Source_8wekyb3d8bbwe" /
                              "LibreHardwareMonitor.exe")
        installed = next((path for path in candidates if path.is_file()), None)
        running = False
        if os.name == "nt":
            try:
                listing = subprocess.run(["tasklist", "/FI", "IMAGENAME eq LibreHardwareMonitor.exe"],
                                         capture_output=True, text=False, timeout=3, check=False)
                stdout = (listing.stdout or b"").decode("utf-8", errors="replace")
                running = "LibreHardwareMonitor.exe".lower() in stdout.lower()
            except (OSError, subprocess.TimeoutExpired):
                pass
        return {"installed": installed is not None, "path": str(installed) if installed else None,
                "running": running, "provider": "LIBRE_HARDWARE_MONITOR"}

    def _snapshot_blocking(self, devices: list[dict[str, Any]]) -> dict[str, Any]:
        nvidia = self._nvidia_smi()
        lhm = self._libre_hardware_monitor()
        queue_rows = (
            _gpu_backlog_rows(self.queue_root, devices)
            if self.queue_root is not None else []
        )
        nvidia_index = 0
        rows = []
        for device_index, device in enumerate(devices):
            vendor = str(device.get("vendor", ""))
            vendor_upper = vendor.upper()
            is_nvidia = "NVIDIA" in vendor_upper
            is_amd = "AMD" in vendor_upper or "ADVANCED MICRO DEVICES" in vendor_upper
            if is_nvidia:
                metrics = nvidia[nvidia_index] if nvidia_index < len(nvidia) else {}
                nvidia_index += 1
                monitor = "NVIDIA_SMI"
            elif is_amd:
                metrics = {}
                monitor = "LIBRE_HARDWARE_MONITOR+OPENCL"
            else:
                metrics = {}
                monitor = "OPENCL"
            queue = queue_rows[device_index] if device_index < len(queue_rows) else {}
            rows.append({"device_index": device_index, "vendor": vendor,
                         "name": device.get("name"), "monitor": monitor,
                         "monitoring_available": bool(metrics) or (is_amd and lhm["installed"]),
                         "hardware_utilization_percent": metrics.get(
                             "utilization_percent"),
                          "hardware_idle": (
                              metrics.get("utilization_percent") is not None
                              and float(metrics.get("utilization_percent", 0.0)) < 1.0),
                          "queue_ticket_count": int(queue.get("ticket_count", 0) or 0),
                          "queue_backlog_seconds": float(
                              queue.get("backlog_seconds", 0.0) or 0.0),
                          "scheduler_idle": int(
                              queue.get("ticket_count", 0) or 0) == 0,
                          "assignment_allowed": True, **metrics})
        snapshot = {"timestamp": time.time(), "devices": rows,
                    "nvidia_smi": bool(nvidia), "libre_hardware_monitor": lhm}
        self.telemetry_path.parent.mkdir(parents=True, exist_ok=True)
        with self.telemetry_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(snapshot, sort_keys=True, default=str) + "\n")
        return snapshot

    def _sample_loop(self) -> None:
        while not self._stop.is_set():
            with self._lock:
                devices = [dict(row) for row in self._devices]
            if devices:
                try:
                    sample = self._snapshot_blocking(devices)
                except Exception as exc:
                    sample = {"timestamp": time.time(), "devices": [],
                              "error": f"{type(exc).__name__}:{exc}"}
                with self._lock:
                    self._latest = sample
            # A quarter-second cadence keeps dispatch responsive without
            # spawning probe processes on every scheduler iteration.
            self._stop.wait(.25)

    def snapshot(self, devices: list[dict[str, Any]]) -> dict[str, Any]:
        """Return the latest probe without blocking the GPU dispatcher."""
        identity = tuple(
            (int(row.get("platform_index", -1)),
             int(row.get("device_index", -1)),
             str(row.get("name", "")))
            for row in devices)
        with self._lock:
            if identity != tuple(
                    (int(row.get("platform_index", -1)),
                     int(row.get("device_index", -1)),
                     str(row.get("name", "")))
                    for row in self._devices):
                self._devices = tuple(dict(row) for row in devices)
                if self._thread is None:
                    self._thread = threading.Thread(
                        target=self._sample_loop,
                        name="dqbd-gpu-hardware-monitor",
                        daemon=True)
                    self._thread.start()
            latest = dict(self._latest or {})
        if not latest:
            latest = {
                "timestamp": time.time(),
                "devices": [
                    {"device_index": index,
                     "vendor": row.get("vendor"),
                     "name": row.get("name"),
                     "monitor": "OPENCL_PENDING",
                     "monitoring_available": False,
                     "queue_ticket_count": 0,
                     "queue_backlog_seconds": 0.0,
                     "scheduler_idle": True,
                     "assignment_allowed": True}
                    for index, row in enumerate(devices)
                ],
                "nvidia_smi": False,
                "libre_hardware_monitor": {
                    "installed": False, "running": False,
                    "provider": "OPENCL_PENDING",
                },
            }
        return latest

    def close(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.0)



def _posix_cgroup_memory() -> tuple[int | None, int | None]:
    """Return the effective cgroup memory limit and current usage.

    SC_PHYS_PAGES describes the host, not necessarily the environment in
    which a hot-started process is running. In a container that mismatch
    lets RAM admission spend memory above the real cgroup ceiling. Read
    cgroup v2 first and v1 as a compatibility fallback; an unlimited or
    unreadable cgroup keeps the normal host behavior.
    """
    candidates = (
        (Path("/sys/fs/cgroup/memory.max"),
         Path("/sys/fs/cgroup/memory.current")),
        (Path("/sys/fs/cgroup/memory/memory.limit_in_bytes"),
         Path("/sys/fs/cgroup/memory/memory.usage_in_bytes")),
    )
    for limit_path, current_path in candidates:
        try:
            raw_limit = limit_path.read_text(encoding="utf-8").strip()
            if not raw_limit or raw_limit.lower() == "max":
                continue
            limit = int(raw_limit)
            if limit <= 0:
                continue
            try:
                current = int(current_path.read_text(encoding="utf-8").strip())
            except (OSError, ValueError):
                current = None
            return limit, current
        except (OSError, ValueError):
            continue
    return None, None

def _current_process_rss_bytes() -> int:
    """Return current RSS for the calling worker without an optional dependency."""
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        get_process_memory_info = psapi.GetProcessMemoryInfo
        # Explicit signatures are required on Windows.  Without them ctypes
        # applies platform-default conversions to the HANDLE/pointer pair;
        # on this host that made the self-process query fail with
        # ERROR_INVALID_HANDLE and silently disabled RSS learning.
        get_process_memory_info.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(PROCESS_MEMORY_COUNTERS),
            wintypes.DWORD,
        ]
        get_process_memory_info.restype = wintypes.BOOL
        counters = PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(counters)
        if not get_process_memory_info(
            ctypes.windll.kernel32.GetCurrentProcess(),
            ctypes.byref(counters), counters.cb
        ):
            raise OSError(ctypes.get_last_error(), "DQBD_GET_SELF_PROCESS_RAM_FAILED")
        return int(counters.WorkingSetSize)
    statm = Path("/proc/self/statm")
    if statm.is_file():
        return int(statm.read_text().split()[1]) * int(os.sysconf("SC_PAGE_SIZE"))
    return 0



_LIVE_JOB_STRUCT = struct.Struct("<QIIQQQQQQ")
_LIVE_JOB_STATE_FREE = 0
_LIVE_JOB_STATE_RUNNING = 1
_LIVE_JOB_STATE_FINISHED = 2


class _LiveJobRamBoard:
    """Small shared-memory board for 10-ms live per-job RSS telemetry."""

    def __init__(self, slots: int) -> None:
        self.slots = max(4, int(slots))
        self._shm = shared_memory.SharedMemory(
            create=True, size=self.slots * _LIVE_JOB_STRUCT.size)
        self._lock = threading.Lock()
        self._free = set(range(self.slots))
        self._descriptors: dict[int, dict[str, Any]] = {}
        self._closed = False
        self._shm.buf[:] = b"\x00" * len(self._shm.buf)

    @property
    def name(self) -> str:
        return self._shm.name

    def allocate(self, descriptor: Mapping[str, Any]) -> dict[str, Any]:
        with self._lock:
            if not self._free:
                raise RuntimeError("DQBD_LIVE_RAM_BOARD_EXHAUSTED")
            slot = min(self._free)
            self._free.remove(slot)
            self._descriptors[slot] = dict(descriptor)
            offset = slot * _LIVE_JOB_STRUCT.size
            self._shm.buf[offset:offset + _LIVE_JOB_STRUCT.size] = (
                b"\x00" * _LIVE_JOB_STRUCT.size)
        return {
            "name": self._shm.name,
            "slot": slot,
            "interval_seconds": .01,
        }

    def release(self, config: Mapping[str, Any] | None) -> None:
        if not config:
            return
        slot = int(config["slot"])
        with self._lock:
            self._descriptors.pop(slot, None)
            if not self._closed:
                offset = slot * _LIVE_JOB_STRUCT.size
                self._shm.buf[offset:offset + _LIVE_JOB_STRUCT.size] = (
                    b"\x00" * _LIVE_JOB_STRUCT.size)
            self._free.add(slot)

    def snapshot(self) -> list[dict[str, Any]]:
        rows = []
        with self._lock:
            descriptors = dict(self._descriptors)
        for slot, descriptor in descriptors.items():
            offset = slot * _LIVE_JOB_STRUCT.size
            values = None
            for _ in range(3):
                try:
                    candidate = _LIVE_JOB_STRUCT.unpack_from(
                        self._shm.buf, offset)
                except (BufferError, ValueError):
                    candidate = None
                if candidate is None:
                    break
                sequence_begin = int(candidate[0])
                sequence_end = int(candidate[-1])
                if sequence_begin == sequence_end and sequence_begin != 0:
                    values = candidate
                    break
            if values is None:
                continue
            (
                sequence_begin, state, phase, timestamp_ns, pid,
                rss, peak, start, sequence_end,
            ) = values
            rows.append({
                **descriptor,
                "slot": slot,
                "sequence": sequence_begin,
                "state": int(state),
                "phase": int(phase),
                "timestamp_ns": int(timestamp_ns),
                "pid": int(pid),
                "rss_bytes": int(rss),
                "peak_bytes": int(peak),
                "start_bytes": int(start),
            })
        return rows

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        try:
            self._shm.close()
        except (BufferError, FileNotFoundError):
            pass
        try:
            self._shm.unlink()
        except FileNotFoundError:
            pass


class _JobMemorySampler:
    """Measure a job every 10 ms and optionally publish into shared memory."""

    def __init__(self, interval_seconds: float = .01,
                 live_config: Mapping[str, Any] | None = None) -> None:
        self.interval_seconds = max(.01, float(interval_seconds))
        self.live_config = dict(live_config or {})
        self.start_rss_bytes = 0
        self.peak_rss_bytes = 0
        self.end_rss_bytes = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._live_shm: shared_memory.SharedMemory | None = None
        self._live_sequence = 0

    def _publish(self, state: int, rss: int) -> None:
        if not self.live_config:
            return
        if self._live_shm is None:
            self._live_shm = shared_memory.SharedMemory(
                name=str(self.live_config["name"]), create=False)
        slot = int(self.live_config["slot"])
        offset = slot * _LIVE_JOB_STRUCT.size
        self._live_sequence += 1
        sequence = int(self._live_sequence)
        _LIVE_JOB_STRUCT.pack_into(
            self._live_shm.buf, offset,
            sequence, int(state), 0, int(time.time_ns()),
            int(os.getpid()), int(rss), int(self.peak_rss_bytes),
            int(self.start_rss_bytes), sequence)

    def __enter__(self) -> "_JobMemorySampler":
        try:
            self.start_rss_bytes = _current_process_rss_bytes()
        except (OSError, MemoryError):
            self.start_rss_bytes = 0
        self.peak_rss_bytes = self.start_rss_bytes
        self._publish(_LIVE_JOB_STATE_RUNNING, self.start_rss_bytes)

        def sample() -> None:
            while not self._stop.wait(self.interval_seconds):
                try:
                    rss = _current_process_rss_bytes()
                    self.peak_rss_bytes = max(self.peak_rss_bytes, rss)
                    self._publish(_LIVE_JOB_STATE_RUNNING, rss)
                except (OSError, FileNotFoundError, BufferError, MemoryError):
                    pass

        self._thread = threading.Thread(
            target=sample, name="dqbd-job-rss-10ms", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(.05, self.interval_seconds * 3.0))
        try:
            self.end_rss_bytes = _current_process_rss_bytes()
        except (OSError, MemoryError):
            self.end_rss_bytes = self.peak_rss_bytes
        self.peak_rss_bytes = max(self.peak_rss_bytes, self.end_rss_bytes)
        try:
            self._publish(_LIVE_JOB_STATE_FINISHED, self.end_rss_bytes)
        except (FileNotFoundError, BufferError):
            pass
        if self._live_shm is not None:
            try:
                self._live_shm.close()
            except BufferError:
                pass
            self._live_shm = None

    def telemetry(self, job_class: str) -> dict[str, Any]:
        gib = float(1024 ** 3)
        return {
            "schema_version": "DQBD_JOB_MEMORY_SAMPLE_V2_10MS",
            "job_class": str(job_class),
            "pid": os.getpid(),
            "sample_interval_ms": int(round(self.interval_seconds * 1000)),
            "rss_start_gib": self.start_rss_bytes / gib,
            "rss_peak_gib": self.peak_rss_bytes / gib,
            "rss_end_gib": self.end_rss_bytes / gib,
            "incremental_peak_gib": max(
                0.0, (self.peak_rss_bytes - self.start_rss_bytes) / gib),
        }


class RamAdmissionScheduler:
    """Closed-loop real-load controller with historical profiles as safety."""

    _DEFAULT_ABSOLUTE_GIB = {
        "candidate_oos:HGB:GPU": 6.00,
        "candidate_oos:HGB:CPU": 6.00,
        "candidate_oos:PREP_RIDGE:CPU": 6.50,
        "candidate_oos:RIDGE:GPU": 5.50,
        "candidate_oos:RIDGE:CPU": 5.50,
        "candidate_oos:CPU:CPU": 4.00,
        "causal:model:HGB:GPU": 6.00,
        "causal:model:HGB:CPU": 6.00,
        "causal:model:RIDGE:GPU": 5.50,
        "causal:model:CPU": 5.00,
        "causal:replay:CPU": 2.00,
        "causal:coverage:CPU": 5.00,
        "causal:evidence:CPU": 1.50,
        "causal:CPU:CPU": 2.50,
    }
    _DEFAULT_INCREMENTAL_GIB = {
        "candidate_oos:HGB:GPU": 1.25,
        "candidate_oos:HGB:CPU": 1.50,
        "candidate_oos:PREP_RIDGE:CPU": 3.00,
        "candidate_oos:RIDGE:GPU": 1.00,
        "candidate_oos:RIDGE:CPU": 1.00,
        "candidate_oos:CPU:CPU": .75,
        "causal:model:HGB:GPU": 1.50,
        "causal:model:HGB:CPU": 1.75,
        "causal:model:RIDGE:GPU": 1.00,
        "causal:model:CPU": 1.00,
        "causal:replay:CPU": .45,
        # v40.0.2 measured ~76 GiB of host growth across 32 simultaneous
        # coverage readers. Cold admission therefore starts near the measured
        # ~2.4 GiB/job cost instead of the generic 0.2-GiB evidence estimate.
        "causal:coverage:CPU": 2.75,
        "causal:evidence:CPU": .20,
        "causal:CPU:CPU": .35,
    }
    _DEFAULT_CLASS_LIMITS = {
        "candidate_oos:HGB:GPU": 2,
        "candidate_oos:HGB:CPU": 2,
        "candidate_oos:PREP_RIDGE:CPU": 4,
        # Do not impose a second static concurrency ceiling on coverage.
        # Real-load projection below remains the authority and stops
        # admission in the 90-95% system-memory corridor.  The former limit
        # of eight stranded a 26/32-lane host around 25-30% RAM even when the
        # causal frontier exposed materially more independent horizons.
        "causal:coverage:CPU": 32,
        "causal:model:HGB:GPU": 2,
        "causal:model:HGB:CPU": 2,
    }

    def __init__(self, *, max_slots: int = 26,
                 fill_floor_fraction: float = .80,
                 target_fraction: float = .86,
                 stop_fraction: float = .90,
                 reclaim_fraction: float = .92,
                 hard_limit_fraction: float = .95,
                 reclaim_slope_gib_per_s: float = .25,
                 estimated_task_gib: float = 1.0,
                 profile_path: str | Path | None = None,
                 telemetry_path: str | Path | None = None,
                 telemetry_rollup_seconds: float = 2.0,
                 trend_lookahead_seconds: float = .25,
                 gpu_companion_slots_per_device: int = 4,
                 class_limits: Mapping[str, int] | None = None) -> None:
        if not (
            0 < fill_floor_fraction < target_fraction < stop_fraction
            < reclaim_fraction < hard_limit_fraction <= 1
        ):
            raise ValueError("DQBD_RAM_REAL_LOAD_FRACTIONS_INVALID")
        self._capacity = max(1, int(max_slots))
        self._default_task_gib = float(estimated_task_gib)
        self._fill_floor_fraction = float(fill_floor_fraction)
        self._target_fraction = float(target_fraction)
        self._stop_fraction = float(stop_fraction)
        self._reclaim_fraction = float(reclaim_fraction)
        self._hard_limit_fraction = float(hard_limit_fraction)
        self._reclaim_slope_gib_per_s = max(
            0.0, float(reclaim_slope_gib_per_s))
        self._trend_lookahead_seconds = max(
            .05, min(1.0, float(trend_lookahead_seconds)))
        self._gpu_companion_slots_per_device = max(
            1, int(gpu_companion_slots_per_device))
        total, available = self._system_memory()
        self._total_bytes = int(total)
        self._baseline_rss_bytes = max(0, int(total) - int(available))
        self._class_limits = dict(self._DEFAULT_CLASS_LIMITS)
        if class_limits:
            for key, value in class_limits.items():
                self._class_limits[str(key)] = max(1, int(value))
        self._condition = threading.Condition()
        self._lock = threading.Lock()
        # SHORT/PRIMARY/LONG share this scheduler. Admission projection and
        # reservation must be one atomic parent-side decision or two seed
        # threads can consume the same projected headroom concurrently.
        self._admission_lock = threading.Lock()
        self._profile_write_lock = threading.Lock()
        self._active_slots = 0
        self._reserved_gib = 0.0
        self._active_by_class: dict[str, int] = {}
        self._reserved_by_class: dict[str, float] = {}
        self._gpu_device_leases: dict[int, set[str]] = {}
        self._gpu_device_assignment_count: dict[int, int] = {}
        self._absolute_samples: dict[str, deque[float]] = {}
        self._incremental_samples: dict[str, deque[float]] = {}
        self._absolute_estimates: dict[str, float] = {}
        self._incremental_estimates: dict[str, float] = {}
        self._profile_history = 64
        self._profile_path = (
            Path(profile_path) if profile_path is not None else None)
        self._telemetry_path = (
            Path(telemetry_path) if telemetry_path is not None else None)
        self._telemetry_rollup_seconds = max(
            .5, float(telemetry_rollup_seconds))
        self._last_telemetry_rollup = 0.0
        self._live_board = _LiveJobRamBoard(self._capacity + 8)
        self._load_profiles()
        self._system_samples: deque[tuple[float, int]] = deque(maxlen=512)
        self._live_snapshot: list[dict[str, Any]] = []
        self._peak = 0
        self._admission_denials = 0
        self._pause_count = 0
        self._worker_pause_count = 0
        self._worker_resume_count = 0
        # v40.0.4 reclaim is edge-triggered. Samples from before a successful
        # reclaim may never trigger another victim, otherwise the old 250-ms
        # positive slope survives the memory drop and causes a kill cascade.
        self._reclaim_epoch_monotonic = 0.0
        self._last_reclaim_monotonic = 0.0
        self._settling_until_monotonic = 0.0
        self._recovery_until_monotonic = 0.0
        self._recovery_stable_since: float | None = None
        self._last_recovery_admission_monotonic = 0.0
        self._reclaim_count = 0
        self._reclaim_released_gib = 0.0
        self._last_reclaim_job_id: str | None = None
        self._settling_seconds = .75
        self._recovery_seconds = 5.0
        self._recovery_lock = threading.Lock()
        # One scheduler is shared by SHORT/PRIMARY/LONG and by their CPU/GPU
        # pools. Reclaim must therefore be globally serialized; otherwise two
        # controllers can observe the same 90% edge and each kill one lane
        # before either publishes its post-reclaim slope epoch.
        self._reclaim_gate = threading.Lock()
        self._reclaim_controllers: dict[int, Any] = {}
        self._stop_controller = threading.Event()
        self._controller_thread = threading.Thread(
            target=self._controller_loop,
            name="dqbd-real-load-controller-10ms", daemon=True)
        self._controller_thread.start()
        atexit.register(self.close)

    @staticmethod
    def _system_memory() -> tuple[int, int]:
        if os.name != "nt":
            page = int(os.sysconf("SC_PAGE_SIZE"))
            host_total = page * int(os.sysconf("SC_PHYS_PAGES"))
            host_available = page * int(os.sysconf("SC_AVPHYS_PAGES"))
            limit, current = _posix_cgroup_memory()
            if limit is not None:
                total = min(host_total, int(limit))
                if current is not None:
                    return total, max(0, total - min(total, int(current)))
                return total, min(host_available, total)
            return host_total, host_available
        import ctypes

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong),
                        ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        status = MEMORYSTATUSEX()
        status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            raise OSError("GLOBAL_MEMORY_STATUS_EX_FAILED")
        return int(status.ullTotalPhys), int(status.ullAvailPhys)

    def _controller_loop(self) -> None:
        while not self._stop_controller.wait(.01):
            try:
                total, available = self._system_memory()
                used = max(0, total - available)
                live = self._live_board.snapshot()
            except Exception:
                continue
            now = time.monotonic()
            with self._lock:
                self._system_samples.append((now, int(used)))
                self._live_snapshot = live
                self._peak = max(self._peak, int(used))
            if (
                self._telemetry_path is not None
                and now - self._last_telemetry_rollup
                    >= self._telemetry_rollup_seconds
            ):
                self._write_runtime_rollup(now)
                self._last_telemetry_rollup = now
            with self._condition:
                self._condition.notify_all()

    def _write_runtime_rollup(self, now: float) -> None:
        """Persist a compact rollup while retaining 10-ms control internally."""
        cutoff = now - self._telemetry_rollup_seconds
        with self._lock:
            points = [
                int(used) for timestamp, used in self._system_samples
                if timestamp >= cutoff]
            live = [dict(row) for row in self._live_snapshot]
            active_by_class = dict(self._active_by_class)
            reserved = float(self._reserved_gib)
        if not points or self._telemetry_path is None:
            return
        fractions = [
            float(value) / max(1, self._total_bytes)
            for value in points]
        live_classes: dict[str, int] = {}
        for row in live:
            if row.get("state") != _LIVE_JOB_STATE_RUNNING:
                continue
            key = str(row.get("job_class", "UNKNOWN"))
            live_classes[key] = live_classes.get(key, 0) + 1
        payload = {
            "schema_version": "DQBD_REAL_LOAD_ROLLUP_V40_0_4",
            "timestamp": time.time(),
            "source_sample_interval_ms": 10,
            "rollup_seconds": self._telemetry_rollup_seconds,
            "sample_count": len(fractions),
            "used_fraction_min": min(fractions),
            "used_fraction_mean": sum(fractions) / len(fractions),
            "used_fraction_max": max(fractions),
            "used_gib_current": points[-1] / float(1024 ** 3),
            "ram_slope_gib_per_s": self._ram_slope_gib_per_s(),
            "live_job_count": sum(live_classes.values()),
            "live_job_classes": live_classes,
            "active_by_job_class": active_by_class,
            "reserved_incremental_gib": reserved,
            "unrealized_live_gib": self._unrealized_live_gib(),
            "unpublished_lease_gib": self._unpublished_lease_gib(),
        }
        self._telemetry_path.parent.mkdir(parents=True, exist_ok=True)
        with self._telemetry_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(payload, sort_keys=True, default=str) + "\n")

    def close(self) -> None:
        if self._stop_controller.is_set():
            return
        self._stop_controller.set()
        if self._controller_thread.is_alive():
            self._controller_thread.join(timeout=.2)
        self._live_board.close()

    def _latest_used_bytes(self) -> int:
        with self._lock:
            if self._system_samples:
                return int(self._system_samples[-1][1])
        total, available = self._system_memory()
        return max(0, total - available)

    def _ram_slope_gib_per_s(self, window_seconds: float = .25) -> float:
        now = time.monotonic()
        with self._lock:
            epoch = float(self._reclaim_epoch_monotonic)
            points = [
                (t, used) for t, used in self._system_samples
                if now - t <= window_seconds and t >= epoch]
        if len(points) < 2:
            return 0.0
        dt = points[-1][0] - points[0][0]
        if dt <= 0:
            return 0.0
        return (
            (float(points[-1][1]) - float(points[0][1]))
            / float(1024 ** 3) / dt)

    def real_load_state(self) -> dict[str, Any]:
        used = self._latest_used_bytes()
        with self._lock:
            live = [dict(row) for row in self._live_snapshot]
        return {
            "used_bytes": used,
            "used_fraction": float(used) / max(1, self._total_bytes),
            "slope_gib_per_s": self._ram_slope_gib_per_s(),
            "live_jobs": live,
            "live_job_count": sum(
                1 for row in live
                if row.get("state") == _LIVE_JOB_STATE_RUNNING),
        }

    def reclaim_state(self) -> dict[str, Any]:
        """Return one edge-triggered reclaim decision from post-reclaim data."""
        state = self.real_load_state()
        usage = float(state["used_fraction"])
        slope = float(state["slope_gib_per_s"])
        now = time.monotonic()
        with self._lock:
            settling_until = float(self._settling_until_monotonic)
            last_reclaim = float(self._last_reclaim_monotonic)
        settling = now < settling_until
        # The normal 92% hard-reclaim boundary is still one-victim-at-a-time.
        # Only a near-Job-Object emergency may break the 750-ms settling gate.
        emergency = usage >= min(
            .945, self._hard_limit_fraction - .005)
        min_interval_clear = (
            last_reclaim <= 0.0 or now - last_reclaim >= .35)
        hard = usage >= self._reclaim_fraction
        rising = (
            usage >= self._stop_fraction
            and slope >= self._reclaim_slope_gib_per_s)
        required = (
            (hard or rising)
            and min_interval_clear
            and (not settling or emergency))
        return {
            **state,
            "reclaim_required": bool(required),
            "reclaim_hard": bool(hard),
            "reclaim_rising": bool(rising),
            "reclaim_settling": bool(settling),
            "reclaim_emergency": bool(emergency),
            "reclaim_epoch_monotonic": self._reclaim_epoch_monotonic,
        }

    def note_targeted_reclaim(
        self, *, job_id: str, released_gib: float,
        usage_fraction: float,
    ) -> None:
        """Reset slope history and enter settling/recovery after one victim."""
        now = time.monotonic()
        with self._lock:
            self._reclaim_epoch_monotonic = now
            self._last_reclaim_monotonic = now
            self._settling_until_monotonic = now + self._settling_seconds
            self._recovery_until_monotonic = now + self._recovery_seconds
            self._recovery_stable_since = None
            self._reclaim_count += 1
            self._reclaim_released_gib += max(0.0, float(released_gib))
            self._last_reclaim_job_id = str(job_id)
        with self._recovery_lock:
            self._last_recovery_admission_monotonic = now

    def register_reclaim_controller(self, controller: Any) -> None:
        with self._lock:
            self._reclaim_controllers[id(controller)] = controller

    def unregister_reclaim_controller(self, controller: Any) -> None:
        with self._lock:
            self._reclaim_controllers.pop(id(controller), None)

    def reclaim_controllers(self) -> tuple[Any, ...]:
        with self._lock:
            return tuple(self._reclaim_controllers.values())

    def try_begin_targeted_reclaim(self) -> dict[str, Any] | None:
        """Acquire the one global reclaim decision for all seed-local pools."""
        if not self._reclaim_gate.acquire(blocking=False):
            return None
        try:
            state = self.reclaim_state()
            if not state["reclaim_required"]:
                self._reclaim_gate.release()
                return None
            return state
        except BaseException:
            self._reclaim_gate.release()
            raise

    def end_targeted_reclaim(self) -> None:
        self._reclaim_gate.release()

    def recovery_state(self) -> dict[str, Any]:
        now = time.monotonic()
        with self._lock:
            return {
                "settling": now < self._settling_until_monotonic,
                "recovering": now < self._recovery_until_monotonic,
                "settling_seconds_remaining": max(
                    0.0, self._settling_until_monotonic - now),
                "recovery_seconds_remaining": max(
                    0.0, self._recovery_until_monotonic - now),
                "last_reclaim_job_id": self._last_reclaim_job_id,
                "reclaim_count": self._reclaim_count,
            }

    def admission_budget(
        self, used_fraction: float | None = None,
    ) -> int:
        """Return a feedback budget with slow refill after one RAM reclaim."""
        # A different seed/pool may currently be terminating the global
        # victim. Do not interpret that transient memory drop as spare
        # capacity until note_targeted_reclaim() has armed recovery.
        if self._reclaim_gate.locked():
            return 0
        state = self.real_load_state()
        used = float(
            state["used_fraction"]
            if used_fraction is None else used_fraction)
        slope = float(state["slope_gib_per_s"])
        now = time.monotonic()
        with self._lock:
            settling_until = float(self._settling_until_monotonic)
            recovery_until = float(self._recovery_until_monotonic)
            stable_since = self._recovery_stable_since
        if now < settling_until:
            return 0
        recovering = now < recovery_until
        if recovering:
            stable = (
                .82 <= used <= .88
                and abs(slope) <= self._reclaim_slope_gib_per_s)
            with self._lock:
                if stable:
                    if stable_since is None:
                        self._recovery_stable_since = now
                        stable_since = now
                    elif now - stable_since >= 2.0:
                        self._recovery_until_monotonic = now
                        recovering = False
                else:
                    self._recovery_stable_since = None
            if recovering:
                # Refill one lane at a time and stop feeding above 87% until
                # the post-reclaim system has demonstrated a stable corridor.
                if used >= .87:
                    return 0
                interval = .10 if used < .80 else (
                    .15 if used < .84 else .25)
                with self._recovery_lock:
                    if (
                        now - self._last_recovery_admission_monotonic
                        < interval
                    ):
                        return 0
                    self._last_recovery_admission_monotonic = now
                return 1
        return 8 if used < .60 else (4 if used < .80 else 2)

    def wait_for_reclaim_clear(self, timeout_seconds: float = 30.0) -> bool:
        """Wait briefly for terminated workers/working sets to leave RAM."""
        deadline = time.monotonic() + max(.1, float(timeout_seconds))
        while time.monotonic() < deadline:
            if self.system_usage_fraction() < self._target_fraction:
                return True
            time.sleep(.05)
        return self.system_usage_fraction() < self._stop_fraction

    def available_gpu_devices(
        self, device_count: int,
        *, exclude: Iterable[int] = (),
    ) -> tuple[int, ...]:
        """Return devices with room for staged GPU-producing futures.

        The actual OpenCL/LightGBM section remains serialized by the
        cross-process v40.0.3 ready-section queue and per-device lock. Parent
        staging capacity exists only to keep enough CPU-prepared work close to
        each GPU; it does not authorize concurrent same-device kernels.
        """
        excluded = {int(value) for value in exclude}
        with self._lock:
            return tuple(
                index for index in range(max(0, int(device_count)))
                if index not in excluded
                and len(self._gpu_device_leases.get(index, set()))
                    < self._gpu_companion_slots_per_device)

    def gpu_device_assignment_count(self, device_index: int) -> int:
        with self._lock:
            return int(
                self._gpu_device_assignment_count.get(
                    int(device_index), 0))

    def try_acquire_gpu_device(
        self, device_index: int, job_id: str,
    ) -> bool:
        index = int(device_index)
        identity = str(job_id)
        with self._lock:
            owners = self._gpu_device_leases.setdefault(index, set())
            if identity in owners:
                return True
            if len(owners) >= self._gpu_companion_slots_per_device:
                return False
            owners.add(identity)
            self._gpu_device_assignment_count[index] = (
                self._gpu_device_assignment_count.get(index, 0) + 1)
        return True

    def release_gpu_device(
        self, device_index: int | None, job_id: str | None = None,
    ) -> None:
        if device_index is None:
            return
        index = int(device_index)
        identity = str(job_id) if job_id is not None else None
        released = False
        with self._lock:
            owners = self._gpu_device_leases.get(index)
            if owners:
                if identity is None:
                    owners.clear()
                    released = True
                elif identity in owners:
                    owners.remove(identity)
                    released = True
                if not owners:
                    self._gpu_device_leases.pop(index, None)
        if released:
            with self._condition:
                self._condition.notify_all()

    @staticmethod
    def _descriptor_profile_key(
        job_class: str, descriptor: Mapping[str, Any] | None,
    ) -> str:
        descriptor = dict(descriptor or {})
        horizon = descriptor.get("horizon")
        h_bucket = (
            f"H{((int(horizon) - 1) // 5) * 5 + 1:02d}_"
            f"{min(30, ((int(horizon) - 1) // 5) * 5 + 5):02d}"
            if horizon else "HNA")
        prepared = "P1" if descriptor.get("prepared") else "P0"
        device = descriptor.get("device_index")
        device_bucket = (
            f"D{int(device)}" if device is not None else "DCPU")
        size_value = (
            descriptor.get("train_sessions")
            or descriptor.get("segment_sessions")
            or descriptor.get("prediction_sessions")
            or 0)
        size_value = max(0, int(size_value or 0))
        if size_value <= 0:
            size_bucket = "SNA"
        elif size_value <= 504:
            size_bucket = "S0001_0504"
        elif size_value <= 1008:
            size_bucket = "S0505_1008"
        elif size_value <= 1512:
            size_bucket = "S1009_1512"
        elif size_value <= 2016:
            size_bucket = "S1513_2016"
        else:
            size_bucket = "S2017_PLUS"
        return (
            f"{job_class}|{h_bucket}|{prepared}|"
            f"{device_bucket}|{size_bucket}")

    def _load_profiles(self) -> None:
        if self._profile_path is None or not self._profile_path.is_file():
            return
        try:
            payload = json.loads(
                self._profile_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        if payload.get("schema_version") not in {
            "DQBD_JOB_MEMORY_PROFILES_V2",
            "DQBD_JOB_MEMORY_PROFILES_V3_REAL_LOAD",
        }:
            return
        absolute = payload.get("absolute_samples") or payload.get("samples") or {}
        incremental = payload.get("incremental_samples") or {}
        for target, source in (
            (self._absolute_samples, absolute),
            (self._incremental_samples, incremental),
        ):
            for key, values in source.items():
                valid = [
                    float(value) for value in values
                    if isinstance(value, (int, float))
                    and 0 <= float(value) <= 32.0]
                target[str(key)] = deque(
                    valid[-self._profile_history:],
                    maxlen=self._profile_history)
        self._rebuild_estimates()

    def _rebuild_estimates(self) -> None:
        def p90(samples: deque[float], fallback: float) -> float:
            if not samples:
                return fallback
            ordered = sorted(samples)
            index = min(
                len(ordered) - 1,
                max(0, int(round(.90 * (len(ordered) - 1)))))
            return float(ordered[index])

        keys = set(self._absolute_samples) | set(self._incremental_samples)
        for key in keys:
            job_class = key.split("|", 1)[0]
            self._absolute_estimates[key] = p90(
                self._absolute_samples.get(key, deque()),
                self._DEFAULT_ABSOLUTE_GIB.get(
                    job_class, self._default_task_gib))
            self._incremental_estimates[key] = p90(
                self._incremental_samples.get(key, deque()),
                self._DEFAULT_INCREMENTAL_GIB.get(
                    job_class, self._default_task_gib))

    def _persist_profiles(self) -> None:
        if self._profile_path is None:
            return
        # Observations may arrive concurrently from SHORT/PRIMARY/LONG.
        # Serialize publication so an older snapshot cannot replace a newer
        # profile file merely because its disk write completed later.
        with self._profile_write_lock:
            with self._lock:
                payload = {
                    "schema_version":
                        "DQBD_JOB_MEMORY_PROFILES_V3_REAL_LOAD",
                    "sample_interval_ms": 10,
                    "absolute_samples": {
                        key: list(values)
                        for key, values
                        in self._absolute_samples.items()},
                    "incremental_samples": {
                        key: list(values)
                        for key, values
                        in self._incremental_samples.items()},
                    "absolute_estimates_gib":
                        dict(self._absolute_estimates),
                    "incremental_estimates_gib":
                        dict(self._incremental_estimates),
                    "updated_at": time.time(),
                }
            self._profile_path.parent.mkdir(
                parents=True, exist_ok=True)
            temporary = self._profile_path.with_name(
                self._profile_path.name
                + f".{os.getpid()}.{threading.get_ident()}."
                + f"{time.time_ns()}.tmp")
            temporary.write_text(
                json.dumps(
                    payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8")
            os.replace(temporary, self._profile_path)

    def observe(self, job_class: str | None,
                peak_process_gib: float | None,
                incremental_peak_gib: float | None = None,
                descriptor: Mapping[str, Any] | None = None) -> None:
        if not job_class:
            return
        key = self._descriptor_profile_key(str(job_class), descriptor)
        with self._lock:
            if peak_process_gib is not None:
                value = float(peak_process_gib)
                if .25 <= value <= 32.0:
                    self._absolute_samples.setdefault(
                        key, deque(maxlen=self._profile_history)).append(value)
            if incremental_peak_gib is not None:
                value = float(incremental_peak_gib)
                if 0 <= value <= 32.0:
                    self._incremental_samples.setdefault(
                        key, deque(maxlen=self._profile_history)).append(value)
            self._rebuild_estimates()
        self._persist_profiles()

    def absolute_estimate_for(
        self, job_class: str,
        descriptor: Mapping[str, Any] | None = None,
    ) -> float:
        key = self._descriptor_profile_key(job_class, descriptor)
        with self._lock:
            value = self._absolute_estimates.get(key)
        return float(
            value if value is not None
            else self._DEFAULT_ABSOLUTE_GIB.get(
                job_class, self._default_task_gib))

    def incremental_estimate_for(
        self, job_class: str,
        descriptor: Mapping[str, Any] | None = None,
    ) -> float:
        key = self._descriptor_profile_key(job_class, descriptor)
        with self._lock:
            value = self._incremental_estimates.get(key)
        cold_default = float(
            self._DEFAULT_INCREMENTAL_GIB.get(
                job_class, self._default_task_gib))
        return max(
            .25 * cold_default,
            float(value if value is not None else cold_default))

    def estimate_for(self, job_class: str | None,
                     fallback_gib: float | None = None) -> float:
        if not job_class:
            return float(fallback_gib or self._default_task_gib)
        return self.incremental_estimate_for(str(job_class))

    def open_live_job(
        self, job_class: str, descriptor: Mapping[str, Any],
    ) -> dict[str, Any]:
        return self._live_board.allocate({
            "job_class": str(job_class),
            **dict(descriptor),
        })

    def close_live_job(self, config: Mapping[str, Any] | None) -> None:
        self._live_board.release(config)

    def _live_expected_incremental_gib(self) -> float:
        with self._lock:
            live = [dict(row) for row in self._live_snapshot]
        return sum(
            self.incremental_estimate_for(
                str(row.get("job_class", "")), row)
            for row in live if row.get("job_class")
        )

    def _unrealized_live_gib(self) -> float:
        """Return expected growth not yet visible in live worker RSS."""
        gib = float(1024 ** 3)
        with self._lock:
            live = [dict(row) for row in self._live_snapshot]
        unrealized = 0.0
        for row in live:
            job_class = str(row.get("job_class", ""))
            if not job_class:
                continue
            expected = self.incremental_estimate_for(job_class, row)
            start = max(0, int(row.get("start_bytes", 0)))
            current = max(0, int(row.get("rss_bytes", 0)))
            observed = max(0.0, (current - start) / gib)
            unrealized += max(0.0, expected - observed)
        return unrealized

    def _unpublished_lease_gib(self) -> float:
        """Cover the tiny lease->shared-slot race across seed scheduler threads."""
        with self._lock:
            reserved = float(self._reserved_gib)
        return max(
            0.0, reserved - self._live_expected_incremental_gib())

    def _safety_projected_fraction(
        self, new_incremental_gib: float,
    ) -> float:
        used = self._latest_used_bytes()
        projected = used + int(
            (
                self._unrealized_live_gib()
                + self._unpublished_lease_gib()
                + float(new_incremental_gib)
            ) * (1024 ** 3))
        return float(projected) / max(1, self._total_bytes)

    def _class_capacity_available(self, key: str) -> bool:
        with self._lock:
            return (
                self._active_slots < self._capacity
                and self._active_by_class.get(key, 0)
                < int(self._class_limits.get(key, self._capacity)))

    def try_acquire(self, estimated_task_gib: float | None = None,
                    wait_callback: Callable[[], None] | None = None,
                    job_class: str | None = None,
                    descriptor: Mapping[str, Any] | None = None) -> float | None:
        if wait_callback is not None:
            wait_callback()
        if self._reclaim_gate.locked():
            with self._lock:
                self._admission_denials += 1
            return None
        key = str(job_class or "default")
        # Projection + lease publication are atomic across all seed scheduler
        # threads. This does not serialize worker execution; it only prevents
        # two concurrent callers from spending the same RAM headroom.
        with self._admission_lock:
            # Close the race between the fast pre-check above and a reclaim
            # edge acquired by another seed thread before reservation publish.
            if self._reclaim_gate.locked():
                with self._lock:
                    self._admission_denials += 1
                return None
            if not self._class_capacity_available(key):
                return None
            state = self.real_load_state()
            used_fraction = float(state["used_fraction"])
            slope = float(state["slope_gib_per_s"])
            recovery = self.recovery_state()
            if recovery["settling"] or (
                recovery["recovering"] and used_fraction >= .87
            ):
                with self._lock:
                    self._admission_denials += 1
                return None
            incremental = (
                self.incremental_estimate_for(key, descriptor)
                if job_class else float(
                    estimated_task_gib or self._default_task_gib))
            safety_projection = self._safety_projected_fraction(incremental)
            trend_guard_gib = (
                max(0.0, slope) * self._trend_lookahead_seconds)
            trend_guard_fraction = (
                trend_guard_gib * (1024 ** 3) / self._total_bytes)
            safety_with_trend = safety_projection + trend_guard_fraction

            allow = False
            if used_fraction < self._fill_floor_fraction:
                # Underfilled: aggressively admit work. A rising RAM slope is
                # converted into a short look-ahead reservation instead of a
                # fixed GiB/s veto, so a brief allocator burst cannot strand
                # an otherwise 30-70%-loaded host.
                allow = safety_with_trend < self._hard_limit_fraction
            elif used_fraction < self._target_fraction:
                allow = safety_with_trend < self._hard_limit_fraction
            elif used_fraction < self._stop_fraction:
                # Above target, the complete projected load is the stop-band
                # authority: current real RAM + outstanding unrealized leases
                # + this job + short positive RAM trend.
                allow = (
                    safety_with_trend <= self._stop_fraction
                    and safety_with_trend < self._hard_limit_fraction)
            if not allow:
                with self._lock:
                    self._admission_denials += 1
                return None

            with self._condition:
                if not self._class_capacity_available(key):
                    return None
                self._active_slots += 1
                self._reserved_gib += incremental
                self._active_by_class[key] = (
                    self._active_by_class.get(key, 0) + 1)
                self._reserved_by_class[key] = (
                    self._reserved_by_class.get(key, 0.0) + incremental)
                return incremental

    def acquire(self, estimated_task_gib: float | None = None,
                wait_callback: Callable[[], None] | None = None,
                job_class: str | None = None,
                descriptor: Mapping[str, Any] | None = None) -> float:
        while True:
            reservation = self.try_acquire(
                estimated_task_gib,
                wait_callback=wait_callback,
                job_class=job_class,
                descriptor=descriptor)
            if reservation is not None:
                return reservation
            with self._condition:
                self._pause_count += 1
                self._condition.wait(timeout=.01)

    def release(self, estimated_task_gib: float | None = None,
                job_class: str | None = None) -> None:
        reservation = float(estimated_task_gib or self._default_task_gib)
        key = str(job_class or "default")
        with self._condition:
            self._active_slots = max(0, self._active_slots - 1)
            self._reserved_gib = max(0.0, self._reserved_gib - reservation)
            if key in self._active_by_class:
                self._active_by_class[key] = max(
                    0, self._active_by_class[key] - 1)
            if key in self._reserved_by_class:
                self._reserved_by_class[key] = max(
                    0.0, self._reserved_by_class[key] - reservation)
            self._condition.notify_all()

    def priority_key(
        self, job_class: str, descriptor: Mapping[str, Any] | None = None,
        *, gpu_idle: bool = False,
    ) -> tuple[float, float, float]:
        """Return a sortable key for content-aware frontier packing."""
        state = self.real_load_state()
        used = float(state["used_fraction"])
        incremental = self.incremental_estimate_for(
            job_class, descriptor)
        gpu_bonus = -1000.0 if gpu_idle else 0.0
        recovery = self.recovery_state()
        if recovery["recovering"]:
            # During refill, fit toward the 86% target instead of selecting
            # the largest available allocation simply because RAM is low.
            headroom_gib = max(
                0.0,
                (self._target_fraction - used)
                * self._total_bytes / float(1024 ** 3))
            return (
                gpu_bonus, abs(headroom_gib - incremental), incremental)
        if used < self._fill_floor_fraction:
            # Normal underfill: GPU starvation first, then larger work.
            return (gpu_bonus, -incremental, 0.0)
        headroom_gib = max(
            0.0,
            (self._target_fraction - used)
            * self._total_bytes / float(1024 ** 3))
        if used < self._target_fraction:
            return (gpu_bonus, abs(headroom_gib - incremental), incremental)
        # Upper band: best-fit small work first.
        return (gpu_bonus, incremental, 0.0)

    def max_admitted_slots(self, estimated_task_gib: float | None = None,
                           job_class: str | None = None) -> int:
        class_limit = int(
            self._class_limits.get(str(job_class or "default"), self._capacity))
        return max(1, min(self._capacity, class_limit))

    def telemetry(self) -> dict[str, Any]:
        state = self.real_load_state()
        with self._lock:
            absolute = dict(self._absolute_estimates)
            incremental = dict(self._incremental_estimates)
            active_by_class = dict(self._active_by_class)
            reserved_by_class = dict(self._reserved_by_class)
            peak = int(self._peak)
            denials = int(self._admission_denials)
            pauses = int(self._pause_count)
            worker_pauses = int(self._worker_pause_count)
            worker_resumes = int(self._worker_resume_count)
            reclaim_count = int(self._reclaim_count)
            reclaim_released_gib = float(self._reclaim_released_gib)
            last_reclaim_job_id = self._last_reclaim_job_id
            reserved = float(self._reserved_gib)
            active_slots = int(self._active_slots)
            gpu_device_leases = {
                index: sorted(owners)
                for index, owners in self._gpu_device_leases.items()}
            gpu_device_assignment_count = dict(
                self._gpu_device_assignment_count)
        return {
            "ram_scheduler_contract":
                "DQBD_SINGLE_LANE_RECLAIM_CLOSED_LOOP_V40_0_4",
            "ram_scheduler_sample_interval_ms": 10,
            "ram_scheduler_total_system_gib":
                self._total_bytes / (1024 ** 3),
            "ram_scheduler_current_used_gib":
                state["used_bytes"] / (1024 ** 3),
            "ram_scheduler_current_used_fraction":
                state["used_fraction"],
            "ram_scheduler_ram_slope_gib_per_s":
                state["slope_gib_per_s"],
            "ram_scheduler_trend_lookahead_seconds":
                self._trend_lookahead_seconds,
            "ram_scheduler_live_job_count":
                state["live_job_count"],
            "ram_scheduler_live_jobs": state["live_jobs"],
            "ram_scheduler_fill_floor_fraction":
                self._fill_floor_fraction,
            "ram_scheduler_target_fraction":
                self._target_fraction,
            "ram_scheduler_stop_fraction":
                self._stop_fraction,
            "ram_scheduler_reclaim_fraction":
                self._reclaim_fraction,
            "ram_scheduler_reclaim_slope_gib_per_s":
                self._reclaim_slope_gib_per_s,
            "ram_scheduler_emergency_throttle_fraction":
                self._stop_fraction,
            "ram_scheduler_hard_limit_fraction":
                self._hard_limit_fraction,
            "ram_scheduler_unrealized_live_gib":
                self._unrealized_live_gib(),
            "ram_scheduler_unpublished_lease_gib":
                self._unpublished_lease_gib(),
            "ram_scheduler_capacity": self._capacity,
            "ram_scheduler_active_slots": active_slots,
            "ram_scheduler_reserved_incremental_gib": reserved,
            "ram_scheduler_active_by_job_class": active_by_class,
            "ram_scheduler_reserved_by_job_class_gib":
                reserved_by_class,
            "ram_scheduler_gpu_device_leases": gpu_device_leases,
            "ram_scheduler_gpu_device_assignment_count":
                gpu_device_assignment_count,
            "ram_scheduler_absolute_profile_gib": absolute,
            "ram_scheduler_incremental_profile_gib": incremental,
            "ram_scheduler_profile_path": (
                str(self._profile_path)
                if self._profile_path is not None else None),
            "ram_scheduler_runtime_telemetry_path": (
                str(self._telemetry_path)
                if self._telemetry_path is not None else None),
            "ram_scheduler_incremental_cold_floor_fraction": .25,
            "ram_scheduler_peak_system_gib":
                peak / (1024 ** 3),
            "ram_scheduler_admission_denial_count": denials,
            "ram_scheduler_pause_count": pauses,
            "ram_scheduler_worker_pause_count": worker_pauses,
            "ram_scheduler_worker_resume_count": worker_resumes,
            "ram_scheduler_targeted_reclaim_count": reclaim_count,
            "ram_scheduler_targeted_reclaim_released_gib":
                reclaim_released_gib,
            "ram_scheduler_last_reclaim_job_id": last_reclaim_job_id,
            "ram_scheduler_reclaim_in_progress":
                self._reclaim_gate.locked(),
            "ram_scheduler_registered_reclaim_pools":
                len(self.reclaim_controllers()),
            "ram_scheduler_recovery": self.recovery_state(),
            "ram_scheduler_admission_policy":
                "REAL_LOAD_10MS_DAMPED_RECOVERY_80_86_90_92_95_V40_0_4",
            "ram_scheduler_safety_policy":
                "SINGLE_LANE_BEST_FIT_RECLAIM_SETTLING_V40_0_4",
        }

    def system_usage_fraction(self) -> float:
        return float(self.real_load_state()["used_fraction"])

    @property
    def _soft_fraction(self) -> float:
        # Compatibility for RamWorkerPauseController: resume below target.
        return self._target_fraction

    @property
    def _hard_fraction(self) -> float:
        # Compatibility for RamWorkerPauseController: emergency pressure.
        return self._stop_fraction


    def note_worker_paused(self) -> None:
        with self._lock:
            self._worker_pause_count += 1

    def note_worker_resumed(self) -> None:
        with self._lock:
            self._worker_resume_count += 1



class RamWorkerPauseController:
    """Targeted RAM reclaim controller for independently replaceable lanes.

    The historical name is retained for checkpoint/code compatibility. v40.0.4
    no longer suspends the largest worker and never raises a pool-wide reclaim
    exception. One progress-aware best-fit lane is reclaimed, the RAM slope
    epoch is reset, and the controller enters a settling interval before it can
    select another victim.
    """

    def __init__(
        self, scheduler: RamAdmissionScheduler, pool, telemetry_root: Path,
    ) -> None:
        self.scheduler = scheduler
        self.pool = pool
        self.telemetry_path = (
            Path(telemetry_root) / "ram-pause-events.jsonl")
        self.spill_root = Path(telemetry_root).parent / "ram-spill"
        self.spill_path = self.spill_root / (
            f"ram-controller-{os.getpid()}-{id(pool):x}.json")
        self.paused: dict[int, dict[str, Any]] = {}
        self._last_pressure_spill_at = 0.0
        self.scheduler.register_reclaim_controller(self)

    def _record(self, payload: dict[str, Any]) -> None:
        self.telemetry_path.parent.mkdir(parents=True, exist_ok=True)
        with self.telemetry_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(payload, sort_keys=True, default=str) + "\n")

    def _spill(self, *, event: str, usage: float) -> None:
        """Persist controller/lane state before or after one reclaim."""
        self.spill_root.mkdir(parents=True, exist_ok=True)
        processes = []
        for process in self._processes(self.pool):
            pid = getattr(process, "pid", None)
            if not pid:
                continue
            try:
                rss = self._rss_bytes(int(pid))
            except OSError:
                rss = None
            processes.append({
                "pid": int(pid),
                "rss_bytes": rss,
                "paused": False,
            })
        payload = {
            "schema_version": "DQBD_RAM_SPILL_RESUME_V2_SINGLE_LANE",
            "event": event,
            "timestamp": time.time(),
            "controller_pid": os.getpid(),
            "system_usage_fraction": float(usage),
            "processes": processes,
            "active_lanes": (
                self.pool.active_lanes()
                if hasattr(self.pool, "active_lanes") else []),
            "scheduler": (
                self.scheduler.telemetry()
                if hasattr(self.scheduler, "telemetry") else {}),
            "evaluation_not_opened": True,
        }
        temporary = self.spill_path.with_name(
            self.spill_path.name + ".tmp")
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8")
        os.replace(temporary, self.spill_path)

    @staticmethod
    def _processes(pool) -> list[Any]:
        return list(getattr(pool, "_processes", {}).values())

    @staticmethod
    def _rss_bytes(pid: int) -> int:
        if os.name == "nt":
            import ctypes
            from ctypes import wintypes

            class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            psapi = ctypes.WinDLL("psapi", use_last_error=True)
            handle = kernel32.OpenProcess(0x0410, False, int(pid))
            if not handle:
                raise OSError(
                    ctypes.get_last_error(),
                    "DQBD_OPEN_PROCESS_FOR_RAM_FAILED")
            try:
                counters = PROCESS_MEMORY_COUNTERS()
                counters.cb = ctypes.sizeof(counters)
                if not psapi.GetProcessMemoryInfo(
                    handle, ctypes.byref(counters), counters.cb
                ):
                    raise OSError(
                        ctypes.get_last_error(),
                        "DQBD_GET_PROCESS_RAM_FAILED")
                return int(counters.WorkingSetSize)
            finally:
                kernel32.CloseHandle(handle)
        statm = Path(f"/proc/{int(pid)}/statm")
        return (
            int(statm.read_text().split()[1])
            * int(os.sysconf("SC_PAGE_SIZE")))

    def _choose_reclaim_victim(
        self, reclaim: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        if not hasattr(self.pool, "active_lanes"):
            return None
        lanes = list(self.pool.active_lanes())
        if not lanes:
            return None
        live_rows = [
            dict(row) for row in reclaim.get("live_jobs", ())
            if row.get("state") == _LIVE_JOB_STATE_RUNNING]
        by_identity = {
            (str(row.get("job_id", "")), int(row.get("pid", -1))): row
            for row in live_rows
            if row.get("pid") is not None}
        by_job = {
            str(row.get("job_id", "")): row
            for row in live_rows}
        usage = float(reclaim["used_fraction"])
        total_gib = (
            self.scheduler._total_bytes / float(1024 ** 3))
        required_gib = max(
            .25,
            (usage - self.scheduler._target_fraction) * total_gib)
        now = time.time()
        candidates = []
        for lane in lanes:
            job_id = str(lane.get("job_id", ""))
            pid_value = lane.get("pid")
            if not job_id or pid_value is None:
                continue
            pid = int(pid_value)
            row = (
                by_identity.get((job_id, pid))
                or by_job.get(job_id)
                or {})
            rss_bytes = int(row.get("rss_bytes", 0) or 0)
            if rss_bytes <= 0:
                try:
                    rss_bytes = self._rss_bytes(pid)
                except OSError:
                    continue
            start_bytes = int(row.get("start_bytes", 0) or 0)
            release_gib = rss_bytes / float(1024 ** 3)
            incremental_gib = max(
                0.0, rss_bytes - start_bytes) / float(1024 ** 3)
            # A fresh replacement lane recreates its imported baseline. For
            # control purposes the sustainable relief is therefore the job's
            # incremental working set, not the whole process RSS. If a sampler
            # has no valid start sample, fall back to the full RSS.
            reclaimable_gib = (
                incremental_gib
                if start_bytes > 0 and incremental_gib >= .05
                else release_gib)
            age_seconds = max(
                0.0, now - float(lane.get("submitted_at", now)))
            reclaim_count = int(
                lane.get("ram_reclaim_count", 0) or 0)
            # Protect already-reclaimed and older work first. GPU work also
            # receives a penalty because its restart cost is usually higher.
            protected_penalty = min(3, max(0, reclaim_count))
            age_penalty = (
                2 if age_seconds >= 30.0
                else (1 if age_seconds >= 15.0 else 0))
            gpu_penalty = (
                1 if lane.get("gpu_device_index") is not None else 0)
            fit_error = abs(reclaimable_gib - required_gib)
            overshoot = max(0.0, reclaimable_gib - required_gib)
            score = (
                protected_penalty,
                age_penalty,
                gpu_penalty,
                fit_error + .35 * overshoot,
                -reclaimable_gib,
                job_id,
            )
            candidates.append({
                **lane,
                "rss_bytes": rss_bytes,
                "release_gib": release_gib,
                "incremental_gib": incremental_gib,
                "reclaimable_gib": reclaimable_gib,
                "age_seconds": age_seconds,
                "required_reclaim_gib": required_gib,
                "score": score,
            })
        if not candidates:
            return None
        return min(candidates, key=lambda value: value["score"])

    def reconcile(self) -> None:
        preview = self.scheduler.reclaim_state()
        usage = float(preview["used_fraction"])
        now = time.time()
        if (
            usage > self.scheduler._stop_fraction
            and now - self._last_pressure_spill_at >= 1.0
        ):
            self._spill(event="RAM_PRESSURE", usage=usage)
            self._last_pressure_spill_at = now
        if not preview["reclaim_required"]:
            return

        reclaim = self.scheduler.try_begin_targeted_reclaim()
        if reclaim is None:
            return
        try:
            usage = float(reclaim["used_fraction"])
            choices = []
            for controller in self.scheduler.reclaim_controllers():
                if not hasattr(controller.pool, "reclaim_job"):
                    continue
                victim = controller._choose_reclaim_victim(reclaim)
                if victim is None:
                    continue
                choices.append((
                    victim["score"],
                    str(victim["job_id"]),
                    id(controller),
                    controller,
                    victim,
                ))
            if not choices:
                self._record({
                    "event": "RAM_RECLAIM_UNAVAILABLE_DRAIN_ONLY",
                    "system_usage_fraction": usage,
                    "ram_slope_gib_per_s": reclaim["slope_gib_per_s"],
                })
                return
            _, _, _, victim_controller, victim = min(
                choices, key=lambda value: value[:3])
            event = (
                "RAM_HARD_RECLAIM_ONE"
                if reclaim["reclaim_hard"]
                else "RAM_RISING_RECLAIM_ONE")
            victim_controller._spill(
                event=f"{event}_SELECTED", usage=usage)
            result = victim_controller.pool.reclaim_job(
                str(victim["job_id"]),
                reason=event,
                released_gib=float(victim["reclaimable_gib"]),
            )
            if result is None:
                victim_controller._record({
                    "event": "RAM_RECLAIM_LANE_TERMINATION_DEFERRED",
                    "system_usage_fraction": usage,
                    "ram_slope_gib_per_s": reclaim["slope_gib_per_s"],
                    "job_id": victim["job_id"],
                    "pid": victim["pid"],
                })
                return
            self.scheduler.note_targeted_reclaim(
                job_id=str(victim["job_id"]),
                released_gib=float(victim["reclaimable_gib"]),
                usage_fraction=usage,
            )
            victim_controller._record({
                "event": event,
                "system_usage_fraction": usage,
                "ram_slope_gib_per_s": reclaim["slope_gib_per_s"],
                "job_id": victim["job_id"],
                "pid": victim["pid"],
                "lane_index": victim["lane_index"],
                "job_class": victim.get("job_class"),
                "job_kind": victim.get("kind"),
                "job_age_seconds": victim["age_seconds"],
                "job_reclaim_count_before": victim.get(
                    "ram_reclaim_count", 0),
                "job_rss_gib": victim["release_gib"],
                "job_incremental_gib": victim["incremental_gib"],
                "job_reclaimable_gib": victim["reclaimable_gib"],
                "required_reclaim_gib": victim["required_reclaim_gib"],
                "settling_seconds": self.scheduler._settling_seconds,
                "recovery_seconds": self.scheduler._recovery_seconds,
                "global_registered_reclaim_pools": len(
                    self.scheduler.reclaim_controllers()),
            })
            victim_controller._spill(
                event=f"{event}_COMPLETE", usage=usage)
        finally:
            self.scheduler.end_targeted_reclaim()

    def resume_all(self) -> None:
        # v40.0.4 performs terminate+requeue, not NtSuspendProcess. Retain the
        # method because the managed-pool cleanup contract calls it.
        self.paused.clear()

@contextmanager
def _managed_pool_with_ram_pause(pool, scheduler: RamAdmissionScheduler | None,
                                  telemetry_root: Path):
    controller = (RamWorkerPauseController(scheduler, pool, telemetry_root)
                  if scheduler is not None else None)
    completed = False
    try:
        yield controller
        completed = True
    except BaseException:
        # ProcessPoolExecutor does not reliably reap already-spawned Windows
        # children after BrokenProcessPool or an interrupted parent.  Stop
        # those exact pool children before returning the exception, otherwise
        # they retain imported panels/BLAS state and poison the next resume.
        if controller is not None:
            controller.resume_all()
        for process in tuple(getattr(pool, "_processes", {}).values()):
            try:
                if process.is_alive():
                    process.terminate()
            except (OSError, RuntimeError):
                pass
        try:
            pool.shutdown(wait=True, cancel_futures=True)
        except TypeError:
            pool.shutdown(wait=True)
        raise
    finally:
        if completed:
            if controller is not None:
                controller.resume_all()
            pool.shutdown(wait=True)
        if controller is not None:
            controller.scheduler.unregister_reclaim_controller(controller)


def _manifested_process_initializer(worker_map: list[dict], slot_counter, role: str,
                                    telemetry_path: str | None = None) -> None:
    """Pin one spawned lane and force one native numerical thread per lane."""
    global _MANIFESTED_WORKER_SLOT, _MANIFESTED_WORKER_CPU
    global _MANIFESTED_WORKER_AFFINITY, _MANIFESTED_WORKER_ROLE
    _MANIFESTED_WORKER_ROLE = str(role)
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS", "POLARS_MAX_THREADS"):
        os.environ[name] = "1"
    if not worker_map or slot_counter is None:
        return
    with slot_counter.get_lock():
        slot = int(slot_counter.value)
        slot_counter.value += 1
    row = worker_map[slot % len(worker_map)]
    _MANIFESTED_WORKER_SLOT = slot
    _MANIFESTED_WORKER_CPU = int(row["logical_processor"])
    os.environ["DQBD_GPU_WORKER_SLOT"] = str(slot)
    _MANIFESTED_WORKER_AFFINITY = set_current_process_logical_affinity(_MANIFESTED_WORKER_CPU)
    if telemetry_path:
        telemetry_file = Path(telemetry_path)
        seed_root = telemetry_file.parent.parent
        shared_root = seed_root.parent / "_shared-compute"
        os.environ["DQBD_SLICE_HASH_SCRATCH_ROOT"] = str(
            shared_root / "slice-hash-scratch")
        os.environ["DQBD_SLICE_HASH_CACHE_ROOT"] = str(
            shared_root / "development-slice-hashes")
        record = {"pid": os.getpid(), "worker_slot": _MANIFESTED_WORKER_SLOT,
                  "logical_processor": _MANIFESTED_WORKER_CPU,
                  "role": _MANIFESTED_WORKER_ROLE,
                  "affinity_pinned": bool(_MANIFESTED_WORKER_AFFINITY),
                  "native_threads_per_worker": 1}
        path = Path(telemetry_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


class _ReclaimableLanePool:
    """Process executor composed of independently replaceable one-worker lanes.

    Killing a child of a multi-worker ProcessPoolExecutor marks the entire
    executor broken. v40.0.3 therefore converted one RAM victim into a
    pool-wide restart. Here each logical lane owns its own one-worker executor:
    a reclaim replaces exactly that lane while every other Future continues.
    """

    def __init__(
        self, *, worker_map: list[dict], role: str,
        telemetry_root: Path | None,
        max_tasks_per_child: int | None = None,
        max_pending: int | None = None,
    ) -> None:
        self.worker_map = [dict(row) for row in worker_map]
        self.role = str(role)
        self.telemetry_root = (
            Path(telemetry_root) if telemetry_root is not None else None)
        self.max_tasks_per_child = max_tasks_per_child
        self.max_pending = max(
            len(self.worker_map),
            int(max_pending) if max_pending is not None
            else len(self.worker_map),
        )
        self._lock = threading.RLock()
        self._closed = False
        self._cursor = 0
        self._queued: deque[tuple[Future, Any, tuple, dict, dict]] = deque()
        self._lanes: list[dict[str, Any]] = []
        for index, row in enumerate(self.worker_map):
            self._lanes.append({
                "index": index,
                "row": row,
                "executor": self._new_executor(index, row),
                "outer": None,
                "inner": None,
                "metadata": None,
                "generation": 0,
                "replacing": False,
            })

    def _new_executor(
        self, lane_index: int, row: Mapping[str, Any],
    ) -> ProcessPoolExecutor:
        spawn_context = mp.get_context("spawn")
        counter = spawn_context.Value("i", int(lane_index))
        executor_kwargs: dict[str, Any] = {}
        if self.max_tasks_per_child is not None:
            executor_kwargs["max_tasks_per_child"] = max(
                1, int(self.max_tasks_per_child))
        telemetry_path = (
            str(self.telemetry_root / f"{self.role}-workers.jsonl")
            if self.telemetry_root is not None else None)
        return ProcessPoolExecutor(
            max_workers=1,
            mp_context=spawn_context,
            initializer=_manifested_process_initializer,
            initargs=([dict(row)], counter, self.role, telemetry_path),
            **executor_kwargs,
        )

    @staticmethod
    def _task_metadata(args: tuple[Any, ...]) -> dict[str, Any]:
        payload = args[0] if args and isinstance(args[0], Mapping) else {}
        job = (
            payload.get("job", {})
            if isinstance(payload, Mapping) else {})
        return {
            "job_id": str(job.get("job_id", "")),
            "kind": str(job.get("kind", "")),
            "job_class": str(payload.get("_ram_job_class", "")),
            "gpu_device_index": payload.get("gpu_device_index"),
            "attempt": int(job.get("attempt", 0) or 0),
            "ram_reclaim_count": int(
                job.get("ram_reclaim_count", 0) or 0),
            "submitted_at": time.time(),
        }

    def _attach_relay(
        self, *, lane_index: int, generation: int,
        outer: Future, inner: Any,
    ) -> None:
        """Relay one inner executor Future into the stable lane Future."""
        def relay(inner_future) -> None:
            with self._lock:
                lane = self._lanes[int(lane_index)]
                if (
                    int(lane["generation"]) != int(generation)
                    or lane["outer"] is not outer
                    or lane["replacing"]
                ):
                    return
                lane["inner"] = None
                lane["outer"] = None
                lane["metadata"] = None
            if outer.done():
                return
            try:
                outer.set_result(inner_future.result())
            except BaseException as exc:
                if isinstance(exc, BrokenProcessPool):
                    # A dead child makes only its one-worker executor
                    # unusable. Replace that lane before dispatching queued
                    # work so one abrupt child cannot poison the whole run.
                    self._replace_broken_lane(
                        lane_index=int(lane_index),
                        generation=int(generation),
                    )
                if not outer.done():
                    outer.set_exception(exc)
            finally:
                # A GPU queue-ahead Future is held outside the worker lanes.
                # Dispatch it only after this lane has become free; this keeps
                # the physical worker count and kernel concurrency unchanged.
                self._dispatch_queued()

        inner.add_done_callback(relay)

    def _replace_broken_lane(
        self, *, lane_index: int, generation: int,
    ) -> None:
        """Replace one broken one-worker executor in-place."""
        with self._lock:
            lane = self._lanes[int(lane_index)]
            if (
                int(lane["generation"]) != int(generation)
                or lane["replacing"]
            ):
                return
            lane["replacing"] = True
            executor = lane["executor"]
            row = dict(lane["row"])
        try:
            try:
                executor.shutdown(wait=False, cancel_futures=True)
            except TypeError:
                executor.shutdown(wait=False)
            replacement = self._new_executor(int(lane_index), row)
        except BaseException:
            with self._lock:
                self._lanes[int(lane_index)]["replacing"] = False
            return
        with self._lock:
            lane = self._lanes[int(lane_index)]
            if (
                int(lane["generation"]) == int(generation)
                and lane["replacing"]
            ):
                lane["executor"] = replacement
                lane["replacing"] = False
            else:
                try:
                    replacement.shutdown(wait=False, cancel_futures=True)
                except TypeError:
                    replacement.shutdown(wait=False)

    def _start_task_locked(
        self, lane: dict[str, Any],
        task: tuple[Future, Any, tuple, dict, dict],
    ) -> tuple[int, int, Future, Any] | None:
        """Bind one queued task to an idle lane while holding ``_lock``."""
        outer, fn, args, kwargs, metadata = task
        if outer.cancelled() or outer.done():
            return None
        lane["generation"] += 1
        generation = int(lane["generation"])
        lane["outer"] = outer
        lane["metadata"] = metadata
        try:
            inner = lane["executor"].submit(fn, *args, **kwargs)
        except BaseException:
            lane["outer"] = None
            lane["metadata"] = None
            raise
        lane["inner"] = inner
        return int(lane["index"]), generation, outer, inner

    def _dispatch_queued(self) -> None:
        """Drain queued Futures into newly available one-worker lanes."""
        attachments: list[tuple[int, int, Future, Any]] = []
        with self._lock:
            if self._closed:
                return
            while self._queued:
                selected = next(
                    (
                        lane for lane in self._lanes
                        if lane["outer"] is None
                        and not lane["replacing"]
                    ),
                    None,
                )
                if selected is None:
                    break
                task = self._queued.popleft()
                try:
                    attachment = self._start_task_locked(selected, task)
                except BaseException as exc:
                    outer = task[0]
                    if not outer.done():
                        outer.set_exception(exc)
                    continue
                if attachment is not None:
                    attachments.append(attachment)
        for lane_index, generation, outer, inner in attachments:
            self._attach_relay(
                lane_index=lane_index,
                generation=generation,
                outer=outer,
                inner=inner,
            )

    def submit(self, fn, /, *args, **kwargs) -> Future:
        with self._lock:
            if self._closed:
                raise RuntimeError("DQBD_RECLAIMABLE_LANE_POOL_CLOSED")
            count = len(self._lanes)
            outer = Future()
            task = (
                outer,
                fn,
                args,
                kwargs,
                self._task_metadata(args),
            )
            selected = None
            for offset in range(count):
                index = (self._cursor + offset) % count
                lane = self._lanes[index]
                if (
                    lane["outer"] is None
                    and not lane["replacing"]
                    and not self._queued
                ):
                    selected = lane
                    self._cursor = (index + 1) % count
                    break
            if selected is None:
                active = sum(
                    1 for lane in self._lanes
                    if lane["outer"] is not None
                    or lane["replacing"]
                )
                if active + len(self._queued) >= self.max_pending:
                    raise RuntimeError("DQBD_RECLAIMABLE_LANE_POOL_FULL")
                self._queued.append(task)
                return outer
            attachment = self._start_task_locked(selected, task)
            if attachment is None:
                return outer
            lane_index, generation, outer, inner = attachment

        self._attach_relay(
            lane_index=lane_index, generation=generation,
            outer=outer, inner=inner)
        return outer

    @property
    def _processes(self) -> dict[int, Any]:
        processes: dict[int, Any] = {}
        with self._lock:
            executors = [
                lane["executor"] for lane in self._lanes
                if lane.get("executor") is not None]
        for executor in executors:
            for pid, process in tuple(
                    getattr(executor, "_processes", {}).items()):
                if pid is not None:
                    processes[int(pid)] = process
        return processes

    def idle_resident_lane_count(self) -> int:
        """Return idle lanes whose worker process already exists.

        GPU keepalive may reuse these lanes without spawning another Python
        process or adding a new imported-worker RAM baseline.
        """
        count = 0
        with self._lock:
            lanes = [
                {
                    "outer": lane["outer"],
                    "executor": lane["executor"],
                    "replacing": bool(lane["replacing"]),
                }
                for lane in self._lanes
            ]
        for lane in lanes:
            outer = lane["outer"]
            if (
                lane["replacing"]
                or (outer is not None and not outer.done())
            ):
                continue
            processes = list(
                getattr(lane["executor"], "_processes", {}).values())
            if any(
                getattr(process, "pid", None)
                and process.is_alive()
                for process in processes
            ):
                count += 1
        return count

    def active_lanes(self) -> list[dict[str, Any]]:
        rows = []
        with self._lock:
            lanes = [
                {
                    "index": int(lane["index"]),
                    "executor": lane["executor"],
                    "outer": lane["outer"],
                    "metadata": dict(lane["metadata"] or {}),
                    "replacing": bool(lane["replacing"]),
                }
                for lane in self._lanes
            ]
        for lane in lanes:
            if (
                lane["outer"] is None
                or lane["outer"].done()
                or lane["replacing"]
            ):
                continue
            processes = list(
                getattr(lane["executor"], "_processes", {}).values())
            pid = next(
                (int(process.pid) for process in processes
                 if getattr(process, "pid", None)),
                None)
            rows.append({
                "lane_index": lane["index"],
                "pid": pid,
                **lane["metadata"],
            })
        return rows

    def reclaim_job(
        self, job_id: str, *, reason: str, released_gib: float,
        failure: BaseException | None = None,
    ) -> dict[str, Any] | None:
        """Terminate and replace exactly the lane running one job.

        ``failure`` is reserved for execution-health recovery.  RAM pressure
        keeps the historical ``RamJobReclaimed`` contract; a dead worker must
        instead surface ``BrokenProcessPool`` so the coordinator records an
        execution incident and retries the exact job.
        """
        identity = str(job_id)
        with self._lock:
            lane = next(
                (
                    value for value in self._lanes
                    if value["outer"] is not None
                    and not value["outer"].done()
                    and str((value["metadata"] or {}).get("job_id", ""))
                        == identity
                ),
                None)
            if lane is None or lane["replacing"]:
                return None
            lane["replacing"] = True
            lane["generation"] += 1
            executor = lane["executor"]
            outer = lane["outer"]
            inner = lane["inner"]
            metadata = dict(lane["metadata"] or {})
            lane_index = int(lane["index"])
            row = dict(lane["row"])
            processes = list(
                getattr(executor, "_processes", {}).values())
            pid = next(
                (int(process.pid) for process in processes
                 if getattr(process, "pid", None)),
                None)

        terminated = True
        for process in processes:
            try:
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=.5)
                if process.is_alive():
                    kill = getattr(process, "kill", None)
                    if callable(kill):
                        kill()
                        process.join(timeout=.5)
                if process.is_alive():
                    terminated = False
            except (OSError, RuntimeError, ValueError):
                terminated = False
        if not terminated:
            # Never reuse a lane while its prior child can still allocate RAM.
            # The first relay was invalidated before terminate(); attach a new
            # generation relay so this lane can still drain normally instead
            # of becoming permanently occupied if termination was refused.
            with self._lock:
                lane = self._lanes[lane_index]
                lane["replacing"] = False
                lane["generation"] += 1
                recovery_generation = int(lane["generation"])
                recovery_outer = lane["outer"]
                recovery_inner = lane["inner"]
            if (
                recovery_outer is not None
                and recovery_inner is not None
            ):
                self._attach_relay(
                    lane_index=lane_index,
                    generation=recovery_generation,
                    outer=recovery_outer,
                    inner=recovery_inner)
            return None
        if pid is not None:
            try:
                cleanup_slice_hash_scratch_for_pid(pid)
            except OSError:
                pass
        try:
            executor.shutdown(wait=False, cancel_futures=True)
        except TypeError:
            executor.shutdown(wait=False)

        replacement = self._new_executor(lane_index, row)
        with self._lock:
            lane = self._lanes[lane_index]
            lane["executor"] = replacement
            lane["inner"] = None
            lane["outer"] = None
            lane["metadata"] = None
            lane["replacing"] = False
        if not outer.done():
            if failure is None:
                outer.set_exception(RamJobReclaimed(
                    job_id=identity, pid=pid, reason=reason,
                    released_gib=float(released_gib)))
            else:
                outer.set_exception(failure)
        self._dispatch_queued()
        return {
            **metadata,
            "lane_index": lane_index,
            "pid": pid,
            "released_gib": float(released_gib),
            "terminated": bool(terminated),
            "reason": str(reason),
        }

    def reclaim_unhealthy_lanes(
        self, *, startup_grace_seconds: float = 120.0,
    ) -> list[dict[str, Any]]:
        """Reclaim active lanes whose child process has disappeared.

        ProcessPoolExecutor does not reliably wake a parent when a Windows
        child vanishes while the parent is blocked in a Future/pipe operation.
        The independent watchdog calls this method.  A lane with no live
        child after its startup grace is therefore an orphaned execution, not
        a scientific failure: replace only that lane and let the normal
        ``BrokenProcessPool`` retry path return its job to ``PENDING``.
        """
        now = time.time()
        candidates: list[tuple[str, str, int | None]] = []
        with self._lock:
            for lane in self._lanes:
                outer = lane.get("outer")
                if (
                    outer is None
                    or outer.done()
                    or lane.get("replacing")
                ):
                    continue
                metadata = dict(lane.get("metadata") or {})
                submitted_at = float(
                    metadata.get("submitted_at", now) or now)
                processes = list(
                    getattr(lane.get("executor"), "_processes", {})
                    .values())
                live = any(
                    getattr(process, "pid", None)
                    and process.is_alive()
                    for process in processes)
                if live:
                    continue
                if now - submitted_at < float(startup_grace_seconds):
                    continue
                candidates.append((
                    str(metadata.get("job_id", "")),
                    f"DQBD_ORPHANED_WORKER:{self.role}",
                    int(lane["index"])))

        reclaimed: list[dict[str, Any]] = []
        for job_id, reason, lane_index in candidates:
            if not job_id:
                continue
            failure = BrokenProcessPool(
                f"{reason}:lane={lane_index}:job={job_id}")
            result = self.reclaim_job(
                job_id,
                reason=reason,
                released_gib=0.0,
                failure=failure,
            )
            if result is not None:
                reclaimed.append(result)
        return reclaimed

    def shutdown(
        self, wait: bool = True, *, cancel_futures: bool = False,
    ) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            queued = list(self._queued)
            self._queued.clear()
            lanes = list(self._lanes)
        for task in queued:
            outer = task[0]
            if not outer.done():
                outer.cancel()
        for lane in lanes:
            outer = lane.get("outer")
            if cancel_futures and outer is not None and not outer.done():
                outer.cancel()
            executor = lane.get("executor")
            if executor is None:
                continue
            try:
                executor.shutdown(
                    wait=bool(wait), cancel_futures=bool(cancel_futures))
            except TypeError:
                executor.shutdown(wait=bool(wait))


class _LaneHealthWatchdog:
    """Keep manifested leases alive and recover disappeared child lanes.

    The dispatch loop is intentionally allowed to block on ordinary Future
    completion.  Heartbeats and worker-health checks must not depend on that
    loop making progress: on Windows a dead ProcessPool child can otherwise
    leave a claimed SQLite job RUNNING forever while its parent waits on a
    result pipe that will never deliver a completion.
    """

    def __init__(
        self, *, store: "ManifestedJobStore", owner: str,
        pools: Iterable[Any], telemetry_root: Path,
        activity_heartbeat_path: str | Path | None = None,
        interval_seconds: float = 10.0,
        startup_grace_seconds: float = 120.0,
        stale_after_seconds: float = 120.0,
    ) -> None:
        self.store = store
        self.owner = str(owner)
        self.pools = tuple(pools)
        self.telemetry_root = Path(telemetry_root)
        self.activity_heartbeat_path = (
            Path(activity_heartbeat_path)
            if activity_heartbeat_path is not None else None)
        self.interval_seconds = max(.5, float(interval_seconds))
        self.startup_grace_seconds = max(
            5.0, float(startup_grace_seconds))
        self.stale_after_seconds = max(30.0, float(stale_after_seconds))
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._events_path = (
            self.telemetry_root / "lane-watchdog-events.jsonl")

    def __enter__(self) -> "_LaneHealthWatchdog":
        self._thread = threading.Thread(
            target=self._run,
            name=f"dqbd-lane-watchdog-{self.owner}",
            daemon=True,
        )
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(5.0, self.interval_seconds * 3))

    def _record(self, payload: Mapping[str, Any]) -> None:
        row = {
            "timestamp_epoch": time.time(),
            "owner": self.owner,
            **dict(payload),
        }
        try:
            self._events_path.parent.mkdir(parents=True, exist_ok=True)
            with self._events_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, sort_keys=True, default=str))
                handle.write("\n")
        except OSError:
            # Telemetry must never turn a recoverable worker incident into a
            # scheduler failure.
            pass

    def _tick(self) -> None:
        # The lane watchdog used to renew leases even when the parent
        # dispatcher had stopped making scheduling progress. That converted
        # a dead bank into a permanently valid-looking bank. The activity
        # file is written by the dispatch loop itself, so it is independent
        # of worker leases and is the correct signal for a stuck dispatcher.
        stale_activity_age = None
        if self.activity_heartbeat_path is not None:
            try:
                activity = json.loads(
                    self.activity_heartbeat_path.read_text(encoding="utf-8"))
                if (
                    activity.get("state") == "ACTIVE"
                    and str(activity.get("owner", "")) == self.owner
                ):
                    stale_activity_age = (
                        time.time() - float(
                            activity.get("updated_at_epoch", 0.0)))
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                stale_activity_age = None
        stale_dispatcher = False
        if (
            stale_activity_age is not None
            and stale_activity_age > self.stale_after_seconds
        ):
            # Only recycle when the bank still has dependency-ready work. An
            # exhausted frontier is normal and must not look like a failure.
            stale_dispatcher = self.store.ready_job_count() > 0
        for pool in self.pools:
            if pool is None:
                continue
            try:
                active = pool.active_lanes()
                for lane in active:
                    job_id = str(lane.get("job_id", ""))
                    if job_id:
                        if stale_dispatcher and hasattr(pool, "reclaim_job"):
                            failure = BrokenProcessPool(
                                "DQBD_STALE_DISPATCHER_ACTIVITY:"
                                f"age={stale_activity_age:.3f}")
                            reclaimed = pool.reclaim_job(
                                job_id,
                                reason="DQBD_STALE_DISPATCHER_ACTIVITY",
                                released_gib=0.0,
                                failure=failure,
                            )
                            if reclaimed is not None:
                                self._record({
                                    "event": "STALE_DISPATCHER_LANE_RECLAIMED",
                                    "job_id": job_id,
                                    "activity_age_seconds": (
                                        float(stale_activity_age)),
                                    **dict(reclaimed),
                                })
                            continue
                        self.store.heartbeat(
                            job_id, self.owner,
                            lease_seconds=JOB_LEASE_SECONDS)
                reclaimed = pool.reclaim_unhealthy_lanes(
                    startup_grace_seconds=self.startup_grace_seconds)
                for row in reclaimed:
                    self._record({
                        "event": "ORPHANED_LANE_RECLAIMED",
                        "role": str(getattr(pool, "role", "")),
                        **row,
                    })
            except BaseException as exc:
                self._record({
                    "event": "WATCHDOG_TICK_ERROR",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                })

    def _run(self) -> None:
        while not self._stop.is_set():
            self._tick()
            self._stop.wait(self.interval_seconds)


_MANIFESTED_SLICE_HASH_CLEANUP_LOCK = threading.Lock()
_MANIFESTED_SLICE_HASH_CLEANED = False


def _manifested_pool(*, workers: int, role: str, telemetry_root: Path | None = None,
                      worker_map_override: list[dict] | None = None,
                      max_pending: int | None = None,
                      max_tasks_per_child: int | None = None):
    global _MANIFESTED_SLICE_HASH_CLEANED
    contract = active_cpu_contract()
    worker_map = list(worker_map_override or contract.get("worker_map") or [])
    if not worker_map:
        worker_map = [{"logical_processor": i, "core_index": i, "role": "fallback"}
                      for i in range(int(contract.get("model_fit_processes", workers)))]
    selected = min(int(workers), len(worker_map))
    selected_map = worker_map[:selected]
    if telemetry_root is not None:
        shared_root = Path(telemetry_root).parent.parent / "_shared-compute"
        os.environ["DQBD_SLICE_HASH_SCRATCH_ROOT"] = str(
            shared_root / "slice-hash-scratch")
        os.environ["DQBD_SLICE_HASH_CACHE_ROOT"] = str(
            shared_root / "development-slice-hashes")
    with _MANIFESTED_SLICE_HASH_CLEANUP_LOCK:
        if not _MANIFESTED_SLICE_HASH_CLEANED:
            try:
                cleanup_legacy_slice_hash_scratch()
                cleanup_orphaned_slice_hash_scratch()
            except OSError:
                pass
            _MANIFESTED_SLICE_HASH_CLEANED = True
    return _ReclaimableLanePool(
        worker_map=selected_map,
        role=role,
        telemetry_root=telemetry_root,
        max_pending=max_pending,
        max_tasks_per_child=max_tasks_per_child,
    ), selected


def _configured_lane_recycle_limit() -> int:
    """Bound resident worker state without reducing the lane target."""
    raw = os.environ.get("DQBD_MAX_TASKS_PER_LANE", "8")
    try:
        return max(1, int(raw))
    except (TypeError, ValueError):
        return 8

def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _cached_development_slice_hash(path: str | Path, *, date_column: str,
                                   development_end: date, holdout_boundary: date) -> str:
    target = Path(path)
    stat = target.stat()
    key = (str(target.resolve()), int(stat.st_size), int(stat.st_mtime_ns),
           date_column, f"{development_end.isoformat()}|{holdout_boundary.isoformat()}")
    value = _DEVELOPMENT_HASH_CACHE.get(key)
    if value is None:
        value = parquet_development_slice_sha256(
            target, date_column=date_column, development_end=development_end,
            holdout_boundary=holdout_boundary)
        _DEVELOPMENT_HASH_CACHE[key] = value
    return value


def parquet_dates(path: str | Path, preferred: tuple[str, ...]) -> pd.Series:
    dataset = ds.dataset(str(path), format="parquet")
    schema = set(dataset.schema.names)
    column = next((name for name in preferred if name in schema), None)
    if column is None:
        raise ValueError(f"MANIFESTED_JOB_DATE_COLUMN_MISSING:{path}:{preferred}")
    predicate = ds.field(column) < pa.scalar(pd.Timestamp(HOLDOUT_BOUNDARY).to_datetime64())
    if "holdout_locked" in schema:
        predicate = predicate & ((ds.field("holdout_locked") == False) | ds.field("holdout_locked").is_null())
    table = dataset.to_table(columns=[column], filter=predicate)
    return pd.to_datetime(table[column].to_pandas())


def read_development_replay_inputs(*, prices_path: str | Path, distributions_path: str | Path,
                                   development_end: date, holdout_boundary: date,
                                   replay_start: date | str | None = None,
                                   replay_end: date | str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read replay inputs with Arrow predicates before materializing rows."""
    if development_end >= holdout_boundary:
        raise PermissionError("REPLAY_DEVELOPMENT_BOUNDARY_OPEN")
    price_ds = ds.dataset(str(prices_path), format="parquet")
    price_names = set(price_ds.schema.names)
    price_date = next((x for x in ("date", "session_date", "decision_date") if x in price_names), None)
    if price_date is None:
        raise ValueError("REPLAY_PRICE_DATE_COLUMN_MISSING")
    price_type = price_ds.schema.field(price_date).type
    price_boundary = pa.scalar(development_end.isoformat()) if pa.types.is_string(price_type) or pa.types.is_large_string(price_type) else pa.scalar(pd.Timestamp(development_end).to_datetime64())
    base_filter = ds.field(price_date) <= price_boundary
    price_target = Path(prices_path)
    price_stat = price_target.stat()
    price_cache_key = (str(price_target.resolve()), int(price_stat.st_size),
                       int(price_stat.st_mtime_ns), development_end.isoformat())
    if replay_start is not None and replay_end is not None and "ticker" in price_names:
        segment_start = pd.Timestamp(replay_start)
        segment_end = pd.Timestamp(replay_end)
        segment_cache_key = price_cache_key + (segment_start.date().isoformat(),
                                               segment_end.date().isoformat())
        with _REPLAY_STATIC_CACHE_LOCK:
            cached_segment = _REPLAY_SEGMENT_PRICE_CACHE.get(segment_cache_key)
            if cached_segment is not None:
                _REPLAY_SEGMENT_PRICE_CACHE.move_to_end(segment_cache_key)
                price_frame = cached_segment
            else:
                price_frame = None
        if pa.types.is_string(price_type) or pa.types.is_large_string(price_type):
            start_scalar = pa.scalar(segment_start.date().isoformat())
            end_scalar = pa.scalar(segment_end.date().isoformat())
        else:
            start_scalar = pa.scalar(segment_start.to_datetime64())
            end_scalar = pa.scalar(segment_end.to_datetime64())
        if price_frame is None:
            with _REPLAY_STATIC_CACHE_LOCK:
                benchmark_prices = _REPLAY_BENCHMARK_CACHE.get(price_cache_key)
                if benchmark_prices is None:
                    benchmark_prices = price_ds.to_table(
                        filter=base_filter & (ds.field("ticker") == "URTH"),
                        columns=list(price_names)).to_pandas()
                    _REPLAY_BENCHMARK_CACHE[price_cache_key] = benchmark_prices
            stock_filter = (base_filter &
                            (ds.field(price_date) >= start_scalar) &
                            (ds.field(price_date) <= end_scalar) &
                            (ds.field("ticker") != "URTH"))
            stock_prices = price_ds.to_table(filter=stock_filter,
                                             columns=list(price_names)).to_pandas()
            price_frame = pd.concat((benchmark_prices, stock_prices), ignore_index=True, copy=False)
            # Invalid stock execution boundaries are not tradable observations.
            # Cache the already-filtered object so replay's id-keyed price-map
            # cache can also be reused by later families in this worker.
            if {"open_quality_ok", "close_quality_ok"} <= set(price_frame.columns):
                price_frame = price_frame[(price_frame["ticker"].eq("URTH")) |
                                          (price_frame["open_quality_ok"] &
                                           price_frame["close_quality_ok"])].copy()
            with _REPLAY_STATIC_CACHE_LOCK:
                _REPLAY_SEGMENT_PRICE_CACHE[segment_cache_key] = price_frame
                _REPLAY_SEGMENT_PRICE_CACHE.move_to_end(segment_cache_key)
                while len(_REPLAY_SEGMENT_PRICE_CACHE) > _REPLAY_SEGMENT_PRICE_CACHE_SIZE:
                    _REPLAY_SEGMENT_PRICE_CACHE.popitem(last=False)
    else:
        price_frame = price_ds.to_table(filter=base_filter,
                                        columns=list(price_names)).to_pandas()
    dist_ds = ds.dataset(str(distributions_path), format="parquet")
    dist_names = set(dist_ds.schema.names)
    if "ex_date" not in dist_names:
        raise ValueError("REPLAY_DISTRIBUTION_EX_DATE_MISSING")
    dist_type = dist_ds.schema.field("ex_date").type
    dist_boundary = pa.scalar(development_end.isoformat()) if pa.types.is_string(dist_type) or pa.types.is_large_string(dist_type) else pa.scalar(pd.Timestamp(development_end).to_datetime64())
    dist_target = Path(distributions_path)
    dist_stat = dist_target.stat()
    dist_cache_key = (str(dist_target.resolve()), int(dist_stat.st_size),
                      int(dist_stat.st_mtime_ns), development_end.isoformat())
    with _REPLAY_STATIC_CACHE_LOCK:
        distribution_frame = _REPLAY_DISTRIBUTION_CACHE.get(dist_cache_key)
        if distribution_frame is None:
            distribution_frame = dist_ds.to_table(
                filter=ds.field("ex_date") <= dist_boundary,
                columns=list(dist_names)).to_pandas()
            _REPLAY_DISTRIBUTION_CACHE[dist_cache_key] = distribution_frame
    return price_frame, distribution_frame


def _cached_replay_signals(path: str | Path) -> pd.DataFrame:
    target = Path(path)
    stat = target.stat()
    key = (str(target.resolve()), int(stat.st_size), int(stat.st_mtime_ns))
    with _REPLAY_STATIC_CACHE_LOCK:
        cached = _REPLAY_SIGNAL_CACHE.get(key)
        if cached is not None:
            _REPLAY_SIGNAL_CACHE.move_to_end(key)
            return cached
    frame = pd.read_parquet(target)[
        ["decision_date", "ticker", "score", "model_artifact_id"]]
    with _REPLAY_STATIC_CACHE_LOCK:
        _REPLAY_SIGNAL_CACHE[key] = frame
        _REPLAY_SIGNAL_CACHE.move_to_end(key)
        while len(_REPLAY_SIGNAL_CACHE) > _REPLAY_SIGNAL_CACHE_SIZE:
            _REPLAY_SIGNAL_CACHE.popitem(last=False)
    return frame


def current_git_sha(repo_root: str | Path) -> str:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(repo_root),
                            capture_output=True, text=True, check=False)
    value = result.stdout.strip()
    if result.returncode or len(value) != 40:
        raise RuntimeError("MANIFESTED_JOB_GIT_SHA_UNAVAILABLE")
    return value


def source_tree_sha256(repo_root: str | Path) -> str:
    root = Path(repo_root)
    # Exclude generated trees at the Git pathspec boundary.  Filtering them
    # only after ``ls-files -o`` has enumerated every cache file makes resume
    # startup scale with old local artifacts rather than with source size.
    result = subprocess.run(
        ["git", "ls-files", "-co", "--exclude-standard", "-z", "--", ".",
         ":(exclude)artifacts/**", ":(exclude)gpu-device-events/**",
         ":(exclude)locks/**", ":(exclude).hotload/**"],
        cwd=root, capture_output=True, check=True)
    digest = hashlib.sha256()
    for raw in result.stdout.split(b"\0"):
        if not raw:
            continue
        relative = raw.decode("utf-8")
        if relative.startswith("artifacts/") or relative.startswith(".hotload/"):
            continue
        path = root / relative
        # ``ls-files -o`` can report an entry Git cannot descend into, for
        # example a directory junction. Opening it raises PermissionError and
        # used to abort initialization, so identity covers regular files only.
        if not path.is_file():
            continue
        digest.update(relative.encode("utf-8")); digest.update(b"\0")
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def input_stat_fingerprint(inputs: "ManifestedJobInputs") -> dict[str, dict[str, int]]:
    """Return cheap resume guards for immutable run inputs."""
    paths = {
        "signal_panel": inputs.signal_panel,
        "candidate_metrics": inputs.candidate_metrics,
        "feature_schema": inputs.feature_schema,
        "benchmark_prices": inputs.benchmark_prices,
        "benchmark_distributions": inputs.benchmark_distributions,
        "stock_execution_prices": inputs.stock_execution_prices,
        "stock_distributions": inputs.stock_distributions,
    }
    result = {}
    for name, path in paths.items():
        if path is None:
            continue
        stat = Path(path).stat()
        result[name] = {"size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)}
    return result


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, default=str, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _coverage_consistency(*, expected_keys: Iterable[tuple[int, str, str]],
                          observed_keys: Iterable[tuple[int, str, str]],
                          duplicate_keys: int, frame_empty: bool,
                          candidate_count: int,
                          candidate_registry_count: int) -> dict[str, Any]:
    """Classify a coverage snapshot without weakening causal admissibility.

    Older resumable DAG manifests can contain fewer dependency tickets than
    the number of partitions that are already causally matured at the same
    cutoff.  Those extra rows are safe: ``read_matured`` has already applied
    the information-availability boundary.  Treat that one-sided mismatch as
    a recoverable stale-manifest condition and use the observed matured set.
    Missing, duplicate, empty, or incomplete candidate evidence remains a
    hard failure.
    """
    expected = set(expected_keys)
    observed = set(observed_keys)
    missing = sorted(expected - observed)
    unexpected = sorted(observed - expected)
    fatal_reasons = []
    if missing:
        fatal_reasons.append("MISSING_EXPECTED_KEYS")
    if int(duplicate_keys):
        fatal_reasons.append("DUPLICATE_OBSERVATIONS")
    if frame_empty:
        fatal_reasons.append("EMPTY_SNAPSHOT")
    if int(candidate_count) < int(candidate_registry_count):
        fatal_reasons.append("INCOMPLETE_CANDIDATE_REGISTRY")
    repaired = bool(unexpected and not fatal_reasons)
    return {
        "status": (
            "STALE_DEPENDENCY_MANIFEST_REPAIRED" if repaired
            else ("FATAL_COVERAGE_INCONSISTENCY" if fatal_reasons
                  else "COVERAGE_CONSISTENT")),
        "repaired": repaired,
        "fatal": bool(fatal_reasons),
        "fatal_reasons": fatal_reasons,
        "expected_key_count": len(expected),
        "observed_key_count": len(observed),
        "missing_key_count": len(missing),
        "unexpected_key_count": len(unexpected),
        "duplicate_key_count": int(duplicate_keys),
        "candidate_count": int(candidate_count),
        "candidate_registry_count": int(candidate_registry_count),
        "missing_keys": [list(value) for value in missing],
        "unexpected_keys": [list(value) for value in unexpected],
    }


@dataclass(frozen=True)
class ManifestedJobInputs:
    repo_root: Path
    signal_panel: Path
    candidate_metrics: Path | None
    feature_schema: Path
    benchmark_prices: Path
    benchmark_distributions: Path
    development_start: date
    development_end: date
    stock_execution_prices: Path | None = None
    stock_price_quality_manifest: Path | None = None
    stock_direct_daily_manifest: Path | None = None
    stock_distributions: Path | None = None
    training_window_sessions: int = 504
    validation_window_sessions: int = 126
    calibration_window_sessions: int = 252
    step_sessions: int = 126
    purge_sessions: int = 30
    embargo_sessions: int = 0
    refit_cadence: str = "MONTH_END"
    maturity_rule: str = "HORIZON_SESSIONS_NEXT_OPEN"
    score_quantile: float = .975
    top_fraction: float = .005
    cost_model: Mapping[str, Any] = None
    tax_contract: Mapping[str, Any] = None
    holdout_boundary: date = HOLDOUT_BOUNDARY
    allow_missing_stock_inputs: bool = False
    allow_dirty_development_fixture: bool = False
    hyperparameter_space: Path | None = None
    random_seed: int = 17

    def __post_init__(self) -> None:
        object.__setattr__(self, "repo_root", Path(self.repo_root))
        for name in ("signal_panel", "candidate_metrics", "feature_schema", "benchmark_prices", "benchmark_distributions",
                     "stock_execution_prices", "stock_price_quality_manifest", "stock_direct_daily_manifest", "stock_distributions"):
            value = getattr(self, name)
            object.__setattr__(self, name, Path(value) if value is not None else None)
        if self.hyperparameter_space is not None:
            object.__setattr__(self, "hyperparameter_space", Path(self.hyperparameter_space))
        if int(self.random_seed) < 0:
            raise ValueError("MANIFESTED_JOB_RANDOM_SEED_INVALID")
        if self.cost_model is None:
            object.__setattr__(self, "cost_model", {"roundtrip_bps": 20.0})
        if self.tax_contract is None:
            object.__setattr__(self, "tax_contract", {
                "mode": "POST_TAX", "enabled": True, "engine": "DE_RETAIL_APPROX",
                "capital_gains_rate": .25, "solidarity_surcharge": .055,
                "allowance_eur": 1000.0, "church_tax_rate": 0.0,
            })
        if self.development_end >= self.holdout_boundary:
            raise ValueError("MANIFESTED_JOB_DEVELOPMENT_CROSSES_HOLDOUT")
        if not 0 < self.top_fraction <= 1 or not 0 < self.score_quantile < 1:
            raise ValueError("MANIFESTED_JOB_INVALID_ENTRY_CONTRACT")

    def resolved_fold_policy(self) -> FoldPolicy:
        return FoldPolicy(training_window_sessions=self.training_window_sessions,
                          calibration_window_sessions=self.calibration_window_sessions,
                          validation_window_sessions=self.validation_window_sessions,
                          step_sessions=self.step_sessions,
                          purge_sessions=self.purge_sessions, embargo_sessions=self.embargo_sessions)

    def resolved_target_contract(self) -> TargetContract:
        return TargetContract(cost_model=dict(self.cost_model))

    def resolved_contracts(self, *, primary_candidate_universe_hash: str = "") -> ResolvedContracts:
        fold = self.resolved_fold_policy()
        target = self.resolved_target_contract()
        selection = RecipeSelectionPolicy()
        training = ModelTrainingContract(
            random_seed=self.random_seed,
            primary_candidate_universe_hash=primary_candidate_universe_hash,
        )
        return ResolvedContracts(fold, target, selection, training)


class ManifestedJobStore:
    """SQLite/WAL persistent job state with a JSON manifest for inspection."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.jobs_path = self.root / "run-state" / "jobs.json"
        self.db_path = self.root / "run-state" / "jobs.sqlite3"
        self.root.joinpath("run-state").mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS jobs (
                job_id TEXT PRIMARY KEY, kind TEXT NOT NULL, state TEXT NOT NULL,
                payload TEXT NOT NULL, attempt INTEGER NOT NULL DEFAULT 0,
                lease_owner TEXT, lease_until REAL, heartbeat REAL,
                last_error TEXT, started_at REAL, finished_at REAL)""")
            # The ordered composite fully subsumes the historical two-column
            # index and avoids a temporary B-tree for every ready-job claim.
            db.execute("DROP INDEX IF EXISTS jobs_ready")
            db.execute("CREATE INDEX IF NOT EXISTS jobs_ready_ordered ON jobs(state, kind, job_id)")
            db.execute("""CREATE TABLE IF NOT EXISTS job_dependencies (
                job_id TEXT NOT NULL, depends_on TEXT NOT NULL,
                PRIMARY KEY(job_id, depends_on))""")
            db.execute("CREATE INDEX IF NOT EXISTS dependencies_parent ON job_dependencies(depends_on, job_id)")
            db.execute("""CREATE TABLE IF NOT EXISTS manifest_metadata (
                key TEXT PRIMARY KEY, value TEXT NOT NULL)""")
            if db.execute("SELECT value FROM manifest_metadata WHERE key='dependency_index_version'").fetchone() is None:
                # Native JSON1 expansion migrates an existing manifest exactly
                # once. Worker-side store construction must not rescan a
                # 200k-node DAG for every submitted job.
                db.execute("BEGIN IMMEDIATE")
                if db.execute("SELECT value FROM manifest_metadata WHERE key='dependency_index_version'").fetchone() is None:
                    db.execute("""INSERT OR IGNORE INTO job_dependencies(job_id, depends_on)
                        SELECT jobs.job_id, CAST(value AS TEXT)
                        FROM jobs, json_each(jobs.payload, '$.depends_on')""")
                    db.execute("INSERT INTO manifest_metadata(key,value) VALUES('dependency_index_version','1')")
                db.execute("COMMIT")

    @contextmanager
    def _connect(self):
        db = sqlite3.connect(self.db_path, timeout=30, isolation_level=None)
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("PRAGMA synchronous=NORMAL")
        try:
            yield db
        finally:
            db.close()

    def write(self, relative: str, payload: Any) -> Path:
        path = self.root / relative
        _write_json(path, payload)
        return path

    def read(self, relative: str) -> Any:
        return json.loads((self.root / relative).read_text(encoding="utf-8"))

    def set_job(self, job_id: str, state: str, **fields: Any) -> None:
        if state not in JOB_STATES:
            raise ValueError(f"MANIFESTED_JOB_INVALID_JOB_STATE:{state}")
        now = time.time()
        with self._connect() as db:
            row = db.execute("SELECT payload, attempt FROM jobs WHERE job_id=?", (job_id,)).fetchone()
            prior = json.loads(row[0]) if row else {"job_id": job_id}
            attempt = int(row[1]) if row else int(fields.pop("attempt", 0))
            prior.update({"job_id": job_id, "state": state, **fields})
            started = now if state == "RUNNING" and not prior.get("started_at") else prior.get("started_at")
            finished = now if state in ("COMPLETE", "FAILED") else prior.get("finished_at")
            db.execute("""INSERT INTO jobs(job_id,kind,state,payload,attempt,lease_owner,lease_until,heartbeat,last_error,started_at,finished_at)
                         VALUES(?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(job_id) DO UPDATE SET
                         kind=excluded.kind,state=excluded.state,payload=excluded.payload,attempt=excluded.attempt,
                         lease_owner=excluded.lease_owner,lease_until=excluded.lease_until,heartbeat=excluded.heartbeat,
                         last_error=excluded.last_error,started_at=excluded.started_at,finished_at=excluded.finished_at""",
                       (job_id, prior.get("kind", ""), state, json.dumps(prior, default=str), attempt,
                        prior.get("lease_owner"), prior.get("lease_until"), now if state == "RUNNING" else prior.get("heartbeat"),
                        prior.get("last_error"), started, finished))

    @staticmethod
    def payload_hash(payload: Mapping[str, Any]) -> str:
        semantic = {k: v for k, v in payload.items() if k not in {"state", "result", "last_error", "started_at", "finished_at", "lease_owner", "lease_until", "heartbeat", "attempt"}}
        return stable_hash(semantic)

    def seed_jobs(
        self, jobs: Mapping[str, Mapping[str, Any]], *,
        invalidate_descendants: bool = False,
    ) -> dict[str, int]:
        """Seed/reconcile manifested jobs without discarding valid checkpoints.

        A slow compatible resume may discover that one existing job's semantic
        payload changed (for example a repaired coverage dependency list).
        Requeue that job and, when requested, only its transitive descendants.
        Independent COMPLETE jobs remain untouched.
        """
        with self._connect() as db:
            existing = {
                row[0]: (row[1], row[2])
                for row in db.execute(
                    "SELECT job_id,state,json_extract(payload,'$.payload_hash') "
                    "FROM jobs").fetchall()
            }
            inserts = []
            updates = []
            for job_id, job in jobs.items():
                incoming = dict(job)
                incoming["logical_job_id"] = str(job_id)
                incoming["payload_hash"] = self.payload_hash(incoming)
                row = existing.get(job_id)
                if row is None:
                    inserts.append((
                        job_id, incoming.get("kind", ""),
                        incoming.get("state", "PENDING"),
                        json.dumps(incoming, default=str),
                    ))
                elif row[1] != incoming["payload_hash"]:
                    incoming["stale_from"] = row[0]
                    incoming["state"] = "PENDING"
                    updates.append((
                        incoming.get("kind", ""),
                        json.dumps(incoming, default=str),
                        job_id,
                    ))
            if inserts:
                db.executemany(
                    "INSERT INTO jobs(job_id,kind,state,payload,attempt) "
                    "VALUES(?,?,?,?,0)", inserts)
            if updates:
                db.executemany(
                    "UPDATE jobs SET kind=?,state='PENDING',payload=?,"
                    "lease_owner=NULL,lease_until=NULL,heartbeat=NULL,"
                    "last_error=NULL,started_at=NULL,finished_at=NULL "
                    "WHERE job_id=?", updates)
            changed_ids = [row[0] for row in inserts] + [
                row[2] for row in updates]
            if changed_ids:
                db.executemany(
                    "DELETE FROM job_dependencies WHERE job_id=?",
                    ((job_id,) for job_id in changed_ids))
                dependency_rows = []
                for job_id in changed_ids:
                    incoming = dict(jobs[job_id])
                    dependency_rows.extend(
                        (job_id, str(parent))
                        for parent in incoming.get("depends_on", ()))
                if dependency_rows:
                    db.executemany(
                        "INSERT OR IGNORE INTO "
                        "job_dependencies(job_id,depends_on) VALUES(?,?)",
                        dependency_rows)

            requeued_descendants = 0
            updated_ids = [row[2] for row in updates]
            if invalidate_descendants and updated_ids:
                db.execute(
                    "CREATE TEMP TABLE IF NOT EXISTS "
                    "resume_changed_jobs(job_id TEXT PRIMARY KEY)")
                db.execute("DELETE FROM resume_changed_jobs")
                db.executemany(
                    "INSERT OR IGNORE INTO resume_changed_jobs(job_id) "
                    "VALUES(?)", ((job_id,) for job_id in updated_ids))
                db.execute(
                    "WITH RECURSIVE descendants(job_id) AS ("
                    " SELECT d.job_id FROM job_dependencies d "
                    " JOIN resume_changed_jobs c "
                    " ON d.depends_on=c.job_id"
                    " UNION "
                    " SELECT d.job_id FROM job_dependencies d "
                    " JOIN descendants x ON d.depends_on=x.job_id"
                    ") "
                    "UPDATE jobs SET state='PENDING',"
                    "payload=json_remove("
                    " json_set(payload,'$.state','PENDING'),"
                    " '$.result','$.last_error','$.started_at',"
                    " '$.finished_at','$.lease_owner','$.lease_until',"
                    " '$.heartbeat'),"
                    "lease_owner=NULL,lease_until=NULL,heartbeat=NULL,"
                    "last_error=NULL,started_at=NULL,finished_at=NULL "
                    "WHERE job_id IN (SELECT job_id FROM descendants)")
                requeued_descendants = int(
                    db.execute("SELECT changes()").fetchone()[0])
                db.execute("DELETE FROM resume_changed_jobs")
        self.materialize_json_manifest()
        return {
            "inserted_jobs": len(inserts),
            "updated_jobs": len(updates),
            "requeued_descendants": requeued_descendants,
        }

    def requeue_descendants(self, job_ids: Iterable[str]) -> int:
        """Invalidate only derived descendants of corrected immutable state."""
        values = tuple(sorted({str(job_id) for job_id in job_ids}))
        if not values:
            return 0
        with self._connect() as db:
            db.execute(
                "CREATE TEMP TABLE IF NOT EXISTS "
                "resume_changed_jobs(job_id TEXT PRIMARY KEY)")
            db.execute("DELETE FROM resume_changed_jobs")
            db.executemany(
                "INSERT OR IGNORE INTO resume_changed_jobs(job_id) VALUES(?)",
                ((job_id,) for job_id in values))
            db.execute(
                "WITH RECURSIVE descendants(job_id) AS ("
                " SELECT d.job_id FROM job_dependencies d "
                " JOIN resume_changed_jobs c ON d.depends_on=c.job_id"
                " UNION "
                " SELECT d.job_id FROM job_dependencies d "
                " JOIN descendants x ON d.depends_on=x.job_id"
                ") "
                "UPDATE jobs SET state='PENDING',"
                "payload=json_remove("
                " json_set(payload,'$.state','PENDING'),"
                " '$.result','$.last_error','$.started_at',"
                " '$.finished_at','$.lease_owner','$.lease_until',"
                " '$.heartbeat'),"
                "lease_owner=NULL,lease_until=NULL,heartbeat=NULL,"
                "last_error=NULL,started_at=NULL,finished_at=NULL "
                "WHERE job_id IN (SELECT job_id FROM descendants)")
            count = int(db.execute("SELECT changes()").fetchone()[0])
            db.execute("DELETE FROM resume_changed_jobs")
        if count:
            self.materialize_json_manifest()
        self.materialize_progress()
        return count

    def claim_ready(self, owner: str, lease_seconds: int = 900, kinds: Iterable[str] | None = None,
                    job_ids: Iterable[str] | None = None,
                    exclude_job_ids: Iterable[str] | None = None) -> dict | None:
        now = time.time()
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute("UPDATE jobs SET state='PENDING', lease_owner=NULL, lease_until=NULL WHERE state='RUNNING' AND lease_until IS NOT NULL AND lease_until<?", (now,))
            kind_clause = ""
            kind_args: tuple[Any, ...] = ()
            if kinds is not None:
                values = tuple(str(x) for x in kinds)
                if not values:
                    db.execute("COMMIT")
                    return None
                kind_clause = " AND j.kind IN (" + ",".join("?" * len(values)) + ")"
                kind_args = values
            if job_ids is not None:
                ids = tuple(str(x) for x in job_ids)
                if not ids:
                    db.execute("COMMIT")
                    return None
                kind_clause += " AND j.job_id IN (" + ",".join("?" * len(ids)) + ")"
                kind_args += ids
            if exclude_job_ids is not None:
                excluded = tuple(str(x) for x in exclude_job_ids)
                if excluded:
                    kind_clause += " AND j.job_id NOT IN (" + ",".join("?" * len(excluded)) + ")"
                    kind_args += excluded
            # Dependency readiness is resolved by indexed anti-join.  The old
            # Python scan revisited most of the large pending DAG per claim.
            row = db.execute(
                "SELECT j.job_id,j.attempt,j.payload FROM jobs j "
                "WHERE j.state='PENDING'" + kind_clause +
                " AND NOT EXISTS (SELECT 1 FROM job_dependencies d "
                "LEFT JOIN jobs p ON p.job_id=d.depends_on "
                "WHERE d.job_id=j.job_id AND (p.job_id IS NULL OR p.state!='COMPLETE')) "
                "ORDER BY j.kind,j.job_id LIMIT 1", kind_args).fetchone()
            if row is not None:
                job_id, attempt, payload = row
                job = json.loads(payload)
                job.update(state="RUNNING", lease_owner=owner, lease_until=now + lease_seconds,
                           started_at=job.get("started_at") or now, attempt=int(attempt) + 1)
                db.execute("UPDATE jobs SET state='RUNNING',payload=?,attempt=?,lease_owner=?,lease_until=?,heartbeat=?,started_at=COALESCE(started_at,?) WHERE job_id=?",
                           (json.dumps(job, default=str), int(attempt) + 1, owner, now + lease_seconds, now,
                            float(job["started_at"]), job_id))
                db.execute("COMMIT")
                return job
            db.execute("COMMIT")
        return None

    def release_claim(self, job_id: str, owner: str) -> None:
        """Return a claimed job to PENDING before a resource is available.

        Keep the persisted JSON payload aligned with the authoritative SQLite
        scheduler columns. Prep-only GPU-feed claims now use this path
        routinely, so leaving payload.state=RUNNING would make jobs.json lie
        about resumable state even though SQLite had already requeued the job.
        """
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT payload,state,lease_owner FROM jobs WHERE job_id=?",
                (str(job_id),),
            ).fetchone()
            if (
                row is None
                or row[1] != "RUNNING"
                or row[2] != str(owner)
            ):
                db.execute("ROLLBACK")
                return
            payload = json.loads(row[0])
            payload["state"] = "PENDING"
            payload.pop("lease_owner", None)
            payload.pop("lease_until", None)
            payload.pop("heartbeat", None)
            payload.pop("started_at", None)
            db.execute(
                "UPDATE jobs SET state='PENDING',payload=?,lease_owner=NULL,"
                "lease_until=NULL,heartbeat=NULL,started_at=NULL "
                "WHERE job_id=? AND state='RUNNING' AND lease_owner=?",
                (
                    json.dumps(payload, default=str),
                    str(job_id), str(owner),
                ),
            )
            db.execute("COMMIT")

    def requeue_reclaimed_job(
        self, job_id: str, owner: str, *, reason: str,
        released_gib: float = 0.0,
        preferred_executor: str | None = None,
    ) -> bool:
        """Return one RAM-reclaimed RUNNING node to PENDING with protection.

        The reclaim counter is part of execution-only job payload state. It
        gives the next admission pass completion priority and prevents the same
        long-running node from being selected as the default victim repeatedly.
        Scientific identity/dependencies are unchanged.
        """
        now = time.time()
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT payload,state,lease_owner FROM jobs WHERE job_id=?",
                (str(job_id),),
            ).fetchone()
            if (
                row is None
                or row[1] != "RUNNING"
                or row[2] != str(owner)
            ):
                db.execute("ROLLBACK")
                return False
            payload = json.loads(row[0])
            payload["state"] = "PENDING"
            payload["ram_reclaim_count"] = (
                int(payload.get("ram_reclaim_count", 0)) + 1)
            payload["ram_last_reclaim_at"] = now
            payload["ram_last_reclaim_reason"] = str(reason)
            payload["ram_last_released_gib"] = float(released_gib)
            if preferred_executor is not None:
                payload["execution_preference"] = str(
                    preferred_executor).upper()
            payload.pop("lease_owner", None)
            payload.pop("lease_until", None)
            payload.pop("started_at", None)
            db.execute(
                "UPDATE jobs SET state='PENDING',payload=?,lease_owner=NULL,"
                "lease_until=NULL,heartbeat=NULL,started_at=NULL,"
                "last_error=NULL,finished_at=NULL WHERE job_id=? "
                "AND state='RUNNING' AND lease_owner=?",
                (json.dumps(payload, default=str), str(job_id), str(owner)),
            )
            if db.total_changes != 1:
                db.execute("ROLLBACK")
                return False
            db.execute("COMMIT")
        return True

    def requeue_execution_failure(
        self, job_id: str, owner: str, *, reason: str,
        preferred_executor: str | None = None,
        count_worker_failure: bool = True,
    ) -> bool:
        """Requeue one abruptly failed lane without losing resumability.

        A child-process failure is an execution incident, not scientific
        evidence and not a reason to discard the complete manifested DAG.
        Preserve the incident in the payload, clear the lease, and let the
        coordinator retry the exact job on the repaired lane.
        """
        now = time.time()
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT payload,state,lease_owner FROM jobs WHERE job_id=?",
                (str(job_id),),
            ).fetchone()
            if (
                row is None
                or row[1] != "RUNNING"
                or row[2] != str(owner)
            ):
                db.execute("ROLLBACK")
                return False
            payload = json.loads(row[0])
            payload["state"] = "PENDING"
            if count_worker_failure:
                payload["worker_failure_count"] = (
                    int(payload.get("worker_failure_count", 0) or 0) + 1)
                payload["last_worker_failure_at"] = now
                payload["last_worker_failure"] = str(reason)
            else:
                # A dispatcher recycle is an orchestrator recovery, not a
                # repeated worker defect. Keep it out of the bounded worker
                # crash budget so a bank can heal indefinitely across resume
                # boundaries without converting healthy jobs to FAILED.
                payload["dispatcher_recovery_count"] = (
                    int(payload.get("dispatcher_recovery_count", 0) or 0)
                    + 1)
                payload["last_dispatcher_recovery_at"] = now
                payload["last_dispatcher_recovery"] = str(reason)
            if preferred_executor is not None:
                payload["execution_preference"] = str(
                    preferred_executor).upper()
            payload.pop("lease_owner", None)
            payload.pop("lease_until", None)
            payload.pop("started_at", None)
            db.execute(
                "UPDATE jobs SET state='PENDING',payload=?,"
                "lease_owner=NULL,lease_until=NULL,heartbeat=NULL,"
                "started_at=NULL,finished_at=NULL,last_error=? "
                "WHERE job_id=? AND state='RUNNING' AND lease_owner=?",
                (
                    json.dumps(payload, default=str), str(reason),
                    str(job_id), str(owner),
                ),
            )
            if db.total_changes != 1:
                db.execute("ROLLBACK")
                return False
            db.execute("COMMIT")
        return True

    def finish_job(self, job_id: str, owner: str, state: str, **fields: Any) -> None:
        if state not in {"COMPLETE", "FAILED", "BLOCKED"}:
            raise ValueError("MANIFESTED_JOB_INVALID_FINISH_STATE")
        now = time.time()
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT payload,state,lease_owner,lease_until FROM jobs WHERE job_id=?", (job_id,)).fetchone()
            if row is None or row[1] != "RUNNING" or row[2] != owner or row[3] is None or float(row[3]) < now:
                db.execute("ROLLBACK")
                raise RuntimeError("MANIFESTED_JOB_JOB_OWNERSHIP_OR_LEASE_INVALID")
            payload = json.loads(row[0])
            payload.update(fields)
            payload["state"] = state
            payload["finished_at"] = now
            db.execute("UPDATE jobs SET state=?,payload=?,lease_owner=NULL,lease_until=NULL,last_error=? ,finished_at=? WHERE job_id=? AND state='RUNNING' AND lease_owner=?",
                       (state, json.dumps(payload, default=str), payload.get("last_error"), now, job_id, owner))
            if db.total_changes != 1:
                db.execute("ROLLBACK")
                raise RuntimeError("MANIFESTED_JOB_JOB_FINISH_RACE")
            db.execute("COMMIT")

    def heartbeat(self, job_id: str, owner: str, lease_seconds: int = 900) -> None:
        with self._connect() as db:
            db.execute("UPDATE jobs SET heartbeat=?,lease_until=? WHERE job_id=? AND lease_owner=? AND state='RUNNING'",
                       (time.time(), time.time() + lease_seconds, job_id, owner))

    def materialize_json_manifest(self) -> None:
        with self._connect() as db:
            target = self.root / "run-state" / "jobs.json"
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + f".{os.getpid()}.tmp")
            # Write incrementally; a complete large DAG must never exist as a
            # second in-memory Python dict alongside the SQLite payloads.
            with temporary.open("w", encoding="utf-8") as handle:
                handle.write("{\n")
                first = True
                for payload, in db.execute("SELECT payload FROM jobs ORDER BY job_id"):
                    job = json.loads(payload)
                    if not first:
                        handle.write(",\n")
                    first = False
                    handle.write(json.dumps(str(job["job_id"])))
                    handle.write(":")
                    handle.write(json.dumps(job, default=str, sort_keys=True))
                handle.write("\n}\n")
            os.replace(temporary, target)

    def materialize_progress(self) -> None:
        """Persist cheap operator telemetry without rewriting the full DAG."""
        payload = self.progress()
        payload["updated_at_epoch"] = time.time()
        _write_json(self.root / "run-state" / "progress.json", payload)

    def all_jobs(self) -> list[dict]:
        with self._connect() as db:
            rows = db.execute(
                "SELECT payload,state,attempt,lease_owner,lease_until,heartbeat,last_error,started_at,finished_at "
                "FROM jobs ORDER BY job_id").fetchall()
        return [self._overlay_sql_state(json.loads(row[0]), row[1:]) for row in rows]

    @staticmethod
    def _overlay_sql_state(payload: dict, sql_state: tuple[Any, ...]) -> dict:
        """Make SQL scheduler columns authoritative over legacy JSON fields."""
        state, attempt, lease_owner, lease_until, heartbeat, last_error, started_at, finished_at = sql_state
        payload.update({"state": state, "attempt": int(attempt),
                        "lease_owner": lease_owner, "lease_until": lease_until,
                        "heartbeat": heartbeat, "last_error": last_error,
                        "started_at": started_at, "finished_at": finished_at})
        return payload

    def jobs_by_kind_state(self, *, kind: str, state: str) -> list[dict]:
        """Read only one indexed scheduler frontier, never the full DAG."""
        with self._connect() as db:
            rows = db.execute(
                "SELECT payload,state,attempt,lease_owner,lease_until,heartbeat,last_error,started_at,finished_at "
                "FROM jobs WHERE state=? AND kind=? ORDER BY job_id",
                (state, kind)).fetchall()
        return [self._overlay_sql_state(json.loads(row[0]), row[1:]) for row in rows]

    def count_jobs_by_kind_state(self, *, kind: str, state: str) -> int:
        """Count one indexed frontier without materializing job payloads."""
        with self._connect() as db:
            row = db.execute(
                "SELECT COUNT(*) FROM jobs WHERE state=? AND kind=?",
                (str(state), str(kind)),
            ).fetchone()
        return int(row[0]) if row else 0

    def ready_jobs(
        self, *, kinds: Iterable[str] | None = None, limit: int = 256,
        exclude_job_ids: Iterable[str] | None = None,
    ) -> list[dict]:
        """Preview dependency-ready PENDING jobs without claiming them."""
        clauses = ["j.state='PENDING'"]
        args: list[Any] = []
        if kinds is not None:
            values = tuple(str(value) for value in kinds)
            if not values:
                return []
            clauses.append(
                "j.kind IN (" + ",".join("?" * len(values)) + ")")
            args.extend(values)
        if exclude_job_ids is not None:
            excluded = tuple(str(value) for value in exclude_job_ids)
            if excluded:
                clauses.append(
                    "j.job_id NOT IN ("
                    + ",".join("?" * len(excluded)) + ")")
                args.extend(excluded)
        sql = (
            "SELECT j.payload,j.state,j.attempt,j.lease_owner,j.lease_until,"
            "j.heartbeat,j.last_error,j.started_at,j.finished_at "
            "FROM jobs j WHERE " + " AND ".join(clauses)
            + " AND NOT EXISTS (SELECT 1 FROM job_dependencies d "
            "LEFT JOIN jobs p ON p.job_id=d.depends_on "
            "WHERE d.job_id=j.job_id "
            "AND (p.job_id IS NULL OR p.state!='COMPLETE')) "
            "ORDER BY j.kind,j.job_id LIMIT ?")
        args.append(max(1, int(limit)))
        with self._connect() as db:
            rows = db.execute(sql, tuple(args)).fetchall()
        return [
            self._overlay_sql_state(json.loads(row[0]), row[1:])
            for row in rows]

    def job_ids_by_kind_prefix(self, *, kind: str, prefix: str) -> list[str]:
        with self._connect() as db:
            rows = db.execute(
                "SELECT job_id FROM jobs WHERE kind=? AND job_id LIKE ? ORDER BY job_id",
                (kind, prefix + "%")).fetchall()
        return [str(row[0]) for row in rows]

    def job(self, job_id: str) -> dict | None:
        with self._connect() as db:
            row = db.execute(
                "SELECT payload,state,attempt,lease_owner,lease_until,heartbeat,last_error,started_at,finished_at "
                "FROM jobs WHERE job_id=?", (job_id,)).fetchone()
        return self._overlay_sql_state(json.loads(row[0]), row[1:]) if row else None

    def progress(self) -> dict:
        with self._connect() as db:
            rows = db.execute("SELECT state,COUNT(*) FROM jobs GROUP BY state").fetchall()
        result = {state: count for state, count in rows}
        result["total"] = sum(result.values())
        return result

    def ready_job_count(self) -> int:
        """Count dependency-ready PENDING jobs without materializing payloads."""
        with self._connect() as db:
            row = db.execute(
                "SELECT COUNT(*) FROM jobs j WHERE j.state='PENDING' "
                "AND NOT EXISTS (SELECT 1 FROM job_dependencies d "
                "LEFT JOIN jobs p ON p.job_id=d.depends_on "
                "WHERE d.job_id=j.job_id "
                "AND (p.job_id IS NULL OR p.state!='COMPLETE'))"
            ).fetchone()
        return int(row[0] or 0) if row else 0

    def running_job_heartbeat_stats(self, owner: str) -> dict[str, float | int | None]:
        """Return cheap liveness facts for one execution owner.

        The Step-9 runner uses this aggregate as a bank-level liveness signal.
        It deliberately reads no job payloads, so a stalled seed cannot make
        the watchdog recreate the old large-DAG scan it is meant to prevent.
        """
        with self._connect() as db:
            row = db.execute(
                "SELECT COUNT(*),MIN(heartbeat),MAX(heartbeat),"
                "MIN(lease_until),MAX(lease_until) "
                "FROM jobs WHERE state='RUNNING' AND lease_owner=?",
                (str(owner),),
            ).fetchone()
        return {
            "running_count": int(row[0] or 0) if row else 0,
            "min_heartbeat": float(row[1]) if row and row[1] is not None else None,
            "max_heartbeat": float(row[2]) if row and row[2] is not None else None,
            "min_lease_until": float(row[3]) if row and row[3] is not None else None,
            "max_lease_until": float(row[4]) if row and row[4] is not None else None,
        }

    def requeue_interrupted_jobs(
        self, *, retry_failed: bool = False, owner: str | None = None,
    ) -> int:
        """Recover unfinished work without touching completed checkpoints.

        When ``owner`` is supplied, recovery is scoped to one independent
        seed bank. This prevents a stalled SHORT/PRIMARY/LONG dispatcher
        from resetting work owned by the other banks.
        """
        states = ("RUNNING", "FAILED") if retry_failed else ("RUNNING",)
        placeholders = ",".join("?" for _ in states)
        owner_clause = ""
        args: tuple[Any, ...] = states
        if owner is not None:
            owner_clause = " AND lease_owner=?"
            args = states + (str(owner),)
        with self._connect() as db:
            cur = db.execute(
                f"UPDATE jobs SET state='PENDING',"
                f"payload=json_remove(json_set(payload,'$.state','PENDING'),"
                f"'$.lease_owner','$.lease_until','$.heartbeat','$.started_at'),"
                f"lease_owner=NULL, lease_until=NULL, heartbeat=NULL,"
                f"last_error=NULL, started_at=NULL, finished_at=NULL "
                f"WHERE state IN ({placeholders}){owner_clause}",
                args)
            count = int(cur.rowcount)
        self.materialize_progress()
        return count

    def ensure_execution_backend_contract(self, contract: Mapping[str, Any]) -> dict[str, Any]:
        """Migrate execution identity without discarding compatible science.

        Numerical backend/kernel changes invalidate computed non-input nodes.
        Scheduler-only changes (RAM admission, queue topology, lane count,
        telemetry) are execution-compatible and keep COMPLETE artifacts.
        """
        marker_path = (
            self.root / "run-state" / "execution-backend-contract.json")
        expected = dict(contract)
        expected["contract_hash"] = stable_hash(expected)
        prior = None
        if marker_path.is_file():
            prior = json.loads(marker_path.read_text(encoding="utf-8"))
        if prior == expected:
            return {
                "status": "UNCHANGED",
                "requeued": 0,
                "contract_hash": expected["contract_hash"],
            }

        semantic_keys = (
            "ridge_execution_backend",
            "hgb_execution_backend",
            "cpu_fallback_backend",
            "hgb_device_policy",
            "hgb_kernel_policy",
            "gpu_pretraining_contract_hash",
        )
        semantic_prior = {
            key: (prior or {}).get(key) for key in semantic_keys}
        semantic_expected = {
            key: expected.get(key) for key in semantic_keys}
        semantic_compatible = (
            prior is not None and semantic_prior == semantic_expected)

        if semantic_compatible:
            _write_json(marker_path, expected)
            return {
                "status": "SCHEDULER_MIGRATED_COMPATIBLE",
                "requeued": 0,
                "previous_contract_hash": (prior or {}).get(
                    "contract_hash"),
                "contract_hash": expected["contract_hash"],
                "semantic_backend_hash": stable_hash(
                    semantic_expected),
            }

        with self._connect() as db:
            cur = db.execute(
                "UPDATE jobs SET state='PENDING', "
                "lease_owner=NULL, lease_until=NULL, heartbeat=NULL, "
                "last_error=NULL, started_at=NULL, finished_at=NULL "
                "WHERE kind!='input'",
            )
            requeued = int(cur.rowcount)
        _write_json(marker_path, expected)
        self.materialize_progress()
        return {
            "status": "NUMERICAL_BACKEND_MIGRATED",
            "requeued": requeued,
            "previous_contract_hash": (prior or {}).get(
                "contract_hash"),
            "contract_hash": expected["contract_hash"],
            "semantic_backend_hash": stable_hash(
                semantic_expected),
        }


def validate_candidate_oos_frame(frame: pd.DataFrame, holdout_boundary: date = HOLDOUT_BOUNDARY,
                                 selection_cutoff: date | None = None) -> None:
    missing = REQUIRED_CANDIDATE_OOS_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"MANIFESTED_JOB_CANDIDATE_OOS_COLUMNS_MISSING:{sorted(missing)}")
    for column in ("decision_date", "terminal_date", "information_available_at"):
        values = pd.to_datetime(frame[column])
        if values.isna().any() or values.ge(pd.Timestamp(holdout_boundary)).any():
            raise PermissionError(f"MANIFESTED_JOB_CANDIDATE_OOS_HOLDOUT_OR_INVALID:{column}")
    decision = pd.to_datetime(frame["decision_date"])
    terminal = pd.to_datetime(frame["terminal_date"])
    available = pd.to_datetime(frame["information_available_at"])
    if not ((decision < terminal) & (terminal <= available)).all():
        raise PermissionError("MANIFESTED_JOB_CANDIDATE_OOS_MATURITY_ORDER_INVALID")
    if not pd.to_numeric(frame["oos_score"], errors="coerce").map(pd.notna).all():
        raise ValueError("MANIFESTED_JOB_CANDIDATE_OOS_NONFINITE_SCORE")
    if not pd.to_numeric(frame["realized_excess"], errors="coerce").map(pd.notna).all():
        raise ValueError("MANIFESTED_JOB_CANDIDATE_OOS_NONFINITE_RETURN")
    keys = frame[["horizon", "fold_id", "candidate_id", "decision_date", "ticker"]]
    if keys.duplicated().any():
        raise ValueError("MANIFESTED_JOB_CANDIDATE_OOS_DUPLICATE_KEY")


def write_candidate_oos_evidence(frame: pd.DataFrame, path: str | Path,
                                 holdout_boundary: date = HOLDOUT_BOUNDARY) -> dict:
    validate_candidate_oos_frame(frame, holdout_boundary)
    ordered = ["fold_id", "horizon", "candidate_id", "recipe_family", "hyperparameters",
               "decision_date", "ticker", "oos_score", "terminal_date", "realized_excess",
               "information_available_at"]
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    frame[ordered].sort_values(["horizon", "fold_id", "candidate_id", "decision_date", "ticker"]).to_parquet(target, index=False)
    return {"path": str(target), "sha256": sha256_file(target), "rows": int(len(frame)),
            "candidate_count": int(frame["candidate_id"].nunique()), "fold_count": int(frame["fold_id"].nunique())}


def write_generation_prediction_store(frame: pd.DataFrame, path: str | Path,
                                      holdout_boundary: date = HOLDOUT_BOUNDARY) -> dict:
    required = {"decision_date", "ticker", "horizon", "generation_id", "model_artifact_id", "recipe_id", "score"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"MANIFESTED_JOB_PREDICTION_COLUMNS_MISSING:{sorted(missing)}")
    decisions = pd.to_datetime(frame["decision_date"])
    if decisions.ge(pd.Timestamp(holdout_boundary)).any():
        raise PermissionError("MANIFESTED_JOB_PREDICTION_HOLDOUT_ACCESS")
    if frame[list(required)].isna().any().any() or not pd.to_numeric(frame["score"], errors="coerce").map(pd.notna).all():
        raise ValueError("MANIFESTED_JOB_PREDICTION_NONFINITE_OR_MISSING")
    key = ["horizon", "generation_id", "decision_date", "ticker"]
    if frame[key].duplicated().any():
        raise ValueError("MANIFESTED_JOB_PREDICTION_DUPLICATE_KEY")
    if "activation_date" in frame and "deactivation_date" in frame:
        active = pd.to_datetime(frame["activation_date"])
        inactive = pd.to_datetime(frame["deactivation_date"])
        if (active > decisions).any() or (inactive.notna() & inactive.lt(decisions)).any():
            raise ValueError("MANIFESTED_JOB_PREDICTION_OUTSIDE_GENERATION_INTERVAL")
    if "artifact_path" in frame and "artifact_sha256" in frame:
        for artifact_path, artifact_sha in frame[["artifact_path", "artifact_sha256"]].drop_duplicates().itertuples(index=False):
            if not Path(str(artifact_path)).is_file() or sha256_file(str(artifact_path)) != str(artifact_sha):
                raise ValueError("MANIFESTED_JOB_PREDICTION_ARTIFACT_HASH_MISMATCH")
    target = Path(path); target.parent.mkdir(parents=True, exist_ok=True)
    keep = sorted(required | {x for x in ("activation_date", "deactivation_date", "artifact_sha256") if x in frame})
    frame[keep].sort_values(["horizon", "generation_id", "decision_date", "ticker"]).to_parquet(target, index=False)
    return {"path": str(target), "sha256": sha256_file(target), "rows": int(len(frame))}


def write_calibration_store(frame: pd.DataFrame, path: str | Path) -> dict:
    required = {"generation_id", "calibration_start", "calibration_end", "maturity_cutoff", "source_prediction_sha256",
                "score_quantile", "resolved_threshold", "resolved_top_fraction", "calibration_fingerprint", "matured_observation_count"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"MANIFESTED_JOB_CALIBRATION_COLUMNS_MISSING:{sorted(missing)}")
    if frame["generation_id"].duplicated().any():
        raise ValueError("MANIFESTED_JOB_CALIBRATION_DUPLICATE_GENERATION")
    if frame.isna().any().any() or not pd.to_numeric(frame["resolved_threshold"], errors="coerce").map(pd.notna).all():
        raise ValueError("MANIFESTED_JOB_CALIBRATION_MISSING_OR_NONFINITE")
    if (pd.to_datetime(frame["calibration_start"]) > pd.to_datetime(frame["calibration_end"])).any():
        raise ValueError("MANIFESTED_JOB_CALIBRATION_INTERVAL_INVALID")
    target = Path(path); target.parent.mkdir(parents=True, exist_ok=True)
    frame[sorted(required)].to_parquet(target, index=False)
    return {"path": str(target), "sha256": sha256_file(target), "generations": int(len(frame))}


def normalized_replay_result(*, assessment_period: str, horizon: int, holding_days: int,
                             max_names: int, recipe: str, generation_id: str,
                             result: Mapping[str, Any]) -> dict:
    metrics = dict(result.get("metrics", {}))
    curve = result.get("curve", pd.DataFrame())
    required_metrics = {
        "terminal_value": "nav", "urth_terminal_value": "urth_nav", "trade_count": "trade_count",
        "turnover": "turnover", "total_cost_eur": "costs", "cagr_excess": "cagr_excess",
        "worst_relative_drawdown": "relative_max_drawdown",
    }
    row = {"assessment_period": assessment_period, "horizon": horizon, "holding_days": holding_days,
           "max_names": max_names, "recipe": recipe, "generation_id": generation_id}
    row.update({output: metrics.get(source) for source, output in required_metrics.items()})
    if not curve.empty:
        relative = curve["strategy_value"].astype(float) / curve["urth_value"].astype(float)
        row["relative_wealth"] = float(relative.iloc[-1])
        row["exposure"] = float(curve.get("stock_exposure", pd.Series(dtype=float)).mean()) if "stock_exposure" in curve else None
    else:
        row.update(relative_wealth=None, exposure=None)
    return row


def stock_dividend_audit(*, trades: pd.DataFrame, distributions: pd.DataFrame) -> pd.DataFrame:
    required_trades = {"ticker", "entry_date", "exit_date"}
    required_distributions = {"ticker", "ex_date", "payable_date", "cash_amount"}
    if required_trades - set(trades) or required_distributions - set(distributions):
        raise ValueError("MANIFESTED_JOB_STOCK_DIVIDEND_AUDIT_COLUMNS_MISSING")
    if trades.empty or distributions.empty:
        return pd.DataFrame(columns=["ticker", "entry_date", "exit_date", "ex_date", "payable_date", "cash_amount"])
    left = trades[["ticker", "entry_date", "exit_date"]].copy()
    right = distributions[["ticker", "ex_date", "payable_date", "cash_amount"]].copy()
    for frame, columns in ((left, ("entry_date", "exit_date")), (right, ("ex_date", "payable_date"))):
        for column in columns:
            frame[column] = pd.to_datetime(frame[column]).dt.normalize()
    merged = left.merge(right, on="ticker", how="inner")
    return merged.loc[(merged["entry_date"] < merged["ex_date"]) & (merged["ex_date"] <= merged["exit_date"])].reset_index(drop=True)


def incubator_is_authoritative(*, birth_date: date, evidence_date: date) -> bool:
    return evidence_date >= birth_date


def propose_incubator_family(*, family_id: str, parent_family_ids: Iterable[str], proposed_at: date,
                             information_cutoff: date, birth_date: date, first_eligible_refit: date,
                             family_spec_hash: str) -> dict:
    if first_eligible_refit < birth_date or information_cutoff > birth_date:
        raise ValueError("MANIFESTED_JOB_INCUBATOR_BIRTH_OR_CUTOFF_INVALID")
    return {"family_id": family_id, "parent_family_ids": sorted(set(parent_family_ids)),
            "proposed_at": str(proposed_at), "information_cutoff": str(information_cutoff),
            "birth_date": str(birth_date), "first_eligible_refit": str(first_eligible_refit),
            "family_spec_hash": family_spec_hash, "status": "PROPOSED"}


def incubator_evidence_allowed(proposal: Mapping[str, Any], evidence_date: date) -> bool:
    return incubator_is_authoritative(birth_date=date.fromisoformat(str(proposal["birth_date"])),
                                      evidence_date=evidence_date)


def append_incubator_proposal(path: str | Path, proposal: Mapping[str, Any]) -> dict:
    """Persist an immutable family proposal; pre-birth evidence is never authoritative."""
    target = Path(path)
    payload = json.loads(target.read_text(encoding="utf-8")) if target.is_file() else {
        "schema_version": "DYNAMIC_QBD_FAMILY_INCUBATOR_V1", "families": []}
    if any(str(x.get("family_id")) == str(proposal.get("family_id")) for x in payload["families"]):
        raise ValueError("MANIFESTED_JOB_INCUBATOR_DUPLICATE_FAMILY")
    payload["families"].append(dict(proposal))
    _write_json(target, payload)
    return dict(proposal)


def input_quality_manifest(path: str | Path, *, date_column_candidates: tuple[str, ...], key_columns: tuple[str, ...],
                           development_end: date, holdout_boundary: date,
                           universe_column: str | None = "ticker",
                           distribution_semantics: bool = False) -> dict:
    """Audit only the development slice; never materialize holdout rows."""
    dataset = ds.dataset(str(path), format="parquet")
    names = set(dataset.schema.names)
    column = next((x for x in date_column_candidates if x in names), None)
    if column is None:
        return {"status": "BLOCKED_DATE_COLUMN_MISSING", "path": str(path), "columns": sorted(names)}
    selected_columns = list(dict.fromkeys([column, *[x for x in key_columns if x in names],
                                           *([universe_column] if universe_column and universe_column in names else [])]))
    field_type = dataset.schema.field(column).type
    as_string = pa.types.is_string(field_type) or pa.types.is_large_string(field_type)
    dev_value = development_end.isoformat() if as_string else pd.Timestamp(development_end).to_datetime64()
    holdout_value = holdout_boundary.isoformat() if as_string else pd.Timestamp(holdout_boundary).to_datetime64()
    predicate = (ds.field(column) <= pa.scalar(dev_value)) & (ds.field(column) < pa.scalar(holdout_value))
    if "holdout_locked" in names:
        predicate = predicate & ((ds.field("holdout_locked") == False) | ds.field("holdout_locked").is_null())
        selected_columns.append("holdout_locked")
    # Stream the quality audit in Arrow batches.  Materializing the complete
    # signal panel here made startup allocate gigabytes before the scheduler
    # even created its first worker.  The audit only needs aggregates and a
    # bounded set of duplicate keys, so preserve the same semantics without a
    # monolithic pandas frame.
    key_names = [x for x in key_columns if x in names]
    seen_keys: set[tuple[Any, ...]] = set()
    duplicate_key_count = 0
    observed_dates: set[pd.Timestamp] = set()
    per_date_counts: dict[pd.Timestamp, int] = {}
    min_date = max_date = None
    row_count = 0
    invalid_date_count = 0
    for batch in dataset.to_batches(columns=list(dict.fromkeys(selected_columns)), filter=predicate,
                                    batch_size=65536):
        frame = batch.to_pandas()
        dates = pd.to_datetime(frame[column], errors="coerce")
        invalid_date_count += int(dates.isna().sum())
        valid_dates = dates.dropna().dt.normalize()
        if len(valid_dates):
            batch_min, batch_max = valid_dates.min(), valid_dates.max()
            min_date = batch_min if min_date is None or batch_min < min_date else min_date
            max_date = batch_max if max_date is None or batch_max > max_date else max_date
            observed_dates.update(valid_dates.tolist())
            for value, count in valid_dates.value_counts().items():
                per_date_counts[value] = per_date_counts.get(value, 0) + int(count)
        row_count += len(frame)
        if key_names:
            for values in frame[key_names].itertuples(index=False, name=None):
                key = tuple(None if pd.isna(value) else value for value in values)
                if key in seen_keys:
                    duplicate_key_count += 1
                else:
                    seen_keys.add(key)
    expected = pd.bdate_range(min_date, max_date) if min_date is not None else pd.DatetimeIndex([])
    observed = pd.DatetimeIndex(sorted(observed_dates))
    missing = expected.difference(observed)
    per_date = pd.Series(per_date_counts, dtype="int64")
    return {"status": "PASS" if duplicate_key_count == 0 and invalid_date_count == 0 else "BLOCKED",
            "path": str(path), "development_content_hash": parquet_development_slice_sha256(
                path, date_column=column, development_end=development_end,
                holdout_boundary=holdout_boundary, projected_columns=selected_columns,
                semantics="EX_DATE_LE_DEVELOPMENT_END_PAYABLE_DATE_MAY_FOLLOW" if distribution_semantics else "DATE_LE_DEVELOPMENT_END"),
            "date_column": column,
            "start": str(min_date.date()) if min_date is not None else None,
            "end": str(max_date.date()) if max_date is not None else None,
            "row_count": int(row_count), "duplicate_key_count": duplicate_key_count,
            "missing_business_date_count": int(len(missing)),
            "universe_min": int(per_date.min()) if not per_date.empty else 0,
            "universe_max": int(per_date.max()) if not per_date.empty else 0,
            "schema_columns": sorted(names), "development_end": str(development_end),
            "holdout_boundary": str(holdout_boundary),
            "distribution_semantics": "EX_DATE_BOUND_PAYABLE_MAY_FOLLOW" if distribution_semantics else None}


def _family_registry(inputs: ManifestedJobInputs, feature_schema_sha256: str) -> dict:
    families = build_family_specs(
        feature_schema_sha256=feature_schema_sha256,
        training_window_sessions=inputs.training_window_sessions,
        calibration_window_sessions=inputs.calibration_window_sessions,
        purge_sessions=inputs.purge_sessions,
        refit_cadence=inputs.refit_cadence,
        score_quantile=inputs.score_quantile,
        top_fraction=inputs.top_fraction,
        cost_contract=dict(inputs.cost_model),
        tax_contract=dict(inputs.tax_contract),
        include_learned_exit=False,
        random_seed=inputs.random_seed,
    )
    model_families = []
    for horizon in range(1, 31):
        model_families.append({
            "model_family_key": f"H{horizon:02d}_RIDGE_HGB_FROZEN_RULE",
            "horizon": horizon,
            "recipe_id": "RIDGE_HGB_FROZEN_RULE",
            "training_contract": {
                "feature_schema_sha256": feature_schema_sha256,
                "training_window_sessions": inputs.training_window_sessions,
                "calibration_window_sessions": inputs.calibration_window_sessions,
                "refit_cadence": inputs.refit_cadence,
                "maturity_rule": inputs.maturity_rule,
                "seed": int(inputs.random_seed),
            },
        })
    portfolio = []
    for family in families:
        payload = asdict(family)
        horizon = int(family.horizon_sessions)
        payload["model_family_key"] = f"H{horizon:02d}_RIDGE_HGB_FROZEN_RULE"
        payload["portfolio_family_key"] = f"H{horizon:02d}_D{family.holding_days:02d}_N{family.max_names:02d}_FIXED"
        portfolio.append(payload)
    hyperparameter_path = inputs.hyperparameter_space or (inputs.repo_root / "stock_predictor" / "v5" / "hyperparameter_space.json")
    if not hyperparameter_path.is_file():
        raise FileNotFoundError(f"MANIFESTED_JOB_HYPERPARAMETER_SPACE_MISSING:{hyperparameter_path}")
    candidates = load_primary_candidate_registry(hyperparameter_path)
    candidate_registry = candidate_registry_document(candidates, source_sha256="QBD_PRIMARY_RIDGE_HGB_V1")
    result = {
        "schema_version": "DYNAMIC_QBD_PRIMARY_FAMILY_REGISTRY_V1",
        "random_seed": int(inputs.random_seed),
        "model_family_count": len(model_families),
        "portfolio_family_count": len(portfolio),
        "model_families": model_families,
        "portfolio_families": portfolio,
        "candidate_registry": candidate_registry,
    }
    result["family_registry_hash"] = stable_hash(result)
    return result


def _candidate_universe(metrics_path: Path) -> dict[tuple[int, str, str], dict]:
    raw = json.loads(metrics_path.read_text(encoding="utf-8"))
    rows = raw if isinstance(raw, list) else raw.get("records", raw.get("candidates", []))
    universe = {}
    for row in rows:
        horizon = int(row.get("horizon_sessions", row.get("horizon", 0)))
        candidate = str(row.get("candidate_id", ""))
        fold = str(row.get("fold_id", ""))
        if 1 <= horizon <= 30 and candidate and fold:
            universe[(horizon, fold, candidate)] = {
                "recipe_family": str(row.get("family", row.get("recipe_family", ""))),
                "hyperparameters": json.dumps(row.get("parameters", row.get("hyperparameters", {})), sort_keys=True, default=str),
            }
    return universe


def select_recipe_from_candidate_oos(frame: pd.DataFrame, *, horizon: int, selection_cutoff: date) -> dict:
    """Legacy diagnostic selector; the primary DAG uses the snapshot selector."""
    validate_candidate_oos_frame(frame, HOLDOUT_BOUNDARY)
    cutoff = pd.Timestamp(selection_cutoff)
    eligible = frame.loc[(frame["horizon"].astype(int) == int(horizon)) &
                         (pd.to_datetime(frame["information_available_at"]) <= cutoff)].copy()
    if eligible.empty:
        raise RuntimeError("MANIFESTED_JOB_RECIPE_SELECTION_NO_MATURED_EVIDENCE")
    rows = []
    for (candidate_id, fold_id), group in eligible.groupby(["candidate_id", "fold_id"], sort=True):
        prediction = pd.to_numeric(group["oos_score"], errors="coerce")
        outcome = pd.to_numeric(group["realized_excess"], errors="coerce")
        rows.append({"candidate_id": str(candidate_id), "fold_id": str(fold_id),
                     "spearman": float(prediction.rank().corr(outcome.rank())) if len(group) > 1 else 0.0,
                     "mean_realized_excess_diagnostic": float(outcome.mean())})
    fold = pd.DataFrame(rows)
    ranking = fold.groupby("candidate_id").agg(
        median_fold_spearman=("spearman", "median"),
        mean_fold_spearman=("spearman", "mean"),
        fold_count=("fold_id", "nunique"),
    ).reset_index()
    ranking = ranking.sort_values(["median_fold_spearman", "mean_fold_spearman", "candidate_id"],
                                  ascending=[False, False, True])
    winner = ranking.iloc[0]
    runner = ranking.iloc[1] if len(ranking) > 1 else None
    winner_rows = eligible.loc[eligible["candidate_id"].eq(winner["candidate_id"])]
    return {"horizon": int(horizon), "selected_candidate_id": str(winner["candidate_id"]),
            "selection_cutoff": str(selection_cutoff), "winner_evidence": winner.to_dict(),
            "runner_up_evidence": runner.to_dict() if runner is not None else None,
            "fold_count": int(winner["fold_count"]), "fold_ids": sorted(winner_rows["fold_id"].astype(str).unique()),
            "diagnostic_only": True}


def persist_recipe_selection(path: str | Path, selection: Mapping[str, Any]) -> dict:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    prior = json.loads(target.read_text(encoding="utf-8")) if target.is_file() else {"schema_version": "DYNAMIC_QBD_RECIPE_SELECTION_V1", "records": []}
    key = (int(selection["horizon"]), str(selection["selection_cutoff"]))
    if any((int(x["horizon"]), str(x["selection_cutoff"])) == key for x in prior["records"]):
        raise ValueError("MANIFESTED_JOB_RECIPE_SELECTION_IMMUTABLE_DUPLICATE")
    prior["records"].append(dict(selection))
    _write_json(target, prior)
    return dict(selection)


def _candidate_oos_audit(path: Path | None, holdout: date, metrics_path: Path | None = None) -> dict:
    if path is None or not path.is_file():
        return {"status": "GENERATABLE_CANDIDATE_OOS_EVIDENCE_MISSING", "path": str(path) if path else None}
    if path.suffix.lower() == ".parquet":
        dataset = ds.dataset(str(path), format="parquet")
        names = set(dataset.schema.names)
        if "information_available_at" not in names:
            return {"status": "BLOCKED_CANDIDATE_OOS_SCHEMA_INCOMPLETE",
                    "missing_columns": ["information_available_at"], "path": str(path)}
        field_type = dataset.schema.field("information_available_at").type
        boundary = holdout.isoformat() if pa.types.is_string(field_type) or pa.types.is_large_string(field_type) else pd.Timestamp(holdout).to_datetime64()
        frame = dataset.to_table(filter=ds.field("information_available_at") < pa.scalar(boundary)).to_pandas()
    else:
        frame = pd.read_csv(path)
    missing = sorted(REQUIRED_CANDIDATE_OOS_COLUMNS - set(frame.columns))
    if missing:
        return {"status": "BLOCKED_CANDIDATE_OOS_EVIDENCE_SCHEMA_INCOMPLETE", "missing_columns": missing,
                "path": str(path)}
    try:
        validate_candidate_oos_frame(frame, holdout)
    except (ValueError, PermissionError) as exc:
        return {"status": "BLOCKED_CANDIDATE_OOS_EVIDENCE_INVALID", "reason": str(exc), "path": str(path)}
    expected = _candidate_universe(metrics_path) if metrics_path and metrics_path.is_file() else {}
    actual = {(int(h), str(f), str(c)): row for (h, f, c), row in frame.groupby(["horizon", "fold_id", "candidate_id"], sort=False).first().iterrows()}
    if expected:
        missing_keys = sorted(set(expected) - set(actual))
        extra_horizons = sorted(set(int(x) for x in frame["horizon"]) - set(range(1, 31)))
        if missing_keys or extra_horizons:
            return {"status": "BLOCKED_CANDIDATE_OOS_EVIDENCE_COVERAGE_INCOMPLETE", "path": str(path),
                    "missing_candidate_keys": len(missing_keys), "sample_missing": [list(x) for x in missing_keys[:10]],
                    "extra_horizons": extra_horizons}
        for key, expected_row in expected.items():
            row = actual[key]
            if str(row["recipe_family"]) != expected_row["recipe_family"] or str(row["hyperparameters"]) != expected_row["hyperparameters"]:
                return {"status": "BLOCKED_CANDIDATE_OOS_CANDIDATE_MAPPING_DRIFT", "key": list(key), "path": str(path)}
    if set(int(x) for x in frame["horizon"]) != set(range(1, 31)):
        return {"status": "BLOCKED_CANDIDATE_OOS_HORIZON_COVERAGE_INCOMPLETE", "path": str(path)}
    return {"status": "COMPLETE", "path": str(path), "rows": int(len(frame)),
            "sha256": sha256_file(path), "candidate_count": int(frame["candidate_id"].nunique()),
            "fold_count": int(frame["fold_id"].nunique()), "horizon_count": int(frame["horizon"].nunique()),
            "candidate_universe_hash": stable_hash({str(k): v for k, v in sorted(expected.items())}) if expected else None,
            "coverage_manifest": {str(h): int((frame["horizon"] == h).sum()) for h in range(1, 31)}}


def _job_graph(registry: dict, refit_dates_by_horizon: Mapping[int, Iterable[date]],
               folds_by_horizon: Mapping[int, Iterable[Any]], *, trading_sessions: Iterable[date],
               fold_policy: FoldPolicy, target_contract: TargetContract,
               recipe_selection_policy: RecipeSelectionPolicy,
               model_training_contract: ModelTrainingContract,
               development_end: date | None = None) -> dict[str, dict]:
    jobs: dict[str, dict] = {}
    families_by_model: dict[str, list[dict]] = {}
    for family in registry["portfolio_families"]:
        families_by_model.setdefault(family["model_family_key"], []).append(family)
    jobs["candidate_registry_ready"] = {"job_id": "candidate_registry_ready", "kind": "input", "state": "COMPLETE", "depends_on": []}
    candidate_records = registry.get("candidate_registry", {}).get("records", [])
    fold_policy_hash = fold_policy.fold_policy_hash
    target_contract_hash = target_contract.target_contract_hash
    # This lookup is called once per candidate×fold and again for every
    # consuming refit.  Cache it at the immutable graph boundary; the old
    # implementation rebuilt a normalized session tuple and index for every
    # call, making graph construction appear hung on the full H1-H30 surface.
    fold_maturity = {
        (int(horizon), str(fold.fold_id)): fold_information_available_at(
            fold=fold, horizon=horizon, trading_sessions=trading_sessions
        )
        for horizon, folds in folds_by_horizon.items()
        for fold in folds
    }
    for horizon in sorted(folds_by_horizon):
      folds = tuple(folds_by_horizon[horizon])
      max_consuming_cutoff = max(tuple(refit_dates_by_horizon.get(horizon, ())), default=None)
      for record in candidate_records:
        candidate_id = str(record["candidate_id"])
        for fold in folds:
            available_at = fold_maturity[(int(horizon), str(fold.fold_id))]
            if available_at is None or max_consuming_cutoff is None or available_at > max_consuming_cutoff:
                continue
            job_id = f"candidate_oos_fold:{horizon}:{fold.fold_id}:{candidate_id}"
            jobs[job_id] = {"job_id": job_id, "kind": "candidate_oos_fold", "horizon": int(horizon),
                            "fold_id": fold.fold_id, "candidate_id": candidate_id, "state": "PENDING",
                            "fold_policy_hash": fold_policy_hash, "target_contract_hash": target_contract_hash,
                            "model_training_contract_hash": model_training_contract.model_training_contract_hash,
                            "depends_on": ["candidate_registry_ready"]}
    for horizon, refit_dates in refit_dates_by_horizon.items():
      model_key = f"H{horizon:02d}_RIDGE_HGB_FROZEN_RULE"
      previous_replay: dict[str, str] = {}
      refit_dates = tuple(refit_dates)
      for refit_index, cutoff in enumerate(refit_dates):
        prediction_end = refit_dates[refit_index + 1] if refit_index + 1 < len(refit_dates) else development_end
        candidate_dependencies = []
        for fold in folds_by_horizon.get(horizon, ()):
            available_at = fold_maturity[(int(horizon), str(fold.fold_id))]
            if available_at is not None and available_at <= cutoff:
                candidate_dependencies.extend(f"candidate_oos_fold:{horizon}:{fold.fold_id}:{record['candidate_id']}" for record in candidate_records)
        coverage = f"candidate_evidence_coverage:{cutoff}:{model_key}"
        jobs[coverage] = {"job_id": coverage, "kind": "candidate_evidence_coverage", "cutoff": str(cutoff),
                          "model_family_key": model_key, "state": "PENDING",
                          "depends_on": sorted(set(candidate_dependencies)) or ["candidate_registry_ready"]}
        selection = f"recipe_selection:{cutoff}:{model_key}"
        jobs[selection] = {"job_id": selection, "kind": "recipe_selection", "cutoff": str(cutoff),
                           "model_family_key": model_key,
                           "recipe_selection_policy_hash": recipe_selection_policy.recipe_selection_policy_hash,
                           "state": "PENDING", "depends_on": [coverage]}
        base = f"model:{cutoff}:{model_key}"
        jobs[base] = {"job_id": base, "kind": "model", "cutoff": str(cutoff),
                      "model_family_key": model_key, "prediction_end": str(prediction_end) if prediction_end else None,
                      "state": "PENDING", "depends_on": [selection]}
        for child_kind in ("calibration", "prediction"):
            child = f"{child_kind}:{cutoff}:{model_key}"
            jobs[child] = {"job_id": child, "kind": child_kind, "cutoff": str(cutoff),
                           "model_family_key": model_key, "state": "PENDING", "depends_on": [base]}
        ready = f"generation_ready:{cutoff}:{model_key}"
        jobs[ready] = {"job_id": ready, "kind": "generation_ready", "cutoff": str(cutoff),
                       "model_family_key": model_key, "state": "PENDING",
                       "depends_on": [f"calibration:{cutoff}:{model_key}", f"prediction:{cutoff}:{model_key}"]}
        for family in families_by_model.get(model_key, ()):
            family_key = family["portfolio_family_key"]
            replay = f"replay:{cutoff}:{family_key}"
            deps = [ready]
            if family_key in previous_replay:
                deps.append(previous_replay[family_key])
            jobs[replay] = {"job_id": replay, "kind": "replay", "cutoff": str(cutoff),
                            "portfolio_family_key": family_key, "generation_ready_job_id": ready,
                            "segment_start": str(cutoff),
                            "segment_end": str(prediction_end) if prediction_end else None,
                            "state": "PENDING", "depends_on": deps}
            evidence = f"evidence:{cutoff}:{family_key}"
            jobs[evidence] = {"job_id": evidence, "kind": "evidence", "cutoff": str(cutoff),
                              "portfolio_family_key": family_key, "state": "PENDING", "depends_on": [replay]}
            previous_replay[family_key] = replay

    return jobs


def _compatible_resume_contract_view(contract: Mapping[str, Any]) -> dict:
    """Return the JSON-semantic scientific contract used for compatible resume.

    Contracts loaded from disk contain JSON arrays, while freshly-built
    dataclass contracts may still contain tuples (notably policy tie-breakers).
    Comparing the native Python objects would reject an otherwise identical
    scientific contract.  Canonical JSON normalization keeps this check
    strict on values while making serialization-only container differences
    irrelevant.
    """
    ignored = {
        "git_sha", "source_tree_sha256", "worktree_status",
        "dirty_patch_sha256", "run_contract_hash",
    }
    def normalize(value: Any, ancestry: tuple[str, ...] = ()) -> Any:
        if isinstance(value, Mapping):
            return {
                str(key): normalize(item, ancestry + (str(key),))
                for key, item in value.items()
                # Input-quality paths can point into an immutable hotload
                # snapshot on resume.  Content hashes, schema and stat
                # fingerprints remain strict, so the materialization path is
                # runtime identity rather than scientific identity.
                if not ("input_quality" in ancestry and str(key) == "path")
            }
        if isinstance(value, (list, tuple)):
            return [normalize(item, ancestry) for item in value]
        return value

    view = {
        key: value for key, value in contract.items() if key not in ignored
    }
    view = normalize(view)
    return json.loads(json.dumps(view, default=str, sort_keys=True, allow_nan=False))


def initialize_manifested_job_coordinator(*, inputs: ManifestedJobInputs, output_root: str | Path,
                        candidate_oos_evidence: str | Path | None = None,
                        allow_compatible_code_resume: bool = False,
                        reconcile_existing_graph: bool = False) -> dict:
    output = Path(output_root)
    store = ManifestedJobStore(output)
    for required in (inputs.signal_panel, inputs.feature_schema,
                     inputs.benchmark_prices, inputs.benchmark_distributions):
        if not required.is_file():
            raise FileNotFoundError(f"MANIFESTED_JOB_INPUT_MISSING:{required}")
    existing_contract = output / "manifested-job-contract.json"
    if existing_contract.is_file() and (output / "run-state" / "jobs.sqlite3").is_file():
        prior = json.loads(existing_contract.read_text(encoding="utf-8"))
        if prior.get("git_sha") != current_git_sha(inputs.repo_root) and not allow_compatible_code_resume:
            raise RuntimeError("MANIFESTED_JOB_EXISTING_STATE_GIT_MISMATCH_USE_NEW_ARTIFACT_ROOT")
        prior_stats = prior.get("input_stat_fingerprint")
        if prior_stats and prior_stats != input_stat_fingerprint(inputs):
            raise RuntimeError("MANIFESTED_JOB_EXISTING_STATE_INPUT_STAT_MISMATCH_USE_NEW_ARTIFACT_ROOT")
        if not reconcile_existing_graph:
            progress = store.progress()
            progress.update({
                "status": (
                    "RESUMED_FAST_COMPATIBLE_CODE"
                    if allow_compatible_code_resume else "RESUMED_FAST"),
                "run_contract_hash": prior.get("run_contract_hash"),
                "prior_git_sha": prior.get("git_sha"),
                "resume_git_sha": current_git_sha(inputs.repo_root),
                "graph_reconciliation": "SKIPPED_VALIDATED_FAST_RESUME",
            })
            store.write("run-state/progress.json", progress)
            return {
                **(json.loads(
                    (output / "summary.json").read_text(encoding="utf-8"))
                   if (output / "summary.json").is_file() else {}),
                "status": "RESUMED_FAST", "resumed": True,
                "progress": progress,
            }
    feature_hash = sha256_file(inputs.feature_schema)
    candidate_path = Path(candidate_oos_evidence) if candidate_oos_evidence else None
    candidate_audit = _candidate_oos_audit(candidate_path, inputs.holdout_boundary, inputs.candidate_metrics)
    stock_quality = input_quality_manifest(inputs.stock_execution_prices, date_column_candidates=("date", "session_date", "decision_date"),
                                           key_columns=("date", "session_date", "ticker"),
                                           development_end=inputs.development_end, holdout_boundary=inputs.holdout_boundary) if inputs.stock_execution_prices and inputs.stock_execution_prices.is_file() else {"status": "BLOCKED_STOCK_EXECUTION_PRICE_INPUT_MISSING"}
    stock_dividend_status = input_quality_manifest(
        inputs.stock_distributions,
        date_column_candidates=("ex_date",),
        key_columns=("ex_date", "ticker"),
        development_end=inputs.development_end,
        holdout_boundary=inputs.holdout_boundary,
        distribution_semantics=True,
    ) if inputs.stock_distributions and inputs.stock_distributions.is_file() else {"status": "BLOCKED_STOCK_DISTRIBUTION_INPUT_MISSING"}
    registry = _family_registry(inputs, feature_hash)
    resolved = inputs.resolved_contracts(
        primary_candidate_universe_hash=registry["candidate_registry"]["candidate_registry_sha256"])
    signal_development_hash = _cached_development_slice_hash(
        inputs.signal_panel, date_column="decision_date", development_end=inputs.development_end,
        holdout_boundary=inputs.holdout_boundary)
    benchmark_development_hash = _cached_development_slice_hash(
        inputs.benchmark_prices, date_column="date", development_end=inputs.development_end,
        holdout_boundary=inputs.holdout_boundary)
    benchmark_distribution_development_hash = _cached_development_slice_hash(
        inputs.benchmark_distributions, date_column="ex_date", development_end=inputs.development_end,
        holdout_boundary=inputs.holdout_boundary)
    stock_execution_development_hash = (_cached_development_slice_hash(
        inputs.stock_execution_prices, date_column="date", development_end=inputs.development_end,
        holdout_boundary=inputs.holdout_boundary) if inputs.stock_execution_prices and inputs.stock_execution_prices.is_file() else None)
    signal_sessions = parquet_dates(inputs.signal_panel, ("decision_date",))
    benchmark_dates = parquet_dates(inputs.benchmark_prices, ("date", "session_date"))
    data_boundary_audit = {
        "signal_panel_start": str(signal_sessions.min().date()),
        "signal_panel_end": str(signal_sessions.max().date()),
        "benchmark_price_start": str(benchmark_dates.min().date()),
        "benchmark_price_end": str(benchmark_dates.max().date()),
        "requested_development_start": str(inputs.development_start),
        "requested_development_end": str(inputs.development_end),
    }
    boundary_errors = []
    if signal_sessions.max().date() < inputs.development_end:
        boundary_errors.append("SIGNAL_PANEL_END_BEFORE_REQUESTED_DEVELOPMENT_END")
    if benchmark_dates.max().date() < inputs.development_end:
        boundary_errors.append("BENCHMARK_PRICE_END_BEFORE_REQUESTED_DEVELOPMENT_END")
    effective_start = max(inputs.development_start, signal_sessions.min().date())
    if signal_sessions.min().date() > inputs.development_start:
        data_boundary_audit["effective_development_start"] = str(effective_start)
        data_boundary_audit["start_adjustment"] = "FEATURE_WARMUP_NO_OBSERVABLE_SIGNAL_BEFORE_PANEL_START"
    else:
        data_boundary_audit["effective_development_start"] = str(inputs.development_start)
        data_boundary_audit["start_adjustment"] = None
    data_boundary_audit["status"] = "PASS" if not boundary_errors else "BLOCKED"
    data_boundary_audit["errors"] = boundary_errors
    sessions = tuple(signal_sessions.dt.date.drop_duplicates().sort_values())
    refit_dates_by_horizon = {h: tuple(x for x in _eligible_refit_dates(sessions, h, inputs.resolved_fold_policy())
                                          if effective_start <= x <= inputs.development_end)
                              for h in range(1, 31)}
    refit_dates = tuple(sorted(set().union(*[set(x) for x in refit_dates_by_horizon.values()])))
    folds_by_horizon = {}
    for horizon in range(1, 31):
        fold_dates = [x.isoformat() for x in sessions if x <= inputs.development_end and x < inputs.holdout_boundary]
        try:
            policy = inputs.resolved_fold_policy()
            folds_by_horizon[horizon] = tuple(expanding_folds(fold_dates,
                                                               minimum_train_dates=policy.training_window_sessions,
                                                               validation_dates=policy.validation_window_sessions,
                                                               step_dates=policy.step_sessions,
                                                               purge_dates=policy.purge_sessions,
                                                               embargo_dates=policy.embargo_sessions))
        except ValueError:
            folds_by_horizon[horizon] = ()
    # Primary refits are evidence-aware: no selection node is created before
    # at least the policy's minimum independent folds are fully mature.
    minimum_primary_folds = resolved.recipe_selection_policy.minimum_candidate_folds
    for horizon, refit_dates in list(refit_dates_by_horizon.items()):
        fold_maturity = {
            str(fold.fold_id): fold_information_available_at(
                fold=fold, horizon=horizon, trading_sessions=sessions
            )
            for fold in folds_by_horizon.get(horizon, ())
        }
        mature_by_cutoff = []
        for cutoff in refit_dates:
            mature_count = sum(
                1 for fold in folds_by_horizon.get(horizon, ())
                if (fold_maturity.get(str(fold.fold_id)) is not None and
                    fold_maturity[str(fold.fold_id)] <= cutoff))
            if mature_count >= minimum_primary_folds:
                mature_by_cutoff.append(cutoff)
        refit_dates_by_horizon[horizon] = tuple(mature_by_cutoff)
    worktree = subprocess.run(["git", "status", "--porcelain"], cwd=inputs.repo_root,
                               capture_output=True, text=True, check=False).stdout
    dirty_patch_sha = hashlib.sha256(worktree.encode("utf-8")).hexdigest() if worktree else None
    signal_quality = input_quality_manifest(inputs.signal_panel, date_column_candidates=("decision_date", "date"),
                                            key_columns=("decision_date", "ticker"),
                                            development_end=inputs.development_end, holdout_boundary=inputs.holdout_boundary)
    benchmark_quality = input_quality_manifest(inputs.benchmark_prices, date_column_candidates=("date", "session_date"),
                                               key_columns=("date", "session_date", "ticker"),
                                               development_end=inputs.development_end, holdout_boundary=inputs.holdout_boundary)
    contract = {
        "schema_version": MANIFESTED_JOB_SCHEMA,
        "git_sha": current_git_sha(inputs.repo_root),
        "source_tree_sha256": source_tree_sha256(inputs.repo_root),
        "signal_panel_development_sha256": signal_development_hash,
        "candidate_metrics_sha256": sha256_file(inputs.candidate_metrics) if inputs.candidate_metrics and inputs.candidate_metrics.is_file() else None,
        "input_stat_fingerprint": input_stat_fingerprint(inputs),
        "feature_schema_sha256": feature_hash,
        "benchmark_prices_development_sha256": benchmark_development_hash,
        "benchmark_distributions_development_sha256": benchmark_distribution_development_hash,
        "candidate_registry_sha256": registry["candidate_registry"]["candidate_registry_sha256"],
        "candidate_oos_evidence_sha256": None,
        "candidate_oos_coverage_manifest": {str(h): {"status": "GENERATABLE", "fold_count": len(folds_by_horizon.get(h, ())) } for h in range(1, 31)},
        "candidate_universe_hash": registry["candidate_registry"]["candidate_registry_sha256"],
        "fold_policy_hash": resolved.fold_policy.fold_policy_hash,
        "target_contract_hash": resolved.target_contract.target_contract_hash,
        "recipe_selection_policy": asdict(resolved.recipe_selection_policy),
        "recipe_selection_policy_hash": resolved.recipe_selection_policy.recipe_selection_policy_hash,
        "model_training_contract": asdict(resolved.model_training_contract),
        "model_training_contract_hash": resolved.model_training_contract.model_training_contract_hash,
        "random_seed": int(inputs.random_seed),
        "stock_execution_prices_development_sha256": stock_execution_development_hash,
        "stock_price_quality_manifest": stock_quality,
        "stock_direct_daily_manifest_sha256": sha256_file(inputs.stock_direct_daily_manifest) if inputs.stock_direct_daily_manifest and inputs.stock_direct_daily_manifest.is_file() else None,
        "stock_distributions_development_sha256": stock_dividend_status.get("development_content_hash"),
        "stock_inputs_policy": "IGNORED_NOT_VALID_FOR_PORTFOLIO_REPLAY" if inputs.allow_missing_stock_inputs else "REQUIRED",
        "training_window_sessions": inputs.training_window_sessions,
        "validation_window_sessions": inputs.validation_window_sessions,
        "calibration_window_sessions": inputs.calibration_window_sessions,
        "step_sessions": inputs.step_sessions, "purge_sessions": inputs.purge_sessions,
        "embargo_sessions": inputs.embargo_sessions,
        "refit_cadence": inputs.refit_cadence,
        "maturity_rule": inputs.maturity_rule,
        "score_quantile": inputs.score_quantile,
        "top_fraction": inputs.top_fraction,
        "cost_model": dict(inputs.cost_model), "tax_contract": dict(inputs.tax_contract),
        "holdout_boundary": str(inputs.holdout_boundary),
        "family_registry_hash": registry["family_registry_hash"],
        "development_start": str(inputs.development_start), "effective_development_start": str(effective_start),
        "development_end": str(inputs.development_end),
        "refit_count": len(refit_dates),
        "data_boundary_audit": data_boundary_audit,
        "input_quality": {"signal_panel": signal_quality, "benchmark_prices": benchmark_quality,
                          "stock_execution_prices": stock_quality, "stock_distributions": stock_dividend_status},
        "worktree_status": "DIRTY" if worktree else "CLEAN",
        "dirty_patch_sha256": dirty_patch_sha,
        "eligible_refit_dates_by_horizon": {str(h): [str(x) for x in dates] for h, dates in refit_dates_by_horizon.items()},
    }
    contract["run_contract_hash"] = stable_hash(contract)
    jobs = _job_graph(
        registry, refit_dates_by_horizon, folds_by_horizon,
        trading_sessions=sessions, fold_policy=resolved.fold_policy,
        target_contract=resolved.target_contract,
        recipe_selection_policy=resolved.recipe_selection_policy,
        model_training_contract=resolved.model_training_contract,
        development_end=inputs.development_end)
    if existing_contract.is_file():
        prior = json.loads(existing_contract.read_text(encoding="utf-8"))
        exact_contract = (
            prior.get("run_contract_hash") == contract["run_contract_hash"])
        semantic_compatible = (
            _compatible_resume_contract_view(prior)
            == _compatible_resume_contract_view(contract))
        if not exact_contract and not (
            allow_compatible_code_resume and semantic_compatible
        ):
            # Preserve an evidence trail for a blocked resume.  The old
            # error only named the remedy, which made a supervised run spend
            # its restart budget without exposing which scientific field
            # actually diverged.
            mismatch = {
                key: {
                    "prior": prior.get(key),
                    "current": contract.get(key),
                }
                for key in sorted(set(prior) | set(contract))
                if _compatible_resume_contract_view(prior).get(key)
                != _compatible_resume_contract_view(contract).get(key)
            }
            (output / "run-state" / "resume-contract-mismatch.json").write_text(
                json.dumps({
                    "status": "BLOCKED_UNSAFE_RESUME",
                    "exact_contract": exact_contract,
                    "allow_compatible_code_resume": allow_compatible_code_resume,
                    "semantic_compatible": semantic_compatible,
                    "mismatch": mismatch,
                }, indent=2, sort_keys=True, default=str),
                encoding="utf-8")
            raise RuntimeError(
                "MANIFESTED_JOB_EXISTING_STATE_CONTRACT_MISMATCH_USE_NEW_ARTIFACT_ROOT")
        reconciliation = store.seed_jobs(
            jobs, invalidate_descendants=True)
        store.write("manifested-job-contract.json", contract)
        progress = store.progress()
        progress.update({
            "status": (
                "RESUMED_RECONCILED_COMPATIBLE_CODE"
                if not exact_contract else "RESUMED_RECONCILED"),
            "run_contract_hash": contract["run_contract_hash"],
            "prior_run_contract_hash": prior.get("run_contract_hash"),
            "prior_git_sha": prior.get("git_sha"),
            "resume_git_sha": contract.get("git_sha"),
            "graph_reconciliation": reconciliation,
        })
        store.write("run-state/progress.json", progress)
        return {
            **(json.loads(
                (output / "summary.json").read_text(encoding="utf-8"))
               if (output / "summary.json").is_file() else {}),
            "status": "RESUMED_RECONCILED",
            "resumed": True,
            "progress": progress,
        }
    store.seed_jobs(jobs)
    store.write("manifested-job-contract.json", contract)
    if worktree:
        dirty_diff = subprocess.run(["git", "diff", "HEAD"], cwd=inputs.repo_root,
                                    capture_output=True, text=True, check=False).stdout
        (output / "provenance").mkdir(parents=True, exist_ok=True)
        (output / "provenance" / "dirty.patch").write_text(dirty_diff, encoding="utf-8")
    store.write("family-registry.json", registry)
    store.write("generation-registry.json", {"schema_version": "DYNAMIC_QBD_GENERATION_REGISTRY_V2_MANIFEST_REQUIRED",
                                              "family_registry_hash": registry["family_registry_hash"], "generations": []})
    store.write("run-state/progress.json", {"status": "INITIALIZED",
                                             "job_count": len(jobs), "complete": 0, "failed": 0,
                                             "candidate_oos_audit": candidate_audit})
    store.write("research/family-incubator/registry.json", {"schema_version": "DYNAMIC_QBD_FAMILY_INCUBATOR_V1", "families": []})
    blocking_reasons = []
    if data_boundary_audit["status"] != "PASS":
        blocking_reasons.append("DATA_BOUNDARY")
    # Candidate-OOS is a generated dependency.  Missing precomputed evidence
    # schedules fold jobs; it is not a global run blocker.
    if stock_quality["status"] != "PASS" and not inputs.allow_missing_stock_inputs:
        blocking_reasons.append("STOCK_EXECUTION_PRICES")
    if stock_dividend_status["status"] != "PASS" and not inputs.allow_missing_stock_inputs:
        blocking_reasons.append("STOCK_DISTRIBUTIONS")
    if worktree:
        blocking_reasons.append("DIRTY_WORKTREE")
    summary = {"status": "BLOCKED_" + "_AND_".join(blocking_reasons) if blocking_reasons else "INITIALIZED",
               "authority": "DEVELOPMENT_ONLY_NO_PROMOTION_NO_HOLDOUT" if not inputs.allow_missing_stock_inputs else "DEVELOPMENT_ONLY_NO_PORTFOLIO_VALIDITY_NO_PROMOTION_NO_HOLDOUT",
               "run_contract_hash": contract["run_contract_hash"],
               "family_registry_hash": registry["family_registry_hash"],
               "model_family_count": registry["model_family_count"],
               "portfolio_family_count": registry["portfolio_family_count"],
               "refit_count": len(refit_dates), "eligible_refit_dates_by_horizon": {str(h): [str(x) for x in v] for h, v in refit_dates_by_horizon.items()}, "job_count": len(jobs),
        "candidate_oos_audit": candidate_audit | {"status": "GENERATABLE_DEPENDENCY"} if candidate_audit["status"] != "COMPLETE" else candidate_audit,
               "data_boundary_audit": data_boundary_audit,
               "input_quality": {"signal_panel": signal_quality, "benchmark_prices": benchmark_quality,
                                 "stock_execution_prices": stock_quality, "stock_distributions": stock_dividend_status},
               "deduplication": {"model_fit_dimensions": ["horizon", "recipe", "refit_date", "feature_schema", "training_window", "calibration_window", "seed"],
                                  "portfolio_dimensions": ["horizon", "holding_days", "max_names", "exit_policy"]},
               "final_holdout_opened": False}
    store.write("summary.json", summary)
    return summary


def _month_end_dates(sessions: Iterable[date]) -> tuple[date, ...]:
    values = pd.Series(pd.to_datetime(tuple(sessions))).drop_duplicates().sort_values()
    return tuple(pd.Timestamp(group.iloc[-1]).date() for _, group in values.groupby(values.dt.to_period("M")))


def _eligible_refit_dates(sessions: tuple[date, ...], horizon: int, fold_policy: FoldPolicy) -> tuple[date, ...]:
    minimum_index = (fold_policy.training_window_sessions + fold_policy.purge_sessions +
                     fold_policy.calibration_window_sessions + int(horizon) - 1)
    if minimum_index >= len(sessions):
        return ()
    eligible = set(sessions[minimum_index:])
    return tuple(x for x in _month_end_dates(sessions) if x in eligible)


def _reuse_peer_candidate_partition(
    *, output_root: Path, job: Mapping[str, Any],
    factory: CandidateOosFactory, expected_execution_backend: str,
) -> dict | None:
    """Publish an identical peer fit only when this seed causally claims it."""
    peer_names = ("short", "primary", "long")
    current = output_root.resolve()
    for name in peer_names:
        peer_root = output_root.parent / name
        if not peer_root.is_dir() or peer_root.resolve() == current:
            continue
        peer_store = ManifestedJobStore(peer_root)
        peer_job = peer_store.job(str(job["job_id"]))
        if not peer_job or peer_job.get("state") != "COMPLETE":
            continue
        result = dict(peer_job.get("result") or {})
        source_path = Path(str(result.get("path", "")))
        source_candidate_root = peer_root / "candidate-oos"
        try:
            relative = source_path.resolve().relative_to(source_candidate_root.resolve())
        except (OSError, ValueError):
            continue
        source_manifest = CandidateOosStore(source_candidate_root).verify_partition(source_path)
        candidate = factory.candidates[str(job["candidate_id"])]
        expected = {
            "horizon": int(job["horizon"]),
            "fold_id": str(job["fold_id"]),
            "candidate_id": str(job["candidate_id"]),
            "signal_panel_sha256": str(factory.signal_panel_sha256),
            "feature_schema_sha256": str(factory.feature_schema_sha256),
            "fold_policy_hash": str(factory.fold_policy.fold_policy_hash),
            "target_contract_hash": str(factory.target_contract.target_contract_hash),
            "model_training_contract_hash": str(factory.model_training_contract_hash),
            "execution_backend": str(expected_execution_backend),
        }
        if any(str(source_manifest.get(key)) != str(value)
               for key, value in expected.items()):
            continue
        target_path = output_root / "candidate-oos" / relative
        target_dir = target_path.parent
        if not target_path.is_file():
            temporary = target_dir.parent / f".{target_dir.name}.peer.{os.getpid()}.tmp"
            if temporary.exists():
                shutil.rmtree(temporary)
            temporary.mkdir(parents=True, exist_ok=False)
            try:
                for source in source_path.parent.iterdir():
                    if not source.is_file():
                        continue
                    destination = temporary / source.name
                    try:
                        os.link(source, destination)
                    except OSError:
                        shutil.copy2(source, destination)
                target_dir.parent.mkdir(parents=True, exist_ok=True)
                try:
                    temporary.replace(target_dir)
                except FileExistsError:
                    shutil.rmtree(temporary, ignore_errors=True)
            except Exception:
                shutil.rmtree(temporary, ignore_errors=True)
                raise
        local_manifest = CandidateOosStore(output_root / "candidate-oos").verify_partition(target_path)
        if local_manifest.get("manifest_sha256") != source_manifest.get("manifest_sha256"):
            raise RuntimeError("CANDIDATE_OOS_CROSS_SEED_MANIFEST_MISMATCH")
        reused = dict(local_manifest)
        reused.update({"cache": "CROSS_SEED_HIT", "path": str(target_path),
                       "source_seed": name.upper()})
        model_path = target_path.with_name("model.pkl")
        if model_path.is_file():
            reused["model_path"] = str(model_path)
        return reused
    return None


def _candidate_oos_process_job(payload: dict) -> Any:
    """Execute one Candidate-OOS node and report worker-local runtime/RSS."""
    job_class = str(payload.get("_ram_job_class", "candidate_oos:CPU:CPU"))
    execution_started = time.monotonic()
    with _JobMemorySampler(
            interval_seconds=.01,
            live_config=payload.get("_live_ram")) as sampler:
        result = _candidate_oos_process_job_inner(payload)
    execution_seconds = max(.001, time.monotonic() - execution_started)
    if isinstance(result, dict):
        result = dict(result)
        result["_runtime_memory"] = sampler.telemetry(job_class)
        result["_runtime_execution_seconds"] = execution_seconds
    return result


def _candidate_oos_process_job_inner(payload: dict) -> Any:
    """Execute one Candidate-OOS node inside an isolated pinned process."""
    global _CANDIDATE_WORKER_FACTORY, _CANDIDATE_WORKER_FACTORY_KEY
    inputs: ManifestedJobInputs = payload["inputs"]
    output_root = Path(payload["output_root"])
    if (_CANDIDATE_WORKER_FACTORY is None or _CANDIDATE_WORKER_FACTORY_KEY is None or
            _CANDIDATE_WORKER_FACTORY_KEY[0] != str(output_root)):
        store = ManifestedJobStore(output_root)
        contract = store.read("manifested-job-contract.json")
        registry = store.read("family-registry.json")
        key = (str(output_root), str(registry["candidate_registry"]["candidate_registry_sha256"]))
        hyperparameter_path = inputs.hyperparameter_space or (inputs.repo_root / "stock_predictor" / "v5" / "hyperparameter_space.json")
        candidates = load_primary_candidate_registry(hyperparameter_path)
        feature_schema = json.loads(inputs.feature_schema.read_text(encoding="utf-8"))
        resolved = inputs.resolved_contracts(
            primary_candidate_universe_hash=registry["candidate_registry"]["candidate_registry_sha256"])
        _CANDIDATE_WORKER_FACTORY = CandidateOosFactory(
            signal_panel=inputs.signal_panel, feature_schema=feature_schema, candidates=candidates,
            output_root=output_root / "candidate-oos",
            signal_panel_sha256=contract["signal_panel_development_sha256"],
            fold_policy=resolved.fold_policy, target_contract=resolved.target_contract,
            development_end=inputs.development_end, holdout_boundary=inputs.holdout_boundary,
            random_seed=resolved.model_training_contract.random_seed,
            model_training_contract_hash=resolved.model_training_contract.model_training_contract_hash,
            prepared_cache_root=(
                output_root.parent / "_shared-compute" / "prepared-folds"))
        _CANDIDATE_WORKER_FACTORY_KEY = key
    factory = _CANDIDATE_WORKER_FACTORY
    job = payload["job"]
    horizon = int(job["horizon"])
    fold_payload = payload.get("fold")
    if isinstance(fold_payload, Fold):
        fold = fold_payload
    elif isinstance(fold_payload, Mapping):
        fold = Fold(
            fold_id=str(fold_payload["fold_id"]),
            train_dates=tuple(str(x) for x in fold_payload["train_dates"]),
            validation_dates=tuple(
                str(x) for x in fold_payload["validation_dates"]),
        )
    else:
        folds = {x.fold_id: x for x in factory.fold_specs(
            horizon=horizon, development_end=inputs.development_end,
            holdout_boundary=inputs.holdout_boundary)}
        fold = folds[str(job["fold_id"])]
    if bool(payload.get("_prepare_only")):
        return factory.prepare_fold(
            horizon=horizon, fold=fold,
            development_end=inputs.development_end,
            holdout_boundary=inputs.holdout_boundary)
    candidate = factory.candidates[str(job["candidate_id"])]
    execution_backend = _execution_backend_for_family(
        candidate.recipe_family, payload.get("execution_backend_override"))
    execution_backend_fingerprint = str(
        payload.get("execution_backend_fingerprint") or
        _gpu_execution_fingerprint(
            "HGB" if candidate.recipe_family == "HIST_GRADIENT_BOOSTING"
            else ("RIDGE" if candidate.recipe_family == "RIDGE_LOGISTIC" else "CPU"),
            payload.get("gpu_device_index"),
        )
    )
    fit_shared_id = canonical_sha256({
        "signal_panel_sha256": factory.signal_panel_sha256,
        "feature_schema_sha256": factory.feature_schema_sha256,
        "fold_policy_hash": factory.fold_policy.fold_policy_hash,
        "target_contract_hash": factory.target_contract.target_contract_hash,
        "model_training_contract_hash": factory.model_training_contract_hash,
        "horizon": horizon,
        "fold_id": fold.fold_id,
    })[:32]
    cache_key = canonical_sha256({
        "signal_panel_sha256": factory.signal_panel_sha256,
        "feature_schema_sha256": factory.feature_schema_sha256,
        "candidate_spec_hash": candidate.candidate_spec_hash,
        "fold_policy_hash": factory.fold_policy.fold_policy_hash,
        "target_contract_hash": factory.target_contract.target_contract_hash,
        "model_training_contract_hash": factory.model_training_contract_hash,
        "execution_backend": execution_backend,
        "execution_backend_fingerprint": execution_backend_fingerprint,
        "horizon": horizon, "fold_id": fold.fold_id,
    })[:32]
    identity = {
        "horizon": horizon, "fold_id": fold.fold_id,
        "candidate_id": candidate.candidate_id, "cache_key": cache_key,
        "execution_backend": execution_backend,
        "execution_backend_fingerprint": execution_backend_fingerprint,
    }
    shared_root = output_root.parent / "_shared-compute"
    shared_store_contract(shared_root)
    shared_store = CandidateOosStore(shared_root / "candidate-oos")
    shared_path = shared_store.partition_path(
        horizon, fold.fold_id, candidate.candidate_id, cache_key)
    local_path = factory.store.partition_path(
        horizon, fold.fold_id, candidate.candidate_id, cache_key)
    prior_gpu_slot = os.environ.get("DQBD_GPU_WORKER_SLOT")
    prior_backend = os.environ.get("DQBD_FORCE_EXECUTION_BACKEND")
    prior_fingerprint = os.environ.get("DQBD_EXECUTION_BACKEND_FINGERPRINT")
    prior_fit_shared_id = os.environ.get("DQBD_FIT_SHARED_ID")
    prior_gpu_job_id = os.environ.get("DQBD_GPU_JOB_ID")
    requested_gpu_slot = payload.get("gpu_device_index")
    if requested_gpu_slot is not None:
        # CPU affinity remains tied to the manifested worker slot.  This
        # separate device slot lets the parent scheduler assign one heavy fit
        # to each physical GPU, independent of the pool process that claims
        # the job.
        os.environ["DQBD_GPU_WORKER_SLOT"] = str(int(requested_gpu_slot))
    backend_override = payload.get("execution_backend_override")
    if backend_override:
        os.environ["DQBD_FORCE_EXECUTION_BACKEND"] = str(backend_override)
    os.environ["DQBD_EXECUTION_BACKEND_FINGERPRINT"] = execution_backend_fingerprint
    os.environ["DQBD_FIT_SHARED_ID"] = fit_shared_id
    os.environ["DQBD_GPU_JOB_ID"] = str(job["job_id"])

    def materialize_verified_shared_result(shared_key: str) -> dict[str, Any]:
        """Reuse an atomically published artifact without taking its lock.

        The lock protects the missing-artifact build.  Once the immutable
        observations file exists, taking that lock again is unnecessary and
        can deadlock a resume when a previous owner disappeared after
        publication but before process cleanup.
        """
        shared_manifest = shared_store.verify_partition(shared_path)
        materialize_immutable_tree(shared_path.parent, local_path.parent)
        local_manifest = factory.store.verify_partition(local_path)
        if local_manifest.get("manifest_sha256") != shared_manifest.get(
                "manifest_sha256"):
            raise RuntimeError("CANDIDATE_OOS_SHARED_VIEW_MANIFEST_MISMATCH")
        result = dict(local_manifest)
        result.update({"cache": "SHARED_COMPUTE_HIT", "path": str(local_path),
                       "shared_compute_key": shared_key,
                       "shared_artifact_path": str(shared_path)})
        model_path = local_path.with_name("model.pkl")
        if model_path.is_file():
            result["model_path"] = str(model_path)
        return result

    def reuse_persisted_job_result() -> dict[str, Any] | None:
        """Reuse a valid result recorded on a prior interrupted attempt.

        Resume may legitimately recompute the execution cache key after a
        scheduler-only contract change.  The manifested result itself is the
        stronger authority: when its immutable partition and execution
        identity still validate, copy/link it into the new shared-key view
        instead of refitting the same Candidate×Horizon×Fold node.
        """
        prior = job.get("result")
        if not isinstance(prior, Mapping):
            return None
        source_value = prior.get("path")
        if not source_value:
            return None
        source_path = Path(str(source_value))
        try:
            source_path = source_path.resolve()
            source_path.relative_to(output_root.resolve() / "candidate-oos")
        except (OSError, ValueError):
            return None
        if not source_path.is_file():
            return None
        try:
            source_manifest = factory.store.verify_partition(source_path)
        except (OSError, ValueError, RuntimeError):
            return None
        expected_backend = f"{execution_backend}:{execution_backend_fingerprint}"
        expected = {
            "horizon": int(job["horizon"]),
            "fold_id": str(job["fold_id"]),
            "candidate_id": str(job["candidate_id"]),
            "signal_panel_sha256": str(factory.signal_panel_sha256),
            "feature_schema_sha256": str(factory.feature_schema_sha256),
            "fold_policy_hash": str(factory.fold_policy.fold_policy_hash),
            "target_contract_hash": str(factory.target_contract.target_contract_hash),
            "model_training_contract_hash": str(factory.model_training_contract_hash),
            "execution_backend": expected_backend,
        }
        if any(str(source_manifest.get(key)) != str(value)
               for key, value in expected.items()):
            return None
        if not shared_path.is_file():
            materialize_immutable_tree(source_path.parent, shared_path.parent)
        return materialize_verified_shared_result(
            compute_identity_key("candidate-oos", identity))

    try:
        # A completed immutable artifact is already a safe publication point.
        # Do this check before the lock so a resume cannot wait behind an
        # orphaned lock for work that does not need to be recomputed.
        shared_key = compute_identity_key("candidate-oos", identity)
        if shared_path.is_file():
            return materialize_verified_shared_result(shared_key)
        persisted = reuse_persisted_job_result()
        if persisted is not None:
            return persisted
        with compute_key_lock(shared_root, "candidate-oos", identity) as locked_key:
            if not shared_path.is_file():
                reused = _reuse_peer_candidate_partition(
                    output_root=output_root, job=job, factory=factory,
                    expected_execution_backend=(
                        f"{execution_backend}:{execution_backend_fingerprint}"))
                if reused is not None:
                    materialize_immutable_tree(Path(reused["path"]).parent, shared_path.parent)
                else:
                    built = factory.build_fold(
                        horizon=horizon, candidate_id=candidate.candidate_id, fold=fold,
                        development_end=inputs.development_end,
                        holdout_boundary=inputs.holdout_boundary)
                    materialize_immutable_tree(Path(built["path"]).parent, shared_path.parent)
            return materialize_verified_shared_result(locked_key)
    finally:
        if prior_gpu_slot is None:
            os.environ.pop("DQBD_GPU_WORKER_SLOT", None)
        else:
            os.environ["DQBD_GPU_WORKER_SLOT"] = prior_gpu_slot
        if prior_backend is None:
            os.environ.pop("DQBD_FORCE_EXECUTION_BACKEND", None)
        else:
            os.environ["DQBD_FORCE_EXECUTION_BACKEND"] = prior_backend
        if prior_fingerprint is None:
            os.environ.pop("DQBD_EXECUTION_BACKEND_FINGERPRINT", None)
        else:
            os.environ["DQBD_EXECUTION_BACKEND_FINGERPRINT"] = prior_fingerprint
        if prior_fit_shared_id is None:
            os.environ.pop("DQBD_FIT_SHARED_ID", None)
        else:
            os.environ["DQBD_FIT_SHARED_ID"] = prior_fit_shared_id
        if prior_gpu_job_id is None:
            os.environ.pop("DQBD_GPU_JOB_ID", None)
        else:
            os.environ["DQBD_GPU_JOB_ID"] = prior_gpu_job_id


def _causal_process_job(payload: dict) -> Any:
    """Run one causal node and report worker-local runtime/RSS."""
    job_class = str(payload.get("_ram_job_class", "causal:CPU:CPU"))
    execution_started = time.monotonic()
    with _JobMemorySampler(
            interval_seconds=.01,
            live_config=payload.get("_live_ram")) as sampler:
        result = _causal_process_job_inner(payload)
    execution_seconds = max(.001, time.monotonic() - execution_started)
    if isinstance(result, dict):
        result = dict(result)
        result["_runtime_memory"] = sampler.telemetry(job_class)
        result["_runtime_execution_seconds"] = execution_seconds
    return result


def _causal_process_job_inner(payload: dict) -> Any:
    """Rebuild the handler context in a worker; no mutable handler is shared."""
    global _CAUSAL_WORKER_HANDLERS, _CAUSAL_WORKER_HANDLERS_KEY
    output_root = Path(payload["output_root"])
    key = str(output_root)
    if _CAUSAL_WORKER_HANDLERS is None or _CAUSAL_WORKER_HANDLERS_KEY != key:
        store = ManifestedJobStore(output_root)
        _CAUSAL_WORKER_HANDLERS = build_manifested_job_handlers(
            store=store, inputs=payload["inputs"], output_root=output_root,
            evidence_snapshot_root=payload.get("evidence_snapshot_root"),
            recipe_selection_root=payload.get("recipe_selection_root"))
        _CAUSAL_WORKER_HANDLERS_KEY = key
    prior_gpu_slot = os.environ.get("DQBD_GPU_WORKER_SLOT")
    prior_backend = os.environ.get("DQBD_FORCE_EXECUTION_BACKEND")
    prior_fingerprint = os.environ.get("DQBD_EXECUTION_BACKEND_FINGERPRINT")
    prior_gpu_job_id = os.environ.get("DQBD_GPU_JOB_ID")
    if payload.get("gpu_device_index") is not None:
        os.environ["DQBD_GPU_WORKER_SLOT"] = str(int(payload["gpu_device_index"]))
    if payload.get("execution_backend_override"):
        os.environ["DQBD_FORCE_EXECUTION_BACKEND"] = str(payload["execution_backend_override"])
    if payload.get("execution_backend_fingerprint"):
        os.environ["DQBD_EXECUTION_BACKEND_FINGERPRINT"] = str(
            payload["execution_backend_fingerprint"])
    os.environ["DQBD_GPU_JOB_ID"] = str(payload["job"]["job_id"])
    try:
        return _CAUSAL_WORKER_HANDLERS[payload["job"]["kind"]](payload["job"])
    finally:
        if prior_gpu_slot is None:
            os.environ.pop("DQBD_GPU_WORKER_SLOT", None)
        else:
            os.environ["DQBD_GPU_WORKER_SLOT"] = prior_gpu_slot
        if prior_backend is None:
            os.environ.pop("DQBD_FORCE_EXECUTION_BACKEND", None)
        else:
            os.environ["DQBD_FORCE_EXECUTION_BACKEND"] = prior_backend
        if prior_fingerprint is None:
            os.environ.pop("DQBD_EXECUTION_BACKEND_FINGERPRINT", None)
        else:
            os.environ["DQBD_EXECUTION_BACKEND_FINGERPRINT"] = prior_fingerprint
        if prior_gpu_job_id is None:
            os.environ.pop("DQBD_GPU_JOB_ID", None)
        else:
            os.environ["DQBD_GPU_JOB_ID"] = prior_gpu_job_id


def _causal_ready_kind_priority(kind: str) -> int:
    """Rank ready nodes by how much causal work they unlock.

    Coverage is deliberately a large backlog. If it wins the CPU frontier
    merely because its job id sorts first, ready recipe-selection nodes never
    finish, model nodes never become visible, and the GPU queues starve.
    This is a scheduling order only; dependency readiness remains authoritative
    and no future information is introduced.
    """
    return {
        "recipe_selection": -2,
        "model": -1,
        "calibration": 0,
        "prediction": 1,
        "generation_ready": 2,
        "replay": 3,
        "evidence": 4,
        "candidate_evidence_coverage": 5,
    }.get(str(kind), 6)


def execute_ready_jobs(store: ManifestedJobStore, handlers: Mapping[str, Callable[[dict], Any]] | None = None,
                       *, owner: str = "local-worker", max_jobs: int | None = None,
                       kinds: Iterable[str] | None = None, workers: int = 1,
                       queue_ahead: int = 2, inputs: ManifestedJobInputs | None = None,
                       max_inflight: int | None = None,
                       worker_map_override: list[dict] | None = None,
                       ram_scheduler: RamAdmissionScheduler | None = None,
                       estimated_task_gib: float = 4.0,
                       evidence_snapshot_root: str | Path | None = None,
                       recipe_selection_root: str | Path | None = None,
                       activity_heartbeat_path: str | Path | None = None,
                       hot_reload_controller: HotReloadController | None = None) -> dict:
    """Run dependency-ready jobs with resumable SQLite claiming.

    Handlers are intentionally injected: the coordinator owns causal ordering,
    while production model/replay implementations can be supplied without
    weakening the state contract.  A missing handler fails the node explicitly.
    """
    handlers = dict(handlers or {})
    processed = 0
    kinds_tuple = tuple(str(value) for value in kinds) if kinds is not None else None
    workers = max(1, int(workers))
    if workers > 1 and inputs is None:
        raise ValueError("MANIFESTED_PROCESS_EXECUTION_INPUTS_REQUIRED")
    active_workers = min(
        MAX_RUNTIME_PROCESS_LANES,
        max(1, min(workers, int(max_inflight or workers))),
    )
    # No ProcessPool future exists beyond the active worker frontier. Parent-
    # side manifested jobs are the queue-ahead mechanism. CPU futures own RAM
    # leases; GPU-capable HGB/Ridge futures are routed independently and own no
    # RamAdmissionScheduler lease.
    pool_workers = active_workers
    queue_capacity = active_workers
    gpu_device_count = len(opencl_fp64_devices())

    def workload_for_job(job: Mapping[str, Any]) -> str:
        """Classify only production model fits as GPU-capable.

        All other causal nodes remain CPU work.  A model fit gets a GPU only
        when a device is free at submission time; otherwise it is submitted
        immediately with the canonical CPU backend.  This keeps CPU lanes
        productive and makes GPU execution opportunistic rather than a gate.
        """
        if str(job.get("kind")) != "model":
            return "CPU"
        for dependency in job.get("depends_on", ()):
            parent = store.job(str(dependency))
            if not parent or parent.get("kind") != "recipe_selection":
                continue
            family = str((parent.get("result") or {}).get("recipe_family", ""))
            if family == "HIST_GRADIENT_BOOSTING":
                return "HGB"
            if family == "RIDGE_LOGISTIC":
                return "RIDGE"
        return "CPU"

    gpu_submitted = 0
    cpu_submitted = 0
    gpu_queue_timeouts = 0
    gpu_device_assignments = [0] * gpu_device_count
    activity_path = (
        Path(activity_heartbeat_path)
        if activity_heartbeat_path is not None else None)
    last_activity_touch = 0.0

    def touch_activity(*, state: str = "ACTIVE", force: bool = False) -> None:
        nonlocal last_activity_touch
        if activity_path is None:
            return
        now = time.time()
        if not force and now - last_activity_touch < 5.0:
            return
        _write_json(activity_path, {
            "schema_version": "DQBD_SEED_BANK_ACTIVITY_V1",
            "state": str(state), "owner": str(owner),
            "pid": os.getpid(), "updated_at_epoch": now,
        })
        last_activity_touch = now

    def ready_frontier() -> list[dict]:
        """Preview every job kind without letting one class starve others.

        The large DAG has many ready coverage nodes.  A single SQL preview
        ordered by ``kind`` can therefore fill the bounded preview entirely
        with CPU coverage work and hide already-causal recipe/model nodes,
        leaving the GPU queues empty.  Keep each per-kind preview bounded and
        merge only that scheduler frontier; dependency readiness remains
        authoritative in the store.
        """
        if kinds_tuple is None:
            return store.ready_jobs(limit=max(64, queue_capacity * 8))
        by_id: dict[str, dict] = {}
        per_kind_limit = max(8, queue_capacity * 2)
        for kind in kinds_tuple:
            for job in store.ready_jobs(
                    kinds=(kind,), limit=per_kind_limit):
                by_id[str(job["job_id"])] = job
        return list(by_id.values())

    def finalize_result(job_id: str, result: Any) -> Any:
        """Apply scheduler-state repairs only in the parent coordinator."""
        if not isinstance(result, dict):
            return result
        finalized = dict(result)
        if finalized.pop("_resume_invalidate_descendants", False):
            finalized["resume_descendants_requeued"] = (
                store.requeue_descendants((str(job_id),)))
        return finalized

    def execute_one(job):
        try:
            handler = handlers.get(job["kind"])
            if handler is None:
                raise RuntimeError(f"MANIFESTED_JOB_HANDLER_MISSING:{job['kind']}")
            result = finalize_result(job["job_id"], handler(job))
            store.finish_job(job["job_id"], owner, "COMPLETE", result=result)
        except Exception as exc:  # explicit failed node; future resume can retry by policy
            try:
                store.finish_job(job["job_id"], owner, "FAILED", last_error=f"{type(exc).__name__}:{exc}")
            except RuntimeError:
                pass
            raise

    if workers == 1:
        touch_activity(force=True)
        while max_jobs is None or processed < max_jobs:
            if hot_reload_controller is not None and hot_reload_controller.changed():
                raise _HotReloadRequested()
            # A serial model fit cannot heartbeat from this controlling thread.
            # Give it a full-day lease; process-pool jobs below retain the short
            # lease and renew it while their futures are alive.
            job = store.claim_ready(owner, lease_seconds=24 * 60 * 60, kinds=kinds)
            if job is None:
                break
            execute_one(job)
            processed += 1
    else:
        # Causal GPU-capable model fits use one staging lane per physical GPU.
        # The CPU budget remains independently capped at 32 Python lanes.
        desired_gpu_workers_per_device = 1
        gpu_pool_count = min(
            gpu_device_count,
            max(
                0,
                (pool_workers - 1)
                // desired_gpu_workers_per_device))
        gpu_workers_per_device = (
            desired_gpu_workers_per_device
            if gpu_pool_count else 0)
        gpu_worker_total = (
            gpu_pool_count * gpu_workers_per_device)
        cpu_workers = pool_workers
        base_worker_map = list(
            worker_map_override
            or active_cpu_contract().get("worker_map")
            or [])
        if not base_worker_map:
            base_worker_map = [
                {
                    "logical_processor": index,
                    "core_index": index,
                    "role": "fallback",
                }
                for index in range(pool_workers)
            ]
        cpu_map = base_worker_map[:cpu_workers]
        gpu_maps: list[list[dict]] = []
        affinity_source = list(active_cpu_contract().get("worker_map") or [])
        cpu_logical = {
            int(entry["logical_processor"])
            for entry in cpu_map
            if entry.get("logical_processor") is not None
        }
        gpu_candidates = [
            entry for entry in affinity_source
            if entry.get("logical_processor") not in cpu_logical
        ] or affinity_source or base_worker_map
        for device_index in range(gpu_pool_count):
            start = device_index * gpu_workers_per_device
            selected = gpu_candidates[start:start + gpu_workers_per_device]
            if len(selected) < gpu_workers_per_device:
                selected = [
                    gpu_candidates[index % len(gpu_candidates)]
                    for index in range(
                        start, start + gpu_workers_per_device)
                ]
            gpu_maps.append(selected)

        causal_gpu_devices = opencl_fp64_devices()[
            :gpu_pool_count]
        causal_shared_compute_root = (
            store.root.parent / "_shared-compute")
        gpu_monitor = GpuRuntimeMonitor(
            store.root / "telemetry",
            queue_root=causal_shared_compute_root)
        gpu_runtime_samples = 0
        last_gpu_monitor_at = 0.0
        healthy_devices = [True] * gpu_pool_count
        duration_profiles = _CandidateExecutionDurationProfiles(
            causal_shared_compute_root)
        gpu_queue_ahead = max(0, int(queue_ahead))
        gpu_queue_capacity = (
            gpu_workers_per_device + gpu_queue_ahead
            if gpu_pool_count else 0)
        gpu_disabled = [False] * gpu_pool_count
        gpu_device_assignments = [0] * gpu_pool_count

        def profile_workload(workload: str) -> str:
            return f"CAUSAL:{str(workload)}"

        with ExitStack() as pool_stack:
            cpu_pool, actual_cpu_workers = _manifested_pool(
                workers=cpu_workers,
                role="causal-cpu",
                telemetry_root=store.root / "telemetry",
                worker_map_override=cpu_map,
                max_tasks_per_child=_configured_lane_recycle_limit())
            cpu_pause = pool_stack.enter_context(
                _managed_pool_with_ram_pause(
                    cpu_pool, ram_scheduler,
                    store.root / "telemetry"))

            gpu_pools = []
            for device_index in range(gpu_pool_count):
                gpu_pool, _ = _manifested_pool(
                    workers=gpu_workers_per_device,
                    role=f"causal-gpu-{device_index}",
                    telemetry_root=store.root / "telemetry",
                    worker_map_override=gpu_maps[device_index],
                    max_pending=gpu_queue_capacity,
                    max_tasks_per_child=_configured_lane_recycle_limit())
                gpu_pools.append(gpu_pool)
                # GPU workers are not registered with RamAdmissionScheduler.
                pool_stack.enter_context(
                    _managed_pool_with_ram_pause(
                        gpu_pool, None,
                        store.root / "telemetry"))
            pool_stack.enter_context(
                _LaneHealthWatchdog(
                    store=store,
                    owner=owner,
                    pools=[cpu_pool, *gpu_pools],
                    telemetry_root=store.root / "telemetry",
                    activity_heartbeat_path=activity_path,
                ))

            actual_workers = (
                actual_cpu_workers
                + gpu_pool_count * gpu_workers_per_device)
            pending: dict[
                Any,
                tuple[
                    dict, str, int | None, str,
                    dict[str, Any]],
            ] = {}
            cpu_pending: set[Any] = set()
            gpu_pending: list[set[Any]] = [
                set() for _ in range(gpu_pool_count)]

            def descriptor_for(
                job: Mapping[str, Any],
                workload: str,
                device_index: int | None,
            ) -> tuple[str, dict[str, Any], str]:
                kind = str(job.get("kind", "CPU"))
                if kind == "model":
                    job_class = (
                        f"causal:model:{workload}:"
                        f"{'GPU' if device_index is not None else 'CPU'}")
                elif "replay" in kind:
                    job_class = "causal:replay:CPU"
                elif kind == "candidate_evidence_coverage":
                    job_class = "causal:coverage:CPU"
                elif "evidence" in kind:
                    job_class = "causal:evidence:CPU"
                else:
                    job_class = "causal:CPU:CPU"
                segment_sessions = 0
                if (
                    job.get("segment_start")
                    and job.get("segment_end")
                ):
                    try:
                        segment_sessions = max(
                            1,
                            (
                                date.fromisoformat(
                                    str(job["segment_end"]))
                                - date.fromisoformat(
                                    str(job["segment_start"]))
                            ).days)
                    except (TypeError, ValueError):
                        segment_sessions = 0
                descriptor = {
                    "job_id": str(job["job_id"]),
                    "kind": kind,
                    "horizon": (
                        int(job["horizon"])
                        if job.get("horizon") is not None
                        else None),
                    "workload": workload,
                    "prepared": True,
                    "device_index": device_index,
                    "segment_sessions": segment_sessions,
                    "ram_reclaim_count": int(
                        job.get("ram_reclaim_count", 0) or 0),
                }
                backend = CPU_EXECUTION_BACKEND
                if device_index is not None:
                    backend = (
                        GPU_HGB_EXECUTION_BACKEND
                        if workload == "HGB"
                        else GPU_EXECUTION_BACKEND)
                return job_class, descriptor, backend

            def remaining_seconds(
                future: Any,
            ) -> float:
                value = pending.get(future)
                if value is None:
                    return 0.0
                descriptor = value[4]
                predicted = float(
                    descriptor.get(
                        "predicted_duration_seconds", 0.0)
                    or 0.0)
                started = float(
                    descriptor.get(
                        "submitted_monotonic",
                        time.monotonic()))
                return max(
                    0.0,
                    predicted
                    - (time.monotonic() - started))

            def gpu_next_start(
                device_index: int,
            ) -> float:
                """Earliest virtual GPU-worker start including queued futures."""
                now = time.monotonic()
                intervals: list[tuple[float, float]] = []
                for future in tuple(gpu_pending[device_index]):
                    value = pending.get(future)
                    if value is None:
                        continue
                    descriptor = value[4]
                    start = float(descriptor.get(
                        "predicted_start_monotonic",
                        descriptor.get("submitted_monotonic", now)))
                    finish = float(descriptor.get(
                        "predicted_finish_monotonic", start))
                    if finish > now:
                        intervals.append((max(now, start), finish))
                candidate = now
                while True:
                    active_finishes = [
                        finish for start, finish in intervals
                        if start <= candidate < finish
                    ]
                    if len(active_finishes) < gpu_workers_per_device:
                        return candidate
                    candidate = min(active_finishes)

            def gpu_predicted_finish(
                workload: str, device_index: int,
            ) -> float:
                # Seed-local futures are only part of the physical-device
                # backlog. Other seed schedulers publish ready GPU sections
                # into the same shared queue, so use that global section
                # backlog as a lower bound for the next useful device start.
                now = time.monotonic()
                global_backlog_seconds = (
                    _gpu_global_backlog_seconds(
                        causal_shared_compute_root,
                        causal_gpu_devices,
                        device_index))
                predicted_start = max(
                    gpu_next_start(device_index),
                    now + global_backlog_seconds)
                return (
                    predicted_start
                    + duration_profiles.estimate(
                        profile_workload(workload),
                        "GPU",
                        causal_gpu_devices[device_index]))

            def best_gpu_predicted_finish(
                workload: str,
            ) -> float:
                values = [
                    gpu_predicted_finish(workload, index)
                    for index in range(gpu_pool_count)
                    if (
                        not gpu_disabled[index]
                        and gpu_workload_allowed(
                            causal_gpu_devices[index], workload)
                    )
                ]
                return (
                    min(values)
                    if values else float("inf"))

            def cpu_predicted_finish(
                workload: str,
            ) -> float:
                now = time.monotonic()
                active_finishes = [
                    now + remaining_seconds(future)
                    for future in tuple(cpu_pending)
                    if future in pending
                ]
                available = (
                    now
                    if len(cpu_pending) < cpu_workers
                    else min(active_finishes or [now]))
                return (
                    available
                    + duration_profiles.estimate(
                        profile_workload(workload),
                        "CPU"))

            def ready_gpu_frontier() -> list[dict]:
                ranked = []
                for job in ready_frontier():
                    workload = workload_for_job(job)
                    priority = _candidate_gpu_workload_priority(
                        workload, prepared=True)
                    if priority >= 99:
                        continue
                    preference = (
                        0 if str(job.get(
                            "execution_preference",
                            "")).upper() == "GPU"
                        else 1)
                    ranked.append(
                        (
                            priority,
                            preference,
                            str(job["job_id"]),
                            job,
                        ))
                ranked.sort(
                    key=lambda value: value[:3])
                return [value[3] for value in ranked]

            def cpu_route_allowed(
                job: Mapping[str, Any],
                workload: str,
            ) -> bool:
                if workload not in {"HGB", "RIDGE"}:
                    return True
                if str(job.get(
                    "execution_preference", "")).upper() == "CPU":
                    return True
                if (
                    gpu_pool_count
                    and not all(gpu_disabled)
                    and str(job.get(
                        "execution_preference",
                        "")).upper() == "GPU"
                ):
                    return False
                free_gpu = any(
                    not gpu_disabled[index]
                    and index < len(healthy_devices)
                    and healthy_devices[index]
                    and len(gpu_pending[index])
                    < gpu_queue_capacity
                    and gpu_workload_allowed(
                        causal_gpu_devices[index], workload)
                    for index in range(gpu_pool_count))
                if free_gpu:
                    return False
                if not gpu_pool_count or all(gpu_disabled):
                    return True
                return (
                    cpu_predicted_finish(workload)
                    <= best_gpu_predicted_finish(workload))

            def ready_cpu_frontier() -> list[dict]:
                ranked = []
                for job in ready_frontier():
                    workload = workload_for_job(job)
                    if not cpu_route_allowed(
                        job, workload
                    ):
                        continue
                    completion_priority = (
                        -1 if int(
                            job.get(
                                "ram_reclaim_count", 0)
                            or 0) > 0
                        else 0)
                    kind_priority = _causal_ready_kind_priority(
                        str(job.get("kind", "")))
                    workload_priority = (
                        1 if workload == "HGB"
                        else (
                            2 if workload == "RIDGE"
                            else 0))
                    ranked.append(
                        (
                            completion_priority,
                            kind_priority,
                            workload_priority,
                            str(job["job_id"]),
                            job,
                        ))
                ranked.sort(
                    key=lambda value: value[:3])
                # The preview job is stored at index 4. Index 3 is the
                # derived workload label and returning it makes the next
                # claim pass call ``job.get(...)`` on a string.
                return [value[4] for value in ranked]

            def claim_gpu(
                device_index: int,
            ) -> tuple[
                dict, str, str, dict[str, Any], str
            ] | None:
                for preview in ready_gpu_frontier():
                    workload = workload_for_job(preview)
                    if not gpu_workload_allowed(
                        causal_gpu_devices[device_index], workload):
                        continue
                    job_class, descriptor, backend = (
                        descriptor_for(
                            preview, workload,
                            device_index))
                    claimed = store.claim_ready(
                        owner,
                        lease_seconds=JOB_LEASE_SECONDS,
                        kinds=kinds,
                        job_ids=(str(preview["job_id"]),))
                    if claimed is None:
                        continue
                    return (
                        claimed, workload, job_class,
                        descriptor, backend)
                return None

            def claim_cpu() -> tuple[
                dict, str, str, dict[str, Any], str, float
            ] | None:
                previews = []
                for preview in ready_cpu_frontier():
                    workload = workload_for_job(preview)
                    job_class, descriptor, backend = (
                        descriptor_for(
                            preview, workload, None))
                    ram_score = (
                        ram_scheduler.priority_key(
                            job_class, descriptor,
                            gpu_idle=False)
                        if ram_scheduler is not None
                        else (0.0, 0.0, 0.0))
                    completion_priority = (
                        -1 if int(
                            preview.get(
                                "ram_reclaim_count", 0)
                            or 0) > 0
                        else 0)
                    kind_priority = _causal_ready_kind_priority(
                        str(preview.get("kind", "")))
                    previews.append(
                        (
                            (
                                completion_priority,
                                kind_priority,
                                *ram_score,
                            ),
                            str(preview["job_id"]),
                            preview,
                            workload,
                            job_class,
                            descriptor,
                            backend,
                        ))
                previews.sort(
                    key=lambda value: (
                        value[0], value[1]))
                for (
                    _, job_id, preview, workload,
                    job_class, descriptor, backend,
                ) in previews:
                    reservation = (
                        ram_scheduler.try_acquire(
                            estimated_task_gib,
                            wait_callback=(
                                cpu_pause.reconcile
                                if cpu_pause is not None
                                else None),
                            job_class=job_class,
                            descriptor=descriptor)
                        if ram_scheduler is not None
                        else 0.0)
                    if reservation is None:
                        continue
                    claimed = store.claim_ready(
                        owner,
                        lease_seconds=JOB_LEASE_SECONDS,
                        kinds=kinds,
                        job_ids=(job_id,))
                    if claimed is None:
                        if ram_scheduler is not None:
                            ram_scheduler.release(
                                reservation,
                                job_class=job_class)
                        continue
                    return (
                        claimed, workload, job_class,
                        descriptor, backend, reservation)
                return None

            def submit_job(
                pool,
                job: dict,
                workload: str,
                device_index: int | None,
                job_class: str,
                descriptor: Mapping[str, Any],
                backend: str,
                reservation: float,
            ) -> Any:
                descriptor = dict(descriptor)
                executor_name = (
                    "GPU"
                    if device_index is not None else "CPU")
                predicted_duration = (
                    duration_profiles.estimate(
                        profile_workload(workload),
                        executor_name,
                        None if device_index is None
                        else causal_gpu_devices[
                            device_index]))
                submitted_monotonic = time.monotonic()
                predicted_start = (
                    gpu_next_start(device_index)
                    if device_index is not None
                    else submitted_monotonic)
                descriptor.update({
                    "executor": executor_name,
                    "submitted_monotonic":
                        submitted_monotonic,
                    "predicted_start_monotonic":
                        predicted_start,
                    "predicted_duration_seconds":
                        predicted_duration,
                    "predicted_finish_monotonic":
                        predicted_start
                        + predicted_duration,
                })
                execution_fingerprint = (
                    _gpu_execution_fingerprint(
                        workload
                        if str(job.get("kind"))
                        == "model" else "CPU",
                        device_index))
                live_config = (
                    ram_scheduler.open_live_job(
                        job_class, descriptor)
                    if (
                        device_index is None
                        and ram_scheduler is not None)
                    else None)
                task_payload = {
                    "output_root": str(store.root),
                    "inputs": inputs,
                    "job": job,
                    "execution_backend_override":
                        backend,
                    "execution_backend_fingerprint":
                        execution_fingerprint,
                    "_ram_job_class": job_class,
                    "_live_ram": live_config,
                }
                if evidence_snapshot_root is not None:
                    task_payload[
                        "evidence_snapshot_root"] = str(
                            evidence_snapshot_root)
                if recipe_selection_root is not None:
                    task_payload[
                        "recipe_selection_root"] = str(
                            recipe_selection_root)
                if device_index is not None:
                    task_payload["gpu_device_index"] = (
                        device_index)
                try:
                    future = pool.submit(
                        _causal_process_job,
                        task_payload)
                except Exception:
                    if (
                        device_index is None
                        and ram_scheduler is not None
                    ):
                        ram_scheduler.close_live_job(
                            live_config)
                        ram_scheduler.release(
                            reservation,
                            job_class=job_class)
                    raise
                if (
                    device_index is None
                    and ram_scheduler is not None
                ):
                    def release_runtime(
                        _future, amount=reservation,
                        cls=job_class,
                        live=live_config,
                    ) -> None:
                        ram_scheduler.close_live_job(
                            live)
                        ram_scheduler.release(
                            amount, job_class=cls)
                    future.add_done_callback(
                        release_runtime)
                pending[future] = (
                    job, workload, device_index,
                    job_class, descriptor)
                if device_index is None:
                    cpu_pending.add(future)
                else:
                    gpu_pending[device_index].add(
                        future)
                return future

            while max_jobs is None or processed < max_jobs:
                if hot_reload_controller is not None and hot_reload_controller.changed():
                    raise _HotReloadRequested()
                now_gpu_monitor = time.monotonic()
                if now_gpu_monitor - last_gpu_monitor_at >= .25:
                    gpu_snapshot = gpu_monitor.snapshot(causal_gpu_devices)
                    gpu_runtime_samples += 1
                    healthy_devices = [
                        bool(row.get("assignment_allowed", False))
                        for row in gpu_snapshot.get("devices", ())]
                    last_gpu_monitor_at = now_gpu_monitor
                touch_activity()
                if cpu_pause is not None:
                    cpu_pause.reconcile()

                # GPU queues are filled independently of RAM admission.
                while True:
                    if (
                        max_jobs is not None
                        and processed + len(pending)
                        >= max_jobs
                    ):
                        break
                    gpu_ready = ready_gpu_frontier()
                    if not gpu_ready:
                        break
                    next_workload = workload_for_job(
                        gpu_ready[0])
                    free_devices = [
                        index
                        for index in range(gpu_pool_count)
                        if (
                            not gpu_disabled[index]
                            and index < len(healthy_devices)
                            and healthy_devices[index]
                            and len(gpu_pending[index])
                            < gpu_queue_capacity)
                    ]
                    if not free_devices:
                        break
                    predicted_finishes = {
                        index: gpu_predicted_finish(
                            next_workload, index)
                        for index in free_devices
                    }
                    device_index = _choose_gpu_feed_device(
                        free_devices,
                        pending_depths={
                            index: len(gpu_pending[index])
                            for index in free_devices
                        },
                        predicted_finishes=predicted_finishes,
                    )
                    if device_index is None:
                        break
                    selected = claim_gpu(
                        device_index)
                    if selected is None:
                        break
                    (
                        job, workload, job_class,
                        descriptor, backend,
                    ) = selected
                    try:
                        submit_job(
                            gpu_pools[device_index],
                            job, workload,
                            device_index,
                            job_class, descriptor,
                            backend, 0.0)
                    except Exception:
                        store.release_claim(
                            str(job["job_id"]), owner)
                        raise
                    gpu_device_assignments[
                        device_index] += 1
                    gpu_submitted += 1

                control_state = (
                    ram_scheduler.real_load_state()
                    if ram_scheduler is not None
                    else {"used_fraction": 0.0})
                used_fraction = float(
                    control_state.get(
                        "used_fraction", 0.0))
                admission_budget = (
                    ram_scheduler.admission_budget(
                        used_fraction)
                    if ram_scheduler is not None
                    else (
                        8 if used_fraction < .60
                        else (
                            4 if used_fraction < .80
                            else 2)))
                # The legacy pacing budget (8/4/2) is a refill rate, not a
                # reason to leave configured CPU lanes idle.  While the
                # system is below the 80% fill corridor and no reclaim is in
                # progress, top up to the complete CPU frontier.  Every
                # individual admission still passes the RAM scheduler, so a
                # rising real load can stop the fill immediately.
                if used_fraction < .80:
                    recovery = (
                        ram_scheduler.recovery_state()
                        if ram_scheduler is not None else {})
                    if not recovery.get("settling", False) and not recovery.get(
                            "recovering", False):
                        admission_budget = max(
                            admission_budget,
                            cpu_workers - len(cpu_pending))
                admitted_cpu = 0
                while (
                    admitted_cpu < admission_budget
                    and len(cpu_pending) < cpu_workers
                    and (
                        max_jobs is None
                        or processed + len(pending)
                        < max_jobs)
                ):
                    selected = claim_cpu()
                    if selected is None:
                        break
                    (
                        job, workload, job_class,
                        descriptor, backend, reservation,
                    ) = selected
                    try:
                        submit_job(
                            cpu_pool, job, workload,
                            None, job_class, descriptor,
                            backend, reservation)
                    except Exception:
                        store.release_claim(
                            str(job["job_id"]), owner)
                        raise
                    cpu_submitted += 1
                    admitted_cpu += 1

                if not pending:
                    if ready_frontier():
                        time.sleep(.01)
                        continue
                    break

                done, _ = wait(
                    tuple(pending), timeout=.01,
                    return_when=FIRST_COMPLETED)
                if cpu_pause is not None:
                    cpu_pause.reconcile()
                if (
                    "last_causal_heartbeat_at"
                    not in locals()
                ):
                    last_causal_heartbeat_at = (
                        time.monotonic())
                now_causal_heartbeat = time.monotonic()
                if (
                    now_causal_heartbeat
                    - last_causal_heartbeat_at >= 30.0
                ):
                    for (
                        future, pending_value
                    ) in tuple(pending.items()):
                        if future not in done:
                            store.heartbeat(
                                pending_value[0]["job_id"],
                                owner,
                                lease_seconds=
                                    JOB_LEASE_SECONDS)
                    last_causal_heartbeat_at = (
                        now_causal_heartbeat)

                for future in done:
                    (
                        job, workload,
                        gpu_device_index,
                        job_class, descriptor,
                    ) = pending[future]
                    if gpu_device_index is None:
                        cpu_pending.discard(future)
                    else:
                        gpu_pending[
                            gpu_device_index].discard(
                                future)
                    pending.pop(future, None)
                    job_id = str(job["job_id"])
                    try:
                        result = future.result()
                        wall_elapsed = max(
                            .001,
                            time.monotonic()
                            - float(descriptor.get(
                                "submitted_monotonic",
                                time.monotonic())))
                        execution_elapsed = (
                            float(result.get(
                                "_runtime_execution_seconds",
                                wall_elapsed))
                            if isinstance(result, dict)
                            else wall_elapsed)
                        duration_profiles.observe(
                            profile_workload(workload),
                            (
                                "GPU"
                                if gpu_device_index
                                is not None
                                else "CPU"),
                            execution_elapsed,
                            (
                                causal_gpu_devices[
                                    gpu_device_index]
                                if gpu_device_index
                                is not None else None))
                        if (
                            gpu_device_index is None
                            and ram_scheduler is not None
                            and isinstance(result, dict)
                        ):
                            memory = (
                                result.get(
                                    "_runtime_memory")
                                or {})
                            ram_scheduler.observe(
                                job_class,
                                memory.get(
                                    "rss_peak_gib"),
                                memory.get(
                                    "incremental_peak_gib"),
                                descriptor=descriptor)
                        result = finalize_result(job_id, result)
                        store.finish_job(
                            job_id, owner,
                            "COMPLETE", result=result)
                    except RamJobReclaimed as exc:
                        prefer_gpu = (
                            gpu_device_index is None
                            and workload
                            in {"HGB", "RIDGE"}
                            and str(job.get("kind"))
                            == "model")
                        store.requeue_reclaimed_job(
                            job_id, owner,
                            reason=exc.reason,
                            released_gib=exc.released_gib,
                            preferred_executor=(
                                "GPU"
                                if prefer_gpu else None))
                        continue
                    except Exception as exc:
                        if (
                            gpu_device_index is not None
                            and isinstance(exc, GPUSectionQueueTimeout)
                        ):
                            # Queue expiry is a bounded, transient wait. Do
                            # not poison the physical device for the rest of
                            # this coordinator invocation; retry this exact
                            # job on a CPU lane and let later GPU work flow.
                            gpu_queue_timeouts += 1
                            requeued = store.requeue_execution_failure(
                                job_id, owner,
                                reason=(
                                    f"{type(exc).__name__}:{exc}"),
                                preferred_executor="CPU")
                            if not requeued:
                                raise
                            continue
                        if (
                            gpu_device_index is not None
                            and _is_gpu_backend_failure(
                                exc)
                        ):
                            gpu_disabled[
                                gpu_device_index] = True
                            store.release_claim(
                                job_id, owner)
                            for queued_future in tuple(
                                gpu_pending[
                                    gpu_device_index]
                            ):
                                queued_value = pending.get(
                                    queued_future)
                                if (
                                    queued_value is None
                                    or not queued_future.cancel()
                                ):
                                    continue
                                gpu_pending[
                                    gpu_device_index].discard(
                                        queued_future)
                                pending.pop(
                                    queued_future, None)
                                store.release_claim(
                                    str(
                                        queued_value[0][
                                            "job_id"]),
                                    owner)
                            continue
                        try:
                            store.finish_job(
                                job_id, owner, "FAILED",
                                last_error=(
                                    f"{type(exc).__name__}:"
                                    f"{exc}"))
                        except RuntimeError:
                            pass
                        raise
                    processed += 1
    if workers > 1:
        gpu_monitor.close()
    touch_activity(force=True)
    store.materialize_progress()
    progress = store.progress()
    progress["processed_this_call"] = processed
    progress["gpu_jobs_submitted"] = gpu_submitted
    progress["cpu_jobs_submitted"] = cpu_submitted
    progress["gpu_device_assignments"] = gpu_device_assignments
    progress["gpu_queue_timeouts"] = gpu_queue_timeouts
    progress["actual_pool_workers"] = actual_workers if workers > 1 else 1
    progress["actual_cpu_pool_workers"] = (
        actual_cpu_workers if workers > 1 else 0)
    progress["actual_gpu_pool_workers"] = (
        gpu_pool_count * gpu_workers_per_device
        if workers > 1 else 0)
    progress["gpu_workers_per_device"] = (
        gpu_workers_per_device if workers > 1 else 0)
    progress["gpu_queue_policy"] = (
        "HGB_FIRST_PREFETCHED_LEARNED_CPU_GPU_ROUTER_V40_0_4_3")
    progress["gpu_queue_ahead"] = (
        gpu_queue_ahead if workers > 1 else 0)
    progress["gpu_queue_capacity"] = (
        gpu_queue_capacity if workers > 1 else 0)
    progress["gpu_ram_scheduler_coupled"] = False
    progress["gpu_global_section_backlog_aware"] = True
    progress["gpu_duration_profile_path"] = (
        str(
            causal_shared_compute_root
            / "candidate-execution-duration-profiles.json")
        if workers > 1 else None)
    return progress


def execute_candidate_oos_jobs(*, store: ManifestedJobStore, inputs: ManifestedJobInputs, output_root: str | Path,
                               max_jobs: int | None = None, owner: str = "candidate-oos-worker",
                               allowed_job_ids: Iterable[str] | None = None, workers: int = 1,
                               queue_ahead: int = 2, max_inflight: int | None = None,
                               worker_map_override: list[dict] | None = None,
                               ram_scheduler: RamAdmissionScheduler | None = None,
                               estimated_task_gib: float = 4.0,
                               activity_heartbeat_path: str | Path | None = None,
                               hot_reload_controller: HotReloadController | None = None) -> dict:
    """Execute generated Candidate×Horizon×Fold jobs and preserve cache hits.

    This is intentionally separate from production fitting: completed
    Candidate-OOS artifacts are evidence only and never become generation
    models.
    """
    contract = store.read("manifested-job-contract.json")
    registry = store.read("family-registry.json")
    hyperparameter_path = inputs.hyperparameter_space or (inputs.repo_root / "stock_predictor" / "v5" / "hyperparameter_space.json")
    candidates = load_primary_candidate_registry(hyperparameter_path)
    feature_schema = json.loads(inputs.feature_schema.read_text(encoding="utf-8"))
    resolved = inputs.resolved_contracts(
        primary_candidate_universe_hash=registry["candidate_registry"]["candidate_registry_sha256"])
    factory = CandidateOosFactory(signal_panel=inputs.signal_panel, feature_schema=feature_schema,
                                  candidates=candidates, output_root=Path(output_root) / "candidate-oos",
                                  signal_panel_sha256=contract["signal_panel_development_sha256"],
                                  fold_policy=resolved.fold_policy,
                                  target_contract=resolved.target_contract,
                                  development_end=inputs.development_end,
                                  holdout_boundary=inputs.holdout_boundary,
                                  random_seed=resolved.model_training_contract.random_seed,
                                  model_training_contract_hash=resolved.model_training_contract.model_training_contract_hash)
    processed = 0
    allowed_ids = (
        tuple(str(value) for value in allowed_job_ids)
        if allowed_job_ids is not None else None)
    candidate_jobs = store.jobs_by_kind_state(
        kind="candidate_oos_fold", state="PENDING")
    if allowed_ids is not None:
        allowed_lookup = set(allowed_ids)
        candidate_jobs = [
            job for job in candidate_jobs
            if str(job["job_id"]) in allowed_lookup
        ]
    fold_lookup: dict[tuple[int, str], Fold] = {}
    for horizon in sorted({int(job["horizon"]) for job in candidate_jobs}):
        for fold in factory.fold_specs(
                horizon=horizon, development_end=inputs.development_end,
                holdout_boundary=inputs.holdout_boundary):
            fold_lookup[(horizon, fold.fold_id)] = fold
    factory.release_frame_cache()

    # A prior interrupted attempt can have a fully written immutable result
    # while the SQLite row is RUNNING/PENDING.  Promote only a manifest- and
    # hash-valid partition whose causal identity still matches this job.  This
    # avoids dispatching a no-op cache hit into a GPU/CPU lane and, crucially,
    # avoids waiting on an old execution lock during resume.
    persisted_resume_count = 0
    remaining_candidate_jobs: list[dict[str, Any]] = []
    candidate_root = factory.store.root.resolve()
    for candidate_job in candidate_jobs:
        prior_result = candidate_job.get("result")
        source_path = (
            Path(str(prior_result.get("path"))).resolve()
            if isinstance(prior_result, Mapping) and prior_result.get("path")
            else None)
        reused = False
        if source_path is not None and source_path.is_file():
            try:
                source_path.relative_to(candidate_root)
                source_manifest = factory.store.verify_partition(source_path)
                expected_values = {
                    "horizon": int(candidate_job["horizon"]),
                    "fold_id": str(candidate_job["fold_id"]),
                    "candidate_id": str(candidate_job["candidate_id"]),
                    "signal_panel_sha256": str(factory.signal_panel_sha256),
                    "feature_schema_sha256": str(factory.feature_schema_sha256),
                    "fold_policy_hash": str(factory.fold_policy.fold_policy_hash),
                    "target_contract_hash": str(factory.target_contract.target_contract_hash),
                    "model_training_contract_hash": str(
                        factory.model_training_contract_hash),
                }
                reused = all(
                    str(source_manifest.get(key)) == str(value)
                    for key, value in expected_values.items()
                )
            except (OSError, ValueError, RuntimeError):
                reused = False
        if reused:
            result = dict(prior_result)
            result.update({
                "cache": "RESUME_VALIDATED_PERSISTED_RESULT",
                "resume_reused": True,
                "path": str(source_path),
            })
            claimed = store.claim_ready(
                owner, lease_seconds=JOB_LEASE_SECONDS,
                kinds=("candidate_oos_fold",),
                job_ids=(str(candidate_job["job_id"]),))
            if claimed is not None:
                store.finish_job(
                    str(candidate_job["job_id"]), owner, "COMPLETE", result=result)
                persisted_resume_count += 1
            else:
                # A concurrent recovery owner won the exact manifest claim;
                # leave this row out of the local frontier and let the next
                # validated resume observe its completed result.
                remaining_candidate_jobs.append(candidate_job)
        else:
            remaining_candidate_jobs.append(candidate_job)
    candidate_jobs = remaining_candidate_jobs

    def fold_for_job(job: Mapping[str, Any]) -> Fold:
        key = (int(job["horizon"]), str(job["fold_id"]))
        fold = fold_lookup.get(key)
        if fold is None:
            raise KeyError(f"DQBD_CANDIDATE_FOLD_NOT_RESOLVED:{key}")
        return fold

    gpu_device_count = len(opencl_fp64_devices())

    def workload_for_job(job: Mapping[str, Any]) -> str:
        family = factory.candidates[str(job["candidate_id"])].recipe_family
        if family == "HIST_GRADIENT_BOOSTING":
            return "HGB"
        if family == "RIDGE_LOGISTIC":
            return "RIDGE"
        return "CPU"

    gpu_submitted = 0
    cpu_submitted = 0
    gpu_device_assignments = [0] * gpu_device_count
    activity_path = (
        Path(activity_heartbeat_path)
        if activity_heartbeat_path is not None else None)

    def touch_activity(*, state: str = "ACTIVE", job_id: str | None = None) -> None:
        if activity_path is None:
            return
        payload = {
            "schema_version": "DQBD_SEED_BANK_ACTIVITY_V1",
            "state": str(state),
            "owner": str(owner),
            "pid": os.getpid(),
            "updated_at_epoch": time.time(),
        }
        if job_id is not None:
            payload["job_id"] = str(job_id)
        _write_json(activity_path, payload)

    def execute_one(job):
        touch_activity(job_id=str(job["job_id"]))
        try:
            horizon = int(job["horizon"])
            folds = {x.fold_id: x for x in factory.fold_specs(horizon=horizon, development_end=inputs.development_end,
                                                               holdout_boundary=inputs.holdout_boundary)}
            result = factory.build_fold(horizon=horizon, candidate_id=str(job["candidate_id"]), fold=folds[str(job["fold_id"])],
                                        development_end=inputs.development_end, holdout_boundary=inputs.holdout_boundary)
            store.finish_job(job["job_id"], owner, "COMPLETE", result=result)
            touch_activity(job_id=str(job["job_id"]))
        except Exception as exc:
            try:
                store.finish_job(job["job_id"], owner, "FAILED", last_error=f"{type(exc).__name__}:{exc}")
            except RuntimeError:
                pass
            touch_activity(state="FAILED", job_id=str(job["job_id"]))
            raise
    workers = max(1, int(workers))
    # Candidate-OOS fitting is the memory-heavy phase. ``workers`` is the
    # CPU-lane budget; GPU staging lanes are additive and do not reduce it.
    # Submit only work that already owns a RAM lease. Manifested PENDING
    # jobs are the only queue-ahead mechanism.
    active_workers = min(
        MAX_RUNTIME_PROCESS_LANES,
        max(1, min(workers, int(max_inflight or workers))),
    )
    pool_workers = active_workers
    if workers == 1:
        while max_jobs is None or processed < max_jobs:
            if hot_reload_controller is not None and hot_reload_controller.changed():
                raise _HotReloadRequested()
            job = store.claim_ready(owner, lease_seconds=24 * 60 * 60,
                                    kinds=("candidate_oos_fold",), job_ids=allowed_ids)
            if job is None or job.get("kind") != "candidate_oos_fold":
                break
            execute_one(job)
            processed += 1
    else:
        # Candidate-OOS uses one RAM-governed CPU pool plus one staging lane per
        # physical GPU. GPU futures never own RAM-scheduler leases; HGB/Ridge
        # routing is handled by the learned execution router.
        # The backend's device lock still serializes actual kernels.
        desired_gpu_workers_per_device = 1
        gpu_pool_count = min(
            gpu_device_count,
            max(0, (pool_workers - 1) // desired_gpu_workers_per_device))
        # The ready-section broker serializes one kernel per physical GPU and
        # the hardware monitor decides which device receives the next job.
        gpu_workers_per_device = (
            desired_gpu_workers_per_device if gpu_pool_count else 0)
        gpu_worker_total = gpu_pool_count * gpu_workers_per_device
        cpu_workers = pool_workers
        base_worker_map = list(worker_map_override or active_cpu_contract().get("worker_map") or [])
        if not base_worker_map:
            base_worker_map = [{"logical_processor": i, "core_index": i, "role": "fallback"}
                               for i in range(pool_workers)]
        cpu_map = base_worker_map[:cpu_workers]
        gpu_maps = []
        affinity_source = list(active_cpu_contract().get("worker_map") or [])
        cpu_logical = {
            int(entry["logical_processor"])
            for entry in cpu_map
            if entry.get("logical_processor") is not None
        }
        gpu_candidates = [
            entry for entry in affinity_source
            if entry.get("logical_processor") not in cpu_logical
        ] or affinity_source or base_worker_map
        for index in range(gpu_pool_count):
            start = index * gpu_workers_per_device
            selected = gpu_candidates[start:start + gpu_workers_per_device]
            if len(selected) < gpu_workers_per_device:
                selected = [
                    gpu_candidates[offset % len(gpu_candidates)]
                    for offset in range(
                        start, start + gpu_workers_per_device)
                ]
            gpu_maps.append(selected)
        gpu_devices = opencl_fp64_devices()[:gpu_pool_count]
        telemetry_root = store.root / "telemetry"
        shared_compute_root = store.root.parent / "_shared-compute"
        gpu_monitor = GpuRuntimeMonitor(
            telemetry_root, queue_root=shared_compute_root)
        gpu_runtime_samples = 0
        gpu_fallbacks = 0
        gpu_queue_timeouts = 0
        gpu_hgb_jobs_submitted = 0
        gpu_ridge_spillover_jobs_submitted = 0
        gpu_queue_depth_peak = [0] * gpu_pool_count
        gpu_disabled = [False] * gpu_pool_count
        gpu_device_assignments = [0] * gpu_pool_count
        allowed_id_set = set(allowed_ids) if allowed_ids is not None else None
        gpu_queue_ahead = max(0, int(queue_ahead))
        gpu_queue_capacity = (
            gpu_workers_per_device + gpu_queue_ahead
            if gpu_pool_count else 0)
        prep_prefetch_target = (
            min(gpu_queue_ahead, cpu_workers)
            if gpu_pool_count else 0)
        cpu_queue_capacity = cpu_workers
        duration_profiles = _CandidateExecutionDurationProfiles(
            shared_compute_root)
        def queue_event(
            queue_name: str, event: str, *,
            job: Mapping[str, Any] | None = None,
            device: int | None = None, depth: int | None = None,
            workload: str | None = None, backend: str | None = None,
            predicted_seconds: float | None = None,
            predicted_finish_monotonic: float | None = None,
            routing_reason: str | None = None,
        ) -> None:
            path = telemetry_root / ("gpu-queue-events.jsonl" if queue_name.startswith("gpu")
                                    else "cpu-queue-events.jsonl")
            payload = {"timestamp": time.time(), "event": event, "queue": queue_name}
            if job is not None:
                payload.update({"job_id": str(job["job_id"]), "horizon": int(job["horizon"]),
                                "fold_id": str(job["fold_id"]),
                                "candidate_id": str(job["candidate_id"])})
            if device is not None:
                payload["gpu_device_index"] = int(device)
            if depth is not None:
                payload["queue_depth"] = int(depth)
            if workload is not None:
                payload["workload"] = workload
            if backend is not None:
                payload["execution_backend"] = backend
            if predicted_seconds is not None:
                payload["predicted_duration_seconds"] = float(
                    predicted_seconds)
            if predicted_finish_monotonic is not None:
                payload["predicted_finish_monotonic"] = float(
                    predicted_finish_monotonic)
            if routing_reason is not None:
                payload["routing_reason"] = str(routing_reason)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, sort_keys=True, default=str) + "\n")

        prepared_root = (
            store.root.parent / "_shared-compute" / "prepared-folds")
        prepared_fold_cache: dict[tuple[int, str], bool] = {}

        def fold_key(job: Mapping[str, Any]) -> tuple[int, str]:
            return int(job["horizon"]), str(job["fold_id"])

        def fold_prepared(job: Mapping[str, Any]) -> bool:
            key = fold_key(job)
            cached = prepared_fold_cache.get(key)
            if cached is not None:
                return bool(cached)
            horizon, fold_id = key
            folder = prepared_root / f"H{horizon:02d}" / fold_id
            prepared = folder.is_dir() and any(folder.glob("*.pkl"))
            prepared_fold_cache[key] = bool(prepared)
            return bool(prepared)

        def unprepared_inflight_keys() -> set[tuple[int, str]]:
            return {
                fold_key(value[0])
                for value in pending.values()
                if not fold_prepared(value[0])
            }

        def cleanup_prepared_fold_if_complete(
            job: Mapping[str, Any],
        ) -> bool:
            horizon, fold_id = fold_key(job)
            prefix = f"candidate_oos_fold:{horizon}:{fold_id}:"
            ids = store.job_ids_by_kind_prefix(
                kind="candidate_oos_fold", prefix=prefix)
            if not ids:
                return False
            if any(
                (store.job(job_id) or {}).get("state") != "COMPLETE"
                for job_id in ids
            ):
                return False
            folder = prepared_root / f"H{horizon:02d}" / fold_id
            if not folder.is_dir():
                return False
            shutil.rmtree(folder, ignore_errors=False)
            return True

        def prep_admissible(
            jobs: list[dict[str, Any]],
        ) -> list[dict[str, Any]]:
            blocked = unprepared_inflight_keys()
            return [
                job for job in jobs
                if fold_prepared(job) or fold_key(job) not in blocked
            ]

        # This is the only broad Candidate-OOS queue read in this executor.
        # The resulting frontier is updated by claim/complete/requeue events;
        # scheduling ticks never call jobs_by_kind_state again.
        ready_frontier_cache = _CandidateOosReadyFrontier(
            candidate_jobs, workload_for_job, fold_prepared)

        def ready_gpu_job_ids() -> tuple[str, ...]:
            """Prepared GPU frontier: always HGB first, then Ridge."""
            return ready_frontier_cache.gpu_ids(
                limit=max(64, gpu_queue_capacity * 4))

        def ready_prepare_job_ids() -> tuple[str, ...]:
            """Distinct unprepared H×Fold producers for GPU feed prefetch."""
            return ready_frontier_cache.prepare_ids(
                unprepared_inflight_keys(),
                limit=max(64, prep_prefetch_target * 8))

        def ready_cpu_job_ids() -> tuple[str, ...]:
            """CPU frontier: prepared spillover; prep is prefetched separately."""
            limit = max(64, cpu_queue_capacity * 4)
            return ready_frontier_cache.cpu_ids(
                limit=limit,
                prepared_gpu_only=bool(prep_prefetch_target),
            )

        with ExitStack() as pool_stack:
            cpu_pool, actual_cpu_workers = _manifested_pool(
                workers=cpu_workers, role="candidate-oos-cpu",
                telemetry_root=telemetry_root, worker_map_override=cpu_map,
                max_tasks_per_child=_configured_lane_recycle_limit())
            cpu_pause = pool_stack.enter_context(
                _managed_pool_with_ram_pause(cpu_pool, ram_scheduler, telemetry_root))
            gpu_pools = []
            gpu_pauses = []
            for device_index in range(gpu_pool_count):
                gpu_pool, actual_gpu_workers = _manifested_pool(
                    workers=gpu_workers_per_device,
                    role=f"candidate-oos-gpu-{device_index}",
                    telemetry_root=telemetry_root,
                    worker_map_override=gpu_maps[device_index],
                    max_pending=gpu_queue_capacity,
                    max_tasks_per_child=_configured_lane_recycle_limit())
                gpu_pools.append(gpu_pool)
                # GPU execution is intentionally independent of RAM admission
                # and RAM reclaim. CPU lanes remain under RamAdmissionScheduler;
                # GPU lanes are owned only by the GPU workload router.
                gpu_pauses.append(pool_stack.enter_context(
                    _managed_pool_with_ram_pause(
                        gpu_pool, None, telemetry_root)))
            pool_stack.enter_context(
                _LaneHealthWatchdog(
                    store=store,
                    owner=owner,
                    pools=[cpu_pool, *gpu_pools],
                    telemetry_root=telemetry_root,
                ))
            actual_workers = actual_cpu_workers + (
                gpu_pool_count * gpu_workers_per_device)
            pending: dict[
                Any, tuple[dict, str, int | None, str, str, dict[str, Any]]
            ] = {}
            cpu_pending: set[Any] = set()
            gpu_pending: list[set[Any]] = [set() for _ in range(gpu_pool_count)]
            spill_root = store.root / "ram-spill"
            spill_path = spill_root / "candidate-oos-progress.json"
            last_spill_at = 0.0
            last_gpu_monitor_at = 0.0
            healthy_devices = [True] * gpu_pool_count

            def write_spill_progress(
                event: str, *, force: bool = False,
            ) -> None:
                nonlocal last_spill_at
                now = time.monotonic()
                if not force and now - last_spill_at < 2.0:
                    return
                last_spill_at = now
                """Persist the small live queue state on SSD for resume/audit."""
                spill_root.mkdir(parents=True, exist_ok=True)
                queued = []
                for future, value in tuple(pending.items()):
                    job, workload, device_index, queue_name, job_class, descriptor = value
                    queued.append({"job_id": str(job["job_id"]), "queue": queue_name,
                                   "workload": workload, "gpu_device_index": device_index,
                                   "future_done": bool(future.done())})
                payload = {
                    "schema_version": "DQBD_RAM_SPILL_CANDIDATE_PROGRESS_V1",
                    "event": event, "timestamp": time.time(),
                    "owner": owner, "processed_this_call": processed,
                    "pending": sorted(queued, key=lambda row: row["job_id"]),
                    "cpu_queue_depth": len(cpu_pending),
                    "gpu_queue_depth": [len(queue) for queue in gpu_pending],
                    "gpu_queue_capacity": gpu_queue_capacity,
                    "unprepared_fold_producers": [
                        {"horizon": key[0], "fold_id": key[1]}
                        for key in sorted(unprepared_inflight_keys())
                    ],
                    "evaluation_not_opened": True,
                }
                temporary = spill_path.with_name(spill_path.name + ".tmp")
                temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                                     encoding="utf-8")
                os.replace(temporary, spill_path)

            def reconcile_pause_controllers() -> None:
                if cpu_pause is not None:
                    cpu_pause.reconcile()

            def job_class_for(
                job: Mapping[str, Any], workload: str,
                device_index: int | None,
            ) -> str:
                if (device_index is None and workload == "RIDGE"
                        and not fold_prepared(job)):
                    return "candidate_oos:PREP_RIDGE:CPU"
                return (
                    f"candidate_oos:{workload}:"
                    f"{'GPU' if device_index is not None else 'CPU'}")

            def descriptor_for(
                job: Mapping[str, Any], workload: str,
                device_index: int | None,
            ) -> dict[str, Any]:
                fold = fold_for_job(job)
                return {
                    "job_id": str(job["job_id"]),
                    "kind": "candidate_oos_fold",
                    "horizon": int(job["horizon"]),
                    "fold_id": str(job["fold_id"]),
                    "candidate_id": str(job["candidate_id"]),
                    "workload": workload,
                    "prepared": bool(fold_prepared(job)),
                    "device_index": device_index,
                    "train_sessions": len(fold.train_dates),
                    "validation_sessions": len(fold.validation_dates),
                    "ram_reclaim_count": int(
                        job.get("ram_reclaim_count", 0) or 0),
                }

            def profile_workload(
                workload: str, descriptor: Mapping[str, Any],
            ) -> str:
                if bool(descriptor.get("prepare_only")):
                    return "FOLD_PREP:RAW"
                return (
                    f"{str(workload)}:"
                    f"{'PREPARED' if descriptor.get('prepared') else 'RAW'}")

            def gpu_next_start(
                device_index: int,
            ) -> float:
                """Earliest virtual worker-lane start including queued futures."""
                now = time.monotonic()
                intervals: list[tuple[float, float]] = []
                for future in tuple(gpu_pending[device_index]):
                    value = pending.get(future)
                    if value is None:
                        continue
                    desc = value[5]
                    start = float(desc.get(
                        "predicted_start_monotonic",
                        desc.get("submitted_monotonic", now)))
                    finish = float(desc.get(
                        "predicted_finish_monotonic", start))
                    if finish > now:
                        intervals.append((max(now, start), finish))
                candidate = now
                while True:
                    active_finishes = [
                        finish for start, finish in intervals
                        if start <= candidate < finish
                    ]
                    if len(active_finishes) < gpu_workers_per_device:
                        return candidate
                    candidate = min(active_finishes)

            def gpu_predicted_finish(
                workload: str, device_index: int,
            ) -> float:
                descriptor = {
                    "prepared": True,
                    "workload": workload,
                }
                duration = duration_profiles.estimate(
                    profile_workload(workload, descriptor),
                    "GPU", gpu_devices[device_index])
                now = time.monotonic()
                global_backlog_seconds = (
                    _gpu_global_backlog_seconds(
                        shared_compute_root,
                        gpu_devices,
                        device_index))
                predicted_start = max(
                    gpu_next_start(device_index),
                    now + global_backlog_seconds)
                return predicted_start + duration

            def best_gpu_predicted_finish(
                workload: str,
            ) -> float:
                values = [
                    gpu_predicted_finish(workload, index)
                    for index in range(gpu_pool_count)
                    if (
                        not gpu_disabled[index]
                        and index < len(healthy_devices)
                        and healthy_devices[index]
                        and gpu_workload_allowed(
                            gpu_devices[index], workload))
                ]
                return min(values) if values else float("inf")

            def cpu_route_allowed(
                preview: Mapping[str, Any], workload: str,
                descriptor: Mapping[str, Any],
            ) -> bool:
                if workload not in {"HGB", "RIDGE"}:
                    return True
                if not bool(descriptor.get("prepared")):
                    return True
                if str(preview.get(
                    "execution_preference", "")).upper() == "CPU":
                    return True
                healthy_gpu = any(
                    not gpu_disabled[index]
                    and index < len(healthy_devices)
                    and healthy_devices[index]
                    and gpu_workload_allowed(
                        gpu_devices[index], workload)
                    for index in range(gpu_pool_count))
                if (
                    healthy_gpu
                    and str(preview.get(
                        "execution_preference", "")).upper() == "GPU"
                ):
                    return False
                cpu_finish = (
                    time.monotonic()
                    + duration_profiles.estimate(
                        profile_workload(workload, descriptor),
                        "CPU"))
                return cpu_finish <= best_gpu_predicted_finish(workload)

            def claim_for_executor(
                job_ids: Iterable[str], *, device_index: int | None,
            ) -> tuple[dict, str, str, float, dict[str, Any]] | None:
                previews = []
                for job_id in job_ids:
                    preview = store.job(str(job_id))
                    if not preview or preview.get("state") != "PENDING":
                        ready_frontier_cache.discard(str(job_id))
                        continue
                    workload = workload_for_job(preview)
                    if not gpu_workload_allowed(
                        gpu_devices[device_index], workload):
                        continue
                    job_class = job_class_for(
                        preview, workload, device_index)
                    descriptor = descriptor_for(
                        preview, workload, device_index)
                    if (
                        device_index is None
                        and not cpu_route_allowed(
                            preview, workload, descriptor)
                    ):
                        continue
                    workload_priority = (
                        _candidate_gpu_workload_priority(
                            workload,
                            prepared=bool(descriptor["prepared"]))
                        if device_index is not None
                        else (
                            0 if (
                                workload == "RIDGE"
                                and not descriptor["prepared"])
                            else (
                                1 if workload == "HGB"
                                else (2 if workload == "RIDGE" else 3))))
                    preference_priority = (
                        0 if (
                            device_index is not None
                            and str(preview.get(
                                "execution_preference", "")).upper()
                                == "GPU")
                        else 1)
                    completion_priority = (
                        -1 if int(
                            preview.get("ram_reclaim_count", 0) or 0) > 0
                        else 0)
                    ram_score = (
                        ram_scheduler.priority_key(
                            job_class, descriptor, gpu_idle=False)
                        if (
                            device_index is None
                            and ram_scheduler is not None)
                        else (0.0, 0.0, 0.0))
                    score = (
                        workload_priority,
                        preference_priority,
                        completion_priority,
                        *ram_score)
                    previews.append(
                        (score, str(job_id), preview, workload,
                         job_class, descriptor))
                previews.sort(key=lambda value: (value[0], value[1]))
                for (
                    _, job_id, preview, workload, job_class, descriptor
                ) in previews:
                    if device_index is None:
                        reservation = (
                            ram_scheduler.try_acquire(
                                estimated_task_gib,
                                wait_callback=reconcile_pause_controllers,
                                job_class=job_class,
                                descriptor=descriptor)
                            if ram_scheduler is not None else 0.0)
                        if reservation is None:
                            continue
                    else:
                        # GPU jobs are governed only by the GPU workload
                        # router. They own no RamAdmissionScheduler lease.
                        reservation = 0.0
                    claimed = store.claim_ready(
                        owner, lease_seconds=JOB_LEASE_SECONDS,
                        kinds=("candidate_oos_fold",),
                        job_ids=(job_id,))
                    if claimed is None:
                        if (
                            device_index is None
                            and ram_scheduler is not None
                        ):
                            ram_scheduler.release(
                                reservation, job_class=job_class)
                        continue
                    ready_frontier_cache.claim(job_id)
                    return (
                        claimed, workload, job_class,
                        reservation, descriptor)
                return None

            def claim_prepare_only(
                job_ids: Iterable[str],
            ) -> tuple[dict, str, str, float, dict[str, Any]] | None:
                for job_id in job_ids:
                    preview = store.job(str(job_id))
                    if not preview or preview.get("state") != "PENDING":
                        ready_frontier_cache.discard(str(job_id))
                        continue
                    workload = workload_for_job(preview)
                    descriptor = descriptor_for(
                        preview, workload, None)
                    descriptor["prepare_only"] = True
                    descriptor["prepared"] = False
                    job_class = "candidate_oos:PREP_RIDGE:CPU"
                    reservation = (
                        ram_scheduler.try_acquire(
                            estimated_task_gib,
                            wait_callback=reconcile_pause_controllers,
                            job_class=job_class,
                            descriptor=descriptor)
                        if ram_scheduler is not None else 0.0)
                    if reservation is None:
                        continue
                    claimed = store.claim_ready(
                        owner, lease_seconds=JOB_LEASE_SECONDS,
                        kinds=("candidate_oos_fold",),
                        job_ids=(str(job_id),))
                    if claimed is None:
                        if ram_scheduler is not None:
                            ram_scheduler.release(
                                reservation, job_class=job_class)
                        continue
                    ready_frontier_cache.claim(str(job_id))
                    return (
                        claimed, workload, job_class,
                        reservation, descriptor)
                return None

            def submit(pool, queue_name: str, job: dict, workload: str,
                       device_index: int | None, reservation: float,
                       job_class: str, descriptor: Mapping[str, Any],
                       *, prepare_only: bool = False):
                backend = (CPU_EXECUTION_BACKEND if device_index is None else
                           (GPU_HGB_EXECUTION_BACKEND if workload == "HGB"
                            else GPU_EXECUTION_BACKEND))
                execution_fingerprint = _gpu_execution_fingerprint(
                    workload, device_index)
                descriptor = dict(descriptor)
                descriptor["prepare_only"] = bool(prepare_only)
                executor_name = (
                    "CPU" if device_index is None else "GPU")
                duration_key = profile_workload(workload, descriptor)
                predicted_duration = duration_profiles.estimate(
                    duration_key, executor_name,
                    None if device_index is None
                    else gpu_devices[device_index])
                submitted_monotonic = time.monotonic()
                predicted_start = (
                    gpu_next_start(device_index)
                    if device_index is not None
                    else submitted_monotonic)
                descriptor.update({
                    "executor": executor_name,
                    "submitted_monotonic": submitted_monotonic,
                    "predicted_start_monotonic": predicted_start,
                    "predicted_duration_seconds": predicted_duration,
                    "predicted_finish_monotonic":
                        predicted_start + predicted_duration,
                })
                live_config = (
                    ram_scheduler.open_live_job(job_class, descriptor)
                    if (
                        device_index is None
                        and ram_scheduler is not None)
                    else None)
                payload = {
                    "output_root": str(store.root), "inputs": inputs, "job": job,
                    "execution_backend_override": backend,
                    "execution_backend_fingerprint": execution_fingerprint,
                    "_ram_job_class": job_class,
                    "_live_ram": live_config,
                    "_prepare_only": bool(prepare_only),
                    "fold": fold_for_job(job),
                }
                if device_index is not None:
                    payload["gpu_device_index"] = device_index
                try:
                    future = pool.submit(_candidate_oos_process_job, payload)
                except Exception:
                    if (
                        device_index is None
                        and ram_scheduler is not None
                    ):
                        ram_scheduler.close_live_job(live_config)
                        ram_scheduler.release(
                            reservation, job_class=job_class)
                    raise
                if (
                    device_index is None
                    and ram_scheduler is not None
                ):
                    def release_runtime(
                        _future, amount=reservation, cls=job_class,
                        live=live_config,
                    ) -> None:
                        ram_scheduler.close_live_job(live)
                        ram_scheduler.release(
                            amount, job_class=cls)
                    future.add_done_callback(release_runtime)
                pending[future] = (
                    job, workload, device_index, queue_name,
                    job_class, descriptor)
                if device_index is None:
                    cpu_pending.add(future)
                    queue_event(
                        "cpu", "SUBMIT", job=job,
                        depth=len(cpu_pending),
                        workload=workload, backend=backend,
                        predicted_seconds=descriptor.get(
                            "predicted_duration_seconds"),
                        predicted_finish_monotonic=descriptor.get(
                            "predicted_finish_monotonic"),
                        routing_reason=(
                            "FOLD_PREP_PREFETCH"
                            if descriptor.get("prepare_only")
                            else (
                                "CPU_PREP_REQUIRED"
                                if not descriptor.get("prepared")
                                else "LEARNED_CPU_FINISH_BEFORE_GPU")))
                else:
                    gpu_pending[device_index].add(future)
                    gpu_queue_depth_peak[device_index] = max(
                        gpu_queue_depth_peak[device_index],
                        len(gpu_pending[device_index]))
                    queue_event(
                        f"gpu-{device_index}", "SUBMIT", job=job,
                        device=device_index,
                        depth=len(gpu_pending[device_index]),
                        workload=workload,
                        backend=f"{backend}:{execution_fingerprint[:12]}",
                        predicted_seconds=descriptor.get(
                            "predicted_duration_seconds"),
                        predicted_finish_monotonic=descriptor.get(
                            "predicted_finish_monotonic"),
                        routing_reason=(
                            "CPU_RECLAIM_GPU_PREFERENCE"
                            if str(job.get(
                                "execution_preference", "")).upper()
                                == "GPU"
                            else (
                                "HGB_FIRST_GPU_QUEUE"
                                if workload == "HGB"
                                else "RIDGE_AFTER_HGB_GPU_QUEUE")))
                return future

            while max_jobs is None or processed < max_jobs:
                if hot_reload_controller is not None and hot_reload_controller.changed():
                    raise _HotReloadRequested()
                reconcile_pause_controllers()
                control_state = (
                    ram_scheduler.real_load_state()
                    if ram_scheduler is not None else {
                        "used_fraction": 0.0})
                used_fraction = float(
                    control_state.get("used_fraction", 0.0))
                admission_budget = (
                    ram_scheduler.admission_budget(used_fraction)
                    if ram_scheduler is not None
                    else (
                        8 if used_fraction < .60
                        else (4 if used_fraction < .80 else 2)))
                # Fill all available CPU lanes below the 80% corridor. The
                # real-load admission check remains authoritative per job;
                # this only removes the old fixed 8-lane refill ceiling.
                if used_fraction < .80:
                    recovery = (
                        ram_scheduler.recovery_state()
                        if ram_scheduler is not None else {})
                    if not recovery.get("settling", False) and not recovery.get(
                            "recovering", False):
                        admission_budget = max(
                            admission_budget,
                            cpu_workers - len(cpu_pending))
                admitted_cpu_this_tick = 0
                now_gpu_monitor = time.monotonic()
                if now_gpu_monitor - last_gpu_monitor_at >= .25:
                    gpu_snapshot = gpu_monitor.snapshot(gpu_devices)
                    gpu_runtime_samples += 1
                    healthy_devices = [
                        bool(row.get("assignment_allowed", False))
                        for row in gpu_snapshot.get("devices", ())]
                    last_gpu_monitor_at = now_gpu_monitor

                # GPU dispatch is independent from RAM admission. Fill every
                # available GPU staging lane from the prepared shared frontier,
                # always HGB first and Ridge second.
                for queue_slot in range(gpu_queue_capacity):
                    job_ids = ready_gpu_job_ids()
                    if not job_ids:
                        break
                    next_preview = store.job(str(job_ids[0]))
                    next_workload = (
                        workload_for_job(next_preview)
                        if next_preview is not None else "HGB")
                    device_order = sorted(
                        range(gpu_pool_count),
                        key=lambda index: (
                            gpu_predicted_finish(
                                next_workload, index),
                            len(gpu_pending[index]),
                            index))
                    for device_index in device_order:
                        if (
                            max_jobs is not None
                            and processed + len(pending) >= max_jobs
                        ):
                            break
                        if (
                            gpu_disabled[device_index]
                            or device_index >= len(healthy_devices)
                            or not healthy_devices[device_index]
                            or len(gpu_pending[device_index]) > queue_slot
                        ):
                            continue
                        leased = claim_for_executor(
                            job_ids, device_index=device_index)
                        if leased is None:
                            continue
                        (
                            job, workload, job_class,
                            reservation, descriptor,
                        ) = leased
                        descriptor["predicted_gpu_seconds"] = (
                            _gpu_job_seconds(
                                gpu_devices[device_index], workload))
                        try:
                            submit(
                                gpu_pools[device_index],
                                f"gpu-{device_index}", job,
                                workload, device_index, reservation,
                                job_class, descriptor)
                        except Exception as exc:
                            store.release_claim(
                                str(job["job_id"]), owner)
                            if _is_gpu_backend_failure(exc):
                                gpu_disabled[device_index] = True
                                queue_event(
                                    f"gpu-{device_index}",
                                    "GPU_SUBMIT_BACKEND_FAILURE",
                                    job=job, device=device_index,
                                    depth=len(
                                        gpu_pending[device_index]),
                                    workload=workload)
                                continue
                            raise
                        gpu_device_assignments[device_index] += 1
                        gpu_submitted += 1
                        if workload == "HGB":
                            gpu_hgb_jobs_submitted += 1
                        elif workload == "RIDGE":
                            gpu_ridge_spillover_jobs_submitted += 1

                # Keep candidate-independent H×Fold preparation ahead of
                # the GPUs. These are execution-only prefetches: the claimed
                # Candidate returns to PENDING immediately after the shared
                # prepared-fold artifact is materialized.
                prep_inflight = sum(
                    1 for value in pending.values()
                    if bool(value[5].get("prepare_only")))
                while (
                    admitted_cpu_this_tick < admission_budget
                    and prep_inflight < prep_prefetch_target
                    and len(cpu_pending) < cpu_queue_capacity
                    and (
                        max_jobs is None
                        or processed + len(pending) < max_jobs)
                ):
                    prep_ids = ready_prepare_job_ids()
                    if not prep_ids:
                        break
                    leased = claim_prepare_only(prep_ids)
                    if leased is None:
                        break
                    (
                        job, workload, job_class,
                        reservation, descriptor,
                    ) = leased
                    try:
                        submit(
                            cpu_pool, "cpu", job, workload, None,
                            reservation, job_class, descriptor,
                            prepare_only=True)
                    except Exception:
                        store.release_claim(
                            str(job["job_id"]), owner)
                        raise
                    cpu_submitted += 1
                    admitted_cpu_this_tick += 1
                    prep_inflight += 1

                # CPU remains governed by the unchanged RAM scheduler. It is
                # opportunistic for prepared HGB/Ridge: claim_for_executor()
                # compares learned completion time against the next GPU finish.
                while (
                    admitted_cpu_this_tick < admission_budget
                    and len(cpu_pending) < cpu_queue_capacity
                    and (
                        max_jobs is None
                        or processed + len(pending) < max_jobs)
                ):
                    cpu_ids = ready_cpu_job_ids()
                    if not cpu_ids:
                        break
                    leased = claim_for_executor(
                        cpu_ids, device_index=None)
                    if leased is None:
                        break
                    (
                        job, workload, job_class,
                        reservation, descriptor,
                    ) = leased
                    try:
                        submit(
                            cpu_pool, "cpu", job, workload, None,
                            reservation, job_class, descriptor)
                    except Exception:
                        store.release_claim(
                            str(job["job_id"]), owner)
                        raise
                    cpu_submitted += 1
                    admitted_cpu_this_tick += 1

                write_spill_progress("QUEUES_FILLED")
                if not pending:
                    if (
                        ready_gpu_job_ids()
                        or ready_cpu_job_ids()
                        or ready_prepare_job_ids()
                    ):
                        time.sleep(.01)
                        continue
                    break
                if "last_heartbeat_at" not in locals():
                    last_heartbeat_at = time.monotonic()
                done, _ = wait(
                    tuple(pending), timeout=.01,
                    return_when=FIRST_COMPLETED)
                reconcile_pause_controllers()
                now_heartbeat = time.monotonic()
                if now_heartbeat - last_heartbeat_at >= 30.0:
                    for future, pending_value in tuple(pending.items()):
                        if future not in done:
                            store.heartbeat(
                                pending_value[0]["job_id"], owner,
                                lease_seconds=JOB_LEASE_SECONDS)
                    last_heartbeat_at = now_heartbeat
                for future in done:
                    (
                        job, workload, device_index, queue_name,
                        job_class, descriptor,
                    ) = pending[future]
                    job_id = str(job["job_id"])
                    if device_index is not None:
                        gpu_pending[device_index].discard(future)
                    else:
                        cpu_pending.discard(future)
                    pending.pop(future, None)
                    try:
                        result = future.result()
                        wall_elapsed = max(
                            .001,
                            time.monotonic() - float(
                                descriptor.get(
                                    "submitted_monotonic",
                                    time.monotonic())))
                        execution_elapsed = (
                            float(result.get(
                                "_runtime_execution_seconds",
                                wall_elapsed))
                            if isinstance(result, dict)
                            else wall_elapsed)
                        executor_name = (
                            "GPU" if device_index is not None else "CPU")
                        duration_profiles.observe(
                            profile_workload(
                                workload, descriptor),
                            executor_name, execution_elapsed,
                            None if device_index is None
                            else gpu_devices[device_index])
                        if (
                            device_index is None
                            and ram_scheduler is not None
                            and isinstance(result, dict)
                        ):
                            memory = result.get("_runtime_memory") or {}
                            ram_scheduler.observe(
                                job_class,
                                memory.get("rss_peak_gib"),
                                memory.get("incremental_peak_gib"),
                                descriptor=descriptor)
                        if bool(descriptor.get("prepare_only")):
                            store.release_claim(job_id, owner)
                            prepared_fold_cache[fold_key(job)] = True
                            ready_frontier_cache.requeue(
                                store.job(job_id) or job)
                            ready_frontier_cache.mark_prepared(
                                fold_key(job))
                            queue_event(
                                "cpu", "PREP_COMPLETE_REQUEUE",
                                job=job, depth=len(cpu_pending),
                                workload="FOLD_PREP",
                                routing_reason="GPU_FEED_PREFETCH_READY")
                            continue
                        store.finish_job(
                            job_id, owner, "COMPLETE", result=result)
                        ready_frontier_cache.complete(job_id)
                        prep_cleaned = (
                            cleanup_prepared_fold_if_complete(job))
                        queue_event(
                            queue_name, "COMPLETE", job=job,
                            device=device_index,
                            depth=(
                                len(gpu_pending[device_index])
                                if device_index is not None
                                else len(cpu_pending)),
                            workload=workload)
                        if prep_cleaned:
                            queue_event(
                                "cpu", "PREP_CACHE_CLEANED",
                                job=job, depth=len(cpu_pending),
                                workload="FOLD_PREP")
                        processed += 1
                    except RamJobReclaimed as exc:
                        prefer_gpu = (
                            device_index is None
                            and workload in {"HGB", "RIDGE"}
                            and bool(descriptor.get("prepared")))
                        store.requeue_reclaimed_job(
                            job_id, owner,
                            reason=exc.reason,
                            released_gib=exc.released_gib,
                            preferred_executor=(
                                "GPU" if prefer_gpu else None))
                        ready_frontier_cache.requeue(
                            store.job(job_id) or job)
                        queue_event(
                            queue_name,
                            (
                                "RAM_RECLAIM_REQUEUE_FOR_GPU"
                                if prefer_gpu
                                else "RAM_RECLAIM_REQUEUE"),
                            job=job, device=device_index,
                            depth=(
                                len(gpu_pending[device_index])
                                if device_index is not None
                                else len(cpu_pending)),
                            workload=workload)
                        continue
                    except Exception as exc:
                        if (
                            device_index is not None
                            and isinstance(exc, GPUSectionQueueTimeout)
                        ):
                            # A bounded physical-section wait is transient:
                            # requeue only this job for CPU and leave the GPU
                            # eligible for subsequent work.  Permanent
                            # device disablement here caused the old
                            # "until coordinator restart" behavior.
                            gpu_queue_timeouts += 1
                            requeued = store.requeue_execution_failure(
                                job_id, owner,
                                reason=(
                                    f"{type(exc).__name__}:{exc}"),
                                preferred_executor="CPU")
                            if not requeued:
                                raise
                            ready_frontier_cache.requeue(
                                store.job(job_id) or job)
                            queue_event(
                                queue_name,
                                "GPU_SECTION_QUEUE_TIMEOUT_REQUEUE",
                                job=job, device=device_index,
                                depth=len(gpu_pending[device_index]),
                                workload=workload,
                                routing_reason="TRANSIENT_TIMEOUT_CPU_RETRY",
                            )
                            continue
                        if isinstance(exc, BrokenProcessPool):
                            failure_count = (
                                int(
                                    job.get(
                                        "worker_failure_count", 0)
                                    or 0)
                                + 1)
                            if failure_count <= WORKER_FAILURE_RETRY_LIMIT:
                                # The one-worker lane has already been
                                # replaced by the pool relay.  Keep the
                                # scientific job PENDING and retry it; a
                                # transient child exit must not abort the
                                # complete causal materialization.
                                preferred_executor = (
                                    "CPU"
                                    if (
                                        device_index is not None
                                        and failure_count >= 2
                                    ) else None)
                                requeued = (
                                    store.requeue_execution_failure(
                                        job_id, owner,
                                        reason=(
                                            f"{type(exc).__name__}:"
                                            f"{exc}"),
                                        preferred_executor=(
                                            preferred_executor),
                                        count_worker_failure=(
                                            "DQBD_STALE_DISPATCHER_ACTIVITY"
                                            not in str(exc))))
                                if not requeued:
                                    # The independent seed-bank watchdog may
                                    # have already returned this exact job to
                                    # PENDING while the lane was being
                                    # recycled. That is a successful external
                                    # recovery, not a reason to abort the
                                    # whole bank.
                                    if (
                                        (store.job(job_id) or {}).get(
                                            "state") != "PENDING"
                                    ):
                                        raise
                                ready_frontier_cache.requeue(
                                    store.job(job_id) or job)
                                queue_event(
                                    queue_name,
                                    "WORKER_PROCESS_CRASH_REQUEUE",
                                    job=job, device=device_index,
                                    depth=(
                                        len(
                                            gpu_pending[device_index])
                                        if device_index is not None
                                        else len(cpu_pending)),
                                    workload=workload,
                                    routing_reason=(
                                        "RETRY_ON_REPLACED_LANE"
                                        if preferred_executor is None
                                        else "GPU_CRASH_CPU_FALLBACK"),
                                )
                                continue
                        if (
                            device_index is not None
                            and _is_gpu_backend_failure(exc)
                        ):
                            gpu_fallbacks += 1
                            gpu_disabled[device_index] = True
                            store.release_claim(job_id, owner)
                            ready_frontier_cache.requeue(
                                store.job(job_id) or job)
                            queue_event(
                                queue_name,
                                "GPU_BACKEND_FAILURE_REQUEUE",
                                job=job, device=device_index,
                                depth=len(
                                    gpu_pending[device_index]),
                                workload=workload)
                            for queued_future in tuple(
                                    gpu_pending[device_index]):
                                queued_value = pending.get(
                                    queued_future)
                                if (
                                    queued_value is None
                                    or not queued_future.cancel()
                                ):
                                    continue
                                queued_job = queued_value[0]
                                gpu_pending[
                                    device_index].discard(
                                        queued_future)
                                pending.pop(
                                    queued_future, None)
                                store.release_claim(
                                    str(queued_job["job_id"]),
                                    owner)
                                ready_frontier_cache.requeue(
                                    store.job(
                                        str(queued_job["job_id"]))
                                    or queued_job)
                                queue_event(
                                    queue_name,
                                    "GPU_CANCEL_REQUEUE",
                                    job=queued_job,
                                    device=device_index,
                                    depth=len(
                                        gpu_pending[
                                            device_index]),
                                    workload=queued_value[1])
                            continue
                        try:
                            store.finish_job(
                                job_id, owner, "FAILED",
                                last_error=(
                                    f"{type(exc).__name__}:{exc}"))
                        except RuntimeError:
                            pass
                        raise
                write_spill_progress("WAIT_CYCLE_COMPLETE")
            write_spill_progress(
                "CANDIDATE_OOS_POOL_COMPLETE", force=True)
    if workers > 1:
        gpu_monitor.close()
    store.materialize_progress()
    return store.progress() | {"processed_this_call": processed,
                               "persisted_resume_reused": persisted_resume_count,
                               "candidate_registry_sha256": registry["candidate_registry"]["candidate_registry_sha256"],
                               "gpu_jobs_submitted": gpu_submitted,
                               "gpu_hgb_jobs_submitted": (
                                   gpu_hgb_jobs_submitted
                                   if workers > 1 else 0),
                               "gpu_ridge_spillover_jobs_submitted": (
                                   gpu_ridge_spillover_jobs_submitted
                                   if workers > 1 else 0),
                               "cpu_jobs_submitted": cpu_submitted,
                               "gpu_device_assignments": gpu_device_assignments,
                               "actual_pool_workers": actual_workers if workers > 1 else 1,
                               "actual_cpu_pool_workers": (actual_cpu_workers if workers > 1 else 0),
                               "actual_gpu_pool_workers": (
                                   gpu_pool_count * gpu_workers_per_device
                                   if workers > 1 else 0),
                               "gpu_workers_per_device": (
                                   gpu_workers_per_device
                                   if workers > 1 else 0),
                               "gpu_queue_policy": "HGB_FIRST_PREFETCHED_LEARNED_CPU_GPU_ROUTER_V40_0_4_3",
                               "gpu_queue_ahead": (
                                   gpu_queue_ahead if workers > 1 else 0),
                               "gpu_fold_prep_prefetch_target": (
                                   prep_prefetch_target if workers > 1 else 0),
                               "manifest_parent_queue_ahead_hint":
                                   max(0, int(queue_ahead)),
                               "gpu_queue_depth_target": (
                                   gpu_queue_capacity if workers > 1 else 0),
                               "gpu_queue_capacity": (
                                   gpu_queue_capacity if workers > 1 else 0),
                               "gpu_queue_timeouts": (
                                   gpu_queue_timeouts if workers > 1 else 0),
                               "gpu_queue_depth_peak":
                                   gpu_queue_depth_peak if workers > 1 else [],
                               "gpu_runtime_samples":
                                   (gpu_runtime_samples if workers > 1 else 0),
                               "gpu_fallbacks_to_cpu":
                                   (gpu_fallbacks if workers > 1 else 0),
                               "gpu_ram_scheduler_coupled": False,
                               "gpu_global_section_backlog_aware": True,
                               "gpu_duration_profile_path": str(
                                   shared_compute_root /
                                   "candidate-execution-duration-profiles.json")
                                   if workers > 1 else None,
                               "candidate_ready_frontier": (
                                   ready_frontier_cache.stats()
                                   if workers > 1 else {
                                       "initial_scan_count": 1,
                                       "frontier_refresh_count": 0,
                                       "claim_count": int(processed),
                                       "complete_count": int(processed),
                                       "active_count": 0,
                                       "known_count": int(processed),
                                   })}


def publish_candidate_oos_seed_views(*, source_store: ManifestedJobStore,
                                     source_root: str | Path,
                                     target_stores: Mapping[str, ManifestedJobStore],
                                     target_roots: Mapping[str, str | Path]) -> dict:
    """Complete seed-local Candidate-OOS nodes from one verified physical fit.

    Candidate folds have no seed-specific input.  Publication does not grant
    later evidence: every coverage snapshot still applies its own causal
    information cutoff to ``information_available_at``.
    """
    source_root = Path(source_root)
    source_candidate_root = source_root / "candidate-oos"
    source_artifacts = CandidateOosStore(source_candidate_root)
    completed = source_store.jobs_by_kind_state(kind="candidate_oos_fold", state="COMPLETE")
    counts = {str(name): 0 for name in target_stores}
    for source_job in completed:
        job_id = str(source_job["job_id"])
        source_result = dict(source_job.get("result") or {})
        source_path = Path(str(source_result.get("path", "")))
        try:
            relative = source_path.resolve().relative_to(source_candidate_root.resolve())
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"CANDIDATE_OOS_GLOBAL_SOURCE_PATH_INVALID:{job_id}") from exc
        publish_targets = []
        for name, target_store in target_stores.items():
            target_root = Path(target_roots[name])
            if target_root.resolve() == source_root.resolve():
                continue
            existing = target_store.job(job_id)
            if not existing or existing.get("state") == "COMPLETE":
                continue
            publish_targets.append((name, target_store, target_root))
        # A Fast-Track resume commonly reaches this function after all target
        # views were already published. Do not re-hash every source partition
        # merely to discover that there is no publication work.
        if not publish_targets:
            continue
        source_manifest = source_artifacts.verify_partition(source_path)
        for name, target_store, target_root in publish_targets:
            claimed = target_store.claim_ready(
                f"global-candidate-publisher-{str(name).lower()}",
                lease_seconds=24 * 60 * 60,
                kinds=("candidate_oos_fold",), job_ids=(job_id,))
            if not claimed:
                continue
            target_path = target_root / "candidate-oos" / relative
            try:
                materialize_immutable_tree(source_path.parent, target_path.parent)
                target_manifest = CandidateOosStore(
                    target_root / "candidate-oos").verify_partition(target_path)
                if target_manifest.get("manifest_sha256") != source_manifest.get("manifest_sha256"):
                    raise RuntimeError("CANDIDATE_OOS_GLOBAL_VIEW_MANIFEST_MISMATCH")
                result = dict(target_manifest)
                result.update({"cache": "GLOBAL_SHARED_COMPUTE_HIT", "path": str(target_path),
                               "source_seed": source_root.name.upper(),
                               "shared_artifact_path": str(source_path)})
                model_path = target_path.with_name("model.pkl")
                if model_path.is_file():
                    result["model_path"] = str(model_path)
                target_store.finish_job(
                    job_id, f"global-candidate-publisher-{str(name).lower()}",
                    "COMPLETE", result=result)
                counts[str(name)] += 1
            except Exception as exc:
                target_store.finish_job(
                    job_id, f"global-candidate-publisher-{str(name).lower()}",
                    "FAILED", last_error=f"{type(exc).__name__}:{exc}")
                raise
    for target_store in target_stores.values():
        target_store.materialize_progress()
    return {"source_complete": len(completed), "published_by_seed": counts}


def build_manifested_job_handlers(*, store: ManifestedJobStore, inputs: ManifestedJobInputs,
                              output_root: str | Path,
                              evidence_snapshot_root: str | Path | None = None,
                              recipe_selection_root: str | Path | None = None) -> dict[str, Callable[[dict], Any]]:
    """Return real handlers for the non-portfolio portion of the DAG.

    The handler set intentionally stops before portfolio replay when stock
    execution inputs are absent.  Such a node is marked BLOCKED by the caller,
    never silently converted into a benchmark-only result.
    """
    root = Path(output_root)
    # A resumed checkpoint can contain immutable snapshots produced by an
    # older materialization algorithm.  Keep those historical artifacts
    # untouched and let a corrected run use its own immutable namespace.
    snapshot_root = (Path(evidence_snapshot_root)
                     if evidence_snapshot_root is not None
                     else root / "evidence-snapshots")
    selection_root = (Path(recipe_selection_root)
                      if recipe_selection_root is not None
                      else root / "recipe-selections")
    contract = store.read("manifested-job-contract.json")
    registry = store.read("family-registry.json")
    candidate_store = CandidateOosStore(root / "candidate-oos")
    candidates = {str(x["candidate_id"]): x for x in registry["candidate_registry"]["records"]}
    feature_schema = json.loads(inputs.feature_schema.read_text(encoding="utf-8"))
    resolved = inputs.resolved_contracts(
        primary_candidate_universe_hash=registry["candidate_registry"]["candidate_registry_sha256"])
    fold_policy = resolved.fold_policy
    target_contract = resolved.target_contract
    selection_policy = resolved.recipe_selection_policy
    training_contract = resolved.model_training_contract
    selection_policy_sha = contract["recipe_selection_policy_hash"]

    def model_payload(job: Mapping[str, Any]) -> tuple[dict, dict]:
        horizon = int(str(job["model_family_key"])[1:3])
        family = next(x for x in registry["portfolio_families"] if x["portfolio_family_key"].startswith(f"H{horizon:02d}_D01_N01"))
        selection_job = store.job(job["depends_on"][0])
        if not selection_job or selection_job.get("state") != "COMPLETE":
            raise RuntimeError("MANIFESTED_JOB_RECIPE_SELECTION_RESULT_MISSING")
        return family, dict(selection_job["result"])

    def coverage(job: Mapping[str, Any]) -> dict:
        horizon = int(str(job["model_family_key"])[1:3])
        cutoff = date.fromisoformat(str(job["cutoff"]))
        # A resumed store can contain several immutable cache directories for
        # one candidate/fold (for example an older CPU artifact plus repeated
        # GPU cache hits).  The completed manifested job result is the
        # canonical artifact selected by the DAG; globbing every cache folder
        # would duplicate observations and make valid coverage look corrupt.
        canonical_partition_paths = []
        for candidate_job in store.jobs_by_kind_state(
                kind="candidate_oos_fold", state="COMPLETE"):
            if int(candidate_job.get("horizon", -1)) != horizon:
                continue
            result = candidate_job.get("result") or {}
            path = result.get("path")
            if path:
                canonical_partition_paths.append(str(path))
        snapshot, frame = build_evidence_snapshot(candidate_store, horizon=horizon,
                                                   selection_cutoff=cutoff,
                                                   candidate_registry_sha256=registry["candidate_registry"]["candidate_registry_sha256"],
                                                   partition_paths=canonical_partition_paths)
        expected_keys = set()
        for dependency in job.get("depends_on", []):
            parts = str(dependency).split(":")
            if len(parts) == 4 and parts[0] == "candidate_oos_fold":
                expected_keys.add((int(parts[1]), str(parts[3]), str(parts[2])))
        observed_keys = set()
        duplicate_keys = 0
        if not frame.empty:
            grouped = frame.groupby(["horizon", "candidate_id", "fold_id"], dropna=False).size()
            observed_keys = set((int(h), str(c), str(f)) for (h, c, f) in grouped.index)
            duplicate_keys = int(frame.duplicated(["horizon", "candidate_id", "fold_id", "symbol", "decision_date"]).sum())
        missing_keys = sorted(expected_keys - observed_keys)
        unexpected_keys = sorted(observed_keys - expected_keys)
        consistency = _coverage_consistency(
            expected_keys=expected_keys,
            observed_keys=observed_keys,
            duplicate_keys=duplicate_keys,
            frame_empty=frame.empty,
            candidate_count=snapshot.candidate_count,
            candidate_registry_count=len(candidates),
        )
        diagnostic = {
            "schema_version": "DQBD_COVERAGE_VALIDATION_EVENT_V1",
            "event": "COVERAGE_VALIDATION",
            "severity": "ERROR" if consistency["repaired"] or consistency["fatal"] else "INFO",
            "horizon": horizon,
            "selection_cutoff": cutoff.isoformat(),
            "job_id": str(job["job_id"]),
            "canonical_partition_path_count": len(canonical_partition_paths),
            "validation": consistency,
            "causal_boundary": "information_available_at <= selection_cutoff",
            "action": (
                "USE_OBSERVED_MATURED_SET" if consistency["repaired"]
                else ("FAIL_CLOSED" if consistency["fatal"] else "CONTINUE")),
        }
        diagnostic_path = (
            root / "telemetry" / "coverage-validation"
            / f"H{horizon:02d}-{cutoff.isoformat()}.json")
        _write_json(diagnostic_path, diagnostic)
        if consistency["repaired"]:
            print(
                "[dynamic-qbd][coverage][ERROR] "
                f"repaired stale dependency manifest for H{horizon} "
                f"at {cutoff.isoformat()}: "
                f"expected={len(expected_keys)} observed={len(observed_keys)} "
                f"extra_matured={len(unexpected_keys)}; "
                "using the causally matured observed set",
                file=sys.stderr, flush=True)
        if consistency["fatal"]:
            print(
                "[dynamic-qbd][coverage][ERROR] "
                f"coverage validation failed for H{horizon} "
                f"at {cutoff.isoformat()}: "
                f"{consistency['fatal_reasons']}",
                file=sys.stderr, flush=True)
            raise RuntimeError(f"MANIFESTED_JOB_CANDIDATE_OOS_COVERAGE_INCOMPLETE:H{horizon}")
        persisted = persist_evidence_snapshot(candidate_store, horizon=horizon, selection_cutoff=cutoff,
                                              candidate_registry_sha256=registry["candidate_registry"]["candidate_registry_sha256"],
                                              fold_policy_hash=fold_policy.fold_policy_hash,
                                              signal_panel_hash=contract["signal_panel_development_sha256"],
                                              target_contract_hash=target_contract.target_contract_hash,
                                              output_root=snapshot_root,
                                              snapshot=snapshot, frame=frame)
        resume_invalidate_descendants = (
            persisted.get("immutable_conflict_resolution") is not None)
        return {"evidence_snapshot": asdict(snapshot), "snapshot_path": persisted["snapshot_path"],
                "rows": int(len(frame)),
                "candidate_count": snapshot.candidate_count, "fold_count": snapshot.fold_count,
                "expected_keys": [list(x) for x in sorted(expected_keys)],
                "observed_keys": [list(x) for x in sorted(observed_keys)],
                "missing_keys": [list(x) for x in missing_keys], "unexpected_keys": [list(x) for x in unexpected_keys],
                "duplicate_keys": duplicate_keys,
                "snapshot_cache": persisted.get("cache"),
                "immutable_conflict_resolution": persisted.get(
                    "immutable_conflict_resolution"),
                "_resume_invalidate_descendants": resume_invalidate_descendants,
                "resume_descendants_requeued": 0,
                "coverage_validation": diagnostic,
                "coverage_validation_path": str(diagnostic_path)}

    def selection(job: Mapping[str, Any]) -> dict:
        coverage_job = store.job(job["depends_on"][0])
        if not coverage_job or coverage_job.get("state") != "COMPLETE":
            raise RuntimeError("MANIFESTED_JOB_COVERAGE_RESULT_MISSING")
        horizon = int(str(job["model_family_key"])[1:3])
        coverage_result = coverage_job["result"]
        manifest, frame = read_evidence_snapshot_artifact(coverage_result["snapshot_path"])
        cutoff = date.fromisoformat(str(manifest["selection_cutoff"]))
        snapshot = type("Snapshot", (), {"horizon": horizon, "selection_cutoff": cutoff,
                                         "evidence_snapshot_sha256": manifest["evidence_snapshot_hash"],
                                         "candidate_registry_sha256": manifest["candidate_registry_hash"]})()
        if job.get("recipe_selection_policy_hash") != selection_policy_sha:
            raise RuntimeError("MANIFESTED_JOB_RECIPE_SELECTION_POLICY_HASH_MISMATCH")
        selected = select_recipe_from_snapshot(snapshot, frame, selection_policy_sha256=selection_policy_sha,
                                                fold_policy_hash=manifest["fold_policy_hash"],
                                                target_contract_hash=manifest["target_contract_hash"],
                                                selection_policy=selection_policy)
        candidate = candidates[selected["selected_candidate_id"]]
        selected.update({"recipe_family": candidate["recipe_family"], "hyperparameters": candidate["hyperparameters"],
                         "candidate_spec_hash": candidate["candidate_spec_hash"]})
        # Recipe selections are immutable evidence products.  The old
        # cutoff-only path allowed a repaired/new evidence snapshot to collide
        # with a selection produced from a different snapshot.  That turned a
        # valid resume into a permanent job failure.  Namespace the artifact
        # by the exact snapshot identity so retries remain idempotent while
        # historical selections remain untouched and auditable.
        snapshot_identity = str(manifest["evidence_snapshot_hash"])
        path = (selection_root / f"H{horizon:02d}" / f"{cutoff}"
                / snapshot_identity / "selection.json")
        if path.exists() and json.loads(path.read_text(encoding="utf-8")).get("selected_recipe_sha256") != selected["selected_recipe_sha256"]:
            raise RuntimeError("MANIFESTED_JOB_RECIPE_SELECTION_IMMUTABLE_CONFLICT")
        _write_json(path, selected)
        return selected | {"path": str(path)}

    def model(job: Mapping[str, Any]) -> dict:
        family, selected = model_payload(job)
        horizon = int(family["horizon_sessions"])
        cutoff = date.fromisoformat(str(job["cutoff"]))
        selected = dict(selected)
        selected["score_quantile"] = float(family["entry_policy_rule"].get("score_quantile", .975))
        selected["top_fraction"] = float(family["entry_policy_rule"].get("top_fraction", .005))
        sessions = tuple(sorted(set(parquet_dates(inputs.signal_panel, ("decision_date",)).dt.date)))
        authorized = tuple(x for x in sessions if x <= cutoff)
        calibration_count = int(family["calibration_window_sessions"])
        purge = max(int(family["horizon_sessions"]), int(fold_policy.purge_sessions))
        if len(authorized) < int(family["training_window_sessions"]) + calibration_count + purge:
            raise RuntimeError("MANIFESTED_JOB_PRODUCTION_HISTORY_INSUFFICIENT")
        mature = list(mature_decision_sessions(trading_sessions=authorized,
                                                information_cutoff=cutoff, horizon=horizon))
        if len(mature) < calibration_count + purge + int(family["training_window_sessions"]):
            raise RuntimeError("MANIFESTED_JOB_PRODUCTION_MATURE_HISTORY_INSUFFICIENT")
        cal = mature[-calibration_count:]
        purge_start = len(mature) - calibration_count - purge
        train = mature[max(0, purge_start - int(family["training_window_sessions"])):purge_start]
        if len(cal) != calibration_count:
            raise RuntimeError("MANIFESTED_JOB_CALIBRATION_NOT_EXACT_MATURE_SESSION_COUNT")
        prediction_end = job.get("prediction_end")
        segment_end = date.fromisoformat(str(prediction_end)) if prediction_end else inputs.development_end
        prediction_dates = [str(x) for x in sessions if x > cutoff and x <= segment_end and
                            x <= inputs.development_end and x < inputs.holdout_boundary]
        training_dates = [str(x) for x in train]
        calibration_dates = [str(x) for x in cal]
        fit_identity = {
            "horizon": horizon,
            "selected_recipe_sha256": str(selected.get("selected_recipe_sha256", "")),
            "recipe_family": str(selected.get("recipe_family", "")),
            "hyperparameters": selected.get("hyperparameters"),
            "information_cutoff": cutoff.isoformat(),
            "training_dates_hash": canonical_sha256(training_dates),
            "calibration_dates_hash": canonical_sha256(calibration_dates),
            "prediction_dates_hash": canonical_sha256(prediction_dates),
            "feature_schema_hash": canonical_sha256(feature_schema),
            "signal_panel_development_hash": contract["signal_panel_development_sha256"],
            "target_contract_hash": target_contract.target_contract_hash,
            "model_training_contract_hash": training_contract.model_training_contract_hash,
            "random_state": int(training_contract.random_seed),
            "execution_backend": (
                _execution_backend_for_family(
                    str(selected.get("recipe_family")))
                + (":" + os.environ["DQBD_EXECUTION_BACKEND_FINGERPRINT"]
                   if os.environ.get("DQBD_EXECUTION_BACKEND_FINGERPRINT")
                   else "")),
        }
        shared_root = root.parent / "_shared-compute"
        shared_store_contract(shared_root)
        with compute_key_lock(shared_root, "production-generation", fit_identity) as fit_key:
            shared_fit_root = shared_root / "production-generations" / fit_key
            shared_leaf = shared_fit_root / f"H{horizon:02d}" / cutoff.isoformat()
            local_leaf = root / "production-generations" / f"H{horizon:02d}" / cutoff.isoformat()
            if not (shared_leaf / "generation-manifest.json").is_file():
                local_manifest_path = local_leaf / "generation-manifest.json"
                if local_manifest_path.is_file():
                    prior = json.loads(local_manifest_path.read_text(encoding="utf-8"))
                    compatible = (
                        int(prior.get("horizon", -1)) == horizon
                        and str(prior.get("information_cutoff")) == cutoff.isoformat()
                        and str(prior.get("selected_recipe_sha256", "")) == fit_identity["selected_recipe_sha256"]
                        and list(prior.get("training_dates", ())) == training_dates
                        and list(prior.get("calibration_dates", ())) == calibration_dates
                        and int(prior.get("random_seed", -1)) == int(training_contract.random_seed)
                        and str(prior.get("target_contract_hash", "")) == target_contract.target_contract_hash
                        and str(prior.get("model_training_contract_hash", "")) == training_contract.model_training_contract_hash
                        and str(prior.get("execution_backend", "")) == fit_identity["execution_backend"]
                    )
                    if compatible:
                        materialize_immutable_tree(local_leaf, shared_leaf)
            cache_hit = (shared_leaf / "generation-manifest.json").is_file()
            shared_result = fit_production_generation(
                signal_panel=inputs.signal_panel, feature_schema=feature_schema,
                selected_recipe=selected, horizon=horizon, information_cutoff=cutoff,
                training_dates=training_dates, calibration_dates=calibration_dates,
                prediction_dates=prediction_dates, target_contract=target_contract,
                random_state=training_contract.random_seed,
                model_training_contract_hash=training_contract.model_training_contract_hash,
                output_root=shared_fit_root)
            # The shared tree is the immutable physical authority; the
            # seed-local tree is only a causal visibility view.  A prior
            # interrupted run can leave a complete-looking local directory
            # with an older execution identity.  ``materialize_immutable_tree``
            # intentionally refuses existing destinations, so repair that
            # exact derived view before publishing the already-validated
            # shared generation instead of converting the stale view into a
            # terminal scientific job failure.
            expected_manifest_sha = str(
                shared_result.get("manifest_sha256", ""))
            if local_leaf.is_dir():
                local_manifest_path = local_leaf / "generation-manifest.json"
                local_manifest_sha = None
                if local_manifest_path.is_file():
                    try:
                        local_manifest_sha = str(json.loads(
                            local_manifest_path.read_text(
                                encoding="utf-8")).get("manifest_sha256", ""))
                    except (OSError, ValueError, TypeError):
                        local_manifest_sha = None
                if local_manifest_sha != expected_manifest_sha:
                    shutil.rmtree(local_leaf)
            materialize_immutable_tree(shared_leaf, local_leaf)
            local_manifest_path = local_leaf / "generation-manifest.json"
            local_manifest = json.loads(local_manifest_path.read_text(encoding="utf-8"))
            if local_manifest.get("manifest_sha256") != shared_result.get("manifest_sha256"):
                raise RuntimeError("PRODUCTION_GENERATION_SHARED_VIEW_MANIFEST_MISMATCH")
            return local_manifest | {
                "model_path": str(local_leaf / "model.joblib"),
                "calibration_path": str(local_leaf / "calibration-predictions.parquet"),
                "prediction_path": str(local_leaf / "predictions.parquet")
                if (local_leaf / "predictions.parquet").is_file() else None,
                "manifest_path": str(local_manifest_path),
                "cache": "SHARED_COMPUTE_HIT" if cache_hit else "SHARED_COMPUTE_MISS",
                "shared_compute_key": fit_key,
                "shared_artifact_path": str(shared_leaf),
            }

    def validate_generation_child(job: Mapping[str, Any]) -> dict:
        parent = store.job(job["depends_on"][0])
        if not parent or parent.get("state") != "COMPLETE":
            raise RuntimeError("MANIFESTED_JOB_PRODUCTION_RESULT_MISSING")
        manifest = json.loads(Path(parent["result"]["manifest_path"]).read_text(encoding="utf-8"))
        if sha256_file(parent["result"]["model_path"]) != manifest["model_artifact_sha256"] or sha256_file(parent["result"]["calibration_path"]) != manifest["calibration_sha256"]:
            raise RuntimeError("MANIFESTED_JOB_PRODUCTION_MANIFEST_HASH_MISMATCH")
        prediction_path = parent["result"].get("prediction_path")
        if manifest.get("prediction_sha256") and (not prediction_path or sha256_file(prediction_path) != manifest["prediction_sha256"]):
            raise RuntimeError("MANIFESTED_JOB_PRODUCTION_PREDICTION_HASH_MISMATCH")
        return {"generation_id": manifest["generation_id"], "artifact_manifest": manifest,
                "model_path": parent["result"]["model_path"], "calibration_path": parent["result"]["calibration_path"],
                "prediction_path": parent["result"].get("prediction_path"), "validated_as": job["kind"]}

    def generation_ready(job: Mapping[str, Any]) -> dict:
        children = [store.job(x) for x in job["depends_on"]]
        if any(not x or x.get("state") != "COMPLETE" for x in children):
            raise RuntimeError("MANIFESTED_JOB_GENERATION_CHILD_MISSING")
        ids = {x["result"]["generation_id"] for x in children}
        if len(ids) != 1:
            raise RuntimeError("MANIFESTED_JOB_GENERATION_ARTIFACT_ID_MISMATCH")
        canonical = children[0]["result"]
        for child in children[1:]:
            if child["result"]["artifact_manifest"].get("manifest_sha256") != canonical["artifact_manifest"].get("manifest_sha256"):
                raise RuntimeError("MANIFESTED_JOB_GENERATION_CANONICAL_MANIFEST_MISMATCH")
        return {"generation_id": ids.pop(), "status": "VALID_GENERATION_READY",
                "generation_manifest": canonical["artifact_manifest"],
                "model_path": canonical["model_path"],
                "calibration_path": canonical["calibration_path"],
                "prediction_path": canonical.get("prediction_path")}

    def replay(job: Mapping[str, Any]) -> dict:
        if inputs.stock_execution_prices is None or inputs.stock_distributions is None:
            raise RuntimeError("PORTFOLIO_REPLAY_BLOCKED_STOCK_INPUTS_REQUIRED")
        horizon = int(str(job["portfolio_family_key"])[1:3])
        family = next(x for x in registry["portfolio_families"] if x["portfolio_family_key"] == job["portfolio_family_key"])
        out = root / "portfolio-replay" / family["family_id"] / str(job["cutoff"])
        ready_job_id = str(job["generation_ready_job_id"])
        ready_cache_key = (str(root.resolve()), ready_job_id)
        ready = _REPLAY_GENERATION_READY_CACHE.get(ready_cache_key)
        if ready is None:
            ready_job = store.job(ready_job_id)
            if not ready_job or ready_job.get("state") != "COMPLETE":
                raise RuntimeError("PORTFOLIO_REPLAY_DIRECT_GENERATION_DEPENDENCY_MISSING")
            ready = dict(ready_job["result"])
            _REPLAY_GENERATION_READY_CACHE[ready_cache_key] = ready
            _REPLAY_GENERATION_READY_CACHE.move_to_end(ready_cache_key)
            while len(_REPLAY_GENERATION_READY_CACHE) > _REPLAY_GENERATION_READY_CACHE_SIZE:
                _REPLAY_GENERATION_READY_CACHE.popitem(last=False)
        manifest = ready["generation_manifest"]
        path = ready.get("prediction_path")
        if not path:
            raise RuntimeError("PORTFOLIO_REPLAY_PREDICTION_ARTIFACT_MISSING")
        signals = _cached_replay_signals(path)
        schedule = pd.DataFrame([{"activation_date": manifest["information_cutoff"], "family_id": family["family_id"],
                                  "generation_id": manifest["generation_id"], "resolved_threshold": float(manifest.get("resolved_threshold", 0.0)),
                                  "entry_policy_id": "ENTRY_" + manifest["generation_id"], "exit_policy_id": "EXIT_FIXED",
                                  "model_artifact_id": manifest["generation_id"]}])
        policy = Policy(horizon, float(family["entry_policy_rule"].get("score_quantile", .975)),
                        float(family["entry_policy_rule"].get("top_fraction", .005)), int(family["max_names"]),
                        int(family["holding_days"]), "FIXED", 0.0, "IGNORE_NEW", "EQUAL_ACTIVE", .50)
        prices, distributions = read_development_replay_inputs(
            prices_path=inputs.stock_execution_prices, distributions_path=inputs.stock_distributions,
            development_end=inputs.development_end, holdout_boundary=inputs.holdout_boundary,
            replay_start=job.get("segment_start"), replay_end=job.get("segment_end"))
        previous = None
        depends = [x for x in job.get("depends_on", ()) if str(x).startswith("replay:")]
        if depends:
            prior = store.job(depends[-1])
            if prior and prior.get("state") == "COMPLETE":
                previous = prior["result"].get("replay_state")
                if previous is not None and prior["result"].get("state_hash") != stable_hash(previous):
                    raise RuntimeError("MANIFESTED_JOB_REPLAY_PREVIOUS_STATE_HASH_MISMATCH")
        if out.exists():
            existing_manifest_path = out / "replay-manifest.json"
            if not existing_manifest_path.is_file():
                shutil.rmtree(out)
            else:
                existing = json.loads(existing_manifest_path.read_text(encoding="utf-8"))
                existing_hash = existing.pop("artifact_hash", None)
                if existing_hash != stable_hash(existing):
                    raise RuntimeError("MANIFESTED_JOB_REPLAY_CACHE_MANIFEST_SELF_HASH_MISMATCH")
                expected = {
                    "family_id": family["family_id"],
                    "segment_start": str(job.get("segment_start")),
                    "segment_end": str(job.get("segment_end")),
                    "generation_id": manifest["generation_id"],
                    "previous_state_hash": stable_hash(previous) if previous is not None else None,
                    "generation_manifest_sha256": manifest.get("manifest_sha256"),
                    "run_contract_hash": contract.get("run_contract_hash"),
                    "fold_policy_hash": fold_policy.fold_policy_hash,
                    "target_contract_hash": target_contract.target_contract_hash,
                    "cost_contract_hash": stable_hash(family["cost_contract"]),
                    "tax_contract_hash": stable_hash(family["tax_contract"]),
                }
                if any(existing.get(key) != value for key, value in expected.items()):
                    raise RuntimeError("MANIFESTED_JOB_REPLAY_CACHE_CONTRACT_MISMATCH")
                for key in ("nav_path", "trades_path", "state_path"):
                    if not Path(existing[key]).is_file():
                        raise RuntimeError("MANIFESTED_JOB_REPLAY_CACHE_FILE_MISSING")
                if (sha256_file(existing["nav_path"]) != existing.get("nav_sha256") or
                        sha256_file(existing["trades_path"]) != existing.get("trades_sha256") or
                        sha256_file(existing["state_path"]) != existing.get("state_sha256")):
                    raise RuntimeError("MANIFESTED_JOB_REPLAY_CACHE_FILE_HASH_MISMATCH")
                state_value = json.loads(Path(existing["state_path"]).read_text(encoding="utf-8"))
                if existing.get("terminal_state_hash") != stable_hash(state_value) or existing.get("state_hash") != stable_hash(state_value):
                    raise RuntimeError("MANIFESTED_JOB_REPLAY_CACHE_TERMINAL_STATE_HASH_MISMATCH")
        result = replay_family(signals=signals, prices=prices, policy=policy, generation_schedule=schedule,
                               start=job.get("segment_start"), end=job.get("segment_end"), resume_state=previous,
                               cost=CostModel(float(family["cost_contract"].get("roundtrip_bps", 20.0))),
                               tax=tax_config_from_contract(family["tax_contract"]), distributions=distributions)
        curve = result["curve"].copy(); curve["family_id"] = family["family_id"]
        if out.exists() and not (out / "replay-manifest.json").is_file():
            shutil.rmtree(out)
        temp_out = out.parent / f".{out.name}.tmp.{os.getpid()}"
        if temp_out.exists():
            shutil.rmtree(temp_out)
        temp_out.mkdir(parents=True, exist_ok=False)
        temp_out.joinpath("nav.parquet")
        curve.to_parquet(temp_out / "nav.parquet", index=False)
        trades = pd.DataFrame(result.get("trades", []))
        trades.to_json(temp_out / "trades.json", orient="records", date_format="iso")
        state = result["replay_state"]
        _write_json(temp_out / "replay-state.json", state)
        artifact = {"schema_version": "QBD_REPLAY_SEGMENT_V2", "family_id": family["family_id"],
                    "segment_start": str(job.get("segment_start")), "segment_end": str(job.get("segment_end")),
                    "generation_id": manifest["generation_id"], "previous_state_hash": stable_hash(previous) if previous is not None else None,
                    "nav_path": str(out / "nav.parquet"), "trades_path": str(out / "trades.json"),
                    "state_path": str(out / "replay-state.json"),
                    "terminal_state_hash": stable_hash(state), "nav_sha256": sha256_file(temp_out / "nav.parquet"),
                    "trades_sha256": sha256_file(temp_out / "trades.json"), "state_sha256": sha256_file(temp_out / "replay-state.json"),
                    "state_hash": stable_hash(state), "generation_manifest_sha256": manifest.get("manifest_sha256"),
                    "run_contract_hash": contract.get("run_contract_hash"),
                    "fold_policy_hash": fold_policy.fold_policy_hash,
                    "target_contract_hash": target_contract.target_contract_hash,
                    "cost_contract_hash": stable_hash(family["cost_contract"]), "tax_contract_hash": stable_hash(family["tax_contract"])}
        artifact["artifact_hash"] = stable_hash(artifact)
        _write_json(temp_out / "replay-manifest.json", artifact)
        if out.exists():
            prior = json.loads((out / "replay-manifest.json").read_text(encoding="utf-8"))
            if prior.get("artifact_hash") != artifact["artifact_hash"]:
                raise RuntimeError("MANIFESTED_JOB_REPLAY_IMMUTABLE_ARTIFACT_CONFLICT")
            shutil.rmtree(temp_out)
        else:
            os.replace(temp_out, out)
        return {"metrics": result.get("metrics", {}), "nav_path": str(out / "nav.parquet"),
                "trades_path": str(out / "trades.json"), "replay_state": state,
                "state_hash": artifact["state_hash"], "replay_manifest_path": str(out / "replay-manifest.json"),
                "generation_ids_used": [str(manifest["generation_id"])], "generation_manifest": manifest}

    def evidence(job: Mapping[str, Any]) -> dict:
        replay_job = store.job(job["depends_on"][0])
        if not replay_job or replay_job.get("state") != "COMPLETE":
            raise RuntimeError("MANIFESTED_JOB_EVIDENCE_REPLAY_RESULT_MISSING")
        result = replay_job["result"]
        replay_manifest_path = Path(result["replay_manifest_path"])
        replay_manifest = json.loads(replay_manifest_path.read_text(encoding="utf-8"))
        artifact_hash = replay_manifest.pop("artifact_hash", None)
        if artifact_hash != stable_hash(replay_manifest):
            raise RuntimeError("MANIFESTED_JOB_EVIDENCE_REPLAY_MANIFEST_HASH_MISMATCH")
        if replay_manifest.get("family_id") != next(x["family_id"] for x in registry["portfolio_families"] if x["portfolio_family_key"] == job["portfolio_family_key"]):
            raise RuntimeError("MANIFESTED_JOB_EVIDENCE_REPLAY_FAMILY_MISMATCH")
        if sha256_file(result["nav_path"]) != replay_manifest.get("nav_sha256") or sha256_file(result["trades_path"]) != replay_manifest.get("trades_sha256"):
            raise RuntimeError("MANIFESTED_JOB_EVIDENCE_REPLAY_FILE_HASH_MISMATCH")
        state_path = Path(replay_manifest["state_path"])
        if sha256_file(state_path) != replay_manifest.get("state_sha256"):
            raise RuntimeError("MANIFESTED_JOB_EVIDENCE_REPLAY_STATE_HASH_MISMATCH")
        curve = pd.read_parquet(result["nav_path"])
        trades = pd.read_json(result["trades_path"])
        schedule = pd.DataFrame([{"activation_date": str(job["cutoff"]),
                                  "family_id": str(curve["family_id"].iloc[0]),
                                  "generation_id": result["generation_ids_used"][0],
                                  "resolved_threshold": float(result["generation_manifest"].get("resolved_threshold", 0.0))}])
        monthly = build_monthly_family_evidence(curve, schedule, trades=trades)
        out = root / "portfolio-evidence" / str(job["portfolio_family_key"]) / str(job["cutoff"])
        existing_manifest_path = out / "evidence-manifest.json"
        if existing_manifest_path.is_file():
            prior = json.loads(existing_manifest_path.read_text(encoding="utf-8"))
            prior_hash = prior.pop("manifest_sha256", None)
            if prior_hash != stable_hash(prior):
                raise RuntimeError("MANIFESTED_JOB_EVIDENCE_CACHE_MANIFEST_SELF_HASH_MISMATCH")
            evidence_file = out / "monthly-family-evidence.parquet"
            if not evidence_file.is_file() or sha256_file(evidence_file) != prior.get("evidence_sha256"):
                raise RuntimeError("MANIFESTED_JOB_EVIDENCE_CACHE_FILE_HASH_MISMATCH")
        if out.exists() and not existing_manifest_path.is_file():
            shutil.rmtree(out)
        temp_out = out.parent / f".{out.name}.tmp.{os.getpid()}"
        if temp_out.exists():
            shutil.rmtree(temp_out)
        temp_out.mkdir(parents=True, exist_ok=False)
        evidence_path = temp_out / "monthly-family-evidence.parquet"
        monthly.to_parquet(evidence_path, index=False)
        manifest = {"schema_version": "QBD_MONTHLY_EVIDENCE_V1", "replay_manifest_hash": stable_hash(replay_manifest),
                    "evidence_sha256": sha256_file(evidence_path), "family_key": job["portfolio_family_key"],
                    "assessment_date": str(job["cutoff"]), "generation_id": result["generation_ids_used"][0],
                    "target_contract_hash": target_contract.target_contract_hash,
                    "run_contract_hash": contract.get("run_contract_hash"),
                    "fold_policy_hash": fold_policy.fold_policy_hash,
                    "recipe_selection_policy_hash": selection_policy.recipe_selection_policy_hash}
        manifest["manifest_sha256"] = stable_hash(manifest)
        _write_json(temp_out / "evidence-manifest.json", manifest)
        if out.exists():
            prior = json.loads(existing_manifest_path.read_text(encoding="utf-8"))
            if prior.get("manifest_sha256") != manifest["manifest_sha256"]:
                shutil.rmtree(temp_out)
                raise RuntimeError("MANIFESTED_JOB_EVIDENCE_IMMUTABLE_ARTIFACT_CONFLICT")
            shutil.rmtree(temp_out)
        else:
            os.replace(temp_out, out)
        return {"evidence_path": str(out / "monthly-family-evidence.parquet"), "manifest_path": str(out / "evidence-manifest.json"),
                "manifest_sha256": manifest["manifest_sha256"]}

    return {"candidate_evidence_coverage": coverage, "recipe_selection": selection,
            "model": model, "calibration": validate_generation_child,
            "prediction": validate_generation_child, "generation_ready": generation_ready,
            "replay": replay, "evidence": evidence}


def execute_manifested_development_jobs(*, store: ManifestedJobStore, inputs: ManifestedJobInputs,
                               output_root: str | Path, candidate_jobs: int | None = None,
                               causal_jobs: int | None = None, workers: int = 1,
                               queue_ahead: int = 2,
                               worker_map_override: list[dict] | None = None,
                               ram_scheduler: RamAdmissionScheduler | None = None,
                               evidence_snapshot_root: str | Path | None = None,
                               recipe_selection_root: str | Path | None = None,
                               activity_heartbeat_path: str | Path | None = None,
                               hot_reload_controller: HotReloadController | None = None,
                               causal_job_budget: int | None = None) -> dict:
    """Run the generated causal DAG in resumable phases.

    Phase 1 materializes Candidate-OOS evidence. Phase 2 uses one
    dependency-ready mixed frontier for model/generation/replay/evidence work.
    The DAG dependencies remain authoritative, so small ready Replay/Evidence
    nodes may fill RAM headroom without opening later causal information.
    Portfolio replay remains explicitly blocked when its stock-price/
    distribution inputs are unavailable.
    """
    summary_path = Path(output_root) / "summary.json"
    if summary_path.is_file() and not inputs.allow_dirty_development_fixture:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if "DIRTY_WORKTREE" in str(summary.get("status", "")):
            raise RuntimeError("MANIFESTED_JOB_DIRTY_WORKTREE_EXECUTION_BLOCKED")
    handlers = build_manifested_job_handlers(
        store=store, inputs=inputs, output_root=output_root,
        evidence_snapshot_root=evidence_snapshot_root,
        recipe_selection_root=recipe_selection_root)
    candidate = {}
    causal = {}
    causal_processed = 0
    # Advance one chronological cutoff at a time. Independent horizons at that
    # same causal cutoff may fit concurrently; later cutoffs remain closed.
    while True:
        pending_coverage = store.jobs_by_kind_state(
            kind="candidate_evidence_coverage", state="PENDING")
        pending_coverage.sort(key=lambda x: (str(x.get("cutoff", "")), str(x.get("job_id", ""))))
        allowed = None
        if pending_coverage:
            frontier_cutoff = str(pending_coverage[0].get("cutoff", ""))
            frontier = [job for job in pending_coverage
                        if str(job.get("cutoff", "")) == frontier_cutoff]
            allowed_ids: set[str] = set()
            for coverage_job in frontier:
                dependencies = tuple(str(x) for x in coverage_job.get("depends_on", ()))
                candidate_dependencies = [x for x in dependencies
                                          if x.startswith("candidate_oos_fold:")]
                if candidate_dependencies:
                    allowed_ids.update(candidate_dependencies)
                    continue
                # A horizon with no matured fold dependency must wait for its
                # own horizon-scoped Candidate-OOS artifacts, never borrow
                # another horizon's later information.
                marker = str(coverage_job.get("job_id", ""))
                horizon_text = next((x[1:3] for x in marker.split(":")
                                     if x.startswith("H") and len(x) >= 3), None)
                if horizon_text and horizon_text.isdigit():
                    allowed_ids.update(store.job_ids_by_kind_prefix(
                        kind="candidate_oos_fold",
                        prefix=f"candidate_oos_fold:{int(horizon_text)}:"))
            allowed = sorted(allowed_ids)
        before = store.progress().copy()
        try:
            candidate = execute_candidate_oos_jobs(
                store=store, inputs=inputs, output_root=output_root,
                owner=owner, max_jobs=candidate_jobs, allowed_job_ids=allowed,
                workers=workers, queue_ahead=queue_ahead,
                max_inflight=min(
                    workers, int(os.environ.get(
                        "DQBD_CANDIDATE_MAX_INFLIGHT", "26"))),
                worker_map_override=worker_map_override,
                ram_scheduler=ram_scheduler,
                estimated_task_gib=4.5,
                hot_reload_controller=hot_reload_controller)
            remaining_causal_budget = (
                None if causal_job_budget is None else max(
                    0, int(causal_job_budget) - causal_processed))
            if remaining_causal_budget == 0:
                break
            mixed = execute_ready_jobs(
                store, handlers,
                max_jobs=(
                    remaining_causal_budget
                    if causal_jobs is None else min(
                        int(causal_jobs), remaining_causal_budget)),
                kinds=(
                    "candidate_evidence_coverage", "recipe_selection", "model",
                    "calibration", "prediction", "generation_ready",
                    "replay", "evidence",
                ),
                owner=owner, workers=workers, queue_ahead=queue_ahead, inputs=inputs,
                max_inflight=min(
                    workers,
                    int(os.environ.get(
                        "DQBD_CAUSAL_MAX_INFLIGHT", str(workers)))),
                worker_map_override=worker_map_override,
                ram_scheduler=ram_scheduler, estimated_task_gib=1.0,
                evidence_snapshot_root=evidence_snapshot_root,
                recipe_selection_root=recipe_selection_root,
                activity_heartbeat_path=activity_heartbeat_path,
                hot_reload_controller=hot_reload_controller)
            causal_processed += int(mixed.get("processed_this_call", 0))
        except _HotReloadRequested:
            # Only claimed/in-flight work is requeued.  COMPLETE rows and all
            # immutable artifacts stay untouched, so source changes never
            # erase progress or force a full graph initialization.
            store.requeue_interrupted_jobs(owner=owner)
            if hot_reload_controller is None:
                raise
            hot_reload_controller.reload_coordinator()
            handlers = build_manifested_job_handlers(
                store=store, inputs=inputs, output_root=output_root,
                evidence_snapshot_root=evidence_snapshot_root,
                recipe_selection_root=recipe_selection_root)
            continue
        causal = {
            "mixed_frontier": mixed,
            "heavy_and_light_merged": True,
            "ram_targeted_reclaims_observed": (
                ram_scheduler.telemetry().get(
                    "ram_scheduler_targeted_reclaim_count", 0)
                if ram_scheduler is not None else 0),
            "progress": store.progress(),
            "processed_this_call": causal_processed,
            "causal_job_budget": causal_job_budget,
        }
        after = store.progress()
        if (causal_job_budget is not None
                and causal_processed >= int(causal_job_budget)):
            break
        if after.get("COMPLETE", 0) == before.get("COMPLETE", 0) and after.get("FAILED", 0) == before.get("FAILED", 0):
            break
        if not pending_coverage and after.get("PENDING", 0) == 0:
            break
    return {"candidate_oos": candidate, "causal": causal,
            "ram_targeted_reclaims_observed": (
                ram_scheduler.telemetry().get(
                    "ram_scheduler_targeted_reclaim_count", 0)
                if ram_scheduler is not None else 0),
            "progress": store.progress(),
            "portfolio_replay_authority": "BLOCKED_UNTIL_STOCK_INPUTS" if inputs.stock_execution_prices is None or inputs.stock_distributions is None else "PRODUCTIVE_DAG_HANDLER"}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--signal-panel", required=True)
    parser.add_argument("--candidate-metrics")
    parser.add_argument("--feature-schema", required=True)
    parser.add_argument("--benchmark-prices", required=True)
    parser.add_argument("--benchmark-distributions", required=True)
    parser.add_argument("--stock-execution-prices")
    parser.add_argument("--stock-price-quality-manifest")
    parser.add_argument("--stock-direct-daily-manifest")
    parser.add_argument("--stock-distributions")
    parser.add_argument("--ignore-stock-inputs", action="store_true")
    parser.add_argument("--execute-candidate-oos", action="store_true")
    parser.add_argument("--execute-causal-jobs", action="store_true")
    parser.add_argument("--hyperparameter-space")
    parser.add_argument("--candidate-oos-evidence")
    parser.add_argument("--output-root", default="artifacts/dynamic-qbd-manifested-job-coordinator-v1")
    parser.add_argument("--start", default="2016-01-01")
    parser.add_argument("--end", default="2026-07-24")
    args = parser.parse_args(argv)
    result = initialize_manifested_job_coordinator(
        inputs=ManifestedJobInputs(repo_root=args.repo_root, signal_panel=args.signal_panel,
                             candidate_metrics=args.candidate_metrics, feature_schema=args.feature_schema,
                             benchmark_prices=args.benchmark_prices, benchmark_distributions=args.benchmark_distributions,
                             stock_execution_prices=args.stock_execution_prices,
                             stock_price_quality_manifest=args.stock_price_quality_manifest,
                             stock_direct_daily_manifest=args.stock_direct_daily_manifest,
                             stock_distributions=args.stock_distributions,
                             allow_missing_stock_inputs=args.ignore_stock_inputs,
                             hyperparameter_space=args.hyperparameter_space,
                             development_start=date.fromisoformat(args.start), development_end=date.fromisoformat(args.end)),
        output_root=args.output_root, candidate_oos_evidence=args.candidate_oos_evidence)
    if args.execute_candidate_oos:
        result["candidate_oos_execution"] = execute_candidate_oos_jobs(
            store=ManifestedJobStore(args.output_root),
            inputs=ManifestedJobInputs(repo_root=args.repo_root, signal_panel=args.signal_panel,
                                 candidate_metrics=args.candidate_metrics, feature_schema=args.feature_schema,
                                 benchmark_prices=args.benchmark_prices, benchmark_distributions=args.benchmark_distributions,
                                 stock_execution_prices=args.stock_execution_prices,
                                 stock_price_quality_manifest=args.stock_price_quality_manifest,
                                 stock_direct_daily_manifest=args.stock_direct_daily_manifest,
                                 stock_distributions=args.stock_distributions,
                                 allow_missing_stock_inputs=args.ignore_stock_inputs,
                                 hyperparameter_space=args.hyperparameter_space if hasattr(args, "hyperparameter_space") else None,
                                 development_start=date.fromisoformat(args.start), development_end=date.fromisoformat(args.end)),
            output_root=args.output_root)
    if args.execute_causal_jobs:
        causal_store = ManifestedJobStore(args.output_root)
        result["causal_execution"] = execute_ready_jobs(
            causal_store, build_manifested_job_handlers(
                store=causal_store,
                inputs=ManifestedJobInputs(repo_root=args.repo_root, signal_panel=args.signal_panel,
                                     candidate_metrics=args.candidate_metrics, feature_schema=args.feature_schema,
                                     benchmark_prices=args.benchmark_prices, benchmark_distributions=args.benchmark_distributions,
                                     stock_execution_prices=args.stock_execution_prices,
                                     stock_price_quality_manifest=args.stock_price_quality_manifest,
                                     stock_direct_daily_manifest=args.stock_direct_daily_manifest,
                                     stock_distributions=args.stock_distributions,
                                     allow_missing_stock_inputs=args.ignore_stock_inputs,
                                     hyperparameter_space=args.hyperparameter_space,
                                     development_start=date.fromisoformat(args.start), development_end=date.fromisoformat(args.end)),
                output_root=args.output_root),
            max_jobs=None)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
