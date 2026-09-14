"""Step-9 coordinator for the fixed causal Model-Store Development replay.

This module is intentionally a coordinator. Model fitting, generation
identity, evidence maturity and portfolio accounting remain owned by the
existing production pipeline and its stores. It never opens evaluation until
all three seed materializations have published a completion marker.
"""
from __future__ import annotations

import argparse
import atexit
import csv
from concurrent.futures import ThreadPoolExecutor, wait
import json
import os
import threading
import time
from datetime import date
from pathlib import Path
from typing import Iterable, Mapping

from .contract_fingerprints import stable_hash
from .dynamic_qbd_manifested_job_coordinator import (
    BoundaryStopRequested,
    ManifestedJobInputs, ManifestedJobStore,
    initialize_manifested_job_coordinator, RamAdmissionScheduler,
    execute_candidate_oos_jobs,
    publish_candidate_oos_seed_views,
    current_git_sha, input_stat_fingerprint,
    HotReloadController,
    MAX_RUNTIME_PROCESS_LANES,
)
from .dynamic_qbd_batched_portfolio_runtime import (
    PORTFOLIO_BATCH_POLICY,
    PORTFOLIO_BATCH_SCHEMA,
    execute_manifested_development_jobs,
)
from .dynamic_qbd_complete_result_validation import (
    COMPLETE_RESULT_VALIDATION_SCHEMA,
    audit_complete_results,
    complete_result_validation_authority_sha256,
)
from .development_slice_hash import (
    cleanup_legacy_slice_hash_scratch,
    cleanup_orphaned_slice_hash_scratch,
)
from .dynamic_qbd_orchestrator_contract import default_pseudolive_experiment_contract
from .dynamic_qbd_run_gate_contract import RunGateContract
from .dynamic_qbd_runtime_resources import configure_cpu_peak
from .dynamic_qbd_gpu_pretraining import (
    CPU_EXECUTION_BACKEND, GPU_EXECUTION_BACKEND, GPU_HGB_DEVICE_POLICY, GPU_HGB_KERNEL_POLICY,
    GPU_HGB_EXECUTION_BACKEND,
    configure_gpu_pretraining,
)

SEEDS = {
    "SHORT": date(2020, 8, 31),
    "PRIMARY": date(2021, 2, 26),
    "LONG": date(2021, 8, 31),
}
SEED_WAVE_CAUSAL_JOB_BUDGET = 64


def _next_ready_seed_wave(*, seed_names: Iterable[str],
                          remaining: set[str],
                          ready_by_seed: Mapping[str, int],
                          next_index: int) -> tuple[str | None, int]:
    """Pick the next ready bank in deterministic round-robin order.

    A wave deliberately owns the complete CPU/GPU pool for one bank, so the
    selected bank can use 32 CPU lanes plus the two physical GPU lanes.  The
    bounded wave budget is what prevents that full-pool execution from
    starving the other seed banks indefinitely.
    """
    names = tuple(str(name) for name in seed_names)
    if not names:
        return None, 0
    start = int(next_index) % len(names)
    for offset in range(len(names)):
        index = (start + offset) % len(names)
        name = names[index]
        if name in remaining and int(ready_by_seed.get(name, 0)) > 0:
            return name, (index + 1) % len(names)
    return None, start


def _allocate_seed_lanes(
    *, worker_map: list[dict], worker_count: int,
    ready_by_seed: Mapping[str, int],
    seed_names: Iterable[str] = SEEDS,
) -> tuple[dict[str, int], dict[str, tuple[int, list[dict]]]]:
    """Allocate lanes only to banks with dependency-ready work.

    The allocation is a scheduler concern, not a scientific choice. A bank
    with no READY frontier receives no process lane in this wave; its unused
    capacity is divided among the active banks. A later wave recomputes the
    allocation after a bank completes, so idle seed capacity is never parked.
    """
    names = tuple(str(name) for name in seed_names)
    active = [name for name in names if int(ready_by_seed.get(name, 0)) > 0]
    if not active:
        return ({name: 0 for name in names},
                {name: (0, []) for name in names})
    base, remainder = divmod(int(worker_count), len(active))
    counts = {name: 0 for name in names}
    for index, name in enumerate(active):
        counts[name] = base + (1 if index < remainder else 0)
    allocations: dict[str, tuple[int, list[dict]]] = {}
    cursor = 0
    for name in names:
        count = counts[name]
        allocations[name] = (count, worker_map[cursor:cursor + count])
        cursor += count
    return counts, allocations
HOLDOUT = date(2026, 7, 25)
# Distinct exit code for "stopped on purpose at a scheduler boundary so a new
# code snapshot can take over".  The supervisor treats it as a normal swap,
# not as a crash.
BOUNDARY_STOP_EXIT_CODE = 75
REQUIRED_COMPACT_FILES = (
    "summary.json", "REPORT.md", "run-contract.json", "contract-audit.json",
    "initial-model-store.csv", "generation-ledger.csv", "evidence-event-ledger.csv",
    "orchestrator-decisions.csv", "arm-comparison.csv", "event-subperiod-summary.csv",
    "oracle-regret-summary.csv", "seed-sensitivity.csv", "runtime-telemetry.json",
)
# The orphan-lane watchdog is an execution-only recovery change.  This
# explicit migration anchor permits already validated v3 initialization
# caches from the immediately preceding scheduler commit to be reused after
# that fix is committed; scientific inputs, graph identity and validation
# authority still have to match byte-for-byte below.
RESUME_COMPATIBLE_RUNTIME_BASE_SHAS = frozenset({
    "816a0ae6178182a672c5df825a9c6ff8ce9211d4",
    "49b138737a05e544394554b266251b1a10be8e2c",
    "1b7cccda284c4723090ced634823b2255041993c",
    # Snapshot-isolated hot swap: the in-process module reload was removed in
    # favour of a boundary stop plus a restart from an immutable code
    # snapshot. Execution runtime only; scientific inputs, graph identity and
    # validation authority are unchanged and still compared byte-for-byte.
    "2992720bcf2a5761133e8604bb5fdf11304d83e8",
})


def _json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    os.replace(temporary, path)


class _SeedBankWatchdog:
    """Detect a stalled seed bank without depending on its worker threads.

    A seed owns a disjoint job store and worker slice, but the three seed
    calls run in threads inside one coordinator. A dead/stuck bank previously
    left RUNNING rows alive until their lease expired while the parent waited
    on the first seed future. This watchdog observes only owner-scoped SQL
    aggregates and records a fail-closed signal for the coordinator loop.
    """

    def __init__(self, *, stores: dict[str, ManifestedJobStore], owners: dict[str, str],
                 telemetry_path: Path, activity_paths: dict[str, Path] | None = None,
                 stale_after_seconds: float = 120.0,
                 interval_seconds: float = 10.0) -> None:
        self.stores = dict(stores)
        self.owners = dict(owners)
        self.activity_paths = {
            str(name): Path(path)
            for name, path in (activity_paths or {}).items()
        }
        self.telemetry_path = Path(telemetry_path)
        self.stale_after_seconds = max(30.0, float(stale_after_seconds))
        self.interval_seconds = max(1.0, float(interval_seconds))
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.failure: RuntimeError | None = None
        self._lock = threading.Lock()
        self.recovery_counts: dict[str, int] = {}
        self._last_recovery_epoch: dict[str, float] = {}
        self._threads: list[threading.Thread] = []

    def _record(self, payload: dict) -> None:
        row = {"timestamp_epoch": time.time(), **payload}
        try:
            self.telemetry_path.parent.mkdir(parents=True, exist_ok=True)
            with self.telemetry_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")
        except OSError:
            pass

    def _recover_bank(self, *, seed_name: str, store: ManifestedJobStore,
                      owner: str, event: str, **facts: object) -> bool:
        """Recover one bank without allowing a transient store error to kill
        the watchdog thread.

        Recovery is deliberately retried on a later tick when the write
        cannot be completed.  The recovery timestamp is advanced only after
        a successful store operation, so a failed SQLite write cannot strand
        a bank behind the watchdog's cooldown window.
        """
        try:
            recovered = store.requeue_interrupted_jobs(owner=owner)
        except BaseException as exc:
            self._record({
                "event": "SEED_BANK_WATCHDOG_RECOVERY_ERROR",
                "seed_name": seed_name,
                "owner": owner,
                "error_type": type(exc).__name__,
                "error": str(exc),
                **facts,
            })
            return False
        now = time.time()
        with self._lock:
            self._last_recovery_epoch[seed_name] = now
            self.recovery_counts[seed_name] = (
                self.recovery_counts.get(seed_name, 0) + recovered)
            recovery_count = self.recovery_counts[seed_name]
        self._record({
            "event": event,
            "seed_name": seed_name,
            "owner": owner,
            "requeued_jobs": recovered,
            "recovery_count": recovery_count,
            **facts,
        })
        return True

    def _tick(self, seed_names: Iterable[str] | None = None) -> None:
        """Check one or all banks without coupling their liveness loops.

        ``_tick()`` remains a deterministic all-bank helper for tests and
        diagnostics.  The daemon uses one loop per bank so a slow SQLite
        operation or an injected I/O fault in one seed cannot prevent the
        other seeds from being recovered.
        """
        now = time.time()
        names = tuple(seed_names) if seed_names is not None else tuple(self.stores)
        for seed_name in names:
            store = self.stores[seed_name]
            owner = self.owners[seed_name]
            try:
                stats = store.running_job_heartbeat_stats(owner)
            except BaseException as exc:
                self._record({"event": "SEED_BANK_WATCHDOG_READ_ERROR",
                              "seed_name": seed_name,
                              "error_type": type(exc).__name__, "error": str(exc)})
                continue
            max_heartbeat = stats.get("max_heartbeat")
            max_lease_until = stats.get("max_lease_until")
            if (
                not stats.get("running_count")
                or max_heartbeat is None
                or max_lease_until is None
            ):
                # A bank can stall before it claims a job. Only treat this as
                # starvation when dependency-ready work exists and the bank's
                # own activity heartbeat has gone stale. An empty frontier is
                # normal and remains untouched.
                activity_path = self.activity_paths.get(seed_name)
                if activity_path is None or not activity_path.is_file():
                    continue
                try:
                    activity = json.loads(
                        activity_path.read_text(encoding="utf-8"))
                    if str(activity.get("owner", "")) != str(owner):
                        # A checkpoint can contain the previous run's
                        # heartbeat while this run is still initializing.
                        # It must never trigger recovery for the new owner.
                        continue
                    activity_age = now - float(
                        activity.get("updated_at_epoch", 0.0))
                except (OSError, ValueError, TypeError, json.JSONDecodeError):
                    continue
                if (
                    activity.get("state") == "COMPLETE"
                    or activity_age <= self.stale_after_seconds
                    or store.ready_job_count() <= 0
                ):
                    continue
                with self._lock:
                    last_recovery = self._last_recovery_epoch.get(
                        seed_name, 0.0)
                if now - last_recovery < self.stale_after_seconds:
                    continue
                self._recover_bank(
                    seed_name=seed_name, store=store, owner=owner,
                    event="SEED_BANK_QUEUE_STARVATION_SELF_HEALED",
                    activity_age_seconds=activity_age,
                    ready_job_count=store.ready_job_count(),
                )
                continue
            age = now - float(max_heartbeat)
            # A valid lease alone is not proof that the bank is making
            # progress.  The coordinator renews its bank activity heartbeat
            # independently of individual worker jobs.  If both signals are
            # stale while dependency-ready work exists, recover immediately;
            # otherwise a blocked dispatcher could strand a bank until the
            # much longer worker lease expires.
            activity_age = None
            activity_path = self.activity_paths.get(seed_name)
            if activity_path is not None and activity_path.is_file():
                try:
                    activity = json.loads(
                        activity_path.read_text(encoding="utf-8"))
                    if str(activity.get("owner", "")) != str(owner):
                        activity_age = None
                    else:
                        activity_age = now - float(
                            activity.get("updated_at_epoch", 0.0))
                except (OSError, ValueError, TypeError, json.JSONDecodeError):
                    activity_age = None
            if (
                activity_age is not None
                and activity_age > self.stale_after_seconds
                and store.ready_job_count() > 0
            ):
                self._recover_bank(
                    seed_name=seed_name, store=store, owner=owner,
                    event="SEED_BANK_ACTIVITY_STALE_SELF_HEALED",
                    heartbeat_age_seconds=age,
                    activity_age_seconds=activity_age,
                    running_count=stats.get("running_count"),
                    max_lease_until=max_lease_until,
                )
                continue
            # A long model fit may temporarily prevent the coordinator's
            # renewal write from becoming visible. Heartbeat age alone is not
            # proof of death while its lease is still valid. Fail closed only
            # after the complete owner bank has lost its leases as well.
            if (
                age <= self.stale_after_seconds
                or float(max_lease_until) > now
            ):
                continue
            # Recover only this bank. A prior implementation converted the
            # observation into a process-wide failure, leaving all three
            # independent seed lanes down until a manual restart. Expired
            # leases are safe to reclaim because the owner-scoped SQL update
            # cannot touch another bank and completed rows are excluded.
            with self._lock:
                last_recovery = self._last_recovery_epoch.get(seed_name, 0.0)
            if now - last_recovery < self.stale_after_seconds:
                continue
            self._recover_bank(
                seed_name=seed_name, store=store, owner=owner,
                event="SEED_BANK_SELF_HEALED",
                heartbeat_age_seconds=age,
                running_count=stats.get("running_count"),
                min_heartbeat=stats.get("min_heartbeat"),
                max_heartbeat=max_heartbeat,
                min_lease_until=stats.get("min_lease_until"),
                max_lease_until=max_lease_until,
            )

    def start(self) -> "_SeedBankWatchdog":
        self._threads = []
        for seed_name in self.stores:
            thread = threading.Thread(
                target=self._run,
                args=(seed_name,),
                name=f"dqbd-seed-bank-watchdog-{seed_name.lower()}",
                daemon=True,
            )
            self._threads.append(thread)
            thread.start()
        self._thread = self._threads[0] if self._threads else None
        return self

    def _run(self, seed_name: str | None = None) -> None:
        while not self._stop.is_set():
            # A watchdog must survive transient SQLite/IO faults.  The next
            # tick retries the affected bank; the main run remains alive and
            # the error is retained in the audit stream.
            try:
                self._tick(
                    (str(seed_name),)
                    if seed_name is not None else None)
            except BaseException as exc:
                self._record({
                    "event": "SEED_BANK_WATCHDOG_TICK_ERROR",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                })
            self._stop.wait(self.interval_seconds)

    def close(self) -> None:
        self._stop.set()
        threads = list(self._threads)
        if not threads and self._thread is not None:
            threads = [self._thread]
        for thread in threads:
            thread.join(timeout=max(5.0, self.interval_seconds * 3))


def _gate(repo_root: Path) -> tuple[dict, RunGateContract]:
    path = repo_root / "research" / "DQBD_CAUSAL_MODEL_STORE_V1_RUN_GATES.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    gate = RunGateContract.from_dict(payload)
    if gate.contract_hash != payload.get("contract_hash"):
        raise ValueError("DQBD_RUN_GATE_HASH_MISMATCH")
    base = default_pseudolive_experiment_contract(baseline_commit=gate.baseline_commit)
    if base.contract_hash != gate.base_contract_hash:
        raise ValueError("DQBD_BASE_CONTRACT_HASH_MISMATCH")
    return payload, gate


def _contract_audit(root: Path, gate_payload: dict, *, seed_name: str, seed: date) -> dict:
    return {
        "schema_version": "DQBD_STEP9_CONTRACT_AUDIT_V1",
        "status": "PASS",
        "seed_name": seed_name,
        "seed": seed.isoformat(),
        "development_end": "2025-12-31",
        "holdout_boundary": HOLDOUT.isoformat(),
        "holdout_reads": 0,
        "performance_evaluation_opened": False,
        "run_gate_hash": gate_payload["contract_hash"],
        "evaluation_contract_hash": stable_hash(gate_payload["evaluation"]),
        "bootstrap_fingerprint": gate_payload["evaluation"]["bootstrap_fingerprint"],
        "fixed_portfolio": {
            "exit": "FIXED", "allocation": "EQUAL_ACTIVE",
            "replacement": "IGNORE_NEW", "execution": "NEXT_OPEN",
            "benchmark": "MSCI_WORLD_IMPLEMENTABLE_PROXY_V1",
        },
        "source_root": str(root.resolve()),
    }


def _write_seed_package(seed_root: Path, seed_name: str, seed: date, gate_payload: dict) -> None:
    development = seed_root / "development"
    if not (development / "summary.json").is_file():
        raise RuntimeError(f"DQBD_STEP9_DEVELOPMENT_RESULT_MISSING:{seed_name}")
    # The existing development evaluation already writes canonical economic
    # metrics. Keep these aliases explicit and machine-readable; no second
    # wealth calculation is introduced here.
    summary = json.loads((development / "summary.json").read_text(encoding="utf-8"))
    _json(seed_root / "summary.json", {
        "schema_version": "DQBD_STEP9_SEED_SUMMARY_V1", "seed_name": seed_name,
        "seed": seed.isoformat(), "evaluation_opened": True,
        "canonical_source": str((development / "summary.json").resolve()),
        "development_summary": summary,
    })
    source_map = {
        "REPORT.md": development / "REPORT.md",
        "arm-comparison.csv": development / "abc_arm_summary.csv",
        "event-subperiod-summary.csv": development / "abc_monthly_evidence.parquet",
        "initial-model-store.csv": seed_root / "valid_generations.parquet",
        "generation-ledger.csv": seed_root / "abc_generation_schedule.parquet",
        "evidence-event-ledger.csv": development / "monthly_family_evidence.parquet",
        "orchestrator-decisions.csv": development / "replay_states.json",
        "oracle-regret-summary.csv": development / "gate1_baseline_comparison.csv",
        "runtime-telemetry.json": seed_root / "telemetry" / "resource-contract.json",
    }
    for name, source in source_map.items():
        destination = seed_root / name
        if destination.exists():
            continue
        if source.suffix == ".parquet" and source.is_file():
            import pandas as pd
            pd.read_parquet(source).to_csv(destination, index=False)
        elif source.is_file():
            destination.write_bytes(source.read_bytes())
        else:
            destination.write_text("", encoding="utf-8")
    _json(seed_root / "run-contract.json", {
        "schema_version": "DQBD_STEP9_RUN_CONTRACT_V1", "seed_name": seed_name,
        "seed": seed.isoformat(), "development_end": "2025-12-31",
        "holdout_boundary": HOLDOUT.isoformat(), "run_gate_hash": gate_payload["contract_hash"],
        "evaluation_contract_hash": stable_hash(gate_payload["evaluation"]),
        "requested_worker_count": 26, "queue_capacity": 52,
        "native_threads_per_worker": 1,
    })
    _json(seed_root / "contract-audit.json", _contract_audit(seed_root, gate_payload, seed_name=seed_name, seed=seed))
    _json(seed_root / "resume-manifest.json", {
        "schema_version": "DQBD_STEP9_RESUME_MANIFEST_V1", "status": "COMPLETE",
        "contract_hash": gate_payload["contract_hash"], "seed_name": seed_name,
        "completed_stages": ["factory", "predictions", "matured_evidence", "replay", "evaluation"],
        "evaluation_not_yet_open_marker": False,
    })


def _resolve_seed_root(*, seed_name: str, output_root: Path,
                       checkpoint_root: Path | None,
                       short_source_root: Path | None) -> tuple[Path, str]:
    """Resolve a seed store without ever silently falling back to a fresh DAG.

    A supplied checkpoint root is an explicit resume contract. Falling back
    to ``output_root/<seed>`` when one seed directory is absent would call the
    graph builder and recreate the already materialized job list. That is
    both expensive and scientifically unsafe because the three seed stores
    would no longer share the intended resume provenance.
    """
    if checkpoint_root is not None:
        seed_root = checkpoint_root / seed_name.lower()
        required = (seed_root / "manifested-job-contract.json",
                     seed_root / "run-state" / "jobs.sqlite3")
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise RuntimeError(
                "DQBD_STEP9_CHECKPOINT_ROOT_INCOMPLETE_NO_FRESH_JOB_GRAPH:" +
                seed_name + ":" + ",".join(missing))
        return seed_root, "EXPLICIT_CHECKPOINT_RESUME"
    if seed_name == "SHORT" and short_source_root is not None:
        return short_source_root, "LEGACY_SHORT_CHECKPOINT_RESUME"
    return output_root / seed_name.lower(), "OUTPUT_ROOT"


def run(*, signal_panel: Path, candidate_metrics: Path, prices: Path, output_root: Path,
        nwinfo_executable: str | None = None, short_source_root: Path | None = None,
        checkpoint_root: Path | None = None, worker_count: int = 26,
        cpu_target_fraction: float = .90, slow_resume: bool = False,
        direct_checkpoint_resume: bool = False) -> int:
    repo_root = Path(__file__).resolve().parents[3]
    gate_payload, gate = _gate(repo_root)
    # Configure the shared runtime contract before any spawned lane is created.
    # v40 defaults to 26 logical lanes. A later 32-lane host test is allowed
    # explicitly; RAM admission, not hidden oversubscription, remains the gate.
    logical_processors = int(os.cpu_count() or 1)
    worker_count = int(worker_count)
    if worker_count < len(SEEDS) or worker_count > logical_processors:
        raise ValueError(
            f"DQBD_STEP9_WORKER_COUNT_INVALID:{worker_count}:"
            f"{logical_processors}")
    cpu_target_fraction = float(cpu_target_fraction)
    if not 0 < cpu_target_fraction <= 1:
        raise ValueError("DQBD_STEP9_CPU_TARGET_FRACTION_INVALID")
    effective_cpu_target = min(
        1.0,
        max(cpu_target_fraction, worker_count / logical_processors),
    )
    runtime_contract = configure_cpu_peak(
        effective_cpu_target,
        process_workers=worker_count,
        native_threads_per_worker=1,
        memory_limit_fraction=.95,
        memory_soft_target_fraction=.90,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    # Hashing a development slice uses bounded external-sort shards. Keep the
    # scratch volume on the run's shared D: tree and make the result reusable
    # by every spawned worker. The startup cleanup removes only this run's
    # named scratch directories left by an earlier hard worker termination;
    # atexit/finally cover orderly exceptions and normal completion.
    slice_hash_temp_root = output_root / "_shared-compute" / "temporary" / "slice-hash"
    slice_hash_cache_root = output_root / "_shared-compute" / "development-slice-hash-cache-v1"
    os.environ["DQBD_SLICE_HASH_SCRATCH_ROOT"] = str(slice_hash_temp_root)
    os.environ["DQBD_SLICE_HASH_CACHE_ROOT"] = str(slice_hash_cache_root)
    cleanup_legacy_slice_hash_scratch()
    cleanup_orphaned_slice_hash_scratch()
    atexit.register(cleanup_orphaned_slice_hash_scratch)
    os.environ["DQBD_ENABLE_GPU_PRETRAINING"] = "1"
    gpu_contract = configure_gpu_pretraining(output_root / "_shared-compute", require=True)
    # Watch source from process start.  A direct checkpoint resume skips the
    # expensive graph validation pass and starts from the existing SQLite
    # frontier; later source edits are adopted by the supervisor through a
    # boundary stop and a fresh immutable snapshot.
    hot_reload_controller = HotReloadController(
        (repo_root / "stock_predictor",),
        boundary_flag=output_root / "_hotload" / "stop-at-boundary.flag")
    root_contract = output_root / "run-contract.json"
    root_payload = {
        "schema_version": "DQBD_STEP9_FIXED_CAUSAL_MODEL_STORE_RUN_V40_1_BATCHED_PORTFOLIO_VALIDATED_RESUME",
        "seeds": {key: value.isoformat() for key, value in SEEDS.items()},
        "development_end": "2025-12-31", "holdout_boundary": HOLDOUT.isoformat(),
        "run_gate_hash": gate.contract_hash, "evaluation_contract_hash": gate.evaluation.contract_hash,
        "requested_worker_count": worker_count,
        "queue_capacity": worker_count,
        "native_threads_per_worker": 1,
        "cpu_target_fraction_requested": cpu_target_fraction,
        "cpu_target_fraction_effective": effective_cpu_target,
        "runtime_contract": runtime_contract,
        "gpu_pretraining_contract_hash": gpu_contract["contract_hash"],
        "ridge_execution_backend": GPU_EXECUTION_BACKEND,
        "hgb_execution_backend": GPU_HGB_EXECUTION_BACKEND,
        "cpu_fallback_backend": CPU_EXECUTION_BACKEND,
        "execution_scheduler_policy": "SINGLE_LANE_RECLAIM_FRONTIER_V40_0_4",
        "portfolio_batch_schema": PORTFOLIO_BATCH_SCHEMA,
        "portfolio_batch_policy": PORTFOLIO_BATCH_POLICY,
        "portfolio_logical_rows_preserved": True,
        "portfolio_physical_scheduling_unit": "HORIZON_X_CUTOFF_BATCH",
        "gpu_queue_policy": "HGB_FIRST_PREFETCHED_LEARNED_CPU_GPU_ROUTER_V40_0_4_3",
        "gpu_workers_per_device": 1,
        "gpu_queue_ahead_per_device": 2,
        "candidate_fold_prep_prefetch": 2,
        "gpu_ram_scheduler_coupled": False,
        "gpu_global_section_backlog_aware": True,
        "gpu_physical_breadth_first": True,
        "gpu_backlog_cache_device_set_keyed": True,
        "gpu_duration_profile":
            "_shared-compute/candidate-execution-duration-profiles.json",
        "ram_admission_policy":
            "REAL_LOAD_10MS_DAMPED_RECOVERY_80_86_90_92_95_V40_0_4",
        "ram_live_sampling_interval_ms": 10,
        "ram_runtime_rollup_interval_ms": 2000,
        "ram_real_load_band": {
            "fill_floor": .80, "target": .86,
            "stop": .90, "reclaim": .92, "hard": .95,
        },
        "ram_reclaim_policy": {
            "rising_trigger_fraction": .90,
            "rising_trigger_gib_per_s": .25,
            "hard_reclaim_fraction": .92,
            "emergency_reclaim_fraction": .945,
            "victims_per_decision": 1,
            "victim_policy":
                "PROGRESS_PROTECTED_BEST_FIT_TO_86_PERCENT",
            "settling_ms": 750,
            "recovery_seconds": 5,
            "action":
                "SSD_SPILL_TERMINATE_ONE_LANE_REQUEUE_PENDING",
        },
        "ram_admission_pacing_per_10ms": {
            "normal_below_60pct": 8,
            "normal_60_to_80pct": 4,
            "normal_80pct_plus": 2,
            "recovery": "ONE_JOB_EVERY_100_TO_250MS_UNTIL_STABLE",
        },
        "ram_trend_lookahead_ms": 250,
        "development_slice_hash_cache": str(slice_hash_cache_root),
        "development_slice_hash_temp_root": str(slice_hash_temp_root),
        "gpu_runtime_monitoring": "NVIDIA_SMI_PLUS_LIBRE_HARDWARE_MONITOR_OPENCL_V1",
        "hgb_device_policy": GPU_HGB_DEVICE_POLICY,
        "hgb_kernel_policy": GPU_HGB_KERNEL_POLICY,
        "complete_result_validation": COMPLETE_RESULT_VALIDATION_SCHEMA,
    }
    prior_root_payload = (
        json.loads(root_contract.read_text(encoding="utf-8"))
        if root_contract.is_file() else None)
    started = time.monotonic()
    feature_schema = repo_root / "stock_predictor" / "v5" / "feature_schema.json"
    benchmark_distributions = repo_root / "artifacts" / "alpaca-urth-daily-repair" / "URTH-distributions.parquet"
    hyperparameter_space = repo_root / "stock_predictor" / "v5" / "hyperparameter_space.json"
    if not feature_schema.is_file() or not benchmark_distributions.is_file():
        raise FileNotFoundError("DQBD_STEP9_CANONICAL_BENCHMARK_OR_FEATURE_SCHEMA_MISSING")
    worker_map = list(runtime_contract.get("worker_map") or [])
    if len(worker_map) < worker_count:
        raise RuntimeError("DQBD_STEP9_WORKER_MAP_BELOW_REQUESTED_LANES")
    seed_names = tuple(SEEDS)
    base_seed_workers, seed_remainder = divmod(
        worker_count, len(seed_names))
    seed_counts = {
        name: base_seed_workers + (1 if index < seed_remainder else 0)
        for index, name in enumerate(seed_names)
    }
    seed_allocations = {}
    cursor = 0
    for name in seed_names:
        count = seed_counts[name]
        seed_allocations[name] = (
            count, worker_map[cursor:cursor + count])
        cursor += count
    root_payload["seed_worker_allocation_target"] = dict(seed_counts)
    root_payload["seed_worker_allocation"] = dict(seed_counts)
    if prior_root_payload is not None and prior_root_payload != root_payload:
        semantic_keys = (
            "seeds", "development_end", "holdout_boundary",
            "run_gate_hash", "evaluation_contract_hash",
            "gpu_pretraining_contract_hash",
            "ridge_execution_backend", "hgb_execution_backend",
            "cpu_fallback_backend", "hgb_device_policy",
            "hgb_kernel_policy",
        )
        prior_semantic = {
            key: prior_root_payload.get(key) for key in semantic_keys}
        current_semantic = {
            key: root_payload.get(key) for key in semantic_keys}
        gpu_migration_path = (
            output_root / "_shared-compute"
            / "gpu-pretraining-contract-migration.json")
        gpu_migration = None
        if gpu_migration_path.is_file():
            try:
                gpu_migration = json.loads(
                    gpu_migration_path.read_text(encoding="utf-8"))
            except (OSError, ValueError, TypeError):
                gpu_migration = None
        semantic_without_gpu_hash = tuple(
            key for key in semantic_keys
            if key != "gpu_pretraining_contract_hash")
        compatible_gpu_migration = (
            isinstance(gpu_migration, dict)
            and gpu_migration.get("prior_contract_hash")
                == prior_semantic.get("gpu_pretraining_contract_hash")
            and gpu_migration.get("current_contract_hash")
                == current_semantic.get("gpu_pretraining_contract_hash"))
        if prior_semantic != current_semantic and not (
            all(
                prior_semantic.get(key) == current_semantic.get(key)
                for key in semantic_without_gpu_hash
            )
            and compatible_gpu_migration
        ):
            raise ValueError("DQBD_STEP9_RESUME_CONTRACT_MISMATCH")
        root_payload["runtime_migration"] = {
            "status": "SCHEDULER_ONLY_COMPATIBLE",
            "from_schema_version":
                prior_root_payload.get("schema_version"),
            "to_schema_version":
                root_payload["schema_version"],
        }
        if compatible_gpu_migration:
            root_payload["runtime_migration"]["gpu_contract"] = {
                "status": "NON_NUMERICAL_RUNTIME_MIGRATION",
                "audit": str(gpu_migration_path),
                "prior_contract_hash": (
                    gpu_migration["prior_contract_hash"]),
                "current_contract_hash": (
                    gpu_migration["current_contract_hash"]),
            }
    _json(root_contract, root_payload)
    ram_scheduler = RamAdmissionScheduler(
        # Size the admission board for the host lane maximum, not for the
        # lane count of this moment, so the runtime lane override below can
        # raise concurrency without rebuilding shared memory. Actual
        # concurrency stays bounded by the per-wave lane allocation; this
        # value only bounds the RAM admission board.
        max_slots=max(worker_count, MAX_RUNTIME_PROCESS_LANES),
        fill_floor_fraction=.80,
        target_fraction=.86,
        stop_fraction=.90,
        reclaim_fraction=.92,
        hard_limit_fraction=.95,
        reclaim_slope_gib_per_s=.25,
        estimated_task_gib=1.0,
        profile_path=(
            output_root / "_shared-compute" / "job-memory-profiles.json"),
        telemetry_path=(
            output_root / "_shared-compute" / "real-load-runtime.jsonl"),
        telemetry_rollup_seconds=2.0,
        trend_lookahead_seconds=.25,
        gpu_companion_slots_per_device=4)

    seed_contexts = {}

    def initialize_seed(item: tuple[str, date]) -> tuple[str, dict]:
        seed_name, seed = item
        # A checkpoint root is an explicit, immutable-resume input. It lets
        # a new run identity reuse already materialized seed DAGs without
        # silently rebuilding their completed work. The legacy SHORT-only
        # option remains supported for existing handoffs.
        seed_root, root_mode = _resolve_seed_root(
            seed_name=seed_name, output_root=output_root,
            checkpoint_root=checkpoint_root, short_source_root=short_source_root)
        inputs = ManifestedJobInputs(
            repo_root=repo_root, signal_panel=signal_panel, candidate_metrics=candidate_metrics,
            feature_schema=feature_schema, benchmark_prices=prices,
            benchmark_distributions=benchmark_distributions,
            stock_execution_prices=prices, stock_distributions=benchmark_distributions,
            hyperparameter_space=hyperparameter_space if hyperparameter_space.is_file() else None,
            development_start=seed, development_end=date(2025, 12, 31),
            holdout_boundary=HOLDOUT, allow_dirty_development_fixture=True,
        )
        init_cache_path = seed_root / "run-state" / (
            "step9-initialization-cache-v3.json")
        contract_path = seed_root / "manifested-job-contract.json"
        prior_contract = None
        if contract_path.is_file():
            try:
                candidate = json.loads(contract_path.read_text(encoding="utf-8"))
                if isinstance(candidate, dict):
                    prior_contract = candidate
            except (OSError, ValueError, TypeError):
                prior_contract = None

        def initialization_cache_key() -> dict:
            return {
                "schema_version": "DQBD_STEP9_INITIALIZATION_CACHE_V3_VALIDATED_COMPLETE",
                "seed_name": seed_name,
                "seed": seed.isoformat(),
                "root_mode": root_mode,
                "current_git_sha": current_git_sha(repo_root),
                "input_stat_fingerprint": input_stat_fingerprint(inputs),
                "manifested_job_contract_sha256": (
                    __import__("hashlib").sha256(
                        contract_path.read_bytes()).hexdigest()
                    if contract_path.is_file() else None),
                "complete_result_validation_authority_sha256":
                    complete_result_validation_authority_sha256(seed_root),
                "complete_result_validation_schema":
                    COMPLETE_RESULT_VALIDATION_SCHEMA,
            }

        cache_key = initialization_cache_key()
        cached = None
        compatible_runtime_cache = False
        if direct_checkpoint_resume:
            cached = {
                "status": "DIRECT_CHECKPOINT_RESUME",
                "seed_name": seed_name,
                "seed_root": str(seed_root),
                "checkpoint_contract": str(contract_path),
            }
        # An explicit direct resume must survive an unreadable cache envelope.
        # Falling back to the validating pass silently requeues COMPLETE
        # nodes, which is how a supervised restart loop can lose more finished
        # work than it produces.
        direct_resume_floor = cached if direct_checkpoint_resume else None
        if not slow_resume and init_cache_path.is_file():
            try:
                candidate = json.loads(
                    init_cache_path.read_text(encoding="utf-8"))
                candidate_key = candidate.get("cache_key")
                exact_cache = candidate_key == cache_key
                compatible_cache = (
                    isinstance(candidate_key, dict)
                    and candidate_key.get("current_git_sha")
                    in RESUME_COMPATIBLE_RUNTIME_BASE_SHAS
                    and cache_key.get("current_git_sha")
                    not in RESUME_COMPATIBLE_RUNTIME_BASE_SHAS
                    and {
                        key: value for key, value in candidate_key.items()
                        if key != "current_git_sha"
                    } == {
                        key: value for key, value in cache_key.items()
                        if key != "current_git_sha"
                    }
                )
                if exact_cache or compatible_cache:
                    cached = (
                        candidate.get("initialized") or direct_resume_floor)
                    compatible_runtime_cache = bool(
                        cached is not None
                        and compatible_cache and not exact_cache)
            except (OSError, ValueError, TypeError):
                cached = direct_resume_floor
        if cached is not None:
            initialized = dict(cached)
            initialized["status"] = "RESUMED_INITIALIZATION_CACHE_HIT"
            initialized["resume_mode"] = (
                "FAST_VALIDATED_RUNTIME_COMPATIBLE"
                if compatible_runtime_cache else "FAST_VALIDATED")
            if compatible_runtime_cache:
                # Migrate only the small cache envelope.  No DAG or result
                # artifact is rewritten, and the validation authority remains
                # part of the exact key comparison above.
                _json(init_cache_path, {
                    "cache_key": cache_key,
                    "initialized": initialized,
                })
            store = ManifestedJobStore(seed_root)
            store.requeue_interrupted_jobs(retry_failed=True)
            store.ensure_execution_backend_contract({
                "ridge_execution_backend": GPU_EXECUTION_BACKEND,
                "hgb_execution_backend": GPU_HGB_EXECUTION_BACKEND,
                "cpu_fallback_backend": CPU_EXECUTION_BACKEND,
                "execution_scheduler_policy": "SINGLE_LANE_RECLAIM_FRONTIER_V40_0_4",
                "gpu_queue_policy": "HGB_FIRST_PREFETCHED_LEARNED_CPU_GPU_ROUTER_V40_0_4_3",
                "gpu_runtime_monitoring": "NVIDIA_SMI_PLUS_LIBRE_HARDWARE_MONITOR_OPENCL_V1",
                "hgb_device_policy": GPU_HGB_DEVICE_POLICY,
                "hgb_kernel_policy": GPU_HGB_KERNEL_POLICY,
                "ram_admission_policy":
                    "REAL_LOAD_10MS_DAMPED_RECOVERY_80_86_90_92_95_V40_0_4",
                "gpu_workers_per_device": 1,
                "gpu_global_section_backlog_aware": True,
                "gpu_physical_breadth_first": True,
                "gpu_backlog_cache_device_set_keyed": True,
                "gpu_pretraining_contract_hash": gpu_contract["contract_hash"],
            })
        else:
            # Cache miss (or explicit --slow-resume) is the one slow,
            # correctness-first reconciliation pass. It first reconciles the
            # manifested graph, then validates every COMPLETE result that
            # survived reconciliation. Only that validated authority can seed
            # the next fast-resume cache.
            initialized = initialize_manifested_job_coordinator(
                inputs=inputs, output_root=seed_root,
                allow_compatible_code_resume=True,
                reconcile_existing_graph=True)
            initialized["resume_mode"] = "SLOW_RECONCILED_VALIDATING"
            store = ManifestedJobStore(seed_root)
            store.requeue_interrupted_jobs(retry_failed=True)
            store.ensure_execution_backend_contract({
                "ridge_execution_backend": GPU_EXECUTION_BACKEND,
                "hgb_execution_backend": GPU_HGB_EXECUTION_BACKEND,
                "cpu_fallback_backend": CPU_EXECUTION_BACKEND,
                "execution_scheduler_policy": "SINGLE_LANE_RECLAIM_FRONTIER_V40_0_4",
                "gpu_queue_policy": "HGB_FIRST_PREFETCHED_LEARNED_CPU_GPU_ROUTER_V40_0_4_3",
                "gpu_runtime_monitoring": "NVIDIA_SMI_PLUS_LIBRE_HARDWARE_MONITOR_OPENCL_V1",
                "hgb_device_policy": GPU_HGB_DEVICE_POLICY,
                "hgb_kernel_policy": GPU_HGB_KERNEL_POLICY,
                "ram_admission_policy":
                    "REAL_LOAD_10MS_DAMPED_RECOVERY_80_86_90_92_95_V40_0_4",
                "gpu_workers_per_device": 1,
                "gpu_global_section_backlog_aware": True,
                "gpu_physical_breadth_first": True,
                "gpu_backlog_cache_device_set_keyed": True,
                "gpu_pretraining_contract_hash": gpu_contract["contract_hash"],
            })
            validation = audit_complete_results(
                store=store, prior_contract=prior_contract)
            initialized["complete_result_validation"] = validation
            initialized["resume_mode"] = "SLOW_RECONCILED_VALIDATED"
            # Recompute after reconciliation and validation because both the
            # manifested-job contract and validation authority may have been
            # migrated. The next default resume may skip the slow path only if
            # both remain byte-identical under the current Git/input identity.
            cache_key = initialization_cache_key()
            _json(init_cache_path, {
                "cache_key": cache_key,
                "initialized": initialized,
            })
        return seed_name, {
            "seed": seed, "root": seed_root, "inputs": inputs,
            "store": store, "initialized": initialized, "root_mode": root_mode,
        }

    # Checkpoint-backed initialization is read-heavy and independent by seed.
    # Do it concurrently so a fresh run does not spend the startup phase
    # serially hashing/validating three equivalent data surfaces.
    with ThreadPoolExecutor(max_workers=3, thread_name_prefix="dqbd-seed-init") as init_pool:
        for seed_name, context in init_pool.map(initialize_seed, SEEDS.items()):
            seed_contexts[seed_name] = context

    # Candidate-OOS is independent of the seed boundary. Select the seed
    # with the most completed checkpoints as the one global producer, finish
    # remaining physical work with the configured lane capacity, then publish
    # verified hardlinked views into the other seed-local DAGs.
    # The runtime lane override is intentionally consulted before this global
    # phase as well as before every seed wave. This makes a 26 -> 32 ramp a
    # real hotpath change for the complete scheduler, not only for the final
    # seed-local phase.
    runtime_override_path = (
        output_root / "_hotload" / "runtime-overrides.json")

    def _requested_worker_count(current: int) -> int:
        """Resolve the operator-requested lane count for the next phase."""
        try:
            payload = json.loads(
                runtime_override_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            return current
        if not isinstance(payload, dict):
            return current
        requested = payload.get("workers")
        if requested is None:
            return current
        try:
            requested = int(requested)
        except (TypeError, ValueError):
            return current
        return max(
            len(SEEDS),
            min(requested, logical_processors, MAX_RUNTIME_PROCESS_LANES))

    requested_workers = _requested_worker_count(worker_count)
    if requested_workers != worker_count:
        previous_workers = worker_count
        worker_count = requested_workers
        effective_cpu_target = min(
            1.0,
            max(cpu_target_fraction, worker_count / logical_processors))
        runtime_contract = configure_cpu_peak(
            effective_cpu_target,
            process_workers=worker_count,
            native_threads_per_worker=1,
            memory_limit_fraction=.95,
            memory_soft_target_fraction=.90,
        )
        worker_map = list(runtime_contract.get("worker_map") or [])
        root_payload["requested_worker_count"] = worker_count
        root_payload["queue_capacity"] = worker_count
        root_payload["cpu_target_fraction_effective"] = effective_cpu_target
        root_payload["runtime_contract"] = runtime_contract
        root_payload["runtime_worker_override"] = {
            "schema_version": "DQBD_RUNTIME_LANE_OVERRIDE_V1",
            "previous_worker_count": previous_workers,
            "worker_count": worker_count,
            "adopted_at_phase": "candidate_oos",
            "adopted_at_epoch": time.time(),
            "process_restarted": False,
            "source": str(runtime_override_path),
        }
        _json(root_contract, root_payload)
        print(
            "[dynamic-qbd][runtime] adopted lane override "
            f"{previous_workers} -> {worker_count} before candidate_oos "
            f"without restarting pid={os.getpid()}",
            flush=True)
    producer_name = max(
        SEEDS, key=lambda name: seed_contexts[name]["store"].count_jobs_by_kind_state(
            kind="candidate_oos_fold", state="COMPLETE"))
    producer = seed_contexts[producer_name]
    candidate_execution = dict(execute_candidate_oos_jobs(
        store=producer["store"], inputs=producer["inputs"],
        output_root=producer["root"],
        workers=worker_count,
        queue_ahead=2,
        max_inflight=min(
            worker_count,
            int(os.environ.get(
                "DQBD_CANDIDATE_MAX_INFLIGHT",
                str(worker_count)))),
        worker_map_override=worker_map,
        ram_scheduler=ram_scheduler,
        estimated_task_gib=5.5,
        hot_reload_controller=hot_reload_controller))
    candidate_execution["ram_targeted_reclaims"] = (
        ram_scheduler.telemetry().get(
            "ram_scheduler_targeted_reclaim_count", 0))
    candidate_failure_count = producer["store"].count_jobs_by_kind_state(
        kind="candidate_oos_fold", state="FAILED")
    if candidate_failure_count:
        raise RuntimeError(
            f"DQBD_GLOBAL_CANDIDATE_JOBS_FAILED:{producer_name}:{candidate_failure_count}")
    publication = publish_candidate_oos_seed_views(
        source_store=producer["store"], source_root=producer["root"],
        target_stores={name: context["store"] for name, context in seed_contexts.items()},
        target_roots={name: context["root"] for name, context in seed_contexts.items()})
    _json(output_root / "shared-compute-runtime.json", {
        "schema_version": "DQBD_STEP9_SHARED_COMPUTE_RUNTIME_V1",
        "candidate_oos": {"producer_seed": producer_name,
                          "execution": candidate_execution,
                          "publication": publication},
        "production_generations": "GLOBAL_IMMUTABLE_FIT_KEY_STORE",
        "causal_visibility": "SEED_LOCAL_MANIFESTED_DAG_ONLY",
        "portfolio_batch_schema": PORTFOLIO_BATCH_SCHEMA,
        "portfolio_batch_policy": PORTFOLIO_BATCH_POLICY,
        "portfolio_logical_rows_preserved": True,
        "portfolio_physical_scheduling_unit": "HORIZON_X_CUTOFF_BATCH",
        "hgb_backend": GPU_HGB_EXECUTION_BACKEND,
        "gpu_queue_policy": "HGB_FIRST_PREFETCHED_LEARNED_CPU_GPU_ROUTER_V40_0_4_3",
        "gpu_workers_per_device": 1,
        "gpu_ram_scheduler_coupled": False,
        "gpu_global_section_backlog_aware": True,
        "gpu_physical_breadth_first": True,
        "gpu_backlog_cache_device_set_keyed": True,
        "gpu_duration_profile": str(
            output_root / "_shared-compute" /
            "candidate-execution-duration-profiles.json"),
        "ram_admission_policy":
            "REAL_LOAD_10MS_DAMPED_RECOVERY_80_86_90_92_95_V40_0_4",
        "ram_profile_path": str(
            output_root / "_shared-compute" / "job-memory-profiles.json"),
        "ram_runtime_telemetry_path": str(
            output_root / "_shared-compute" / "real-load-runtime.jsonl"),
        "ram_runtime_rollup_interval_ms": 2000,
        "gpu_runtime_monitoring": "NVIDIA_SMI_PLUS_LIBRE_HARDWARE_MONITOR_OPENCL_V1",
        "gpu_pretraining": gpu_contract,
        "worker_target": worker_count, "queue_ahead": 2,
        "gpu_queue_ahead_per_device": 2,
        "candidate_fold_prep_prefetch": 2,
        "seed_root_modes": {name: context["root_mode"] for name, context in seed_contexts.items()},
        "complete_result_validation": COMPLETE_RESULT_VALIDATION_SCHEMA,
        "complete_result_validation_authority": {
            name: complete_result_validation_authority_sha256(context["root"])
            for name, context in seed_contexts.items()
        },
    })

    # Never reuse the generic historical owner. A run/seed-specific owner
    # makes stale leases attributable and prevents a later run from being
    # mistaken for the process that claimed them.
    seed_owners = {
        name: f"dqbd-step9-{os.getpid()}-{name.lower()}"
        for name in SEEDS
    }
    seed_watchdog = _SeedBankWatchdog(
        stores={name: context["store"] for name, context in seed_contexts.items()},
        owners=seed_owners,
        activity_paths={
            name: context["root"] / "seed-bank-heartbeat.json"
            for name, context in seed_contexts.items()
        },
        telemetry_path=output_root / "_shared-compute" / "seed-bank-watchdog-events.jsonl",
    ).start()
    def run_seed(seed_item: tuple[str, date]) -> None:
        seed_name, seed = seed_item
        seed_workers, seed_worker_map = seed_allocations[seed_name]
        context = seed_contexts[seed_name]
        seed_root = context["root"]
        marker = seed_root / "completion-manifest.json"
        if marker.is_file() and json.loads(marker.read_text(encoding="utf-8")).get("run_gate_hash") == gate.contract_hash:
            prior_progress = ManifestedJobStore(seed_root).progress()
            if prior_progress.get("FAILED", 0) == 0 and prior_progress.get("PENDING", 0) == 0:
                return
        inputs = context["inputs"]
        initialized = context["initialized"]
        result = execute_manifested_development_jobs(
            store=context["store"], inputs=inputs, output_root=seed_root,
            workers=seed_workers, queue_ahead=2, owner=seed_owners[seed_name],
            worker_map_override=seed_worker_map,
            ram_scheduler=ram_scheduler,
            # The v13 checkpoint contains immutable snapshots from the
            # pre-deduplication reader. Keep them as historical evidence and
            # materialize corrected snapshots in a new immutable namespace.
            evidence_snapshot_root=(
                seed_root / "evidence-snapshots-v40.0.3-canonical"),
            # Selection artifacts are immutable. The resumed v2 checkpoint
            # already contains v40.0.3 selections from the prior snapshot
            # semantics; a repaired/stale-dependency snapshot must use a new
            # namespace instead of colliding with those historical files.
            recipe_selection_root=(
                seed_root / "recipe-selections-v40.0.4-canonical"),
            activity_heartbeat_path=seed_root / "seed-bank-heartbeat.json",
            hot_reload_controller=hot_reload_controller,
            causal_job_budget=SEED_WAVE_CAUSAL_JOB_BUDGET,
        )
        # ManifestedJobStore uses the canonical uppercase state names. A
        # lowercase lookup previously allowed failed jobs to be published as
        # complete and was the direct reason evaluation never opened safely.
        if result.get("progress", {}).get("FAILED", 0):
            raise RuntimeError(f"DQBD_STEP9_CAUSAL_JOBS_FAILED:{seed_name}:{result['progress']}")
        if result.get("progress", {}).get("PENDING", 0):
            # This was a bounded fair-share quantum, not a completed seed.
            # Leave the bank in ``remaining`` so the next scheduler wave can
            # rotate to the next ready bank without publishing a false
            # completion marker.
            return
        _json(seed_root / "causal-run-result.json", {"initialized": initialized, "execution": result})
        _json(marker, {"status": "COMPLETE", "seed_name": seed_name, "run_gate_hash": gate.contract_hash,
                       "evaluation_opened": False, "causal_materialization_complete": True,
                       "portfolio_batch_policy": PORTFOLIO_BATCH_POLICY,
                       "elapsed_seconds": time.monotonic() - started})
        _json(seed_root / "seed-bank-heartbeat.json", {
            "schema_version": "DQBD_SEED_BANK_ACTIVITY_V1",
            "state": "COMPLETE", "owner": seed_owners[seed_name],
            "pid": os.getpid(), "updated_at_epoch": time.time(),
        })

    # Seeds have independent causal information sets and advance in parallel.
    # Allocate lanes in waves from the current READY frontier. A bank with no
    # available work receives zero lanes and its capacity is lent to active
    # banks (for example 15/16/0), then recomputed after the next completion.

    try:
        remaining = set(SEEDS)
        wave = 0
        next_seed_index = 0
        while remaining:
            complete_now = {
                name for name in remaining
                if (
                    (seed_root := Path(seed_contexts[name]["root"]))
                    / "completion-manifest.json"
                ).is_file()
                and seed_contexts[name]["store"].progress().get(
                    "FAILED", 0) == 0
                and seed_contexts[name]["store"].progress().get(
                    "PENDING", 0) == 0
            }
            remaining -= complete_now
            if not remaining:
                break
            ready_by_seed = {
                name: seed_contexts[name]["store"].ready_job_count()
                for name in remaining
            }
            selected_seed, next_seed_index = _next_ready_seed_wave(
                seed_names=SEEDS, remaining=remaining,
                ready_by_seed=ready_by_seed, next_index=next_seed_index)
            if selected_seed is None:
                raise RuntimeError(
                    f"DQBD_STEP9_SEED_FRONTIER_STARVED:{ready_by_seed}")
            # Only one seed bank owns pools in a wave to prevent multiplying
            # 32 CPU lanes plus two GPU lanes across all three banks. The
            # bounded causal quantum above rotates ownership, so a bank can
            # use the complete pool without starving its siblings.
            active_names = (selected_seed,)
            allocation_ready_by_seed = {
                name: ready_by_seed.get(name, 0)
                if name in active_names else 0
                for name in SEEDS
            }
            requested_workers = _requested_worker_count(worker_count)
            if requested_workers != worker_count:
                previous_workers = worker_count
                worker_count = requested_workers
                effective_cpu_target = min(
                    1.0,
                    max(cpu_target_fraction,
                        worker_count / logical_processors))
                # Re-pin affinity and refresh the worker map so the new lanes
                # get real logical processors. Pools are created per wave, so
                # the change takes effect on the wave submitted below.
                runtime_contract = configure_cpu_peak(
                    effective_cpu_target,
                    process_workers=worker_count,
                    native_threads_per_worker=1,
                    memory_limit_fraction=.95,
                    memory_soft_target_fraction=.90,
                )
                worker_map = list(runtime_contract.get("worker_map") or [])
                root_payload["requested_worker_count"] = worker_count
                root_payload["queue_capacity"] = worker_count
                root_payload["cpu_target_fraction_effective"] = (
                    effective_cpu_target)
                root_payload["runtime_contract"] = runtime_contract
                root_payload["runtime_worker_override"] = {
                    "schema_version": "DQBD_RUNTIME_LANE_OVERRIDE_V1",
                    "previous_worker_count": previous_workers,
                    "worker_count": worker_count,
                    "adopted_at_wave": wave + 1,
                    "adopted_at_epoch": time.time(),
                    "process_restarted": False,
                    "source": str(runtime_override_path),
                }
                print(
                    "[dynamic-qbd][runtime] adopted lane override "
                    f"{previous_workers} -> {worker_count} at wave {wave + 1} "
                    f"without restarting pid={os.getpid()}",
                    flush=True)
            counts, allocations = _allocate_seed_lanes(
                worker_map=worker_map,
                worker_count=min(worker_count, MAX_RUNTIME_PROCESS_LANES),
                ready_by_seed=allocation_ready_by_seed, seed_names=SEEDS)
            seed_allocations.clear()
            seed_allocations.update(allocations)
            wave += 1
            root_payload["seed_worker_allocation"] = dict(counts)
            root_payload["seed_worker_allocation_ready_jobs"] = dict(
                ready_by_seed)
            root_payload["seed_worker_allocation_wave"] = wave
            _json(root_contract, root_payload)
            _json(output_root / "_shared-compute" / "seed-scheduler-state.json", {
                "schema_version": "DQBD_SEED_LANE_SCHEDULER_V1",
                "wave": wave, "ready_by_seed": ready_by_seed,
                "allocation": counts,
                "active_seeds": list(active_names),
                "requested_worker_count": worker_count,
            })
            # Each seed scheduler owns CPU and GPU staging pools. Running all
            # three at once multiplies those pools and can exhaust RAM before
            # admission gets a chance to react. Process one ready bank at a
            # time; the selected bank still feeds both physical GPUs.
            seed_pool = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix=f"dqbd-seed-wave-{wave}")
            futures = [
                seed_pool.submit(run_seed, (name, SEEDS[name]))
                for name in active_names
            ]
            pending = set(futures)
            try:
                while pending:
                    if seed_watchdog.failure is not None:
                        raise seed_watchdog.failure
                    done, pending = wait(pending, timeout=5.0)
                    if seed_watchdog.failure is not None:
                        raise seed_watchdog.failure
                    for future in done:
                        future.result()
            finally:
                seed_pool.shutdown(wait=not pending)
        missing = [key for key, context in seed_contexts.items()
                   if not (Path(context["root"]) / "completion-manifest.json").is_file()]
        if missing:
            raise RuntimeError(f"DQBD_STEP9_SEEDS_INCOMPLETE:{missing}")
        # Economic evaluation is intentionally not opened by this coordinator.
        # A separate post-materialization audit consumes only the three complete
        # causal stores and writes the final S0/S1/S2/S3/Oracle package.
        final_ram_telemetry = ram_scheduler.telemetry()
        _json(output_root / "completion-manifest.json", {
            "status": "COMPLETE", "run_gate_hash": gate.contract_hash,
            "seeds_completed": list(SEEDS), "holdout_reads": 0,
            "evaluation_opened_only_after_all_seeds": False,
            "portfolio_batch_policy": PORTFOLIO_BATCH_POLICY,
            "runtime_seconds": time.monotonic() - started,
            "ram_scheduler": final_ram_telemetry,
        })
        return 0
    except BoundaryStopRequested:
        # The supervisor asked for a boundary stop so a validated new code
        # snapshot can take over. The scheduler already requeued its own
        # claims; every COMPLETE checkpoint stays reusable, so the next start
        # resumes exactly here instead of re-initializing the graph.
        _json(output_root / "_hotload" / "last-boundary-stop.json", {
            "schema_version": "DQBD_BOUNDARY_STOP_V1",
            "stopped_at": time.time(),
            "elapsed_seconds": time.monotonic() - started,
        })
        return BOUNDARY_STOP_EXIT_CODE
    finally:
        seed_watchdog.close()
        if "seed_pool" in locals():
            seed_pool.shutdown(
                wait=seed_watchdog.failure is None,
                cancel_futures=seed_watchdog.failure is not None,
            )
        ram_scheduler.close()
        cleanup_orphaned_slice_hash_scratch()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--signal-panel", required=True)
    parser.add_argument("--candidate-metrics", required=True)
    parser.add_argument("--prices", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--nwinfo-executable")
    parser.add_argument("--short-source-root", help="Existing verified SHORT cache to resume in place.")
    parser.add_argument("--checkpoint-root", help="Existing verified seed-root tree (SHORT/PRIMARY/LONG) to resume in place.")
    parser.add_argument(
        "--workers", type=int, default=26,
        help="Logical process-lane capacity. Keep 26 for the v40.0.4.3 execution baseline; 32 remains an explicit later host test.")
    parser.add_argument(
        "--cpu-target-fraction", type=float, default=.90,
        help="Requested CPU affinity fraction; raised automatically when --workers requires more logical lanes.")
    parser.add_argument(
        "--slow-resume", action="store_true",
        help=(
            "Force graph reconciliation plus COMPLETE-result validation. "
            "Subsequent default resumes use the validation-authority-bound "
            "v3 initialization cache."))
    parser.add_argument(
        "--direct-checkpoint-resume", action="store_true",
        help=(
            "Resume the supplied checkpoint frontier immediately without the "
            "slow graph/result initialization pass."))
    args = parser.parse_args(argv)
    return run(signal_panel=Path(args.signal_panel), candidate_metrics=Path(args.candidate_metrics),
               prices=Path(args.prices), output_root=Path(args.output_root),
               nwinfo_executable=args.nwinfo_executable,
               short_source_root=Path(args.short_source_root) if args.short_source_root else None,
               checkpoint_root=Path(args.checkpoint_root) if args.checkpoint_root else None,
               worker_count=args.workers,
               cpu_target_fraction=args.cpu_target_fraction,
               slow_resume=args.slow_resume,
               direct_checkpoint_resume=args.direct_checkpoint_resume)


if __name__ == "__main__":
    raise SystemExit(main())
