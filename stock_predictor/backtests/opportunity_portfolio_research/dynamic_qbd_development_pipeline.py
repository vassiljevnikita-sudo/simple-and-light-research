"""Canonical causal Dynamic-QBD Development pipeline.

Raw locked H1-H30 panel -> monthly production generations -> model-specific
recalibration -> generation-tagged scores -> A/B/C schedules -> authoritative
portfolio replay -> evidence -> Gate 1.  Final holdout access is fail closed.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import date
import gc
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import subprocess
import os
import shutil
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import time

import exchange_calendars as xcals
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .contract_fingerprints import stable_hash
from .dynamic_qbd_abc_schedules import build_abc_generation_schedules
from .dynamic_qbd_factory import monthly_refit_dates, refit_family
from .dynamic_qbd_algorithm_freeze import build_algorithm_manifest, write_algorithm_manifest
from .dynamic_qbd_h1_30_adapter import H130ProductionGenerationBuilder, ParquetH130DatasetMaterializer
from .dynamic_qbd_maturity import HorizonMaturityResolver
from .dynamic_qbd_model_jobs import canonical_queue_sort_key, fixed_portfolio_groups
from .dynamic_qbd_prediction_store import DiskBackedPredictionStore, MaturedPredictionStore
from .dynamic_qbd_generation_recalibration import recalibrate_generation
from .dynamic_qbd_evidence import build_monthly_market_state
from .dynamic_qbd_generation_registry import GenerationRegistry
from .dynamic_qbd_development_evaluation import run_development, tax_config_from_contract
from .dynamic_qbd_family_surface import build_family_specs
from .dynamic_qbd_runtime_resources import active_cpu_contract
from .cpu_topology import set_current_process_logical_affinity
from .dynamic_qbd_runtime_telemetry import NWInfoSampler
from .learned_exit_qbd_profit import ProfitTaxConfig
from .learned_exit_qbd_provider import LearnedExitProvider
from .learned_exit_qbd_replay import configure_replay

HOLDOUT_CONTRACTS = {"PRESERVE_HISTORICAL_LOCKBOX", "PROSPECTIVE_FROM_2026_07_25"}
DEFAULT_MODEL_WORKERS = 26

_MODEL_WORKER_SLOT = -1
_MODEL_WORKER_LOGICAL_PROCESSOR: int | None = None
_MODEL_WORKER_AFFINITY_PINNED = False
_MODEL_WORKER_LAST_FINISH = 0.0


def _model_fit_process_initializer(worker_map: list[dict], slot_counter) -> None:
    """Pin each spawned fit worker to one topology-planned CPU slot."""
    global _MODEL_WORKER_SLOT, _MODEL_WORKER_LOGICAL_PROCESSOR
    global _MODEL_WORKER_AFFINITY_PINNED, _MODEL_WORKER_LAST_FINISH
    # Model jobs are deliberately one native numerical thread per process.
    # The parent configures these before spawn; reassert them for libraries
    # imported by a spawned child before task execution.
    for name in (
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS", "POLARS_MAX_THREADS",
    ):
        os.environ[name] = "1"
    if slot_counter is None or not worker_map:
        _MODEL_WORKER_LAST_FINISH = time.perf_counter()
        return
    with slot_counter.get_lock():
        slot = int(slot_counter.value)
        slot_counter.value += 1
    _MODEL_WORKER_SLOT = slot
    row = worker_map[slot % len(worker_map)]
    _MODEL_WORKER_LOGICAL_PROCESSOR = int(row["logical_processor"])
    _MODEL_WORKER_AFFINITY_PINNED = set_current_process_logical_affinity(
        _MODEL_WORKER_LOGICAL_PROCESSOR
    )
    _MODEL_WORKER_LAST_FINISH = time.perf_counter()
    print(
        "[dynamic-qbd-model-worker] "
        f"pid={os.getpid()} slot={slot} core={row['core_index']} "
        f"logical_cpu={row['logical_processor']} role={row.get('role', 'worker')} "
        f"affinity={'OK' if _MODEL_WORKER_AFFINITY_PINNED else 'FALLBACK'}",
        flush=True,
    )


def _run_horizon_lane(payload: dict) -> dict:
    """Run one horizon's causal monthly lane in an isolated process."""
    horizon = int(payload["horizon"])
    families = tuple(x for x in payload["families"] if int(x.horizon_sessions) == horizon)
    all_families = tuple(payload["all_families"])
    signal_panel = Path(payload["signal_panel"])
    candidate_metrics = Path(payload["candidate_metrics"])
    output_root = Path(payload["output_root"])
    development_end = date.fromisoformat(payload["development_end"])
    holdout_boundary = date.fromisoformat(payload["holdout_boundary"])
    sessions = tuple(date.fromisoformat(x) for x in payload["sessions"])
    maturity = HorizonMaturityResolver(tuple(x for x in sessions if x < holdout_boundary))
    materializer = ParquetH130DatasetMaterializer(
        signal_panel, candidate_metrics, development_end,
        Path(payload["learned_exit_candidate_metrics"]) if payload.get("learned_exit_candidate_metrics") else None,
    )
    global_registry_path = output_root / "factory" / "generation-registry.json"
    shard_path = output_root / "factory" / "horizon-registries" / f"H{horizon:02d}.json"
    shard_path.parent.mkdir(parents=True, exist_ok=True)
    if not shard_path.is_file() and global_registry_path.is_file():
        # Resume only this horizon's state.  Copying the complete global
        # registry makes every lane carry all historical records and also
        # leaves stale current-generation entries that can overwrite another
        # lane during the final merge.
        global_registry = GenerationRegistry(global_registry_path, all_families)
        shard = GenerationRegistry(shard_path, all_families)
        shard.records = {
            generation_id: generation
            for generation_id, generation in global_registry.records.items()
            if int(generation.family_id[1:3]) == horizon
        }
        shard.current = {
            family_id: generation_id
            for family_id, generation_id in global_registry.current.items()
            if int(family_id[1:3]) == horizon
        }
        shard.save()
    registry = GenerationRegistry(shard_path, all_families, autosave=False)
    # Older runs may contain a full-registry shard.  Migrate it in place to
    # the horizon-local representation before resuming; generation artifacts
    # remain reusable and only the redundant registry envelope is removed.
    local_records = {
        generation_id: generation
        for generation_id, generation in registry.records.items()
        if int(generation.family_id[1:3]) == horizon
    }
    local_current = {
        family_id: generation_id
        for family_id, generation_id in registry.current.items()
        if int(family_id[1:3]) == horizon
    }
    if len(local_records) != len(registry.records) or local_current != registry.current:
        registry.records = local_records
        registry.current = local_current
        registry.save()
    builder = H130ProductionGenerationBuilder(
        materializer=materializer, root=output_root / "factory" / "generations",
        code_commit=str(payload["code_commit"]),
    )
    for generation in registry.records.values():
        if generation.lifecycle_status.value == "VALID" and int(generation.family_id[1:3]) == horizon:
            builder.restore_generation(generation)
    lane_records = 0
    checkpoint_counter = 0
    non_fixed_families = []
    for family in families:
        if str(family.exit_policy.get("family", "FIXED")).upper() != "FIXED":
            non_fixed_families.append(family)

    # Fixed-exit model identity is H×cutoff×recipe.  Holding period D and
    # capacity N are portfolio dimensions and must not trigger another fit.
    model_groups = {group[0].family_id: tuple(group) for group in fixed_portfolio_groups(families).values()}
    model_representatives = [group[0] for group in model_groups.values()]
    for family in [x for x in non_fixed_families] + model_representatives:
        portfolio_group = model_groups.get(family.family_id, (family,))
        cadence = str(family.refit_cadence).upper()
        refit_dates = tuple(x for x in monthly_refit_dates(sessions, day=cadence) if x >= date.fromisoformat(payload["start"]))
        prior_valid_cutoffs = [record.information_cutoff
                               for record in registry.records.values()
                               if record.family_id in {x.family_id for x in portfolio_group}
                               and record.lifecycle_status.value == "VALID"]
        first_valid_cutoff = min(prior_valid_cutoffs) if prior_valid_cutoffs else None
        for cutoff_index, cutoff in enumerate(refit_dates):
            if maturity.latest_matured_decision(cutoff, family.horizon_sessions) is None:
                # No model job exists before the first fully matured label.
                # The migration manifest records this as a horizon-level
                # methodological reject; do not create repeated D/N rejects.
                continue
            existing = next((registry.generation_for_family_cutoff(group_family.family_id, cutoff)
                             for group_family in portfolio_group
                             if registry.generation_for_family_cutoff(group_family.family_id, cutoff) is not None), None)
            if existing is not None:
                base = existing
            else:
                next_cutoff = refit_dates[cutoff_index + 1] if cutoff_index + 1 < len(refit_dates) else None
                base = refit_family(family=family, information_cutoff=cutoff, maturity=maturity,
                                    builder=builder, registry=registry,
                                    prediction_end=(next_cutoff if first_valid_cutoff is not None and next_cutoff is not None
                                                    else development_end))
                lane_records += 1
                checkpoint_counter += 1
            if base is not None and base.lifecycle_status.value == "VALID" and first_valid_cutoff is None:
                first_valid_cutoff = cutoff
            if family.family_id in model_groups and base is not None and base.lifecycle_status.value == "VALID":
                for portfolio_family in model_groups[family.family_id]:
                    if registry.generation_for_family_cutoff(portfolio_family.family_id, cutoff) is not None:
                        continue
                    clone_id = stable_hash({"canonical_model_generation": base.generation_id,
                                            "portfolio_family": portfolio_family.family_id})[:24]
                    clone = replace(base, generation_id=clone_id, family_id=portfolio_family.family_id,
                                    exit_policy_fingerprint=stable_hash(portfolio_family.exit_policy))
                    registry.register(clone)
                    checkpoint_counter += 1
            if checkpoint_counter >= 25:
                registry.checkpoint()
                checkpoint_counter = 0
    registry.checkpoint()
    return {"horizon": horizon, "families": len(families), "records_written": lane_records}


def _run_canonical_model_job(payload: dict):
    """Execute one independent H×cutoff fixed-exit model job.

    The worker returns the generation to the coordinator; only the
    coordinator mutates the horizon registry. This gives all Horizons one
    global queue without introducing concurrent writers for one shard.
    """
    global _MODEL_WORKER_LAST_FINISH
    family = payload["family"]
    output_root = Path(payload["output_root"])
    signal_panel = Path(payload["signal_panel"])
    candidate_metrics = Path(payload["candidate_metrics"])
    cutoff = date.fromisoformat(payload["cutoff"])
    latest = date.fromisoformat(payload["latest_matured"])
    development_end = date.fromisoformat(payload["development_end"])
    materializer = ParquetH130DatasetMaterializer(signal_panel, candidate_metrics, development_end)
    builder = H130ProductionGenerationBuilder(
        materializer=materializer,
        root=output_root / "factory" / "generations",
        code_commit=str(payload["code_commit"]),
    )
    frozen_choice = payload.get("frozen_choice")
    if frozen_choice:
        builder._frozen_choice_by_family[family.family_id] = (cutoff, dict(frozen_choice))
    maturity = HorizonMaturityResolver(tuple(date.fromisoformat(x) for x in payload["sessions"]
                                              if date.fromisoformat(x) < date.fromisoformat(payload["holdout_boundary"])))
    started = time.perf_counter()
    idle_seconds = max(0.0, started - _MODEL_WORKER_LAST_FINISH)
    if _MODEL_WORKER_LOGICAL_PROCESSOR is not None:
        # Re-assert the binding before every task so the telemetry proves the
        # task was run on its assigned slot even if an external tool changed
        # process affinity between jobs.
        affinity_pinned = set_current_process_logical_affinity(
            _MODEL_WORKER_LOGICAL_PROCESSOR
        )
    else:
        affinity_pinned = False
    generation = refit_family(
        family=family, information_cutoff=cutoff, maturity=maturity, builder=builder, registry=None,
        prediction_end=(date.fromisoformat(payload["prediction_end"]) if payload.get("prediction_end") else development_end),
    )
    finished = time.perf_counter()
    _MODEL_WORKER_LAST_FINISH = finished
    return {
        "generation": generation,
        "telemetry": {
            "event": "dynamic_qbd_model_fit_worker_task",
            "pid": int(os.getpid()),
            "worker_slot": int(_MODEL_WORKER_SLOT),
            "logical_processor": _MODEL_WORKER_LOGICAL_PROCESSOR,
            "affinity_pinned": bool(affinity_pinned),
            "idle_seconds": float(idle_seconds),
            "compute_seconds": float(finished - started),
            "family_id": str(family.family_id),
            "information_cutoff": str(cutoff),
        },
    }


def _run_global_canonical_model_queue(*, families, all_families, signal_panel, candidate_metrics,
                                      output_root, sessions, development_end, holdout_start,
                                      start, code_fingerprint, model_workers, registry):
    """Run one globally balanced queue of independent fixed H×cutoff jobs."""
    groups = {group[0].family_id: tuple(group) for group in fixed_portfolio_groups(families).values()}
    if not groups:
        return registry
    registry.autosave = False
    materializer = ParquetH130DatasetMaterializer(signal_panel, candidate_metrics, development_end)
    maturity = HorizonMaturityResolver(tuple(x for x in sessions if x < holdout_start))
    jobs = []
    for representative_id, portfolio_group in groups.items():
        representative = portfolio_group[0]
        refit_dates = tuple(x for x in monthly_refit_dates(sessions, day=str(representative.refit_cadence).upper())
                            if start <= x <= development_end)
        bootstrap_cutoff = None
        frozen_choice = None
        bootstrap_builder = H130ProductionGenerationBuilder(
            materializer=materializer, root=output_root / "factory" / "generations",
            code_commit=code_fingerprint)
        for cutoff in refit_dates:
            latest = maturity.latest_matured_decision(cutoff, representative.horizon_sessions)
            if latest is None:
                continue
            try:
                frozen_choice = bootstrap_builder._select_candidate(representative, candidate_metrics, latest)
                bootstrap_cutoff = cutoff
                break
            except ValueError:
                continue
        if bootstrap_cutoff is None:
            continue
        for index, cutoff in enumerate(refit_dates):
            if cutoff < bootstrap_cutoff:
                continue
            if any(registry.generation_for_family_cutoff(portfolio_family.family_id, cutoff) is not None
                   for portfolio_family in portfolio_group):
                continue
            latest = maturity.latest_matured_decision(cutoff, representative.horizon_sessions)
            if latest is None:
                continue
            next_cutoff = refit_dates[index + 1] if index + 1 < len(refit_dates) else None
            jobs.append({
                "family": representative,
                "signal_panel": str(signal_panel), "candidate_metrics": str(candidate_metrics),
                "output_root": str(output_root), "cutoff": str(cutoff), "latest_matured": str(latest),
                "development_end": str(development_end),
                "prediction_end": str(development_end if cutoff == bootstrap_cutoff or next_cutoff is None else next_cutoff),
                "holdout_boundary": str(holdout_start), "sessions": [str(x) for x in sessions],
                "code_commit": code_fingerprint, "frozen_choice": frozen_choice,
            })
    jobs.sort(key=canonical_queue_sort_key)
    workers = max(1, min(int(model_workers), len(jobs) or 1))
    if jobs:
        # Keep twice as many jobs queued as workers so uneven H×cutoff fit
        # durations do not starve a worker.  Results are committed as soon as
        # they complete; the queue order remains deterministic because the
        # submitted jobs are canonically sorted.
        queue_capacity = max(workers, workers * 2)
        worker_map = list(active_cpu_contract().get("worker_map") or [])[:workers]
        if len(worker_map) != workers:
            raise RuntimeError(
                f"DYNAMIC_QBD_WORKER_AFFINITY_PLAN_INCOMPLETE:{len(worker_map)}:{workers}"
            )
        telemetry_path = output_root / "telemetry" / "model-fit-workers.jsonl"
        telemetry_path.parent.mkdir(parents=True, exist_ok=True)
        spawn_context = mp.get_context("spawn")
        slot_counter = spawn_context.Value("i", 0, lock=True)

        def submit_next(pool, pending):
            nonlocal next_index
            while next_index < len(jobs) and len(pending) < queue_capacity:
                future = pool.submit(_run_canonical_model_job, jobs[next_index])
                pending[future] = next_index
                next_index += 1

        pending = {}
        next_index = 0
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=spawn_context,
            initializer=_model_fit_process_initializer,
            initargs=(worker_map, slot_counter),
        ) as pool:
            submit_next(pool, pending)
            while pending:
                done, _ = wait(tuple(pending), return_when=FIRST_COMPLETED)
                for future in done:
                    job_index = pending.pop(future)
                    result = future.result()
                    generation = result["generation"]
                    worker_telemetry = dict(result.get("telemetry") or {})
                    worker_telemetry["job_index"] = int(job_index)
                    worker_telemetry["queue_depth_after_completion"] = int(len(pending))
                    with telemetry_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(worker_telemetry, sort_keys=True) + "\n")
                    registry.register(generation)
                    group = groups.get(generation.family_id, (generation.family_id,))
                    if generation.lifecycle_status.value == "VALID":
                        for portfolio_family in group:
                            if registry.generation_for_family_cutoff(
                                portfolio_family.family_id, generation.information_cutoff
                            ) is not None:
                                continue
                            clone_id = stable_hash({
                                "canonical_model_generation": generation.generation_id,
                                "portfolio_family": portfolio_family.family_id,
                            })[:24]
                            registry.register(replace(
                                generation,
                                generation_id=clone_id,
                                family_id=portfolio_family.family_id,
                                exit_policy_fingerprint=stable_hash(portfolio_family.exit_policy),
                            ))
                    if len(registry.records) % 25 == 0:
                        registry.checkpoint()
                submit_next(pool, pending)
    registry.checkpoint()
    return registry


def _git_sha() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()


def _git_code_identity() -> dict:
    head=_git_sha()
    diff=subprocess.run(["git","diff","--binary","--no-ext-diff"],check=True,capture_output=True).stdout
    source_root=Path(__file__).resolve().parent
    source_files=sorted(source_root.glob("dynamic_qbd*.py")) + [source_root/"next_open_portfolio_replay.py",source_root/"learned_exit_qbd_replay.py"]
    digest=hashlib.sha256()
    for path in source_files:
        digest.update(path.name.encode("utf-8")); digest.update(path.read_bytes())
    source_status=subprocess.run(
        ["git","status","--porcelain","--untracked-files=all","--",
         str(source_root.relative_to(Path.cwd()))],check=True,capture_output=True,text=True,
    ).stdout
    return {"commit":head,"tracked_diff_sha256":hashlib.sha256(diff).hexdigest(),
            "source_tree_sha256":digest.hexdigest(),"clean":not bool(source_status.strip())}


def _write_atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(payload,default=str,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    os.replace(temp,path)


def _sha256_file(path: Path) -> str:
    digest=hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b""):
            digest.update(chunk)
    return digest.hexdigest()


def _semantic_contract(payload: dict) -> dict:
    """Strip execution-only provenance from a run contract for resume checks."""
    execution_only = {"git_sha", "code_identity", "model_workers", "execution_provenance",
                      "semantic_run_contract_hash", "run_contract_sha256"}
    return {key: value for key, value in payload.items() if key not in execution_only}


def _reuse_verified_completed_run(output_root: Path, run_contract: dict) -> dict | None:
    manifest_path=output_root/"manifest.json"
    summary_path=output_root/"pipeline-summary.json"
    if not manifest_path.is_file() or not summary_path.is_file():
        return None
    manifest=json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("run_contract_sha256") != run_contract["run_contract_sha256"]:
        return None
    root=output_root.resolve(); hash_cache={}
    for relative,expected in manifest.get("artifact_sha256",{}).items():
        path=(root/relative).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"DYNAMIC_QBD_MANIFEST_PATH_ESCAPE:{relative}") from exc
        if not path.is_file():
            raise FileNotFoundError(f"DYNAMIC_QBD_COMPLETED_RUN_ARTIFACT_MISSING:{relative}")
        stat=path.stat(); identity=(stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns)
        actual=hash_cache.get(identity)
        if actual is None:
            actual=_sha256_file(path); hash_cache[identity]=actual
        if actual != expected:
            raise ValueError(f"DYNAMIC_QBD_COMPLETED_RUN_ARTIFACT_HASH_MISMATCH:{relative}")
    result=json.loads(summary_path.read_text(encoding="utf-8"))
    if stable_hash({key:value for key,value in result.items() if key!="pipeline_fingerprint"}) != result.get("pipeline_fingerprint"):
        raise ValueError("DYNAMIC_QBD_PIPELINE_SUMMARY_HASH_MISMATCH")
    return result | {"completed_run_reused":True,"verified_artifact_count":len(manifest.get("artifact_sha256",{}))}


def _hardlink_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True,exist_ok=True)
    if destination.exists():
        destination.unlink()
    try:
        os.link(source,destination)
    except OSError:
        import shutil
        shutil.copy2(source,destination)


def _prediction_artifact_cache_key(sources: list[tuple[Path, str]]) -> str:
    records = []
    for path, model_artifact_id in sources:
        stat = path.stat()
        records.append({"path": str(path.resolve()), "size": stat.st_size,
                        "mtime_ns": stat.st_mtime_ns,
                        "model_artifact_id": str(model_artifact_id)})
    return stable_hash({"schema_version": "DYNAMIC_QBD_PREDICTION_STORE_V4_REGISTRY_TAGGED_SOURCES",
                        "columns": list(DiskBackedPredictionStore.columns),
                        "sources": records})


def _unique_prediction_artifact_sources(generations: pd.DataFrame) -> list[tuple[Path, str]]:
    """Select one immutable source per registry model identity.

    The fixed-N surface can reference the same model artifact from several
    portfolio families.  Those generations carry separate, byte-identical
    prediction paths.  Concatenating every alias duplicates each authoritative
    score and makes replay ambiguous.  A conflicting content hash is an
    integrity error; identical aliases are safely represented once.  Some
    factory aliases also contain a stale embedded model tag; the registry
    identity is authoritative and the streaming writer repairs that tag in the
    canonical store without changing any score value.
    """
    required = {"model_artifact_id", "prediction_artifact_path"}
    missing = required - set(generations)
    if missing:
        raise ValueError(f"DYNAMIC_QBD_PREDICTION_SOURCE_COLUMNS_MISSING:{sorted(missing)}")
    source = generations.loc[:, [
        "model_artifact_id", "prediction_artifact_path",
        *(["prediction_artifact_sha256"] if "prediction_artifact_sha256" in generations else []),
    ]].dropna(subset=["model_artifact_id", "prediction_artifact_path"]).copy()
    selected: dict[str, tuple[Path, str]] = {}
    for model_id, group in source.groupby(source["model_artifact_id"].astype(str), sort=True):
        paths = sorted({Path(str(value)) for value in group["prediction_artifact_path"]})
        if not paths:
            continue
        if "prediction_artifact_sha256" in group:
            hashes = {str(value) for value in group["prediction_artifact_sha256"].dropna()
                      if str(value)}
            if len(hashes) > 1:
                raise ValueError(f"DYNAMIC_QBD_PREDICTION_ARTIFACT_CONFLICT:{model_id}")
        selected[model_id] = (paths[0], model_id)
    return sorted(selected.values(), key=lambda item: (str(item[0]), item[1]))


def _ensure_streamed_prediction_store(*, sources: list[tuple[Path, str]], destination: Path) -> DiskBackedPredictionStore:
    """Create/reuse the canonical prediction parquet without a global concat."""
    if not sources:
        raise RuntimeError("DYNAMIC_QBD_PREDICTION_ARTIFACTS_EMPTY")
    cache_key = _prediction_artifact_cache_key(sources)
    metadata_path = destination.with_name(destination.name + ".cache.json")
    expected_rows = sum(int(pq.ParquetFile(path).metadata.num_rows) for path, _ in sources)
    if destination.is_file() and metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        parquet = pq.ParquetFile(destination)
        schema_ok = set(DiskBackedPredictionStore.columns).issubset(set(parquet.schema_arrow.names))
        row_count = int(parquet.metadata.num_rows)
        parquet.close()
        if (metadata.get("cache_key") == cache_key and
                int(metadata.get("rows", -1)) == expected_rows and row_count == expected_rows and schema_ok):
            print(f"[cache] model predictions hit rows={expected_rows}", flush=True)
            return DiskBackedPredictionStore(destination, row_group_cache_size=0)
    elif destination.is_file():
        # The interrupted pre-streaming run already published this canonical
        # file.  Validate its immutable shape once, then adopt it by writing
        # the new cache receipt instead of rebuilding valid factory output.
        parquet = pq.ParquetFile(destination)
        schema_ok = set(DiskBackedPredictionStore.columns).issubset(set(parquet.schema_arrow.names))
        row_count = int(parquet.metadata.num_rows)
        parquet.close()
        if schema_ok and row_count == expected_rows:
            _write_atomic_json(metadata_path, {"schema_version": "DYNAMIC_QBD_PREDICTION_STORE_V4_REGISTRY_TAGGED_SOURCES",
                                                "cache_key": cache_key, "rows": row_count,
                                                "adopted_existing": True})
            print(f"[cache] adopted validated model predictions rows={row_count}", flush=True)
            return DiskBackedPredictionStore(destination, row_group_cache_size=0)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + f".{os.getpid()}.tmp")
    writer = None
    rows = 0
    try:
        repaired_tags = 0
        for source_path, expected_model_artifact_id in sources:
            source = pq.ParquetFile(source_path)
            for row_group in range(source.num_row_groups):
                table = source.read_row_group(row_group, columns=list(DiskBackedPredictionStore.columns))
                model_column = table.column("model_artifact_id")
                embedded_ids = {str(value) for value in model_column.unique().to_pylist() if value is not None}
                if embedded_ids != {str(expected_model_artifact_id)}:
                    if len(embedded_ids) > 1:
                        raise ValueError(f"DYNAMIC_QBD_PREDICTION_EMBEDDED_ID_CONFLICT:{source_path}")
                    replacement = pa.array([str(expected_model_artifact_id)] * table.num_rows,
                                           type=model_column.type)
                    table = table.set_column(table.schema.get_field_index("model_artifact_id"),
                                             "model_artifact_id", replacement)
                    repaired_tags += 1
                if writer is None:
                    writer = pq.ParquetWriter(temporary, table.schema, compression="zstd")
                writer.write_table(table)
                rows += table.num_rows
        if writer is None or rows != expected_rows:
            raise ValueError("DYNAMIC_QBD_PREDICTION_STREAM_ROWCOUNT_MISMATCH")
        writer.close()
        writer = None
        os.replace(temporary, destination)
        _write_atomic_json(metadata_path, {"schema_version": "DYNAMIC_QBD_PREDICTION_STORE_V4_REGISTRY_TAGGED_SOURCES",
                                            "cache_key": cache_key, "rows": rows,
                                            "repaired_embedded_model_tag_row_groups": repaired_tags})
    finally:
        if writer is not None:
            writer.close()
        if temporary.exists():
            temporary.unlink()
    return DiskBackedPredictionStore(destination, row_group_cache_size=0)


def _sessions_and_holdout(panel_path: Path) -> tuple[tuple[date, ...], date | None]:
    columns = ["decision_date", "holdout_locked"]
    frame = pd.read_parquet(panel_path, columns=columns).drop_duplicates()
    frame["decision_date"] = pd.to_datetime(frame["decision_date"]).dt.date
    locked = frame.loc[frame["holdout_locked"].astype(bool), "decision_date"] if "holdout_locked" in frame else pd.Series(dtype=object)
    return tuple(sorted(frame["decision_date"].unique())), (min(locked) if len(locked) else None)


def _holdout_boundary(contract: str, historical_start: date | None) -> date:
    if contract not in HOLDOUT_CONTRACTS:
        raise ValueError("EXPLICIT_DYNAMIC_QBD_HOLDOUT_CONTRACT_REQUIRED")
    if contract == "PRESERVE_HISTORICAL_LOCKBOX":
        if historical_start is None:
            raise ValueError("HISTORICAL_LOCKBOX_REQUESTED_BUT_DATASET_HAS_NONE")
        return historical_start
    return date(2026, 7, 25)


def materialize_daily_store_prices(*, daily_store_root: str | Path, signal_panel: str | Path,
                                   start: date, end: date, output_path: str | Path,
                                   benchmark_daily_path: str | Path | None = None,
                                   direct_daily_stock_root: str | Path | None = None) -> Path:
    """Project the canonical partitioned daily store into the replay price contract."""
    daily_store_root=Path(daily_store_root); signal_panel=Path(signal_panel); output_path=Path(output_path)
    universe=pd.read_parquet(signal_panel,columns=["decision_date","ticker"])
    universe["decision_date"]=pd.to_datetime(universe["decision_date"])
    tickers=set(universe.loc[universe["decision_date"].between(pd.Timestamp(start),pd.Timestamp(end)),"ticker"].astype(str))
    tickers.add("URTH")
    benchmark_daily_path=Path(benchmark_daily_path) if benchmark_daily_path is not None else None
    direct_daily_stock_root=Path(direct_daily_stock_root) if direct_daily_stock_root is not None else None
    if benchmark_daily_path is not None and not benchmark_daily_path.is_file():
        raise FileNotFoundError(f"EXPLICIT_BENCHMARK_DAILY_NOT_FOUND:{benchmark_daily_path}")
    if benchmark_daily_path is not None:
        benchmark_manifest_path=benchmark_daily_path.with_suffix(benchmark_daily_path.suffix+".manifest.json")
        if not benchmark_manifest_path.is_file():
            raise FileNotFoundError("EXPLICIT_BENCHMARK_DAILY_MANIFEST_MISSING")
        benchmark_manifest=json.loads(benchmark_manifest_path.read_text(encoding="utf-8"))
        expected_manifest={"schema_version":"ALPACA_DIRECT_DAILY_REPLAY_V1","feed":"sip","timeframe":"1Day",
                           "usage":"EXPLICIT_REPLAY_BENCHMARK_DAILY_OPEN_CLOSE","ticker":"URTH"}
        if any(benchmark_manifest.get(key)!=value for key,value in expected_manifest.items()):
            raise ValueError("EXPLICIT_BENCHMARK_DAILY_MANIFEST_CONTRACT_MISMATCH")
        if benchmark_manifest.get("output_sha256")!=_sha256_file(benchmark_daily_path):
            raise ValueError("EXPLICIT_BENCHMARK_DAILY_HASH_MISMATCH")
    if direct_daily_stock_root is not None:
        tree_manifest_path=direct_daily_stock_root/"manifest.json"
        if not tree_manifest_path.is_file(): raise FileNotFoundError("DIRECT_DAILY_STOCK_TREE_MANIFEST_MISSING")
        tree_manifest=json.loads(tree_manifest_path.read_text(encoding="utf-8"))
        if (tree_manifest.get("schema_version")!="ALPACA_DIRECT_DAILY_TREE_V1"
                or tree_manifest.get("usage")!="ROW_LEVEL_FALLBACK_FOR_INVALID_MINUTE_EXECUTION_BOUNDARIES"):
            raise ValueError("DIRECT_DAILY_STOCK_TREE_CONTRACT_MISMATCH")
    partitions=[]; missing=[]
    for ticker in sorted(tickers):
        if ticker=="URTH" and benchmark_daily_path is not None:
            partitions.append((ticker,benchmark_daily_path,"ALPACA_SIP_1DAY_DIRECT"))
            continue
        direct=daily_store_root/"schema=v1"/"source=alpaca"/f"ticker={ticker}"/"variant=sip_split"/"bars.parquet"
        candidates=[direct] if direct.is_file() else sorted(
            daily_store_root.glob(f"schema=*/source=*/ticker={ticker}/variant=*/bars.parquet"),key=lambda path:str(path.resolve()))
        if not candidates:
            missing.append(ticker); continue
        if len(candidates)>1:
            raise RuntimeError(f"AMBIGUOUS_DAILY_PRICE_PROVIDER:{ticker}:{[str(x) for x in candidates]}")
        partitions.append((ticker,candidates[0],"MINUTE_DERIVED_DAILY"))
    if missing:
        raise FileNotFoundError(f"DYNAMIC_QBD_DAILY_PRICES_MISSING:{missing[:20]}:COUNT={len(missing)}")
    source_identity=[]
    for ticker,path,provider in partitions:
        stat=path.stat()
        manifest_path=path.with_suffix(path.suffix+".manifest.json")
        source_identity.append((ticker,provider,str(path.resolve()),stat.st_size,stat.st_mtime_ns,
                                _sha256_file(path),
                                _sha256_file(manifest_path) if manifest_path.is_file() else None))
        if ticker!="URTH" and direct_daily_stock_root is not None:
            fallback=direct_daily_stock_root/f"ticker={ticker}"/"bars.parquet"
            fallback_manifest=fallback.with_suffix(fallback.suffix+".manifest.json")
            if not fallback.is_file() or not fallback_manifest.is_file():
                raise FileNotFoundError(f"DIRECT_DAILY_STOCK_FALLBACK_MISSING:{ticker}")
            fallback_payload=json.loads(fallback_manifest.read_text(encoding="utf-8"))
            if (fallback_payload.get("ticker")!=ticker
                    or fallback_payload.get("usage")!="ROW_LEVEL_FALLBACK_FOR_INVALID_MINUTE_EXECUTION_BOUNDARIES"
                    or fallback_payload.get("output_sha256")!=_sha256_file(fallback)):
                raise ValueError(f"DIRECT_DAILY_STOCK_FALLBACK_CONTRACT_MISMATCH:{ticker}")
            source_identity.append((ticker,"ALPACA_SIP_1DAY_ROW_FALLBACK",str(fallback.resolve()),
                                    fallback.stat().st_size,_sha256_file(fallback),_sha256_file(fallback_manifest)))
    cache_contract={"schema_version":"DYNAMIC_QBD_PRICE_PROJECTION_V5_ROW_LEVEL_STOCK_FALLBACK",
                    "daily_store_root":str(daily_store_root.resolve()),"start":str(start),"end":str(end),
                    "ticker_count":len(tickers),"source_identity_sha256":stable_hash(source_identity),
                    "benchmark_daily_path":str(benchmark_daily_path.resolve()) if benchmark_daily_path else None,
                    "direct_daily_stock_root":str(direct_daily_stock_root.resolve()) if direct_daily_stock_root else None,
                    "minute_boundary_rule":{"open_delay_minutes":[0,5],"absolute_close_offset_minutes":5},
                    "benchmark_contract":"ALPACA_SIP_1DAY_DIRECT" if benchmark_daily_path else "MINUTE_DERIVED_DAILY"}
    cache_path=output_path.with_suffix(output_path.suffix+".contract.json")
    quality_path=output_path.with_suffix(output_path.suffix+".quality.json")
    boundary_comparison_path=output_path.with_suffix(output_path.suffix+".benchmark-boundary-comparison.json")
    if output_path.is_file() and cache_path.is_file():
        cached=json.loads(cache_path.read_text(encoding="utf-8"))
        comparison_ready=(benchmark_daily_path is None or boundary_comparison_path.is_file())
        if (cached==cache_contract and quality_path.is_file() and comparison_ready
                and {"date","ticker","open","close","open_quality_ok","close_quality_ok"} <= set(pq.ParquetFile(output_path).schema_arrow.names)):
            return output_path
    rows=[]
    for ticker,path,provider in partitions:
        available=set(pq.ParquetFile(path).schema_arrow.names)
        projected=[name for name in ("session_date","ticker","open","close","volume","minute_count",
                                      "first_timestamp_ms","last_timestamp_ms") if name in available]
        frame=pd.read_parquet(path,columns=projected)
        frame=frame.rename(columns={"session_date":"date"})
        frame["date"]=pd.to_datetime(frame["date"])
        frame["price_boundary_contract"]=provider
        rows.append(frame.loc[frame["date"].between(pd.Timestamp(start),pd.Timestamp(end))])
    result=pd.concat(rows,ignore_index=True).sort_values(["date","ticker"])
    if result.duplicated(["date","ticker"]).any():
        raise ValueError("DYNAMIC_QBD_DUPLICATE_DAILY_PRICE_ROWS")
    minute_mask=result["price_boundary_contract"].eq("MINUTE_DERIVED_DAILY")
    if minute_mask.any() and not {"first_timestamp_ms","last_timestamp_ms"} <= set(result):
        raise ValueError("DYNAMIC_QBD_EXECUTION_BOUNDARY_TIMESTAMPS_MISSING")
    for column in ("first_timestamp_ms","last_timestamp_ms"):
        if column not in result:
            result[column]=pd.NA
    calendar=xcals.get_calendar("XNYS")
    schedule=calendar.schedule.loc[str(start):str(end),["open","close"]].copy().rename(
        columns={"open":"expected_market_open","close":"expected_market_close"})
    schedule["date"]=pd.to_datetime(schedule.index).tz_localize(None)
    boundaries=result.merge(schedule,on="date",how="left",validate="many_to_one")
    first=pd.to_datetime(boundaries.get("first_timestamp_ms"),unit="ms",utc=True)
    last=pd.to_datetime(boundaries.get("last_timestamp_ms"),unit="ms",utc=True)
    expected_open=pd.to_datetime(boundaries["expected_market_open"],utc=True)
    expected_last=pd.to_datetime(boundaries["expected_market_close"],utc=True)-pd.Timedelta(minutes=1)
    boundaries["open_delay_minutes"]=(first-expected_open).dt.total_seconds()/60.0
    boundaries["close_lead_minutes"]=(expected_last-last).dt.total_seconds()/60.0
    boundaries["open_quality_ok"]=boundaries["open_delay_minutes"].between(0.0,5.0,inclusive="both")
    # Some vendors timestamp the final early-close bar at the close boundary
    # itself (for example 13:00) rather than at the minute start (12:59).
    boundaries["close_quality_ok"]=boundaries["close_lead_minutes"].abs().le(5.0)
    boundaries["open_provider_contract"]=boundaries["price_boundary_contract"]
    boundaries["close_provider_contract"]=boundaries["price_boundary_contract"]
    direct_daily=boundaries["price_boundary_contract"].eq("ALPACA_SIP_1DAY_DIRECT")
    boundaries.loc[direct_daily,["open_quality_ok","close_quality_ok"]]=True
    repaired_open=0; repaired_close=0
    if direct_daily_stock_root is not None:
        fallback_rows=[]
        for ticker,_,_ in partitions:
            if ticker=="URTH": continue
            fallback=direct_daily_stock_root/f"ticker={ticker}"/"bars.parquet"
            frame=pd.read_parquet(fallback,columns=["session_date","ticker","open","close"]).rename(
                columns={"session_date":"date","open":"fallback_open","close":"fallback_close"})
            frame["date"]=pd.to_datetime(frame["date"])
            fallback_rows.append(frame.loc[frame["date"].between(pd.Timestamp(start),pd.Timestamp(end))])
        fallback_frame=pd.concat(fallback_rows,ignore_index=True)
        boundaries=boundaries.merge(fallback_frame,on=["date","ticker"],how="left",validate="one_to_one")
        open_repair=(~boundaries["open_quality_ok"] & boundaries["fallback_open"].notna())
        close_repair=(~boundaries["close_quality_ok"] & boundaries["fallback_close"].notna())
        boundaries.loc[open_repair,"open"]=boundaries.loc[open_repair,"fallback_open"]
        boundaries.loc[close_repair,"close"]=boundaries.loc[close_repair,"fallback_close"]
        boundaries.loc[open_repair,"open_quality_ok"]=True
        boundaries.loc[close_repair,"close_quality_ok"]=True
        boundaries.loc[open_repair,"open_provider_contract"]="ALPACA_SIP_1DAY_INVALID_MINUTE_FALLBACK"
        boundaries.loc[close_repair,"close_provider_contract"]="ALPACA_SIP_1DAY_INVALID_MINUTE_FALLBACK"
        repaired_open=int(open_repair.sum()); repaired_close=int(close_repair.sum())
        boundaries=boundaries.drop(columns=["fallback_open","fallback_close"])
    result=boundaries
    benchmark=result.loc[result["ticker"].eq("URTH")]
    provider_contract_valid=bool(len(benchmark)) and bool(
        benchmark["price_boundary_contract"].eq("ALPACA_SIP_1DAY_DIRECT").all())
    quality_summary={"schema_version":"DYNAMIC_QBD_PRICE_BOUNDARY_QUALITY_V3","start":str(start),"end":str(end),
        "open_tolerance_minutes":5,"close_tolerance_minutes":5,"ticker_count":int(result["ticker"].nunique()),
        "rows":int(len(result)),"benchmark":"URTH","benchmark_sessions":int(len(benchmark)),
        "benchmark_provider_contract_valid":provider_contract_valid,
        "benchmark_boundary_quality_empirically_measured":False if benchmark_daily_path else True,
        "benchmark_invalid_open_dates":[str(x.date()) for x in benchmark.loc[~benchmark["open_quality_ok"],"date"]],
        "benchmark_invalid_close_dates":[str(x.date()) for x in benchmark.loc[~benchmark["close_quality_ok"],"date"]],
        "benchmark_provider_contract":str(benchmark["price_boundary_contract"].iloc[0]) if len(benchmark) else None,
        "stock_invalid_minute_open_rows_repaired_from_direct_daily":repaired_open,
        "stock_invalid_minute_close_rows_repaired_from_direct_daily":repaired_close,
        "replay_contract":"FAIL_ON_ANY_INVALID_BENCHMARK_CLOSE_OR_ANY_TRADED_OPEN/CLOSE"}
    output_path.parent.mkdir(parents=True,exist_ok=True)
    result.to_parquet(output_path,index=False)
    _write_atomic_json(cache_path,cache_contract)
    _write_atomic_json(quality_path,quality_summary)
    if benchmark_daily_path is not None:
        minute_urth=daily_store_root/"schema=v1"/"source=alpaca"/"ticker=URTH"/"variant=sip_split"/"bars.parquet"
        if not minute_urth.is_file(): raise FileNotFoundError("URTH_MINUTE_DERIVED_PROVIDER_MISSING_FOR_CROSS_AUDIT")
        minute=pd.read_parquet(minute_urth,columns=["session_date","open","close"]).rename(
            columns={"session_date":"date","open":"minute_open","close":"minute_close"})
        minute["date"]=pd.to_datetime(minute["date"])
        comparison=benchmark[["date","open","close"]].rename(columns={"open":"direct_open","close":"direct_close"}).merge(
            minute,on="date",how="inner",validate="one_to_one")
        def difference_metrics(field):
            values=(comparison[f"direct_{field}"]/comparison[f"minute_{field}"]-1.0)*10000.0
            absolute=values.abs()
            return {"count":int(len(values)),"median_difference_bps":float(values.median()),
                    "median_absolute_difference_bps":float(absolute.median()),
                    "p95_absolute_difference_bps":float(absolute.quantile(.95)),
                    "p99_absolute_difference_bps":float(absolute.quantile(.99)),
                    "max_absolute_difference_bps":float(absolute.max()),
                    "count_absolute_gt_5_bps":int(absolute.gt(5).sum()),
                    "count_absolute_gt_10_bps":int(absolute.gt(10).sum())}
        _write_atomic_json(boundary_comparison_path,{"schema_version":"URTH_DIRECT_DAILY_VS_MINUTE_BOUNDARY_V1",
            "authority":"DIAGNOSTIC_ONLY_PROVIDER_DEFINITION_COMPARISON","start":str(start),"end":str(end),
            "direct_provider":"ALPACA_SIP_1DAY_DIRECT","comparison_provider":"ALPACA_SIP_FIRST_LAST_REPORTED_MINUTE",
            "open":difference_metrics("open"),"close":difference_metrics("close")})
    return output_path


def _generation_rows(registry: GenerationRegistry) -> pd.DataFrame:
    rows = []
    for generation in registry.records.values():
        if str(generation.lifecycle_status.value) != "VALID":
            continue
        row = asdict(generation)
        row["lifecycle_status"] = generation.lifecycle_status.value
        row["exit_model_recipes_json"] = json.dumps(row.pop("exit_model_recipes", {}), sort_keys=True)
        row["entry_policy_id"] = generation.entry_policy_fingerprint
        row["exit_policy_id"] = generation.exit_generation_id or generation.exit_policy_fingerprint
        rows.append(row)
    return pd.DataFrame(rows)


def _write_atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    """Publish a validated parquet cache without exposing a partial file."""
    import pyarrow as pa
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    table = pa.Table.from_pandas(frame, preserve_index=False)
    pq.write_table(table, temporary, compression="zstd")
    check = pq.ParquetFile(temporary)
    rows = int(check.metadata.num_rows)
    del check
    if rows != len(frame):
        temporary.unlink()
        raise ValueError(f"ATOMIC_PARQUET_ROWCOUNT_MISMATCH:{path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    os.replace(temporary, path)


def _repair_cached_generation_bundle(*, output_root: Path, families, all_families):
    """Recover valid H04 shard records omitted by the interrupted global merge.

    This is a registry/checkpoint repair only: it consumes already published
    model, calibration and prediction artifacts and never calls ``refit``.
    """
    path = output_root / "valid_generations.parquet"
    generations = pd.read_parquet(path)
    requested_ids = {str(x.family_id) for x in families}
    present_ids = set(generations["family_id"].astype(str))
    missing_ids = requested_ids - present_ids
    if not missing_ids:
        return generations, {}
    recovered = []
    recovered_records = {}
    for shard_path in sorted((output_root / "factory" / "horizon-registries").glob("H*.json")):
        shard = GenerationRegistry(shard_path, all_families)
        for generation in shard.records.values():
            if (generation.family_id in missing_ids and
                    generation.lifecycle_status.value == "VALID"):
                recovered_records[generation.generation_id] = generation
        if missing_ids.issubset({x.family_id for x in recovered_records.values()}):
            break
    recovered = _generation_rows(type("RecoveredRegistry", (), {"records": recovered_records})())
    recovered = recovered.loc[recovered["family_id"].astype(str).isin(missing_ids)]
    recovered_ids = set(recovered["family_id"].astype(str)) if not recovered.empty else set()
    if recovered_ids != missing_ids:
        raise RuntimeError(f"DYNAMIC_QBD_CACHED_GENERATION_SHARDS_INCOMPLETE:{sorted(missing_ids - recovered_ids)}")
    generations = pd.concat([generations, recovered], ignore_index=True)
    generations = generations.sort_values(["family_id", "activation_date", "generation_id"]).reset_index(drop=True)
    _write_atomic_parquet(generations, path)
    print(f"[repair] recovered valid shard generations families={len(missing_ids)} rows={len(recovered)}", flush=True)
    return generations, recovered_records


class _CachedRegistryIdentity:
    """Minimal registry identity for a post-factory-only resume."""
    def __init__(self, families):
        self.family_registry_hash = stable_hash(tuple(asdict(x) for x in sorted(families, key=lambda item: item.family_id)))
        self.records = {}


def _matured_generation_predictions(*, predictions, generation_schedule: pd.DataFrame,
                                    families, signal_panel: Path,
                                    maturity: HorizonMaturityResolver, cutoff: date,
                                    cache_root: Path | None = None,
                                    telemetry_path: Path | None = None):
    # Keep the large Arrow-backed schedule out of the per-family hot loop.
    # Selecting all columns for every family can force pyarrow to materialize
    # a multi-GB boolean take.  The matured output needs only these columns.
    schedule_columns = ["family_id", "arm", "activation_date", "generation_id", "model_artifact_id"]
    active_schedule = generation_schedule.loc[
        generation_schedule["arm"].eq("C_ROLLING_REFIT_ROLLING_RECALIBRATION"), schedule_columns
    ].copy()
    active_schedule["family_id"] = active_schedule["family_id"].astype(str)
    active_schedule["model_artifact_id"] = active_schedule["model_artifact_id"].astype(str)
    schedule_fingerprint = hashlib.sha256(
        pd.util.hash_pandas_object(active_schedule, index=True).values.tobytes()
    ).hexdigest()
    cache_fingerprint = stable_hash({
        "schema_version": "DYNAMIC_QBD_MATURED_AGGREGATION_V2",
        "schedule_fingerprint": schedule_fingerprint,
        "signal_panel": _sha256_file(signal_panel),
        "cutoff": str(cutoff),
        "families": sorted(str(x.family_id) for x in families),
        "code": "STREAM_FAMILY_PARTITIONS_V2",
    })
    store = None
    if cache_root is not None:
        cache_root = Path(cache_root)
        manifest = cache_root / "manifest.json"
        if manifest.is_file():
            candidate = MaturedPredictionStore(cache_root)
            if candidate.cache_fingerprint != cache_fingerprint:
                raise ValueError("MATURED_CACHE_FINGERPRINT_MISMATCH")
            store = candidate
            if candidate.manifest.get("status") == "COMPLETE":
                print(f"[cache] matured predictions hit families={len(candidate.families)} rows={candidate.row_count}", flush=True)
                return candidate
        if store is None:
            import pyarrow as pa
            schema = pa.schema([pa.field(name, pa.string()) for name in MaturedPredictionStore.columns])
            store = MaturedPredictionStore.empty(cache_root, cache_fingerprint=cache_fingerprint, schema=schema)
    def memory_event(*, families_done: int, rows_processed: int, cache_hits: int) -> None:
        if telemetry_path is None:
            return
        try:
            import ctypes
            from ctypes import wintypes
            class _MemoryCounters(ctypes.Structure):
                _fields_ = [("cb", wintypes.DWORD), ("page_fault_count", wintypes.DWORD),
                            ("peak_working_set", ctypes.c_size_t), ("working_set", ctypes.c_size_t),
                            ("quota_peak_paged_pool", ctypes.c_size_t), ("quota_paged_pool", ctypes.c_size_t),
                            ("quota_peak_nonpaged_pool", ctypes.c_size_t), ("quota_nonpaged_pool", ctypes.c_size_t),
                            ("pagefile_usage", ctypes.c_size_t), ("peak_pagefile_usage", ctypes.c_size_t)]
            counters = _MemoryCounters()
            counters.cb = ctypes.sizeof(counters)
            get_info = ctypes.windll.psapi.GetProcessMemoryInfo
            get_info.argtypes = [wintypes.HANDLE, ctypes.POINTER(_MemoryCounters), wintypes.DWORD]
            get_info.restype = wintypes.BOOL
            ok = get_info(ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb)
            rss_gb = counters.working_set / (1024 ** 3) if ok else None
        except Exception:
            rss_gb = None
        telemetry_path.parent.mkdir(parents=True, exist_ok=True)
        record = {"ts": time.time(), "stage": "matured_generation_predictions",
                  "families_done": families_done, "rows_processed": rows_processed,
                  "cache_hits": cache_hits, "rss_gb": rss_gb}
        with telemetry_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        print("[MEM] " + json.dumps(record, sort_keys=True), flush=True)
    schedule_by_family = {
        str(family_id): group.sort_values("activation_date")
        for family_id, group in active_schedule.groupby("family_id", sort=False)
    }
    prediction_model_ids = None if hasattr(predictions, "iter_model_ids") else predictions["model_artifact_id"].astype(str)
    prediction_columns = ["decision_date", "ticker", "model_artifact_id", "score"]
    outcomes_by_horizon: dict[int, pd.DataFrame] = {}
    active_horizon = None
    # D/N cells sharing a fixed H schedule must reuse the same expensive
    # prediction/outcome merge.  Without this cache the aggregation is
    # effectively O(number of portfolio families × prediction rows).
    rows = []
    rows_processed = 0
    cache_hits = 0
    cached_families = set(store.families) if store is not None else set()
    base_cache_root = (Path(cache_root) / "base" if cache_root is not None else None)
    base_columns = ("decision_date", "terminal_date", "ticker",
                    "model_artifact_id", "score", "realized_excess")
    family_base_keys = {}
    base_specs = {}
    base_groups = {}
    for candidate in families:
        candidate_schedule = schedule_by_family.get(str(candidate.family_id))
        if candidate_schedule is None or candidate_schedule.empty:
            continue
        candidate_schedule = candidate_schedule.copy()
        candidate_schedule["activation_date"] = pd.to_datetime(
            candidate_schedule["activation_date"],
        ).dt.normalize().astype("datetime64[ns]")
        candidate_horizon = int(candidate.horizon_sessions)
        candidate_model_ids = set(candidate_schedule["model_artifact_id"].tolist())
        candidate_schedule_fingerprint = hashlib.sha256(
            pd.util.hash_pandas_object(
                candidate_schedule[["activation_date", "model_artifact_id"]], index=False,
            ).values.tobytes()
        ).hexdigest()
        candidate_base_key = stable_hash({
            "schema_version": "DYNAMIC_QBD_MATURED_BASE_V2",
            "cache_fingerprint": cache_fingerprint,
            "horizon": candidate_horizon,
            "model_ids": sorted(candidate_model_ids),
            "schedule_fingerprint": candidate_schedule_fingerprint,
        })
        family_base_keys[str(candidate.family_id)] = candidate_base_key
        if candidate_base_key not in base_specs:
            base_specs[candidate_base_key] = (
                candidate_schedule, candidate_horizon, candidate_model_ids,
                base_cache_root / f"base-{candidate_base_key}.parquet" if base_cache_root is not None else None,
            )
        base_groups.setdefault(candidate_base_key, []).append(str(candidate.family_id))
    processed_family_ids = set()
    for family_index, family in enumerate(families, start=1):
        family_id = str(family.family_id)
        if family_id in processed_family_ids:
            continue
        horizon = int(family.horizon_sessions)
        if active_horizon is not None and horizon != active_horizon:
            # Outcome panels are horizon-specific. Keeping prior horizons in
            # memory is unnecessary and can overlap the next large merge.
            outcomes_by_horizon.clear()
        active_horizon = horizon
        if store is not None and family_id in cached_families:
            cache_hits += 1
            processed_family_ids.add(family_id)
            memory_event(families_done=family_index, rows_processed=rows_processed, cache_hits=cache_hits)
            continue
        base_key = family_base_keys.get(family_id)
        if base_key is None:
            continue
        schedule, horizon, model_ids, base_path = base_specs[base_key]
        group_ids = [candidate_id for candidate_id in base_groups[base_key]
                     if candidate_id not in cached_families and candidate_id not in processed_family_ids]
        for candidate_id in base_groups[base_key]:
            if candidate_id != family_id and candidate_id in cached_families:
                processed_family_ids.add(candidate_id)
        base_valid = False
        base_chunks = ()
        if base_path is not None and base_path.is_file():
            try:
                check = pq.ParquetFile(base_path)
                base_valid = set(base_columns).issubset(set(check.schema_arrow.names)) and int(check.metadata.num_rows) > 0
                check.close()
            except Exception:
                base_valid = False
                base_path.unlink(missing_ok=True)

        def transformed_base_chunks(source_chunks, outcomes):
            for source in source_chunks:
                if source.empty:
                    continue
                source["decision_date"] = pd.to_datetime(source["decision_date"]).dt.normalize().astype("datetime64[ns]")
                decisions = pd.DataFrame({"decision_date": sorted(source["decision_date"].unique())})
                authority = pd.merge_asof(
                    decisions,
                    schedule[["activation_date", "model_artifact_id"]],
                    left_on="decision_date", right_on="activation_date", direction="backward",
                ).dropna(subset=["model_artifact_id"])
                source = source.merge(
                    authority[["decision_date", "model_artifact_id"]],
                    on=["decision_date", "model_artifact_id"], how="inner",
                )
                if source.empty:
                    continue
                merged = source.merge(
                    outcomes, on=["decision_date", "ticker"], how="inner", validate="many_to_one",
                )
                terminal = []
                for value in pd.to_datetime(merged["decision_date"]):
                    try:
                        terminal.append(maturity.terminal_date(value, horizon))
                    except ValueError:
                        terminal.append(pd.NaT)
                merged["terminal_date"] = terminal
                base_frame = merged.loc[
                    pd.to_datetime(merged["terminal_date"]).le(pd.Timestamp(cutoff)), list(base_columns),
                ].copy()
                if not base_frame.empty:
                    yield base_frame
                del merged, source, authority, decisions, terminal, base_frame

        if not base_valid:
            target = f"net_excess_return_{horizon}__BASELINE_20_BPS"
            outcomes = outcomes_by_horizon.get(horizon)
            if outcomes is None:
                outcomes = pd.read_parquet(signal_panel, columns=["decision_date", "ticker", target])
                outcomes["decision_date"] = pd.to_datetime(outcomes["decision_date"]).dt.normalize().astype("datetime64[ns]")
                outcomes = outcomes.rename(columns={target: "realized_excess"})
                outcomes_by_horizon[horizon] = outcomes
            if hasattr(predictions, "iter_model_ids"):
                source_chunks = predictions.iter_model_ids(model_ids)
            else:
                source = predictions.loc[prediction_model_ids.isin(model_ids), prediction_columns].copy()
                source_chunks = (source,)
            transformed = transformed_base_chunks(source_chunks, outcomes)
            if base_path is not None:
                import pyarrow as pa
                base_cache_root.mkdir(parents=True, exist_ok=True)
                temporary = base_path.with_name(base_path.name + f".{os.getpid()}.tmp")
                writer = None
                try:
                    for base_frame in transformed:
                        table = pa.Table.from_pandas(base_frame, preserve_index=False)
                        if writer is None:
                            writer = pq.ParquetWriter(temporary, table.schema, compression="zstd")
                        writer.write_table(table)
                    if writer is not None:
                        writer.close()
                        writer = None
                        check = pq.ParquetFile(temporary)
                        if int(check.metadata.num_rows) <= 0 or not set(base_columns).issubset(set(check.schema_arrow.names)):
                            check.close()
                            raise ValueError("MATURED_BASE_CACHE_VALIDATION_FAILED")
                        check.close()
                        os.replace(temporary, base_path)
                        base_valid = True
                finally:
                    if writer is not None:
                        writer.close()
                    if temporary.exists():
                        temporary.unlink()
            else:
                base_chunks = transformed
        if base_valid:
            def read_base_chunks():
                parquet = pq.ParquetFile(base_path)
                try:
                    for index in range(parquet.num_row_groups):
                        yield parquet.read_row_group(index, columns=list(base_columns)).to_pandas()
                finally:
                    parquet.close()
            base_chunks = read_base_chunks()

        def family_chunks():
            nonlocal rows_processed
            for base_frame in base_chunks:
                decisions = pd.DataFrame({"decision_date": sorted(base_frame["decision_date"].unique())})
                authority = pd.merge_asof(
                    decisions,
                    schedule[["activation_date", "generation_id", "model_artifact_id"]],
                    left_on="decision_date", right_on="activation_date", direction="backward",
                ).dropna(subset=["generation_id"])
                family_rows = base_frame.merge(
                    authority[["decision_date", "generation_id", "model_artifact_id"]],
                    on=["decision_date", "model_artifact_id"], how="inner",
                )
                if family_rows.empty:
                    del decisions, authority, family_rows, base_frame
                    continue
                family_rows["family_id"] = str(family.family_id)
                family_rows = family_rows[[
                    "decision_date", "terminal_date", "ticker", "family_id",
                    "generation_id", "model_artifact_id", "score", "realized_excess",
                ]]
                rows_processed += len(family_rows)
                yield family_rows
                del decisions, authority, family_rows, base_frame

        if store is not None and len(group_ids) > 1:
            family_specs = {
                candidate_id: base_specs[family_base_keys[candidate_id]][0]
                for candidate_id in group_ids
            }
            published = store.publish_family_group_from_base(family_specs, base_chunks)
            rows_processed += sum(published.values())
            processed_family_ids.update(group_ids)
            cached_families.update(group_ids)
        else:
            chunks = family_chunks()
            if store is not None:
                store.publish_family_chunks(family_id, chunks)
                processed_family_ids.add(family_id)
                cached_families.add(family_id)
            else:
                rows.extend(chunks)
            del chunks
        if family_index % 25 == 0 or family_index == len(families):
            memory_event(families_done=family_index, rows_processed=rows_processed, cache_hits=cache_hits)
    if store is not None:
        store.finalize()
        return store
    # Backwards-compatible fixture path only; production always supplies a
    # cache_root and therefore never materializes the global panel.
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=list(MaturedPredictionStore.columns))


def run_pipeline(*, signal_panel: str | Path, candidate_metrics: str | Path, prices: str | Path | None = None,
                 daily_store_root: str | Path | None = None,
                 output_root: str | Path, family_ids, start: date, end: date,
                 holdout_contract: str, feature_schema_sha256: str,
                 learned_exit_candidate_metrics: str | Path | None = None,
                 market_state: str | Path | None = None,
                 training_window_sessions: int = 504, calibration_window_sessions: int = 252,
                 initial: float = 10000.0, include_policy_optimization_arms: bool = False,
                 score_quantile: float = .975, top_fraction: float = .005,
                 model_workers: int = 1) -> dict:
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    # Persist the actual host/worker plan used by this invocation.  This is
    # execution telemetry, not part of the scientific semantic contract, and
    # is deliberately written before any fit/replay work starts.
    _write_atomic_json(
        output_root / "telemetry" / "resource-contract.json",
        active_cpu_contract(),
    )
    signal_panel, candidate_metrics = Path(signal_panel), Path(candidate_metrics)
    if (prices is None) == (daily_store_root is None):
        raise ValueError("EXACTLY_ONE_OF_PRICES_OR_DAILY_STORE_ROOT_REQUIRED")
    prices_path=(Path(prices) if prices is not None else materialize_daily_store_prices(
        daily_store_root=daily_store_root,signal_panel=signal_panel,start=start,end=end,
        output_path=output_root/"inputs"/"prices.parquet"))
    sessions, historical_holdout = _sessions_and_holdout(signal_panel)
    holdout_start = _holdout_boundary(holdout_contract, historical_holdout)
    if end >= holdout_start:
        raise PermissionError(f"DYNAMIC_QBD_FINAL_HOLDOUT_LOCKED:{holdout_start}")
    requested = set(str(x) for x in family_ids)
    include_learned_exit = any(str(x).endswith("_LEARNED_EXIT") for x in requested)
    materializer = ParquetH130DatasetMaterializer(
        signal_panel, candidate_metrics, end,
        Path(learned_exit_candidate_metrics) if learned_exit_candidate_metrics else None,
    )
    if feature_schema_sha256 == "AUTO":
        feature_schema_sha256 = materializer.feature_schema_fingerprint
    code_identity=_git_code_identity()
    code_fingerprint=f"{code_identity['commit']}:{code_identity['source_tree_sha256']}"
    semantic_contract={"schema_version":"DYNAMIC_QBD_FACTORY_RUN_CONTRACT_V1",
                  "signal_panel_sha256":materializer.dataset_fingerprint,
                  "candidate_metrics_sha256":stable_hash(json.loads(candidate_metrics.read_text(encoding="utf-8"))),
                  "prices_sha256":_sha256_file(prices_path),
                  "market_state_sha256":_sha256_file(Path(market_state)) if market_state else "DERIVED_FROM_PRICES_V1",
                  "holdout_contract":holdout_contract,"holdout_start":holdout_start,
                  "development_start":start,"development_end":end,"family_ids":sorted(requested),
                  "feature_schema_sha256":feature_schema_sha256,
                  "training_window_sessions":training_window_sessions,
                  "calibration_window_sessions":calibration_window_sessions,
                  "score_quantile":float(score_quantile),"top_fraction":float(top_fraction),
                  "include_policy_optimization_arms":bool(include_policy_optimization_arms)}
    execution_provenance={"git_sha":code_identity["commit"], "code_identity":code_identity,
                          "model_workers":int(model_workers),
                  "parallel_scheduler_version":"PROCESS_POOL_2X_QUEUE_AHEAD_V1",
                          "queue_capacity": int(model_workers) * 2,
                          "native_threads_per_worker": 1}
    run_contract={**semantic_contract,
                  "semantic_run_contract_hash":stable_hash(semantic_contract),
                  "execution_provenance":execution_provenance,
                  # Retain these top-level fields for backwards-readable provenance.
                  "git_sha":code_identity["commit"], "code_identity":code_identity,
                  "model_workers":int(model_workers)}
    run_contract["run_contract_sha256"]=stable_hash(run_contract)
    contract_path=output_root/"factory-run-contract.json"
    if contract_path.is_file():
        existing=json.loads(contract_path.read_text(encoding="utf-8"))
        existing_semantic=_semantic_contract(existing)
        current_semantic=json.loads(json.dumps(semantic_contract,default=str))
        if existing_semantic != current_semantic:
            raise ValueError("DYNAMIC_QBD_RESTART_RUN_CONTRACT_MISMATCH")
        migration_path=output_root/"execution-provenance-migrations.json"
        migration_rows=json.loads(migration_path.read_text(encoding="utf-8")) if migration_path.is_file() else []
        migration=migration_rows[-1] if migration_rows else None
        if (existing.get("git_sha") != code_identity["commit"] or
                int(existing.get("model_workers", 1)) != int(model_workers)):
            migration_rows.append({"schema_version":"DYNAMIC_QBD_EXECUTION_ONLY_RESUME_MIGRATION_V1",
                                   "from_git_sha":existing.get("git_sha"),
                                   "to_git_sha":code_identity["commit"],
                                   "from_model_workers":int(existing.get("model_workers", 1)),
                                   "to_model_workers":int(model_workers),
                                   "semantic_run_contract_hash":stable_hash(semantic_contract),
                                   "from_contract_sha256":existing.get("run_contract_sha256"),
                                   "to_execution_provenance":execution_provenance})
            _write_atomic_json(migration_path,migration_rows)
    else:
        _write_atomic_json(contract_path,run_contract)
    completed=_reuse_verified_completed_run(output_root,run_contract)
    if completed is not None:
        return completed
    all_families = build_family_specs(feature_schema_sha256=feature_schema_sha256,
                                      training_window_sessions=training_window_sessions,
                                      calibration_window_sessions=calibration_window_sessions,
                                      include_learned_exit=include_learned_exit,
                                      score_quantile=score_quantile,top_fraction=top_fraction)
    families = tuple(x for x in all_families if x.family_id in requested)
    if not requested or {x.family_id for x in families} != requested:
        missing = sorted(requested - {x.family_id for x in families})
        raise ValueError(f"DYNAMIC_QBD_FAMILY_SELECTION_INVALID:{missing or 'EMPTY'}")
    development_sessions = tuple(x for x in sessions if start <= x <= end and x < holdout_start)
    if not development_sessions:
        raise ValueError("DYNAMIC_QBD_DEVELOPMENT_SESSIONS_EMPTY")
    maturity = HorizonMaturityResolver(tuple(x for x in sessions if x < holdout_start))
    downstream_cache_ready = all((output_root / name).is_file() for name in (
        "valid_generations.parquet", "rolling_frozen_model_calibrations.parquet",
        "abc_generation_schedule.parquet", "model_predictions.parquet",
    ))
    registry = (_CachedRegistryIdentity(families) if downstream_cache_ready else
                GenerationRegistry(output_root/"factory"/"generation-registry.json", families))
    builder = H130ProductionGenerationBuilder(
        materializer=materializer,
        root=output_root/"factory"/"generations", code_commit=code_fingerprint,
    )
    recovered_records = {}
    cached_generations = None
    if downstream_cache_ready:
        cached_generations, recovered_records = _repair_cached_generation_bundle(
            output_root=output_root, families=families, all_families=all_families)
        for generation in recovered_records.values():
            builder.restore_generation(generation)
    if downstream_cache_ready:
        # The factory registry is authoritative for generation identity, but
        # no model object has to be reloaded or re-hashed to execute only the
        # post-factory stages.  This is the normal Development-pipeline resume path.
        print("[cache] downstream-only resume: factory artifacts retained", flush=True)
    else:
        for generation in registry.records.values():
            if generation.lifecycle_status.value=="VALID":
                builder.restore_generation(generation)
    all_fixed_exit = all(str(x.exit_policy.get("family", "FIXED")).upper() == "FIXED" for x in families)
    if downstream_cache_ready:
        generations = cached_generations
        if (set(generations.get("family_id", pd.Series(dtype=str)).astype(str)) !=
                {str(x.family_id) for x in families} or
                not set(generations.get("lifecycle_status", pd.Series(["VALID"])).astype(str)).issubset({"VALID", "GenerationStatus.VALID"})):
            raise ValueError("DYNAMIC_QBD_CACHED_GENERATIONS_INVALID")
    elif int(model_workers) > 1 and all_fixed_exit:
        registry = _run_global_canonical_model_queue(
            families=families, all_families=all_families, signal_panel=signal_panel,
            candidate_metrics=candidate_metrics, output_root=output_root, sessions=sessions,
            development_end=end, holdout_start=holdout_start, start=start,
            code_fingerprint=code_fingerprint, model_workers=model_workers, registry=registry)
        builder = H130ProductionGenerationBuilder(
            materializer=materializer, root=output_root/"factory"/"generations", code_commit=code_fingerprint)
        for generation in registry.records.values():
            if generation.lifecycle_status.value == "VALID":
                builder.restore_generation(generation)
    elif int(model_workers) > 1:
        workers = min(int(model_workers), len(set(int(x.horizon_sessions) for x in families)))
        payloads = [{"horizon": horizon, "families": families, "all_families": all_families,
                     "signal_panel": str(signal_panel), "candidate_metrics": str(candidate_metrics),
                     "learned_exit_candidate_metrics": str(learned_exit_candidate_metrics) if learned_exit_candidate_metrics else None,
                     "output_root": str(output_root), "development_end": str(end),
                     "holdout_boundary": str(holdout_start), "sessions": [str(x) for x in sessions],
                     "start": str(start), "code_commit": code_fingerprint}
                    for horizon in sorted(set(int(x.horizon_sessions) for x in families))]
        worker_map = list(active_cpu_contract().get("worker_map") or [])[:workers]
        if len(worker_map) != workers:
            raise ValueError(f"DYNAMIC_QBD_HORIZON_WORKER_AFFINITY_PLAN_INCOMPLETE:{len(worker_map)}:{workers}")
        slot_counter = mp.Value("i", 0)
        # Keep the same bounded two-ahead scheduler as the fixed-exit path;
        # pool.map would eagerly submit the complete horizon surface and would
        # also omit the topology-aware lane initializer.
        queue_capacity = max(workers, workers * 2)
        with ProcessPoolExecutor(
            max_workers=workers, mp_context=mp.get_context("spawn"),
            initializer=_model_fit_process_initializer,
            initargs=(worker_map, slot_counter),
        ) as pool:
            pending = {}
            next_index = 0
            while next_index < len(payloads) and len(pending) < queue_capacity:
                future = pool.submit(_run_horizon_lane, payloads[next_index])
                pending[future] = next_index
                next_index += 1
            while pending:
                done, _ = wait(tuple(pending), return_when=FIRST_COMPLETED)
                for future in done:
                    future.result()
                    pending.pop(future, None)
                    if next_index < len(payloads):
                        replacement = pool.submit(_run_horizon_lane, payloads[next_index])
                        pending[replacement] = next_index
                        next_index += 1
        # Horizon lanes write independently; merge their append-only shards
        # once, outside the hot model-generation loop.
        registry = GenerationRegistry(output_root/"factory"/"generation-registry.json", families)
        for shard_path in sorted((output_root/"factory"/"horizon-registries").glob("H*.json")):
            shard = GenerationRegistry(shard_path, families)
            shard_horizon = int(shard_path.stem[1:])
            for generation_id, generation in shard.records.items():
                prior = registry.records.get(generation_id)
                if prior is not None and prior != generation:
                    raise ValueError("DYNAMIC_QBD_HORIZON_REGISTRY_GENERATION_COLLISION")
                registry.records[generation_id] = generation
            # A shard may have been bootstrapped from an older global
            # registry.  Its records are useful for collision checking, but
            # its activation map must contribute only this horizon; otherwise
            # a stale H02 copy could overwrite a freshly merged H01 current
            # generation during the final merge.
            for family_id, generation_id in shard.current.items():
                if int(family_id[1:3]) != shard_horizon:
                    continue
                registry.current[family_id] = generation_id
        registry.save()
        builder = H130ProductionGenerationBuilder(
            materializer=materializer, root=output_root/"factory"/"generations", code_commit=code_fingerprint)
        for generation in registry.records.values():
            if generation.lifecycle_status.value == "VALID":
                builder.restore_generation(generation)
    else:
        for family in families:
            cadence = str(family.refit_cadence).upper()
            if cadence not in {"MONTH_END", "MONTH_START"}:
                raise ValueError(f"UNSUPPORTED_DYNAMIC_QBD_REFIT_CADENCE:{cadence}")
            refit_dates = tuple(x for x in monthly_refit_dates(development_sessions,day=cadence) if x >= start)
            for cutoff in refit_dates:
                if registry.generation_for_family_cutoff(family.family_id, cutoff) is not None:
                    continue
                refit_family(family=family, information_cutoff=cutoff, maturity=maturity,
                             builder=builder, registry=registry)
    if not downstream_cache_ready:
        generations = _generation_rows(registry)
    if generations.empty:
        failures = {x.generation_id: x.failure_reason for x in registry.records.values()}
        raise RuntimeError(f"DYNAMIC_QBD_NO_VALID_GENERATIONS:{failures}")
    prediction_sources = _unique_prediction_artifact_sources(generations)
    prediction_path = output_root / "model_predictions.parquet"
    disk_predictions = _ensure_streamed_prediction_store(sources=prediction_sources, destination=prediction_path)
    exit_paths = [str(x) for x in generations.get("exit_prediction_artifact_path", pd.Series(dtype=str)) if str(x)]
    if include_learned_exit:
        if not exit_paths:
            raise RuntimeError("DYNAMIC_QBD_LEARNED_EXIT_GENERATIONS_MISSING")
        exit_predictions = pd.concat([pd.read_parquet(path) for path in sorted(set(exit_paths))], ignore_index=True)
        exit_path = output_root/"generation_exit_predictions.parquet"
        exit_predictions.to_parquet(exit_path, index=False)
        tax_contracts={stable_hash(family.tax_contract):family.tax_contract for family in families}
        if len(tax_contracts) != 1:
            raise ValueError("LEARNED_EXIT_REQUIRES_ONE_SHARED_TAX_CONTRACT")
        tax=tax_config_from_contract(next(iter(tax_contracts.values())))
        configure_replay(LearnedExitProvider(exit_path), ProfitTaxConfig(
            enabled=tax.enabled,capital_gains_rate=tax.capital_gains_rate,
            solidarity_surcharge=tax.solidarity_surcharge,allowance_eur=tax.allowance_eur,
            church_tax_rate=tax.church_tax_rate))
    cached_generations_path = output_root / "valid_generations.parquet"
    cached_calibrations_path = output_root / "rolling_frozen_model_calibrations.parquet"
    cached_schedule_path = output_root / "abc_generation_schedule.parquet"
    cached_optimized_path = output_root / "rolling_policy_optimized_calibrations.parquet"
    expected_generation_ids = set(generations["generation_id"].astype(str))
    use_calibration_cache = all(path.is_file() for path in (
        cached_generations_path, cached_calibrations_path, cached_schedule_path,
    ))
    if use_calibration_cache:
        cached_generations_index = pd.read_parquet(cached_generations_path, columns=["generation_id", "family_id"])
        cached_schedule = pd.read_parquet(cached_schedule_path)
        use_calibration_cache = (
            set(cached_generations_index["generation_id"].astype(str)) == expected_generation_ids and
            set(cached_generations_index["family_id"].astype(str)) == set(str(x.family_id) for x in families) and
            set(cached_schedule.get("family_id", pd.Series(dtype=str)).astype(str)) <= set(str(x.family_id) for x in families) and
            {"family_id", "arm", "activation_date", "generation_id", "model_artifact_id"}.issubset(cached_schedule.columns) and
            (not include_policy_optimization_arms or cached_optimized_path.is_file())
        )
    if use_calibration_cache:
        print(f"[cache] calibration/schedule hit generations={len(expected_generation_ids)}", flush=True)
        calibrations = pd.read_parquet(cached_calibrations_path)
        optimized_calibrations = pd.read_parquet(cached_optimized_path) if cached_optimized_path.is_file() else pd.DataFrame()
        cached_schedule_family_ids = set(cached_schedule["family_id"].astype(str))
        missing_schedule_family_ids = set(str(x.family_id) for x in families) - cached_schedule_family_ids
        if missing_schedule_family_ids:
            print(f"[repair] rebuilding calibration/schedule rows for families={len(missing_schedule_family_ids)}", flush=True)
            repair_calibration_rows = []
            repair_optimized_rows = []
            for family in families:
                if str(family.family_id) not in missing_schedule_family_ids:
                    continue
                fg = generations.loc[generations["family_id"].eq(family.family_id)].sort_values("activation_date")
                frozen = fg.iloc[0]
                frozen_cutoff = pd.Timestamp(frozen["information_cutoff"]).date()
                for current in fg.itertuples(index=False):
                    cutoff = pd.Timestamp(current.information_cutoff).date()
                    latest = maturity.latest_matured_decision(cutoff, family.horizon_sessions)
                    frame = builder.recalibration_predictions(
                        family=family, model_information_cutoff=frozen_cutoff, information_cutoff=cutoff,
                        latest_matured_label_cutoff=latest, cache_result=False,
                    )
                    record = recalibrate_generation(family, str(frozen.generation_id), frame,
                                                    information_cutoff=cutoff, maturity=maturity)
                    repair_calibration_rows.append({**asdict(record), "activation_date": cutoff,
                                                    "entry_policy_id": str(frozen.entry_policy_id),
                                                    "exit_policy_id": str(frozen.exit_policy_id)})
                    if include_policy_optimization_arms:
                        optimized_frozen = recalibrate_generation(
                            family, str(frozen.generation_id), frame, information_cutoff=cutoff,
                            maturity=maturity, optimize_policy=True,
                        )
                        repair_optimized_rows.append({**asdict(optimized_frozen), "activation_date": cutoff,
                                                      "optimization_scope":"FROZEN_MODEL",
                                                      "model_artifact_id":str(frozen.model_artifact_id)})
                        own_frame = builder.generation_calibration_predictions(
                            family=family, model_information_cutoff=cutoff,
                        )
                        optimized_rolling = recalibrate_generation(
                            family, str(current.generation_id), own_frame, information_cutoff=cutoff,
                            maturity=maturity, optimize_policy=True,
                        )
                        repair_optimized_rows.append({**asdict(optimized_rolling), "activation_date": cutoff,
                                                      "optimization_scope":"ROLLING_MODEL",
                                                      "model_artifact_id":str(current.model_artifact_id)})
                    del frame
                builder._recalibration_cache.clear()
                builder._panel_cache.clear()
                gc.collect()
            calibrations = pd.concat([calibrations, pd.DataFrame(repair_calibration_rows)], ignore_index=True)
            if repair_optimized_rows:
                optimized_calibrations = pd.concat([optimized_calibrations, pd.DataFrame(repair_optimized_rows)], ignore_index=True)
            schedules = build_abc_generation_schedules(generations, calibrations, optimized_calibrations)
        else:
            schedules = cached_schedule
    else:
        calibration_rows = []
        optimized_calibration_rows = []
        for family in families:
            fg = generations.loc[generations["family_id"].eq(family.family_id)].sort_values("activation_date")
            if fg.empty:
                continue
            frozen = fg.iloc[0]
            frozen_cutoff = pd.Timestamp(frozen["information_cutoff"]).date()
            for current in fg.itertuples(index=False):
                cutoff = pd.Timestamp(current.information_cutoff).date()
                latest = maturity.latest_matured_decision(cutoff, family.horizon_sessions)
                frame = builder.recalibration_predictions(
                    family=family, model_information_cutoff=frozen_cutoff, information_cutoff=cutoff,
                    latest_matured_label_cutoff=latest,
                )
                record = recalibrate_generation(family, str(frozen.generation_id), frame,
                                                information_cutoff=cutoff, maturity=maturity)
                calibration_rows.append({**asdict(record), "activation_date": cutoff,
                                         "entry_policy_id": str(frozen.entry_policy_id),
                                         "exit_policy_id": str(frozen.exit_policy_id)})
                if include_policy_optimization_arms:
                    optimized_frozen = recalibrate_generation(
                        family, str(frozen.generation_id), frame, information_cutoff=cutoff,
                        maturity=maturity, optimize_policy=True,
                    )
                    optimized_calibration_rows.append({**asdict(optimized_frozen), "activation_date": cutoff,
                                                       "optimization_scope":"FROZEN_MODEL",
                                                       "model_artifact_id":str(frozen.model_artifact_id)})
                    own_frame = builder.generation_calibration_predictions(
                        family=family, model_information_cutoff=cutoff,
                    )
                    optimized_rolling = recalibrate_generation(
                        family, str(current.generation_id), own_frame, information_cutoff=cutoff,
                        maturity=maturity, optimize_policy=True,
                    )
                    optimized_calibration_rows.append({**asdict(optimized_rolling), "activation_date": cutoff,
                                                       "optimization_scope":"ROLLING_MODEL",
                                                       "model_artifact_id":str(current.model_artifact_id)})
        calibrations = pd.DataFrame(calibration_rows)
        optimized_calibrations = pd.DataFrame(optimized_calibration_rows)
        schedules = build_abc_generation_schedules(generations, calibrations, optimized_calibrations)

    # Calibration replay may transiently materialize a large source panel.  It
    # is no longer needed once schedules are built; release adapter caches
    # before the prediction and matured-aggregation stages start.
    builder._recalibration_cache.clear()
    builder._panel_cache.clear()
    gc.collect()
    generations.to_parquet(output_root/"valid_generations.parquet", index=False)
    calibrations.to_parquet(output_root/"rolling_frozen_model_calibrations.parquet", index=False)
    if not optimized_calibrations.empty:
        optimized_calibrations.to_parquet(output_root/"rolling_policy_optimized_calibrations.parquet", index=False)
    schedules.to_parquet(output_root/"abc_generation_schedule.parquet", index=False)
    matured_cache_root = output_root / "cache" / "matured_generation_predictions"
    matured_predictions = _matured_generation_predictions(
        predictions=disk_predictions, generation_schedule=schedules, families=families,
        signal_panel=signal_panel, maturity=maturity, cutoff=end,
        cache_root=matured_cache_root,
        telemetry_path=output_root / "telemetry" / "memory.jsonl",
    )
    matured_flat_path = output_root / "matured_generation_predictions.parquet"
    if (not matured_flat_path.is_file() or
            pq.ParquetFile(matured_flat_path).metadata.num_rows != matured_predictions.row_count):
        matured_predictions.copy_to_parquet(matured_flat_path)
    price_frame = pd.read_parquet(prices_path)
    price_frame = price_frame.loc[pd.to_datetime(price_frame["date"]).le(pd.Timestamp(end))].copy()
    market_frame=(pd.read_parquet(market_state) if market_state else build_monthly_market_state(price_frame))
    summary = run_development(
        families=families, predictions=disk_predictions, prices=price_frame,
        generation_schedule=schedules, output_root=output_root/"development", initial=initial,
        matured_predictions=matured_predictions,
        final_holdout_start=holdout_start, holdout_contract=holdout_contract,
        market_state=market_frame,
    )
    provenance = {"schema_version": "DYNAMIC_QBD_PIPELINE_V1", "git_sha": code_identity["commit"],
                  "code_identity":code_identity,
                  "holdout_contract": holdout_contract, "holdout_start": holdout_start,
                  "development_start": start, "development_end": end,
                  "signal_panel_sha256": builder.materializer.materialize(
                      family=families[0], information_cutoff=end,
                      latest_matured_label_cutoff=maturity.latest_matured_decision(end, families[0].horizon_sessions),
                      output_root=output_root/"provenance")["source_panel_sha256"],
                  "family_registry_hash": registry.family_registry_hash,
                  "family_ids": sorted(requested), "result": summary}
    provenance["pipeline_fingerprint"] = stable_hash(provenance)
    _write_atomic_json(output_root/"pipeline-summary.json",provenance)
    _write_atomic_json(output_root/"generations"/"generation-index.json",{
        "schema_version":"DYNAMIC_QBD_GENERATION_INDEX_V1",
        "registry":str((output_root/"factory"/"generation-registry.json").resolve()),
        "generation_ids":sorted(generations["generation_id"].astype(str).unique()),
        "model_artifact_ids":sorted(generations["model_artifact_id"].astype(str).unique()),
    })
    aliases = {
        output_root/"valid_generations.parquet": output_root/"generations"/"valid_generations.parquet",
        output_root/"model_predictions.parquet": output_root/"predictions"/"model_predictions.parquet",
        output_root/"matured_generation_predictions.parquet": output_root/"predictions"/"matured_generation_predictions.parquet",
        output_root/"rolling_frozen_model_calibrations.parquet": output_root/"calibration"/"rolling_frozen_model_calibrations.parquet",
        output_root/"abc_generation_schedule.parquet": output_root/"abc"/"generation_schedule.parquet",
    }
    if (output_root/"rolling_policy_optimized_calibrations.parquet").is_file():
        aliases[output_root/"rolling_policy_optimized_calibrations.parquet"] = output_root/"calibration"/"rolling_policy_optimized_calibrations.parquet"
    for source,destination in aliases.items():
        _hardlink_or_copy(source,destination)
    for directory in ("portfolio_paths","abc","evidence","gate1","gate1b","gate2","gate3"):
        source_root=output_root/"development"/directory
        if source_root.is_dir():
            for source in source_root.iterdir():
                if source.is_file():
                    _hardlink_or_copy(source,output_root/directory/source.name)
    for name in ("REPORT.md","summary.json","pre_holdout_state.json"):
        source=output_root/"development"/name
        if source.is_file():
            _hardlink_or_copy(source,output_root/name)
    nav_path=output_root/"development"/"family_shadow_nav.parquet"
    pre_holdout_path=output_root/"development"/"pre_holdout_state.json"
    if not pre_holdout_path.is_file():
        raise RuntimeError("DYNAMIC_QBD_PRE_HOLDOUT_STATE_MISSING")
    pre_holdout_state=json.loads(pre_holdout_path.read_text(encoding="utf-8"))
    supplied_state_hash=pre_holdout_state.pop("state_hash",None)
    if supplied_state_hash!=stable_hash(pre_holdout_state):
        raise ValueError("DYNAMIC_QBD_PRE_HOLDOUT_STATE_HASH_MISMATCH")
    initial_state_hash=supplied_state_hash
    freeze=build_algorithm_manifest(
        code_commit=code_fingerprint,family_registry_hash=registry.family_registry_hash,
        candidate_design_space={"family_ids":sorted(requested),"surface":"H1_H30_D_LE_H_N1_N6_EXIT_VARIANTS"},
        feature_schema=feature_schema_sha256,training_recipe="FROZEN_WITHIN_FAMILY_MONTHLY_FIT_ONLY",
        hyperparameter_rule="CAUSAL_PRIOR_OOS_FROZEN_RECIPE_SELECTION",
        training_window_rule={"sessions":training_window_sessions,"horizon_specific_maturity":True},
        refit_cadence="MONTHLY",horizon_maturity_rule="TERMINAL_DATE_LE_INFORMATION_CUTOFF",
        calibration_window_rule={"sessions":calibration_window_sessions,"disjoint_from_train":True},
        recalibration_algorithm="GENERATION_SPECIFIC_DAILY_TOP_SCORE_QUANTILE",
        threshold_rule="FROZEN_Q_TOP_ROLLING_THRESHOLD_PRIMARY_B2_C2_OPTIONAL",
        exit_runtime_rule="ENTRY_GENERATION_LINEAGE_NEXT_OPEN_EXECUTION",
        benchmark_contract="URTH_IDLE_CAPITAL_AND_RELATIVE_WEALTH",
        cost_contract="FAMILY_SPEC_AUTHORITATIVE",tax_contract="FAMILY_SPEC_AUTHORITATIVE",
        evidence_panel_schema="DYNAMIC_QBD_MONTHLY_FAMILY_EVIDENCE_V1",
        gate_rules="GATE1_PERFORMANCE_THEN_INCREMENTAL_1B_2_3_SHADOW_ONLY",
        final_holdout_boundaries={"contract":holdout_contract,"start":holdout_start},
        initial_pre_holdout_state_hash=initial_state_hash,
    )
    write_algorithm_manifest(output_root/"freeze"/"algorithm-freeze-manifest.json",freeze)
    artifact_paths=sorted(path for path in output_root.rglob("*") if path.is_file()
                          and path.name not in {"manifest.json"})
    hash_cache={}
    artifact_hashes={}
    for path in artifact_paths:
        stat=path.stat(); identity=(stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns)
        digest=hash_cache.get(identity)
        if digest is None:
            digest=_sha256_file(path); hash_cache[identity]=digest
        artifact_hashes[str(path.relative_to(output_root)).replace("\\","/")]=digest
    nav_semantic_hash=""
    if nav_path.is_file():
        nav=pd.read_parquet(nav_path).sort_values(["family_id","arm","date"]).reset_index(drop=True)
        nav_semantic_hash=stable_hash(nav.astype(object).where(pd.notna(nav),None).to_dict(orient="records"))
    manifest={"schema_version":"DYNAMIC_QBD_LOCAL_RUN_MANIFEST_V1","execution_environment":"LOCAL_AUTHORITATIVE",
              "execution_resources":active_cpu_contract(),
              "git_commit":code_identity["commit"],"code_identity":code_identity,
              "random_seeds":sorted({int(x.random_seed) for x in families}),
              "date_range":{"start":start,"end":end},"data_fingerprints":{
                  "signal_panel":_sha256_file(signal_panel),"candidate_metrics":_sha256_file(candidate_metrics),
                  "prices":_sha256_file(prices_path),
                  "market_state":_sha256_file(Path(market_state)) if market_state else "DERIVED_FROM_PRICES_V1"},
              "family_registry_hash":registry.family_registry_hash,
              "run_contract_sha256":run_contract["run_contract_sha256"],"freeze_manifest_hash":freeze["manifest_hash"],
              "artifact_sha256":artifact_hashes,"capital_authority":False,"router_mode":"SHADOW_RESEARCH_ONLY"}
    manifest["nav_semantic_hash"]=nav_semantic_hash
    manifest["reproducibility_fingerprint"]=stable_hash({
        "run_contract_sha256":run_contract["run_contract_sha256"],
        "pipeline_fingerprint":provenance["pipeline_fingerprint"],
        "freeze_manifest_hash":freeze["manifest_hash"],
        "nav_semantic_hash":nav_semantic_hash,
        "generation_fingerprints":sorted(generations["generation_fingerprint"].astype(str))
            if "generation_fingerprint" in generations else sorted(generations["generation_id"].astype(str)),
    })
    manifest_path=output_root/"manifest.json"
    if manifest_path.is_file():
        prior=json.loads(manifest_path.read_text(encoding="utf-8"))
        if prior.get("reproducibility_fingerprint") != manifest["reproducibility_fingerprint"]:
            raise ValueError("DYNAMIC_QBD_RESTART_OUTPUT_MISMATCH")
    _write_atomic_json(manifest_path,manifest)
    return provenance


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--signal-panel", required=True)
    parser.add_argument("--candidate-metrics", required=True)
    price_group=parser.add_mutually_exclusive_group(required=True)
    price_group.add_argument("--prices")
    price_group.add_argument("--daily-store-root")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--family-id", action="append", default=[])
    surface_group=parser.add_mutually_exclusive_group()
    surface_group.add_argument("--full-fixed-surface", action="store_true")
    surface_group.add_argument("--full-surface-with-learned-exit", action="store_true")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--holdout-contract", choices=sorted(HOLDOUT_CONTRACTS), required=True)
    parser.add_argument("--feature-schema-sha256", required=True)
    parser.add_argument("--learned-exit-candidate-metrics")
    parser.add_argument("--market-state")
    parser.add_argument("--include-policy-optimization-arms", action="store_true")
    parser.add_argument("--training-window-sessions",type=int,default=504)
    parser.add_argument("--calibration-window-sessions",type=int,default=252)
    parser.add_argument("--score-quantile",type=float,default=.975)
    parser.add_argument("--top-fraction",type=float,default=.005)
    parser.add_argument("--model-workers",type=int,default=DEFAULT_MODEL_WORKERS,
                        help="Independent model-fit workers; default 26 on the current 24-core/32-logical host.")
    parser.add_argument("--native-threads-per-worker", type=int, default=None,
                        help="Native numerical threads per model worker; defaults to logical CPUs / workers.")
    parser.add_argument("--cpu-peak-fraction",type=float,default=.85,
                        help="Target runnable CPU capacity; 0.85 maps to 26/32 logical CPUs on the current host.")
    parser.add_argument("--nwinfo-executable", default=None,
                        help="Optional NWinfo executable path for diagnostic host telemetry.")
    parser.add_argument("--nwinfo-interval-seconds", type=float, default=60.0,
                        help="NWinfo diagnostic sample interval; minimum 10 seconds.")
    args = parser.parse_args(argv)
    from .dynamic_qbd_runtime_resources import configure_cpu_peak
    configure_cpu_peak(args.cpu_peak_fraction, process_workers=args.model_workers,
                       native_threads_per_worker=args.native_threads_per_worker)
    family_ids=list(args.family_id)
    if args.full_fixed_surface or args.full_surface_with_learned_exit:
        family_ids.extend(x.family_id for x in build_family_specs(feature_schema_sha256="AUTO",
                          include_learned_exit=args.full_surface_with_learned_exit))
    with NWInfoSampler(Path(args.output_root) / "telemetry",
                       interval_seconds=args.nwinfo_interval_seconds,
                       executable=args.nwinfo_executable):
        run_pipeline(signal_panel=args.signal_panel, candidate_metrics=args.candidate_metrics, prices=args.prices,
                     daily_store_root=args.daily_store_root,
                     output_root=args.output_root, family_ids=family_ids, start=date.fromisoformat(args.start),
                     end=date.fromisoformat(args.end), holdout_contract=args.holdout_contract,
                     feature_schema_sha256=args.feature_schema_sha256,
                     learned_exit_candidate_metrics=args.learned_exit_candidate_metrics,
                     market_state=args.market_state,
                     include_policy_optimization_arms=args.include_policy_optimization_arms,
                     training_window_sessions=args.training_window_sessions,
                     calibration_window_sessions=args.calibration_window_sessions,
                     score_quantile=args.score_quantile,top_fraction=args.top_fraction,
                     model_workers=args.model_workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
