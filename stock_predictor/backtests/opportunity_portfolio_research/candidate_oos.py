"""Causal Candidate-OOS factory and immutable evidence store.

The factory deliberately reuses ``stock_predictor.v5.train`` for model
construction and prediction.  It owns only panel adaptation, fold boundaries,
label maturity, content addressing and artifact lifecycle.
"""
from __future__ import annotations

import gc

from dataclasses import dataclass, asdict
from datetime import date
import hashlib
import json
import math
import os
import pickle
from pathlib import Path
import shutil
from typing import Any, Iterable, Mapping

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from stock_predictor.v5 import train as v5_train
from stock_predictor.v5.dataset_builder import canonical_sha256
from stock_predictor.v5.walk_forward import Fold, expanding_folds
from .qbd_training_selection_contracts import (FoldPolicy, TargetContract, RecipeSelectionPolicy,
                             frozen_primary_candidate_registry)
from .development_slice_hash import parquet_development_slice_sha256
from .dynamic_qbd_shared_compute_store import compute_key_lock


OBSERVATION_COLUMNS = (
    "horizon", "fold_id", "candidate_id", "recipe_family", "hyperparameters",
    "symbol", "decision_date", "prediction", "terminal_date",
    "information_available_at", "realized_return", "benchmark_return",
    "realized_excess", "status", "candidate_spec_hash", "fold_spec_hash",
    "signal_panel_sha256", "feature_schema_sha256", "model_artifact_sha256",
    "fold_policy_hash", "target_contract_hash",
)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_development_panel(*, path: str | Path, columns: Iterable[str], development_end: date,
                           holdout_boundary: date = date(2026, 7, 25),
                           decision_dates: Iterable[date | str] | None = None) -> pd.DataFrame:
    """Read only development rows, with the physical holdout excluded by Arrow."""
    if development_end >= holdout_boundary or holdout_boundary > date(2026, 7, 25):
        raise PermissionError("QBD_HOLDOUT_BOUNDARY_OPEN")
    dataset = ds.dataset(str(path), format="parquet")
    names = set(dataset.schema.names)
    requested = list(dict.fromkeys(str(x) for x in columns))
    missing = sorted(set(requested) - names)
    if missing:
        raise ValueError(f"QBD_PANEL_COLUMNS_MISSING:{missing}")
    date_type = dataset.schema.field("decision_date").type
    if pa.types.is_string(date_type) or pa.types.is_large_string(date_type):
        boundary = pa.scalar(holdout_boundary.isoformat())
        end = pa.scalar(development_end.isoformat())
    else:
        boundary = pa.scalar(pd.Timestamp(holdout_boundary).to_datetime64())
        end = pa.scalar(pd.Timestamp(development_end).to_datetime64())
    predicate = (ds.field("decision_date") <= end) & (ds.field("decision_date") < boundary)
    if decision_dates is not None:
        values = [pd.Timestamp(x).date().isoformat() if pa.types.is_string(date_type) or pa.types.is_large_string(date_type)
                  else pd.Timestamp(x).to_datetime64() for x in decision_dates]
        predicate = predicate & ds.field("decision_date").isin(values)
    if "holdout_locked" in names:
        predicate = predicate & ((ds.field("holdout_locked") == False) | ds.field("holdout_locked").is_null())
    table = dataset.to_table(columns=requested, filter=predicate)
    frame = table.to_pandas()
    if not frame.empty:
        dates = pd.to_datetime(frame["decision_date"]).dt.date
        if (dates >= holdout_boundary).any():
            raise PermissionError("QBD_HOLDOUT_ROW_READ_DETECTED")
    return frame


@dataclass(frozen=True)
class CandidateSpec:
    candidate_id: str
    recipe_family: str
    hyperparameters: Mapping[str, Any]
    candidate_spec_hash: str

    @classmethod
    def create(cls, recipe_family: str, hyperparameters: Mapping[str, Any]) -> "CandidateSpec":
        params = json.loads(json.dumps(dict(hyperparameters), sort_keys=True, default=str))
        candidate_id = canonical_sha256({"recipe_family": recipe_family, "hyperparameters": params})[:16]
        return cls(candidate_id, recipe_family, params, canonical_sha256({"candidate_id": candidate_id, "recipe_family": recipe_family, "hyperparameters": params}))


def load_candidate_registry(path: str | Path) -> tuple[CandidateSpec, ...]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    families = raw.get("families") if isinstance(raw, Mapping) else None
    if not isinstance(families, Mapping):
        raise ValueError("CANDIDATE_REGISTRY_FAMILIES_MISSING")
    specs = []
    for family in sorted(families):
        values = families[family]
        if not isinstance(values, list):
            raise ValueError(f"CANDIDATE_REGISTRY_PARAMETERS_NOT_LIST:{family}")
        for params in values:
            if not isinstance(params, Mapping):
                raise ValueError(f"CANDIDATE_REGISTRY_PARAMETERS_NOT_OBJECT:{family}")
            specs.append(CandidateSpec.create(str(family), params))
    if len({x.candidate_id for x in specs}) != len(specs):
        raise ValueError("CANDIDATE_REGISTRY_DUPLICATE_ID")
    return tuple(specs)


def load_primary_candidate_registry(path: str | Path) -> tuple[CandidateSpec, ...]:
    """Load only the frozen Ridge/HGB primary universe.

    The generic hyperparameter file is an input source, not the authority for
    the primary universe.  The filter is intentionally explicit so adding a
    shadow/incubator family cannot silently change a primary run.
    """
    all_specs = load_candidate_registry(path)
    records = [asdict(x) for x in all_specs if x.recipe_family in {"RIDGE_LOGISTIC", "HIST_GRADIENT_BOOSTING"}]
    # The primary contract identity is intentionally independent of unrelated
    # shadow entries appended to the generic source file.
    frozen = frozen_primary_candidate_registry(records, source_contract="QBD_PRIMARY_RIDGE_HGB_V1")
    return tuple(CandidateSpec(**record) for record in frozen["records"])


def candidate_registry_document(specs: Iterable[CandidateSpec], *, source_sha256: str) -> dict:
    records = [asdict(x) for x in sorted(specs, key=lambda x: x.candidate_id)]
    document = {"schema_version": "DYNAMIC_QBD_CANDIDATE_REGISTRY_V1", "source_sha256": source_sha256, "records": records}
    document["candidate_registry_sha256"] = canonical_sha256(document)
    return document


def resolve_information_available_at(*, decision_date: str, terminal_date: str) -> str:
    decision = pd.Timestamp(decision_date).normalize()
    terminal = pd.Timestamp(terminal_date).normalize()
    if not decision < terminal:
        raise ValueError("CANDIDATE_OOS_TERMINAL_NOT_AFTER_DECISION")
    return terminal.date().isoformat()


def mature_decision_sessions(*, trading_sessions: Iterable[date], information_cutoff: date,
                             horizon: int) -> tuple[date, ...]:
    """Return decision sessions whose exact H-terminal session is available."""
    sessions = tuple(sorted(pd.Timestamp(x).date() for x in trading_sessions))
    cutoff = pd.Timestamp(information_cutoff).date()
    return tuple(decision for index, decision in enumerate(sessions)
                 if index + int(horizon) < len(sessions)
                 and sessions[index + int(horizon)] <= cutoff)


def fold_information_available_at(*, fold: Fold, horizon: int,
                                  trading_sessions: Iterable[date | str]) -> date | None:
    """Return exact session maturity of a fold, or None if terminal is unknown."""
    sessions = tuple(pd.Timestamp(x).normalize() for x in trading_sessions)
    index = {value: position for position, value in enumerate(sessions)}
    terminals = []
    for value in fold.validation_dates:
        decision = pd.Timestamp(value).normalize()
        position = index.get(decision)
        if position is None or position + int(horizon) >= len(sessions):
            return None
        terminals.append(sessions[position + int(horizon)])
    return terminals[-1].date() if terminals else None


def _fold_hash(fold: Fold, horizon: int) -> str:
    return canonical_sha256({"horizon": horizon, "fold": {"fold_id": fold.fold_id, "train_dates": fold.train_dates,
                                                           "validation_dates": fold.validation_dates, "purged_dates": fold.purged_dates,
                                                           "embargoed_dates": fold.embargoed_dates}})


class CandidateOosStore:
    """Immutable partitioned Parquet store with manifest hash verification."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def partition_path(self, horizon: int, fold_id: str, candidate_id: str, cache_key: str | None = None) -> Path:
        suffix = str(cache_key or "legacy")
        return self.root / f"H{int(horizon):02d}" / f"fold_{fold_id}" / candidate_id / suffix / "observations.parquet"

    def write(self, frame: pd.DataFrame, *, horizon: int, fold: Fold, candidate: CandidateSpec,
              signal_panel_sha256: str, feature_schema_sha256: str, model_artifact_sha256: str,
              model_bytes: bytes | None = None, fold_policy_hash: str = "",
              target_contract_hash: str = "", model_training_contract_hash: str = "",
              execution_backend: str = "SKLEARN_CANONICAL_CPU") -> dict:
        missing = sorted(set(OBSERVATION_COLUMNS) - set(frame.columns))
        if missing:
            raise ValueError(f"CANDIDATE_OOS_SCHEMA_MISSING:{missing}")
        keys = ["horizon", "fold_id", "candidate_id", "symbol", "decision_date"]
        if frame[keys].duplicated().any():
            raise ValueError("CANDIDATE_OOS_DUPLICATE_OBSERVATION")
        for column in ("decision_date", "terminal_date", "information_available_at"):
            frame[column] = pd.to_datetime(frame[column]).dt.date.astype(str)
        decision = pd.to_datetime(frame["decision_date"])
        terminal = pd.to_datetime(frame["terminal_date"])
        available = pd.to_datetime(frame["information_available_at"])
        if not ((decision < terminal) & (terminal <= available)).all():
            raise ValueError("CANDIDATE_OOS_MATURITY_ORDER_INVALID")
        if frame["status"].isin(["INVALID"]).any():
            raise ValueError("CANDIDATE_OOS_INVALID_OBSERVATION")
        cache_key = canonical_sha256({"signal_panel_sha256": signal_panel_sha256,
                                      "feature_schema_sha256": feature_schema_sha256,
                                      "candidate_spec_hash": candidate.candidate_spec_hash,
                                      "fold_policy_hash": fold_policy_hash,
                                      "target_contract_hash": target_contract_hash,
                                      "model_training_contract_hash": model_training_contract_hash,
                                      "execution_backend": execution_backend,
                                      "horizon": int(horizon), "fold_id": fold.fold_id})[:32]
        target = self.partition_path(horizon, fold.fold_id, candidate.candidate_id, cache_key)
        manifest = target.with_name("manifest.json")
        payload = frame.loc[:, OBSERVATION_COLUMNS].sort_values(list(keys)).copy()
        final_dir = target.parent
        model_path = target.with_name("model.pkl")
        if model_bytes is not None and hashlib.sha256(model_bytes).hexdigest() != model_artifact_sha256:
            raise ValueError("CANDIDATE_OOS_MODEL_HASH_MISMATCH")
        if target.exists() or manifest.exists():
            if not target.exists() or not manifest.exists():
                raise ValueError("CANDIDATE_OOS_IMMUTABLE_PARTITION_CONFLICT")
            existing = json.loads(manifest.read_text(encoding="utf-8"))
            if existing.get("artifact_sha256") != file_sha256(target) or existing.get("cache_key") != cache_key:
                raise ValueError("CANDIDATE_OOS_IMMUTABLE_PARTITION_CONFLICT")
            return existing | {"cache": "HIT", "path": str(target)}
        temp_dir = final_dir.parent / f".{final_dir.name}.tmp.{os.getpid()}"
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        temp_dir.mkdir(parents=True, exist_ok=False)
        temp_target = temp_dir / "observations.parquet"
        temp_model = temp_dir / "model.pkl"
        temp_manifest = temp_dir / "manifest.json"
        if model_bytes is not None:
            temp_model.write_bytes(model_bytes)
        payload.to_parquet(temp_target, index=False)
        artifact_sha = file_sha256(temp_target)
        metadata = {"schema_version": "DYNAMIC_QBD_CANDIDATE_OOS_PARTITION_V1", "horizon": int(horizon),
                    "fold_id": fold.fold_id, "fold_spec_hash": _fold_hash(fold, horizon),
                    "candidate_id": candidate.candidate_id, "candidate_spec_hash": candidate.candidate_spec_hash,
                    "signal_panel_sha256": signal_panel_sha256, "feature_schema_sha256": feature_schema_sha256,
                    "model_artifact_sha256": model_artifact_sha256, "fold_policy_hash": fold_policy_hash,
                    "target_contract_hash": target_contract_hash,
                    "model_training_contract_hash": model_training_contract_hash,
                    "execution_backend": str(execution_backend),
                    "cache_key": cache_key, "artifact_sha256": artifact_sha,
                    "rows": int(len(payload))}
        metadata["manifest_sha256"] = canonical_sha256(metadata)
        temp_manifest.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        final_dir.parent.mkdir(parents=True, exist_ok=True)
        try:
            temp_dir.replace(final_dir)
        except FileExistsError:
            shutil.rmtree(temp_dir, ignore_errors=True)
            if not target.is_file() or not manifest.is_file() or file_sha256(target) != artifact_sha:
                raise ValueError("CANDIDATE_OOS_IMMUTABLE_PARTITION_CONFLICT")
            return metadata | {"cache": "HIT", "path": str(target)}
        return metadata | {"cache": "MISS", "path": str(target), "model_path": str(model_path)}

    def observation_paths(self, *, horizon: int,
                          partition_paths: Iterable[str | Path] | None = None) -> tuple[Path, ...]:
        """Return the immutable observation partitions for one horizon.

        A resumed store can retain older cache directories for the same
        candidate/fold.  The manifested job result is the canonical selector
        for those directories; callers may pass those paths explicitly.  The
        historical glob remains available for standalone store inspection.
        """
        if partition_paths is None:
            values = (self.root / f"H{int(horizon):02d}").glob(
                "fold_*/**/observations.parquet")
        else:
            values = partition_paths
        return tuple(sorted({Path(value) for value in values}))

    def read_matured(self, *, horizon: int, selection_cutoff: date,
                     partition_paths: Iterable[str | Path] | None = None) -> pd.DataFrame:
        """Read only matured rows required by one causal snapshot.

        v40.0.2 loaded every full Candidate-OOS partition and filtered it in
        pandas. Under parallel coverage jobs that multiplied Arrow/pandas
        working sets before the RAM controller could learn the class. Push the
        maturity predicate and column projection into Arrow first, then retain
        only the already-filtered frames required by the selection contract.
        """
        parts = self.observation_paths(
            horizon=horizon, partition_paths=partition_paths)
        frames: list[pd.DataFrame] = []
        cutoff_text = pd.Timestamp(selection_cutoff).date().isoformat()
        for part in parts:
            try:
                frame = pd.read_parquet(
                    part,
                    columns=list(OBSERVATION_COLUMNS),
                    filters=[
                        ("status", "==", "MATURED"),
                        ("information_available_at", "<=", cutoff_text),
                    ],
                )
            except (TypeError, ValueError):
                # Compatibility for historical partitions whose Arrow physical
                # date type cannot compare directly to the string cutoff.
                frame = pd.read_parquet(
                    part, columns=list(OBSERVATION_COLUMNS))
                available = pd.to_datetime(
                    frame["information_available_at"], errors="coerce")
                frame = frame.loc[
                    frame["status"].eq("MATURED")
                    & available.le(pd.Timestamp(selection_cutoff))
                ].copy()
            if frame.empty:
                continue
            frame["information_available_at"] = pd.to_datetime(
                frame["information_available_at"])
            frames.append(frame)
        if not frames:
            return pd.DataFrame(columns=OBSERVATION_COLUMNS)
        if len(frames) == 1:
            return frames[0].reset_index(drop=True)
        return pd.concat(frames, ignore_index=True, copy=False)

    def verify_partition(self, path: str | Path) -> dict:
        target = Path(path)
        manifest_path = target.with_name("manifest.json")
        if not target.is_file() or not manifest_path.is_file():
            raise ValueError("CANDIDATE_OOS_PARTITION_OR_MANIFEST_MISSING")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        actual = file_sha256(target)
        if actual != manifest.get("artifact_sha256"):
            raise ValueError("CANDIDATE_OOS_PARTITION_HASH_MISMATCH")
        if canonical_sha256({k: v for k, v in manifest.items() if k != "manifest_sha256"}) != manifest.get("manifest_sha256"):
            raise ValueError("CANDIDATE_OOS_MANIFEST_HASH_MISMATCH")
        frame = pd.read_parquet(target)
        missing = sorted(set(OBSERVATION_COLUMNS) - set(frame.columns))
        if missing:
            raise ValueError(f"CANDIDATE_OOS_PARTITION_SCHEMA_MISSING:{missing}")
        if len(frame) != int(manifest.get("rows", -1)):
            raise ValueError("CANDIDATE_OOS_PARTITION_ROW_COUNT_MISMATCH")
        expected = {"horizon": int(manifest["horizon"]), "fold_id": str(manifest["fold_id"]),
                    "candidate_id": str(manifest["candidate_id"]),
                    "candidate_spec_hash": str(manifest["candidate_spec_hash"])}
        for column, value in (("horizon", expected["horizon"]), ("fold_id", expected["fold_id"]),
                              ("candidate_id", expected["candidate_id"]), ("candidate_spec_hash", expected["candidate_spec_hash"])):
            if frame[column].astype(str).nunique() != 1 or str(frame[column].iloc[0]) != str(value):
                raise ValueError("CANDIDATE_OOS_PARTITION_IDENTITY_MISMATCH")
        if (pd.to_datetime(frame["decision_date"]) >= pd.to_datetime(frame["terminal_date"])).any():
            raise ValueError("CANDIDATE_OOS_PARTITION_MATURITY_INVALID")
        for column in ("prediction", "realized_return", "benchmark_return", "realized_excess"):
            values = pd.to_numeric(frame[column], errors="coerce")
            if not values.map(lambda value: pd.notna(value) and math.isfinite(float(value))).all():
                raise ValueError(f"CANDIDATE_OOS_PARTITION_NONFINITE:{column}")
        model_path = target.with_name("model.pkl")
        if model_path.is_file() and file_sha256(model_path) != str(manifest.get("model_artifact_sha256")):
            raise ValueError("CANDIDATE_OOS_MODEL_HASH_MISMATCH")
        return manifest

    def verify_partition_metadata(self, path: str | Path) -> dict:
        """Verify immutable identity without materializing the parquet frame.

        Snapshot construction has already consumed the filtered observations.
        Re-reading every full partition solely to recover its immutable
        identity doubled the coverage working set in v40.0.2. Artifact hash,
        manifest self-hash, parquet schema/row count and model hash are enough
        to prove that the exact previously-written immutable partition is the
        one named by the snapshot.
        """
        target = Path(path)
        manifest_path = target.with_name("manifest.json")
        if not target.is_file() or not manifest_path.is_file():
            raise ValueError("CANDIDATE_OOS_PARTITION_OR_MANIFEST_MISSING")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if file_sha256(target) != manifest.get("artifact_sha256"):
            raise ValueError("CANDIDATE_OOS_PARTITION_HASH_MISMATCH")
        if canonical_sha256({
            k: v for k, v in manifest.items()
            if k != "manifest_sha256"
        }) != manifest.get("manifest_sha256"):
            raise ValueError("CANDIDATE_OOS_MANIFEST_HASH_MISMATCH")
        parquet = pq.ParquetFile(target)
        missing = sorted(
            set(OBSERVATION_COLUMNS) - set(parquet.schema_arrow.names))
        if missing:
            raise ValueError(
                f"CANDIDATE_OOS_PARTITION_SCHEMA_MISSING:{missing}")
        if int(parquet.metadata.num_rows) != int(manifest.get("rows", -1)):
            raise ValueError("CANDIDATE_OOS_PARTITION_ROW_COUNT_MISMATCH")
        model_path = target.with_name("model.pkl")
        if (
            model_path.is_file()
            and file_sha256(model_path)
                != str(manifest.get("model_artifact_sha256"))
        ):
            raise ValueError("CANDIDATE_OOS_MODEL_HASH_MISMATCH")
        return manifest


@dataclass(frozen=True)
class EvidenceSnapshot:
    horizon: int
    selection_cutoff: date
    partition_identities: tuple[str, ...]
    candidate_registry_sha256: str
    evidence_snapshot_sha256: str
    rows: int
    candidate_count: int
    fold_count: int


def persist_evidence_snapshot(store: CandidateOosStore, *, horizon: int, selection_cutoff: date,
                              candidate_registry_sha256: str, fold_policy_hash: str,
                              signal_panel_hash: str, target_contract_hash: str,
                              output_root: str | Path,
                              snapshot: EvidenceSnapshot | None = None,
                              frame: pd.DataFrame | None = None) -> dict:
    """Materialize one immutable snapshot without letting valid old leaves kill resume.

    The canonical H×cutoff leaf remains immutable.  If a compatible slow
    resume discovers that the causally matured evidence set has changed while
    a valid older leaf already exists, the older leaf is preserved and the
    corrected snapshot is published under a deterministic manifest-hash
    version.  Corrupt/partial artifacts still fail closed.
    """
    if snapshot is None or frame is None:
        snapshot, frame = build_evidence_snapshot(
            store, horizon=horizon, selection_cutoff=selection_cutoff,
            candidate_registry_sha256=candidate_registry_sha256)
    expected_keys = sorted({(int(x), str(y), str(z)) for x, y, z in frame[["horizon", "candidate_id", "fold_id"]].drop_duplicates().itertuples(index=False, name=None)}) if not frame.empty else []
    manifest = {"schema_version": "DYNAMIC_QBD_EVIDENCE_SNAPSHOT_V1", "horizon": int(horizon),
                "selection_cutoff": str(selection_cutoff), "expected_keys": [list(x) for x in expected_keys],
                "observation_count": int(len(frame)), "candidate_count": snapshot.candidate_count,
                "fold_count": snapshot.fold_count, "candidate_registry_hash": candidate_registry_sha256,
                "fold_policy_hash": fold_policy_hash, "signal_panel_hash": signal_panel_hash,
                "target_contract_hash": target_contract_hash, "evidence_snapshot_hash": snapshot.evidence_snapshot_sha256,
                "partition_identities": list(snapshot.partition_identities)}
    root = Path(output_root) / f"H{int(horizon):02d}" / str(selection_cutoff)
    temp = root.parent / f".{root.name}.tmp.{os.getpid()}"
    if temp.exists():
        shutil.rmtree(temp)
    temp.mkdir(parents=True, exist_ok=False)
    frame.to_parquet(temp / "snapshot.parquet", index=False)
    manifest["snapshot_sha256"] = file_sha256(temp / "snapshot.parquet")
    manifest["manifest_sha256"] = canonical_sha256(manifest)

    def validated_existing(target_root: Path) -> dict | None:
        artifact = target_root / "snapshot.parquet"
        manifest_path = target_root / "manifest.json"
        if not artifact.exists() and not manifest_path.exists():
            return None
        if not artifact.is_file() or not manifest_path.is_file():
            raise ValueError("EVIDENCE_SNAPSHOT_IMMUTABLE_CONFLICT")
        prior = json.loads(manifest_path.read_text(encoding="utf-8"))
        if canonical_sha256({k: v for k, v in prior.items() if k != "manifest_sha256"}) != prior.get("manifest_sha256"):
            raise ValueError("EVIDENCE_SNAPSHOT_MANIFEST_HASH_MISMATCH")
        if file_sha256(artifact) != prior.get("snapshot_sha256"):
            raise ValueError("EVIDENCE_SNAPSHOT_CONTENT_HASH_MISMATCH")
        return prior

    target_root = root
    cache = "MISS"
    conflict_resolution = None
    prior = validated_existing(root)
    if prior is not None:
        if prior.get("manifest_sha256") == manifest["manifest_sha256"]:
            shutil.rmtree(temp)
            return prior | {
                "snapshot_path": str(root / "snapshot.parquet"),
                "manifest_path": str(root / "manifest.json"),
                "cache": "HIT",
            }
        # A valid immutable leaf may describe an older resumable DAG state.
        # Never overwrite it and never abort the whole run merely because a
        # corrected causally matured set now exists.  The new manifest hash is
        # a deterministic derived-state identity, so subsequent fast resumes
        # hit exactly this corrected version and skip the expensive job.
        target_root = root / "versions" / manifest["manifest_sha256"]
        conflict_resolution = {
            "status": "VALID_PRIOR_LEAF_PRESERVED",
            "superseded_manifest_sha256": prior.get("manifest_sha256"),
            "resolution": "PUBLISH_CONTENT_VERSION_AND_SKIP_CONFLICTING_LEAF",
        }
        versioned = validated_existing(target_root)
        if versioned is not None:
            shutil.rmtree(temp)
            if versioned.get("manifest_sha256") != manifest["manifest_sha256"]:
                raise ValueError("EVIDENCE_SNAPSHOT_IMMUTABLE_CONFLICT")
            return versioned | {
                "snapshot_path": str(target_root / "snapshot.parquet"),
                "manifest_path": str(target_root / "manifest.json"),
                "cache": "HIT_VERSIONED_IMMUTABLE_CONFLICT",
                "immutable_conflict_resolution": conflict_resolution,
            }
        cache = "MISS_VERSIONED_IMMUTABLE_CONFLICT"

    (temp / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    target_root.parent.mkdir(parents=True, exist_ok=True)
    try:
        temp.replace(target_root)
    except OSError:
        # Another resumable worker may have won the deterministic publication
        # race.  Accept only the exact validated manifest; unrelated I/O
        # failures and partial artifacts still propagate/fail closed.
        raced = validated_existing(target_root)
        if raced is None or raced.get("manifest_sha256") != manifest["manifest_sha256"]:
            raise
        if temp.exists():
            shutil.rmtree(temp)
        return raced | {
            "snapshot_path": str(target_root / "snapshot.parquet"),
            "manifest_path": str(target_root / "manifest.json"),
            "cache": (
                "HIT_VERSIONED_IMMUTABLE_CONFLICT"
                if conflict_resolution is not None else "HIT"),
            **({"immutable_conflict_resolution": conflict_resolution}
               if conflict_resolution is not None else {}),
        }
    return manifest | {
        "snapshot_path": str(target_root / "snapshot.parquet"),
        "manifest_path": str(target_root / "manifest.json"),
        "cache": cache,
        **({"immutable_conflict_resolution": conflict_resolution}
           if conflict_resolution is not None else {}),
    }

def read_evidence_snapshot_artifact(snapshot_path: str | Path) -> tuple[dict, pd.DataFrame]:
    artifact = Path(snapshot_path)
    manifest_path = artifact.with_name("manifest.json")
    if not artifact.is_file() or not manifest_path.is_file():
        raise ValueError("EVIDENCE_SNAPSHOT_ARTIFACT_MISSING")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if canonical_sha256({k: v for k, v in manifest.items() if k != "manifest_sha256"}) != manifest.get("manifest_sha256"):
        raise ValueError("EVIDENCE_SNAPSHOT_MANIFEST_HASH_MISMATCH")
    if file_sha256(artifact) != manifest.get("snapshot_sha256"):
        raise ValueError("EVIDENCE_SNAPSHOT_CONTENT_HASH_MISMATCH")
    frame = pd.read_parquet(artifact)
    if len(frame) != int(manifest["observation_count"]):
        raise ValueError("EVIDENCE_SNAPSHOT_ROW_COUNT_MISMATCH")
    if pd.to_datetime(frame["information_available_at"]).gt(pd.Timestamp(manifest["selection_cutoff"])).any():
        raise PermissionError("EVIDENCE_SNAPSHOT_FUTURE_INFORMATION")
    return manifest, frame


def build_evidence_snapshot(store: CandidateOosStore, *, horizon: int, selection_cutoff: date,
                            candidate_registry_sha256: str,
                            partition_paths: Iterable[str | Path] | None = None) -> tuple[EvidenceSnapshot, pd.DataFrame]:
    parts = store.observation_paths(
        horizon=horizon, partition_paths=partition_paths)
    frame = store.read_matured(
        horizon=horizon, selection_cutoff=selection_cutoff,
        partition_paths=parts)
    identities = []
    eligible_partitions = set()
    if not frame.empty:
        eligible_partitions = {(str(x), str(y)) for x, y in frame[["fold_id", "candidate_id"]].drop_duplicates().itertuples(index=False, name=None)}
    for part in parts:
        manifest = part.with_name("manifest.json")
        if not manifest.is_file():
            raise ValueError("CANDIDATE_OOS_SNAPSHOT_MANIFEST_MISSING")
        metadata = store.verify_partition_metadata(part)
        # The snapshot's admissibility is determined row-wise by
        # information_available_at.  Include only partitions contributing
        # matured rows; the existence of future partitions must not alter a
        # historical snapshot identity.
        if (str(metadata["fold_id"]), str(metadata["candidate_id"])) in eligible_partitions:
            identities.append(f"{metadata['fold_id']}:{metadata['candidate_id']}:{metadata['artifact_sha256']}")
    identities = tuple(sorted(identities))
    payload = {"horizon": int(horizon), "selection_cutoff": str(selection_cutoff),
               "partition_identities": identities, "candidate_registry_sha256": candidate_registry_sha256,
               "rows": int(len(frame))}
    snapshot = EvidenceSnapshot(int(horizon), selection_cutoff, identities, candidate_registry_sha256,
                                canonical_sha256(payload), int(len(frame)),
                                int(frame["candidate_id"].nunique()) if not frame.empty else 0,
                                int(frame["fold_id"].nunique()) if not frame.empty else 0)
    return snapshot, frame


def select_recipe_from_snapshot(snapshot: EvidenceSnapshot, evidence: pd.DataFrame,
                                *, selection_policy_sha256: str, fold_policy_hash: str = "",
                                target_contract_hash: str = "",
                                selection_policy: RecipeSelectionPolicy | None = None) -> dict:
    """Select a recipe from one explicit, already-cut EvidenceSnapshot."""
    if evidence.empty:
        raise ValueError("RECIPE_SELECTION_SNAPSHOT_EMPTY")
    if int(snapshot.horizon) not in set(evidence["horizon"].astype(int)):
        raise ValueError("RECIPE_SELECTION_HORIZON_MISMATCH")
    if pd.to_datetime(evidence["information_available_at"]).gt(pd.Timestamp(snapshot.selection_cutoff)).any():
        raise PermissionError("RECIPE_SELECTION_FUTURE_EVIDENCE")
    policy = selection_policy or RecipeSelectionPolicy()
    rows = []
    for (candidate_id, fold_id), group in evidence.groupby(["candidate_id", "fold_id"], sort=True):
        if len(group) < policy.minimum_fold_observations:
            continue
        prediction = pd.to_numeric(group["prediction"], errors="coerce")
        outcome = pd.to_numeric(group["realized_excess"], errors="coerce")
        rank_prediction = prediction.rank(method="average")
        rank_outcome = outcome.rank(method="average")
        spearman = float(rank_prediction.corr(rank_outcome)) if rank_prediction.nunique() > 1 and rank_outcome.nunique() > 1 else 0.0
        mae = float((prediction - outcome).abs().mean())
        centered = outcome - outcome.mean()
        sse = float(((prediction - outcome) ** 2).sum())
        sst = float((centered ** 2).sum())
        rows.append({"candidate_id": str(candidate_id), "fold_id": str(fold_id),
                     "observation_count": int(len(group)), "spearman": spearman,
                     "mae": mae, "r2_diagnostic": float(1.0 - sse / sst) if sst > 0 else 0.0,
                     "mean_realized_excess_diagnostic": float(outcome.mean())})
    fold_metrics = pd.DataFrame(rows)
    if fold_metrics.empty:
        raise ValueError("RECIPE_SELECTION_NO_ELIGIBLE_PREDICTIVE_FOLDS")
    ranking = fold_metrics.groupby("candidate_id").agg(
        median_fold_spearman=("spearman", "median"), mean_fold_spearman=("spearman", "mean"),
        positive_fold_fraction=("spearman", lambda x: float((x > 0).mean())),
        mean_mae=("mae", "mean"), fold_count=("fold_id", "nunique"),
    ).reset_index()
    ranking = ranking.loc[ranking["fold_count"] >= policy.minimum_candidate_folds]
    if ranking.empty:
        raise ValueError("RECIPE_SELECTION_NO_CANDIDATE_WITH_MINIMUM_FOLDS")
    ranking = ranking.sort_values(["median_fold_spearman", "mean_fold_spearman", "positive_fold_fraction", "mean_mae", "candidate_id"],
                                  ascending=[False, False, False, True, True])
    winner = ranking.iloc[0].to_dict()
    runner = ranking.iloc[1].to_dict() if len(ranking) > 1 else None
    selected_rows = evidence.loc[evidence["candidate_id"].eq(winner["candidate_id"])]
    selected_hash = str(selected_rows["candidate_spec_hash"].iloc[0])
    recipe_family = str(selected_rows["recipe_family"].iloc[0])
    hyperparameters = json.loads(str(selected_rows["hyperparameters"].iloc[0]))
    artifact = {"schema_version": "DYNAMIC_QBD_SELECTED_RECIPE_V1", "horizon": snapshot.horizon,
                "refit_date": str(snapshot.selection_cutoff), "selection_cutoff": str(snapshot.selection_cutoff),
                "selected_candidate_id": str(winner["candidate_id"]), "candidate_spec_hash": selected_hash,
                "recipe_family": recipe_family, "hyperparameters": hyperparameters,
                "evidence_snapshot_sha256": snapshot.evidence_snapshot_sha256,
                "candidate_registry_sha256": snapshot.candidate_registry_sha256,
                "fold_policy_hash": fold_policy_hash, "target_contract_hash": target_contract_hash,
                "selection_algorithm_version": policy.version,
                "recipe_selection_policy_hash": policy.recipe_selection_policy_hash,
                "selection_policy_sha256": selection_policy_sha256, "winner_evidence": winner,
                "runner_up_evidence": runner,
                "fold_metrics": fold_metrics.to_dict(orient="records")}
    artifact["selected_recipe_sha256"] = canonical_sha256(artifact)
    return artifact


class CandidateOosFactory:
    """Build one Candidate×Horizon×Fold partition using V5 train primitives."""

    def __init__(self, *, signal_panel: str | Path, feature_schema: Mapping[str, Any], candidates: Iterable[CandidateSpec],
                 output_root: str | Path, signal_panel_sha256: str | None = None,
                 fold_policy: FoldPolicy, target_contract: TargetContract,
                 development_end: date | None = None, holdout_boundary: date = date(2026, 7, 25),
                 random_seed: int = 17, model_training_contract_hash: str = "",
                 prepared_cache_root: str | Path | None = None):
        self.signal_panel = Path(signal_panel)
        self.feature_schema = dict(feature_schema)
        self.candidates = {x.candidate_id: x for x in candidates}
        self.store = CandidateOosStore(output_root)
        self.development_end = development_end
        self.holdout_boundary = holdout_boundary
        self.signal_panel_sha256 = signal_panel_sha256
        self.feature_schema_sha256 = canonical_sha256(feature_schema)
        self.fold_policy = fold_policy
        self.target_contract = target_contract
        self.target_contract.validate_target_cost()
        self.random_seed = int(random_seed)
        self.model_training_contract_hash = str(model_training_contract_hash)
        self.prepared_cache_root = (
            Path(prepared_cache_root) if prepared_cache_root is not None else None)
        self._prepared_fold_cache: dict[tuple[Any, ...], dict[str, Any]] = {}
        # Shared prepared folds are large. When the immutable SSD cache is
        # enabled, retaining even one fold per long-lived worker multiplies
        # RAM by process count for no scientific benefit.
        self._prepared_fold_cache_limit = (
            0 if self.prepared_cache_root is not None else 2)
        allowed = set(self.feature_schema.get("v2_features", ())) | set(self.feature_schema.get("v3_features", ())) | set(self.feature_schema.get("execution_features", ())) | set(self.feature_schema.get("optional_unvalidated_diagnostics", {}).get("features", ()))
        self.feature_columns = tuple(sorted(allowed))
        self._frame: pd.DataFrame | None = None

    def _load(self, horizon: int, *, development_end: date | None = None,
              holdout_boundary: date | None = None) -> pd.DataFrame:
        if self._frame is None:
            cols = ["decision_date", "ticker", "sector", "sub_industry", "holdout_locked", *self.feature_columns]
            available = set(pq.read_schema(self.signal_panel).names)
            cols = list(dict.fromkeys([x for x in cols if x in available]))
            effective_end = development_end or self.development_end
            effective_boundary = holdout_boundary or self.holdout_boundary
            if effective_end is None:
                raise ValueError("CANDIDATE_OOS_DEVELOPMENT_END_REQUIRED")
            self._frame = read_development_panel(path=self.signal_panel, columns=cols,
                                                 development_end=effective_end,
                                                 holdout_boundary=effective_boundary)
        return self._frame

    def release_frame_cache(self) -> None:
        """Release the large process-local panel after fold preparation."""
        self._frame = None
        if self.prepared_cache_root is not None:
            self._prepared_fold_cache.clear()
        gc.collect()

    def fold_specs(self, *, horizon: int, development_end: date, holdout_boundary: date) -> tuple[Fold, ...]:
        if development_end >= date(2026, 7, 25) or holdout_boundary > date(2026, 7, 25):
            raise PermissionError("CANDIDATE_OOS_HOLDOUT_BOUNDARY_OPEN")
        if not self.signal_panel_sha256:
            self.signal_panel_sha256 = parquet_development_slice_sha256(
                self.signal_panel, date_column="decision_date", development_end=development_end,
                holdout_boundary=holdout_boundary)
        frame = self._load(horizon, development_end=development_end, holdout_boundary=holdout_boundary)
        sessions = sorted(set(pd.to_datetime(frame["decision_date"]).dt.date))
        development = [x.isoformat() for x in sessions if x < holdout_boundary and x <= development_end]
        return tuple(expanding_folds(development,
                                     minimum_train_dates=self.fold_policy.training_window_sessions,
                                     validation_dates=self.fold_policy.validation_window_sessions,
                                     step_dates=self.fold_policy.step_sessions,
                                     purge_dates=self.fold_policy.purge_sessions,
                                     embargo_dates=self.fold_policy.embargo_sessions))

    def _prepare_fold_inputs(
        self, *, horizon: int, fold: Fold, development_end: date,
        holdout_boundary: date,
    ) -> dict[str, Any]:
        """Prepare candidate-independent H×Fold rows exactly once per key."""
        if (development_end >= date(2026, 7, 25)
                or holdout_boundary > date(2026, 7, 25)):
            raise PermissionError("CANDIDATE_OOS_HOLDOUT_BOUNDARY_OPEN")
        identity = {
            "signal_panel_sha256": self.signal_panel_sha256,
            "feature_schema_sha256": self.feature_schema_sha256,
            "fold_policy_hash": self.fold_policy.fold_policy_hash,
            "target_contract_hash": self.target_contract.target_contract_hash,
            "model_training_contract_hash": self.model_training_contract_hash,
            "horizon": int(horizon),
            "fold_spec_hash": _fold_hash(fold, horizon),
            "development_end": development_end.isoformat(),
            "holdout_boundary": holdout_boundary.isoformat(),
        }
        cache_key = canonical_sha256(identity)
        local_key = (int(horizon), fold.fold_id, cache_key)
        cached = self._prepared_fold_cache.get(local_key)
        if cached is not None:
            return cached

        def remember(prepared: dict[str, Any]) -> dict[str, Any]:
            self._prepared_fold_cache[local_key] = prepared
            while len(self._prepared_fold_cache) > self._prepared_fold_cache_limit:
                self._prepared_fold_cache.pop(
                    next(iter(self._prepared_fold_cache)))
            return prepared

        def compute_prepared() -> dict[str, Any]:
            frame = self._load(
                horizon, development_end=development_end,
                holdout_boundary=holdout_boundary)
            date_values = pd.to_datetime(
                frame["decision_date"]).dt.date.astype(str)
            train = frame.loc[date_values.isin(fold.train_dates)].copy()
            test = frame.loc[
                date_values.isin(fold.validation_dates)].copy()
            label_col = self.target_contract.target_column(horizon)
            benchmark_col = self.target_contract.benchmark_column(horizon)
            schema_names = set(pq.read_schema(self.signal_panel).names)
            if benchmark_col not in schema_names:
                raise ValueError(
                    f"CANDIDATE_OOS_BENCHMARK_COLUMN_MISSING:"
                    f"{benchmark_col}")
            sessions = sorted(set(date_values))
            session_index = {x: i for i, x in enumerate(sessions)}

            def eligible_rows(source: pd.DataFrame) -> pd.DataFrame:
                dates = []
                for value in pd.to_datetime(
                        source["decision_date"]).dt.date.astype(str):
                    position = session_index[value] + int(horizon)
                    if (position < len(sessions)
                            and date.fromisoformat(
                                sessions[position]) <= development_end
                            and date.fromisoformat(
                                sessions[position]) < holdout_boundary):
                        dates.append(value)
                if not dates:
                    return source.iloc[0:0].copy()
                labels = read_development_panel(
                    path=self.signal_panel,
                    columns=[
                        "decision_date", "ticker", label_col, benchmark_col],
                    development_end=development_end,
                    holdout_boundary=holdout_boundary,
                    decision_dates=dates)
                return source.merge(
                    labels, on=["decision_date", "ticker"], how="inner",
                    validate="one_to_one")

            train = eligible_rows(train)
            test = eligible_rows(test)
            if label_col not in train or benchmark_col not in test:
                raise ValueError(
                    f"CANDIDATE_OOS_PANEL_COLUMNS_MISSING:"
                    f"{label_col},{benchmark_col}")

            def adapt(row: Mapping[str, Any]) -> dict[str, Any]:
                decision_date = pd.Timestamp(
                    row["decision_date"]).date().isoformat()
                snapshot = {
                    x: row[x] for x in self.feature_columns
                    if x in row and pd.notna(row[x])
                }
                labels = {
                    str(horizon): {
                        "y_excess_net": float(row[label_col]),
                        "y_positive_edge":
                            int(float(row[label_col]) > .002),
                        "y_downside": int(float(row[label_col]) < -.03),
                        "modeled_total_cost_bps": float(
                            self.target_contract.cost_model[
                                "roundtrip_bps"]),
                    }
                }
                benchmark_value = row.get(
                    f"benchmark_forward_return_{horizon}")
                if (pd.isna(benchmark_value)
                        or not math.isfinite(float(benchmark_value))):
                    raise ValueError(
                        f"CANDIDATE_OOS_BENCHMARK_NONFINITE:"
                        f"{horizon}:{decision_date}")
                return {
                    "decision_date": decision_date,
                    "ticker": str(row["ticker"]),
                    "isin": str(row["ticker"]),
                    "sector": str(row.get("sector", "Unknown")),
                    "sub_industry":
                        str(row.get("sub_industry", "Unknown")),
                    "feature_snapshot": snapshot,
                    "labels": labels,
                    f"benchmark_forward_return_{horizon}":
                        float(benchmark_value),
                    "feature_snapshot_hash":
                        canonical_sha256(snapshot),
                }

            return {
                "train_rows": [
                    adapt(row) for row in train.to_dict("records")],
                "test_rows": [
                    adapt(row) for row in test.to_dict("records")],
                "sessions": sessions,
                "session_index": session_index,
                "label_col": label_col,
                "benchmark_col": benchmark_col,
            }

        if self.prepared_cache_root is None:
            return remember(compute_prepared())

        artifact_path = (
            self.prepared_cache_root / f"H{int(horizon):02d}" /
            fold.fold_id / f"{cache_key}.pkl")
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        # Hold the per-H×Fold lock across first-time preparation, not only
        # publication. This deliberately makes all five primary Candidate
        # consumers wait for the single producer instead of duplicating the
        # expensive panel filtering / merge / row adaptation.
        with compute_key_lock(
            self.prepared_cache_root, "candidate-fold-prep", identity
        ):
            if artifact_path.is_file():
                payload = pickle.loads(artifact_path.read_bytes())
                if payload.get("cache_key") != cache_key:
                    raise ValueError(
                        "CANDIDATE_FOLD_PREP_CACHE_KEY_MISMATCH")
                self.release_frame_cache()
                return remember(payload["prepared"])

            prepared = compute_prepared()
            payload = {
                "schema_version": "DQBD_CANDIDATE_FOLD_PREP_V2",
                "cache_key": cache_key,
                "identity": identity,
                "prepared": prepared,
            }
            temporary = artifact_path.with_name(
                artifact_path.name + f".{os.getpid()}.tmp")
            temporary.write_bytes(
                pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL))
            os.replace(temporary, artifact_path)
            # The prepared artifact now owns the candidate-independent rows.
            # Keeping the full panel resident in this spawned worker multiplies
            # RAM by worker count and is unnecessary for subsequent fits.
            self.release_frame_cache()
            return remember(prepared)

    def prepare_fold(self, *, horizon: int, fold: Fold,
                     development_end: date, holdout_boundary: date) -> dict:
        """Materialize one candidate-independent H×Fold cache artifact only.

        This is an execution-only prefetch primitive. It does not fit a
        Candidate, publish OOS evidence, or change causal visibility.
        """
        prepared = self._prepare_fold_inputs(
            horizon=horizon, fold=fold, development_end=development_end,
            holdout_boundary=holdout_boundary)
        return {
            "schema_version": "DQBD_CANDIDATE_FOLD_PREP_PREFETCH_V1",
            "horizon": int(horizon),
            "fold_id": str(fold.fold_id),
            "train_rows": len(prepared["train_rows"]),
            "test_rows": len(prepared["test_rows"]),
            "prepared": True,
        }

    def build_fold(self, *, horizon: int, candidate_id: str, fold: Fold,
                   development_end: date, holdout_boundary: date) -> dict:
        candidate = self.candidates[candidate_id]
        prepared = self._prepare_fold_inputs(
            horizon=horizon, fold=fold, development_end=development_end,
            holdout_boundary=holdout_boundary)
        train_rows = prepared["train_rows"]
        test_rows = prepared["test_rows"]
        sessions = prepared["sessions"]
        session_index = prepared["session_index"]
        benchmark_col = prepared["benchmark_col"]

        bundle = v5_train.build_bundle(
            candidate.recipe_family, candidate.hyperparameters,
            self.random_seed).fit(train_rows, horizon)
        expected, _, _ = bundle.predict(test_rows)
        model_bytes = pickle.dumps(bundle, protocol=4)
        model_sha = hashlib.sha256(model_bytes).hexdigest()
        records = []
        for raw, prediction in zip(test_rows, expected):
            decision = str(raw["decision_date"])
            index = session_index[decision] + horizon
            if index >= len(sessions):
                continue
            terminal = date.fromisoformat(sessions[index])
            if terminal > development_end or terminal >= holdout_boundary:
                continue
            realized = float(
                raw["labels"][str(horizon)]["y_excess_net"])
            benchmark = float(raw[benchmark_col])
            if not math.isfinite(realized) or not math.isfinite(benchmark):
                raise ValueError(
                    f"CANDIDATE_OOS_REALIZED_OR_BENCHMARK_NONFINITE:"
                    f"{decision}")
            net_stock_return = realized + benchmark
            records.append({
                "horizon": horizon, "fold_id": fold.fold_id,
                "candidate_id": candidate.candidate_id,
                "recipe_family": candidate.recipe_family,
                "hyperparameters": json.dumps(
                    candidate.hyperparameters, sort_keys=True),
                "symbol": raw["ticker"], "decision_date": decision,
                "prediction": float(prediction),
                "terminal_date": terminal,
                "information_available_at":
                    resolve_information_available_at(
                        decision_date=decision, terminal_date=terminal),
                "realized_return": net_stock_return,
                "benchmark_return": benchmark,
                "realized_excess": realized,
                "status": "MATURED",
                "candidate_spec_hash": candidate.candidate_spec_hash,
                "fold_spec_hash": _fold_hash(fold, horizon),
                "signal_panel_sha256": self.signal_panel_sha256,
                "feature_schema_sha256": self.feature_schema_sha256,
                "model_artifact_sha256": model_sha,
                "fold_policy_hash": self.fold_policy.fold_policy_hash,
                "target_contract_hash":
                    self.target_contract.target_contract_hash,
            })
        execution_backend = v5_train.execution_backend_for_family(
            candidate.recipe_family)
        execution_fingerprint = os.environ.get(
            "DQBD_EXECUTION_BACKEND_FINGERPRINT", "").strip()
        if execution_fingerprint:
            execution_backend = (
                f"{execution_backend}:{execution_fingerprint}")
        return self.store.write(
            pd.DataFrame(records, columns=OBSERVATION_COLUMNS),
            horizon=horizon, fold=fold, candidate=candidate,
            signal_panel_sha256=self.signal_panel_sha256,
            feature_schema_sha256=self.feature_schema_sha256,
            model_artifact_sha256=model_sha, model_bytes=model_bytes,
            fold_policy_hash=self.fold_policy.fold_policy_hash,
            target_contract_hash=self.target_contract.target_contract_hash,
            model_training_contract_hash=(
                self.model_training_contract_hash or canonical_sha256({
                    "module": "stock_predictor.v5.train",
                    "version": "V5_TRAINING_PIPELINE_IMPLEMENTED",
                    "random_seed": self.random_seed,
                })),
            execution_backend=execution_backend)


def fit_production_generation(*, signal_panel: str | Path, feature_schema: Mapping[str, Any],
                              selected_recipe: Mapping[str, Any], horizon: int, information_cutoff: date,
                              training_dates: Iterable[str], calibration_dates: Iterable[str],
                              output_root: str | Path, prediction_dates: Iterable[str] = (),
                              target_contract: TargetContract,
                              random_state: int = 17, model_training_contract_hash: str = "") -> dict:
    """Fit one causal production generation from an already selected recipe.

    Candidate-OOS artifacts are evidence only.  This function deliberately
    refits a fresh V5 bundle on the authorized production window and writes a
    separate model/prediction/calibration manifest.  No rows after the
    information cutoff are read into the fit.
    """
    panel = Path(signal_panel)
    if not panel.is_file():
        raise FileNotFoundError(f"PRODUCTION_SIGNAL_PANEL_MISSING:{panel}")
    if information_cutoff >= date(2026, 7, 25):
        raise PermissionError("PRODUCTION_HOLDOUT_BOUNDARY_OPEN")
    target_contract.validate_target_cost()
    candidate_id = str(selected_recipe.get("selected_candidate_id", ""))
    if not candidate_id:
        raise ValueError("PRODUCTION_SELECTED_RECIPE_MISSING")
    family = str(selected_recipe.get("recipe_family", selected_recipe.get("winner_evidence", {}).get("recipe_family", "")))
    params = selected_recipe.get("hyperparameters") or selected_recipe.get("winner_evidence", {}).get("hyperparameters", {})
    if isinstance(params, str):
        params = json.loads(params)
    if not family:
        raise ValueError("PRODUCTION_SELECTED_RECIPE_FAMILY_MISSING")
    schema = set(pq.read_schema(panel).names)
    allowed = set(feature_schema.get("v2_features", ())) | set(feature_schema.get("v3_features", ())) | set(feature_schema.get("execution_features", ())) | set(feature_schema.get("optional_unvalidated_diagnostics", {}).get("features", ()))
    features = tuple(sorted(x for x in allowed if x in schema))
    target = target_contract.target_column(horizon)
    benchmark_column = target_contract.benchmark_column(horizon)
    required = {"decision_date", "ticker", target, benchmark_column, *features}
    if not required <= schema:
        raise ValueError(f"PRODUCTION_PANEL_COLUMNS_MISSING:{sorted(required - schema)}")
    dates = tuple(sorted(set(str(x)[:10] for x in training_dates)))
    calibration = tuple(sorted(set(str(x)[:10] for x in calibration_dates)))
    prediction_dates = tuple(sorted(set(str(x)[:10] for x in prediction_dates)))
    cutoff = pd.Timestamp(information_cutoff).normalize()
    if any(pd.Timestamp(x) >= pd.Timestamp("2026-07-25") for x in (*dates, *calibration, *prediction_dates)):
        raise PermissionError("PRODUCTION_WINDOW_AFTER_INFORMATION_CUTOFF")
    if set(dates) & set(calibration) or not dates or not calibration:
        raise ValueError("PRODUCTION_WINDOWS_OVERLAP_OR_EMPTY")

    # The H/cutoff leaf is immutable. Validate a complete prior leaf before
    # reading the panel or fitting anything; the old implementation performed
    # this check only after the expensive production fit, so every restarted
    # worker retrained a generation that already existed. A leaf without a
    # manifest is an interrupted write and can be safely reclaimed now.
    root = Path(output_root) / f"H{int(horizon):02d}" / str(information_cutoff)
    model_path = root / "model.joblib"
    calibration_path = root / "calibration-predictions.parquet"
    prediction_path = root / "predictions.parquet"
    manifest_path = root / "generation-manifest.json"
    feature_schema_sha256 = canonical_sha256(feature_schema)
    signal_panel_sha256 = parquet_development_slice_sha256(
        panel, date_column="decision_date", development_end=information_cutoff,
        holdout_boundary=date(2026, 7, 25))
    execution_backend = (
        v5_train.execution_backend_for_family(family)
        + (":" + os.environ["DQBD_EXECUTION_BACKEND_FINGERPRINT"]
           if os.environ.get("DQBD_EXECUTION_BACKEND_FINGERPRINT") else ""))
    if manifest_path.is_file():
        prior = json.loads(manifest_path.read_text(encoding="utf-8"))
        if canonical_sha256({k: v for k, v in prior.items() if k != "manifest_sha256"}) != prior.get("manifest_sha256"):
            raise ValueError("PRODUCTION_GENERATION_CACHE_MANIFEST_HASH_MISMATCH")
        expected_identity = {
            "horizon": int(horizon), "information_cutoff": str(information_cutoff),
            "candidate_id": candidate_id,
            "selected_recipe_sha256": str(selected_recipe.get("selected_recipe_sha256", "")),
            "recipe_family": family, "hyperparameters": dict(params),
            "training_dates": list(dates), "calibration_dates": list(calibration),
            "target_contract_hash": target_contract.target_contract_hash,
            "model_training_contract_hash": str(model_training_contract_hash),
            "random_seed": int(random_state),
            "feature_schema_sha256": feature_schema_sha256,
            "signal_panel_sha256": signal_panel_sha256,
            "signal_panel_development_sha256": signal_panel_sha256,
            "execution_backend": execution_backend,
        }
        mismatches = [key for key, value in expected_identity.items()
                      if key not in prior or prior.get(key) != value]
        if "prediction_dates_hash" in prior:
            if prior["prediction_dates_hash"] != canonical_sha256(prediction_dates):
                mismatches.append("prediction_dates_hash")
        else:
            # Older manifests did not persist the prediction-date identity.
            # Validate their materialized decision-date set before reusing
            # them; never infer compatibility from a missing field.
            if prediction_path.is_file():
                existing_dates = tuple(sorted(
                    pd.to_datetime(pd.read_parquet(prediction_path,
                                                   columns=["decision_date"])["decision_date"])
                    .dt.normalize().dt.date.astype(str).unique()))
            else:
                existing_dates = ()
            if existing_dates != prediction_dates:
                mismatches.append("prediction_dates_hash")
        if mismatches:
            raise ValueError("PRODUCTION_GENERATION_IMMUTABLE_MANIFEST_CONFLICT:" + ",".join(mismatches))
        required_paths = [(model_path, "model_artifact_sha256"),
                          (calibration_path, "calibration_sha256")]
        if prior.get("prediction_sha256"):
            required_paths.append((prediction_path, "prediction_sha256"))
        if any(not path.is_file() or file_sha256(path) != prior.get(key)
               for path, key in required_paths):
            raise ValueError("PRODUCTION_GENERATION_CACHE_FILE_HASH_MISMATCH")
        return prior | {"model_path": str(model_path),
                         "calibration_path": str(calibration_path),
                         "prediction_path": str(prediction_path) if prediction_path.is_file() else None,
                         "manifest_path": str(manifest_path), "cache": "GENERATION_MANIFEST_HIT"}
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    # Session discovery is a projected, development-only read.  The fit read
    # contains labels; the prediction read below never requests them.
    session_frame = read_development_panel(path=panel, columns=["decision_date"],
                                           development_end=information_cutoff,
                                           holdout_boundary=date(2026, 7, 25))
    all_sessions = tuple(sorted(pd.to_datetime(session_frame["decision_date"]).dt.normalize().unique()))
    session_index = {x: i for i, x in enumerate(all_sessions)}
    eligible_dates = []
    requested_fit_dates = tuple(sorted(set(dates) | set(calibration)))
    for value in requested_fit_dates:
        decision = pd.Timestamp(value).normalize()
        position = session_index.get(decision)
        if position is not None and position + int(horizon) < len(all_sessions) and all_sessions[position + int(horizon)] <= cutoff:
            eligible_dates.append(value)
    frame = read_development_panel(path=panel,
                                   columns=["decision_date", "ticker", *features, target, benchmark_column],
                                   development_end=information_cutoff,
                                   holdout_boundary=date(2026, 7, 25), decision_dates=eligible_dates)
    # A forward label is usable for training/calibration only after its exact
    # terminal session is available.  This filter is based on the sessions
    # observed by the projected reader, never on future/holdout rows.
    frame["decision_date"] = pd.to_datetime(frame["decision_date"]).dt.normalize()
    frame["_terminal_date"] = frame["decision_date"].map(
        lambda x: all_sessions[session_index[x] + int(horizon)] if session_index[x] + int(horizon) < len(all_sessions) else pd.NaT
    )
    frame = frame.loc[frame["_terminal_date"].notna() & (frame["_terminal_date"] <= cutoff)].copy()
    train = frame.loc[frame["decision_date"].isin(pd.to_datetime(dates))].copy()
    cal = frame.loc[frame["decision_date"].isin(pd.to_datetime(calibration))].copy()
    if not cal.empty and pd.to_datetime(cal["_terminal_date"]).gt(cutoff).any():
        raise PermissionError("PRODUCTION_CALIBRATION_BEFORE_LABEL_MATURITY")
    dates = tuple(sorted(pd.to_datetime(train["decision_date"]).dt.date.astype(str).unique()))
    calibration = tuple(sorted(pd.to_datetime(cal["decision_date"]).dt.date.astype(str).unique()))
    if train.empty or cal.empty:
        raise ValueError("PRODUCTION_WINDOWS_NOT_PRESENT")

    def adapt_training_row(row: Mapping[str, Any]) -> dict[str, Any]:
        snapshot = {x: row[x] for x in features if pd.notna(row[x])}
        value = float(row[target])
        return {"decision_date": pd.Timestamp(row["decision_date"]).date().isoformat(),
                "ticker": str(row["ticker"]), "isin": str(row["ticker"]),
                "feature_snapshot": snapshot,
                "labels": {str(horizon): {"y_excess_net": value, "y_positive_edge": int(value > .002),
                                           "y_downside": int(value < -.03)}},
                "sector": "Unknown", "sub_industry": "Unknown"}
    def adapt_prediction_row(row: Mapping[str, Any]) -> dict[str, Any]:
        snapshot = {x: row[x] for x in features if pd.notna(row[x])}
        return {"decision_date": pd.Timestamp(row["decision_date"]).date().isoformat(),
                "ticker": str(row["ticker"]), "isin": str(row["ticker"]),
                "feature_snapshot": snapshot, "sector": "Unknown", "sub_industry": "Unknown"}
    train_rows = [adapt_training_row(x) for x in train.to_dict("records")]
    cal_rows = [adapt_training_row(x) for x in cal.to_dict("records")]
    bundle = v5_train.build_bundle(family, dict(params), random_state).fit(train_rows, int(horizon))
    scores, _, _ = bundle.predict(cal_rows)
    model_bytes = __import__("pickle").dumps(bundle, protocol=4)
    model_sha = hashlib.sha256(model_bytes).hexdigest()
    generation_id = canonical_sha256({"candidate_id": candidate_id, "recipe": {"family": family, "parameters": params},
                                      "horizon": int(horizon), "information_cutoff": str(information_cutoff),
                                      "train_dates": dates, "calibration_dates": calibration,
                                      "model_sha256": model_sha,
                                      "model_training_contract_hash": str(model_training_contract_hash)})[:24]
    if model_path.exists() and file_sha256(model_path) != model_sha:
        raise ValueError("PRODUCTION_GENERATION_IMMUTABLE_MODEL_CONFLICT")
    if not model_path.exists():
        model_path.write_bytes(model_bytes)
    predictions = pd.DataFrame({"decision_date": [x["decision_date"] for x in cal_rows],
                                "ticker": [x["ticker"] for x in cal_rows], "horizon": int(horizon),
                                "generation_id": generation_id, "model_artifact_id": generation_id,
                                "candidate_id": candidate_id, "score": [float(x) for x in scores],
                                "realized_excess": [float(x["labels"][str(horizon)]["y_excess_net"]) for x in cal_rows],
                                "information_available_at": [pd.Timestamp(x).date().isoformat() for x in cal["_terminal_date"]]})
    prediction_sha = hashlib.sha256(pd.util.hash_pandas_object(predictions, index=False).to_numpy(dtype="uint64").tobytes()).hexdigest()
    if calibration_path.exists():
        existing = pd.read_parquet(calibration_path)
        existing_sha = hashlib.sha256(pd.util.hash_pandas_object(existing, index=False).to_numpy(dtype="uint64").tobytes()).hexdigest()
        if existing_sha != prediction_sha:
            raise ValueError("PRODUCTION_GENERATION_IMMUTABLE_CALIBRATION_CONFLICT")
    else:
        predictions.to_parquet(calibration_path, index=False)
    prediction_rows = pd.DataFrame()
    if prediction_dates:
        prediction_source = read_development_panel(
            path=panel, columns=["decision_date", "ticker", *features],
            development_end=max(pd.Timestamp(x).date() for x in prediction_dates),
            holdout_boundary=date(2026, 7, 25))
        prediction_source["decision_date"] = pd.to_datetime(prediction_source["decision_date"]).dt.normalize()
        predict_frame = prediction_source.loc[prediction_source["decision_date"].isin(pd.to_datetime(prediction_dates))].copy()
        predict_rows = [adapt_prediction_row(x) for x in predict_frame.to_dict("records")]
        pred_scores, _, _ = bundle.predict(predict_rows)
        prediction_rows = pd.DataFrame({"decision_date": [x["decision_date"] for x in predict_rows],
                                        "ticker": [x["ticker"] for x in predict_rows], "horizon": int(horizon),
                                        "generation_id": generation_id, "model_artifact_id": generation_id,
                                        "candidate_id": candidate_id, "score": [float(x) for x in pred_scores],
                                        "information_available_at": [None] * len(predict_rows)})
    if prediction_path.exists():
        existing = pd.read_parquet(prediction_path)
        if not prediction_rows.empty and not existing.equals(prediction_rows):
            raise ValueError("PRODUCTION_GENERATION_IMMUTABLE_PREDICTION_CONFLICT")
    elif not prediction_rows.empty:
        prediction_rows.to_parquet(prediction_path, index=False)
    calibration_sha = file_sha256(calibration_path)
    prediction_sha256 = file_sha256(prediction_path) if prediction_path.is_file() else None
    score_quantile = float(selected_recipe.get("score_quantile", .975))
    resolved_threshold = float(predictions["score"].quantile(score_quantile))
    manifest = {"schema_version": "DYNAMIC_QBD_PRODUCTION_GENERATION_MANIFEST_V1", "generation_id": generation_id,
                "horizon": int(horizon), "information_cutoff": str(information_cutoff), "candidate_id": candidate_id,
                "selected_recipe_sha256": str(selected_recipe.get("selected_recipe_sha256", "")),
                "recipe_family": family, "hyperparameters": dict(params), "training_dates": list(dates),
                "calibration_dates": list(calibration), "model_artifact_sha256": model_sha,
                "score_quantile": score_quantile, "resolved_threshold": resolved_threshold,
                "calibration_sha256": calibration_sha, "prediction_sha256": prediction_sha256,
                "raw_predictions_sha256": prediction_sha256 or calibration_sha,
                "training_start": dates[0], "training_end": dates[-1],
                "calibration_start": calibration[0], "calibration_end": calibration[-1],
                "calibration_observation_sessions": len(calibration),
                "maturity_cutoff": str(information_cutoff), "horizon": int(horizon),
                 "feature_schema_sha256": feature_schema_sha256,
                 "signal_panel_sha256": signal_panel_sha256,
                 "signal_panel_development_sha256": signal_panel_sha256,
                 "prediction_dates_hash": canonical_sha256(prediction_dates),
                "target_contract_hash": target_contract.target_contract_hash,
                "random_seed": int(random_state),
                 "execution_backend": execution_backend,
                "model_training_contract_hash": str(model_training_contract_hash),
                "usage": "FRESH_PRODUCTION_FIT_FOR_GENERATION_NOT_CANDIDATE_OOS"}
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    if manifest_path.exists():
        prior = json.loads(manifest_path.read_text(encoding="utf-8"))
        if prior.get("manifest_sha256") != manifest["manifest_sha256"]:
            raise ValueError("PRODUCTION_GENERATION_IMMUTABLE_MANIFEST_CONFLICT")
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest | {"model_path": str(model_path), "calibration_path": str(calibration_path),
                       "prediction_path": str(prediction_path) if prediction_path.is_file() else None,
                       "manifest_path": str(manifest_path)}
