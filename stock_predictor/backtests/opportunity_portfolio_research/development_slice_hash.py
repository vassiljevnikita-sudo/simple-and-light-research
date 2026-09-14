"""Causal, representation-independent hashes for development Parquet slices."""
from __future__ import annotations

from datetime import date
import heapq
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Iterable

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds

from .dynamic_qbd_shared_compute_store import compute_key_lock


def _scalar_boundary(field_type, value: date):
    if pa.types.is_string(field_type) or pa.types.is_large_string(field_type):
        return pa.scalar(value.isoformat())
    return pa.scalar(pd.Timestamp(value).to_datetime64())


def _json_value(value):
    if pd.isna(value):
        return None
    if isinstance(value, (pd.Timestamp,)):
        return value.normalize().isoformat()
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float):
        return None if pd.isna(value) else value
    return value


def _compute_parquet_development_slice_sha256(
    path: str | Path,
    *,
    date_column: str,
    development_end: date,
    holdout_boundary: date,
    projected_columns: Iterable[str] | None = None,
    holdout_locked_column: str = "holdout_locked",
    semantics: str = "DATE_LE_DEVELOPMENT_END_LT_HOLDOUT",
) -> str:
    """Hash only canonical rows permitted by the causal development contract."""
    dataset = ds.dataset(str(path), format="parquet")
    names = set(dataset.schema.names)
    if date_column not in names:
        raise ValueError(f"DEVELOPMENT_HASH_DATE_COLUMN_MISSING:{date_column}")
    columns = list(projected_columns) if projected_columns is not None else list(dataset.schema.names)
    columns = [x for x in columns if x in names]
    if date_column not in columns:
        columns.insert(0, date_column)
    if holdout_locked_column in names and holdout_locked_column not in columns:
        columns.append(holdout_locked_column)
    predicate = ((ds.field(date_column) <= _scalar_boundary(dataset.schema.field(date_column).type, development_end)) &
                 (ds.field(date_column) < _scalar_boundary(dataset.schema.field(date_column).type, holdout_boundary)))
    if holdout_locked_column in names:
        predicate = predicate & ((ds.field(holdout_locked_column) == False) | ds.field(holdout_locked_column).is_null())
    columns = [x for x in columns if x != holdout_locked_column]
    # External-sort canonical rows in bounded batches.  The previous table ->
    # pandas -> Python-list path multiplied a 1.5-GB signal panel into many GB
    # of transient memory before any worker could start.
    encoder = json.JSONEncoder(sort_keys=True, default=str, separators=(",", ":"))
    scratch_root = Path(
        os.environ.get("DQBD_SLICE_HASH_SCRATCH_ROOT", tempfile.gettempdir()))
    scratch_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
            prefix=f"p{os.getpid()}-", dir=str(scratch_root)) as temp:
        shard_paths = []
        for batch in dataset.to_batches(columns=columns + ([holdout_locked_column] if holdout_locked_column in names else []),
                                        filter=predicate, batch_size=65536):
            frame = batch.to_pandas()
            if holdout_locked_column in frame:
                frame = frame.drop(columns=[holdout_locked_column])
            frame[date_column] = pd.to_datetime(frame[date_column], errors="coerce").dt.normalize().map(
                lambda x: x.isoformat() if pd.notna(x) else None)
            rows = sorted(encoder.encode([_json_value(value) for value in values])
                          for values in frame.reindex(columns=columns).itertuples(index=False, name=None))
            if rows:
                shard = Path(temp) / f"shard-{len(shard_paths):06d}.jsonl"
                shard.write_text("\n".join(rows) + "\n", encoding="utf-8")
                shard_paths.append(shard)
        base = {"columns": columns, "date_column": date_column,
                "development_end": development_end.isoformat(),
                "holdout_boundary": holdout_boundary.isoformat(), "schema_version": "QBD_DEVELOPMENT_SLICE_HASH_V1",
                "semantics": semantics}
        empty = encoder.encode({**base, "rows": []})
        digest = hashlib.sha256()
        digest.update((empty[:-2] + "[").encode())
        first = True
        handles = [path.open("r", encoding="utf-8") for path in shard_paths]
        try:
            for line in heapq.merge(*(map(str.rstrip, handle) for handle in handles)):
                if not first:
                    digest.update(b",")
                digest.update(line.encode("utf-8"))
                first = False
        finally:
            for handle in handles:
                handle.close()
        digest.update(b"]}")
        return digest.hexdigest()


_SLICE_HASH_CACHE_SCHEMA = "DQBD_DEVELOPMENT_SLICE_HASH_CACHE_V1"
_SLICE_HASH_SCRATCH_PATTERN = re.compile(r"^p(?P<pid>\d+)-")


def _process_alive(pid: int) -> bool:
    if int(pid) == os.getpid():
        return True
    if os.name == "nt":
        import ctypes
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        handle = kernel32.OpenProcess(0x1000, False, int(pid))
        if not handle:
            return False
        kernel32.CloseHandle(handle)
        return True
    try:
        os.kill(int(pid), 0)
    except OSError:
        return False
    return True


def slice_hash_scratch_root() -> Path:
    return Path(
        os.environ.get("DQBD_SLICE_HASH_SCRATCH_ROOT", tempfile.gettempdir()))


def cleanup_slice_hash_scratch_for_pid(pid: int) -> int:
    """Delete external-sort scratch left by one terminated worker."""
    root = slice_hash_scratch_root()
    if not root.is_dir():
        return 0
    removed = 0
    for path in root.glob(f"p{int(pid)}-*"):
        if not path.is_dir():
            continue
        shutil.rmtree(path, ignore_errors=True)
        if not path.exists():
            removed += 1
    return removed


def cleanup_orphaned_slice_hash_scratch() -> int:
    """Delete scratch directories belonging to processes that no longer exist."""
    root = slice_hash_scratch_root()
    if not root.is_dir():
        return 0
    removed = 0
    for path in root.iterdir():
        if not path.is_dir():
            continue
        match = _SLICE_HASH_SCRATCH_PATTERN.match(path.name)
        if match is None:
            continue
        pid = int(match.group("pid"))
        if _process_alive(pid):
            continue
        shutil.rmtree(path, ignore_errors=True)
        if not path.exists():
            removed += 1
    return removed


def cleanup_legacy_slice_hash_scratch() -> int:
    """Remove pre-v40.0.4 scratch directories without PID ownership metadata.

    v40.0.3 used TemporaryDirectory(prefix="dqbd-slice-hash-") in the system
    temp volume. Forced pool termination bypassed its cleanup and the published
    run eventually failed with ENOSPC. v40.0.4 never creates this prefix, so
    directories with it are historical orphan state for this runtime.
    """
    root = Path(tempfile.gettempdir())
    if not root.is_dir():
        return 0
    removed = 0
    for path in root.glob("dqbd-slice-hash-*"):
        if not path.is_dir():
            continue
        shutil.rmtree(path, ignore_errors=True)
        if not path.exists():
            removed += 1
    return removed


def parquet_development_slice_sha256(
    path: str | Path,
    *,
    date_column: str,
    development_end: date,
    holdout_boundary: date,
    projected_columns: Iterable[str] | None = None,
    holdout_locked_column: str = "holdout_locked",
    semantics: str = "DATE_LE_DEVELOPMENT_END_LT_HOLDOUT",
) -> str:
    """Hash one causal Development slice with an optional shared disk cache.

    Production Generations repeatedly request the same signal-panel slice.
    v40.0.3 rebuilt the external sort in many workers and abrupt pool reclaim
    could strand its temporary shards. v40.0.4 binds the cache key to the
    source file identity and causal hash contract, serializes the first
    computation cross-process, and puts PID-addressable scratch on the shared
    compute volume when configured by the runner.
    """
    target = Path(path).resolve()
    stat = target.stat()
    projected = (
        tuple(str(value) for value in projected_columns)
        if projected_columns is not None else None)
    identity = {
        "schema_version": _SLICE_HASH_CACHE_SCHEMA,
        "path": str(target),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "date_column": str(date_column),
        "development_end": development_end.isoformat(),
        "holdout_boundary": holdout_boundary.isoformat(),
        "projected_columns": list(projected) if projected is not None else None,
        "holdout_locked_column": str(holdout_locked_column),
        "semantics": str(semantics),
    }
    cache_root_value = os.environ.get("DQBD_SLICE_HASH_CACHE_ROOT", "").strip()
    if not cache_root_value:
        return _compute_parquet_development_slice_sha256(
            target,
            date_column=date_column,
            development_end=development_end,
            holdout_boundary=holdout_boundary,
            projected_columns=projected,
            holdout_locked_column=holdout_locked_column,
            semantics=semantics,
        )
    cache_root = Path(cache_root_value)
    identity_bytes = json.dumps(
        identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    cache_key = hashlib.sha256(identity_bytes).hexdigest()
    cache_path = cache_root / "values" / f"{cache_key}.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with compute_key_lock(cache_root, "development-slice-hash", identity):
        if cache_path.is_file():
            try:
                payload = json.loads(cache_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                payload = {}
            digest = str(payload.get("sha256", ""))
            if (
                payload.get("schema_version") == _SLICE_HASH_CACHE_SCHEMA
                and payload.get("identity") == identity
                and len(digest) == 64
                and all(ch in "0123456789abcdef" for ch in digest)
            ):
                return digest
        digest = _compute_parquet_development_slice_sha256(
            target,
            date_column=date_column,
            development_end=development_end,
            holdout_boundary=holdout_boundary,
            projected_columns=projected,
            holdout_locked_column=holdout_locked_column,
            semantics=semantics,
        )
        payload = {
            "schema_version": _SLICE_HASH_CACHE_SCHEMA,
            "identity": identity,
            "sha256": digest,
        }
        temporary = cache_path.with_name(
            cache_path.name + f".{os.getpid()}.tmp")
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
        os.replace(temporary, cache_path)
        return digest
