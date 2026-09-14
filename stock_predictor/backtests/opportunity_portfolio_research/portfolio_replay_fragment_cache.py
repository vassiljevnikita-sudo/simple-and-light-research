from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sqlite3
from threading import RLock
import time
import zlib

import numpy as np
import pandas as pd


FRAGMENT_CACHE_SCHEMA_VERSION = "OPPORTUNITY_PORTFOLIO_FRAGMENT_CACHE_V1"
_CRITICAL_CODE_FILES = (
    "contracts.py",
    "portfolio.py",
    "search.py",
    "tax_de.py",
    "fragment_cache.py",
)


def _sha256_json(value: dict) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def critical_code_fingerprint() -> str:
    root = Path(__file__).resolve().parent
    h = hashlib.sha256()
    for name in _CRITICAL_CODE_FILES:
        path = root / name
        h.update(name.encode("utf-8"))
        h.update(b"\0")
        h.update(path.read_bytes())
        h.update(b"\0")
    return h.hexdigest()


def fingerprint_data_sources(predictions_path: Path, daily_store_root: Path) -> dict:
    """Fast, deterministic source fingerprint for safe fragment reuse.

    The cache is an acceleration layer only. We intentionally use file metadata rather
    than reading every parquet byte; the critical code hash, relative path, size and
    nanosecond mtime must all match before a fragment namespace can be reused.
    """
    predictions_path = Path(predictions_path).resolve()
    daily_store_root = Path(daily_store_root).resolve()

    def meta(path: Path, relative_name: str) -> tuple[str, int, int]:
        st = path.stat()
        return relative_name.replace("\\", "/"), int(st.st_size), int(st.st_mtime_ns)

    prediction_meta = meta(predictions_path, predictions_path.name)
    daily_files = sorted(p for p in daily_store_root.rglob("*.parquet") if p.is_file())
    daily_meta = [meta(p, str(p.relative_to(daily_store_root))) for p in daily_files]
    payload = {
        "prediction": prediction_meta,
        "daily_file_count": len(daily_meta),
        "daily_files": daily_meta,
    }
    return {
        "fingerprint": _sha256_json(payload),
        "prediction_file": str(predictions_path),
        "daily_store_root": str(daily_store_root),
        "daily_file_count": len(daily_meta),
    }


def build_fragment_namespace(input_fingerprint: str) -> str:
    return _sha256_json({
        "schema": FRAGMENT_CACHE_SCHEMA_VERSION,
        "inputs": str(input_fingerprint),
        "critical_code": critical_code_fingerprint(),
        "python_stack": {
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
    })[:40]


def clear_fragment_cache_files(path: Path) -> None:
    path = Path(path)
    for candidate in (path, Path(str(path) + "-wal"), Path(str(path) + "-shm")):
        try:
            candidate.unlink()
        except FileNotFoundError:
            pass


def _encode(value) -> bytes:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return zlib.compress(raw, level=1)


def _decode(blob: bytes):
    return json.loads(zlib.decompress(blob).decode("utf-8"))


class FragmentStore:
    """Thread-safe, restart-safe cache for deterministic research fragments.

    Values are held in memory for hot-path lookup and checkpointed to SQLite in small
    batches. Normal Ctrl+C/close flushes immediately; an abrupt process kill can lose
    only the most recent small batch, never previously committed fragments.
    """

    def __init__(self, path: Path, namespace: str, commit_batch: int = 16) -> None:
        self.path = Path(path)
        self.namespace = str(namespace)
        self.commit_batch = max(1, int(commit_batch))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()
        self._conn = sqlite3.connect(str(self.path), timeout=60.0, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA temp_store=MEMORY")
        self._conn.execute("PRAGMA busy_timeout=60000")
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS fragments (
                namespace TEXT NOT NULL,
                kind TEXT NOT NULL,
                cache_key TEXT NOT NULL,
                payload BLOB NOT NULL,
                created_at REAL NOT NULL,
                PRIMARY KEY(namespace, kind, cache_key)
            )
            """
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_fragments_namespace ON fragments(namespace)"
        )
        self._conn.commit()
        self._memory: dict[tuple[str, str], object] = {}
        self._hits = 0
        self._misses = 0
        self._writes = 0
        self._pending_writes = 0
        self._preloaded = 0
        self._last_commit = time.monotonic()
        self._load_namespace()

    def _load_namespace(self) -> None:
        rows = self._conn.execute(
            "SELECT kind, cache_key, payload FROM fragments WHERE namespace = ?",
            (self.namespace,),
        ).fetchall()
        for kind, key, blob in rows:
            self._memory[(str(kind), str(key))] = _decode(blob)
        self._preloaded = len(rows)

    def get(self, kind: str, cache_key: str) -> tuple[bool, object | None]:
        with self._lock:
            k = (str(kind), str(cache_key))
            if k in self._memory:
                self._hits += 1
                return True, self._memory[k]
            self._misses += 1
            return False, None

    def put(self, kind: str, cache_key: str, value) -> None:
        kind = str(kind)
        cache_key = str(cache_key)
        blob = _encode(value)
        with self._lock:
            self._memory[(kind, cache_key)] = value
            self._conn.execute(
                """
                INSERT INTO fragments(namespace, kind, cache_key, payload, created_at)
                VALUES(?, ?, ?, ?, ?)
                ON CONFLICT(namespace, kind, cache_key)
                DO UPDATE SET payload=excluded.payload, created_at=excluded.created_at
                """,
                (self.namespace, kind, cache_key, blob, time.time()),
            )
            self._writes += 1
            self._pending_writes += 1
            if self._pending_writes >= self.commit_batch or time.monotonic() - self._last_commit >= 1.0:
                self._commit_locked()

    def _commit_locked(self) -> None:
        if self._pending_writes:
            self._conn.commit()
            self._pending_writes = 0
            self._last_commit = time.monotonic()

    def flush(self) -> None:
        with self._lock:
            self._commit_locked()

    def stats(self) -> dict:
        with self._lock:
            return {
                "enabled": True,
                "path": str(self.path),
                "namespace": self.namespace,
                "entries_loaded_at_start": int(self._preloaded),
                "entries_in_namespace": int(len(self._memory)),
                "hits": int(self._hits),
                "misses": int(self._misses),
                "writes": int(self._writes),
                "pending_writes": int(self._pending_writes),
            }

    def close(self) -> None:
        with self._lock:
            self._commit_locked()
            self._conn.close()

    def __enter__(self) -> "FragmentStore":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
