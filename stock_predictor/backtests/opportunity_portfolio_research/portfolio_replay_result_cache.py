from __future__ import annotations

from pathlib import Path
import pickle
import sqlite3
from threading import RLock
import time
import zlib


FINAL_REPLAY_CACHE_SCHEMA_VERSION = "OPPORTUNITY_FINAL_REPLAY_CACHE_V1"


class ReplayResultStore:
    """Local-only persistent cache for complete replay result objects.

    Unlike FragmentStore, this cache must preserve pandas DataFrames in `curve`, so it
    uses pickle protocol 5 plus fast zlib compression. It only reads entries created
    under the exact externally supplied research namespace and schema version.
    """

    def __init__(self, path: Path, namespace: str) -> None:
        self.path = Path(path)
        self.namespace = f"{namespace}:{FINAL_REPLAY_CACHE_SCHEMA_VERSION}"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()
        self._conn = sqlite3.connect(str(self.path), timeout=60.0, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA temp_store=MEMORY")
        self._conn.execute("PRAGMA busy_timeout=60000")
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS replay_results (
                namespace TEXT NOT NULL,
                cache_key TEXT NOT NULL,
                payload BLOB NOT NULL,
                created_at REAL NOT NULL,
                PRIMARY KEY(namespace, cache_key)
            )
            """
        )
        self._conn.commit()
        self._hits = 0
        self._misses = 0
        self._writes = 0
        self._entries_at_start = int(self._conn.execute(
            "SELECT COUNT(*) FROM replay_results WHERE namespace = ?", (self.namespace,)
        ).fetchone()[0])

    @staticmethod
    def _encode(value) -> bytes:
        return zlib.compress(pickle.dumps(value, protocol=5), level=1)

    @staticmethod
    def _decode(blob: bytes):
        return pickle.loads(zlib.decompress(blob))

    def get(self, cache_key: str):
        with self._lock:
            row = self._conn.execute(
                "SELECT payload FROM replay_results WHERE namespace = ? AND cache_key = ?",
                (self.namespace, str(cache_key)),
            ).fetchone()
            if row is None:
                self._misses += 1
                return False, None
            self._hits += 1
            return True, self._decode(row[0])

    def put(self, cache_key: str, value) -> None:
        payload = self._encode(value)
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO replay_results(namespace, cache_key, payload, created_at)
                VALUES(?, ?, ?, ?)
                ON CONFLICT(namespace, cache_key)
                DO UPDATE SET payload=excluded.payload, created_at=excluded.created_at
                """,
                (self.namespace, str(cache_key), payload, time.time()),
            )
            # Final replays are comparatively few and expensive. Commit each one so
            # Ctrl+C never loses a completed final/tax/diagnostic replay.
            self._conn.commit()
            self._writes += 1

    def stats(self) -> dict:
        with self._lock:
            entries = int(self._conn.execute(
                "SELECT COUNT(*) FROM replay_results WHERE namespace = ?", (self.namespace,)
            ).fetchone()[0])
            return {
                "enabled": True,
                "path": str(self.path),
                "namespace": self.namespace,
                "entries_loaded_at_start": self._entries_at_start,
                "entries": entries,
                "hits": self._hits,
                "misses": self._misses,
                "writes": self._writes,
            }

    def close(self) -> None:
        with self._lock:
            self._conn.commit()
            self._conn.close()
