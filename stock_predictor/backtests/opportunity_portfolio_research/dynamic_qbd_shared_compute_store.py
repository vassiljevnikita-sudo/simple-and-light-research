"""Cross-seed immutable compute artifacts with causal seed-local views.

Physical computation is shared only when the full execution identity matches.
The store never grants causal visibility: a seed's manifested DAG must still
claim and complete its own node before any shared artifact can be consumed.
"""
from __future__ import annotations

from contextlib import contextmanager
import json
import errno
import os
from pathlib import Path
import shutil
import time
from typing import Iterator

from .contract_fingerprints import stable_hash


SCHEMA_VERSION = "DQBD_SHARED_COMPUTE_STORE_V1"


def compute_identity_key(namespace: str, identity: object) -> str:
    return stable_hash({"namespace": str(namespace), "identity": identity})


@contextmanager
def compute_key_lock(root: str | Path, namespace: str, identity: object) -> Iterator[str]:
    """Hold one crash-safe OS file lock for a deterministic compute identity."""
    key = compute_identity_key(namespace, identity)
    path = Path(root) / "locks" / str(namespace) / f"{key}.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    # Create the lock-file payload before opening a shared handle.  The old
    # ``a+b`` + write/flush sequence had a Windows race: two processes could
    # both observe a zero-length file and one could flush while the other had
    # already acquired the byte range.  Windows then raised PermissionError
    # from ``flush()`` instead of from the retryable lock acquisition below.
    try:
        descriptor = os.open(
            str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            0o666)
    except FileExistsError:
        pass
    else:
        try:
            os.write(descriptor, b"0")
        finally:
            os.close(descriptor)
    handle = path.open("r+b")
    if os.name == "nt":
        import msvcrt
        # ``LK_LOCK`` retries only a bounded number of times on Windows and
        # then raises ``ERROR_POSSIBLE_DEADLOCK``/errno 36.  That is a normal
        # collision here: 26 spawned lanes intentionally share two physical
        # GPUs.  Retry the one-byte lock until the owning worker releases it;
        # never convert ordinary GPU contention into a failed research job.
        try:
            handle.seek(0)
            while True:
                try:
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError as error:
                    if error.errno not in (errno.EACCES, errno.EDEADLK, errno.EAGAIN,
                                           errno.EBUSY, 36):
                        raise
                    time.sleep(.25)
            # A pre-existing zero-length file can only come from an older
            # interrupted implementation.  Repair it while owning the lock.
            if path.stat().st_size == 0:
                handle.write(b"0")
                handle.flush()
            yield key
        finally:
            try:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            finally:
                handle.close()
    else:
        import fcntl
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            if path.stat().st_size == 0:
                handle.write(b"0")
                handle.flush()
            yield key
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            handle.close()


def _link_or_copy(source: Path, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
        return "HARDLINK"
    except OSError:
        shutil.copy2(source, destination)
        return "COPY"


def materialize_immutable_tree(source: str | Path, destination: str | Path) -> dict:
    """Atomically publish a directory using hardlinks when the volume permits."""
    source = Path(source).resolve()
    destination = Path(destination)
    if not source.is_dir():
        raise FileNotFoundError(f"DQBD_SHARED_SOURCE_TREE_MISSING:{source}")
    if destination.is_dir():
        return {"status": "EXISTS", "path": str(destination.resolve())}
    if destination.exists():
        raise RuntimeError(f"DQBD_SHARED_DESTINATION_NOT_DIRECTORY:{destination}")
    temporary = destination.parent / f".{destination.name}.shared.{os.getpid()}.{time.time_ns()}.tmp"
    temporary.mkdir(parents=True, exist_ok=False)
    modes: set[str] = set()
    try:
        for item in source.rglob("*"):
            relative = item.relative_to(source)
            if item.is_dir():
                (temporary / relative).mkdir(parents=True, exist_ok=True)
            elif item.is_file():
                modes.add(_link_or_copy(item, temporary / relative))
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            temporary.replace(destination)
        except (FileExistsError, OSError):
            if not destination.is_dir():
                raise
            shutil.rmtree(temporary, ignore_errors=True)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return {"status": "MATERIALIZED", "path": str(destination.resolve()),
            "mode": "+".join(sorted(modes)) if modes else "EMPTY"}


def shared_store_contract(root: str | Path) -> dict:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "physical_compute_shared": True,
        "causal_visibility_shared": False,
        "publication": "ATOMIC_HARDLINK_OR_COPY",
        "locking": "OS_FILE_LOCK_PER_COMPUTE_KEY",
    }
    payload["contract_hash"] = stable_hash(payload)
    path = Path(root) / "shared-compute-contract.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        prior = json.loads(path.read_text(encoding="utf-8"))
        if prior != payload:
            raise RuntimeError("DQBD_SHARED_COMPUTE_CONTRACT_MISMATCH")
    else:
        temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    return payload
