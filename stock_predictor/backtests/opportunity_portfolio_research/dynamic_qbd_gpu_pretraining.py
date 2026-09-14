"""Persistent OpenCL pretraining for the canonical Ridge and HGB components.

The public V5 bundle contract is unchanged.  Ridge uses the numerically
equivalent FP64 OpenCL implementation, while HGB uses the sklearn-compatible
LightGBM GPU tree learner compiled with OpenCL.  Both paths keep contexts
process-local and serialize access to each physical device across manifested
workers.  The canonical sklearn HistGradientBoosting backend remains
available as a separately identified CPU backend, but is never mixed into a
GPU-contract run.
"""
from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
import sys
import threading
import time
from typing import Any, Iterator, Mapping

import numpy as np

from .dynamic_qbd_shared_compute_store import compute_key_lock
from .contract_fingerprints import stable_hash


GPU_PRETRAINING_CONTRACT = "DQBD_GPU_RIDGE_HGB_PRETRAINING_V6"
GPU_EXECUTION_BACKEND = "OPENCL_FP64_RIDGE_REGRESSOR_SKLEARN_BUNDLE_V2_SHARED_GRAM"
GPU_HGB_EXECUTION_BACKEND = "OPENCL_LIGHTGBM_GPU_HGB_SKLEARN_BUNDLE_V5_DEVICE_FINGERPRINTED"
CPU_EXECUTION_BACKEND = "SKLEARN_CANONICAL_CPU_HGB_UNCHANGED"
GPU_HGB_PACKAGE_ENV = "DQBD_LIGHTGBM_GPU_PYTHON_PATH"
# This run is intentionally tied to the locally built OpenCL LightGBM bundle.
# Set it at import time so spawned workers and restarts do not depend on the
# shell that launched the coordinator.
GPU_HGB_PACKAGE_PATH = r"D:\lightgbm-opencl-python-v470"
os.environ[GPU_HGB_PACKAGE_ENV] = GPU_HGB_PACKAGE_PATH
# v40.1 keeps the established dual-device routing identity. Display-safety
# eligibility remains controlled separately by DQBD_GPU_HGB_DISPLAY_POLICY.
GPU_HGB_DEVICE_POLICY = "DUAL_GPU_OPENCL_HGB_AMD_TDR_GUARD_V4"
GPU_HGB_KERNEL_POLICY = "AMD_MAX_BIN_15_NVIDIA_MAX_BIN_63_SINGLE_PRECISION_V4_DEVICE_IDENTITY"
GPU_SECTION_QUEUE_TIMEOUT_SECONDS = 120.0
GPU_SECTION_QUEUE_MIN_TIMEOUT_SECONDS = 0.1


class GPUSectionQueueTimeout(TimeoutError):
    """A bounded wait for a physical GPU section expired.

    This is a transient scheduler condition, not proof that the device or
    numerical backend is broken.  Coordinators must requeue the exact job and
    keep the device eligible for later work.
    """

_ENGINES: dict[str, tuple[Any, Any, Any]] = {}
_DEVICE_ROWS: list[dict[str, Any]] | None = None
_GPU_SECTION_QUEUE_SNAPSHOT_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}
_GPU_SECTION_QUEUE_SCHEMA = "DQBD_GPU_SECTION_READY_QUEUE_V40_0_3"

_OPENCL_SOURCE = r"""
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void gram_rhs(__global const double *x, __global const double *y,
                       __global double *gram, __global double *rhs,
                       const int rows, const int cols) {
    int i = get_global_id(0);
    int j = get_global_id(1);
    if (i >= cols || j >= cols) return;
    double gram_value = 0.0;
    for (int row = 0; row < rows; ++row)
        gram_value += x[row * cols + i] * x[row * cols + j];
    gram[i * cols + j] = gram_value;
    if (j == 0) {
        double rhs_value = 0.0;
        for (int row = 0; row < rows; ++row)
            rhs_value += x[row * cols + i] * y[row];
        rhs[i] = rhs_value;
    }
}
"""


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True, default=str) + "\n",
                    encoding="utf-8")


def _gpu_contract_device_identities(rows: Any) -> tuple[tuple[Any, ...], ...]:
    """Normalize device identities while ignoring process-local enumeration."""
    if not isinstance(rows, list):
        return ()
    return tuple(sorted(
        (
            str(row.get("platform", "")).strip(),
            str(row.get("vendor", "")).strip(),
            str(row.get("name", "")).strip(),
            int(row.get("compute_units", 0) or 0),
            round(float(row.get("global_memory_gib", 0.0) or 0.0), 6),
            bool(row.get("double_precision", False)),
        )
        for row in rows if isinstance(row, Mapping)
    ))


def _compatible_gpu_contract_runtime_migration(
    prior: Mapping[str, Any], current: Mapping[str, Any],
) -> bool:
    """Allow only non-numerical contract migrations on an existing checkpoint."""
    immutable_keys = (
        "schema_version", "enabled", "backend", "hgb_backend",
        "hgb_device_policy", "hgb_kernel_policy",
        "assignment", "context_lifetime",
        "native_numerical_threads_per_cpu_worker", "causal_visibility",
    )
    if any(prior.get(key) != current.get(key) for key in immutable_keys):
        return False
    for key in ("hgb_display_policy",):
        if key in prior and key in current and prior[key] != current[key]:
            return False
    prior_runtime = prior.get("hgb_runtime")
    current_runtime = current.get("hgb_runtime")
    if not isinstance(prior_runtime, Mapping) or not isinstance(current_runtime, Mapping):
        return False
    if prior_runtime.get("backend") != current_runtime.get("backend"):
        return False
    return (
        _gpu_contract_device_identities(prior.get("devices"))
        == _gpu_contract_device_identities(current.get("devices"))
    )


def opencl_fp64_devices() -> list[dict[str, Any]]:
    """Return a deterministic list of usable double-precision GPU devices.

    OpenCL implementations do not promise a stable platform enumeration
    order between processes.  The contract therefore uses canonical indices
    derived from platform/device identity, while ``_device`` resolves those
    identities back to the process-local OpenCL handles.
    """
    try:
        import pyopencl as cl
    except ImportError:
        return []
    platform_rows: list[tuple[str, str, list[Any]]] = []
    for platform in cl.get_platforms():
        candidates = []
        for device in platform.get_devices(device_type=cl.device_type.GPU):
            if "cl_khr_fp64" not in str(device.extensions).split():
                continue
            candidates.append(device)
        if candidates:
            platform_rows.append((
                platform.name.strip(),
                str(getattr(platform, "vendor", "")).strip(),
                candidates,
            ))
    platform_rows.sort(key=lambda row: (row[1].casefold(), row[0].casefold()))
    devices: list[dict[str, Any]] = []
    for platform_index, (platform_name, _platform_vendor, candidates) in enumerate(platform_rows):
        candidates.sort(key=lambda device: (
            str(device.vendor).strip().casefold(),
            str(device.name).strip().casefold(),
            int(device.global_mem_size),
            int(device.max_compute_units),
        ))
        for device_index, device in enumerate(candidates):
            devices.append({
                "platform_index": int(platform_index),
                "device_index": int(device_index),
                "platform": platform_name,
                "vendor": device.vendor.strip(),
                "name": device.name.strip(),
                "global_memory_gib": float(device.global_mem_size) / (1024 ** 3),
                "compute_units": int(device.max_compute_units),
                "double_precision": True,
            })
    return devices


def _device(row: Mapping[str, Any]):
    import pyopencl as cl
    matching_platforms = [
        platform for platform in cl.get_platforms()
        if platform.name.strip() == str(row["platform"])
    ]
    if not matching_platforms:
        raise RuntimeError("DQBD_GPU_DEVICE_PLATFORM_NOT_FOUND")
    for platform in matching_platforms:
        devices = [
            device for device in platform.get_devices(device_type=cl.device_type.GPU)
            if "cl_khr_fp64" in str(device.extensions).split()
        ]
        devices.sort(key=lambda device: (
            str(device.vendor).strip().casefold(),
            str(device.name).strip().casefold(),
            int(device.global_mem_size),
            int(device.max_compute_units),
        ))
        for device_index, device in enumerate(devices):
            if (
                device.vendor.strip() == str(row["vendor"])
                and device.name.strip() == str(row["name"])
                and device_index == int(row["device_index"])
            ):
                return device
    raise RuntimeError("DQBD_GPU_DEVICE_IDENTITY_NOT_FOUND")


def _engine(row: Mapping[str, Any]):
    import pyopencl as cl
    key = stable_hash({"platform": row["platform_index"], "device": row["device_index"],
                      "name": row["name"]})
    cached = _ENGINES.get(key)
    if cached is not None:
        return key, cached
    device = _device(row)
    context = cl.Context([device])
    queue = cl.CommandQueue(context)
    program = cl.Program(context, _OPENCL_SOURCE).build()
    cached = (context, queue, program)
    _ENGINES[key] = cached
    return key, cached


def configured() -> bool:
    return os.environ.get("DQBD_ENABLE_GPU_PRETRAINING", "0") == "1"


def active_backend() -> str:
    return GPU_EXECUTION_BACKEND if configured() else CPU_EXECUTION_BACKEND


def _gpu_lightgbm_root() -> Path | None:
    raw = os.environ.get(GPU_HGB_PACKAGE_ENV, "").strip()
    if not raw:
        return None
    root = Path(raw).expanduser().resolve()
    if not (root / "lightgbm" / "libpath.py").is_file():
        return None
    return root


def lightgbm_gpu_module() -> Any:
    """Load the explicitly provisioned GPU LightGBM Python package.

    A CPU-only wheel must never silently satisfy the GPU HGB contract.  The
    package root is supplied by the host runtime and is inherited by spawned
    manifested workers through ``PYTHONPATH``.
    """
    root = _gpu_lightgbm_root()
    if root is None:
        raise RuntimeError("DQBD_GPU_HGB_PYTHON_PACKAGE_NOT_CONFIGURED")
    existing = sys.modules.get("lightgbm")
    if existing is not None:
        module_path = Path(str(getattr(existing, "__file__", ""))).resolve()
        try:
            module_path.relative_to(root)
        except ValueError as exc:
            raise RuntimeError("DQBD_GPU_HGB_CPU_LIGHTGBM_ALREADY_IMPORTED") from exc
        return existing
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import importlib
    module = importlib.import_module("lightgbm")
    module_path = Path(str(getattr(module, "__file__", ""))).resolve()
    try:
        module_path.relative_to(root)
    except ValueError as exc:
        raise RuntimeError("DQBD_GPU_HGB_WRONG_LIGHTGBM_PACKAGE") from exc
    return module


def gpu_hgb_available() -> dict[str, Any]:
    """Return the provisioned GPU HGB runtime identity or raise if absent."""
    module = lightgbm_gpu_module()
    if not all(hasattr(module, name) for name in ("LGBMRegressor", "LGBMClassifier")):
        raise RuntimeError("DQBD_GPU_HGB_SKLEARN_API_MISSING")
    root = _gpu_lightgbm_root()
    assert root is not None
    version_path = root / "lightgbm" / "VERSION.txt"
    version = version_path.read_text(encoding="utf-8").strip() if version_path.is_file() else "UNKNOWN"
    return {"backend": GPU_HGB_EXECUTION_BACKEND, "package_root": str(root),
            "version": version, "module": str(Path(module.__file__).resolve())}


def active_hgb_backend() -> str:
    return GPU_HGB_EXECUTION_BACKEND if os.environ.get("DQBD_ENABLE_GPU_HGB", "0") == "1" else CPU_EXECUTION_BACKEND


def configure_gpu_pretraining(root: str | Path, *, require: bool = True) -> dict[str, Any]:
    """Freeze the GPU execution contract before any spawned lane is created."""
    global _DEVICE_ROWS
    _DEVICE_ROWS = opencl_fp64_devices()
    if require and len(_DEVICE_ROWS) < 2:
        raise RuntimeError(f"DQBD_GPU_PRETRAINING_REQUIRES_TWO_FP64_GPUS:{len(_DEVICE_ROWS)}")
    hgb_runtime = gpu_hgb_available() if require else None
    payload: dict[str, Any] = {
        "schema_version": GPU_PRETRAINING_CONTRACT,
        "enabled": True,
        "backend": GPU_EXECUTION_BACKEND,
        "hgb_backend": GPU_HGB_EXECUTION_BACKEND if hgb_runtime else CPU_EXECUTION_BACKEND,
        "hgb_device_policy": GPU_HGB_DEVICE_POLICY,
        "hgb_display_policy": os.environ.get(
            "DQBD_GPU_HGB_DISPLAY_POLICY", "AMD_ONLY").strip().upper(),
        "hgb_kernel_policy": GPU_HGB_KERNEL_POLICY,
        "gpu_section_queue_timeout_seconds": GPU_SECTION_QUEUE_TIMEOUT_SECONDS,
        "hgb_runtime": hgb_runtime,
        "device_count": len(_DEVICE_ROWS),
        "devices": _DEVICE_ROWS,
        "assignment": "worker_slot_modulo_device_count_with_exclusive_device_lock",
        "context_lifetime": "persistent_per_spawned_worker_process",
        "native_numerical_threads_per_cpu_worker": 1,
        "causal_visibility": "unchanged_seed_local_manifested_dag",
    }
    payload["contract_hash"] = stable_hash(payload)
    path = Path(root) / "gpu-pretraining-contract.json"
    if path.is_file():
        prior = json.loads(path.read_text(encoding="utf-8"))
        if prior != payload:
            if not _compatible_gpu_contract_runtime_migration(prior, payload):
                raise RuntimeError("DQBD_GPU_PRETRAINING_CONTRACT_MISMATCH")
            _write_json(path.with_name("gpu-pretraining-contract-migration.json"), {
                "schema_version": "DQBD_GPU_PRETRAINING_CONTRACT_MIGRATION_V1",
                "migrated_at": time.time(),
                "reason": "NON_NUMERICAL_RUNTIME_PATH_OR_DEVICE_ENUMERATION",
                "prior_contract_hash": prior.get("contract_hash"),
                "current_contract_hash": payload["contract_hash"],
                "prior_hgb_runtime": prior.get("hgb_runtime"),
                "current_hgb_runtime": payload.get("hgb_runtime"),
                "prior_devices": prior.get("devices"),
                "current_devices": payload.get("devices"),
            })
            _write_json(path, payload)
    else:
        _write_json(path, payload)
    os.environ["DQBD_ENABLE_GPU_PRETRAINING"] = "1"
    if hgb_runtime:
        os.environ["DQBD_ENABLE_GPU_HGB"] = "1"
        package_root = str(hgb_runtime["package_root"])
        pythonpath = [part for part in os.environ.get("PYTHONPATH", "").split(os.pathsep) if part]
        if package_root not in pythonpath:
            os.environ["PYTHONPATH"] = os.pathsep.join([package_root, *pythonpath])
    os.environ["DQBD_GPU_PRETRAINING_ROOT"] = str(Path(root).resolve())
    os.environ["DQBD_GPU_DEVICE_COUNT"] = str(len(_DEVICE_ROWS))
    return payload


def _selected_device(*, workload: str | None = None) -> dict[str, Any]:
    global _DEVICE_ROWS
    if _DEVICE_ROWS is None:
        _DEVICE_ROWS = opencl_fp64_devices()
    if not _DEVICE_ROWS:
        raise RuntimeError("DQBD_GPU_FP64_DEVICE_UNAVAILABLE")
    rows = list(_DEVICE_ROWS)
    # Both GPUs are eligible for HGB.  The AMD-specific kernel limits are
    # applied by ``hgb_kernel_parameters`` below; keeping device assignment
    # modulo the worker slot gives one exclusive HGB fit per physical device
    # while allowing AMD and NVIDIA to work concurrently.
    slot = int(os.environ.get("DQBD_GPU_WORKER_SLOT", "0"))
    return rows[slot % len(rows)]


def selected_gpu_device(*, workload: str | None = None) -> dict[str, Any]:
    """Return the deterministic device assigned to the current worker."""
    return dict(_selected_device(workload=workload))


def hgb_kernel_parameters(row: Mapping[str, Any]) -> dict[str, Any]:
    """Return the bounded OpenCL HGB kernel policy for one physical device.

    AMD's WDDM stack previously hit a watchdog reset and then exposed a
    zero-right-child OpenCL histogram edge case at 31 bins.  The AMD path
    therefore uses LightGBM's documented 15-bin OpenCL setting; NVIDIA
    retains the slightly wider bounded setting.  Both remain GPU LightGBM
    fits, and the policy is part of the versioned run contract.
    """
    identity = f"{row.get('vendor', '')} {row.get('name', '')}".upper()
    return {
        "max_bin": 15 if "AMD" in identity or "ADVANCED MICRO DEVICES" in identity else 63,
        "gpu_use_dp": False,
        "n_jobs": 1,
    }


def gpu_workload_allowed(row: Mapping[str, Any], workload: str) -> bool:
    """Keep long HGB kernels off the host's display adapter.

    The RTX 3070 is also the Windows display adapter on the run host. A long
    LightGBM OpenCL histogram section on that WDDM device can starve desktop
    composition and can trigger the driver TDR path. Ridge sections remain
    eligible there because they are short, while HGB is reserved for the AMD
    compute adapter. ``ALL`` is an explicit opt-out for a headless host.
    """
    if str(workload).upper() != "HGB":
        return True
    policy = os.environ.get(
        "DQBD_GPU_HGB_DISPLAY_POLICY", "AMD_ONLY").strip().upper()
    if policy == "ALL":
        return True
    if policy != "AMD_ONLY":
        raise RuntimeError("DQBD_GPU_HGB_DISPLAY_POLICY_INVALID")
    identity = f"{row.get('vendor', '')} {row.get('name', '')}".upper()
    return "AMD" in identity or "ADVANCED MICRO DEVICES" in identity


def _section_workload() -> str:
    backend = os.environ.get("DQBD_FORCE_EXECUTION_BACKEND", "")
    if GPU_HGB_EXECUTION_BACKEND in backend:
        return "HGB"
    if GPU_EXECUTION_BACKEND in backend:
        return "RIDGE"
    return "GPU"


def _expected_section_seconds(
    row: Mapping[str, Any], workload: str,
) -> float:
    """Cold section-time estimate used only for queue/backlog scheduling."""
    identity = f"{row.get('vendor', '')} {row.get('name', '')}".upper()
    amd = "AMD" in identity or "ADVANCED MICRO DEVICES" in identity
    if workload == "HGB":
        return 4.0 if amd else 3.0
    if workload == "RIDGE":
        return 1.5 if amd else .6
    return 1.0


def expected_gpu_section_seconds(
    row: Mapping[str, Any], workload: str,
) -> float:
    """Public execution-only estimate for parent backlog scheduling."""
    return _expected_section_seconds(row, str(workload))


def _section_queue_dir(root: Path, row: Mapping[str, Any]) -> Path:
    return (
        Path(root) / "gpu-section-ready-queue"
        / f"platform-{int(row['platform_index'])}-device-{int(row['device_index'])}"
    )


def _pid_alive(pid: int) -> bool:
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


def _queue_identity(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "platform_index": int(row["platform_index"]),
        "device_index": int(row["device_index"]),
        "name": str(row.get("name", "")),
    }


def _read_section_tickets(
    root: Path, row: Mapping[str, Any], *, cleanup: bool = True,
) -> list[dict[str, Any]]:
    folder = _section_queue_dir(root, row)
    if not folder.is_dir():
        return []
    now = time.time()
    rows: list[dict[str, Any]] = []
    for path in folder.glob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        pid = int(payload.get("pid", -1))
        stale = (
            now - float(payload.get("created_at", 0.0)) > 900.0
            or pid <= 0
            or not _pid_alive(pid))
        if stale:
            if cleanup:
                try:
                    path.unlink()
                except OSError:
                    pass
            continue
        payload["_path"] = str(path)
        rows.append(payload)
    rows.sort(key=lambda value: (
        int(value.get("priority", 99)),
        float(value.get("created_at", 0.0)),
        str(value.get("ticket_id", "")),
    ))
    return rows


def gpu_section_queue_snapshot(
    root: str | Path,
    devices: list[dict[str, Any]] | None = None,
    *, cache_seconds: float = .05,
) -> dict[str, Any]:
    """Return actual GPU-ready ticket backlog in predicted device seconds."""
    base = str(Path(root).resolve())
    rows = list(devices or opencl_fp64_devices())
    device_identity = [
        {
            "platform_index": int(row["platform_index"]),
            "device_index": int(row["device_index"]),
            "vendor": str(row.get("vendor", "")),
            "name": str(row.get("name", "")),
        }
        for row in rows
    ]
    cache_key = f"{base}:{stable_hash(device_identity)}"
    now = time.monotonic()
    cached = _GPU_SECTION_QUEUE_SNAPSHOT_CACHE.get(cache_key)
    if cached is not None and now - cached[0] < max(0.0, cache_seconds):
        return dict(cached[1])
    device_rows = []
    for logical_index, row in enumerate(rows):
        tickets = _read_section_tickets(Path(root), row)
        by_workload: dict[str, float] = {}
        for ticket in tickets:
            workload = str(ticket.get("workload", "GPU"))
            by_workload[workload] = (
                by_workload.get(workload, 0.0)
                + float(ticket.get("expected_seconds", 0.0)))
        device_rows.append({
            "device_index": logical_index,
            "platform_index": int(row["platform_index"]),
            "opencl_device_index": int(row["device_index"]),
            "name": str(row.get("name", "")),
            "vendor": str(row.get("vendor", "")),
            "ticket_count": len(tickets),
            "backlog_seconds": sum(by_workload.values()),
            "backlog_seconds_by_workload": by_workload,
        })
    payload = {
        "schema_version": _GPU_SECTION_QUEUE_SCHEMA,
        "timestamp": time.time(),
        "devices": device_rows,
    }
    _GPU_SECTION_QUEUE_SNAPSHOT_CACHE[cache_key] = (now, payload)
    return dict(payload)


@contextmanager
def _queued_device_turn(
    root: Path, row: Mapping[str, Any],
) -> Iterator[dict[str, Any]]:
    """Queue one actual GPU section, HGB first then Ridge by arrival time."""
    workload = _section_workload()
    priority = 0 if workload == "HGB" else (1 if workload == "RIDGE" else 2)
    folder = _section_queue_dir(root, row)
    folder.mkdir(parents=True, exist_ok=True)
    ticket_id = (
        f"{time.time_ns()}-{os.getpid()}-{threading.get_ident()}")
    path = folder / f"{priority:02d}-{ticket_id}.json"
    payload = {
        "schema_version": _GPU_SECTION_QUEUE_SCHEMA,
        "ticket_id": ticket_id,
        "_ticket_path": str(path),
        "created_at": time.time(),
        "pid": os.getpid(),
        "thread_id": threading.get_ident(),
        "workload": workload,
        "priority": priority,
        "expected_seconds": _expected_section_seconds(row, workload),
        "job_id": os.environ.get("DQBD_GPU_JOB_ID", "").strip() or None,
        "gpu_worker_slot": os.environ.get(
            "DQBD_GPU_WORKER_SLOT", "").strip() or None,
        **_queue_identity(row),
    }
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)
    identity = _queue_identity(row)
    active_path = folder / "_active-section.token"
    owns_active = False
    timeout_seconds = max(
        GPU_SECTION_QUEUE_MIN_TIMEOUT_SECONDS,
        float(os.environ.get(
            "DQBD_GPU_SECTION_QUEUE_TIMEOUT_SECONDS",
            str(GPU_SECTION_QUEUE_TIMEOUT_SECONDS))),
    )
    deadline = time.monotonic() + timeout_seconds
    try:
        while True:
            with compute_key_lock(root, "gpu-section-queue-admin", identity):
                if active_path.is_file():
                    try:
                        active = json.loads(
                            active_path.read_text(encoding="utf-8"))
                    except (OSError, json.JSONDecodeError):
                        active = {}
                    active_pid = int(active.get("pid", -1))
                    active_stale = (
                        time.time() - float(
                            active.get("claimed_at", 0.0)) > 900.0
                        or active_pid <= 0
                        or not _pid_alive(active_pid))
                    if active_stale:
                        try:
                            active_path.unlink()
                        except OSError:
                            pass
                tickets = _read_section_tickets(root, row)
                first = tickets[0] if tickets else None
                if (
                    not active_path.exists()
                    and first
                    and first.get("ticket_id") == ticket_id
                ):
                    active_payload = {
                        "schema_version": _GPU_SECTION_QUEUE_SCHEMA,
                        "ticket_id": ticket_id,
                        "pid": os.getpid(),
                        "thread_id": threading.get_ident(),
                        "workload": workload,
                        "claimed_at": time.time(),
                        **identity,
                    }
                    active_temp = active_path.with_name(
                        active_path.name + f".{os.getpid()}.tmp")
                    active_temp.write_text(
                        json.dumps(active_payload, sort_keys=True) + "\n",
                        encoding="utf-8")
                    os.replace(active_temp, active_path)
                    owns_active = True
                    try:
                        path.unlink()
                    except OSError:
                        pass
                    break
            if time.monotonic() >= deadline:
                raise GPUSectionQueueTimeout(
                    "DQBD_GPU_SECTION_QUEUE_TIMEOUT:"
                    f"device={row.get('name', '')}:"
                    f"timeout_seconds={timeout_seconds:.1f}")
            time.sleep(.005)
        yield payload
    finally:
        try:
            path.unlink()
        except OSError:
            pass
        if owns_active:
            with compute_key_lock(root, "gpu-section-queue-admin", identity):
                try:
                    active = json.loads(
                        active_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    active = {}
                if active.get("ticket_id") == ticket_id:
                    try:
                        active_path.unlink()
                    except OSError:
                        pass


def _record_device_section(
    root: Path, row: Mapping[str, Any], event: str,
    *, started_at: float | None = None,
) -> None:
    """Persist actual GPU lock occupancy per physical device."""
    now = time.time()
    path = (
        Path(root) / "gpu-device-events"
        / f"platform-{int(row['platform_index'])}-device-{int(row['device_index'])}.jsonl"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "DQBD_GPU_DEVICE_SECTION_EVENTS_V1",
        "timestamp": now,
        "event": str(event),
        "pid": os.getpid(),
        "job_id": os.environ.get("DQBD_GPU_JOB_ID", "").strip() or None,
        "gpu_worker_slot": os.environ.get(
            "DQBD_GPU_WORKER_SLOT", "").strip() or None,
        "platform_index": int(row["platform_index"]),
        "device_index": int(row["device_index"]),
        "vendor": str(row.get("vendor", "")),
        "name": str(row.get("name", "")),
        "execution_backend": os.environ.get(
            "DQBD_FORCE_EXECUTION_BACKEND", ""),
    }
    if started_at is not None:
        payload["duration_seconds"] = max(0.0, now - float(started_at))
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


@contextmanager
def _exclusive_device(root: Path, row: Mapping[str, Any]) -> Iterator[str]:
    """Run one actual GPU section through the v40.0.3 ready-section queue."""
    identity = {
        "platform_index": row["platform_index"],
        "device_index": row["device_index"],
        "name": row["name"],
    }
    with _queued_device_turn(root, row) as ticket:
        # The queue's active-section token guarantees that only the selected
        # highest-priority ready ticket can contend for the physical device
        # lock. Later Ridge waiters therefore cannot get ahead of a ready HGB
        # section merely by blocking earlier in the OS lock queue.
        with compute_key_lock(root, "gpu-device", identity) as key:
            queue_wait = max(
                0.0, time.time() - float(ticket["created_at"]))
            started_at = time.time()
            _record_device_section(root, row, "ACQUIRE")
            queue_path = (
                Path(root) / "gpu-section-queue-events.jsonl")
            with queue_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({
                    "schema_version": _GPU_SECTION_QUEUE_SCHEMA,
                    "timestamp": started_at,
                    "event": "ACQUIRE",
                    "ticket_id": ticket["ticket_id"],
                    "workload": ticket["workload"],
                    "job_id": ticket.get("job_id"),
                    "gpu_worker_slot": ticket.get("gpu_worker_slot"),
                    "queue_wait_seconds": queue_wait,
                    **identity,
                }, sort_keys=True) + "\n")
            try:
                yield key
            finally:
                _record_device_section(
                    root, row, "RELEASE", started_at=started_at)


@contextmanager
def exclusive_gpu_device(root: str | Path, row: Mapping[str, Any] | None = None) -> Iterator[str]:
    """Serialize one complete fit on the selected physical GPU."""
    selected = dict(row or _selected_device())
    with _exclusive_device(Path(root).resolve(), selected) as key:
        yield key


def fit_ridge_regressor(matrix: Any, target: Any, *, alpha: float) -> tuple[np.ndarray, float, dict[str, Any]]:
    """Fit Ridge with shared H×Fold Gram/RHS reuse and a narrow GPU lock."""
    if not configured():
        raise RuntimeError("DQBD_GPU_PRETRAINING_NOT_ENABLED")
    import pyopencl as cl

    x = np.asarray(matrix, dtype=np.float64, order="C")
    y = np.asarray(target, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 1 or len(x) != len(y) or not len(x):
        raise ValueError("DQBD_GPU_RIDGE_MATRIX_SHAPE_INVALID")
    rows, cols = x.shape
    row = _selected_device()
    root = Path(os.environ.get("DQBD_GPU_PRETRAINING_ROOT", ".")).resolve()
    device_key, engine = _engine(row)
    context, queue, program = engine

    shared_id = os.environ.get("DQBD_FIT_SHARED_ID", "").strip()
    cache_hit = False
    gram = rhs = x_mean = None
    y_mean = None
    cache_path: Path | None = None
    identity = None
    if shared_id:
        identity = {
            "schema_version": "DQBD_SHARED_RIDGE_GRAM_V1",
            "fit_shared_id": shared_id,
            "rows": int(rows),
            "features": int(cols),
            "precision": "FP64",
        }
        key = stable_hash(identity)
        cache_path = root / "subfit-cache" / "ridge-gram" / f"{key}.npz"
        cache_path.parent.mkdir(parents=True, exist_ok=True)

    def load_cache(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        with np.load(path, allow_pickle=False) as payload:
            return (
                np.asarray(payload["gram"], dtype=np.float64),
                np.asarray(payload["rhs"], dtype=np.float64),
                np.asarray(payload["x_mean"], dtype=np.float64),
                float(np.asarray(payload["y_mean"]).item()),
            )

    if cache_path is not None and identity is not None:
        with compute_key_lock(root, "ridge-gram", identity):
            if cache_path.is_file():
                gram, rhs, x_mean, y_mean = load_cache(cache_path)
                cache_hit = True
            else:
                x_mean = x.mean(axis=0)
                y_mean = float(y.mean())
                centered_x = np.ascontiguousarray(x - x_mean, dtype=np.float64)
                centered_y = np.ascontiguousarray(y - y_mean, dtype=np.float64)
                gram = np.empty((cols, cols), dtype=np.float64)
                rhs = np.empty(cols, dtype=np.float64)
                flags = cl.mem_flags
                with _exclusive_device(root, row):
                    x_buffer = cl.Buffer(
                        context, flags.READ_ONLY | flags.COPY_HOST_PTR,
                        hostbuf=centered_x)
                    y_buffer = cl.Buffer(
                        context, flags.READ_ONLY | flags.COPY_HOST_PTR,
                        hostbuf=centered_y)
                    gram_buffer = cl.Buffer(
                        context, flags.WRITE_ONLY, gram.nbytes)
                    rhs_buffer = cl.Buffer(
                        context, flags.WRITE_ONLY, rhs.nbytes)
                    kernel = cl.Kernel(program, "gram_rhs")
                    kernel(
                        queue, (cols, cols), None,
                        x_buffer, y_buffer, gram_buffer, rhs_buffer,
                        np.int32(rows), np.int32(cols))
                    cl.enqueue_copy(queue, gram, gram_buffer)
                    cl.enqueue_copy(queue, rhs, rhs_buffer).wait()
                temporary = cache_path.with_name(
                    cache_path.name + f".{os.getpid()}.tmp")
                with temporary.open("wb") as handle:
                    np.savez(
                        handle, gram=gram, rhs=rhs, x_mean=x_mean,
                        y_mean=np.asarray([y_mean], dtype=np.float64))
                os.replace(temporary, cache_path)
    else:
        x_mean = x.mean(axis=0)
        y_mean = float(y.mean())
        centered_x = np.ascontiguousarray(x - x_mean, dtype=np.float64)
        centered_y = np.ascontiguousarray(y - y_mean, dtype=np.float64)
        gram = np.empty((cols, cols), dtype=np.float64)
        rhs = np.empty(cols, dtype=np.float64)
        flags = cl.mem_flags
        with _exclusive_device(root, row):
            x_buffer = cl.Buffer(
                context, flags.READ_ONLY | flags.COPY_HOST_PTR,
                hostbuf=centered_x)
            y_buffer = cl.Buffer(
                context, flags.READ_ONLY | flags.COPY_HOST_PTR,
                hostbuf=centered_y)
            gram_buffer = cl.Buffer(
                context, flags.WRITE_ONLY, gram.nbytes)
            rhs_buffer = cl.Buffer(
                context, flags.WRITE_ONLY, rhs.nbytes)
            kernel = cl.Kernel(program, "gram_rhs")
            kernel(
                queue, (cols, cols), None,
                x_buffer, y_buffer, gram_buffer, rhs_buffer,
                np.int32(rows), np.int32(cols))
            cl.enqueue_copy(queue, gram, gram_buffer)
            cl.enqueue_copy(queue, rhs, rhs_buffer).wait()

    if gram is None or rhs is None or x_mean is None or y_mean is None:
        raise RuntimeError("DQBD_GPU_RIDGE_SHARED_GRAM_NOT_MATERIALIZED")
    gram_for_solve = np.array(gram, dtype=np.float64, copy=True)
    gram_for_solve.flat[:: cols + 1] += float(alpha)
    coefficients = np.linalg.solve(gram_for_solve, rhs)
    intercept = y_mean - float(x_mean @ coefficients)
    diagnostics = {
        "schema_version": GPU_PRETRAINING_CONTRACT,
        "backend": GPU_EXECUTION_BACKEND,
        "device_key": device_key,
        "device": dict(row),
        "rows": int(rows),
        "features": int(cols),
        "alpha": float(alpha),
        "cpu_small_solve": True,
        "exclusive_device_lock": True,
        "device_lock_scope": "GPU_BUFFER_KERNEL_COPY_ONLY",
        "shared_gram_cache": bool(shared_id),
        "shared_gram_cache_hit": bool(cache_hit),
        "shared_fit_id": shared_id or None,
    }
    return np.asarray(coefficients, dtype=np.float64), float(intercept), diagnostics
