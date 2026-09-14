"""Host-aware CPU and memory budget for the Dynamic-QBD research suite.

The authoritative local Windows runner is constrained to one aggregate Job
Object memory budget so worker children cannot collectively consume more than
the suite limit. Other platforms keep the same declared contract but use
best-effort process limits only.
"""
from __future__ import annotations

import ctypes
from ctypes import wintypes
import os
from pathlib import Path
import sys
from typing import Any

from threadpoolctl import threadpool_limits

from .cpu_topology import physical_core_affinity_plan, set_current_process_affinity


# Kept as a compatibility name for older callers; the active limit is now
# derived from the host's physical memory or an explicitly supplied fraction.
# There is intentionally no suite-wide 90-GiB ceiling anymore.
MAX_MEMORY_LIMIT_GB = None
DEFAULT_MEMORY_LIMIT_GB = float(os.environ.get("DYNAMIC_QBD_MEMORY_LIMIT_GB", "90.0"))
DEFAULT_MEMORY_SOFT_TARGET_GB = float(os.environ.get("DYNAMIC_QBD_MEMORY_SOFT_TARGET_GB", "84.0"))
DEFAULT_MEMORY_LIMIT_FRACTION = .95
DEFAULT_MEMORY_SOFT_TARGET_FRACTION = .90
_BYTES_PER_GIB = 1024 ** 3

_ACTIVE_LIMITER = None
_ACTIVE_CONTRACT: dict[str, Any] | None = None
_WINDOWS_JOB_HANDLE = None


def _posix_cgroup_memory_limit_bytes() -> int | None:
    """Return a finite container memory limit when one is configured."""
    candidates = (
        Path("/sys/fs/cgroup/memory.max"),
        Path("/sys/fs/cgroup/memory/memory.limit_in_bytes"),
    )
    for path in candidates:
        try:
            raw = path.read_text(encoding="utf-8").strip()
            if not raw or raw.lower() == "max":
                continue
            value = int(raw)
            if value > 0:
                return value
        except (OSError, ValueError):
            continue
    return None

def _physical_memory_bytes() -> int:
    if sys.platform == "win32":
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
        status.dwLength = ctypes.sizeof(status)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            raise OSError("GLOBAL_MEMORY_STATUS_EX_FAILED")
        return int(status.ullTotalPhys)
    host_total = int(os.sysconf("SC_PAGE_SIZE")) * int(os.sysconf("SC_PHYS_PAGES"))
    container_limit = _posix_cgroup_memory_limit_bytes()
    if container_limit is not None:
        return min(host_total, container_limit)
    return host_total


def _memory_contract_name(limit_gb: float, *, fraction: float | None = None) -> str:
    if fraction is not None:
        return f"MAX_{fraction * 100:.0f}PCT_SYSTEM_MEMORY_SUITE_WIDE_WINDOWS_JOB_OBJECT"
    value = float(limit_gb)
    if value.is_integer():
        label = str(int(value))
    else:
        label = str(value).replace(".", "P")
    return f"MAX_{label}_GIB_SUITE_WIDE_WINDOWS_JOB_OBJECT"


def cpu_budget(
    target_fraction: float = .90,
    logical_processors: int | None = None,
    *,
    enforce_capacity_fraction: bool = False,
) -> dict[str, Any]:
    """Return the CPU execution budget.

    Historical Dynamic-QBD callers use ``target_fraction`` only as an observed
    utilization target and therefore retain access to every logical processor.
    Large bounded local suites may set ``enforce_capacity_fraction=True`` to
    turn the fraction into an affinity-capacity budget.  On a 32-logical CPU,
    0.80 then selects 26 logical processors and leaves six for the OS and other
    coordinator work.
    """
    logical = max(1, int(logical_processors or os.cpu_count() or 1))
    fraction = float(target_fraction)
    if not 0 < fraction <= 1:
        raise ValueError("DYNAMIC_QBD_CPU_TARGET_FRACTION_OUT_OF_RANGE")
    if enforce_capacity_fraction:
        target = max(1, min(logical, int(round(logical * fraction))))
        reserve = max(0, logical - target)
    else:
        # Utilization is not the same as runnable-thread capacity. Historical
        # suites intentionally expose all CPUs and use this as a measured target.
        target = logical
        reserve = 0
    return {
        "logical_processors": logical,
        "target_fraction": fraction,
        "target_logical_processors": target,
        "reserve_logical_processors": reserve,
        "available_capacity_fraction": target / logical,
        "capacity_fraction_enforced": bool(enforce_capacity_fraction),
    }


def _windows_current_process_is_in_job() -> bool:
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    is_in_job = kernel32.IsProcessInJob
    is_in_job.argtypes = [wintypes.HANDLE, wintypes.HANDLE, ctypes.POINTER(wintypes.BOOL)]
    is_in_job.restype = wintypes.BOOL
    get_current = kernel32.GetCurrentProcess
    get_current.restype = wintypes.HANDLE
    result = wintypes.BOOL()
    if not is_in_job(get_current(), None, ctypes.byref(result)):
        error = ctypes.get_last_error()
        raise OSError(error, "DYNAMIC_QBD_QUERY_MEMORY_JOB_FAILED")
    return bool(result.value)


def _windows_job_memory_limit(limit_bytes: int) -> dict[str, Any]:
    """Apply one aggregate committed-memory ceiling to this process tree."""
    global _WINDOWS_JOB_HANDLE

    if _WINDOWS_JOB_HANDLE is not None:
        return {
            "memory_enforcement": "WINDOWS_JOB_OBJECT_JOB_MEMORY",
            "memory_limit_applied": True,
            "memory_limit_reused": True,
            "memory_limit_inherited": False,
        }

    # Windows spawned workers import this module from scratch. The marker lets
    # them retain parent Job membership instead of nesting another Job Object.
    if os.environ.get("DYNAMIC_QBD_MEMORY_JOB_INHERITED") == "1" and _windows_current_process_is_in_job():
        return {
            "memory_enforcement": "WINDOWS_JOB_OBJECT_INHERITED_FROM_PARENT",
            "memory_limit_applied": True,
            "memory_limit_reused": True,
            "memory_limit_inherited": True,
        }

    ULONG_PTR = ctypes.c_size_t
    SIZE_T = ctypes.c_size_t
    DWORD = wintypes.DWORD
    BOOL = wintypes.BOOL
    HANDLE = wintypes.HANDLE

    class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_longlong),
            ("PerJobUserTimeLimit", ctypes.c_longlong),
            ("LimitFlags", DWORD),
            ("MinimumWorkingSetSize", SIZE_T),
            ("MaximumWorkingSetSize", SIZE_T),
            ("ActiveProcessLimit", DWORD),
            ("Affinity", ULONG_PTR),
            ("PriorityClass", DWORD),
            ("SchedulingClass", DWORD),
        ]

    class IO_COUNTERS(ctypes.Structure):
        _fields_ = [
            ("ReadOperationCount", ctypes.c_ulonglong),
            ("WriteOperationCount", ctypes.c_ulonglong),
            ("OtherOperationCount", ctypes.c_ulonglong),
            ("ReadTransferCount", ctypes.c_ulonglong),
            ("WriteTransferCount", ctypes.c_ulonglong),
            ("OtherTransferCount", ctypes.c_ulonglong),
        ]

    class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION),
            ("IoInfo", IO_COUNTERS),
            ("ProcessMemoryLimit", SIZE_T),
            ("JobMemoryLimit", SIZE_T),
            ("PeakProcessMemoryUsed", SIZE_T),
            ("PeakJobMemoryUsed", SIZE_T),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    create_job = kernel32.CreateJobObjectW
    create_job.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    create_job.restype = HANDLE
    set_information = kernel32.SetInformationJobObject
    set_information.argtypes = [HANDLE, ctypes.c_int, ctypes.c_void_p, DWORD]
    set_information.restype = BOOL
    assign = kernel32.AssignProcessToJobObject
    assign.argtypes = [HANDLE, HANDLE]
    assign.restype = BOOL
    get_current = kernel32.GetCurrentProcess
    get_current.restype = HANDLE
    close_handle = kernel32.CloseHandle
    close_handle.argtypes = [HANDLE]
    close_handle.restype = BOOL

    job = create_job(None, None)
    if not job:
        error = ctypes.get_last_error()
        raise OSError(error, "DYNAMIC_QBD_CREATE_MEMORY_JOB_FAILED")

    info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
    JOB_OBJECT_LIMIT_JOB_MEMORY = 0x00000200
    JobObjectExtendedLimitInformation = 9
    info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_JOB_MEMORY
    info.JobMemoryLimit = int(limit_bytes)

    if not set_information(
        job,
        JobObjectExtendedLimitInformation,
        ctypes.byref(info),
        ctypes.sizeof(info),
    ):
        error = ctypes.get_last_error()
        close_handle(job)
        raise OSError(error, "DYNAMIC_QBD_SET_JOB_MEMORY_LIMIT_FAILED")

    if not assign(job, get_current()):
        error = ctypes.get_last_error()
        close_handle(job)
        raise OSError(error, "DYNAMIC_QBD_ASSIGN_PROCESS_TO_MEMORY_JOB_FAILED")

    _WINDOWS_JOB_HANDLE = job
    os.environ["DYNAMIC_QBD_MEMORY_JOB_INHERITED"] = "1"
    return {
        "memory_enforcement": "WINDOWS_JOB_OBJECT_JOB_MEMORY",
        "memory_limit_applied": True,
        "memory_limit_reused": False,
        "memory_limit_inherited": False,
    }


def _posix_process_memory_limit(limit_bytes: int) -> dict[str, Any]:
    """Best-effort fallback for non-Windows development/CI hosts."""
    try:
        import resource
        current_soft, current_hard = resource.getrlimit(resource.RLIMIT_AS)
        requested = int(limit_bytes)
        hard = current_hard
        if hard not in (resource.RLIM_INFINITY, -1):
            requested = min(requested, int(hard))
        resource.setrlimit(resource.RLIMIT_AS, (requested, hard))
        return {
            "memory_enforcement": "POSIX_RLIMIT_AS_PER_PROCESS_BEST_EFFORT",
            "memory_limit_applied": True,
            "memory_limit_reused": False,
            "memory_limit_inherited": False,
        }
    except Exception as exc:
        return {
            "memory_enforcement": "DECLARED_ONLY_UNSUPPORTED_HOST",
            "memory_limit_applied": False,
            "memory_limit_reused": False,
            "memory_limit_inherited": False,
            "memory_limit_error": f"{type(exc).__name__}:{exc}",
        }


def configure_memory_peak(
    memory_limit_gb: float | None = DEFAULT_MEMORY_LIMIT_GB,
    *,
    soft_target_gb: float | None = DEFAULT_MEMORY_SOFT_TARGET_GB,
    memory_limit_fraction: float | None = None,
    soft_target_fraction: float | None = None,
) -> dict[str, Any]:
    """Configure the suite-wide memory contract.

    ``memory_limit_gb`` is an absolute ceiling. ``soft_target_gb`` leaves
    allocator/transient headroom below the ceiling. Existing suites default to
    60/56 GiB; the surface-wide Fold-Clock suite opts into 90/84 GiB before any
    Dynamic-QBD surface import installs the Windows Job Object.
    """
    total_bytes = _physical_memory_bytes()
    if memory_limit_fraction is not None or soft_target_fraction is not None:
        if memory_limit_fraction is None or soft_target_fraction is None:
            raise ValueError("DYNAMIC_QBD_MEMORY_FRACTIONS_MUST_BE_PAIRED")
        if not 0 < float(soft_target_fraction) < float(memory_limit_fraction) <= 1:
            raise ValueError("DYNAMIC_QBD_MEMORY_FRACTION_LIMITS_INVALID")
        limit = total_bytes / _BYTES_PER_GIB * float(memory_limit_fraction)
        soft = total_bytes / _BYTES_PER_GIB * float(soft_target_fraction)
    else:
        limit = float(memory_limit_gb)
        soft = float(soft_target_gb)
    if not 0 < limit:
        raise ValueError("DYNAMIC_QBD_MEMORY_LIMIT_MUST_BE_POSITIVE")
    system_limit_gb = total_bytes / _BYTES_PER_GIB
    if limit > system_limit_gb:
        raise ValueError("DYNAMIC_QBD_MEMORY_LIMIT_MUST_NOT_EXCEED_SYSTEM_MEMORY")
    if not 0 < soft < limit:
        raise ValueError("DYNAMIC_QBD_MEMORY_SOFT_TARGET_MUST_BE_BELOW_LIMIT")

    limit_bytes = int(limit * _BYTES_PER_GIB)
    if sys.platform == "win32":
        enforcement = _windows_job_memory_limit(limit_bytes)
    else:
        enforcement = _posix_process_memory_limit(limit_bytes)

    os.environ["DYNAMIC_QBD_MEMORY_LIMIT_GB"] = str(limit)
    os.environ["DYNAMIC_QBD_MEMORY_SOFT_TARGET_GB"] = str(soft)
    return {
        "memory_limit_gb": limit,
        "memory_limit_bytes": limit_bytes,
        "memory_soft_target_gb": soft,
        "memory_soft_target_bytes": int(soft * _BYTES_PER_GIB),
        "memory_total_system_gb": total_bytes / _BYTES_PER_GIB,
        "memory_limit_fraction": memory_limit_fraction,
        "memory_soft_target_fraction": soft_target_fraction,
        "memory_headroom_gb": limit - soft,
        "memory_contract": _memory_contract_name(limit, fraction=memory_limit_fraction),
        **enforcement,
    }


def configure_cpu_peak(
    target_fraction: float = .90,
    *,
    process_workers: int = 1,
    native_threads_per_worker: int | None = None,
    memory_limit_gb: float = DEFAULT_MEMORY_LIMIT_GB,
    memory_soft_target_gb: float = DEFAULT_MEMORY_SOFT_TARGET_GB,
    memory_limit_fraction: float | None = None,
    memory_soft_target_fraction: float | None = None,
    enforce_capacity_fraction: bool = False,
) -> dict[str, Any]:
    """Configure numerical CPU threads, affinity and the suite memory ceiling."""
    global _ACTIVE_LIMITER, _ACTIVE_CONTRACT
    workers = max(1, int(process_workers))
    budget = cpu_budget(
        target_fraction, enforce_capacity_fraction=enforce_capacity_fraction
    )
    memory = configure_memory_peak(
        memory_limit_gb, soft_target_gb=memory_soft_target_gb,
        memory_limit_fraction=memory_limit_fraction,
        soft_target_fraction=memory_soft_target_fraction,
    )
    target = int(budget["target_logical_processors"])
    # A multi-process fit uses one native numerical thread per worker.  Pin
    # the worker pool to the requested capacity instead of leaving every
    # child eligible for all logical CPUs.  On the current host 26 workers
    # means 24 physical-core slots plus two distributed SMT slots, i.e.
    # 81.25% of the 32 logical CPUs and within the 80-90% load target.
    requested_capacity = max(
        1,
        min(
            int(budget["logical_processors"]),
            int(round(int(budget["logical_processors"]) * float(target_fraction))),
        ),
    )
    effective_target = target if workers <= 1 else min(requested_capacity, workers)
    affinity_target = effective_target
    affinity_reserve = max(0, int(budget["logical_processors"]) - affinity_target)
    plan = physical_core_affinity_plan(
        affinity_target,
        reserve_logical_processors=affinity_reserve,
    )
    selected = [int(row["logical_processor"]) for row in plan.get("worker_map", [])]
    affinity_set = set_current_process_affinity(selected) if selected else False
    native = int(native_threads_per_worker or max(1, effective_target // workers))
    if native < 1:
        raise ValueError("DYNAMIC_QBD_NATIVE_THREADS_MUST_BE_POSITIVE")
    for name in (
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS", "POLARS_MAX_THREADS",
    ):
        os.environ[name] = str(native)
    if _ACTIVE_LIMITER is not None:
        _ACTIVE_LIMITER.restore_original_limits()
    _ACTIVE_LIMITER = threadpool_limits(limits=native)
    capacity_label = "BOUNDED_CPU_CAPACITY" if workers > 1 or enforce_capacity_fraction else "FULL_CPU_CAPACITY"
    result = {
        **budget,
        "target_logical_processors": int(effective_target),
        "reserve_logical_processors": int(affinity_reserve),
        "available_capacity_fraction": float(
            effective_target / max(1, int(budget["logical_processors"]))
        ),
        "capacity_fraction_enforced": bool(workers > 1 or enforce_capacity_fraction),
        **memory,
        "configured": True,
        "selected_logical_processors": selected,
        "worker_map": list(plan.get("worker_map", [])),
        "reserved_logical_processors": plan.get("reserved_logical_processors", []),
        "affinity_target_logical_processors": int(affinity_target),
        "effective_capacity_fraction": float(len(selected) / max(1, int(budget["logical_processors"]))),
        "effective_reserved_logical_processors": int(
            max(0, int(budget["logical_processors"]) - len(selected))
        ),
        "worker_slot_count": int(len(selected)),
        "physical_primary_worker_slots": int(sum(
            1 for row in plan.get("worker_map", [])
            if row.get("role") == "physical_primary"
        )),
        "smt_overflow_worker_slots": int(sum(
            1 for row in plan.get("worker_map", [])
            if row.get("role") == "smt_overflow"
        )),
        "affinity_enabled": bool(affinity_set),
        "native_threads_per_worker": native,
        "model_fit_processes": workers,
        "contract": f"{capacity_label}_WITH_{memory['memory_contract']}",
    }
    _ACTIVE_CONTRACT = dict(result)
    print(
        "[dynamic-qbd-resources] "
        + " ".join(f"{key}={value}" for key, value in result.items()),
        flush=True,
    )
    return result


def active_cpu_contract() -> dict[str, Any]:
    if _ACTIVE_CONTRACT is not None:
        return dict(_ACTIVE_CONTRACT)
    budget = cpu_budget(.90)
    return {
        **budget,
        "configured": False,
        "memory_limit_gb": DEFAULT_MEMORY_LIMIT_GB,
        "memory_soft_target_gb": DEFAULT_MEMORY_SOFT_TARGET_GB,
        "memory_headroom_gb": DEFAULT_MEMORY_LIMIT_GB - DEFAULT_MEMORY_SOFT_TARGET_GB,
        "memory_contract": _memory_contract_name(DEFAULT_MEMORY_LIMIT_GB),
        "memory_enforcement": "NOT_CONFIGURED",
        "memory_limit_applied": False,
        "memory_limit_inherited": False,
        "contract": f"FULL_CPU_CAPACITY_WITH_{_memory_contract_name(DEFAULT_MEMORY_LIMIT_GB)}",
    }
