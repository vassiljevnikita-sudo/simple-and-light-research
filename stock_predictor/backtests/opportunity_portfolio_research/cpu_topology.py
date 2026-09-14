from __future__ import annotations

from collections import defaultdict
import ctypes
from ctypes import wintypes
import os
from typing import Any


CPU_SET_INFORMATION_TYPE_CPU_SET = 0


def windows_cpu_topology() -> dict[str, Any]:
    """Return Windows CPU-set topology without external tools.

    On Windows 10+ GetSystemCpuSetInformation exposes both LogicalProcessorIndex
    and CoreIndex. Logical processors sharing a CoreIndex are SMT siblings. On
    non-Windows platforms this returns a conservative no-affinity description.
    """
    logical_count = int(os.cpu_count() or 1)
    if os.name != "nt":
        return {
            "available": False,
            "source": "non_windows_fallback",
            "logical_processors": logical_count,
            "physical_cores": None,
            "groups": None,
            "cores": [],
            "affinity_supported": False,
        }

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    fn = kernel32.GetSystemCpuSetInformation
    fn.argtypes = [ctypes.c_void_p, wintypes.ULONG, ctypes.POINTER(wintypes.ULONG), wintypes.HANDLE, wintypes.ULONG]
    fn.restype = wintypes.BOOL

    required = wintypes.ULONG(0)
    fn(None, 0, ctypes.byref(required), None, 0)
    if required.value <= 0:
        return {
            "available": False,
            "source": "GetSystemCpuSetInformation_unavailable",
            "logical_processors": logical_count,
            "physical_cores": None,
            "groups": None,
            "cores": [],
            "affinity_supported": False,
            "winerror": int(ctypes.get_last_error()),
        }

    buffer = ctypes.create_string_buffer(required.value)
    returned = wintypes.ULONG(required.value)
    ok = fn(buffer, required.value, ctypes.byref(returned), None, 0)
    if not ok:
        return {
            "available": False,
            "source": "GetSystemCpuSetInformation_failed",
            "logical_processors": logical_count,
            "physical_cores": None,
            "groups": None,
            "cores": [],
            "affinity_supported": False,
            "winerror": int(ctypes.get_last_error()),
        }

    raw = memoryview(buffer.raw)[: returned.value]
    offset = 0
    cpus: list[dict[str, int | bool]] = []
    while offset + 8 <= len(raw):
        size = int.from_bytes(raw[offset:offset + 4], "little")
        info_type = int.from_bytes(raw[offset + 4:offset + 8], "little")
        if size < 8 or offset + size > len(raw):
            break
        if info_type == CPU_SET_INFORMATION_TYPE_CPU_SET and size >= 24:
            base = offset + 8
            cpu_id = int.from_bytes(raw[base:base + 4], "little")
            group = int.from_bytes(raw[base + 4:base + 6], "little")
            logical = int(raw[base + 6])
            core = int(raw[base + 7])
            llc = int(raw[base + 8])
            numa = int(raw[base + 9])
            efficiency = int(raw[base + 10])
            flags = int(raw[base + 11])
            cpus.append({
                "cpu_set_id": cpu_id,
                "group": group,
                "logical_processor": logical,
                "core_index": core,
                "last_level_cache_index": llc,
                "numa_node_index": numa,
                "efficiency_class": efficiency,
                "parked": bool(flags & 0x01),
                "allocated": bool(flags & 0x02),
                "allocated_to_target_process": bool(flags & 0x04),
                "realtime": bool(flags & 0x08),
            })
        offset += size

    by_core: dict[tuple[int, int], list[dict[str, int | bool]]] = defaultdict(list)
    for cpu in cpus:
        by_core[(int(cpu["group"]), int(cpu["core_index"]))].append(cpu)

    cores = []
    for (group, core_index), members in sorted(by_core.items()):
        members = sorted(members, key=lambda x: (bool(x["parked"]), int(x["logical_processor"])))
        logicals = [int(x["logical_processor"]) for x in members]
        active = [x for x in members if not bool(x["parked"])] or members
        primary = int(active[0]["logical_processor"])
        siblings = [x for x in logicals if x != primary]
        cores.append({
            "group": group,
            "core_index": core_index,
            "logical_processors": logicals,
            "primary_logical_processor": primary,
            "sibling_logical_processors": siblings,
            "smt_width": len(logicals),
            # Preserve the Windows efficiency class so the affinity planner
            # can fill higher-performance P-core primaries before E-core/SMT
            # capacity on hybrid hosts.
            "efficiency_class": min(
                int(x.get("efficiency_class", 0)) for x in members
            ),
            "parked_logical_processors": [
                int(x["logical_processor"]) for x in members if bool(x["parked"])
            ],
        })
    cores.sort(key=lambda core: (
        -int(core.get("efficiency_class", 0)),
        int(core["group"]),
        int(core["core_index"]),
    ))

    groups = sorted({int(x["group"]) for x in cpus})
    # The Ryzen 7 5800X target has one processor group. Fail closed on multi-group
    # systems rather than applying a SetProcessAffinityMask mask to the wrong group.
    affinity_supported = bool(cores) and groups == [0] and all(
        int(lp) < 64 for c in cores for lp in c["logical_processors"]
    )
    return {
        "available": bool(cpus),
        "source": "GetSystemCpuSetInformation",
        "logical_processors": len(cpus) if cpus else logical_count,
        "physical_cores": len(cores) if cores else None,
        "groups": groups,
        "cores": cores,
        "affinity_supported": affinity_supported,
    }


def _spread_smt_candidates(cores: list[dict]) -> list[tuple[dict, int]]:
    """Interleave SMT siblings across cores before using a second sibling on a core."""
    per_core = [
        (core, [int(x) for x in core.get("sibling_logical_processors", [])])
        for core in cores
    ]
    result: list[tuple[dict, int]] = []
    max_width = max((len(items) for _, items in per_core), default=0)
    for sibling_index in range(max_width):
        layer = [
            (core, items[sibling_index])
            for core, items in per_core
            if sibling_index < len(items)
        ]
        # Alternate cores first so a partial SMT expansion is distributed rather
        # than concentrated on adjacent physical cores.
        result.extend(layer[::2])
        result.extend(layer[1::2])
    return result


def physical_core_affinity_plan(
    max_workers: int,
    reserve_logical_processors: int = 4,
) -> dict[str, Any]:
    """Build a topology-aware worker plan with explicit coordinator reserve.

    Workers first occupy one logical processor per physical core. Requests beyond
    the physical-core count use SMT siblings, while up to reserve_logical_processors
    remain unused by replay workers for the main/coordinator process and the OS.
    """
    topology = windows_cpu_topology()
    requested = max(1, int(max_workers))
    reserve_requested = max(0, int(reserve_logical_processors))
    cores = list(topology.get("cores") or [])
    logical_count = int(topology.get("logical_processors") or os.cpu_count() or 1)
    if not topology.get("affinity_supported") or not cores:
        usable = max(1, logical_count - min(reserve_requested, max(0, logical_count - 1)))
        return {
            "enabled": False,
            "workers": min(requested, usable),
            "primary_logical_processors": [],
            "overflow_logical_processors": [],
            "reserved_logical_processors": [],
            "spare_smt_logical_processors": [],
            "worker_map": [],
            "reserve_requested": reserve_requested,
            "topology": topology,
        }

    all_logicals = sorted({
        int(lp) for core in cores for lp in core.get("logical_processors", [])
    })
    reserve = min(reserve_requested, max(0, len(all_logicals) - 1))
    max_worker_capacity = max(1, len(all_logicals) - reserve)
    target_workers = min(requested, max_worker_capacity)

    worker_map: list[dict[str, Any]] = []
    selected_cpus: set[int] = set()

    # First fill every physical core once.
    for core in cores:
        if len(worker_map) >= target_workers:
            break
        primary = int(core["primary_logical_processor"])
        siblings = [int(x) for x in core.get("sibling_logical_processors", [])]
        worker_map.append({
            "worker_slot": len(worker_map),
            "role": "physical_primary",
            "group": int(core["group"]),
            "core_index": int(core["core_index"]),
            "logical_processor": primary,
            "efficiency_class": int(core.get("efficiency_class", 0)),
            "smt_siblings": siblings,
        })
        selected_cpus.add(primary)

    # Then add SMT workers, spread across cores, while preserving the reserve.
    if len(worker_map) < target_workers:
        for core, logical in _spread_smt_candidates(cores):
            if len(worker_map) >= target_workers:
                break
            if logical in selected_cpus:
                continue
            worker_map.append({
                "worker_slot": len(worker_map),
                "role": "smt_overflow",
                "group": int(core["group"]),
                "core_index": int(core["core_index"]),
                "logical_processor": int(logical),
                "efficiency_class": int(core.get("efficiency_class", 0)),
                "smt_siblings": [
                    int(x) for x in core.get("logical_processors", []) if int(x) != int(logical)
                ],
            })
            selected_cpus.add(int(logical))

    reserved = [lp for lp in all_logicals if lp not in selected_cpus]
    primary = [
        int(x["logical_processor"])
        for x in worker_map
        if x.get("role") == "physical_primary"
    ]
    overflow = [
        int(x["logical_processor"])
        for x in worker_map
        if x.get("role") == "smt_overflow"
    ]
    return {
        "enabled": True,
        "workers": len(worker_map),
        "primary_logical_processors": primary,
        "overflow_logical_processors": overflow,
        "reserved_logical_processors": reserved,
        # Compatibility name used by existing logging/tests.
        "spare_smt_logical_processors": reserved,
        "worker_map": worker_map,
        "reserve_requested": reserve_requested,
        "topology": topology,
    }


def set_current_process_affinity(logical_processors: list[int] | tuple[int, ...]) -> bool:
    if os.name != "nt":
        return False
    logicals = sorted({int(x) for x in logical_processors})
    if not logicals or min(logicals) < 0 or max(logicals) >= 64:
        return False
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    get_current = kernel32.GetCurrentProcess
    get_current.argtypes = []
    get_current.restype = wintypes.HANDLE
    set_mask = kernel32.SetProcessAffinityMask
    set_mask.argtypes = [wintypes.HANDLE, ctypes.c_size_t]
    set_mask.restype = wintypes.BOOL
    mask_value = 0
    for logical in logicals:
        mask_value |= 1 << logical
    return bool(set_mask(get_current(), ctypes.c_size_t(mask_value)))


def set_current_process_logical_affinity(logical_processor: int) -> bool:
    return set_current_process_affinity([int(logical_processor)])
