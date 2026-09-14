from __future__ import annotations

import os

from .cpu_topology import physical_core_affinity_plan, windows_cpu_topology


def main() -> int:
    topo = windows_cpu_topology()
    plan = physical_core_affinity_plan(12, reserve_logical_processors=4)

    if os.name != "nt":
        assert not topo["affinity_supported"]
        assert not plan["enabled"]
        expected = max(1, int(topo["logical_processors"]) - min(4, max(0, int(topo["logical_processors"]) - 1)))
        assert int(plan["workers"]) == min(12, expected)
        print("CPU_TOPOLOGY_SELF_TEST_PASS: non-Windows scheduler fallback")
        return 0

    assert topo["available"], topo
    assert int(topo["logical_processors"]) >= 1
    assert int(topo["physical_cores"]) >= 1
    cores = topo["cores"]
    assert len(cores) == int(topo["physical_cores"])

    logicals = [int(lp) for core in cores for lp in core["logical_processors"]]
    assert len(logicals) == len(set(logicals))
    assert len(logicals) == int(topo["logical_processors"])

    if topo["affinity_supported"]:
        worker_map = list(plan["worker_map"])
        selected = {int(x["logical_processor"]) for x in worker_map}
        reserved = {int(x) for x in plan["reserved_logical_processors"]}
        expected_workers = min(12, max(1, len(logicals) - min(4, max(0, len(logicals) - 1))))

        assert len(worker_map) == expected_workers
        assert len(selected) == expected_workers
        assert selected.isdisjoint(reserved)
        assert selected | reserved == set(logicals)
        assert len(reserved) == len(logicals) - expected_workers

        covered_cores = {int(x["core_index"]) for x in worker_map}
        assert len(covered_cores) == min(int(topo["physical_cores"]), expected_workers)

        physical_primary = [x for x in worker_map if x.get("role") == "physical_primary"]
        smt_overflow = [x for x in worker_map if x.get("role") == "smt_overflow"]
        assert len(physical_primary) == min(int(topo["physical_cores"]), expected_workers)
        assert len(smt_overflow) == max(0, expected_workers - int(topo["physical_cores"]))

        # Exact Ryzen 7 5800X target: 8C/16T -> 12 replay workers + 4 reserve CPUs.
        if int(topo["physical_cores"]) == 8 and int(topo["logical_processors"]) == 16:
            assert len(worker_map) == 12
            assert len(physical_primary) == 8
            assert len(smt_overflow) == 4
            assert len(reserved) == 4

    print("CPU_TOPOLOGY_SELF_TEST_PASS")
    print(f"source={topo['source']}")
    print(f"physical_cores={topo['physical_cores']} logical_processors={topo['logical_processors']}")
    print(f"worker_map={plan['worker_map']}")
    print(f"reserved_logical_processors={plan['reserved_logical_processors']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
