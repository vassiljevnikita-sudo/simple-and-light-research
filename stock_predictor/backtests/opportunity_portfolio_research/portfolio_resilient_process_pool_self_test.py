from __future__ import annotations

from concurrent.futures import Future
from concurrent.futures.process import BrokenProcessPool

from . import portfolio_replay_process_backend as backend
from . import portfolio_resilient_process_pool as resilient


class _FakePool:
    def __init__(self, workers: int, broken: bool, empty: bool = False) -> None:
        self._max_workers = workers
        self.broken = broken
        self.empty = empty

    def submit(self, _func, payload):
        future = Future()
        if self.broken:
            future.set_exception(BrokenProcessPool("synthetic child crash"))
        else:
            value = {
                "payload": payload,
                "_worker_telemetry": {
                    "worker_index": int(payload),
                    "compute_seconds": 0.1,
                    "idle_seconds_before_job": 0.2,
                },
            }
            if self.empty:
                value = {"_worker_empty_result": True, "_worker_telemetry": value["_worker_telemetry"]}
            future.set_result(value)
        return future

    def shutdown(self, wait=True, cancel_futures=False):
        return None


def main() -> int:
    original_recover = resilient._recover_pool
    original_write = backend._write_telemetry
    telemetry = []
    committed = {}
    good_pool = _FakePool(12, False)

    try:
        resilient._recover_pool = lambda failed_pool, label, pending_jobs, exc: good_pool
        backend._write_telemetry = lambda record: telemetry.append(dict(record))
        resilient.resilient_run_process_futures(
            _FakePool(12, True),
            [(0, 0), (1, 1), (2, 2)],
            lambda x: x,
            "synthetic recovery",
            lambda index, value: committed.__setitem__(index, value),
            jobs_reused=4,
            main_cache_seconds=0.01,
        )
    finally:
        resilient._recover_pool = original_recover
        backend._write_telemetry = original_write

    assert sorted(committed) == [0, 1, 2]
    assert all("_worker_telemetry" not in value for value in committed.values())
    assert [committed[i]["payload"] for i in sorted(committed)] == [0, 1, 2]
    final = next(row for row in reversed(telemetry) if row.get("batch") == "synthetic recovery")
    assert final["jobs_submitted"] == 3
    assert final["jobs_reused"] == 4
    assert final["jobs_computed"] == 3
    assert final["pool_recoveries"] == 1
    assert sum(final["worker_job_counts"].values()) == 3

    empty_committed = {}
    empty_pool = _FakePool(12, False, empty=True)
    backend._write_telemetry = lambda record: telemetry.append(dict(record))
    try:
        resilient.resilient_run_process_futures(
            empty_pool,
            [(11, 11)],
            lambda x: x,
            "synthetic empty result",
            lambda index, value: empty_committed.__setitem__(index, value),
        )
    finally:
        backend._write_telemetry = original_write
    assert empty_committed == {11: None}
    empty_final = next(row for row in reversed(telemetry) if row.get("batch") == "synthetic empty result")
    assert empty_final["worker_job_counts"]["11"] == 1
    assert set(empty_final["worker_job_counts"]) == {str(i) for i in range(12)}
    assert empty_final["worker_slots_expected"] == [str(i) for i in range(12)]
    assert empty_final["worker_slots_with_jobs"] == ["11"]
    cumulative = next(row for row in reversed(telemetry) if row.get("event") == "cumulative_telemetry")
    assert cumulative["jobs_computed"] >= 4
    assert "6" in cumulative["worker_utilization"]

    # Custom-initialized pools (used by the V4.5 high/low/ATR overlay) must use the
    # same checkpoint-before-retry semantics and 12->12 first-recovery policy.
    external_committed = {}
    external_calls = []
    external_telemetry = []

    def external_factory(workers: int):
        external_calls.append(int(workers))
        return _FakePool(workers, broken=len(external_calls) == 1)

    backend._write_telemetry = lambda record: external_telemetry.append(dict(record))
    resilient._RECOVERY_COUNT = 0
    try:
        resilient.resilient_run_external_process_futures(
            external_factory,
            12,
            [(0, 0), (1, 1), (2, 2)],
            lambda x: x,
            "synthetic external recovery",
            lambda index, value: external_committed.__setitem__(index, value),
            jobs_reused=5,
            telemetry_extra={"research_contract": "TEST_EXTERNAL"},
        )
    finally:
        backend._write_telemetry = original_write

    assert external_calls == [12, 12]
    assert sorted(external_committed) == [0, 1, 2]
    assert [external_committed[i]["payload"] for i in sorted(external_committed)] == [0, 1, 2]
    external_final = next(row for row in reversed(external_telemetry) if row.get("batch") == "synthetic external recovery")
    assert external_final["jobs_submitted"] == 3
    assert external_final["jobs_reused"] == 5
    assert external_final["jobs_computed"] == 3
    assert external_final["pool_recoveries"] == 1
    assert external_final["research_contract"] == "TEST_EXTERNAL"
    assert any(row.get("event") == "pool_recovery" for row in external_telemetry)

    print("RESILIENT_PROCESS_POOL_SELF_TEST_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
