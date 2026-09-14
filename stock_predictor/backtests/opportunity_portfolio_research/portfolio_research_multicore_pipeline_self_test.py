from __future__ import annotations

import multiprocessing as mp
import os

from . import portfolio_policy_search as search
from . import portfolio_policy_walk_forward as multicore_walk_forward
from .affinity_coordinator_pool import AffinityCoordinatorPool
from .portfolio_replay_process_backend import install_multicore_backend, shutdown_multicore_backend
from .portfolio_replay_process_backend_self_test import _fixture, _outer_signature


def main() -> int:
    prices, signals = _fixture()
    original_parallel_map = search._parallel_map
    old_fragment_path = os.environ.pop("OPPORTUNITY_FRAGMENT_CACHE_PATH", None)
    old_pipeline = os.environ.get("OPPORTUNITY_WINDOW_PIPELINE")
    os.environ["OPPORTUNITY_WINDOW_PIPELINE"] = "2"

    try:
        install_multicore_backend(2)

        serial_outer, _serial_history, serial_meta = search.run_walk_forward(
            signals,
            prices,
            horizons=(10,),
            budget=8,
            max_workers=2,
            fragment_cache_path=None,
            fragment_namespace=None,
        )

        multicore_walk_forward.ThreadPoolExecutor = AffinityCoordinatorPool
        pipelined_outer, _pipeline_history, pipeline_meta = multicore_walk_forward.run_walk_forward(
            signals,
            prices,
            horizons=(10,),
            budget=8,
            max_workers=2,
            fragment_cache_path=None,
            fragment_namespace=None,
        )

        assert _outer_signature(pipelined_outer) == _outer_signature(serial_outer)
        assert pipeline_meta["final_policies"][10].policy_id == serial_meta["final_policies"][10].policy_id
        assert pipeline_meta["final_thresholds"] == serial_meta["final_thresholds"]
        perf = pipeline_meta.get("performance", {})
        assert perf.get("outer_window_pipeline") is True
        hmeta = perf.get("outer_window_pipeline_by_horizon", {}).get(10, {})
        assert int(hmeta.get("coordinator_workers", 0)) >= 1
    finally:
        shutdown_multicore_backend()
        search._parallel_map = original_parallel_map
        if old_fragment_path is not None:
            os.environ["OPPORTUNITY_FRAGMENT_CACHE_PATH"] = old_fragment_path
        else:
            os.environ.pop("OPPORTUNITY_FRAGMENT_CACHE_PATH", None)
        if old_pipeline is not None:
            os.environ["OPPORTUNITY_WINDOW_PIPELINE"] = old_pipeline
        else:
            os.environ.pop("OPPORTUNITY_WINDOW_PIPELINE", None)

    print("MULTICORE_PIPELINE_SELF_TEST_PASS")
    return 0


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())
