from __future__ import annotations

"""Explicit executable runner for the Prediction Horizon x Holding Days QbD surface.

The public runner keeps the established QbD research contract but uses the
Prediction x Hold throughput runtime: a deeper shared cell queue and bounded
replay worker pool.
"""

import multiprocessing as mp
import sys

from .prediction_hold_qbd_surface import parse_args
from .prediction_hold_qbd_process_pool_readiness import qbd_process_pool_readiness_contract
from .prediction_hold_qbd_throughput_runtime import (
    DEFAULT_QBD_COORDINATORS,
    DEFAULT_QBD_WORKERS,
    run_qbd_throughput,
)


def run() -> int:
    args = parse_args()
    # Keep direct `python -m ...prediction_hold_qbd_runner` aligned with the
    # PowerShell launcher even though the legacy parser still documents 12/4.
    if "--max-workers" not in sys.argv:
        args.max_workers = DEFAULT_QBD_WORKERS
    if "--coordinator-threads" not in sys.argv:
        args.coordinator_threads = DEFAULT_QBD_COORDINATORS

    # Hold research dispatch until every configured worker is initialized. Once
    # ready, all independent window coordinators feed the same persistent pool.
    with qbd_process_pool_readiness_contract():
        return run_qbd_throughput(args)


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(run())
