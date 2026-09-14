from __future__ import annotations

import argparse
from pathlib import Path

from .portfolio_evidence_expansion import run_evidence


def parse_args():
    parser = argparse.ArgumentParser(description="Run diagnostics on frozen OOS outer-fold policies.")
    parser.add_argument("--v5-predictions", required=True)
    parser.add_argument("--daily-store-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--max-workers", type=int, default=24)
    parser.add_argument("--coordinator-threads", type=int, default=4)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_evidence(
        Path(args.v5_predictions),
        Path(args.daily_store_root),
        Path(args.output_root),
        max_workers=args.max_workers,
        coordinator_threads=args.coordinator_threads,
    )
    print(result)
    return 0 if result.get("baseline_consistency_pass") else 2


if __name__ == "__main__":
    raise SystemExit(main())
