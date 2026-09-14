from __future__ import annotations

"""Explicit executable runner for Phase-3 Replacement QbD."""

import multiprocessing as mp

from .replacement_qbd_surface import main, parse_args


def run() -> int:
    return main(parse_args())


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(run())
