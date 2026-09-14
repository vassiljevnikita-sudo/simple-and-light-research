"""Fast unit/contract checks for the paired selector tournament."""
from __future__ import annotations

from .dynamic_qbd_paired_selector_tournament import _self_test


def test_paired_selector_tournament_self_test() -> None:
    _self_test()


if __name__ == "__main__":
    _self_test()
