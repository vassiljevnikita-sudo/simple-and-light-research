"""Deterministic contract self-test for the Fold-Clock Surface Fold-Clock suite."""
from __future__ import annotations

from datetime import date

from .dynamic_qbd_fold_clock_surface_validation_2016_2025 import (
    _fold_evidence_expanded,
    _fold_evidence_fingerprint,
)


def main() -> None:
    states = [
        (date(2020, 8, 31), ("F1", "F2")),
        (date(2020, 9, 30), ("F1", "F2")),
        (date(2020, 10, 30), ("F1", "F2")),
        (date(2020, 11, 30), ("F1", "F2", "F3")),
        (date(2020, 12, 31), ("F1", "F2", "F3")),
        (date(2021, 1, 29), ("F1", "F2", "F3", "F4")),
    ]
    previous = None
    expanded = []
    fingerprints = []
    for _, fold_ids in states:
        expanded.append(_fold_evidence_expanded(previous, fold_ids))
        fingerprints.append(_fold_evidence_fingerprint(fold_ids))
        previous = fold_ids

    assert expanded == [True, False, False, True, False, True]
    assert fingerprints[0] == fingerprints[1] == fingerprints[2]
    assert fingerprints[3] == fingerprints[4]
    assert fingerprints[0] != fingerprints[3]
    assert _fold_evidence_fingerprint(("F2", "F1")) == fingerprints[0]

    try:
        _fold_evidence_expanded(("F1", "F2", "F3"), ("F1", "F3"))
    except AssertionError:
        pass
    else:
        raise AssertionError("non-monotonic fold evidence was accepted")

    # Calendar progress alone cannot create a new evidence fingerprint.
    assert _fold_evidence_fingerprint(("F1", "F2")) == _fold_evidence_fingerprint(
        ["F1", "F2"]
    )
    print("DYNAMIC_QBD_FOLD_CLOCK_SURFACE_FOLD_CLOCK_SELF_TEST_PASS")


if __name__ == "__main__":
    main()
