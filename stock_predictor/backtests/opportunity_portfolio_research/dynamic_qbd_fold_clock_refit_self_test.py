"""Deterministic contract-only self-test for the fold-clock plan builder.

This module is intentionally not executed by the research runner. It provides
a small executable check for the calendar-independent fold-clock transition
contract before a real development run is launched.
"""
from __future__ import annotations

from datetime import date

from .dynamic_qbd_fold_clock_refit_experiment import (
    ARM_A,
    ARM_F,
    ARM_M,
    _build_plans,
    _fold_evidence_fingerprint,
)


def _choice(fold_ids: tuple[str, ...]) -> dict:
    return {
        "candidate_id": "SELF_TEST_CANDIDATE",
        "family": "RIDGE",
        "parameters": {"alpha": 1.0},
        "fold_count": len(fold_ids),
        "fold_ids": fold_ids,
        "robust_score": 0.5,
    }


def main() -> None:
    fold_states = (
        (date(2020, 8, 31), date(2020, 8, 28), ("F1", "F2")),
        (date(2020, 9, 30), date(2020, 9, 25), ("F1", "F2")),
        (date(2020, 10, 30), date(2020, 10, 23), ("F1", "F2")),
        (date(2021, 2, 26), date(2021, 2, 19), ("F1", "F2", "F3")),
        (date(2021, 3, 31), date(2021, 3, 26), ("F1", "F2", "F3")),
        (date(2021, 8, 31), date(2021, 8, 27), ("F1", "F2", "F3", "F4")),
    )
    assessments = []
    for assessment_date, matured_date, fold_ids in fold_states:
        choice = _choice(fold_ids)
        assessments.append({
            "assessment_date": assessment_date,
            "latest_matured_evidence_date": matured_date,
            "eligible_choices": (choice,),
            "selector_winner": dict(choice),
        })

    plans, audit = _build_plans(assessments)
    assert [row["refit_required"] for row in plans[ARM_A]] == [True, False, False, False, False, False]
    assert [row["refit_required"] for row in plans[ARM_F]] == [True, False, False, True, False, True]
    assert [row["refit_required"] for row in plans[ARM_M]] == [True, True, True, True, True, True]
    assert audit[1]["evidence_expanded"] is False
    assert audit[3]["evidence_expanded"] is True
    assert audit[5]["evidence_expanded"] is True
    assert _fold_evidence_fingerprint(("F1", "F2")) == _fold_evidence_fingerprint(("F2", "F1"))
    assert len({row["frozen_recipe"] for row in audit}) == 1

    # A calibration value is allowed to change only at a F event. This models
    # the frozen-between-events property without invoking the fit pipeline.
    thresholds = []
    current = None
    for row in plans[ARM_F]:
        if row["refit_required"]:
            current = 0.10 + len(thresholds) * 0.01
        thresholds.append(current)
    expected_thresholds = [0.10, 0.10, 0.10, 0.13, 0.13, 0.15]
    assert all(abs(actual - expected) <= 1e-12 for actual, expected in zip(thresholds, expected_thresholds))

    print("DYNAMIC_QBD_FOLD_CLOCK_REFIT_SELF_TEST_PASS")


if __name__ == "__main__":
    main()
