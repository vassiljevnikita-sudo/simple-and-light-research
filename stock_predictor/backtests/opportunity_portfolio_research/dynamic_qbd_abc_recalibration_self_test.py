"""Deterministic contract test for the matched A/B/C recalibration suite."""
from __future__ import annotations

from datetime import date

from .dynamic_qbd_abc_recalibration_experiment import (
    ARM_A,
    ARM_B,
    ARM_C,
    _build_plans,
    _contrast,
    _pair_diagnosis,
)


def _choice(candidate: str, family: str, score: float, folds: int) -> dict:
    return {
        "candidate_id": candidate,
        "family": family,
        "parameters": {"p": candidate},
        "fold_count": folds,
        "fold_ids": tuple(f"WF_{i:03d}" for i in range(folds)),
        "robust_score": score,
    }


def main() -> None:
    ridge2 = _choice("RIDGE_A", "RIDGE", 0.01, 2)
    ridge3 = {**ridge2, "fold_count": 3, "fold_ids": (*ridge2["fold_ids"], "WF_002"), "robust_score": 0.02}
    hgb3 = _choice("HGB_A", "HIST_GRADIENT_BOOSTING", 0.03, 3)

    assessments = [
        {
            "assessment_date": date(2020, 8, 31),
            "latest_matured_evidence_date": date(2020, 8, 26),
            "evidence_fingerprint": "E2",
            "eligible_choices": (ridge2,),
            "selector_winner": ridge2,
        },
        {
            "assessment_date": date(2020, 9, 30),
            "latest_matured_evidence_date": date(2020, 9, 25),
            "evidence_fingerprint": "E2",
            "eligible_choices": (ridge2,),
            "selector_winner": ridge2,
        },
        {
            "assessment_date": date(2021, 2, 26),
            "latest_matured_evidence_date": date(2021, 2, 23),
            "evidence_fingerprint": "E3",
            "eligible_choices": (ridge3, hgb3),
            "selector_winner": hgb3,
        },
    ]

    plans, audit = _build_plans(assessments)
    assert len(audit) == 3
    assert [x["refit_required"] for x in plans[ARM_A]] == [True, False, False]
    assert [x["refit_required"] for x in plans[ARM_B]] == [True, False, False]
    assert [x["refit_required"] for x in plans[ARM_C]] == [True, True, True]

    # Recipe is frozen to the first causal Ridge choice even after HGB later wins.
    for arm in (ARM_A, ARM_B, ARM_C):
        assert all(row["choice"]["candidate_id"] == "RIDGE_A" for row in plans[arm])
    assert audit[-1]["selector_winner_is_frozen_recipe"] is False

    a = {
        "strategy": ARM_A,
        "terminal_value": 12000.0,
        "terminal_relative_return": 0.10,
        "cagr": 0.06,
        "cagr_excess": 0.01,
        "relative_max_drawdown": -0.20,
        "trade_count": 40,
        "total_cost_eur": 300.0,
    }
    b = {
        "strategy": ARM_B,
        "terminal_value": 11000.0,
        "terminal_relative_return": 0.00,
        "cagr": 0.03,
        "cagr_excess": -0.02,
        "relative_max_drawdown": -0.30,
        "trade_count": 80,
        "total_cost_eur": 500.0,
    }
    c = {
        "strategy": ARM_C,
        "terminal_value": 11500.0,
        "terminal_relative_return": 0.05,
        "cagr": 0.045,
        "cagr_excess": -0.005,
        "relative_max_drawdown": -0.25,
        "trade_count": 65,
        "total_cost_eur": 450.0,
    }

    ba = _contrast(b, a)
    cb = _contrast(c, b)
    assert ba["terminal_value_delta_eur"] == -1000.0
    assert ba["relative_max_drawdown_delta"] < 0.0
    assert cb["terminal_value_delta_eur"] == 500.0
    assert cb["relative_max_drawdown_delta"] > 0.0
    assert _pair_diagnosis(b, a, "BA") == "BA_PARETO_WORSENS"
    assert _pair_diagnosis(c, b, "CB") == "CB_PARETO_IMPROVES"

    print("DYNAMIC_QBD_ABC_RECALIBRATION_SELF_TEST_PASS")


if __name__ == "__main__":
    main()
