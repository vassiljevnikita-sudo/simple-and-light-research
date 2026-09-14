"""Deterministic contract tests for Dynamic-QBD recipe-switch hysteresis."""
from __future__ import annotations

from datetime import date
import json

from .dynamic_qbd_recipe_hysteresis import (
    HysteresisConfig,
    RecipeHysteresisState,
    recipe_key,
    stable_evidence_fingerprint,
    state_from_dict,
    state_to_dict,
    update_recipe_hysteresis,
)


def _choice(candidate: str, family: str, score: float, folds: int = 2) -> dict:
    return {
        "candidate_id": candidate,
        "family": family,
        "parameters": {"p": candidate},
        "fold_count": folds,
        "fold_ids": tuple(f"WF_{i:03d}_2020-0{i+1}-01" for i in range(folds)),
        "robust_score": score,
    }


def _step(state, *, at: str, winner: dict, choices: tuple[dict, ...]):
    evidence_date = date.fromisoformat(at)
    fingerprint = stable_evidence_fingerprint(evidence_date=evidence_date, choices=choices)
    return update_recipe_hysteresis(
        state=state,
        assessment_date=evidence_date,
        evidence_date=evidence_date,
        evidence_fingerprint=fingerprint,
        selector_winner=winner,
        eligible_choices=choices,
        config=HysteresisConfig(confirmations_required=2, minimum_oos_folds=2),
    )


def main() -> None:
    ridge2 = _choice("RIDGE_A", "RIDGE", 0.10, folds=2)
    hgb2 = _choice("HGB_A", "HIST_GRADIENT_BOOSTING", 0.12, folds=2)

    state = RecipeHysteresisState()
    state, d0 = _step(state, at="2020-01-31", winner=ridge2, choices=(ridge2, hgb2))
    assert d0.reason_code == "INITIAL_CAUSAL_RECIPE"
    assert state.incumbent == recipe_key(ridge2)

    # Calendar progress with the exact same fold evidence is not a vote.
    state_same, d_same = _step(state, at="2020-02-29", winner=hgb2, choices=(ridge2, hgb2))
    assert d_same.reason_code == "NO_NEW_RECIPE_EVIDENCE"
    assert d_same.confirmations_after == 0
    assert state_same == state

    # A real fold-set expansion can provide the first challenger confirmation.
    ridge3 = {**ridge2, "robust_score": 0.09, "fold_count": 3,
              "fold_ids": (*ridge2["fold_ids"], "WF_002_2020-03-01")}
    hgb3 = {**hgb2, "robust_score": 0.14, "fold_count": 3,
            "fold_ids": (*hgb2["fold_ids"], "WF_002_2020-03-01")}
    state, d1 = _step(state, at="2020-03-31", winner=hgb3, choices=(ridge3, hgb3))
    assert d1.reason_code == "CHALLENGER_CONFIRMATION_PENDING"
    assert d1.confirmations_after == 1
    assert state.incumbent == recipe_key(ridge3)

    # Another month with unchanged 3-fold evidence must remain at one vote.
    state_unchanged, d_unchanged = _step(
        state, at="2020-04-30", winner=hgb3, choices=(ridge3, hgb3)
    )
    assert d_unchanged.reason_code == "NO_NEW_RECIPE_EVIDENCE"
    assert d_unchanged.confirmations_after == 1
    assert state_unchanged == state

    # A second independent fold expansion supporting the same exact recipe switches.
    ridge4 = {**ridge3, "robust_score": 0.08, "fold_count": 4,
              "fold_ids": (*ridge3["fold_ids"], "WF_003_2020-04-01")}
    hgb4 = {**hgb3, "robust_score": 0.15, "fold_count": 4,
            "fold_ids": (*hgb3["fold_ids"], "WF_003_2020-04-01")}
    state, d2 = _step(state, at="2020-05-31", winner=hgb4, choices=(ridge4, hgb4))
    assert d2.reason_code == "HYSTERESIS_SWITCH_CONFIRMED"
    assert d2.switch_allowed
    assert state.incumbent == recipe_key(hgb4)
    assert state.switch_count == 1

    # A tie-break-only challenger cannot dislodge the incumbent.
    ridge5 = {**ridge4, "robust_score": 0.15, "fold_count": 5,
              "fold_ids": (*ridge4["fold_ids"], "WF_004_2020-05-01")}
    hgb5 = {**hgb4, "robust_score": 0.15, "fold_count": 5,
            "fold_ids": (*hgb4["fold_ids"], "WF_004_2020-05-01")}
    state, d3 = _step(state, at="2020-06-30", winner=ridge5, choices=(ridge5, hgb5))
    assert d3.reason_code == "NO_STRICT_ROBUST_SCORE_IMPROVEMENT"
    assert state.incumbent == recipe_key(hgb5)

    # State serialization/resume is deterministic.
    restored = state_from_dict(json.loads(json.dumps(state_to_dict(state))))
    assert restored == state
    ridge6 = {**ridge5, "robust_score": 0.17, "fold_count": 6,
              "fold_ids": (*ridge5["fold_ids"], "WF_005_2020-06-01")}
    hgb6 = {**hgb5, "robust_score": 0.16, "fold_count": 6,
            "fold_ids": (*hgb5["fold_ids"], "WF_005_2020-06-01")}
    a_state, a_decision = _step(state, at="2020-07-31", winner=ridge6, choices=(ridge6, hgb6))
    b_state, b_decision = _step(restored, at="2020-07-31", winner=ridge6, choices=(ridge6, hgb6))
    assert a_state == b_state
    assert a_decision == b_decision

    # Non-monotonic evidence is fail-closed.
    try:
        _step(a_state, at="2020-06-01", winner=hgb5, choices=(ridge5, hgb5))
    except ValueError as exc:
        assert "NON_MONOTONIC" in str(exc)
    else:
        raise AssertionError("non-monotonic matured evidence accepted")

    # One-fold evidence cannot enter the state machine.
    weak = _choice("HGB_WEAK", "HIST_GRADIENT_BOOSTING", 1.0, folds=1)
    try:
        update_recipe_hysteresis(
            state=RecipeHysteresisState(),
            assessment_date=date(2020, 8, 31),
            evidence_date=date(2020, 8, 31),
            evidence_fingerprint=stable_evidence_fingerprint(
                evidence_date=date(2020, 8, 31), choices=(weak,)
            ),
            selector_winner=weak,
            eligible_choices=(weak,),
            config=HysteresisConfig(),
        )
    except ValueError as exc:
        assert "INSUFFICIENT_OOS_FOLDS" in str(exc)
    else:
        raise AssertionError("single-fold challenger accepted")

    print("DYNAMIC_QBD_RECIPE_HYSTERESIS_SELF_TEST_PASS")


if __name__ == "__main__":
    main()
