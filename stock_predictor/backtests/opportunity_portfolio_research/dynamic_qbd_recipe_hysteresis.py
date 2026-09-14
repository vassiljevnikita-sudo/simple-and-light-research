"""Causal recipe-switch hysteresis for Dynamic-QBD shadow research.

The state machine consumes already-causal recipe assessments. A challenger vote
may advance only when the underlying recipe evidence changes. Calendar progress
without a new OOS-fold evidence set is explicitly non-evidence.

No live/capital authority is granted by this state machine.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
import hashlib
import json
from typing import Any, Mapping, Sequence


RecipeKey = tuple[str, str, str]


@dataclass(frozen=True)
class HysteresisConfig:
    confirmations_required: int = 2
    minimum_oos_folds: int = 2
    require_strict_robust_score_improvement: bool = True

    def __post_init__(self) -> None:
        if self.confirmations_required < 2:
            raise ValueError("HYSTERESIS_CONFIRMATIONS_MUST_BE_AT_LEAST_TWO")
        if self.minimum_oos_folds < 2:
            raise ValueError("HYSTERESIS_MINIMUM_OOS_FOLDS_MUST_BE_AT_LEAST_TWO")


@dataclass(frozen=True)
class RecipeHysteresisState:
    incumbent: RecipeKey | None = None
    pending_challenger: RecipeKey | None = None
    pending_confirmations: int = 0
    last_evidence_date: date | None = None
    last_evidence_fingerprint: str | None = None
    switch_count: int = 0
    last_switch_assessment: date | None = None


@dataclass(frozen=True)
class RecipeHysteresisDecision:
    assessment_date: date
    evidence_date: date
    evidence_fingerprint: str
    selector_winner: RecipeKey
    incumbent_before: RecipeKey | None
    incumbent_after: RecipeKey
    challenger: RecipeKey | None
    incumbent_robust_score: float | None
    challenger_robust_score: float | None
    robust_score_margin: float | None
    confirmations_before: int
    confirmations_after: int
    switch_allowed: bool
    reason_code: str


def _json_normalize(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _json_normalize(value[k]) for k in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        return [_json_normalize(x) for x in value]
    if isinstance(value, date):
        return value.isoformat()
    return value


def stable_evidence_fingerprint(*, evidence_date: date, choices: Sequence[Mapping[str, Any]]) -> str:
    """Hash recipe evidence independently of the assessment/maturity date.

    ``evidence_date`` remains in the signature for backwards-compatible call
    sites and audit readability, but is deliberately excluded from the hash.
    A new month with the same eligible recipes, fold ids and robust scores is
    therefore the same evidence and cannot supply another confirmation.
    """
    del evidence_date
    payload = {
        "schema_version": "DYNAMIC_QBD_RECIPE_EVIDENCE_FINGERPRINT_V2_FOLD_CONTENT",
        "choices": [
            {
                "candidate_id": str(x["candidate_id"]),
                "family": str(x["family"]),
                "parameters": _json_normalize(dict(x.get("parameters", {}))),
                "fold_count": int(x["fold_count"]),
                "fold_ids": tuple(sorted(str(v) for v in x.get("fold_ids", ()))),
                "robust_score": float(x["robust_score"]),
            }
            for x in choices
        ],
    }
    payload["choices"].sort(
        key=lambda x: (x["candidate_id"], x["family"], json.dumps(x["parameters"], sort_keys=True))
    )
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def recipe_key(choice: Mapping[str, Any]) -> RecipeKey:
    params = json.dumps(
        _json_normalize(dict(choice.get("parameters", {}))),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return (
        str(choice["candidate_id"]),
        str(choice["family"]),
        hashlib.sha256(params.encode("utf-8")).hexdigest(),
    )


def _choice_by_key(choices: Sequence[Mapping[str, Any]], key: RecipeKey) -> Mapping[str, Any] | None:
    return next((x for x in choices if recipe_key(x) == key), None)


def _ignored_evidence_decision(
    *,
    state: RecipeHysteresisState,
    assessment_date: date,
    evidence_date: date,
    evidence_fingerprint: str,
    winner_key: RecipeKey,
    incumbent_choice: Mapping[str, Any],
    winner_current: Mapping[str, Any],
    reason_code: str,
) -> tuple[RecipeHysteresisState, RecipeHysteresisDecision]:
    return state, RecipeHysteresisDecision(
        assessment_date=assessment_date,
        evidence_date=evidence_date,
        evidence_fingerprint=evidence_fingerprint,
        selector_winner=winner_key,
        incumbent_before=state.incumbent,
        incumbent_after=state.incumbent,
        challenger=state.pending_challenger,
        incumbent_robust_score=float(incumbent_choice["robust_score"]),
        challenger_robust_score=(
            float(winner_current["robust_score"]) if winner_key != state.incumbent else None
        ),
        robust_score_margin=(
            float(winner_current["robust_score"]) - float(incumbent_choice["robust_score"])
            if winner_key != state.incumbent else None
        ),
        confirmations_before=state.pending_confirmations,
        confirmations_after=state.pending_confirmations,
        switch_allowed=False,
        reason_code=reason_code,
    )


def update_recipe_hysteresis(
    *,
    state: RecipeHysteresisState,
    assessment_date: date,
    evidence_date: date,
    evidence_fingerprint: str,
    selector_winner: Mapping[str, Any],
    eligible_choices: Sequence[Mapping[str, Any]],
    config: HysteresisConfig = HysteresisConfig(),
) -> tuple[RecipeHysteresisState, RecipeHysteresisDecision]:
    """Advance hysteresis using one causal assessment.

    The same exact recipe must win on ``confirmations_required`` consecutive
    *evidence-content expansions*. A later calendar date with an unchanged
    evidence fingerprint is ignored. The challenger must also beat the current
    incumbent on robust score, not merely on a downstream tie-break.
    """
    if not eligible_choices:
        raise ValueError("HYSTERESIS_ELIGIBLE_CHOICES_EMPTY")
    winner_key = recipe_key(selector_winner)
    winner_current = _choice_by_key(eligible_choices, winner_key)
    if winner_current is None:
        raise ValueError("HYSTERESIS_WINNER_NOT_IN_ELIGIBLE_CHOICES")
    if int(winner_current["fold_count"]) < config.minimum_oos_folds:
        raise ValueError("HYSTERESIS_WINNER_INSUFFICIENT_OOS_FOLDS")

    if state.last_evidence_date is not None and evidence_date < state.last_evidence_date:
        raise ValueError("HYSTERESIS_NON_MONOTONIC_MATURED_EVIDENCE")

    if state.incumbent is None:
        next_state = RecipeHysteresisState(
            incumbent=winner_key,
            last_evidence_date=evidence_date,
            last_evidence_fingerprint=evidence_fingerprint,
            switch_count=0,
            last_switch_assessment=None,
        )
        return next_state, RecipeHysteresisDecision(
            assessment_date=assessment_date,
            evidence_date=evidence_date,
            evidence_fingerprint=evidence_fingerprint,
            selector_winner=winner_key,
            incumbent_before=None,
            incumbent_after=winner_key,
            challenger=None,
            incumbent_robust_score=None,
            challenger_robust_score=None,
            robust_score_margin=None,
            confirmations_before=0,
            confirmations_after=0,
            switch_allowed=False,
            reason_code="INITIAL_CAUSAL_RECIPE",
        )

    incumbent_choice = _choice_by_key(eligible_choices, state.incumbent)
    if incumbent_choice is None:
        raise ValueError("HYSTERESIS_INCUMBENT_MISSING_FROM_CAUSAL_CHOICES")
    if int(incumbent_choice["fold_count"]) < config.minimum_oos_folds:
        raise ValueError("HYSTERESIS_INCUMBENT_INSUFFICIENT_OOS_FOLDS")

    if state.last_evidence_date == evidence_date:
        if (
            state.last_evidence_fingerprint is not None
            and evidence_fingerprint != state.last_evidence_fingerprint
        ):
            raise ValueError("HYSTERESIS_SAME_DATE_EVIDENCE_CHANGED")
        return _ignored_evidence_decision(
            state=state,
            assessment_date=assessment_date,
            evidence_date=evidence_date,
            evidence_fingerprint=evidence_fingerprint,
            winner_key=winner_key,
            incumbent_choice=incumbent_choice,
            winner_current=winner_current,
            reason_code="DUPLICATE_MATURED_EVIDENCE_IGNORED",
        )

    if (
        state.last_evidence_fingerprint is not None
        and evidence_fingerprint == state.last_evidence_fingerprint
    ):
        return _ignored_evidence_decision(
            state=state,
            assessment_date=assessment_date,
            evidence_date=evidence_date,
            evidence_fingerprint=evidence_fingerprint,
            winner_key=winner_key,
            incumbent_choice=incumbent_choice,
            winner_current=winner_current,
            reason_code="NO_NEW_RECIPE_EVIDENCE",
        )

    incumbent_score = float(incumbent_choice["robust_score"])
    winner_score = float(winner_current["robust_score"])

    if winner_key == state.incumbent:
        next_state = RecipeHysteresisState(
            incumbent=state.incumbent,
            pending_challenger=None,
            pending_confirmations=0,
            last_evidence_date=evidence_date,
            last_evidence_fingerprint=evidence_fingerprint,
            switch_count=state.switch_count,
            last_switch_assessment=state.last_switch_assessment,
        )
        return next_state, RecipeHysteresisDecision(
            assessment_date=assessment_date,
            evidence_date=evidence_date,
            evidence_fingerprint=evidence_fingerprint,
            selector_winner=winner_key,
            incumbent_before=state.incumbent,
            incumbent_after=state.incumbent,
            challenger=None,
            incumbent_robust_score=incumbent_score,
            challenger_robust_score=None,
            robust_score_margin=None,
            confirmations_before=state.pending_confirmations,
            confirmations_after=0,
            switch_allowed=False,
            reason_code="INCUMBENT_RECONFIRMED",
        )

    margin = winner_score - incumbent_score
    if config.require_strict_robust_score_improvement and margin <= 0.0:
        next_state = RecipeHysteresisState(
            incumbent=state.incumbent,
            pending_challenger=None,
            pending_confirmations=0,
            last_evidence_date=evidence_date,
            last_evidence_fingerprint=evidence_fingerprint,
            switch_count=state.switch_count,
            last_switch_assessment=state.last_switch_assessment,
        )
        return next_state, RecipeHysteresisDecision(
            assessment_date=assessment_date,
            evidence_date=evidence_date,
            evidence_fingerprint=evidence_fingerprint,
            selector_winner=winner_key,
            incumbent_before=state.incumbent,
            incumbent_after=state.incumbent,
            challenger=winner_key,
            incumbent_robust_score=incumbent_score,
            challenger_robust_score=winner_score,
            robust_score_margin=margin,
            confirmations_before=state.pending_confirmations,
            confirmations_after=0,
            switch_allowed=False,
            reason_code="NO_STRICT_ROBUST_SCORE_IMPROVEMENT",
        )

    confirmations = state.pending_confirmations + 1 if state.pending_challenger == winner_key else 1
    if confirmations >= config.confirmations_required:
        next_state = RecipeHysteresisState(
            incumbent=winner_key,
            pending_challenger=None,
            pending_confirmations=0,
            last_evidence_date=evidence_date,
            last_evidence_fingerprint=evidence_fingerprint,
            switch_count=state.switch_count + 1,
            last_switch_assessment=assessment_date,
        )
        return next_state, RecipeHysteresisDecision(
            assessment_date=assessment_date,
            evidence_date=evidence_date,
            evidence_fingerprint=evidence_fingerprint,
            selector_winner=winner_key,
            incumbent_before=state.incumbent,
            incumbent_after=winner_key,
            challenger=winner_key,
            incumbent_robust_score=incumbent_score,
            challenger_robust_score=winner_score,
            robust_score_margin=margin,
            confirmations_before=state.pending_confirmations,
            confirmations_after=0,
            switch_allowed=True,
            reason_code="HYSTERESIS_SWITCH_CONFIRMED",
        )

    next_state = RecipeHysteresisState(
        incumbent=state.incumbent,
        pending_challenger=winner_key,
        pending_confirmations=confirmations,
        last_evidence_date=evidence_date,
        last_evidence_fingerprint=evidence_fingerprint,
        switch_count=state.switch_count,
        last_switch_assessment=state.last_switch_assessment,
    )
    return next_state, RecipeHysteresisDecision(
        assessment_date=assessment_date,
        evidence_date=evidence_date,
        evidence_fingerprint=evidence_fingerprint,
        selector_winner=winner_key,
        incumbent_before=state.incumbent,
        incumbent_after=state.incumbent,
        challenger=winner_key,
        incumbent_robust_score=incumbent_score,
        challenger_robust_score=winner_score,
        robust_score_margin=margin,
        confirmations_before=state.pending_confirmations,
        confirmations_after=confirmations,
        switch_allowed=False,
        reason_code="CHALLENGER_CONFIRMATION_PENDING",
    )


def state_to_dict(state: RecipeHysteresisState) -> dict[str, Any]:
    payload = asdict(state)
    for key in ("last_evidence_date", "last_switch_assessment"):
        if payload[key] is not None:
            payload[key] = payload[key].isoformat()
    for key in ("incumbent", "pending_challenger"):
        if payload[key] is not None:
            payload[key] = list(payload[key])
    return payload


def state_from_dict(payload: Mapping[str, Any]) -> RecipeHysteresisState:
    def parse_key(value: Any) -> RecipeKey | None:
        if value is None:
            return None
        if len(value) != 3:
            raise ValueError("HYSTERESIS_RECIPE_KEY_INVALID")
        return (str(value[0]), str(value[1]), str(value[2]))

    return RecipeHysteresisState(
        incumbent=parse_key(payload.get("incumbent")),
        pending_challenger=parse_key(payload.get("pending_challenger")),
        pending_confirmations=int(payload.get("pending_confirmations", 0)),
        last_evidence_date=(date.fromisoformat(str(payload["last_evidence_date"])) if payload.get("last_evidence_date") else None),
        last_evidence_fingerprint=(str(payload["last_evidence_fingerprint"]) if payload.get("last_evidence_fingerprint") else None),
        switch_count=int(payload.get("switch_count", 0)),
        last_switch_assessment=(date.fromisoformat(str(payload["last_switch_assessment"])) if payload.get("last_switch_assessment") else None),
    )


def decision_to_dict(decision: RecipeHysteresisDecision) -> dict[str, Any]:
    payload = asdict(decision)
    payload["assessment_date"] = decision.assessment_date.isoformat()
    payload["evidence_date"] = decision.evidence_date.isoformat()
    for key in ("selector_winner", "incumbent_before", "incumbent_after", "challenger"):
        if payload[key] is not None:
            payload[key] = list(payload[key])
    return payload
