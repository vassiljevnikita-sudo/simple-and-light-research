"""Immutable contracts for the causal Dynamic-QBD model-store experiment."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, Mapping

from .contract_fingerprints import stable_hash


DEVELOPMENT_END = date(2025, 12, 31)
HOLDOUT_BOUNDARY = date(2026, 7, 25)
EXPERIMENT_ID = "DQBD_CAUSAL_MODEL_STORE_V1"
GenerationTrigger = Literal["NEW_MATURED_FOLD"]
DecisionAction = Literal["KEEP", "ACTIVATE_NEW", "SWITCH_RECIPE"]
PseudoLiveGenerationTrigger = Literal["NEW_MATURED_FOLD_STATE"]


def _as_date(value: date | str) -> date:
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value))


def _require_text(name: str, value: str) -> str:
    value = str(value)
    if not value:
        raise ValueError(f"{name}_REQUIRED")
    return value


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (tuple, list)):
        return [_plain(v) for v in value]
    if isinstance(value, date):
        return value.isoformat()
    return value


def _freeze_mapping(value: Mapping[str, Any] | None) -> Mapping[str, Any]:
    return MappingProxyType(dict(_plain(value or {})))


@dataclass(frozen=True)
class ModelGenerationRecord:
    """One append-only generation record; a later generation cannot overwrite it."""

    generation_id: str
    recipe_id: str
    horizon: int
    created_at: date
    training_start: date
    training_end: date
    target_maturity_cutoff: date
    model_artifact_id: str
    model_hash: str
    calibration_id: str
    evidence_fingerprint_at_creation: str
    source_generation_id: str = ""
    source_generation_fingerprint: str = ""
    source_generation_provenance: Mapping[str, Any] = None

    def __post_init__(self) -> None:
        for name in ("generation_id", "recipe_id", "model_artifact_id", "model_hash",
                     "calibration_id", "evidence_fingerprint_at_creation"):
            _require_text(name.upper(), getattr(self, name))
        if not 1 <= int(self.horizon) <= 30:
            raise ValueError("MODEL_GENERATION_HORIZON_OUT_OF_RANGE")
        for name in ("created_at", "training_start", "training_end", "target_maturity_cutoff"):
            object.__setattr__(self, name, _as_date(getattr(self, name)))
        if self.training_start > self.training_end:
            raise ValueError("MODEL_GENERATION_TRAINING_WINDOW_INVALID")
        if self.training_end > self.target_maturity_cutoff:
            raise ValueError("MODEL_GENERATION_TRAINING_AFTER_MATURITY_CUTOFF")
        if self.target_maturity_cutoff > self.created_at:
            raise ValueError("MODEL_GENERATION_CREATED_BEFORE_MATURITY")
        if self.created_at > DEVELOPMENT_END:
            raise ValueError("MODEL_GENERATION_PROSPECTIVE_HOLDOUT_CLOSED")
        object.__setattr__(self, "source_generation_provenance",
                           _freeze_mapping(self.source_generation_provenance))

    @property
    def fingerprint(self) -> str:
        return stable_hash(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "generation_id": self.generation_id,
            "recipe_id": self.recipe_id,
            "horizon": int(self.horizon),
            "created_at": self.created_at.isoformat(),
            "training_start": self.training_start.isoformat(),
            "training_end": self.training_end.isoformat(),
            "target_maturity_cutoff": self.target_maturity_cutoff.isoformat(),
            "model_artifact_id": self.model_artifact_id,
            "model_hash": self.model_hash,
            "calibration_id": self.calibration_id,
            "evidence_fingerprint_at_creation": self.evidence_fingerprint_at_creation,
            "source_generation_id": self.source_generation_id,
            "source_generation_fingerprint": self.source_generation_fingerprint,
            "source_generation_provenance": _plain(self.source_generation_provenance),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ModelGenerationRecord":
        return cls(**dict(value))


@dataclass(frozen=True)
class OrchestratorDecision:
    decision_time: date
    incumbent_generation_id: str
    candidate_generation_ids: tuple[str, ...]
    selected_generation_id: str
    action: DecisionAction
    evidence_fingerprint: str
    decision_contract_hash: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "decision_time", _as_date(self.decision_time))
        candidates = tuple(str(x) for x in self.candidate_generation_ids)
        object.__setattr__(self, "candidate_generation_ids", candidates)
        _require_text("INCUMBENT_GENERATION_ID", self.incumbent_generation_id)
        _require_text("SELECTED_GENERATION_ID", self.selected_generation_id)
        _require_text("EVIDENCE_FINGERPRINT", self.evidence_fingerprint)
        _require_text("DECISION_CONTRACT_HASH", self.decision_contract_hash)
        if self.action not in ("KEEP", "ACTIVATE_NEW", "SWITCH_RECIPE"):
            raise ValueError("ORCHESTRATOR_ACTION_INVALID")
        if len(candidates) != len(set(candidates)):
            raise ValueError("ORCHESTRATOR_CANDIDATES_NOT_UNIQUE")
        if self.selected_generation_id not in (self.incumbent_generation_id, *candidates):
            raise ValueError("ORCHESTRATOR_SELECTED_GENERATION_NOT_VISIBLE")
        if self.action == "KEEP" and self.selected_generation_id != self.incumbent_generation_id:
            raise ValueError("ORCHESTRATOR_KEEP_MUST_SELECT_INCUMBENT")
        if self.action in ("ACTIVATE_NEW", "SWITCH_RECIPE") and self.selected_generation_id == self.incumbent_generation_id:
            raise ValueError("ORCHESTRATOR_ACTIVATION_MUST_CHANGE_GENERATION")
        if self.decision_time > DEVELOPMENT_END:
            raise ValueError("ORCHESTRATOR_PROSPECTIVE_HOLDOUT_CLOSED")

    @property
    def fingerprint(self) -> str:
        return stable_hash(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision_time": self.decision_time.isoformat(),
            "incumbent_generation_id": self.incumbent_generation_id,
            "candidate_generation_ids": list(self.candidate_generation_ids),
            "selected_generation_id": self.selected_generation_id,
            "action": self.action,
            "evidence_fingerprint": self.evidence_fingerprint,
            "decision_contract_hash": self.decision_contract_hash,
        }


@dataclass(frozen=True)
class RunContract:
    """Hashable experiment definition persisted before any performance run."""

    baseline_commit: str
    recipe_universe: str
    seed: Mapping[str, Any]
    generation_trigger: GenerationTrigger = "NEW_MATURED_FOLD"
    training_window: str = "EXPANDING"
    calibration: str = "GENERATION_SPECIFIC"
    experiment: str = EXPERIMENT_ID
    development_end: date = DEVELOPMENT_END
    holdout_boundary: date = HOLDOUT_BOUNDARY
    portfolio: Mapping[str, Any] = None

    def __post_init__(self) -> None:
        _require_text("BASELINE_COMMIT", self.baseline_commit)
        _require_text("RECIPE_UNIVERSE", self.recipe_universe)
        object.__setattr__(self, "seed", _freeze_mapping(self.seed))
        object.__setattr__(self, "portfolio", _freeze_mapping(self.portfolio or {
            "exit_mode": "FIXED",
            "allocation": "EQUAL_ACTIVE",
            "replacement": "IGNORE_NEW",
        }))
        object.__setattr__(self, "development_end", _as_date(self.development_end))
        object.__setattr__(self, "holdout_boundary", _as_date(self.holdout_boundary))
        if self.development_end != DEVELOPMENT_END or self.holdout_boundary != HOLDOUT_BOUNDARY:
            raise ValueError("DQBD_RUN_CONTRACT_BOUNDARY_MISMATCH")
        if self.development_end >= self.holdout_boundary:
            raise ValueError("DQBD_RUN_CONTRACT_BOUNDARIES_INVALID")

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment": self.experiment,
            "baseline_commit": self.baseline_commit,
            "development_end": self.development_end.isoformat(),
            "holdout_boundary": self.holdout_boundary.isoformat(),
            "seed": _plain(self.seed),
            "recipe_universe": self.recipe_universe,
            "generation_trigger": self.generation_trigger,
            "training_window": self.training_window,
            "calibration": self.calibration,
            "portfolio": _plain(self.portfolio),
        }

    @property
    def run_contract_hash(self) -> str:
        return stable_hash(self.to_dict())

    def write(self, path: str | Path) -> str:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = self.to_dict() | {"run_contract_hash": self.run_contract_hash}
        if destination.is_file():
            existing = json.loads(destination.read_text(encoding="utf-8"))
            if existing.get("run_contract_hash") != self.run_contract_hash:
                raise ValueError("DQBD_RUN_CONTRACT_IMMUTABLE")
            return self.run_contract_hash
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(destination)
        return self.run_contract_hash


@dataclass(frozen=True)
class PseudoLiveExperimentContract:
    """The frozen, machine-readable contract required before Step 4 runs."""

    baseline_commit: str
    seeds: Mapping[str, date]
    expected_fold_states: Mapping[str, int]
    recipe_universe: Mapping[str, Any]
    recipe_eligibility: Mapping[str, Any]
    time_semantics: Mapping[str, str]
    portfolio: Mapping[str, Any]
    experiment_version: str = EXPERIMENT_ID
    development_end: date = DEVELOPMENT_END
    prospective_holdout_start: date = HOLDOUT_BOUNDARY
    generation_trigger: PseudoLiveGenerationTrigger = "NEW_MATURED_FOLD_STATE"
    training_window: str = "EXPANDING"
    calibration: str = "GENERATION_SPECIFIC"

    def __post_init__(self) -> None:
        _require_text("BASELINE_COMMIT", self.baseline_commit)
        if self.experiment_version != EXPERIMENT_ID:
            raise ValueError("DQBD_EXPERIMENT_VERSION_INVALID")
        if self.generation_trigger != "NEW_MATURED_FOLD_STATE":
            raise ValueError("DQBD_GENERATION_TRIGGER_INVALID")
        if self.training_window != "EXPANDING":
            raise ValueError("DQBD_TRAINING_WINDOW_MUST_BE_EXPANDING")
        if self.calibration != "GENERATION_SPECIFIC":
            raise ValueError("DQBD_CALIBRATION_MUST_BE_GENERATION_SPECIFIC")
        object.__setattr__(self, "development_end", _as_date(self.development_end))
        object.__setattr__(self, "prospective_holdout_start", _as_date(self.prospective_holdout_start))
        if self.development_end != DEVELOPMENT_END or self.prospective_holdout_start != HOLDOUT_BOUNDARY:
            raise ValueError("DQBD_PSEUDOLIVE_BOUNDARY_MISMATCH")
        if self.development_end >= self.prospective_holdout_start:
            raise ValueError("DQBD_PSEUDOLIVE_BOUNDARIES_INVALID")
        expected_seeds = {"short", "primary", "long"}
        if set(self.seeds) != expected_seeds:
            raise ValueError("DQBD_SEED_KEYS_MUST_BE_SHORT_PRIMARY_LONG")
        seed_values = {key: _as_date(value) for key, value in self.seeds.items()}
        if not seed_values["short"] < seed_values["primary"] < seed_values["long"] <= self.development_end:
            raise ValueError("DQBD_SEED_ORDER_INVALID")
        object.__setattr__(self, "seeds", MappingProxyType(seed_values))
        expected_fold_states = {str(key): int(value) for key, value in self.expected_fold_states.items()}
        if expected_fold_states != {"short": 2, "primary": 3, "long": 4}:
            raise ValueError("DQBD_SEED_FOLD_STATE_EXPECTATIONS_INVALID")
        object.__setattr__(self, "expected_fold_states", MappingProxyType(expected_fold_states))
        object.__setattr__(self, "recipe_universe", _freeze_mapping(self.recipe_universe))
        object.__setattr__(self, "recipe_eligibility", _freeze_mapping(self.recipe_eligibility))
        object.__setattr__(self, "time_semantics", _freeze_mapping(self.time_semantics))
        object.__setattr__(self, "portfolio", _freeze_mapping(self.portfolio))
        required_time_fields = {
            "observation_date", "prediction_date", "target_terminal_date", "matured_at",
            "generation_created_at", "decision_at", "execution_at",
        }
        if set(self.time_semantics) != required_time_fields:
            raise ValueError("DQBD_TIME_SEMANTICS_INCOMPLETE")
        required_portfolio = {"exit", "allocation", "replacement", "execution", "benchmark", "transaction_costs", "sleeve_semantics"}
        if not required_portfolio <= set(self.portfolio):
            raise ValueError("DQBD_PORTFOLIO_CONTRACT_INCOMPLETE")

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline_commit": self.baseline_commit,
            "experiment_version": self.experiment_version,
            "development_end": self.development_end.isoformat(),
            "prospective_holdout_start": self.prospective_holdout_start.isoformat(),
            "seeds": {key: value.isoformat() for key, value in sorted(self.seeds.items())},
            "expected_fold_states": dict(sorted(self.expected_fold_states.items())),
            "training_window": self.training_window,
            "generation_trigger": self.generation_trigger,
            "calibration": self.calibration,
            "recipe_universe": _plain(self.recipe_universe),
            "recipe_eligibility": _plain(self.recipe_eligibility),
            "time_semantics": _plain(self.time_semantics),
            "portfolio": _plain(self.portfolio),
        }

    @property
    def contract_hash(self) -> str:
        return stable_hash(self.to_dict())

    @property
    def run_contract_hash(self) -> str:
        return self.contract_hash

    def write(self, path: str | Path) -> str:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = self.to_dict() | {"contract_hash": self.contract_hash}
        if destination.is_file():
            existing = json.loads(destination.read_text(encoding="utf-8"))
            if existing.get("contract_hash") != self.contract_hash:
                raise ValueError("DQBD_PSEUDOLIVE_CONTRACT_IMMUTABLE")
            return self.contract_hash
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(destination)
        return self.contract_hash

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PseudoLiveExperimentContract":
        payload = dict(value)
        payload.pop("contract_hash", None)
        return cls(**payload)


def default_pseudolive_experiment_contract(*, baseline_commit: str) -> PseudoLiveExperimentContract:
    """Return the predeclared Step-3 contract; no performance data is consulted."""
    return PseudoLiveExperimentContract(
        baseline_commit=baseline_commit,
        seeds={
            "short": date(2020, 8, 31),
            "primary": date(2021, 2, 26),
            "long": date(2021, 8, 31),
        },
        expected_fold_states={"short": 2, "primary": 3, "long": 4},
        recipe_universe={
            "source": "existing_dynamic_qbd_factory_contract",
            "candidate_source": "candidate_oos.load_primary_candidate_registry",
            "recipe_families": ["RIDGE_LOGISTIC", "HIST_GRADIENT_BOOSTING"],
            "horizons": "H1-H30",
            "selection_rule": "ALL_STRUCTURALLY_ELIGIBLE_RECIPES",
            "performance_based_exclusion": False,
        },
        recipe_eligibility={
            "required_features_available": True,
            "minimum_training_sessions": 504,
            "minimum_calibration_sessions": 252,
            "minimum_matured_candidate_folds": 2,
            "minimum_fold_observations": 2,
            "calibration_feasible": True,
            "finite_model_fit": True,
            "identity_and_provenance_complete": True,
            "allowed_exclusion": "PREDECLARED_TECHNICAL_OR_CAUSAL_UNTRAINABILITY_ONLY",
            "forbidden_inputs": ["CAGR", "SHARPE", "FUTURE_RANK", "POST_SEED_PERFORMANCE"],
        },
        time_semantics={
            "observation_date": "source observation/session date",
            "prediction_date": "causal model prediction/decision date",
            "target_terminal_date": "horizon-specific target terminal date",
            "matured_at": "date on which target/evidence becomes observable",
            "generation_created_at": "append-only ModelStore publication date",
            "decision_at": "orchestrator point-in-time decision date",
            "execution_at": "next-open portfolio execution date",
        },
        portfolio={
            "exit": "FIXED",
            "allocation": "EQUAL_ACTIVE",
            "replacement": "IGNORE_NEW",
            "execution": "NEXT_OPEN",
            "benchmark": "MSCI_WORLD_IMPLEMENTABLE_PROXY_V1",
            "transaction_costs": "EXISTING_CANONICAL_COST_CONTRACT",
            "sleeve_semantics": "UNCHANGED",
            "learned_exit": False,
        },
    )
