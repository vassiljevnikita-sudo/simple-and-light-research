"""Canonical contracts shared by Candidate-OOS and production generation."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping

from .contract_fingerprints import stable_hash


@dataclass(frozen=True)
class FoldPolicy:
    training_window_sessions: int = 504
    validation_window_sessions: int = 126
    calibration_window_sessions: int = 252
    step_sessions: int = 126
    purge_sessions: int = 30
    embargo_sessions: int = 0
    outer_fold_definition: str = "EXPANDING_TRAIN_VALIDATION_PURGED"
    maturity_policy: str = "TERMINAL_DATE_LE_INFORMATION_AVAILABLE_AT"

    def __post_init__(self) -> None:
        if min(self.training_window_sessions, self.validation_window_sessions,
               self.calibration_window_sessions, self.step_sessions) <= 0:
            raise ValueError("QBD_FOLD_POLICY_NONPOSITIVE_WINDOW")
        if min(self.purge_sessions, self.embargo_sessions) < 0:
            raise ValueError("QBD_FOLD_POLICY_NEGATIVE_PURGE_OR_EMBARGO")

    @property
    def fold_policy_hash(self) -> str:
        return stable_hash(asdict(self))


@dataclass(frozen=True)
class TargetContract:
    horizon_semantics: str = "FORWARD_SESSION_RETURN"
    entry_execution: str = "DECISION_CLOSE_TO_NEXT_OPEN"
    exit_execution: str = "HORIZON_NEXT_OPEN"
    benchmark: str = "URTH"
    benchmark_return_definition: str = "BENCHMARK_FORWARD_RETURN_COLUMN_REQUIRED"
    cost_model: Mapping[str, Any] = None
    target_column_template: str = "net_excess_return_{horizon}__BASELINE_20_BPS"
    benchmark_column_template: str = "benchmark_forward_return_{horizon}"
    excess_return_definition: str = "NET_STOCK_RETURN_MINUS_BENCHMARK;COST_INCLUDED_IN_NET_STOCK_RETURN"
    algebra_version: str = "QBD_EXPLICIT_RETURN_ALGEBRA_V1"
    maturity_semantics: str = "DECISION_LT_TERMINAL_LE_INFORMATION_AVAILABLE"

    def __post_init__(self) -> None:
        if self.cost_model is None:
            object.__setattr__(self, "cost_model", {"roundtrip_bps": 20.0})
        if not self.benchmark or "REQUIRED" not in self.benchmark_return_definition:
            raise ValueError("QBD_TARGET_CONTRACT_BENCHMARK_NOT_REQUIRED")

    @property
    def target_contract_hash(self) -> str:
        return stable_hash(asdict(self))

    def target_column(self, horizon: int) -> str:
        return self.target_column_template.format(horizon=int(horizon))

    def benchmark_column(self, horizon: int) -> str:
        return self.benchmark_column_template.format(horizon=int(horizon))

    def validate_target_cost(self) -> None:
        if "BASELINE_20_BPS" in self.target_column_template and float(self.cost_model.get("roundtrip_bps", -1)) != 20.0:
            raise ValueError("QBD_TARGET_COST_CONTRACT_MISMATCH_BASELINE_20_BPS")


@dataclass(frozen=True)
class RecipeSelectionPolicy:
    primary_metric: str = "FOLD_SPEARMAN_PREDICTION_VS_REALIZED_EXCESS"
    aggregation: str = "MEDIAN_FOLD_SPEARMAN"
    tie_breakers: tuple[str, ...] = ("MEAN_FOLD_SPEARMAN", "POSITIVE_FOLD_FRACTION", "LOWER_MEAN_MAE", "CANDIDATE_ID")
    minimum_fold_observations: int = 2
    # A single mature fold is retained as diagnostic evidence only.  Primary
    # recipe authority starts at two independent folds.
    minimum_candidate_folds: int = 2
    version: str = "QBD_RECIPE_SELECTION_POLICY_V1"

    @property
    def recipe_selection_policy_hash(self) -> str:
        return stable_hash(asdict(self))


@dataclass(frozen=True)
class ModelTrainingContract:
    """Identity of the actual model-building implementation and seed."""

    implementation: str = "stock_predictor.v5.train"
    implementation_version: str = "V5_TRAINING_PIPELINE_IMPLEMENTED"
    random_seed: int = 17
    model_builder: str = "V5_BUILD_BUNDLE_FIT"
    primary_candidate_universe_hash: str = ""
    deterministic_runtime: str = "SINGLE_PROCESS_DETERMINISTIC"

    def __post_init__(self) -> None:
        if int(self.random_seed) < 0:
            raise ValueError("QBD_MODEL_TRAINING_SEED_INVALID")

    @property
    def model_training_contract_hash(self) -> str:
        return stable_hash(asdict(self))


@dataclass(frozen=True)
class ResolvedContracts:
    fold_policy: FoldPolicy
    target_contract: TargetContract
    recipe_selection_policy: RecipeSelectionPolicy
    model_training_contract: ModelTrainingContract


def frozen_primary_candidate_registry(records: Iterable[Mapping[str, Any]], *, source_contract: str) -> dict:
    """Build the explicit primary universe; unrelated hyperparameter entries are ignored."""
    allowed = {"RIDGE_LOGISTIC", "HIST_GRADIENT_BOOSTING"}
    selected = []
    for record in records:
        family = str(record.get("recipe_family", record.get("family", "")))
        if family not in allowed:
            continue
        params = record.get("hyperparameters", record.get("parameters", {}))
        selected.append({"candidate_id": str(record["candidate_id"]), "recipe_family": family,
                         "hyperparameters": dict(params), "candidate_spec_hash": str(record["candidate_spec_hash"])})
    selected.sort(key=lambda x: x["candidate_id"])
    payload = {"schema_version": "QBD_PRIMARY_CANDIDATE_CONTRACT_V1", "source_contract": source_contract,
               "records": selected}
    payload["primary_candidate_universe_hash"] = stable_hash(payload)
    return payload
