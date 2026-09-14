"""Immutable identities for the Dynamic-QBD causal model factory."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import Any, Mapping

from .contract_fingerprints import stable_hash


class GenerationStatus(str, Enum):
    BUILDING = "BUILDING"
    VALID = "VALID"
    FAILED = "FAILED"
    RETIRED = "RETIRED"


class FactoryArm(str, Enum):
    A_FROZEN_MODEL_FROZEN_CALIBRATION = "A_FROZEN_MODEL_FROZEN_CALIBRATION"
    B_FROZEN_MODEL_ROLLING_RECALIBRATION = "B_FROZEN_MODEL_ROLLING_RECALIBRATION"
    B2_FROZEN_MODEL_ROLLING_POLICY_AND_RECALIBRATION = "B2_FROZEN_MODEL_ROLLING_POLICY_AND_RECALIBRATION"
    C_ROLLING_REFIT_ROLLING_RECALIBRATION = "C_ROLLING_REFIT_ROLLING_RECALIBRATION"
    C2_ROLLING_REFIT_ROLLING_POLICY_AND_RECALIBRATION = "C2_ROLLING_REFIT_ROLLING_POLICY_AND_RECALIBRATION"


PRIMARY_FACTORY_ARMS = (
    FactoryArm.A_FROZEN_MODEL_FROZEN_CALIBRATION,
    FactoryArm.B_FROZEN_MODEL_ROLLING_RECALIBRATION,
    FactoryArm.C_ROLLING_REFIT_ROLLING_RECALIBRATION,
)


@dataclass(frozen=True)
class ExpertFamilySpec:
    family_id: str
    horizon_sessions: int
    holding_days: int
    max_names: int
    entry_policy_rule: Mapping[str, Any]
    exit_policy: Mapping[str, Any]
    feature_schema_sha256: str
    model_family: str
    training_recipe: Mapping[str, Any]
    hyperparameter_rule: Mapping[str, Any]
    training_window_sessions: int
    calibration_window_sessions: int
    threshold_rule: Mapping[str, Any]
    refit_cadence: str
    benchmark_contract: Mapping[str, Any]
    cost_contract: Mapping[str, Any]
    tax_contract: Mapping[str, Any]
    random_seed: int

    def __post_init__(self) -> None:
        if not self.family_id or not 1 <= int(self.horizon_sessions) <= 30:
            raise ValueError("DYNAMIC_QBD_INVALID_FAMILY_ID_OR_HORIZON")
        if not 1 <= int(self.holding_days) <= int(self.horizon_sessions):
            raise ValueError("DYNAMIC_QBD_INVALID_HOLDING_DAYS")
        if int(self.max_names) not in range(1, 7):
            raise ValueError("DYNAMIC_QBD_MAX_NAMES_MUST_BE_N1_TO_N6")
        if self.training_window_sessions <= self.horizon_sessions or self.calibration_window_sessions <= 0:
            raise ValueError("DYNAMIC_QBD_INVALID_WINDOW")

    @property
    def family_hash(self) -> str:
        return stable_hash(asdict(self))


@dataclass(frozen=True)
class ModelGeneration:
    generation_id: str
    family_id: str
    refit_timestamp: datetime
    information_cutoff: date
    latest_matured_label_cutoff: date
    train_start: date
    train_end: date
    calibration_start: date
    calibration_end: date
    model_artifact_sha256: str
    dataset_fingerprint: str
    feature_schema_sha256: str
    training_recipe_fingerprint: str
    random_seed: int
    calibration_fingerprint: str
    resolved_threshold: float
    exit_policy_fingerprint: str
    validation_status: str
    lifecycle_status: GenerationStatus
    activation_date: date | None = None
    failure_reason: str | None = None
    model_artifact_id: str = ""
    selected_model_family: str = ""
    selected_hyperparameters: Mapping[str, Any] = field(default_factory=dict)
    selection_oos_fold_count: int = 0
    selection_oos_positive_fold_fraction: float = 0.0
    selection_oos_fold_ids: tuple[str, ...] = ()
    selection_metric_contract: str = ""
    model_combination_contract: str = ""
    prediction_artifact_path: str = ""
    prediction_artifact_sha256: str = ""
    resolved_score_quantile: float = 0.975
    resolved_top_fraction: float = 0.005
    entry_policy_fingerprint: str = ""
    exit_generation_id: str = ""
    exit_model_artifact_sha256: str = ""
    exit_calibration_fingerprint: str = ""
    exit_prediction_artifact_path: str = ""
    exit_prediction_artifact_sha256: str = ""
    exit_model_recipes: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.latest_matured_label_cutoff > self.information_cutoff:
            raise ValueError("GENERATION_USES_UNMATURED_LABEL")
        if self.train_end > self.latest_matured_label_cutoff or self.calibration_end > self.information_cutoff:
            raise ValueError("GENERATION_WINDOW_AFTER_INFORMATION_CUTOFF")
        if self.lifecycle_status == GenerationStatus.VALID and (
            not self.model_artifact_sha256 or not self.calibration_fingerprint or self.activation_date is None
        ):
            raise ValueError("VALID_GENERATION_MISSING_PROVENANCE")
        if self.lifecycle_status == GenerationStatus.VALID and (
            not self.model_artifact_id or not self.prediction_artifact_sha256 or not self.entry_policy_fingerprint
        ):
            raise ValueError("VALID_GENERATION_MISSING_EXECUTION_PROVENANCE")
        if self.lifecycle_status == GenerationStatus.VALID and not 0 < float(self.resolved_top_fraction) <= 1:
            raise ValueError("VALID_GENERATION_INVALID_TOP_FRACTION")
        if self.lifecycle_status == GenerationStatus.VALID and self.exit_generation_id and (
            not self.exit_model_artifact_sha256 or not self.exit_calibration_fingerprint
            or not self.exit_prediction_artifact_sha256
        ):
            raise ValueError("VALID_EXIT_GENERATION_MISSING_PROVENANCE")

    @property
    def generation_fingerprint(self) -> str:
        payload = asdict(self)
        # Historical batch prediction files contain post-refit feature rows.
        # They are replay evidence, not information available when the model
        # generation was created, and therefore cannot define its identity.
        for key in ("prediction_artifact_path", "prediction_artifact_sha256",
                    "exit_prediction_artifact_path", "exit_prediction_artifact_sha256"):
            payload.pop(key, None)
        return stable_hash(payload)


@dataclass(frozen=True)
class CalibrationRecord:
    family_id: str
    generation_id: str
    information_cutoff: date
    calibration_start: date
    calibration_end: date
    latest_terminal_date_used: date | None
    score_quantile: float
    resolved_threshold: float
    observations: int
    calibration_fingerprint: str
    resolved_top_fraction: float = 0.005
    active_dates: int = 0
    historical_month_count: int = 0
    historical_positive_month_fraction: float = 0.0
    historical_median_month_excess: float = 0.0
    historical_q25_month_excess: float = 0.0


@dataclass(frozen=True)
class PositionLineage:
    family_id: str
    generation_id: str
    entry_policy_id: str
    exit_policy_id: str
    entry_model_artifact_id: str = ""
    exit_generation_id: str = ""


@dataclass(frozen=True)
class DynamicFactoryState:
    schema_version: str = "DYNAMIC_QBD_FACTORY_STATE_V1"
    as_of: date = date.min
    family_registry_hash: str = ""
    current_generation_by_family: Mapping[str, str] = field(default_factory=dict)
    refit_history_cursor: str = ""
    calibration_history_cursor: str = ""
    family_evidence_cursor: str = ""
    open_position_lineage: Mapping[str, PositionLineage] = field(default_factory=dict)
