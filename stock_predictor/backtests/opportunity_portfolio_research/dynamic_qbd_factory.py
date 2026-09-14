"""Causal refit scheduler and generation lifecycle orchestrator."""
from __future__ import annotations

from dataclasses import replace
from datetime import date, datetime
from typing import Protocol

import pandas as pd

from .contract_fingerprints import stable_hash
from .dynamic_qbd_generation_contracts import ExpertFamilySpec, GenerationStatus, ModelGeneration
from .dynamic_qbd_maturity import HorizonMaturityResolver
from .dynamic_qbd_generation_recalibration import recalibrate_generation
from .dynamic_qbd_generation_registry import GenerationRegistry


class GenerationBuilder(Protocol):
    def build(self, *, family: ExpertFamilySpec, information_cutoff: date,
              latest_matured_label_cutoff: date, prediction_end: date | None = None,
              prediction_end_exclusive: date | None = None) -> dict: ...
    def calibration_predictions(self, *, family: ExpertFamilySpec, build: dict, information_cutoff: date) -> pd.DataFrame: ...
    def finalize_generation(self, *, family: ExpertFamilySpec, build: dict, generation_id: str) -> dict: ...


def monthly_refit_dates(sessions, *, day: str = "MONTH_END") -> tuple[date, ...]:
    values = pd.Series(pd.to_datetime(tuple(sessions))).drop_duplicates().sort_values()
    if values.empty:
        return ()
    grouped = values.groupby(values.dt.to_period("M"))
    if day == "MONTH_START":
        return tuple(pd.Timestamp(group.iloc[0]).date() for _, group in grouped)
    return tuple(pd.Timestamp(group.iloc[-1]).date() for _, group in grouped)


def deterministic_generation_id(family: ExpertFamilySpec, cutoff: date, build: dict) -> str:
    return stable_hash({
        "family_hash": family.family_hash,
        "information_cutoff": cutoff,
        "dataset_fingerprint": build["dataset_fingerprint"],
        "model_artifact_sha256": build["model_artifact_sha256"],
        "training_recipe_fingerprint": build["training_recipe_fingerprint"],
        "seed": family.random_seed,
    })[:24]


def refit_family(
    *, family: ExpertFamilySpec, information_cutoff: date, maturity: HorizonMaturityResolver,
    builder: GenerationBuilder, registry: GenerationRegistry | None,
    prediction_end: date | None = None, prediction_end_exclusive: date | None = None,
) -> ModelGeneration:
    latest = maturity.latest_matured_decision(information_cutoff, family.horizon_sessions)
    if latest is None:
        raise ValueError("INSUFFICIENT_MATURED_HISTORY_FOR_REFIT")
    try:
        build_kwargs = {}
        if prediction_end is not None:
            build_kwargs["prediction_end"] = prediction_end
        if prediction_end_exclusive is not None:
            build_kwargs["prediction_end_exclusive"] = prediction_end_exclusive
        build = builder.build(family=family, information_cutoff=information_cutoff,
                              latest_matured_label_cutoff=latest, **build_kwargs)
        generation_id = deterministic_generation_id(family, information_cutoff, build)
        predictions = builder.calibration_predictions(family=family, build=build, information_cutoff=information_cutoff)
        calibration = recalibrate_generation(family, generation_id, predictions, information_cutoff=information_cutoff, maturity=maturity)
        finalized = builder.finalize_generation(family=family, build=build, generation_id=generation_id)
        generation = ModelGeneration(
            generation_id=generation_id, family_id=family.family_id,
            refit_timestamp=datetime.combine(information_cutoff, datetime.min.time()),
            information_cutoff=information_cutoff, latest_matured_label_cutoff=latest,
            train_start=pd.Timestamp(build["train_start"]).date(), train_end=pd.Timestamp(build["train_end"]).date(),
            calibration_start=calibration.calibration_start, calibration_end=calibration.calibration_end,
            model_artifact_sha256=build["model_artifact_sha256"], dataset_fingerprint=build["dataset_fingerprint"],
            feature_schema_sha256=family.feature_schema_sha256,
            training_recipe_fingerprint=build["training_recipe_fingerprint"], random_seed=family.random_seed,
            calibration_fingerprint=calibration.calibration_fingerprint,
            resolved_threshold=calibration.resolved_threshold, exit_policy_fingerprint=stable_hash(family.exit_policy),
            validation_status="CAUSAL_FACTORY_VALIDATED", lifecycle_status=GenerationStatus.VALID,
            activation_date=information_cutoff, model_artifact_id=str(build["model_artifact_id"]),
            selected_model_family=str(build["selected_model_family"]),
            selected_hyperparameters=dict(build["selected_hyperparameters"]),
            selection_oos_fold_count=int(build.get("selection_evidence",{}).get("fold_count",0)),
            selection_oos_positive_fold_fraction=float(1.0-build.get("selection_evidence",{}).get("negative_fold_rate",1.0)),
            selection_oos_fold_ids=tuple(build.get("selection_evidence",{}).get("fold_ids",())),
            selection_metric_contract=str(build.get("selection_evidence",{}).get("selection_metric_contract","")),
            model_combination_contract=str(build.get("selection_evidence",{}).get("model_combination_contract","")),
            prediction_artifact_path=str(finalized["prediction_artifact_path"]),
            prediction_artifact_sha256=str(finalized["prediction_artifact_sha256"]),
            resolved_score_quantile=calibration.score_quantile,
            resolved_top_fraction=calibration.resolved_top_fraction,
            entry_policy_fingerprint=stable_hash({"quantile": calibration.score_quantile,
                                                   "top_fraction": calibration.resolved_top_fraction,
                                                   "threshold": calibration.resolved_threshold}),
            exit_generation_id=str(finalized.get("exit_generation_id", "")),
            exit_model_artifact_sha256=str(finalized.get("exit_model_artifact_sha256", "")),
            exit_calibration_fingerprint=str(finalized.get("exit_calibration_fingerprint", "")),
            exit_prediction_artifact_path=str(finalized.get("exit_prediction_artifact_path", "")),
            exit_prediction_artifact_sha256=str(finalized.get("exit_prediction_artifact_sha256", "")),
            exit_model_recipes=dict(build.get("exit_model_recipes", {})),
        )
    except Exception as exc:
        failed_id = stable_hash((family.family_hash, information_cutoff, type(exc).__name__, str(exc)))[:24]
        prior = registry.current_generation(family.family_id) if registry is not None else None
        fallback_date = latest
        generation = ModelGeneration(
            generation_id=failed_id, family_id=family.family_id,
            refit_timestamp=datetime.combine(information_cutoff, datetime.min.time()), information_cutoff=information_cutoff,
            latest_matured_label_cutoff=latest, train_start=fallback_date, train_end=fallback_date,
            calibration_start=fallback_date, calibration_end=fallback_date, model_artifact_sha256="",
            dataset_fingerprint="", feature_schema_sha256=family.feature_schema_sha256,
            training_recipe_fingerprint=stable_hash(family.training_recipe), random_seed=family.random_seed,
            calibration_fingerprint="", resolved_threshold=0.0, exit_policy_fingerprint=stable_hash(family.exit_policy),
            validation_status="REFIT_FAILED_PRIOR_VALID_RETAINED" if prior else "REFIT_FAILED_NO_VALID_FALLBACK",
            lifecycle_status=GenerationStatus.FAILED, activation_date=None,
            failure_reason=f"{type(exc).__name__}:{exc}",
        )
    if registry is not None:
        registry.register(generation)
    return generation


def run_refit_schedule(*, families, refit_dates, maturity, builder, registry):
    history = []
    for cutoff in refit_dates:
        for family in families:
            history.append(refit_family(family=family, information_cutoff=cutoff, maturity=maturity, builder=builder, registry=registry))
    return tuple(history)
