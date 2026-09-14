"""Factory-value A/B/C economic comparison."""
from __future__ import annotations

import pandas as pd

from .top10_qbd_overfitting_diagnostics import block_bootstrap_mean
from .dynamic_qbd_family_surface import structural_plateau_id_from_family_id


def build_abc_generation_schedules(generations: pd.DataFrame, calibrations: pd.DataFrame,
                                   optimized_calibrations: pd.DataFrame | None = None) -> pd.DataFrame:
    required_g={"family_id","generation_id","model_artifact_id","activation_date","resolved_threshold","entry_policy_id","exit_policy_id"}
    required_c={"family_id","activation_date","resolved_threshold"}
    if required_g-set(generations): raise ValueError(f"ABC_GENERATION_COLUMNS_MISSING:{sorted(required_g-set(generations))}")
    if required_c-set(calibrations): raise ValueError(f"ABC_CALIBRATION_COLUMNS_MISSING:{sorted(required_c-set(calibrations))}")
    rows=[]
    for family_id, fg in generations.groupby("family_id",sort=True):
        fg=fg.sort_values("activation_date"); frozen=fg.iloc[0]
        base={"family_id":family_id,"entry_policy_id":frozen.entry_policy_id,"exit_policy_id":frozen.exit_policy_id,
              "model_artifact_id":frozen.model_artifact_id,
              "exit_generation_id":str(getattr(frozen,"exit_generation_id","") or frozen.exit_policy_id),
              "selection_oos_fold_count":int(getattr(frozen,"selection_oos_fold_count",0)),
              "selection_oos_positive_fold_fraction":float(getattr(frozen,"selection_oos_positive_fold_fraction",float("nan")))}
        rows.append({**base,"arm":"A_FROZEN_MODEL_FROZEN_CALIBRATION","generation_id":frozen.generation_id,
                     "activation_date":frozen.activation_date,"resolved_threshold":float(frozen.resolved_threshold),
                     "resolved_top_fraction":float(getattr(frozen,"resolved_top_fraction",.005))})
        fc=calibrations.loc[calibrations["family_id"].eq(family_id)].sort_values("activation_date")
        for calibration in fc.itertuples(index=False):
            rows.append({**base,"arm":"B_FROZEN_MODEL_ROLLING_RECALIBRATION","generation_id":frozen.generation_id,
                         "activation_date":calibration.activation_date,"resolved_threshold":float(calibration.resolved_threshold),
                         "resolved_top_fraction":float(getattr(calibration,"resolved_top_fraction",getattr(frozen,"resolved_top_fraction",.005)))})
        for generation in fg.itertuples(index=False):
            rows.append({"family_id":family_id,"arm":"C_ROLLING_REFIT_ROLLING_RECALIBRATION",
                         "generation_id":generation.generation_id,"activation_date":generation.activation_date,
                         "model_artifact_id":generation.model_artifact_id,
                         "resolved_threshold":float(generation.resolved_threshold),
                         "resolved_top_fraction":float(getattr(generation,"resolved_top_fraction",.005)),
                         "entry_policy_id":generation.entry_policy_id,
                         "exit_policy_id":generation.exit_policy_id,
                         "exit_generation_id":str(getattr(generation,"exit_generation_id","") or generation.exit_policy_id),
                         "selection_oos_fold_count":int(getattr(generation,"selection_oos_fold_count",0)),
                         "selection_oos_positive_fold_fraction":float(getattr(generation,"selection_oos_positive_fold_fraction",float("nan")))})
        if optimized_calibrations is not None and not optimized_calibrations.empty:
            optimized = optimized_calibrations.loc[
                optimized_calibrations["family_id"].eq(family_id)
            ].sort_values("activation_date")
            frozen_optimized = optimized.loc[optimized["optimization_scope"].eq("FROZEN_MODEL")]
            for calibration in frozen_optimized.itertuples(index=False):
                rows.append({**base,"arm":"B2_FROZEN_MODEL_ROLLING_POLICY_AND_RECALIBRATION",
                             "generation_id":frozen.generation_id,"activation_date":calibration.activation_date,
                             "resolved_threshold":float(calibration.resolved_threshold),
                             "resolved_top_fraction":float(calibration.resolved_top_fraction)})
            rolling_optimized = optimized.loc[optimized["optimization_scope"].eq("ROLLING_MODEL")]
            optimized_by_generation = rolling_optimized.set_index("generation_id")
            for generation in fg.itertuples(index=False):
                if generation.generation_id not in optimized_by_generation.index:
                    continue
                calibration = optimized_by_generation.loc[generation.generation_id]
                if isinstance(calibration, pd.DataFrame):
                    calibration = calibration.iloc[-1]
                rows.append({"family_id":family_id,"arm":"C2_ROLLING_REFIT_ROLLING_POLICY_AND_RECALIBRATION",
                             "generation_id":generation.generation_id,"activation_date":generation.activation_date,
                             "model_artifact_id":generation.model_artifact_id,
                             "resolved_threshold":float(calibration.resolved_threshold),
                             "resolved_top_fraction":float(calibration.resolved_top_fraction),
                             "entry_policy_id":generation.entry_policy_id,"exit_policy_id":generation.exit_policy_id,
                             "exit_generation_id":str(getattr(generation,"exit_generation_id","") or generation.exit_policy_id),
                             "selection_oos_fold_count":int(getattr(generation,"selection_oos_fold_count",0)),
                             "selection_oos_positive_fold_fraction":float(getattr(generation,"selection_oos_positive_fold_fraction",float("nan")))})
    result=pd.DataFrame(rows).sort_values(["family_id","arm","activation_date"]).reset_index(drop=True)
    validate_abc_generation_schedules(result)
    return result


def compare_abc(arm_summary: pd.DataFrame) -> dict:
    required = {"family_id", "arm", "assessment_date", "relative_return"}
    missing = required - set(arm_summary)
    if missing:
        raise ValueError(f"ABC_COLUMNS_MISSING:{sorted(missing)}")
    values = arm_summary.copy()
    values["assessment_date"] = pd.to_datetime(values["assessment_date"]).dt.normalize()
    values["plateau_id"] = values["family_id"].map(structural_plateau_id_from_family_id)
    policy_level = values.pivot_table(index=["assessment_date", "plateau_id", "family_id"],
                                      columns="arm", values="relative_return", aggfunc="first")
    names = {
        "A": "A_FROZEN_MODEL_FROZEN_CALIBRATION",
        "B": "B_FROZEN_MODEL_ROLLING_RECALIBRATION",
        "C": "C_ROLLING_REFIT_ROLLING_RECALIBRATION",
    }
    for value in names.values():
        if value not in policy_level:
            raise ValueError(f"ABC_ARM_MISSING:{value}")
    policy_level = policy_level.dropna(subset=list(names.values()))
    policy_level["delta_ba"] = policy_level[names["B"]] - policy_level[names["A"]]
    policy_level["delta_cb"] = policy_level[names["C"]] - policy_level[names["B"]]
    plateau = policy_level[["delta_ba", "delta_cb"]].groupby(level=[0, 1]).mean()
    temporal = plateau.groupby(level=0).median()
    delta_ba = temporal["delta_ba"]
    delta_cb = temporal["delta_cb"]
    bootstrap_ba = block_bootstrap_mean(delta_ba.tolist(), block_size=3, repetitions=1000, seed=17) if len(delta_ba) else {}
    bootstrap_cb = block_bootstrap_mean(delta_cb.tolist(), block_size=3, repetitions=1000, seed=19) if len(delta_cb) else {}
    ba_pass = bool(len(delta_ba) and delta_ba.median() > 0 and (delta_ba > 0).mean() >= .6
                   and float(bootstrap_ba.get("q05", -1)) > 0)
    cb_pass = bool(len(delta_cb) and delta_cb.median() > 0 and (delta_cb > 0).mean() >= .6
                   and float(bootstrap_cb.get("q05", -1)) > 0)
    return {
        "primary_statistical_unit": "ASSESSMENT_MONTH",
        "policy_variants_clustered_within_plateau": True,
        "recalibration_incremental_monthly_excess_median": float(delta_ba.median()) if len(delta_ba) else 0.0,
        "refit_incremental_monthly_excess_median": float(delta_cb.median()) if len(delta_cb) else 0.0,
        "rolling_recalibration_promoted": ba_pass,
        "rolling_refit_promoted": cb_pass,
        "recalibration_block_bootstrap": bootstrap_ba,
        "refit_block_bootstrap": bootstrap_cb,
        "assessment_month_count": int(len(temporal)),
        "plateau_count": int(plateau.index.get_level_values(1).nunique()) if len(plateau) else 0,
        "policy_count": int(values["family_id"].nunique()),
    }


def validate_abc_generation_schedules(schedule: pd.DataFrame) -> None:
    required={"family_id","arm","generation_id","model_artifact_id","resolved_threshold","activation_date"}
    missing=required-set(schedule)
    if missing: raise ValueError(f"ABC_SCHEDULE_COLUMNS_MISSING:{sorted(missing)}")
    for family_id, group in schedule.groupby("family_id"):
        a=group.loc[group["arm"].eq("A_FROZEN_MODEL_FROZEN_CALIBRATION")]
        b=group.loc[group["arm"].eq("B_FROZEN_MODEL_ROLLING_RECALIBRATION")]
        c=group.loc[group["arm"].eq("C_ROLLING_REFIT_ROLLING_RECALIBRATION")]
        if a["generation_id"].nunique()!=1 or a["model_artifact_id"].nunique()!=1 or a["resolved_threshold"].nunique()!=1:
            raise ValueError(f"ABC_A_NOT_FROZEN:{family_id}")
        if b["generation_id"].nunique()!=1 or b["model_artifact_id"].nunique()!=1:
            raise ValueError(f"ABC_B_MODEL_NOT_FROZEN:{family_id}")
        if c.empty:
            raise ValueError(f"ABC_C_MISSING:{family_id}")
        if c.groupby("generation_id")["model_artifact_id"].nunique().gt(1).any():
            raise ValueError(f"ABC_C_GENERATION_MODEL_AMBIGUOUS:{family_id}")
        for optional in ("B2_FROZEN_MODEL_ROLLING_POLICY_AND_RECALIBRATION",
                         "C2_ROLLING_REFIT_ROLLING_POLICY_AND_RECALIBRATION"):
            arm = group.loc[group["arm"].eq(optional)]
            if not arm.empty and ("resolved_top_fraction" not in arm or arm["resolved_top_fraction"].isna().any()):
                raise ValueError(f"ABC_OPTIONAL_POLICY_ARM_INCOMPLETE:{family_id}:{optional}")
