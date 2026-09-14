"""Required real-data H11/H24/H28 Dynamic-QBD integration smoke."""
from __future__ import annotations

import argparse
from datetime import date
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .dynamic_qbd_development_pipeline import run_pipeline


SMOKE_HORIZONS = (11, 24, 28)
SMOKE_MAX_NAMES = (1, 3, 6)


def smoke_family_ids(*, holding_days: int = 5, learned_exit: bool = False) -> tuple[str, ...]:
    suffix = "LEARNED_EXIT" if learned_exit else "FIXED"
    return tuple(
        f"H{horizon:02d}_D{min(holding_days, horizon):02d}_N{names:02d}_{suffix}"
        for horizon in SMOKE_HORIZONS for names in SMOKE_MAX_NAMES
    )


def standard_smoke_family_ids(*, holding_days: int = 5, fixed_only: bool = False) -> tuple[str, ...]:
    fixed=smoke_family_ids(holding_days=holding_days)
    return fixed if fixed_only else fixed+smoke_family_ids(holding_days=holding_days,learned_exit=True)


def resolve_smoke_window(signal_panel: str | Path, start: str | None, end: str | None,
                         holdout_contract: str) -> tuple[date,date]:
    if (start is None) != (end is None):
        raise ValueError("REAL_SMOKE_START_AND_END_MUST_BOTH_BE_SET_OR_OMITTED")
    if start is not None:
        return date.fromisoformat(start),date.fromisoformat(end)
    panel=pd.read_parquet(signal_panel,columns=["decision_date","holdout_locked"]).drop_duplicates()
    panel["decision_date"]=pd.to_datetime(panel["decision_date"]).dt.normalize()
    if holdout_contract=="PRESERVE_HISTORICAL_LOCKBOX":
        locked=panel.loc[panel["holdout_locked"].astype(bool),"decision_date"]
        if locked.empty: raise ValueError("REAL_SMOKE_HISTORICAL_LOCKBOX_MISSING")
        eligible=panel.loc[panel["decision_date"].lt(locked.min()),"decision_date"]
    else:
        eligible=panel.loc[panel["decision_date"].lt(pd.Timestamp("2026-07-25")),"decision_date"]
    if eligible.empty: raise ValueError("REAL_SMOKE_NO_PRE_HOLDOUT_DATES")
    resolved_end=eligible.max()
    resolved_start=(resolved_end.to_period("M")-2).start_time
    return resolved_start.date(),resolved_end.date()


def validate_real_smoke(output_root: str | Path, expected_families) -> dict:
    root=Path(output_root)
    generations=pd.read_parquet(root/"valid_generations.parquet")
    schedules=pd.read_parquet(root/"abc_generation_schedule.parquet")
    predictions=pd.read_parquet(root/"model_predictions.parquet")
    trades=pd.read_parquet(root/"development"/"family_shadow_trades.parquet")
    expected=set(expected_families)
    if set(generations["family_id"]) != expected:
        raise AssertionError("REAL_SMOKE_FAMILY_SURFACE_MISMATCH")
    counts=generations.groupby("family_id")["generation_id"].nunique()
    if counts.lt(2).any():
        raise AssertionError(f"REAL_SMOKE_REQUIRES_MULTIPLE_REFITS:{counts[counts.lt(2)].to_dict()}")
    if generations.groupby("family_id")["model_artifact_id"].nunique().lt(2).any():
        raise AssertionError("REAL_SMOKE_MONTHLY_MODEL_ARTIFACT_DID_NOT_CHANGE")
    if not (pd.to_datetime(generations["train_end"]).le(pd.to_datetime(generations["latest_matured_label_cutoff"])).all()
            and pd.to_datetime(generations["calibration_end"]).le(pd.to_datetime(generations["information_cutoff"])).all()):
        raise AssertionError("REAL_SMOKE_MATURITY_CONTRACT_BROKEN")
    if generations.groupby("family_id")["calibration_fingerprint"].nunique().lt(2).any():
        raise AssertionError("REAL_SMOKE_GENERATION_CALIBRATION_DID_NOT_CHANGE")
    if generations.groupby("family_id")["resolved_threshold"].nunique().lt(2).any():
        raise AssertionError("REAL_SMOKE_RESOLVED_THRESHOLD_DID_NOT_CHANGE")
    changed=[]
    for family_id, family_generations in generations.groupby("family_id"):
        ordered=family_generations.sort_values("activation_date").head(2)
        first,second=ordered.iloc[0],ordered.iloc[1]
        left=predictions.loc[predictions["model_artifact_id"].astype(str).eq(str(first.model_artifact_id)),
                             ["decision_date","ticker","score"]]
        right=predictions.loc[predictions["model_artifact_id"].astype(str).eq(str(second.model_artifact_id)),
                              ["decision_date","ticker","score"]]
        overlap=left.merge(right,on=["decision_date","ticker"],suffixes=("_first","_second"))
        changed.append(bool(len(overlap) and not np.allclose(overlap["score_first"],overlap["score_second"])))
    if not all(changed):
        raise AssertionError("REAL_SMOKE_ACTIVE_GENERATION_PREDICTIONS_DID_NOT_CHANGE")
    primary=schedules.loc[schedules["arm"].eq("C_ROLLING_REFIT_ROLLING_RECALIBRATION")]
    if primary.groupby("family_id")["generation_id"].nunique().lt(2).any():
        raise AssertionError("REAL_SMOKE_C_SCHEDULE_NOT_ROLLING")
    lineage_columns={"entry_family_id","entry_generation_id","entry_model_artifact_id",
                     "entry_policy_id","exit_generation_id"}
    if trades.empty or lineage_columns-set(trades):
        raise AssertionError(f"REAL_SMOKE_POSITION_LINEAGE_NOT_OBSERVED:{sorted(lineage_columns-set(trades))}")
    known_generations=set(generations["generation_id"].astype(str))
    known_models=set(generations["model_artifact_id"].astype(str))
    if not set(trades["entry_generation_id"].dropna().astype(str)).issubset(known_generations):
        raise AssertionError("REAL_SMOKE_UNKNOWN_ENTRY_GENERATION")
    if not set(trades["entry_model_artifact_id"].dropna().astype(str)).issubset(known_models):
        raise AssertionError("REAL_SMOKE_UNKNOWN_ENTRY_MODEL")
    learned_expected={family_id for family_id in expected if family_id.endswith("_LEARNED_EXIT")}
    if learned_expected:
        learned=trades.loc[trades["entry_family_id"].astype(str).isin(learned_expected)]
        if learned.empty or learned["exit_generation_id"].isna().any():
            raise AssertionError("REAL_SMOKE_LEARNED_EXIT_LINEAGE_NOT_OBSERVED")
        summary=pd.read_csv(root/"development"/"abc_arm_summary.csv")
        learned_summary=summary.loc[summary["family_id"].isin(learned_expected)]
        if learned_summary.empty:
            raise AssertionError("REAL_SMOKE_LEARNED_EXIT_RESULTS_MISSING")
    manifest=json.loads((root/"manifest.json").read_text(encoding="utf-8"))
    if manifest.get("router_mode") != "SHADOW_RESEARCH_ONLY" or manifest.get("capital_authority") is not False:
        raise AssertionError("REAL_SMOKE_ROUTER_ACQUIRED_CAPITAL_AUTHORITY")
    result={"status":"DYNAMIC_QBD_REAL_SMOKE_PASS","families":len(expected),
            "generations":int(len(generations)),"trades":int(len(trades)),
            "nav_semantic_hash":manifest["nav_semantic_hash"],
            "reproducibility_fingerprint":manifest["reproducibility_fingerprint"]}
    (root/"REAL_SMOKE_PASS.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    return result


def main(argv=None) -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--signal-panel",required=True)
    parser.add_argument("--candidate-metrics",required=True)
    parser.add_argument("--daily-store-root",required=True)
    parser.add_argument("--output-root",required=True)
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--holdout-contract",choices=("PRESERVE_HISTORICAL_LOCKBOX","PROSPECTIVE_FROM_2026_07_25"),required=True)
    parser.add_argument("--fixed-only",action="store_true")
    parser.add_argument("--learned-exit-candidate-metrics")
    parser.add_argument("--training-window-sessions",type=int,default=504)
    parser.add_argument("--calibration-window-sessions",type=int,default=252)
    parser.add_argument("--score-quantile",type=float,default=.75)
    parser.add_argument("--top-fraction",type=float,default=.01)
    args=parser.parse_args(argv)
    families=standard_smoke_family_ids(fixed_only=args.fixed_only)
    if not args.fixed_only and not args.learned_exit_candidate_metrics:
        parser.error("standard smoke requires --learned-exit-candidate-metrics; use --fixed-only only for diagnostics")
    start,end=resolve_smoke_window(args.signal_panel,args.start,args.end,args.holdout_contract)
    common=dict(signal_panel=args.signal_panel,candidate_metrics=args.candidate_metrics,
                 daily_store_root=args.daily_store_root,output_root=args.output_root,
                 family_ids=families,start=start,end=end,
                 holdout_contract=args.holdout_contract,feature_schema_sha256="AUTO",
                 learned_exit_candidate_metrics=args.learned_exit_candidate_metrics,
                 training_window_sessions=args.training_window_sessions,
                 calibration_window_sessions=args.calibration_window_sessions,
                 score_quantile=args.score_quantile,top_fraction=args.top_fraction)
    run_pipeline(**common)
    first_manifest=json.loads((Path(args.output_root)/"manifest.json").read_text(encoding="utf-8"))
    run_pipeline(**common)
    second_manifest=json.loads((Path(args.output_root)/"manifest.json").read_text(encoding="utf-8"))
    for key in ("nav_semantic_hash","reproducibility_fingerprint"):
        if first_manifest.get(key)!=second_manifest.get(key):
            raise AssertionError(f"REAL_SMOKE_RESTART_{key.upper()}_MISMATCH")
    result=validate_real_smoke(args.output_root,families)
    result["restart_reproduction_verified"]=True
    result["window"]={"start":str(start),"end":str(end)}
    Path(args.output_root,"REAL_SMOKE_PASS.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps(result,sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
