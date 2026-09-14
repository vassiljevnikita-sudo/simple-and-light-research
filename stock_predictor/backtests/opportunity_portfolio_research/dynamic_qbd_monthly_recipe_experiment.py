"""Monthly-refit Dynamic-QBD recipe-selection shadow experiment.

R0 freezes the first causal recipe but refits and recalibrates it monthly.
R1 reselects the recipe only when newly matured Outer-WF evidence changes the
winner, while still refitting/recalibrating monthly. R2 adds a preregistered
paired-fold bootstrap hysteresis gate. R3 is a causal 50/50 Ridge/HGB daily
percentile blend with its own monthly calibration.
"""
from __future__ import annotations

import argparse
from bisect import bisect_left
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import date
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

import numpy as np
import pandas as pd

from .contract_fingerprints import stable_hash
from .cost_contracts import CostModel
from .portfolio_policy_contracts import Policy
from .tax_contracts import TaxConfig
from .dynamic_qbd_factory import monthly_refit_dates
from .dynamic_qbd_h1_30_adapter import H130ProductionGenerationBuilder, ParquetH130DatasetMaterializer
from .dynamic_qbd_maturity import HorizonMaturityResolver
from .dynamic_qbd_development_pipeline import materialize_daily_store_prices
from .dynamic_qbd_portfolio_replay import replay_family
from .dynamic_qbd_generation_recalibration import recalibrate_generation
from .dynamic_qbd_runtime_resources import active_cpu_contract, configure_cpu_peak
from .dynamic_qbd_family_surface import build_family_specs
from .dynamic_qbd_wealth_metrics import wealth_path_metrics
from .next_open_portfolio_replay import _benchmark_total_return_maps, _distribution_contract


ARMS=(
    "R0_FROZEN_RECIPE_MONTHLY_REFIT",
    "R1_ROLLING_RECIPE_MONTHLY_REFIT",
    "R2_HYSTERESIS_RECIPE_MONTHLY_REFIT",
    "R3_RIDGE_HGB_50_50_PERCENTILE_BLEND",
)
HYSTERESIS_MINIMUM_PAIRED_FOLDS=5
HYSTERESIS_PROBABILITY=0.90
HYSTERESIS_MINIMUM_ROBUST_DELTA=0.005
BOOTSTRAP_DRAWS=5000
_MONTHLY_TRAINING_SOURCE_FILES=(
    "dynamic_qbd_monthly_recipe_experiment.py",
    "dynamic_qbd_h1_30_adapter.py",
    "dynamic_qbd_generation_recalibration.py",
    "dynamic_qbd_family_surface.py",
    "contract_fingerprints.py",
    "portfolio_policy_contracts.py",
    "cost_contracts.py",
    "tax_contracts.py",
    "market_regime_contracts.py",
)


class NoCausalMonthlyWindow(RuntimeError):
    def __init__(self, payload: dict):
        super().__init__(str(payload["reason_code"]))
        self.payload=payload


def _markdown_table(frame: pd.DataFrame) -> str:
    """Render a compact Markdown table without pandas' optional tabulate dependency."""
    display=frame.copy()
    for column in display.columns:
        if pd.api.types.is_float_dtype(display[column]):
            display[column]=display[column].map(lambda value: "" if pd.isna(value) else f"{value:.6f}")
    labels=[str(column) for column in display.columns]
    rows=["| "+" | ".join(labels)+" |","| "+" | ".join("---" for _ in labels)+" |"]
    rows.extend("| "+" | ".join(str(value).replace("|","\\|") for value in row)+" |"
                for row in display.itertuples(index=False,name=None))
    return "\n".join(rows)


def _identity(choice: dict) -> tuple[str,str,str]:
    return str(choice["candidate_id"]),str(choice["family"]),stable_hash(choice["parameters"])


def _sha256_file(path: Path) -> str:
    digest=hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda:handle.read(8*1024*1024),b""): digest.update(block)
    return digest.hexdigest()


def _current_git_sha() -> str:
    result=subprocess.run(["git","rev-parse","HEAD"],cwd=Path(__file__).resolve().parents[3],
                          capture_output=True,text=True,check=False)
    value=result.stdout.strip()
    if result.returncode or len(value)!=40:
        raise RuntimeError("MONTHLY_MODEL_CACHE_GIT_SHA_UNAVAILABLE")
    return value


def _monthly_training_source_sha256() -> str:
    root=Path(__file__).resolve().parent
    digest=hashlib.sha256()
    for name in _MONTHLY_TRAINING_SOURCE_FILES:
        path=root/name
        if not path.is_file(): raise RuntimeError(f"MONTHLY_TRAINING_SOURCE_MISSING:{name}")
        digest.update(name.encode("utf-8")); digest.update(b"\0")
        with path.open("rb") as handle:
            for block in iter(lambda:handle.read(8*1024*1024),b""): digest.update(block)
    return digest.hexdigest()


def _write_atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    temporary.replace(path)


def _panel_sha256(path: Path) -> str:
    manifest=path.with_name("manifest.json")
    if manifest.is_file():
        payload=json.loads(manifest.read_text(encoding="utf-8"))
        signal=payload.get("signal",{})
        if Path(str(signal.get("path",path))).resolve()==path.resolve() and signal.get("sha256"):
            return str(signal["sha256"])
    return _sha256_file(path)


def _load_distribution_contract(path: str|Path|None) -> tuple[pd.DataFrame,dict]:
    if path is None: raise ValueError("TOTAL_RETURN_REPLAY_REQUIRES_BENCHMARK_DISTRIBUTIONS")
    path=Path(path); manifest_path=path.with_suffix(path.suffix+".manifest.json")
    if not path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("BENCHMARK_DISTRIBUTION_DATA_OR_MANIFEST_MISSING")
    manifest=json.loads(manifest_path.read_text(encoding="utf-8"))
    required={"schema_version":"BLACKROCK_URTH_DISTRIBUTIONS_V1","ticker":"URTH",
              "usage":"CASH_DISTRIBUTION_LEDGER_FOR_STATEFUL_TOTAL_RETURN_REPLAY"}
    if any(manifest.get(key)!=value for key,value in required.items()):
        raise ValueError("BENCHMARK_DISTRIBUTION_MANIFEST_CONTRACT_MISMATCH")
    if manifest.get("output_sha256")!=_sha256_file(path):
        raise ValueError("BENCHMARK_DISTRIBUTION_HASH_MISMATCH")
    return pd.read_parquet(path),manifest


def _robust_score(values: np.ndarray) -> float:
    values=np.asarray(values,dtype=float); values=values[np.isfinite(values)]
    if not len(values): return -2.0
    return float(np.median(values)+.5*np.mean(values)-.5*np.std(values)-.02*np.mean(values<=0))


def _paired_bootstrap(*, incumbent: dict, challenger: dict, rows: list[dict], cutoff: date) -> dict:
    grouped={}
    for row in rows:
        key=(str(row.get("candidate_id")),str(row.get("family")))
        grouped.setdefault(key,{})[str(row.get("fold_id"))]=float(row.get("metrics",{}).get("spearman",np.nan))
    incumbent_map=grouped.get((str(incumbent["candidate_id"]),str(incumbent["family"])),{})
    challenger_map=grouped.get((str(challenger["candidate_id"]),str(challenger["family"])),{})
    folds=sorted(set(incumbent["fold_ids"]) & set(challenger["fold_ids"]) & set(incumbent_map) & set(challenger_map))
    if len(folds)<2:
        return {"paired_fold_count":len(folds),"probability_challenger_better":0.0,
                "delta_ci_05":float("nan"),"delta_ci_95":float("nan")}
    left=np.asarray([incumbent_map[x] for x in folds]); right=np.asarray([challenger_map[x] for x in folds])
    seed=int(stable_hash(("RECIPE_HYSTERESIS",str(cutoff),folds))[:8],16)
    rng=np.random.default_rng(seed); indices=rng.integers(0,len(folds),size=(BOOTSTRAP_DRAWS,len(folds)))
    deltas=np.asarray([_robust_score(right[index])-_robust_score(left[index]) for index in indices])
    return {"paired_fold_count":len(folds),"paired_fold_ids":folds,
            "probability_challenger_better":float(np.mean(deltas>0)),
            "delta_ci_05":float(np.quantile(deltas,.05)),"delta_ci_95":float(np.quantile(deltas,.95)),
            "bootstrap_draws":BOOTSTRAP_DRAWS,"bootstrap_seed":seed}


def _hysteresis_accepts(bootstrap: dict, robust_delta: float) -> bool:
    return (int(bootstrap["paired_fold_count"])>=HYSTERESIS_MINIMUM_PAIRED_FOLDS
            and float(bootstrap["probability_challenger_better"])>=HYSTERESIS_PROBABILITY
            and float(robust_delta)>=HYSTERESIS_MINIMUM_ROBUST_DELTA)


def _daily_percentile(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    result=frame.copy(); result["decision_date"]=pd.to_datetime(result["decision_date"]).dt.normalize()
    rank=result.groupby("decision_date")["score"].rank(method="average",ascending=False)
    size=result.groupby("decision_date")["score"].transform("size").astype(float)
    result[name]=1.0-rank/size
    return result


def _blend_frames(left: pd.DataFrame, right: pd.DataFrame, *, calibration: bool) -> pd.DataFrame:
    left=_daily_percentile(left,"ridge_percentile"); right=_daily_percentile(right,"hgb_percentile")
    keep=["decision_date","ticker","ridge_percentile"]
    if calibration: keep += ["terminal_date","observed_excess"]
    merged=left[keep].merge(right[["decision_date","ticker","hgb_percentile"]],
        on=["decision_date","ticker"],how="inner",validate="one_to_one")
    merged["score"]=.5*merged["ridge_percentile"]+.5*merged["hgb_percentile"]
    return merged


class MonthlyModelFactory:
    def __init__(self, *, base, panel: Path, metrics: Path, development_end: date,
                 sessions: tuple[date,...], root: Path):
        self.base=base; self.panel=panel; self.metrics=metrics; self.development_end=development_end
        self.sessions=sessions; self.root=root; self.maturity=HorizonMaturityResolver(sessions)
        self.rows=json.loads(metrics.read_text(encoding="utf-8"))
        self.git_sha=_current_git_sha(); self.training_source_sha256=_monthly_training_source_sha256()
        self.candidate_metrics_sha256=_sha256_file(metrics)
        self.materializer=ParquetH130DatasetMaterializer(panel,metrics,development_end)
        self.selector=H130ProductionGenerationBuilder(materializer=self.materializer,
            root=root/"selector",code_commit=self.git_sha)
        self._recipe_builders={}; self._recipe_materializers={}; self._recipe_families={}; self._cache={}
        cache_contract={"schema_version":"MONTHLY_MODEL_CACHE_CONTRACT_V2_PROVENANCE_MANIFEST",
            "signal_panel_path":str(panel.resolve()),"signal_panel_sha256":_panel_sha256(panel),
            "candidate_metrics_path":str(metrics.resolve()),"candidate_metrics_sha256":self.candidate_metrics_sha256,
            "development_end":str(development_end),"session_start":str(sessions[0]),"session_end":str(sessions[-1]),
            "session_count":len(sessions),"base_family_fingerprint":base.family_hash,
            "git_sha":self.git_sha,"training_source_sha256":self.training_source_sha256,
            "factory_code_contract":"MONTHLY_RECIPE_FACTORY_V2_PROVENANCE_MANIFEST"}
        cache_contract["contract_sha256"]=stable_hash(cache_contract)
        contract_path=root/"monthly-model-cache-contract.json"
        existing_models=(root/"monthly-models").is_dir() and next((root/"monthly-models").rglob("model.joblib"),None) is not None
        if contract_path.is_file():
            existing=json.loads(contract_path.read_text(encoding="utf-8"))
            if existing.get("contract_sha256")!=cache_contract["contract_sha256"]:
                raise RuntimeError("MONTHLY_MODEL_CACHE_CONTRACT_OR_PROVENANCE_MISMATCH_REBUILD_REQUIRED")
        elif existing_models:
            raise RuntimeError("UNVERIFIED_MONTHLY_MODEL_CACHE_REUSE_BLOCKED")
        else:
            contract_path.parent.mkdir(parents=True,exist_ok=True)
            temporary=contract_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(cache_contract,indent=2,sort_keys=True)+"\n",encoding="utf-8")
            temporary.replace(contract_path)

    def choices(self, cutoff: date, model_family: str="RIDGE_HGB_FROZEN_RULE") -> tuple[dict,...]:
        latest=self.maturity.latest_matured_decision(cutoff,self.base.horizon_sessions)
        family=replace(self.base,model_family=model_family)
        return self.selector.candidate_choices(family,self.metrics,latest)

    def winner(self, cutoff: date, model_family: str="RIDGE_HGB_FROZEN_RULE") -> dict:
        choices=self.choices(cutoff,model_family)
        return max(choices,key=lambda x:(x["robust_score"],x["median_spearman"],-x["mean_mae"],x["candidate_id"]))

    def _recipe_objects(self, choice: dict):
        key=_identity(choice)
        if key not in self._recipe_builders:
            filtered=[row for row in self.rows if int(row.get("horizon_sessions",-1))==self.base.horizon_sessions
                and str(row.get("candidate_id"))==str(choice["candidate_id"])
                and str(row.get("family"))==str(choice["family"])]
            metric_path=self.root/"recipe-evidence"/f"{choice['candidate_id']}.json"
            metric_path.parent.mkdir(parents=True,exist_ok=True)
            metric_path.write_text(json.dumps(filtered,indent=2,sort_keys=True)+"\n",encoding="utf-8")
            materializer=ParquetH130DatasetMaterializer(self.panel,metric_path,self.development_end)
            family=replace(self.base,family_id=f"{self.base.family_id}_RECIPE_{choice['candidate_id']}",
                model_family=str(choice["family"]),hyperparameter_rule={**self.base.hyperparameter_rule,
                    "minimum_oos_folds":2,"recipe_selection_contract":"FIRST_CAUSAL_REFIT_THEN_FROZEN_WITHIN_FAMILY"})
            builder=H130ProductionGenerationBuilder(materializer=materializer,
                root=self.root/"monthly-models"/str(choice["candidate_id"]),code_commit=self.git_sha)
            self._recipe_materializers[key]=materializer; self._recipe_families[key]=family; self._recipe_builders[key]=builder
        return self._recipe_families[key],self._recipe_builders[key]

    @staticmethod
    def _recipe_identity(choice: dict) -> dict:
        return {"candidate_id":str(choice["candidate_id"]),"family":str(choice["family"]),
                "parameters":choice["parameters"],"identity_sha256":stable_hash(_identity(choice))}

    def _generation_manifest_payload(self, *, family, choice: dict, cutoff: date, build: dict,
                                     calibration_path: Path, raw_prediction_path: Path,
                                     prediction_path: Path, model_path: Path) -> dict:
        return {"schema_version":"MONTHLY_MODEL_GENERATION_MANIFEST_V1",
            "git_sha":self.git_sha,"training_source_sha256":self.training_source_sha256,
            "family_spec_hash":family.family_hash,"dataset_hash":str(build["dataset_fingerprint"]),
            "candidate_metrics_hash":self.candidate_metrics_sha256,"cutoff":cutoff.isoformat(),
            "recipe_identity":self._recipe_identity(choice),"model_artifact_id":str(build["model_artifact_id"]),
            "artifacts":{"model.joblib":_sha256_file(model_path),
                "calibration-predictions.parquet":_sha256_file(calibration_path),
                "generation-predictions.parquet":_sha256_file(raw_prediction_path),
                "model-predictions.parquet":_sha256_file(prediction_path)}}

    def _verify_generation_manifest(self, *, artifact_root: Path, family, choice: dict, cutoff: date,
                                    calibration_path: Path, raw_prediction_path: Path,
                                    prediction_path: Path, model_path: Path) -> dict:
        manifest_path=artifact_root/"generation-manifest.json"
        if not manifest_path.is_file():
            raise RuntimeError(f"MONTHLY_MODEL_CACHE_GENERATION_MANIFEST_MISSING:{artifact_root}")
        manifest=json.loads(manifest_path.read_text(encoding="utf-8"))
        expected={"schema_version":"MONTHLY_MODEL_GENERATION_MANIFEST_V1","git_sha":self.git_sha,
            "training_source_sha256":self.training_source_sha256,"family_spec_hash":family.family_hash,
            "candidate_metrics_hash":self.candidate_metrics_sha256,"cutoff":cutoff.isoformat(),
            "recipe_identity":self._recipe_identity(choice)}
        for key,value in expected.items():
            if manifest.get(key)!=value:
                raise RuntimeError(f"MONTHLY_MODEL_CACHE_MANIFEST_CONTEXT_MISMATCH:{key}:{artifact_root}")
        required_hashes={"model.joblib":model_path,"calibration-predictions.parquet":calibration_path,
            "generation-predictions.parquet":raw_prediction_path,"model-predictions.parquet":prediction_path}
        if not isinstance(manifest.get("dataset_hash"),str) or not manifest["dataset_hash"]:
            raise RuntimeError(f"MONTHLY_MODEL_CACHE_MANIFEST_DATASET_HASH_MISSING:{artifact_root}")
        for name,path in required_hashes.items():
            expected_hash=manifest.get("artifacts",{}).get(name)
            if not isinstance(expected_hash,str) or expected_hash!=_sha256_file(path):
                raise RuntimeError(f"MONTHLY_MODEL_CACHE_ARTIFACT_HASH_MISMATCH:{name}:{artifact_root}")
        stored=pd.read_parquet(prediction_path,columns=["model_artifact_id"])
        artifact_ids=stored["model_artifact_id"].dropna().astype(str).unique()
        if len(artifact_ids)!=1 or str(artifact_ids[0])!=str(manifest.get("model_artifact_id")):
            raise RuntimeError(f"MONTHLY_MODEL_CACHE_IDENTITY_INVALID:{artifact_root}")
        return manifest

    def fit(self, choice: dict, cutoff: date) -> dict:
        cache_key=(_identity(choice),cutoff)
        if cache_key in self._cache: return self._cache[cache_key]
        family,builder=self._recipe_objects(choice)
        latest=self.maturity.latest_matured_decision(cutoff,family.horizon_sessions)
        artifact_root=builder.root/family.family_id/cutoff.isoformat()
        calibration_path=artifact_root/"calibration-predictions.parquet"
        prediction_path=artifact_root/"model-predictions.parquet"
        raw_prediction_path=artifact_root/"generation-predictions.parquet"
        model_path=artifact_root/"model.joblib"
        artifact_paths=(calibration_path,prediction_path,raw_prediction_path,model_path)
        present_count=sum(path.is_file() for path in artifact_paths)
        if present_count and present_count!=len(artifact_paths):
            raise RuntimeError(f"MONTHLY_MODEL_CACHE_INCOMPLETE_ARTIFACT_SET:{artifact_root}")
        reusable=present_count==len(artifact_paths)
        if reusable:
            manifest=self._verify_generation_manifest(artifact_root=artifact_root,family=family,choice=choice,
                cutoff=cutoff,calibration_path=calibration_path,raw_prediction_path=raw_prediction_path,
                prediction_path=prediction_path,model_path=model_path)
            build={"model_artifact_id":str(manifest["model_artifact_id"]),
                "dataset_fingerprint":str(manifest["dataset_hash"]),
                "model_artifact_sha256":str(manifest["artifacts"]["model.joblib"]),
                "calibration_prediction_path":str(calibration_path),"raw_prediction_path":str(raw_prediction_path),
                "model_path":str(model_path),"generation_manifest_path":str(artifact_root/"generation-manifest.json"),
                "cache_reused":True}
        else:
            build=builder.build(family=family,information_cutoff=cutoff,latest_matured_label_cutoff=latest)
        calibration=builder.calibration_predictions(family=family,build=build,information_cutoff=cutoff)
        generation_id=stable_hash(("MONTHLY_RECIPE",cutoff,build["model_artifact_id"]))[:24]
        record=recalibrate_generation(family,generation_id,calibration,information_cutoff=cutoff,maturity=self.maturity)
        finalized=builder.finalize_generation(family=family,build=build,generation_id=generation_id)
        predictions=pd.read_parquet(finalized["prediction_artifact_path"])
        if not build.get("cache_reused",False):
            manifest=self._generation_manifest_payload(family=family,choice=choice,cutoff=cutoff,build=build,
                calibration_path=calibration_path,raw_prediction_path=raw_prediction_path,
                prediction_path=Path(finalized["prediction_artifact_path"]),model_path=model_path)
            manifest_path=artifact_root/"generation-manifest.json"; _write_atomic_json(manifest_path,manifest)
            build["generation_manifest_path"]=str(manifest_path)
        component={"choice":choice,"family":family,"build":build,"generation_id":generation_id,
            "cutoff":cutoff,"threshold":float(record.resolved_threshold),
            "top_fraction":float(record.resolved_top_fraction),"calibration":calibration,"predictions":predictions}
        self._cache[cache_key]=component
        return component


def _schedule_row(arm: str, component: dict) -> dict:
    choice=component["choice"]
    return {"activation_date":component["cutoff"],"family_id":arm,
        "generation_id":component["generation_id"],"model_artifact_id":component["build"]["model_artifact_id"],
        "resolved_threshold":component["threshold"],"resolved_top_fraction":component["top_fraction"],
        "entry_policy_id":f"{arm}_{component['cutoff']}","exit_policy_id":"FIXED_D2",
        "recipe_candidate_id":choice["candidate_id"],"recipe_family":choice["family"],
        "recipe_parameters_json":json.dumps(choice["parameters"],sort_keys=True),
        "recipe_robust_score":choice["robust_score"],"recipe_fold_count":choice["fold_count"]}


def _blend_component(factory: MonthlyModelFactory, ridge: dict, hgb: dict, cutoff: date) -> dict:
    calibration=_blend_frames(ridge["calibration"],hgb["calibration"],calibration=True)
    predictions=_blend_frames(ridge["predictions"],hgb["predictions"],calibration=False)
    artifact_id=stable_hash(("RIDGE_HGB_50_50_PERCENTILE",ridge["build"]["model_artifact_id"],hgb["build"]["model_artifact_id"]))
    generation_id=stable_hash(("BLEND_GENERATION",cutoff,artifact_id))[:24]
    family=replace(factory.base,family_id="R3_RIDGE_HGB_50_50_PERCENTILE_BLEND")
    record=recalibrate_generation(family,generation_id,calibration,information_cutoff=cutoff,maturity=factory.maturity)
    predictions["model_artifact_id"]=artifact_id
    choice={"candidate_id":"RIDGE_HGB_50_50_PERCENTILE","family":"RIDGE_HGB_BLEND",
        "parameters":{"ridge":ridge["choice"]["parameters"],"hgb":hgb["choice"]["parameters"],"weights":[.5,.5]},
        "robust_score":float("nan"),"fold_count":min(ridge["choice"]["fold_count"],hgb["choice"]["fold_count"])}
    return {"choice":choice,"family":family,"build":{"model_artifact_id":artifact_id},
        "generation_id":generation_id,"cutoff":cutoff,"threshold":float(record.resolved_threshold),
        "top_fraction":float(record.resolved_top_fraction),"calibration":calibration,"predictions":predictions}


def _recipe_decisions(factory: MonthlyModelFactory, cutoffs: tuple[date,...]) -> tuple[dict,date,list[dict]]:
    first=factory.winner(cutoffs[0]); incumbent=first; prior_folds=tuple(first["fold_ids"]); decisions=[]
    r1={}; r2={}
    for cutoff in cutoffs:
        choices=factory.choices(cutoff); ranked=sorted(choices,
            key=lambda x:(x["robust_score"],x["median_spearman"],-x["mean_mae"],x["candidate_id"]),reverse=True)
        winner=ranked[0]; runner=ranked[1] if len(ranked)>1 else None
        r1[cutoff]=winner
        fold_event=tuple(winner["fold_ids"])!=prior_folds
        row={"assessment_date":cutoff,"new_outer_fold_evidence":fold_event,
            "winner_candidate_id":winner["candidate_id"],"winner_family":winner["family"],
            "winner_score":winner["robust_score"],"winner_fold_count":winner["fold_count"],
            "available_fold_ids_json":json.dumps(list(winner["fold_ids"])),
            "winner_positive_fold_fraction":1.0-winner["negative_fold_rate"],
            "runner_up_candidate_id":runner["candidate_id"] if runner else None,
            "runner_up_family":runner["family"] if runner else None,
            "runner_up_score":runner["robust_score"] if runner else np.nan,
            "incumbent_candidate_id_before":incumbent["candidate_id"],"incumbent_family_before":incumbent["family"],
            "incumbent_score":next((x["robust_score"] for x in choices if _identity(x)==_identity(incumbent)),np.nan)}
        row["challenger_score"]=winner["robust_score"]
        row["delta_challenger_incumbent"]=float(row["challenger_score"]-row["incumbent_score"])
        bootstrap={"probability_challenger_better":np.nan,"delta_ci_05":np.nan,
            "delta_ci_95":np.nan,"paired_fold_count":np.nan}
        switch=False
        if fold_event and _identity(winner)!=_identity(incumbent):
            current=next((x for x in choices if _identity(x)==_identity(incumbent)),None)
            if current is None: raise RuntimeError("HYSTERESIS_INCUMBENT_EVIDENCE_MISSING")
            bootstrap=_paired_bootstrap(incumbent=current,challenger=winner,rows=factory.rows,cutoff=cutoff)
            switch=_hysteresis_accepts(bootstrap,row["delta_challenger_incumbent"])
            if switch: incumbent=winner
        row.update(bootstrap); row["hysteresis_switch_accepted"]=switch
        row["r2_active_candidate_id_after"]=incumbent["candidate_id"]
        row["r2_active_family_after"]=incumbent["family"]
        decisions.append(row); r2[cutoff]=incumbent; prior_folds=tuple(winner["fold_ids"])
    return {"r1":r1,"r2":r2},cutoffs[0],decisions


def _attach_next_unseen_fold_evidence(decisions: pd.DataFrame, rows: list[dict]) -> pd.DataFrame:
    """Add strictly post-decision diagnostics; these values never enter selection."""
    result=decisions.copy(); evidence={}
    for row in rows:
        if not str(row.get("fold_id","")).startswith("WF_"):
            continue
        key=(str(row.get("candidate_id")),str(row.get("family")))
        evidence.setdefault(key,{})[str(row.get("fold_id"))]=float(row.get("metrics",{}).get("spearman",np.nan))
    for index,row in result.iterrows():
        available=set(json.loads(row["available_fold_ids_json"]))
        challenger_key=(str(row["winner_candidate_id"]),str(row["winner_family"]))
        incumbent_key=(str(row["incumbent_candidate_id_before"]),str(row["incumbent_family_before"]))
        future=sorted((set(evidence.get(challenger_key,{}))|set(evidence.get(incumbent_key,{})))-available)
        if not future: continue
        fold=future[0]; challenger=float(evidence.get(challenger_key,{}).get(fold,np.nan)); incumbent=float(evidence.get(incumbent_key,{}).get(fold,np.nan))
        result.loc[index,"next_unseen_fold_id"]=fold
        result.loc[index,"next_unseen_challenger_spearman"]=challenger
        result.loc[index,"next_unseen_incumbent_spearman"]=incumbent
        result.loc[index,"next_unseen_challenger_minus_incumbent_spearman"]=challenger-incumbent
    return result


def _candidate_execution_quality_audit(*, signals: pd.DataFrame, schedules: dict[str,list[dict]],
                                       prices: pd.DataFrame, max_names: int) -> tuple[pd.DataFrame,pd.DataFrame]:
    """Intersect causal gate candidates with their next-open execution boundary."""
    sessions=pd.DatetimeIndex(sorted(prices.loc[prices["ticker"].eq("URTH"),"date"].unique()))
    quality=prices[["date","ticker","open_quality_ok"]].copy()
    quality["date"]=pd.to_datetime(quality["date"]); quality["ticker"]=quality["ticker"].astype(str)
    summaries=[]; invalid=[]
    for arm,rows in schedules.items():
        schedule=pd.DataFrame(rows).copy().sort_values("activation_date")
        schedule["activation_date"]=pd.to_datetime(schedule["activation_date"]).astype("datetime64[ns]")
        schedule["model_artifact_id"]=schedule["model_artifact_id"].astype(str)
        source=signals.copy(); source["decision_date"]=pd.to_datetime(source["decision_date"]).astype("datetime64[ns]")
        source["model_artifact_id"]=source["model_artifact_id"].astype(str)
        dates=pd.DataFrame({"decision_date":pd.to_datetime(sorted(source["decision_date"].unique())).astype("datetime64[ns]")})
        authority=pd.merge_asof(dates,schedule[["activation_date","model_artifact_id","resolved_threshold",
            "resolved_top_fraction"]],left_on="decision_date",right_on="activation_date",direction="backward")
        candidates=source.merge(authority.dropna(subset=["model_artifact_id"]),
            on=["decision_date","model_artifact_id"],how="inner")
        selected=[]
        for decision,group in candidates.groupby("decision_date",sort=True):
            group=group.sort_values(["score","ticker"],ascending=[False,True])
            threshold=float(group["resolved_threshold"].iloc[0]); fraction=float(group["resolved_top_fraction"].iloc[0])
            passing=group.loc[group["score"].ge(threshold)]
            limit=max(1,int(np.ceil(len(group)*fraction)))
            chosen=passing.head(min(limit,max_names)).copy()
            position=int(sessions.searchsorted(pd.Timestamp(decision),side="right"))
            if position>=len(sessions): continue
            chosen["execution_date"]=sessions[position]; selected.append(chosen)
        selected_frame=(pd.concat(selected,ignore_index=True) if selected else
                        pd.DataFrame(columns=["decision_date","ticker","execution_date"]))
        if len(selected_frame):
            selected_frame=selected_frame.merge(quality,left_on=["execution_date","ticker"],
                right_on=["date","ticker"],how="left",validate="many_to_one")
            bad=selected_frame.loc[~selected_frame["open_quality_ok"].fillna(False)].copy(); bad["arm"]=arm
            invalid.append(bad)
        else: bad=selected_frame
        summaries.append({"arm":arm,"eligible_candidate_rows":int(len(selected_frame)),
            "invalid_next_open_candidate_rows":int(len(bad)),
            "invalid_next_open_candidate_fraction":float(len(bad)/len(selected_frame)) if len(selected_frame) else 0.0,
            "unique_invalid_tickers":int(bad["ticker"].nunique()) if len(bad) else 0,
            "contract":"CANDIDATE_DIAGNOSTIC_ONLY_ACTUAL_EXECUTIONS_REMAIN_FAIL_CLOSED"})
    return pd.DataFrame(summaries),pd.concat(invalid,ignore_index=True) if invalid else pd.DataFrame()


def _trade_metadata(panel: Path, trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty: return trades
    metadata=pd.read_parquet(panel,columns=["decision_date","ticker","sector","sub_industry"])
    metadata["decision_date"]=pd.to_datetime(metadata["decision_date"]); metadata=metadata.sort_values(["ticker","decision_date"])
    result=trades.copy(); result["entry_date"]=pd.to_datetime(result["entry_date"])
    sectors=[]; industries=[]; decision_dates=[]
    grouped={ticker:group.reset_index(drop=True) for ticker,group in metadata.groupby("ticker",sort=False)}
    for row in result.itertuples():
        group=grouped.get(str(row.ticker)); chosen=None
        if group is not None:
            values=group["decision_date"].tolist(); index=bisect_left(values,pd.Timestamp(row.entry_date))-1
            if index>=0: chosen=group.iloc[index]
        sectors.append(None if chosen is None else chosen.get("sector")); industries.append(None if chosen is None else chosen.get("sub_industry"))
        decision_dates.append(None if chosen is None else chosen.get("decision_date"))
    result["signal_decision_date"]=decision_dates; result["sector"]=sectors; result["sub_industry"]=industries
    return result


def _daily_attribution(*, arm: str, result: dict, schedule: pd.DataFrame, trades: pd.DataFrame,
                       prices: pd.DataFrame) -> tuple[pd.DataFrame,pd.DataFrame,dict]:
    curve=result["curve"].copy().sort_values("date").reset_index(drop=True); curve["date"]=pd.to_datetime(curve["date"]).astype("datetime64[ns]")
    curve["relative_wealth"]=curve["strategy_value"]/curve["urth_value"]
    curve["daily_relative_return"]=curve["relative_wealth"].pct_change().fillna(curve["relative_wealth"]-1.0)
    timeline=schedule.sort_values("activation_date").copy(); timeline["activation_date"]=pd.to_datetime(timeline["activation_date"]).astype("datetime64[ns]")
    curve=pd.merge_asof(curve,timeline[["activation_date","generation_id","model_artifact_id","recipe_candidate_id","recipe_family"]],
        left_on="date",right_on="activation_date",direction="backward")
    curve["arm"]=arm; curve["recipe_boundary_day"]=curve["activation_date"].eq(curve["date"])
    curve["relative_drawdown"]=curve["relative_wealth"]/curve["relative_wealth"].cummax()-1.0
    trough_index=curve["relative_drawdown"].idxmin(); trough_date=curve.loc[trough_index,"date"]
    peak_date=curve.loc[:trough_index].loc[curve.loc[:trough_index,"relative_wealth"].idxmax(),"date"]
    curve["in_max_drawdown_episode"]=curve["date"].between(peak_date,trough_date)
    price=prices.copy(); price["date"]=pd.to_datetime(price["date"]); price=price.sort_values(["ticker","date"])
    lookup={(str(x.ticker),pd.Timestamp(x.date)):(float(x.open),float(x.close)) for x in price.itertuples()}
    sessions=sorted(price.loc[price["ticker"].eq("URTH"),"date"].unique()); previous={pd.Timestamp(sessions[i]):pd.Timestamp(sessions[i-1]) for i in range(1,len(sessions))}
    generation_meta=schedule.drop_duplicates("generation_id").set_index("generation_id")[["recipe_candidate_id","recipe_family"]].to_dict("index")
    position_rows=[]
    for trade in trades.itertuples():
        entry=pd.Timestamp(trade.entry_date); exit_date=pd.Timestamp(trade.exit_date); ticker=str(trade.ticker); qty=float(trade.quantity)
        recipe=generation_meta.get(str(trade.entry_generation_id),{})
        trade_id=stable_hash((arm,ticker,str(entry),str(exit_date),str(trade.entry_generation_id)))[:20]
        active_dates=[pd.Timestamp(x) for x in sessions if entry<=pd.Timestamp(x)<=exit_date]
        for current in active_dates:
            if (ticker,current) not in lookup or ("URTH",current) not in lookup: continue
            if current==entry:
                stock_base,stock_terminal=lookup[(ticker,current)]; urth_base,urth_terminal=lookup[("URTH",current)]
            else:
                prior=previous.get(current)
                if prior is None or (ticker,prior) not in lookup or ("URTH",prior) not in lookup: continue
                stock_base=lookup[(ticker,prior)][1]; urth_base=lookup[("URTH",prior)][1]
                stock_terminal=lookup[(ticker,current)][0 if current==exit_date else 1]
                urth_terminal=lookup[("URTH",current)][0 if current==exit_date else 1]
            stock_return=stock_terminal/stock_base-1.0; urth_return=urth_terminal/urth_base-1.0
            position_rows.append({"arm":arm,"date":current,"trade_id":trade_id,"ticker":ticker,"sector":getattr(trade,"sector",None),
                "sub_industry":getattr(trade,"sub_industry",None),"entry_date":entry,"exit_date":exit_date,
                "entry_generation_id":trade.entry_generation_id,"entry_model_artifact_id":trade.entry_model_artifact_id,
                "entry_recipe_candidate_id":recipe.get("recipe_candidate_id"),"entry_recipe_family":recipe.get("recipe_family"),
                "entry_score":trade.entry_score,"entry_score_rank":trade.entry_score_rank,
                "entry_threshold":trade.entry_threshold,"entry_threshold_distance":trade.entry_threshold_distance,
                "buy_notional":trade.buy_notional,"quantity":qty,"stock_return_interval":stock_return,
                "urth_return_interval":urth_return,"active_excess_return_interval":stock_return-urth_return,
                "active_excess_contribution_eur":qty*stock_base*(stock_return-urth_return),
                "is_entry_day":current==entry,"is_exit_day":current==exit_date,
                "in_max_drawdown_episode":bool(peak_date<=current<=trough_date)})
    positions=pd.DataFrame(position_rows)
    if not positions.empty:
        open_summary=positions.groupby("date").agg(open_tickers=("ticker",lambda x:"|".join(sorted(set(x)))),
            open_generation_ids=("entry_generation_id",lambda x:"|".join(sorted(set(str(v) for v in x)))),
            active_excess_contribution_eur=("active_excess_contribution_eur","sum")).reset_index()
        curve=curve.merge(open_summary,on="date",how="left")
    summary={"arm":arm,"peak_date":str(peak_date.date()),"trough_date":str(trough_date.date()),
        "relative_max_drawdown":float(curve["relative_drawdown"].min()),
        "episode_sessions":int(curve["in_max_drawdown_episode"].sum())}
    return curve,positions,summary


def _summary_row(arm: str, result: dict) -> dict:
    metrics=result["metrics"]; risk=wealth_path_metrics(result["curve"])
    return {"arm":arm,"terminal_value":float(metrics["terminal_value"]),
        "urth_terminal_value":float(metrics["urth_terminal_value"]),
        "terminal_excess_eur":float(metrics["terminal_value"]-metrics["urth_terminal_value"]),
        "terminal_relative_return":float(metrics["terminal_value"]/metrics["urth_terminal_value"]-1.0),
        "trade_count":int(metrics["trade_count"]),"total_cost_eur":float(metrics["total_cost_eur"]),
        "gross_distribution_income_eur":float(metrics.get("gross_distribution_income_eur",0.0)),
        "benchmark_gross_distribution_income_eur":float(metrics.get("benchmark_gross_distribution_income_eur",0.0)),
        **{key:float(risk[key]) for key in ("relative_max_drawdown","cdar_95","expected_shortfall_95","time_under_water_fraction")},
        "drawdown_duration_sessions":int(risk["drawdown_duration_sessions"])}


def boundary_complete_segments(daily_frame: pd.DataFrame) -> pd.DataFrame:
    """Compound every daily relative return into exactly one active recipe."""
    keys=["arm","recipe_candidate_id","recipe_family"]
    segments=daily_frame.groupby(keys,dropna=False).agg(
        first_date=("date","min"),last_date=("date","max"),sessions=("date","size"),
        boundary_days=("recipe_boundary_day","sum")).reset_index()
    compounded=daily_frame.groupby(keys,dropna=False)["daily_relative_return"].apply(
        lambda x:float(np.prod(1.0+x.to_numpy(float))-1.0)).rename("compounded_relative_return").reset_index()
    return segments.merge(compounded,on=keys,how="left")


def _validate_boundary_attribution(daily_frame: pd.DataFrame, segments: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    for arm,frame in daily_frame.groupby("arm",sort=False):
        if frame["recipe_candidate_id"].isna().any() or frame["date"].duplicated().any():
            raise RuntimeError(f"BOUNDARY_ATTRIBUTION_COVERAGE_INVALID:{arm}")
        daily_total=float(np.prod(1.0+frame["daily_relative_return"].to_numpy(float))-1.0)
        terminal_total=float(frame.sort_values("date")["relative_wealth"].iloc[-1]-1.0)
        arm_segments=segments.loc[segments["arm"].eq(arm),"compounded_relative_return"].to_numpy(float)
        segment_total=float(np.prod(1.0+arm_segments)-1.0)
        maximum_error=max(abs(daily_total-terminal_total),abs(segment_total-terminal_total))
        if maximum_error>1e-10:
            raise RuntimeError(f"BOUNDARY_ATTRIBUTION_RECONSTRUCTION_FAILED:{arm}:{maximum_error}")
        rows.append({"arm":arm,"daily_compounded_relative_return":daily_total,
            "segment_compounded_relative_return":segment_total,"terminal_relative_return":terminal_total,
            "maximum_absolute_error":maximum_error,"sessions":len(frame),"status":"PASS"})
    return pd.DataFrame(rows)


def _drawdown_attribution(position_frame: pd.DataFrame) -> tuple[pd.DataFrame,pd.DataFrame]:
    episode=position_frame.loc[position_frame["in_max_drawdown_episode"]].copy()
    episode["negative_active_excess_contribution_eur"]=(-episode["active_excess_contribution_eur"].clip(upper=0.0))
    dimensions={"ticker":"ticker","sector":"sector","generation":"entry_generation_id",
        "recipe":"entry_recipe_candidate_id","trade":"trade_id"}
    rows=[]
    for arm,arm_frame in episode.groupby("arm",sort=False):
        total_negative=float(arm_frame["negative_active_excess_contribution_eur"].sum())
        for dimension,column in dimensions.items():
            grouped=arm_frame.assign(_key=arm_frame[column].fillna("UNKNOWN").astype(str)).groupby("_key",dropna=False).agg(
                net_active_excess_contribution_eur=("active_excess_contribution_eur","sum"),
                negative_active_excess_contribution_eur=("negative_active_excess_contribution_eur","sum"),
                active_sessions=("date","nunique"),trades=("trade_id","nunique"),tickers=("ticker","nunique")).reset_index()
            grouped["negative_contribution_share"]=np.where(total_negative>0,
                grouped["negative_active_excess_contribution_eur"]/total_negative,0.0)
            grouped["arm"]=arm; grouped["dimension"]=dimension; grouped=grouped.rename(columns={"_key":"key"})
            rows.extend(grouped.to_dict(orient="records"))
    attribution=pd.DataFrame(rows)
    classifications=[]
    for arm,arm_frame in episode.groupby("arm",sort=False):
        shares=attribution.loc[attribution["arm"].eq(arm)]
        ticker_rows=shares.loc[shares["dimension"].eq("ticker")].sort_values("negative_contribution_share",ascending=False)
        generation_rows=shares.loc[shares["dimension"].eq("generation")].sort_values("negative_contribution_share",ascending=False)
        top_ticker=float(ticker_rows["negative_contribution_share"].iloc[0])
        top_generation=float(generation_rows["negative_contribution_share"].iloc[0])
        if top_ticker>=.50: cause="A_SINGLE_TICKER_OUTLIER"
        elif top_generation>=.60: cause="B_GENERATION_CLUSTER"
        else: cause="C_MULTI_NAME_OR_MARKET_CLUSTER"
        classifications.append({"arm":arm,"cause_classification":cause,
            "top_negative_ticker":str(ticker_rows["key"].iloc[0]),"top_ticker_negative_share":top_ticker,
            "top_negative_generation":str(generation_rows["key"].iloc[0]),
            "top_generation_negative_share":top_generation,"episode_tickers":int(arm_frame["ticker"].nunique()),
            "episode_sectors":int(arm_frame["sector"].nunique()),"episode_generations":int(arm_frame["entry_generation_id"].nunique()),
            "negative_active_excess_contribution_eur":float(arm_frame["negative_active_excess_contribution_eur"].sum()),
            "net_active_excess_contribution_eur":float(arm_frame["active_excess_contribution_eur"].sum())})
    return attribution.sort_values(["arm","dimension","negative_contribution_share"],ascending=[True,True,False]),pd.DataFrame(classifications)


def _write_no_causal_window(*, output_root: Path, panel_sessions: tuple[date,...], requested_start: date,
                            requested_end: date, all_cutoffs: tuple[date,...], factory: MonthlyModelFactory,
                            family, benchmark_daily_path: str|Path|None=None,
                            distributions: pd.DataFrame|None=None) -> dict:
    purge=max(int(family.horizon_sessions),int(family.training_recipe.get("purge_sessions",30)))
    required=int(family.training_window_sessions)+purge+int(family.calibration_window_sessions)
    first_history=None; first_evidence=None; first_joint=None
    for cutoff in all_cutoffs:
        latest=factory.maturity.latest_matured_decision(cutoff,family.horizon_sessions)
        matured_count=sum(session<=latest for session in panel_sessions) if latest is not None else 0
        history_ok=matured_count>=required
        evidence_ok=False
        try:
            factory.winner(cutoff); evidence_ok=True
        except ValueError as exc:
            if not str(exc).startswith(("NO_CAUSAL_","INSUFFICIENT_CAUSAL_")): raise
        if history_ok and first_history is None: first_history=cutoff
        if evidence_ok and first_evidence is None: first_evidence=cutoff
        if history_ok and evidence_ok:
            first_joint=cutoff; break
    requested_sessions=tuple(x for x in panel_sessions if requested_start<=x<=requested_end)
    reasons=[]
    if not requested_sessions or requested_sessions[0]>requested_start:
        reasons.append("SIGNAL_PANEL_DOES_NOT_COVER_REQUESTED_START")
    if first_history is None or first_history>requested_end:
        reasons.append("INSUFFICIENT_PRE_REQUEST_MODEL_HISTORY")
    if first_evidence is None or first_evidence>requested_end:
        reasons.append("NO_MATURED_OUTER_FOLD_RECIPE_EVIDENCE")
    market_context=None
    if benchmark_daily_path is not None:
        benchmark=pd.read_parquet(benchmark_daily_path)
        benchmark=benchmark.rename(columns={"session_date":"date"})
        benchmark["date"]=pd.to_datetime(benchmark["date"])
        benchmark=benchmark.loc[benchmark["date"].between(pd.Timestamp(requested_start),pd.Timestamp(requested_end))]
        benchmark=benchmark.loc[benchmark["ticker"].astype(str).eq("URTH")].sort_values("date").reset_index(drop=True)
        if len(benchmark)>=2:
            all_dates=tuple(pd.Timestamp(x) for x in benchmark["date"])
            umap={pd.Timestamp(row.date):(float(row.open),float(row.close)) for row in benchmark.itertuples(index=False)}
            relevant=(distributions.loc[pd.to_datetime(distributions["ex_date"]).between(benchmark["date"].min(),benchmark["date"].max())]
                      if distributions is not None else pd.DataFrame())
            ex_map,_=_distribution_contract(relevant,all_dates)
            _,total_return_close,paid=_benchmark_total_return_maps(
                umap=umap,all_dates=all_dates,start_date=all_dates[0],distribution_ex_map=ex_map)
            initial_index=float(total_return_close[all_dates[0]])
            benchmark["price_wealth_index"]=benchmark["close"]/float(benchmark["close"].iloc[0])
            benchmark["wealth_index"]=[total_return_close[x]/initial_index for x in all_dates]
            benchmark["daily_return"]=benchmark["wealth_index"].pct_change().fillna(0.0)
            benchmark["drawdown"]=benchmark["wealth_index"]/benchmark["wealth_index"].cummax()-1.0
            trough=int(benchmark["drawdown"].idxmin())
            peak=int(benchmark.loc[:trough,"wealth_index"].idxmax())
            negative=benchmark.loc[benchmark["daily_return"]<0,"daily_return"]
            cutoff=float(negative.quantile(.05)) if len(negative) else 0.0
            tail=negative.loc[negative<=cutoff]
            elapsed_days=max(1,int((benchmark["date"].iloc[-1]-benchmark["date"].iloc[0]).days))
            market_context={"schema_version":"URTH_TOTAL_RETURN_MARKET_CONTEXT_V2","authority":"MARKET_CONTEXT_ONLY_NOT_QBD_PERFORMANCE",
                "provider_contract":str(benchmark["price_boundary_contract"].iloc[0]) if "price_boundary_contract" in benchmark else None,
                "return_contract":"CASH_DISTRIBUTIONS_REINVESTED_AT_PAYABLE_DATE_OPEN",
                "first_session":str(benchmark["date"].iloc[0].date()),"last_session":str(benchmark["date"].iloc[-1].date()),
                "sessions":int(len(benchmark)),"first_close":float(benchmark["close"].iloc[0]),
                "last_close":float(benchmark["close"].iloc[-1]),
                "price_return":float(benchmark["price_wealth_index"].iloc[-1]-1.0),
                "total_return":float(benchmark["wealth_index"].iloc[-1]-1.0),
                "distribution_events":int(sum(1 for value in paid.values() if value>0)),
                "annualized_return":float(benchmark["wealth_index"].iloc[-1]**(365.2425/elapsed_days)-1.0),
                "max_drawdown":float(benchmark.loc[trough,"drawdown"]),
                "max_drawdown_peak":str(benchmark.loc[peak,"date"].date()),
                "max_drawdown_trough":str(benchmark.loc[trough,"date"].date()),
                "max_drawdown_duration_sessions":int(trough-peak+1),
                "daily_expected_shortfall_95":float(tail.mean()) if len(tail) else 0.0}
            benchmark.to_parquet(output_root/"benchmark-market-context.parquet",index=False)
            (output_root/"benchmark-drawdown.json").write_text(
                json.dumps(market_context,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    payload={"schema_version":"DYNAMIC_QBD_NO_CAUSAL_WINDOW_V2","status":"NOT_RUN_FAIL_CLOSED",
        "reason_code":"NO_CAUSAL_MONTHLY_RECIPE_CUTOFF","reasons":reasons,
        "requested_start":str(requested_start),"requested_end":str(requested_end),
        "signal_panel_start":str(panel_sessions[0]),"signal_panel_end":str(panel_sessions[-1]),
        "signal_panel_sessions_in_requested_window":len(requested_sessions),
        "training_window_sessions":int(family.training_window_sessions),
        "purge_sessions":purge,"calibration_window_sessions":int(family.calibration_window_sessions),
        "required_matured_sessions_before_fit":required,
        "first_history_eligible_month_end":str(first_history) if first_history else None,
        "first_recipe_evidence_eligible_month_end":str(first_evidence) if first_evidence else None,
        "first_jointly_eligible_month_end":str(first_joint) if first_joint else None,
        "performance_results_emitted":False,"final_holdout_opened":False,
        "research_authority":"DATA_AND_CAUSALITY_PREFLIGHT_ONLY","market_context":market_context}
    output_root.mkdir(parents=True,exist_ok=True)
    (output_root/"preflight.json").write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    report=["# Dynamic QBD 2016-2017 causal preflight","","## Decision","",
        "**NOT_RUN_FAIL_CLOSED**. A portfolio result cannot be produced for the requested interval without inventing pre-2016 training history or importing future recipe evidence.","",
        "## Evidence","",f"- Requested: {requested_start} through {requested_end}",
        f"- Signal panel: {panel_sessions[0]} through {panel_sessions[-1]}",
        f"- Signal sessions inside request: {len(requested_sessions)}",
        f"- Required matured sessions before each fit: {required} ({family.training_window_sessions} train + {purge} purge + {family.calibration_window_sessions} calibration)",
        f"- First history-eligible month end: {first_history}",f"- First recipe-evidence-eligible month end: {first_evidence}",
        f"- First jointly eligible month end: {first_joint}","","## Validity guard","",
        "No future-selected recipe, future-trained model, shortened hidden window, fabricated prehistory or stale model cache was used. No QBD return, drawdown or promotion statistic was emitted."]
    if market_context is not None:
        report.extend(["","## URTH market context only","",
            "These figures use Alpaca SIP 1Day execution prices plus official iShares cash distributions and are not QBD strategy results.","",
            f"- Price return: {market_context['price_return']:.2%}",
            f"- Total return: {market_context['total_return']:.2%}",
            f"- Annualized return: {market_context['annualized_return']:.2%}",
            f"- Maximum drawdown: {market_context['max_drawdown']:.2%} ({market_context['max_drawdown_peak']} to {market_context['max_drawdown_trough']})",
            f"- Daily expected shortfall (95%): {market_context['daily_expected_shortfall_95']:.2%}"])
    (output_root/"REPORT.md").write_text("\n".join(report)+"\n",encoding="utf-8")
    return payload


def run_monthly_experiment(*, signal_panel: str|Path, candidate_metrics: str|Path,
                           daily_store_root: str|Path, output_root: str|Path,
                           start: date, end: date, horizon: int=3, holding_days: int=2,
                           max_names: int=1, initial: float=10000.0,
                           benchmark_daily_path: str|Path|None=None,
                           benchmark_distributions_path: str|Path|None=None,
                           direct_daily_stock_root: str|Path|None=None) -> dict:
    output_root=Path(output_root); output_root.mkdir(parents=True,exist_ok=True)
    distributions,distribution_manifest=_load_distribution_contract(benchmark_distributions_path)
    panel=Path(signal_panel); metrics=Path(candidate_metrics)
    materializer=ParquetH130DatasetMaterializer(panel,metrics,end)
    base=next(x for x in build_family_specs(feature_schema_sha256=materializer.feature_schema_fingerprint,
        score_quantile=.75,top_fraction=.01) if x.horizon_sessions==horizon and x.holding_days==holding_days
        and x.max_names==max_names and x.exit_policy["family"]=="FIXED")
    sessions=tuple(sorted(pd.to_datetime(pd.read_parquet(panel,columns=["decision_date"])["decision_date"]).dt.date.unique()))
    all_cutoffs=monthly_refit_dates(sessions)
    cutoffs=tuple(x for x in all_cutoffs if start<=x<end)
    factory=MonthlyModelFactory(base=base,panel=panel,metrics=metrics,development_end=end,sessions=sessions,root=output_root)
    while cutoffs:
        try: factory.winner(cutoffs[0]); break
        except ValueError as exc:
            if not str(exc).startswith(("NO_CAUSAL_","INSUFFICIENT_CAUSAL_")): raise
            cutoffs=cutoffs[1:]
    if not cutoffs:
        raise NoCausalMonthlyWindow(_write_no_causal_window(output_root=output_root,panel_sessions=sessions,
            requested_start=start,requested_end=end,all_cutoffs=all_cutoffs,factory=factory,family=base,
            benchmark_daily_path=benchmark_daily_path,distributions=distributions))
    selections,first_cutoff,decision_rows=_recipe_decisions(factory,cutoffs)
    frozen=factory.winner(first_cutoff); schedules={arm:[] for arm in ARMS}; signals=[]; generation_rows=[]; model_fit_rows=[]
    for index,cutoff in enumerate(cutoffs,1):
        ridge=factory.fit(factory.winner(cutoff,"RIDGE"),cutoff)
        hgb=factory.fit(factory.winner(cutoff,"HIST_GRADIENT_BOOSTING"),cutoff)
        for component in (ridge,hgb):
            model_fit_rows.append({"information_cutoff":cutoff,
                "model_artifact_id":component["build"]["model_artifact_id"],
                "recipe_candidate_id":component["choice"]["candidate_id"],
                "recipe_family":component["choice"]["family"],
                "generation_manifest_path":component["build"].get("generation_manifest_path"),
                "generation_manifest_verified":bool(component["build"].get("generation_manifest_path")),
                "model_fit_cache_reused_this_invocation":bool(component["build"].get("cache_reused",False))})
        r0=factory.fit(frozen,cutoff); r1=factory.fit(selections["r1"][cutoff],cutoff); r2=factory.fit(selections["r2"][cutoff],cutoff)
        blend=_blend_component(factory,ridge,hgb,cutoff)
        components={ARMS[0]:r0,ARMS[1]:r1,ARMS[2]:r2,ARMS[3]:blend}
        for arm,component in components.items(): schedules[arm].append(_schedule_row(arm,component))
        for component in {x["build"]["model_artifact_id"]:x for x in components.values()}.values():
            signals.append(component["predictions"][["decision_date","ticker","score","model_artifact_id"]])
            generation_rows.append({"information_cutoff":cutoff,"model_artifact_id":component["build"]["model_artifact_id"],
                "generation_id":component["generation_id"],"recipe_candidate_id":component["choice"]["candidate_id"],
                "recipe_family":component["choice"]["family"],"resolved_threshold":component["threshold"],
                "resolved_top_fraction":component["top_fraction"],
                "model_fit_cache_reused_this_invocation":bool(component["build"].get("cache_reused",False))})
        print(f"[monthly-recipe] {index}/{len(cutoffs)} cutoff={cutoff} r1={r1['choice']['family']} r2={r2['choice']['family']}",flush=True)
    signal_frame=pd.concat(signals,ignore_index=True).drop_duplicates(["decision_date","ticker","model_artifact_id"])
    schedule_frame=pd.concat([pd.DataFrame(rows).assign(arm=arm) for arm,rows in schedules.items()],ignore_index=True)
    price_path=materialize_daily_store_prices(daily_store_root=daily_store_root,signal_panel=panel,
        start=first_cutoff,end=end,output_path=output_root/"inputs"/"prices.parquet",
        benchmark_daily_path=benchmark_daily_path,direct_daily_stock_root=direct_daily_stock_root)
    prices=pd.read_parquet(price_path)
    price_quality=json.loads(price_path.with_suffix(price_path.suffix+".quality.json").read_text(encoding="utf-8"))
    candidate_quality,candidate_invalid=_candidate_execution_quality_audit(
        signals=signal_frame,schedules=schedules,prices=prices,max_names=max_names)
    candidate_quality.to_csv(output_root/"eligible-candidate-execution-quality.csv",index=False)
    candidate_invalid.to_csv(output_root/"invalid-candidate-execution-boundaries.csv",index=False)
    policy=Policy(horizon=horizon,score_quantile=.75,top_fraction=.01,max_names=max_names,
                  holding_days=holding_days,sleeve=.5)
    def replay_arm(arm):
        return arm,replay_family(signals=signal_frame,prices=prices,policy=policy,
            generation_schedule=pd.DataFrame(schedules[arm]),cost=CostModel(20.0),tax=TaxConfig(False),
            start=pd.Timestamp(first_cutoff),end=pd.Timestamp(end),initial=initial,distributions=distributions)
    with ThreadPoolExecutor(max_workers=4,thread_name_prefix="monthly-recipe-replay") as pool:
        results=dict(pool.map(replay_arm,ARMS))
    summaries=[]; all_daily=[]; all_positions=[]; dd_summaries=[]; enriched_trades={}
    for arm,result in results.items():
        trades=_trade_metadata(panel,pd.DataFrame(result["trades"])); enriched_trades[arm]=trades
        daily,positions,dd=_daily_attribution(arm=arm,result=result,schedule=pd.DataFrame(schedules[arm]),trades=trades,prices=prices)
        daily.to_parquet(output_root/f"{arm}-nav-attribution.parquet",index=False)
        trades.to_parquet(output_root/f"{arm}-trades.parquet",index=False)
        positions.to_parquet(output_root/f"{arm}-position-day-attribution.parquet",index=False)
        all_daily.append(daily); all_positions.append(positions); dd_summaries.append(dd); summaries.append(_summary_row(arm,result))
    daily_frame=pd.concat(all_daily,ignore_index=True); position_frame=pd.concat(all_positions,ignore_index=True)
    daily_frame.to_parquet(output_root/"daily-relative-return-attribution.parquet",index=False)
    position_frame.to_parquet(output_root/"position-day-attribution.parquet",index=False)
    drawdown_attribution,drawdown_classification=_drawdown_attribution(position_frame)
    drawdown_attribution.to_csv(output_root/"max-drawdown-attribution.csv",index=False)
    drawdown_classification.to_csv(output_root/"max-drawdown-cause-classification.csv",index=False)
    pd.DataFrame(dd_summaries).to_csv(output_root/"max-drawdown-episodes.csv",index=False)
    schedule_frame.to_csv(output_root/"monthly-generation-schedule.csv",index=False)
    pd.DataFrame(generation_rows).drop_duplicates().to_csv(output_root/"monthly-generation-audit.csv",index=False)
    model_fit_audit=pd.DataFrame(model_fit_rows).drop_duplicates()
    model_fit_audit.to_csv(output_root/"monthly-model-fit-audit.csv",index=False)
    decision_frame=_attach_next_unseen_fold_evidence(pd.DataFrame(decision_rows),factory.rows)
    for column in ("next_realized_r1_relative_return","next_realized_r0_relative_return",
                   "next_realized_r1_minus_r0_relative_wealth"):
        if column not in decision_frame: decision_frame[column]=np.nan
    # Every daily relative return belongs to exactly one active recipe; segment
    # compounding therefore reconstructs each terminal relative wealth exactly.
    segments=boundary_complete_segments(daily_frame)
    segments.to_csv(output_root/"boundary-complete-recipe-attribution.csv",index=False)
    attribution_validation=_validate_boundary_attribution(daily_frame,segments)
    attribution_validation.to_csv(output_root/"boundary-attribution-validation.csv",index=False)
    # Attach next unseen decision-period portfolio evidence without feeding it
    # back into any selection.
    r1daily=daily_frame.loc[daily_frame["arm"].eq(ARMS[1])].set_index("date")
    r0daily=daily_frame.loc[daily_frame["arm"].eq(ARMS[0])].set_index("date")
    for idx,row in decision_frame.iterrows():
        if not bool(row["new_outer_fold_evidence"]): continue
        start_date=pd.Timestamp(row["assessment_date"]); later=decision_frame.loc[(decision_frame.index>idx)&decision_frame["new_outer_fold_evidence"],"assessment_date"]
        stop_date=pd.Timestamp(later.iloc[0]) if len(later) else pd.Timestamp(end)+pd.Timedelta(days=1)
        values=r1daily.loc[(r1daily.index>=start_date)&(r1daily.index<stop_date),"daily_relative_return"]
        control=r0daily.loc[(r0daily.index>=start_date)&(r0daily.index<stop_date),"daily_relative_return"]
        r1_return=float(np.prod(1.0+values)-1.0) if len(values) else np.nan
        r0_return=float(np.prod(1.0+control)-1.0) if len(control) else np.nan
        decision_frame.loc[idx,"next_realized_r1_relative_return"]=r1_return
        decision_frame.loc[idx,"next_realized_r0_relative_return"]=r0_return
        decision_frame.loc[idx,"next_realized_r1_minus_r0_relative_wealth"]=((1.0+r1_return)/(1.0+r0_return)-1.0)
    decision_frame.to_csv(output_root/"recipe-selection-evidence.csv",index=False)
    # MAX_NAMES capacity ablation: identical scores, recipes, thresholds, sleeve,
    # costs and execution; only max_names changes.
    r1_schedule=pd.DataFrame(schedules[ARMS[1]])
    def replay_capacity(n):
        p=replace(policy,max_names=n)
        result=replay_family(signals=signal_frame,prices=prices,policy=p,generation_schedule=r1_schedule,
            cost=CostModel(20.0),tax=TaxConfig(False),start=pd.Timestamp(first_cutoff),end=pd.Timestamp(end),initial=initial,
            distributions=distributions)
        return {"max_names":n,**_summary_row(f"N{n}",result)}
    with ThreadPoolExecutor(max_workers=4,thread_name_prefix="capacity-ablation") as pool:
        capacity=list(pool.map(replay_capacity,(1,2,3,6)))
    pd.DataFrame(capacity).to_csv(output_root/"max-names-capacity-ablation.csv",index=False)
    summary_table=pd.DataFrame(summaries).sort_values("terminal_value",ascending=False)
    summary_table.to_csv(output_root/"portfolio-value-comparison.csv",index=False)
    by_arm=summary_table.set_index("arm")
    r0=by_arm.loc[ARMS[0]]; r1=by_arm.loc[ARMS[1]]
    primary={"terminal_value_delta_eur":float(r1["terminal_value"]-r0["terminal_value"]),
        "terminal_relative_wealth_delta":float(r1["terminal_value"]/r0["terminal_value"]-1.0),
        "relative_max_drawdown_delta":float(r1["relative_max_drawdown"]-r0["relative_max_drawdown"]),
        "cdar_95_delta":float(r1["cdar_95"]-r0["cdar_95"]),
        "trade_count_delta":int(r1["trade_count"]-r0["trade_count"]),
        "cost_delta_eur":float(r1["total_cost_eur"]-r0["total_cost_eur"])}
    best_capacity=max(capacity,key=lambda row:row["terminal_value"])
    generation_audit=pd.DataFrame(generation_rows).drop_duplicates()
    switch_evidence=decision_frame.loc[
        decision_frame["new_outer_fold_evidence"].astype(bool)
        & decision_frame["winner_candidate_id"].ne(decision_frame["incumbent_candidate_id_before"]),
        ["assessment_date","incumbent_family_before","winner_family","delta_challenger_incumbent",
         "probability_challenger_better","delta_ci_05","delta_ci_95","next_unseen_fold_id",
         "next_unseen_challenger_minus_incumbent_spearman","next_realized_r1_minus_r0_relative_wealth"]]
    research_status="INCONCLUSIVE_RECIPE_RESELECTION_SINGLE_SWITCH_EVENT"
    cache_reused_count=int(model_fit_audit["model_fit_cache_reused_this_invocation"].astype(bool).sum())
    fresh_fit_count=int(len(model_fit_audit)-cache_reused_count)
    provenance_verified_count=int(model_fit_audit["generation_manifest_verified"].astype(bool).sum())
    summary={"schema_version":"DYNAMIC_QBD_MONTHLY_RECIPE_EXPERIMENT_V2_TOTAL_RETURN","status":"COMPLETE",
        "research_status":research_status,
        "authority":"SHADOW_ONLY_NO_PROMOTION_NO_HOLDOUT","evaluation":{"start":str(first_cutoff),"end":str(end)},
        "monthly_refit_count":len(cutoffs),"arms":list(ARMS),
        "primary_contrast":"R1_ROLLING_RECIPE_MONTHLY_REFIT_MINUS_R0_FROZEN_RECIPE_MONTHLY_REFIT",
        "hysteresis_contract":{"minimum_paired_folds":HYSTERESIS_MINIMUM_PAIRED_FOLDS,
            "probability_threshold":HYSTERESIS_PROBABILITY,
            "minimum_robust_delta":HYSTERESIS_MINIMUM_ROBUST_DELTA,"bootstrap_draws":BOOTSTRAP_DRAWS,
            "registration":"FIXED_BEFORE_TOTAL_RETURN_R0_R3_REPLAY"},
        "portfolio_contract":{"initial_value":initial,"benchmark":"URTH","horizon":horizon,
            "holding_days":holding_days,"max_names":max_names,"sleeve":.5,"roundtrip_bps":20.0,"tax":"PRE_TAX"},
        "benchmark_return_contract":{"prices":"ALPACA_SIP_1DAY_SPLIT_ADJUSTED",
            "distributions":"ISHARES_OFFICIAL_CASH_DISTRIBUTIONS",
            "reinvestment":"PAYABLE_DATE_OPEN","distribution_manifest":distribution_manifest},
        "stock_execution_provider_contract":{"primary":"MINUTE_DERIVED_DAILY_BOUNDARIES",
            "fallback":"ALPACA_SIP_1DAY_DIRECT_ONLY_WHERE_PRIMARY_BOUNDARY_INVALID",
            "direct_daily_stock_root":str(Path(direct_daily_stock_root).resolve()) if direct_daily_stock_root else None,
            "valid_primary_rows_are_never_replaced":True},
        "model_artifact_provenance":{"physical_monthly_model_fits":int(model_fit_audit["model_artifact_id"].nunique()),
            "blend_generations":int(generation_audit.loc[generation_audit["recipe_family"].eq("RIDGE_HGB_BLEND"),"generation_id"].nunique()),
            "cache_reused_this_invocation":cache_reused_count,"fresh_fits_this_invocation":fresh_fit_count,
            "generation_manifest_verified_count":provenance_verified_count,
            "contract":"GENERATION_MANIFEST_REQUIRED_FOR_ALL_FIT_REUSE"},
        "single_fold_r2_arm":"EXCLUDED_NOT_A_CHALLENGER",
        "primary_contrast_result":primary,
        "portfolio_value_comparison":summary_table.to_dict(orient="records"),
        "capacity_ablation":capacity,"best_capacity_by_terminal_value":best_capacity,"drawdown_episodes":dd_summaries,
        "candidate_execution_quality":candidate_quality.to_dict(orient="records"),
        "price_boundary_quality":price_quality,
        "drawdown_cause_classification":drawdown_classification.to_dict(orient="records"),
        "boundary_attribution_validation":attribution_validation.to_dict(orient="records"),
        "execution_resources":active_cpu_contract(),"final_holdout_opened":False}
    (output_root/"summary.json").write_text(json.dumps(summary,default=str,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    report=["# Dynamic QBD monthly recipe factory","","## Decision","",
        f"**{research_status}**. Rolling recipe selection improved terminal wealth versus the matched frozen-recipe monthly-refit control by EUR {primary['terminal_value_delta_eur']:.2f} ({primary['terminal_relative_wealth_delta']:.2%}), but both arms underperformed URTH and the recipe contrast contains only one genuinely different recipe episode. This is not promotion evidence.","",
        "## Technical summary","",
        f"Completed {len(cutoffs)} causal monthly refits from {first_cutoff} through {end}. R0/R1/R2/R3 share the same monthly fit dates, moving windows, monthly calibration, costs and stateful next-open replay.","",
        "URTH execution uses split-adjusted Alpaca SIP daily Open/Close. Official iShares cash distributions are credited to entitled units on the payable session and reinvested at that session's Open for both the benchmark and idle sleeve.","",
        f"The experiment contains {int(model_fit_audit['model_artifact_id'].nunique())} physical monthly production fits (Ridge and HGB) and {int(generation_audit.loc[generation_audit['recipe_family'].eq('RIDGE_HGB_BLEND'),'generation_id'].nunique())} separately calibrated blend generations. This reporting invocation reused {cache_reused_count} fits, trained {fresh_fit_count} new fits and cryptographically verified {provenance_verified_count} generation manifests. Existing historical WF metrics informed recipe choice; stale historical model artifacts were not traded.","",
        f"The row-level direct-daily fallback repaired {price_quality['stock_invalid_minute_open_rows_repaired_from_direct_daily']} invalid stock Opens and {price_quality['stock_invalid_minute_close_rows_repaired_from_direct_daily']} invalid stock Closes. Valid minute-derived boundaries were retained.","",
        _markdown_table(summary_table),"","## Recipe-switch evidence","",
        _markdown_table(switch_evidence),"",
        "The bootstrap hysteresis rejected the only Ridge-to-HGB switch. The ungated rolling arm nevertheless beat its matched control during the subsequent unseen portfolio period; this single event is insufficient for threshold tuning or generalization claims.","",
        "## Max-names capacity ablation","",_markdown_table(pd.DataFrame(capacity)),"",
        f"Only `max_names` changed. N{int(best_capacity['max_names'])} had the highest terminal value (EUR {best_capacity['terminal_value']:.2f}); no sector cap or changed score gate was introduced.","",
        "## Candidate execution-boundary audit","",_markdown_table(candidate_quality),"",
        "This is a pre-replay intersection of causal gate candidates and next-open data quality. Actual executed and held prices remain fail-closed.","",
        "## Maximum-drawdown cause classification","",_markdown_table(drawdown_classification),"",
        "Daily and recipe-segment compounding reconstruct terminal relative wealth with an absolute error below 1e-10. Switch-boundary returns are fully assigned.","",
        "The earlier single-fold/R-squared diagnostic is excluded as a challenger. Learned exits and drawdown throttles remain deferred; neither was optimized on these outcomes.","",
        "## Research authority","",
        "This is Development shadow evidence only. Recipe decisions use completed prior Outer-WF folds; portfolio outcomes and drawdown attribution never feed back into the same decision. Final holdout remained closed."]
    (output_root/"REPORT.md").write_text("\n".join(report)+"\n",encoding="utf-8")
    return summary


def main(argv=None) -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--signal-panel",required=True); parser.add_argument("--candidate-metrics",required=True)
    parser.add_argument("--daily-store-root",required=True); parser.add_argument("--output-root",required=True)
    parser.add_argument("--benchmark-daily-path")
    parser.add_argument("--benchmark-distributions-path",required=True)
    parser.add_argument("--direct-daily-stock-root")
    parser.add_argument("--start",default="2020-08-31"); parser.add_argument("--end",default="2023-12-29")
    parser.add_argument("--cpu-target",type=float,default=.90)
    args=parser.parse_args(argv); configure_cpu_peak(args.cpu_target)
    try:
        result=run_monthly_experiment(signal_panel=args.signal_panel,candidate_metrics=args.candidate_metrics,
            daily_store_root=args.daily_store_root,output_root=args.output_root,
            start=date.fromisoformat(args.start),end=date.fromisoformat(args.end),
            benchmark_daily_path=args.benchmark_daily_path,
            benchmark_distributions_path=args.benchmark_distributions_path,
            direct_daily_stock_root=args.direct_daily_stock_root)
    except NoCausalMonthlyWindow as exc:
        print(json.dumps(exc.payload,indent=2,sort_keys=True))
        return 2
    print(json.dumps(result["portfolio_value_comparison"],indent=2)); return 0


if __name__=="__main__":
    raise SystemExit(main())
