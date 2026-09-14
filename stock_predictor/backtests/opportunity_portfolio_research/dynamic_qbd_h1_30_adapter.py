"""Concrete causal H1-H30 production-generation adapter.

This module consumes the locked H1-H30 signal panel and historical OOS
candidate evidence.  It never calls the legacy H5/H10/H20 V5 entrypoint.
By default the first refit freezes a candidate recipe using only completed
prior OOS folds.  Shadow families may explicitly request causal recipe
reselection at each refit; every resulting generation is still freshly fit
on the moving FamilySpec window, calibrated on a disjoint later window, and
writes generation-specific scores.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
from typing import Any, Mapping
from functools import cached_property

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .contract_fingerprints import stable_hash


IDENTITY_COLUMNS = ("decision_date", "ticker", "sector", "sub_industry")
FEATURE_COLUMNS = (
    "open", "high", "low", "close", "volume", "mom5", "mom20", "mom60", "mom120",
    "trend20", "trend100", "volatility20", "volatility60", "avg_dollar_volume20",
    "volume_ratio20_60", "relative_strength20", "relative_strength60", "benchmark_mom20",
    "benchmark_mom60", "benchmark_mom120", "benchmark_trend100", "benchmark_volatility20",
    "market_breadth_mom20", "market_breadth_mom120", "ohlcv_open_gap", "ohlcv_session_return",
    "ohlcv_intraday_range", "ohlcv_close_location", "ohlcv_first5_impact", "ohlcv_first5_range",
    "ohlcv_first5_volume_share", "ohlcv_log_session_volume", "ohlcv_log_first5_dollar_volume",
    "ohlcv_target_participation_bps", "ohlcv_regular_minute_count", "ohlcv_first5_minute_coverage",
)
_FOLD_END = re.compile(r"_(\d{4}-\d{2}-\d{2})$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _frame_sha256(frame: pd.DataFrame, columns) -> str:
    ordered = frame.loc[:, list(columns)].sort_values(["decision_date", "ticker"]).reset_index(drop=True)
    values = pd.util.hash_pandas_object(ordered, index=False).to_numpy(dtype=np.uint64)
    digest = hashlib.sha256(values.tobytes())
    digest.update(json.dumps([(x, str(ordered[x].dtype)) for x in ordered], sort_keys=True).encode())
    return digest.hexdigest()


def _robust_statistics(rows: list[Mapping[str, Any]]) -> dict[str, float]:
    values = np.asarray([float(x["metrics"]["spearman"]) for x in rows
                         if x.get("metrics", {}).get("spearman") is not None], dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"mean_spearman": -1.0, "median_spearman": -1.0, "std_spearman": 1.0,
                "negative_fold_rate": 1.0, "robust_score": -2.0, "mean_mae": np.inf}
    mean, median, std = float(values.mean()), float(np.median(values)), float(values.std())
    negative = float(np.mean(values <= 0))
    r2=np.asarray([float(x["metrics"].get("r2",np.nan)) for x in rows],dtype=float)
    r2=r2[np.isfinite(r2)]
    return {"mean_spearman": mean, "median_spearman": median, "std_spearman": std,
            "negative_fold_rate": negative,
            "robust_score": median + .5 * mean - .5 * std - .02 * negative,
            "mean_mae": float(np.mean([float(x["metrics"]["mae"]) for x in rows])),
            "mean_r2_diagnostic":float(r2.mean()) if len(r2) else float("nan")}


@dataclass(frozen=True)
class ParquetH130DatasetMaterializer:
    signal_panel_path: Path
    candidate_metrics_path: Path
    development_end: date | None = None
    learned_exit_candidate_metrics_path: Path | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "signal_panel_path", Path(self.signal_panel_path))
        object.__setattr__(self, "candidate_metrics_path", Path(self.candidate_metrics_path))
        if self.learned_exit_candidate_metrics_path is not None:
            object.__setattr__(self, "learned_exit_candidate_metrics_path", Path(self.learned_exit_candidate_metrics_path))
        if not self.signal_panel_path.is_file() or not self.candidate_metrics_path.is_file():
            raise FileNotFoundError("H1_30_MATERIALIZER_INPUT_MISSING")

    @cached_property
    def dataset_fingerprint(self) -> str:
        return _sha256(self.signal_panel_path)

    @cached_property
    def feature_schema_fingerprint(self) -> str:
        schema = pq.ParquetFile(self.signal_panel_path).schema_arrow
        return stable_hash([(name, str(schema.field(name).type)) for name in FEATURE_COLUMNS if name in schema.names])

    def materialize(self, *, family, information_cutoff, latest_matured_label_cutoff, output_root: Path) -> dict:
        target = f"net_excess_return_{family.horizon_sessions}__BASELINE_20_BPS"
        schema = set(pq.ParquetFile(self.signal_panel_path).schema_arrow.names)
        columns = [x for x in (*IDENTITY_COLUMNS, *FEATURE_COLUMNS, target) if x in schema]
        missing = {"decision_date", "ticker", target, *FEATURE_COLUMNS} - set(columns)
        if missing:
            raise ValueError(f"H1_30_SIGNAL_PANEL_COLUMNS_MISSING:{sorted(missing)}")
        if str(family.feature_schema_sha256) != self.feature_schema_fingerprint:
            raise ValueError("H1_30_FEATURE_SCHEMA_FINGERPRINT_MISMATCH")
        output_root.mkdir(parents=True, exist_ok=True)
        return {
            "signal_panel_path": str(self.signal_panel_path),
            "candidate_metrics_path": str(self.candidate_metrics_path),
            "target": target, "feature_columns": list(FEATURE_COLUMNS),
            "source_panel_sha256": self.dataset_fingerprint,
            "latest_matured_label_cutoff": latest_matured_label_cutoff.isoformat(),
            "development_end": self.development_end.isoformat() if self.development_end else None,
            "learned_exit_candidate_metrics_path": (str(self.learned_exit_candidate_metrics_path)
                                                       if self.learned_exit_candidate_metrics_path else None),
        }


class H130ProductionGenerationBuilder:
    def __init__(self, *, materializer: ParquetH130DatasetMaterializer, root: str | Path, code_commit: str):
        if not code_commit:
            raise ValueError("DYNAMIC_QBD_CODE_COMMIT_REQUIRED")
        self.materializer = materializer
        self.root = Path(root)
        self.code_commit = str(code_commit)
        self._builds: dict[tuple[str, date], dict] = {}
        self._shared_signal_fits: dict[str, dict] = {}
        self._shared_learned_exit_fits: dict[str, dict] = {}
        self._panel_cache: dict[tuple[str, tuple[str, ...]], pd.DataFrame] = {}
        self._candidate_cache: dict[str, dict] = {}
        self._frozen_choice_by_family: dict[str, tuple[date, dict]] = {}
        self._frozen_exit_choice_by_family_horizon: dict[tuple[str, int], tuple[date, dict]] = {}
        self._recalibration_cache: dict[str, pd.DataFrame] = {}
        self._file_hash_cache: dict[Path, str] = {}

    @property
    def _shared_fit_root(self) -> Path:
        """Persistent cache for H×cutoff fits shared by D/N families/processes."""
        return self.root.parent / "shared-fits"

    @staticmethod
    def _link_or_copy(source: Path, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            return
        try:
            os.link(source, destination)
        except OSError:
            shutil.copy2(source, destination)

    def _restore_shared_fit(self, shared_key: str, destination_root: Path) -> dict | None:
        cache_root = self._shared_fit_root / shared_key
        manifest_path = cache_root / "fit-result.json"
        if not manifest_path.is_file():
            return None
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("shared_fit_key") != shared_key:
                return None
            if stable_hash({k: v for k, v in manifest.items() if k != "manifest_sha256"}) != manifest.get("manifest_sha256"):
                return None
            result = dict(manifest["result"])
            for field, filename, digest_key in (
                ("model_path", "model.joblib", "model_sha256"),
                ("calibration_prediction_path", "calibration-predictions.parquet", "calibration_sha256"),
                ("raw_prediction_path", "generation-predictions.parquet", "prediction_sha256"),
            ):
                source = cache_root / filename
                if not source.is_file() or _sha256(source) != manifest.get(digest_key):
                    return None
                target = destination_root / filename
                self._link_or_copy(source, target)
                result[field] = str(target)
            return result
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            return None

    def _persist_shared_fit(self, shared_key: str, result: dict) -> None:
        """Publish a completed fixed-exit fit atomically for future workers."""
        if result.get("raw_exit_prediction_path"):
            return
        cache_root = self._shared_fit_root / shared_key
        manifest_path = cache_root / "fit-result.json"
        if manifest_path.is_file():
            return
        staging = self._shared_fit_root / f".{shared_key}.{os.getpid()}.tmp"
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True, exist_ok=False)
        files = {
            "model_path": ("model.joblib", "model_sha256"),
            "calibration_prediction_path": ("calibration-predictions.parquet", "calibration_sha256"),
            "raw_prediction_path": ("generation-predictions.parquet", "prediction_sha256"),
        }
        published = dict(result)
        hashes = {}
        for field, (filename, digest_key) in files.items():
            source = Path(result[field])
            target = staging / filename
            shutil.copy2(source, target)
            hashes[digest_key] = _sha256(target)
            published[field] = str(cache_root / filename)
        manifest = {"schema_version": "DYNAMIC_QBD_SHARED_SIGNAL_FIT_V1",
                    "shared_fit_key": shared_key, "result": published, **hashes}
        manifest["manifest_sha256"] = stable_hash(manifest)
        (staging / "fit-result.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        try:
            cache_root.parent.mkdir(parents=True, exist_ok=True)
            os.replace(staging, cache_root)
        except FileExistsError:
            shutil.rmtree(staging, ignore_errors=True)

    def _file_hash(self, path: Path) -> str:
        resolved = path.resolve()
        if resolved not in self._file_hash_cache:
            self._file_hash_cache[resolved] = _sha256(resolved)
        return self._file_hash_cache[resolved]

    def restore_generation(self, generation) -> None:
        """Restore the minimum builder state needed after a factory restart."""
        prediction_path = Path(generation.prediction_artifact_path)
        model_path = prediction_path.with_name("model.joblib")
        calibration_path = prediction_path.with_name("calibration-predictions.parquet")
        for path in (prediction_path, model_path, calibration_path):
            if not path.is_file():
                raise FileNotFoundError(f"GENERATION_RESTART_ARTIFACT_MISSING:{path}")
        # Many immutable generations intentionally point at the same shared
        # model artifact. Reuse the builder's content-hash cache instead of
        # rereading the same multi-megabyte joblib once per generation.
        if self._file_hash(model_path) != generation.model_artifact_sha256:
            raise ValueError("GENERATION_RESTART_MODEL_HASH_MISMATCH")
        self._builds[(generation.family_id, generation.information_cutoff)] = {
            "train_start":str(generation.train_start),"train_end":str(generation.train_end),
            "dataset_fingerprint":generation.dataset_fingerprint,
            "model_artifact_sha256":generation.model_artifact_sha256,
            "model_artifact_id":generation.model_artifact_id,
            "selected_model_family":generation.selected_model_family,
            "selected_hyperparameters":dict(generation.selected_hyperparameters),
            "training_recipe_fingerprint":generation.training_recipe_fingerprint,
            "model_path":str(model_path),"calibration_prediction_path":str(calibration_path),
            "raw_prediction_path":str(prediction_path),
        }
        prior=self._frozen_choice_by_family.get(generation.family_id)
        if prior is None or generation.information_cutoff < prior[0]:
            self._frozen_choice_by_family[generation.family_id]=(generation.information_cutoff,{
                "family":generation.selected_model_family,
                "parameters":dict(generation.selected_hyperparameters),
                "fold_count":int(getattr(generation,"selection_oos_fold_count",0)),
                "negative_fold_rate":1.0-float(getattr(generation,"selection_oos_positive_fold_fraction",0.0)),
                "fold_ids":tuple(getattr(generation,"selection_oos_fold_ids",())),
                "selection_metric_contract":str(getattr(generation,"selection_metric_contract","")),
                "model_combination_contract":str(getattr(generation,"model_combination_contract","")),
                "selection_contract":"RESTORED_FROZEN_FAMILY_RECIPE",
            })
        for horizon, choice in dict(getattr(generation,"exit_model_recipes",{})).items():
            key=(generation.family_id,int(horizon))
            prior_exit=self._frozen_exit_choice_by_family_horizon.get(key)
            if prior_exit is None or generation.information_cutoff < prior_exit[0]:
                self._frozen_exit_choice_by_family_horizon[key]=(generation.information_cutoff,dict(choice))

    @staticmethod
    def _candidate_allowed(family, row: Mapping[str, Any]) -> bool:
        requested = str(family.model_family).upper()
        actual = str(row["family"]).upper()
        if requested not in {"RIDGE_HGB_FROZEN_RULE", "RIDGE_HGB"} and actual != requested:
            return False
        allowed = tuple(str(x).upper() for x in family.hyperparameter_rule.get("candidate_models", ("RIDGE", "HGB")))
        normalized = "HGB" if actual == "HIST_GRADIENT_BOOSTING" else actual
        return normalized in allowed

    def candidate_choices(self, family, metrics_path: Path, latest_matured: date, *, horizon: int | None = None) -> tuple[dict, ...]:
        horizon = int(horizon or family.horizon_sessions)
        rows = json.loads(metrics_path.read_text(encoding="utf-8"))
        grouped: dict[str, list[dict]] = {}
        metadata: dict[str, dict] = {}
        for row in rows:
            match = _FOLD_END.search(str(row.get("fold_id", "")))
            if (int(row.get("horizon_sessions", -1)) != horizon or row.get("selection_only")
                    or match is None or pd.Timestamp(match.group(1)).date() > latest_matured
                    or not self._candidate_allowed(family, row)):
                continue
            identifier = str(row["candidate_id"])
            grouped.setdefault(identifier, []).append(row)
            metadata[identifier] = row
        choices = []
        for identifier, evidence in grouped.items():
            row = metadata[identifier]
            choices.append({"candidate_id": identifier, "family": str(row["family"]),
                            "parameters": dict(row["parameters"]), "fold_count": len(evidence),
                            "fold_ids":tuple(sorted(str(x["fold_id"]) for x in evidence)),
                            "selection_metric_contract":"ROBUST_FOLD_SPEARMAN_THEN_MAE_TIEBREAK_R2_DIAGNOSTIC_ONLY",
                            "model_combination_contract":"SINGLE_SELECTED_RECIPE_NO_ENSEMBLE_NO_MODEL_INTERSECTION",
                            **_robust_statistics(evidence)})
        if not choices:
            raise ValueError(f"NO_CAUSAL_H1_30_CANDIDATE_EVIDENCE:H{horizon}")
        minimum_folds = int(family.hyperparameter_rule.get("minimum_oos_folds", 2))
        eligible = [x for x in choices if x["fold_count"] >= minimum_folds]
        if not eligible:
            raise ValueError(f"INSUFFICIENT_CAUSAL_MODEL_SELECTION_FOLDS:H{horizon}")
        return tuple(sorted(eligible,key=lambda x:(x["candidate_id"],x["family"])))

    def _select_candidate(self, family, metrics_path: Path, latest_matured: date, *, horizon: int | None = None) -> dict:
        horizon = int(horizon or family.horizon_sessions)
        cache_key = stable_hash({"metrics":str(metrics_path.resolve()),"metrics_sha256":self._file_hash(metrics_path),
                                 "latest_matured":latest_matured,"horizon":horizon,
                                 "model_family":family.model_family,"rule":family.hyperparameter_rule})
        if cache_key in self._candidate_cache:
            return dict(self._candidate_cache[cache_key])
        eligible=list(self.candidate_choices(family,metrics_path,latest_matured,horizon=horizon))
        selected = max(eligible, key=lambda x: (x["robust_score"], x["median_spearman"], -x["mean_mae"], x["candidate_id"]))
        self._candidate_cache[cache_key] = dict(selected)
        return selected

    @staticmethod
    def _model(choice: Mapping[str, Any], seed: int):
        parameters = choice["parameters"]
        if choice["family"] == "RIDGE":
            pipeline = Pipeline([("imputer", SimpleImputer(strategy="median")),
                                 ("scaler", StandardScaler()),
                                 ("model", Ridge(alpha=float(parameters["alpha"])))])
            return TransformedTargetRegressor(regressor=pipeline, transformer=StandardScaler())
        if choice["family"] == "HIST_GRADIENT_BOOSTING":
            model = HistGradientBoostingRegressor(
                learning_rate=float(parameters["learning_rate"]), max_iter=int(parameters["max_iter"]),
                max_leaf_nodes=int(parameters["max_leaf_nodes"]),
                l2_regularization=float(parameters["l2_regularization"]), random_state=int(seed),
            )
            return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", model)])
        raise ValueError(f"UNSUPPORTED_H1_30_PRODUCTION_MODEL:{choice['family']}")

    def build(self, *, family, information_cutoff, latest_matured_label_cutoff,
              prediction_end=None, prediction_end_exclusive=None):
        root = self.root / family.family_id / information_cutoff.isoformat()
        paths = self.materializer.materialize(family=family, information_cutoff=information_cutoff,
                                              latest_matured_label_cutoff=latest_matured_label_cutoff,
                                              output_root=root)
        recipe_selection_contract=str(family.hyperparameter_rule.get(
            "recipe_selection_contract", "FIRST_CAUSAL_REFIT_THEN_FROZEN_WITHIN_FAMILY"
        ))
        allowed_selection_contracts={
            "FIRST_CAUSAL_REFIT_THEN_FROZEN_WITHIN_FAMILY",
            "CAUSAL_RESELECT_AT_EACH_REFIT",
        }
        if recipe_selection_contract not in allowed_selection_contracts:
            raise ValueError(f"UNSUPPORTED_RECIPE_SELECTION_CONTRACT:{recipe_selection_contract}")
        frozen=self._frozen_choice_by_family.get(family.family_id)
        if frozen is None or recipe_selection_contract == "CAUSAL_RESELECT_AT_EACH_REFIT":
            choice=self._select_candidate(family,Path(paths["candidate_metrics_path"]),latest_matured_label_cutoff)
            choice={**choice,"recipe_frozen_at":information_cutoff.isoformat(),
                    "selection_contract":recipe_selection_contract}
            if recipe_selection_contract == "FIRST_CAUSAL_REFIT_THEN_FROZEN_WITHIN_FAMILY":
                self._frozen_choice_by_family[family.family_id]=(information_cutoff,dict(choice))
        else:
            choice=dict(frozen[1])
        shared_key = stable_hash({
            "cutoff": information_cutoff, "horizon": family.horizon_sessions,
            "feature_schema": family.feature_schema_sha256, "model_family": family.model_family,
            "training_recipe": family.training_recipe, "hyperparameter_rule": family.hyperparameter_rule,
            "training_window_sessions": family.training_window_sessions,
            "calibration_window_sessions": family.calibration_window_sessions,
            "choice": choice, "source_contract": (str(Path(paths["signal_panel_path"]).resolve()),
                                                     family.feature_schema_sha256), "seed": family.random_seed,
        })
        if family.exit_policy.get("family") != "LEARNED_EXIT" and shared_key in self._shared_signal_fits:
            result = dict(self._shared_signal_fits[shared_key])
            self._builds[(family.family_id, information_cutoff)] = result
            return result
        if family.exit_policy.get("family") != "LEARNED_EXIT":
            restored = self._restore_shared_fit(shared_key, root)
            if restored is not None:
                self._shared_signal_fits[shared_key] = dict(restored)
                self._builds[(family.family_id, information_cutoff)] = restored
                return restored
        learned_key = stable_hash({"signal_fit":shared_key,"holding_days":family.holding_days,
                                   "exit_policy":family.exit_policy})
        if family.exit_policy.get("family") == "LEARNED_EXIT" and learned_key in self._shared_learned_exit_fits:
            result = dict(self._shared_learned_exit_fits[learned_key])
            self._builds[(family.family_id, information_cutoff)] = result
            return result
        exit_horizons = tuple(range(1, family.holding_days)) if family.exit_policy.get("family") == "LEARNED_EXIT" else ()
        exit_targets = [f"gross_excess_return_{h}" for h in exit_horizons]
        columns = ["decision_date", "ticker", *[x for x in IDENTITY_COLUMNS[2:]], *paths["feature_columns"],
                   paths["target"], *exit_targets]
        panel_key = (str(Path(paths["signal_panel_path"]).resolve()), tuple(columns))
        frame = self._panel_cache.get(panel_key)
        if frame is None:
            frame = pd.read_parquet(paths["signal_panel_path"], columns=columns)
            frame["decision_date"] = pd.to_datetime(frame["decision_date"]).dt.normalize()
            self._panel_cache[panel_key] = frame
        all_dates = np.asarray(sorted(frame["decision_date"].unique()))
        dates = np.asarray([x for x in all_dates if pd.Timestamp(x) <= pd.Timestamp(latest_matured_label_cutoff)])
        calibration_count = int(family.calibration_window_sessions)
        purge = max(int(family.horizon_sessions), int(family.training_recipe.get("purge_sessions", 30)))
        needed = int(family.training_window_sessions) + purge + calibration_count
        if len(dates) < needed:
            raise ValueError(f"INSUFFICIENT_H1_30_PRODUCTION_HISTORY:{len(dates)}<{needed}")
        calibration_dates = dates[-calibration_count:]
        train_end_index = len(dates) - calibration_count - purge
        train_dates = dates[max(0, train_end_index-int(family.training_window_sessions)):train_end_index]
        train = frame.loc[frame["decision_date"].isin(train_dates)]
        calibration = frame.loc[frame["decision_date"].isin(calibration_dates)]
        if train.empty or calibration.empty:
            raise ValueError("H1_30_PRODUCTION_WINDOWS_EMPTY")
        causal_columns = ["decision_date", "ticker", *paths["feature_columns"], paths["target"]]
        dataset_fingerprint = stable_hash({"train":_frame_sha256(train,causal_columns),
                                           "calibration":_frame_sha256(calibration,causal_columns),
                                           "latest_matured_label_cutoff":latest_matured_label_cutoff})
        model = self._model(choice, family.random_seed)
        model.fit(train[paths["feature_columns"]].to_numpy(float), train[paths["target"]].to_numpy(float))
        model_path = root / "model.joblib"; model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_path, compress=3)
        calibration_values = model.predict(calibration[paths["feature_columns"]].to_numpy(float))
        cal = calibration[["decision_date", "ticker", paths["target"]]].copy()
        cal = cal.rename(columns={paths["target"]: "observed_excess"})
        cal["score"] = calibration_values
        sessions = list(all_dates)
        terminal_by_date = {pd.Timestamp(value): pd.Timestamp(sessions[index+family.horizon_sessions])
                            for index, value in enumerate(sessions) if index+family.horizon_sessions < len(sessions)}
        cal["terminal_date"] = cal["decision_date"].map(terminal_by_date)
        calibration_path = root / "calibration-predictions.parquet"
        cal.to_parquet(calibration_path, index=False)
        development_end = pd.Timestamp(paths["development_end"]) if paths.get("development_end") else frame["decision_date"].max()
        scoring_end = pd.Timestamp(prediction_end) if prediction_end is not None else development_end
        scoring_mask = frame["decision_date"].gt(pd.Timestamp(information_cutoff))
        if prediction_end_exclusive is not None:
            scoring_mask &= frame["decision_date"].lt(pd.Timestamp(prediction_end_exclusive))
        else:
            scoring_mask &= frame["decision_date"].le(scoring_end)
        scoring = frame.loc[scoring_mask]
        scores = scoring[["decision_date", "ticker"]].copy()
        scores["score"] = model.predict(scoring[paths["feature_columns"]].to_numpy(float)) if len(scoring) else np.asarray([], dtype=float)
        raw_prediction_path = root / "generation-predictions.parquet"
        scores.to_parquet(raw_prediction_path, index=False)
        exit_payload = {}
        if exit_horizons:
            exit_metrics = paths.get("learned_exit_candidate_metrics_path")
            if not exit_metrics:
                raise ValueError("LEARNED_EXIT_GENERATION_REQUIRES_CAUSAL_CANDIDATE_EVIDENCE")
            exit_rows, exit_hashes, exit_choices, exit_calibration = [], [], {}, {}
            for exit_horizon in exit_horizons:
                exit_key=(family.family_id,int(exit_horizon))
                frozen_exit=self._frozen_exit_choice_by_family_horizon.get(exit_key)
                if frozen_exit is None:
                    choice_exit=self._select_candidate(family,Path(exit_metrics),latest_matured_label_cutoff,
                                                       horizon=exit_horizon)
                    choice_exit={**choice_exit,"recipe_frozen_at":information_cutoff.isoformat(),
                                 "selection_contract":"FIRST_CAUSAL_REFIT_THEN_FROZEN_WITHIN_FAMILY"}
                    self._frozen_exit_choice_by_family_horizon[exit_key]=(information_cutoff,dict(choice_exit))
                else:
                    choice_exit=dict(frozen_exit[1])
                exit_choices[str(exit_horizon)] = choice_exit
                exit_model = self._model(choice_exit, family.random_seed + exit_horizon)
                target = f"gross_excess_return_{exit_horizon}"
                exit_model.fit(train[paths["feature_columns"]].to_numpy(float), train[target].to_numpy(float))
                exit_model_path = root/"exit-models"/f"E{exit_horizon:02d}.joblib"
                exit_model_path.parent.mkdir(parents=True, exist_ok=True)
                joblib.dump(exit_model, exit_model_path, compress=3); exit_hashes.append(_sha256(exit_model_path))
                calibration_scores = exit_model.predict(calibration[paths["feature_columns"]].to_numpy(float))
                exit_calibration[str(exit_horizon)] = {
                    "rows": int(len(calibration_scores)), "score_mean": float(np.mean(calibration_scores)),
                    "score_std": float(np.std(calibration_scores)),
                    "observed_mean": float(calibration[target].mean()),
                    "direction_accuracy": float(np.mean((calibration_scores > 0) == (calibration[target].to_numpy(float) > 0))),
                }
                part = scoring[["decision_date", "ticker"]].copy()
                part["exit_horizon_sessions"] = exit_horizon
                part["predicted_continuation_excess"] = exit_model.predict(
                    scoring[paths["feature_columns"]].to_numpy(float)) if len(scoring) else np.asarray([], dtype=float)
                part["holdout_locked"] = False
                exit_rows.append(part)
            raw_exit_path = root/"generation-exit-predictions.parquet"
            pd.concat(exit_rows, ignore_index=True).to_parquet(raw_exit_path, index=False)
            exit_payload = {"raw_exit_prediction_path": str(raw_exit_path),
                            "exit_model_artifact_sha256": stable_hash(tuple(exit_hashes)),
                            "exit_generation_id": stable_hash({"horizon":family.horizon_sessions,
                                                               "holding_days":family.holding_days,
                                                               "feature_schema":family.feature_schema_sha256,
                                                               "cutoff":information_cutoff,"dataset":dataset_fingerprint,
                                                               "choices":exit_choices,"artifacts":exit_hashes})[:24],
                            "exit_calibration_fingerprint": stable_hash({"cutoff":information_cutoff,
                                                                         "choices":exit_choices,
                                                                         "calibration_statistics":exit_calibration,
                                                                         "calibration_dates":(str(calibration_dates[0]),str(calibration_dates[-1]))}),
                            "exit_model_recipes":exit_choices}
        model_hash = _sha256(model_path)
        model_artifact_id = stable_hash({"horizon":family.horizon_sessions,"feature_schema":family.feature_schema_sha256,
                                         "model_family":family.model_family,"training_recipe":family.training_recipe,
                                         "hyperparameter_rule":family.hyperparameter_rule,
                                         "training_window_sessions":family.training_window_sessions,
                                         "calibration_window_sessions":family.calibration_window_sessions,
                                         "cutoff": information_cutoff,
                                         "choice": choice, "train_dates": (str(train_dates[0]), str(train_dates[-1])),
                                         "dataset": dataset_fingerprint, "seed": family.random_seed})
        result = {
            "train_start": str(pd.Timestamp(train_dates[0]).date()), "train_end": str(pd.Timestamp(train_dates[-1]).date()),
            "dataset_fingerprint": dataset_fingerprint, "model_artifact_sha256": model_hash,
            "model_artifact_id": model_artifact_id, "selected_model_family": choice["family"],
            "selected_hyperparameters": choice["parameters"], "selection_evidence": choice,
            "training_recipe_fingerprint": stable_hash({"code_commit": self.code_commit,
                "horizon": family.horizon_sessions, "model_family":family.model_family,
                "training_recipe":family.training_recipe,"hyperparameter_rule":family.hyperparameter_rule,
                "choice": choice, "features": paths["feature_columns"],
                "train_window": family.training_window_sessions, "calibration_window": family.calibration_window_sessions}),
            "model_path":str(model_path), "shared_signal_fit_key":shared_key,
            "calibration_prediction_path": str(calibration_path), "raw_prediction_path": str(raw_prediction_path),
            **exit_payload,
        }
        if not exit_horizons:
            self._shared_signal_fits[shared_key] = dict(result)
            self._persist_shared_fit(shared_key, result)
        else:
            self._shared_learned_exit_fits[learned_key] = dict(result)
        self._builds[(family.family_id, information_cutoff)] = result
        return result

    def calibration_predictions(self, *, family, build, information_cutoff):
        frame = pd.read_parquet(build["calibration_prediction_path"])
        if frame.empty:
            raise RuntimeError("H1_30_CALIBRATION_PREDICTIONS_EMPTY")
        return frame

    def recalibration_predictions(self, *, family, model_information_cutoff, information_cutoff,
                                  latest_matured_label_cutoff, cache_result=True):
        """Score a new calibration window with one already frozen model."""
        source_build = self._builds.get((family.family_id, model_information_cutoff))
        if source_build is None:
            raise ValueError("FROZEN_MODEL_BUILD_NOT_AVAILABLE_FOR_RECALIBRATION")
        model_path = Path(source_build["model_path"])
        cache_key = stable_hash({"model_artifact_id":source_build["model_artifact_id"],
                                 "information_cutoff":information_cutoff,
                                 "latest_matured":latest_matured_label_cutoff,
                                 "calibration_window":family.calibration_window_sessions})
        if cache_result and cache_key in self._recalibration_cache:
            return self._recalibration_cache[cache_key].copy()
        model = joblib.load(model_path)
        paths = self.materializer.materialize(family=family, information_cutoff=information_cutoff,
                                              latest_matured_label_cutoff=latest_matured_label_cutoff,
                                              output_root=self.root/family.family_id/information_cutoff.isoformat()/"recalibration")
        columns = ["decision_date", "ticker", *paths["feature_columns"], paths["target"]]
        frame = pd.read_parquet(paths["signal_panel_path"], columns=columns)
        frame["decision_date"] = pd.to_datetime(frame["decision_date"]).dt.normalize()
        all_dates = np.asarray(sorted(frame["decision_date"].unique()))
        matured_dates = [x for x in all_dates if pd.Timestamp(x) <= pd.Timestamp(latest_matured_label_cutoff)]
        keep = set(matured_dates[-int(family.calibration_window_sessions):])
        calibration = frame.loc[frame["decision_date"].isin(keep)].copy()
        calibration["score"] = model.predict(calibration[paths["feature_columns"]].to_numpy(float))
        calibration = calibration.rename(columns={paths["target"]: "observed_excess"})
        terminal = {pd.Timestamp(value): pd.Timestamp(all_dates[index+family.horizon_sessions])
                    for index, value in enumerate(all_dates) if index+family.horizon_sessions < len(all_dates)}
        calibration["terminal_date"] = calibration["decision_date"].map(terminal)
        result = calibration[["decision_date", "terminal_date", "ticker", "score", "observed_excess"]]
        if cache_result:
            self._recalibration_cache[cache_key] = result.copy()
        return result

    def generation_calibration_predictions(self, *, family, model_information_cutoff):
        """Return only the calibration scores made by the requested fitted generation."""
        source_build = self._builds.get((family.family_id, model_information_cutoff))
        if source_build is None:
            raise ValueError("GENERATION_BUILD_NOT_AVAILABLE_FOR_CALIBRATION")
        frame = pd.read_parquet(source_build["calibration_prediction_path"])
        if frame.empty:
            raise RuntimeError("GENERATION_CALIBRATION_PREDICTIONS_EMPTY")
        return frame

    def finalize_generation(self, *, family, build, generation_id):
        source = pd.read_parquet(build["raw_prediction_path"])
        source["model_artifact_id"] = build["model_artifact_id"]
        destination = Path(build["raw_prediction_path"]).with_name("model-predictions.parquet")
        if not destination.is_file():
            source.to_parquet(destination, index=False)
        result = {"prediction_artifact_path": str(destination),
                  "prediction_artifact_sha256": _sha256(destination)}
        if build.get("raw_exit_prediction_path"):
            exits = pd.read_parquet(build["raw_exit_prediction_path"])
            exits["exit_generation_id"] = build["exit_generation_id"]
            exit_destination = Path(build["raw_exit_prediction_path"]).with_name("exit-predictions.parquet")
            exits.to_parquet(exit_destination, index=False)
            result.update({"exit_prediction_artifact_path": str(exit_destination),
                           "exit_prediction_artifact_sha256": _sha256(exit_destination),
                           "exit_generation_id": build["exit_generation_id"],
                           "exit_model_artifact_sha256": build["exit_model_artifact_sha256"],
                           "exit_calibration_fingerprint": build["exit_calibration_fingerprint"]})
        return result
