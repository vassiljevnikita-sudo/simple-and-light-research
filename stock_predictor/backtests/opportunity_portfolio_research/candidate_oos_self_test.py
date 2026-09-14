"""Tiny regression tests for canonical Candidate-OOS generation."""
from __future__ import annotations

from datetime import date
from pathlib import Path
import tempfile
import json

import pandas as pd

from .candidate_oos import (CandidateOosFactory, CandidateOosStore, CandidateSpec, EvidenceSnapshot,
                            build_evidence_snapshot, candidate_registry_document,
                            persist_evidence_snapshot, read_evidence_snapshot_artifact,
                            select_recipe_from_snapshot, fit_production_generation)
from .candidate_oos import load_primary_candidate_registry, candidate_registry_document
from .qbd_training_selection_contracts import FoldPolicy, TargetContract, RecipeSelectionPolicy


def main() -> int:
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        dates = pd.bdate_range("2010-01-01", periods=800)
        panel = root / "panel.parquet"
        pd.DataFrame({"decision_date": dates, "ticker": ["AAA"] * len(dates), "sector": ["X"] * len(dates),
                      "sub_industry": ["Y"] * len(dates), "mom20": [float(i % 13) for i in range(len(dates))],
                      "net_excess_return_1__BASELINE_20_BPS": [0.01 if i % 2 else -0.002 for i in range(len(dates))],
                      "benchmark_forward_return_1": [0.001] * len(dates)}).to_parquet(panel, index=False)
        candidate = CandidateSpec.create("RIDGE_LOGISTIC", {"alpha": 1.0, "positive_C": 1.0, "downside_C": 1.0})
        document = candidate_registry_document((candidate,), source_sha256="a" * 64)
        assert document["records"][0]["candidate_spec_hash"] == candidate.candidate_spec_hash
        generic = root / "hyperparameters.json"
        generic.write_text(json.dumps({"families": {"RIDGE_LOGISTIC": [candidate.hyperparameters],
                                                     "HIST_GRADIENT_BOOSTING": [{"learning_rate": .05, "max_iter": 10,
                                                                                   "max_leaf_nodes": 5, "l2_regularization": 1.0}],
                                                     "SHADOW_NEW_FAMILY": [{"foo": 1}]}}), encoding="utf-8")
        primary_before = load_primary_candidate_registry(generic)
        generic.write_text(json.dumps({"families": {"RIDGE_LOGISTIC": [candidate.hyperparameters],
                                                     "HIST_GRADIENT_BOOSTING": [{"learning_rate": .05, "max_iter": 10,
                                                                                   "max_leaf_nodes": 5, "l2_regularization": 1.0}],
                                                     "SHADOW_NEW_FAMILY": [{"foo": 1}, {"foo": 2}]}}), encoding="utf-8")
        primary_after = load_primary_candidate_registry(generic)
        assert [x.candidate_spec_hash for x in primary_before] == [x.candidate_spec_hash for x in primary_after]
        prepared_cache = root / "shared-prepared-folds"
        factory = CandidateOosFactory(signal_panel=panel, feature_schema={"v2_features": ["mom20"]}, candidates=(candidate,), output_root=root / "candidate-oos",
                                      fold_policy=FoldPolicy(), target_contract=TargetContract(), development_end=date(2013, 12, 31),
                                      prepared_cache_root=prepared_cache)
        fold = factory.fold_specs(horizon=1, development_end=date(2013, 12, 31), holdout_boundary=date(2014, 1, 1))[0]
        prefetched = factory.prepare_fold(
            horizon=1, fold=fold,
            development_end=date(2013, 12, 31),
            holdout_boundary=date(2014, 1, 1))
        assert prefetched["prepared"] is True
        assert prefetched["train_rows"] > 0
        assert prefetched["test_rows"] > 0
        assert len(list(prepared_cache.rglob("*.pkl"))) == 1
        assert not list((root / "candidate-oos").rglob("observations.parquet"))
        result = factory.build_fold(horizon=1, candidate_id=candidate.candidate_id, fold=fold,
                                    development_end=date(2013, 12, 31), holdout_boundary=date(2014, 1, 1))
        assert result["cache"] == "MISS"
        assert factory._frame is None
        assert factory._prepared_fold_cache == {}
        result_hit = factory.build_fold(horizon=1, candidate_id=candidate.candidate_id, fold=fold,
                                        development_end=date(2013, 12, 31), holdout_boundary=date(2014, 1, 1))
        assert result_hit["cache"] == "HIT"
        prepared_artifacts = list(prepared_cache.rglob("*.pkl"))
        assert len(prepared_artifacts) == 1
        # A fresh factory / output store must reuse the same candidate-
        # independent H×Fold preparation artifact rather than materializing a
        # second prepared copy.
        second_factory = CandidateOosFactory(
            signal_panel=panel,
            feature_schema={"v2_features": ["mom20"]},
            candidates=(candidate,),
            output_root=root / "candidate-oos-second",
            fold_policy=FoldPolicy(), target_contract=TargetContract(),
            development_end=date(2013, 12, 31),
            prepared_cache_root=prepared_cache)
        second_fold = second_factory.fold_specs(
            horizon=1, development_end=date(2013, 12, 31),
            holdout_boundary=date(2014, 1, 1))[0]
        second_factory.build_fold(
            horizon=1, candidate_id=candidate.candidate_id,
            fold=second_fold, development_end=date(2013, 12, 31),
            holdout_boundary=date(2014, 1, 1))
        assert second_factory._frame is None
        assert second_factory._prepared_fold_cache == {}
        assert len(list(prepared_cache.rglob("*.pkl"))) == 1
        partition = Path(result["path"])
        assert factory.store.verify_partition(partition)["artifact_sha256"] == result["artifact_sha256"]
        snapshot, frame = build_evidence_snapshot(CandidateOosStore(root / "candidate-oos"), horizon=1,
                                                   selection_cutoff=date(2014, 1, 1), candidate_registry_sha256=document["candidate_registry_sha256"])
        assert snapshot.rows == len(frame) > 0
        assert pd.to_datetime(frame["information_available_at"]).le(pd.Timestamp("2014-01-01")).all()

        # A valid older immutable H×cutoff snapshot must never kill a resume.
        # Preserve the old leaf and publish the corrected derived state under
        # a deterministic manifest-hash version; the next attempt is a hit.
        snapshot_root = root / "evidence-snapshots"
        first_snapshot = persist_evidence_snapshot(
            CandidateOosStore(root / "candidate-oos"),
            horizon=1, selection_cutoff=date(2014, 1, 1),
            candidate_registry_sha256=document["candidate_registry_sha256"],
            fold_policy_hash="f" * 64, signal_panel_hash="s" * 64,
            target_contract_hash="t" * 64, output_root=snapshot_root,
            snapshot=snapshot, frame=frame)
        assert first_snapshot["cache"] == "MISS"
        changed_frame = frame.copy()
        changed_frame["candidate_id"] = (
            changed_frame["candidate_id"].astype(str) + "-RESUME")
        changed_snapshot = EvidenceSnapshot(
            1, date(2014, 1, 1),
            tuple(snapshot.partition_identities) + ("resume-extra",),
            document["candidate_registry_sha256"], "c" * 64,
            len(changed_frame), int(changed_frame["candidate_id"].nunique()),
            int(changed_frame["fold_id"].nunique()))
        versioned_snapshot = persist_evidence_snapshot(
            CandidateOosStore(root / "candidate-oos"),
            horizon=1, selection_cutoff=date(2014, 1, 1),
            candidate_registry_sha256=document["candidate_registry_sha256"],
            fold_policy_hash="f" * 64, signal_panel_hash="s" * 64,
            target_contract_hash="t" * 64, output_root=snapshot_root,
            snapshot=changed_snapshot, frame=changed_frame)
        assert versioned_snapshot["cache"] == (
            "MISS_VERSIONED_IMMUTABLE_CONFLICT")
        assert versioned_snapshot["immutable_conflict_resolution"]["status"] == (
            "VALID_PRIOR_LEAF_PRESERVED")
        assert Path(versioned_snapshot["snapshot_path"]).parent != Path(
            first_snapshot["snapshot_path"]).parent
        first_manifest, first_frame = read_evidence_snapshot_artifact(
            first_snapshot["snapshot_path"])
        versioned_manifest, versioned_frame = read_evidence_snapshot_artifact(
            versioned_snapshot["snapshot_path"])
        assert first_manifest["manifest_sha256"] != (
            versioned_manifest["manifest_sha256"])
        assert len(first_frame) == len(frame)
        assert versioned_frame["candidate_id"].str.endswith(
            "-RESUME").all()
        versioned_hit = persist_evidence_snapshot(
            CandidateOosStore(root / "candidate-oos"),
            horizon=1, selection_cutoff=date(2014, 1, 1),
            candidate_registry_sha256=document["candidate_registry_sha256"],
            fold_policy_hash="f" * 64, signal_panel_hash="s" * 64,
            target_contract_hash="t" * 64, output_root=snapshot_root,
            snapshot=changed_snapshot, frame=changed_frame)
        assert versioned_hit["cache"] == (
            "HIT_VERSIONED_IMMUTABLE_CONFLICT")
        selected = select_recipe_from_snapshot(snapshot, frame, selection_policy_sha256="b" * 64,
                                               selection_policy=RecipeSelectionPolicy(minimum_candidate_folds=1))
        assert selected["selected_recipe_sha256"] and selected["evidence_snapshot_sha256"] == snapshot.evidence_snapshot_sha256
        production = fit_production_generation(
            signal_panel=panel, feature_schema={"v2_features": ["mom20"]},
            selected_recipe={**selected, "recipe_family": candidate.recipe_family,
                             "hyperparameters": candidate.hyperparameters},
            horizon=1, information_cutoff=date(2013, 12, 31),
            training_dates=[str(x.date()) for x in dates[0:504]],
            calibration_dates=[str(x.date()) for x in dates[600:650]], target_contract=TargetContract(),
            output_root=root / "generations")
        assert production["usage"] == "FRESH_PRODUCTION_FIT_FOR_GENERATION_NOT_CANDIDATE_OOS"
        assert Path(production["model_path"]).is_file()
        assert Path(production["manifest_path"]).is_file()
        production_hit = fit_production_generation(
            signal_panel=panel, feature_schema={"v2_features": ["mom20"]},
            selected_recipe={**selected, "recipe_family": candidate.recipe_family,
                             "hyperparameters": candidate.hyperparameters},
            horizon=1, information_cutoff=date(2013, 12, 31),
            training_dates=[str(x.date()) for x in dates[0:504]],
            calibration_dates=[str(x.date()) for x in dates[600:650]], target_contract=TargetContract(),
            output_root=root / "generations")
        assert production_hit["cache"] == "GENERATION_MANIFEST_HIT"
        assert production_hit["manifest_path"] == production["manifest_path"]
        print("PRODUCTION_GENERATION_REUSE_BEFORE_REFIT_PASS")
        calibration_rows = pd.read_parquet(production["calibration_path"])
        assert (pd.to_datetime(calibration_rows["information_available_at"]) >
                pd.to_datetime(calibration_rows["decision_date"])).all()
        # Prediction rows are target-free.  Mutating only their future labels
        # must not alter scores or the generation prediction artifact.
        prediction_dates = [str(x.date()) for x in dates[700:710]]
        changed_panel = root / "panel-target-changed.parquet"
        changed = pd.read_parquet(panel)
        changed.loc[changed["decision_date"].isin(pd.to_datetime(prediction_dates)),
                    "net_excess_return_1__BASELINE_20_BPS"] = 987654.321
        changed.to_parquet(changed_panel, index=False)
        generation_kwargs = dict(
            feature_schema={"v2_features": ["mom20"]},
            selected_recipe={**selected, "recipe_family": candidate.recipe_family,
                             "hyperparameters": candidate.hyperparameters},
            horizon=1, information_cutoff=date(2013, 12, 31),
            training_dates=[str(x.date()) for x in dates[0:504]],
            calibration_dates=[str(x.date()) for x in dates[600:650]],
            target_contract=TargetContract(),
            prediction_dates=prediction_dates)
        first = fit_production_generation(signal_panel=panel, output_root=root / "gen-a", **generation_kwargs)
        second = fit_production_generation(signal_panel=changed_panel, output_root=root / "gen-b", **generation_kwargs)
        assert pd.read_parquet(first["prediction_path"]).equals(pd.read_parquet(second["prediction_path"]))
        # Selection must use predictive OOS quality, never the candidate-
        # independent realized outcome.
        rows = []
        outcomes = [0.1, 0.2, 0.3, 0.4]
        for candidate_id, predictions in (("A", outcomes), ("B", list(reversed(outcomes)))):
            for index, (prediction, outcome) in enumerate(zip(predictions, outcomes)):
                rows.append({"horizon": 1, "candidate_id": candidate_id, "fold_id": "F1",
                             "prediction": prediction, "realized_excess": outcome,
                             "information_available_at": "2013-01-01", "candidate_spec_hash": candidate_id,
                             "recipe_family": "RIDGE_LOGISTIC", "hyperparameters": "{}"})
        metric_frame = pd.DataFrame(rows)
        metric_snapshot = EvidenceSnapshot(1, date(2013, 1, 2), (), "r" * 64, "s" * 64, len(metric_frame), 2, 1)
        selected_metric = select_recipe_from_snapshot(metric_snapshot, metric_frame,
                                                       selection_policy_sha256="p" * 64,
                                                       selection_policy=RecipeSelectionPolicy(minimum_candidate_folds=1))
        assert selected_metric["selected_candidate_id"] == "A"
        assert selected_metric["winner_evidence"]["median_fold_spearman"] > 0
        assert selected_metric["recipe_selection_policy_hash"]
    print("CANDIDATE_OOS_SELF_TEST_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
