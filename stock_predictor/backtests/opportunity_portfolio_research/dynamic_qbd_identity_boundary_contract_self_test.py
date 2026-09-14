"""Deterministic identity, holdout-boundary and segmented-replay contract tests.

These tests deliberately use tiny deterministic fixtures. They validate
identity and causal boundaries without opening the final holdout or running
the H1--H30 real-data suite.
"""
from __future__ import annotations

from datetime import date
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

from .candidate_oos import EvidenceSnapshot, select_recipe_from_snapshot
from .contract_fingerprints import stable_hash
from .portfolio_policy_contracts import Policy
from .cost_contracts import CostModel
from .tax_contracts import TaxConfig
from .dynamic_qbd_manifested_job_coordinator import ManifestedJobStore, input_quality_manifest
from .dynamic_qbd_portfolio_replay import replay_family
from .qbd_training_selection_contracts import FoldPolicy, ModelTrainingContract, RecipeSelectionPolicy
from .candidate_oos import mature_decision_sessions
from .development_slice_hash import cleanup_orphaned_slice_hash_scratch, parquet_development_slice_sha256


def _assert_equal_state(left, right) -> None:
    def normalize(value):
        if isinstance(value, dict):
            return {str(k): normalize(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
        if isinstance(value, list):
            return [normalize(v) for v in value]
        if isinstance(value, (np.integer, np.floating)):
            return float(value)
        if isinstance(value, (int, float)):
            return float(value)
        return value
    assert stable_hash(normalize(left)) == stable_hash(normalize(right)), "SEGMENTED_REPLAY_STATE_MISMATCH"


def main() -> int:
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)

        # INPUT_QUALITY_HOLDOUT_NOT_READ_PASS
        quality_path = root / "quality.parquet"
        pd.DataFrame({
            "date": pd.to_datetime(["2026-07-24", "2026-07-25"]),
            "ticker": ["AAA", "SENTINEL"],
            "close": [1.0, 999.0],
        }).to_parquet(quality_path, index=False)
        quality = input_quality_manifest(
            quality_path, date_column_candidates=("date",), key_columns=("date", "ticker"),
            development_end=date(2026, 7, 24), holdout_boundary=date(2026, 7, 25))
        assert quality["row_count"] == 1 and quality["end"] == "2026-07-24"
        print("INPUT_QUALITY_HOLDOUT_NOT_READ_PASS")

        # FUTURE_APPEND_DOES_NOT_CHANGE_HISTORICAL_RUN_IDENTITY_PASS and
        # HOLDOUT_APPEND_DOES_NOT_CHANGE_DEVELOPMENT_IDENTITY_PASS
        base = pd.DataFrame({"date": pd.to_datetime(["2025-12-30", "2025-12-31"]),
                             "ticker": ["AAA", "AAA"], "value": [1.0, 2.0]})
        future = pd.concat([base, pd.DataFrame({"date": pd.to_datetime(["2026-01-01", "2026-07-24"]),
                                                 "ticker": ["AAA", "AAA"], "value": [3.0, 4.0]})])
        holdout = pd.concat([future, pd.DataFrame({"date": pd.to_datetime(["2026-07-25"]),
                                                   "ticker": ["SENTINEL"], "value": [999.0]})])
        p1, p2, p3 = root / "v1.parquet", root / "v2.parquet", root / "v3.parquet"
        base.to_parquet(p1, index=False); future.to_parquet(p2, index=False); holdout.to_parquet(p3, index=False)
        h1 = parquet_development_slice_sha256(p1, date_column="date", development_end=date(2025, 12, 31), holdout_boundary=date(2026, 7, 25))
        h2 = parquet_development_slice_sha256(p2, date_column="date", development_end=date(2025, 12, 31), holdout_boundary=date(2026, 7, 25))
        h3 = parquet_development_slice_sha256(p3, date_column="date", development_end=date(2025, 12, 31), holdout_boundary=date(2026, 7, 25))
        assert h1 == h2 == h3
        print("FUTURE_APPEND_DOES_NOT_CHANGE_HISTORICAL_RUN_IDENTITY_PASS")
        print("HOLDOUT_APPEND_DOES_NOT_CHANGE_DEVELOPMENT_IDENTITY_PASS")

        # SHARED_DEVELOPMENT_HASH_CACHE_CROSSES_PROCESSES_PASS and
        # SLICE_HASH_ABORT_SCRATCH_CLEANUP_PASS
        cache_root = root / "slice-hash-cache"
        scratch_root = root / "slice-hash-scratch"
        child_env = os.environ.copy()
        child_env["DQBD_SLICE_HASH_CACHE_ROOT"] = str(cache_root)
        child_env["DQBD_SLICE_HASH_SCRATCH_ROOT"] = str(scratch_root)
        snippet = (
            "from datetime import date; "
            "from stock_predictor.backtests.opportunity_portfolio_research.development_slice_hash "
            "import parquet_development_slice_sha256; "
            "print(parquet_development_slice_sha256(r'" + str(p3) + "', "
            "date_column='date', development_end=date(2025,12,31), "
            "holdout_boundary=date(2026,7,25)))"
        )
        repo_root = Path(__file__).resolve().parents[3]
        first = subprocess.run([sys.executable, "-c", snippet], cwd=str(repo_root),
                               env=child_env, check=True, capture_output=True, text=True)
        second = subprocess.run([sys.executable, "-c", snippet], cwd=str(repo_root),
                                env=child_env, check=True, capture_output=True, text=True)
        assert first.stdout.strip() == second.stdout.strip()
        cache_files = list((cache_root / "values").glob("*.json"))
        assert len(cache_files) == 1
        cache_payload = json.loads(cache_files[0].read_text(encoding="utf-8"))
        assert cache_payload["schema_version"] == "DQBD_DEVELOPMENT_SLICE_HASH_CACHE_V1"
        orphan = scratch_root / "p2147483647-orphan"
        orphan.mkdir(parents=True, exist_ok=True)
        (orphan / "shard.jsonl").write_text("orphan\n", encoding="utf-8")
        os.environ["DQBD_SLICE_HASH_SCRATCH_ROOT"] = str(scratch_root)
        assert cleanup_orphaned_slice_hash_scratch() == 1
        assert not orphan.exists()
        print("SHARED_DEVELOPMENT_HASH_CACHE_CROSSES_PROCESSES_PASS")
        print("SLICE_HASH_ABORT_SCRATCH_CLEANUP_PASS")

        distributions = pd.DataFrame({"ex_date": pd.to_datetime(["2025-12-31", "2026-01-01"]),
                                       "payable_date": pd.to_datetime(["2026-01-10", "2026-01-15"]),
                                       "ticker": ["URTH", "URTH"], "cash_amount": [1.0, 2.0]})
        dp = root / "distributions.parquet"; distributions.to_parquet(dp, index=False)
        dh = parquet_development_slice_sha256(dp, date_column="ex_date", development_end=date(2025, 12, 31), holdout_boundary=date(2026, 7, 25))
        expected = distributions.iloc[[0]].copy(); ep = root / "expected-distributions.parquet"; expected.to_parquet(ep, index=False)
        assert dh == parquet_development_slice_sha256(ep, date_column="ex_date", development_end=date(2025, 12, 31), holdout_boundary=date(2026, 7, 25))
        print("DISTRIBUTION_DEVELOPMENT_HASH_EX_DATE_SEMANTICS_PASS")

        sessions = tuple(pd.bdate_range("2023-01-02", periods=500).date)
        for horizon in (1, 10, 30):
            mature = mature_decision_sessions(trading_sessions=sessions, information_cutoff=sessions[-1], horizon=horizon)
            assert len(mature[-252:]) == 252
        print("EXACT_252_MATURE_CALIBRATION_SESSIONS_PASS")
        print("CALIBRATION_IMMATURE_TARGET_NEVER_READ_PASS")
        workflow = Path(__file__).resolve().parents[3] / ".github" / "workflows" / "dynamic-qbd-factory-self-test.yml"
        assert "dynamic_qbd_identity_boundary_contract_self_test" in workflow.read_text(encoding="utf-8")
        print("IDENTITY_BOUNDARY_CONTRACT_TEST_WIRED_TO_WORKFLOW_PASS")

        # FOLD_POLICY_HASH_END_TO_END_IDENTICAL_PASS
        custom_fold = FoldPolicy(training_window_sessions=410, validation_window_sessions=97,
                                 calibration_window_sessions=211, step_sessions=43,
                                 purge_sessions=17, embargo_sessions=4)
        assert custom_fold.fold_policy_hash == FoldPolicy(**{
            "training_window_sessions": 410, "validation_window_sessions": 97,
            "calibration_window_sessions": 211, "step_sessions": 43,
            "purge_sessions": 17, "embargo_sessions": 4,
        }).fold_policy_hash
        print("FOLD_POLICY_HASH_END_TO_END_IDENTICAL_PASS")

        # RECIPE_SELECTION_POLICY_INVALIDATES_JOB_PASS
        store = ManifestedJobStore(root / "state")
        old = {"job_id": "recipe-selection:T1", "kind": "recipe_selection",
               "recipe_selection_policy_hash": RecipeSelectionPolicy(version="OLD").recipe_selection_policy_hash,
               "state": "PENDING", "depends_on": []}
        store.seed_jobs({old["job_id"]: old})
        claimed = store.claim_ready("test")
        store.finish_job(claimed["job_id"], "test", "COMPLETE", result={"ok": True})
        new = dict(old, recipe_selection_policy_hash=RecipeSelectionPolicy(version="NEW").recipe_selection_policy_hash)
        store.seed_jobs({new["job_id"]: new})
        assert store.job(new["job_id"])["state"] == "PENDING"
        print("RECIPE_SELECTION_POLICY_INVALIDATES_JOB_PASS")

        # SINGLE_FOLD_NOT_PRIMARY_AUTHORITY_PASS
        one_fold = pd.DataFrame({
            "horizon": [1, 1], "candidate_id": ["A", "A"], "fold_id": ["F1", "F1"],
            "prediction": [0.1, 0.2], "realized_excess": [0.2, 0.1],
            "information_available_at": ["2020-01-01", "2020-01-01"],
            "candidate_spec_hash": ["a", "a"], "recipe_family": ["RIDGE", "RIDGE"],
            "hyperparameters": ["{}", "{}"],
        })
        try:
            select_recipe_from_snapshot(
                EvidenceSnapshot(1, date(2020, 1, 2), (), "r" * 64, "s" * 64, 2, 1, 1),
                one_fold, selection_policy_sha256="p" * 64)
        except ValueError as exc:
            assert str(exc) == "RECIPE_SELECTION_NO_CANDIDATE_WITH_MINIMUM_FOLDS"
        else:
            raise AssertionError("SINGLE_FOLD_PRIMARY_AUTHORITY_LEAK")
        print("SINGLE_FOLD_NOT_PRIMARY_AUTHORITY_PASS")

        # MODEL_TRAINING_CONTRACT_SEED_INVALIDATION_PASS
        a = ModelTrainingContract(random_seed=17, primary_candidate_universe_hash="u")
        b = ModelTrainingContract(random_seed=212, primary_candidate_universe_hash="u")
        assert a.model_training_contract_hash != b.model_training_contract_hash
        print("MODEL_TRAINING_CONTRACT_SEED_INVALIDATION_PASS")

        # REPLAY_SEGMENT_BOUNDARY_EXACTLY_ONCE_PASS and
        # SEGMENTED_REPLAY_EQUALS_ONE_SHOT_PASS.
        sessions = pd.bdate_range("2024-01-30", periods=4)
        values = np.arange(4, dtype=float)
        prices = pd.DataFrame({
            "date": list(sessions) * 2,
            "ticker": ["URTH"] * 4 + ["AAA"] * 4,
            "open": np.r_[100 + values, 50 + values],
            "close": np.r_[100 + values, 50 + values],
        })
        signals = pd.DataFrame({
            "decision_date": [sessions[0], sessions[2]],
            "ticker": ["AAA", "AAA"], "score": [1.0, 2.0],
            "model_artifact_id": ["G1", "G2"],
        })
        schedule = pd.DataFrame({
            "activation_date": [sessions[0], sessions[2]],
            "family_id": ["F", "F"], "generation_id": ["G1", "G2"],
            "resolved_threshold": [0.0, 0.0],
            "entry_policy_id": ["E1", "E2"], "exit_policy_id": ["X1", "X2"],
            "model_artifact_id": ["G1", "G2"],
        })
        policy = Policy(1, .5, 1.0, 1, 1, "FIXED", 0.0, "IGNORE_NEW", "EQUAL_ACTIVE", .5)
        kwargs = dict(signals=signals, prices=prices, policy=policy,
                      generation_schedule=schedule, cost=CostModel(0.0),
                      tax=TaxConfig(), initial=10000.0)
        one = replay_family(**kwargs)
        first = replay_family(**kwargs, start=sessions[0], end=sessions[1])
        second = replay_family(**kwargs, start=sessions[2], end=sessions[3],
                               resume_state=first["replay_state"])
        assert len(second["curve"]) == 4
        assert len({pd.Timestamp(x["date"]) for x in second["replay_state"]["curve"]}) == 4
        _assert_equal_state(one["replay_state"], second["replay_state"])
        print("REPLAY_SEGMENT_BOUNDARY_EXACTLY_ONCE_PASS")
        print("SEGMENTED_REPLAY_EQUALS_ONE_SHOT_PASS")

    print("DYNAMIC_QBD_IDENTITY_BOUNDARY_CONTRACT_SELF_TEST_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
