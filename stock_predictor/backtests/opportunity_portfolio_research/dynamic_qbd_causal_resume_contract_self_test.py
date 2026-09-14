"""Focused regression tests for the causal/resume hardening layer."""
from __future__ import annotations

from datetime import date
from pathlib import Path
import tempfile
import time
import json

import pandas as pd

from .candidate_oos import CandidateOosFactory, CandidateSpec, resolve_information_available_at
from .dynamic_qbd_manifested_job_coordinator import ManifestedJobStore
from .qbd_training_selection_contracts import FoldPolicy, TargetContract, frozen_primary_candidate_registry


def main() -> int:
    assert resolve_information_available_at(decision_date="2020-01-01", terminal_date="2020-02-12") == "2020-02-12"
    for terminal in ("2020-01-02", "2020-01-11", "2020-01-31"):
        assert resolve_information_available_at(decision_date="2020-01-01", terminal_date=terminal)
    try:
        resolve_information_available_at(decision_date="2020-01-01", terminal_date="2019-12-31")
    except ValueError:
        pass
    else:
        raise AssertionError("IMMATURE_LABEL_NOT_REJECTED")
    primary = frozen_primary_candidate_registry([
        {"candidate_id": "R", "recipe_family": "RIDGE_LOGISTIC", "hyperparameters": {"alpha": 1}, "candidate_spec_hash": "r"},
        {"candidate_id": "H", "recipe_family": "HIST_GRADIENT_BOOSTING", "hyperparameters": {"max_iter": 2}, "candidate_spec_hash": "h"},
        {"candidate_id": "X", "recipe_family": "TRANSFORMER", "hyperparameters": {}, "candidate_spec_hash": "x"},
    ], source_contract="QBD_PRIMARY_RIDGE_HGB_V1")
    assert [x["candidate_id"] for x in primary["records"]] == ["H", "R"]
    assert primary["primary_candidate_universe_hash"]
    assert FoldPolicy().fold_policy_hash != TargetContract().target_contract_hash

    with tempfile.TemporaryDirectory() as folder:
        store = ManifestedJobStore(Path(folder))
        store.seed_jobs({"A": {"job_id": "A", "kind": "input", "state": "PENDING", "depends_on": []},
                         "B": {"job_id": "B", "kind": "input", "state": "PENDING", "depends_on": []},
                         "C": {"job_id": "C", "kind": "derived", "state": "PENDING", "depends_on": ["A", "B", "MISSING"]}})
        a = store.claim_ready("worker-a", lease_seconds=60); b = store.claim_ready("worker-b", lease_seconds=60)
        store.finish_job(a["job_id"], "worker-a", "COMPLETE")
        store.finish_job(b["job_id"], "worker-b", "COMPLETE")
        assert store.claim_ready("worker-c") is None

        store.seed_jobs({"LEASE": {"job_id": "LEASE", "kind": "input", "state": "PENDING", "depends_on": []}})
        first = store.claim_ready("worker-a", lease_seconds=0)
        second = store.claim_ready("worker-b", lease_seconds=60)
        assert first and second and first["job_id"] == second["job_id"]
        try:
            store.finish_job(first["job_id"], "worker-a", "COMPLETE")
        except RuntimeError:
            pass
        else:
            raise AssertionError("STALE_WORKER_FINISH_ACCEPTED")
        store.finish_job(second["job_id"], "worker-b", "COMPLETE")

        dates = pd.bdate_range("2020-01-01", periods=700)
        candidate = CandidateSpec.create("RIDGE_LOGISTIC", {"alpha": 1.0, "positive_C": 1.0, "downside_C": 1.0})
        panel = Path(folder) / "panel.parquet"
        pd.DataFrame({"decision_date": dates, "ticker": ["AAA"] * len(dates), "mom20": [0.1] * len(dates),
                      "net_excess_return_1__BASELINE_20_BPS": [0.01] * len(dates)}).to_parquet(panel, index=False)
        factory = CandidateOosFactory(signal_panel=panel, feature_schema={"v2_features": ["mom20"]},
                                      candidates=(candidate,), output_root=Path(folder) / "evidence",
                                      fold_policy=FoldPolicy(), target_contract=TargetContract())
        try:
            factory.fold_specs(horizon=1, development_end=date(2026, 7, 25), holdout_boundary=date(2026, 7, 25))
        except PermissionError:
            pass
        else:
            raise AssertionError("DEEP_HOLDOUT_GUARD_MISSING")
        folds = factory.fold_specs(horizon=1, development_end=date(2022, 9, 30), holdout_boundary=date(2026, 7, 25))
        try:
            factory.build_fold(horizon=1, candidate_id=candidate.candidate_id, fold=folds[0],
                               development_end=date(2022, 9, 30), holdout_boundary=date(2026, 7, 25))
        except ValueError as exc:
            assert "BENCHMARK_COLUMN_MISSING" in str(exc)
        else:
            raise AssertionError("BENCHMARK_ZERO_FALLBACK_ACCEPTED")

        store.seed_jobs({"CACHE": {"job_id": "CACHE", "kind": "input", "state": "PENDING", "semantic": "v1"}})
        claimed = store.claim_ready("worker", lease_seconds=60)
        store.finish_job(claimed["job_id"], "worker", "COMPLETE")
        store.seed_jobs({"CACHE": {"job_id": "CACHE", "kind": "input", "state": "PENDING", "semantic": "v2"}})
        assert store.job("CACHE")["state"] == "PENDING" and store.job("CACHE").get("stale_from") == "COMPLETE"

    print("QBD_HARDENING_SELF_TEST_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
