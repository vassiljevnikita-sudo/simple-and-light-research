"""Contract tests for the manifested Manifested Job Coordinator coordinator."""
from __future__ import annotations

from datetime import date
import json
import os
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace

import pandas as pd
from concurrent.futures.process import BrokenProcessPool

from .dynamic_qbd_manifested_job_coordinator import (
    GpuRuntimeMonitor, ManifestedJobInputs, ManifestedJobStore,
    RamAdmissionScheduler, RamJobReclaimed, RamWorkerPauseController,
    _CandidateExecutionDurationProfiles,
    _CandidateOosReadyFrontier,
    _configured_lane_recycle_limit,
    _candidate_gpu_workload_priority, _choose_fair_gpu_device,
    _choose_gpu_feed_device, _gpu_global_backlog_seconds,
    _manifested_pool,
    _coverage_consistency, _gpu_execution_fingerprint, _is_gpu_backend_failure,
    execute_ready_jobs, incubator_is_authoritative, initialize_manifested_job_coordinator,
    propose_incubator_family, select_recipe_from_candidate_oos,
    stock_dividend_audit, write_calibration_store, write_candidate_oos_evidence,
    write_generation_prediction_store, _compatible_resume_contract_view,
)
from .dynamic_qbd_causal_model_store_run import _resolve_seed_root
from .dynamic_qbd_development_evaluation import tax_config_from_contract
from .development_slice_hash import (
    cleanup_orphaned_slice_hash_scratch,
    parquet_development_slice_sha256,
)


def _lane_reclaim_self_test_worker(payload: dict) -> dict:
    import os as _os
    import time as _time
    _time.sleep(float(payload.get("sleep_seconds", 0.0)))
    return {
        "job_id": str(payload.get("job", {}).get("job_id", "")),
        "pid": _os.getpid(),
    }


def main() -> int:
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        repo = Path(__file__).resolve().parents[3]
        panel = root / "signal-panel.parquet"
        dates = pd.date_range("2012-01-01", "2017-12-29", freq="B")
        pd.DataFrame({"decision_date": dates, "ticker": ["AAA"] * len(dates)}).to_parquet(panel, index=False)

        # Development-slice hashes are now shared execution artifacts. Repeated
        # identical requests must hit one persistent cache value and leave no
        # external-sort scratch behind. Orphan PID scratch is removable after
        # an abruptly terminated lane.
        slice_cache = root / "slice-hash-cache"
        slice_scratch = root / "slice-hash-scratch"
        os.environ["DQBD_SLICE_HASH_CACHE_ROOT"] = str(slice_cache)
        os.environ["DQBD_SLICE_HASH_SCRATCH_ROOT"] = str(slice_scratch)
        slice_hash_1 = parquet_development_slice_sha256(
            panel, date_column="decision_date",
            development_end=date(2017, 12, 29),
            holdout_boundary=date(2026, 7, 25))
        slice_hash_2 = parquet_development_slice_sha256(
            panel, date_column="decision_date",
            development_end=date(2017, 12, 29),
            holdout_boundary=date(2026, 7, 25))
        assert slice_hash_1 == slice_hash_2
        assert len(list((slice_cache / "values").glob("*.json"))) == 1
        assert not list(slice_scratch.glob("p*-*"))
        orphan = slice_scratch / "p2147483647-orphan"
        orphan.mkdir(parents=True, exist_ok=False)
        assert cleanup_orphaned_slice_hash_scratch() == 1
        assert not orphan.exists()

        metrics = root / "candidate-metrics.json"
        metrics.write_text("[]\n", encoding="utf-8")
        schema = root / "feature-schema.json"
        schema.write_text('{"schema":"synthetic"}\n', encoding="utf-8")
        prices = root / "benchmark-prices.parquet"
        pd.DataFrame({"date": dates, "ticker": ["URTH"] * len(dates), "open": [100.0] * len(dates),
                      "close": [100.0] * len(dates)}).to_parquet(prices, index=False)
        distributions = root / "distributions.parquet"
        pd.DataFrame({"ticker": pd.Series(dtype="string"),
                      "ex_date": pd.Series(dtype="datetime64[ns]"),
                      "payable_date": pd.Series(dtype="datetime64[ns]"),
                      "cash_amount": pd.Series(dtype="float64")}).to_parquet(distributions, index=False)
        output = root / "manifested-job"
        result = initialize_manifested_job_coordinator(
            inputs=ManifestedJobInputs(repo_root=repo, signal_panel=panel, candidate_metrics=metrics,
                                 feature_schema=schema, benchmark_prices=prices,
                                 benchmark_distributions=distributions,
                                 development_start=date(2016, 1, 1), development_end=date(2017, 12, 29)),
            output_root=output)
        assert result["status"].startswith("BLOCKED_")
        assert result["candidate_oos_audit"]["status"] == "GENERATABLE_DEPENDENCY"
        assert result["model_family_count"] == 30
        assert result["portfolio_family_count"] == 2790
        assert result["refit_count"] == 24
        assert result["final_holdout_opened"] is False
        assert result["input_quality"]["stock_execution_prices"]["status"] == "BLOCKED_STOCK_EXECUTION_PRICE_INPUT_MISSING"
        contract = json.loads((output / "manifested-job-contract.json").read_text(encoding="utf-8"))
        required = {"git_sha", "source_tree_sha256", "signal_panel_development_sha256", "candidate_metrics_sha256",
                    "feature_schema_sha256", "benchmark_prices_development_sha256", "benchmark_distributions_development_sha256",
                    "training_window_sessions", "calibration_window_sessions", "refit_cadence",
                    "maturity_rule", "score_quantile", "top_fraction", "cost_model", "tax_contract",
                    "holdout_boundary", "family_registry_hash"}
        assert required <= set(contract)
        registry = json.loads((output / "family-registry.json").read_text(encoding="utf-8"))
        assert len({x["model_family_key"] for x in registry["portfolio_families"]}) == 30
        assert len({x["portfolio_family_key"] for x in registry["portfolio_families"]}) == 2790
        jobs = json.loads((output / "run-state" / "jobs.json").read_text(encoding="utf-8"))
        assert any(value["kind"] == "recipe_selection" for value in jobs.values())
        assert any(value["kind"] == "generation_ready" for value in jobs.values())
        assert all(value["depends_on"] for value in jobs.values() if value["kind"] not in ("model", "input"))

        # Default existing-state initialization stays fast.  An explicit slow
        # reconciliation rebuilds graph metadata once and preserves all
        # unchanged COMPLETE checkpoints.
        resume_inputs = ManifestedJobInputs(
            repo_root=repo, signal_panel=panel, candidate_metrics=metrics,
            feature_schema=schema, benchmark_prices=prices,
            benchmark_distributions=distributions,
            development_start=date(2016, 1, 1),
            development_end=date(2017, 12, 29))
        fast_resume = initialize_manifested_job_coordinator(
            inputs=resume_inputs, output_root=output,
            allow_compatible_code_resume=True)
        assert fast_resume["status"] == "RESUMED_FAST"
        assert fast_resume["progress"]["graph_reconciliation"] == (
            "SKIPPED_VALIDATED_FAST_RESUME")
        slow_resume = initialize_manifested_job_coordinator(
            inputs=resume_inputs, output_root=output,
            allow_compatible_code_resume=True,
            reconcile_existing_graph=True)
        assert slow_resume["status"] == "RESUMED_RECONCILED"
        assert slow_resume["progress"]["graph_reconciliation"] == {
            "inserted_jobs": 0,
            "updated_jobs": 0,
            "requeued_descendants": 0,
        }
        # JSON-loaded contracts expose arrays as lists while a freshly-built
        # dataclass contract may still expose the same field as a tuple.  The
        # compatible scientific resume comparison must use JSON semantics.
        contract_view = {"recipe_selection_policy": {
            "tie_breakers": ("MEAN_FOLD_SPEARMAN", "CANDIDATE_ID")}}
        loaded_view = {"recipe_selection_policy": {
            "tie_breakers": ["MEAN_FOLD_SPEARMAN", "CANDIDATE_ID"]}}
        assert _compatible_resume_contract_view(contract_view) == (
            _compatible_resume_contract_view(loaded_view))
        # A hotloaded resume may materialize the same validated input under a
        # snapshot path.  The content/stat fingerprints still bind the input;
        # only that non-scientific path spelling is ignored.
        assert _compatible_resume_contract_view({
            "input_quality": {"stock_distributions": {
                "path": r"D:\repo\artifacts\input.parquet",
                "development_content_hash": "same",
            }},
        }) == _compatible_resume_contract_view({
            "input_quality": {"stock_distributions": {
                "path": r"D:\repo\.hotload\code\snapshot\artifacts\input.parquet",
                "development_content_hash": "same",
            }},
        })

        # Candidate-OOS dispatch materializes its broad pending frontier only
        # once.  Claims/completions and a newly prepared fold update bounded
        # in-memory queues; a scheduling tick must not reload the manifest.
        frontier_jobs = [
            {"job_id": "candidate_oos_fold:1:F1:HGB", "horizon": 1,
             "fold_id": "F1", "candidate_id": "HGB", "state": "PENDING"},
            {"job_id": "candidate_oos_fold:1:F1:RIDGE", "horizon": 1,
             "fold_id": "F1", "candidate_id": "RIDGE", "state": "PENDING"},
            {"job_id": "candidate_oos_fold:1:F2:HGB", "horizon": 1,
             "fold_id": "F2", "candidate_id": "HGB", "state": "PENDING"},
        ]
        prepared_folds: set[tuple[int, str]] = set()
        frontier = _CandidateOosReadyFrontier(
            frontier_jobs,
            lambda job: str(job["candidate_id"]),
            lambda job: (int(job["horizon"]), str(job["fold_id"]))
            in prepared_folds,
        )
        assert frontier.stats()["initial_scan_count"] == 1
        assert frontier.gpu_ids() == ()
        prep_ids = frontier.prepare_ids(set(), limit=8)
        assert len(prep_ids) == 2
        prep_id = prep_ids[0]
        frontier.claim(prep_id)
        frontier.requeue(next(
            job for job in frontier_jobs if job["job_id"] == prep_id))
        prepared_folds.add((1, "F1"))
        frontier.mark_prepared((1, "F1"))
        assert frontier.gpu_ids(limit=8) == (
            "candidate_oos_fold:1:F1:HGB",
            "candidate_oos_fold:1:F1:RIDGE",
        )
        assert frontier.stats()["initial_scan_count"] == 1
        frontier.complete("candidate_oos_fold:1:F1:HGB")
        assert "candidate_oos_fold:1:F1:HGB" not in frontier.gpu_ids()

        # If a slow resume repairs one job payload, only that node and its
        # transitive derived subtree are invalidated. Independent COMPLETE
        # checkpoints remain reusable.
        reconcile_root = root / "resume-reconcile"
        reconcile_store = ManifestedJobStore(reconcile_root)
        graph_v1 = {
            "input-a": {
                "job_id": "input-a", "kind": "input", "state": "COMPLETE",
                "depends_on": []},
            "input-b": {
                "job_id": "input-b", "kind": "input", "state": "COMPLETE",
                "depends_on": []},
            "coverage": {
                "job_id": "coverage", "kind": "candidate_evidence_coverage",
                "state": "PENDING", "depends_on": ["input-a"]},
            "selection": {
                "job_id": "selection", "kind": "recipe_selection",
                "state": "PENDING", "depends_on": ["coverage"]},
            "model": {
                "job_id": "model", "kind": "model",
                "state": "PENDING", "depends_on": ["selection"]},
        }
        reconcile_store.seed_jobs(graph_v1)
        reconcile_store.set_job(
            "coverage", "COMPLETE", result={"snapshot": "old"})
        reconcile_store.set_job(
            "selection", "COMPLETE", result={"recipe": "old"})
        reconcile_store.set_job(
            "model", "COMPLETE", result={"generation": "old"})
        graph_v2 = {
            **graph_v1,
            "coverage": {
                "job_id": "coverage", "kind": "candidate_evidence_coverage",
                "state": "PENDING",
                "depends_on": ["input-a", "input-b"]},
        }
        reconciled = reconcile_store.seed_jobs(
            graph_v2, invalidate_descendants=True)
        assert reconciled == {
            "inserted_jobs": 0,
            "updated_jobs": 1,
            "requeued_descendants": 2,
        }
        assert reconcile_store.job("input-a")["state"] == "COMPLETE"
        assert reconcile_store.job("input-b")["state"] == "COMPLETE"
        assert reconcile_store.job("coverage")["state"] == "PENDING"
        assert reconcile_store.job("selection")["state"] == "PENDING"
        assert reconcile_store.job("model")["state"] == "PENDING"
        assert "result" not in reconcile_store.job("selection")
        assert "result" not in reconcile_store.job("model")
        reconcile_store.set_job(
            "coverage", "COMPLETE", result={"snapshot": "corrected"})
        reconcile_store.set_job(
            "selection", "COMPLETE", result={"recipe": "stale-again"})
        reconcile_store.set_job(
            "model", "COMPLETE", result={"generation": "stale-again"})
        runtime_requeued = reconcile_store.requeue_descendants(("coverage",))
        assert runtime_requeued == 2
        assert reconcile_store.job("coverage")["state"] == "COMPLETE"
        assert reconcile_store.job("selection")["state"] == "PENDING"
        assert reconcile_store.job("model")["state"] == "PENDING"
        assert "result" not in reconcile_store.job("selection")
        assert "result" not in reconcile_store.job("model")
        candidate = pd.DataFrame({
            "fold_id": ["F1"], "horizon": [3], "candidate_id": ["C1"], "recipe_family": ["RIDGE"],
            "hyperparameters": ["{}"], "decision_date": [pd.Timestamp("2017-01-03")], "ticker": ["AAA"],
            "oos_score": [.1], "terminal_date": [pd.Timestamp("2017-01-06")], "realized_excess": [.01],
            "information_available_at": [pd.Timestamp("2017-01-07")],
        })
        evidence = root / "candidate-oos.parquet"
        assert write_candidate_oos_evidence(candidate, evidence)["rows"] == 1
        selection = select_recipe_from_candidate_oos(candidate, horizon=3, selection_cutoff=date(2017, 1, 10))
        assert selection["selected_candidate_id"] == "C1" and selection["fold_count"] == 1
        predictions = root / "predictions.parquet"
        assert write_generation_prediction_store(pd.DataFrame({
            "decision_date": [pd.Timestamp("2017-01-03")], "ticker": ["AAA"], "horizon": [3],
            "generation_id": ["G1"], "model_artifact_id": ["M1"], "recipe_id": ["R1"], "score": [.1],
        }), predictions)["rows"] == 1
        calibration = root / "calibration.parquet"
        assert write_calibration_store(pd.DataFrame({
            "generation_id": ["G1"], "calibration_start": [pd.Timestamp("2016-12-01")],
            "calibration_end": [pd.Timestamp("2016-12-31")], "maturity_cutoff": [pd.Timestamp("2016-12-31")],
            "source_prediction_sha256": ["P"], "score_quantile": [.975],
            "resolved_threshold": [.1], "resolved_top_fraction": [.005], "calibration_fingerprint": ["C"],
            "matured_observation_count": [10],
        }), calibration)["generations"] == 1
        trades = pd.DataFrame({"ticker": ["AAA"], "entry_date": ["2017-01-01"], "exit_date": ["2017-01-05"]})
        distributions = pd.DataFrame({"ticker": ["AAA"], "ex_date": ["2017-01-03"],
                                       "payable_date": ["2017-01-10"], "cash_amount": [.2]})
        assert len(stock_dividend_audit(trades=trades, distributions=distributions)) == 1
        assert incubator_is_authoritative(birth_date=date(2017, 1, 3), evidence_date=date(2017, 1, 2)) is False
        assert incubator_is_authoritative(birth_date=date(2017, 1, 3), evidence_date=date(2017, 1, 3)) is True
        proposal = propose_incubator_family(family_id="NEW", parent_family_ids=["H01"], proposed_at=date(2017, 1, 1),
                                            information_cutoff=date(2016, 12, 31), birth_date=date(2017, 1, 3),
                                            first_eligible_refit=date(2017, 2, 1), family_spec_hash="S")
        assert not incubator_is_authoritative(birth_date=date.fromisoformat(proposal["birth_date"]), evidence_date=date(2017, 1, 2))
        post_tax = tax_config_from_contract({"mode": "POST_TAX", "enabled": True,
                                             "engine": "DE_RETAIL_APPROX", "capital_gains_rate": .25,
                                             "solidarity_surcharge": .055, "allowance_eur": 1000.0,
                                             "church_tax_rate": 0.0})
        assert post_tax.enabled is True and abs(post_tax.capital_gains_rate - .25) < 1e-12
        # The contract must reach every portfolio cell, including POST_TAX and costs.
        registry = json.loads((output / "family-registry.json").read_text(encoding="utf-8"))
        assert {x["tax_contract"]["mode"] for x in registry["portfolio_families"]} == {"POST_TAX"}
        assert {x["cost_contract"]["roundtrip_bps"] for x in registry["portfolio_families"]} == {20.0}
        # Resume-safe claiming: a COMPLETE node is never processed twice.
        state = ManifestedJobStore(output)
        state.set_job("resume-test", "PENDING", kind="input_test", depends_on=[])
        called = []
        progress = execute_ready_jobs(
            state, {"input_test": lambda job: called.append(job["job_id"])},
            max_jobs=1, kinds=("input_test",)
        )
        assert progress["processed_this_call"] == 1
        assert called == ["resume-test"]
        assert state.job("resume-test")["state"] == "COMPLETE"
        # An abrupt child exit is an execution incident.  The exact claimed
        # node is returned to the resumable frontier with an auditable retry
        # counter instead of being lost or aborting the whole DAG.
        failure_store = ManifestedJobStore(root / "worker-failure-requeue")
        failure_store.set_job(
            "worker-failure", "PENDING", kind="input_test", depends_on=[])
        failure_claim = failure_store.claim_ready(
            "worker-failure-test", lease_seconds=60)
        assert failure_claim is not None
        assert failure_store.requeue_execution_failure(
            "worker-failure", "worker-failure-test",
            reason="BrokenProcessPool:synthetic") is True
        failure_job = failure_store.job("worker-failure")
        assert failure_job["state"] == "PENDING"
        assert failure_job["worker_failure_count"] == 1
        assert failure_job["last_worker_failure"] == (
            "BrokenProcessPool:synthetic")
        assert _configured_lane_recycle_limit() >= 1
        # A live worker must be able to extend ownership beyond the original
        # lease.  Long real-data HGB fits routinely exceed the default lease.
        lease_root = root / "lease-heartbeat"
        lease_store = ManifestedJobStore(lease_root)
        lease_store.set_job("slow", "PENDING", kind="input", depends_on=[])
        claimed = lease_store.claim_ready("heartbeat-test", lease_seconds=1)
        assert claimed is not None and claimed["job_id"] == "slow"
        time.sleep(.6)
        lease_store.heartbeat("slow", "heartbeat-test", lease_seconds=2)
        time.sleep(.6)  # finish after the original one-second lease expired
        lease_store.finish_job("slow", "heartbeat-test", "COMPLETE", result={"ok": True})
        assert lease_store.job("slow")["state"] == "COMPLETE"
        # GPU assignment remains fair, while the process topology keeps the
        # full 26-lane capacity. Job-aware RAM admission controls live work
        # instead of permanently shrinking the pool from a static estimate.
        device, cursor = _choose_fair_gpu_device([False, False], [8, 2], 0)
        assert device == 1 and cursor == 0
        device, _ = _choose_fair_gpu_device([False, False], [2, 2], cursor)
        assert device == 0
        # Candidate GPU queues prefer HGB, then use prepared Ridge as
        # spillover rather than leaving an otherwise healthy GPU idle.
        assert _candidate_gpu_workload_priority(
            "HGB", prepared=True) == 0
        assert _candidate_gpu_workload_priority(
            "RIDGE", prepared=True) == 1
        assert _candidate_gpu_workload_priority(
            "RIDGE", prepared=False) == 99
        assert _candidate_gpu_workload_priority(
            "CPU", prepared=True) == 99
        # A legacy checkpoint may omit a fold dependency even though its
        # complete partition is already matured at the causal cutoff.  That is
        # a recoverable stale-manifest condition; missing evidence is not.
        repaired = _coverage_consistency(
            expected_keys={(2, "C1", "WF01")},
            observed_keys={(2, "C1", "WF01"), (2, "C1", "WF05")},
            duplicate_keys=0, frame_empty=False,
            candidate_count=5, candidate_registry_count=5)
        assert repaired["status"] == "STALE_DEPENDENCY_MANIFEST_REPAIRED"
        assert repaired["repaired"] is True and repaired["fatal"] is False
        fatal = _coverage_consistency(
            expected_keys={(2, "C1", "WF01"), (2, "C1", "WF02")},
            observed_keys={(2, "C1", "WF01")},
            duplicate_keys=0, frame_empty=False,
            candidate_count=5, candidate_registry_count=5)
        assert fatal["status"] == "FATAL_COVERAGE_INCONSISTENCY"
        assert fatal["fatal"] is True and fatal["repaired"] is False
        profile_path = root / "job-memory-profiles.json"
        admission = RamAdmissionScheduler(
            max_slots=26,
            fill_floor_fraction=.80,
            target_fraction=.86,
            stop_fraction=.90,
            hard_limit_fraction=.95,
            profile_path=profile_path)

        # v40.0.4 retains four staged GPU-producing futures per physical
        # device. Actual kernels are still serialized by the section queue and
        # cross-process device lock; a fifth future is rejected.
        assert admission.available_gpu_devices(2) == (0, 1)
        for index in range(4):
            assert admission.try_acquire_gpu_device(
                0, f"gpu0-staged-{index}")
            if index < 3:
                assert admission.available_gpu_devices(2) == (0, 1)
        assert admission.available_gpu_devices(2) == (1,)
        assert not admission.try_acquire_gpu_device(
            0, "gpu0-fifth")
        for index in range(4):
            assert admission.try_acquire_gpu_device(
                1, f"gpu1-staged-{index}")
        assert admission.available_gpu_devices(2) == ()
        admission.release_gpu_device(0, "gpu0-staged-0")
        assert admission.available_gpu_devices(2) == (0,)
        for index in range(1, 4):
            admission.release_gpu_device(0, f"gpu0-staged-{index}")
        for index in range(4):
            admission.release_gpu_device(1, f"gpu1-staged-{index}")
        assert admission.available_gpu_devices(2) == (0, 1)

        assert admission.max_admitted_slots(
            job_class="candidate_oos:HGB:GPU") == 2
        assert abs(
            admission.incremental_estimate_for(
                "candidate_oos:HGB:GPU") - 1.25
        ) < 1e-12
        assert admission.max_admitted_slots(
            job_class="causal:coverage:CPU") == 26
        assert abs(
            admission.incremental_estimate_for(
                "causal:coverage:CPU") - 2.75
        ) < 1e-12

        # Low real load is authoritative for scheduling. Historical absolute
        # worker RSS must not hold a 30%-loaded host at 12-13 jobs.
        gib = 1024 ** 3
        admission._total_bytes = 100 * gib
        admission._latest_used_bytes = lambda: 30 * gib
        admission.real_load_state = lambda: {
            "used_bytes": 30 * gib,
            "used_fraction": .30,
            "slope_gib_per_s": 0.0,
            "live_jobs": [],
            "live_job_count": 0,
        }
        first_lease = admission.try_acquire(
            job_class="candidate_oos:HGB:GPU",
            descriptor={"horizon": 11, "prepared": True})
        second_lease = admission.try_acquire(
            job_class="candidate_oos:HGB:GPU",
            descriptor={"horizon": 11, "prepared": True})
        third_lease = admission.try_acquire(
            job_class="candidate_oos:HGB:GPU",
            descriptor={"horizon": 11, "prepared": True})
        assert first_lease is not None and second_lease is not None
        assert third_lease is None  # two physical GPUs remain the class cap

        # A lease must enter the safety projection before the child has written
        # its first 10-ms shared-memory sample. This closes the submit->RSS
        # visibility race across concurrent seed scheduler threads.
        race = RamAdmissionScheduler(max_slots=26)
        race._total_bytes = 100 * gib
        race._latest_used_bytes = lambda: 80 * gib
        race.real_load_state = lambda: {
            "used_bytes": 80 * gib,
            "used_fraction": .80,
            "slope_gib_per_s": 0.0,
            "live_jobs": [],
            "live_job_count": 0,
        }
        race_lease = race.try_acquire(
            job_class="candidate_oos:PREP_RIDGE:CPU",
            descriptor={
                "horizon": 11, "prepared": False,
                "train_sessions": 1000,
            })
        assert race_lease is not None
        assert race._unpublished_lease_gib() >= race_lease - 1e-12
        assert race._safety_projected_fraction(0.0) > .80
        race.release(
            race_lease, job_class="candidate_oos:PREP_RIDGE:CPU")
        race.close()

        admission.release(
            first_lease, job_class="candidate_oos:HGB:GPU")
        admission.release(
            second_lease, job_class="candidate_oos:HGB:GPU")

        # In the upper corridor, content-aware packing must reject a large job
        # that would cross 90% while still allowing a small Evidence job.
        packing = RamAdmissionScheduler(max_slots=26)
        packing._total_bytes = 10 * gib
        packing._latest_used_bytes = lambda: int(8.7 * gib)
        packing.real_load_state = lambda: {
            "used_bytes": int(8.7 * gib),
            "used_fraction": .87,
            "slope_gib_per_s": 0.0,
            "live_jobs": [],
            "live_job_count": 0,
        }
        large = packing.try_acquire(
            job_class="candidate_oos:HGB:GPU",
            descriptor={"horizon": 11, "prepared": True})
        small = packing.try_acquire(
            job_class="causal:evidence:CPU",
            descriptor={"kind": "evidence", "prepared": True})
        assert large is None
        assert small is not None
        packing.release(small, job_class="causal:evidence:CPU")

        # Outstanding leases count against the 90% stop band before their
        # children publish RSS. Shared seed schedulers therefore cannot spend
        # the same headroom twice.
        atomic = RamAdmissionScheduler(max_slots=26)
        atomic._total_bytes = 100 * gib
        atomic._latest_used_bytes = lambda: 89 * gib
        atomic.real_load_state = lambda: {
            "used_bytes": 89 * gib,
            "used_fraction": .89,
            "slope_gib_per_s": 0.0,
            "live_jobs": [],
            "live_job_count": 0,
        }
        atomic_first = atomic.try_acquire(estimated_task_gib=.6)
        atomic_second = atomic.try_acquire(estimated_task_gib=.6)
        assert atomic_first is not None
        assert atomic_second is None
        atomic.release(atomic_first)
        atomic.close()

        # RAM slope is a short look-ahead reservation, not a fixed veto. A
        # moderate allocator burst may still admit work when projected load
        # stays inside the corridor, while an extreme rise remains blocked.
        trend = RamAdmissionScheduler(
            max_slots=26, trend_lookahead_seconds=.25)
        trend._total_bytes = 100 * gib
        trend._latest_used_bytes = lambda: 70 * gib
        trend.real_load_state = lambda: {
            "used_bytes": 70 * gib,
            "used_fraction": .70,
            "slope_gib_per_s": 5.0,
            "live_jobs": [],
            "live_job_count": 0,
        }
        moderate = trend.try_acquire(
            job_class="causal:evidence:CPU",
            descriptor={"kind": "evidence", "prepared": True})
        assert moderate is not None
        trend.release(moderate, job_class="causal:evidence:CPU")
        trend.real_load_state = lambda: {
            "used_bytes": 70 * gib,
            "used_fraction": .70,
            "slope_gib_per_s": 120.0,
            "live_jobs": [],
            "live_job_count": 0,
        }
        assert trend.try_acquire(
            job_class="candidate_oos:PREP_RIDGE:CPU",
            descriptor={"horizon": 21, "prepared": False}) is None
        trend.close()

        # Learned profiles keep absolute RSS for safety and incremental RSS for
        # scheduling. The latter is the quantity used for best-fit packing.
        descriptor = {
            "horizon": 11, "prepared": True,
            "kind": "candidate_oos_fold"}
        admission.observe(
            "candidate_oos:RIDGE:CPU",
            peak_process_gib=5.8,
            incremental_peak_gib=.72,
            descriptor=descriptor)
        admission.observe(
            "candidate_oos:RIDGE:CPU",
            peak_process_gib=6.1,
            incremental_peak_gib=.80,
            descriptor=descriptor)
        assert profile_path.is_file()
        resumed_admission = RamAdmissionScheduler(
            max_slots=26, profile_path=profile_path)
        learned_incremental = resumed_admission.incremental_estimate_for(
            "candidate_oos:RIDGE:CPU", descriptor)
        learned_absolute = resumed_admission.absolute_estimate_for(
            "candidate_oos:RIDGE:CPU", descriptor)
        assert .72 <= learned_incremental <= .80
        assert 5.8 <= learned_absolute <= 6.1

        # Repeated tiny/cache-like samples may lower the p90, but the scheduling
        # estimate keeps 25% of the cold default as a floor for a future real fit.
        cache_descriptor = {
            "horizon": 11, "prepared": True,
            "kind": "candidate_oos_fold",
            "train_sessions": 900,
        }
        for _ in range(40):
            admission.observe(
                "candidate_oos:RIDGE:CPU",
                peak_process_gib=1.0,
                incremental_peak_gib=.001,
                descriptor=cache_descriptor)
        assert admission.incremental_estimate_for(
            "candidate_oos:RIDGE:CPU",
            cache_descriptor) >= .25

        # Shared-memory job telemetry is live at 10 ms and carries actual RSS.
        live = admission.open_live_job(
            "causal:evidence:CPU",
            {"job_id": "live-test", "kind": "evidence"})
        with __import__(
            "stock_predictor.backtests.opportunity_portfolio_research.dynamic_qbd_manifested_job_coordinator",
            fromlist=["_JobMemorySampler"],
        )._JobMemorySampler(
            interval_seconds=.01, live_config=live
        ):
            time.sleep(.03)
            snapshot = admission._live_board.snapshot()
            assert any(
                row.get("job_id") == "live-test"
                and int(row.get("rss_bytes", 0)) > 0
                for row in snapshot)
        admission.close_live_job(live)

        telemetry = admission.telemetry()
        assert telemetry["ram_scheduler_contract"] == (
            "DQBD_SINGLE_LANE_RECLAIM_CLOSED_LOOP_V40_0_4")
        assert telemetry["ram_scheduler_sample_interval_ms"] == 10
        assert telemetry[
            "ram_scheduler_incremental_cold_floor_fraction"] == .25
        assert telemetry["ram_scheduler_fill_floor_fraction"] == .80
        assert telemetry["ram_scheduler_target_fraction"] == .86
        assert telemetry["ram_scheduler_stop_fraction"] == .90
        assert telemetry["ram_scheduler_reclaim_fraction"] == .92
        assert telemetry["ram_scheduler_hard_limit_fraction"] == .95
        assert telemetry["ram_scheduler_admission_policy"] == (
            "REAL_LOAD_10MS_DAMPED_RECOVERY_80_86_90_92_95_V40_0_4")

        # Content identity must distinguish materially different fold sizes.
        small_key = admission._descriptor_profile_key(
            "candidate_oos:RIDGE:CPU",
            {
                "horizon": 11, "prepared": True,
                "device_index": None, "train_sessions": 400,
            })
        large_key = admission._descriptor_profile_key(
            "candidate_oos:RIDGE:CPU",
            {
                "horizon": 11, "prepared": True,
                "device_index": None, "train_sessions": 1400,
            })
        assert small_key != large_key

        # Persisted diagnostics are compact rollups of the 10-ms controller,
        # not a 100-Hz disk stream.
        rollup_path = root / "real-load-runtime.jsonl"
        rollup = RamAdmissionScheduler(
            max_slots=4,
            telemetry_path=rollup_path,
            telemetry_rollup_seconds=.5)
        time.sleep(.03)
        rollup._write_runtime_rollup(time.monotonic())
        assert rollup_path.is_file()
        rollup_rows = [
            json.loads(line)
            for line in rollup_path.read_text(
                encoding="utf-8").splitlines()
            if line.strip()
        ]
        assert rollup_rows
        assert rollup_rows[-1]["schema_version"] == (
            "DQBD_REAL_LOAD_ROLLUP_V40_0_4")
        assert rollup_rows[-1]["source_sample_interval_ms"] == 10
        rollup.close()

        # v40.0.4 reclaim is edge-triggered. After one victim, old 250-ms
        # samples must not trigger a second kill during the settling window.
        reclaim = RamAdmissionScheduler(max_slots=4)
        reclaim._total_bytes = 100 * gib
        reclaim.real_load_state = lambda: {
            "used_bytes": int(90.5 * gib),
            "used_fraction": .905,
            "slope_gib_per_s": .5,
            "live_jobs": [],
            "live_job_count": 0,
        }
        rising = reclaim.reclaim_state()
        assert rising["reclaim_required"] is True
        assert rising["reclaim_rising"] is True
        reclaim.note_targeted_reclaim(
            job_id="victim", released_gib=4.0,
            usage_fraction=.905)
        settling = reclaim.reclaim_state()
        assert settling["reclaim_required"] is False
        assert settling["reclaim_settling"] is True
        assert reclaim.admission_budget(.70) == 0
        reclaim._settling_until_monotonic = time.monotonic() - .01
        reclaim._last_recovery_admission_monotonic = 0.0
        assert reclaim.admission_budget(.70) == 1
        assert reclaim.admission_budget(.70) == 0
        assert reclaim.admission_budget(.88) == 0
        reclaim._last_reclaim_monotonic = 0.0
        reclaim._recovery_until_monotonic = 0.0
        reclaim.real_load_state = lambda: {
            "used_bytes": int(92.1 * gib),
            "used_fraction": .921,
            "slope_gib_per_s": 0.0,
            "live_jobs": [],
            "live_job_count": 0,
        }
        hard = reclaim.reclaim_state()
        assert hard["reclaim_required"] is True
        assert hard["reclaim_hard"] is True
        reclaim.close()

        # Victim selection is progress-protected and best-fit to the 86%
        # target. It must not simply choose the largest RSS worker, and a job
        # already reclaimed once is protected behind an equivalent fresh job.
        victim_scheduler = RamAdmissionScheduler(max_slots=4)
        victim_scheduler._total_bytes = 100 * gib
        now = time.time()
        victim_rows = [
            {
                "job_id": "old-large", "pid": 101,
                "state": 1, "rss_bytes": 12 * gib,
                "start_bytes": 2 * gib,
            },
            {
                "job_id": "young-fit", "pid": 102,
                "state": 1, "rss_bytes": int(5 * gib),
                "start_bytes": int(.5 * gib),
            },
            {
                "job_id": "already-reclaimed", "pid": 103,
                "state": 1, "rss_bytes": int(5 * gib),
                "start_bytes": int(.5 * gib),
            },
        ]
        victim_scheduler.reclaim_state = lambda: {
            "used_bytes": int(90.5 * gib),
            "used_fraction": .905,
            "slope_gib_per_s": .5,
            "live_jobs": victim_rows,
            "live_job_count": 3,
            "reclaim_required": True,
            "reclaim_hard": False,
            "reclaim_rising": True,
            "reclaim_settling": False,
        }

        class FakeLanePool:
            def __init__(self):
                self._processes = {}
                self.reclaimed = []

            def active_lanes(self):
                return [
                    {
                        "lane_index": 0, "pid": 101,
                        "job_id": "old-large", "kind": "model",
                        "job_class": "causal:model:RIDGE:CPU",
                        "gpu_device_index": None,
                        "ram_reclaim_count": 0,
                        "submitted_at": now - 40.0,
                    },
                    {
                        "lane_index": 1, "pid": 102,
                        "job_id": "young-fit", "kind": "model",
                        "job_class": "causal:model:RIDGE:CPU",
                        "gpu_device_index": None,
                        "ram_reclaim_count": 0,
                        "submitted_at": now - 4.0,
                    },
                    {
                        "lane_index": 2, "pid": 103,
                        "job_id": "already-reclaimed", "kind": "model",
                        "job_class": "causal:model:RIDGE:CPU",
                        "gpu_device_index": None,
                        "ram_reclaim_count": 1,
                        "submitted_at": now - 4.0,
                    },
                ]

            def reclaim_job(self, job_id, *, reason, released_gib):
                self.reclaimed.append(job_id)
                return {
                    "job_id": job_id, "pid": 102,
                    "lane_index": 1, "reason": reason,
                    "released_gib": released_gib,
                }

        fake_pool = FakeLanePool()
        victim_controller = RamWorkerPauseController(
            victim_scheduler, fake_pool, root / "targeted-reclaim")
        victim_controller.reconcile()
        assert fake_pool.reclaimed == ["young-fit"]
        assert victim_scheduler.telemetry()[
            "ram_scheduler_targeted_reclaim_count"] == 1
        victim_scheduler.close()

        # The reclaim edge is global across all seed/CPU/GPU controllers.
        # Two callers may not independently consume the same >90% RAM edge.
        gate_scheduler = RamAdmissionScheduler(max_slots=4)
        gate_scheduler.reclaim_state = lambda: {
            "used_bytes": int(90.5 * gib),
            "used_fraction": .905,
            "slope_gib_per_s": .5,
            "live_jobs": [],
            "live_job_count": 0,
            "reclaim_required": True,
            "reclaim_hard": False,
            "reclaim_rising": True,
            "reclaim_settling": False,
        }
        first_gate = gate_scheduler.try_begin_targeted_reclaim()
        assert first_gate is not None
        assert gate_scheduler.try_begin_targeted_reclaim() is None
        assert gate_scheduler.admission_budget(.50) == 0
        assert gate_scheduler.try_acquire(
            .25, job_class="causal:evidence:CPU",
            descriptor={"job_id": "blocked-during-reclaim"}) is None
        gate_scheduler.end_targeted_reclaim()
        assert gate_scheduler.admission_budget(.50) > 0
        second_gate = gate_scheduler.try_begin_targeted_reclaim()
        assert second_gate is not None
        gate_scheduler.end_targeted_reclaim()
        gate_scheduler.close()

        # Victim ranking must also be global, not dependent on which pool's
        # reconcile loop happened to observe pressure first.
        global_scheduler = RamAdmissionScheduler(max_slots=4)
        global_scheduler._total_bytes = 100 * gib
        global_rows = [
            {
                "job_id": "pool-old-large", "pid": 201,
                "state": 1, "rss_bytes": 12 * gib,
                "start_bytes": 2 * gib,
            },
            {
                "job_id": "pool-young-fit", "pid": 202,
                "state": 1, "rss_bytes": int(5 * gib),
                "start_bytes": int(.5 * gib),
            },
        ]
        global_scheduler.reclaim_state = lambda: {
            "used_bytes": int(90.5 * gib),
            "used_fraction": .905,
            "slope_gib_per_s": .5,
            "live_jobs": global_rows,
            "live_job_count": 2,
            "reclaim_required": True,
            "reclaim_hard": False,
            "reclaim_rising": True,
            "reclaim_settling": False,
        }

        class OneLaneFakePool:
            def __init__(self, row):
                self.row = dict(row)
                self._processes = {}
                self.reclaimed = []

            def active_lanes(self):
                return [dict(self.row)]

            def reclaim_job(
                self, job_id, *, reason, released_gib,
            ):
                self.reclaimed.append(str(job_id))
                return {
                    "job_id": str(job_id),
                    "pid": self.row["pid"],
                    "lane_index": self.row["lane_index"],
                    "reason": reason,
                    "released_gib": released_gib,
                }

        old_pool = OneLaneFakePool({
            "lane_index": 0, "pid": 201,
            "job_id": "pool-old-large", "kind": "model",
            "job_class": "causal:model:RIDGE:CPU",
            "gpu_device_index": None,
            "ram_reclaim_count": 0,
            "submitted_at": time.time() - 40.0,
        })
        fit_pool = OneLaneFakePool({
            "lane_index": 0, "pid": 202,
            "job_id": "pool-young-fit", "kind": "model",
            "job_class": "causal:model:RIDGE:CPU",
            "gpu_device_index": None,
            "ram_reclaim_count": 0,
            "submitted_at": time.time() - 4.0,
        })
        old_controller = RamWorkerPauseController(
            global_scheduler, old_pool, root / "global-old")
        fit_controller = RamWorkerPauseController(
            global_scheduler, fit_pool, root / "global-fit")
        old_controller.reconcile()
        assert old_pool.reclaimed == []
        assert fit_pool.reclaimed == ["pool-young-fit"]
        global_scheduler.unregister_reclaim_controller(
            old_controller)
        global_scheduler.unregister_reclaim_controller(
            fit_controller)
        global_scheduler.close()

        # GPU routing learns full-job service times independently from RAM
        # admission. HGB remains ahead of Ridge on the GPU frontier.
        assert _candidate_gpu_workload_priority(
            "HGB", prepared=True) < _candidate_gpu_workload_priority(
                "RIDGE", prepared=True)
        duration_root = root / "duration-profiles"
        durations = _CandidateExecutionDurationProfiles(
            duration_root)
        device = {
            "vendor": "NVIDIA", "name": "RTX TEST",
            "platform_index": 0, "device_index": 0,
        }
        cold_gpu = durations.estimate(
            "HGB:PREPARED", "GPU", device)
        durations.observe(
            "HGB:PREPARED", "GPU", 8.0, device)
        durations.observe(
            "HGB:PREPARED", "GPU", 10.0, device)
        learned_gpu = durations.estimate(
            "HGB:PREPARED", "GPU", device)
        assert learned_gpu != cold_gpu
        assert 8.0 <= learned_gpu <= 10.0
        reloaded_durations = _CandidateExecutionDurationProfiles(
            duration_root)
        assert reloaded_durations.estimate(
            "HGB:PREPARED", "GPU", device) == learned_gpu
        # Physical-device breadth wins over raw learned speed: a slower empty
        # GPU must receive work before a faster GPU gets another staged Future.
        assert _choose_gpu_feed_device(
            (0, 1),
            pending_depths={0: 1, 1: 0},
            predicted_finishes={0: 10.0, 1: 20.0},
        ) == 1
        # Once queue depth is equal, learned finish time breaks the tie.
        assert _choose_gpu_feed_device(
            (0, 1),
            pending_depths={0: 1, 1: 1},
            predicted_finishes={0: 10.0, 1: 20.0},
        ) == 0

        # Killing one physical lane must not break another lane's Future.
        # This is the regression that made v40.0.3 pool-wide reclaim unusable.
        lane_pool, lane_count = _manifested_pool(
            workers=2, role="lane-reclaim-self-test",
            telemetry_root=root / "lane-seed" / "telemetry")
        assert lane_count == 2
        lane_a = lane_pool.submit(
            _lane_reclaim_self_test_worker,
            {
                "job": {"job_id": "lane-a", "kind": "test"},
                "_ram_job_class": "causal:CPU:CPU",
                "sleep_seconds": 5.0,
            })
        lane_b = lane_pool.submit(
            _lane_reclaim_self_test_worker,
            {
                "job": {"job_id": "lane-b", "kind": "test"},
                "_ram_job_class": "causal:CPU:CPU",
                "sleep_seconds": .35,
            })
        deadline = time.monotonic() + 5.0
        active = []
        while time.monotonic() < deadline:
            active = lane_pool.active_lanes()
            if (
                len(active) == 2
                and all(row.get("pid") for row in active)
            ):
                break
            time.sleep(.02)
        assert len(active) == 2
        reclaimed_lane = lane_pool.reclaim_job(
            "lane-a", reason="SELF_TEST", released_gib=1.0)
        assert reclaimed_lane is not None
        try:
            lane_a.result(timeout=2.0)
        except RamJobReclaimed:
            pass
        else:
            raise AssertionError(
                "targeted lane must surface RamJobReclaimed")
        assert lane_b.result(timeout=3.0)["job_id"] == "lane-b"
        assert lane_pool.idle_resident_lane_count() >= 1
        lane_c = lane_pool.submit(
            _lane_reclaim_self_test_worker,
            {
                "job": {"job_id": "lane-c", "kind": "test"},
                "_ram_job_class": "causal:CPU:CPU",
                "sleep_seconds": .01,
            })
        assert lane_c.result(timeout=3.0)["job_id"] == "lane-c"
        lane_pool.shutdown(wait=True)

        # A child can disappear without the parent receiving a usable relay
        # completion on Windows.  The health watchdog's primitive must detect
        # that orphaned lane, replace only it, and leave a retryable Future.
        orphan_pool, orphan_count = _manifested_pool(
            workers=1, role="orphan-lane-self-test",
            telemetry_root=root / "orphan-seed" / "telemetry")
        assert orphan_count == 1
        orphan_future = orphan_pool.submit(
            _lane_reclaim_self_test_worker,
            {
                "job": {"job_id": "orphan-a", "kind": "test"},
                "_ram_job_class": "causal:CPU:CPU",
                "sleep_seconds": 5.0,
            })
        deadline = time.monotonic() + 5.0
        orphan_pid = None
        while time.monotonic() < deadline:
            active = orphan_pool.active_lanes()
            if active and active[0].get("pid"):
                orphan_pid = int(active[0]["pid"])
                break
            time.sleep(.02)
        assert orphan_pid is not None
        for process in tuple(orphan_pool._processes.values()):
            if int(process.pid) == orphan_pid:
                process.terminate()
                process.join(timeout=1.0)
        reclaimed = orphan_pool.reclaim_unhealthy_lanes(
            startup_grace_seconds=0.0)
        assert len(reclaimed) == 1
        try:
            orphan_future.result(timeout=2.0)
        except BrokenProcessPool:
            pass
        else:
            raise AssertionError(
                "orphaned lane must surface BrokenProcessPool")
        orphan_retry = orphan_pool.submit(
            _lane_reclaim_self_test_worker,
            {
                "job": {"job_id": "orphan-retry", "kind": "test"},
                "_ram_job_class": "causal:CPU:CPU",
                "sleep_seconds": .01,
            })
        assert orphan_retry.result(timeout=3.0)["job_id"] == "orphan-retry"
        orphan_pool.shutdown(wait=True)

        # A targeted reclaim returns only that manifested job to PENDING and
        # persists anti-starvation history for its next admission.
        reclaim_root = root / "targeted-reclaim-store"
        reclaim_store = ManifestedJobStore(reclaim_root)
        reclaim_store.set_job(
            "reclaim-me", "PENDING", kind="model", depends_on=[])
        reclaim_claim = reclaim_store.claim_ready(
            "reclaim-test", lease_seconds=60,
            job_ids=("reclaim-me",))
        assert reclaim_claim is not None
        assert reclaim_store.requeue_reclaimed_job(
            "reclaim-me", "reclaim-test",
            reason="RAM_RISING_RECLAIM_ONE",
            released_gib=4.5,
            preferred_executor="GPU")
        reclaimed = reclaim_store.job("reclaim-me")
        assert reclaimed["state"] == "PENDING"
        assert reclaimed["ram_reclaim_count"] == 1
        assert reclaimed["execution_preference"] == "GPU"
        reclaim_claim2 = reclaim_store.claim_ready(
            "reclaim-test-2", lease_seconds=60,
            job_ids=("reclaim-me",))
        assert reclaim_claim2 is not None
        assert reclaim_claim2["ram_reclaim_count"] == 1
        reclaim_store.release_claim(
            "reclaim-me", "reclaim-test-2")
        reclaim_store.materialize_json_manifest()
        released_manifest = json.loads(
            (reclaim_root / "run-state" / "jobs.json").read_text(
                encoding="utf-8"))
        assert released_manifest["reclaim-me"]["state"] == "PENDING"

        # Cross-seed GPU routing must see the shared physical section backlog,
        # not only one scheduler's local Future set.
        backlog_root = root / "global-gpu-backlog"
        backlog_dir = (
            backlog_root / "gpu-section-ready-queue"
            / "platform-0-device-0")
        backlog_dir.mkdir(parents=True, exist_ok=True)
        (backlog_dir / "00-selftest.json").write_text(
            json.dumps({
                "schema_version": "DQBD_GPU_SECTION_READY_QUEUE_V40_0_3",
                "ticket_id": "selftest",
                "created_at": time.time(),
                "pid": os.getpid(),
                "thread_id": 0,
                "workload": "HGB",
                "priority": 0,
                "expected_seconds": 7.0,
                "platform_index": 0,
                "device_index": 0,
                "name": "TEST_GPU",
            }),
            encoding="utf-8")
        assert _gpu_global_backlog_seconds(
            backlog_root,
            [{
                "platform_index": 0,
                "device_index": 0,
                "name": "TEST_GPU",
                "vendor": "TEST",
            }],
            0,
        ) == 7.0

        # The Windows RSS sampler must remain live; a silent zero fallback
        # would make adaptive profiles look valid while learning nothing.
        from .dynamic_qbd_manifested_job_coordinator import _current_process_rss_bytes
        assert _current_process_rss_bytes() > 0
        assert _is_gpu_backend_failure(MemoryError("host oom")) is False
        assert _is_gpu_backend_failure(
            RuntimeError("OpenCL kernel device failure")) is True

        admission.close()
        resumed_admission.close()
        packing.close()

        # Runtime monitoring must remain best-effort when tasklist returns no
        # stdout (for example while the Windows service is restarting).
        monitor = GpuRuntimeMonitor(root / "gpu-telemetry")
        assert isinstance(monitor._libre_hardware_monitor(), dict)
        # Execution identity must include the device-specific HGB kernel
        # policy.  In particular Radeon max_bin=15 and NVIDIA max_bin=63 may
        # never share an immutable Candidate-OOS / Generation cache identity.
        try:
            devices = __import__(
                "stock_predictor.backtests.opportunity_portfolio_research.dynamic_qbd_gpu_pretraining",
                fromlist=["opencl_fp64_devices"],
            ).opencl_fp64_devices()
        except Exception:
            devices = []
        if len(devices) >= 2:
            fingerprint0 = _gpu_execution_fingerprint("HGB", 0)
            fingerprint1 = _gpu_execution_fingerprint("HGB", 1)
            assert fingerprint0 != fingerprint1
        # Scheduler-only contract changes must preserve completed scientific
        # checkpoints; numerical backend changes still invalidate them.
        migration_root = root / "execution-contract-migration"
        migration_store = ManifestedJobStore(migration_root)
        migration_store.set_job(
            "input", "COMPLETE", kind="input", depends_on=[])
        migration_store.set_job(
            "fit", "COMPLETE", kind="model", depends_on=["input"])
        base_backend = {
            "ridge_execution_backend": "RIDGE_V1",
            "hgb_execution_backend": "HGB_V1",
            "cpu_fallback_backend": "CPU_V1",
            "hgb_device_policy": "DEVICES_V1",
            "hgb_kernel_policy": "KERNEL_V1",
            "gpu_pretraining_contract_hash": "a" * 64,
            "execution_scheduler_policy": "SCHEDULER_V1",
        }
        first_migration = migration_store.ensure_execution_backend_contract(
            base_backend)
        assert first_migration["status"] == "NUMERICAL_BACKEND_MIGRATED"
        migration_store.set_job(
            "fit", "COMPLETE", kind="model", depends_on=["input"])
        compatible_backend = {
            **base_backend,
            "execution_scheduler_policy": "SCHEDULER_V2",
            "gpu_queue_policy":
                "HGB_FIRST_PREFETCHED_LEARNED_CPU_GPU_ROUTER_V40_0_4_3",
            "ram_admission_policy":
                "REAL_LOAD_10MS_DAMPED_RECOVERY_80_86_90_92_95_V40_0_4",
        }
        compatible = migration_store.ensure_execution_backend_contract(
            compatible_backend)
        assert compatible["status"] == "SCHEDULER_MIGRATED_COMPATIBLE"
        assert compatible["requeued"] == 0
        assert migration_store.job("fit")["state"] == "COMPLETE"
        numerical = migration_store.ensure_execution_backend_contract({
            **compatible_backend,
            "hgb_kernel_policy": "KERNEL_V2",
        })
        assert numerical["status"] == "NUMERICAL_BACKEND_MIGRATED"
        assert migration_store.job("fit")["state"] == "PENDING"

        # An explicit checkpoint tree may never silently fall back to a new
        # output-root DAG.  That fallback used to rebuild a large job list
        # when one seed directory was misspelled or missing.
        checkpoint = root / "checkpoint"
        try:
            _resolve_seed_root(seed_name="PRIMARY", output_root=root / "fresh",
                               checkpoint_root=checkpoint, short_source_root=None)
        except RuntimeError as error:
            assert str(error).startswith(
                "DQBD_STEP9_CHECKPOINT_ROOT_INCOMPLETE_NO_FRESH_JOB_GRAPH:PRIMARY:")
        else:
            raise AssertionError("explicit checkpoint root must fail closed")
    print("DYNAMIC_QBD_MANIFESTED_JOB_COORDINATOR_SELF_TEST_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
