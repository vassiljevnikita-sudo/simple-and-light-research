> **Current-state note — 2026-08-27:** This file documents the Dynamic-QBD implementation contract and historical design. For the current scientific verdict, completed-result index, current 96-GB runtime contract and next Model-Store architecture, read [../../../research/DYNAMIC_QBD_CURRENT_STATE.md](../../../research/DYNAMIC_QBD_CURRENT_STATE.md), [../../../research/DYNAMIC_QBD_RESULTS_INDEX.md](../../../research/DYNAMIC_QBD_RESULTS_INDEX.md), [../../../research/DYNAMIC_QBD_DATA_AND_RUNTIME.md](../../../research/DYNAMIC_QBD_DATA_AND_RUNTIME.md) and [../../../research/CAUSAL_MODEL_STORE_ORCHESTRATOR_PLAN.md](../../../research/CAUSAL_MODEL_STORE_ORCHESTRATOR_PLAN.md). Where old hardware/runtime wording conflicts, the data/runtime document wins.

# Dynamic QBD implementation

This package implements the current build boundary in the normative Dynamic-QBD architecture.

## Runtime chain

`dynamic_qbd_family_surface.py` reuses the existing H1-H30/HxD search dimensions and adds the Dynamic-QBD N1-N6 design space without changing the legacy 48-policy contracts.

`dynamic_qbd_generation_contracts.py`, `dynamic_qbd_generation_registry.py`, `dynamic_qbd_maturity.py`, `dynamic_qbd_factory.py`, `dynamic_qbd_generation_recalibration.py`, `dynamic_qbd_h1_30_adapter.py`, and `dynamic_qbd_v5_adapter.py` implement durable family identity, concrete generation identity, horizon-specific outcome maturity, deterministic refits, generation-specific recalibration, failed-refit fallback, and the production-fit adapter to the locked H1-H30 panel. The legacy H5/H10/H20 V5 walk-forward entrypoint is not used as a monthly production fit.

The first production generation causally selects one Ridge/HGB candidate from every already completed, non-selection-only OOS fold and freezes that single recipe inside the family. It does not average model families, intersect their picks, or build an ensemble. Selection uses robust fold Spearman statistics with MAE only as a later tie-break; R-squared is retained as diagnostics and has no selection weight. The exact contributing fold IDs and these contracts are persisted with every generation. Later months only call `.fit()` on the exact moving `FamilySpec` training window; they do not silently reselect an algorithm or hyperparameters. Every fit leaves a purge gap, calibrates on a disjoint matured window, and writes a model-specific score artifact. Identical H/cutoff/training recipes are deduplicated across D/N policy variants. A generation maps to its `model_artifact_id`; C therefore trades the active model's scores rather than relabelling one static prediction series. A separate shadow-only `CAUSAL_RESELECT_AT_EACH_REFIT` contract may measure whether newly matured OOS folds justify a different recipe. It creates a fresh model and threshold for every selected recipe and has no promotion or holdout authority.

The authoritative local CLI targets approximately 90% measured CPU utilization while leaving all logical processors available. On the 8C/16T reference host this means a 16-thread native budget: limiting affinity to 14 threads would confuse runnable capacity with observed utilization and underutilize the host during I/O, scheduling and synchronization gaps. Memory-heavy model fits remain serial so full CPU access does not multiply panel RAM; independent lightweight portfolio comparisons may run concurrently. This resource contract changes execution only, never research selection or replay semantics.

`next_open_portfolio_replay.py`, `learned_exit_qbd_replay.py`, and `dynamic_qbd_portfolio_replay.py` remain the economic execution authority. They support generation-specific model scores, thresholds, top fractions and immutable family/generation/entry/exit lineage while retaining next-open execution, URTH/cash accounting, NAV-dependent sizing, costs, taxes, max-names and replacement mechanics. Learned-exit families receive monthly E1-E(D-1) artifacts and the provider resolves every decision through the position's original exit-generation identity.

The 50% stock sleeve is an entry-notional cap, not a continuous-rebalancing target. New buys cannot exceed the sleeve at execution; subsequent mark-to-market drift is retained economically and reported through `max_stock_exposure` and `sleeve_mark_to_market_breach_days`. No hidden trimming trade is introduced.

The current generated family surface is explicitly `PRE_TAX` (`enabled=false`). A future `DE_RETAIL_APPROX` family must set `enabled=true` and carry its exact rate/allowance parameters; contradictory contracts fail closed.

`dynamic_qbd_wealth_metrics.py` and `dynamic_qbd_evidence.py` produce the benchmark-relative wealth-path and monthly causal family/current-generation evidence layers.

`dynamic_qbd_abc_schedules.py`, `dynamic_qbd_gate1.py`, `dynamic_qbd_incremental_gates.py`, and `dynamic_qbd_development_evaluation.py` implement the A/B/C factory-value ablation and Gates 1/1B/2/3. A/B/C hold q/top fixed, so B-A isolates threshold recalibration and C-B isolates refitting. Optional B2/C2 arms expose rolling q/top optimization as a separate experiment. Assessment time is the primary inference unit; policies are aggregated through ex-ante three-session H/D neighborhoods with the same exit family. Gate 1 emits independent 1M and 3M decisions. Gate 1B adds downside state; Gate 2 adds matured fit health on top of performance and accepted downside evidence; Gate 3 then adds market state on top of the complete preceding base.

`dynamic_qbd_development_pipeline.py` is the authoritative one-shot local development runner: locked H1-H30 panel -> monthly generations -> frozen-model recalibrations -> A/B/C schedules -> generation-authoritative replay -> matured evidence -> gates. It accepts either a consolidated price parquet or the canonical partitioned daily store, requires an explicit holdout contract, and writes `generations/`, `predictions/`, `calibration/`, `portfolio_paths/`, `abc/`, `evidence/`, `gate1/`, `gate1b/`, `gate2/`, `gate3/`, `freeze/`, `manifest.json`, and `REPORT.md`. `PRESERVE_HISTORICAL_LOCKBOX` uses the panel's locked boundary; `PROSPECTIVE_FROM_2026_07_25` treats all data through 2026-07-24 as development. No default silently chooses between them. GitHub Actions may run lightweight checks but are not part of this runtime contract.

Daily-store projections are reused only when their date, universe and source-partition identity contract matches. A completed run may resume without recomputation only after its run contract, pipeline-summary fingerprint and every manifest artifact hash have been verified.

`dynamic_qbd_factory_state_store.py`, `dynamic_qbd_family_replay_store.py`, and `dynamic_qbd_algorithm_freeze.py` provide atomic restart state and algorithm-level freeze semantics. Replay checkpoints restore the next-session cursor, cash, URTH units/basis, positions, pending orders, tax ledger, learned-exit plans and immutable lineage; tests require resumed and continuous NAV/trades to match. The freeze hashes the actual terminal pre-holdout operational state, active generations, evidence cursor and gate cursors. Future concrete generations are append-only registry records and are not frozen in advance.

## Authority boundary

The legacy Top-10 V4/activity/changepoint/promotion/router stack remains present for reproducibility but has no Dynamic-QBD capital-allocation authority. `dynamic_qbd_development_evaluation.py` never invokes it. Gate 1 failure is emitted as `GATE1_FAIL_NO_PERFORMANCE_SELECTOR_AUTHORITY`.

Champion/hysteresis remains disabled. Every gate output is `SHADOW_RESEARCH_ONLY`; even a positive gate has no capital-allocation authority in this implementation phase.

## Verification

Run:

```text
python -m stock_predictor.backtests.opportunity_portfolio_research.dynamic_qbd_self_test
python -m stock_predictor.backtests.opportunity_portfolio_research.dynamic_qbd_h1_30_self_test
```

The required real-data smoke is:

```text
python -m stock_predictor.backtests.opportunity_portfolio_research.dynamic_qbd_real_smoke \
  --signal-panel <locked-signal-panel.parquet> \
  --candidate-metrics <signal-candidate-metrics.json> \
  --learned-exit-candidate-metrics <exit-candidate-metrics.json> \
  --daily-store-root <daily-parquet-root> \
  --output-root <smoke-output> \
  --holdout-contract PRESERVE_HISTORICAL_LOCKBOX
```

The standard smoke derives the final three pre-lockbox months from the panel, runs fixed and learned-exit variants, asserts threshold/model/score changes, and performs its own second restart run. `--start` and `--end` may override the derived window together. `--fixed-only` is diagnostic and is not the standard smoke contract.

The tests cover future mutation, H1/H30 maturity, deterministic generations, frozen family recipes, failed-refit fallback, N1/N6 portfolio differences, accounting, model/entry/exit lineage across generation switches, family-evidence continuity, current-generation isolation, time-clustered A/B/B2/C/C2, all gate contracts, restart parity and algorithm freeze. The H1-H30 integration test additionally runs real Ridge/HGB selection, production fit, model-pure calibration, generation score materialisation, learned-exit generation materialisation and the complete synthetic one-shot pipeline. The real smoke requires H11/H24/H28 × N1/N3/N6, multiple monthly generations, changed scores and thresholds, observed trade lineage, locked-holdout protection, and an identical NAV semantic hash on restart.
