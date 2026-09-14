# Opportunity Portfolio Research

This package contains the active portfolio/QbD research stack behind Dynamic-QBD, plus preserved Top-10 and sequential QbD experiments used for audit and regression.

Do not infer current architecture from old artifact names or historical experiment documents. Start with:

- [../../../TOBECONTINUED.md](../../../TOBECONTINUED.md)
- [../../../research/DYNAMIC_QBD_CURRENT_STATE.md](../../../research/DYNAMIC_QBD_CURRENT_STATE.md)
- [../../../research/DYNAMIC_QBD_RESULTS_INDEX.md](../../../research/DYNAMIC_QBD_RESULTS_INDEX.md)
- [../../../research/DYNAMIC_QBD_DATA_AND_RUNTIME.md](../../../research/DYNAMIC_QBD_DATA_AND_RUNTIME.md)
- [../../../research/DYNAMIC_QBD_NAMING.md](../../../research/DYNAMIC_QBD_NAMING.md)

## Canonical vocabulary

Agents and maintainers should use the terms below consistently. Do not invent synonyms when one of these names applies.

| Term | Canonical meaning | Canonical module examples |
|---|---|---|
| **Contract** | Immutable rule, parameter contract, identity definition or allowed research surface. A contract module must not secretly execute research. | `cost_contracts.py`, `tax_contracts.py`, `portfolio_policy_contracts.py`, `market_regime_contracts.py`, `qbd_training_selection_contracts.py`, `dynamic_qbd_generation_contracts.py` |
| **Cost Contract** | Trading-cost assumptions/types only. Say “cost contract”, not generic “cost settings”. | `cost_contracts.py` |
| **Tax Contract** | Tax configuration types/assumptions only. | `tax_contracts.py` |
| **Portfolio Policy Contract** | Portfolio-policy identity and search-space constants such as horizon, threshold/quantile, holding, capacity, replacement and allocation. | `portfolio_policy_contracts.py` |
| **Training / Selection Contract** | Fold, target, Recipe-selection and model-training rules. | `qbd_training_selection_contracts.py` |
| **Generation Contract** | Immutable Dynamic-QBD family, Generation and factory identities/lifecycle semantics. | `dynamic_qbd_generation_contracts.py` |
| **Contract Fingerprint** | Stable hash used to identify a contract or immutable input definition. | `contract_fingerprints.py` |
| **Candidate OOS** | Causal out-of-sample Candidate evidence used for Recipe selection. `candidate_oos.py` owns the Candidate-OOS factory and immutable evidence store; “Candidate OOS” is one defined subsystem, not a generic label for every OOS result. | `candidate_oos.py`, `candidate_oos_self_test.py` |
| **Factory** | Creates fitted model artifacts / Generations from causally available information. It does not decide portfolio performance after the fact. | `dynamic_qbd_factory.py` |
| **Generation** | One immutable fitted model + calibration identity at one information cutoff. A later refit creates a new Generation; it does not mutate the old one. | `dynamic_qbd_generation_contracts.py`, `dynamic_qbd_generation_registry.py` |
| **Registry** | Persistent index/lifecycle mapping for Generations. A registry does not fit models. | `dynamic_qbd_generation_registry.py` |
| **Store** | Persistence/cache/state only. A Store must not introduce selection policy. | `dynamic_qbd_prediction_store.py`, `dynamic_qbd_factory_state_store.py`, `dynamic_qbd_family_replay_store.py`, `portfolio_replay_result_cache.py` |
| **Surface** | Explicit parameter/design-space coverage such as H×D×N×Exit. “Surface” describes breadth in parameter space, not time coverage. | `dynamic_qbd_family_surface.py`, `prediction_hold_qbd_surface.py`, `replacement_qbd_surface.py` |
| **Replay** | Portfolio simulation from already-defined signals/policies/generations. A replay must not fit/select a new model unless its contract explicitly says so. | `next_open_portfolio_replay.py`, `dynamic_qbd_portfolio_replay.py`, `learned_exit_qbd_replay.py` |
| **Search** | Searches a declared policy/Recipe space. The search space must come from a contract. | `portfolio_policy_search.py` |
| **Evaluation / Evaluator** | Reads completed result rows and computes summaries, paired comparisons or gates. It must not silently enlarge the search surface. | `dynamic_qbd_development_evaluation.py`, `prediction_hold_qbd_evaluate.py`, `replacement_qbd_evaluate.py` |
| **Runner** | Thin executable/CLI orchestration entrypoint. Business/research logic belongs in named modules below it. | `prediction_hold_qbd_runner.py`, `allocation_qbd_runner.py`, `portfolio_research_cli.py` |
| **Pipeline** | Multi-stage canonical orchestration where stage order is part of the architecture. Use only when the module really owns a pipeline. | `dynamic_qbd_development_pipeline.py` |
| **Coordinator** | Schedules jobs/dependencies/processes. “Coordinator” describes execution control, not scientific meaning. | `dynamic_qbd_manifested_job_coordinator.py`, `affinity_coordinator_pool.py` |
| **Runtime** | Execution/concurrency/resource machinery that must not alter the research contract. | `prediction_hold_qbd_throughput_runtime.py`, `portfolio_research_validation_runtime.py`, `dynamic_qbd_runtime_resources.py`, `dynamic_qbd_runtime_telemetry.py` |
| **Process-pool readiness** | Worker initialization/barrier infrastructure for one specific runtime. It is not a research run. | `prediction_hold_qbd_process_pool_readiness.py` |
| **Adapter** | Converts between concrete interfaces/data contracts without inventing new research policy. | `dynamic_qbd_h1_30_adapter.py`, `dynamic_qbd_v5_adapter.py` |
| **Diagnostic** | Measures/explains a mechanism or association. It has no selection, promotion or capital authority. | `dynamic_qbd_regime_diagnostics.py`, `dynamic_qbd_capacity_diagnostic.py`, `dynamic_qbd_recipe_objective_diagnostic.py` |
| **Experiment** | Predeclared causal comparison intended to answer one research question. | `dynamic_qbd_abc_recalibration_experiment.py`, `dynamic_qbd_fold_clock_refit_experiment.py` |
| **Validation** | Fixed contract + explicit gate/hurdle. Validation does not imply promotion. | `dynamic_qbd_fold_clock_surface_validation_2016_2025.py` |
| **Counterfactual** | Matched OLD/NEW branch from identical pre-event state. | `dynamic_qbd_fold_event_counterfactual.py` |
| **Self-test** | Deterministic contract/plumbing regression test. A self-test is not economic evidence. | files ending in `_self_test.py` |
| **Development Run** | Iterative historical research restricted to Development. Current canonical completed run: 2016-06-24 through 2025-12-31. | `DQBD_DEVELOPMENT_RUN_2016_2025_V1` |
| **Full Run** | Reserved exclusively for the entire defined 2016–2026 temporal dataset. It is currently `RESERVED_NOT_RUN`. Never use “full” to mean “large”, “all families”, “complete” or “final”. | `DQBD_FULL_RUN_2016_2026_V1` |

## Naming rules

A module name should answer what it owns. Prefer `<domain>_<mechanism>_<role>.py` over generic names.

Examples:

- costs → `cost_contracts.py`, not `contracts.py`;
- taxes → `tax_contracts.py`, not `tax.py`;
- portfolio simulation → `next_open_portfolio_replay.py`, not `portfolio.py`;
- policy search → `portfolio_policy_search.py`, not `search.py`;
- market regime classification → `urth_market_regime_classifier.py`, not `regimes.py`;
- research inputs → `portfolio_research_inputs.py`, not `data_loader.py`;
- Development-slice hashing → `development_slice_hash.py`, not `development_hash.py`;
- Dynamic-QBD generation identities → `dynamic_qbd_generation_contracts.py`, not `dynamic_qbd_contract.py`;
- Dynamic-QBD Generation registry → `dynamic_qbd_generation_registry.py`, not `dynamic_qbd_registry.py`;
- Dynamic-QBD portfolio replay → `dynamic_qbd_portfolio_replay.py`, not `dynamic_qbd_portfolio.py`;
- Prediction×Hold throughput runtime → `prediction_hold_qbd_throughput_runtime.py`, not `qbd_throughput_runtime.py`.

If two files can be described with the same one-line purpose, their names or responsibilities are still too ambiguous.

## Legacy import aliases

Two small compatibility aliases remain so older frozen scripts/surfaces can resolve without rewriting large historical modules solely for one import line:

- `replacement_qbd.py` → canonical `replacement_qbd_contract.py`;
- `concentration_replacement_qbd.py` → canonical `concentration_replacement_qbd_contract.py`.

These files contain no independent contract implementation. **Do not add new imports of the alias modules.** New code must import the canonical `*_contract.py` modules. They may be deleted once all preserved historical importers are migrated.

## Package layers

### 1. Shared portfolio contracts and replay infrastructure

Canonical modules include:

- `portfolio_policy_contracts.py` — portfolio policy identity/search constants;
- `cost_contracts.py` — cost assumptions;
- `tax_contracts.py` — tax configuration;
- `contract_fingerprints.py` — stable hashes;
- `portfolio_allocation_weights.py` — allocation arithmetic;
- `next_open_portfolio_replay.py` — stateful next-open portfolio accounting;
- `urth_market_regime_classifier.py` + `market_regime_contracts.py` — benchmark regime classification;
- `portfolio_research_inputs.py` — research prediction/price loading and holdout guards;
- `portfolio_policy_search.py` — portfolio-policy search;
- `portfolio_policy_walk_forward.py` — chronological policy walk-forward;
- `portfolio_replay_process_backend.py` — replay process backend;
- `portfolio_replay_fragment_cache.py` / `portfolio_replay_result_cache.py` — replay caches;
- `portfolio_resilient_process_pool.py` — resilient process-pool overlay;
- `affinity_coordinator_pool.py` — affinity-aware coordinator pool;
- `portfolio_research_validation_runtime.py` — validation execution runtime;
- `portfolio_research_cli.py` — general portfolio-research CLI.

### 2. Sequential portfolio-policy QbD

These experiments established the portfolio-policy surface before Dynamic-QBD:

- Prediction × Holding: `prediction_hold_qbd_*`;
- Allocation: `allocation_qbd_*`;
- Replacement: `replacement_qbd_*` with canonical contract `replacement_qbd_contract.py`;
- Concentration × Replacement: `concentration_replacement_qbd_*` with canonical contract `concentration_replacement_qbd_contract.py`;
- Learned Exit: `learned_exit_qbd_*`.

The Prediction × Holding execution-only modules are explicitly scoped:

- `prediction_hold_qbd_throughput_runtime.py`;
- `prediction_hold_qbd_process_pool_readiness.py`.

Important frozen lessons include `EQUAL_ACTIVE` allocation and `IGNORE_NEW` replacement.

### 3. Frozen Top-10 / controller research

Files prefixed `top10_` contain frozen forward replay, threshold diagnostics, memory/cohort/adaptation controllers and QBD Router V1 work.

Do not build another generic threshold/memory/regime/changepoint/consensus controller before checking [../../../research/DYNAMIC_QBD_RESULTS_INDEX.md](../../../research/DYNAMIC_QBD_RESULTS_INDEX.md).

### 4. Dynamic-QBD causal factory and evidence stack

Canonical modules include:

- `dynamic_qbd_generation_contracts.py` — Family / Generation / factory identity;
- `qbd_training_selection_contracts.py` — fold, target, Recipe-selection and training contracts;
- `candidate_oos.py` — causal Candidate-OOS factory + immutable evidence store;
- `dynamic_qbd_factory.py` — model/Recipe generation factory;
- `dynamic_qbd_generation_registry.py` — Generation registry/lifecycle;
- `dynamic_qbd_generation_registry_migration.py` — legacy registry migration audit;
- `dynamic_qbd_generation_recalibration.py` — Generation calibration;
- `dynamic_qbd_evidence.py` — matured family evidence;
- `dynamic_qbd_maturity.py` — label-maturity boundaries;
- `dynamic_qbd_prediction_store.py` — prediction persistence;
- `dynamic_qbd_factory_state_store.py` — factory-state persistence;
- `dynamic_qbd_family_replay_store.py` — segmented family replay persistence;
- `dynamic_qbd_portfolio_replay.py` — Generation-authoritative portfolio replay;
- `dynamic_qbd_family_surface.py` — H/D/N/exit Family surface;
- `dynamic_qbd_algorithm_freeze.py` — algorithm freeze manifest;
- `dynamic_qbd_abc_schedules.py` — A/B/C Generation schedules;
- `dynamic_qbd_development_evaluation.py` — Development result evaluation/gates;
- `dynamic_qbd_development_pipeline.py` — canonical Development orchestration;
- `dynamic_qbd_runtime_resources.py` — CPU/memory resource contract;
- `dynamic_qbd_runtime_telemetry.py` — optional NWinfo telemetry;
- `dynamic_qbd_manifested_job_coordinator.py` — older manifest/DAG coordinator/preflight architecture.

### 5. Dynamic-QBD research experiments and diagnostics

Current named entrypoints include:

- `dynamic_qbd_regime_diagnostics.py`;
- `dynamic_qbd_opportunity_state.py`;
- `dynamic_qbd_true_consensus.py`;
- `dynamic_qbd_paired_selector_tournament.py`;
- `dynamic_qbd_recipe_hysteresis_experiment.py`;
- `dynamic_qbd_conditional_refit_experiment.py`;
- `dynamic_qbd_abc_recalibration_experiment.py`;
- `dynamic_qbd_fold_clock_refit_experiment.py`;
- `dynamic_qbd_fold_clock_surface_validation_2016_2025.py`;
- `dynamic_qbd_fold_event_counterfactual.py`.

Focused contract tests now use descriptive names, including:

- `dynamic_qbd_end_to_end_self_test.py`;
- `dynamic_qbd_candidate_generation_portfolio_fixture_self_test.py`;
- `dynamic_qbd_causal_resume_contract_self_test.py`;
- `dynamic_qbd_identity_boundary_contract_self_test.py`.

## Current research surface

The H/D/N Family surface is:

- H1-H30;
- D1-D_H;
- N1-N6.

FIXED count:

```text
6 × sum(H=1..30) H = 2,790
```

FIXED + LEARNED_EXIT Fold-Clock surface validation:

- 5,400 families;
- 16,200 arm results;
- 10,800 contrasts;
- 110 structural plateaus.

Do not infer model-fit count from Family count. Fits are deduplicated across portfolio dimensions when H/Recipe/cutoff/training contract is identical.

## Historical data boundary

Current research data:

- raw historical coverage starts around 2016-01-01;
- benchmark daily begins 2016-01-04;
- canonical signal panel begins 2016-06-24 after feature warm-up;
- iterative Development ends 2025-12-31;
- prospective final holdout begins 2026-07-25 and remains closed.

The lack of pre-2016 history is a central limitation for the upcoming Model-Store pseudo-live experiment.

## Large-data architecture

The completed Development Run 2016-2025 processed:

- 1,632,350,262 matured prediction rows;
- 179,376 compact evidence rows.

A previous implementation collected matured fragments and performed one giant `pd.concat`, causing a multi-gigabyte allocation failure. The repaired architecture uses streaming/bounded caching and resumable stores.

When working here:

- partition heavy work by Horizon/Event/Generation;
- project only needed Parquet columns;
- keep caches bounded;
- preserve checkpoints;
- avoid model duplication across D/N;
- never load the complete prediction universe into one pandas object.

## Runtime baseline

Current large-run Development host/contract:

- 96 GB RAM;
- 32 logical CPUs;
- target 80% CPU capacity = 26 logical slots;
- normally 24 heavy process lanes;
- one native numerical thread per worker;
- 90 GiB suite-wide hard memory limit;
- 84 GiB soft target;
- Windows Job Object enforcement;
- process-affinity/topology-aware scheduling.

See [../../../research/DYNAMIC_QBD_DATA_AND_RUNTIME.md](../../../research/DYNAMIC_QBD_DATA_AND_RUNTIME.md) before changing concurrency.

## Research chronology

```text
Prediction × Hold
    ↓
Allocation
    ↓
Replacement / Concentration
    ↓
Learned Exit
    ↓
Frozen Top-10 forward diagnostics
    ↓
Threshold / memory / cohort / adaptation controllers
    ↓
QBD Router V1
    ↓
Dynamic-QBD causal factory
    ↓
Development Run 2016-2025
    ↓
Regime / opportunity / consensus diagnostics
    ↓
Paired selector tournament
    ↓
Recipe/refit/recalibration decomposition
    ↓
Fold-Clock H3 cell experiment
    ↓
Fold-Clock Surface Validation 2016-2025
    ↓
Fold-Event Counterfactual
    ↓
next: causal historical Model Store + Evidence Store + Orchestrator
```

## Scientific limitation

The latest Fold-Clock and Counterfactual suites use Horizon Recipes selected from broader Development evidence overlapping the period later analyzed. Their fixed-Recipe OLD/NEW comparisons are useful mechanistic diagnostics, but the Recipe assignments are hindsight-conditioned.

Do not claim that a historical live system would have selected the same Recipe/Generation. The next experiment must construct its initial model bank from a historical prefix and evolve chronologically. See [../../../research/CAUSAL_MODEL_STORE_ORCHESTRATOR_PLAN.md](../../../research/CAUSAL_MODEL_STORE_ORCHESTRATOR_PLAN.md).

## Execution and self-tests

Self-tests validate plumbing/contracts, not economic performance. Do not automatically execute all tests or large suites. Expensive real-data runs are normally launched locally by the user.

For a focused code task that does not explicitly request a large run:

- update the relevant focused self-test;
- perform static review;
- provide the exact command if useful;
- do not claim the economic run passed.

## Committed artifacts versus local state

Committed `artifacts/` packages usually contain compact evidence: summaries, reports, audits, contracts, contrasts and hashes. Large local state can additionally contain model artifacts, prediction stores, NAV/trades, learned-exit providers, Family-level Parquet and checkpoints.

Verify required local source artifacts before implementing a follow-up experiment. A compact GitHub summary may be insufficient.

## Safety

This package remains research/shadow only under the current Dynamic-QBD contract. Without explicit new authority, do not:

- place live orders or mutate broker state;
- promote a model/selector;
- open the prospective holdout for iterative tuning;
- use future information in model/calibration/orchestrator decisions;
- silently alter execution timing, costs or benchmark definitions to improve a result.

Experiment-specific Markdown files in this directory preserve exact historical contracts. For current status, use the root tracker/current-state/results-index/naming hierarchy first.
