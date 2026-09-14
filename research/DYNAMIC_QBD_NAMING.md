# Dynamic-QBD Naming and Run IDs

Status: **CANONICAL NAMING CONTRACT**  
Effective: 2026-08-27

This document defines the canonical names for Dynamic-QBD modules, runs, validations, diagnostics and artifacts.

The purpose is to make it impossible to confuse temporal scope, design-space scope and completion status.

## 1. Reserved term: Full Run

**Full Run** is reserved exclusively for a run over the entire currently available historical dataset.

Canonical ID:

`DQBD_FULL_RUN_2016_2026_V1`

Temporal meaning:

- raw historical market coverage begins around 2016-01-01;
- benchmark coverage begins 2016-01-04;
- first usable Dynamic-QBD signal date is 2016-06-24 after feature warm-up;
- latest currently available forward data reaches 2026-07-24;
- therefore a signal-based Full Run has an effective decision window of 2016-06-24 through 2026-07-24.

A Full Run includes the prospective 2026 segment. Under the current holdout contract it is **not executable for iterative research** because the final prospective holdout remains closed.

Current status:

`DQBD_FULL_RUN_2016_2026_V1 = RESERVED_NOT_RUN`

There is currently **no completed artifact that may be called "the Full Run."**

## 2. Terms that are not Full Run

### Development Run

A chronological economic/research run restricted to the Development period.

Current completed canonical run:

`DQBD_DEVELOPMENT_RUN_2016_2025_V1`

Effective signal window:

- start: 2016-06-24;
- end: 2025-12-31.

Artifact:

`artifacts/dynamic-qbd-development-run-2016-2025-final/`

This is the completed 2,790-FIXED-family Development run with 1,632,350,262 matured prediction rows.

### Development Preflight

A setup/contract/DAG/data-boundary check that does not produce a valid completed economic portfolio result.

Canonical ID:

`DQBD_DEVELOPMENT_PREFLIGHT_2016_2025_NO_STOCK_V2`

Artifact:

`artifacts/dynamic-qbd-development-preflight-2016-2025-no-stock-v2/`

This must never be called a Development Run or Full Run.

### Surface Validation

A test across a broad H/D/N/exit design surface. "Surface" describes **parameter-space breadth**, not temporal breadth.

Canonical ID:

`DQBD_FOLD_CLOCK_SURFACE_VALIDATION_2016_2025_V1`

Artifact:

`artifacts/dynamic-qbd-fold-clock-surface-validation-2016-2025-20260826/`

This is the 5,400-family A/F/M Fold-Clock validation.

It is not a Full Run because it is Development-only and evaluates a specific refit mechanism.

### Cell Experiment

A matched experiment on one explicit H/D/N/exit cell.

Example:

`DQBD_FOLD_CLOCK_CELL_H03_D02_N01_FIXED_V1`

Artifact:

`artifacts/dynamic-qbd-fold-clock-refit-h3-d2-n1-rerun2/`

A cell experiment can motivate a surface validation. It cannot inherit the word "full."

### Counterfactual

A branched comparison from identical pre-event state.

Canonical current experiment:

`DQBD_FOLD_EVENT_COUNTERFACTUAL_DEVELOPMENT_V1`

Artifact:

`artifacts/dynamic-qbd-fold-event-counterfactual/`

Its Development contract is 2016-06-24 through 2025-12-31, while the actual non-initial Fold events occur in 2021-2023.

### Diagnostic

A research analysis that measures a mechanism or association without being a deployable selector/run.

Examples:

- `DQBD_REGIME_DIAGNOSTICS_DEVELOPMENT_V1`;
- `DQBD_OPPORTUNITY_STATE_DIAGNOSTIC_DEVELOPMENT_V1`;
- `DQBD_TRUE_CONSENSUS_DIAGNOSTIC_DEVELOPMENT_V1`.

### Preview

A chronological decision replay that lacks one or more ingredients required for an economic validation.

Example:

`TOP10_QBD_ROUTER_PREVIEW_NO_REALIZED_OUTCOMES_V1`.

### Validation

Use "validation" only when there is an explicit fixed contract and a defined gate/hurdle.

Validation does not imply promotion or holdout authority.

## 3. Completion words

These words have separate meanings:

- **COMPLETE** — the requested computation finished and required artifacts exist.
- **PASS** — a specific predeclared contract/gate passed.
- **FINAL artifact** — the last published revision of that historical run package; does not mean final holdout.
- **PROMOTED** — explicit model/architecture authority was granted. None of the current Dynamic-QBD results are promoted.
- **FULL** — must not be used as a synonym for COMPLETE, FINAL, broad, large or expensive.

Do not write "full run completed" when the intended meaning is merely "the requested Development run completed."

## 4. Canonical active module names

| Module | Exact role |
|---|---|
| `dynamic_qbd_development_pipeline.py` | Canonical large Development orchestration pipeline used for the completed 2016-2025 Development research path. |
| `dynamic_qbd_manifested_job_coordinator.py` | Older manifest/DAG job coordinator and preflight architecture. It is not the canonical completed Development pipeline. |
| `dynamic_qbd_fold_clock_surface_validation_2016_2025.py` | 5,400-family Development-only Fold-Clock surface validation. |
| `dynamic_qbd_fold_event_counterfactual.py` | FIXED-family OLD-vs-NEW Fold-event counterfactual. |
| `qbd_process_pool_readiness.py` | Prediction×Hold process-pool readiness/warm-up overlay. It is execution infrastructure, not a research run. |

Matching self-tests use the same base module name followed by `_self_test.py`.

## 5. Module naming pattern

New research modules should follow:

`dynamic_qbd_<mechanism>_<role>[_<explicit_scope>].py`

Good:

- `dynamic_qbd_fold_clock_surface_validation_2016_2025.py`
- `dynamic_qbd_fold_event_counterfactual.py`
- `dynamic_qbd_regime_diagnostics.py`
- `dynamic_qbd_development_pipeline.py`

Bad:

- `dynamic_qbd_full_run.py` unless it really is `DQBD_FULL_RUN_2016_2026`;
- `dynamic_qbd_full_development.py`;
- `dynamic_qbd_full_space_*.py`;
- `dynamic_qbd_pipeline2.py`;
- `dynamic_qbd_final.py`;
- `run_new.py`.

If two modules could be described with the same one-line purpose, their names or architecture are not sufficiently separated.

## 6. Artifact directory pattern

Canonical artifact directories use:

`<system>-<mechanism>-<role>-<scope>[-<revision/date>]`

Examples:

- `dynamic-qbd-development-run-2016-2025-final`
- `dynamic-qbd-development-preflight-2016-2025-no-stock-v2`
- `dynamic-qbd-fold-clock-surface-validation-2016-2025-20260826`
- `dynamic-qbd-fold-event-counterfactual`

Do not create another artifact directory whose name can reasonably be interpreted as the same run.

## 7. Legacy aliases

Historical commits and immutable result payloads may contain old names. These are provenance aliases only.

| Legacy name | Canonical meaning now |
|---|---|
| `dynamic_qbd_full_development.py` | `dynamic_qbd_manifested_job_coordinator.py` |
| `dynamic_qbd_pipeline.py` | `dynamic_qbd_development_pipeline.py` |
| `dynamic_qbd_full_space_fold_clock_validation.py` | `dynamic_qbd_fold_clock_surface_validation_2016_2025.py` |
| `qbd_full_pool_queue.py` | `qbd_process_pool_readiness.py` |
| `artifacts/dynamic-qbd-full-development-2025-final/` | `artifacts/dynamic-qbd-development-run-2016-2025-final/` |
| `artifacts/dynamic-qbd-full-development-2025-no-stock-v2/` | `artifacts/dynamic-qbd-development-preflight-2016-2025-no-stock-v2/` |
| `artifacts/dynamic-qbd-full-space-fold-clock-validation-20260826/` | `artifacts/dynamic-qbd-fold-clock-surface-validation-2016-2025-20260826/` |
| historical `DYNAMIC_QBD_FULL_SPACE_*` schema fields | legacy schema IDs for the Fold-Clock surface validation payload |
| historical `source_full_space_*` fields | provenance fields pointing to the Fold-Clock surface validation source run |
| historical `DYNAMIC_QBD_FULL_DEVELOPMENT_V1` | legacy schema ID from the older manifested job coordinator |

Do not rewrite immutable historical JSON merely to change these schema strings. Current code and documentation must use the canonical terms.

## 8. Local legacy paths

Some large local directories were created before this naming contract, for example:

`D:\simple-and-light-opportunity-portfolio-qbd-fullrun-2025\run`

That path is a **legacy filesystem path**, not a canonical run name.

Do not infer research scope from a legacy local directory name.

New local output roots must follow the canonical artifact/run terminology.

## 9. Full Run creation rule

A future artifact may use `full-run` only if all of these are true:

1. it consumes the complete currently defined 2016-2026 temporal dataset;
2. the prospective holdout has been explicitly authorized for that final evaluation;
3. the run contract states the exact effective start/end dates;
4. no earlier Development-only result is relabelled;
5. the artifact ID is `DQBD_FULL_RUN_2016_2026_...`;
6. `TOBECONTINUED.md` records that the holdout was intentionally opened.

Until then, "Full Run" remains reserved and unexecuted.

## 10. Agent rule

Before naming a new module or artifact, answer four questions:

1. **What mechanism?**
2. **What role?** run / preflight / diagnostic / validation / counterfactual / infrastructure.
3. **What time scope?**
4. **What design-space scope?**

The name must make the first two explicit and must make the latter two explicit whenever omission could create ambiguity.

If an agent encounters an old ambiguous name, it must consult this file rather than inventing a second interpretation.
