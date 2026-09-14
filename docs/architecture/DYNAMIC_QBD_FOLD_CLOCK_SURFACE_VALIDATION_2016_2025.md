> **Result interpretation update — 2026-08-27:** The completed surface-wide run is valuable fixed-Recipe mechanistic evidence: monthly refit was robustly worse than Fold-Clock, while Fold-Clock vs frozen failed full robustness. The horizon Recipes were selected from broader Development evidence overlapping the analyzed era, so this run must not be treated as causal historical Recipe-selection proof. See [../../research/DYNAMIC_QBD_CURRENT_STATE.md](../../research/DYNAMIC_QBD_CURRENT_STATE.md).

# Dynamic-QBD Fold-Clock Surface Validation 2016-2025

This is a development-only validation of sparse, evidence-clocked entry-model
refreshes across the complete H1-H30 H/D/N/exit surface. It does not open the prospective final
holdout, grant promotion authority, or create capital authority.

## Scientific question

Can a model be refreshed when a new fully matured out-of-sample fold becomes
available, without the damage observed under monthly fresh refitting?

The suite evaluates 5,400 predeclared families: H1-H30, every D1-D_H, N1-N6,
and both FIXED and LEARNED_EXIT. Learned-exit providers are fit once at the
first causal activation and then matched and frozen across the entry arms.

## Matched arms

- `A_FROZEN_MODEL_FROZEN_CALIBRATION`: one causal fit and one calibration,
  frozen for the complete replay.
- `F_FOLD_CLOCK_REFIT_FROZEN_BETWEEN_FOLDS`: same initial recipe; refit and
  recalibrate exactly once for each strict expansion of sorted matured fold IDs,
  then freeze until the next expansion.
- `M_MONTHLY_REFIT_FROZEN_RECIPE`: same initial recipe and monthly fresh fit
  plus calibration, as the frequency control.

The primary contrasts are `F - A` and `M - F`. Recipe identity is never
switched. The fold fingerprint contains only fold count and sorted fold IDs;
calendar progress alone is not evidence.

## Resource and holdout contract

The runner applies a bounded suite-wide 90 GiB hard / 84 GiB soft memory
contract on the 96-GiB development host and an 80% CPU capacity target. On a
32-logical-CPU host this is 26 worker slots, with one native numerical thread
per worker. The replay stage uses twenty-four isolated horizon processes and
dynamically queues the next horizon as soon as one process finishes; with 26
total worker slots the default split is one family thread per horizon process.
Each horizon task is pinned to one selected logical processor, so the
Python-heavy replay gets independent process-level execution lanes instead of
relying on GIL-limited threads inside one process. On Windows hybrid CPUs,
processor efficiency classes are ordered so higher-performance P-core
primaries are filled before E-core and SMT capacity; six logical processors
remain reserved. Twenty-four processes are the memory-safe default on the
observed 96-GiB host; a larger explicit value must remain subject to the
90 GiB hard cap. NWinfo sampling is diagnostic and fail-open.

The holdout contract is `PROSPECTIVE_FROM_2026_07_25`. Historical
`holdout_locked` markers dated before that boundary are audited as legacy
metadata and do not block a development replay ending before the boundary.

## CLI

```powershell
$sha = git rev-parse HEAD
python -m stock_predictor.backtests.opportunity_portfolio_research.dynamic_qbd_fold_clock_surface_validation_2016_2025 `
  --signal-panel "D:\\simple-and-light-v5-h1-30-artifacts\\datasets\\primary\\signal-panel.parquet" `
  --candidate-metrics "D:\\simple-and-light-v5-h1-30-artifacts\\training\\signal-candidate-metrics.json" `
  --learned-exit-candidate-metrics "D:\\simple-and-light-v5-h1-30-artifacts\\training\\e1-30-learned-exit-20260809\\exit-candidate-metrics.json" `
  --daily-store-root "D:\\simple-and-light-v5-h1-30\\artifacts\\daily-parquet" `
  --benchmark-daily-path "D:\\simple-and-light-opportunity-portfolio-qbd-60gib-run\\artifacts\\alpaca-urth-daily-repair\\URTH-direct-daily.parquet" `
  --direct-daily-stock-root "D:\\simple-and-light-opportunity-portfolio-qbd\\artifacts\\alpaca-stock-direct-daily-store" `
  --output-root "artifacts\\dynamic-qbd-surface-wide-fold-clock-validation" `
  --code-commit $sha
```

The defaults are the fixed Development 2016-2025 cell (`2016-06-24` through
`2025-12-31`, score quantile `.975`, top fraction `.005`, EUR 10,000, 20 bps,
50% sleeve). No parameter search is performed on this evaluation window.

## Outputs and interpretation

Compact outputs include `summary.json`, `REPORT.md`, `gates.json`,
`contract-audit.json`, `fit-audit.json`, `calibration-audit.csv`,
`plan-audit.csv`, `fold-clock-events.csv`, contrast CSVs, exit audit,
`nwinfo-summary.json`, and worker telemetry. Large family results, model
artifacts, signals, inputs, checkpoints, and sensor streams remain local.

The result is evidence about this development window only. A positive gate is
not a promotion decision. If F is near A and clearly better than M, sparse
fold-clock refresh is a candidate architecture for a separately designed
prospective test; otherwise the report records the relevant trade-off without
optimizing a new trigger.
