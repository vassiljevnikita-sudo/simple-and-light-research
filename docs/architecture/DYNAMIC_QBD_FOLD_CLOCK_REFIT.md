# Dynamic-QBD Fold-Clock Refit Suite

## Problem and hypothesis

The matched A/B/C development experiment showed that monthly recalibration and
monthly fresh fitting can destroy the performance of a stale frozen model. The
fold-clock suite isolates whether a model can be refreshed sparsely when a new,
fully matured out-of-sample fold becomes available.

The refresh clock is evidence-based only. Calendar age, portfolio performance,
drawdown, regime labels, selector momentum, model age and optimized thresholds
are not triggers.

This is development evidence only. It creates no promotion or capital
authority.

## Fixed development cell

The default cell is the matched H3/D2/N1 replay:

- assessment window: `2020-08-31` through `2023-12-29`
- horizon: `H=3`
- holding period: `D=2`
- maximum names: `N=1`
- score quantile: `0.75`
- top fraction: `0.01`
- sleeve: `50%`
- round-trip cost: `20 bps`
- benchmark: `URTH`
- initial wealth: `EUR 10,000`
- exit: `FIXED_D2`

The first causally available recipe is frozen across all arms. There is no
recipe search on the evaluation window and no recipe switching.

## Arm contracts

`A_FROZEN_MODEL_FROZEN_CALIBRATION` fits one initial model and resolves one
initial calibration. Its model artifact, threshold and top fraction are then
frozen for the complete replay.

`F_FOLD_CLOCK_REFIT_FROZEN_BETWEEN_FOLDS` uses the same frozen recipe. It fits
and calibrates at the initial assessment, then does so again only when the
content of the fully matured fold set expands. Between those events the model
artifact and calibration are reused exactly.

`M_MONTHLY_REFIT_FROZEN_RECIPE` is the frequency control. It uses the same
frozen recipe but freshly fits and calibrates at every assessment.

The primary causal contrasts are:

- `F - A`: sparse evidence-clocked refresh versus a permanently frozen model
- `M - F`: monthly refit frequency versus the fold clock

The runner reports terminal value, terminal excess, CAGR, excess CAGR, relative
maximum drawdown, trade count and total costs for each contrast.

## Fold-clock definition

For every assessment, the selected frozen-recipe choice supplies the mature
fold state. The evidence fingerprint is a stable hash of exactly:

```text
fold_count
sorted(fold_ids)
```

The assessment date and latest matured date are deliberately excluded. If the
fingerprint is unchanged, calendar progress alone cannot cause an F refit. A
new fingerprint causes exactly one refit and exactly one calibration for F.

The `fold-clock-events.csv` ledger records the fold state, fingerprint,
expansion flag, refit reason, active artifact, fit cutoff, model age and active
calibration at every assessment.

## Causal and authority rules

The final holdout beginning `2026-07-25` remains closed. The runner rejects an
end date on or after that boundary. `promotion_allowed` is always false and
the authority is `SHADOW_ONLY_NO_PROMOTION_NO_HOLDOUT`.

No V5/factory retraining, prediction regeneration, H>30 extension, capital
authority, performance trigger, recipe hysteresis or parameter optimization is
introduced by this suite.

## Outputs and telemetry

The output root contains the compact audit and result files:

`summary.json`, `manifest.json`, `REPORT.md`, `portfolio-value-comparison.csv`,
`causal-contrasts.json`, `contract-audit.json`, `fold-clock-events.csv`,
`fit-audit.json`, `calibration-audit.csv`, `plan-audit.csv`, and one schedule,
NAV and trade file per arm.

Independent fit tasks use the existing suite resource contract: up to 32
workers with a bounded queue of up to 64 tasks and one native thread per
worker. `worker-telemetry.jsonl` records task compute time, idle time, worker
utilization and queue summary. The existing 60-GiB suite-wide resource policy
is reused.

## Interpretation fixed before the run

If F is close to A, sparse evidence-clocked refreshes are compatible with the
frozen-baseline architecture. If F is worse than A but better than M, refresh
frequency is implicated. If F is as poor as M, the fresh-fit transformation
itself is implicated. If F is better than A, it is a development candidate for
later, separately validated adaptation research. None of these outcomes is a
promotion decision.

## Commands

The deterministic contract self-test is deliberately provided for local
execution but is not run as part of this repository change:

```powershell
python -m stock_predictor.backtests.opportunity_portfolio_research.dynamic_qbd_fold_clock_refit_self_test
```

Run the real development experiment with the same compact input artifacts used
by the matched A/B/C suite:

```powershell
python -m stock_predictor.backtests.opportunity_portfolio_research.dynamic_qbd_fold_clock_refit_experiment `
  --signal-panel "<SIGNAL_PANEL>" `
  --candidate-metrics "<CANDIDATE_METRICS>" `
  --daily-store-root "<DAILY_STORE_ROOT>" `
  --benchmark-daily-path "<BENCHMARK_DAILY_PATH>" `
  --direct-daily-stock-root "<DIRECT_DAILY_STOCK_ROOT>" `
  --output-root "artifacts\dynamic-qbd-fold-clock-refit-h3-d2-n1" `
  --code-commit "<CODE_COMMIT_SHA>"
```

The expected fit counts are derived, not hardcoded: A has one fit, M has one
fit per assessment, and F has one initial fit plus one fit for each distinct
post-initial matured-fold evidence fingerprint.
