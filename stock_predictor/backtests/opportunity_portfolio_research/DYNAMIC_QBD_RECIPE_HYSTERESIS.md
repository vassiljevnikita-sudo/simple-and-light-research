# Dynamic-QBD Recipe-Switch Hysteresis

## Status

`SHADOW_ONLY_NO_PROMOTION_NO_HOLDOUT`

The final holdout begins at `2026-07-25` and remains fail-closed.

## Corrected evidence contract

The original V1 experiment incorrectly allowed calendar progression to create a
new evidence fingerprint because `latest_matured_evidence_date` was part of the
fingerprint. That meant two monthly assessments could count as two confirmations
even when the causal OOS fold set and recipe scores were unchanged.

That behavior is superseded.

The current fingerprint is content-based and excludes assessment/maturity date.
It includes the eligible recipe identities, exact hyperparameters, fold ids,
fold count and robust score.

Therefore:

```text
2022-08: HGB wins on folds 1..6 -> confirmation 1
2022-09: same folds 1..6       -> still confirmation 1
2022-10: same folds 1..6       -> still confirmation 1
```

A second confirmation requires a genuinely changed causal fold-content evidence
state. A tie-break-only winner still cannot switch, and both incumbent and
challenger require at least two causal OOS folds.

## State-machine regression test

```powershell
python -m stock_predictor.backtests.opportunity_portfolio_research.dynamic_qbd_recipe_hysteresis_self_test
```

The self-test now explicitly verifies that a later month with identical fold
content yields `NO_NEW_RECIPE_EVIDENCE` and cannot advance confirmation.

## Economic experiment

The previous three-arm monthly-refit experiment is retained only as historical
Development evidence. It refit every model every month and therefore does not
answer whether refitting itself should be conditional.

The active scientific test is:

`DYNAMIC_QBD_CONDITIONAL_REFIT.md`

Runner:

```powershell
python -m stock_predictor.backtests.opportunity_portfolio_research.dynamic_qbd_conditional_refit_experiment `
  --signal-panel "D:\path\to\signal-panel.parquet" `
  --candidate-metrics "D:\path\to\candidate-metrics.json" `
  --daily-store-root "D:\path\to\daily-parquet" `
  --benchmark-daily-path "D:\path\to\URTH-direct-daily.parquet" `
  --direct-daily-stock-root "D:\path\to\alpaca-stock-direct-daily-store" `
  --output-root "artifacts\dynamic-qbd-conditional-refit-h3-d2-n1"
```

The new four-arm test separates:

```text
C - B   pure unconditional monthly-refit value
D0 - B  immediate recipe-change conditional-refit value
D1 - B  hysteresis-gated conditional-refit value
```

No result from either experiment grants capital or promotion authority.
