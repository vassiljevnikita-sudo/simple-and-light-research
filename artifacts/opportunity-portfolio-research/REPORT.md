# Chronological V5 Opportunity Portfolio Research

Decision: `RESEARCH_CONTRACT_CHANGED_RERUN_REQUIRED` / `NOT_READY_FOR_FINAL_HOLDOUT`

The locked final holdout was not opened.

## Why the dad597c Development replay is superseded

The accounting fix was valid, but the subsequent research contract was still inconsistent with Opportunity-Core:

- outer walk-forward policy selection resolved `score_quantile` across all stock rows rather than the distribution of daily top scores;
- the Development run used `StageABudget=4`, which collapsed the sampled coarse search to `max_names=1` and `holding_days=1` and skipped dynamic exits;
- the later full-Development replay implicitly recalibrated the threshold from daily top scores, so it was not policy-identical to the outer-fold tests;
- readiness was effectively gated only by accounting/exposure invariants;
- `parameter_plateau.csv` was not a real neighborhood test;
- regime aggregation used last/first NAV across disjoint regime days;
- concentration diagnostics did not test ticker dependence or explicit exclusion replays.

Therefore the previous 18 fold portfolio values and frozen H5/H10/H20 policies must not be used to decide whether the model has tradable alpha.

## Corrected suite now on the branch

The code now:

- calibrates score thresholds from historical **daily top scores**, matching Opportunity-Core;
- persists and reuses the exact numeric `resolved_threshold` in all Development/cost/tax replays;
- uses balanced Stage-A coverage across score quantile, top fraction, max names and holding days (including H10-relevant 5/7/10-day holds);
- ranks candidates lexicographically by chronological prior-fold robustness rather than one combined historical CAGR;
- only explores dynamic exits around robust coarse candidates;
- bounds Python search concurrency to a conservative default of two workers and a hard cap of four;
- limits BLAS/OpenMP native threads per worker in `run_local.ps1` to prevent nested CPU oversubscription;
- calculates mathematically correct fold medians;
- performs actual parameter-neighbor replays;
- attributes daily returns to regimes rather than comparing disjoint first/last NAVs;
- adds trade/ticker concentration diagnostics, EXCLUDE_BEST_1, EXCLUDE_BEST_5 and EXCLUDE_TOP_TICKER replays;
- requires research-readiness gates before any `FROZEN_POLICY_READY_FOR_FINAL_HOLDOUT` status can be emitted.

## CI status

GitHub Actions run `31263920860` was created for the corrected self-test workflow, but GitHub did not start a runner because it reported a Billing/Spending-Limit account condition. This is not a code-test result. Local compile/self-test is still required.

## Next required run

Use the canonical local V5 predictions and daily store. Recommended starting configuration for the user's CPU constraint:

```powershell
.\stock_predictor\backtests\opportunity_portfolio_research\run_local.ps1 `
  -V5Predictions "D:\simple-and-light-v5-local-training\artifacts\v5-local\training\signal\selected-walk-forward-predictions.parquet" `
  -DailyStoreRoot "D:\simple-and-light-v5-local-training\artifacts\daily-parquet" `
  -OutputRoot "artifacts\opportunity-portfolio-research" `
  -StageABudget 48 `
  -MaxWorkers 2 `
  -NativeThreadsPerWorker 1
```

For the most conservative CPU usage, use `-MaxWorkers 1`.

Do not pass `-OpenFinalHoldout`.
