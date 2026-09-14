# Top-10 Entry Activation and Alpha Diagnostic

Contract: `TOP10_ENTRY_ACTIVATION_AND_ALPHA_DIAGNOSTIC_V1`

## Purpose

Explain why the causal expanding-live Top-10 replay produced only three trades without retraining models, regenerating predictions, changing policies, or opening the final holdout.

The audit separates three questions:

1. **Activation** — do daily model scores cross the already-produced model-specific threshold?
2. **Calibration / score scale** — how does the threshold relate to the current score distribution (`max`, `p99`, margin, ratio)?
3. **Ranking alpha** — do high-ranked names subsequently outperform URTH even when the absolute threshold is not crossed?

The test is diagnostic only. It must not be used to select a new score quantile, top fraction, horizon, holding period, max-names value, or champion model from the 2023-2026 evaluation window.

## Inputs

Read-only inputs:

- completed `causal-expanding-entry-predictions.parquet` from the 17,020-checkpoint causal run;
- historical selected walk-forward Entry predictions;
- historical reproduction summary containing exact fold thresholds;
- frozen Top-10 policy manifest;
- daily market data for the prediction tickers and URTH;
- optional causal trade CSV for the three actually executed trades.

No Exit predictions are required for the diagnostic itself.

## Activation audit

For every frozen `(H, score_quantile, top_fraction)` activation contract and trading day, the audit records:

- universe size;
- score min / median / p90 / p95 / p97.5 / p99 / max;
- resolved threshold;
- `max_score - threshold`;
- `max_score / threshold`;
- `threshold / p99`;
- threshold crossing count and fraction;
- top-fraction limit;
- final eligible count before portfolio state constraints;
- whether relative candidates exist despite zero absolute crossings.

The same calculations are made for the historical WF folds using their exact reproduced fold thresholds.

## Alpha audit

For H11/H24/H28, future outcomes are attached only after prediction for evaluation:

- decision at session close;
- hypothetical entry at next session open;
- terminal value at the H-th future session close;
- gross relative excess = stock growth / URTH growth - 1;
- diagnostic net excess = gross relative excess - 20 bps.

Rows whose complete forward horizon is unavailable are right-censored and excluded, never treated as losses.

Per horizon the audit reports:

- Spearman rank correlation;
- mean realized net excess and hit rate for daily top 0.5%, 1%, and 5% score ranks;
- top-decile and bottom-decile mean realized excess;
- top-minus-bottom decile spread;
- monthly versions of the same diagnostics.

Future returns never feed a threshold, model, feature, or historical prediction.

## Heuristic diagnosis labels

The report may assign one of:

- `SCORE_SCALE_DRIFT`
- `CALIBRATION_TOO_RESTRICTIVE`
- `RANKING_ALPHA_DECAY`
- `BOTH_CALIBRATION_AND_ALPHA_DECAY`
- `HEALTHY_BUT_RARE_SIGNAL`
- `INCONCLUSIVE`

These are conservative diagnostic labels, not promotion decisions.

## Outputs

- `entry_activation_daily.parquet`
- `entry_activation_monthly.csv`
- `entry_activation_quarterly.csv`
- `entry_alpha_monthly.csv`
- `entry_alpha_horizon_summary.csv`
- `actual_crossing_examples.csv`
- `diagnostic_summary.json`
- `REPORT.md`

No model, checkpoint, prediction parquet, cache, or existing portfolio artifact is overwritten.

## Run

```powershell
.\scripts\run_top10_entry_activation_alpha_diagnostic.ps1 `
  -CausalPredictionRoot "D:\path\to\completed-causal-predictions" `
  -DailyStoreRoot "D:\path\to\daily-parquet"
```

The runner first executes the module self-test and verifies `prediction_summary.json` has status `CAUSAL_EXPANDING_PREDICTIONS_COMPLETE`.

## Methodological boundary

This test can determine whether the observed inactivity is primarily consistent with score/calibration drift or with deterioration of cross-sectional ranking skill. It does not resolve the separate point-in-time-universe promotion blocker and cannot promote a strategy by itself.
