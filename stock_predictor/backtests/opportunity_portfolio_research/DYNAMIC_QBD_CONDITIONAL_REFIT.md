# Dynamic-QBD Conditional Refit / Recipe Hysteresis

Status: `SHADOW_ONLY_NO_PROMOTION_NO_HOLDOUT`.

This test supersedes the scientific interpretation of the earlier monthly-refit
hysteresis experiment. It does not tune any parameter and does not open the
final holdout.

## Four fixed arms

```text
B_FROZEN_MODEL_ROLLING_RECALIBRATION
    initial causal model fit once
    same model artifact thereafter
    monthly threshold recalibration only

C_MONTHLY_REFIT_FROZEN_RECIPE
    same initial frozen recipe as B
    fresh model fit every monthly assessment
    monthly recalibration
    => C-B isolates unconditional model refitting

D0_NEW_EVIDENCE_RECIPE_CHANGE_REFIT
    starts from B
    no refit while recipe evidence is unchanged
    when a genuinely new OOS-fold evidence state changes the exact causal
    recipe, fit that recipe once
    otherwise only recalibrate
    => D0-B measures immediate conditional-refit value

D1_HYSTERESIS_GATED_CONDITIONAL_REFIT
    starts from B
    no refit while recipe evidence is unchanged
    challenger must win on two consecutive independent OOS-fold evidence
    expansions and have a strict robust-score edge
    only then is a fresh model fitted
    otherwise only recalibrate
    => D1-B measures hysteresis-gated conditional-refit value
```

## Evidence definition

Calendar progress is not evidence.

The evidence fingerprint excludes `assessment_date` and
`latest_matured_evidence_date`. It is derived from the causal recipe set,
including candidate identity, model family, exact hyperparameters, fold ids,
fold count and robust score.

Therefore this sequence cannot switch:

```text
2022-08: 6 folds, HGB winner -> confirmation 1
2022-09: same 6 folds, same scores -> still confirmation 1
2022-10: same 6 folds, same scores -> still confirmation 1
```

Only a changed fold-content evidence state can provide another confirmation.

## Default reproduction contract

```text
start             2020-08-31
end               2023-12-29
H                 3
D                 2
N                 1
score_quantile    0.75
top_fraction      0.01
sleeve            50%
round-trip cost   20 bps
initial capital   EUR 10,000
benchmark         URTH
```

## Run

```powershell
python -m stock_predictor.backtests.opportunity_portfolio_research.dynamic_qbd_recipe_hysteresis_self_test

python -m stock_predictor.backtests.opportunity_portfolio_research.dynamic_qbd_conditional_refit_experiment `
  --signal-panel "D:\path\to\signal-panel.parquet" `
  --candidate-metrics "D:\path\to\candidate-metrics.json" `
  --daily-store-root "D:\path\to\daily-parquet" `
  --benchmark-daily-path "D:\path\to\URTH-direct-daily.parquet" `
  --direct-daily-stock-root "D:\path\to\alpaca-stock-direct-daily-store" `
  --output-root "artifacts\dynamic-qbd-conditional-refit-h3-d2-n1"
```

The Dynamic-QBD resource contract is applied by the runner, including the
60-GiB Windows process-tree memory ceiling.

## Required interpretation

The primary contrasts are fixed before the run:

```text
C - B   unconditional monthly refit value
D0 - B  immediate conditional-refit value
D1 - B  hysteresis-gated conditional-refit value
```

Do not tune confirmation count, thresholds, evaluation dates or recipe margins
from these results.

## Outputs

- `portfolio-value-comparison.csv`
- `conditional-refit-decisions.csv`
- `hysteresis-decisions.csv`
- `fit-audit.json`
- `calibration-audit.csv`
- one schedule CSV per arm
- one NAV Parquet and trades Parquet per arm
- `summary.json`
- `REPORT.md`

Large fresh-model and raw prediction artifacts remain below the selected output
root and need not be committed.
