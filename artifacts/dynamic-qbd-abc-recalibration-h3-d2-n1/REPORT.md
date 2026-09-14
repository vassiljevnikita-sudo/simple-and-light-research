# Dynamic-QBD matched A/B/C recalibration decomposition

Rolling recalibration diagnosis: **B_MINUS_A_ROLLING_RECALIBRATION_PARETO_WORSENS**
Fresh-refit diagnosis: **C_MINUS_B_FRESH_REFIT_PARETO_IMPROVES**

Authority: `SHADOW_ONLY_NO_PROMOTION_NO_HOLDOUT`. The final holdout remains closed.

## Scientific contract

- `A_FROZEN_MODEL_FROZEN_CALIBRATION`: one initial model fit and one initial calibration; both remain frozen.
- `B_FROZEN_MODEL_ROLLING_RECALIBRATION`: exactly the same initial model, but threshold/top-fraction are recalibrated monthly.
- `C_MONTHLY_REFIT_FROZEN_RECIPE`: exactly the same frozen recipe is freshly fit and calibrated every monthly assessment.
- Recipe identity is fixed across every arm and every assessment.
- Initial model artifact and initial threshold must match across A/B/C; the runner fails closed otherwise.
- Same replay inputs, portfolio policy, costs, benchmark and evaluation dates.
- No recipe switching, grid search, evaluation-window tuning, promotion or holdout access.

## Portfolio comparison

| Strategy | Terminal EUR | URTH EUR | Excess EUR | Relative | Trades | Costs EUR | Relative MaxDD |
|---|---:|---:|---:|---:|---:|---:|---:|
| A_FROZEN_MODEL_FROZEN_CALIBRATION | 13,461.75 | 12,919.58 | +542.17 | +4.20% | 35 | 417.94 | -28.79% |
| C_MONTHLY_REFIT_FROZEN_RECIPE | 10,634.57 | 12,919.58 | -2,285.01 | -17.69% | 94 | 974.40 | -31.04% |
| B_FROZEN_MODEL_ROLLING_RECALIBRATION | 10,384.74 | 12,919.58 | -2,534.84 | -19.62% | 108 | 833.99 | -59.34% |

## Causal contrasts

- B-A terminal delta: -3077.01 EUR; CAGR delta: -8.2064%; relative-MaxDD delta: -30.5518%.
- C-B terminal delta: +249.82 EUR; CAGR delta: +0.7254%; relative-MaxDD delta: +28.3016%.

Positive relative-MaxDD delta means the candidate has a less severe relative drawdown.

Exact provenance is persisted in `contract-audit.json`, `fit-audit.json`, `calibration-audit.csv`, each arm schedule/NAV/trades, and `causal-contrasts.json`.

This is Development/Research evidence only.
