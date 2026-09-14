# Dynamic-QBD Conditional Refit / Recipe Hysteresis

Status: **D1_MATCHES_FROZEN_MODEL_B**

Authority: `SHADOW_ONLY_NO_PROMOTION_NO_HOLDOUT`.
The final holdout remains closed.

## Scientific contract

- `B_FROZEN_MODEL_ROLLING_RECALIBRATION` fits the initial model once and recalibrates it monthly.
- `C_MONTHLY_REFIT_FROZEN_RECIPE` uses the same initial frozen recipe but freshly refits it every month; C-B isolates unconditional refitting.
- `D0_NEW_EVIDENCE_RECIPE_CHANGE_REFIT` refits only when a real new OOS-fold evidence expansion changes the exact causal recipe.
- `D1_HYSTERESIS_GATED_CONDITIONAL_REFIT` refits only after two independent OOS-fold evidence expansions support the same exact challenger with a strict robust-score edge.
- Calendar progress alone cannot increment hysteresis confirmation.
- All arms use the same monthly recalibration dates, portfolio policy, costs and benchmark.
- No grid search, evaluation-window tuning, promotion or holdout access.

## Portfolio comparison

| Strategy | Terminal EUR | URTH EUR | Excess EUR | Relative | Trades | Costs EUR | Relative MaxDD |
|---|---:|---:|---:|---:|---:|---:|---:|
| C_MONTHLY_REFIT_FROZEN_RECIPE | 10,634.57 | 12,919.58 | -2,285.01 | -17.69% | 94 | 974.40 | -31.04% |
| B_FROZEN_MODEL_ROLLING_RECALIBRATION | 10,384.74 | 12,919.58 | -2,534.84 | -19.62% | 108 | 833.99 | -59.34% |
| D1_HYSTERESIS_GATED_CONDITIONAL_REFIT | 10,384.74 | 12,919.58 | -2,534.84 | -19.62% | 108 | 833.99 | -59.34% |
| D0_NEW_EVIDENCE_RECIPE_CHANGE_REFIT | 8,707.13 | 12,919.58 | -4,212.45 | -32.61% | 94 | 622.69 | -63.64% |

## Interpretation

The decomposition is C-B (pure unconditional refit value), D0-B (immediate recipe-change conditional-refit value), and D1-B (hysteresis-gated conditional-refit value).

Observed monthly assessments: 41; distinct fold-content evidence states: 7.

Exact fit provenance is in `fit-audit.json`; monthly model age, thresholds and refit flags are in `calibration-audit.csv`; authority decisions are in `conditional-refit-decisions.csv` and `hysteresis-decisions.csv`.

This remains Development/Research evidence only.
