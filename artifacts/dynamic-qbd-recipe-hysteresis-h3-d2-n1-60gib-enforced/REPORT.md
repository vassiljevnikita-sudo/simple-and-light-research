# Dynamic-QBD causal recipe-switch hysteresis

Status: **HYSTERESIS_ROLLING_TRADEOFF**

Authority: `SHADOW_ONLY_NO_PROMOTION_NO_HOLDOUT`.
The final holdout remains closed.

## Fixed scientific contract

- Same monthly refit/recalibration dates in all three arms.
- Frozen arm keeps the first causal recipe but refits it every assessment.
- Rolling arm uses the existing causal selector winner every assessment.
- Hysteresis arm requires two consecutive distinct matured-evidence expansions supporting the same exact challenger.
- Challenger robust score must be strictly greater than the incumbent robust score; a selector tie-break alone cannot switch.
- Duplicate matured evidence cannot add a confirmation.
- Minimum two completed causal OOS folds for incumbent and challenger.
- No hysteresis parameter grid, no evaluation-period tuning, no final holdout.

## Portfolio comparison

| Strategy | Terminal EUR | URTH EUR | Excess EUR | Relative | Trades | Costs EUR | Relative MaxDD |
|---|---:|---:|---:|---:|---:|---:|---:|
| HYSTERESIS_CAUSAL_RECIPE_MONTHLY_REFIT | 11,586.73 | 12,919.58 | -1,332.85 | -10.32% | 104 | 1,068.74 | -39.39% |
| ROLLING_CAUSAL_RECIPE_MONTHLY_REFIT | 11,441.68 | 12,919.58 | -1,477.90 | -11.44% | 105 | 1,078.82 | -38.67% |
| FROZEN_FIRST_CAUSAL_RECIPE_MONTHLY_REFIT | 10,634.57 | 12,919.58 | -2,285.01 | -17.69% | 94 | 974.40 | -31.04% |

Recipe-change and model-family-switch counts are persisted in `portfolio-value-comparison.csv`; exact assessment decisions are in `hysteresis-decisions.csv`.

## Interpretation

The diagnosis uses strict Pareto logic only: hysteresis dominates rolling only if terminal value is no lower, relative MaxDD is no worse, and recipe changes do not increase. Otherwise the result is a trade-off or inferiority; no tolerance was tuned on this evaluation window.

This experiment is Development/Research evidence only and cannot promote a Family, recipe, router, or capital allocation.
