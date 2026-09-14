# Dynamic-QBD Fold-Clock Refit Suite

Fold-clock vs frozen diagnosis: **FOLD_CLOCK_REFIT_PARETO_IMPROVES**
Monthly vs fold-clock diagnosis: **MONTHLY_REFIT_VS_FOLD_CLOCK_PARETO_WORSENS**

Authority: `SHADOW_ONLY_NO_PROMOTION_NO_HOLDOUT`. The final holdout remains closed.

## Scientific contract

- `A_FROZEN_MODEL_FROZEN_CALIBRATION` freezes its first causal model and calibration for the complete replay.
- `F_FOLD_CLOCK_REFIT_FROZEN_BETWEEN_FOLDS` uses the same frozen recipe and refits only when matured fold content expands; calibration is frozen between those events.
- `M_MONTHLY_REFIT_FROZEN_RECIPE` uses the same frozen recipe but freshly fits and calibrates every assessment.
- Fold evidence is fingerprinted from sorted fold IDs and fold count only; calendar progress alone is not evidence.
- No recipe switching, performance trigger, parameter search, promotion, capital authority or holdout access.

## Portfolio comparison

| Strategy | Terminal EUR | URTH EUR | Excess EUR | Relative | Trades | Costs EUR | Relative MaxDD |
|---|---:|---:|---:|---:|---:|---:|---:|
| F_FOLD_CLOCK_REFIT_FROZEN_BETWEEN_FOLDS | 15,773.36 | 12,919.58 | +2,853.78 | +22.09% | 77 | 871.70 | -24.26% |
| A_FROZEN_MODEL_FROZEN_CALIBRATION | 13,461.75 | 12,919.58 | +542.17 | +4.20% | 35 | 417.94 | -28.79% |
| M_MONTHLY_REFIT_FROZEN_RECIPE | 10,634.57 | 12,919.58 | -2,285.01 | -17.69% | 94 | 974.40 | -31.04% |

## Causal contrasts

- F-A terminal delta: +2311.61 EUR; CAGR delta: +5.3353%; relative-MaxDD delta: +4.5303%.
- M-F terminal delta: -5138.79 EUR; CAGR delta: -12.8164%; relative-MaxDD delta: -6.7805%.

The fold-clock event ledger and calibration audit explicitly show frozen months between matured-fold events.
Parallel fit execution: 32 active workers, queue capacity 64, 41 completed fit tasks; see `worker-telemetry.jsonl`.
This is Development/Research evidence only.
