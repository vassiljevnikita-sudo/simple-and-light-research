# URTH Core → Stock Opportunity → URTH

Decision: `ROBUST_CANDIDATE_FOR_FILL_MARK_REPLAY`

- Default: 100% URTH / MSCI-World proxy.
- V2 is excluded as a signal source.
- V5 selected walk-forward signal is primary; forward labels are not compounded.
- Policy selection uses only earlier OOS folds; rare policies require <=25% activation.
- No sleeve-size optimization is performed in this stage.

## Diagnostic single-/low-fold results

These are not eligible for a research conclusion.

| Signal | Horizon | Outer folds | Status | Median excess/date | Median activation |
|---|---:|---:|---|---:|---:|
| HIST_GRADIENT_BOOSTING | 20 | 0 | DIAGNOSTIC_INSUFFICIENT_OUTER_FOLDS | 0.000% | 0.0% |
| RIDGE | 10 | 0 | DIAGNOSTIC_INSUFFICIENT_OUTER_FOLDS | 0.000% | 0.0% |
| HIST_GRADIENT_BOOSTING | 10 | 4 | DIAGNOSTIC_INSUFFICIENT_OUTER_FOLDS | 0.116% | 16.3% |
| RIDGE | 20 | 4 | DIAGNOSTIC_INSUFFICIENT_OUTER_FOLDS | 0.000% | 0.0% |

## Eligible multi-fold robustness

| Signal | Horizon | Folds | Positive | Median excess/date | Q25 | Worst | Median active excess | Median rate | Median hit rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| RIDGE | 5 | 6 | 4 | 0.109% | 0.003% | -0.065% | 0.853% | 6.0% | 50.0% |

## Decision

`ROBUST_CANDIDATE_FOR_FILL_MARK_REPLAY`

A robust candidate is suitable only for a later chronological fill/mark replay; it is not production approval.
