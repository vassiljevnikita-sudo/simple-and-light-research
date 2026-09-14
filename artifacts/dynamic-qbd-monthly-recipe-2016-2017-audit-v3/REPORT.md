# Dynamic QBD 2016-2017 causal preflight

## Decision

**NOT_RUN_FAIL_CLOSED**. A portfolio result cannot be produced for the requested interval without inventing pre-2016 training history or importing future recipe evidence.

## Evidence

- Requested: 2016-01-01 through 2017-12-31
- Signal panel: 2016-06-24 through 2025-12-31
- Signal sessions inside request: 383
- Required matured sessions before each fit: 786 (504 train + 30 purge + 252 calibration)
- First history-eligible month end: 2019-08-30
- First recipe-evidence-eligible month end: 2020-08-31
- First jointly eligible month end: 2020-08-31

## Validity guard

No future-selected recipe, future-trained model, shortened hidden window, fabricated prehistory or stale model cache was used. No QBD return, drawdown or promotion statistic was emitted.

## URTH market context only

These figures use the explicit Alpaca SIP 1Day benchmark and are not QBD strategy results.

- Total return: 28.44%
- Annualized return: 13.44%
- Maximum drawdown: -10.76% (2016-01-04 to 2016-02-11)
- Daily expected shortfall (95%): -2.48%
