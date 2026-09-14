# Phase 4 — Concentration × Replacement QbD

## Purpose

Phase 3 showed that `REPLACE_WEAKEST` is active but unstable while the selected Outer-OOS policies were usually concentrated at `max_names=1`. Phase 4 separates those two degrees of freedom so that concentration is no longer hidden inside the entry-policy optimizer.

The question is not “which single cell has the highest CAGR?” It is:

1. which `max_names` values remain broadly robust across the frozen Phase-1 H/D design space, and
2. whether `REPLACE_WEAKEST` improves the corresponding `IGNORE_NEW` policy at the **same** concentration.

## Frozen dimensions

- Phase-1 prediction-horizon / holding-day design space: the committed 14-cell plateau.
- Phase-2 allocation: `EQUAL_ACTIVE`.
- Sleeve: `0.50`.
- Exit family/value: `FIXED / 0.0`.
- Final holdout: closed.
- Interpolation: disabled.
- V4.5 exit overlay: disabled.

Phase-3 replacement is **not** frozen. Phase 3 is provenance only; Phase 4 deliberately reopens `IGNORE_NEW` vs `REPLACE_WEAKEST` while conditioning on concentration.

## Phase-4 factors

`max_names` is an explicit cell factor:

```text
1, 2, 3, 4, 5
```

Replacement is an explicit cell factor:

```text
IGNORE_NEW
REPLACE_WEAKEST
```

Default surface:

```text
14 H/D cells × 5 max_names × 2 replacement treatments = 140 cells
```

## Inner entry search

Inside every Phase-4 cell, `max_names` is fixed and **cannot** move. Only:

```text
score_quantile × top_fraction
```

is searched.

Current grid size:

```text
4 score quantiles × 3 top fractions = 12 entry policies per cell
```

Therefore the default measured surface contains 1,680 cell-level entry-policy candidates before fold-level evaluation.

`max_names=4` is intentionally included even though the legacy `search.SEARCH_MAX_NAMES` tuple is `(1,2,3,5)`; the Phase-4 contract must prove that `4` is a first-class fixed concentration cell rather than silently skipping it.

## Pairing and selection

Every `REPLACE_WEAKEST` cell is paired against `IGNORE_NEW` with identical:

- prediction horizon,
- holding days,
- `max_names`,
- allocation,
- sleeve,
- exit contract.

The evaluator reports broad H/D stability, median/Q25/worst active CAGR excess, trade count and turnover. Replacement must beat `IGNORE_NEW` broadly rather than only in an isolated peak:

- complete across all Phase-1 H/D design cells,
- robust-cell fraction ≥ 50%,
- positive-cell fraction ≥ 50%,
- beats `IGNORE_NEW` in ≥ 60% of H/D cells,
- median delta vs `IGNORE_NEW` > 0,
- Q25 delta vs `IGNORE_NEW` ≥ 0.

`IGNORE_NEW` at each concentration is retained as the direct reference and only needs the broad-stability gate.

The evaluator additionally reports deltas against `max_names=1` under the same replacement rule. This makes concentration effects visible without converting `max_names=1` into a mandatory winner.

## Cost and tax boundary

Phase 4 does **not** reopen roundtrip cost or tax assumptions. That is deliberate: concentration/replacement must first be identified without adding another high-dimensional factor.

Phase-4 artifacts retain trade count and turnover so that the surviving concentration/replacement region can be stress-tested later against:

- roundtrip bps,
- break-even cost,
- realized gains/losses,
- tax paid / tax drag,
- realization timing / deferral.

Those are marked `cost_and_tax_stress_deferred=true` in the Phase-4 contract.

## Self-test contract

The deterministic suite verifies:

- the exact committed 14-cell Phase-1 plateau,
- the committed Phase-2 `EQUAL_ACTIVE` lock,
- committed Phase-3 provenance with Final Holdout still closed,
- exactly 12 inner entry policies per Phase-4 cell,
- `max_names` fixed to exactly one of `1..5` inside each cell,
- explicit support for `max_names=4`,
- checkpoint separation across H/D, `max_names`, and replacement,
- policy-ID separation across concentration and replacement,
- `EQUAL_ACTIVE`, sleeve `0.50`, and FIXED exits cannot escape,
- `REPLACE_WEAKEST` changes real synthetic replay behavior,
- max-position, sleeve, exposure, and accounting invariants,
- stale/incompatible cell artifacts are rejected,
- missing/failed/duplicate/unexpected surface cells are detected,
- an isolated replacement CAGR spike fails the broad-stability gate,
- Final Holdout remains locked.

## Commands

Self-test:

```powershell
python -m stock_predictor.backtests.opportunity_portfolio_research.concentration_replacement_qbd_self_test
python -m stock_predictor.backtests.opportunity_portfolio_research.concentration_replacement_qbd_runner --self-test
```

Full default Phase 4:

```powershell
.\stock_predictor\backtests\opportunity_portfolio_research\run_concentration_replacement_qbd.ps1 `
  -V5Predictions <H1-H30-predictions.parquet> `
  -DailyStoreRoot artifacts\daily-parquet `
  -Phase1DesignSpace artifacts\prediction-hold-qbd-surface\qbd_design_space.csv `
  -Phase2Summary artifacts\allocation-qbd\allocation_qbd_summary.json `
  -Phase2TreatmentSummary artifacts\allocation-qbd\allocation_qbd_treatment_summary.csv `
  -Phase3Summary artifacts\replacement-qbd\replacement_qbd_summary.json `
  -Phase3TreatmentSummary artifacts\replacement-qbd\replacement_qbd_treatment_summary.csv `
  -OutputRoot artifacts\concentration-replacement-qbd
```
