# Phase 3 — Replacement QbD

Contract: `ENTRY_REPLACEMENT_QBD_V1`

## Objective

Measure whether replacing the weakest open position with a newly stronger signal improves Outer-OOS performance broadly across the already validated design region.

Phase 3 does not re-optimize dimensions that were decided earlier.

## Frozen inputs

- Phase-1 Prediction × Holding design space: the published 14-cell primary plateau from `artifacts/prediction-hold-qbd-surface/qbd_design_space.csv`.
- Phase-2 allocation: `EQUAL_ACTIVE`, verified from the published Phase-2 summary and treatment table.
- Sleeve: `0.50`.
- Exit family: `FIXED` with `exit_value=0`.
- Final Holdout: closed.
- Interpolation: not used.
- V4.5 exit overlay: not used.

## Replacement treatments

1. `IGNORE_NEW` — baseline. When the portfolio is full, a new signal does not displace an existing position.
2. `REPLACE_WEAKEST` — when the portfolio is full, the strongest eligible new signal may replace the open position with the weakest entry score if the new score is strictly higher.

The replacement behavior already exists in `portfolio.py`; Phase 3 is additive and does not alter replay semantics.

## Search surface

Inside every frozen H/D/replacement cell the established entry search remains active:

- score quantiles: 4
- top fractions: 3
- max names: 4
- total entry policies per cell: 48

Default Phase-3 surface:

- 14 frozen H/D cells
- 2 replacement treatments
- 28 measured cells
- 1,344 cell-level entry-policy candidates before walk-forward fold replay work

## Selection rule

`IGNORE_NEW` is the paired baseline. A non-baseline replacement treatment passes only if it:

- is complete across all 14 frozen H/D cells,
- is robust in at least 50% of the cells,
- beats `IGNORE_NEW` on at least 60% of H/D cells by median active CAGR excess,
- has positive median paired delta,
- has non-negative Q25 paired delta.

Ranking remains QbD-style: broad paired stability across the design space is preferred over an isolated maximum-CAGR cell. Turnover delta versus `IGNORE_NEW` is reported explicitly but is not a separate hard gate because transaction costs are already included in replay economics.

## Contract guards

The Phase-3 search context rejects any policy that escapes:

- frozen prediction horizon,
- frozen holding period,
- selected replacement treatment,
- `EQUAL_ACTIVE`,
- sleeve `0.50`,
- `FIXED` exit / `0.0`.

Checkpoint keys include the Phase-3 contract, H/D, replacement treatment and frozen allocation. Resume accepts only artifacts that preserve both Phase-1 and Phase-2 locks and keep the Final Holdout closed.

## Self-test coverage

`replacement_qbd_self_test.py` checks:

- replacement parser and allowed treatment set,
- committed 14-cell Phase-1 plateau provenance,
- committed Phase-2 result provenance and sole `EQUAL_ACTIVE` pass,
- exact 48-policy entry grid per replacement cell,
- policy/checkpoint identity across replacement and H/D changes,
- restoration of patched search functions,
- `IGNORE_NEW` preservation against the default replay behavior,
- synthetic early replacement behavior for `REPLACE_WEAKEST`,
- accounting/exposure/max-name invariants,
- resume rejection for wrong allocation, sleeve, treatment, Phase-2 lock or opened Holdout,
- complete/missing/failed/duplicate surface detection,
- paired evaluator behavior,
- broad-stability selection and rejection of an isolated spike winner.

## Run

```powershell
.\stock_predictor\backtests\opportunity_portfolio_research\run_replacement_qbd.ps1 `
  -V5Predictions <H1-H30-predictions.parquet> `
  -DailyStoreRoot artifacts\daily-parquet `
  -Phase1DesignSpace artifacts\prediction-hold-qbd-surface\qbd_design_space.csv `
  -Phase2Summary artifacts\allocation-qbd\allocation_qbd_summary.json `
  -Phase2TreatmentSummary artifacts\allocation-qbd\allocation_qbd_treatment_summary.csv `
  -OutputRoot artifacts\replacement-qbd
```

Self-tests only:

```powershell
python -m stock_predictor.backtests.opportunity_portfolio_research.replacement_qbd_self_test
python -m stock_predictor.backtests.opportunity_portfolio_research.replacement_qbd_runner --self-test
```

## Outputs after a measured run

- `REPLACEMENT_QBD_RESULTS_SUMMARY.md`
- `replacement_qbd_summary.json`
- `replacement_qbd_run_summary.json`
- `replacement_qbd_cell_status.csv`
- `replacement_qbd_outer_fold_results.csv`
- `replacement_qbd_final_policies.csv`
- `replacement_qbd_surface_cells.csv`
- `replacement_qbd_treatment_summary.csv`
- `replacement_qbd_paired_vs_ignore_new.csv`
- `replacement_qbd_design_space.csv`
- treatment heatmaps under `heatmaps/`
