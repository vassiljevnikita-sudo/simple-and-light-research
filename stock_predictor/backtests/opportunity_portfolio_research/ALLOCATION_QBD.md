# Phase 2 — Entry Allocation QbD

Contract: `ENTRY_ALLOCATION_QBD_V1`

This suite is the second optimization phase after the completed Prediction-Horizon × Holding-Days QbD surface.

## Frozen inputs

Phase 2 does **not** reopen Prediction Horizon or Holding Days. By default it reads `artifacts/prediction-hold-qbd-surface/qbd_design_space.csv` and selects the largest connected Phase-1 plateau. For the published Phase-1 result at commit `5fdd94a63b493cbd6d8559397d15d026650f0106`, that plateau contains 14 measured H/D cells. The loader selects those actual rows; it does not construct the rectangular cross-product H23–H26 × D3–D8.

Also frozen: sleeve=`0.50`, replacement=`IGNORE_NEW`, exit family=`FIXED`, exit value=`0`, no V4.5 dynamic-exit overlay, final holdout closed. The established 48-policy entry search remains active inside every H/D/allocation cell (4 quantiles × 3 top fractions × 4 max_names).

The Phase-2 search contract rejects attempts to change sleeve inside this phase. Chosen policies and outer-fold rows are validated against frozen H/D, allocation treatment, replacement, sleeve, and exit family before a cell can be marked complete. Resume checkpoints are accepted only when the same invariants and closed-holdout flags are present.

## Allocation treatments

Default treatments: `EQUAL_ACTIVE`, `RANK_POWER:1.0`, `RANK_POWER:1.5`, `RANK_POWER:2.0`, `SCORE_EXCESS_POWER:1.0`, `SCORE_EXCESS_POWER:1.5`, `SCORE_EXCESS_POWER:2.0`, `SCORE_SOFTMAX:0.5`, `SCORE_SOFTMAX:1.0`, `SCORE_SOFTMAX:2.0`.

With 14 Phase-1 H/D cells this is a **140-cell Phase-2 surface**. Every cell runs the full 48-policy entry grid. Weighting applies only to simultaneous new entries; existing positions are not continuously rebalanced, so rebalancing/turnover remains a separate later decision. All weighting inputs exist at decision time; no future volatility or returns are used.

## Baseline compatibility

`EQUAL_ACTIVE` retains the original replay branch exactly. New weighting is called only for explicit Phase-2 treatment prefixes. Unknown legacy allocation strings retain the prior equal-slot fallback. `Policy.policy_id` hashes `allocation`, and Phase-2 checkpoint keys additionally include H, D, and treatment.

The deterministic replay integration test explicitly replaces the Phase-2 weighting helper with a failing stub and proves that `EQUAL_ACTIVE` still completes without calling it. Separate synthetic replay checks prove that rank/score weighting changes simultaneous-entry notionals while preserving total sleeve capacity, accounting identity, max-name limits, and total exposure.

## Selection rule

Primary comparison is paired against `EQUAL_ACTIVE` at the same H/D cell. A non-baseline treatment passes only when it is complete on all frozen H/D cells, at least 50% of cells pass the established robustness gate, it beats `EQUAL_ACTIVE` on at least 60% of H/D cells, and the median paired `median_active_cagr_excess` delta is positive. Ranking favors Q25 paired delta, median paired delta, robust-cell fraction, then worst paired delta. An isolated maximum-CAGR spike therefore cannot win Phase 2.

## Testsuite

`allocation_qbd_self_test.py` is a data-free deterministic test suite and covers:

- all default allocation families: parsing, normalization, non-negativity, monotonicity, one-name and tied-score behavior;
- rejection of malformed/unsupported treatment strings and missing score-excess thresholds;
- exact loading of the committed 14-cell primary Phase-1 plateau;
- exactly 48 unique entry policies per Phase-2 cell;
- frozen H/D, allocation, replacement, sleeve and fixed-exit invariants;
- checkpoint-key isolation across both allocation treatment and holding period;
- explicit rejection of sleeve changes inside Phase 2;
- synthetic end-to-end replay for `EQUAL_ACTIVE`, rank weighting and softmax weighting;
- proof that the baseline replay path does not call the new weighting helper;
- stale/mismatched checkpoint rejection, including changed sleeve or opened final holdout;
- complete-surface, failed-cell and duplicate-cell detection;
- paired broad-stability selection, including rejection of an isolated high-CAGR spike.

The GitHub workflow `.github/workflows/allocation-qbd-self-test.yml` additionally recompiles the research package, exercises the explicit runner, reruns the complete Phase-1 Prediction×Hold QbD self-test, reruns base replay/accounting invariants, reruns prepared-replay equivalence/search coverage, and proves the final holdout command remains locked.

## Outputs

Default runtime root: `artifacts/allocation-qbd`.

Outputs include per-cell JSON resume checkpoints, `allocation_qbd_cell_status.csv`, `allocation_qbd_outer_fold_results.csv`, `allocation_qbd_final_policies.csv`, `allocation_qbd_surface_cells.csv`, `allocation_qbd_paired_vs_equal.csv`, `allocation_qbd_treatment_summary.csv`, `allocation_qbd_design_space.csv`, `allocation_qbd_summary.json`, `allocation_qbd_run_summary.json`, `ALLOCATION_QBD_RESULTS_SUMMARY.md`, and per-treatment heatmap CSVs under `heatmaps/`.

## Run

```powershell
.\stock_predictor\backtests\opportunity_portfolio_research\run_allocation_qbd.ps1 `
  -V5Predictions <H1-H30-predictions.parquet> `
  -DailyStoreRoot artifacts\daily-parquet `
  -Phase1DesignSpace artifacts\prediction-hold-qbd-surface\qbd_design_space.csv `
  -OutputRoot artifacts\allocation-qbd
```

Self-test only:

```powershell
python -m stock_predictor.backtests.opportunity_portfolio_research.allocation_qbd_self_test
python -m stock_predictor.backtests.opportunity_portfolio_research.allocation_qbd_runner --self-test
```

Phase 2 never opens the final holdout.
