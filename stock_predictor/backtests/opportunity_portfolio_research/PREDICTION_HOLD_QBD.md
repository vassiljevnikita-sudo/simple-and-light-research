# Prediction Horizon x Holding Days QbD Surface

Contract: `PREDICTION_HOLD_QBD_SURFACE_V1`

This Development runner measures Prediction Horizon and Holding Days independently. The default triangular H1-H30 design contains 465 measured cells (`1 <= holding_days <= prediction_horizon`). It does not interpolate missing cells and does not open the final holdout.

For each `(H, D)` cell, H and D are fixed. The runner optimizes only the established entry dimensions: four score quantiles (`0.90, 0.95, 0.975, 0.99`), three top fractions (`0.005, 0.01, 0.025`), and four max-name values (`1, 2, 3, 5`). Therefore every historical selection window evaluates the complete 48-policy entry grid. Sleeve remains 0.50, allocation remains `EQUAL_ACTIVE`, and replacement remains `IGNORE_NEW`.

Dynamic V4.5 exit families are disabled. The only exit dimension under study is the fixed holding period D. Score thresholds continue to be calibrated from historical daily top scores only.

The QbD response is built from chronological outer-OOS folds. The primary response is `median_active_cagr_excess`. The evaluator also records q25 active excess, positive active-fold fraction, trade count, turnover and drawdown. A base robust cell requires at least three active folds, at least 60% positive active folds, positive median active excess, and at least eight trades.

Local stability is evaluated from actually measured neighboring H/D cells. Connected locally stable cells form plateaus; by default a plateau needs at least three cells to count as a design space. Plateau ranking emphasizes worst-cell and lower-quartile behavior before central performance rather than simply selecting maximum CAGR.

Run the full surface with:

```powershell
.\stock_predictor\backtests\opportunity_portfolio_research\run_prediction_hold_qbd.ps1 `
  -V5Predictions <H1-H30-predictions.parquet> `
  -DailyStoreRoot artifacts\daily-parquet `
  -OutputRoot artifacts\prediction-hold-qbd-surface
```

Main outputs are `qbd_cell_status.csv`, `qbd_outer_fold_results.csv`, `qbd_final_policies.csv`, `qbd_surface_cells.csv`, `qbd_plateaus.csv`, `qbd_design_space.csv`, the `heatmap_*.csv` matrices, `qbd_summary.json`, `qbd_run_summary.json`, resumable `cells/Hxx_Dxx.json` artifacts, and the persistent `qbd_fragment_cache.sqlite3`.

Cache/checkpoint identities include the QbD contract plus prediction horizon and holding days. The V4.5 overlay and old H1-H30 two-arm suite are not invoked. Material changes to QbD cell semantics should bump the contract version before cached results are reused.
