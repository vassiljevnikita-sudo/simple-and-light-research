# Dynamic QBD model recipe and combination experiment

## Technical summary

The causal selector changed H3 recipes as additional completed Outer WF folds matured: Ridge (August 2020), HGB (August 2022), then Ridge (February 2023). Every activation used a fresh production fit and generation-specific calibration; historical model artifacts were not reused.

The primary matched-refit comparison ended at EUR 14,703.12 for rolling recipe reselection versus EUR 13,398.25 for the frozen Ridge recipe refitted on the same dates. The rolling arm added EUR 1,304.87, but relative maximum drawdown worsened from -24.27% to -36.11%. This is development-only shadow evidence, not promotion authority.

## Portfolio-value evidence

### Matched-date recipe-selection test

| Strategy | Terminal EUR | URTH EUR | Excess EUR | Relative | Trades | Costs EUR | Relative MaxDD |
|---|---:|---:|---:|---:|---:|---:|---:|
| ROLLING_CAUSAL_RECIPE_RESELECTION | 14,703.12 | 12,889.25 | +1,813.86 | +14.07% | 55 | 514.67 | -36.11% |
| FROZEN_FIRST_CAUSAL_RECIPE | 13,452.30 | 12,889.25 | +563.04 | +4.37% | 35 | 417.45 | -28.79% |
| FROZEN_RECIPE_MATCHED_REFIT_DATES | 13,398.25 | 12,889.25 | +508.99 | +3.95% | 46 | 446.39 | -24.27% |

The stale-model arm is diagnostic only because it confounds recipe choice with model age. The scientific recipe contrast is rolling reselection versus the frozen recipe refitted on identical dates.

### 2023-Q4 model-combination diagnostic

| Strategy | Terminal EUR | URTH EUR | Excess EUR | Relative | Trades | Costs EUR | Relative MaxDD |
|---|---:|---:|---:|---:|---:|---:|---:|
| SINGLE_POSITIVE_SPEARMAN_R2_FOLD | 11,582.26 | 11,129.42 | +452.83 | +4.07% | 11 | 113.04 | -5.57% |
| SELECTED_SINGLE_MODEL | 11,039.46 | 11,129.42 | -89.97 | -0.81% | 5 | 48.58 | -5.90% |
| RIDGE_HGB_INTERSECTION | 10,973.56 | 11,129.42 | -155.86 | -1.40% | 1 | 9.60 | -1.70% |

The positive-Spearman/positive-R² single-fold arm led this quarter, but one fold and one quarter are insufficient research evidence. The hard Ridge/HGB intersection generated only one trade and reduced terminal value, so hard consensus is not supported by this run.

## Scope and definitions

- Capital: EUR 10,000; benchmark and idle capital: URTH.
- Contract: H3 signal, D2 holding, N1, 50% entry sleeve, 20 bps round-trip costs, pre-tax.
- Recipe evidence: only completed prior Outer WF folds; R² is diagnostic and does not select the robust production recipe.
- Primary long-window comparison: 31 August 2020 through 29 December 2023.
- Q4 combination diagnostic: 2 October through 29 December 2023.

## Method and robustness

The matched-refit control holds refit dates, moving training windows, calibration, costs and portfolio rules constant. Only the recipe chosen at a switch date differs. Existing positions retain generation lineage, and all executions occur under the stateful next-open portfolio accounting contract.

No chart is included because the five portfolio rows span two different evaluation windows; combining them in one visual would imply a false common denominator. Exact tables and the saved daily NAV paths are the more honest evidence surface.

## Limitations

- H3/D2/N1 is one policy cell; results do not establish H1-H30 generality.
- Recipe changes are sparse, so statistical power is low and the 2022 HGB interval may be regime-specific.
- Rolling reselection improved terminal wealth but materially worsened drawdown; it is not a free dominance result.
- The Q4 single-fold winner is explicitly diagnostic and must not influence the production picker.
- Final holdout remained closed; no capital or promotion authority is granted.

## Recommended next experiments

1. **N2_N3_DIVERSIFICATION** — Reduce idiosyncratic drawdown while preserving the 50% total sleeve. Guard: Same entries, total sleeve and costs; predeclare N2/N3 and sector cap.
2. **CAUSAL_RECIPE_SWITCH_HYSTERESIS** — Avoid weak recipe changes and reduce transition risk. Guard: Switch only on matured multi-fold improvement; no evaluation-period tuning.
3. **SOFT_RIDGE_HGB_AGREEMENT** — Retain more alpha than the one-trade hard intersection while filtering disagreement. Guard: Calibrate percentile blend and minimum agreement solely on prior OOS folds.
4. **DRAWDOWN_AWARE_SLEEVE_THROTTLE** — Cut left-tail exposure without permanently lowering opportunity participation. Guard: Causal trailing relative-wealth state; fixed de-risk/re-risk ladder; compare terminal value non-inferiority.
5. **LEARNED_EXIT_DOWNSIDE_OVERLAY** — Exit failed opportunities earlier while retaining winners. Guard: Generation-specific E-models, next-open execution, coverage gate and fixed-H fallback.
6. **RECIPE_PLATEAU_CLUSTERING** — Prefer stable recipe plateaus over a single noisy candidate winner. Guard: Cluster correlated candidates; time/fold remains inference unit; R2 diagnostic only.

The first test should be N2/N3 diversification because N1 concentration is the cleanest likely drawdown source and can be varied without changing the signal or total stock sleeve. Recipe hysteresis and soft model agreement follow; drawdown throttles and learned exits should be tested only with explicit terminal-value non-inferiority gates.

## Further questions

- Does rolling recipe reselection remain positive under N2/N3 and across H1-H30 plateaus?
- Which trades produced the additional HGB drawdown, and is the loss concentrated by ticker, sector or market regime?
- Can a soft agreement score preserve the selected-model trade breadth while improving left-tail outcomes?
- What hysteresis margin remains stable under block-bootstrap resampling of time folds?

Status: `SHADOW_ONLY_NO_PROMOTION_NO_HOLDOUT`.
