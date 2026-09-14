# Dynamic QBD monthly recipe factory

## Decision

**SUPERSEDED_PROVENANCE_MANIFEST_REQUIRED**. These portfolio figures are diagnostic only and must not be treated as final research evidence until the monthly fits have been rebuilt or revalidated under the generation-manifest contract. Rolling recipe selection improved terminal wealth versus the matched frozen-recipe monthly-refit control by EUR 805.30 (7.19%), but both arms underperformed URTH and the recipe contrast contains only one genuinely different recipe episode. This is not promotion evidence.

## Technical summary

Completed 40 causal monthly refits from 2020-08-31 through 2023-12-29. R0/R1/R2/R3 share the same monthly fit dates, moving windows, monthly calibration, costs and stateful next-open replay.

URTH execution uses split-adjusted Alpaca SIP daily Open/Close. Official iShares cash distributions are credited to entitled units on the payable session and reinvested at that session's Open for both the benchmark and idle sleeve.

The experiment contains 80 legacy monthly production artifacts (Ridge and HGB) and 40 separately calibrated blend generations. This reporting invocation reused 80 artifacts under the former presence-and-input-contract cache; no per-fit generation manifests existed. The current code now requires manifest-verified artifact hashes, Git SHA and training-source hash before reuse, so these artifacts require a fresh manifest-backed build before any final claim.

The row-level direct-daily fallback repaired 157 invalid stock Opens and 3312 invalid stock Closes. Valid minute-derived boundaries were retained.

| arm | terminal_value | urth_terminal_value | terminal_excess_eur | terminal_relative_return | trade_count | total_cost_eur | gross_distribution_income_eur | benchmark_gross_distribution_income_eur | relative_max_drawdown | cdar_95 | expected_shortfall_95 | time_under_water_fraction | drawdown_duration_sessions |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R3_RIDGE_HGB_50_50_PERCENTILE_BLEND | 12297.694895 | 13680.860842 | -1383.165947 | -0.101102 | 143 | 1478.886583 | 459.397818 | 688.795860 | -0.496989 | -0.445985 | -0.045649 | 0.970203 | 695 |
| R1_ROLLING_RECIPE_MONTHLY_REFIT | 12013.341890 | 13680.860842 | -1667.518952 | -0.121887 | 105 | 1107.026480 | 521.980866 | 688.795860 | -0.386690 | -0.269270 | -0.041535 | 0.985697 | 356 |
| R0_FROZEN_RECIPE_MONTHLY_REFIT | 11208.040141 | 13680.860842 | -2472.820700 | -0.180750 | 94 | 1000.195913 | 551.524558 | 688.795860 | -0.310390 | -0.290702 | -0.036520 | 0.985697 | 356 |
| R2_HYSTERESIS_RECIPE_MONTHLY_REFIT | 11208.040141 | 13680.860842 | -2472.820700 | -0.180750 | 94 | 1000.195913 | 551.524558 | 688.795860 | -0.310390 | -0.290702 | -0.036520 | 0.985697 | 356 |

## Recipe-switch evidence

| assessment_date | incumbent_family_before | winner_family | delta_challenger_incumbent | probability_challenger_better | delta_ci_05 | delta_ci_95 | next_unseen_fold_id | next_unseen_challenger_minus_incumbent_spearman | next_realized_r1_minus_r0_relative_wealth |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2022-08-31 | RIDGE | HIST_GRADIENT_BOOSTING | 0.003968 | 0.315400 | -0.128062 | 0.046890 | WF_006_2022-08-10_2023-02-08 | -0.013734 | 0.057262 |

The bootstrap hysteresis rejected the only Ridge-to-HGB switch. The ungated rolling arm nevertheless beat its matched control during the subsequent unseen portfolio period; this single event is insufficient for threshold tuning or generalization claims.

## Max-names capacity ablation

| max_names | arm | terminal_value | urth_terminal_value | terminal_excess_eur | terminal_relative_return | trade_count | total_cost_eur | gross_distribution_income_eur | benchmark_gross_distribution_income_eur | relative_max_drawdown | cdar_95 | expected_shortfall_95 | time_under_water_fraction | drawdown_duration_sessions |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | N1 | 12013.341890 | 13680.860842 | -1667.518952 | -0.121887 | 105 | 1107.026480 | 521.980866 | 688.795860 | -0.386690 | -0.269270 | -0.041535 | 0.985697 | 356 |
| 2 | N2 | 16715.319997 | 13680.860842 | 3034.459155 | 0.221803 | 132 | 1132.036824 | 637.165639 | 688.795860 | -0.332700 | -0.242804 | -0.038790 | 0.992849 | 229 |
| 3 | N3 | 17121.065588 | 13680.860842 | 3440.204746 | 0.251461 | 144 | 1154.749276 | 648.995999 | 688.795860 | -0.324685 | -0.231205 | -0.037757 | 0.991657 | 229 |
| 6 | N6 | 16240.538331 | 13680.860842 | 2559.677489 | 0.187099 | 154 | 1103.658367 | 629.617302 | 688.795860 | -0.357408 | -0.260492 | -0.037838 | 0.992849 | 229 |

Only `max_names` changed. N3 had the highest terminal value (EUR 17121.07); no sector cap or changed score gate was introduced.

## Candidate execution-boundary audit

| arm | eligible_candidate_rows | invalid_next_open_candidate_rows | invalid_next_open_candidate_fraction | unique_invalid_tickers | contract |
| --- | --- | --- | --- | --- | --- |
| R0_FROZEN_RECIPE_MONTHLY_REFIT | 176 | 0 | 0.000000 | 0 | CANDIDATE_DIAGNOSTIC_ONLY_ACTUAL_EXECUTIONS_REMAIN_FAIL_CLOSED |
| R1_ROLLING_RECIPE_MONTHLY_REFIT | 198 | 0 | 0.000000 | 0 | CANDIDATE_DIAGNOSTIC_ONLY_ACTUAL_EXECUTIONS_REMAIN_FAIL_CLOSED |
| R2_HYSTERESIS_RECIPE_MONTHLY_REFIT | 176 | 0 | 0.000000 | 0 | CANDIDATE_DIAGNOSTIC_ONLY_ACTUAL_EXECUTIONS_REMAIN_FAIL_CLOSED |
| R3_RIDGE_HGB_50_50_PERCENTILE_BLEND | 248 | 0 | 0.000000 | 0 | CANDIDATE_DIAGNOSTIC_ONLY_ACTUAL_EXECUTIONS_REMAIN_FAIL_CLOSED |

This is a pre-replay intersection of causal gate candidates and next-open data quality. Actual executed and held prices remain fail-closed.

## Maximum-drawdown cause classification

| arm | cause_classification | top_negative_ticker | top_ticker_negative_share | top_negative_generation | top_generation_negative_share | episode_tickers | episode_sectors | episode_generations | negative_active_excess_contribution_eur | net_active_excess_contribution_eur |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R0_FROZEN_RECIPE_MONTHLY_REFIT | A_SINGLE_TICKER_OUTLIER | CVNA | 0.718426 | 817be492db020a442459d595 | 0.258341 | 17 | 6 | 14 | 15157.331928 | -3202.044682 |
| R1_ROLLING_RECIPE_MONTHLY_REFIT | A_SINGLE_TICKER_OUTLIER | CVNA | 0.863644 | 0910c3634152f18beaaea52f | 0.443741 | 5 | 3 | 6 | 11115.257344 | -4290.021912 |
| R2_HYSTERESIS_RECIPE_MONTHLY_REFIT | A_SINGLE_TICKER_OUTLIER | CVNA | 0.718426 | 817be492db020a442459d595 | 0.258341 | 17 | 6 | 14 | 15157.331928 | -3202.044682 |
| R3_RIDGE_HGB_50_50_PERCENTILE_BLEND | A_SINGLE_TICKER_OUTLIER | CVNA | 0.601918 | 4199bb93f45037498461a289 | 0.114709 | 15 | 6 | 20 | 22278.365364 | -5674.739422 |

Daily and recipe-segment compounding reconstruct terminal relative wealth with an absolute error below 1e-10. Switch-boundary returns are fully assigned.

The earlier single-fold/R-squared diagnostic is excluded as a challenger. Learned exits and drawdown throttles remain deferred; neither was optimized on these outcomes.

## Research authority

This is Development shadow evidence only. Recipe decisions use completed prior Outer-WF folds; portfolio outcomes and drawdown attribution never feed back into the same decision. Final holdout remained closed.
