# Top-10 Entry Activation and Alpha Diagnostic

Status: **COMPLETE**

This audit reuses existing historical and causal-expanding predictions. It does not fit models, regenerate predictions, or optimize policies.
Realized future returns are attached only after prediction for evaluation and never feed thresholds or model inputs.

## Diagnosis by frozen activation contract

| contract_id | horizon | historical_crossing_day_rate | live_crossing_day_rate | historical_median_threshold_over_p99 | live_median_threshold_over_p99 | historical_top1_mean_realized_net_excess | live_top1_mean_realized_net_excess | historical_top_minus_bottom_decile_spread | live_top_minus_bottom_decile_spread | diagnosis | diagnosis_is_heuristic |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| H11_Q0_975_T0_005 | 11 | 3.1746% | 1.0811% | 6.104122430755202 | 35.93069044139949 | 1.2267% | 2.5490% | 0.004450513773811124 | 0.009288908508653805 | HEALTHY_BUT_RARE_SIGNAL | True |
| H24_Q0_95_T0_005 | 24 | 5.5556% | 0.0000% | 4.358875335641444 | 19.850878226031444 | 2.6587% | 4.5346% | 0.01815445153728349 | 0.015439366474220588 | CALIBRATION_TOO_RESTRICTIVE | True |
| H24_Q0_975_T0_005 | 24 | 3.1746% | 0.0000% | 4.638438058848543 | 25.854368942643134 | 2.6587% | 4.5346% | 0.01815445153728349 | 0.015439366474220588 | CALIBRATION_TOO_RESTRICTIVE | True |
| H28_Q0_975_T0_005 | 28 | 3.1746% | 0.0000% | 4.747945596504142 | 25.073417425905674 | 2.4159% | 5.0890% | 0.02130479932925681 | 0.020543562548189228 | CALIBRATION_TOO_RESTRICTIVE | True |

## Interpretation labels

- `SCORE_SCALE_DRIFT`: activation collapsed while relative ranking still works and current scores sit materially below calibration scale.
- `CALIBRATION_TOO_RESTRICTIVE`: activation collapsed but high-ranked names retain positive realized alpha.
- `RANKING_ALPHA_DECAY`: activation is not the main issue; realized ranking skill deteriorated.
- `BOTH_CALIBRATION_AND_ALPHA_DECAY`: activation collapsed and ranking alpha also failed.
- `HEALTHY_BUT_RARE_SIGNAL`: rare activation with positive ranking evidence.
- `INCONCLUSIVE`: evidence does not satisfy conservative heuristic gates.

Final holdout remained closed.
