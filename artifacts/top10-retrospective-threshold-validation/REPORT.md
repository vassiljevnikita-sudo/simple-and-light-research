# Top-10 Retrospective Threshold Validation

Status: **COMPLETE**

This is a backward-looking, maturity-aware diagnostic. At each historical assessment date, only outcomes whose terminal date had already occurred are visible.
It does not fit models, regenerate predictions, optimize thresholds, or choose a replacement threshold from the evaluation period.

Primary window: last 126 fully evaluable decision sessions.
63 and 252 sessions are sensitivity views only and must not be used to pick the best-looking result.

## Latest primary assessment

| contract_id | horizon | matured_decision_days | historical_crossing_day_rate | recent_crossing_day_rate | shadow_rejected_candidate_count | shadow_mean_realized_net_excess | shadow_hit_rate | median_threshold_over_p99 | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| H11_Q0_975_T0_005 | 11 | 126 | 3.1746% | 0.0000% | 378.0 | 6.9390% | 62.1693% | 14.633620622992005 | RETROSPECTIVE_TOO_RESTRICTIVE_EVIDENCE |
| H24_Q0_95_T0_005 | 24 | 126 | 5.5556% | 0.0000% | 378.0 | 10.1232% | 58.4656% | 13.19697452169947 | RETROSPECTIVE_TOO_RESTRICTIVE_EVIDENCE |
| H24_Q0_975_T0_005 | 24 | 126 | 3.1746% | 0.0000% | 378.0 | 10.1232% | 58.4656% | 17.657235038918863 | RETROSPECTIVE_TOO_RESTRICTIVE_EVIDENCE |
| H28_Q0_975_T0_005 | 28 | 126 | 3.1746% | 0.0000% | 378.0 | 9.1143% | 58.7302% | 18.1760359750251 | RETROSPECTIVE_TOO_RESTRICTIVE_EVIDENCE |

`RETROSPECTIVE_TOO_RESTRICTIVE_EVIDENCE` means the old activation rate collapsed while top-fraction names rejected by the threshold had positive subsequently realized net excess in the fully matured backward window.
It is evidence for a calibration review, not authorization to change the live threshold.

Final holdout remained closed.
