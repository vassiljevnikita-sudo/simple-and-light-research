# Dynamic-QBD Paired Selector Model-Pool Tournament

## Scope and fixed design

This is a research-only, cross-fitted tournament on the compact Opportunity-State evidence. The primary unit is `decision_date × ticker × arm` with at least two deduplicated active buckets and exactly one candidate per bucket. The fixed outer walk-forward uses the prior monthly fold structure, with training targets satisfying `target_matured_date < test_start`. Every independent pool sees the identical candidate sets.

No Factory model was retrained, no prediction was regenerated, the final holdout boundary `2026-07-25` remained closed, and no capital authority was granted. No pseudo-CAGR, Sharpe, drawdown or terminal-wealth metric is used.

P9 is not an additional fitted learner: P1 and P2 already use exactly the complete True-Consensus feature contract (score baseline, shape, presence, agreement and bucket-specific means). P9 is retained as a scientific feature-class label in the manifest and deliberately excluded from independent winner selection to avoid duplicate evidence.

Pairwise pools use canonical bucket pairs and expected wins. Pairwise ties are excluded from training. P5/P6/P7 use fixed 50/50 component weights. Horizon normalization is the deterministic per-event `forward_excess_return / horizon_h`, aggregated once per bucket; it is not annualized.

## Primary pooled results

| Pool | Selected Excess | Δ vs Score | Regret | Pairwise Accuracy | Same-set IC | Override Rate | Override Value | Calendar q05 | Ticker q05 | Episode q05 | Target |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| P1 | 0.021104 | -0.142075 | 0.266779 | 0.3871 | -0.2419 | 0.476 | -0.298597 | -0.153450 | -0.079514 | -0.113766 | RAW_EXCESS |
| P2 | 0.116890 | -0.046289 | 0.170993 | 0.4946 | 0.1849 | 0.298 | -0.155131 | -0.044457 | -0.020281 | -0.018937 | RAW_EXCESS |
| P3 | 0.127365 | -0.035814 | 0.160518 | 0.6075 | 0.1815 | 0.306 | -0.116867 | -0.079376 | -0.036268 | -0.034626 | RAW_EXCESS |
| P4 | 0.094133 | -0.069046 | 0.193750 | 0.4194 | -0.1815 | 0.589 | -0.117284 | -0.146645 | -0.069153 | -0.114572 | RAW_EXCESS |
| P5 | 0.025478 | -0.137700 | 0.262404 | 0.3414 | -0.0924 | 0.516 | -0.266794 | -0.152354 | -0.079322 | -0.116740 | RAW_EXCESS |
| P6 | 0.025478 | -0.137700 | 0.262404 | 0.3414 | -0.0924 | 0.516 | -0.266794 | -0.150737 | -0.079788 | -0.111901 | RAW_EXCESS |
| P7 | 0.122654 | -0.040524 | 0.165229 | 0.5914 | 0.1532 | 0.371 | -0.109240 | -0.101807 | -0.039142 | -0.037784 | RAW_EXCESS |
| P8 | 0.018913 | -0.144265 | 0.268969 | 0.3844 | -0.2500 | 0.476 | -0.303201 | -0.151431 | -0.081650 | -0.107115 | RAW_EXCESS |

The full normalized-target table is in `pool_metrics.csv`; all comparisons are paired in `pool_vs_score.csv` and `pool_pairwise_matrix.csv`.

## Scientific review

- Primary diagnosis: **G_SCORE_ONLY_REMAINS_BEST_ROBUST_SELECTOR**.
- Highest selected excess (descriptive only): `P3`.
- Lowest regret (descriptive only): `P3`.
- Robust-gate pools: `[]`.
- Pairwise versus regression: `PAIRWISE_HIGHER_MEAN`.
- Ridge versus HGB: `HGB_HIGHER_MEAN`.
- Ensemble effect: `NO_MEAN_ENSEMBLE_GAIN`.
- Score-shape versus full consensus: `FULL_CONSENSUS_HIGHER_MEAN`.
- Two- versus three-bucket and early/late evidence is in `set_type_robustness.csv`; `SHORT_LONG` remains descriptive because its fixed count is four.
- Concentration is in `ticker_contribution.csv` and `episode_contribution.csv`; top-share summaries are in `summary.json`.
- Raw versus normalized target comparison is in `target_variant_comparison.csv`.

## Contract and recommendation

All contract checks in `diagnostic_contract_checks.json` must be true. The winner is selected by paired-vs-score, ticker/episode bootstrap gates, regret, override value, target robustness, subperiod stability and then complexity—not by raw mean excess alone. The final holdout remains closed and there is no Capital Authority.

Recommendation: **retain Score-only and stop selector promotion research until more independent evidence exists**.

Runtime: `138.07` seconds; measured peak RSS: `0.22040176391601562` GB.
