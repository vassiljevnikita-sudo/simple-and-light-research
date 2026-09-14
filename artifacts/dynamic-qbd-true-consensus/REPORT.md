# Dynamic-QBD True Cross-Horizon Consensus

## Scope and contract

This is a research-only identification test built from the existing compact Opportunity-State evidence. No Factory model was retrained, no prediction was regenerated, and the final holdout boundary `2026-07-25` remained closed. The primary unit is a `decision_date × ticker × arm` set with at least two active economic H buckets. One candidate vote is used per bucket; raw Family multiplicity is not treated as independent consensus.

The old T1 same-ticker result is not interpreted as selector evidence. T1 features are set-level constants, and prediction variation inside a set is only floating-point noise; therefore T1 same-ticker selection is `NOT_APPLICABLE`.

## Evidence counts

- Events: `10836`; active sets: `495`; active Families: `108`.
- Active events are all in arm C: `{'C_ROLLING_REFIT_ROLLING_RECALIBRATION': 10836}`. Arms A/B remain in the prior eligibility denominator but generated no active events under the stored causal threshold/activation contract; they were not dropped by the walk-forward code.
- True multi-bucket sets: `134`; two-bucket: `79`; three-bucket: `55`.
- Bucket combinations: `{'SHORT_MID': 58, 'SHORT_MID_LONG': 55, 'MID_LONG': 17, 'SHORT_LONG': 4}`.
- Test months: `19`; paired T2/T3 rows: `248`.

## Paired scientific result

All T3-vs-T2 excess and regret values are paired on the same set, candidate buckets, fold and model. No CAGR, Sharpe, drawdown or terminal-wealth claim is made from overlapping H-specific forward returns.

- Same-ticker Rank-IC, pairwise accuracy, selected-excess difference and regret are reported in the CSV/JSON artifacts.
- Three-month calendar-block bootstrap (1000 repetitions) is the inference unit. Pooled results: same-ticker IC delta `mean=0.112975, median=0.109372, q05=-0.119928, q95=0.367806, positive_fraction=0.778`; pairwise-accuracy delta `mean=0.031770, median=0.029549, q05=-0.064295, q95=0.140688, positive_fraction=0.679`; selected-excess delta `mean=0.026691, median=0.025702, q05=0.001122, q95=0.058103, positive_fraction=0.971`; regret delta `mean=-0.026691, median=-0.026006, q05=-0.057212, q95=-0.001749, positive_fraction=0.974`. Full JSON is persisted in the four bootstrap files.
- Consensus and score-shape results are separated in `score_shape_comparison.csv`; robustness is split by set type, subperiod and reconstructed H30 removals.
- Model-separated paired bootstrap evidence:
- **RIDGE**: paired sets `124`, selected-excess delta median `0.014889`, q05 `-0.001717`; same-ticker IC delta median `0.017544`, q05 `-0.150088`; regret delta q95 `0.001566`.
- **HGB**: paired sets `124`, selected-excess delta median `0.037204`, q05 `-0.002335`; same-ticker IC delta median `0.096216`, q05 `-0.275481`; regret delta q95 `0.002528`.
- Early/late paired selected-excess deltas: `EARLY/HGB: Δexcess=0.0130; EARLY/RIDGE: Δexcess=0.0087; LATE/HGB: Δexcess=0.0103; LATE/RIDGE: Δexcess=-0.0003`.
- Reconstructed anti-long-horizon results: `FULL_UNIVERSE/HGB: sets=134, Δexcess=0.0119; FULL_UNIVERSE/RIDGE: sets=134, Δexcess=0.0050; REMOVE_H30/HGB: sets=134, Δexcess=0.0119; REMOVE_H30/RIDGE: sets=134, Δexcess=0.0050; REMOVE_H28_H30/HGB: sets=133, Δexcess=0.0119; REMOVE_H28_H30/RIDGE: sets=133, Δexcess=0.0050; REMOVE_H_GE25_D_GE25/HGB: sets=134, Δexcess=0.0119; REMOVE_H_GE25_D_GE25/RIDGE: sets=134, Δexcess=0.0050`. The ablation reclassifies candidate sets and excludes sets that fall below two buckets; it is an evaluation robustness check using the fixed causal fold predictions, not a new model search.

## Diagnosis

**TRUE_CROSS_HORIZON_SIGNAL_SUGGESTIVE_BUT_NOT_ROBUST**

This diagnosis is research evidence only. It does not grant any Selector or Family capital authority. If the evidence is suggestive but the bootstrap interval crosses zero, the result is not treated as robust feasibility.

## Self-review

- Final holdout opened: **NO**.
- Factory retraining: **NO**.
- Prediction regeneration: **NO**.
- Ex-ante state only: **YES**; forward realized excess is target-only.
- Inactive models treated as negative: **NO**; only active signals enter candidates.
- Family and economic H-bucket levels separated: **YES**.
- H30/long-horizon proxy risk reconstructed: **YES**, see `anti_h30_true_consensus.csv`.
- Equal-active candidate mean is the paired hurdle; no pseudo-NAV inference.
- Time dependence: calendar 3-month block bootstrap.
- Peak measured RSS: `0.223 GB`; runtime: `16.76 seconds`.
- Promotion authority: **NONE**.
