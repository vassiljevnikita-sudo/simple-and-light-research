# Dynamic-QBD Opportunity-State Predictability / Cross-Family Consensus

## Technical summary

Primary diagnosis: **NO_ROBUST_OPPORTUNITY_STATE_PREDICTABILITY**. The deduplicated consensus state does not clear the causal robustness and economic hurdles against Score-only; no selector authority is granted.
The run contains `10836` active events from `108` of `2790` design Families. Only `27.1%` of active date×ticker×arm sets contain more than one deduplicated active Family, limiting effective consensus evidence.

## Scope and safeguards

- Active event count: `10836`; valid targets: `10836`; active date×ticker×arm sets: `495`.
- Holdout boundary: `2026-07-25 00:00:00`; opened: **no**.
- Models retrained: **no**. Predictions regenerated: **no**. Only Ridge/HGB diagnostics were fit on historical opportunity evidence.
- SIGNAL_ACTIVE means `score >= resolved_threshold` and `decision_date >= activation_date`; forward `realized_excess` is target-only.
- Stock-return and separate benchmark-return components were unavailable in authoritative matured evidence and remain missing; no imputation was performed.

## Predictability result

- Mean fold Rank-IC: T2 Score-only `0.059665`, T3 Score+Consensus `0.118226`, T4 + Market `0.121491`.
- T3 minus T2 incremental Rank-IC: fold median `0.000000`, date-block bootstrap q05/q50/q95 `-0.036671` / `0.070806` / `0.172299`.
- T3 minus T2 top-quartile spread increment: fold median `0.000000`, date-block bootstrap q05/q50/q95 `-0.002779` / `0.000115` / `0.019586`.
- T1/T3 Consensus features use one primary economic vote per H bucket; missing H buckets are causal state information and are imputed only from the training fold for Ridge. Raw Family-density diagnostics are retained separately.

## Economic hurdle

- B0 Equal-active is the hurdle; B1 Score-only, B2 Consensus-only, B3 Score+Consensus, B4 +Market and B5 Oracle use the same active candidate sets.
- At 20 bp additional diagnostic cost, mean excess is B0 `0.080814`, B1 `0.112898`, B3 `0.086990`.
- Cost stress is reported at 0/10/20/30 bp. These are diagnostic cross-sectional replays, not production portfolios or capital-authorized selectors.

## Robustness and multiple-counting control

- H30, H28–H30, same-ticker, activity-count, consensus/disagreement, score and early/late ablations are persisted in `anti_h30_ablation.csv` and `subperiod_robustness.csv`.
- Raw active Family density is materially larger than deduplicated H-bucket breadth in the persisted RAW-vs-DEDUP table; raw density is therefore not accepted as independent consensus evidence.
- Inference is time-block based; no IID interpretation over the millions of correlated opportunity rows is used.

## Limitations and self-review

- T2 contains only concrete score/Family H-D-N information; T3 adds only predeclared consensus features; T4 adds only lagged URTH market state.
- Different H regimes are reported by SHORT/MID/LONG buckets; the original H-specific realized excess remains the primary target.
- All conclusions are Development/Research-only. No selector receives authority.

- The authoritative matured table exposes `realized_excess` but not separate stock- and benchmark-return components; those fields remain unavailable and were not imputed.
- Economic replay metrics are research diagnostics on overlapping H-specific opportunity returns, not deployable portfolio NAV statistics.

## Conclusion

**NO_ROBUST_OPPORTUNITY_STATE_PREDICTABILITY**
