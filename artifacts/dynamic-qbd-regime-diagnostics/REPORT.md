# Dynamic-QBD Regime Persistence / Selector Feasibility

Primary diagnosis: **PERSISTENCE_EXISTS_BUT_TOO_WEAK_AFTER_COSTS**

## Scope and research safeguards

- Assessment coverage: `2016-02-29 00:00:00` to `2025-12-31 00:00:00`; holdout boundary is exclusive at `2026-07-25 00:00:00`.
- Input universe: `2790` Families, arms A/B/C; cluster counts: `{'FAMILY': 2790, 'H_BAND': 4, 'HD_GRID': 60, 'ECONOMIC_REGION': 10}`.
- Models retrained: **no**. Predictions regenerated: **no**. Final holdout opened: **no**.
- State variables are current/trailing, fully matured monthly evidence. Forward 1M/3M values are evaluation targets only.
- Opportunity count was not present in authoritative evidence and remains explicitly unavailable; it was not replaced by zero.

## Persistence results

- Family-level mean rank autocorrelation: 1M `0.454818`, 3M `0.521432`.
- Cluster-level mean rank autocorrelation across H-band, H×D×N×exit-grid and economic-region levels: 1M `0.446856`, 3M `0.482703`.
- Family top-quartile forward spread (top minus rest): 1M `0.001915`, 3M `-0.001279`.
- Cluster top-quartile forward spread: 1M `0.001555`, 3M `-0.000293`.
- Duration and transition tables show that apparent rank persistence is substantially longer at coarse H/D levels than at individual-family level; this is consistent with smoothing and is not by itself predictive evidence.

## Activity-aware interpretation

- Rank autocorrelation conditional on active state: 1M `0.070356`, 3M `0.049095`.
- Rank autocorrelation for no-opportunity state: 1M `0.440095`, 3M `0.562289`.
- `NO_OPPORTUNITY` is a separate state from `ACTIVE_NEGATIVE`; sparse models are not penalized merely for producing zero return in an inactive month.

## Selector-feasibility benchmarks

- F0 Oracle is retrospective and diagnostic only; it has no authority.
- At 20 bp additional switch cost, the strongest non-family diagnostic cell was `HD_GRID / F4_INCUMBENT / 3M` with incremental-vs-equal mean `0.007096` and bootstrap q05 `-0.000765`.
- Equal weight is the hurdle. Results are reported at 0/10/20 bp additional selector cost; existing portfolio costs are not added a second time.
- These are feasibility measurements, not parameter tuning and not a production selector recommendation.

## Anti-H30 / anti-proxy result

- Economic-region full universe mean 1M spread: `0.002080`; after removing H28–H30: `0.001017`.
- Economic-region full universe mean 3M spread: `0.000063`; after removing H28–H30: `-0.000984`.
- The ablation does not show a collapse to zero at the regional level, but neither does it establish a cost-robust selector edge; the known long-horizon region is therefore not treated as sufficient evidence.

## Statistical uncertainty

- All bootstrap intervals use monthly blocks of `3` months and `1000` repetitions; overlapping 3M targets are evaluated through non-overlapping 3M cohorts in selector feasibility.
- Percentiles and positive fractions are uncertainty summaries over time blocks, not IID p-values.

## Scientific self-review

1. No models retrained; no predictions regenerated; no final holdout opened.
2. Selector state uses only information available by assessment time; forward values are targets.
3. Inactivity and negative active performance are separate states.
4. Family and cluster levels are analyzed separately.
5. H30/long-horizon ablations were executed.
6. Equal weight is the hurdle and no result receives capital authority.
7. Temporal dependence is handled through monthly block bootstrap.
8. The result distinguishes unstable families, smoothed cluster persistence, and weak forward predictiveness; it does not justify promotion.
9. Inputs, config, seed and executable code are fingerprinted in `manifest.json`.

## Conclusion

**PERSISTENCE_EXISTS_BUT_TOO_WEAK_AFTER_COSTS**. This is a Development/Research-only feasibility diagnosis. No Family, cluster, or selector is promoted, and no capital authority is granted.

See the CSV/JSON/Parquet artifacts in this directory for full transitions, durations, cost stress, activity conditioning, bootstrap distributions and ablations.
