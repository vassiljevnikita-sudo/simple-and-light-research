> **Historical point-in-time audit (2026-08-15):** The scientific checks below are preserved, but the host/runtime baseline (Ryzen 5800X / 32 GB / 16 logical processors) is obsolete for current large runs. Current data/scale/runtime context is [DYNAMIC_QBD_DATA_AND_RUNTIME.md](DYNAMIC_QBD_DATA_AND_RUNTIME.md); current research state is [DYNAMIC_QBD_CURRENT_STATE.md](DYNAMIC_QBD_CURRENT_STATE.md).

# Dynamic-QBD system audit — 2026-08-15

## Scope

This audit covers the Dynamic-QBD generation, model-selection, calibration,
A/B/C replay, evidence, gate, restart, freeze and local-runtime chain.  It does
not open the historical final holdout and grants no capital authority.

## Host and resource baseline

- Windows 11 Pro, build 26200
- AMD Ryzen 7 5800X: 8 physical cores / 16 logical processors
- 32 GB physical RAM
- approximately 17.6 GB free before the measured run

The measured four-generation audit run peaked at 1.65 GB working set and
3.07 GB private bytes.  Minimum system-free RAM remained approximately
16.1 GB.

## Contract results

The fast factory, H1–H30 adapter and portfolio performance self-tests pass.
The real-data audit additionally verified:

- four valid monthly generations;
- only causally completed OOS folds contribute to model selection;
- purged and disjoint training/calibration windows;
- horizon-matured training labels;
- changing generation thresholds;
- model-pure prediction keys;
- A/B/C schedule completeness and rolling C generations;
- exact NAV accounting identity;
- hard `max_names` enforcement;
- explicit entry-notional sleeve contract with reported mark-to-market drift;
- positive costs on executed trades;
- known entry/exit generation lineage;
- explicit pre-tax execution;
- a verified terminal pre-holdout state hash;
- complete active-generation, replay-state, evidence and gate cursors;
- no historical holdout access.

The requested 2016-01-01 through 2017-12-31 run remains correctly fail-closed.
The locked candidate evidence has no completed causal H3 OOS fold in that
period, so all 19 monthly attempts return
`NO_CAUSAL_H1_30_CANDIDATE_EVIDENCE:H3`.  Future fold evidence was not used to
manufacture an early result.

## Model-family and fold audit

The selector consumes every completed, non-selection-only Outer WF fold as of
the first generation cutoff.  It excludes `INNER_FOR_WF_000`.

- In the 2020-Q4 audit window the only available folds were `WF_000` and
  `WF_001`.
- At a 2023-12-29 cutoff the available set is `WF_000` through `WF_007` — eight
  folds, not only `WF_001` through `WF_007`.
- The selected H3 recipe at that later cutoff is `RIDGE(alpha=10)`.
- One recipe is selected and refitted.  There is no ensemble, averaging, or
  intersection across Ridge/HGB model families.
- Robust fold Spearman drives selection, with MAE as a later tie-break.
  R-squared is diagnostic only.  A regression test that reverses the R-squared
  ordering leaves the selected candidate unchanged.

Exact fold IDs, the selection-metric contract and the no-ensemble contract are
now persisted in every `ModelGeneration`.

## Performance profile and changes

Initial profile for one family and four monthly generations:

- first run: 14.9 seconds;
- model-generation/refit work: approximately 6.5 seconds;
- partitioned price projection: approximately 2.3 seconds / 483 source files;
- repeated artifact hashing and calibration reads were the next largest costs.

Implemented runtime improvements:

1. Daily-store price projections now have an explicit date/universe/source
   identity contract and are reused only after that contract matches.
2. Hard-linked output aliases share one content-hash calculation.
3. A completed run resumes only after matching its run contract, pipeline
   fingerprint and every artifact hash.

Measured verified resume:

- before completed-run reuse: approximately 7.1–7.6 seconds;
- after hardening: 3.0 seconds;
- first-run versus verified-resume speedup: 5.0x.

These changes alter runtime and provenance only.  They do not change features,
targets, model selection, thresholds, trades, NAV, evidence or gate decisions.

## Real-data audit result

Audit window: 2020-09-01 through 2020-12-31, H3/D2/N1 Fixed, diagnostic
score quantile 0.50.

- generations: 4
- executed A/B/C trades: 15
- maximum positions: 1
- maximum accounting error: 0.0 EUR
- maximum observed stock exposure: 50.865%
- mark-to-market sleeve-drift observations across arms: 9
- Gate 1: fail/no selector authority, as expected for the short evidence window
- final holdout opened: no

Artifacts are under `artifacts/dynamic-qbd-audit-2020q4-final-v2` and are local
diagnostic outputs, not production or final-holdout evidence.

## Causal recipe-reselection shadow

The completed H3 evidence would have selected different recipes as additional
Outer WF folds matured:

- 2020-08-31: Ridge, alpha 10, using WF_000–WF_001;
- 2022-08-31: HGB, learning rate 0.05 / 31 leaves / L2 1, using WF_000–WF_005;
- 2023-02-28: Ridge, alpha 10, using WF_000–WF_006.

All three generations were freshly fitted on their causally permitted moving
training windows; no historical model artifact was reused.  In the shadow
portfolio from 2020-08-31 through 2023-12-29, the frozen Ridge recipe refitted
on the same activation dates ended at EUR 13,398.25 and the rolling
causal-reselection arm at EUR 14,703.12, against EUR 12,889.25 for URTH.
Rolling reselection therefore added EUR 1,304.87 in the matched-refit contrast,
but also increased relative maximum drawdown from -24.27% to -36.11%.  A stale
first-generation Ridge arm ended at EUR 13,452.30 but is diagnostic only because
it confounds recipe choice with model age.  This is development-only evidence
and does not grant promotion or holdout authority.

The separate 2023-Q4 three-way diagnostic ended at EUR 11,582.26 for the
eligible positive-Spearman/positive-R² single-fold model, EUR 11,039.46 for the
robust selected single model, EUR 10,973.56 for the Ridge/HGB intersection,
and EUR 11,129.42 for URTH.  The single-fold result is explicitly high-risk
diagnostics rather than research selection evidence.

The suite CPU resource contract now targets approximately 90% measured CPU
utilization while exposing all 16 logical processors.  A 14-thread affinity
mask would cap runnable capacity, not actual utilization, and would undershoot
when jobs pause for I/O, scheduling or synchronization.  Model fits remain
serial to avoid multiplying panel RAM; each fit may use all 16 native threads.

In a short real-data verification sample the corrected 16-thread contract
reached 98.1% measured whole-system CPU at peak (91.7% in the preceding sample)
with a 1.76 GB peak working set.  This confirms that all host capacity is
available and the requested approximately 90% operating peak is reachable.
Short I/O, serialization and portfolio phases are not expected to saturate the
CPU continuously.
