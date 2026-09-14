# Frozen Top-10 Backward + Forward External Validation

## Question

Test whether the ten best portfolio cells selected on the existing 2020-08-10..2023-08-10 development OOS surface retain their CAGR excess outside that selection window.

This is external validation, not a new H/D/N search. The Top-10 identities and the known-window results are immutable inputs.

## Raw-data boundary

The guard reads timestamps from the JSONL payloads in `alpaca-minute/*.jsonl.gz` and requires exactly:

- first observed minute: `2016-01-01T00:00:00Z`
- last observed minute: `2026-07-24T23:59:00Z`

Filename dates are not treated as evidence. Files named through 2026-07-26 are allowed; they do not move the observed maximum unless their payload contains a later timestamp.

## Canonical prediction artifacts

The official runner consumes the existing selected walk-forward artifacts. It does **not** require separately manufactured Backward/Forward prediction files or external training-manifest JSONs.

Under the supplied artifact root the canonical paths are:

- V5 signal: `training/signal/selected-walk-forward-predictions.parquet`
- Learned Exit: `training/e1-30-learned-exit-20260809/exit/selected-walk-forward-predictions.parquet`
- Stop Execution companion: `training/stop-execution/selected-walk-forward-predictions.parquet`

The V5 and Learned-Exit files are the economic prediction inputs. The Stop-Execution parquet is retained in `prediction_provenance_audit.json` as a canonical companion/hash artifact; it is not injected into the Top-10 profit replay because the existing QbD-profit evaluator does not consume it.

Backward and Forward are temporal **whole-fold views of the same V5 selected-WF artifact**:

- BACKWARD keeps complete folds ending on or before `2020-08-07`.
- FORWARD evidence keeps complete folds starting on or after `2023-08-11` and ending on or before `2026-07-24`.
- FORWARD additionally keeps complete earlier folds ending on or before `2023-08-10` as causal calibration history.
- Any fold straddling the Known/Forward boundary is excluded rather than clipped.

No prediction values are extrapolated. If the canonical V5 parquet has no complete Forward OOS fold, the run fails with `NO_FORWARD_OOS_FOLDS_IN_CANONICAL_V5`. Equivalent explicit coverage failures apply to Backward and Learned Exit.

The official strict wrapper internally materializes temporary segment parquet views only to satisfy the existing replay API. These are derived execution files, not new research artifacts and are deleted after the run.

## Frozen Top 10

| rank | exit | H | D | N | known median CAGR excess |
|---:|---|---:|---:|---:|---:|
| 1 | LEARNED_EXIT | 11 | 3 | 1 | 97.61% |
| 2 | FIXED | 24 | 5 | 1 | 50.38% |
| 3 | LEARNED_EXIT | 28 | 21 | 1 | 30.69% |
| 4 | LEARNED_EXIT | 28 | 21 | 5 | 30.69% |
| 5 | LEARNED_EXIT | 28 | 21 | 4 | 30.69% |
| 6 | LEARNED_EXIT | 28 | 21 | 6 | 30.69% |
| 7 | LEARNED_EXIT | 28 | 21 | 2 | 30.69% |
| 8 | LEARNED_EXIT | 28 | 21 | 3 | 30.69% |
| 9 | LEARNED_EXIT | 24 | 21 | 5 | 8.99% |
| 10 | LEARNED_EXIT | 24 | 21 | 6 | 8.99% |

The existing `cell_contract` is reused: H, D, N, exit family, replacement, allocation and sleeve stay fixed. `score_quantile` and `top_fraction` are recalibrated only from prior folds, exactly as in the existing portfolio walk-forward procedure. No external result can change membership of the Top 10.

## Time split

- BACKWARD: all reportable outer folds must end on or before `2020-08-07`.
- KNOWN: `2020-08-10..2023-08-10`, loaded from the existing QbD-profit cell JSON and SHA-256 fingerprinted. It is never replaced by a recomputed value.
- FORWARD: reportable outer folds must start on or after `2023-08-11` and end on or before `2026-07-24`.

Every reported portfolio fold must satisfy `calibration_end < fold_start`.

The first reportable backward date is determined by the available selected-WF folds. The suite does not pretend that a model can be evaluated from the first raw minute in 2016 without prior training history.

## Causal-provenance gate

The previous version required two hand-authored external training manifests. That requirement was incorrect for this pipeline because the real research outputs are already the canonical `selected-walk-forward-predictions.parquet` artifacts.

The strict gate now verifies:

- exact canonical artifact path suffixes;
- SHA-256 hashes of all three canonical parquets;
- no locked Learned-Exit holdout rows;
- required V5 horizons for both external evidence windows;
- required Learned-Exit horizons;
- actual Backward and Forward date coverage;
- no prediction dates after the raw-data maximum;
- whole-fold segmentation with no Known/Forward fold clipping;
- portfolio causality through `calibration_end < fold_start`.

The audit deliberately does **not** claim an independently reconstructed `train_end` proof when the parquet does not expose a universal `train_end` field. Instead it records `independent_train_end_manifest_proof=false` and identifies the inputs as the canonical selected-WF artifacts. No fabricated manifest is accepted as stronger evidence than the pipeline's real artifacts.

## Primary classification

For each frozen model:

`back_retention = backward_median_CAGR_excess / known_median_CAGR_excess`

`forward_retention = forward_median_CAGR_excess / known_median_CAGR_excess`

Predeclared labels:

- `STABLE`: both external medians are positive and both retention ratios are at least 50%.
- `DECAY`: both are positive and the smaller retention ratio is at least 25%, but below 50%.
- `COLLAPSE`: otherwise, including sign reversal or retention below 25%.
- `NO_ACTIVITY`: at least one external segment has no active outer fold.
- `UNCLASSIFIED`: known median CAGR excess is not positive.

The main output is `retention_matrix.csv`, not the full-period euro ranking.

## Full-period wealth

The known result is chained multiplicatively:

`V_full = 10000 * G_backward * G_known * G_forward`

The benchmark is chained the same way. Euro profits are never added across segments.

As in the existing profit research, fold tax ledgers are independent. Therefore full-period tax-aware wealth is a research comparison, not a literal single-depot tax simulation running continuously from 2016 to 2026.

## Outputs

- `raw_coverage_audit.json`
- `frozen_top10.json`
- `known_bridge_manifest.json`
- `prediction_provenance_audit.json`
- `external_outer_rows.json`
- `retention_matrix.csv`
- `full_period_wealth.csv`
- `evaluation_summary.json`

`evaluation_summary.json` keeps `promotion_eligible=false` and `survivorship_bias_promotion_block=true`. The current universe is not yet verified point-in-time, so a strong external result cannot by itself promote the strategy.

## Self-test

```powershell
python -m stock_predictor.backtests.opportunity_portfolio_research.top10_external_validation --self-test
python -m stock_predictor.backtests.opportunity_portfolio_research.top10_external_validation_strict --self-test
```

The strict self-test additionally verifies that one canonical V5 artifact is partitioned into whole-fold Backward/Forward views and that a fold straddling the Known/Forward boundary is not admitted as Forward evidence.

## Run

If all three canonical files share the artifact root:

```powershell
.\stock_predictor\backtests\opportunity_portfolio_research\run_top10_external_validation.ps1 `
  -RawMinuteRoot "D:\...\alpaca-minute" `
  -ArtifactRoot "D:\...\artifacts-root" `
  -KnownProfitRoot "artifacts\learned-exit-qbd-profit" `
  -DailyStoreRoot "D:\simple-and-light-v5-local-training\artifacts\daily-parquet"
```

The runner resolves automatically:

```text
<ArtifactRoot>\training\signal\selected-walk-forward-predictions.parquet
<ArtifactRoot>\training\e1-30-learned-exit-20260809\exit\selected-walk-forward-predictions.parquet
<ArtifactRoot>\training\stop-execution\selected-walk-forward-predictions.parquet
```

If the files are stored under different roots, use the optional `-V5Predictions`, `-LearnedExitPredictions`, and `-StopExecutionPredictions` overrides. The canonical suffixes remain enforced.
