# Dynamic-QBD Matched A/B/C Recalibration Suite

## Purpose

This Development-only suite tests the remaining adaptation decomposition on the
same H3/D2/N1 replay contract used by the conditional-refit experiment.

It freezes the **exact same first causal recipe in every arm** and permits no
recipe switching.

Authority remains:

```text
SHADOW_ONLY_NO_PROMOTION_NO_HOLDOUT
```

The final holdout begins at `2026-07-25` and is fail-closed.

## Arms

### A — frozen model + frozen calibration

`A_FROZEN_MODEL_FROZEN_CALIBRATION`

- fit the first causal recipe once;
- resolve threshold/top-fraction once at the initial activation;
- reuse the same model artifact, generation calibration and threshold for the
  full replay;
- no later recalibration and no later refit.

### B — frozen model + rolling recalibration

`B_FROZEN_MODEL_ROLLING_RECALIBRATION`

- use the exact same initial model artifact as A;
- no later refit;
- resolve threshold/top-fraction at every monthly assessment using only the
  matured calibration window available then.

### C — monthly refit + monthly calibration

`C_MONTHLY_REFIT_FROZEN_RECIPE`

- freeze exactly the same recipe identity as A/B;
- freshly refit that recipe at every monthly assessment;
- calibrate the fresh model at that assessment;
- recipe selection is not allowed to change the recipe.

## Primary causal contrasts

```text
B - A = pure rolling-recalibration effect
C - B = fresh-refit effect conditional on monthly calibration
```

No tolerance, parameter grid or alternative policy is selected from the
evaluation period.

## Hard matching checks

The runner fails closed unless all are true:

- A, B and C use the same initial model artifact;
- A, B and C use the same initial threshold;
- one exact recipe identity is used across all arms and all assessments;
- A has one schedule generation and one calibration resolution;
- B uses one model artifact over the full replay;
- B and C have one schedule row per assessment;
- C has one distinct freshly fit model artifact per assessment.

These checks are persisted in `contract-audit.json`.

## Default reproduction cell

```text
start          2020-08-31
end            2023-12-29
H              3
D              2
N              1
score_quantile 0.75
top_fraction   0.01
sleeve         50%
cost           20 bps round-trip
benchmark      URTH
initial        EUR 10,000
```

## Run

Use the same input paths as the immediately preceding conditional-refit run.
Pass the checked-out commit explicitly so the result manifest is tied to the
code that produced it.

```powershell
$sha = git rev-parse HEAD

python -m stock_predictor.backtests.opportunity_portfolio_research.dynamic_qbd_abc_recalibration_experiment `
  --signal-panel "D:\path\to\signal-panel.parquet" `
  --candidate-metrics "D:\path\to\candidate-metrics.json" `
  --daily-store-root "D:\path\to\daily-parquet" `
  --benchmark-daily-path "D:\path\to\URTH-direct-daily.parquet" `
  --direct-daily-stock-root "D:\path\to\alpaca-stock-direct-daily-store" `
  --output-root "artifacts\dynamic-qbd-abc-recalibration-h3-d2-n1" `
  --code-commit $sha
```

The suite calls the existing Dynamic-QBD resource manager and therefore retains
the suite-wide Windows 60 GiB hard memory cap.

## Outputs

- `summary.json`
- `REPORT.md`
- `portfolio-value-comparison.csv`
- `causal-contrasts.json`
- `contract-audit.json`
- `abc-plan-audit.csv`
- `fit-audit.json`
- `calibration-audit.csv`
- one schedule CSV per arm
- one NAV Parquet per arm
- one trades Parquet per arm
- fresh-model and assessment provenance below the output root

Large raw model/prediction artifacts may remain local; the summary, report,
audits and comparison files are sufficient for the remote result review.

## Interpretation

The first question is whether B-A is economically negative. If A recovers while
B remains weak, rolling recalibration is directly implicated within this matched
cell.

The second question is C-B. It tells us whether fresh refitting helps or hurts
once monthly calibration is already allowed.

Neither contrast grants promotion or capital authority. This is only a causal
Development decomposition.
