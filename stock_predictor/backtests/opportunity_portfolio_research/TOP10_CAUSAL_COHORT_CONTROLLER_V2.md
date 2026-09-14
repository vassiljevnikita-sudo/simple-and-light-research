# Top-10 Causal Cohort Controller V2

## Goal

Replace the V1 lag-slot controller with a controller whose weights are attached to
concrete historical 21-session cohorts.

V1 learned that `M3` or `M6` was good as an age slot. V2 instead learns that a
specific historical cohort was good and keeps that cohort's weight while it ages
from `M1 -> M2 -> ... -> M12`.

## Methodological status

This is **development reuse**, not a fresh independent OOS test.

V2 was designed after inspecting the V1 results from 2023-2026. Therefore the
2023-08-11 through 2026-07-24 replay may be used to compare controller designs
during development, but it must not be described as untouched validation.

The final frozen holdout remains closed.

## Data contract

Warm-up:

- start: 2020-09-01
- end: 2023-08-10
- source: historical `WF_001..WF_007` OOS entry predictions
- `WF_000` excluded
- purpose: controller-state calibration only

Live development replay:

- start: 2023-08-11
- end: 2026-07-24
- source: existing causal-expanding entry/exit prediction Parquets
- reuses the existing ~17k causal artifacts
- no model retraining
- no prediction regeneration

## Fixed cohorts

The full decision-session calendar is partitioned once into consecutive fixed
21-session blocks:

```text
C001 = sessions 1..21
C002 = sessions 22..42
...
```

At each assessment, the newest twelve fully matured available cohorts form the
controller memory.

The newest cohort is labeled `M1`, the oldest `M12`, but the **weight belongs to
the cohort ID**, not to `M1..M12`.

Example:

```text
assessment t:
C020 = M1, weight 18%
C019 = M2, weight 12%

assessment t+1:
C021 = M1
C020 = M2, still carries its prior cohort weight before the new update
C019 = M3, still carries its prior cohort weight before the new update
```

When the oldest cohort leaves memory, its released weight mass is assigned to the
new cohort entering M1 before the next quality update. Retained cohorts keep their
weights exactly through the roll step. If no newly matured cohort entered since the
last state, V2 records the assessment but does not relearn the same evidence twice.

## Cohort evidence

For each model and cohort:

1. rank the entry-score cross-section on each decision date,
2. apply frozen `top_fraction`,
3. cap by frozen `max_names`,
4. wait until the cohort's horizon is fully matured,
5. evaluate realized net excess only from outcomes already visible at the
   assessment date.

Stored evidence includes:

- realized candidate count,
- raw mean net excess,
- median net excess,
- hit rate,
- mean net excess after clipping individual observations to +/-50%,
- median daily p99 score,
- confidence,
- cohort start/end dates.

## Magnitude-aware quality

V1 reduced quality to a rank. V2 preserves relative magnitude.

For the twelve current cohorts:

```text
q_k = clipped mean realized net excess
median_q = median(q)
scale = 1.4826 * MAD(q)

robust_z_k =
    clip((q_k - median_q) / scale, -2, +2)
```

If MAD degenerates, standard deviation is used as a deterministic fallback.

## Confidence

Sparse cohorts receive less influence:

```text
target_count = min(30, 21 * max_names)

confidence_k =
    min(1, sqrt(realized_candidate_count / target_count))

controller_signal_k =
    robust_z_k * confidence_k
```

Confidence changes only the update signal. Raw evidence remains stored separately.

## Online weight update

Initial state:

```text
Cohort weights = 1/12 each
```

Target:

```text
target_k ∝ rolled_weight_k * exp(eta * controller_signal_k)

eta = 1.0
```

The target is projected onto the constrained simplex.

Hard limits:

```text
weight floor = 1%
weight ceiling = 35%
max absolute change per assessment = 5 percentage points
sum(weights) = 100%
```

These parameters are predeclared for V2. No grid search is performed on the
2023-2026 replay.

## Threshold scale

Each current cohort supplies a median daily p99 score.

```text
weighted_p99 =
    sum(cohort_weight_k * cohort_median_p99_k)

replacement_threshold =
    historical_threshold_over_p99 * weighted_p99

effective_threshold =
    min(raw_threshold, replacement_threshold)
```

The controller may relax an obsolete raw threshold but may never tighten it.

A state produced on assessment date `t` can only affect a strictly later decision
session.

## Replay arms

For each of the ten frozen Top-10 models:

1. `RAW`
2. `COHORT_UNIFORM_12M`
3. `COHORT_CONTROLLER_V2`

The prior V1 `CONTROLLER_12M` result is loaded as a frozen reference and is **not**
replayed in this suite.

No winning arm may be promoted from this development replay as if it were a fresh
OOS selection.

## Required invariants

- no model fit,
- no prediction generation,
- no threshold search,
- no weight-grid search,
- historical warm-up uses only `WF_001..WF_007`,
- `WF_000` excluded,
- live replay uses existing causal artifacts,
- outcome must be fully mature before it can affect a controller update,
- controller update affects only later sessions,
- cohort weights age with the data cohort,
- no repeated update without a newly matured cohort,
- max +/-5 pp reweighting enforced,
- 1% floor enforced,
- 35% ceiling enforced,
- effective threshold never exceeds raw threshold,
- final holdout closed.

## Outputs

The suite writes:

- `cohort_controller_state.csv`
- `cohort_controller_state.parquet`
- `cohort_controller_evidence.parquet`
- `cohort_controller_live_thresholds.csv`
- `cohort_controller_live_thresholds.parquet`
- `cohort_weights_at_live_start.csv`
- `cohort_controller_results.csv`
- `cohort_controller_trades.csv`
- `cohort_controller_curves.parquet`
- `cohort_controller_v1_comparison.csv`
- `cohort_controller_summary.json`
- `REPORT.md`

`cohort_weights_at_live_start.csv` proves that the controller entered 2023-08-11
with state learned solely from the pre-live warm-up.
