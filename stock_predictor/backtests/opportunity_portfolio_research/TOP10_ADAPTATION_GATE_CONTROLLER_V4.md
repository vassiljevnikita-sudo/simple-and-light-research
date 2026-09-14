# Top-10 Adaptation Gate Controller V4

## Goal

V4 adds a causal **adaptation-strength gate** on top of the model-specific V3 memory controller.

V3 showed that adaptive monthly-memory weights are valuable for some model systems, especially the H28/D21 group, but can reduce performance for others. V4 therefore separates two questions:

1. What adaptive memory weights would V3 choose?
2. How much of those adaptive weights should be allowed to override the model's frozen static prior right now?

The second question is controlled by a model-specific `lambda_t`.

## Methodological status

This suite is **development reuse, not independent OOS**.

The V4 architecture was designed after inspecting the 2023-2026 V3 development results. Therefore the same 2023-08-11 through 2026-07-24 interval can be used only for controller development comparison.

The Final Holdout remains closed.

## Data contract

Warm-up:

- 2020-09-01 through 2023-08-10
- historical `WF_001..WF_007` OOS entry predictions
- `WF_000` excluded
- controller-state calibration only

Development replay:

- 2023-08-11 through 2026-07-24
- existing causal-expanding entry/exit prediction artifacts
- reuses the existing ~17k causal artifacts
- no model retraining
- no prediction regeneration

## Frozen V3 prior classes

V4 keeps the exact V3 model-specific prior classes:

- `SHORT`: M1-M3 dominant
- `MID`: M4-M6 dominant
- `LONG`: M7-M12 dominant

The prior assignment is not changed online.

The V3 adaptive learner also remains unchanged and continues to update in shadow mode even when the V4 gate is closed.

## Core gate idea

The gate must not evaluate the new adaptive weights on the same monthly-quality snapshot that created those weights.

At assessment `t`:

```text
adaptive weights from t-1
        |
        v
newly matured M1..M12 quality snapshot at t
        |
        +--> score static prior
        |
        +--> score previous adaptive weights
        |
        v
one-step adaptive advantage
        |
        v
update lambda_t
        |
        v
NOW update V3 adaptive weights using snapshot t
        |
        v
blend static prior and new adaptive weights
        |
        v
threshold effective only after t
```

This ordering prevents the gate from rewarding an adaptive state merely because it was fit to the current evidence.

## Gate observation

For each fully available monthly snapshot:

```text
q_k = clipped mean realized net excess of expert Mk

prior_score = sum(prior_weight_k * q_k)
adaptive_score = sum(previous_adaptive_weight_k * q_k)

advantage = adaptive_score - prior_score
```

The advantage is normalized by robust cross-expert dispersion:

```text
scale = 1.4826 * MAD(q)
```

with deterministic standard-deviation fallback if MAD degenerates.

```text
advantage_z = clip(advantage / scale, -2, +2)
```

Only fully matured outcomes are visible:

```text
terminal_date <= assessment_date
```

## Gate memory

The gate uses the most recent six monthly one-step observations.

Predeclared constants:

```text
lookback assessments = 6
minimum observations = 3
initial lambda = 0
max lambda change per assessment = 0.20
```

The rolling evidence is:

```text
rolling_advantage_z = mean(last up to 6 one-step advantage_z observations)
```

Target adaptation strength:

```text
lambda_target =
    clip(max(0, rolling_advantage_z) / 1.0, 0, 1)
```

Interpretation:

- no demonstrated adaptive advantage -> target lambda 0
- +0.25 sigma rolling advantage -> target lambda 0.25
- +0.50 sigma -> target lambda 0.50
- +1.00 sigma or more -> target lambda 1.00
- negative evidence never creates negative adaptation; it closes the gate toward the static prior

The actual lambda may move by at most 0.20 per monthly assessment.

These constants are fixed before the V4 replay. No parameter grid is searched.

## Weight blend

The V3 adaptive learner remains fully active internally.

Production-facing memory weights are:

```text
gated_weights_t =
    (1 - lambda_t) * static_prior_weights
    + lambda_t * v3_adaptive_weights_t
```

Therefore:

```text
lambda = 0   -> pure static prior
lambda = 1   -> full V3 adaptive controller
```

Intermediate values provide shrinkage toward the static prior.

## Threshold contract

The gated weights combine the same twelve monthly p99 score-scale experts:

```text
gated_weighted_p99 =
    sum(gated_weight_k * monthly_p99_k)

replacement_threshold =
    historical_threshold_over_p99 * gated_weighted_p99

effective_threshold =
    min(raw_threshold, replacement_threshold)
```

V4 can only relax the raw threshold. It may never tighten it.

A state produced at assessment date `t` can affect only a strictly later decision session.

## Replay arms

Exactly four arms are replayed for each frozen Top-10 model:

1. `RAW`
2. `STATIC_MODEL_PRIOR`
3. `ADAPTIVE_V3_REFERENCE`
4. `GATED_ADAPTIVE_V4`

This gives 40 portfolio replays.

`ADAPTIVE_V3_REFERENCE` must exactly reproduce the already-published V3 adaptive result for every model, including CAGR excess and trade count. The suite fails closed if reproduction differs.

## Required invariants

- no model fit
- no prediction generation
- existing ~17k causal artifacts reused
- historical warm-up uses only `WF_001..WF_007`
- `WF_000` excluded
- no threshold optimization
- no gate-parameter grid search
- no online prior-class switching
- previous adaptive weights are scored before current adaptive weights are updated
- gate outcomes fully matured before use
- gate state affects only later decision sessions
- lambda constrained to `[0, 1]`
- max lambda movement 0.20 per assessment
- effective threshold never exceeds raw threshold
- frozen V3 replay reproduced exactly
- Final Holdout closed
- development-period winner selection prohibited

## Outputs

The suite writes:

- `adaptation_gate_state.csv`
- `adaptation_gate_state.parquet`
- `adaptation_gate_expert_evidence.parquet`
- `adaptation_gate_live_thresholds.csv`
- `adaptation_gate_live_thresholds.parquet`
- `adaptation_gate_state_at_live_start.csv`
- `adaptation_gate_prior_manifest.csv`
- `adaptation_gate_results.csv`
- `adaptation_gate_trades.csv`
- `adaptation_gate_curves.parquet`
- `adaptation_gate_v4_vs_v3.csv`
- `adaptation_gate_summary.json`
- `REPORT.md`

The key diagnostic columns are:

- `gate_advantage_z`
- `rolling_advantage_z`
- `lambda_before`
- `lambda_target`
- `lambda_after`
- `adaptive_l1_distance_from_prior`
- `gated_l1_distance_from_prior`

These show whether the gate learned to open for model systems where adaptation is predictively useful and to remain near the anchor where it is not.
