# Top-10 Causal Monthly Weight Controller V1

## Goal

Replace fixed 1M/3M/6M weighting profiles with a causal online controller over twelve discrete monthly age buckets `M1..M12`.

The controller is allowed to learn only from outcomes that were fully known at the assessment date. It never sees a future outcome for an earlier decision and never retrains the entry or exit models.

## Time contract

- Controller initialization: **2020-09-01**.
- Warm-up/calibration: 2020-09-01 through **2023-08-10**.
- Warm-up source: historical `WF_001..WF_007` OOS entry predictions only. `WF_000` remains excluded.
- Warm-up is development calibration only. No performance claim is made for it.
- Live evaluation: **2023-08-11 through 2026-07-24**.
- Live source: the already-completed causal-expanding entry and exit prediction artifacts (~17k checkpoint artifacts). They are read, not regenerated.
- Final frozen holdout remains closed.

## Monthly experts

Each model has twelve controller experts:

- `M1`: newest 21 fully matured decision sessions.
- `M2`: the 21 matured sessions immediately before M1.
- ...
- `M12`: the oldest 21-session block in the trailing 252 fully matured decision sessions.

An outcome is visible only when `terminal_date <= assessment_date`.

The controller does not update until all twelve 21-session experts are fully available. This avoids giving unavailable older experts an artificial disadvantage during the first year of warm-up.

## Initial state

Every model starts with uniform weights:

```text
M1 = M2 = ... = M12 = 1/12
```

The controller state is persistent and evolves sequentially. It is never recomputed by globally optimizing the entire 2020-2026 history.

## Model-specific evidence

Past performance is evaluated from relative entry candidates for that frozen Top-10 model:

1. Rank the model's entry score cross-sectionally for each decision date.
2. Apply the frozen `top_fraction`.
3. Cap the candidate count by the frozen `max_names`.
4. Evaluate the model-predicted horizon return only after the outcome has fully matured.

This makes the controller state model-specific where the frozen portfolio constraints differ.

For each monthly expert the suite records:

- candidate count,
- mean realized net excess,
- clipped mean realized net excess,
- median excess,
- hit rate,
- median daily p99 score,
- exact window start/end,
- latest terminal date used.

Individual realized net-excess observations are clipped to +/-50% **only for the controller quality update** to stop one extreme stock from dominating an entire expert. Raw mean/median/hit-rate values remain in the audit artifact.

## Online weight update

The only quality value used to move weights is the clipped mean realized net excess of each expert.

At each monthly assessment:

1. Rank the 12 expert qualities cross-sectionally.
2. Convert percentile ranks to a bounded `[-1, 1]` signal.
3. Form a multiplicative-weights/Hedge target:

```text
target_k ∝ current_weight_k * exp(eta * rank_signal_k)
eta = 1.0
```

4. Project the target onto the constrained simplex.

Hard constraints:

```text
weight floor = 2%
weight ceiling = 25%
max absolute change per assessment = 2 percentage points
sum(weights) = 100%
```

The 2 percentage-point step limit is why the controller begins in September 2020 rather than at the 2023 live boundary.

## Threshold scale

The controller does not choose a threshold by maximizing historical PnL.

Each expert supplies its median daily p99 score scale. The current controller weights produce:

```text
weighted_p99_t = sum(w_k,t * p99_k,t)

replacement_threshold_t =
    historical_threshold_over_p99
    * weighted_p99_t
```

The production-facing threshold remains monotone-safe:

```text
effective_threshold_t = min(raw_threshold_t, replacement_threshold_t)
```

The controller can therefore relax an obsolete absolute threshold but never make it more restrictive.

Any state update at assessment date `t` becomes effective only on a strictly later decision session.

## Predeclared live arms

The 2023-2026 evaluation runs exactly three arms for each of the ten frozen Top-10 models:

1. `RAW` — original daily threshold.
2. `UNIFORM_12M` — twelve monthly score-scale experts fixed at `1/12` each.
3. `CONTROLLER_12M` — persistent online weights learned from fully matured past performance.

No arm may be promoted solely because it wins this evaluation period. The purpose is to test whether online past-performance weighting adds value over both the broken raw threshold and a non-learning 12M calibration baseline.

## Required invariants

- no model fit,
- no prediction generation,
- historical warm-up uses only `WF_001..WF_007`,
- `WF_000` excluded,
- live evaluation reuses the existing causal entry/exit prediction Parquets,
- no threshold search,
- no weight-grid search,
- no future outcome in an earlier controller update,
- controller state at day `t` can affect only later sessions,
- maximum monthly reweighting enforced,
- weight floor/ceiling enforced,
- final holdout closed.

## Outputs

The suite writes:

- `monthly_controller_state.csv`
- `monthly_controller_state.parquet`
- `monthly_controller_expert_evidence.parquet`
- `monthly_controller_live_thresholds.csv`
- `monthly_controller_live_thresholds.parquet`
- `controller_weights_at_live_start.csv`
- `monthly_controller_results.csv`
- `monthly_controller_trades.csv`
- `monthly_controller_curves.parquet`
- `monthly_controller_summary.json`
- `REPORT.md`

The most important audit artifact is `controller_weights_at_live_start.csv`: it proves that the 2023 evaluation begins with a controller state learned exclusively from the 2020-2023 development warm-up rather than from the later evaluation outcomes.
