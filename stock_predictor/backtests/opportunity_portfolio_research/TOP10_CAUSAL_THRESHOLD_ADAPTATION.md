# Top-10 Causal Threshold Adaptation V1

## Purpose

Test whether the frozen Top-10 entry policies can recover live activation when the absolute score threshold is allowed to adapt **only from information already available at the time**.

This suite does not train models, regenerate predictions, optimize thresholds, optimize weights, open the final holdout, or choose a winning weighting profile from forward performance.

## Predeclared weighting profiles

Each Top-10 model is evaluated under all three profiles. The weights are for the 1-month / 3-month / 6-month retrospective evidence signals.

| Profile | 1M | 3M | 6M | Intent |
|---|---:|---:|---:|---|
| `W_RECENT` | 0.50 | 0.30 | 0.20 | strongest recency emphasis |
| `W_MID` | 0.20 | 0.50 | 0.30 | strongest 3-month emphasis |
| `W_LONG` | 0.20 | 0.30 | 0.50 | strongest 6-month emphasis |

These are three fixed hypothesis arms, not a tuning grid. The report may compare them but must not automatically promote the best-looking arm.

## Causal assessment clock

Assessment occurs every 21 decision sessions.

For an assessment on date `A`:

- prediction scores with `decision_date < A` are known;
- outcome evidence is allowed only when `terminal_date <= A`;
- any resulting threshold change becomes effective only on a later decision session (`decision_date > A`).

Thus an outcome that matures on `A` can influence the next decision session but never a decision before or on `A`.

## 1M / 3M / 6M evidence signals

Lookbacks are fixed at:

- 1M = 21 sessions
- 3M = 63 sessions
- 6M = 126 sessions

For each window the suite reproduces the retrospective shadow-candidate logic:

`shadow candidate = top_fraction relative candidate AND score < original threshold`

A window emits `signal = 1` only when all of the following are true:

1. enough matured decision days are present (at least two-thirds of the nominal window),
2. at least 30 shadow candidates are available,
3. activation has collapsed relative to the historical WF crossing rate,
4. the fully matured shadow candidates have positive mean realized net excess.

Otherwise the signal is `0` when evidence is sufficient, or unavailable when the maturity/sample gate fails.

Unavailable windows have their weight set to zero and the remaining profile weights are renormalized. This is a deterministic missing-evidence rule, not optimization.

## Adaptation trigger

For profile `P`:

`weighted_evidence = sum(effective_weight_window * signal_window)`

Adaptation is active when:

`weighted_evidence >= 0.50`

This makes the dominant window of each profile independently capable of triggering adaptation:

- 1M alone can trigger `W_RECENT`,
- 3M alone can trigger `W_MID`,
- 6M alone can trigger `W_LONG`.

## Replacement-threshold contract

The suite does **not** search outcome data for a profitable replacement threshold.

Instead it performs a score-scale correction. For each lookback it calculates the median daily score `p99` using only scores known before the assessment. The same profile weights combine the 1M/3M/6M p99 anchors.

The target pressure is frozen from historical WF evidence:

`historical_threshold_over_p99 = median historical threshold / p99 relationship`

The candidate replacement is then:

`replacement_threshold = historical_threshold_over_p99 * weighted_recent_p99_anchor`

The effective threshold for a later decision date is:

`effective_threshold = min(original_daily_threshold, replacement_threshold)`

Therefore adaptation can only make the gate less restrictive. It can never raise the threshold above the original causal threshold.

## Portfolio replay

For each of 10 frozen Top-10 models, replay all three profiles:

- 30 causal portfolio arms total,
- same existing causal entry predictions,
- same existing causal exit E1-E20 predictions,
- same current replay engine,
- same frozen policy semantics (`top_fraction`, `max_names`, holding/exit family),
- same next-session execution, costs, taxes and learned-exit coverage gates.

The only changed input is `resolved_threshold_by_date`.

## Outputs

`artifacts/top10-causal-threshold-adaptation/`

- `causal_threshold_adaptation_schedule.csv`
- `causal_threshold_adaptation_schedule.parquet`
- `causal_threshold_daily_thresholds.csv`
- `causal_threshold_daily_thresholds.parquet`
- `causal_threshold_adaptation_results.csv`
- `causal_threshold_adaptation_trades.csv`
- `causal_threshold_adaptation_curves.parquet`
- `causal_threshold_adaptation_summary.json`
- `REPORT.md`

## Interpretation

The three profiles are predeclared parallel tests. A stronger result for one profile is evidence about time-scale sensitivity, but is not by itself authorization to select that profile for production. Any production promotion requires a separately defined rule or untouched validation period.

Final Holdout remains closed throughout.
