# Top-10 Retrospective Threshold Validation

Contract: `TOP10_RETROSPECTIVE_THRESHOLD_VALIDATION_V1`

## Purpose

Determine whether the entry thresholds used by the causal expanding-live Top-10 system were retrospectively too restrictive, without pretending that future outcomes were available in live operation.

This suite is **backward-looking only**. It is a diagnostic/audit, not a new trading policy.

## Causality rule

For every historical assessment date `A`, a prediction row may be evaluated only when its complete H-session outcome is already observable:

`terminal_date <= A`

Rows whose H11/H24/H28 outcome has not yet matured at `A` are invisible to that assessment.

The suite therefore reconstructs the information set that would actually have existed at each historical date. Future realized returns never influence an earlier assessment.

## Frozen inputs

The suite only reads:

- completed causal-expanding entry predictions;
- the frozen Top-10 policy manifest;
- the completed Entry Activation and Alpha Diagnostic for the historical WF crossing-rate baseline;
- existing daily market prices for retrospective outcome measurement.

It does not:

- fit or update a model;
- regenerate a prediction;
- open the Final Holdout;
- optimize `score_quantile`, `top_fraction`, `max_names`, H/D/N, or exit policy;
- search for a replacement threshold;
- replay portfolios under alternative thresholds.

## Shadow-candidate definition

For each decision date and frozen activation contract, rank the model scores cross-sectionally.

The frozen `top_fraction` determines the relative candidates that would have been considered if the absolute threshold had not rejected them.

A **shadow rejected candidate** is:

`relative top_fraction candidate AND score < then-used threshold`

Its subsequently realized H-session net excess is used only after that outcome has matured.

This directly tests the relevant counterfactual question without changing the live rule:

> Did the threshold reject names that the model ranked at the top and that later delivered positive excess?

## Assessment windows

Historical assessment anchors are sampled every 21 trading sessions and at the final available decision date.

For each anchor:

- 63 fully evaluable decision sessions: sensitivity only;
- **126 fully evaluable decision sessions: primary view**;
- 252 fully evaluable decision sessions: sensitivity only.

The window ends at the most recent decision whose H-session outcome is fully known at the assessment date. Therefore H24/H28 naturally lag the assessment date.

No window is selected because it produces the best result.

## Metrics

Per activation contract, assessment date, and lookback:

- historical WF crossing-day rate;
- recent matured crossing-day rate;
- activation ratio vs historical;
- median threshold / daily p99 score;
- relative-candidate count;
- accepted relative-candidate count;
- rejected shadow-candidate count;
- shadow mean realized net excess;
- shadow median realized net excess;
- shadow hit rate;
- accepted mean realized net excess;
- all-relative-candidate mean realized net excess.

Realized net excess uses the same diagnostic contract as the preceding Entry Activation/Alpha audit: next-session open to H-session close, relative to URTH, minus 20 bps round-trip costs.

## Heuristic verdicts

`RETROSPECTIVE_TOO_RESTRICTIVE_EVIDENCE`

- historical crossing rate was at least 1%;
- recent crossing rate collapsed below `max(0.5%, 25% of historical)`;
- at least 42 matured decision days and 30 shadow candidates are available;
- rejected shadow candidates have positive mean realized net excess.

`RARE_SIGNAL_WITHOUT_POSITIVE_SHADOW_ALPHA`

- activation collapsed, but rejected top-ranked names did not have positive mean realized net excess.

`NO_RETROSPECTIVE_RESTRICTION_EVIDENCE`

- recent activation did not satisfy the collapse definition.

`INSUFFICIENT_MATURED_EVIDENCE`

- not enough fully matured decisions/shadow candidates exist yet.

All verdicts are diagnostic heuristics, not promotion gates and not threshold-change instructions.

## Outputs

- `retrospective_threshold_windows.csv`
- `retrospective_threshold_windows.parquet`
- `latest_primary_126_session_assessment.csv`
- `retrospective_threshold_summary.json`
- `REPORT.md`

## Live implication

This test can justify a later **calibration-review process**, but it cannot choose the future threshold. Any future live recalibration rule must be separately specified before evaluation and must use only information available at the live decision time.
