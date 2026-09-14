> **Scientific interpretation update — 2026-08-27:** This suite is a valid matched OLD-vs-NEW mechanism diagnostic **conditional on the fixed horizon Recipe**, but those Recipes were selected using broader Development evidence overlapping the analyzed era. It therefore does not prove which Recipe/Generation a historical live orchestrator would have selected. The next causal test is the Model Store + Evidence Store + Orchestrator replay in [../../../research/CAUSAL_MODEL_STORE_ORCHESTRATOR_PLAN.md](../../../research/CAUSAL_MODEL_STORE_ORCHESTRATOR_PLAN.md).

# Dynamic-QBD Fold-Event Counterfactual Suite

## Purpose

This suite diagnoses why the Fold-Clock surface-validation arm improves median return
versus a forever-frozen model but fails the predeclared temporal and drawdown
robustness gates.

It consumes the completed Fold-Clock Surface Validation 2016-2025 and reuses its already
fitted entry models, calibrations, authoritative F signals, prices, and
checkpoints. It does not retrain the 1,936 source entry models.

For every non-initial Fold-Clock event the suite creates one matched
counterfactual branch:

OLD_KEEP_INCUMBENT_ONE_MORE_FOLD_INTERVAL

- Keep the incumbent model.
- Keep the incumbent calibration threshold/top fraction.
- Use the same fixed entry recipe.
- Continue only until the next Fold-Clock event.

NEW_ACTIVATE_FOLD_CLOCK_REFIT

- Activate the actual new Fold-Clock fit.
- Activate the new generation calibration.
- Use the same fixed entry recipe.
- This NEW branch becomes the canonical state used at the next event.

Both branches start from the exact same pre-event portfolio state.

## Why FIXED exits are primary

The source Fold-Clock surface validation also contains LEARNED_EXIT families, but the
counterfactual OLD model can create entries that never existed under A/F/M.
The previously generated learned-exit provider therefore does not necessarily
contain exit predictions for those new positions.

Using that provider would mix the value of an entry refit with an exit-model
coverage artifact. The first counterfactual suite therefore uses all 2,790
FIXED families only:

- H1-H30
- D1-D_H
- N1-N6

Learned Exit can be revisited after the entry-refit event mechanism is
understood.

## Event interval

The initial Fold-Clock fit is not an event comparison because there is no prior
incumbent.

Every later real fold expansion is an event.

The comparison interval runs after the event cutoff through the final market
session before the next Fold-Clock event. The old model is scored on exactly
the same post-cutoff decision dates as the new model so a one-session model
deployment gap is not mistaken for model value.

Pending orders already created before the event are inherited identically by
OLD and NEW.

At the next event, NEW is the canonical history.

## Diagnostics

Activation-safe, ex-ante fields include:

- incumbent_model_age_days
- fold_count_increment
- threshold_delta
- threshold_pct_delta
- calibration_observation_delta
- train_start_shift_days
- train_end_shift_days

Post-event fields include:

- post_event_score_spearman
- post_event_daily_top_overlap_median
- post_event_threshold_pass_jaccard_median

Post-event fields use behavior from the future event interval. They are
diagnostic only and are explicitly forbidden as future activation inputs.

The suite computes Spearman associations but does not select a threshold,
classifier, activation gate, model, or feature combination.

## Economic outputs per event and family

For every event x FIXED family:

- OLD and NEW terminal value
- OLD and NEW interval return
- OLD and NEW excess return
- OLD and NEW relative MaxDD
- OLD and NEW trades
- OLD and NEW transaction costs
- NEW minus OLD deltas
- branch-start state fingerprint

The suite then aggregates by:

- event
- structural H/D plateau
- horizon
- event year

## Event diagnoses

RETURN_AND_RISK_HELPFUL

- median NEW minus OLD excess return > 0
- median relative-MaxDD delta >= 0

RETURN_HELPFUL_RISK_WORSE

- return improves
- drawdown worsens

RETURN_WORSE_RISK_HELPFUL

- return worsens
- drawdown improves

RETURN_AND_RISK_HARMFUL

- both return and drawdown worsen

These are diagnostic labels only.

## Resource contract

The runner uses the current proven machine architecture:

- 96 GB system RAM
- 90 GiB aggregate Windows Job Object hard ceiling
- 84 GiB soft target
- 80% CPU capacity target
- on a 32-logical-CPU host: 26 selected logical processors
- default 24 isolated Horizon processes
- one logical processor pinned per active Horizon task
- one native numerical thread per process
- one family lane per Horizon process
- Horizon-level checkpoints
- OLD signal cache per event
- source surface-validation entry models are reused, not refit

NWinfo remains diagnostic-only and fail-open. The runner now accepts an exact
NWinfo executable path because the previous run showed that the installed
binary was not visible through PATH.

## Required source artifacts

The source root must be the original local output directory of the completed
Fold-Clock Surface Validation 2016-2025, not only the compact GitHub result folder.

Required at minimum:

- summary.json
- run-contract.json
- entry-fit-audit.parquet
- inputs/prices.parquet
- authoritative-signals/
- entry-models/

The signal panel must be the exact panel used by that run. Its SHA256 is
validated against the source run contract.

## Self-test

Run locally from the repository root:

    python -m stock_predictor.backtests.opportunity_portfolio_research.dynamic_qbd_fold_event_counterfactual_self_test

Expected terminal marker:

    DYNAMIC_QBD_FOLD_EVENT_COUNTERFACTUAL_SELF_TEST_PASS

The implementation workflow does not execute this self-test automatically.

## Local counterfactual execution

PowerShell example:

    python -m stock_predictor.backtests.opportunity_portfolio_research.dynamic_qbd_fold_event_counterfactual `
      --source-surface-validation-root "D:\PATH\TO\dynamic-qbd-fold-clock-surface-validation-2016-2025-20260826" `
      --signal-panel "D:\PATH\TO\signal-panel.parquet" `
      --output-root "D:\PATH\TO\dynamic-qbd-fold-event-counterfactual" `
      --code-commit "<FINAL_IMPLEMENTATION_SHA>" `
      --horizon-workers 24 `
      --nwinfo-executable "C:\PATH\TO\nwinfo.exe" `
      --nwinfo-interval-seconds 60

If NWinfo is already available on PATH, omit --nwinfo-executable. Alternatively
set the NWINFO_EXE environment variable.

Do not point --source-surface-validation-root at the compact GitHub artifacts directory
unless it also contains the large local source artifacts listed above.

## Compact results to push

- summary.json
- REPORT.md
- contract-audit.json
- source-contract-audit.json
- event-ledger.csv
- event-summary.csv
- event-plateau-summary.csv
- horizon-summary.csv
- year-summary.csv
- risk-problem-events.csv
- feature-associations.json
- run-contract.json
- nwinfo-summary.json

Keep local:

- event-family-counterfactual.parquet
- old-counterfactual-signals/
- checkpoints/
- nwinfo-sensors.jsonl

## Interpretation

The first question is whether harmful Fold-Clock events are concentrated by
Horizon, time, or a simple activation-safe shift metric.

If a simple ex-ante characteristic is stable across events and explains a
large part of harmful refreshes, that becomes a candidate for a separately
predeclared activation-gate experiment.

If the ex-ante associations are weak while post-event score/rank disruption is
strong, do not train another selector. The next decomposition should instead
test what changes inside the refit: training-window composition, parameter
stability, calibration shift, and model-family sensitivity.

No result from this suite opens the final holdout or grants promotion/capital
authority.
