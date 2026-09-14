# Candidate-OOS / QBD Runtime Contract

The QBD coordinator owns scheduling, while the V5 model layer remains the
authoritative Ridge/HGB implementation. `candidate_oos.py` adapts the
point-in-time signal panel into the row contract consumed by
`stock_predictor.v5.train.build_bundle`; it does not implement a second model.

## Lifecycle

```text
candidate registry
  -> candidate_oos_fold(H, candidate, fold)
  -> immutable CandidateOosStore partition
  -> EvidenceSnapshot(H, selection_cutoff)
  -> SelectedRecipe artifact
  -> production generation
  -> prediction/calibration
  -> generation_ready
  -> H/D/N replay
```

Candidate-OOS observations are immutable and content-addressed by candidate
spec, fold spec, signal-panel hash, feature-schema hash and model artifact
hash. A historical observation contains `decision_date`, `terminal_date` and
`information_available_at`; it does not contain a selection cutoff. A snapshot
is the only object allowed to expose matured evidence to recipe selection.

The maturity invariant is:

```text
decision_date < terminal_date <= information_available_at
```

The full-run contract contains the candidate registry hash and the semantic
input hashes. Candidate-OOS partition and snapshot hashes are derived
artifacts and are not circular inputs to the initial contract. Missing
Candidate-OOS is therefore a schedulable dependency, not a global external
blocker.

Candidate-OOS evidence is model evidence only. Stock execution, distributions,
tax and cost contracts belong to the downstream portfolio replay. A missing
portfolio input may block replay while leaving valid Candidate-OOS evidence and
recipe selection available.

`qbd_contracts.py` is the shared authority for `FoldPolicy`, `TargetContract`
and the frozen Ridge/HGB Primary Candidate Contract. Their hashes are carried
through the run contract, Candidate-OOS manifests, snapshots, selected recipes
and production-generation manifests. Benchmark forward-return columns are
required; a missing benchmark is a hard error rather than a zero-return
fallback.

The production job state is SQLite with WAL, transactional claiming, leases,
heartbeats and explicit stale-worker recovery. `jobs.json` is only a human
inspection projection and is never the authoritative state store.

`fit_production_generation()` always creates a fresh V5 bundle from the
selected recipe.  It writes `model.joblib`, calibration predictions and a
`generation-manifest.json` containing the source-panel, feature-schema,
model, calibration and recipe hashes.  Reuse is permitted only when the
manifest and every referenced artifact hash agree.  Candidate-OOS models are
never used as trading generations.

The full coordinator exposes two resumable execution phases: candidate OOS
materialization and causal recipe/production jobs.  Portfolio replay is a
separate phase and reports an explicit stock-input blocker if execution prices
or distributions are not supplied; it is never downgraded to a benchmark-only
claim.
