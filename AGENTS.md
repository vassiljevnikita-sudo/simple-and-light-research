# AGENTS.md

This is the canonical repository-specific instruction file for coding and research agents working in the `simple-and-light` research program.

> **You are in the public mirror.** `simple-and-light-research` is the curated
> public export of the private `simple-and-light` repository, which remains the
> internal source of truth. The published file tree is sanitized and the private
> source history is not imported. Public-mirror publication blockers, including
> commit-metadata sanitation, are tracked in `TOBECONTINUED.md` and
> `QA_TOOL_FAILURE_LOG.md`; do not call the mirror fully sanitized while such an
> item is open. Every document the start sequence below requires is present here.
>
> Before acting, read [PUBLIC_SCOPE.md](PUBLIC_SCOPE.md): it defines what may ever be
> published to this repository. Material it excludes — large artifact stores, private
> runtime incident history, operational and business automation — is absent by
> contract, not by oversight. An absent document is **not** evidence that the
> contract it defines does not exist; do not reconstruct, guess at, or silently
> replace one you cannot read, and never publish excluded material here.

The repository contains multiple historical research tracks. Do not assume the oldest root model documentation describes the active task. Resolve the current branch, read the current tracker, and identify the research track before changing code.

## Mandatory start sequence

Before making changes:

1. Resolve the actual repository, current branch and current HEAD. Do not infer the Development head from a PR title, branch name, old handoff or previous conversation.
2. Read [TOBECONTINUED.md](TOBECONTINUED.md). This is mandatory for **every agent**.
3. Read [DOCUMENTATION_AUTHORITY.md](DOCUMENTATION_AUTHORITY.md).
4. Read [research/DYNAMIC_QBD_NAMING.md](research/DYNAMIC_QBD_NAMING.md) before naming or renaming QBD modules, runs or artifacts.
5. For Dynamic-QBD / Opportunity-Portfolio work, read:
   - [research/DYNAMIC_QBD_CURRENT_STATE.md](research/DYNAMIC_QBD_CURRENT_STATE.md)
   - [research/DYNAMIC_QBD_RESULTS_INDEX.md](research/DYNAMIC_QBD_RESULTS_INDEX.md)
   - [research/DYNAMIC_QBD_DATA_AND_RUNTIME.md](research/DYNAMIC_QBD_DATA_AND_RUNTIME.md)
   - [research/DYNAMIC_QBD_EXECUTION_MINIMUM_BASELINE.md](research/DYNAMIC_QBD_EXECUTION_MINIMUM_BASELINE.md) before changing any execution/runtime code;
   - the detailed plan linked by the relevant open item in `TOBECONTINUED.md`.
6. For execution/runtime/resume/hotstart/provenance work, read [QA_TOOL_FAILURE_LOG.md](QA_TOOL_FAILURE_LOG.md) before editing so known operational failures and reusable-state decisions are not rediscovered.
7. Inspect the exact existing result artifacts relevant to the proposed experiment before proposing or implementing another research path.
8. Inspect the current implementation files, not only documentation. Documentation can lag code; exact committed result artifacts and contracts are evidence for completed runs.
9. Only then plan or edit.

In the public mirror, steps 4-7 are all satisfiable: the research documents, the
architecture docs and the compact run artifacts they refer to are present. What is
not present is listed in [PUBLIC_SCOPE.md](PUBLIC_SCOPE.md) under
`Deliberately not transferred` — chiefly the bulk artifact stores, so a step-7
inspection of a *large* run's raw state must happen in the private repository.
Proceeding without a mandatory contract, or inferring its content, is a failure
mode — not a shortcut.

If the task is only a question, investigation or result review, treat it as read-only unless the user explicitly asks for changes.

## Repository-specific failure modes to avoid

These are not hypothetical style preferences. They have already caused confusion or wasted work in this project.

1. **Wrong Development head.** Agents have oriented from an old branch/PR/doc instead of the actual current QBD head. Always resolve branch + SHA first.
2. **Result blindness.** Several completed experiments were missed because the agent read only the latest report or a few familiar artifact folders. Check the Results Index and the relevant exact artifact before proposing a mechanism.
3. **Ex-post winner conditioning.** A Recipe selected with broad Development evidence was later analyzed retroactively as if it had been known historically. Treat such results as conditional diagnostics, not live-selection proof.
4. **Single-cell generalization.** H3 Fold-Clock looked excellent and failed to generalize robustly over the full 5,400-family space. Never promote a mechanism from one H/D/N cell without the intended breadth test.
5. **Implementation/run/science conflation.** A runner can be implemented, COMPLETE and contract-PASS while the scientific gate FAILS. Report these states separately.
6. **Compact-artifact confusion.** GitHub may contain only a summary while the model/prediction/checkpoint state required by the next experiment remains local. Verify source artifacts before designing reuse.
7. **Memory-naive aggregation.** A giant final matured-prediction concat already failed. Do not restore monolithic pandas materialization for billion-row flows.
8. **Stale hardware assumptions.** Older docs describe a 5800X/32-GB machine and older worker layouts. Current large-run limits live in the data/runtime SSOT.
9. **Unasked heavy execution.** Expensive research runs are normally launched by the user locally. Do not start one merely to “verify” a documentation or implementation task.
10. **Repeating exhausted selector ideas.** Regime, persistence, changepoint, consensus, threshold/memory/cohort controllers and fitted selector pools have existing evidence. A new attempt needs a materially new information contract, not a renamed implementation.
11. **Execution-history amnesia.** The v1 -> v40.0.4.3 runtime lineage accumulated specific failures that must not recur. `v40.0.4.3` is the minimum acceptable baseline, not a ban on better architecture. A redesign may replace its mechanisms only if it preserves the failure-derived minimum properties and states which historical defect each replacement continues to prevent. Read [research/DYNAMIC_QBD_EXECUTION_MINIMUM_BASELINE.md](research/DYNAMIC_QBD_EXECUTION_MINIMUM_BASELINE.md).

12. **Fast-resume derived-state amnesia.** A compatible fast resume once reused a
stale Coverage DAG, repaired the matured set only inside the worker, and then
aborted on `EVIDENCE_SNAPSHOT_IMMUTABLE_CONFLICT` against an older valid
snapshot leaf. Do not grant fast-resume authority after code/graph migration
until one slow reconciliation has compared current job payloads. Preserve
valid old immutable leaves; content-version corrected derived state; invalidate
only the changed node's transitive derived descendants; keep corrupt/partial
artifacts fail-closed.

13. **Runtime-incident amnesia.** A real run can complete substantial valid work and then fail in a non-scientific boundary such as Git provenance, text decoding, serialization, snapshot publication or OS-specific I/O. Do not let the next agent rediscover the same failure or discard reusable state. Material runtime incidents belong in [QA_TOOL_FAILURE_LOG.md](QA_TOOL_FAILURE_LOG.md), including exactly which databases/checkpoints/artifacts remain valid and reusable.

When an agent notices a new recurring failure mode, update this section or the appropriate focused instruction instead of relying on future agents to rediscover it.

## Mandatory QA incident logging

[QA_TOOL_FAILURE_LOG.md](QA_TOOL_FAILURE_LOG.md) is the durable operational incident ledger despite its historical filename. It is mandatory context for execution/runtime/resume/hotstart/provenance work.

Record all of the following:

- every failed tool invocation before the next retry;
- every real runtime, initialization, resume, hotstart, provenance, serialization, encoding or OS-specific failure that interrupts a run, consumes material work, risks repeated expensive computation, or establishes a new non-regression rule;
- concrete incidents reported by the user or another agent when they can affect follow-up work. Mark these `REPORTED_UNVERIFIED` until the current branch/code or a reproduced run confirms the correction.

Each incident entry must preserve enough state for the next agent to act correctly:

1. exact context/run stage and elapsed useful work when known;
2. observed error and the distinction between confirmed root cause and hypothesis;
3. classification (`TOOL`, `IMPLEMENTATION`, `RUNTIME`, `EXPECTED_NEGATIVE`, `REPORTED_UNVERIFIED`, or a more specific compatible label);
4. **state impact**: which SQLite databases, checkpoints, graph materializations, caches or artifacts remain valid/reusable, and which are invalidated;
5. correction or containment applied/planned;
6. verification status and evidence still required;
7. the non-regression rule when the incident should constrain future implementation.

Example: if a real initialization run spends about 17 minutes successfully producing three reusable databases and then fails only while publishing Git provenance because a parallel `git diff` reader uses the Windows CP-1252 default and returns `None` after a decode error, log the failure as a provenance/encoding runtime incident. The entry must say that the three databases remain reusable, that Git text reads must use explicit UTF-8 with loss-tolerant replacement, and that the next snapshot should reuse the existing graphs rather than recompute them merely because provenance publication failed.

A material incident may require more than the QA log:

- update `TOBECONTINUED.md` if it changes the current gate/blocker;
- update `research/DYNAMIC_QBD_EXECUTION_MINIMUM_BASELINE.md` if it establishes a durable failure-derived runtime invariant;
- update the relevant focused runtime/architecture document if the implementation contract changes.

Do not promote an unverified report into a proven root cause or a validated fix merely because it appeared in another agent's reasoning.

## TOBECONTINUED.md is the continuity contract

`TOBECONTINUED.md` is intentionally short and operational. Every agent must keep it truthful.

When work changes project state:

- completed items become `[x]`;
- unfinished items remain `[ ]`;
- do not delete an open item merely because a different approach is preferred;
- add newly agreed plans or blockers;
- link detailed design/task documents instead of expanding the tracker into a long specification;
- record supersession explicitly when a plan is replaced;
- do not mark an experiment complete until the evidence/artifact required by the item actually exists;
- do not mark a scientific question solved merely because the implementation or run completed.

A task that materially changes the current plan is not complete until the tracker is updated.

## Two different kinds of authority exist in this repository

Do not conflate them.

### Protected model identity

`stock_predictor/research_model_registry.json` and [CURRENT_MODEL.md](CURRENT_MODEL.md) define the protected V4/N25, execution, V4.5 and V5 research identities.

They remain authoritative for those model IDs and safety constraints. Dynamic-QBD work does not silently replace those IDs.

### Active Dynamic-QBD research state

The current scientific program on the QBD branch is tracked by:

- `TOBECONTINUED.md`;
- `research/DYNAMIC_QBD_CURRENT_STATE.md`;
- `research/DYNAMIC_QBD_RESULTS_INDEX.md`;
- exact committed result artifacts under `artifacts/`.

Do not use the July N25 registry timestamp as evidence that Dynamic-QBD stopped in July. Conversely, do not rename Dynamic-QBD research into a protected V4/V5 model ID without an explicit promotion decision.

## Project glossary

Use these terms consistently.

- **H / horizon** — prediction target horizon in trading sessions, currently H1-H30 in the full QBD surface.
- **D / holding period** — portfolio holding period; normally D1-D_H.
- **N / max names / capacity** — maximum simultaneously held stock names, currently N1-N6. N is a portfolio dimension, not a separate predictive training target.
- **Family** — an economic portfolio configuration identified by H, D, N and exit mode. Do not assume every Family requires a distinct model fit.
- **Recipe** — predictive model family and hyperparameters used to fit one horizon model.
- **Generation** — one immutable fitted model artifact for a Recipe at a specific causal training cutoff.
- **Model Store** — persistent collection of recipes and immutable generations available as of a historical time.
- **Evidence Store** — matured causal OOS predictions/outcomes and lineage available as of a historical time.
- **Fold Clock** — an information clock that advances only when genuinely new matured fold evidence appears; calendar progress alone is not new evidence.
- **Orchestrator** — causal policy that selects an active Recipe/Generation from the Model Store using only information available at that time.
- **Development** — research data available for iterative research. It is not the final prospective holdout.
- **Prospective holdout** — starts 2026-07-25 under the current contract. It remains closed unless an explicitly approved final evaluation opens it.
- **Shadow** — research inference/decision with no capital authority.
- **Authority** — permission to influence capital, promotion or production. A positive research metric does not grant authority.
- **Artifact** — persisted run evidence. A compact Git artifact may summarize a much larger local run.

## Current Dynamic-QBD scale

**Full Run is a reserved term.** It means the complete 2016-2026 temporal dataset and is currently `RESERVED_NOT_RUN`. Development-only, surface-wide or completed runs must not use that name. See `research/DYNAMIC_QBD_NAMING.md`.


Agents must understand the scale before changing aggregation, caching or scheduling.

Current historical limits:

- market history begins around 2016-01-01;
- benchmark data begins 2016-01-04;
- canonical signal panel begins 2016-06-24 after feature warm-up;
- current Development ends 2025-12-31;
- prospective holdout begins 2026-07-25.

Relevant completed runs include:

- 2,790 FIXED families;
- 5,400 FIXED + LEARNED_EXIT families in the Fold-Clock Surface Validation 2016-2025;
- 119 ABC assessment months;
- 55 FIXED structural plateaus / 110 by exit mode;
- the completed Development Run 2016-2025 with **1,632,350,262 matured prediction rows** and **179,376 compact evidence rows**.

Do not introduce a design that materializes all matured predictions in one pandas DataFrame. This already caused an ArrayMemoryError and was repaired with streamed/cache-aware aggregation.

The current 96-GiB host runtime contract for large Dynamic-QBD runs is documented in [research/DYNAMIC_QBD_DATA_AND_RUNTIME.md](research/DYNAMIC_QBD_DATA_AND_RUNTIME.md). Do not copy old Ryzen-5800X/32-GB or 12-worker instructions from historical docs into a new heavy runner.

## Research history is a guardrail, not optional reading

Before proposing another selector or controller, check [research/DYNAMIC_QBD_RESULTS_INDEX.md](research/DYNAMIC_QBD_RESULTS_INDEX.md).

The following broad ideas have already been tested in one or more forms and must not be casually reintroduced as a new solution:

- trailing performance / family momentum;
- threshold adaptation;
- monthly memory weighting;
- cohort memory;
- model-specific memory controllers;
- adaptation gates;
- regime routing;
- changepoint routing;
- opportunity-state routing;
- cross-horizon consensus;
- additional fitted selector pools;
- recipe-switch hysteresis;
- calendar-month refitting;
- unconditional rolling recalibration.

A new version is justified only if it addresses a documented flaw or changes the information contract in a material, predeclared way. Explain that distinction.

## Important scientific correction

The Fold-Clock Surface Validation 2016-2025 and Fold-Event Counterfactual suites are useful diagnostics, but their fixed Recipes were selected using the broader Development evidence that overlaps the period being analyzed.

Therefore they answer conditional/mechanistic questions such as:

> Given this Development-selected Recipe, what happens under a different refresh schedule?

They do **not** prove:

> A causal historical orchestrator would have known to select this Recipe at that time.

Do not use those runs to justify ex-ante HGB-vs-Ridge, horizon or generation selection without a causal Model-Store replay.

The active next architecture is a pseudo-live Model Store + Evidence Store + Orchestrator simulation. See [research/CAUSAL_MODEL_STORE_ORCHESTRATOR_PLAN.md](research/CAUSAL_MODEL_STORE_ORCHESTRATOR_PLAN.md).

## Data and causal rules

- No target/outcome may enter training, calibration, model selection or orchestration before its terminal date has matured.
- Different H values mature at different times.
- Never equalize horizons by borrowing future outcomes from longer horizons.
- Random row splits are not acceptable for time-dependent model-selection claims.
- A model generated at t may use only inputs and matured outcomes available at t.
- Old generations remain distinct; do not overwrite history.
- A refit may reuse earlier training observations in an expanding window. That creates correlated generations; do not treat them as independent evidence.
- Never choose a historical seed cutoff, Recipe, gate threshold or orchestrator rule after inspecting the same pseudo-live performance and then call it OOS.
- The exact final prospective holdout remains closed during iterative architecture work.

## Result discipline

For a completed experiment:

1. Read the exact `summary.json` / `REPORT.md` and contract audit at the result commit.
2. Verify the result commit and the code/run SHA are distinct and correctly referenced when applicable.
3. Distinguish:
   - implementation complete;
   - run complete;
   - contract audit pass;
   - scientific gate pass;
   - promotion/capital authority.
4. Preserve negative and superseded results.
5. If an old artifact has a provenance warning, repeat that warning when using its numbers.
6. Do not summarize a large run from filenames or a diff snippet alone.
7. After publishing a meaningful new result, update the Results Index and `TOBECONTINUED.md`.

## Code and artifact locations

For Dynamic-QBD work:

- `stock_predictor/backtests/opportunity_portfolio_research/` — implementation.
- `research/` — current research handoffs and detailed plans.
- `docs/architecture/` — implementation/runtime contracts.
- `artifacts/` — committed compact result evidence.
- large local model/prediction/NAV/checkpoint stores are not expected to be committed.

Prefer linking an exact detailed doc over duplicating a long contract in multiple files.

## Heavy-run policy

Do not automatically run large backtests, full model factories, or full repository test suites.

For this project, the user normally runs expensive suites locally. Unless explicitly asked to run them:

- implement the runner and deterministic self-test if needed;
- review code statically;
- provide the exact local command;
- do not claim a self-test or real-data research run passed unless it was actually executed;
- do not delete or invalidate local caches/checkpoints;
- keep resume compatibility where the research contract is unchanged.

If the user asks only to inspect results, do not start a new run.

## Verification

Match verification ceremony to the task.

- Documentation-only work: verify links, names, branch/HEAD and internal consistency. Do not launch unrelated model tests.
- Small code change: use the narrowest relevant static/self-test path.
- Heavy research runner: only execute if explicitly requested; otherwise supply the command.
- Never equate a zero exit code with scientific validity.
- Never equate fixture/self-test evidence with real-data runtime evidence.

The older `stock_predictor/project_manifest.json` and `agent_cli` are useful for protected N25/control-plane checks, but they do not yet contain the complete Dynamic-QBD research map. Do not rely on `agent_cli context` alone for QBD orientation.

## Safety

Without explicit user authorization and a separately valid contract:

- no live order submission;
- no broker writes/state mutation;
- no model promotion;
- no capital authority;
- no final-holdout opening;
- no rewriting historical evidence to make a result look current;
- no silently changing costs, benchmark, signal timing or execution timing;
- no destructive deletion of historical artifacts or checkpoints.

## Branch and provenance discipline

- Resolve the actual branch and SHA before editing.
- Read current code at that SHA.
- Do not assume `main` is the active Development head.
- Do not merge/rebase unrelated branches merely to make docs look current.
- Keep result commits separate from implementation commits when practical.
- Persist code SHA/run-contract hashes in research artifacts.
- When comparing old and new results, compare the research contract as well as execution code.

## Dynamic-QBD execution minimum baseline

`v40.0.4.3` is the minimum acceptable execution baseline. More efficient architectures are allowed and should be considered when justified, but they may not reintroduce any historical failure captured in the failure ledger.

Every redesign must identify the current mechanism it replaces, the historical defect that mechanism fixed, how the replacement still prevents that defect, the focused regression proving it, and the measurable efficiency/complexity improvement.

The authoritative minimum requirements and `failed because X` history are in [research/DYNAMIC_QBD_EXECUTION_MINIMUM_BASELINE.md](research/DYNAMIC_QBD_EXECUTION_MINIMUM_BASELINE.md).

## Scope control

Fight scope creep. A review comment, old TODO or historical document is not permission to expand the task.

When changing one research mechanism, keep other dimensions frozen unless the experiment explicitly exists to vary them. Prefer matched controls and causal contrasts.

If a historical assumption conflicts with the current task, state the conflict and follow the current user instruction while preserving audit history.

## Communication

Lead with the problem and scientific consequence, not an implementation inventory.

When reporting research results, make clear:

- what was tested;
- what the causal information set was;
- what passed/failed;
- what remains conditional or development-only;
- what this changes about the next step.

Do not call an ex-post diagnostic a live-selection result.
