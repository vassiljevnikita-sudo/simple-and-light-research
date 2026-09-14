# Documentation Authority

> **Public mirror note.** This precedence order describes the full private
> `simple-and-light` tree. Several documents named below have not been transferred
> to this public mirror yet; [PUBLIC_SCOPE.md](PUBLIC_SCOPE.md) lists which. Absence
> here does not lower a document's authority — it means the authoritative copy must
> be read in the private repository.

This repository contains multiple research generations. A file can be technically correct for its historical track and still be the wrong source for the current Dynamic-QBD task.

Use the following authority order.

## 1. Direct user instruction

The current user's explicit instruction overrides repository defaults when the two conflict, subject to safety constraints.

Do not use an old roadmap or agent file to defeat the current task.

## 2. Agent work instructions

Canonical repository-specific agent behavior:

1. [AGENTS.md](AGENTS.md)
2. this file

Compatibility handoff:

- [agent.md](agent.md) points agents into the same current hierarchy.

Do not maintain a separate competing instruction set in `agent.md`.

## 3. Current work tracker

[TOBECONTINUED.md](TOBECONTINUED.md) is the mandatory current work-state tracker.

It answers:

- what is complete;
- what remains open;
- what plan is currently next;
- what blockers still exist.

It is not a detailed design specification. Open items should link detailed documents.

Every agent must read it before starting work and update it when project state changes.

## 4. Dynamic-QBD current scientific state

For current Opportunity-Portfolio / Dynamic-QBD research:

- [research/DYNAMIC_QBD_NAMING.md](research/DYNAMIC_QBD_NAMING.md) — canonical module, run, scope and artifact naming contract.
- [research/DYNAMIC_QBD_CURRENT_STATE.md](research/DYNAMIC_QBD_CURRENT_STATE.md) — current scientific/technical handoff.
- [research/DYNAMIC_QBD_RESULTS_INDEX.md](research/DYNAMIC_QBD_RESULTS_INDEX.md) — chronological experiment map and consequences.
- [research/DYNAMIC_QBD_DATA_AND_RUNTIME.md](research/DYNAMIC_QBD_DATA_AND_RUNTIME.md) — data availability, scale, local-artifact and current runtime constraints.
- [research/DYNAMIC_QBD_EXECUTION_MINIMUM_BASELINE.md](research/DYNAMIC_QBD_EXECUTION_MINIMUM_BASELINE.md) — authoritative `v40.0.4.3` minimum execution baseline, chronological `failed because X` ledger and failure-derived runtime non-regression requirements.
- [research/CAUSAL_MODEL_STORE_ORCHESTRATOR_PLAN.md](research/CAUSAL_MODEL_STORE_ORCHESTRATOR_PLAN.md) — current causal scientific Model-Store/Orchestrator plan; execution improvements remain allowed when they preserve the minimum baseline.

These documents describe the active research program. They do not promote a protected model.

### QA operational incident evidence

[QA_TOOL_FAILURE_LOG.md](QA_TOOL_FAILURE_LOG.md) is the durable operational incident ledger despite its historical filename. For runtime/resume/hotstart/provenance work it must be read alongside the current runtime SSOT. It records both failed tool invocations and material real-run failures, including the state-reuse decision needed to avoid recomputing valid databases/checkpoints after a non-scientific failure.

A QA incident is not by itself scientific result authority. A user/other-agent report remains marked unverified until current-code or reproduced-run evidence confirms it. When an incident proves a durable execution invariant, that invariant should also be promoted into `research/DYNAMIC_QBD_EXECUTION_MINIMUM_BASELINE.md`.

## 5. Exact completed-run evidence

For a numerical claim about a completed experiment, the highest authority is the exact artifact at the result commit:

- `artifacts/<run>/summary.json`;
- `artifacts/<run>/REPORT.md`;
- `contract-audit.json`;
- `run-contract.json`;
- compact result tables/ledgers.

The Results Index is a navigation and interpretation layer. If a number in the index conflicts with the exact committed run artifact, inspect the run artifact and fix the index.

A code implementation commit and a result-publication commit may be different. Preserve both in provenance.

## 6. Protected operational/research model identity

For the protected V4/N25/V4.5/V5 research stack:

1. `stock_predictor/research_model_registry.json` — machine-readable model identity.
2. [CURRENT_MODEL.md](CURRENT_MODEL.md) — human-readable protected model status.
3. exact runner/model contracts referenced by the registry.

The currently protected identities are `V4_N25_T212_RESEARCH_V1`,
`EXEC_N25_DUAL_VENUE_RESEARCH_V1`, `V45_N25_HYBRID_STOP_RESEARCH_V1`, and
`MSCI_WORLD_IMPLEMENTABLE_PROXY_V1`.
The historical comparison arm remains isolated as `V4_FROZEN_10K_LEGACY`.

Dynamic-QBD experiments do not silently replace these identities.

The registry being last updated in July is not a current Dynamic-QBD status marker; it is a protected model-identity record.

## 7. Project/control-plane manifest

`stock_predictor/project_manifest.json` remains authoritative for the control-plane/N25 architecture facts it explicitly defines:

- protected safety defaults;
- allowlisted control-plane tasks;
- N25 architecture paths;
- protected expected IDs.

It is currently **not a complete map of the Aug-2026 Dynamic-QBD research program**.

Therefore:

- `python -m stock_predictor.devtools.agent_cli context` can be useful for protected N25/control-plane work;
- it is not sufficient orientation for Dynamic-QBD work;
- do not overwrite current QBD state with its older N25-centric map.

Extending the machine-readable manifest is an open tracked task, not something to do silently during unrelated research.

## 8. Human project overview

[README.md](README.md) is the human-facing repository map.

It should explain:

- what the repository contains;
- the active research program;
- data/safety boundaries;
- where to find current state and results.

It should not contain detailed agent-behavior rules.

## 9. Package documentation

For code in `stock_predictor/backtests/opportunity_portfolio_research/`, the package README and focused architecture docs describe implementation contracts.

They may contain historical sections for reproducibility. When their status differs from the root current-state documents, use:

- exact current code for implementation behavior;
- exact run artifacts for completed result evidence;
- current-state/tracker docs for what to do next.

## 10. Historical documents

The following classes are preserved as historical evidence and must not automatically steer new work:

- dated audits such as `research/DYNAMIC_QBD_SYSTEM_AUDIT_2026-08-15.md`;
- earlier QBD roadmaps written before the Development Run/Fold-Clock/Counterfactual results;
- N25-focused architecture documents from July;
- experiment-specific plans whose run is already complete;
- superseded artifacts that explicitly carry a provenance warning.

Do not delete them simply because they are old. Add a supersession/current-state pointer when confusion is likely.

## 11. Conflict resolution

When documents disagree:

1. verify the current branch and HEAD;
2. determine whether each document is current-state, protected-identity, historical-plan or exact-run evidence;
3. inspect current code for implementation questions;
4. inspect the exact artifact for numerical result questions;
5. follow `TOBECONTINUED.md` for open work;
6. report the conflict rather than silently choosing the convenient source.

## 12. Updating documentation

When a meaningful research result or plan change occurs:

- update `TOBECONTINUED.md`;
- update `QA_TOOL_FAILURE_LOG.md` for every material operational incident, recording reusable versus invalidated state, correction and verification status;
- update `research/DYNAMIC_QBD_RESULTS_INDEX.md` if a new experiment completed;
- update `research/DYNAMIC_QBD_CURRENT_STATE.md` if the scientific conclusion or next architecture changed;
- update `research/DYNAMIC_QBD_EXECUTION_MINIMUM_BASELINE.md` when a newly proven runtime failure adds a minimum requirement or an explicitly validated redesign supersedes a mechanism without weakening the failure-derived property;
- update the focused detailed plan if its contract changed;
- avoid copying the same large result table into multiple root docs.

The goal is one clear current handoff plus durable exact evidence, not many competing “latest” summaries.
