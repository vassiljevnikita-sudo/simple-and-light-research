# TOBECONTINUED.md

This public mirror tracks the current Dynamic-QBD research state. Detailed private-machine runtime notes and operational incident history are intentionally excluded.

The documents linked under [Authority map](#authority-map) are the authoritative
sources for this program. Those not yet transferred to the public mirror are listed
in [PUBLIC_SCOPE.md](PUBLIC_SCOPE.md) and must be read in the private repository.

## Current focus

- [x] Canonical naming contract: `research/DYNAMIC_QBD_NAMING.md`.
- [x] Causal pseudo-live Model Store + Evidence Store + Orchestrator architecture is the active research direction.
- [x] Existing Development research remains research/shadow only.
- [x] Prospective holdout beginning 2026-07-25 remains closed.
- [x] No live-order, capital, or silent-promotion authority exists.
- [ ] Complete local causal pseudo-live Development evaluation under the current information contract.
- [ ] Publish only compact, auditable result artifacts after local runtime acceptance.
- [ ] Reassess whether a more complex orchestrator is justified only after the causal baseline exists.
- [ ] Before promotion-grade claims, explicitly bound point-in-time universe, corporate-action, and historical-listing limitations.

## Public mirror completion

- [x] Sanitation contract defined and published: `PUBLIC_SCOPE.md`.
- [x] Published tree verified against the known problem classes (no matches).
- [x] History-free start from source snapshot `c196dba4`; private history not imported.
- [x] Root documentation, package markers and `contract_fingerprints.py` transferred.
- [x] Research documents under `research/` transferred (18 files).
- [x] `opportunity_portfolio_research` implementation and self-tests transferred (241 files).
- [x] Compact artifacts under `artifacts/` transferred (64 evidence files).
- [x] `docs/architecture/`, `docs/external-validation/` and the protected-identity
      files under `stock_predictor/` transferred.
- [x] `CURRENT_MODEL.md` sizing and named-broker details reviewed and approved for
      publication as research configuration parameters (see `QA_TOOL_FAILURE_LOG.md` P8).
- [ ] Decide whether the mirror's own commit metadata should be corrected
      (see `QA_TOOL_FAILURE_LOG.md` P9) — requires rewriting published history.

The documented research scope is now fully mirrored. Everything still absent is
excluded by contract, not pending; `PUBLIC_SCOPE.md` lists it. Large artifact
stores, private runtime incident history and all business/customer automation
remain out of scope by design.

## Authority map

- Naming: `research/DYNAMIC_QBD_NAMING.md`
- Current state: `research/DYNAMIC_QBD_CURRENT_STATE.md`
- Results index: `research/DYNAMIC_QBD_RESULTS_INDEX.md`
- Data/runtime context: `research/DYNAMIC_QBD_DATA_AND_RUNTIME.md`
- Causal architecture: `research/CAUSAL_MODEL_STORE_ORCHESTRATOR_PLAN.md`
- Execution minimum baseline: `research/DYNAMIC_QBD_EXECUTION_MINIMUM_BASELINE.md`

## Safety state

- Prospective final holdout remains closed.
- Dynamic-QBD promotion authority remains disabled.
- Dynamic-QBD capital authority remains disabled.
- Live broker writes/orders remain outside this research program.
- No future information may enter model, calibration, routing, or orchestrator decisions.
