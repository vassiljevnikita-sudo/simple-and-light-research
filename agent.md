# Agent handoff

This file is a compatibility entry point for agents or harnesses that look for `agent.md`.

The canonical instructions are in [AGENTS.md](AGENTS.md). Do not maintain an independent workflow here.

## Mandatory first reads

Before changing this repository:

1. verify current branch and HEAD;
2. read [TOBECONTINUED.md](TOBECONTINUED.md);
3. read [DOCUMENTATION_AUTHORITY.md](DOCUMENTATION_AUTHORITY.md);
4. read [research/DYNAMIC_QBD_NAMING.md](research/DYNAMIC_QBD_NAMING.md);
5. read [AGENTS.md](AGENTS.md).

For Dynamic-QBD / Opportunity-Portfolio work also read:

- [research/DYNAMIC_QBD_CURRENT_STATE.md](research/DYNAMIC_QBD_CURRENT_STATE.md);
- [research/DYNAMIC_QBD_RESULTS_INDEX.md](research/DYNAMIC_QBD_RESULTS_INDEX.md);
- [research/DYNAMIC_QBD_DATA_AND_RUNTIME.md](research/DYNAMIC_QBD_DATA_AND_RUNTIME.md);
- the detailed plan linked from the relevant open tracker item.

## Current project distinction

The protected V4/N25, V4.5 and V5 identities remain defined by `stock_predictor/research_model_registry.json` and [CURRENT_MODEL.md](CURRENT_MODEL.md).

The active research program on the QBD branch is Dynamic-QBD. It has separate current-state/result documentation and no promotion/capital authority.

Do not mistake the July protected-model registry for the current Dynamic-QBD development head.

## Current Dynamic-QBD direction

The latest completed research includes:

- full 2,790-family Development;
- regime/opportunity/consensus/selector diagnostics;
- matched recalibration/refit experiments;
- full 5,400-family Fold-Clock validation;
- 180-event Fold-Event Counterfactual.

The latest methodological correction is that the fixed Recipes in the Fold-Clock/Counterfactual diagnostics were selected from broader Development evidence overlapping the later evaluation era. Those runs are therefore conditional/mechanistic diagnostics, not proof of historical causal Recipe selection.

The next architecture is a causal Model Store + Evidence Store + Orchestrator pseudo-live replay. See [research/CAUSAL_MODEL_STORE_ORCHESTRATOR_PLAN.md](research/CAUSAL_MODEL_STORE_ORCHESTRATOR_PLAN.md).

## Continuity rule

Every agent must maintain `TOBECONTINUED.md`:

- check off completed work;
- leave unfinished work open;
- add newly agreed plans/blockers;
- link details elsewhere.

Do not finish a state-changing task without updating the tracker.

## Mandatory QA incident logging

[QA_TOOL_FAILURE_LOG.md](QA_TOOL_FAILURE_LOG.md) is the durable operational incident ledger. Its historical filename is retained for compatibility, but its scope is broader than tool errors.

Agents must log:

- failed tool invocations before retry;
- material runtime/initialization/resume/hotstart/provenance/encoding/serialization/OS-specific failures;
- concrete user/other-agent reported incidents that affect follow-up work, marked `REPORTED_UNVERIFIED` until independently confirmed.

Every runtime incident must record the **state impact**: what remains reusable versus what was invalidated, the correction, verification status and any non-regression rule. This prevents a later agent from discarding valid checkpoints or repeating expensive computation.

Example: a ~17-minute initialization may successfully create three reusable databases and then fail only during Git-provenance publication because a Windows CP-1252 decode error makes a parallel `git diff` reader return `None`. The QA entry should preserve those databases as reusable, require explicit UTF-8 plus loss-tolerant replacement for provenance reads, and require the next snapshot to prove reuse instead of rebuilding the graphs.

The complete rule is canonical in [AGENTS.md](AGENTS.md). If the incident adds a durable runtime invariant, also update the execution minimum baseline; if it changes the current gate, update `TOBECONTINUED.md`.

## Heavy runs

Do not automatically execute full research suites or broad test runs. The user normally runs expensive Dynamic-QBD work locally. Implement/review the runner and provide exact commands unless execution was explicitly requested.

## Safety

Research/shadow only unless explicitly changed:

- no live orders;
- no broker writes;
- no silent model promotion;
- no final prospective holdout opening;
- no future leakage;
- no historical result relabelling.
