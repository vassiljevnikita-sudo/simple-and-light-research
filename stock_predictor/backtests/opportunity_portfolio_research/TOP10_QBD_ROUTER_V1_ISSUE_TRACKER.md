# TOP10 QbD Router V1 — GitHub Work Tracker

Parent epic: #28  
Architecture branch: `agent/top10-qbd-router-v1-architecture`  
Baseline: `9d387eca9f6f119ba226b9fb49224b7ae143a794`

The canonical detailed implementation is `TOP10_QBD_ROUTER_V1_IMPLEMENTATION_PLAN.md`. This file maps each implementation step to its GitHub issue.

## Ordered work items

- [ ] #29 — Step 00: Freeze router research contract and policy
- [ ] #30 — Step 01: Build immutable expert registry
- [ ] #31 — Step 02: Build permanent shadow engine and ledger
- [ ] #32 — Step 03: Build expert health store
- [ ] #33 — Step 04: Implement opportunity-time activity monitor
- [ ] #34 — Step 05: Implement alpha health and conservative uncertainty
- [ ] #35 — Step 06: Implement regime-conditioned health with shrinkage
- [ ] #36 — Step 07: Implement alpha decay and change-point detection
- [ ] #37 — Step 08: Implement decomposed kill switches
- [ ] #38 — Step 09: Build superior candidate set
- [ ] #39 — Step 10: Implement router lifecycle state machine
- [ ] #40 — Step 11: Implement equal-weight, soft DMA and hard-champion allocation
- [ ] #41 — Step 12: Make MSCI and Cash first-class meta-experts
- [ ] #42 — Step 13: Implement persistent router state store
- [ ] #43 — Step 14: Build single causal prequential replay engine
- [ ] #44 — Step 15: Implement A-K router ablation arms
- [ ] #45 — Step 16: Build synthetic oracle and null validation suite
- [ ] #46 — Step 17: Run historical pseudo-OOS evaluation and reporting
- [ ] #47 — Step 18: Add multiple-testing diagnostics and cost stress
- [ ] #48 — Step 19: Freeze manifest and Final Holdout gate
- [ ] #49 — Step 20: Execute final independent adaptive holdout

## Dependency graph

```text
#29 Contract
 ↓
#30 Registry
 ↓
#31 Shadow Ledger
 ↓
#32 Health Store
 ├─ #33 Activity
 ├─ #34 Alpha/Uncertainty
 ├─ #35 Regime
 └─ #36 Change Point
       ↓
#37 Kill Switches
 ↓
#38 Candidate Set
 ↓
#39 Lifecycle
 ↓
#40 Allocation/Promotion
 ↓
#41 MSCI/Cash
 ↓
#42 Persistent State
 ↓
#43 Prequential Engine
 ↓
#44 A-K Arms
 ↓
#45 Synthetic + Null Validation
 ↓
#46 Historical Pseudo-OOS
 ↓
#47 Multiple Testing + Cost Stress
 ↓
#48 Freeze Manifest
 ↓
#49 Final Holdout
```

## Parallelization allowed

After #32 establishes the health-store interfaces, #33, #34, #35 and #36 can be implemented in parallel branches/worktrees provided each targets the same frozen contract and no shared implementation file is independently forked.

Synthetic scenario definitions in #45 may be drafted earlier, but acceptance tests for the full router depend on #43/#44.

## Mandatory gates before proceeding

### Before #38 Candidate Set

- matured-evidence invariant passes;
- activity/alpha/regime/change-point outputs are causal and deterministic.

### Before #43 Prequential Engine is accepted

- lifecycle and allocation functions are pure/testable;
- switch-cost/hysteresis rules are implemented;
- benchmark/Cash fallback exists;
- restart-state schema is defined.

### Before #46 historical pseudo-OOS conclusions

- future-mutation test passes;
- A-K parity test passes;
- synthetic known-truth and null scenarios have been executed;
- router decisions contain a complete evidence trace.

### Before #48 freeze

- attempted variants inventory is complete;
- cost stress is complete;
- multiple-testing diagnostics are generated;
- no unresolved causal/restart invariant failure remains.

### Before #49 Final Holdout

- `QBD_ROUTER_V1_FROZEN_MANIFEST.json` exists;
- manifest validation passes against code/config/artifacts;
- Final Holdout runner exposes no tuning path;
- all adaptation rules are frozen.

## Completion rule

Closing a step issue requires both implementation and the relevant tests from `TOP10_QBD_ROUTER_V1_TEST_CONTRACT.md`. Code-only completion is not sufficient.
