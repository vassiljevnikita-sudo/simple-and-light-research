# simple-and-light-research



> Public research snapshot of the Dynamic-QBD / Opportunity-Portfolio program. This repository starts from source snapshot `c196dba496203e803917fd4fff9a03d4df9dc3a2` and intentionally excludes private operational material, credentials, customer data, local machine state, and large licensed/local datasets.

> **Scope.** The documented research scope is mirrored here: research documents,
> architecture docs, the Dynamic-QBD implementation and self-tests, and compact run
> evidence. Large artifact stores, private runtime incident history and all
> business/customer automation are excluded by contract.
> [PUBLIC_SCOPE.md](PUBLIC_SCOPE.md) states what is included, what is excluded and why.

`simple-and-light` is a research repository for chronological stock-selection, portfolio and execution experiments. It contains several generations of work. The protected V4/N25, V4.5, V5 and broker contracts remain in the repository, but the active research program on the current QBD branch is the **Opportunity-Portfolio / Dynamic-QBD system**.

This README is the human-facing map. Coding agents should start with [AGENTS.md](AGENTS.md).

## Current research focus

Naming contract: [research/DYNAMIC_QBD_NAMING.md](research/DYNAMIC_QBD_NAMING.md). **Full Run** is reserved for the entire 2016-2026 dataset. No Full Run has been executed under the current closed-holdout contract.


The public mirror is published from `main`. Development branch names from the private source repository are provenance only and are not public-mirror branches. Resolve the current private Development head in the private repository before doing source-development work.

The current Dynamic-QBD program asks how a large family of causal stock-selection models should be trained, refreshed, stored and selected through time without future leakage or repeated hindsight optimization.

The research surface is materially larger than the older Top-10 and H5/H10/H20 experiments:

- prediction horizons `H1-H30`;
- holding periods `D1-D_H`;
- capacity `N1-N6`;
- FIXED and, where valid, LEARNED_EXIT variants;
- 2,790 FIXED families and 5,400 families in the full FIXED + LEARNED_EXIT validation;
- 55 FIXED structural H/D plateaus, 110 when separated by exit mode;
- Development evidence through 2025-12-31;
- the completed **Development Run 2016-2025** contained **1,632,350,262 matured prediction rows** and **179,376 compact evidence rows**.

The final prospective holdout starts at **2026-07-25** and remains closed. Dynamic-QBD has no capital, promotion or live-order authority.

### Execution minimum baseline

The Dynamic-QBD execution suite uses `v40.0.4.3` as the **minimum acceptable
baseline**, not as a permanently frozen final architecture. More efficient
runtime designs may be tested.

The historical value of the predecessor sequence is the concrete failure
record: each stage is documented as **failed because X**, together with the
minimum property required to prevent the same defect from returning.

The authoritative failure ledger and minimum contract are
[research/DYNAMIC_QBD_EXECUTION_MINIMUM_BASELINE.md](research/DYNAMIC_QBD_EXECUTION_MINIMUM_BASELINE.md).
This baseline decision is separate from the still-pending local runtime
validation and from scientific/economic validation.

## Data boundary

The repository does not have a clean pre-2016 historical feature/training archive for this program.

- Raw historical market coverage used by the current research starts around `2016-01-01`.
- Benchmark daily data starts `2016-01-04`.
- The canonical signal panel begins `2016-06-24` after feature warm-up.
- Current Development ends `2025-12-31`.
- Prospective holdout boundary: `2026-07-25`.

This matters for the next architecture. A production system can eventually begin with roughly ten years of 2016-2025 history, but a historical pseudo-live replay cannot honestly pretend that the same ten-year seed existed before 2016.

See [research/DYNAMIC_QBD_DATA_AND_RUNTIME.md](research/DYNAMIC_QBD_DATA_AND_RUNTIME.md).

## What has already been learned

The important conclusions are not stored in one experiment. Read the chronological [Dynamic-QBD Results Index](research/DYNAMIC_QBD_RESULTS_INDEX.md) before proposing another selector, controller or refit rule.

High-level findings:

- The original Prediction × Hold surface contains broad robust regions; one isolated best cell is not the research conclusion.
- Equal-active allocation survived the allocation QbD; score/rank-weighted allocation did not add robust value.
- `IGNORE_NEW` survived the replacement test; `REPLACE_WEAKEST` did not.
- Frozen historical Top-10 winners generalized poorly in the later true-forward replay.
- Historical threshold calibration became too restrictive for several frozen models.
- Several memory, cohort, threshold, regime, changepoint and adaptive-controller ideas were tested; none has received selector authority.
- Full Dynamic-QBD Development completed over 2,790 FIXED families. Gates 1, 1B, 2 and 3 all failed conservatively.
- Regime persistence exists, but was too weak after costs to support routing.
- Opportunity-state predictability was not robust.
- Cross-horizon consensus was suggestive but not robust.
- A paired selector tournament ended with `Score-only` as the strongest robust selector; no tested learned pool cleared the promotion gates.
- Monthly refitting/recalibration is broadly destructive relative to sparse evidence-clocked refitting.
- A single H3 Fold-Clock result looked excellent, but the full 5,400-family validation did **not** justify automatic Fold-Clock promotion: F-vs-A failed temporal and drawdown robustness while monthly-vs-Fold-Clock was strongly negative.
- The Fold-Event Counterfactual run is useful mechanistic evidence, but it conditions on recipes selected from the same broad Development era. It must **not** be treated as proof that an ex-ante live orchestrator would have selected those recipes.

The current scientific correction is therefore important:

> Existing Fold-Clock Surface Validation 2016-2025 and Fold-Event Counterfactual results are conditional/mechanistic diagnostics given a Development-selected recipe. The next valid architecture must build a model store and make recipe/generation decisions only from information available at that historical time.

## Intended next architecture

The research direction is a causal **Model Store + Evidence Store + Orchestrator**:

```text
historical prefix available at t0
        ↓
initial model bank
        ↓
pseudo-live boundary
        ↓
incoming observations and matured outcomes
        ↓
new immutable generations / expanding-window refits
        ↓
model + evidence store
        ↓
causal orchestrator
        ↓
active recipe / generation
        ↓
portfolio
```

Old generations remain available. A newly refitted model is a challenger, not an automatic replacement. The orchestrator may keep an older generation or choose another causally available recipe.

The exact historical seed cutoff is still open because no pre-2016 data exists. Do not silently pick the seed after looking at performance.

Detailed plan: [research/CAUSAL_MODEL_STORE_ORCHESTRATOR_PLAN.md](research/CAUSAL_MODEL_STORE_ORCHESTRATOR_PLAN.md).

## Repository map

The most relevant areas for the current research are:

- `stock_predictor/backtests/opportunity_portfolio_research/` — QbD surfaces, Dynamic-QBD factory/replay/evidence/gates and research experiments.
- `research/` — research architecture, scientific interpretation and current handoffs.
- `docs/architecture/` — implementation/runtime architecture documents.
- `artifacts/` — compact committed results and audit evidence. Large model/prediction/NAV/checkpoint stores usually remain local.
- `stock_predictor/research_model_registry.json` — protected V4/N25/V4.5/V5 model identity. It is **not** the Dynamic-QBD project-status tracker.
- `TOBECONTINUED.md` — mandatory current work tracker.

For the Opportunity-Portfolio package itself, see [stock_predictor/backtests/opportunity_portfolio_research/README.md](stock_predictor/backtests/opportunity_portfolio_research/README.md).

## Documentation entry points

Use these in order for current Dynamic-QBD work:

1. [TOBECONTINUED.md](TOBECONTINUED.md) — what is done, open and planned.
2. [research/DYNAMIC_QBD_CURRENT_STATE.md](research/DYNAMIC_QBD_CURRENT_STATE.md) — current scientific/technical state.
3. [research/DYNAMIC_QBD_RESULTS_INDEX.md](research/DYNAMIC_QBD_RESULTS_INDEX.md) — experiments already run and their consequences.
4. [research/DYNAMIC_QBD_DATA_AND_RUNTIME.md](research/DYNAMIC_QBD_DATA_AND_RUNTIME.md) — data limits, scale and current machine/runtime contract.
5. [QA_TOOL_FAILURE_LOG.md](QA_TOOL_FAILURE_LOG.md) — durable operational QA incident ledger (historical filename) for tool failures and material runtime/resume/provenance failures, including reusable-state decisions.
6. [research/DYNAMIC_QBD_EXECUTION_MINIMUM_BASELINE.md](research/DYNAMIC_QBD_EXECUTION_MINIMUM_BASELINE.md) — v40.0.4.3 minimum execution baseline, `failed because X` history and failure-derived non-regression requirements.
7. [research/CAUSAL_MODEL_STORE_ORCHESTRATOR_PLAN.md](research/CAUSAL_MODEL_STORE_ORCHESTRATOR_PLAN.md) — current causal scientific architecture plan.
8. Exact `artifacts/<run>/summary.json` / `REPORT.md` — authoritative evidence for a specific completed run.

[DOCUMENTATION_AUTHORITY.md](DOCUMENTATION_AUTHORITY.md) defines how these relate to the protected model registry and historical documents.

## Safety and research authority

Unless the user explicitly changes the scope and an appropriate contract is created:

- paper/research/shadow only;
- no live orders;
- no broker state mutation;
- no silent model promotion;
- no final-holdout opening for iterative tuning;
- no future information in model, route, calibration or orchestrator decisions;
- negative results remain preserved;
- a successful process exit is not economic validation.

Large research runs are performed locally and publish only compact, auditable result packages to GitHub.
