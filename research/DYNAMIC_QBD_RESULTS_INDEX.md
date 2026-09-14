# Dynamic-QBD Results Index

This is the navigation index for completed Opportunity-Portfolio, Top-10 and Dynamic-QBD experiments relevant to the current research program.

It exists because the repository contains many valid but differently scoped results. Agents must read this index before proposing a new selector, controller, refit rule or portfolio-policy search.

This file is a **map**, not a substitute for the exact run artifact. For numerical claims, the authoritative source is the linked `REPORT.md` / `summary.json` at the result commit.

Current authority across all rows unless explicitly stated otherwise:

`RESEARCH/SHADOW ONLY — NO PROMOTION — NO CAPITAL — FINAL PROSPECTIVE HOLDOUT CLOSED`

## How to read the table

- **Result** — what the run actually established.
- **Consequence** — what later work should assume.
- **Do not misread as** — common overclaim or repeat failure.
- **Status**:
  - `ESTABLISHED_CONSTRAINT` — a useful constraint for later research;
  - `NEGATIVE_RESULT` — tested and failed to justify the mechanism;
  - `SUGGESTIVE_ONLY` — interesting but not robust;
  - `SUPERSEDED_DIAGNOSTIC` — historical evidence with a known provenance/design limitation;
  - `MECHANISTIC_ONLY` — conditional diagnostic, not deployable causal selection evidence.

---

## A. Portfolio-policy QbD foundations

| Experiment | Artifact | Result | Consequence | Status |
|---|---|---|---|---|
| Prediction × Holding QbD | `artifacts/prediction-hold-qbd-surface/QBD_RESULTS_SUMMARY.md` | COMPLETE 465/465 H/D cells. 75 robust-gate cells, 22 locally stable cells, 18 design-space cells, 5 connected plateaus. Largest plateau: 14 cells H23-H26 / D3-D8. H6/D6 was isolated. | Treat H/D as a surface/plateau problem, not a one-cell winner problem. | `ESTABLISHED_CONSTRAINT` |
| Allocation QbD | `artifacts/allocation-qbd/ALLOCATION_QBD_RESULTS_SUMMARY.md` | 140/140 cells. `EQUAL_ACTIVE` retained. Rank-power, score-excess-power and softmax treatments did not produce robust paired improvement. | Keep equal-active allocation frozen unless a new experiment specifically targets allocation with new information. | `ESTABLISHED_CONSTRAINT` |
| Replacement QbD | `artifacts/replacement-qbd/REPLACEMENT_QBD_RESULTS_SUMMARY.md` | 28/28 cells. `IGNORE_NEW` robust; `REPLACE_WEAKEST` failed paired robustness. | Do not silently reintroduce replacement turnover into later model-selection experiments. | `ESTABLISHED_CONSTRAINT` |
| Concentration / replacement Phase 4 | `artifacts/concentration-replacement-qbd/` | Concentration and replacement robustness phase completed after the frozen earlier choices. | Preserve as supporting portfolio-policy evidence; do not reopen capacity/replacement casually. | `ESTABLISHED_CONSTRAINT` |
| Learned Exit QbD | `artifacts/learned-exit-qbd-profit-qbd-runtime/evaluation_summary.json` | 2,790 learned-exit points. Best Development point H11/D3/N1; broad H28/D21 islands for higher N. Promotion false. | Learned Exit remains a legitimate research dimension, but strong Development islands are not promotion evidence. | `SUGGESTIVE_ONLY` |

### Important early lesson

The QbD foundation intentionally froze dimensions sequentially. Later research should not vary H, D, N, allocation, replacement, exit, Recipe and orchestrator simultaneously unless the experiment is explicitly designed for that multiplicity.

---

## B. Frozen Top-10 forward and activation diagnostics

| Experiment | Artifact | Result | Consequence | Status |
|---|---|---|---|---|
| Frozen Top-10 live replay | `artifacts/top10-frozen-live-replay/REPORT.md` | True forward replay 2024-01-31 to 2026-07-24 with frozen model/policy. Most historical Top-10 models produced little activity or negative CAGR excess; point-in-time universe not fully verified. | Historical Development winners do not automatically remain winners. Dynamic maintenance is scientifically motivated, but this run does not identify the correct maintenance rule. | `ESTABLISHED_CONSTRAINT` |
| Dual-live reproduction preflight | `artifacts/top10-dual-live-validation/dual_live_validation_summary.json` | Historical entry/exit model reproduction was checked against stored folds and feature coverage reached 2026-07-24; expanding-live portion was still `READY_TO_RUN` in this preflight artifact. | Useful reproduction/data-contract evidence only; do not confuse the preflight with the later completed expanding-live portfolio. | `SUPERSEDED_DIAGNOSTIC` |
| Causal expanding Top-10 portfolio | `artifacts/top10-causal-expanding-portfolio/REPORT.md` | `DUAL_VALIDATION_COMPLETE`. 2023-08-11 to 2026-07-24 expanding-live replay produced only 3 trades; 0/10 models had positive CAGR excess. | Daily refit + model-specific threshold calibration did not rescue the frozen Top-10 system. | `NEGATIVE_RESULT` |
| Entry activation alpha diagnostic | `artifacts/top10-entry-activation-alpha-diagnostic/REPORT.md` | H24/H28 activation collapsed while high-ranked names retained positive realized alpha; H11 was rare but healthier. | Absolute calibration scale drift can suppress otherwise useful rankings. | `ESTABLISHED_CONSTRAINT` |
| Retrospective threshold validation | `artifacts/top10-retrospective-threshold-validation/REPORT.md` | Maturity-aware historical diagnostic labelled the frozen thresholds `RETROSPECTIVE_TOO_RESTRICTIVE_EVIDENCE`. | Threshold review was justified; no replacement threshold was authorized from the same period. | `ESTABLISHED_CONSTRAINT` |

Do not misread these as proof that “relax thresholds” is the general solution. Later threshold/controller runs were mixed.

---

## C. Top-10 adaptive-controller family

These experiments are important mainly because they prevent repeated rediscovery of the same controller ideas.

| Experiment | Artifact | Result | Consequence | Status |
|---|---|---|---|---|
| Causal threshold adaptation | `artifacts/top10-causal-threshold-adaptation/REPORT.md` | W_RECENT/W_MID/W_LONG profiles produced strongly model-specific outcomes. No forward winner was promoted. | Generic causal relaxation is not a universal fix. | `NEGATIVE_RESULT` |
| Monthly weight controller | `artifacts/top10-causal-monthly-weight-controller/REPORT.md` | Twelve-memory-month controller produced mixed forward results versus raw/uniform controls. | Do not assume learned memory weighting dominates simple history. | `NEGATIVE_RESULT` |
| Cohort controller V2 | `artifacts/top10-causal-cohort-controller-v2/REPORT.md` | Development-reuse warning. Controller was much worse than uniform history for many models, with exceptions at H24/D21. | Generic cohort relearning is unstable; do not repeat it without a materially new information contract. | `NEGATIVE_RESULT` |
| Model-specific memory V3 | `artifacts/top10-model-specific-memory-controller-v3/REPORT.md` | Model-specific frozen priors + adaptive controller; already-inspected Development profile, not independent OOS. Mixed results. | Model-specificity alone did not solve adaptation and cannot be claimed OOS. | `SUPERSEDED_DIAGNOSTIC` |
| Adaptation Gate V4 | `artifacts/top10-adaptation-gate-controller-v4/REPORT.md` | Gating improved several models but materially harmed H28/D21/N1; same Development reuse limitation. | An adaptation gate can move risk around but did not establish a general authority mechanism. | `SUPERSEDED_DIAGNOSTIC` |
| Top10 QBD Router V1 preview | `artifacts/top10-qbd-router-v1/REPORT.md` | 2,504 chronological decisions, but status `LIVE_PREVIEW_NO_REALIZED_OUTCOMES`; no realized outcome provider was opened. | Architecture/preflight evidence only. Never use this artifact for selector performance. | `SUPERSEDED_DIAGNOSTIC` |
| Top10 QBD Router V1 repaired all-arms | `artifacts/top10-qbd-router-v1-repaired-all-arms/summary.json` | 2,504 chronological preview decisions, A-K ablation arms, validation status `PSEUDO_OOS_DEVELOPMENT_WITH_REALIZED_DAILY_OUTCOMES`. | Preserve as router architecture/economic Development evidence, not a production selector verdict. | `SUPERSEDED_DIAGNOSTIC` |

Broad ideas already represented here include trailing winners, discounted weighting, superior sets, regime shrinkage, changepoint, opportunity activity and uncertainty/LCB. Do not propose them as untouched ideas.

---

## D. Dynamic-QBD factory / Development Run 2016-2025

| Experiment | Artifact | Result | Consequence | Status |
|---|---|---|---|---|
| Dynamic-QBD causal factory/fidelity stack | `stock_predictor/backtests/opportunity_portfolio_research/DYNAMIC_QBD.md` plus Aug-15 commits | Causal Recipe selection from matured folds, immutable Generation identity, model-specific score materialization, calibration, lineage, restart/freeze contracts implemented. | This is the technical basis for later Dynamic-QBD experiments. | `ESTABLISHED_CONSTRAINT` |
| Development preflight 2016-2025 | `artifacts/dynamic-qbd-development-preflight-2016-2025-no-stock-v2/summary.json` | Data boundary passed but portfolio validity was explicitly disabled and the run blocked on missing complete Candidate×Fold OOS evidence. | This was an initializer/preflight, not the final 2,790-family economic result. | `SUPERSEDED_DIAGNOSTIC` |
| Development Run 2016-2025 | `artifacts/dynamic-qbd-development-run-2016-2025-final/summary.json` | 2,790 FIXED families, 119 ABC months, 55 structural plateaus. Gate1 FAIL, Gate1B FAIL, Gate2 FAIL, Gate3 FAIL. Router shadow only. | Existing performance/health/regime selector stack has no authority. | `NEGATIVE_RESULT` |
| Development Run scale / streaming repair | `artifacts/dynamic-qbd-development-run-2016-2025-final/README.md` and commit `9da9ac0` | 1,632,350,262 matured prediction rows; 179,376 evidence rows; peak RSS ~5.64 GB after streaming/resumable aggregation repair. | Never restore all-matured `pd.concat` materialization. Scale and resume contracts are architectural constraints. | `ESTABLISHED_CONSTRAINT` |

### Development Run gate meaning

A completed run is not a passed selector. The gate stack conservatively rejected capital authority even though individual families/periods could look strong.

---

## E. Meta-predictability / selector feasibility

| Experiment | Artifact | Result | Consequence | Status |
|---|---|---|---|---|
| Regime persistence diagnostics | `artifacts/dynamic-qbd-regime-diagnostics/REPORT.md` | `PERSISTENCE_EXISTS_BUT_TOO_WEAK_AFTER_COSTS`. Family rank autocorrelation exists, but active-conditioned persistence was weak and selector feasibility did not clear cost-robust hurdles. | Do not build another generic regime/persistence router from the same state variables. | `NEGATIVE_RESULT` |
| Opportunity-state predictability | `artifacts/dynamic-qbd-opportunity-state/REPORT.md` | `NO_ROBUST_OPPORTUNITY_STATE_PREDICTABILITY`. 10,836 active events but only 108/2,790 Families and 495 active sets. Consensus additions did not robustly beat Score-only. | Opportunity-state density/consensus is not established selector authority. | `NEGATIVE_RESULT` |
| True cross-horizon consensus | `artifacts/dynamic-qbd-true-consensus/REPORT.md` | 134 true multi-bucket sets. Paired selected-excess effect looked positive in aggregate, but Ridge/HGB-specific bootstrap lower bounds crossed zero. Diagnosis: suggestive but not robust. | Preserve as signal, not a routing rule. | `SUGGESTIVE_ONLY` |
| Paired selector tournament | `artifacts/dynamic-qbd-paired-selector-tournament/REPORT.md` | No fitted pool cleared robust gates. Diagnosis `G_SCORE_ONLY_REMAINS_BEST_ROBUST_SELECTOR`. | Stop generic selector-pool promotion research until genuinely new independent evidence/information arrives. | `NEGATIVE_RESULT` |

This sequence is a direct reason the next work should improve the **causal chronology/model-store experiment**, not add more feature engineering to the selector.

---

## F. Recipe switching, recalibration and refit frequency

### H3 recipe reselection / model combination

Artifact: `artifacts/dynamic-qbd-model-combination-2023q4/REPORT.md`

Matched refit dates:

- rolling causal Recipe reselection: EUR 14,703.12;
- frozen Recipe refitted on same dates: EUR 13,398.25;
- rolling added EUR 1,304.87;
- relative MaxDD worsened from -24.27% to -36.11%.

Hard Ridge/HGB intersection in the Q4 diagnostic produced one trade and underperformed the selected single model.

**Consequence:** causal Recipe changes can add terminal wealth while increasing transition/drawdown risk. Hard consensus was not supported. This one H3 cell was not enough for surface-wide claims.

Status: `SUGGESTIVE_ONLY`.

### 2016-2017 causal preflight

Artifact: `artifacts/dynamic-qbd-monthly-recipe-2016-2017-audit-v3/REPORT.md`

Decision: **NOT_RUN_FAIL_CLOSED**.

The requested 2016-2017 strategy replay could not be produced causally:

- signal panel starts 2016-06-24;
- required history was 504 train + 30 purge + 252 calibration sessions;
- first history-eligible month end was 2019-08-30;
- first jointly history-and-Recipe-evidence-eligible month end was 2020-08-31.

No future Recipe evidence or fabricated prehistory was used.

**Consequence:** this is direct evidence for the current “no true ten-year historical seed before 2016” limitation.

Status: `ESTABLISHED_CONSTRAINT`.

### Monthly Recipe factory

Artifact: `artifacts/dynamic-qbd-monthly-recipe-h3-total-return/REPORT.md`

Status explicitly: `SUPERSEDED_PROVENANCE_MANIFEST_REQUIRED`.

The rolling monthly Recipe arm beat the frozen monthly Recipe arm, but both underperformed URTH and only one genuinely different Recipe episode drove the contrast. N2/N3 capacity improved this historical diagnostic, and CVNA dominated drawdown attribution.

**Consequence:** retain only as historical diagnostic. Do not use its raw portfolio figures as final research evidence.

Status: `SUPERSEDED_DIAGNOSTIC`.

### Recipe hysteresis

Artifact: `artifacts/dynamic-qbd-recipe-hysteresis-h3-d2-n1-60gib-enforced/REPORT.md`

Diagnosis: `HYSTERESIS_ROLLING_TRADEOFF`. Hysteresis modestly improved terminal value versus rolling but did not create a clean Pareto-dominant result.

**Consequence:** “add hysteresis” is already tested and not a solved general mechanism.

Status: `NEGATIVE_RESULT`.

### Conditional refit

Artifact: `artifacts/dynamic-qbd-conditional-refit-h3-d2-n1-enforced/REPORT.md`

- C monthly same-Recipe refit: EUR 10,634.57;
- B frozen model + rolling recalibration: EUR 10,384.74;
- D1 hysteresis conditional refit: identical to B;
- D0 immediate new-evidence Recipe-change refit: EUR 8,707.13.

Diagnosis: `D1_MATCHES_FROZEN_MODEL_B`.

**Consequence:** merely waiting for Recipe-change evidence/hysteresis did not produce a useful causal refit mechanism in this setup.

Status: `NEGATIVE_RESULT`.

---

## G. Matched A/B/C recalibration decomposition

Artifact: `artifacts/dynamic-qbd-abc-recalibration-h3-d2-n1/REPORT.md`

H3/D2/N1:

| Arm | Terminal EUR | Relative MaxDD | Trades |
|---|---:|---:|---:|
| A frozen model + frozen calibration | 13,461.75 | -28.79% | 35 |
| B frozen model + rolling recalibration | 10,384.74 | -59.34% | 108 |
| C monthly fresh fit + rolling calibration | 10,634.57 | -31.04% | 94 |

Causal contrasts:

- B-A terminal delta: -EUR 3,077.01; CAGR delta -8.2064 pp;
- C-B terminal delta: +EUR 249.82; CAGR delta +0.7254 pp.

**Correct interpretation:**

Rolling recalibration was severely destructive. C only looks better than B because B is badly damaged; monthly C remained far below frozen A.

**Do not misread as:** “monthly refitting works.”

Status: `ESTABLISHED_CONSTRAINT`.

The process-parallel reproduction at `artifacts/dynamic-qbd-abc-recalibration-h3-d2-n1-parallel32/REPORT.md` reproduced the same A/B/C economic result under the parallel execution path. Treat it as execution/reproducibility evidence, not an independent statistical sample.

---

## H. H3 Fold-Clock refit

Artifact: `artifacts/dynamic-qbd-fold-clock-refit-h3-d2-n1-rerun2/REPORT.md`

H3/D2/N1:

| Arm | Terminal EUR | Relative MaxDD | Trades |
|---|---:|---:|---:|
| F Fold-Clock | 15,773.36 | -24.26% | 77 |
| A frozen | 13,461.75 | -28.79% | 35 |
| M monthly | 10,634.57 | -31.04% | 94 |

- F-A CAGR delta: +5.3353 pp;
- M-F CAGR delta: -12.8164 pp.

At this single cell, Fold-Clock Pareto-improved A and monthly refit strongly worsened F.

**Consequence:** justified a Fold-Clock surface validation.

**Do not misread as:** proof that Fold-Clock works across H1-H30.

Status: `SUGGESTIVE_ONLY`.

---

## I. Fold-Clock Surface Validation 2016-2025

Artifact root: `artifacts/dynamic-qbd-fold-clock-surface-validation-2016-2025-20260826/`  
Primary: `summary.json`, `gates.json`, `REPORT.md`.  
Current published rerun result commit: `79987c115d966edc829c3aa84c26dd18711ac4d5`.  
Run/code SHA: `92a8cc048d2ac669425675a629c031a140673a9a`.

The scheduler/process-pool rerun reproduced the prior economic/research payload exactly aside from code/run-contract/execution provenance. This is valuable execution reproducibility evidence, not another independent economic sample.

Scope:

- 5,400 families;
- 2,790 FIXED;
- 2,610 LEARNED_EXIT;
- 16,200 family-arm results;
- 10,800 family contrasts;
- 30 horizons;
- 110 plateaus;
- 1,936 monthly fresh fits;
- 210 Fold-Clock fits.

### F minus A

ALL:

- median family CAGR-excess delta: +0.5888 pp;
- median plateau delta: +1.1104 pp;
- positive plateau fraction: 58.18%;
- positive-year fraction: 33.33%;
- median plateau relative-MaxDD delta: **-23.23 pp**;
- gate: FAIL.

FIXED:

- median family delta: +1.1580 pp;
- median plateau delta: +1.9728 pp;
- positive plateau fraction: 63.64%;
- positive-year fraction: 33.33%;
- median relative-MaxDD delta: **-22.18 pp**;
- gate: FAIL.

LEARNED_EXIT:

- median family delta: -0.1333 pp;
- median plateau delta: +0.8203 pp;
- positive plateau fraction: 52.73%;
- bootstrap q05 negative;
- median relative-MaxDD delta: **-24.24 pp**;
- gate: FAIL.

### M minus F

Robustly negative across all strata:

- ALL median family delta: -3.9122 pp;
- FIXED: -4.8383 pp;
- LEARNED_EXIT: -3.3350 pp;
- negative plateaus: 81.82%-89.09%;
- negative years: 66.67%;
- all M-F gates: PASS in the predeclared negative direction.

### Scientific consequence

Established:

> Calendar-monthly fresh refitting is broadly worse than sparse Fold-Clock refitting under the fixed-Recipe contract.

Not established:

> Fold-Clock is robustly better than keeping the model frozen.

Not established:

> Any specific horizon or model family should automatically be refit.

Status: `MECHANISTIC_ONLY` because of the later identified ex-post Recipe-selection limitation.

---

## J. Fold-Event Counterfactual

Artifact root: `artifacts/dynamic-qbd-fold-event-counterfactual/`  
Result commit: `26dda9607304769e4f134681ab380388668af5d2`  
Run code commit: `7598dbf000898f2632267e921c7d2790e40e1714`

Contract:

- FIXED only;
- 2,790 families;
- 30 horizons;
- 180 non-initial Fold events;
- 16,740 Event-Family rows;
- OLD and NEW branch from identical pre-event portfolio state;
- NEW becomes canonical state for the next event;
- contract audit PASS.

Aggregate:

- positive median event fraction: 25.56%;
- return-and-risk helpful: 10.00%;
- return-and-risk harmful: 10.56%;
- 77/180 event medians exactly zero;
- among 103 non-zero events: 46 positive, 57 negative;
- median non-zero event effect about -1.94%.

Year-level:

- 2021: median 0; 11.7% positive events;
- 2022: median 0; 36.7% positive; median drawdown delta around -3.31 pp;
- 2023: median NEW-minus-OLD excess -4.88%; 28.3% positive; median drawdown delta +2.53 pp.

Simple ex-ante associations with event benefit were weak:

- incumbent model age Spearman ~+0.20;
- threshold delta ~+0.05;
- threshold-percent delta ~0;
- train-start shift ~+0.17;
- train-end shift ~+0.11.

Post-event diagnostics were also limited; threshold-pass Jaccard was the strongest of them at roughly -0.25.

### Scientific consequence

Established:

> A new Fold is an information event, not an automatic activation command.

Not established:

> Which Recipe/Generation a historical live system should choose.

### Critical limitation

The fixed Recipe for a horizon came from broader Development evidence overlapping the analyzed interval. This creates hindsight conditioning. Any apparent HGB/Ridge or horizon-specific refresh pattern is conditional on those ex-post Recipe assignments.

Status: `MECHANISTIC_ONLY`.

---

## K. Current research conclusion

The research program has moved through:

```text
portfolio surface
→ frozen forward failure
→ threshold/adaptation controllers
→ router/meta-predictability
→ causal factory
→ Development Run 2016-2025
→ regime/opportunity/consensus
→ selector tournament
→ recalibration/refit decomposition
→ Fold Clock
→ Fold-Clock Surface Validation 2016-2025
→ event counterfactual
```

The next step is **not** another post-hoc selector.

The next experiment must change the information contract:

> Build a Model Store from only the historical prefix available at pseudo-live start, create immutable expanding-window Generations as evidence arrives, keep old Generations, and let an Orchestrator choose only among models that existed at that historical time.

See:

- [DYNAMIC_QBD_CURRENT_STATE.md](DYNAMIC_QBD_CURRENT_STATE.md)
- [CAUSAL_MODEL_STORE_ORCHESTRATOR_PLAN.md](CAUSAL_MODEL_STORE_ORCHESTRATOR_PLAN.md)
- [../TOBECONTINUED.md](../TOBECONTINUED.md)

---

## L. Results that must remain visibly negative

Do not hide or “supersede away” these facts:

1. Frozen historical winners did not robustly generalize forward.
2. Monthly recalibration can be severely destructive.
3. Calendar-monthly refitting is broadly worse than sparse Fold-Clock refitting.
4. Fold-Clock did not pass surface-wide F-vs-A robustness.
5. Existing performance/health/regime gates failed.
6. Regime persistence was too weak after costs.
7. Opportunity-state predictability was not robust.
8. True consensus was suggestive, not robust.
9. The fitted selector tournament did not beat Score-only robustly.
10. Simple event-level ex-ante features did not explain which refresh would help.
11. The latest Fold-Clock/Counterfactual Recipe assignments are hindsight-conditioned and therefore cannot be used as causal historical recipe-selection evidence.

These failures narrow the valid next design and are valuable research output.
