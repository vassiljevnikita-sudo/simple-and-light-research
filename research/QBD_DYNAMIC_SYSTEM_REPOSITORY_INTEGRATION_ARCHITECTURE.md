# QBD Dynamic System — repository integration architecture

## Status

Normative repository-integration architecture for the Dynamic-QBD research programme.

This document must be read together with:

- `research/QBD_DYNAMIC_SYSTEM_RESEARCH_ROADMAP_AND_TARGET_ARCHITECTURE.md`
- `research/QBD_DYNAMIC_SYSTEM_SCIENTIFIC_EVIDENCE.md`

The roadmap defines the research sequence and falsification gates. This document defines **how the existing repository components are to be reused and connected** to implement that programme without rebuilding already validated causal/replay infrastructure.

The central repository decision is:

> **Dynamic QBD is not a greenfield rewrite. The existing causal portfolio, H×D QBD, V5 point-in-time, learned-exit, shadow-state and freeze infrastructure is reused. The principal new layer is Family → ModelGeneration → GenerationRegistry → causal refit/recalibration → family evidence → predictability gates.**

The old Top-10 Router/V4/Activity/Changepoint/Promotion stack remains implementation inventory only and has no capital-allocation authority until the required meta-predictability gates pass.

---

# 1. Repository audit — what already exists

The repository already contains most of the expensive low-level research infrastructure required by the Dynamic-QBD design.

## 1.1 QBD candidate surface

The existing Prediction-Horizon × Holding-Days research machinery already supports:

```text
H = 1 ... 30
D = 1 ... H
```

which yields all 465 valid H/D cells.

The existing implementation searches the established entry dimensions inside each cell rather than requiring a new H×D engine.

Current `max_names` design-space gap:

```text
current search:
N = 1, 2, 3, 5

target Dynamic-QBD design space:
N = 1, 2, 3, 4, 5, 6
```

N4 and N6 therefore require an explicit design-space extension. H×D itself does not.

## 1.2 Stateful portfolio accounting

`stock_predictor/backtests/opportunity_portfolio_research/portfolio.py` already implements the economically important stateful path:

```text
current portfolio NAV
        ↓
current stock/URTH/cash state
        ↓
NAV-dependent available active capital
        ↓
position sizing
        ↓
P&L / costs / taxes
        ↓
new NAV
        ↓
next opportunities use the new capital base
```

The implementation already includes:

- stock positions;
- URTH benchmark sleeve;
- cash;
- current-equity-dependent position sizing;
- `max_names` enforcement;
- exits;
- freed capital;
- later/replacement entries;
- transaction costs;
- German tax approximation where enabled;
- accounting reconciliation;
- terminal wealth;
- benchmark terminal wealth;
- benchmark-relative wealth diagnostics;
- relative maximum drawdown.

This engine remains the **economic accounting authority** for Dynamic-QBD research unless explicitly superseded by a separately validated implementation.

## 1.3 Learned exits

The repository already contains a stateful learned-exit replay path in the opportunity-portfolio research stack.

This should be reused for Dynamic-QBD families whose exit policy is learned.

The historical V4.5 frozen-entry exit overlay remains useful as historical research evidence, but it is **not** the final production-policy replay when an early exit should free capacity and permit a replacement entry.

## 1.4 V5 causal primitives

The V5 stack already supplies reusable lower-level controls for:

- point-in-time feature construction;
- `as_of` / `available_at` checks;
- prohibited future-feature rejection;
- locked final holdout;
- chronological walk-forward folds;
- purge/embargo contracts;
- deterministic seeds;
- dataset fingerprints;
- feature-schema fingerprints;
- model-artifact hashes;
- registry collision protection;
- prediction provenance;
- OOS-only uncertainty construction.

These are **factory primitives**, not a replacement for the new Dynamic-QBD GenerationRegistry.

## 1.5 Existing QBD router infrastructure

The old Top-10 QBD branch already contains substantial generic machinery:

- `top10_qbd_contract.py`
- `top10_qbd_expert_registry.py`
- `top10_qbd_shadow_engine.py`
- `top10_qbd_shadow_ledger.py`
- `top10_qbd_health_store.py`
- `top10_qbd_health_metrics.py`
- `top10_qbd_activity_monitor.py`
- `top10_qbd_changepoint.py`
- `top10_qbd_candidate_set.py`
- `top10_qbd_allocation.py`
- `top10_qbd_promotion.py`
- `top10_qbd_router.py`
- `top10_qbd_state.py`
- `top10_qbd_state_store.py`
- `top10_qbd_prequential_replay.py`
- `top10_qbd_overfitting_diagnostics.py`
- `top10_qbd_freeze.py`
- `top10_qbd_test_suite.py`

This code is not discarded. It is split into:

1. reusable causal/state infrastructure;
2. legacy static-R01–R10 semantics;
3. deferred selector/controller mechanisms.

---

# 2. Authoritative Dynamic-QBD execution chain

The target repository wiring is:

```text
                EXISTING POINT-IN-TIME / V5 PRIMITIVES
          feature availability + maturity + hashes + holdout
                                 │
                                 ▼
                         ExpertFamilySpec
                                 │
                                 ▼
                      causal refit scheduler
                                 │
                    latest matured information
                                 │
                                 ▼
                  model-training build primitive
                                 │
                                 ▼
                         ModelGeneration
                                 │
                                 ▼
                generation-specific recalibration
                                 │
                       threshold / policy
                                 │
                                 ▼
                       GenerationRegistry
                                 │
                 ┌───────────────┴───────────────┐
                 │                               │
                 ▼                               ▼
        existing H1-H30/QBD              learned-exit runtime
        signal/entry machinery                    │
                 │                                │
                 └───────────────┬────────────────┘
                                 ▼
                AUTHORITATIVE STATEFUL PORTFOLIO REPLAY
                   opportunity_portfolio_research
                                 │
                    independent family shadow NAV
                                 │
                                 ▼
                     maturity-aware shadow evidence
                                 │
                                 ▼
                       Monthly Family Evidence Panel
                                 │
                         A / B / C experiment
                                 │
                                 ▼
                       META-PREDICTABILITY GATES
                                 │
                  ┌──────────────┴───────────────┐
                  ▼                              ▼
                FAIL                            PASS
                  │                              │
       no performance-driven           enable selector research
             switching                         │
                                               ▼
                                 adapter into reusable router tools
                                               │
                                               ▼
                                    optional Champion layer
```

No selector/controller layer sits between ModelGeneration and family shadow evidence during the factory-validation phases.

---

# 3. Durable identity model

## 3.1 `ExpertFamilySpec`

`ExpertFamilySpec` identifies a durable structural QBD strategy rather than one fitted model artifact.

At minimum it owns:

- `family_id`;
- prediction horizon H;
- holding-days / exit-policy definition;
- `max_names` contract;
- entry-policy rule;
- feature-schema identity;
- model family / model-training recipe;
- frozen hyperparameters or frozen hyperparameter-selection rule;
- training-window rule;
- calibration-window rule;
- threshold-calibration rule;
- refit cadence;
- benchmark contract;
- cost contract;
- tax contract;
- deterministic seed/configuration contract.

Example structural identity:

```text
H24_D05_N01_LEARNED_EXIT
```

This identity persists while concrete fitted artifacts change.

## 3.2 `ModelGeneration`

A `ModelGeneration` is one concrete historical fit produced under an `ExpertFamilySpec`.

Example:

```text
H24_D05_N01_LEARNED_EXIT
        ├── 2025-01 generation
        ├── 2025-02 generation
        ├── 2025-03 generation
        └── ...
```

Required fields include:

- `generation_id`;
- `family_id`;
- refit timestamp;
- information cutoff;
- latest matured label cutoff;
- train start/end;
- calibration start/end;
- model artifact hash;
- dataset fingerprint;
- feature-schema hash;
- code/training-recipe fingerprint;
- random seed;
- calibration fingerprint;
- generation-specific resolved threshold;
- exit artifact / exit policy fingerprint;
- validation status;
- lifecycle status.

Recommended lifecycle values:

```text
BUILDING
VALID
FAILED
RETIRED
```

## 3.3 Refit and recalibration are inseparable

The Dynamic-QBD contract is:

```text
new model fit
        ↓
new causal calibration
        ↓
new generation-specific threshold
```

A fresh generation must never silently inherit the old absolute threshold simply because the family identity is unchanged.

## 3.4 `GenerationRegistry`

The GenerationRegistry owns:

- every historical generation;
- current valid generation per family;
- failed refit attempts;
- activation timestamp;
- artifact/config fingerprints;
- prior-valid-generation fallback.

Exactly one current `VALID` generation may be active for a family.

Failed refit behaviour is fail-safe:

```text
new generation build fails
        ↓
mark generation FAILED
        ↓
last VALID generation remains current
```

No failed or incompletely validated generation can silently acquire entry authority.

---

# 4. Family evidence and generation evidence are different

The system maintains two different evidence concepts.

## 4.1 Family evidence

Family evidence answers:

> Does this structural QBD family remain useful across many causally created generations and market periods?

It persists across refits.

Examples:

- long-run benchmark-relative wealth history;
- fold/time-block stability;
- behaviour across generations;
- robustness to single historical winners;
- long-run downside path.

## 4.2 Current-generation evidence

Current-generation evidence answers:

> How healthy is the exact fit currently produced by the factory?

Examples:

- current-fit matured Rank IC;
- calibration error;
- score monotonicity;
- top-score realised excess;
- residual drift;
- generation-specific uncertainty.

Historical family success must not be rewritten as if it were evidence generated by the fresh fit.

Conversely, creating a fresh fit must not erase the accumulated history of the family.

---

# 5. Existing components that remain authoritative

## 5.1 Portfolio engine

Authoritative component:

`stock_predictor/backtests/opportunity_portfolio_research/portfolio.py`

Responsibility:

- capital accounting;
- stock/URTH/cash state;
- NAV;
- positions;
- fills;
- costs;
- tax world;
- `max_names`;
- replacement capacity;
- final portfolio wealth.

Dynamic-QBD must not create a simplified second economic replay and use it as final evidence.

## 5.2 Learned-exit replay

Authoritative learned-exit mechanics are reused from the existing learned-exit opportunity-portfolio stack.

Family replay must allow an early exit to affect later portfolio state and future entries.

## 5.3 H×D candidate surface

Authoritative surface machinery is reused from the existing H1–H30 / Prediction-Hold QBD stack.

Do not rebuild the 465-cell surface.

The required extension is the N dimension from the current `(1,2,3,5)` to the intended `1..6`, subject to the same causal development-only research discipline.

## 5.4 Point-in-time and training primitives

Reuse V5 primitives for:

- data availability;
- maturity;
- holdout guards;
- folds;
- fingerprints;
- deterministic training;
- artifact registration;
- OOS uncertainty.

Wrap these primitives behind the new Family/Generation lifecycle rather than duplicating them.

---

# 6. Legacy components reused through adapters

The following old Router-V1 concepts remain useful once stripped of static R01–R10 assumptions.

## 6.1 Shadow maturity barrier

Reuse the `PendingOutcome → MaturedOutcome` principle from `top10_qbd_shadow_ledger.py`.

Before maturity, router/evidence-facing state contains only an opaque outcome reference.

Only when:

```text
outcome_available_at <= current_information_cutoff
```

may realised outcome values enter evidence.

## 6.2 State persistence / restart

Reuse the deterministic state-store pattern from:

- `top10_qbd_state.py`
- `top10_qbd_state_store.py`
- `top10_qbd_prequential_replay.py`

but extend persisted state to include:

- current generation per family;
- refit history cursor;
- calibration history cursor;
- generation-specific open-position lineage;
- family-evidence cursor.

## 6.3 Freeze-manifest mechanics

Reuse canonical serialization/hash mechanics from the old freeze system, but change what is frozen.

The new final-holdout manifest freezes the **algorithm**, not all future concrete model hashes.

---

# 7. Static R01–R10 registry is Legacy

The current `top10_qbd_expert_registry.py` is a historical static Top-10 registry.

Its semantics are:

```text
R01...R10
        ↓
fixed concrete model artifact
        ↓
fixed entry policy / threshold
        ↓
V4 adaptation-controller identity
```

This is not the Dynamic-QBD family registry.

Normative status:

```text
LEGACY_STATIC_TOP10_RESEARCH_BASELINE
```

It remains available for reproducing old Top-10/controller experiments.

Replacement identity:

```text
LEGACY
R01
→ fixed artifact
→ fixed threshold
→ V4 controller

DYNAMIC
ExpertFamilySpec
→ ModelGeneration(t)
→ recalibration(t)
→ threshold(t)
→ family shadow NAV(t)
```

---

# 8. Components explicitly deferred until gate pass

The following code may remain in the repository but MUST NOT have capital-allocation authority during Phase 0 through Gate 3:

- `top10_qbd_router.py`;
- `top10_qbd_promotion.py`;
- `top10_qbd_activity_monitor.py`;
- `top10_qbd_changepoint.py`;
- old candidate-set logic;
- old router allocation logic;
- old V4 adaptation integration;
- Champion/probation/suspension lifecycle;
- regime-conditioned routing;
- uncertainty-LCB promotion.

Normative status:

```text
LEGACY_DEFERRED_UNTIL_META_GATE_PASS
```

Reason:

these mechanisms presuppose that information observable at time `t` can usefully predict future relative family quality.

That proposition is exactly what the meta-predictability programme must first prove or falsify.

A code path existing in the repository is not evidence that the mechanism belongs in production.

---

# 9. Monthly Family Evidence Panel

The factory must produce one common causal evidence panel before selector research begins.

Conceptual row:

```text
assessment_date
family_id
generation_id
generation_refit_date
information_cutoff
resolved_threshold

NAV
benchmark_NAV
relative_wealth
1M_excess
3M_excess
6M_excess
12M_excess

relative_max_drawdown
relative_sortino
CDaR
expected_shortfall
drawdown_duration
recovery_duration
time_under_water
capital_impairment

trade_count
matured_trade_count
matured_prediction_count

rank_ic
top_score_realised_excess
calibration_error
score_spread
prediction_residual_drift

top_ticker_contribution
top_trade_contribution
fold_stability
generation_stability
```

This panel is a causal information store. It is not one unrestricted meta-model feature matrix.

Each research gate receives only its predeclared subset.

---

# 10. Risk/wealth-path metrics missing from current authoritative replay

The existing replay already provides terminal wealth, benchmark wealth and relative maximum drawdown.

The Dynamic-QBD evidence layer still needs explicit implementations for:

- benchmark-relative Sortino/downside deviation;
- Conditional Drawdown at Risk (CDaR);
- expected shortfall of benchmark excess;
- current drawdown depth;
- drawdown duration;
- recovery time;
- time under water;
- capital-impairment buckets / area.

These are diagnostics and possible Gate-1B inputs.

They are **not** objectives to minimise total volatility.

Upside volatility remains unpenalised unless a later experiment establishes an economic reason otherwise.

---

# 11. Sparse trades and concentration semantics

## 11.1 Sparse trades

Few trades do not mean bad model quality.

They primarily reduce certainty that observed alpha generalises.

Therefore:

```text
few trades
→ higher epistemic uncertainty
```

not:

```text
few trades
→ automatic demotion
```

Calendar inactivity is not a negative routing signal by itself.

## 11.2 Concentration

Ticker/sector concentration is not an optimisation penalty.

If the strongest causal opportunities all belong to one sector, the portfolio may legitimately hold them.

Concentration diagnostics answer questions such as:

- are apparent bets strongly dependent?;
- does one trade explain most historical alpha?;
- does one ticker explain most claimed general edge?;
- how strong is the scientific evidence after removing one ex-post winner?

They downgrade evidence strength where appropriate; they do not force lower-quality diversification trades.

## 11.3 Threshold / `max_names`

Do not loosen threshold or increase N merely to create more breadth.

The economic principle is:

```text
admit a marginal opportunity only if
expected benchmark excess after costs remains positive
```

Trade count and diversification are outcomes, not primary optimisation targets.

---

# 12. Required A/B/C factory-value experiment

Before any family selector is promoted, compare under identical portfolio execution:

```text
A  frozen model + frozen calibration

B  frozen model + rolling causal recalibration

C  rolling causal refit + rolling causal recalibration
```

Interpretation:

```text
B - A = incremental value of recalibration
C - B = incremental value of retraining
```

A future `D` arm may add dynamic family selection only after the required predictability evidence exists.

Primary economic comparison uses actual portfolio wealth paths, not isolated prediction accuracy.

---

# 13. Gate wiring

## Gate 1 — past economic performance

Allowed predictors are restricted to predeclared historical performance variables such as:

- 1M excess;
- 3M excess;
- 6M excess;
- 12M excess;
- EWMA excess;
- recent-minus-long-run excess;
- positive-period fraction.

Do not include prediction-health or regime information in Gate 1.

Targets:

```text
future 1M family benchmark excess
future 3M family benchmark excess
future family ranking
```

For the 3M target, the family follows its frozen future refit schedule. It is not one fixed generation held for three months unless that is the actual production policy.

## Gate 1B — downside / wealth path

Adds only predeclared capital-path features:

- relative downside deviation / Sortino;
- relative MaxDD;
- CDaR;
- expected shortfall;
- drawdown duration;
- recovery duration;
- time under water;
- capital impairment.

Question:

> Does past capital-impairment shape add prospective information beyond past return?

If no, these remain diagnostics only.

## Gate 2 — current-fit health

Possible causal inputs:

- matured Rank IC;
- top-score realised excess;
- calibration drift;
- score monotonicity;
- residual drift;
- score dispersion;
- top-vs-median realised outcome.

This gate may succeed even if Gate 1 fails.

## Gate 3 — market state

Only after prior gates are separately measured may market/regime state be added.

Potential frozen inputs include trend, stress/volatility state, breadth, dispersion and causally available liquidity measures.

No regime router is promoted merely because market regimes are economically plausible.

---

# 14. Gate result → architecture mapping

Architecture is conditional on evidence.

```text
Phase 0 factory fidelity fails
→ stop; no later result admissible

A/B/C: recalibration fails
→ remove rolling recalibration

A/B/C: refit fails
→ remove recurring refit

Gate 1 fails
→ no trailing-performance Champion–Challenger

Gate 1B fails
→ downside metrics remain diagnostics

Gate 2 fails
→ current-fit prediction health not used for selection

Gate 3 fails
→ no regime layer

selector comparison fails
→ no dynamic family selection

hard Champion loses to simpler selector/hold rule
→ no hard Champion deployment
```

The smallest architecture surviving its own causal ablations is the desired endpoint.

---

# 15. Mandatory new Dynamic-QBD tests

Existing generic causality/accounting tests remain in force.

The new factory layer additionally requires the following.

## 15.1 Full-factory future-mutation test

Mutating all data after cutoff `t` must not alter any object that should already be determined at `t`, including:

- generation artifact;
- calibration;
- threshold;
- prediction;
- trade decision;
- portfolio NAV through `t`;
- family evidence through `t`;
- selector feature row at `t`.

## 15.2 H1–H30 maturity test

For every consumed training/calibration/evidence outcome:

```text
terminal_date <= information_cutoff
```

must hold.

H1 and H30 necessarily have different most-recent usable outcomes at the same wall-clock date.

## 15.3 Generation reproducibility

Identical:

- code;
- data;
- FamilySpec;
- cutoff;
- seed;
- calibration rule

must create identical:

- generation identity;
- model artifact hash;
- calibration fingerprint;
- resolved threshold;
- predictions.

## 15.4 Failed-refit fallback

A failed new generation cannot become current. The last valid generation remains active until a valid replacement exists.

## 15.5 Family evidence continuity

Refitting must not reset historical family evidence.

## 15.6 Generation evidence isolation

Historical success of an old generation cannot be written into the fresh generation as current-fit evidence.

## 15.7 Position lineage

Every position must carry at minimum:

```text
family_id
generation_id
entry policy identity
exit policy identity
```

A later Champion switch must not rewrite the lineage or exit authority of an already-open position.

## 15.8 N-policy behavioural difference

Where sufficient opportunities exist, N1 and N6 must create materially different portfolio states.

## 15.9 Exit / replacement fidelity

An early exit that releases a slot/capital must allow a replacement entry whenever the frozen production policy permits it.

## 15.10 Full continuous-vs-restart parity

Continuous execution and interrupted/resumed execution must produce identical:

- refit schedule;
- generations;
- calibrations;
- thresholds;
- predictions;
- positions;
- trades;
- NAV;
- matured evidence;
- later selector decisions.

---

# 16. Final holdout freeze semantics

The old static router freeze pins concrete expert artifact hashes. That is appropriate for reproducing the historical static-R01–R10 system, but not for Dynamic-QBD.

Before the Dynamic-QBD final holdout, freeze the **algorithmic recipe**:

- exact code commit;
- FamilyRegistry hash;
- candidate design space;
- feature schema;
- training recipe;
- hyperparameter rule;
- training-window rule;
- refit cadence;
- H-specific maturity rule;
- calibration-window rule;
- recalibration algorithm;
- threshold rule;
- exit runtime/training rule;
- benchmark/cost/tax contract;
- evidence-panel schema;
- risk metrics allowed to affect decisions;
- gate/selector features and model/rule, if any;
- switch/hysteresis rules, if any;
- all pass/fail thresholds;
- final-holdout boundaries;
- initial pre-holdout state.

Future monthly concrete generation artifacts are **not** frozen beforehand.

Inside the holdout they are created causally by the already-frozen algorithm and recorded append-only with their fingerprints.

The evaluated object is:

```text
TrainingAlgorithm
+ RefitSchedule
+ RecalibrationPolicy
+ PortfolioExecutionPolicy
+ optional frozen SelectionRule
```

---

# 17. Legacy document status

The following old Router-V1 documents remain useful historical implementation records:

- `TOP10_QBD_ROUTER_V1_TARGET_ARCHITECTURE.md`
- `TOP10_QBD_ROUTER_V1_IMPLEMENTATION_PLAN.md`
- `TOP10_QBD_ROUTER_V1_TEST_CONTRACT.md`
- `TOP10_QBD_ROUTER_V1_CONCRETE_DESIGN.md`

Their normative role for current development is:

```text
LEGACY_DEFERRED_ROUTER_ARCHITECTURE
```

They may be consulted for reusable state/router components, but they do not define the current next implementation phase.

Current normative documents are:

1. `QBD_DYNAMIC_SYSTEM_RESEARCH_ROADMAP_AND_TARGET_ARCHITECTURE.md`
2. `QBD_DYNAMIC_SYSTEM_REPOSITORY_INTEGRATION_ARCHITECTURE.md`
3. `QBD_DYNAMIC_SYSTEM_SCIENTIFIC_EVIDENCE.md`

If a legacy document conflicts with the Dynamic-QBD documents, the Dynamic-QBD documents control for new research.

---

# 18. Concrete implementation boundary

## Build now

Only the missing factory/evidence layer should be implemented next:

```text
ExpertFamilySpec
ModelGeneration
GenerationRegistry
horizon-specific maturity resolver
causal refit scheduler
generation-specific recalibration
generation-specific threshold provenance
failed-refit fallback
generation/family position lineage
missing downside/wealth-path metrics
Monthly Family Evidence Panel
A/B/C experiment
Gate 1
```

## Reuse now

Reuse without rebuilding:

```text
V5 point-in-time / holdout / hash primitives
H1-H30 / 465-cell QBD surface
opportunity_portfolio_research stateful NAV engine
URTH sleeve accounting
cost/tax accounting
learned-exit stateful replay
replacement-entry mechanics
shadow outcome maturity concept
state/restart primitives
freeze/hash primitives
```

## Do not activate yet

```text
V4 adaptation
activity-based demotion
changepoint routing
promotion lifecycle
regime routing
Champion/hysteresis
uncertainty-LCB capital switching
```

These may be adapted later only if the relevant gate passes.

---

# 19. Final repository principle

The repository should not be organised around the question:

> Which previously successful frozen expert should be kept alive or switched away from?

It should be organised around:

> **Can a deterministic causal model factory repeatedly create valid QBD generations from newly matured information, replay each family with the capital actually available through time, and demonstrate that any additional adaptive mechanism creates incremental future benchmark-relative wealth before that mechanism receives decision authority?**

Therefore the implementation order is:

```text
reuse causal/replay foundation
→ add Family/Generation factory layer
→ establish generation-specific recalibration
→ produce production-faithful family shadow NAV
→ produce Monthly Family Evidence Panel
→ isolate recalibration and refit value
→ run Gate 1
→ add Gate 1B / 2 / 3 only as separate hypotheses
→ reconnect legacy router components only after a gate pass
→ freeze the surviving algorithm
→ one final adaptive holdout
```

No legacy component is removed merely because it is currently deferred, and no legacy component gains authority merely because it already exists.