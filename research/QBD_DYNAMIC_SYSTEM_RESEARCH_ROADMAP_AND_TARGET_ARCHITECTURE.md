> **Historical roadmap:** This document predates the completed Full Development, regime/opportunity/consensus diagnostics, full-space Fold-Clock validation and Fold-Event Counterfactual. Preserve it as design history, but use [DYNAMIC_QBD_CURRENT_STATE.md](DYNAMIC_QBD_CURRENT_STATE.md), [DYNAMIC_QBD_RESULTS_INDEX.md](DYNAMIC_QBD_RESULTS_INDEX.md) and [CAUSAL_MODEL_STORE_ORCHESTRATOR_PLAN.md](CAUSAL_MODEL_STORE_ORCHESTRATOR_PLAN.md) for current decisions.

# QBD Dynamic System — research roadmap, falsification gates, and conditional target architecture

## Status

Normative research roadmap for the next QBD research phase.

This document consolidates the architectural conclusions reached after the Top-10/QBD audits, the dynamic-refit discussion, the scientific-evidence review, and the subsequent risk/portfolio discussion.

It must be read together with:

- `research/QBD_DYNAMIC_SYSTEM_SCIENTIFIC_EVIDENCE.md`

The scientific-evidence note establishes that dynamic model/factor performance is plausible but that hard winner selection is not scientifically established. This document defines how the repository should obtain the missing empirical evidence before committing to a final routing architecture.

The key rule is:

> **Do not build a more sophisticated Champion–Challenger system until the repository demonstrates that information available at time `t` has reliable forward value for selecting QBD families after `t`.**

The final architecture is therefore conditional on research gates. Components that fail their gate are removed rather than retained by default.

---

# 1. Consolidated design decisions

## 1.1 The trading system is the model factory, not a single fitted model

The long-term research object is:

```text
QBD_DYNAMIC_SYSTEM

= Training Policy
+ Refit Cadence
+ Calibration Policy
+ QBD Candidate Families
+ Shadow Evaluation
+ Optional Family Selection
+ Optional Switch Policy
+ Portfolio Execution
```

A fitted H11, H24, H28, or any other concrete artifact is replaceable. The durable algorithm is the deterministic process that creates, evaluates, and possibly selects new generations.

Intended causal process:

```text
new fully matured data
        ↓
causal refit of every allowed QBD family
        ↓
causal recalibration of every new fit
        ↓
new generation-specific threshold / policy
        ↓
all current generations execute in shadow
        ↓
only matured outcomes become evidence
        ↓
optional selection decision
        ↓
stateful portfolio execution
```

This replaces the previous architectural emphasis on keeping one old fitted model alive through Memory Controllers, lambda adaptation, adaptation gates, and post-hoc threshold repair.

The old controller work remains useful historical evidence, but it is not the target adaptation mechanism.

---

## 1.2 Refit and recalibration are one causal unit

A new model generation must not blindly inherit an old absolute score threshold.

A new fit can change:

- score distribution;
- score dispersion;
- ranking behaviour;
- calibration;
- optimal operating point.

Therefore:

```text
new fit
→ new causal calibration
→ new generation-specific threshold
```

The calibration policy itself is frozen before final evaluation. Only the data entering that policy change through time.

---

## 1.3 Family identity and generation identity are separate

With recurring refits, `H24` is not a single artifact.

Example:

```text
H24_fit_2025_01
H24_fit_2025_02
H24_fit_2025_03
...
```

Two levels of evidence are required.

### Family evidence

Question:

> Is H24, or another structurally defined QBD family, robust across many causal refits and market periods?

### Current-generation evidence

Question:

> How healthy is the exact current H24 fit created from the latest matured data?

Historical family evidence must not be fully attributed to a fresh generation, but a fresh generation must also not erase everything previously learned about the family.

---

## 1.4 Champion–Challenger is a hypothesis, not the next implementation step

A hard capital switch only makes sense if relative family performance is prospectively predictable.

The primary gate question is:

> **Can information observable up to assessment time `t` predict which QBD family will achieve greater benchmark excess over the next 1M and/or 3M?**

If the answer is no, performance-driven Champion–Challenger development stops.

A failed Champion gate does **not** automatically invalidate causal refit and recalibration. The factory can still be valuable even if dynamic family selection is not.

---

# 2. Risk principles for QBD

## 2.1 Sharpe and total volatility are not target objectives

QBD does not treat upside volatility as risk.

The research programme therefore must not optimise for smoothness merely because returns fluctuate strongly upward.

Primary risk concepts are asymmetric and path dependent:

- relative downside deviation;
- relative Sortino-type measures;
- relative maximum drawdown;
- Conditional Drawdown at Risk (CDaR);
- expected shortfall of benchmark excess;
- drawdown duration;
- recovery time;
- capital impairment / time under water.

Where possible these are defined relative to the benchmark sleeve rather than as absolute market-risk measures.

---

## 2.2 Few trades are not automatically investment risk

Sparse trading can be correct behaviour.

A family that produces only a small number of entries may simply wait for rare high-conviction opportunities.

Therefore:

```text
few trades
≠ bad model
≠ automatic confidence decay
≠ automatic demotion
```

Few matured outcomes instead imply primarily **epistemic uncertainty**: the true edge is estimated with less evidence.

Calendar inactivity alone must not be used as a negative health signal.

---

## 2.3 Diversification is not an optimisation target

QBD must not force inferior positions merely to diversify ticker or sector exposure.

If the best causal opportunities are concentrated in one sector, the system is allowed to take those opportunities.

Therefore no production rule should say:

```text
sector concentration high
→ reject otherwise valid alpha
```

Concentration is still useful as information about:

- common downside exposure;
- dependence between apparent bets;
- the strength of the scientific evidence supporting a claimed general edge.

It is a diagnostic and uncertainty input, not an automatic portfolio penalty.

---

## 2.4 Historical winner concentration is a robustness diagnostic

If a large fraction of historical alpha comes from one ticker or one trade, this does not imply that the trade was wrong.

A leave-one-ticker/trade/fold analysis answers a different question:

> Would the evidence for a generalisable model-family edge still exist without this one historical outcome?

Therefore concentration tests are used to classify evidence strength, not to rewrite historical trades or impose ex-post diversification constraints.

---

## 2.5 Threshold and `max_names` are not breadth controls

The system must not loosen its threshold merely to create more trades.

Likewise, a larger `max_names` is not better because it creates diversification.

The correct economic question is whether the marginal opportunity still carries positive expected benchmark excess after costs.

Conceptually:

```text
admit marginal trade only if
expected excess after costs > 0
```

The optimal policy may therefore be sparse or dense depending on the realised signal structure.

---

# 3. Portfolio wealth is the economic source of truth

## 3.1 Stateful NAV is mandatory

The production-faithful replay must update the actual available capital through time:

```text
current NAV
    ↓
available buying power
    ↓
position sizing
    ↓
realised / marked P&L
    ↓
new NAV
    ↓
next opportunities use the new capital base
```

A drawdown therefore matters economically even if the final return is positive, because subsequent opportunities are taken with a smaller capital base until recovery.

Example:

```text
100 → 70 → 115
```

The final wealth is +15%, but the portfolio had to earn approximately +64.3% from the trough to reach 115. During the recovery phase all percentage-based position sizes operated on the impaired capital base.

This effect must emerge naturally from stateful replay rather than being approximated by a post-hoc risk penalty.

---

## 3.2 Evaluation hierarchy

The preferred economic hierarchy is:

```text
1. Final portfolio wealth
2. Final wealth relative to MSCI/URTH benchmark
3. Full relative-wealth path
4. Relative drawdown / capital impairment
5. CDaR / downside / recovery duration
6. Trade-level and prediction-level metrics as diagnostics
```

CAGR excess remains useful, but it is not sufficient on its own.

The portfolio-level question is:

> How much real wealth did the causal system create from the capital that was actually available at every point in time, and how severely did it impair its ability to exploit later opportunities?

---

## 3.3 Investment risk vs epistemic risk

These must be kept separate.

### Investment risk

Risk to actual capital and the future ability to deploy it:

- relative drawdown;
- CDaR;
- downside deviation;
- expected shortfall;
- drawdown duration;
- recovery time;
- joint tail exposure of simultaneously held positions.

### Epistemic risk

Uncertainty that the measured edge is real or persistent:

- number of matured independent time blocks;
- number of matured trades/outcomes;
- fold stability;
- generation/refit stability;
- sensitivity to single winners;
- parameter-surface robustness;
- prediction uncertainty;
- selection uncertainty.

A model can have low realised investment risk but high epistemic uncertainty, or vice versa.

---

# 4. Research questions to answer before the final architecture is frozen

The research programme must separate the value of different mechanisms rather than testing one giant adaptive system.

## RQ0 — replay fidelity

Can the repository reproduce a fully causal, production-faithful portfolio path for every candidate family/generation?

If not, all later results are invalid.

## RQ1 — value of recalibration

Does rolling causal recalibration improve economic OOS performance versus reusing a frozen calibration?

## RQ2 — value of refitting

Does rolling causal refit plus recalibration improve economic OOS performance versus a frozen model with rolling recalibration?

## RQ3 — performance predictability

Does historical family performance predict future 1M/3M benchmark excess or relative ranking?

## RQ4 — downside/wealth-path predictability

Does past downside and capital-path information add forward predictive value beyond raw performance?

## RQ5 — current-fit health

Do matured prediction-level diagnostics of the current generation predict future family performance before sparse trades reveal deterioration?

## RQ6 — market/regime state

Does current market state add incremental predictive value after performance and current-fit health are already accounted for?

## RQ7 — capital selection

If predictive information exists, does a dynamic selector create more net portfolio wealth than simple static/hold/combination baselines?

## RQ8 — Champion–Challenger

If dynamic selection works, does hard one-Champion allocation with hysteresis outperform softer or simpler selection rules after switching costs and full stateful execution?

---

# 5. Phase 0 — establish the causal model-factory and replay contract

No meta-model result is admissible until this phase passes.

## 5.1 Required durable entities

### `ExpertFamilySpec`

Frozen structural recipe, for example:

- family ID;
- model family;
- prediction horizon;
- holding/exit policy;
- `max_names` contract;
- feature schema;
- model hyperparameters or frozen selection rule;
- training-window rule;
- calibration-window rule;
- threshold-calibration rule;
- benchmark/cost/tax contracts.

### `ModelGeneration`

Concrete historical fitted instance:

- generation ID;
- family ID;
- refit timestamp;
- train start/end;
- latest matured label cutoff;
- calibration start/end;
- model artifact fingerprint;
- threshold/policy fingerprint;
- data fingerprints;
- validation status.

---

## 5.2 Causal maturity rule

For every assessment/refit date:

```text
terminal_date <= information_cutoff
```

must hold for every outcome used in training, calibration, family health, and selector features.

H1 and H30 therefore have different latest usable labels at the same wall-clock date.

A single naive global recent-data cutoff is not acceptable.

---

## 5.3 Full portfolio replay requirements

The replay must include:

- exits before entries according to the actual execution contract;
- freed capital becoming available according to the production timing rule;
- replacement entries where the real policy permits them;
- fixed and learned exit policies as actually deployed;
- `max_names` enforcement;
- benchmark/URTH sleeve;
- existing cost model;
- existing tax approximation where applicable;
- actual NAV-dependent sizing;
- deterministic state persistence and restart.

Frozen-entry early-exit overlays that do not create replacement trades are not sufficient as final production-policy evidence.

---

## 5.4 Phase-0 mandatory tests

1. **Future-mutation test** — modifying all post-cutoff data cannot change any pre-cutoff fit, calibration, decision, or NAV path.
2. **Target-maturity test** — an outcome with `terminal_date > cutoff` is impossible to consume.
3. **Generation reproducibility** — same code/data/config/seed creates the same generation fingerprint and predictions.
4. **Failed-refit fallback** — a failed generation cannot silently replace the last valid generation.
5. **Continuous vs restart parity** — refits, predictions, trades, NAV, and evidence are identical after resume.
6. **N-policy behavioural difference** — where opportunities permit it, N1 and N6 produce genuinely different portfolio states.
7. **Exit behavioural difference** — learned/fixed exit policies actually alter position lifecycle when their rules differ.
8. **Benchmark-sleeve accounting** — idle capital and active capital reconcile exactly to total NAV.
9. **Cost/tax reconciliation** — portfolio-level P&L equals constituent accounting.
10. **Open-position lineage** — every live/shadow position records the generation/family and exit policy that opened it.

### Phase-0 output

A monthly historical panel where every family has a current causal generation and a production-faithful shadow NAV path.

No Champion research proceeds if Phase 0 is not green.

---

# 6. Phase 1 — isolate recalibration and refit value

The first economic experiment compares three core arms under identical execution.

## Arm A — frozen model + frozen calibration

Historical baseline.

## Arm B — frozen model + rolling causal recalibration

Isolates the effect of recalibration.

## Arm C — rolling causal refit + rolling causal recalibration

Measures the incremental value of the model factory before any dynamic family selection.

Interpretation:

```text
B - A = incremental value of recalibration
C - B = incremental value of refitting
```

This interpretation is only valid if all other contracts are identical.

### Required outputs per arm

- terminal portfolio wealth;
- benchmark-relative terminal wealth;
- CAGR excess;
- full daily/periodic NAV series;
- relative MaxDD;
- CDaR;
- downside deviation / relative Sortino diagnostic;
- expected shortfall diagnostic;
- drawdown duration and recovery time;
- turnover and costs;
- trade count;
- per-family/per-generation provenance.

### Decision rule

If C does not produce robust improvement over B under outer OOS evaluation, recurring refit is not promoted merely because it is conceptually attractive.

If B improves A but C does not improve B, retain recalibration without mandatory recurring model refit.

---

# 7. Phase 2 — Gate 1: can past performance predict future performance?

This is the primary gate for performance-based Champion–Challenger.

## 7.1 Predictors

Use only historical economic performance information observable at assessment time.

Candidate features:

- trailing 1M benchmark excess;
- trailing 3M benchmark excess;
- trailing 6M benchmark excess;
- trailing 12M benchmark excess;
- EWMA benchmark excess;
- recent-minus-long-term excess;
- positive-period fraction;
- historical relative drawdown features only in a separately labelled Gate 1B arm.

Do not initially use:

- H/D/N identity as a predictive shortcut;
- ticker identity;
- market regime;
- VIX/volatility state;
- Rank IC;
- calibration health;
- prediction residuals;
- feature drift;
- sector composition.

The purpose is to determine whether **past performance itself** contains prospective information.

---

## 7.2 Targets

Separate experiments:

```text
Future 1M benchmark excess
Future 3M benchmark excess
```

For a dynamically refitted family, the target represents the forward performance of the family under the frozen refit/recalibration policy, not the performance of one permanently frozen monthly artifact.

For a 3M target, each scheduled new generation that would causally appear during those three months must appear in the replay exactly as it would live.

---

## 7.3 Tasks

Evaluate separately:

1. **Direction** — probability future family excess is positive.
2. **Magnitude** — expected future excess.
3. **Cross-sectional ranking** — whether predicted better families actually rank better after the assessment date.

Ranking is the most important task for later Champion selection.

---

## 7.4 Statistical unit

Time is the effective independent sample.

```text
10,000 policy rows × 50 months
≠ 500,000 independent observations
```

Policies share the same market and many are near duplicates.

No random row split is permitted.

The selector itself must use expanding/rolling temporal walk-forward evaluation.

Inference must use time-aware methods such as temporal blocks, block bootstrap, or suitable HAC treatment rather than IID row assumptions.

---

## 7.5 Family-first evaluation

The primary unit is the structurally defined QBD family/plateau, not every tiny policy variation as an independent challenger.

Family definitions must be frozen without using future evaluation data.

Policy-level selection within a family is secondary research only.

---

## 7.6 Baselines

At minimum compare against:

- benchmark only;
- static best-development family;
- previous incumbent held;
- trailing 1M winner;
- trailing 3M winner;
- trailing 6M winner;
- long-run historical best under the same information cutoff;
- equal-weight family combination as a scientific null/counterfactual baseline.

Equal weighting is a comparator, not a statement that diversification is an objective of the final system.

---

## 7.7 Primary economic metrics

Do not accept classification accuracy alone.

Report:

- monthly cross-sectional Spearman rank IC;
- realised excess of predicted top family;
- top-minus-bottom family spread;
- selected-family portfolio wealth;
- selected-family benchmark-relative wealth;
- regret versus the ex-post oracle;
- fraction of oracle alpha captured;
- switching turnover and costs where relevant;
- result stability across time blocks.

### Gate-1 pass concept

Before looking at the outer result, preregister numerical pass thresholds based on the realised number of independent assessment periods.

At minimum, a pass requires all of the following qualitatively:

1. forward ranking information is positive and not confined to one isolated time block;
2. the selected top family creates positive incremental net wealth versus simple no-selector baselines;
3. the result is not an artefact of random row dependence or an invalid effective sample size;
4. the conclusion survives reasonable cost stress;
5. evidence strength is explicitly downgraded if one ticker/trade/fold explains nearly the entire effect.

If Gate 1 fails, do **not** build a trailing-performance Champion–Challenger router.

---

# 8. Phase 3 — Gate 1B: does downside / wealth-path information add value?

This gate tests the risk concept actually relevant to QBD.

It must not optimise Sharpe or total volatility.

Candidate historical features:

- relative downside deviation;
- relative Sortino-type statistic;
- relative MaxDD;
- CDaR;
- expected shortfall of benchmark excess;
- current relative drawdown;
- drawdown duration;
- recovery duration;
- time under previous relative-wealth peak;
- capital-impairment buckets.

Question:

> Given the same historical return information, does the **shape of capital impairment** improve prediction of future family performance or future capital loss?

This can succeed even if raw past-return ranking is weak.

A positive result would justify Downside/Wealth-Path Health as a future selector input.

A negative result means these metrics remain portfolio diagnostics rather than routing features.

---

# 9. Phase 4 — Gate 2: current-generation prediction health

Only after the performance-only experiment is known should the richer prediction panel be introduced.

Potential causal features from fully matured shadow predictions:

- recent Rank IC;
- top-score realised excess;
- calibration error/slope;
- score monotonicity;
- prediction residual drift;
- score dispersion;
- top-vs-median realised outcome;
- generation-level degradation versus its own calibration expectations.

Purpose:

> Detect deterioration or improvement in a sparse trading model before another actual trade occurs.

A failure of Gate 1 does not imply Gate 2 must fail.

If Gate 2 works while Gate 1 fails, the future system may use current-fit health rather than trailing realised portfolio performance as its selection signal.

---

# 10. Phase 5 — Gate 3: market / regime information

Only after the previous gates are measured may current market state be added.

Potential predeclared state variables:

- benchmark trend;
- volatility/stress state;
- market breadth;
- cross-sectional dispersion;
- liquidity state where causally available.

Question:

> Does market state add incremental forward information beyond historical family performance and current-fit health?

The regime layer is retained only if its incremental OOS value is demonstrated.

Do not build a complex regime controller merely because regimes are economically plausible.

---

# 11. Phase 6 — selector comparison

Only if at least one predictive gate passes should capital-selection rules be tested.

Compare increasingly complex selectors:

```text
S0  static best-development family
S1  previous incumbent held
S2  simple trailing winner
S3  performance-only predicted ranking
S4  + downside/wealth-path health, if Gate 1B passed
S5  + current-generation health, if Gate 2 passed
S6  + market/regime state, if Gate 3 passed
```

Each added layer must show incremental outer-OOS economic value.

A layer that fails its ablation is removed from the target architecture.

---

# 12. Phase 7 — Champion–Challenger and hysteresis

Hard one-Champion deployment is tested only after forward family ranking has demonstrated value.

Candidate rules may then require a challenger to satisfy several conditions before receiving new-entry authority:

- positive predicted incremental excess over incumbent;
- sufficient uncertainty-adjusted evidence;
- persistence over multiple assessment dates if justified;
- switching benefit greater than switching cost/hurdle;
- no disqualifying capital-integrity failure;
- generation and data provenance valid.

Do not introduce a calendar inactivity penalty merely because the Champion has not traded.

### Open-position rule

When a Champion switch occurs at time `t`:

```text
new entries from t onward
→ new Champion

positions opened by old Champion
→ retain original family/generation lineage
→ exit under the exit policy that governs that position
```

No forced liquidation is created solely by a Champion identity change unless a separately researched safety rule requires it.

---

# 13. Optional active-risk scaling is a separate hypothesis

Risk-aware capital sizing may be tested, but it is not automatically part of QBD.

The research question is not whether volatility should be reduced. It is:

> Can causal downside/capital-impairment information improve final portfolio wealth by changing the amount of active capital deployed without destroying valid high-upside opportunities?

If tested, compare it as a separate arm so its value is not confused with refit or family selection.

If no incremental wealth benefit exists, retain risk metrics as diagnostics/guards only.

---

# 14. Required result artefacts

Every research phase should emit machine-readable and human-readable evidence.

Suggested artefacts:

## Factory / lineage

- `family_registry.json`
- `generation_registry.parquet`
- `refit_history.parquet`
- `calibration_history.parquet`
- `generation_fingerprints.json`

## Shadow execution

- `family_shadow_nav.parquet`
- `family_shadow_positions.parquet`
- `family_shadow_trades.parquet`
- `family_matured_outcomes.parquet`

## Risk / wealth path

- `family_relative_wealth.parquet`
- `family_drawdown_metrics.csv`
- `family_downside_metrics.csv`
- `family_epistemic_diagnostics.csv`

## Predictability gates

- `gate1_predictions.parquet`
- `gate1_monthly_rank_metrics.csv`
- `gate1_baseline_comparison.csv`
- `gate1_block_bootstrap.json`
- corresponding Gate 1B / Gate 2 / Gate 3 files

## Selector evaluation

- `selector_nav.parquet`
- `selector_decisions.parquet`
- `selector_switch_log.csv`
- `selector_baseline_comparison.csv`
- `selector_cost_stress.csv`

## Human-readable reports

Each phase produces a concise Markdown report containing:

- exact Git SHA;
- data fingerprints;
- information cutoff rules;
- primary hypotheses;
- predeclared primary metrics;
- result tables;
- failed/passed gates;
- known limitations;
- explicit decision: continue / simplify / stop.

---

# 15. Stopping and falsification rules

The programme must accept negative results.

## Stop/simplify rule A — recalibration

If rolling recalibration does not improve or stabilise OOS economics versus frozen calibration, do not retain rolling calibration merely by assumption.

## Stop/simplify rule B — refit

If rolling refit + recalibration does not improve on rolling recalibration alone, the final system does not require recurring full refits.

## Stop rule C — performance selector

If past family performance does not produce robust forward 1M/3M ranking/economic value, stop performance-driven Champion–Challenger development.

## Stop rule D — downside selector

If downside/wealth-path state has no incremental forward value, retain it only as reporting/risk diagnostics.

## Stop rule E — current-fit health

If prediction-health metrics do not forecast forward family quality, do not add them to the selector.

## Stop rule F — regime layer

If regime state adds no incremental OOS value, omit the regime layer.

## Stop rule G — hard Champion

If one-Champion/hysteresis does not beat simpler selection/hold baselines after costs, do not deploy hard switching.

A simpler final architecture is a successful scientific outcome if the more complex layers fail.

---

# 16. Development evaluation and final holdout

## 16.1 Development phase

Allowed:

- build/fix causal infrastructure;
- compare predeclared research arms;
- determine whether gates pass;
- choose a small number of system-level design constants such as refit cadence;
- run robustness and leakage tests.

All such iterations are development evidence, not final OOS evidence.

---

## 16.2 Freeze the algorithm, not future monthly artifacts

Before final holdout, freeze:

- Family Specs;
- candidate universe;
- feature schema;
- training recipe;
- refit cadence;
- label-maturity logic;
- training-window rule;
- calibration-window rule;
- recalibration algorithm;
- threshold rule;
- portfolio/execution contract;
- cost/tax contract;
- risk metrics used by decisions;
- selector features;
- selector model/rule;
- switch/hysteresis rule if applicable;
- all pass/fail thresholds;
- holdout boundaries;
- initial pre-holdout state.

Do **not** freeze the concrete future monthly generations, because the live algorithm is explicitly allowed to create them from newly matured data.

During final holdout, every new generation is created causally under the frozen algorithm and receives append-only lineage/provenance.

---

## 16.3 Final holdout is opened once

The final holdout evaluates the complete adaptive algorithm:

```text
frozen system rules
        ↓
causal refits/recalibrations inside holdout
        ↓
shadow evidence as it matures
        ↓
selector decisions, if enabled
        ↓
stateful portfolio NAV
```

No architecture or threshold is changed after inspecting the final result.

If final holdout fails, the result is recorded as failure rather than repaired against the same holdout.

---

# 17. Conditional target architecture

The most complete target architecture is intentionally conditional:

```text
FROZEN QBD FAMILY / TRAINING SPECS
                ↓
     Causal Matured-Data Builder
                ↓
       Periodic Family Refit
                ↓
     Per-Generation Recalibration
                ↓
       Model Generation Registry
                ↓
   Current Generation per Family
                ↓
    Production-Faithful Shadow NAV
                ↓
       Matured Causal Evidence
                ↓
    ┌───────────┼──────────────┐
    │           │              │
Family      Downside /     Current-Fit
History     Wealth Path      Health
    │           │              │
    └───────────┼──────────────┘
                │
         [Market State]
        only if Gate 3 passes
                ↓
       Predictive Family Model
       only if gates pass
                ↓
      Candidate / Challenger Set
                ↓
    Champion + Hysteresis
 only if hard-selection test passes
                ↓
        NEW ENTRY AUTHORITY
                ↓
      Stateful Portfolio Engine
                ↓
       Total Portfolio Wealth
                ↺
```

Square-bracketed / conditional layers are not guaranteed components. They exist only if their research gate provides incremental outer-OOS economic value.

---

# 18. Minimum viable architecture if selection fails

A negative selector result does not collapse the project.

A scientifically defensible simpler endpoint is:

```text
Frozen Family/Training Policy
        ↓
Periodic Causal Refit
        ↓
Per-Generation Recalibration
        ↓
Chosen static family / predefined policy
        ↓
Stateful Portfolio Engine
        ↓
Benchmark Sleeve + valid opportunities
```

If refit itself fails but recalibration works, simplify further.

The architecture should contain only mechanisms that survive their own causal ablation.

---

# 19. Recommended implementation order

## P0 — evidence engine

1. Family specification and generation registry.
2. Horizon-specific maturity-aware causal dataset builder.
3. Periodic refit runner.
4. Per-generation recalibration.
5. Full production-faithful shadow portfolio replay.
6. Stateful NAV / benchmark-sleeve reconciliation.
7. Generation/family lineage and fingerprints.
8. Future-mutation, maturity, restart, accounting tests.

## P1 — factory-value experiment

9. Run Arms A/B/C.
10. Produce wealth-path and downside evidence.
11. Decide whether recalibration and refit deserve promotion.

## P2 — predictability gates

12. Build performance-only family panel.
13. Gate 1: 1M and 3M walk-forward direction/magnitude/ranking.
14. Gate 1B: add downside/wealth-path state.
15. Gate 2: add current-fit prediction health.
16. Gate 3: add market state only if still justified.

## P3 — selector research

17. Compare static/hold/trailing/meta-selector baselines.
18. Test hard Champion only if predictive selection exists.
19. Add hysteresis/switch costs/open-position lineage.
20. Test optional risk-aware sizing separately.

## P4 — freeze and final evaluation

21. Freeze the complete algorithm and manifests.
22. Verify final-holdout isolation.
23. Run the adaptive holdout once.
24. Publish final result without post-hoc repair against the same holdout.

---

# 20. Success is not defined as maximum complexity

The research programme succeeds if it identifies the smallest causal architecture that produces robust net wealth improvement.

Possible valid conclusions include:

```text
A. Refit + recalibration + Champion works
B. Refit + recalibration works, Champion does not
C. Recalibration works, refit/Champion do not
D. Static family remains strongest
E. Dynamic selection works only with current-fit health, not past returns
F. Hard Champion loses to a simpler combination/hold rule
```

Every one of these is an informative result.

The repository must prefer the simplest architecture supported by strict outer-OOS evidence.

---

# 21. Final research principle

The core question is no longer:

> Which QBD model had the best historical return?

It is:

> **Can the complete causal system use only information genuinely available at time `t` to create more future benchmark-relative portfolio wealth, while correctly accounting for capital impairment and model uncertainty, than simple static alternatives?**

The order is therefore:

```text
causal factory fidelity
→ isolate recalibration value
→ isolate refit value
→ prove or falsify forward family predictability
→ test downside/current-fit/regime information incrementally
→ only then build selection/hysteresis
→ freeze the full algorithm
→ one final adaptive holdout
```

No later layer is entitled to exist merely because it is theoretically attractive.