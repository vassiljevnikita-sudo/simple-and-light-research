# Causal Model Store + Evidence Store + Orchestrator Plan

Status: **NEXT RESEARCH ARCHITECTURE — DESIGN OPEN, NOT YET RUN**  
Authority: Development / Shadow only  
Final prospective holdout: closed

This document replaces the idea that a Development-selected Recipe can be fixed retroactively and then treated as if a historical live system would have known to use it.

Open execution items are tracked in [../TOBECONTINUED.md](../TOBECONTINUED.md).

## 1. Research question

The next valid question is:

> Can a system that begins with only the models buildable from the historical data available at pseudo-live start, then receives new observations and matured evidence through time, create new immutable model Generations and causally choose an appropriate active model better than simple baselines?

This is the architecture we eventually want to operate, so the historical experiment should reproduce its information flow as closely as the available data permits.

## 2. Why the current Fold-Clock diagnostics are not enough

The Fold-Clock Surface Validation 2016-2025 and Fold-Event Counterfactual suites freeze a Recipe that was selected using broader Development evidence overlapping the later evaluation era.

That makes them useful for:

- refit-frequency mechanics;
- calibration/refit contrasts;
- event-level OLD-vs-NEW behavior;
- runtime and replay validation.

It does not make them valid for:

- historical Recipe choice;
- historical model-family choice;
- orchestrator performance;
- “HGB was the right model at date t” claims.

The next suite must build every Recipe/Generation choice from the information set that existed at that date.

## 3. Production concept versus historical approximation

### Intended production concept

At a future production start after 2025, the system can begin with approximately ten years of 2016-2025 data.

Conceptually:

```text
2016 ───────────────────────── 2025
          initial history
                 ↓
          MODEL STORE G0
                 ↓
2026+ incoming observations/evidence
                 ↓
new immutable generations
                 ↓
orchestrator selects active model
```

### Historical limitation

The repository has no clean pre-2016 training history.

Therefore the pseudo-live experiment cannot honestly do:

```text
10-year seed
→ 2016-2025 pseudo-live
```

because the seed data does not exist.

The historical run must use a shorter initial prefix of the available 2016+ data. That weakens realism of the initial model quality but does not weaken causal chronology if handled correctly.

### Unresolved design decision

The exact seed boundary is deliberately **not fixed in this document yet**.

It must be selected using methodological considerations such as:

- minimum viable training history;
- enough remaining pseudo-live time;
- enough matured fold/evidence events;
- comparable availability across H1-H30;
- no inspection of future pseudo-live portfolio performance.

Once chosen, it must be written into the run contract before the first performance run.

## 4. Core architecture

```text
HISTORICAL PREFIX AVAILABLE AT t0
                │
                ▼
       INITIAL MODEL FACTORY
                │
                ▼
     IMMUTABLE MODEL STORE
                │
        pseudo-live begins
                │
                ▼
      incoming observations
                │
                ▼
       matured outcome store
                │
                ▼
          EVIDENCE CLOCK
                │
                ├─────────────► new Generation allowed
                │                         │
                │                         ▼
                │                expanding-window fit
                │                         │
                │                         ▼
                └──────────────── MODEL STORE grows
                                          │
                                          ▼
                                   CAUSAL ORCHESTRATOR
                                          │
                          ┌───────────────┼───────────────┐
                          │               │               │
                      keep old G      use new G      switch Recipe
                          │               │               │
                          └───────────────┴───────────────┘
                                          │
                                          ▼
                                       PORTFOLIO
                                          │
                                          ▼
                                   future outcomes
                                          │
                                          └────► Evidence Store
```

## 5. Initial Model Store

At pseudo-live time `t0`, the factory may only use data available through `t0`.

The initial store should contain a predeclared Recipe universe rather than the ex-post Development winner only.

For each allowed horizon/Recipe:

- Recipe identity;
- algorithm/model family;
- hyperparameters;
- feature schema;
- training start/end;
- target maturity cutoff;
- calibration window;
- calibration parameters;
- model artifact hash;
- training-source hash;
- creation timestamp/cutoff;
- generation ID;
- evidence available at creation.

### Important rule

The Recipe universe must be frozen before pseudo-live outcomes are evaluated.

Do not inspect 2020-2025 and then choose only the Recipes that happened to work there.

The first implementation should prefer the already bounded Ridge/HGB candidate family used by the causal factory unless a separate candidate-universe design is explicitly approved.

## 6. Immutable Generations

A refit creates a new Generation. It never overwrites the old model.

Example:

```text
H10 / Recipe R
├── G0  train through t0
├── G1  train through t1
├── G2  train through t2
└── G3  train through t3
```

Every Generation remains addressable.

This lets the orchestrator decide:

- keep G0;
- move to G1;
- later return to G0/G1 if the contract allows;
- switch to another Recipe generation;
- leave the portfolio unchanged.

A newly built model is a **challenger**, not an automatic replacement.

## 7. Refit training window

The first experiment should use an **expanding window** unless a different window is predeclared before results.

Example:

```text
G0: available historical prefix
G1: same prefix + newly available data through t1
G2: same prefix + data through t2
...
```

Earlier observations are intentionally reused. That is normal for expanding-window refitting.

The consequence is that Generations are highly correlated. Do not count them as independent statistical samples.

A rolling/forgetting window is a separate future hypothesis and should not be mixed into the first causal-store test.

## 8. When new Generations may be created

Calendar-month refitting has already been shown broadly destructive.

Therefore the initial generation-creation clock should be information-based.

Candidate trigger:

> A genuinely new matured causal fold/evidence state becomes available.

The trigger must be derived from the sorted matured evidence/fold identity, not from calendar passage.

However, **generation creation** and **generation activation** are different:

```text
new evidence
    ↓
new model may be built
    ↓
orchestrator may still keep incumbent
```

This is one of the main lessons from the Fold-Event Counterfactual.

## 9. Evidence Store

For every Recipe/Generation, preserve evidence chronologically.

Minimum evidence fields:

- model/generation ID;
- Recipe ID;
- H;
- prediction date;
- target terminal date;
- matured-at date;
- prediction score/rank;
- realized benchmark-relative outcome;
- calibration state used at prediction time;
- active/eligible signal state;
- portfolio/trade lineage where relevant;
- evidence source/fold ID;
- hashes/provenance.

At assessment date t, the orchestrator may only see rows with:

```text
matured_at <= t
```

No future outcome from H30 may be visible merely because an H1 target from the same decision date has already matured.

## 10. Orchestrator role

The orchestrator selects among Models/Generations that existed at that historical time.

It may decide:

- KEEP incumbent;
- ACTIVATE newer Generation of same Recipe;
- SWITCH to another causally available Recipe;
- possibly choose no stock model / benchmark-only state if that is a predeclared arm.

The orchestrator must not fit or tune itself on the same future interval it is evaluated on.

### First-version principle

Do not begin with another high-capacity ML selector.

The repository has already tested many selector/controller families and found weak robustness.

The first Model-Store suite should prioritize a transparent predeclared policy so the architecture can be falsified cleanly.

Examples of permissible first-stage signals might include matured:

- benchmark-relative model performance;
- ranking quality;
- downside/risk;
- evidence amount/confidence;
- generation age.

The exact formula and any thresholds must be frozen **before** pseudo-live results.

A nested/prequential learned orchestrator can be a later experiment if the baseline architecture shows exploitable oracle/model-store headroom.

## 11. Required baseline arms

The suite should compare matched causal policies.

### S0 — Initial Static

Build the initial Model Store at t0. Choose the predeclared initial selection rule and never change active model afterwards.

Purpose: “do nothing after launch” baseline.

### S1 — Auto Refit Same Recipe

Keep the initial Recipe. At every allowed evidence event, create and automatically activate the newest Generation.

Purpose: isolates automatic expanding-window maintenance.

This is the Model-Store analogue of Fold-Clock activation without Recipe switching.

### S2 — Model-Store Orchestrator

Model Store grows. Old and new Generations remain available. The causal orchestrator selects among allowed models.

Purpose: target architecture.

### S3 — Simple pooling/combination baseline

If computationally feasible, include a simple predeclared combination/equal model pool that does not require fitted routing.

Purpose: strong benchmark against noisy selection, given prior evidence that pooling can be hard to beat.

### O — Ex-post Oracle diagnostic

After the fact, choose the best model/Generation for each future interval.

The Oracle:

- has zero authority;
- is never a deployable arm;
- exists only to measure whether the Model Store contained useful alternatives.

Interpretation:

```text
Oracle >> Orchestrator
→ store contains useful choices; selection is weak

Oracle ≈ Orchestrator ≈ poor
→ model store/factory lacks useful alternatives

Orchestrator > Auto Refit
→ selective activation has demonstrated value
```

## 12. Recipe reselection decomposition

The architecture should keep two changes distinguishable:

1. model age / new Generation of same Recipe;
2. Recipe identity.

A useful matched decomposition at each event is:

```text
incumbent Recipe + incumbent Generation
        vs
same Recipe + fresh Generation
        vs
new causal Recipe + fresh Generation
```

This lets us estimate:

- value of refitting;
- value of Recipe reselection;
- value of orchestration.

Do not combine them into one opaque “dynamic” arm only.

## 13. Portfolio dimensions to freeze

The first orchestrator experiment is not another portfolio-policy search.

Unless the final run contract says otherwise, keep the existing established portfolio dimensions fixed:

- equal-active allocation;
- ignore-new replacement;
- current cost contract;
- current benchmark;
- same next-open timing;
- same sleeve semantics;
- FIXED exits for the cleanest entry-model orchestration test.

Learned Exit can be added later once causal prediction coverage exists for counterfactual entries.

N/H/D may still define the model/family universe, but do not tune their inclusion from the pseudo-live result.

## 14. Causal calibration

Calibration must follow the active Generation's information set.

No calibration observation may have a terminal date after the calibration cutoff.

The first suite must predeclare whether:

- each fresh Generation gets a fresh causal calibration;
- older Generations retain their original calibration when reused;
- an old model may receive a new calibration without a new fit.

Given the previous evidence that monthly recalibration is destructive, the safest default hypothesis is:

> Calibration is generation-specific and changes only when a new Generation is created, not every calendar month.

This remains a design choice to encode in the run contract before execution.

## 15. Evaluation units and statistics

Do not treat Model×Family×Day rows as independent samples.

Primary inference units should be temporal/evidence events and, where relevant, predeclared structural plateaus.

Report at least:

- benchmark excess;
- terminal wealth;
- relative MaxDD;
- expected shortfall / downside where available;
- turnover/trades/costs;
- selection switch count;
- generation age while active;
- regret versus Oracle;
- captured Oracle alpha;
- fraction of periods where newest model is chosen;
- fraction where an older Generation is retained;
- performance by event/subperiod;
- model/Recipe concentration;
- robustness across reasonable predeclared seed sensitivity runs.

Use time/block bootstrap rather than IID row bootstrap for system-level claims.

## 16. Seed sensitivity without seed selection

Because no true ten-year historical seed exists, it may be useful to run more than one predeclared seed length.

If so, the purpose must be **robustness**, not choosing the best historical seed.

Before results, specify a small set such as:

```text
short seed
primary seed
long seed
```

with exact dates derived from training sufficiency and remaining pseudo-live evidence, not from performance.

Then report all runs.

Do not pick the best-performing seed and call it the architecture.

## 17. Holdout and time-window discipline

The pseudo-live run itself is still Development research because the architecture is being developed now.

Therefore:

- do not call it fresh final OOS;
- do not open 2026-07-25+;
- do not tune on one pseudo-live period and re-label the same period as validation;
- if the architecture is changed after seeing pseudo-live results, record the change and treat prior outcomes as Development evidence.

The goal is to make the causal mechanics valid now so a later prospective window can become meaningful.

## 18. Data/implementation reuse

Reuse existing components where their contracts are compatible:

- H1-H30 feature/target adapters;
- causal Recipe candidate evidence;
- model fitting;
- generation registry/identity;
- score materialization;
- maturity logic;
- portfolio replay;
- cost/benchmark accounting;
- streaming aggregation;
- process-pool scheduling;
- checkpoint/resume;
- run-contract hashing.

Do not rewrite the full Dynamic-QBD pipeline from scratch just to create the Model Store.

The main new abstractions should be:

- historical initial-store builder;
- immutable multi-generation store view as-of t;
- evidence-store cursor as-of t;
- generation-creation event planner;
- orchestrator decision contract;
- pseudo-live replay coordinator;
- matched baseline/oracle reporting.

## 19. Required audits

The suite should fail closed unless it can prove:

### Time

- every Generation training end <= creation cutoff;
- every target used in training/calibration/evidence matured before the relevant decision;
- pseudo-live outcomes never feed an earlier choice.

### Store

- old Generations are immutable;
- model IDs/hashes are unique and stable;
- newly created models do not overwrite incumbents;
- a model cannot be selected before creation.

### Recipe

- candidate universe frozen before pseudo-live;
- Recipe reselection uses only evidence available as-of the event;
- no Development-final winner is injected retroactively.

### Orchestrator

- decision inputs have explicit timestamps;
- no target-only/post-event feature is present;
- rule/weights are predeclared;
- previous decision state is replayable.

### Portfolio

- OLD/NEW comparisons share the same pre-decision state where matched;
- next-open execution preserved;
- accounting identity preserved;
- costs preserved;
- max names/sleeve preserved.

### Holdout

- no 2026-07-25+ final holdout read;
- no promotion/capital authority.

## 20. Compact result package

Expected compact artifacts should include:

- `summary.json`;
- `REPORT.md`;
- `run-contract.json`;
- `contract-audit.json`;
- `initial-model-store.csv/json`;
- `generation-ledger.csv`;
- `evidence-event-ledger.csv`;
- `orchestrator-decisions.csv`;
- `arm-comparison.csv`;
- `event/subperiod-summary.csv`;
- `oracle-regret-summary.csv`;
- `seed-sensitivity.csv` if predeclared;
- runtime/telemetry summary.

Large model stores, predictions, per-family NAV and checkpoints remain local.

## 21. Falsification

The target architecture should be rejected or simplified if:

- the Oracle itself has little advantage over the static store;
- automatic refits do not improve and the orchestrator cannot avoid harmful ones;
- orchestrator gains vanish after costs;
- gains depend on one seed split;
- gains are concentrated in one event/ticker/Recipe;
- the orchestrator does not beat a simple static or pooled baseline;
- selection frequently relies on evidence too sparse to distinguish candidates;
- results require post-hoc horizon/Recipe exclusions;
- apparent gains disappear under temporal robustness.

A negative result is useful. It means the store needs better models or the system should remain simpler.

## 22. Immediate next design decisions

These remain open and should be resolved before implementation is called final:

1. exact historical seed cutoff(s);
2. minimum training-history requirement per H;
3. frozen initial Recipe universe;
4. generation-creation trigger;
5. generation-specific calibration rule;
6. first deterministic orchestrator scoring rule;
7. baseline/Oracle set;
8. evaluation gates and block-bootstrap contract;
9. whether first run is FIXED-only;
10. exact local source-artifact reuse contract.

Track those decisions in [../TOBECONTINUED.md](../TOBECONTINUED.md). Once agreed, encode them in the runner's semantic run contract so they cannot drift during the pseudo-live replay.
