# TOP10 QbD Router V1 — Implementation Plan

Status: `IMPLEMENTED_RESEARCH_PREVIEW`
Parent architecture: `TOP10_QBD_ROUTER_V1_TARGET_ARCHITECTURE.md`  
Baseline commit: `9d387eca9f6f119ba226b9fb49224b7ae143a794`

## Implementation principle

The goal is to reach the full target architecture directly. Intermediate steps exist only to make the build testable and auditable; they are not separate model-selection research projects.

The existing Top-10 expert implementations and `top10_adaptation_gate_controller_v4.py` are treated as frozen dependencies. The new work is a meta-routing layer around them.

Every step below has a concrete deliverable and Definition of Done (DoD). No step is considered complete because code exists; the required tests and invariants must also pass.

---

## Step 0 — Freeze the router research contract

### Files

Create:

```text
stock_predictor/backtests/opportunity_portfolio_research/top10_qbd_contract.py
stock_predictor/backtests/opportunity_portfolio_research/top10_qbd_policy.py
```

### Implement

Define typed immutable contracts for:

- `TOP10_QBD_ROUTER_V1`;
- expert universe identity;
- policy version;
- evidence cadence;
- health windows/discounting;
- promotion/demotion thresholds;
- lifecycle transitions;
- regime shrinkage;
- change-point configuration;
- opportunity-time activity configuration;
- transaction-cost assumptions;
- fallback policy;
- holdout boundaries.

Expose deterministic serialization and SHA-256 fingerprinting.

### DoD

- same logical policy serializes byte-for-byte identically;
- changed policy field changes policy hash;
- baseline expert/V4 identifiers are explicit;
- Final Holdout boundaries cannot be omitted;
- live-trading mode is not representable by this research contract.

---

## Step 1 — Immutable Expert Registry

### Files

Create:

```text
stock_predictor/backtests/opportunity_portfolio_research/top10_qbd_expert_registry.py
```

### Implement

`ExpertSpec` containing:

```text
expert_id
model_artifact_hash
horizon
holding_days
max_names
entry_policy_id
exit_policy_id
feature_schema_hash
adaptation_controller_id
adaptation_controller_hash
expert_type
```

Register:

```text
R01 ... R10
MSCI_WORLD
CASH
```

Validate loaded implementation against declared identity.

### DoD

- duplicate expert IDs fail;
- missing hash/identity fields fail for stock experts;
- artifact mismatch fails closed;
- registry has deterministic hash;
- R01-R10 mapping is pinned to the exact existing research identities.

---

## Step 2 — Permanent Shadow Engine and Ledger

### Files

Create:

```text
stock_predictor/backtests/opportunity_portfolio_research/top10_qbd_shadow_engine.py
stock_predictor/backtests/opportunity_portfolio_research/top10_qbd_shadow_ledger.py
```

### Implement

Run all experts on every eligible decision date independent of allocation state.

Typed records:

```text
ShadowDecision
PendingOutcome
MaturedOutcome
```

Required fields:

```text
decision_date
expert_id
eligible_opportunity
prediction
threshold
would_enter
would_exit
shadow_position
shadow_trade
market_regime
market_state_features
prediction_uncertainty
outcome_available_at
realized_return
benchmark_return
net_excess
transaction_cost
outcome_matured
```

The engine stores pending outcomes without revealing their realized values to the router before maturity.

### DoD

- suspended experts still produce shadow decisions;
- router receives no future-return field;
- outcome maturation is deterministic;
- duplicate decision/expert records fail;
- ledger can be replayed from persisted records.

---

## Step 3 — Expert Health Store

### Files

Create:

```text
stock_predictor/backtests/opportunity_portfolio_research/top10_qbd_health_store.py
stock_predictor/backtests/opportunity_portfolio_research/top10_qbd_health_metrics.py
```

### Implement

`ExpertHealthState` with distinct sub-states:

```text
StructuralHealth
ActivityHealth
AlphaHealth
RegimeHealth
UncertaintyHealth
```

Health updates consume only matured shadow outcomes.

Expose compact machine-readable snapshots per expert/date.

### DoD

- future/pending outcomes cannot update health;
- repeated same update is idempotent or rejected deterministically;
- health history is auditable;
- structural failures are separate from statistical weakness;
- no single black-box health scalar is authoritative.

---

## Step 4 — Opportunity-Time Activity Monitor

### Files

Create:

```text
stock_predictor/backtests/opportunity_portfolio_research/top10_qbd_activity_monitor.py
```

### Implement

Track:

```text
eligible_opportunities
actual_trades
opportunity_time_since_last_trade
calendar_time_since_last_trade
Gamma-Poisson posterior parameters
posterior predictive P(N=0 | T)
```

States:

```text
NORMAL
WATCH
ANOMALOUS
CRITICAL
```

The policy thresholds are frozen in Step 0.

### DoD

- calendar-only inactivity cannot cause anomaly;
- non-opportunity days do not increment opportunity time;
- rare historical strategy is not penalized using a high-frequency prior;
- statistically implausible zero-trade streak under high opportunity exposure raises anomaly;
- posterior state persists across restart.

---

## Step 5 — Alpha Health and Conservative Uncertainty

### Files

Extend:

```text
top10_qbd_health_metrics.py
```

Create if useful:

```text
stock_predictor/backtests/opportunity_portfolio_research/top10_qbd_alpha_health.py
```

### Implement

For matured net excess relative to MSCI:

- discounted/EWMA net excess;
- rolling mean/median;
- hit rate;
- relative drawdown;
- effective sample size;
- block-bootstrap or posterior uncertainty;
- lower confidence bound (LCB) for relative edge.

Avoid using raw point CAGR as the switching statistic.

### DoD

- alpha metric uses net excess, not absolute return;
- low sample size widens uncertainty;
- LCB is reproducible with frozen seed/config where stochastic methods are used;
- all cost deductions occur before router alpha scoring.

---

## Step 6 — Regime-Conditioned Health with Shrinkage

### Files

Create:

```text
stock_predictor/backtests/opportunity_portfolio_research/top10_qbd_regime_health.py
```

### Implement

Reuse existing market-state/regime features already available in the opportunity research stack.

Compute local and global expert evidence and shrink:

```text
S_j,r,t = rho_r,t * S_local_j,r,t + (1-rho_r,t) * S_global_j,t
```

`rho` is a deterministic function of matured local evidence/effective sample size.

### DoD

- no new learned regime classifier is introduced;
- unseen/sparse regime falls back mostly/entirely to global evidence;
- local regime evidence cannot use future labels/outcomes;
- regime-health result is reproducible.

---

## Step 7 — Alpha Decay and Change-Point Detection

### Files

Create:

```text
stock_predictor/backtests/opportunity_portfolio_research/top10_qbd_alpha_decay.py
stock_predictor/backtests/opportunity_portfolio_research/top10_qbd_changepoint.py
```

### Implement

Two layers:

1. discounted/EWMA evidence for gradual drift;
2. BOCPD or equivalent causal change-point detector for abrupt changes.

Input series is expert-level matured net-excess/residual evidence, not raw benchmark returns.

Output:

```text
change_probability
estimated_run_length
forgetting_multiplier
probation_pressure
```

### DoD

- change point never directly chooses a new champion;
- no future observations enter current run-length posterior;
- detector state survives restart;
- synthetic sudden-decay test raises change probability faster than stable-null test.

---

## Step 8 — Kill-Switch Decomposition

### Files

Create:

```text
stock_predictor/backtests/opportunity_portfolio_research/top10_qbd_kill_switches.py
```

### Implement

Return independent determinations:

```text
structural_kill
activity_kill
alpha_kill
regime_kill
```

Each determination includes:

```text
status
reason_code
evidence_timestamp
policy_rule_id
```

### DoD

- structural kill can fail closed immediately;
- activity kill requires opportunity-conditioned evidence;
- regime kill means zero/currently reduced allocation, not permanent model death;
- reason codes are persisted and reportable.

---

## Step 9 — Superior Candidate Set

### Files

Create:

```text
stock_predictor/backtests/opportunity_portfolio_research/top10_qbd_candidate_set.py
```

### Implement

Causal filtration pipeline:

```text
all experts
→ structural eligibility
→ minimum matured evidence
→ uncertainty gate
→ relative-performance elimination
→ superior candidate set
```

Start with a deterministic conservative confidence-bound procedure. Add MCS-style multiple-comparison-aware elimination behind the same interface once validated.

### DoD

- two statistically indistinguishable experts may both survive;
- clearly dominated synthetic expert is removed;
- high-uncertainty lucky expert is not automatically promoted;
- MSCI/Cash remain available as fallbacks even if no stock expert survives.

---

## Step 10 — Router Lifecycle State Machine

### Files

Create:

```text
stock_predictor/backtests/opportunity_portfolio_research/top10_qbd_router.py
stock_predictor/backtests/opportunity_portfolio_research/top10_qbd_state.py
```

### Implement

States:

```text
SHADOW
CANDIDATE
ENSEMBLE
CHAMPION
PROBATION
SUSPENDED
```

Transitions are pure/deterministic functions of current state, matured evidence and frozen policy.

Persist transition reason and source evidence timestamp.

### DoD

- every transition is explicit and test-covered;
- illegal transitions fail;
- `SUSPENDED` has zero real allocation but keeps shadow execution;
- reactivation path `SUSPENDED → CANDIDATE` exists;
- structural failure removes real allocation immediately.

---

## Step 11 — Allocation: Equal Weight, Soft DMA and Hard Champion

### Files

Extend:

```text
top10_qbd_router.py
```

Create if separation improves testability:

```text
stock_predictor/backtests/opportunity_portfolio_research/top10_qbd_allocation.py
stock_predictor/backtests/opportunity_portfolio_research/top10_qbd_promotion.py
```

### Implement

Three core policies in one interface:

```text
EQUAL_WEIGHT
DISCOUNTED_SOFTMAX / DMA
HARD_CHAMPION
```

Soft weights:

```text
S_j,t = lambda*S_j,t-1 + (1-lambda)*q_j,t
w_j,t ∝ exp(eta*S_j,t)
```

Hard promotion criterion:

```text
LCB(alpha_challenger - alpha_incumbent - switch_cost) > 0
```

Add hysteresis:

- promotion threshold;
- hold range;
- probation threshold;
- suspension threshold;
- minimum champion duration.

### DoD

- small score noise does not cause repeated switching;
- challenger gross edge below switching cost does not switch;
- weights sum correctly after eligibility gates;
- no eligible alpha expert triggers benchmark/Cash fallback;
- deterministic tie handling.

---

## Step 12 — Benchmark and Cash Meta-Experts

### Files

Extend registry/router/allocation modules.

### Implement

Treat benchmark and Cash as first-class router outcomes rather than implicit emergency branches.

Define frozen policy for when:

- `MSCI_WORLD` receives allocation;
- `CASH` receives allocation;
- partial stock-expert ensemble + benchmark is allowed or forbidden.

### DoD

- router is never forced to select a negative-edge stock expert;
- fallback is recorded in state and performance reporting;
- benchmark return is not double-counted in excess calculation;
- Cash return convention is explicit and frozen.

---

## Step 13 — Persistent Router State Store

### Files

Create:

```text
stock_predictor/backtests/opportunity_portfolio_research/top10_qbd_state_store.py
```

### Implement

Persist at least:

```text
policy_hash
registry_hash
current champion
candidate set
ensemble weights
lifecycle state per expert
health state
Gamma-Poisson state
EWMA state
change-point state
probation counters
last switch
last update
matured-outcome cursor
```

Use atomic writes/versioned schema suitable for replay and crash recovery.

### DoD

- continuous run equals save/restart/resume run;
- incompatible policy/registry hash refuses resume;
- partial/corrupted state fails closed;
- state schema version is explicit.

---

## Step 14 — Single End-to-End Prequential Replay Engine

### Files

Create:

```text
stock_predictor/backtests/opportunity_portfolio_research/top10_qbd_prequential_replay.py
```

### Implement

Single causal loop:

```text
mature outcomes
→ update ledger
→ update health
→ activity/alpha/regime/change
→ candidate set
→ lifecycle
→ allocation
→ execution/cost simulation
→ register pending outcomes
→ persist state
```

Do not pass complete future returns into selector APIs.

### DoD

- future mutation test leaves past decisions unchanged;
- replay output includes evidence timestamp for each router decision;
- identical inputs + policy hash produce identical replay;
- all router arms use exactly this engine.

---

## Step 15 — A–K Ablation Arms

### Files

Extend prequential engine and policy definitions.

### Implement

Arms:

```text
A STATIC_BEST_DEV
B HARD_CHAMPION_3M
C HARD_CHAMPION_6M
D EQUAL_WEIGHT
E DISCOUNTED_SOFTMAX
F SUPERIOR_SET_EQUAL
G SUPERIOR_SET_DMA
H G + REGIME_SHRINKAGE
I H + CHANGEPOINT
J I + OPPORTUNITY_TIME_ACTIVITY
K J + UNCERTAINTY_LCB
```

Use feature flags/composition, not separate copies of the replay engine.

### DoD

- all arms consume identical frozen expert shadow data;
- only explicitly named router capability differs between adjacent arms;
- result manifest contains arm configuration/hash;
- no arm-specific hidden preprocessing.

---

## Step 16 — Synthetic Oracle and Null Test Suite

### Files

Create:

```text
stock_predictor/backtests/opportunity_portfolio_research/top10_qbd_synthetic_scenarios.py
stock_predictor/backtests/opportunity_portfolio_research/top10_qbd_oracle_diagnostics.py
```

### Implement

Known-truth experts/scenarios:

- A strong then decays;
- B becomes strong after regime shift;
- C rare but profitable;
- D persistent loser;
- E lucky/noise expert;
- F abrupt structural failure;
- G zero trades despite high opportunity exposure;
- H suspended then later recovers;
- null universe where all experts have identical expected return.

Metrics:

```text
entry_detection_delay
exit_detection_delay
reactivation_delay
false_promotion_rate
false_suspension_rate
captured_alpha_fraction
```

### DoD

- full router promotes/allocates to known signal with bounded delay;
- decayed expert is demoted without using future data;
- rare expert is not suspended solely due to calendar inactivity;
- null scenario quantifies false champion discovery;
- recovery scenario proves reactivation path.

---

## Step 17 — Historical Pseudo-OOS Evaluation and Reporting

### Files

Create:

```text
stock_predictor/backtests/opportunity_portfolio_research/top10_qbd_evaluation.py
```

Artifacts under a versioned directory such as:

```text
artifacts/top10-qbd-router-v1/
```

### Implement

Report:

Performance:

```text
net excess CAGR
Information Ratio
Sortino
relative MaxDD
trade count
turnover
costs
```

Adaptation:

```text
switch_count
switch_cost
champion_duration
probation_count
suspension_count
reactivation_count
benchmark_fallback_days
cash_days
entry/exit/reactivation delay
false promotion/suspension proxies
captured oracle alpha
```

Concentration:

```text
top-1/top-5/top-10 trade contribution
month/quarter/year
ticker
sector
regime
OOS fold
time in market
```

### DoD

- each result references commit, policy hash, registry hash and data fingerprint;
- tables distinguish gross, cost and net excess;
- router decisions are traceable to matured evidence;
- V4 development-reuse results are not relabelled independent OOS.

---

## Step 18 — Multiple-Testing and Cost-Stress Layer

### Files

Extend evaluation module; create dedicated diagnostics if needed:

```text
stock_predictor/backtests/opportunity_portfolio_research/top10_qbd_overfitting_diagnostics.py
```

### Implement

Track full attempted-variant count and compute appropriate diagnostics:

- Deflated Sharpe Ratio;
- PBO where feasible;
- block bootstrap;
- Reality-Check-style comparison / family-wise evidence;
- router-arm multiple comparison summary.

Cost stresses:

```text
native
10 bps
20 bps
30 bps
50 bps roundtrip
```

Include incremental switching cost.

### DoD

- attempted variants cannot silently disappear from denominator/log;
- same trades replayed under all cost stresses;
- router-specific switching cost is explicit;
- conclusions distinguish nominal performance from robust performance.

---

## Step 19 — Freeze Manifest and Final Holdout Gate

### Files

Create generated manifest:

```text
stock_predictor/backtests/opportunity_portfolio_research/QBD_ROUTER_V1_FROZEN_MANIFEST.json
```

Create validation utility:

```text
stock_predictor/backtests/opportunity_portfolio_research/top10_qbd_freeze.py
```

### Implement

Freeze/hash:

```text
code commit
expert registry
model artifacts
feature schemas
V4 controller
router policy
health rules
regime rules
change-point rules
activity model
allocation rules
cost model
review cadence
fallback policy
evaluation metrics
holdout boundaries
```

Final Holdout runner refuses to execute without matching manifest.

### DoD

- any frozen field/code mismatch blocks holdout run;
- holdout runner cannot tune thresholds/configuration;
- algorithmic adaptation during holdout remains allowed using matured evidence only;
- holdout output records manifest hash.

---

## Step 20 — Final independent adaptive holdout run

### Implement

Run the frozen `TOP10_QBD_ROUTER_V1` exactly once under the independent holdout contract.

The router may, by design:

- change ensemble weights;
- change champion;
- move experts to probation/suspension;
- reactivate experts;
- use MSCI/Cash fallback;

provided each decision uses only evidence available at that date and the adaptation rules are unchanged from the frozen manifest.

### DoD

- no post-hoc parameter change during the run;
- all decisions replay deterministically from recorded inputs/state;
- result is clearly labelled independent holdout or failed holdout;
- any subsequent architecture/policy modification requires a new version, e.g. `TOP10_QBD_ROUTER_V2`.

---

# Dependency order

```text
0 Contract
↓
1 Registry
↓
2 Shadow Ledger
↓
3 Health Store
├─ 4 Activity
├─ 5 Alpha/Uncertainty
├─ 6 Regime
└─ 7 Change Point
↓
8 Kill Switches
↓
9 Candidate Set
↓
10 Lifecycle
↓
11 Allocation/Promotion
↓
12 MSCI/Cash
↓
13 Persistent State
↓
14 Prequential Engine
↓
15 A–K Arms
↓
16 Synthetic/Null Validation
↓
17 Historical Pseudo-OOS
↓
18 Overfitting + Cost Stress
↓
19 Freeze Manifest
↓
20 Final Holdout
```

## Stop conditions

Implementation should stop and correct the architecture before moving to holdout if any of the following is true:

- future evidence can alter a prior router decision;
- restart parity fails;
- suspended experts stop producing shadow evidence;
- inactivity uses calendar time without opportunity conditioning;
- the router cannot choose MSCI/Cash;
- switching is driven by point CAGR without uncertainty/switch-cost hurdle in the full arm;
- A–K arms do not share the exact same causal replay engine;
- synthetic null test shows uncontrolled false champion discovery and no frozen rule mitigates it;
- the freeze manifest cannot detect code/config drift.
