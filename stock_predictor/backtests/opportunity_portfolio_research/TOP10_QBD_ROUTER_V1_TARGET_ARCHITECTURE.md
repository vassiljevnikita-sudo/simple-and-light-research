# TOP10 QbD Router V1 — Target Architecture

Status: `TARGET_ARCHITECTURE_NOT_IMPLEMENTED`  
Contract ID: `TOP10_QBD_ROUTER_V1`  
Baseline branch: `agent/top10-external-validation`  
Baseline commit: `9d387eca9f6f119ba226b9fb49224b7ae143a794`  
Baseline result: `research: publish adaptation gate controller v4 results`

## 1. Purpose

This document is the normative target architecture for the next research layer above the existing Top-10 opportunity models and `TOP10_ADAPTATION_GATE_CONTROLLER_V4`.

The existing V4 controller adapts memory/threshold behaviour **inside one expert**. It does not decide which expert should receive capital. `TOP10_QBD_ROUTER_V1` adds that missing meta-layer.

The router must answer, causally and repeatedly:

> Given only evidence that has matured by decision time, which expert or ensemble should receive capital now, should the incumbent remain active, should a model be put on probation or suspended, and when should a previously suspended model be allowed to return?

The system is designed specifically for low-frequency models where calendar inactivity alone is not evidence of decay. A model may wait months for a valid opportunity, while another model may have stopped responding to opportunities entirely. The architecture must distinguish those cases.

## 2. Non-goals

This contract does **not**:

- retrain R01-R10;
- change their frozen entry/exit model identities;
- reopen the Final Holdout for iterative tuning;
- use V4 development-reuse performance to select a production winner;
- replace the current benchmark definition;
- enable live broker writes;
- create a new market-regime classifier as the first step;
- treat absence of trades as failure without conditioning on opportunity time.

The existing `top10_adaptation_gate_controller_v4.py` remains unchanged unless a separately versioned research contract explicitly replaces it.

## 3. System boundary

The router operates above a frozen expert universe.

```text
R01 + V4 ─┐
R02 + V4 ─┤
R03 + V4 ─┤
...       ├──> Permanent Shadow Execution
R10 + V4 ─┤
MSCI      ─┤
CASH      ─┘
             ↓
        Shadow Outcome Ledger
             ↓
        Matured Evidence Gate
             ↓
        Expert Health Store
        ├─ Structural health
        ├─ Activity health
        ├─ Alpha health
        ├─ Regime health
        └─ Uncertainty health
             ↓
        Superior Candidate Set
             ↓
        QbD Meta Router
        ├─ soft ensemble
        ├─ hard champion
        ├─ probation
        ├─ suspension
        └─ MSCI/Cash fallback
             ↓
        Portfolio allocation
             ↓
        Realized outcomes
             ↺
```

## 4. Frozen expert universe

Initial expert universe:

- `R01` … `R10`, each including its existing frozen model/policy identity and V4 controller behaviour;
- `MSCI_WORLD` as benchmark meta-expert;
- `CASH` as abstention/fail-safe meta-expert.

`NO_TRADE` is represented through abstention/Cash and is not required as a separate alpha model.

Every stock expert must have an immutable identity containing at least:

```python
ExpertSpec(
    expert_id,
    model_artifact_hash,
    horizon,
    holding_days,
    max_names,
    entry_policy_id,
    exit_policy_id,
    feature_schema_hash,
    adaptation_controller_id,
    adaptation_controller_hash,
)
```

Any mismatch between declared and loaded identity is a structural failure and must fail closed.

## 5. Permanent shadow execution

Every expert continues to generate shadow decisions independent of its capital-allocation state.

A `SUSPENDED` expert receives zero real allocation but must continue to produce shadow evidence so that reactivation is possible.

The authoritative shadow record is keyed by `decision_date × expert_id` and includes at minimum:

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

## 6. Matured-evidence invariant

The central causality rule is:

```text
An outcome may affect router state only when
outcome_available_at <= current_router_date.
```

A replay that exposes future outcomes to the router, even if later filtered, is not an acceptable implementation.

The selector must receive only already-matured evidence.

## 7. Expert health

Health is intentionally decomposed. There is no single opaque `healthy/unhealthy` scalar.

### 7.1 Structural health

Examples:

- artifact/hash mismatch;
- missing or stale features;
- missing required price data;
- invalid venue mapping;
- impossible execution state;
- unavailable required exit component;
- non-causal source date;
- schema incompatibility.

Structural failure is fail-closed and can immediately remove real allocation.

### 7.2 Activity health

Activity health measures whether an expert behaves as expected **conditional on opportunity time**.

The router tracks:

```text
eligible_opportunities
expected_trade_rate
actual_trades
opportunity_time_since_last_trade
calendar_time_since_last_trade
```

Historical trade-arrival intensity is modelled with a Gamma-Poisson process:

```text
lambda_j ~ Gamma(a_j, b_j)
```

For future opportunity exposure `T`, the posterior-predictive zero-trade probability is:

```text
P(N_future = 0 | T) = (b_j / (b_j + T)) ** a_j
```

Calendar inactivity without opportunities must not trigger suspension.

### 7.3 Alpha health

Primary signal: realized **net excess relative to MSCI**, not absolute market return.

Health may include:

- discounted/EWMA net excess;
- mean/median matured net excess;
- hit rate;
- information-ratio style stability;
- relative drawdown;
- rank/selection quality metrics where available;
- conservative lower confidence bound of relative edge.

The router should prefer conservative evidence over point-estimate CAGR ranking.

### 7.4 Regime health

Use already available market-state features first. Do not introduce a new regime classifier until the simpler architecture is validated.

Expert evidence is shrunk from local regime evidence toward global evidence:

```text
S_j,r,t = rho_r,t * S_local_j,r,t + (1 - rho_r,t) * S_global_j,t
```

`rho` increases only with matured evidence in the regime. Sparse local evidence therefore cannot dominate.

### 7.5 Uncertainty health

Uncertainty is tracked separately from expected edge. Sources may include:

- sample sufficiency;
- bootstrap/posterior uncertainty;
- ensemble dispersion;
- prediction/residual uncertainty already emitted by lower layers.

High uncertainty may reduce allocation or block promotion without declaring the model structurally broken.

## 8. Alpha-decay and change detection

Two complementary mechanisms are allowed:

1. discounted/EWMA evidence for gradual decay;
2. BOCPD/change-point evidence for abrupt changes.

Change detection is applied to model-level performance evidence such as matured net-excess residuals, not merely to raw market returns.

A change point must not directly switch the champion. It can:

- reduce confidence;
- accelerate forgetting;
- move an expert toward `PROBATION`;
- increase the hurdle for continued allocation.

## 9. Superior Candidate Set

The router does not default to `argmax(score)`.

Pipeline:

```text
all experts
→ remove structural failures
→ apply minimum evidence / uncertainty gates
→ remove clearly inferior experts
→ retain statistically plausible superior set
```

When multiple experts are not distinguishable with adequate confidence, the correct action is an ensemble, not an arbitrary hard winner.

An MCS-style or equivalent multiple-comparison-aware procedure may be used once the causal evidence pipeline exists.

## 10. Router lifecycle

Required lifecycle states:

```text
SHADOW
CANDIDATE
ENSEMBLE
CHAMPION
PROBATION
SUSPENDED
```

Core transitions:

```text
SHADOW → CANDIDATE
  enough matured evidence

CANDIDATE → CHAMPION
  conservative dominance over incumbent / alternatives

CANDIDATE → ENSEMBLE
  several experts remain statistically plausible

CHAMPION → PROBATION
  alpha/activity/regime/change warning

PROBATION → CHAMPION
  evidence recovers

PROBATION → SUSPENDED
  deterioration is confirmed

SUSPENDED → CANDIDATE
  shadow evidence supports recovery
```

Structural failure may force any allocating state to `SUSPENDED` immediately.

Permanent retirement is outside V1 unless the failure is explicitly structural and irrecoverable.

## 11. Allocation policy

### 11.1 Default under ambiguous evidence

When no clear dominant expert exists, use a soft ensemble within the qualified candidate set.

A baseline discounted soft-weight rule is:

```text
S_j,t = lambda * S_j,t-1 + (1-lambda) * q_j,t
w_j,t ∝ exp(eta * S_j,t)
```

Weights must respect health gates, fallback rules and concentration limits defined in the frozen policy.

### 11.2 Hard champion promotion

Hard switching requires conservative relative edge after switch costs:

```text
LCB(alpha_challenger - alpha_incumbent - C_switch) > 0
```

Promotion and demotion thresholds must differ to create hysteresis.

The frozen policy must define:

- minimum matured evidence;
- minimum eligible opportunities;
- minimum champion duration;
- promotion margin;
- probation margin;
- suspension margin;
- reactivation requirements;
- switch-cost hurdle.

### 11.3 Benchmark and Cash fallback

The router must be allowed to conclude that no stock expert currently has sufficient evidence of positive net excess.

Fallback choices:

- `MSCI_WORLD` when benchmark exposure is preferred;
- `CASH` when abstention/fail-safe policy requires no market exposure.

The router must never be forced to select the least-bad stock expert.

## 12. Kill-switch decomposition

Required independent outputs:

```text
structural_kill
activity_kill
alpha_kill
regime_kill
```

These signals feed a deterministic router policy. They must be logged independently for diagnosis.

## 13. Persistent router state

The router must persist all state needed for deterministic restart parity, including:

```text
current champion
current candidate set
current ensemble weights
lifecycle state per expert
health state per expert
Gamma-Poisson parameters
EWMA/discounted alpha state
change-point state
probation counters
last switch date
last health update
last matured-evidence cursor
policy hash
expert registry hash
```

Invariant:

```text
continuous replay == stop/save/restart/resume replay
```

for identical inputs and frozen policy.

## 14. Prequential execution contract

The end-to-end engine processes time in causal order:

```text
for current_date in dates:
    1. ingest outcomes that have matured by current_date
    2. update shadow ledger
    3. update health states
    4. update activity/alpha/regime/change evidence
    5. compute candidate set
    6. update lifecycle/router state
    7. decide allocation
    8. simulate orders/fills/costs
    9. register new shadow outcomes with future maturity timestamps
   10. persist router state
```

No router method may receive future return arrays as an input convenience.

## 15. Required comparison arms

All arms run in the same engine and use the same frozen experts, data, costs and matured-outcome rules:

```text
A  STATIC_BEST_DEV
B  HARD_CHAMPION_3M
C  HARD_CHAMPION_6M
D  EQUAL_WEIGHT
E  DISCOUNTED_SOFTMAX
F  SUPERIOR_SET_EQUAL
G  SUPERIOR_SET_DMA
H  G + REGIME_SHRINKAGE
I  H + CHANGEPOINT
J  I + OPPORTUNITY_TIME_ACTIVITY
K  J + UNCERTAINTY_LCB
```

`K` is the full V1 target. The other arms are ablations, not separate research projects.

## 16. Evaluation outputs

Performance metrics:

- net excess CAGR relative to MSCI;
- Information Ratio;
- Sortino;
- relative MaxDD;
- turnover and transaction cost;
- trade count.

Adaptation metrics:

- switch count;
- switch cost;
- champion duration distribution;
- probation/suspension/reactivation counts;
- benchmark/Cash fallback share;
- entry detection delay;
- exit detection delay;
- reactivation delay;
- false promotion rate;
- false suspension rate;
- captured alpha fraction relative to a hindsight oracle upper bound.

The oracle is diagnostic only and never a tradable comparison arm.

## 17. Concentration and false-discovery diagnostics

Every research run must report alpha concentration by:

- top-1/top-5/top-10 trade contribution;
- month/quarter/year;
- ticker;
- sector;
- regime;
- OOS fold;
- time in market.

Before Final Holdout, the research report must also include multiple-testing/overfitting diagnostics appropriate to the number of attempted variants, including DSR/PBO and block-bootstrap/Reality-Check-style evidence where feasible.

## 18. Cost stress

Required replay cost variants:

```text
native cost model
10 bps roundtrip
20 bps roundtrip
30 bps roundtrip
50 bps roundtrip
```

Router-induced switching costs are included explicitly.

## 19. Holdout discipline

Before opening Final Holdout, freeze and hash:

- expert universe;
- expert artifacts;
- V4 controller identity;
- router arms;
- score definitions;
- discount factors;
- regime logic and shrinkage;
- health rules;
- promotion/probation/suspension/reactivation rules;
- inactivity model;
- change-point configuration;
- costs;
- review cadence;
- fallback policy;
- metric definitions.

The resulting manifest is `QBD_ROUTER_V1_FROZEN_MANIFEST.json`.

During Final Holdout, the **algorithm may adapt** using only information available at each date. It may change champions, weights, lifecycle states and fallback exposure. What may not change are the frozen adaptation rules themselves.

## 20. Safety and research status

Until an independent validation contract explicitly promotes this architecture:

```text
Trading: PAPER_TRADING_ONLY
Live broker writes: FORBIDDEN
Final Holdout tuning: FORBIDDEN
V4 mutation from this contract: FORBIDDEN
```

This target architecture supersedes informal QbD-router planning notes for `TOP10_QBD_ROUTER_V1`, but it does not alter the repository-wide current production/research model identity defined by the higher-authority registry and `CURRENT_MODEL.md`.
