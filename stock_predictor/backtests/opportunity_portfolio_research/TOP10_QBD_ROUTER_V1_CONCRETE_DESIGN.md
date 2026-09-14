# TOP10 QbD Router V1 — Concrete Design

Status: `IMPLEMENTED_RESEARCH_PREVIEW`
Parent: `TOP10_QBD_ROUTER_V1_TARGET_ARCHITECTURE.md`

This document turns the target architecture into concrete Python interfaces, state records and module ownership. Field names may be extended during implementation, but semantic changes require an explicit contract revision.

## 1. Package layout

All new V1 research code stays scoped under:

```text
stock_predictor/backtests/opportunity_portfolio_research/
```

Target module set:

```text
top10_qbd_contract.py
top10_qbd_policy.py
top10_qbd_expert_registry.py
top10_qbd_shadow_engine.py
top10_qbd_shadow_ledger.py
top10_qbd_health_store.py
top10_qbd_health_metrics.py
top10_qbd_activity_monitor.py
top10_qbd_alpha_health.py
top10_qbd_regime_health.py
top10_qbd_alpha_decay.py
top10_qbd_changepoint.py
top10_qbd_kill_switches.py
top10_qbd_candidate_set.py
top10_qbd_allocation.py
top10_qbd_promotion.py
top10_qbd_router.py
top10_qbd_state.py
top10_qbd_state_store.py
top10_qbd_prequential_replay.py
top10_qbd_synthetic_scenarios.py
top10_qbd_oracle_diagnostics.py
top10_qbd_evaluation.py
top10_qbd_overfitting_diagnostics.py
top10_qbd_freeze.py
```

Do not fork/copy the replay engine per arm. Composition and frozen feature flags must be used.

## 2. Core enums

```python
from enum import StrEnum

class ExpertType(StrEnum):
    STOCK = "STOCK"
    BENCHMARK = "BENCHMARK"
    CASH = "CASH"

class LifecycleState(StrEnum):
    SHADOW = "SHADOW"
    CANDIDATE = "CANDIDATE"
    ENSEMBLE = "ENSEMBLE"
    CHAMPION = "CHAMPION"
    PROBATION = "PROBATION"
    SUSPENDED = "SUSPENDED"

class HealthLevel(StrEnum):
    PASS = "PASS"
    WATCH = "WATCH"
    FAIL = "FAIL"

class ActivityLevel(StrEnum):
    NORMAL = "NORMAL"
    WATCH = "WATCH"
    ANOMALOUS = "ANOMALOUS"
    CRITICAL = "CRITICAL"

class RouterArm(StrEnum):
    A_STATIC_BEST_DEV = "A_STATIC_BEST_DEV"
    B_HARD_CHAMPION_3M = "B_HARD_CHAMPION_3M"
    C_HARD_CHAMPION_6M = "C_HARD_CHAMPION_6M"
    D_EQUAL_WEIGHT = "D_EQUAL_WEIGHT"
    E_DISCOUNTED_SOFTMAX = "E_DISCOUNTED_SOFTMAX"
    F_SUPERIOR_SET_EQUAL = "F_SUPERIOR_SET_EQUAL"
    G_SUPERIOR_SET_DMA = "G_SUPERIOR_SET_DMA"
    H_REGIME_SHRINKAGE = "H_REGIME_SHRINKAGE"
    I_CHANGEPOINT = "I_CHANGEPOINT"
    J_OPPORTUNITY_TIME_ACTIVITY = "J_OPPORTUNITY_TIME_ACTIVITY"
    K_UNCERTAINTY_LCB = "K_UNCERTAINTY_LCB"
```

## 3. Frozen contract

```python
@dataclass(frozen=True)
class FrozenQbdRouterPolicyV1:
    contract_id: str
    policy_version: str
    router_arm: RouterArm

    assessment_cadence_sessions: int
    allocation_cadence_sessions: int

    alpha_discount: float
    softmax_eta: float
    min_matured_observations: int
    min_eligible_opportunities: int

    promotion_lcb_margin: float
    probation_lcb_margin: float
    suspension_lcb_margin: float
    reactivation_lcb_margin: float
    min_champion_sessions: int

    activity_prior_a: float
    activity_prior_b: float
    activity_watch_zero_prob: float
    activity_anomalous_zero_prob: float
    activity_critical_zero_prob: float

    regime_shrinkage_k: float

    changepoint_enabled: bool
    changepoint_config: Mapping[str, Any]

    stock_roundtrip_bps: float
    benchmark_roundtrip_bps: float
    switch_cost_bps: float

    fallback_order: tuple[str, ...]

    final_holdout_start: date
    final_holdout_end: date
```

Exact numeric values are **not** to be invented during coding. They must be explicitly frozen from a development-only design decision and recorded in the manifest before independent holdout.

Serialization:

```python
policy_json = canonical_json(policy)
policy_hash = sha256(policy_json.encode()).hexdigest()
```

Canonical JSON requires stable key order, ISO dates and no non-deterministic floating representation.

## 4. Expert registry

```python
@dataclass(frozen=True)
class ExpertSpec:
    expert_id: str
    expert_type: ExpertType
    model_artifact_hash: str | None
    horizon: int | None
    holding_days: int | None
    max_names: int | None
    entry_policy_id: str | None
    exit_policy_id: str | None
    feature_schema_hash: str | None
    adaptation_controller_id: str | None
    adaptation_controller_hash: str | None
```

Functions:

```python
def build_top10_qbd_registry() -> tuple[ExpertSpec, ...]: ...
def validate_registry(registry: Sequence[ExpertSpec]) -> None: ...
def registry_hash(registry: Sequence[ExpertSpec]) -> str: ...
```

`MSCI_WORLD` and `CASH` may have null model fields but their benchmark/cash policy identity must be present elsewhere in the frozen contract/manifest.

## 5. Shadow records

```python
@dataclass(frozen=True)
class ShadowDecision:
    decision_date: date
    expert_id: str
    eligible_opportunity: bool
    prediction: float | None
    threshold: float | None
    would_enter: bool
    would_exit: bool
    market_regime: str | None
    market_state_features: Mapping[str, float]
    prediction_uncertainty: float | None
    outcome_available_at: date | None

@dataclass(frozen=True)
class PendingOutcome:
    decision_date: date
    expert_id: str
    outcome_available_at: date
    opaque_outcome_key: str

@dataclass(frozen=True)
class MaturedOutcome:
    decision_date: date
    expert_id: str
    outcome_available_at: date
    realized_return: float
    benchmark_return: float
    transaction_cost: float
    net_excess: float
```

Important design rule: router-facing code stores only `PendingOutcome.opaque_outcome_key` before maturity. Realized values are resolved only by the maturity step.

This avoids accidentally carrying future returns inside an object that is technically filtered later.

## 6. Shadow ledger API

```python
class ShadowLedger(Protocol):
    def append_decision(self, decision: ShadowDecision) -> None: ...
    def append_pending(self, pending: PendingOutcome) -> None: ...
    def mature(self, current_date: date) -> tuple[MaturedOutcome, ...]: ...
    def matured_since(self, cursor: str | None) -> tuple[MaturedOutcome, ...]: ...
    def cursor(self) -> str: ...
```

In-memory and persisted implementations must have parity tests.

## 7. Structural health

```python
@dataclass(frozen=True)
class StructuralHealth:
    level: HealthLevel
    reason_codes: tuple[str, ...]
    checked_at: date
```

Stable reason-code namespace, for example:

```text
ARTIFACT_HASH_MISMATCH
FEATURE_SCHEMA_MISMATCH
STALE_FEATURES
PRICE_DATA_MISSING
NONCAUSAL_SOURCE_DATE
EXECUTION_STATE_INVALID
EXIT_COMPONENT_UNAVAILABLE
```

Do not use free-text-only failure reasons.

## 8. Activity health

```python
@dataclass(frozen=True)
class ActivityPosterior:
    a: float
    b: float
    eligible_opportunities: int
    actual_trades: int
    opportunity_time_since_last_trade: int
    last_trade_date: date | None

@dataclass(frozen=True)
class ActivityHealth:
    level: ActivityLevel
    zero_trade_probability: float
    posterior: ActivityPosterior
    evaluated_at: date
```

Update sketch:

```python
def update_activity(
    previous: ActivityPosterior,
    *,
    eligible_opportunity: bool,
    traded: bool,
) -> ActivityPosterior:
    ...
```

Predictive zero probability over accumulated opportunity exposure `T`:

```python
def p_zero(a: float, b: float, T: float) -> float:
    return (b / (b + T)) ** a
```

## 9. Alpha health

```python
@dataclass(frozen=True)
class AlphaHealth:
    discounted_net_excess: float
    mean_net_excess: float | None
    median_net_excess: float | None
    relative_drawdown: float
    effective_n: float
    lcb_net_excess: float | None
    evaluated_at: date
```

Interface:

```python
def update_alpha_health(
    previous: AlphaHealthState,
    matured: Sequence[MaturedOutcome],
    policy: FrozenQbdRouterPolicyV1,
) -> AlphaHealthState:
    ...
```

LCB implementation must be isolated behind a deterministic interface so bootstrap/posterior mechanics can evolve only under a new policy version.

## 10. Regime health

```python
@dataclass(frozen=True)
class RegimeScore:
    regime_id: str
    local_score: float | None
    global_score: float
    rho: float
    shrunk_score: float
    effective_local_n: float
```

Frozen default functional form:

```python
rho = n_local / (n_local + k)
shrunk = rho * local + (1-rho) * global
```

If `local` is unavailable, `rho=0` and `shrunk=global`.

## 11. Change-point state

```python
@dataclass(frozen=True)
class ChangePointState:
    change_probability: float
    expected_run_length: float
    forgetting_multiplier: float
    evaluated_at: date
    opaque_detector_state: Mapping[str, Any]
```

Interface:

```python
def update_changepoint(
    previous: ChangePointState,
    matured_net_excess_observation: float,
    config: Mapping[str, Any],
) -> ChangePointState:
    ...
```

No allocation function may be imported into `top10_qbd_changepoint.py`.

## 12. Combined health snapshot

```python
@dataclass(frozen=True)
class ExpertHealthSnapshot:
    expert_id: str
    as_of: date
    structural: StructuralHealth
    activity: ActivityHealth
    alpha: AlphaHealth
    regime: RegimeScore
    uncertainty_score: float
    changepoint: ChangePointState | None
    matured_evidence_cursor: str
```

Health snapshots are immutable once emitted.

## 13. Kill-switch result

```python
@dataclass(frozen=True)
class KillDecision:
    triggered: bool
    reason_code: str | None
    evidence_date: date
    policy_rule_id: str

@dataclass(frozen=True)
class KillVector:
    structural: KillDecision
    activity: KillDecision
    alpha: KillDecision
    regime: KillDecision
```

`KillVector` is diagnostic input to the deterministic router. It is not itself the lifecycle state.

## 14. Candidate-set result

```python
@dataclass(frozen=True)
class CandidateAssessment:
    expert_id: str
    eligible: bool
    score: float
    lcb: float | None
    uncertainty: float
    exclusion_reason: str | None

@dataclass(frozen=True)
class CandidateSet:
    as_of: date
    members: tuple[str, ...]
    assessments: tuple[CandidateAssessment, ...]
```

The initial implementation should support a conservative confidence-bound filter. MCS-style logic implements the same interface later without changing router call sites.

## 15. Lifecycle state

```python
@dataclass(frozen=True)
class ExpertLifecycle:
    expert_id: str
    state: LifecycleState
    entered_state_at: date
    prior_state: LifecycleState | None
    reason_code: str
    evidence_date: date
```

Transition function must be pure:

```python
def transition_expert(
    previous: ExpertLifecycle,
    health: ExpertHealthSnapshot,
    kills: KillVector,
    candidate_set: CandidateSet,
    policy: FrozenQbdRouterPolicyV1,
    current_date: date,
) -> ExpertLifecycle:
    ...
```

## 16. Promotion assessment

```python
@dataclass(frozen=True)
class PromotionAssessment:
    challenger_id: str
    incumbent_id: str | None
    point_edge: float
    switch_cost: float
    lcb_net_edge_after_switch_cost: float
    minimum_evidence_met: bool
    minimum_duration_met: bool
    promotion_allowed: bool
    reason_code: str
```

Rule:

```python
promotion_allowed = (
    minimum_evidence_met
    and minimum_duration_met
    and lcb_net_edge_after_switch_cost > policy.promotion_lcb_margin
)
```

## 17. Allocation decision

```python
@dataclass(frozen=True)
class AllocationDecision:
    decision_date: date
    weights: Mapping[str, float]
    champion_id: str | None
    candidate_members: tuple[str, ...]
    fallback_used: str | None
    reason_code: str
    policy_hash: str
    registry_hash: str
    matured_evidence_cursor: str
```

Validation:

```text
all weights finite
all weights >= 0
sum(weights) == 1 within frozen tolerance
SUSPENDED stock expert weight == 0
structural-failed expert weight == 0
```

## 18. Router state

```python
@dataclass(frozen=True)
class RouterStateV1:
    schema_version: str
    as_of: date
    policy_hash: str
    registry_hash: str
    champion_id: str | None
    candidate_members: tuple[str, ...]
    current_weights: Mapping[str, float]
    lifecycle: Mapping[str, ExpertLifecycle]
    health: Mapping[str, ExpertHealthSnapshot]
    last_switch_date: date | None
    matured_evidence_cursor: str
    persisted_component_states: Mapping[str, Any]
```

`persisted_component_states` contains versioned Gamma-Poisson, EWMA and change-point internals if those are not directly contained in the snapshot structures.

## 19. State-store API

```python
class RouterStateStore(Protocol):
    def save_atomic(self, state: RouterStateV1) -> None: ...
    def load(self) -> RouterStateV1 | None: ...
```

Load validation must compare:

```text
schema_version
policy_hash
registry_hash
```

Mismatch blocks resume.

## 20. Router interface

```python
class QbdRouterV1:
    def decide(
        self,
        *,
        current_date: date,
        previous_state: RouterStateV1,
        matured_outcomes: Sequence[MaturedOutcome],
        shadow_decisions: Sequence[ShadowDecision],
    ) -> tuple[RouterStateV1, AllocationDecision]:
        ...
```

The interface intentionally does not accept future return tables.

## 21. Prequential engine interface

```python
@dataclass(frozen=True)
class ReplayResult:
    decisions: tuple[AllocationDecision, ...]
    final_state: RouterStateV1
    performance_artifact: Mapping[str, Any]
    run_manifest: Mapping[str, Any]


def run_qbd_router_replay(
    *,
    policy: FrozenQbdRouterPolicyV1,
    registry: Sequence[ExpertSpec],
    start_date: date,
    end_date: date,
    state_store: RouterStateStore,
    data_provider: CausalDataProvider,
) -> ReplayResult:
    ...
```

Daily/session loop ordering is fixed:

```text
1 resolve only outcomes maturing now/past
2 update health
3 create current shadow decisions
4 compute kill vector
5 compute candidate set
6 update lifecycle
7 compute allocation
8 simulate current execution/costs
9 register new pending outcomes
10 persist state
```

If lower-layer predictions require a different exact ordering, the implementation must preserve the same causality semantics and document the justified ordering change.

## 22. A–K composition

Do not duplicate router classes per arm.

Example capability flags derived from `RouterArm`:

```python
@dataclass(frozen=True)
class RouterCapabilities:
    hard_window_selector: bool
    equal_weight: bool
    discounted_softmax: bool
    superior_set: bool
    regime_shrinkage: bool
    changepoint: bool
    opportunity_activity: bool
    uncertainty_lcb: bool
```

Each arm resolves to one immutable capability set.

## 23. Evaluation manifest

Every run emits:

```python
@dataclass(frozen=True)
class QbdRunManifest:
    contract_id: str
    commit_sha: str
    policy_hash: str
    registry_hash: str
    data_fingerprint: str
    router_arm: str
    cost_policy: Mapping[str, Any]
    start_date: date
    end_date: date
    validation_status: str
```

Allowed validation status examples:

```text
SYNTHETIC
DEVELOPMENT_REUSE_NOT_INDEPENDENT_OOS
PSEUDO_OOS_DEVELOPMENT
FINAL_HOLDOUT_INDEPENDENT
FAILED_CONTRACT
```

Do not use ambiguous `OOS` labels without the contract status.

## 24. Freeze manifest

`QBD_ROUTER_V1_FROZEN_MANIFEST.json` should contain at least:

```json
{
  "contract_id": "TOP10_QBD_ROUTER_V1",
  "code_commit": "<sha>",
  "policy_hash": "<sha256>",
  "registry_hash": "<sha256>",
  "expert_artifact_hashes": {},
  "feature_schema_hashes": {},
  "adaptation_controller_hashes": {},
  "cost_policy": {},
  "evaluation_contract_hash": "<sha256>",
  "final_holdout": {
    "start": "YYYY-MM-DD",
    "end": "YYYY-MM-DD"
  }
}
```

Final Holdout runner validates the manifest before reading holdout outcomes.

## 25. Required decision trace

For every allocation date, persist enough information to reconstruct:

```text
what was known
what had matured
which experts were eligible
which health/kill rules fired
which candidate set survived
why a switch did or did not occur
what fallback was selected
what weights were applied
```

A compact JSONL decision trace is preferred over unstructured log prose.

## 26. Implementation constraints

- use typed dataclasses/enums/protocols or equivalent explicit schemas;
- prefer pure deterministic state-transition functions;
- keep mutable I/O at the orchestration boundary;
- no hidden global state;
- no model fitting inside the router replay;
- no future-return convenience object passed through router APIs;
- no live broker integration in V1;
- keep V4 implementation immutable under this contract;
- preserve historical artifacts for audit.
