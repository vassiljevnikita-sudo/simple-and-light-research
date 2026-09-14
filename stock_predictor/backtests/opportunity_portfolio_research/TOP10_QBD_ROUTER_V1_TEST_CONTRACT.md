# TOP10 QbD Router V1 — Test Contract

Status: `IMPLEMENTED_WITH_PREVIEW_GATES`
Parent architecture: `TOP10_QBD_ROUTER_V1_TARGET_ARCHITECTURE.md`  
Implementation plan: `TOP10_QBD_ROUTER_V1_IMPLEMENTATION_PLAN.md`

## 1. Purpose

This file defines the mandatory validation layers for `TOP10_QBD_ROUTER_V1`. A module is not complete when it merely runs. It is complete only when the tests required by its layer pass and the causal/restart invariants remain intact.

The full suite must separate:

1. unit/contract tests;
2. causality/leakage tests;
3. state/persistence tests;
4. synthetic known-truth tests;
5. null false-discovery tests;
6. historical pseudo-OOS replay;
7. cost and overfitting stress;
8. frozen Final Holdout validation.

## 2. Mandatory global invariants

### G1 — Matured evidence only

For decision date `t`, no outcome with `outcome_available_at > t` may modify:

- health state;
- candidate set;
- lifecycle state;
- allocation;
- promotion/demotion decision;
- change-point state.

### G2 — Future mutation invariance

If all data after date `t` are modified arbitrarily, every router decision on or before `t` must remain byte-for-byte identical.

### G3 — Determinism

Given identical:

- code commit;
- expert registry hash;
- policy hash;
- data fingerprint;
- initial router state;

all decisions and artifacts must be reproducible.

### G4 — Restart parity

```text
continuous replay
==
replay to cut date → persist → restart → continue
```

for decisions, state and final performance.

### G5 — Suspended shadow continuity

`SUSPENDED` means zero real allocation, not disabled inference. Shadow decisions and matured shadow outcomes must continue.

### G6 — Fail closed on structural mismatch

Artifact, policy, schema or registry mismatch must prevent real allocation and must not be silently repaired.

### G7 — Benchmark-relative alpha

Alpha health and router performance use net excess relative to the frozen MSCI benchmark definition. Market drawdown alone is not model decay.

### G8 — Opportunity-time inactivity

Non-opportunity days cannot advance the statistical inactivity clock.

### G9 — Switch-cost awareness

A hard switch cannot occur merely because challenger point estimate exceeds incumbent. The frozen switch-cost and confidence hurdle must be applied.

### G10 — Frozen holdout policy

Final Holdout execution must refuse a configuration that does not match the frozen manifest.

## 3. Unit test map

Expected test directory:

```text
stock_predictor/backtests/opportunity_portfolio_research/tests/qbd_router_v1/
```

Suggested files:

```text
test_contract.py
test_expert_registry.py
test_shadow_ledger.py
test_health_store.py
test_activity_monitor.py
test_alpha_health.py
test_regime_health.py
test_changepoint.py
test_kill_switches.py
test_candidate_set.py
test_router_state_machine.py
test_allocation.py
test_state_store.py
test_prequential_causality.py
test_ablation_arms.py
test_synthetic_scenarios.py
test_freeze_manifest.py
```

## 4. Contract and registry tests

Required cases:

- deterministic policy serialization;
- deterministic registry serialization;
- hash changes when any material field changes;
- duplicate expert IDs rejected;
- missing stock-expert artifact identity rejected;
- V4 controller identity mismatch rejected;
- missing holdout boundary rejected;
- live-trading mode impossible under this contract.

## 5. Shadow ledger tests

Required cases:

### S1 — Pending outcome isolation

Create a shadow trade whose result matures at `t+20`. Verify that the result cannot be queried through the router-facing matured-evidence API before `t+20`.

### S2 — Exact maturity boundary

At `outcome_available_at == current_date`, the result becomes available exactly once.

### S3 — Duplicate protection

Duplicate `(decision_date, expert_id)` decision record must fail or be idempotently identical by explicit policy.

### S4 — Suspended expert

Set expert state to `SUSPENDED`; verify:

```text
real allocation = 0
shadow decision = generated
shadow outcome = later matured normally
```

## 6. Health-store tests

Required cases:

- pending outcome does not update health;
- matured outcome updates once;
- cost is deducted before net-excess metric;
- structural health cannot be overwritten by strong alpha;
- low sample count produces larger uncertainty than large sample count for comparable observations;
- health snapshot contains evidence timestamp/source cursor.

## 7. Opportunity-time activity tests

### A1 — Long calendar inactivity, zero opportunity

Example:

```text
120 calendar days
0 eligible opportunities
0 trades
```

Expected: no activity anomaly solely from elapsed time.

### A2 — Rare strategy consistency

Expert historically trades at very low rate. A long zero-trade interval with limited opportunity exposure must remain plausible.

### A3 — Missing expected trades

Expert historically trades frequently when opportunities occur. Provide many eligible opportunities and zero trades.

Expected: posterior predictive zero-trade probability falls; state moves through configured warning levels.

### A4 — Posterior persistence

Save/restart produces the same Gamma-Poisson posterior as continuous execution.

## 8. Alpha/uncertainty tests

Required cases:

- positive gross return but negative benchmark-relative return is not positive alpha;
- positive gross alpha erased by costs is not positive net alpha;
- outlier-only alpha produces weaker conservative evidence than broad repeated alpha with same point mean where uncertainty method supports this distinction;
- LCB reacts to effective sample size;
- deterministic/bootstrap seed discipline produces repeatable results.

## 9. Regime-shrinkage tests

### R1 — No local evidence

Expected `rho ≈ 0` or frozen minimum; score is effectively global.

### R2 — Sparse local evidence

One or a few lucky observations cannot dominate global evidence.

### R3 — Mature local evidence

As effective local sample size grows, local score receives more weight according to the frozen shrinkage function.

### R4 — Future regime leakage

Future regime outcomes cannot alter past local score.

## 10. Change-point tests

### C1 — Stable null

Stationary synthetic net-excess series should not continuously trigger high change probability.

### C2 — Abrupt decay

Series switches from positive to negative edge. Detector should respond with bounded delay under frozen configuration.

### C3 — No direct champion switch

High change probability alone cannot call allocation/promotion code directly; only health/probation inputs may change.

### C4 — Restart parity

Persisted BOCPD state must reproduce continuous posterior path.

## 11. Kill-switch tests

Required matrix:

| Structural | Activity | Alpha | Regime | Expected |
|---|---|---|---|---|
| FAIL | any | any | any | no real allocation |
| PASS | WARN | PASS | PASS | policy-defined watch/probation |
| PASS | PASS | FAIL | PASS | alpha demotion path |
| PASS | PASS | PASS | FAIL | regime-specific zero/reduced allocation, not permanent death |

Every output must contain a stable reason code.

## 12. Candidate-set tests

Required cases:

- clearly inferior expert excluded;
- indistinguishable experts both retained;
- lucky high-mean/high-uncertainty expert not automatically selected;
- structurally failed expert excluded regardless of return history;
- no qualified stock experts leaves `MSCI_WORLD`/`CASH` available.

## 13. Lifecycle tests

Test every allowed transition and several forbidden transitions.

Required paths:

```text
SHADOW → CANDIDATE
CANDIDATE → ENSEMBLE
CANDIDATE → CHAMPION
CHAMPION → PROBATION
PROBATION → CHAMPION
PROBATION → SUSPENDED
SUSPENDED → CANDIDATE
```

Required properties:

- transition reason persisted;
- evidence timestamp is <= transition date;
- minimum-duration/hysteresis rules enforced;
- `SUSPENDED` cannot receive real capital;
- reactivation requires fresh shadow evidence.

## 14. Allocation/promotion tests

### P1 — Switch cost blocks weak challenger

```text
challenger edge > incumbent edge
but challenger edge - incumbent edge < switch cost
```

Expected: no hard switch.

### P2 — Confidence blocks weak evidence

Point estimate superior, LCB <= 0 after switch cost.

Expected: no hard switch.

### P3 — Strong challenger

LCB relative edge after switch cost > 0 and minimum evidence met.

Expected: promotion allowed subject to lifecycle policy.

### P4 — Hysteresis

Oscillating near-tie scores must not generate repeated daily switching.

### P5 — Ensemble fallback

Several qualified indistinguishable experts produce ensemble allocation rather than arbitrary hard winner.

### P6 — Benchmark/Cash fallback

No stock expert meets minimum net-edge evidence.

Expected: frozen fallback logic chooses benchmark/Cash.

## 15. End-to-end causality tests

### E1 — Future return poisoning

Run replay to completion. Mutate all realized returns after cutoff `t`. Rerun.

Expected:

```text
all router outputs <= t identical
```

### E2 — Outcome-delay poisoning

Change maturity delay of a future trade. Verify health changes only from the new maturity date onward.

### E3 — Full-table API prohibition

Selector/router API must not accept an unrestricted future returns table/array. Test interface structure as well as behaviour.

### E4 — Decision trace

Every allocation decision must be traceable to:

```text
policy hash
registry hash
health snapshot
candidate set
lifecycle state
matured evidence cursor
```

## 16. Synthetic known-truth scenarios

Synthetic suite must include at least:

### Scenario 1 — Champion decay

```text
Year 1: A strong, B neutral
Year 2: A decays, B strong
```

Measure exit delay for A and entry delay for B.

### Scenario 2 — Rare profitable expert

C trades rarely but has valid opportunities only occasionally.

Expected: no suspension from calendar silence alone.

### Scenario 3 — Opportunity-response failure

G historically responds frequently when opportunity exists, then receives many opportunities but produces no trades.

Expected: activity anomaly develops.

### Scenario 4 — Recovery/reactivation

H performs, deteriorates, is suspended, later regains positive shadow evidence.

Expected: `SUSPENDED → CANDIDATE` path is exercised.

### Scenario 5 — Lucky noise expert

D receives a short lucky streak without true positive expected edge.

Measure false promotion behaviour.

### Scenario 6 — Structural failure

Artifact/schema identity changes midstream.

Expected: immediate fail-closed allocation removal.

## 17. Null false-discovery test

Construct a universe where all stock experts have identical expected return/alpha.

Repeated seeded simulations measure:

```text
false champion discovery frequency
unnecessary switch frequency
average champion duration
benchmark fallback frequency
```

This is a mandatory gate for assessing whether the router manufactures apparent winners from noise.

## 18. A–K ablation parity tests

For arms A-K:

- identical expert predictions;
- identical shadow ledger;
- identical outcome maturity rules;
- identical costs;
- identical benchmark series;
- identical execution simulation.

Adjacent arms may differ only by the named capability in the architecture.

The engine should expose a normalized arm manifest so parity can be asserted automatically.

## 19. Historical pseudo-OOS validation

At each date:

```text
use state from past matured evidence only
→ decide allocation
→ advance time
→ mature later outcomes
→ update state
```

Required output metadata:

```text
commit SHA
contract ID
policy hash
registry hash
data fingerprint
router arm
cost policy
start/end date
holdout status label
```

No historical replay may be labelled independent OOS merely because the code is causal if its rules were designed using that same window.

## 20. Metrics validation

Tests must verify formulas for:

- net excess CAGR;
- Information Ratio;
- Sortino;
- relative MaxDD;
- switch cost;
- champion duration;
- entry/exit/reactivation delay;
- captured alpha fraction;
- concentration shares.

Hindsight oracle calculations must be isolated from router inputs.

## 21. Cost-stress tests

Replay identical decisions/trades under:

```text
native cost model
10 bps
20 bps
30 bps
50 bps roundtrip
```

Verify:

- only cost assumptions change;
- switch costs are included;
- performance is reported gross, cost and net;
- stress result cannot feed back into the same frozen run's router decisions unless that cost policy was the predeclared router policy.

## 22. Multiple-testing/overfitting validation

The research artifact must preserve the attempted configuration inventory.

Tests/checks should prevent:

- dropping failed arms from summary denominator;
- selecting only favourable seeds;
- silently changing metric definition across arms;
- reporting DSR/PBO inputs that exclude tested variants.

## 23. Freeze-manifest tests

Required mismatch tests:

- changed code commit;
- changed policy;
- changed expert registry;
- changed feature schema;
- changed V4 controller hash;
- changed cost policy;
- changed holdout boundary;
- changed evaluation metric definition.

Every mismatch must block Final Holdout execution.

## 24. Final Holdout acceptance contract

A Final Holdout result is valid only if:

1. freeze manifest was created before opening holdout;
2. all manifest hashes match;
3. no tuning configuration is accepted by the holdout runner;
4. router decisions use matured evidence only;
5. algorithmic adaptation follows the frozen state machine/policy;
6. the full decision log is replayable;
7. result is labelled once as independent holdout outcome for this contract version.

Any material change after observing the result creates a new research version rather than retroactively modifying `TOP10_QBD_ROUTER_V1`.
