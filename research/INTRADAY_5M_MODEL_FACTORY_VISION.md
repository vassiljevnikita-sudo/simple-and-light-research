# Intraday 5-Minute Model Factory — Vision Track

## Status

**VISION / ADJACENT RESEARCH TRACK — NOT PART OF THE CURRENT DYNAMIC-QBD IMPLEMENTATION PLAN**

This document captures a possible future research direction discussed alongside the Dynamic-QBD programme.

It must not be interpreted as an instruction to expand the current H1–H30 design space or to delay the existing Dynamic-QBD evidence programme.

The key architectural decision is:

> **A genuine 5-minute trading system should be treated as a separate research architecture, not as `H31` or another extension of the daily H-model family.**

The motivation is that moving from daily/next-session trading to 5-minute holding horizons changes the dominant problem from medium-horizon signal selection to short-horizon market microstructure, execution quality and rapid evidence accumulation.

---

# 1. Why this is a separate project

The current QBD programme is built around daily stock signals, H1–H30 prediction horizons, holding-day policies, causal refits, recalibration and stateful benchmark-relative portfolio replay.

A 5-minute system changes several assumptions simultaneously:

- target horizon;
- feature cadence;
- data source;
- transaction-cost scale relative to gross alpha;
- fill uncertainty;
- spread importance;
- market-impact importance;
- time-of-day effects;
- number of observations;
- speed at which model-health evidence matures.

Therefore the future research object should be conceptually separate:

```text
INTRADAY_5M_MODEL_FACTORY

= Intraday Data Contract
+ Microstructure Feature Policy
+ Executable-Net-Return Target
+ Intraday Training Policy
+ Refit/Recalibration Policy
+ Execution Model
+ Shadow Evaluation
+ Optional Dynamic Model Selection
+ Stateful Portfolio Execution
```

---

# 2. Central hypothesis

The main speculative hypothesis is not simply:

> smaller exchanges are better.

It is:

> **Less efficient but still sufficiently liquid markets may offer larger and/or longer-lived short-horizon pricing inefficiencies, while extremely illiquid markets may lose that advantage once spread, slippage, impact and fill risk are modeled realistically.**

The expected relation is therefore a possible efficiency–liquidity trade-off rather than monotonic preference for smaller markets.

Conceptually:

```text
very large / ultra-liquid markets
→ excellent execution
→ strongest professional competition
→ potentially smaller exploitable short-horizon alpha

moderately smaller but liquid markets
→ somewhat worse execution
→ potentially slower price discovery / less competition
→ possible net-alpha sweet spot

very illiquid markets
→ potentially large apparent inefficiencies
→ spread / impact / stale quotes / fill risk may dominate
```

This is an empirical hypothesis and must be tested rather than assumed.

---

# 3. The first economic target must be executable net alpha

For a 5-minute system, close-to-close or mid-price prediction is not sufficient evidence of tradable alpha.

A model should ideally be trained/evaluated on an economically executable target.

For a long trade, a conceptual target is:

```text
entry = next executable ask / conservative executable entry price
exit  = executable bid after the chosen holding horizon

net_return
= exit / entry - 1
- explicit fees
- modeled slippage
- modeled impact
```

The exact implementation depends on available data, but the guiding principle is fixed:

> **The prediction target should approximate realizable net return, not an unreachable reference-price movement.**

A system that predicts the next 5-minute close accurately but cannot capture the move after crossing the spread is not economically successful.

---

# 4. Data requirements

Pure 5-minute OHLCV is useful for initial exploratory work but should not be considered sufficient for production-grade execution evidence if materially better market-microstructure data are available.

Preferred hierarchy:

```text
minimum exploratory layer
- timestamped intraday OHLCV
- volume
- time-of-day

better
- trades
- bid
- ask
- bid/ask sizes
- spread

preferred for serious execution research
- quote stream
- trade stream
- Level-2 / order-book state where available
```

Potential feature groups:

### Own-security state

- recent returns;
- realized volatility;
- relative volume;
- spread;
- bid/ask imbalance;
- order-flow imbalance;
- depth;
- short-horizon momentum/reversal;
- local liquidity state.

### Cross-sectional / related-security state

- sector returns;
- related-stock returns;
- index/factor returns;
- cross-stock lead/lag relationships;
- cross-sectional dispersion.

### Cross-market state

Potentially important hypothesis:

```text
larger / faster market moves
        ↓
related smaller-market securities react with delay
        ↓
short-lived lead/lag opportunity
```

This should be tested causally and must account for overlapping market hours, timestamp synchronization and actual execution latency.

### Time-of-day state

Intraday behavior should not be assumed stationary across the trading session.

Potential time-state variables include:

- minutes since open;
- minutes to close;
- opening period;
- midday period;
- closing period;
- market-overlap windows;
- scheduled event windows where causally known.

---

# 5. Market-universe research axis

The initial research should not optimize simultaneously over an enormous exchange/security universe.

Use a small number of predeclared universe buckets to test the core hypothesis.

Conceptual buckets:

```text
U1  ultra-liquid US large caps
U2  liquid US mid caps
U3  liquid US small caps
U4  liquid German / major European equities
U5  liquid equities on smaller developed exchanges
U6  deliberately low-liquidity boundary sample
```

The purpose of U6 is primarily diagnostic: determine whether apparent gross alpha disappears after realistic execution costs.

For each universe report at least:

- gross predicted alpha;
- gross realized alpha;
- spread cost;
- slippage estimate;
- market-impact estimate;
- net executable alpha;
- trade frequency;
- fill rate / executable-opportunity rate;
- turnover;
- capacity;
- final portfolio wealth;
- downside wealth path.

The core comparison is:

```text
gross inefficiency
versus
net economically realizable alpha
```

---

# 6. 5-minute trading may solve part of the sparse-evidence problem

One major weakness of the current daily QBD candidates is sparse realized-trade evidence.

A 5-minute system can produce many more prediction opportunities and therefore potentially much faster shadow-evidence accumulation.

This does **not** mean millions of stock-time rows are independent observations.

Time dependence, cross-sectional dependence, overlapping targets and common market shocks remain critical.

However, compared with a model that may execute only a few dozen trades over several years, an intraday system could provide much faster evidence about:

- calibration drift;
- ranking deterioration;
- loss of net edge;
- time-of-day instability;
- market-state dependence;
- generation degradation.

This makes dynamic model-health and Champion–Challenger ideas potentially more statistically plausible in this future research track than in extremely sparse daily families.

---

# 7. Dynamic-model-selection implication

The Dynamic-QBD research principle still applies:

> Do not build a sophisticated selector before proving that current/past information predicts future relative model performance.

But the intraday environment may provide a materially denser evidence stream.

A possible future sequence is:

```text
multiple intraday models run in full shadow
        ↓
thousands of matured predictions accumulate rapidly
        ↓
current-fit health is measured frequently
        ↓
test whether health predicts future executable net alpha
        ↓
only if positive:
optional Champion / Challenger / allocation layer
```

Possible current-fit diagnostics include:

- executable prediction calibration;
- realized net return by score bucket;
- Rank IC;
- spread-adjusted score monotonicity;
- residual drift;
- fill-rate drift;
- adverse-selection drift;
- performance by time-of-day;
- performance by liquidity state.

---

# 8. Execution is a first-class model component

For a 5-minute system:

```text
Prediction Model
≈ only half of the economic problem

Execution Model
≈ equally important
```

The replay must model, as far as data permit:

- bid/ask crossing;
- explicit fees;
- slippage;
- latency assumptions;
- order size versus available liquidity;
- market impact;
- failed/partial fills where relevant;
- opening/closing auction behavior where relevant;
- stale/missing quotes;
- trading halts;
- corporate-action/security-identity continuity.

A strategy whose profitability disappears under a small realistic execution perturbation should not be considered robust alpha.

---

# 9. Capacity must be measured explicitly

A short-horizon strategy may have high percentage returns at small capital and much lower returns when position size increases relative to available liquidity.

Therefore performance should eventually be treated as a function of starting/deployed capital:

```text
W_T(C_0)
```

Possible capacity grid:

```text
€10k
€50k
€100k
€250k
€500k
€1m
```

The exact amounts are not normative; the principle is.

For every capital level measure:

- executable participation rate;
- spread/slippage/impact;
- rejected opportunities due to capacity;
- final wealth;
- benchmark-relative wealth where meaningful;
- capital impairment;
- turnover.

This connects directly to the existing project principle that the real objective is total portfolio wealth generated from the capital actually available through time.

---

# 10. Risk concept remains asymmetric and wealth-path based

The future intraday project should inherit the current risk philosophy:

- upside volatility is not automatically risk;
- diversification is not an objective by itself;
- concentration is information, not an automatic penalty;
- few trades are epistemic uncertainty, not automatically investment risk.

Primary economic risk remains capital impairment.

Relevant portfolio metrics include:

- final wealth;
- relative wealth where a valid benchmark exists;
- maximum drawdown;
- CDaR;
- expected shortfall;
- downside deviation / Sortino-type metrics;
- drawdown duration;
- recovery duration;
- time under water;
- capacity loss after drawdowns.

---

# 11. Initial research gates

This vision should only become an implementation programme if the earliest gates are promising.

## Gate I0 — executable 5-minute alpha exists

Question:

> Is there robust 5-minute predictive alpha after realistic executable spread/cost assumptions?

If no, stop before building a large architecture.

## Gate I1 — market-size / efficiency hypothesis

Question:

> After execution costs, do moderately smaller / less efficient but still liquid markets produce greater net alpha than ultra-liquid major markets?

If no, do not retain market-size complexity.

## Gate I2 — capacity

Question:

> Does the net-alpha result remain economically useful at the capital scale relevant to the project?

## Gate I3 — dynamic adaptation

Question:

> Does rolling refit/recalibration or current-fit health improve future net alpha versus a simpler frozen/periodically refit model?

## Gate I4 — model selection

Question:

> Can relative intraday-model performance/health be predicted well enough to justify dynamic Champion–Challenger allocation?

Each layer is conditional on the previous economic evidence.

---

# 12. Minimal future experiment

The first experiment should be deliberately small and falsifiable.

Example structure:

```text
3–5 predeclared market/liquidity universes
×
one or a very small set of model classes
×
one fixed 5-minute target
×
identical causal feature contract
×
identical execution assumptions
```

Primary outputs:

```text
GrossAlpha
ExecutionCost
NetAlpha
TradeFrequency
Capacity
FinalWealth
DownsideWealthPath
```

Only after this experiment demonstrates meaningful net alpha should the project expand into a large intraday Model Factory.

---

# 13. Relationship to Dynamic QBD

The two research tracks share principles but not necessarily models or data pipelines.

Shared principles:

- strict causal information cutoffs;
- point-in-time data;
- matured outcomes only;
- reproducible ModelGenerations;
- recalibration after refit;
- stateful portfolio accounting;
- algorithm-level final holdout freeze;
- negative-result acceptance;
- selectors only after meta-predictability evidence.

Different implementation concerns:

```text
Dynamic QBD / daily
- sparse trade evidence
- H1–H30
- holding-day / exit policy
- daily portfolio opportunity process

Intraday 5M
- dense prediction stream
- market microstructure
- executable quote-aware targets
- spread / fill / impact modeling
- time-of-day
- capacity constraints
- potentially much faster health estimation
```

The intraday project should therefore reuse generic causal/provenance/state concepts where appropriate, but should not be forced into the daily H-model interfaces if doing so distorts the economic problem.

---

# 14. Visionary target architecture

If all early gates eventually pass, a plausible endpoint is:

```text
        POINT-IN-TIME TRADE / QUOTE DATA
                     ↓
        Intraday Feature-State Builder
                     ↓
     Executable-Net-Return Target Builder
                     ↓
             Family Specifications
                     ↓
          Causal Model Generations
                     ↓
       Per-Generation Recalibration
                     ↓
      Intraday Generation Registry
                     ↓
        Full-Shadow Model Evaluation
                     ↓
      Execution-Aware Stateful Replay
                     ↓
    Matured Prediction / Fill Evidence
                     ↓
      Current-Fit / Economic Health
                     ↓
       [Dynamic Model Selection]
        only if evidence supports it
                     ↓
       Execution / Position Authority
                     ↓
          Stateful Portfolio NAV
                     ↓
          Final Portfolio Wealth
```

The square-bracketed selection layer is conditional, exactly as in Dynamic QBD.

---

# 15. Current decision

For now:

```text
Dynamic QBD
→ active research programme

Intraday 5M Model Factory
→ documented future vision / separate research track
```

Do not add 5-minute universes, microstructure features or intraday model-selection degrees of freedom to the current QBD research space.

The intraday idea should be revisited as a separate project when there is a deliberate decision to obtain/validate the required intraday execution data and run Gate I0.