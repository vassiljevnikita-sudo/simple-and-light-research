> **Historical scientific background:** This literature/evidence review remains useful context, but it is not the current empirical verdict of the repository. Later completed experiments are indexed in [DYNAMIC_QBD_RESULTS_INDEX.md](DYNAMIC_QBD_RESULTS_INDEX.md); current scientific state is [DYNAMIC_QBD_CURRENT_STATE.md](DYNAMIC_QBD_CURRENT_STATE.md).

# QBD Dynamic System — scientific evidence, counter-evidence, and empirical gate

## Status

Research note for the `simple-and-light` QBD programme.

This document records the scientific rationale for the proposed dynamic QBD model factory and, equally importantly, the evidence against assuming that dynamic champion selection will work.

The architecture is treated as **plausible but empirically open**. The immediate research objective is therefore not to optimise a Champion–Challenger router, but to falsify or support the premise that relative historical QBD-family performance contains useful information about future relative QBD-family performance.

---

## 1. Core hypothesis

The long-term system under consideration is:

```text
QBD_DYNAMIC_SYSTEM

= Training Policy
+ Refit Cadence
+ Calibration Policy
+ QBD Candidate Families
+ Shadow Evaluation
+ Champion Selection
+ Switch Hysteresis
+ Portfolio Execution
```

The intended causal process is:

```text
new fully matured data
        ↓
causal refit of the QBD suite
        ↓
separate recalibration of every new fit
        ↓
new fit-specific threshold
        ↓
all families continue in shadow
        ↓
only matured OOS evidence becomes observable
        ↓
optional champion selection / switch
```

The important conceptual shift is that a particular H11, H24 or H28 fitted artifact is no longer the trading system. The **training-and-selection process itself** becomes the trading system.

However, the Champion–Challenger layer is only justified if a weaker proposition can first be established:

> Relative QBD-family performance observed up to time `t` contains information about relative QBD-family performance after `t`.

Formally, the target is approximately:

\[
P(Excess_{i,t+1:t+h} \mid Information_{\leq t}, PastPerformance_{i,\leq t})
\]

for horizons such as one and three months.

The central null hypothesis is:

> **H0:** Past relative QBD performance has no robust forward predictive information. Apparent winners are mainly sampling noise, sparse trades, threshold effects, exit timing, correlated variants, and concentrated lucky outcomes.

If H0 cannot be rejected in a strict temporal OOS design, a performance-driven Champion–Challenger router should not be built.

---

## 2. The scientific literature is broader than four papers, but the exact QBD hypothesis remains weakly established

The scientific basis is not limited to a handful of generic Mixture-of-Experts papers. There is a meaningful literature on persistence in forecast-model performance, factor timing, factor momentum, performance-dependent forecast combinations, model instability, and the difficulty of beating simple combinations.

At the same time, there is no strong established literature proving the exact QBD architecture:

```text
cross-sectional equity ML families
× H1–H30
× different holding / exit / max_names policies
× periodic causal refit
× fit-specific recalibration
× sparse trading
× family/plateau grouping
× past OOS family performance
→ next 1M / 3M family excess
× hard Champion–Challenger deployment
```

Therefore the literature provides **credible H1 mechanisms and credible H0 mechanisms**, but not a direct validation of the full design.

---

## 3. Direct evidence that relative model performance can be persistent

### 3.1 Aiolfi & Timmermann (2006): Persistence in forecasting performance and conditional combination strategies

Aiolfi and Timmermann study persistence in the **relative forecasting performance** of linear and nonlinear time-series models over a broad set of G7 macroeconomic variables.

They report strong evidence that both top and bottom forecasting models can exhibit persistence. They then use historical model performance to form clusters and construct conditional forecast combinations, with shrinkage toward equal weights, and test these procedures out of sample.

This is one of the closest direct analogues to the QBD gate question:

```text
past relative model performance
        ↓
conditional model grouping / weighting
        ↓
future OOS forecasting performance
```

Reference:

Aiolfi, M. & Timmermann, A. (2006), *Persistence in forecasting performance and conditional combination strategies*, Journal of Econometrics 135(1–2), 31–53. DOI: https://doi.org/10.1016/j.jeconom.2005.07.015

### Interpretation for QBD

This paper supports the proposition that **past relative model quality can contain information**. It does not support naive winner rotation. The proposed method pools models and shrinks toward equal weights, which itself is evidence that estimation noise is a central problem.

The relevant QBD implication is:

> Testing whether historical family ranking has forward value is scientifically justified, but the natural baseline is pooling / averaging, not `argmax(previous performance)`.

---

## 4. Directly related financial evidence: strategy/factor performance can exhibit momentum

### 4.1 Ehsani & Linnainmaa (2022): Factor Momentum and the Momentum Factor

Ehsani and Linnainmaa document positive autocorrelation in factor returns. In their sample, factors that performed positively over the previous year subsequently earn substantially higher average monthly returns than factors following a negative year.

Reference:

Ehsani, S. & Linnainmaa, J. T. (2022), *Factor Momentum and the Momentum Factor*, Journal of Finance 77(3), 1877–1919. DOI: https://doi.org/10.1111/jofi.13131

### Interpretation for QBD

This is not model-selection research, but it provides an important economic analogue:

```text
past strategy/factor performance
        ↓
future strategy/factor performance
```

can display persistence.

This means H0 is not trivially true. Strategy-level performance momentum is a legitimate empirical phenomenon to test in QBD families.

It does **not** imply that sparse, highly correlated QBD policy returns will show the same effect or that a one-month Champion rule will work.

---

## 5. Factor timing provides evidence that expected returns of strategy families can vary predictably

### 5.1 Haddad, Kozak & Santosh (2020): Factor Timing

Haddad, Kozak and Santosh show that market-neutral equity-factor returns are strongly and robustly predictable and develop dynamic factor-timing portfolios that materially improve performance relative to static factor investing.

Reference:

Haddad, V., Kozak, S. & Santosh, S. (2020), *Factor Timing*, Review of Financial Studies 33(5), 1980–2018. NBER DOI: https://doi.org/10.3386/w26708

### Interpretation for QBD

This supports the more general proposition that:

> The expected return of a strategy family need not be constant through time.

This is relevant to the observation that H11 may dominate in one period and fail in another.

Again, this does not establish that trailing QBD performance is the correct timing signal. It establishes that **time-varying strategy attractiveness is economically plausible**.

---

## 6. Performance-dependent model combinations can work OOS

### 6.1 Pettenuzzo & Ravazzolo (2016): Optimal Portfolio Choice Under Decision-Based Model Combinations

Pettenuzzo and Ravazzolo construct model-combination weights that depend explicitly on the **past forecast performance** of the component models through a utility-based objective. They apply the approach to stock-return forecasting and report improvements in both statistical and economic OOS predictability relative to competing combination schemes.

Reference:

Pettenuzzo, D. & Ravazzolo, F. (2016), *Optimal Portfolio Choice Under Decision-Based Model Combinations*, Journal of Applied Econometrics 31(7), 1312–1332. DOI: https://doi.org/10.1002/jae.2502

### Interpretation for QBD

This is highly relevant evidence for a dynamic QBD system because it demonstrates that historical forecast performance can be used in a **financial** setting to alter model weights prospectively.

However, it favours performance-dependent combination rather than proving that hard winner-take-all Champion selection is superior.

---

## 7. Model instability matters — and simple combinations are difficult to beat

### 7.1 Zhang, He, Jacobsen & Jiang (2020): Forecasting stock returns with model uncertainty and parameter instability

Zhang et al. compare sophisticated model averaging and variable-selection methods for stock-return forecasts. Their baseline results confirm the strength of simple combinations. More sophisticated approaches improve when parameter instability is explicitly accommodated.

Reference:

Zhang, H., He, Q., Jacobsen, B. & Jiang, F. (2020), *Forecasting stock returns with model uncertainty and parameter instability*, Journal of Applied Econometrics 35, 629–644. DOI: https://doi.org/10.1002/jae.2747

### Interpretation for QBD

This is directly relevant to the QBD premise that old fitted models can become stale.

It supports:

```text
model instability is real
→ static forever is questionable
```

but also reinforces the main counterhypothesis:

```text
simple pooling
can outperform estimated dynamic selection
because selector estimation is noisy
```

---

## 8. Evidence from large trading-rule universes: persistence can be short-lived

Sermpinis, Hassanniakalager, Stasinakis and Psaradellis examine more than 21,000 technical trading rules over 12 MSCI market categories and more than 240,000 hypotheses, explicitly applying false-discovery controls.

They find evidence of short-term value and persistence, largely linked to short-term momentum, rather than universal long-lived superiority.

Reference:

Sermpinis, G., Hassanniakalager, A., Stasinakis, C. & Psaradellis, I. (2021), *Technical analysis profitability and Persistence: A discrete false discovery approach on MSCI indices*, Journal of International Financial Markets, Institutions and Money 73, 101353. DOI: https://doi.org/10.1016/j.intfin.2021.101353

### Interpretation for QBD

This is useful because QBD also contains a large correlated rule/model universe.

It supports both sides:

- short-horizon persistence is plausible;
- large rule universes create severe multiple-testing risk;
- persistence may decay quickly;
- a strategy that was historically best need not remain best for long.

---

## 9. Strong counter-evidence: simple diversification and combination are hard to beat

### 9.1 DeMiguel, Garlappi & Uppal (2009)

DeMiguel, Garlappi and Uppal compare 14 portfolio-allocation models across seven empirical datasets with the naive 1/N rule. No evaluated optimized model consistently beats 1/N in Sharpe ratio, certainty-equivalent return, and turnover.

Reference:

DeMiguel, V., Garlappi, L. & Uppal, R. (2009), *Optimal Versus Naive Diversification: How Inefficient is the 1/N Portfolio Strategy?*, Review of Financial Studies 22(5), 1915–1953. DOI: https://doi.org/10.1093/rfs/hhm075

### Interpretation for QBD

This is not a direct model-routing test, but the statistical lesson is highly relevant:

> Theoretical gains from estimating the best allocation can be more than offset by estimation error.

For QBD, the corresponding null is:

```text
estimated dynamic Champion
may be worse than
static/equal-weight family exposure
```

This is why every dynamic selector must be compared with simple baselines.

---

## 10. The evidence therefore does NOT justify naive recent-winner selection

The literature supports the possibility of persistence, instability, and dynamic weighting. It does not justify:

```text
R04 best this month → trade R04
R09 best next month → immediately switch to R09
R07 best afterwards → immediately switch to R07
```

The expected failure mode is performance chasing caused by estimation error.

The evidence instead motivates a spectrum of increasingly aggressive approaches:

```text
static best family
        ↓
equal-weight / pooled families
        ↓
performance-conditioned combination
        ↓
family ranking
        ↓
Champion–Challenger with hysteresis
```

A hard Champion should therefore be treated as an empirical hypothesis, not as the default implied by the literature.

---

## 11. Scientific assessment of the individual QBD hypotheses

| Hypothesis | Scientific assessment |
|---|---|
| Model/strategy performance varies through time | **Well supported** |
| Factor/strategy performance can exhibit persistence | **Supported in multiple settings** |
| Past model forecast performance can improve conditional combinations | **Supported by direct forecasting evidence** |
| Periodic causal refitting is preferable to permanently frozen models | **Plausible and common, but optimal cadence is not established for QBD** |
| A newly refitted model should be recalibrated instead of blindly reusing the old absolute threshold | **Methodologically strong** |
| Past QBD-family performance predicts future QBD-family excess | **Empirically open** |
| The best recent QBD family should receive all capital | **Weakly supported / high noise risk** |
| Champion–Challenger with hysteresis beats static or pooled baselines | **Plausible but unproven** |
| Equal-weight / pooled models are a strong baseline | **Strongly supported** |
| Complex routing automatically improves OOS performance | **Not supported** |

Overall classification for `QBD_DYNAMIC_SYSTEM`:

> **Plausible, but empirically open.**

---

## 12. The correct next experiment is a predictability gate, not a router optimisation

Before implementing further Champion–Challenger machinery, test:

> Can only information observable at assessment date `t` predict which QBD family will outperform over the next 1M or 3M?

The first gate should deliberately use only historical performance information.

### Gate 1 — performance only

Candidate features:

```text
trailing 1M excess
trailing 3M excess
trailing 6M excess
trailing 12M excess
EWMA excess
information ratio
relative drawdown
recent-minus-long-term performance
positive-period fraction
```

Do **not** initially include:

```text
market regime
VIX / volatility state
H/D/N identifiers as predictive features
prediction-score health
Rank IC
calibration diagnostics
ticker identities
```

These can be introduced only in later gates so that the information source remains interpretable.

### Targets

Run separate experiments for:

```text
future 1M benchmark excess
future 3M benchmark excess
```

and evaluate three tasks separately:

1. **Direction:** will family excess be positive?
2. **Magnitude:** how large will future family excess be?
3. **Ranking:** which family will rank highest relative to other families?

The ranking task is the most directly relevant to Champion–Challenger.

---

## 13. Time, not policy rows, is the effective statistical sample

A QBD panel such as:

```text
10,000 policies × 50 months
```

must not be treated as 500,000 independent observations.

The policies share the same markets, many policies are near-duplicates, and observations within a time block are strongly dependent.

Therefore:

```text
TRAIN = past only
TEST  = future time block
```

No random row split is acceptable for the meta-selector.

The meta-model itself must be walk-forward/prequential.

Standard errors and uncertainty should be based on appropriate temporal blocks, HAC methods, or block bootstrap rather than IID row assumptions.

---

## 14. Family-level analysis should be primary

Several QBD policies may be almost the same strategy:

```text
H28 D21 N2
H28 D21 N3
H28 D21 N4
H28 D21 N5
H28 D21 N6
```

Treating them as independent challengers exaggerates the amount of evidence and makes meta-overfitting easier.

Preferred hierarchy:

```text
QBD policy surface
        ↓
structurally defined family / plateau
        ↓
family-level forward predictability
        ↓
optional policy selection within winning family
```

Family definitions themselves must not use future OOS information. For Gate 1, structurally predefined families are safer than clusters discovered using the entire history.

---

## 15. Sparse trading changes the interpretation of inactivity

A sparse QBD family can go for months without a trade while behaving exactly as designed.

Therefore:

```text
calendar inactivity
≠
negative evidence
```

and a rule such as:

```text
confidence *= decay_per_month
```

is not justified.

In addition, if idle capital stays predominantly in the benchmark sleeve, the economic cost of keeping a sleeping Champion is mainly **missed challenger alpha**, not missing the entire market return.

This supports conservative switching rules if a Champion–Challenger system is eventually justified.

---

## 16. Prediction-level health is a separate second hypothesis

Even if Gate 1 fails, this does not prove that dynamic family selection is impossible.

There is much richer causal information in matured shadow predictions than in sparse executed trades.

A second gate may test whether current-fit diagnostics such as:

```text
recent Rank IC
calibration error
calibration slope
top-score realized excess
score monotonicity
prediction residual drift
score dispersion
```

predict future family performance.

Therefore:

```text
Gate 1 FAIL
→ reject trailing-performance Champion selection
```

but not necessarily:

```text
Gate 1 FAIL
→ reject every possible dynamic selector
```

---

## 17. Regime information belongs only in a later gate

If historical performance and/or current-fit health contain useful information, a third gate can test whether market state adds incremental predictive power:

```text
performance history
+ current-fit health
+ market state
→ future family performance
```

Candidate state variables could include trend, volatility, breadth, dispersion, and stress.

They should **not** be introduced before the simpler hypotheses are evaluated, because doing so creates a much larger research search space and makes a false-positive selector easier to discover.

---

## 18. Refit, recalibration and switching must be tested separately

The clean architectural ablation is:

```text
A  Frozen model
   + frozen calibration

B  Frozen model
   + rolling recalibration

C  Rolling causal refit
   + rolling recalibration

D  Rolling causal refit
   + rolling recalibration
   + dynamic family selection
```

Interpretation:

```text
B - A = incremental value of recalibration
C - B = incremental value of refitting
D - C = incremental value of dynamic selection
```

This separation is essential. A failure of an old controller based on frozen models does not imply that causal refit + recalibration fails.

Likewise, successful refitting does not imply that dynamic switching adds value.

---

## 19. Family evidence and current-generation evidence must remain separate

With repeated refitting, `H24` is not a single model artifact:

```text
H24_fit_Jan
H24_fit_Feb
H24_fit_Mar
...
```

Two evidence layers are required:

### Family evidence

```text
How robust has H24 been across many causal refits?
```

### Current-fit evidence

```text
How healthy is the current H24 generation?
```

Historical success of old generations should not be attributed completely to a newly trained generation, but family history should not be discarded either.

This is naturally compatible with hierarchical / partial-pooling logic if later evidence shows that such complexity is justified.

---

## 20. Horizon-specific outcome maturity is mandatory

At assessment time `t`, an observation may only enter the selector if:

```text
terminal_date <= assessment_date
```

H1 therefore accumulates ground truth faster than H30.

The selector must never equalise information sets by accidentally using outcomes that have not yet matured for the longer horizon.

This maturity rule is more important than nominal calendar lookbacks.

---

## 21. Economic success criteria

Do not declare success from classification accuracy alone.

The key metrics should include:

```text
monthly cross-sectional rank IC
future excess of predicted top family
top-minus-bottom future family excess
selected-family benchmark excess
information ratio
relative max drawdown
switch turnover / costs
regret versus ex-post oracle
captured oracle alpha
```

Every dynamic model must beat simple alternatives such as:

```text
benchmark
static best-development family
previous champion held
trailing 1M winner
trailing 3M winner
trailing 6M winner
equal-weight family portfolio
simple forecast combination / pooling
```

A selector with positive excess but inferior performance to a simple pooled baseline has not demonstrated useful routing skill.

---

## 22. Falsification criteria

A performance-based Champion–Challenger architecture should be rejected if, under a strict temporal outer OOS design:

1. historical relative family performance has no stable positive relationship with future 1M or 3M relative excess;
2. predicted top families do not subsequently outperform lower-ranked families;
3. the selector does not beat static and pooled baselines after costs;
4. gains are concentrated in a very small number of periods, tickers, or trades;
5. apparent gains disappear under temporal/block bootstrap or multiple-testing correction;
6. selector performance is unstable across reasonable pre-registered lookback choices;
7. switching gains disappear after realistic turnover costs.

A failed Gate 1 means:

> **Do not build a trailing-performance Champion–Challenger.**

It does **not** invalidate causal periodic refitting, recalibration, or a later test of current-fit prediction-health signals.

---

## 23. Research architecture implied by the evidence

```text
             QBD MODEL FACTORY
                    │
       causal refit + recalibration
                    │
                    ▼
         FAMILY GENERATION PANEL
                    │
                    ▼
           SHADOW REPLAY ENGINE
                    │
             matured causal OOS
                    │
        ┌───────────┴───────────┐
        ▼                       ▼
 FAMILY HISTORY          CURRENT FIT HEALTH
        │                       │
        └───────────┬───────────┘
                    ▼
          META-PREDICTABILITY GATES
                    │
             ┌──────┴──────┐
             │             │
           FAIL           PASS
             │             │
             ▼             ▼
 no performance-     Champion–Challenger
 based switching          research
```

The next implementation priority is therefore **factory/replay fidelity and the predictability gate**, not a more sophisticated router.

---

## 24. Bottom line

The literature does **not** establish that the QBD Champion–Challenger architecture will work.

It does establish enough to make the hypothesis scientifically legitimate:

- relative forecasting-model performance can be persistent;
- factor/strategy returns can exhibit persistence and time variation;
- performance-dependent forecast combination can improve OOS financial forecasts;
- parameter/model instability matters;
- but simple pooling/equal weighting is a very strong benchmark because estimated dynamic decisions suffer from estimation error;
- large candidate universes create severe multiple-testing and data-snooping risks.

Therefore the current scientific classification is:

> **QBD_DYNAMIC_SYSTEM: plausible, but empirically open.**

The most defensible next question is not:

> Which model should be Champion?

It is:

> **Is there any ex-ante observable property — starting with past causal family performance — that reliably identifies which QBD family will deliver higher benchmark excess over the next one to three months?**

Only a positive answer to that question justifies investing further in performance-based Champion–Challenger logic.

---

## Key references

1. Aiolfi, M. & Timmermann, A. (2006). *Persistence in forecasting performance and conditional combination strategies*. Journal of Econometrics 135(1–2), 31–53. https://doi.org/10.1016/j.jeconom.2005.07.015
2. Haddad, V., Kozak, S. & Santosh, S. (2020). *Factor Timing*. Review of Financial Studies 33(5), 1980–2018. https://doi.org/10.3386/w26708
3. Ehsani, S. & Linnainmaa, J. T. (2022). *Factor Momentum and the Momentum Factor*. Journal of Finance 77(3), 1877–1919. https://doi.org/10.1111/jofi.13131
4. Pettenuzzo, D. & Ravazzolo, F. (2016). *Optimal Portfolio Choice Under Decision-Based Model Combinations*. Journal of Applied Econometrics 31(7), 1312–1332. https://doi.org/10.1002/jae.2502
5. Zhang, H., He, Q., Jacobsen, B. & Jiang, F. (2020). *Forecasting stock returns with model uncertainty and parameter instability*. Journal of Applied Econometrics 35, 629–644. https://doi.org/10.1002/jae.2747
6. Sermpinis, G., Hassanniakalager, A., Stasinakis, C. & Psaradellis, I. (2021). *Technical analysis profitability and Persistence: A discrete false discovery approach on MSCI indices*. Journal of International Financial Markets, Institutions and Money 73, 101353. https://doi.org/10.1016/j.intfin.2021.101353
7. DeMiguel, V., Garlappi, L. & Uppal, R. (2009). *Optimal Versus Naive Diversification: How Inefficient is the 1/N Portfolio Strategy?* Review of Financial Studies 22(5), 1915–1953. https://doi.org/10.1093/rfs/hhm075
8. White, H. (2000). *A Reality Check for Data Snooping*. Econometrica 68(5), 1097–1126.
9. Hansen, P. R. (2005). *A Test for Superior Predictive Ability*. Journal of Business & Economic Statistics 23(4), 365–380.
10. Harvey, C. R., Liu, Y. & Zhu, H. (2016). *…and the Cross-Section of Expected Returns*. Review of Financial Studies 29(1), 5–68.
