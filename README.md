# simple-and-light-research

Can a stock-picking model be kept up to date over ten years of data without using information it could not have had at the time? This repository holds my research on that question, and most of the answers are "no" or "not yet".

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/readme/results-chart-dark.png">
  <img alt="Left: one test setting, EUR 10,000 invested from 2020-08-31 to 2023-12-29. Retraining only when a new block of outcomes is complete ends at EUR 15,773, training once ends at EUR 13,462, retraining every month at EUR 10,635, re-tuning the buy threshold every month at EUR 10,385; the URTH index fund ends at EUR 12,920. Right: across the 2,790 model families with a fixed holding period, retraining on new outcomes changes the yearly return above benchmark by a median of +1.2 percentage points but is better in only 2 of 6 years; retraining every month changes it by -4.8 points and is worse in 4 of 6 years." src="docs/readme/results-chart.png">
</picture>

The chart is drawn by [docs/readme/make_results_chart.py](docs/readme/make_results_chart.py) from three committed `summary.json` files. Research only: no real money, no live orders, not investment advice.

## Try it

- Demo (web app): [simple-light-qbd.expo.app](https://simple-light-qbd.expo.app/)
- Rebuild the chart from the committed results: `pip install matplotlib`, then `python docs/readme/make_results_chart.py` from the repository root.
- Read one result: [the Development Run report](artifacts/dynamic-qbd-development-run-2016-2025-final/REPORT.md) and its [summary.json](artifacts/dynamic-qbd-development-run-2016-2025-final/summary.json).

## My role

I am Nikita Vassiljev ([GitHub](https://github.com/vassiljevnikita-sudo)), the sole human author. I designed the research and ran the experiments; I built the code and documentation with AI coding agents (OpenAI Codex, Claude).

## Tech stack

Python 3 with pandas, NumPy, scikit-learn (Ridge regression and histogram gradient boosting) and PyArrow/Parquet. Large runs use process pools with a fixed memory ceiling. The chart uses matplotlib.

## Limitations

- This is a curated public copy. Market data, model files and some private modules (for example `stock_predictor.v5`) are not included, so the self-tests stop at import and the experiments cannot be rerun from this repository. The chart script is the one thing that runs from a fresh clone. The [chart check](.github/workflows/chart.yml) workflow reruns it on every push, compares the images with the committed ones and checks that the files this README links to exist. It does not rerun any experiment.
- Every number here comes from data before 2026-07-25. The sealed test period that starts on that date has not been opened, so no result has been checked on truly unseen data.
- Usable history starts in 2016. A replay of the early years cannot pretend that ten years of history already existed.
- The Fold-Clock studies (retrain only when a new block of outcomes is complete) used model recipes chosen with knowledge of the whole 2016-2025 period. They explain mechanisms; they do not prove that a live system would have made the same choices.
- The prediction row count below is recorded in [research/DYNAMIC_QBD_DATA_AND_RUNTIME.md](research/DYNAMIC_QBD_DATA_AND_RUNTIME.md). The run's committed `summary.json` holds the family count but not the row count.

## The question

Every trading day a model scores stocks, and a simple rule buys the top-scored ones. Markets change, so the model goes stale. Should it be retrained every month, only when enough new outcomes have arrived, or never? And can a second layer learn which of many models to trust at a given time?

## Method

- Chronological replay of data and outcomes. On each simulated day, training and scoring use only market data and outcomes that existed on that day. A prediction counts as evidence only after its outcome is known. Training and calibration windows are purged and do not overlap. This covers the data, not the choice of model recipes (see the Fold-Clock note under Limitations).
- Trading costs of 20 basis points per round trip are charged. Money not invested in stocks sits in the URTH world index fund, which is also the benchmark.
- A model family is one combination of three settings: how many trading days ahead the model looks (H, 1 to 30), how long a bought stock is held (D, 1 to H days) and how many stocks are held at most (N, 1 to 6). That gives 6 × (1 + 2 + ... + 30) = 2,790 families.
- The Development Run over 2016-2025 evaluated all 2,790 families over 119 monthly checkpoints and produced 1,632,350,262 matured prediction rows.
- Each experiment states its pass/fail gates before it runs. The Fold-Clock gates, for example, require the effect in at least 60% of years and 60% of groups of neighbouring settings, not just on average.

## What worked

- The full evaluation runs on one workstation. The first version ran out of memory when it tried to load all 1.63 billion rows into one table; streamed, resumable aggregation finished the run with a peak of about 5.6 GB.
- Retraining every calendar month is clearly worse than retraining only when a new block of outcomes is complete. Across the 2,790 families with a fixed holding period, the median change was -4.8 percentage points of yearly return above benchmark, worse in 4 of 6 years and in 89% of setting groups. This gate passed. The same study also ran 2,610 families in which a model decides when to sell (5,400 in total); all surface-study figures in this README are for the 2,790 fixed-holding families.
- Both refit comparisons (this one and the Fold-Clock result under What failed) use model recipes selected from the full Development period. They compare training schedules, but do not establish a fully causal model-selection process.
- Simple portfolio rules held up. Weighting stocks by score did not beat equal weights, and swapping out the weakest holding for a new pick did not beat keeping current holdings.
- The data and holdout controls work. The surface-validation run's holdout audit counts 0 records after 2026-07-25, and a requested 2016-2017 replay refused to run because the required history did not exist yet. These checks cover data and outcomes, not how the model recipes were chosen.

## What failed, and why

- No model-picking layer earned the right to choose. In the Development Run all four gates (1, 1B, 2, 3) failed. Single families looked strong in single periods, but those gains did not hold up under the gate checks.
- Past winners did not keep winning. The ten best historical models were frozen and replayed forward from 2024-01-31 to 2026-07-24: each made 0 to 3 trades, and none beat the benchmark. In a separate study from 2023-08-11 to 2026-07-24 the same ten models were retrained daily; together they made 3 trades, and 0 of 10 beat the benchmark.
- Re-tuning only the buy threshold every month hurt. In one setting it ended at EUR 10,385 instead of EUR 13,462, and the largest fall against the benchmark deepened from -28.8% to -59.3%.
- Fold-Clock retraining looked excellent in that one setting (EUR 15,773). Across the 2,790 fixed-holding families the median gain shrank to +1.2 percentage points a year, it was better in only 2 of 6 years, and in the median setting group the largest fall against the benchmark got 22 points deeper. The gate failed.
- Signals for choosing between models were too weak: regime persistence did not survive costs, market-state prediction was not robust, and agreement between horizons was suggestive only. The plain model score remained the best selector.

Next step: a model store that keeps every past model version and lets a selector choose only among versions that existed at that point in time. Plan: [research/CAUSAL_MODEL_STORE_ORCHESTRATOR_PLAN.md](research/CAUSAL_MODEL_STORE_ORCHESTRATOR_PLAN.md). Full list of runs: [research/DYNAMIC_QBD_RESULTS_INDEX.md](research/DYNAMIC_QBD_RESULTS_INDEX.md).

## For coding agents

Coding agents start with [AGENTS.md](AGENTS.md). The research map below is the original technical README.

---

## Research map

> Public research snapshot of the Dynamic-QBD / Opportunity-Portfolio program. This repository starts from source snapshot `c196dba496203e803917fd4fff9a03d4df9dc3a2` and contains only the research documents, code and compact results listed in [PUBLIC_SCOPE.md](PUBLIC_SCOPE.md). Everything else, including operational configuration, credentials, third-party and business data, local machine state and large licensed/local datasets, is out of scope.

> **Scope.** The documented research scope is mirrored here: research documents,
> architecture docs, the Dynamic-QBD implementation and self-tests, and compact run
> evidence. Large artifact stores, private runtime incident history and anything
> outside the research program are excluded by contract.
> [PUBLIC_SCOPE.md](PUBLIC_SCOPE.md) states what is included, what is excluded and why.

`simple-and-light` is a research repository for chronological stock-selection, portfolio and execution experiments. It contains several generations of work. The protected V4/N25, V4.5, V5 and broker contracts remain in the repository, but the active research program on the current QBD branch is the **Opportunity-Portfolio / Dynamic-QBD system**.

This README is the human-facing map. Coding agents should start with [AGENTS.md](AGENTS.md).

## Current research focus

Naming contract: [research/DYNAMIC_QBD_NAMING.md](research/DYNAMIC_QBD_NAMING.md). **Full Run** is reserved for the entire 2016-2026 dataset. No Full Run has been executed under the current closed-holdout contract.


The public mirror is published from `main`. Development branch names from the private source repository are provenance only and are not public-mirror branches. Resolve the current private Development head in the private repository before doing source-development work.

The current Dynamic-QBD program asks how a large family of causal stock-selection models should be trained, refreshed, stored and selected through time without future leakage or repeated hindsight optimization.

The research surface is materially larger than the older Top-10 and H5/H10/H20 experiments:

- prediction horizons `H1-H30`;
- holding periods `D1-D_H`;
- capacity `N1-N6`;
- FIXED and, where valid, LEARNED_EXIT variants;
- 2,790 FIXED families and 5,400 families in the full FIXED + LEARNED_EXIT validation;
- 55 FIXED structural H/D plateaus, 110 when separated by exit mode;
- Development evidence through 2025-12-31;
- the completed **Development Run 2016-2025** contained **1,632,350,262 matured prediction rows** and **179,376 compact evidence rows**.

The final prospective holdout starts at **2026-07-25** and remains closed. Dynamic-QBD has no capital, promotion or live-order authority.

### Execution minimum baseline

The Dynamic-QBD execution suite uses `v40.0.4.3` as the **minimum acceptable
baseline**, not as a permanently frozen final architecture. More efficient
runtime designs may be tested.

The historical value of the predecessor sequence is the concrete failure
record: each stage is documented as **failed because X**, together with the
minimum property required to prevent the same defect from returning.

The authoritative failure ledger and minimum contract are
[research/DYNAMIC_QBD_EXECUTION_MINIMUM_BASELINE.md](research/DYNAMIC_QBD_EXECUTION_MINIMUM_BASELINE.md).
This baseline decision is separate from the still-pending local runtime
validation and from scientific/economic validation.

## Data boundary

The repository does not have a clean pre-2016 historical feature/training archive for this program.

- Raw historical market coverage used by the current research starts around `2016-01-01`.
- Benchmark daily data starts `2016-01-04`.
- The canonical signal panel begins `2016-06-24` after feature warm-up.
- Current Development ends `2025-12-31`.
- Prospective holdout boundary: `2026-07-25`.

This matters for the next architecture. A production system can eventually begin with roughly ten years of 2016-2025 history, but a historical pseudo-live replay cannot honestly pretend that the same ten-year seed existed before 2016.

See [research/DYNAMIC_QBD_DATA_AND_RUNTIME.md](research/DYNAMIC_QBD_DATA_AND_RUNTIME.md).

## What has already been learned

The important conclusions are not stored in one experiment. Read the chronological [Dynamic-QBD Results Index](research/DYNAMIC_QBD_RESULTS_INDEX.md) before proposing another selector, controller or refit rule.

High-level findings:

- The original Prediction × Hold surface contains broad robust regions; one isolated best cell is not the research conclusion.
- Equal-active allocation survived the allocation QbD; score/rank-weighted allocation did not add robust value.
- `IGNORE_NEW` survived the replacement test; `REPLACE_WEAKEST` did not.
- Frozen historical Top-10 winners generalized poorly in the later true-forward replay.
- Historical threshold calibration became too restrictive for several frozen models.
- Several memory, cohort, threshold, regime, changepoint and adaptive-controller ideas were tested; none has received selector authority.
- Full Dynamic-QBD Development completed over 2,790 FIXED families. Gates 1, 1B, 2 and 3 all failed conservatively.
- Regime persistence exists, but was too weak after costs to support routing.
- Opportunity-state predictability was not robust.
- Cross-horizon consensus was suggestive but not robust.
- A paired selector tournament ended with `Score-only` as the strongest robust selector; no tested learned pool cleared the promotion gates.
- Monthly refitting/recalibration is broadly destructive relative to sparse evidence-clocked refitting.
- A single H3 Fold-Clock result looked excellent, but the full 5,400-family validation did **not** justify automatic Fold-Clock promotion: F-vs-A failed temporal and drawdown robustness while monthly-vs-Fold-Clock was strongly negative.
- The Fold-Event Counterfactual run is useful mechanistic evidence, but it conditions on recipes selected from the same broad Development era. It must **not** be treated as proof that an ex-ante live orchestrator would have selected those recipes.

The current scientific correction is therefore important:

> Existing Fold-Clock Surface Validation 2016-2025 and Fold-Event Counterfactual results are conditional/mechanistic diagnostics given a Development-selected recipe. The next valid architecture must build a model store and make recipe/generation decisions only from information available at that historical time.

## Intended next architecture

The research direction is a causal **Model Store + Evidence Store + Orchestrator**:

```text
historical prefix available at t0
        ↓
initial model bank
        ↓
pseudo-live boundary
        ↓
incoming observations and matured outcomes
        ↓
new immutable generations / expanding-window refits
        ↓
model + evidence store
        ↓
causal orchestrator
        ↓
active recipe / generation
        ↓
portfolio
```

Old generations remain available. A newly refitted model is a challenger, not an automatic replacement. The orchestrator may keep an older generation or choose another causally available recipe.

The exact historical seed cutoff is still open because no pre-2016 data exists. Do not silently pick the seed after looking at performance.

Detailed plan: [research/CAUSAL_MODEL_STORE_ORCHESTRATOR_PLAN.md](research/CAUSAL_MODEL_STORE_ORCHESTRATOR_PLAN.md).

## Repository map

The most relevant areas for the current research are:

- `stock_predictor/backtests/opportunity_portfolio_research/`: QbD surfaces, Dynamic-QBD factory/replay/evidence/gates and research experiments.
- `research/`: research architecture, scientific interpretation and current handoffs.
- `docs/architecture/`: implementation/runtime architecture documents.
- `artifacts/`: compact committed results and audit evidence. Large model/prediction/NAV/checkpoint stores usually remain local.
- `stock_predictor/research_model_registry.json`: protected V4/N25/V4.5/V5 model identity. It is **not** the Dynamic-QBD project-status tracker.
- `TOBECONTINUED.md`: mandatory current work tracker.

For the Opportunity-Portfolio package itself, see [stock_predictor/backtests/opportunity_portfolio_research/README.md](stock_predictor/backtests/opportunity_portfolio_research/README.md).

## Documentation entry points

Use these in order for current Dynamic-QBD work:

1. [TOBECONTINUED.md](TOBECONTINUED.md): what is done, open and planned.
2. [research/DYNAMIC_QBD_CURRENT_STATE.md](research/DYNAMIC_QBD_CURRENT_STATE.md): current scientific/technical state.
3. [research/DYNAMIC_QBD_RESULTS_INDEX.md](research/DYNAMIC_QBD_RESULTS_INDEX.md): experiments already run and their consequences.
4. [research/DYNAMIC_QBD_DATA_AND_RUNTIME.md](research/DYNAMIC_QBD_DATA_AND_RUNTIME.md): data limits, scale and current machine/runtime contract.
5. [QA_TOOL_FAILURE_LOG.md](QA_TOOL_FAILURE_LOG.md): durable operational QA incident ledger (historical filename) for tool failures and material runtime/resume/provenance failures, including reusable-state decisions.
6. [research/DYNAMIC_QBD_EXECUTION_MINIMUM_BASELINE.md](research/DYNAMIC_QBD_EXECUTION_MINIMUM_BASELINE.md): v40.0.4.3 minimum execution baseline, `failed because X` history and failure-derived non-regression requirements.
7. [research/CAUSAL_MODEL_STORE_ORCHESTRATOR_PLAN.md](research/CAUSAL_MODEL_STORE_ORCHESTRATOR_PLAN.md): current causal scientific architecture plan.
8. Exact `artifacts/<run>/summary.json` / `REPORT.md`: authoritative evidence for a specific completed run.

[DOCUMENTATION_AUTHORITY.md](DOCUMENTATION_AUTHORITY.md) defines how these relate to the protected model registry and historical documents.

## Safety and research authority

Unless the user explicitly changes the scope and an appropriate contract is created:

- paper/research/shadow only;
- no live orders;
- no broker state mutation;
- no silent model promotion;
- no final-holdout opening for iterative tuning;
- no future information in model, route, calibration or orchestrator decisions;
- negative results remain preserved;
- a successful process exit is not economic validation.

Large research runs are performed locally and publish only compact, auditable result packages to GitHub.
