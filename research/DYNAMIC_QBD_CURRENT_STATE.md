# Dynamic-QBD Current State

Status date: 2026-09-05  
Research authority: `SHADOW_ONLY_NO_PROMOTION_NO_HOLDOUT`  
Prospective holdout: starts 2026-07-25 and remains closed.

This is the current handoff for the Dynamic-QBD / Opportunity-Portfolio research program. It summarizes the present scientific state. Exact result artifacts remain authoritative for the numerical result of a specific run.

For open work, always read [../TOBECONTINUED.md](../TOBECONTINUED.md). Current execution/runtime authority is [DYNAMIC_QBD_DATA_AND_RUNTIME.md](DYNAMIC_QBD_DATA_AND_RUNTIME.md); detailed v40.1 scheduler authority is [DYNAMIC_QBD_V40_1_SCHEDULER_EFFICIENCY.md](DYNAMIC_QBD_V40_1_SCHEDULER_EFFICIENCY.md).

## 1. What this program is

Dynamic-QBD is the current research program for a large stock-selection portfolio family. It separates:

1. predictive model/Recipe selection;
2. concrete fitted model Generations;
3. score calibration;
4. portfolio policy dimensions H/D/N/exit;
5. causal matured evidence;
6. optional model/family selection or orchestration.

The long-term goal is not a single permanently frozen winner. It is a causal system that can retain a bank of historical model Generations, create new Generations as data and evidence arrive, and choose an active model without knowing future performance.

## 2. Current design space and scale

Core dimensions:

- H1-H30 prediction horizon;
- D1-D_H holding period;
- N1-N6 maximum names;
- FIXED and LEARNED_EXIT where the exit contract is valid.

Counts from completed research:

- FIXED families: 2,790;
- FIXED + LEARNED_EXIT surface-validation families: 5,400;
- FIXED structural H/D plateaus: 55;
- plateaus separated by exit mode: 110;
- surface-validation arm results: 16,200;
- surface-validation family contrasts: 10,800;
- Fold-Clock entry fits: 210;
- monthly fresh-fit assessments: 1,936;
- Development Run 2016-2025 ABC assessment months: 119.

The completed Development Run 2016-2025 produced:

- **1,632,350,262 matured prediction rows**;
- **179,376 compact evidence rows**;
- peak observed RSS around 5.64 GB after streaming/resume repairs.

The size is not incidental. Aggregation and replay code must remain streaming/cache-aware.

## 3. Data boundary

The available historical program data does not extend far enough backward to simulate the intended production seed exactly.

Current observed boundaries:

- historical market data: approximately 2016-01-01 onward;
- benchmark daily data: 2016-01-04 onward;
- canonical signal panel: 2016-06-24 onward after feature warm-up;
- current Development end: 2025-12-31;
- prospective holdout boundary: 2026-07-25.

A real system starting after 2025 can have about ten years of 2016-2025 history in its initial Model Store. A historical pseudo-live replay starting earlier cannot. The backtest must use a shorter prefix and label that as an approximation rather than inventing pre-2016 evidence.

## 4. What the early QbD phases established

The QbD program began by mapping portfolio-policy dimensions before Dynamic-QBD orchestration.

### Prediction × Holding

The H1-H30, D<=H surface completed all 465 H/D cells. It identified multiple robust/stable regions rather than one unique magic cell. The largest connected plateau covered H23-H26 and D3-D8. The isolated H6/D6 point remained diagnostic.

### Allocation

With the Phase-1 surface frozen, ten allocation treatments were compared. `EQUAL_ACTIVE` remained the robust choice. Rank-power, score-excess-power and softmax weighting did not produce a robust paired improvement.

### Replacement

With allocation frozen, `IGNORE_NEW` remained the robust replacement policy. `REPLACE_WEAKEST` did not survive paired robustness.

### Learned Exit

A 2,790-point Learned-Exit surface produced strong Development islands, especially H11/D3/N1 and H28/D21 regions. These results motivated retaining learned exits as a research dimension but never granted promotion authority.

These phases are completed research constraints, not an invitation to re-open all portfolio dimensions during each later model-selection experiment.

## 5. Frozen Top-10 evidence and why it mattered

A frozen true-forward replay from 2024-01-31 through 2026-07-24 showed that the historically selected Top-10 models did not simply continue their historical dominance. Most produced little activity or negative benchmark excess.

Separate diagnostics showed that several frozen absolute score thresholds became much too restrictive even while the highest-ranked candidates continued to contain positive realized alpha.

This was a key motivation for investigating:

- recalibration;
- model refitting;
- model-generation memory;
- dynamic selection.

It also established a recurring warning: sparse trading and threshold drift make trade-count-only evaluation misleading.

## 6. Selector/controller research already exhausted

Several causal or pseudo-causal controller families were tested.

The central conclusions are:

- threshold adaptation: mixed across models, no general promotion;
- monthly memory weights: mixed;
- cohort controller V2: often materially worse than simple uniform history, with a few model-specific exceptions;
- model-specific memory V3: Development reuse, not independent OOS;
- adaptation-gate V4: improved some models but harmed others; Development reuse;
- regime persistence: measurable persistence, but too weak after costs;
- opportunity-state features: no robust incremental predictability;
- true cross-horizon consensus: suggestive but not robust;
- paired selector model-pool tournament: no learned pool cleared robust gates; `Score-only` remained the strongest robust selector.

Therefore another generic selector, memory controller, regime switch or consensus model is **not** the default next step.

The problem is now experimental design: can a real causal historical Model Store create and choose Generations using only evidence that had arrived by that time?

## 7. Full Dynamic-QBD Development result

The Development Run 2016-2025 covered 2,790 FIXED families and 119 monthly assessment dates.

Important full-run properties:

- family/model fit deduplication;
- generation-specific scores and calibration;
- stateful next-open replay;
- matured evidence only;
- streaming/resumable aggregation;
- final holdout closed.

Gate results:

- Gate 1: FAIL;
- Gate 1B: FAIL;
- Gate 2: FAIL;
- Gate 3: FAIL;
- router: `SHADOW_RESEARCH_ONLY`;
- scientific selection authority: false;
- capital authority: false.

This means the existing performance/health/regime gate stack did not establish a deployable family selector.

## 8. Refit/recalibration decomposition

### H3 matched A/B/C

For H3/D2/N1:

- A = frozen model + frozen calibration: terminal EUR 13,461.75;
- B = frozen model + monthly recalibration: EUR 10,384.74;
- C = monthly fresh refit of same Recipe + monthly calibration: EUR 10,634.57.

The important interpretation is:

- monthly recalibration materially damaged the frozen model;
- C-B was positive only relative to the already damaged B arm;
- C remained far below A.

This rejected the idea that monthly calendar recalibration/refit should be the default.

### H3 Fold Clock

A sparse information-clocked arm was then tested:

- A frozen: EUR 13,461.75;
- F Fold-Clock: EUR 15,773.36;
- M monthly fresh refit: EUR 10,634.57.

At this one cell, Fold-Clock strongly improved both return and relative drawdown versus A, while monthly refit was much worse than F.

This justified surface-wide validation. It did **not** by itself justify a production rule.

## 9. Fold-Clock Surface Validation 2016-2025

The full validation expanded to all 5,400 FIXED + LEARNED_EXIT families.

Primary results:

### F minus A

Overall and FIXED medians were economically positive, but the predeclared robustness gate failed.

ALL:

- median family CAGR-excess delta: +0.5888 percentage points;
- median plateau delta: +1.1104 pp;
- 58.18% positive plateaus;
- only 33.33% positive years;
- median plateau relative-MaxDD delta: -23.23 pp;
- FAIL.

FIXED:

- median family delta: +1.1580 pp;
- median plateau delta: +1.9728 pp;
- 63.64% positive plateaus;
- only 33.33% positive years;
- median relative-MaxDD delta: -22.18 pp;
- FAIL.

LEARNED_EXIT:

- median family delta: -0.1333 pp;
- plateau delta: +0.8203 pp;
- 52.73% positive plateaus;
- bootstrap lower bound negative;
- median relative-MaxDD delta: -24.24 pp;
- FAIL.

### Monthly minus Fold Clock

This was the robust result.

M-F passed the negative-direction gate for ALL, FIXED and LEARNED_EXIT:

- ALL median family delta: -3.9122 pp;
- FIXED: -4.8383 pp;
- LEARNED_EXIT: -3.3350 pp;
- 81.8%-89.1% of structural plateaus were negative depending on stratum;
- 66.7% of years were negative.

Conclusion that survives this run:

> Calendar-monthly refitting is broadly worse than sparse Fold-Clock refitting under the fixed-Recipe contract.

Conclusion that does **not** survive:

> Fold-Clock should automatically replace the incumbent at every new fold.

## 10. Fold-Event Counterfactual

The follow-up branched OLD versus NEW at each non-initial Fold event across all 2,790 FIXED families.

Run:

- 30/30 horizons;
- 180 events;
- 16,740 Event-Family rows;
- contract audit PASS;
- holdout closed.

Aggregate:

- median event effect: 0;
- only 25.56% of events had positive median NEW-minus-OLD excess;
- 77/180 events had zero median effect;
- among 103 non-zero events, 46 were positive and 57 negative;
- median non-zero event effect was about -1.94%;
- simple ex-ante event features had weak rank association with subsequent benefit.

This established that a new Fold is an **information event**, not an automatic activation command.

## 11. Critical scientific correction: ex-post Recipe conditioning

The current Fold-Clock Surface Validation and Fold-Event results have an important limitation that must govern future work.

The fixed Recipe associated with a horizon was selected using the broader Development evidence that overlaps the time interval later analyzed. Therefore those experiments are conditional on a Recipe that the historical system would not necessarily have known was the eventual Development winner.

They are valid for questions such as:

> If this Recipe were fixed, how does refit frequency affect it?

They are not sufficient for:

> Which Recipe/Generation should a historical live system have selected at this date?

This limitation also weakens post-hoc statements such as “HGB handles refits better than Ridge” when the HGB/Ridge Recipe assignment itself came from the same broad Development period.

The next experiment must remove that hindsight conditioning.

## 12. Current target: causal Model Store + Evidence Store + Orchestrator

The intended architecture is:

1. Choose a historical prefix available at pseudo-live start.
2. Build the initial Model Store only from that prefix.
3. Freeze the allowed Recipe/candidate universe before pseudo-live evaluation.
4. Advance time chronologically.
5. Add only newly matured causal evidence.
6. At predeclared evidence events, create new immutable Generations using an expanding training window.
7. Retain all older Generations.
8. Let a causal Orchestrator choose among only the Models/Generations that exist as of that date.
9. Replay the portfolio.
10. Compare against simple predeclared baselines and an ex-post Oracle diagnostic that has no authority.

The production concept can start after 2025 with a roughly ten-year seed. The historical test cannot. Exact seed design is still open and must be predeclared before reading pseudo-live performance.

See [CAUSAL_MODEL_STORE_ORCHESTRATOR_PLAN.md](CAUSAL_MODEL_STORE_ORCHESTRATOR_PLAN.md).

## 13. What not to do next

For execution/runtime work, do not reintroduce a historical failure merely to make the scheduler simpler. A replacement architecture is allowed, but it must state which failure-derived property it replaces and how the new mechanism still prevents the same defect.

For scientific work, do not respond to the current weakness by:

- adding another generic ML selector;
- retuning Fold-Clock thresholds on the same Development outcomes;
- selecting only the horizons that looked good in the Fold-Clock surface-validation result;
- declaring HGB good/Ridge bad from the ex-post-conditioned Recipe assignments;
- reopening allocation/replacement dimensions during the orchestration test;
- using 2020-2025 performance to decide the historical seed cutoff;
- opening the prospective final holdout.

Those would add another layer of hindsight rather than test the target system.

## 14. Runtime and local-artifact state

`v40.0.4.3` remains the minimum acceptable execution baseline, not the final scheduler architecture. The failure-derived requirements are preserved in [DYNAMIC_QBD_EXECUTION_MINIMUM_BASELINE.md](DYNAMIC_QBD_EXECUTION_MINIMUM_BASELINE.md).

The active source-level Step-9 scheduler is now v40.1 scheduler efficiency:

- **32 shared physical CPU lanes** across SHORT/PRIMARY/LONG;
- one native numerical thread per CPU worker;
- unchanged RAM control corridor 80% fill / 86% target / 90% stop / 92% reclaim / 95% hard ceiling;
- persistent shared CPU/GPU worker pools across scheduler waves;
- physical Replay/Evidence execution by multiple dependency-ready H×cutoff batches rather than `max_physical_batches=1`;
- parent-only once-per-SQLite-path schema/WAL bootstrap and lightweight worker store opens;
- bounded cached mixed-causal READY frontier and earliest-ready Coverage cutoff;
- watchdog Boolean liveness via indexed EXISTS rather than full dependency-ready counts;
- **4 GPU staging workers + 2 parent queue-ahead = 6 staged Futures per physical GPU**;
- one unsafe kernel section per physical GPU remains enforced by the existing device-section lock;
- Radeon VII remains `max_bin=15`; RTX 3070 remains `max_bin=63`.

The prior 26-lane default and “32 only after the 26-lane gate” plan are superseded, not retroactively validated. The user explicitly set 32 lanes as the current target after the later one-live-job / 3-4% CPU scheduler failure. The minimum v40.0.4.3 RAM/GPU/recovery properties still have to survive the current 32-lane acceptance.

v40.1 recovery hardening remains active beneath the efficiency layer: exact attempt fencing, owner-scoped recovery, explicit ONLINE/REPAIRING/OFFLINE lane state, replacement-failure recovery, advisory telemetry and collision-safe atomic publication.

The focused non-heavy scheduler regression is `dynamic_qbd_scheduler_efficiency_self_test.py`. Its source is committed and syntax-checked, but this agent has not executed it on the configured Windows target. Heavy Step-9 remains user-local and no GitHub CI result is claimed.

Detailed operational authority: [DYNAMIC_QBD_DATA_AND_RUNTIME.md](DYNAMIC_QBD_DATA_AND_RUNTIME.md) and [DYNAMIC_QBD_V40_1_SCHEDULER_EFFICIENCY.md](DYNAMIC_QBD_V40_1_SCHEDULER_EFFICIENCY.md).

NWinfo telemetry remains diagnostic-only. Recent published runs recorded zero samples when the executable was not discovered.

## 15. Current authority and next gate

Nothing in the current Dynamic-QBD research can:

- allocate live capital;
- promote a model into the protected registry;
- open the final prospective holdout;
- mutate broker state;
- convert Development evidence into an OOS claim by relabelling it.

The next milestone is not “choose the champion.” It is:

> Complete the causal Step-9 chronology and prove that every model-generation/orchestration decision was made from information available at that time, under the current 32-lane v40.1 execution contract.

The immediate runtime gate is the configured-Windows-host acceptance in `TOBECONTINUED.md`: prove shared <=32 CPU lanes, broad concurrent RUNNING work instead of the prior single-lane state, persistent pools, lightweight SQLite opens, bounded READY/watchdog queries, 4+2 GPU staging with one actual same-device kernel, unchanged RAM/reclaim safety, checkpoint reuse and zero holdout reads. Only after all three causal seed stores are complete may post-materialization economic evaluation open.
