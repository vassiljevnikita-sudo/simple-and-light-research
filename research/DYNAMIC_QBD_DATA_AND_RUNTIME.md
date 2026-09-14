# Dynamic-QBD Data, Scale and Runtime

This document is the current operational source of truth for large Opportunity-Portfolio / Dynamic-QBD research runs.

It is not a model-selection document. Scientific state: [DYNAMIC_QBD_CURRENT_STATE.md](DYNAMIC_QBD_CURRENT_STATE.md). Open work: [../TOBECONTINUED.md](../TOBECONTINUED.md). Failure-derived minimum execution properties: [DYNAMIC_QBD_EXECUTION_MINIMUM_BASELINE.md](DYNAMIC_QBD_EXECUTION_MINIMUM_BASELINE.md). Current scheduler-efficiency contract: [DYNAMIC_QBD_V40_1_SCHEDULER_EFFICIENCY.md](DYNAMIC_QBD_V40_1_SCHEDULER_EFFICIENCY.md).

## 1. Historical data and holdout boundary

The current research history begins in 2016. There is no clean pre-2016 feature/training history available for Dynamic-QBD.

| Data | Earliest relevant date | Current Development end |
|---|---:|---:|
| historical market source | approximately 2016-01-01 | 2025-12-31 |
| benchmark daily | 2016-01-04 | extends beyond Development for other diagnostics |
| canonical H1-H30 signal panel | 2016-06-24 | 2025-12-31 |
| prospective final holdout | **2026-07-25** | closed |

The signal panel starts later than raw market data because feature construction needs warm-up history.

A production system initialized after 2025 can legitimately use roughly ten years of 2016-2025 history as its initial Model Store. A historical pseudo-live test cannot pretend that the full decade existed at an earlier cutoff. The historical seed and every later evidence/refit decision must therefore remain chronological and maturity-aware.

Current prospective contract: `PROSPECTIVE_FROM_2026_07_25`.

Hard rules:

- Development research may use data through 2025-12-31 under the frozen Development contract.
- Prospective 2026-07-25+ performance must not select historical seeds, Recipes, gates or scheduler policies.
- Routine debugging never opens the prospective holdout.
- A final holdout read requires an explicit final-evaluation decision.

## 2. Research scale

### FIXED Opportunity-Portfolio surface

For H1-H30, D ranges 1..H and N ranges 1..6:

```text
6 × (1 + 2 + ... + 30) = 2,790 FIXED families
```

The completed FIXED + LEARNED_EXIT surface-validation space contained 5,400 families: 2,790 FIXED + 2,610 LEARNED_EXIT.

N/D are portfolio dimensions. They do not imply distinct predictive fits when H, Recipe, training identity and causal cutoff are identical.

### Development evidence scale

The completed Development Run 2016-2025 recorded:

- 2,790 FIXED families;
- 119 ABC assessment months;
- 55 structural plateaus;
- **1,632,350,262 matured prediction rows**;
- **179,376 compact evidence rows**.

The early implementation attempted a final `pd.concat(rows, ignore_index=True)` over large matured partials and failed with `ArrayMemoryError`. The repaired streamed/resumable path completed with observed peak RSS around 5.64 GB.

Permanent rule: never require all matured predictions, all H/D/N families or all generations to coexist in one in-memory DataFrame. Prefer Parquet projection/filter pushdown, partitioning, bounded caches, immutable artifacts, streamed aggregation and deterministic checkpoints.

## 3. Local versus committed artifacts

Large research state remains local. GitHub contains compact, auditable evidence and code.

Typical local-only state:

- model artifacts;
- full prediction Parquet stores;
- portfolio NAV/trade Parquet;
- large family stores;
- prepared-fold caches;
- per-horizon/seed checkpoints;
- worker/sensor telemetry streams;
- replay fragments.

Typical committed evidence:

- `summary.json`;
- `REPORT.md`;
- contract/gate audits;
- compact contrast CSVs;
- event ledgers;
- manifests/hashes;
- small telemetry summaries.

A compact committed summary does not replace the local scientific artifacts needed for a new experiment.

## 4. Current development host

Current large-run host:

- Windows 11;
- Intel Core i9-14900KF class CPU;
- **32 logical processors**;
- **96 GB physical RAM**;
- RTX 3070 8 GB;
- Radeon VII 16 GB.

Older Ryzen 7 5800X / 32-GB runtime documents are historical evidence only and are not current resource guidance.

## 5. Current Step-9 execution contract

The active v40.1 Step-9 runtime targets **32 shared CPU process lanes**.

This replaces the historical 26-lane default. The earlier plan to validate 26 first and benchmark 32 later was superseded after resumed v40.1 runs exposed a structural one-batch / pool-lifecycle / SQLite-initialization bottleneck. The failure-derived v40.0.4.3 minimum properties remain mandatory; only the physical scheduler topology changed.

Current CPU/RAM contract:

```text
physical CPU lane target = 32 total across SHORT/PRIMARY/LONG
native numerical threads = 1 per worker
RAM fill floor           = 80%
RAM target               = 86%
new-large-work stop      = 90%
normal reclaim           = 92%
Windows hard ceiling     = 95%
RAM sample cadence       = 10 ms
runtime rollup cadence   = 2 s
```

The 32 lane count is a physical ceiling, not guaranteed simultaneous execution. `RamAdmissionScheduler` remains authoritative and can admit fewer lanes when real RAM, slope, learned class profiles or recovery state require it.

Execution identities retained from the minimum baseline:

```text
DQBD_SINGLE_LANE_RECLAIM_CLOSED_LOOP_V40_0_4
REAL_LOAD_10MS_DAMPED_RECOVERY_80_86_90_92_95_V40_0_4
SINGLE_LANE_RECLAIM_FRONTIER_V40_0_4
HGB_FIRST_PREFETCHED_LEARNED_CPU_GPU_ROUTER_V40_0_4_3
DQBD_JOB_MEMORY_PROFILES_V3_REAL_LOAD
DQBD_REAL_LOAD_ROLLUP_V40_0_4
```

The current additive scheduler layer is `DQBD_SCHEDULER_EFFICIENCY_V41`.

## 6. Single-lane RAM reclaim remains unchanged

The v40.0.3 `--workers 32` diagnostic established why pool-wide reclaim is forbidden: RAM repeatedly collapsed from the intended upper corridor to roughly 20-30%, producing a 20-90% sawtooth. The current controller therefore reclaims exactly one independently replaceable lane.

A pressure event remains:

```text
select one progress-aware victim
-> persist spill/audit state where possible
-> terminate exactly that worker lane
-> clean that PID's scratch
-> replace only that lane
-> reclaimed RUNNING node -> PENDING
-> unrelated Futures continue
```

Normal triggers remain:

```text
RAM >= 90% and post-epoch slope >= 0.25 GiB/s
or RAM >= 92%
```

A successful reclaim resets the slope epoch and starts a 750-ms settling interval. Only >=94.5% emergency pressure may break settling. Recovery admission remains damped and does not immediately refill the host above 87%.

Coverage remains a separate heavy class:

```text
causal:coverage:CPU
cold incremental estimate = 2.75 GiB
cold absolute estimate    = 5.00 GiB
global class cap          = 8
```

Development-slice hashes remain shared persistent compute artifacts and PID-addressable scratch must be cleaned after targeted worker death.

## 7. v40.1 scheduler-efficiency topology

The current source implementation is documented in [DYNAMIC_QBD_V40_1_SCHEDULER_EFFICIENCY.md](DYNAMIC_QBD_V40_1_SCHEDULER_EFFICIENCY.md).

### 7.1 Persistent shared CPU pool

Step-9 owns one physical 32-lane CPU pool for the runtime root. SHORT/PRIMARY/LONG seed schedulers submit into this same pool. A bounded scheduler quantum no longer creates and destroys a new Windows `spawn` pool.

Per-seed lane budgets are logical admission caps whose sum is <=32. Physical pool processes persist across scheduler waves until the shared RAM scheduler closes or the process exits.

### 7.2 Cross-seed allocation

All READY seed banks may run in one wave because they no longer each own a full physical pool.

Allocation is based on physical schedulable units:

- ordinary causal node = one unit;
- Replay/Evidence = one unit per distinct Horizon at the earliest ready cutoff;
- Candidate-OOS is excluded from seed-local capacity because one global Candidate producer runs first.

This prevents one narrow seed frontier from parking a fixed one-third share while another seed has runnable work.

### 7.3 Portfolio H×cutoff breadth

Logical Replay/Evidence rows remain one row per family/cutoff for scientific and resume identity. Physical execution groups the earliest dependency-ready cutoff by Horizon.

The previous bounded-wave `max_physical_batches=1` restriction is removed. A wave may execute multiple independent H×cutoff groups up to its available seed CPU-lane budget and RAM admission.

Each logical Replay/Evidence row still commits independently. A lane failure requeues only unfinished rows; already COMPLETE rows remain reusable.

### 7.4 Lightweight SQLite worker open

`ManifestedJobStore` schema/index/dependency/WAL bootstrap is parent-only and once per SQLite path. Worker construction attaches to an existing `jobs.sqlite3` without DDL, index rebuild, dependency migration or `journal_mode` renegotiation.

Hot connections use a 60-second SQLite busy timeout and `synchronous=NORMAL`. Portfolio batch workers construct one store and reuse it for their handler set instead of opening the same store twice.

### 7.5 Bounded READY frontier

Mixed-causal `ready_jobs()` uses a bounded cache keyed by store and requested kind set. Cache entries expire after two seconds, are invalidated by state mutations and discard claimed IDs immediately. Broad scans are bounded to 4,096-8,192 rows instead of being rebuilt for every CPU admission attempt.

Coverage resume exposes all dependency-ready coverage rows at the **earliest ready cutoff**. It does not scan the complete pending Coverage backlog and it does not collapse the earliest cutoff to one row.

### 7.6 Watchdog liveness

Boolean liveness uses an indexed dependency-ready `SELECT 1 ... LIMIT 1`. A bounded count is used only for telemetry. Watchdogs do not evaluate the complete dependency-ready DAG merely to answer “is any runnable work present?”.

Shared-pool lane watchdogs filter by store root so a SHORT watchdog cannot mutate a PRIMARY/LONG lane.

## 8. Current GPU contract

Per healthy physical GPU:

```text
actual GPU staging worker processes = 4
parent queue-ahead                  = 2
total staged capacity              = 6
unsafe physical kernel concurrency = 1
```

The four staging workers are **not** permission for four same-device LightGBM/OpenCL kernels. The shared device-section lock remains the kernel authority. Staging overlap exists so CPU/materialization portions of one GPU-producing job can overlap another job's device section and keep HGB-first/Ridge-spillover work queued.

GPU pools are persistent and shared by the seed schedulers. The 4+2 capacity is per physical device, not multiplied by SHORT/PRIMARY/LONG.

GPU execution remains outside CPU RAM-admission leases as established by v40.0.4.2/4.3. GPU queue order remains HGB-first, then prepared Ridge spillover.

Physical devices may run concurrently with one another: RTX and Radeon can overlap. Same-device kernel sections remain serialized.

Device-specific HGB numerical policy is unchanged:

```text
Radeon VII: max_bin = 15
RTX 3070:   max_bin = 63
```

Numerical identities remain:

```text
DQBD_GPU_RIDGE_HGB_PRETRAINING_V6
OPENCL_FP64_RIDGE_REGRESSOR_SKLEARN_BUNDLE_V2_SHARED_GRAM
OPENCL_LIGHTGBM_GPU_HGB_SKLEARN_BUNDLE_V5_DEVICE_FINGERPRINTED
DUAL_GPU_OPENCL_HGB_AMD_TDR_GUARD_V4
AMD_MAX_BIN_15_NVIDIA_MAX_BIN_63_SINGLE_PRECISION_V4_DEVICE_IDENTITY
```

Scheduler-only migration may preserve compatible COMPLETE artifacts only when those numerical identities remain unchanged.

## 9. Candidate preparation and device telemetry

Candidate-independent H×Fold preparation remains separated from Candidate fitting. `CandidateOosFactory.prepare_fold()` publishes the shared prepared-fold artifact, then returns the claimed Candidate node to PENDING so HGB/Ridge routing can consume the prepared frontier.

Authoritative actual GPU occupancy remains repository-owned device-section evidence:

```text
_shared-compute/gpu-device-events/
  platform-<p>-device-<d>.jsonl
```

Each ACQUIRE/RELEASE carries timestamp, PID, job ID, worker slot, backend and duration. Queue assignment is not accepted as proof of device execution.

Ready-section scheduling and queue telemetry remain under:

```text
_shared-compute/gpu-section-ready-queue/
_shared-compute/gpu-section-queue-events.jsonl
```

A host RAM/data exception does not disable a GPU. Only an actual OpenCL/LightGBM/device/kernel failure justifies device disable/requeue.

## 10. Resume and recovery requirements

Long Dynamic-QBD runs must resume from immutable checkpoints rather than rebuild completed work.

A valid scientific reuse decision binds to relevant signal-panel, candidate/Recipe evidence, registry, causal cutoff, training/calibration, model/backend and research-contract identities.

The v40.0.4.4/v40.1 resume path distinguishes scientific identity from execution-only migration:

- a slow resume reconciles graph semantics and validates surviving COMPLETE results;
- changed scientific nodes requeue only themselves plus transitive derived descendants;
- unrelated COMPLETE checkpoints remain reusable;
- corrupt/partial/hash-invalid artifacts fail closed;
- exact attempt fencing prevents an old Future from publishing into a newer same-owner claim;
- stale-dispatcher/lane-repair incidents remain execution recovery and do not consume the genuine worker-crash budget;
- lane replacement failure produces explicit OFFLINE/repair state rather than an unresolved Future.

Do not clear working checkpoints merely because a process was interrupted or scheduler source changed compatibly.

## 11. Data-fidelity constraints

### Feature warm-up

Do not fabricate signals before the canonical panel begins on 2016-06-24.

### Benchmark total return

URTH is the implementable benchmark/idle sleeve. Do not compare a total-return strategy path against a price-only benchmark accidentally.

### Stock daily boundaries

Execution inputs may require row-level source repair/fallback for invalid daily opens/closes. Actual execution remains fail closed when authoritative inputs are absent.

### Corporate actions and ticker lineage

Symbols change. Do not resolve historical source gaps through future-looking aliasing or unverified forward-fill.

### Point-in-time universe

The frozen Top-10 forward replay explicitly did not prove a complete point-in-time universe. Do not promote that evidence beyond its data contract.

### Learned exits

Do not apply learned-exit predictions to counterfactual positions for which causal exit predictions did not exist. The first causal baseline remains FIXED-only unless an explicit causal provider is defined.

## 12. Model/evidence maturity and fit deduplication

For Horizon H, a prediction becomes evidence only after its terminal date/information-availability boundary. Longer Horizons mature later. Every selector/refit/orchestrator must consume maturity-filtered evidence.

A physical fit is identified by H, Recipe, causal cutoff, feature schema, training/calibration windows, target/training contract, random state and exact numerical backend fingerprint. D/N do not normally require new predictive fits. Historical seed is a visibility dimension, not automatically a physical-fit dimension; exact shared fit identities may reuse one immutable Generation while each seed retains separate visibility/provenance state.

## 13. Heavy-run workflow

For heavy runners:

1. resolve current branch/HEAD and authority docs;
2. define/freeze semantic run contract;
3. implement deterministic checkpoint/resume;
4. bound native numerical threads;
5. partition by H/event/generation;
6. write compact progress/telemetry;
7. avoid monolithic aggregation;
8. preserve local large artifacts;
9. publish compact evidence only after completion;
10. retain code/source-run lineage.

Heavy Step-9 remains user-local unless explicitly requested. The reserved `Full Run` is the separate complete 2016-2026 dataset and remains not run.

## 14. Current runtime gate

Source implementation for v40.1 scheduler efficiency is complete and the focused non-heavy regression test is committed as:

`stock_predictor/backtests/opportunity_portfolio_research/dynamic_qbd_scheduler_efficiency_self_test.py`

That test source was syntax-checked before commit. This agent has **not** executed the target-host selftests or Heavy Step-9 and no GitHub CI result is claimed.

The next configured-Windows-host acceptance must demonstrate:

- holdout reads = 0;
- total shared CPU lane ceiling <=32 across all seed banks;
- broad runnable periods produce multiple simultaneous RUNNING jobs and materially exceed the prior ~3-4% single-worker state;
- unused seed capacity is borrowed without creating per-seed 32-lane pools;
- multiple earliest-cutoff H×cutoff batches execute concurrently when available;
- CPU/GPU worker PIDs persist across scheduler waves;
- worker SQLite opens do not repeat DDL/index/WAL bootstrap and lock contention does not kill the run;
- READY-cache hit/refresh telemetry confirms bounded frontier reuse;
- watchdog Boolean liveness does not execute unbounded full-ready counts;
- four GPU staging workers/device exist and queue depth may reach six including parent lookahead;
- actual device-section evidence still proves one unsafe kernel section/device while RTX/Radeon may overlap;
- RAM remains in the unchanged 80/86/90/92/95 control contract and targeted reclaim remains one lane;
- no pool-wide collapse, orphan workers, stale-attempt publication, unresolved replacement Future, duplicate logical completion or loss of COMPLETE checkpoints;
- Radeon `max_bin=15`, NVIDIA `max_bin=63`;
- compatible checkpoint reuse remains intact.

Only after that acceptance should post-materialization economic evaluation be opened.
