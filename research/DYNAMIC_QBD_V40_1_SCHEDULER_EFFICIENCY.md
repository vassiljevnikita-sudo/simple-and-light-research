# Dynamic-QBD v40.1 Scheduler Efficiency Contract

Status: source implementation complete; focused source/self-test committed; target-host Step-9 runtime acceptance still required.

This document is execution-only authority for the scheduler-efficiency changes added after the v40.1 recovery-hardening layer. It does not change the scientific DAG, causal visibility, Recipe/candidate selection, portfolio accounting, prospective holdout authority, RAM thresholds, or numerical Ridge/HGB kernels.

The minimum non-regression authority remains `DYNAMIC_QBD_EXECUTION_MINIMUM_BASELINE.md`. The RAM safety identity remains `REAL_LOAD_10MS_DAMPED_RECOVERY_80_86_90_92_95_V40_0_4`; the GPU routing identity remains `HGB_FIRST_PREFETCHED_LEARNED_CPU_GPU_ROUTER_V40_0_4_3`.

## Triggering failure

The resumed Step-9 process could be alive while doing almost no useful compute. Observed CPU load around 3-4% matched a single logical worker on the 32-logical-processor host. The remote v40.1 runtime contained several structural causes:

1. bounded portfolio waves forced `max_physical_batches=1`;
2. CPU/GPU pools were created and torn down again across scheduler quanta;
3. every `ManifestedJobStore` construction repeated schema/index/WAL initialization, and portfolio workers opened the same store twice;
4. one seed bank owned a whole wave while the other banks could not borrow unused physical capacity;
5. mixed-causal readiness repeatedly rebuilt per-kind SQL frontiers;
6. watchdog liveness used full dependency-ready counts where an existence check was sufficient;
7. runtime code had regressed to one GPU staging worker/device although the v40.0.4.3 minimum requires four workers plus two parent-queued Futures/device.

The RAM scheduler was not the primary defect: if the physical frontier exposes one batch, a perfect RAM admission policy can still run only one job.

## Active execution contract

### 1. Portfolio breadth

The earliest dependency-ready cutoff remains the causal boundary. Within that cutoff, independent H×cutoff Replay/Evidence groups may execute concurrently. A bounded scheduler quantum may admit up to the seed's currently assigned CPU-lane capacity instead of one physical batch.

Logical scientific identity is unchanged: SQLite still contains one Replay/Evidence row per logical family/cutoff. A physical H×cutoff batch is only an execution unit.

### 2. Persistent shared CPU/GPU pools

Step-9 enables `DQBD_SCHEDULER_EFFICIENCY_V41=1` and owns one persistent physical CPU pool for the runtime root:

- CPU lane target: **32**;
- native numerical threads per worker: 1;
- the RAM scheduler still determines how many of the 32 lanes may execute concurrently;
- SHORT/PRIMARY/LONG calls receive logical lane budgets whose sum never exceeds 32;
- pool lifetime spans scheduler waves and seed quanta;
- pool shutdown occurs only when the shared `RamAdmissionScheduler` closes or at process exit.

The shared pool uses store-scoped lane identities so same job IDs from different seed stores cannot collide in reclaim/watchdog handling.

### 3. Lightweight SQLite worker open

`ManifestedJobStore` has two execution modes:

- parent/bootstrap open: schema/index/dependency migration is performed once per SQLite path;
- worker open: requires an existing `jobs.sqlite3` and performs no DDL, index rebuild, WAL-mode change, or dependency migration.

Ordinary hot connections use `busy_timeout=60000` and `synchronous=NORMAL`; WAL is established during bootstrap rather than renegotiated on every worker connection.

Portfolio batch workers construct one store and pass that exact instance to their handlers. They no longer construct the same store twice.

### 4. Cross-seed physical-capacity sharing

All dependency-ready seed banks may participate in one scheduler wave because they submit into the same physical pools rather than each constructing a 32-lane pool.

Lane allocation is based on **physical schedulable units**, not raw logical row counts:

- ordinary causal nodes count as one physical unit;
- Replay/Evidence at the earliest ready cutoff count by distinct Horizon group;
- Candidate-OOS is excluded from this seed-local capacity count because the global Candidate producer runs before seed-local causal waves;
- total allocated CPU lanes across seed banks is bounded by 32.

There is one deliberate execution floor: an active seed receives at least **two logical scheduler lanes**. `workers==1` selects the legacy serial path and disables the process/GPU router entirely, so a one-unit seed must still enter the asynchronous scheduler through `workers=2`. This may reserve at most one otherwise-unused logical lane for a very narrow seed, instead of parking the historical 10-32 lane share. The second lane is not permission to invent another scientific job and the shared physical pool still never exceeds 32 CPU workers.

This prevents a seed whose frontier contains only one or two physical batches from parking roughly one third of the host while another bank has runnable work.

### 5. Bounded mixed-causal READY frontier

`ManifestedJobStore.ready_jobs()` is backed by a bounded two-second execution cache keyed by store root and requested kind set. The cache is invalidated after state-changing store operations and claimed IDs are discarded immediately. Broad scans are bounded to 4,096-8,192 dependency-ready rows rather than being rebuilt per claim.

Coverage resume uses `pending_cutoff_frontier()` and exposes only the earliest **dependency-ready** coverage cutoff. It must not collapse that cutoff to one row: all ready Horizon coverage rows at that cutoff remain eligible.

### 6. Bounded watchdog liveness

Watchdogs use:

- `ready_job_exists()` -> indexed dependency-ready `SELECT 1 ... LIMIT 1` for Boolean liveness;
- `ready_job_count_bounded(limit=4096)` only when telemetry needs an approximate count.

They do not run a full dependency-ready `COUNT(*)` merely to decide whether a stalled bank has work.

Shared-pool lane watchdogs are store-scoped. A SHORT watchdog cannot reclaim or heartbeat a PRIMARY/LONG lane merely because all seed schedulers reference the same physical pool.

### 7. GPU staging topology restored

Per healthy physical GPU:

- actual GPU worker processes: **4**;
- parent queue-ahead: **2**;
- total staged capacity: **6**;
- actual unsafe kernel concurrency: **1** per physical GPU.

The physical device-section lock remains the kernel authority. Four worker processes do not permit four simultaneous HGB/OpenCL kernels. They permit CPU/materialization portions of one GPU-producing job to overlap with the device section of another so HGB-first/Ridge-spillover queues stay fed.

The GPU pools are persistent and shared by the seed schedulers, so 4+2 is a physical-device capacity, not 4+2 multiplied by the number of seed banks.

Device-specific numerical policy is unchanged:

- Radeon VII HGB: `max_bin=15`;
- RTX 3070 HGB: `max_bin=63`;
- Ridge/HGB backend fingerprints unchanged;
- GPU jobs remain independent of CPU RAM admission leases as defined by v40.0.4.2/4.3.

## Source implementation

The additive implementation is in:

- `dynamic_qbd_scheduler_efficiency.py`;
- package bootstrap `opportunity_portfolio_research/__init__.py`;
- focused regression test `dynamic_qbd_scheduler_efficiency_self_test.py`.

The package install order is deliberate:

1. scheduler-efficiency pre-hardening patches original coordinator functions;
2. v40.1 runtime hardening wraps them with attempt fencing/recovery;
3. runtime-hardening extensions bind runner/import hooks;
4. scheduler-efficiency post-hardening patches shared-pool watchdog, batched runtime and Step-9 runner.

This preserves the existing recovery layer instead of bypassing it.

## Focused source/self-test contract

`dynamic_qbd_scheduler_efficiency_self_test.py` is intentionally light and must not spawn model workers or launch Step-9. It checks:

- 32 CPU-lane contract;
- 4 GPU workers + 2 queue-ahead = 6 staged/device;
- scheduler-efficiency and recovery wrappers are installed in the expected chain;
- persistent shared pool hooks are active;
- worker-side lightweight store-open hook is active;
- earliest-cutoff coverage frontier behavior;
- physical Replay H×cutoff capacity counting;
- READY-cache reuse;
- watchdog EXISTS hooks;
- cross-seed allocations sum to one 32-lane budget;
- portfolio single-store and widened-batch hooks are active.

The test source was syntax-checked before commit. It has not been executed by this agent on the configured Windows target host.

## Required target-host acceptance

Source completion is not runtime proof. The next local resumed Step-9 acceptance must demonstrate all of the following before this scheduler can be called runtime-validated:

- no prospective holdout reads;
- 32 is the physical CPU lane ceiling, not 32 per seed;
- during a broad runnable frontier, more than one CPU job becomes simultaneously RUNNING and sustained CPU duty materially exceeds the prior 3-4% single-lane state;
- when multiple seed banks are ready, unused physical capacity is borrowed without exceeding 32 CPU lanes;
- earliest-cutoff Replay/Evidence exposes multiple independent H batches where available;
- process IDs show CPU/GPU workers persist across scheduler waves rather than respawning every bounded quantum;
- worker-side SQLite opens do not repeatedly perform schema/index/WAL initialization and do not recreate the resume initialization stall;
- SQLite lock errors do not terminate the run;
- mixed-causal READY-query volume is bounded and cache hit telemetry becomes non-zero;
- watchdog liveness does not perform unbounded full-ready counts;
- each physical GPU has four actual staging workers and may reach six staged Futures including queue-ahead;
- actual device-section events still prove at most one unsafe kernel section per physical GPU;
- RTX and Radeon may execute concurrently with one another;
- RAM remains governed by the unchanged 80/86/90/92/95 corridor and single-lane reclaim behavior;
- no pool-wide collapse, orphan workers, duplicate logical completion, stale-attempt publication, or lost COMPLETE checkpoint occurs;
- Radeon `max_bin=15`, NVIDIA `max_bin=63`;
- compatible COMPLETE checkpoints are reused.

No GitHub CI result is part of this acceptance. Heavy Step-9 remains user-local.
