# Dynamic-QBD Execution Minimum Baseline and Failure Ledger

Status date: 2026-09-03  
Scope: Dynamic-QBD / Opportunity-Portfolio execution runtime  
Minimum baseline identity: `v40.0.4.3` / `HGB_FIRST_PREFETCHED_LEARNED_CPU_GPU_ROUTER_V40_0_4_3`

## Decision

`v40.0.4.3` is **not** the final architecture and is **not** frozen against
future efficiency improvements.

It is the **minimum acceptable execution baseline**.

A future scheduler/runtime architecture may be materially simpler, faster or
more efficient, but it must preserve or demonstrably improve every
non-regression property in this document. A new design is not acceptable if it
reintroduces a failure that was already paid for during the v1 -> v40.0.4.3
development sequence.

The engineering question for future changes is therefore not:

> Is the new architecture different from v40.0.4.3?

It is:

> Does the new architecture retain every safety, causality, resume,
> observability and resource-efficiency property that had to be learned by
> v40.0.4.3, while producing a measurable improvement?

## Version-provenance rule

The repository preserves exact failure evidence for many later revisions
(v39/v40/v40.0.x) and for several earlier implementation phases, but it does
**not** preserve a trustworthy one-to-one mapping from every historical minor
label `v1`, `v2`, ... `v38` to one exact defect.

Do not invent that mapping.

For the early suite history this ledger therefore records the chronological
**failure phase** and the exact defect evidenced by code/history. Where an
exact version is known, it is named. Where it is not, the revision field says
so explicitly.

This still covers the failures that occurred across the v1 -> v40.0.4.3
architecture evolution; it simply avoids false precision about old minor
labels.

## Failure ledger

| Revision / phase | Classification | Failed because | Required correction / minimum property |
| --- | --- | --- | --- |
| Early Dynamic-QBD aggregation, before the streamed matured store | **FAILED — monolithic aggregation** | Matured prediction partitions were collected and finally merged with one giant `pd.concat`. At Development scale this required roughly another 12 GB allocation and produced `ArrayMemoryError`, preventing final NAV/summary/report generation. | Billion-row paths must remain streamed, partitioned and resumable. Never require a monolithic final pandas frame. |
| Early manifested DAG / pre-v13 resume path | **FAILED — resume itself could exhaust RAM** | Dependency claiming scanned large pending DAG payloads in a way that could materialize too much state; interrupted/failed work was not cleanly requeued while preserving COMPLETE checkpoints, and compatible code fixes could force a fresh root. | Claim/scan large DAGs incrementally; preserve COMPLETE immutable work; support compatible-code resume; requeue interrupted work without rebuilding the entire computation. |
| Early manifested state semantics / v13 lineage | **FAILED — scheduler state could lie about completion** | Manifested job states are canonical uppercase. A lowercase progress lookup could miss FAILED jobs and allow a seed to be treated as complete enough to publish downstream state; this was explicitly recorded as the reason evaluation could not open safely. | Scheduler state names are canonical and fail closed. A seed with FAILED work cannot be published as complete. |
| Early explicit-checkpoint resume | **FAILED — missing checkpoint input could silently trigger a fresh DAG** | If an explicitly requested seed checkpoint directory was missing/misspelled, the old fallback could rebuild a large new output-root DAG instead of failing the resume request. | Explicit checkpoint inputs fail closed when incomplete. Never silently replace a requested resume with fresh recomputation. |
| Early parallel Candidate/Causal execution | **FAILED — concurrency was bounded by process count, not memory cost** | Candidate fits and causal replay could fill the configured process pool even when the active workload was too memory-heavy. A nominal worker count was treated as safe concurrency. | Heavy frontiers require an independent `max_inflight` / admission contract based on actual resource cost, not only the number of logical lanes. |
| Early process scheduler before global resource admission | **FAILED — jobs could be claimed/submitted before resource authority existed** | Work ownership and process submission were not consistently preceded by a global memory reservation. Queue-ahead could therefore create executable work outside the intended resource envelope. | Resource authority must exist before an expensive job becomes executable. Claim/reservation/submission ordering must fail closed and release cleanly on submit failure. |
| Early RAM telemetry/admission implementation | **FAILED — incomplete or unreliable host memory truth** | Initial admission relied on process-tree RSS and optional probes. Later fixes were required for native Windows system-memory telemetry, fail-closed probe behavior and reserving only actually executable lanes. | Admission must use reliable host-level memory truth; telemetry failure must not silently authorize work; reservations must represent executable work rather than abstract queue slots. |
| Early global RAM reservation | **FAILED — static reservation ignored transient and unrealized growth** | Fixed per-task estimates did not cover interpreter imports, allocator arenas, Arrow/sklearn conversion spikes and already-submitted jobs whose RSS had not yet appeared. Nominal reservations could still cross the 95% host limit. | Include safety headroom and unrealized reservations in projected peak. Publication of reservations across seeds must be atomic. |
| Early RAM pressure response | **FAILED — process suspension was treated as memory reclaim** | Suspending Windows workers stopped CPU execution but often kept their working sets resident. Multiple suspended workers could accumulate above the hard limit and deadlock the run. | CPU suspension is not RAM eviction. Pressure control must verify actual relief; ineffective suspension must be abandoned rather than accumulating paused workers. |
| Early worker lifetime / Candidate fit memory | **FAILED — long-lived workers accumulated heap/allocator state** | Candidate fitting retained fold/cache/interpreter state long enough that working sets grew beyond simple job estimates. Conversion spikes could occur on top of resident worker memory. | Large prepared fold state must leave the worker heap when no longer needed; heavy worker lifetime/cache growth must be bounded; scheduler estimates must include real observed worker memory. |
| Early multi-seed execution | **FAILED — identical physical work was repeated per seed** | Candidate fits and common numerical subfits were recomputed independently although their scientific/data identity was identical across SHORT/PRIMARY/LONG. | Share physical compute only by immutable identity: global Candidate-OOS producer, generation/subfit/fold caches and per-key locks, while causal visibility remains seed-local. |
| Early GPU implementation | **FAILED — GPU identity and fallback were too weakly bound** | GPU numerical work evolved from Ridge-only acceleration to HGB/Ridge with device-specific kernels. Without explicit backend/device/kernel identity, artifacts could be reused across numerically different GPU policies or a host/data error could incorrectly disable a GPU. | GPU backend + physical device + kernel policy belong to immutable execution identity. Disable/requeue a GPU only on actual backend/device/kernel failure. |
| Early GPU locking | **FAILED — lock scope was wider than the actual kernel section** | Holding device ownership across CPU preparation/materialization serialized work that did not use the device, leaving GPU gaps and preventing another ready workload from entering. | Physical GPU authority covers only the unsafe device section. CPU prep/materialization stays outside the device lock where numerically safe. |
| Early cross-seed GPU scheduling | **FAILED — seeds could independently target the same nominally free GPU** | SHORT/PRIMARY/LONG had separate scheduler views, so multiple schedulers could prefer one device while the other card remained underused. | Physical GPU occupancy/backlog is shared execution state. Device routing must account for cross-seed contention before assigning new work. |
| Adaptive V6 / v39 | **FAILED — learned incremental RAM + unleased queue-ahead over-admitted the host** | Learned profiles were initially empty/optimistic and ProcessPool queue-ahead Futures could become executable without a matching RAM lease. The diagnostic admitted 5 HGB + 5 Ridge jobs, reached about 96% system RAM and ended in `ArrayMemoryError`. | Lease before submit; include cold-start safety; bound queue-ahead by resource authority; projected host peak must include unrealized submitted work. |
| v40 safety predecessor | **FAILED — safe enough to prevent the v39 pattern, but too estimate-driven to be an effective controller** | Full-RSS leases and projected-peak guards improved safety, but the next runtime revision explicitly had to replace conservative profile/slope-driven admission with live 10-ms real-load control and later relax over-conservative slope vetoes. GPU execution also still lived inside the same broad resource-control topology. | Real host load must close the control loop at runtime. Historical profiles are safety/forecast inputs, not a substitute for current measured load. |
| v40.0.1 shared-compute locking | **FAILED — concurrent lock-file initialization raced on Windows** | Concurrent workers initializing the same shared logistic-head lock file could raise `PermissionError: [Errno 13]` during lock creation/flush and terminate the run. | Shared lock/token creation must be atomic before concurrent handles are opened; per-key publication/locking must tolerate simultaneous Windows processes. |
| v40.0.1 | **FAILED — whole-job GPU executor ownership caused starvation gaps** | The 10-ms real-load controller improved RAM utilization, but one HGB Future could occupy a GPU executor through CPU/materialization intervals. Prepared Ridge therefore could not enter released device gaps. Diagnostics showed only ~31.4% RTX / ~32.2% Radeon device-section duty with repeated 20-125 s gaps. | GPU scheduling granularity must align with actual device-section availability; prepared filler work must be able to enter gaps without unsafe same-device kernel concurrency. |
| v40.0.1 cross-seed GPU subphase | **FAILED — local GPU leases were not a global physical-device contract** | Independent seed schedulers could spend the same nominal GPU availability simultaneously. | GPU lease/backlog state must be global to the physical cards, not seed-local. |
| v40.0.2 | **FAILED — one workload class was mis-sized by orders of magnitude** | Prepared Ridge reached both GPUs, but `candidate_evidence_coverage` was treated as generic lightweight Evidence. Thirty-two coverage jobs were admitted with only ~6.4 GiB reserved while real RAM rose to ~88.6 GiB / ~92.5%, ending in PyArrow `MemoryError`. GPU duty still remained only ~28.4% RTX / ~34.9% Radeon. | Memory profiles/classes must be content-aware. Coverage is its own heavy class, snapshot reads must be projected/filtered, and global concurrency must be capped independently. |
| v40.0.3 RAM reclaim | **FAILED — reclaim granularity was the whole process pool** | A single pressure event escaped through a multi-worker ProcessPool cleanup and terminated unrelated children. RAM repeatedly collapsed from the intended upper corridor to ~20-40%, producing a 20-90% sawtooth rather than stable utilization. | Reclaim one exact logical lane/job, not a pool. Unrelated Futures must survive. |
| v40.0.3 victim selection | **FAILED — max-RSS victim choice punished mature work** | Suspension/reclaim fallback selected the largest resident process instead of the job whose incremental working set best matched required relief. Mature high-memory jobs were repeatedly attractive victims. | Victim selection must be progress-aware and best-fit to required relief, with protection for mature/already-reclaimed work. |
| v40.0.3 reclaim control loop | **FAILED — stale pre-reclaim slope could trigger repeated kills** | The positive RAM slope measured before a successful reclaim remained in the controller epoch after memory had already fallen, so the same old growth episode could trigger another victim. | Reset the slope epoch after successful relief; enforce settling/hysteresis and damped refill before another normal reclaim. |
| v40.0.3 scratch lifecycle | **FAILED — abrupt pool death leaked external-sort scratch** | Development-slice hashes were recomputed in temporary directories; abrupt pool termination could bypass cleanup. The run later ended with `ENOSPC`. | Shared persistent slice-hash values, PID-addressable scratch, crash-safe locking and startup/reclaim orphan cleanup are mandatory. |
| v40.0.4 | **FAILED — RAM architecture was corrected, but GPU progress was still coupled to CPU/RAM control** | Single-lane reclaim, hysteresis, best-fit victims and damped recovery solved the major RAM failures. The suite still allowed GPU-capable Candidate/Causal work to depend on `RamAdmissionScheduler`, so closing CPU admission could starve otherwise free GPUs; CPU-reclaimed GPU-capable jobs could return to CPU again. | CPU/RAM admission and GPU execution are separate control domains. GPU pools are not RAM-reclaim clients; GPU dispatch must continue while CPU admission is closed; reclaimed prepared GPU work prefers GPU. |
| v40.0.4.2 | **FAILED — GPU router was decoupled but underfed** | Learned RTX/Radeon/CPU service times correctly showed GPUs were much faster (about 61 s RTX / 81 s Radeon / 360 s CPU HGB P75), but Step-9 still forced `queue_ahead=0`; GPU eligibility required already-prepared folds; only four staged Futures/device existed. GPUs could drain between prepared batches. | Produce GPU-capable work ahead of demand: separate H×Fold prep, maintain prep lookahead, and preserve parent queue-ahead without increasing unsafe kernel concurrency. |
| v40.0.4.2 observability | **FAILED — queue assignment was mistaken for execution evidence** | Both device queues could report HGB acquires while run-attributed NVIDIA compute was not proven. Scheduler events described assignment, not the physical device section. | Physical `ACQUIRE/RELEASE` events must carry `job_id`, worker slot and device identity. Queue assignment alone is never proof of GPU execution. |
| v40.0.4.3 initial feed implementation | **FAILED SUBCASES FIXED WITHIN THE SAME BASELINE IDENTITY** | Final audit still found cross-seed device prediction based too heavily on local Futures, a 50-ms backlog cache keyed only by root, stale JSON payload state after prep requeue, and learned speed capable of stacking RTX Futures while Radeon was empty. | Device prediction uses shared physical backlog; backlog cache is keyed by exact device set; prep requeue keeps SQLite and JSON payload synchronized; physical breadth wins before learned-finish tie breaking. |
| v40.0.4.3 fast-resume derived-state phase | **FAILED — fast compatible resume skipped DAG reconciliation and a corrected snapshot collided with an older immutable leaf** | The v40.0.4.3 diagnostic repaired stale Coverage dependencies at runtime (for example expected 25 Candidate×Fold keys while 30 were already causally matured), then attempted to persist the corrected Evidence Snapshot under an H×cutoff leaf that already contained the older valid snapshot. The immutable guard raised `EVIDENCE_SNAPSHOT_IMMUTABLE_CONFLICT` and aborted the whole run even though RAM/GPU execution was healthy. | A cache miss / explicit slow resume must reconcile current graph payloads and requeue only changed nodes plus transitive derived descendants. Valid older snapshot leaves remain immutable and are preserved; corrected snapshots publish under a deterministic manifest-hash version. A validated slow reconcile writes the v2 initialization cache so later fast resumes skip already-complete jobs. Partial/corrupt snapshot artifacts still fail closed. |
| v40.0.4.3 current | **MINIMUM_ACCEPTABLE_BASELINE** | The known failure classes above are addressed in the current baseline. This does **not** prove no more efficient architecture exists and does not by itself validate the local Step-9 run. | Future designs may improve architecture, but may not regress any property in the minimum requirements below. |

## Minimum requirements inherited from the failure history

### Large-data execution

1. Matured prediction/evidence aggregation is streaming, bounded and resumable.
2. No billion-row path depends on one final in-memory pandas concat.
3. Large DAG scans/claims do not materialize the entire pending payload set.
4. COMPLETE immutable artifacts survive compatible runtime changes and resume.

### RAM control

5. Host-level real memory load is measured continuously; profiles do not replace live truth.
6. Heavy job classes are content-aware and separately profiled/capped.
7. Resource admission exists before an expensive job becomes executable.
8. Outstanding/unrealized submissions count toward projected load.
9. Cross-seed reservation publication is atomic.
10. Normal pressure relief targets one exact CPU lane/job.
11. Reclaim victim selection is progress-aware and best-fit.
12. Post-reclaim slope state is reset; settling and damped recovery prevent sawtooth refill.
13. Ineffective process suspension is never treated as successful memory eviction.
14. GPU pools are not RAM-reclaim victims.

### State, resume and worker/cache lifecycle

15. Manifested scheduler state names are canonical and fail closed; FAILED work cannot be mistaken for completion.
16. Explicit checkpoint/resume inputs never silently fall back to a fresh large DAG.
17. Heavy worker caches and prepared-fold state do not grow without bound.
18. Shared immutable computation is deduplicated by scientific/execution identity.
19. Shared per-key locks/tokens are initialized atomically under concurrent Windows processes.
20. Temporary scratch is PID-addressable, crash-cleanable and kept on the intended workspace volume.
21. Persistent hash/cache state uses crash-safe per-key publication.

### CPU/GPU separation

22. CPU/RAM admission and GPU execution are separate control domains.
23. CPU admission closure does not block healthy ready GPU work.
24. CPU-reclaimed prepared HGB/Ridge work is not allowed to fall into a CPU-reclaim loop.

### GPU feed and safety

25. Candidate-independent H×Fold prep can run ahead of fitting.
26. GPU-capable prepared work is staged ahead of device demand.
27. Parent lookahead is preserved; queue depth is staging only and does not imply concurrent physical kernels.
28. Exactly one unsafe kernel section executes per physical GPU.
29. RTX and Radeon can execute concurrently with each other.
30. HGB remains ahead of Ridge at the device-side priority boundary.
31. Device/kernel-specific parameters remain part of immutable execution identity.
32. Radeon VII keeps its safe HGB `max_bin=15`; RTX 3070 keeps `max_bin=63` unless a separately validated numerical/runtime contract supersedes them.

### GPU routing

33. Physical device choice sees shared cross-seed backlog, not only one scheduler's local Futures.
34. Backlog cache identity includes the exact physical device set.
35. An empty/shallower healthy physical GPU is filled before a deeper staged queue is stacked purely because another card has a faster learned P75.
36. Once staged depth is comparable, learned service time may favor the faster card.
37. Learned service-time profiles exclude parent queue wait.

### Observability and resume truth

38. Queue assignment is not execution evidence.
39. Actual GPU device sections emit job-correlated ACQUIRE/RELEASE evidence.
40. Persisted scheduler payload state and SQLite state remain consistent after requeue/reclaim/prep-only cycles.
41. Scheduler-only changes preserve compatible COMPLETE scientific checkpoints.
42. Seed-local causal visibility remains intact even when physical computation is shared.
43. A compatible-code cache miss, or explicit slow resume, reconciles the current manifested DAG before fast-resume cache authority is granted.
44. A changed job payload invalidates only that job and its transitive derived descendants; unrelated COMPLETE checkpoints remain reusable.
45. A valid immutable Evidence Snapshot conflict never overwrites the old leaf and does not abort the run: preserve the prior leaf and publish the corrected snapshot under deterministic content-bound versioning. Partial or hash-invalid artifacts still fail closed.
46. Fast resume may skip reconciled COMPLETE jobs only when its cache key binds the current Git SHA, input-stat fingerprint and current manifested-job-contract hash.

## How future architecture changes are judged

A future redesign is allowed.

It should be preferred when it is measurably simpler or more efficient **and**
retains the minimum requirements above.

Before replacing the baseline, the change must state:

- which current mechanism it replaces;
- which historical failure originally required that mechanism;
- how the replacement prevents the same failure;
- which focused regression test proves that;
- what measurable efficiency/complexity improvement justifies the change.

The historical failure ledger is therefore a design test suite, not a ban on
new architecture.

## Validation status

The current `v40.0.4.3` architecture remains pending local 26-lane runtime
validation. The minimum-baseline decision does not claim runtime success,
economic success, promotion authority or final-holdout evidence.

Prospective holdout remains closed. Promotion and capital authority remain
disabled.
