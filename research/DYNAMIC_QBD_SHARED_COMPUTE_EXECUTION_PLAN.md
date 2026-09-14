# Dynamic-QBD Shared Compute Execution

Status: V40.0.4.3 MINIMUM ACCEPTABLE EXECUTION BASELINE; LOCAL 26-LANE VALIDATION PENDING

> Minimum-baseline decision — 2026-09-02: `v40.0.4.3` is not the final
> execution architecture. More efficient designs may replace its mechanisms,
> but they must preserve the concrete requirements learned from the historical
> failure sequence. The authoritative `failed because X` ledger is
> [DYNAMIC_QBD_EXECUTION_MINIMUM_BASELINE.md](DYNAMIC_QBD_EXECUTION_MINIMUM_BASELINE.md).

The Step-9 causal pseudo-live Development evaluation keeps the frozen
Candidate universe, causal DAG, FIXED portfolio contract and evaluation
authority unchanged. This execution layer removes duplicate physical work,
uses CPU/GPU resources more continuously, and preserves seed-local causal
visibility. It does not open the prospective holdout and does not create a new
promotion decision.

## Physical sharing

Candidate-OOS folds are independent of the historical seed boundary when the
following scientific/data identity is equal:

```text
horizon + fold + candidate + signal panel + feature schema
+ fold policy + target contract + training contract
```

One global producer materializes a Candidate-OOS partition and publishes the
verified immutable artifact into each seed-local store. Seed-local DAG state
and each seed's `information_available_at` cutoff remain authoritative.

Production generations use an immutable fit key containing horizon, selected
Recipe, information cutoff, training/calibration/prediction date hashes,
feature and target contracts, random state, model-training contract and exact
execution-backend fingerprint. Equal keys share one physical Generation tree;
causal visibility remains seed-local.

## Candidate-fold preparation

The expensive Candidate-independent H×Fold preparation is materialized once
under a per-key crash-safe lock:

```text
signal panel
  -> H×Fold maturity/filter/label join
  -> adapted train/test rows
  -> immutable prepared-fold artifact
  -> all Candidate consumers
```

The first producer holds the H×Fold preparation lock across the expensive
materialization. The parent scheduler admits only one producer for an
unprepared H×Fold. As soon as the prepared artifact exists, the remaining
Candidates for that fold can execute concurrently. This prevents several
worker slots from waiting on or recomputing the same preparation.

The prepared-fold artifact is an ephemeral execution cache, not durable
scientific evidence. After all Candidate-OOS jobs for that H×Fold are COMPLETE,
the parent removes that fold's prepared cache directory. A resume may
deterministically rebuild it from the authoritative panel if unfinished
Candidates still require it.

## Shared sub-fits

Ridge Candidates for one H×Fold share the expensive FP64 Gram/RHS work.
`X'X`, `X'y`, feature means and target mean are materialized once; each
alpha then performs only its own small CPU solve. Identical
`LogisticRegression` heads are content-addressed by fit identity, classifier
parameters and label hash and are reused across Candidates.

These caches are physical execution caches only. Candidate identities,
predictions and immutable model artifacts remain distinct where their Recipe
parameters differ.

## Job-aware RAM admission and single-lane reclaim

v40.0.2 established the need for a heavy Coverage class. v40.0.3 fixed that
classification and could repeatedly reach the upper RAM corridor, but the
published `--workers 32` selection-fix diagnostic exposed a second controller
defect: reclaim was applied to a whole ProcessPool rather than to one job.

The committed v40.0.3 rollups contain repeated abrupt drops from roughly
85-87 GiB to 19-37 GiB. Across 736 two-second rollups, 30 intervals dropped
more than 15 GiB and mean-RAM p10/median/p90 were about 15.2%/65.6%/83.7%.
That is execution oscillation, not a valid stable-86% result.

The causes were:

1. `RamPressureReclaim` escaped a multi-worker `ProcessPoolExecutor`, and
   managed cleanup terminated every child in that pool;
2. the older suspend fallback ranked victims by maximum RSS, so mature jobs
   near their peak could be selected repeatedly;
3. the 250-ms RAM-slope window still contained samples from before the memory
   drop, so one successful reclaim could immediately be followed by another;
4. abrupt termination could strand external-sort Development-slice-hash
   scratch, and the run ultimately ended with `ENOSPC`.

v40.0.4 replaces the process topology with independently replaceable lanes.
The logical worker count is unchanged, but each lane owns a one-worker
`ProcessPoolExecutor`. Only the lane selected for reclaim is terminated and
recreated. Other lane Futures remain valid and continue executing.

The normal control bands remain:

```text
fill floor = 80%
target     = 86%
stop       = 90%
reclaim    = 92%
hard       = 95%
```

Pressure triggers are:

```text
>=90% and post-reclaim-epoch slope >=0.25 GiB/s
or
>=92% regardless of slope
```

Each decision may reclaim **one lane only**. Required relief is computed toward
the 86% target. Sustainable job relief is based on incremental job RSS because
a newly spawned replacement lane recreates its baseline imported process RSS.

Victim order is progress-protected:

```text
never reclaimed
  before already-reclaimed

<15 s old
  before 15-30 s
  before >=30 s

CPU
  before GPU when otherwise comparable

then best-fit incremental RAM to required relief
```

A reclaimed manifested node returns from RUNNING to PENDING with
`ram_reclaim_count`, last reclaim time/reason and released-RAM evidence in its
execution payload. It is given completion priority on the next admission. This
prevents the previous cycle in which the same large mature job could become the
default victim every time it approached its memory peak.

Reclaim arbitration is global across all active seed-local CPU/GPU pools. The
shared scheduler exposes one non-blocking reclaim gate; all lane controllers
register with it, and one winning decision ranks candidates from every
registered pool before touching a process. SHORT, PRIMARY and LONG therefore
cannot each react to the same >90% edge and remove three workers in parallel.\nNo new RAM reservation may publish while the reclaim gate is held. Admission\nchecks the gate again from inside the atomic reservation lock, so the memory\ndrop created by the terminating victim cannot trigger an early refill before\nthe 750-ms settling / damped-recovery state is installed.\n

A lane is replaced only after its prior child is confirmed dead. If the
terminate/kill attempt cannot prove that, no replacement is started; the
original Future is reattached to a fresh relay generation and the lane remains
occupied until that work drains or fails normally.

After one reclaim the scheduler resets the RAM-slope epoch; samples from before
that timestamp cannot participate in the next 250-ms slope. A 750-ms settling
period blocks another normal victim. Only an emergency at >=94.5% may reclaim
again during settling, still one lane at a time.

Admission then enters a five-second damped-recovery state:

```text
settling             0 admissions
RAM >=87%            0 admissions
RAM <80%             <=1 job / 100 ms
80-84%               <=1 job / 150 ms
84-87%               <=1 job / 250 ms
```

Recovery exits early after two seconds stable between 82-88% with absolute
RAM slope <=0.25 GiB/s. Only then does normal 8/4/2 admission pacing resume.
During recovery, best-fit selection targets 86% rather than deliberately
choosing the largest work item merely because the host is underfilled.

Coverage retains its bounded v40.0.3 contract:

```text
causal:coverage:CPU
absolute cold estimate    = 5.00 GiB
incremental cold estimate = 2.75 GiB
global class cap          = 8
```

The v40.0.3 diagnostic's final failure was
`OSError: [Errno 28] No space left on device` in
`development_slice_hash.py`. v40.0.4 moves new hash-sort scratch to the
shared-compute volume with PID-addressable directories, cleans the reclaimed
PID immediately, removes dead-PID orphans at startup, and purges the old
v40.0.3 `dqbd-slice-hash-*` system-temp directories.

Identical Development-slice hashes are now persisted under a content-bound
cross-process cache. The cache identity binds source path, size, mtime,
Development cutoff, holdout boundary, projected columns and hash semantics;
the existing representation-independent digest algorithm itself is unchanged.

Runtime identities:

```text
DQBD_SINGLE_LANE_RECLAIM_CLOSED_LOOP_V40_0_4
REAL_LOAD_10MS_DAMPED_RECOVERY_80_86_90_92_95_V40_0_4
SINGLE_LANE_RECLAIM_FRONTIER_V40_0_4
HGB_FIRST_PREFETCHED_LEARNED_CPU_GPU_ROUTER_V40_0_4_3
DQBD_JOB_MEMORY_PROFILES_V3_REAL_LOAD
DQBD_REAL_LOAD_ROLLUP_V40_0_4
```

The numerical Ridge/HGB backend and device-kernel fingerprints remain
unchanged, so this is execution-only and compatible COMPLETE checkpoints are
reused.

## Candidate preparation and GPU section scheduling

For an unprepared H×Fold, only a CPU Ridge candidate may become the producer.
It publishes the Candidate-independent prepared-fold artifact before GPU work
is eligible. Workers do not retain the full signal panel or a large prepared
fold LRU when the shared SSD cache is active.

v40.0.2 successfully proved that Ridge can reach both GPUs: its published
32-lane diagnostic submitted 427 HGB GPU jobs and 85 Ridge GPU spillover jobs
with zero GPU fallback. However whole-job companions did not solve utilization:
repository-owned device-section duty was still about 28.4% on the RTX and
34.9% on the Radeon, with repeated tens-of-seconds idle gaps.

v40.0.3 therefore places the queue at the actual GPU section boundary. When
Ridge/HGB code reaches the physical device boundary, it publishes a tiny
SSD-backed ready ticket containing workload, device identity, priority and
expected section duration. Arrays and models are not copied into the queue.

Per physical device:

```text
ready HGB section
    -> first priority

otherwise ready Ridge section
    -> filler

otherwise
    -> parent feed controller stages more GPU-producing work
```

The existing cross-process `gpu-device` lock remains the kernel authority.
Only one LightGBM/OpenCL section may run on one physical GPU at a time; RTX and
Radeon may still execute concurrently.

Candidate-OOS permits up to four staged GPU-producing Futures per physical GPU
inside the fixed total lane budget. Slot 0 is HGB-first. Additional feed slots
prefer prepared Ridge while the actual waiting-section backlog is below target;
if no Ridge is ready, more HGB work may be staged. Any ready HGB section still
overtakes Ridge at the device-side queue.

Cold ready-backlog targets are expressed in predicted GPU seconds rather than
Future count:

```text
RTX 3070   ~9 ready GPU-seconds
Radeon VII ~12 ready GPU-seconds
```

Later seed-local model scheduling assigns a GPU by the lowest predicted finish
time:

```text
actual ready-section backlog
+ staged Future GPU seconds
+ predicted seconds for the new job
```

This allows the faster device to receive more work instead of forcing a
50/50 assignment count.

Queue evidence is persisted under:

```text
_shared-compute/gpu-section-ready-queue/
_shared-compute/gpu-section-queue-events.jsonl
_shared-compute/gpu-device-events/
```

The first two describe ready/waiting work and queue latency. The existing
device-event stream remains the authority for actual physical GPU occupancy.

Queue policy identity:

```text
HGB_FIRST_LEARNED_CPU_GPU_ROUTER_V40_0_4_2
```

Scheduler policy identity:

```text
SINGLE_LANE_RECLAIM_FRONTIER_V40_0_4
```

A host RAM/data exception does not disable a GPU. Only an actual
OpenCL/LightGBM/device/kernel failure triggers GPU disable/requeue.

## v40.0.2 host diagnostic that motivated v40.0.3

The published v40.0.2 `--workers 32` run is a failed runtime diagnostic, not
a scientific result and not a validation pass.

It established:

- prepared Ridge GPU spillover was real: 427 HGB and 85 Ridge GPU submissions;
- RTX/Radeon assignments were 272/240 with zero GPU fallback;
- device-section duty remained only ~28.4% RTX / ~34.9% Radeon;
- the causal coverage frontier admitted 32 jobs as lightweight evidence;
- only 6.4 GiB total incremental RAM was reserved for those 32 jobs;
- real host RAM reached ~92.5%;
- `candidate_evidence_coverage -> build_evidence_snapshot ->
  CandidateOosStore.read_matured -> pd.read_parquet` failed with
  `MemoryError`.

This diagnostic is the direct execution evidence for both v40.0.3 changes:
active RAM reclaim plus a device-section backlog queue.

## v40.0.1 host diagnostic that motivated v40.0.2

The first resumed v40.0.1 host diagnostic did not reach evaluation because of
the separately fixed Windows shared-lock initialization race. Before that
failure it produced enough repository-owned GPU occupancy evidence to expose a
second runtime defect:

- RTX device-section duty cycle was about 31.4%;
- Radeon device-section duty cycle was about 32.2%;
- both devices showed repeated tens-of-seconds idle gaps, with the largest
  gaps above 100 seconds;
- no `candidate_oos:RIDGE:GPU` memory profile was learned.

The two devices did overlap for substantial periods, so dual-GPU concurrency
itself worked. The missing behavior was same-device handoff between an HGB
future's estimator sections and prepared Ridge work. v40.0.2 changes only the
execution scheduler/topology required to make that handoff possible.

## v40.0.4.2 independent GPU workload router

The v40.0.4 RAM controller is retained unchanged. Candidate-OOS and causal
HGB/Ridge GPU execution are no longer admission clients of that controller:

- GPU pools are not registered with RAM reclaim;
- GPU Candidate-OOS and causal-model jobs own no RAM admission lease;
- GPU queue fill is not limited by the CPU admission budget;
- only RAM-governed CPU lanes can be stopped/requeued by the RAM controller.

For Candidate-OOS, the GPU router consumes only already-prepared H×Fold
artifacts. The causal frontier routes dependency-ready production model fits.
Both use the same strict workload order:

```text
HGB prepared
    ↓
RIDGE prepared
    ↓
no suitable prepared GPU work -> device waits
```

Each healthy physical GPU keeps four staged worker lanes. The physical device
lock still permits only one actual kernel section per GPU at a time.

Full Candidate-OOS service time and causal-model service time are learned
separately for:

```text
HGB:PREPARED × CPU
RIDGE:PREPARED × CPU
HGB:PREPARED × each GPU device
RIDGE:PREPARED × each GPU device
RIDGE:RAW × CPU
```

The bounded recent profile is persisted at:

```text
_shared-compute/candidate-execution-duration-profiles.json
```

GPU assignment minimizes predicted completion time using the device-specific
learned duration and the predicted finish of currently staged work. The faster
device may therefore receive more jobs.

CPU remains opportunistic. After GPU staging lanes have been filled, a free
RAM-governed CPU lane may take prepared HGB/Ridge only when:

```text
now + predicted CPU duration
<=
predicted next GPU availability + predicted GPU duration
```

Unprepared Ridge fold production remains CPU-only because the immutable
prepared-fold artifact does not yet exist.

If RAM reclaim terminates a CPU execution of an already-prepared HGB/Ridge
Candidate job, the manifested node returns to PENDING with
`execution_preference=GPU`. A healthy GPU then receives that job on a later
routing pass instead of immediately returning it to the CPU path.

Queue policy identity:

```text
HGB_FIRST_LEARNED_CPU_GPU_ROUTER_V40_0_4_2
```

RAM identities remain unchanged at v40.0.4.

## v40.0.4.3 GPU feed prefetch

The v40.0.4.2 router correctly detached GPU execution from RAM admission, but
the local host diagnostic exposed a remaining feed bottleneck: Step-9 still
passed `queue_ahead=0`, Candidate GPU routing could consume only already
prepared folds, and four Futures per device were insufficient to hide the
preparation/materialization gaps. Learned HGB P75 service time was about 61 s
on NVIDIA, 81 s on AMD and 360 s on CPU, so the remaining starvation was not
explained by CPU being competitive with GPU.

v40.0.4.3 changes only execution scheduling:

- `RamAdmissionScheduler` and `RamWorkerPauseController` are unchanged;
- Step-9 passes `queue_ahead=2` to Candidate-OOS and causal execution;
- each physical GPU keeps four actual worker processes and may hold two
  additional parent-queued Futures, for queue capacity six;
- the device-side lock remains the authority and still serializes one actual
  kernel section per physical GPU;
- `CandidateOosFactory.prepare_fold()` materializes only the immutable
  Candidate-independent H×Fold artifact;
- the Candidate scheduler keeps up to two distinct prep-only CPU tasks ahead,
  under the existing RAM admission policy, then releases the claimed Candidate
  back to PENDING so the prepared HGB/Ridge job can enter GPU routing;
- virtual per-device worker intervals include queued predicted starts when
  comparing CPU and GPU completion;
- Candidate and seed-local causal device choice also uses the shared physical
  `gpu-section-ready-queue` backlog as a lower bound on next device
  availability, so concurrent SHORT/PRIMARY/LONG schedulers account for each
  other's ready work instead of routing from local Future state alone;
- that 50-ms shared-backlog cache is keyed by the exact requested device set,
  preventing a one-GPU seed snapshot from being reused as a two-GPU view;
- seed-local causal assignment prefers lower staged depth before learned
  finish time. An empty healthy physical GPU therefore receives work before a
  faster GPU gets another staged Future; learned finish breaks ties once depth
  is equal;
- duration profiles learn worker-local service time, not submit-to-complete
  wall time, so intentional queue wait does not inflate RTX/Radeon estimates;
- prep-only claim release updates both SQLite state and the persisted job
  payload before returning the Candidate to PENDING, keeping resumable
  `run-state/jobs.json` consistent;
- actual device-section tickets/events carry `job_id` and
  `gpu_worker_slot`, allowing an assignment to be correlated with a real
  RTX/Radeon `ACQUIRE/RELEASE`.

Prepared HGB remains the first GPU workload and prepared Ridge remains the
fallback. CPU execution of prepared HGB/Ridge remains opportunistic only when
its learned predicted completion is no later than the next predicted GPU
completion. A CPU-reclaimed prepared job keeps `execution_preference=GPU`.

Queue policy identity:

```text
HGB_FIRST_PREFETCHED_LEARNED_CPU_GPU_ROUTER_V40_0_4_3
```

This is scheduler-only. Numerical GPU fingerprints, immutable scientific
artifacts, seed-local causal visibility and the closed prospective holdout
contract are unchanged.

## GPU occupancy evidence

Vendor utilization telemetry remains diagnostic and can be unavailable.
v40 additionally records the authoritative occupancy of the repository's own
per-device lock:

```text
_shared-compute/gpu-device-events/
  platform-<p>-device-<d>.jsonl
```

Each actual device section writes `ACQUIRE` and `RELEASE` with timestamp,
PID, `job_id`, `gpu_worker_slot`, device identity and duration. Queue
assignment by itself is not execution evidence; the matching `job_id` in this
stream is the authoritative proof that the assigned job reached the physical
device. Overlapping intervals across the two files are direct evidence that
RTX and Radeon GPU sections ran concurrently.

## Device-specific HGB stability policy

The Radeon VII must not use the NVIDIA bin setting. The known-safe execution
policy remains intentionally asymmetric:

```text
AMD / Radeon VII: max_bin = 15
NVIDIA / RTX 3070: max_bin = 63
gpu_use_dp = false
n_jobs = 1
```

The AMD `max_bin=15` guard is retained because wider settings previously
triggered OpenCL kernel / Windows watchdog failures. This difference is not
hidden: device identity plus kernel parameters are part of the immutable
execution fingerprint used by Candidate-OOS and production-Generation cache
identity. An AMD-15 artifact therefore cannot be reused as an NVIDIA-63
artifact, or vice versa.

The actual per-device GPU fit remains serialized. Two simultaneous LightGBM
kernels on one physical GPU are not enabled by this change. Any future
same-device GPU concurrency requires a separate runtime benchmark, especially
on the Radeon path.

## GPU lock scope

For Ridge, CPU centering and the small linear solve occur outside the
per-device lock. The lock covers only GPU buffer creation, kernel execution
and transfers. For HGB, the lock is acquired separately for the regressor,
positive classifier and downside classifier, permitting another prepared job
to use the device between estimator fits without running two unsafe kernels
at once.

## Resume and audit

The 2026-09-03 resumed v40.0.4.3 diagnostic exposed a derived-state resume
failure after healthy Candidate-OOS execution: Coverage repaired stale
dependency payloads to the larger causally matured observed set, then the
corrected Evidence Snapshot collided with an older valid immutable H×cutoff
leaf.

Resume therefore has two explicit stages:

```text
cache miss or --slow-resume
    -> rebuild current manifested graph metadata
    -> compare semantic job payloads
    -> requeue changed nodes + transitive derived descendants only
    -> preserve unrelated COMPLETE checkpoints
    -> preserve old immutable snapshot leaf
    -> publish corrected snapshot at
       <H>/<cutoff>/versions/<manifest_sha256>/
    -> write step9-initialization-cache-v2.json

next default resume
    -> validate Git SHA + input-stat fingerprint + manifested-job-contract hash
    -> cache HIT
    -> skip reconciliation
    -> skip COMPLETE jobs
```

The parent coordinator owns descendant invalidation. Worker processes may
report that a valid immutable snapshot conflict was content-versioned, but do
not mutate central scheduler state themselves. Partial, missing-pair or
hash-invalid snapshot artifacts remain hard failures.

The execution backend is versioned:

```text
DQBD_GPU_RIDGE_HGB_PRETRAINING_V6
OPENCL_FP64_RIDGE_REGRESSOR_SKLEARN_BUNDLE_V2_SHARED_GRAM
OPENCL_LIGHTGBM_GPU_HGB_SKLEARN_BUNDLE_V5_DEVICE_FINGERPRINTED
DUAL_GPU_OPENCL_HGB_AMD_TDR_GUARD_V4
AMD_MAX_BIN_15_NVIDIA_MAX_BIN_63_SINGLE_PRECISION_V4_DEVICE_IDENTITY
```

A new local run identity must be used for this execution contract. Existing
v38 artifacts/logs remain evidence about the older runtime and must not be
silently resumed under V6.

Local v40.0.4.3 validation must verify, without opening economic results early:

- one `--slow-resume` completes graph reconciliation without rebuilding
  unrelated COMPLETE Candidate-OOS/Generation work;
- a subsequent default resume reports the v2 initialization-cache hit and
  skips reconciled COMPLETE jobs;
- stale Coverage dependency payloads no longer terminate the run with
  `EVIDENCE_SNAPSHOT_IMMUTABLE_CONFLICT`; valid old leaves remain intact and
  corrected snapshots resolve to deterministic manifest-hash version paths;
- corrupt/partial/hash-invalid Evidence Snapshot state still fails closed;
- holdout reads remain zero;
- no RAM OOM, ENOSPC or 95% hard-ceiling violation;
- normal runnable periods settle around the 82-88% target corridor instead of
  repeating the v40.0.3 20-90% sawtooth;
- each pressure decision emits one `RAM_*_RECLAIM_ONE` and only one logical
  lane disappears; unrelated Futures remain alive;
- `ram_scheduler_targeted_reclaim_count` and per-job
  `ram_reclaim_count` are recorded, and reclaimed nodes resume from PENDING
  with completion priority rather than FAILED;
- no second normal victim occurs from pre-reclaim slope samples during the
  750-ms settling interval;
- recovery does not admit above 87% and otherwise refills at no more than one
  job per 100-250 ms until stable;
- `causal:coverage:CPU` remains <=8 and learns its own memory profile;
- shared Development-slice-hash values are reused and neither PID-addressable
  nor legacy system-temp scratch accumulates;
- Candidate-OOS and causal-model GPU jobs continue to dispatch while CPU RAM
  admission is closed, proving that GPU routing is independent from the RAM
  scheduler;
- GPU staging remains HGB-first then Ridge and no unprepared job enters GPU;
- learned CPU/GPU/device duration profiles populate and device assignment
  follows predicted completion rather than fixed 50/50 counts;
- a prepared HGB/Ridge job reclaimed from CPU returns with GPU preference and
  can later complete on GPU;
- the GPU section queue emits HGB/Ridge tickets and drains without stale
  tickets;
- when unprepared Candidate folds exist, up to two distinct prep-only tasks
  stay ahead without publishing Candidate-OOS evidence;
- per-device parent queue depth may reach six while the actual GPU worker count
  remains four and physical kernel concurrency remains one;
- learned duration samples exclude parent queue wait;
- an assigned HGB/Ridge job that reaches a GPU section carries the same
  `job_id` in `gpu-device-events`, including real NVIDIA evidence rather
  than queue assignment alone;
- Radeon HGB remains `max_bin=15` and NVIDIA remains `max_bin=63`;
- exact execution fingerprints, immutable artifacts and seed-local causal
  visibility remain valid.

The host integration checks are
`dynamic_qbd_gpu_pretraining_self_test.py`,
`dynamic_qbd_manifested_job_coordinator_self_test.py` and
`candidate_oos_self_test.py`. Their execution on the configured Windows
dual-GPU host remains pending.
