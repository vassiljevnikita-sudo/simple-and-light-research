# Dynamic-QBD v40.1 Runtime Hardening

Date: 2026-09-05

Scope: execution/runtime only. This document supplements
`DYNAMIC_QBD_DATA_AND_RUNTIME.md` and the BrokenProcessPool incident audit.
It does not change the manifested scientific DAG, causal visibility, FIXED
Recipe/model contracts, portfolio accounting, RAM thresholds, GPU numerical
policy, prospective holdout boundary or promotion/capital authority.

## Status

The remaining recovery defects identified after the 2026-09-05
`BrokenProcessPool` incident are implemented in the v40.1 Hotstart execution
layer. The implementation is additive to the v40.0.4.3 minimum execution
baseline.

The target-PC acceptance is still open. Source/self-test presence is not a
substitute for the resumed Step-9 runtime evidence.

## 1. Attempt fencing and same-owner ABA prevention

Every Candidate-OOS and mixed-causal scheduler invocation now receives an
`AttemptFencedManifestedJobStore` facade.

The underlying SQLite/WAL `ManifestedJobStore` remains authoritative. The
facade records the exact `attempt` returned by each local claim and requires
that same attempt in the SQL predicate for:

- `finish_job`;
- `heartbeat`;
- `release_claim`;
- RAM-reclaim requeue;
- execution-failure requeue.

It also automatically excludes all locally in-flight job IDs from subsequent
`ready_jobs` and `claim_ready` operations. Therefore an external watchdog may
return an old attempt to `PENDING`, but the still-alive old Future cannot
publish into or heartbeat a newer same-owner attempt.

Stale-attempt publication remains fail-closed with the existing manifested
ownership/lease error. No scientific result from the stale Future is accepted.

## 2. Lane replacement is explicitly repairable

`_ReclaimableLanePool` now has explicit execution states:

```text
ONLINE
REPAIRING
OFFLINE
```

If a selected child is proven terminated but construction of its replacement
executor fails:

1. the old outer Future is always settled with a transient
   `BrokenProcessPool:DQBD_LANE_REPLACEMENT_FAILED...` exception;
2. the lane releases its old task pointers and becomes `OFFLINE`;
3. no work is submitted into that lane while it is offline;
4. other ONLINE lanes continue draining queued work;
5. the lane watchdog and later submissions independently call
   `repair_offline_lanes()`;
6. successful executor construction returns the lane to `ONLINE` and queued
   work may use it again.

A failed replacement constructor therefore cannot leave a hidden
`replacing=True` lane with an unresolved Future indefinitely.

The earlier v40.0.4 safety property is retained: when termination of the old
child itself cannot be proven, no replacement child is constructed beside it.

## 3. Deferred termination preserves the lease

When stale-dispatcher recovery selects a lane but termination cannot yet be
proved, the lane watchdog now explicitly refreshes the current manifested
attempt's heartbeat/lease before leaving the lane in drain mode.

This closes the race where the lease could expire while the old Future was
still alive, allowing the same owner to claim a new attempt before the old
Future settled.

## 4. Dispatcher and repair incidents do not consume worker-crash budget

The following execution markers are orchestration recovery, not worker-quality
failures:

```text
DQBD_STALE_DISPATCHER_ACTIVITY
DQBD_LANE_REPLACEMENT_FAILED
```

The attempt-fenced store forces these markers onto the dispatcher/execution
recovery counter path even if an older Candidate-OOS caller supplied
`count_worker_failure=True`.

This also closes the boundary case where a job already had
`WORKER_FAILURE_RETRY_LIMIT` genuine crashes: a later stale-dispatcher incident
cannot be interpreted as crash `N+1` and converted to terminal scientific
`FAILED` state.

Genuine child crashes still consume `worker_failure_count` and retain the
existing bounded fail-closed limit.

## 5. RAM/profile/JSON telemetry is advisory

The v40.0.4 RAM algorithm and thresholds are unchanged.

The 10-ms controller now separates sampling/admission state from telemetry
publication:

- successful system/live-board samples update in-memory scheduling state;
- sampling failures are counted and do not terminate the controller thread;
- runtime-rollup write failures are counted and do not terminate the controller;
- memory-profile observations update in-memory learned estimates before disk
  persistence;
- profile persistence failures are counted and cannot retroactively fail a
  completed worker result;
- controller telemetry exposes thread-alive state, last successful sample,
  consecutive sample failures and profile/runtime telemetry write failures.

RAM reclaim event/spill writes are likewise advisory. Failure to publish an
operator JSON/JSONL telemetry record does not cancel the actual single-lane
recovery decision or a scientifically valid committed result.

## 6. Thread-unique atomic JSON writers

The runtime paths identified by the incident audit no longer use a fixed
PID-only sibling temp file.

The hardening layer uses:

```text
<TARGET>.<PID>.<THREAD_ID>.<TIME_NS>.tmp
```

with atomic `os.replace` and bounded sharing-violation retries for:

- coordinator `_write_json` publications;
- manifested `jobs.json` materialization;
- Step-9 runner `_json` publications;
- RAM reclaim spill JSON.

SQLite remains authoritative for manifested scheduler state.

The Step-9 runner writer is patched after module execution by a one-shot import
loader hook, because package bootstrap executes before the runner defines its
local `_json` function. The hook changes only that function after normal module
loading and then removes itself.

## 7. Focused regression evidence

New focused source test:

`dynamic_qbd_runtime_hardening_self_test.py`

It covers:

1. stale attempt 1 cannot publish into a re-claimed attempt 2;
2. dispatcher recovery at an already exhausted genuine worker-failure budget
   remains `PENDING` and does not increment that budget;
3. synthetic replacement-constructor failure settles the Future, leaves the
   lane `OFFLINE`, then repairs it back to `ONLINE`;
4. deferred stale-dispatcher termination explicitly preserves the job lease;
5. a synthetic profile persistence error remains advisory while in-memory RAM
   learning and the controller thread continue;
6. the Step-9 runner receives the collision-safe JSON writer.

This test is intentionally lightweight. It does not run model fitting, replay,
portfolio evaluation or any prospective holdout read.

## 8. Acceptance still required

Before the resumed long Step-9 run is called stall-proof, the target PC must
still demonstrate:

- all focused recovery/hotstart self-tests PASS locally;
- no prospective holdout reads;
- stale-dispatcher recovery returns work to `PENDING` and later `COMPLETE`;
- fourth genuine worker crash remains terminal while dispatcher/repair recovery
  does not consume that budget;
- no duplicate completion or stale-attempt publication;
- forced replacement-constructor failure produces no unresolved Future;
- `OFFLINE` lane repair restores capacity without overlapping a still-live old
  child;
- simulated transient telemetry/profile write errors do not stop RAM sampling
  or admission;
- Hotstart distinguishes a long active portfolio batch from a logical stall;
- existing compatible COMPLETE checkpoints remain reusable;
- no orphan worker process remains after boundary swap/restart.

No GitHub CI result is accepted as evidence for this gate.
