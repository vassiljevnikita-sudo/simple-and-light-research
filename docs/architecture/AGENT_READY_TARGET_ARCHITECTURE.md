> **Scope note:** This architecture document describes the protected N25/control-plane refactor track. It is not the complete current Dynamic-QBD research map. Agents must start at [../../AGENTS.md](../../AGENTS.md) and [../../TOBECONTINUED.md](../../TOBECONTINUED.md); current QBD state is [../../research/DYNAMIC_QBD_CURRENT_STATE.md](../../research/DYNAMIC_QBD_CURRENT_STATE.md).

# Agent-ready target architecture

Status: Phase 1 complete; Phase 2 implemented for the V4 N25 bundled runner  
Scope: branch derived from `research/v4-n25-multifactor`  
Safety: research and paper trading only

## 1. Goal

The repository must remain scientifically auditable while becoming faster and safer for humans, language models, and local coding agents to understand and modify.

The target is not a language rewrite. Python remains the orchestration and research language. TypeScript workers and future native kernels are permitted only behind explicit contracts and only when a measured bottleneck justifies them.

Existing V4/N25 model semantics, frozen parameters, historical results, and promotion status remain unchanged.

## 2. Non-negotiable constraints

- `V4_N25_T212_RESEARCH_V1` remains the current V4 research candidate.
- `EXEC_N25_DUAL_VENUE_RESEARCH_V1` remains the execution research candidate.
- `V45_N25_HYBRID_STOP_RESEARCH_V1` remains the V4.5 overlay candidate.
- `V4_FROZEN_10K_LEGACY` remains a historical baseline only.
- No live order, broker write, model promotion, holdout reopening, or result relabelling is introduced.
- Negative and incomplete research results remain preserved.
- Existing production V2 code on `main` is outside this architecture branch.
- The authoritative model identity remains `stock_predictor/research_model_registry.json`.
- The separate V4.5 × N25 model-family branch owns its replay engine and remains outside this refactor.

## 3. Decision filter: could an agent do this faster or safer?

Every architectural change must answer these questions before implementation:

1. **Discovery:** Can an agent find the authoritative file without searching the whole repository?
2. **Scope:** Can an agent identify which model, data source, workflow, and validation gate the change affects?
3. **Execution:** Is there one canonical, allowlisted task instead of several undocumented commands?
4. **Feedback:** Does the command return a deterministic exit code and machine-readable output?
5. **Impact:** Can changed files be mapped to the smallest sufficient validation set?
6. **Safety:** Are forbidden actions and promotion boundaries machine-checkable?
7. **Context size:** Can an agent obtain a compact repository summary without loading large reports or artifacts?
8. **Duplication:** Is a fact defined once and referenced elsewhere?
9. **Parallel work:** Can branches and worktrees modify isolated modules without sharing generated state?
10. **Rollback:** Is the change additive or compatibility-preserving until validated?

A change that does not improve at least one item and weakens another must be rejected or redesigned.

## 4. Implemented layers

Dependency direction is top to bottom. Lower layers must not import higher layers.

```text
devtools / CI
    |
    +--> control_plane
    |      |- project and model registries
    |      |- architecture validation
    |      `- allowlisted task catalog
    |
    +--> research applications
    |      `- typed V4 N25 application and thin CLI
    |
    `--> research_runtime
           |- verified immutable source bundles
           `- atomic hashed execution receipts

research applications --> adapters only through explicit boundaries
```

### 4.1 Control plane

Path: `stock_predictor/control_plane/`

Responsibilities:

- load the machine-readable project manifest;
- resolve repository paths;
- validate model identity and safety invariants;
- validate architectural layer references;
- load and execute allowlisted tasks without a shell;
- expose compact context to humans and agents.

It uses the Python standard library only. The fastest repository checks therefore work before optional research dependencies are installed.

### 4.2 Developer and agent tools

Path: `stock_predictor/devtools/`

Responsibilities:

- one CLI entry point;
- repository health checks;
- compact context export;
- changed-file impact analysis;
- task discovery and task execution;
- deterministic JSON output;
- optional hashed execution receipts.

Canonical entry point:

```bash
python -m stock_predictor.devtools.agent_cli <command>
```

### 4.3 Shared research runtime

Path: `stock_predictor/research_runtime/`

Responsibilities:

- normalize text-form Base64 artifacts;
- validate Base64 and gzip structure;
- verify the uncompressed source SHA-256;
- compile before execution;
- restore `sys.argv` and `sys.path` after execution;
- preserve legacy direct-import behavior while supporting package entrypoints;
- write small artifacts atomically;
- store hashes rather than raw task output in receipts.

This layer contains no model, portfolio, provider, broker, or promotion logic.

### 4.4 V4 N25 application boundary

Path: `stock_predictor/backtests/v4_n25_multifactor_runtime/`

The V4 runner is now split into:

```text
backtest_v4_n25_multifactor_runner.py
    compatibility only
        |
        v
v4_n25_multifactor_runtime/cli.py
    wrapper argument handling only
        |
        v
v4_n25_multifactor_runtime/application.py
    contract and registry validation
        |
        v
research_runtime/bundles.py
    source verification and execution
        |
        v
backtest_v4_n25_multifactor.py.gz.b64
    unchanged immutable research source
```

The explicit runner contract records:

- runner and model identity;
- execution and benchmark policies;
- source and bundle paths;
- pinned uncompressed source hash;
- mismatch error code;
- paper-trading and broker-write restrictions;
- canonical and legacy entrypoints;
- frozen workflow compatibility arguments.

The historical V4 workflow passes cost assumptions that the immutable source does not parse itself. The typed wrapper now consumes and validates these values before source execution:

```text
primary stock roundtrip: 20 bps
stress stock roundtrip:  30 bps
ETF roundtrip:            5 bps
```

Matching values preserve the historical workflow. Any changed value fails before the source is executed. The wrapper therefore restores compatibility without silently allowing model or cost retuning.

The old command remains valid, but it no longer owns source-verification logic.

### 4.5 Data and artifacts

Large downloaded data and generated results must not define architecture or model identity. They are inputs or evidence.

Preferred direction:

- immutable raw data outside source modules;
- versioned manifests and checksums;
- Parquet for large tabular data where appropriate;
- small JSON summaries committed only when they are audit evidence;
- generated caches, receipts and bytecode ignored;
- raw command output excluded from receipts.

Artifact upload is an evidence-transport step, not the validation itself. Workflows keep tests and contract checks fail-closed, but treat upload failure as non-blocking when GitHub storage quota is unavailable. The underlying evidence files are still generated in the job workspace.

## 5. Machine-readable repository map

`stock_predictor/project_manifest.json` is the architecture index. It does not replace the model registry. It records:

- safety boundaries;
- architectural layers;
- required paths;
- canonical commands;
- allowlisted tasks;
- changed-file impact rules;
- expected current model IDs;
- explicit runner-contract locations.

The manifest is deliberately small enough for an agent to load in one tool call.

## 6. Canonical commands and tasks

```bash
# Compact model, safety, architecture, command and task context
python -m stock_predictor.devtools.agent_cli context

# Repository and architecture validation
python -m stock_predictor.devtools.agent_cli check

# Environment and Git diagnostics
python -m stock_predictor.devtools.agent_cli doctor

# List exact allowlisted commands
python -m stock_predictor.devtools.agent_cli tasks

# Minimal validation plan for the current diff
python -m stock_predictor.devtools.agent_cli impact --base research/v4-n25-multifactor

# Verify the immutable V4 runner without running the model
python -m stock_predictor.devtools.agent_cli run-task v4-n25-verify

# Execute the bundled source self-test through the typed runtime
python -m stock_predictor.devtools.agent_cli run-task v4-n25-self-test

# Run runtime, compatibility and architecture tests
python -m stock_predictor.devtools.agent_cli run-task v4-n25-focused-tests
```

Task execution uses argument arrays and `shell=False`. Tasks cannot be invented at runtime; they must first be added to the reviewed manifest.

## 7. Parallel branch and worktree model

This architecture branch is based on `research/v4-n25-multifactor` and targets that branch in its pull request.

Parallel agents use separate branches and separate Git worktrees. They must not share one working tree by switching branches during long-running jobs.

```bash
git worktree add ../simple-and-light-agent-a agent/task-a
git worktree add ../simple-and-light-agent-b agent/task-b
```

Generated outputs must be written under branch-specific or run-specific artifact directories. A worker must never write directly to another worker's checkout.

## 8. Migration status

### Phase 1 — complete

- target architecture and `AGENTS.md`;
- project manifest;
- control-plane validation;
- compact context, doctor and impact CLI;
- shared model-registry validation;
- architecture CI;
- cache and local-environment ignore rules.

### Phase 2A — complete for the V4 N25 bundled runner

- shared source-artifact verification;
- shared atomic receipt helper;
- typed V4 runner contract;
- thin package CLI and application boundary;
- historical command retained as compatibility wrapper;
- frozen legacy workflow arguments validated rather than forwarded;
- allowlisted manifest tasks;
- compatibility, source-hash and process-state parity tests;
- CI and local agents use the same tasks;
- validation status separated from optional artifact transport.

### Phase 2B — remaining

- migrate other bundled runners to the shared runtime when their active branches are integrated;
- move repeated provider and result-path resolution into explicit adapters;
- introduce typed records at provider, portfolio and artifact boundaries;
- replace remaining implicit data discovery with explicit dataset manifests;
- reduce package `__init__` side effects where imports pull optional dependencies unnecessarily;
- profile V4/V4.5 backtests before changing concurrency or numerical kernels.

### Phase 3 — only after profiling

- use NumPy/Numba or process pools for measured numerical hot paths;
- retain TypeScript worker implementations only where benchmarks show value;
- introduce Rust/PyO3 only for a stable, isolated, CPU-bound kernel with parity tests.

## 9. Current definition of done

This branch is architecturally complete when:

- the project manifest and this document agree;
- the model registry validates through the shared control plane;
- the V4 runner contract matches the current registry;
- the compressed V4 source hash and compilation validate;
- old and new entrypoints are behaviorally compatible;
- frozen workflow arguments are accepted only at their contracted values;
- allowlisted tasks are discoverable and executable without a shell;
- changed files map to focused checks;
- CI runs the same task entrypoints as local agents;
- no V4/N25 parameter or economic result has changed;
- no live-trading capability has been added.
