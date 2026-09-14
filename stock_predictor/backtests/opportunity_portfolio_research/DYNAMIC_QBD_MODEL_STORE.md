# Dynamic-QBD Causal Model Store V1

This branch adds only the causal infrastructure for a historical model-store
experiment. It does not add a selector, a second factory, or a second
portfolio replay.

The boundaries are explicit:

- Development is closed at `2025-12-31`.
- The prospective holdout begins at `2026-07-25` and is fail-closed.
- `ModelStore.generations_as_of(t)` exposes only generations with
  `created_at <= t`.
- `EvidenceCursor.as_of(t)` exposes only records with `matured_at <= t`.
- A `ModelGenerationRecord` is append-only and immutable by generation ID.
- `RunContract.write()` refuses to replace a previously persisted contract
  with a different hash.

The modules are deliberately separated:

| Responsibility | Module |
| --- | --- |
| Immutable records and run hash | `dynamic_qbd_orchestrator_contract.py` |
| Generation visibility and append-only storage | `dynamic_qbd_model_store.py` |
| Matured evidence cursor | `dynamic_qbd_evidence_store_cursor.py` |
| `NEW_MATURED_FOLD` event planning | `dynamic_qbd_generation_event_planner.py` |
| Development-only bundle construction | `dynamic_qbd_historical_store_builder.py` |
| Selector-free pseudo-live trace | `dynamic_qbd_pseudolive_replay.py` |
| Non-economic visibility diagnostics | `dynamic_qbd_model_store_evaluation.py` |
| Existing generation -> ModelStore provenance adapter | `dynamic_qbd_generation_store_adapter.py` |
| Frozen initial/S2/S3/evaluation run gates | `dynamic_qbd_run_gate_contract.py` |
| Structural recipe registry and eligibility | `dynamic_qbd_recipe_store.py` |
| Selector-minimal S0/S1/S2a/S2b/S3/Oracle arms | `dynamic_qbd_baseline_arms.py` |
| H3/H11/H23 chronology gate | `dynamic_qbd_chronology_smoke.py` |
| Real-provider H3/H11/H23 integration gate | `dynamic_qbd_model_store_integration_smoke.py` |

The causal Development-run gate artifact is
`research/DQBD_CAUSAL_MODEL_STORE_V1_RUN_GATES.json`. Its
`decision_contract_hash` must be supplied to `PseudoLiveCoordinator`; a
different hash is rejected. The contract freezes the existing Candidate-OOS
selection rule at the seed, a no-performance-score newest-visible-generation
S2 policy, a latest-generation-per-recipe equal-weight S3 pool, and the
time-block bootstrap/evaluation definitions. For a same-horizon pool with `n`
visible members, each member has explicit weight `1/n`; no visible
same-horizon generation is fail-closed. The productive route is
`state_as_of_rule(...)` with a required horizon and the frozen rule, so an
arbitrary caller-selected generation list is not the S3 evaluation path. The
S3 rule hash is embedded in the immutable run-gate contract. It does not
contain performance results.

Step 8 freezes the evaluation layer before any economic run. The primary
scientific question is `ORACLE_HEADROOM_EXISTS`; `S0`, `S1`, and `S3` are
causal baselines and S2 is a mechanical newest-visible reference, not an
evidence-based challenger selector. The historical contrast retained for
backward-compatible reporting is `S2_MINUS_S0`; secondary contrasts are `S1_MINUS_S0`,
`S3_MINUS_S0`, and `S2_MINUS_S1`. All contrasts use paired path returns and
the canonical cost/accounting replay. The moving-block bootstrap resamples
complete paired paths and computes canonical `cagr_excess` (strategy CAGR
minus benchmark CAGR) for every arm; it does not use mean daily relative
returns. The frozen metrics include terminal
wealth, CAGR/excess, relative and absolute drawdown, turnover, trades, costs,
downside/expected-shortfall measures, switching, generation age/retention,
recipe/ticker concentration, and diagnostic Oracle regret/captured alpha.
The result schema is fixed in `EvaluationContract` before results exist.

Existing `dynamic_qbd_factory.py`, `candidate_oos.py`, and
`dynamic_qbd_portfolio_replay.py` remain providers. They are not duplicated or
modified by this layer.
