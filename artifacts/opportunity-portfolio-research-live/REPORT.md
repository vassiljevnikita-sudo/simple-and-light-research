# Chronological V5 Opportunity Portfolio Research

Decision: `DEVELOPMENT_SUITE_COMPLETE` / `MIXED_PORTFOLIO_EVIDENCE_NOT_READY_FOR_FINAL_HOLDOUT`

The locked final holdout was not opened.

## Corrected research contract

- Score thresholds are calibrated on historical daily top scores, matching Opportunity-Core.
- The numeric resolved threshold is persisted and reused in every Development/cost/tax replay.
- Stage-A sampling is balanced across score quantile, top fraction, max_names and holding_days.
- Policy ranking is lexicographic across chronological prior-fold robustness, not a single historical CAGR.
- CPU concurrency is bounded: requested 12, effective 12, hard cap 12.
- Regime results compound daily returns attributed to each regime; they do not use last/first NAV across disjoint dates.
- Concentration diagnostics include ticker dependence and explicit exclude-top-trade/top-ticker replays.
- parameter_plateau.csv contains actual adjacent-policy replays.

## Outer-fold summary

[
  {
    "active_fold_fraction": 0.5,
    "active_folds": 3,
    "chained_oos_cagr": 0.10798390715687334,
    "chained_oos_cagr_excess": 0.019824104742758086,
    "chained_oos_diagnostic_only": true,
    "chained_oos_total_return": 0.3599060729096857,
    "chained_oos_total_return_excess": 0.07164853702204099,
    "chained_urth_cagr": 0.08815980241411525,
    "chained_urth_total_return": 0.2882575358876447,
    "folds": 6,
    "horizon": 5,
    "inactive_folds": 3,
    "median_active_cagr_excess": 0.027479225337586177,
    "median_cagr_excess": 0.0,
    "positive_active_fold_fraction": 0.6666666666666666,
    "positive_active_folds": 2,
    "positive_fold_fraction": 0.3333333333333333,
    "positive_folds": 2,
    "q25_active_cagr_excess": -0.12153584209830026,
    "q25_cagr_excess": 0.0,
    "trade_count": 19,
    "worst_active_cagr_excess": -0.2705509095341867,
    "worst_cagr_excess": -0.2705509095341867
  },
  {
    "active_fold_fraction": 0.5,
    "active_folds": 3,
    "chained_oos_cagr": 0.2039082403725221,
    "chained_oos_cagr_excess": 0.11574843795840684,
    "chained_oos_diagnostic_only": true,
    "chained_oos_total_return": 0.7442738571525813,
    "chained_oos_total_return_excess": 0.45601632126493663,
    "chained_urth_cagr": 0.08815980241411525,
    "chained_urth_total_return": 0.2882575358876447,
    "folds": 6,
    "horizon": 10,
    "inactive_folds": 3,
    "median_active_cagr_excess": 0.13450060326586943,
    "median_cagr_excess": 0.04439135699424468,
    "positive_active_fold_fraction": 1.0,
    "positive_active_folds": 3,
    "positive_fold_fraction": 0.5,
    "positive_folds": 3,
    "q25_active_cagr_excess": 0.1116416586271794,
    "q25_cagr_excess": 0.0,
    "trade_count": 41,
    "worst_active_cagr_excess": 0.08878271398848936,
    "worst_cagr_excess": 0.0
  },
  {
    "active_fold_fraction": 0.3333333333333333,
    "active_folds": 2,
    "chained_oos_cagr": 0.02388726140541464,
    "chained_oos_cagr_excess": -0.06427254100870061,
    "chained_oos_diagnostic_only": true,
    "chained_oos_total_return": 0.07333518889088975,
    "chained_oos_total_return_excess": -0.21492234699675494,
    "chained_urth_cagr": 0.08815980241411525,
    "chained_urth_total_return": 0.2882575358876447,
    "folds": 6,
    "horizon": 20,
    "inactive_folds": 4,
    "median_active_cagr_excess": -0.17470000650510253,
    "median_cagr_excess": 0.0,
    "positive_active_fold_fraction": 0.0,
    "positive_active_folds": 0,
    "positive_fold_fraction": 0.0,
    "positive_folds": 0,
    "q25_active_cagr_excess": -0.2289567005661065,
    "q25_cagr_excess": -0.04963996378732094,
    "trade_count": 37,
    "worst_active_cagr_excess": -0.2832133946271105,
    "worst_cagr_excess": -0.2832133946271105
  }
]

## Readiness gates

{
  "accounting_invariants_pass": true,
  "all_readiness_gates_pass": false,
  "concentration_pass": false,
  "cost_tax_stress_pass": true,
  "cross_regime_context_matrix_complete": true,
  "development_execution_complete": true,
  "dynamic_exit_stage_complete": true,
  "exposure_invariants_pass": true,
  "horizon_execution_coverage": {
    "5": {
      "active_fold_fraction": 0.5,
      "active_folds": 3,
      "actual_outer_folds": 6,
      "dynamic_stage_complete": true,
      "evidence_coverage_complete": false,
      "execution_complete": true,
      "execution_failure_detected": false,
      "execution_failure_messages": [],
      "expected_outer_folds": 6,
      "final_policy_present": true,
      "frozen_threshold_present": true,
      "inactive_folds": 3,
      "nominal_grid_coverage_complete": true,
      "oos_trade_count": 19,
      "positive_active_fold_fraction": 0.6666666666666666,
      "sparse_outer_robustness_pass": false
    },
    "10": {
      "active_fold_fraction": 0.5,
      "active_folds": 3,
      "actual_outer_folds": 6,
      "dynamic_stage_complete": true,
      "evidence_coverage_complete": false,
      "execution_complete": true,
      "execution_failure_detected": false,
      "execution_failure_messages": [],
      "expected_outer_folds": 6,
      "final_policy_present": true,
      "frozen_threshold_present": true,
      "inactive_folds": 3,
      "nominal_grid_coverage_complete": true,
      "oos_trade_count": 41,
      "positive_active_fold_fraction": 1.0,
      "sparse_outer_robustness_pass": false
    },
    "20": {
      "active_fold_fraction": 0.3333333333333333,
      "active_folds": 2,
      "actual_outer_folds": 6,
      "dynamic_stage_complete": true,
      "evidence_coverage_complete": false,
      "execution_complete": true,
      "execution_failure_detected": false,
      "execution_failure_messages": [],
      "expected_outer_folds": 6,
      "final_policy_present": true,
      "frozen_threshold_present": true,
      "inactive_folds": 4,
      "nominal_grid_coverage_complete": true,
      "oos_trade_count": 37,
      "positive_active_fold_fraction": 0.0,
      "sparse_outer_robustness_pass": false
    }
  },
  "inactive_outer_folds_are_negative_evidence": false,
  "minimum_active_fold_fraction": 0.5,
  "minimum_active_outer_folds": 4,
  "minimum_oos_roundtrips": 20,
  "nominal_grid_coverage_complete": true,
  "outer_fold_robustness_pass": false,
  "parameter_plateau_pass": false,
  "search_coverage_complete": true,
  "sparse_evidence_coverage_complete": false
}

## Frozen Development policy

{
  "decision": "MIXED_PORTFOLIO_EVIDENCE_NOT_READY_FOR_FINAL_HOLDOUT",
  "final_holdout_locked": true,
  "final_holdout_opened": false,
  "policies": [
    {
      "allocation": "EQUAL_ACTIVE",
      "exit_family": "FIXED",
      "exit_value": 0.0,
      "holding_days": 5,
      "horizon": 5,
      "max_names": 3,
      "policy_id": "9e73f5f04774e04b",
      "replacement": "IGNORE_NEW",
      "resolved_threshold": 0.047775843455748125,
      "score_quantile": 0.95,
      "sleeve": 0.5,
      "top_fraction": 0.01
    },
    {
      "allocation": "EQUAL_ACTIVE",
      "exit_family": "FIXED",
      "exit_value": 0.0,
      "holding_days": 5,
      "horizon": 10,
      "max_names": 3,
      "policy_id": "a7d06b454b7402b0",
      "replacement": "IGNORE_NEW",
      "resolved_threshold": 0.1858273999939213,
      "score_quantile": 0.95,
      "sleeve": 0.5,
      "top_fraction": 0.01
    },
    {
      "allocation": "EQUAL_ACTIVE",
      "exit_family": "FIXED",
      "exit_value": 0.0,
      "holding_days": 5,
      "horizon": 20,
      "max_names": 5,
      "policy_id": "932404fbdfe81392",
      "replacement": "IGNORE_NEW",
      "resolved_threshold": 0.6483168101188825,
      "score_quantile": 0.99,
      "sleeve": 0.5,
      "top_fraction": 0.005
    }
  ],
  "policy_contract": "V5_SELECTED:H5/H10/H20; daily-top-score threshold; family/candidate provenance only",
  "policy_hash": "b3e75b5eebed2689fe0b34eff698bf53715c81aa8196838a6e41435f9771b3f8",
  "readiness_gates": {
    "accounting_invariants_pass": true,
    "all_readiness_gates_pass": false,
    "concentration_pass": false,
    "cost_tax_stress_pass": true,
    "cross_regime_context_matrix_complete": true,
    "development_execution_complete": true,
    "dynamic_exit_stage_complete": true,
    "exposure_invariants_pass": true,
    "horizon_execution_coverage": {
      "5": {
        "active_fold_fraction": 0.5,
        "active_folds": 3,
        "actual_outer_folds": 6,
        "dynamic_stage_complete": true,
        "evidence_coverage_complete": false,
        "execution_complete": true,
        "execution_failure_detected": false,
        "execution_failure_messages": [],
        "expected_outer_folds": 6,
        "final_policy_present": true,
        "frozen_threshold_present": true,
        "inactive_folds": 3,
        "nominal_grid_coverage_complete": true,
        "oos_trade_count": 19,
        "positive_active_fold_fraction": 0.6666666666666666,
        "sparse_outer_robustness_pass": false
      },
      "10": {
        "active_fold_fraction": 0.5,
        "active_folds": 3,
        "actual_outer_folds": 6,
        "dynamic_stage_complete": true,
        "evidence_coverage_complete": false,
        "execution_complete": true,
        "execution_failure_detected": false,
        "execution_failure_messages": [],
        "expected_outer_folds": 6,
        "final_policy_present": true,
        "frozen_threshold_present": true,
        "inactive_folds": 3,
        "nominal_grid_coverage_complete": true,
        "oos_trade_count": 41,
        "positive_active_fold_fraction": 1.0,
        "sparse_outer_robustness_pass": false
      },
      "20": {
        "active_fold_fraction": 0.3333333333333333,
        "active_folds": 2,
        "actual_outer_folds": 6,
        "dynamic_stage_complete": true,
        "evidence_coverage_complete": false,
        "execution_complete": true,
        "execution_failure_detected": false,
        "execution_failure_messages": [],
        "expected_outer_folds": 6,
        "final_policy_present": true,
        "frozen_threshold_present": true,
        "inactive_folds": 4,
        "nominal_grid_coverage_complete": true,
        "oos_trade_count": 37,
        "positive_active_fold_fraction": 0.0,
        "sparse_outer_robustness_pass": false
      }
    },
    "inactive_outer_folds_are_negative_evidence": false,
    "minimum_active_fold_fraction": 0.5,
    "minimum_active_outer_folds": 4,
    "minimum_oos_roundtrips": 20,
    "nominal_grid_coverage_complete": true,
    "outer_fold_robustness_pass": false,
    "parameter_plateau_pass": false,
    "search_coverage_complete": true,
    "sparse_evidence_coverage_complete": false
  },
  "ready_for_final_holdout": false,
  "status": "DEVELOPMENT_SUITE_COMPLETE",
  "usage": "FROZEN_DEVELOPMENT_INPUT_NOT_INDEPENDENT_OOS_RESULT"
}

Any previous artifacts generated under the all-stock threshold or incomplete Stage-A coverage contract are superseded and must not be used as evidence.

## Sparse-opportunity execution/evidence coverage

- BrokenProcessPool/child-process failures are execution failures, not negative alpha evidence.
- Zero-trade outer folds are inactive evidence, not negative folds.
- Readiness requires all expected horizons to finish, >=4 active folds and >=20 OOS roundtrips per horizon.
- Chained OOS fold return is diagnostic only and is not a readiness gate.
- Yearly trade counts are completed roundtrips by entry year.

{
  "10": "INTERESTING_BUT_SPARSE_AND_UNSTABLE",
  "20": "NO_ROBUST_AFTER_COST_ALPHA",
  "5": "INTERESTING_BUT_SPARSE_AND_UNSTABLE"
}
