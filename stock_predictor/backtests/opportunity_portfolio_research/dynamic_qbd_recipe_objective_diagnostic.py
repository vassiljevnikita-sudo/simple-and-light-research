"""Audit whether the available recipe evidence measures the traded objective.

This module is deliberately diagnostic-only.  The historical H1--H30 artifact
contains only the selected candidate per walk-forward fold, so it can describe
the selected-series relationship between rank skill and tail outcomes, but it
cannot compare competing candidates.  That distinction is emitted as a hard
status rather than silently turning an incomplete artifact into selection
evidence.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


STATUS = "COMPARATIVE_RECIPE_TEST_BLOCKED_CANDIDATE_LEVEL_OOS_SCORES_UNAVAILABLE"


def _fold_end(value: str) -> str:
    return str(value).rsplit("_", 1)[-1]


def run(*, predictions_path: str | Path, candidate_metrics_path: str | Path,
        output_root: str | Path, horizon: int = 3) -> dict:
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    metrics = json.loads(Path(candidate_metrics_path).read_text(encoding="utf-8"))
    metric_rows = [row for row in metrics if int(row.get("horizon_sessions", -1)) == horizon
                   and not bool(row.get("selection_only", False))]
    metric_by_key = {(str(row["fold_id"]), str(row["candidate_id"])): row for row in metric_rows}
    columns = ["decision_date", "ticker", "fold_id", "candidate_id", "family",
               "predicted_net_excess_return", f"net_excess_return_{horizon}__BASELINE_20_BPS"]
    selected = pd.read_parquet(predictions_path, filters=[("horizon_sessions", "=", horizon)], columns=columns)
    selected["decision_date"] = pd.to_datetime(selected["decision_date"]).dt.normalize()
    target = f"net_excess_return_{horizon}__BASELINE_20_BPS"
    rows: list[dict] = []
    for (fold_id, candidate_id), frame in selected.groupby(["fold_id", "candidate_id"], sort=True):
        evidence = metric_by_key.get((str(fold_id), str(candidate_id)))
        if evidence is None:
            raise ValueError(f"OBJECTIVE_AUDIT_METRIC_MISSING:{fold_id}:{candidate_id}")
        winners = frame.sort_values(["decision_date", "predicted_net_excess_return", "ticker"],
                                    ascending=[True, False, True]).groupby("decision_date", sort=True).head(1)
        daily = winners.groupby("decision_date", sort=True)[target].mean()
        rows.append({
            "fold_id": str(fold_id), "fold_end": _fold_end(str(fold_id)),
            "selected_candidate_id": str(candidate_id), "selected_family": str(frame["family"].iloc[0]),
            "fold_spearman": float(evidence["metrics"]["spearman"]),
            "fold_r2_diagnostic": float(evidence["metrics"].get("r2", np.nan)),
            "top_score_active_dates": int(len(daily)),
            "top_score_mean_net_excess": float(daily.mean()),
            "top_score_median_net_excess": float(daily.median()),
            "top_score_positive_fraction": float((daily > 0).mean()),
            "top_score_compounded_excess_proxy": float(np.prod(1.0 + daily.to_numpy(float)) - 1.0),
        })
    fold_frame = pd.DataFrame(rows).sort_values("fold_end").reset_index(drop=True)
    fold_frame.to_csv(output / "selected-fold-objective-evidence.csv", index=False)
    correlations = []
    for outcome in ("top_score_mean_net_excess", "top_score_median_net_excess",
                    "top_score_compounded_excess_proxy"):
        correlations.append({"selector_metric": "fold_spearman", "outcome": outcome,
                             "pearson_correlation": float(fold_frame["fold_spearman"].corr(fold_frame[outcome])),
                             "observation_unit": "SELECTED_WALK_FORWARD_FOLD",
                             "interpretation": "DESCRIPTIVE_ONLY_NOT_CANDIDATE_COMPARATIVE"})
    pd.DataFrame(correlations).to_csv(output / "selected-fold-objective-correlations.csv", index=False)
    payload = {
        "schema_version": "DYNAMIC_QBD_RECIPE_OBJECTIVE_AUDIT_V1",
        "status": STATUS,
        "authority": "DIAGNOSTIC_ONLY_NO_RECIPE_SELECTION_NO_PROMOTION_NO_HOLDOUT",
        "horizon_sessions": horizon,
        "selected_fold_count": int(len(fold_frame)),
        "available_prediction_contract": "ONE_SELECTED_CANDIDATE_PER_FOLD",
        "missing_contract": "ALL_CANDIDATES_X_FOLDS_OOS_SCORES_WITH_CAUSAL_TAIL_AND_PORTFOLIO_REPLAYS",
        "required_before_selector_change": [
            "store candidate-level OOS scores for every candidate and fold",
            "pre-register top-tail and stateful QBD portfolio objectives",
            "compare candidates by fold/time blocks, not correlated QBD cells",
        ],
        "correlations": correlations,
    }
    (output / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--candidate-metrics", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--horizon", type=int, default=3)
    args = parser.parse_args(argv)
    print(json.dumps(run(predictions_path=args.predictions, candidate_metrics_path=args.candidate_metrics,
                         output_root=args.output_root, horizon=args.horizon), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
