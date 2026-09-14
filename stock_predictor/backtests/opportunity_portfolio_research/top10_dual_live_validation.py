"""Historical WF reproduction gate for the causal Top-10 live suite.

The companion expanding-live run is deliberately gated by this module.  It is
not valid to interpret newly fitted live models until the shared feature path
can reproduce every historical WF model on its original OOS interval.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .top10_external_validation import TOP10


CONTRACT_ID = "TOP10_HISTORICAL_REPRODUCTION_AND_CAUSAL_EXPANDING_LIVE_V1"
WF_PATTERN = re.compile(r"(WF_00[1-7])_(\d{4}-\d{2}-\d{2})_(\d{4}-\d{2}-\d{2})")
REQUIRED_FEATURE_END = pd.Timestamp("2026-07-24")
SCORE_TOLERANCE = 1e-12


SECURITY_SERIES = (
    {
        "security_id": "CHK_PRE_2021",
        "source_tickers": ("CHK", "EXE"),
        "valid_from": None,
        "valid_through": "2020-06-26",
        "series_break": "2020-06-29",
        "terminal_value": 0.0,
        "terminal_value_source": "FROZEN_ZERO_RECOVERY_NO_VERIFIED_DISTRIBUTION",
    },
    {
        "security_id": "CHK_POST_2021_EXE",
        "source_tickers": ("CHK", "EXE"),
        "valid_from": "2021-02-10",
        "valid_through": None,
        "ticker_change": "CHK_TO_EXE_2024-10-02",
        "terminal_value": None,
    },
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    tmp.replace(path)


def _manifest(panel_root: Path) -> dict:
    path = panel_root / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"PANEL_MANIFEST_MISSING:{path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _fold(path: Path) -> tuple[str, pd.Timestamp, pd.Timestamp]:
    match = WF_PATTERN.search(path.name)
    if not match:
        raise RuntimeError(f"WF_MODEL_FILENAME_INVALID:{path.name}")
    return match.group(1), pd.Timestamp(match.group(2)), pd.Timestamp(match.group(3))


def _one_model(model_path: Path, panel_path: Path, features: list[str], originals_path: Path,
               *, horizon_column: str, horizon: int, prediction_column: str) -> dict:
    fold_id, start, end = _fold(model_path)
    full_fold_id = f"{fold_id}_{start.date()}_{end.date()}"
    columns = ["decision_date", "ticker", *features]
    source = pd.read_parquet(
        panel_path,
        columns=columns,
        filters=[
            ("decision_date", ">=", start.date()),
            ("decision_date", "<=", end.date()),
        ],
    )
    original = pd.read_parquet(
        originals_path,
        columns=["decision_date", "ticker", "fold_id", horizon_column, prediction_column],
        filters=[(horizon_column, "=", horizon), ("fold_id", "=", full_fold_id)],
    )
    if source.empty or original.empty:
        raise RuntimeError(f"WF_REPRODUCTION_INPUT_EMPTY:{model_path.name}:{len(source)}:{len(original)}")
    model = joblib.load(model_path)
    reproduced = np.asarray(model.predict(source[features].to_numpy(dtype=float, copy=False)), dtype=float)
    source = source[["decision_date", "ticker"]].copy()
    source["reproduced"] = reproduced
    joined = source.merge(
        original[["decision_date", "ticker", prediction_column]],
        on=["decision_date", "ticker"],
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    counts = joined["_merge"].value_counts().to_dict()
    # Exit panels can contain right-edge feature rows whose future target was
    # not yet observable and which were therefore absent from the historical
    # OOS prediction artifact.  Missing historical keys remain a hard error;
    # additional causal feature rows are explicitly diagnostic.
    if int(counts.get("right_only", 0)):
        raise RuntimeError(f"WF_REPRODUCTION_KEY_MISMATCH:{model_path.name}:{counts}")
    joined = joined.loc[joined["_merge"].eq("both")].copy()
    expected = joined[prediction_column].to_numpy(dtype=float)
    observed = joined["reproduced"].to_numpy(dtype=float)
    delta = np.abs(expected - observed)
    finite = np.isfinite(expected) & np.isfinite(observed)
    if not finite.all():
        raise RuntimeError(f"WF_REPRODUCTION_NONFINITE:{model_path.name}:{int((~finite).sum())}")
    max_abs = float(delta.max(initial=0.0))
    # Quantile crossings prove that score-scale equality also holds where the
    # portfolio gate consumes it, not merely in a global RMSE statistic.
    crossings = {}
    frame = joined.assign(date=pd.to_datetime(joined["decision_date"]).dt.normalize())
    expected_tops = frame.groupby("date")[prediction_column].max().to_numpy(dtype=float)
    observed_tops = frame.groupby("date")["reproduced"].max().to_numpy(dtype=float)
    for quantile in (0.90, 0.95, 0.975, 0.99):
        expected_threshold = float(np.quantile(expected_tops, quantile))
        observed_threshold = float(np.quantile(observed_tops, quantile))
        expected_pass = expected >= expected_threshold
        observed_pass = observed >= observed_threshold
        crossings[str(quantile)] = {
            "expected_threshold": expected_threshold,
            "reproduced_threshold": observed_threshold,
            "threshold_abs_delta": abs(expected_threshold - observed_threshold),
            "crossing_mismatches": int(np.count_nonzero(expected_pass != observed_pass)),
        }
    sample = pd.read_parquet(
        panel_path,
        columns=["decision_date", "ticker", *features],
        filters=[
            ("decision_date", ">=", start.date()),
            ("decision_date", "<=", min(end, start + pd.Timedelta(days=10)).date()),
        ],
    ).sort_values(["decision_date", "ticker"]).head(256)
    feature_hash = hashlib.sha256(
        pd.util.hash_pandas_object(sample[["decision_date", "ticker", *features]], index=False)
        .to_numpy(dtype=np.uint64).tobytes()
    ).hexdigest()
    passed = max_abs <= SCORE_TOLERANCE and all(x["crossing_mismatches"] == 0 for x in crossings.values())
    return {
        "fold_id": full_fold_id,
        "horizon": horizon,
        "model_path": str(model_path),
        "rows": int(len(joined)),
        "causal_feature_rows_without_historical_prediction": int(counts.get("left_only", 0)),
        "max_abs_score_delta": max_abs,
        "mean_abs_score_delta": float(delta.mean()),
        "feature_sample_sha256": feature_hash,
        "threshold_crossings": crossings,
        "status": "EXACT_REPRODUCTION" if passed else "REPRODUCTION_MISMATCH",
    }


def _models(root: Path, pattern: str) -> list[Path]:
    paths = sorted(root.glob(pattern), key=lambda path: _fold(path)[0])
    folds = [_fold(path)[0] for path in paths]
    if folds != [f"WF_{index:03d}" for index in range(1, 8)]:
        raise RuntimeError(f"WF_001_007_MODEL_SET_INCOMPLETE:{pattern}:{folds}")
    return paths


def feature_coverage_gate(panel_root: Path) -> dict:
    rows = []
    for inference_name, training_name in (
        ("signal-inference-panel.parquet", "signal-panel.parquet"),
        ("stop-execution-inference-panel.parquet", "stop-execution-panel.parquet"),
    ):
        inference_path = panel_root / inference_name
        path = inference_path if inference_path.is_file() else panel_root / training_name
        values = pq.read_table(path, columns=["decision_date"]).column(0)
        maximum = pd.Timestamp(max(values.to_pylist()))
        rows.append({
            "path": str(path), "max_feature_date": str(maximum.date()),
            "inference_only_panel": path == inference_path,
        })
    observed = min(pd.Timestamp(row["max_feature_date"]) for row in rows)
    return {
        "required_feature_end": str(REQUIRED_FEATURE_END.date()),
        "observed_common_feature_end": str(observed.date()),
        "target_availability_must_not_truncate_features": True,
        "rows": rows,
        "status": "PASSED" if observed >= REQUIRED_FEATURE_END else "FAILED_FEATURES_END_BEFORE_MARKET_DATA",
    }


def run(args: argparse.Namespace) -> dict:
    panel_root = Path(args.panel_root)
    training_root = Path(args.training_root)
    output = Path(args.output_root)
    output.mkdir(parents=True, exist_ok=True)
    manifest = _manifest(panel_root)
    signal_features = list(manifest["signal"]["features"])
    exit_features = list(manifest["stop_execution"]["features"])
    required_entry_horizons = sorted({model.h for model in TOP10})
    entry_rows = []
    for horizon in required_entry_horizons:
        paths = _models(
            training_root / "signal" / "models",
            f"net_excess_return_{horizon}__BASELINE_20_BPS__*__WF_00[1-7]_*.joblib",
        )
        for path in paths:
            print(f"[historical-reproduction] entry H{horizon} {path.name}", flush=True)
            entry_rows.append(_one_model(
                path,
                panel_root / "signal-panel.parquet",
                signal_features,
                training_root / "signal" / "selected-walk-forward-predictions.parquet",
                horizon_column="horizon_sessions",
                horizon=horizon,
                prediction_column="predicted_net_excess_return",
            ))
    max_exit = max(model.d - 1 for model in TOP10 if model.mode == "LEARNED_EXIT")
    exit_root = training_root / "e1-30-learned-exit-20260809" / "exit"
    exit_rows = []
    for horizon in range(1, max_exit + 1):
        paths = _models(exit_root / "models", f"E{horizon:02d}__*__WF_00[1-7]_*.joblib")
        for path in paths:
            print(f"[historical-reproduction] exit E{horizon} {path.name}", flush=True)
            exit_rows.append(_one_model(
                path,
                panel_root / "stop-execution-panel.parquet",
                exit_features,
                exit_root / "selected-walk-forward-predictions.parquet",
                horizon_column="exit_horizon_sessions",
                horizon=horizon,
                prediction_column="predicted_continuation_excess",
            ))
    pd.DataFrame(entry_rows).drop(columns=["threshold_crossings"]).to_csv(output / "entry_wf_reproduction.csv", index=False)
    pd.DataFrame(exit_rows).drop(columns=["threshold_crossings"]).to_csv(output / "exit_wf_reproduction.csv", index=False)
    coverage = feature_coverage_gate(panel_root)
    reproduction_passed = all(row["status"] == "EXACT_REPRODUCTION" for row in entry_rows + exit_rows)
    status = "HISTORICAL_REPRODUCTION_PASSED_EXPANDING_LIVE_READY" if reproduction_passed and coverage["status"] == "PASSED" else "BLOCKED_BEFORE_EXPANDING_LIVE"
    summary = {
        "contract_id": CONTRACT_ID,
        "status": status,
        "historical_reproduction": {
            "status": "PASSED" if reproduction_passed else "FAILED",
            "entry_models": len(entry_rows),
            "exit_models": len(exit_rows),
            "score_tolerance": SCORE_TOLERANCE,
            "entry_rows": entry_rows,
            "exit_rows": exit_rows,
        },
        "feature_coverage": coverage,
        "security_series_contract": SECURITY_SERIES,
        "expanding_live": {
            "status": "READY_TO_RUN" if status.endswith("READY") else "NOT_RUN_FAIL_CLOSED",
            "reason": None if status.endswith("READY") else "Historical reproduction and feature-end gates must both pass",
            "daily_model_refit": True,
            "daily_model_specific_threshold_calibration": True,
            "labels_must_be_complete_at_decision_time": True,
            "policy_hyperparameters_frozen": True,
            "final_holdout_opened": False,
        },
    }
    _write_json(output / "dual_live_validation_summary.json", summary)
    return summary


def self_test() -> None:
    assert len(SECURITY_SERIES) == 2
    assert SECURITY_SERIES[0]["security_id"] != SECURITY_SERIES[1]["security_id"]
    assert SECURITY_SERIES[0]["valid_through"] == "2020-06-26"
    assert SECURITY_SERIES[1]["valid_from"] == "2021-02-10"
    assert REQUIRED_FEATURE_END == pd.Timestamp("2026-07-24")
    assert WF_PATTERN.search("x__WF_007_2023-02-09_2023-08-10.joblib")
    print("TOP10_DUAL_LIVE_VALIDATION_SELF_TEST_OK")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel-root")
    parser.add_argument("--training-root")
    parser.add_argument("--output-root", default="artifacts/top10-dual-live-validation")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return
    if not args.panel_root or not args.training_root:
        raise SystemExit("--panel-root and --training-root are required")
    summary = run(args)
    print(json.dumps({"contract_id": CONTRACT_ID, "status": summary["status"]}, indent=2))


if __name__ == "__main__":
    main()
