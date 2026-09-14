from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

import pandas as pd

from . import top10_external_validation as core
from .portfolio_research_inputs import load_predictions
from .learned_exit_qbd_provider import LearnedExitProvider

PREDICTION_PROVENANCE_CONTRACT = "TOP10_EXTERNAL_CANONICAL_ARTIFACTS_V2"
CANONICAL_V5 = Path("training/signal/selected-walk-forward-predictions.parquet")
CANONICAL_LEARNED_EXIT = Path(
    "training/e1-30-learned-exit-20260809/exit/selected-walk-forward-predictions.parquet"
)
CANONICAL_STOP_EXECUTION = Path("training/stop-execution/selected-walk-forward-predictions.parquet")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _path_has_suffix(path: Path, suffix: Path) -> bool:
    parts = tuple(str(x).lower() for x in path.parts)
    wanted = tuple(str(x).lower() for x in suffix.parts)
    return len(parts) >= len(wanted) and parts[-len(wanted):] == wanted


def _assert_canonical_path(path: Path, suffix: Path, *, layer: str) -> None:
    if not path.is_file():
        raise RuntimeError(f"CANONICAL_ARTIFACT_MISSING:{layer}:{path}")
    if not _path_has_suffix(path, suffix):
        raise RuntimeError(
            f"CANONICAL_ARTIFACT_PATH_MISMATCH:{layer}:{path}:expected_suffix={suffix}"
        )


def _resolve_artifacts(a: argparse.Namespace) -> tuple[Path, Path, Path]:
    root = Path(a.artifact_root) if a.artifact_root else None

    def pick(explicit: str | None, suffix: Path, layer: str) -> Path:
        if explicit:
            path = Path(explicit)
        elif root is not None:
            path = root / suffix
        else:
            raise RuntimeError(f"ARTIFACT_ROOT_OR_EXPLICIT_PATH_REQUIRED:{layer}")
        _assert_canonical_path(path, suffix, layer=layer)
        return path

    return (
        pick(a.v5_predictions, CANONICAL_V5, "v5"),
        pick(a.learned_exit_predictions, CANONICAL_LEARNED_EXIT, "learned_exit"),
        pick(a.stop_execution_predictions, CANONICAL_STOP_EXECUTION, "stop_execution"),
    )


def _normalize_dates(frame: pd.DataFrame) -> pd.Series:
    dates = pd.to_datetime(frame["decision_date"])
    if getattr(dates.dt, "tz", None) is not None:
        dates = dates.dt.tz_convert(None)
    return dates


def _fold_table(pred: pd.DataFrame) -> pd.DataFrame:
    if pred.empty:
        raise RuntimeError("CANONICAL_V5_EMPTY")
    x = pred[["fold_id", "decision_date"]].copy()
    x["decision_date"] = _normalize_dates(x)
    folds = (
        x.groupby("fold_id", as_index=False)["decision_date"]
        .agg(fold_start="min", fold_end="max")
        .sort_values(["fold_start", "fold_end", "fold_id"])
        .reset_index(drop=True)
    )
    return folds


def _horizon_coverage(frame: pd.DataFrame, *, segment: str, evidence_mask: pd.Series) -> dict:
    needed = sorted({m.h for m in core.TOP10})
    observed = sorted({int(x) for x in frame.loc[evidence_mask, "horizon"].unique()})
    missing = sorted(set(needed) - set(observed))
    if missing:
        raise RuntimeError(f"{segment}_V5_HORIZONS_MISSING:{missing}")
    return {"required_horizons": needed, "observed_horizons": observed}


def _select_v5_segments(pred: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    pred = pred.copy()
    pred["decision_date"] = _normalize_dates(pred)
    if pred["decision_date"].max().date() > core.FORWARD_END.date():
        raise RuntimeError(
            f"CANONICAL_V5_AFTER_RAW_MAX:{pred['decision_date'].max()}>{core.FORWARD_END.date()}"
        )

    folds = _fold_table(pred)
    back_ids = set(
        folds.loc[folds["fold_end"].dt.date <= core.BACKWARD_END.date(), "fold_id"].astype(str)
    )
    forward_evidence_ids = set(
        folds.loc[
            (folds["fold_start"].dt.date >= core.FORWARD_START.date())
            & (folds["fold_end"].dt.date <= core.FORWARD_END.date()),
            "fold_id",
        ].astype(str)
    )
    # Forward replay keeps complete earlier folds as causal calibration history, but
    # never keeps a fold that straddles the Known/Forward boundary.
    forward_history_ids = set(
        folds.loc[folds["fold_end"].dt.date <= core.KNOWN_END.date(), "fold_id"].astype(str)
    )
    forward_ids = forward_history_ids | forward_evidence_ids

    if not back_ids:
        raise RuntimeError("NO_BACKWARD_OOS_FOLDS_IN_CANONICAL_V5")
    if not forward_evidence_ids:
        raise RuntimeError("NO_FORWARD_OOS_FOLDS_IN_CANONICAL_V5")

    fold_id = pred["fold_id"].astype(str)
    backward = pred.loc[fold_id.isin(back_ids)].copy()
    forward = pred.loc[fold_id.isin(forward_ids)].copy()

    back_evidence = backward["decision_date"].dt.date <= core.BACKWARD_END.date()
    fwd_evidence = (
        (forward["decision_date"].dt.date >= core.FORWARD_START.date())
        & (forward["decision_date"].dt.date <= core.FORWARD_END.date())
    )
    back_h = _horizon_coverage(backward, segment="BACKWARD", evidence_mask=back_evidence)
    fwd_h = _horizon_coverage(forward, segment="FORWARD", evidence_mask=fwd_evidence)

    excluded_straddlers = folds.loc[
        (folds["fold_start"].dt.date < core.FORWARD_START.date())
        & (folds["fold_end"].dt.date >= core.FORWARD_START.date()),
        "fold_id",
    ].astype(str).tolist()

    audit = {
        "source_rows": int(len(pred)),
        "source_min_decision_date": str(pred["decision_date"].min().date()),
        "source_max_decision_date": str(pred["decision_date"].max().date()),
        "source_fold_count": int(len(folds)),
        "backward_fold_ids": sorted(back_ids),
        "backward_fold_count": len(back_ids),
        "backward_rows": int(len(backward)),
        "forward_history_fold_ids": sorted(forward_history_ids),
        "forward_evidence_fold_ids": sorted(forward_evidence_ids),
        "forward_evidence_fold_count": len(forward_evidence_ids),
        "forward_rows_with_history": int(len(forward)),
        "excluded_known_forward_straddling_folds": excluded_straddlers,
        "backward_horizon_coverage": back_h,
        "forward_horizon_coverage": fwd_h,
        "segments_derived_from_single_selected_walk_forward_artifact": True,
        "no_prediction_extrapolation": True,
    }
    return backward, forward, audit


def _audit_v5(path: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    pred, loader_audit = load_predictions(path)
    backward, forward, segment_audit = _select_v5_segments(pred)
    raw = pd.read_parquet(path)
    locked_rows = 0
    if "holdout_locked" in raw.columns:
        locked_rows = int(raw["holdout_locked"].fillna(False).astype(bool).sum())
        if locked_rows:
            raise RuntimeError(f"V5_FROZEN_VALIDATION_ROWS_FORBIDDEN:{locked_rows}")
    audit = {
        "path": str(path),
        "sha256": _sha256(path),
        "canonical_suffix": str(CANONICAL_V5),
        "selected_walk_forward_artifact": True,
        "loader_audit": loader_audit,
        "locked_rows": locked_rows,
        "segments": segment_audit,
    }
    return backward, forward, audit


def _audit_learned_exit(path: Path) -> dict:
    provider = LearnedExitProvider(path)
    frame = pd.read_parquet(path, columns=["decision_date", "exit_horizon_sessions"])
    frame["decision_date"] = pd.to_datetime(frame["decision_date"])
    if getattr(frame["decision_date"].dt, "tz", None) is not None:
        frame["decision_date"] = frame["decision_date"].dt.tz_convert(None)

    required_exit_h = set(range(1, max(m.d for m in core.TOP10 if m.mode == "LEARNED_EXIT")))
    observed_exit_h = set(int(x) for x in frame["exit_horizon_sessions"].unique())
    missing_exit_h = sorted(required_exit_h - observed_exit_h)
    if missing_exit_h:
        raise RuntimeError(f"LEARNED_EXIT_HORIZONS_MISSING:{missing_exit_h}")

    backward_rows = int((frame["decision_date"].dt.date <= core.BACKWARD_END.date()).sum())
    forward_rows = int(
        (
            (frame["decision_date"].dt.date >= core.FORWARD_START.date())
            & (frame["decision_date"].dt.date <= core.FORWARD_END.date())
        ).sum()
    )
    if backward_rows == 0:
        raise RuntimeError("LEARNED_EXIT_BACKWARD_COVERAGE_MISSING")
    if forward_rows == 0:
        raise RuntimeError("LEARNED_EXIT_FORWARD_COVERAGE_MISSING")
    if frame["decision_date"].max().date() > core.FORWARD_END.date():
        raise RuntimeError(f"LEARNED_EXIT_AFTER_RAW_MAX:{frame['decision_date'].max()}")

    return {
        "path": str(path),
        "sha256": _sha256(path),
        "canonical_suffix": str(CANONICAL_LEARNED_EXIT),
        "selected_walk_forward_artifact": True,
        "provider_audit": provider.audit.__dict__,
        "required_exit_horizons": sorted(required_exit_h),
        "backward_rows": backward_rows,
        "forward_rows": forward_rows,
        "no_prediction_extrapolation": True,
    }


def _audit_stop_execution(path: Path) -> dict:
    frame = pd.read_parquet(path)
    audit = {
        "path": str(path),
        "sha256": _sha256(path),
        "canonical_suffix": str(CANONICAL_STOP_EXECUTION),
        "rows": int(len(frame)),
        "columns": [str(x) for x in frame.columns],
        "economic_input_to_top10_profit_replay": False,
        "role": "canonical companion provenance/hash artifact",
    }
    if "decision_date" in frame.columns and len(frame):
        dates = pd.to_datetime(frame["decision_date"])
        if getattr(dates.dt, "tz", None) is not None:
            dates = dates.dt.tz_convert(None)
        audit["min_decision_date"] = str(dates.min().date())
        audit["max_decision_date"] = str(dates.max().date())
    return audit


def validate_canonical_artifacts(a: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    v5_path, learned_path, stop_path = _resolve_artifacts(a)
    backward, forward, v5_audit = _audit_v5(v5_path)
    learned_audit = _audit_learned_exit(learned_path)
    stop_audit = _audit_stop_execution(stop_path)
    provenance = {
        "contract_id": PREDICTION_PROVENANCE_CONTRACT,
        "artifact_root": str(a.artifact_root) if a.artifact_root else None,
        "v5": v5_audit,
        "learned_exit": learned_audit,
        "stop_execution": stop_audit,
        "artificial_backward_forward_prediction_files_required": False,
        "external_training_manifests_required": False,
        "selected_walk_forward_contract_verified": True,
        "portfolio_fold_causality_still_enforced": "calibration_end < fold_start",
        "independent_train_end_manifest_proof": False,
        "independent_train_end_manifest_note": (
            "No fabricated external manifest is required. Model-training causality is inherited from the canonical "
            "selected-walk-forward artifacts; the files themselves do not expose a universal train_end field."
        ),
    }
    return backward, forward, provenance


def _write_segment_parquets(backward: pd.DataFrame, forward: pd.DataFrame, root: Path) -> tuple[Path, Path]:
    back_path = root / "backward-v5-selected-wf.parquet"
    forward_path = root / "forward-v5-with-history-selected-wf.parquet"
    backward.to_parquet(back_path, index=False)
    forward.to_parquet(forward_path, index=False)
    return back_path, forward_path


def self_test() -> None:
    core.self_test()
    rows = []
    fold_specs = [
        ("B0", "2018-01-01", "2018-03-31"),
        ("B1", "2019-01-01", "2019-03-31"),
        ("B2", "2020-01-01", "2020-03-31"),
        ("K0", "2021-01-01", "2021-03-31"),
        ("S0", "2023-08-01", "2023-08-20"),
        ("F0", "2024-01-01", "2024-03-31"),
    ]
    for fold_id, start, end in fold_specs:
        for horizon in sorted({m.h for m in core.TOP10}):
            rows.extend(
                [
                    {"fold_id": fold_id, "decision_date": pd.Timestamp(start), "horizon": horizon},
                    {"fold_id": fold_id, "decision_date": pd.Timestamp(end), "horizon": horizon},
                ]
            )
    pred = pd.DataFrame(rows)
    backward, forward, audit = _select_v5_segments(pred)
    assert set(backward["fold_id"]) == {"B0", "B1", "B2"}
    assert "F0" in set(forward["fold_id"])
    assert "S0" not in set(forward["fold_id"])
    assert audit["forward_evidence_fold_ids"] == ["F0"]

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        good = root / CANONICAL_V5
        good.parent.mkdir(parents=True, exist_ok=True)
        good.touch()
        _assert_canonical_path(good, CANONICAL_V5, layer="v5")
        bad = root / "training" / "signal" / "predictions.parquet"
        bad.touch()
        try:
            _assert_canonical_path(bad, CANONICAL_V5, layer="v5")
        except RuntimeError as exc:
            assert "PATH_MISMATCH" in str(exc)
        else:
            raise AssertionError("non-canonical artifact path was accepted")
    print("TOP10_EXTERNAL_VALIDATION_STRICT_SELF_TEST_OK")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Strict frozen Top-10 backward/forward external validation")
    p.add_argument("--raw-minute-root", required=False)
    p.add_argument("--artifact-root", required=False)
    p.add_argument("--v5-predictions", required=False)
    p.add_argument("--learned-exit-predictions", required=False)
    p.add_argument("--stop-execution-predictions", required=False)
    p.add_argument("--known-profit-root", required=False)
    p.add_argument("--daily-store-root", required=False)
    p.add_argument("--output-root", default="artifacts/top10-external-validation")
    p.add_argument("--entry-budget", type=int, default=12)
    p.add_argument("--max-workers", type=int, default=8)
    p.add_argument("--initial-capital", type=float, default=core.INITIAL_CAPITAL_EUR)
    p.add_argument("--tax-allowance-eur", type=float, default=1000.0)
    p.add_argument("--church-tax-rate", type=float, default=0.0)
    p.add_argument("--benchmark-partial-exemption", type=float, default=0.30)
    p.add_argument("--self-test", action="store_true")
    return p.parse_args()


def main() -> None:
    a = parse_args()
    if a.self_test:
        self_test()
        return
    required = ("raw_minute_root", "known_profit_root", "daily_store_root")
    missing = [x for x in required if not getattr(a, x)]
    if missing:
        raise SystemExit("missing required args: " + ", ".join("--" + x.replace("_", "-") for x in missing))
    if not a.artifact_root and not (a.v5_predictions and a.learned_exit_predictions and a.stop_execution_predictions):
        raise SystemExit(
            "provide --artifact-root or all three canonical artifact paths: "
            "--v5-predictions, --learned-exit-predictions, --stop-execution-predictions"
        )

    backward, forward, provenance = validate_canonical_artifacts(a)
    out = Path(a.output_root)
    out.mkdir(parents=True, exist_ok=True)
    core._write_json(out / "prediction_provenance_audit.json", provenance)

    _, learned_path, _ = _resolve_artifacts(a)
    with tempfile.TemporaryDirectory(prefix="top10-external-v5-") as td:
        back_path, forward_path = _write_segment_parquets(backward, forward, Path(td))
        core_args = argparse.Namespace(
            raw_minute_root=a.raw_minute_root,
            backward_v5_predictions=str(back_path),
            backward_learned_exit_predictions=str(learned_path),
            forward_v5_predictions=str(forward_path),
            forward_learned_exit_predictions=str(learned_path),
            known_profit_root=a.known_profit_root,
            daily_store_root=a.daily_store_root,
            output_root=a.output_root,
            entry_budget=a.entry_budget,
            max_workers=a.max_workers,
            initial_capital=a.initial_capital,
            tax_allowance_eur=a.tax_allowance_eur,
            church_tax_rate=a.church_tax_rate,
            benchmark_partial_exemption=a.benchmark_partial_exemption,
        )
        result = core.run(core_args)

    result["prediction_provenance"] = provenance
    result["canonical_selected_walk_forward_contract_verified"] = True
    result["artificial_external_artifacts_required"] = False
    core._write_json(out / "evaluation_summary.json", result)
    print(
        json.dumps(
            {
                "contract_id": result["contract_id"],
                "status": result["status"],
                "canonical_selected_walk_forward_contract_verified": True,
                "output_root": a.output_root,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
