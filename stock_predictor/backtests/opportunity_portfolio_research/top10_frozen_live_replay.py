"""Frozen-model Top-10 live replay.

This is deliberately not a walk-forward training suite.  It loads already
trained weights, projects only causal feature columns, obtains predictions in
chronological order and replays the already frozen entry policies unchanged.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import re
import tempfile
from bisect import bisect_right
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from . import portfolio_policy_search as search
from .cost_contracts import CostModel
from .portfolio_policy_contracts import Policy
from .tax_contracts import TaxConfig
from .portfolio_research_inputs import load_price_panel
from .learned_exit_qbd_profit import ProfitTaxConfig
from .learned_exit_qbd_provider import LearnedExitProvider
from .learned_exit_qbd_replay import configure_replay, replay
from . import top10_external_validation as known


CONTRACT_ID = "TOP10_FROZEN_MODEL_LIVE_REPLAY_V1"
REPLAY_START = pd.Timestamp("2016-01-01")
# This is the originally requested historical start.  The exact frozen system
# may only begin once the final learned-exit training labels are observable.
REQUESTED_HISTORICAL_FORWARD_START = pd.Timestamp("2023-08-11")
REPLAY_END = pd.Timestamp("2026-07-24")
DELISTING_RULES = {
    "EXE": {
        "security_id": "CHK_PRE_2021",
        "ticker_at_entry": "CHK",
        "series_end": "2020-06-26",
        "exit_reason": "DELISTING_ZERO_RECOVERY_ASSUMPTION",
        "terminal_value": 0.0,
        "terminal_value_source": "FROZEN_ZERO_RECOVERY_NO_VERIFIED_DISTRIBUTION",
    }
}
SIGNAL_FEATURE_COUNT = 31
EXIT_FEATURE_COUNT = 25

# These are the exact binary artifacts selected before this replay.  A matching
# filename is insufficient: a replacement model must fail closed.
EXPECTED_SIGNAL_SHA256 = {
    11: "985d1fe3f137a9e6b3c117826115fdc83ccf1359215e87af99350d9e34e30de1",
    24: "091ad389be1b06be4a99f91e734d9694747c9db858d35f04677b43057eaa77b5",
    28: "2300578dac95954118fcb08696c62fb0e7c451d14d0a7e7ab254e4bca22f89a4",
}
EXPECTED_EXIT_SHA256 = {
    1: "95d1c6fb42c768847adf9500397afeeca8fa8eb3475ab05c252e66efbf5f2968",
    2: "cb15e248345c4bed5e8480080d5f6514358ab5144b6adfe19059dcb64510f77f",
    3: "74a2649dccbac33d94379d2ca4e29082d29249c3b847d11f45aaec07fc1ed6c7",
    4: "c44228e4bd6cb810a6bdc7680fae72d148485e25d8570bf562b1771871b86742",
    5: "fc2458a2030eb1162745a5bb75dc263f30ff19d27517f36f0c895fd8db6f4201",
    6: "a295877112cc39d89b0b7d892f7bacde478cf30e0df073d7904b0e3c9b80cc7e",
    7: "bbf1c0c9415778128effbb46accd26c9f1cdd84bd1464d62f7740d9dc978bb35",
    8: "0c29434a9629e7f5330e81e48c098e1611ac104ba88366a437c66fe227f8d1d7",
    9: "f164d0fb7cba58d173ca7a425997e49a1017d4b1ccffdef1c460c03c2317eb33",
    10: "01edf559bfd22ad17fe7cef2e6648274af9ffc8357c7f81dcd9d8c9d306169d7",
    11: "3b4387df2bb649c4c39ac1c37b1a26437a4bec5209c3cb9073bab47eb5375f56",
    12: "0f14e9e15e3673a4f167acf10216f2101cd97e6c2597067935dd55a63ccaf026",
    13: "b8279a9e7843fa2bc304952d7e8e983c6355cf011b80088a984f37bb7d9966c7",
    14: "e750684430f977c3ca9d3b56c60c2d4086f4a82c51cb7902ae29bb9b3563dbc6",
    15: "13d6ff5ddd256d23ac75fea63a795045b01cdbcee9e112b1fa78b2bfcd512383",
    16: "83a583648df1a9f1320fd17d4e51f16fe550e25dcba21ec191caeab38ab11cb7",
    17: "e26f8671303fbcd8bdc813bdb8cd0a68fe662fdf189586f330f10ecd327beee4",
    18: "c59b207a513bc9a5f19e9884dadba8cc497692d7e0f25dc1c6ddc855c40163f6",
    19: "2ef91086666e30347ba3640ecc1887a1c476953bb6a9c20b33d972030f7cb39f",
    20: "51eb4de95beefe3ceadd80e06d06a19dd7309aa9c958878e9bfdbc754d601ad2",
}
EXPECTED_FEATURE_BUILDER_SHA256 = "6ad49e90ed1120c17140c240c29b1914a34bf2cf3d55539e08471b4016a5c0e6"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    temp.replace(path)


def _read_manifest(panel_root: Path) -> dict:
    path = panel_root / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"FROZEN_LIVE_PANEL_MANIFEST_MISSING:{path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _last_wf_model(model_root: Path, horizon: int) -> Path:
    matches = sorted(model_root.glob(f"net_excess_return_{horizon}__BASELINE_20_BPS__*__WF_007_*.joblib"))
    if len(matches) != 1:
        raise RuntimeError(f"FROZEN_SIGNAL_MODEL_AMBIGUOUS:H{horizon}:{[p.name for p in matches]}")
    return matches[0]


def _exit_models(exit_root: Path, maximum: int) -> dict[int, Path]:
    models: dict[int, Path] = {}
    for horizon in range(1, maximum + 1):
        matches = sorted((exit_root / "exit" / "models").glob(f"E{horizon:02d}__*__DEVELOPMENT_FULL.joblib"))
        if len(matches) != 1:
            raise RuntimeError(f"FROZEN_EXIT_MODEL_AMBIGUOUS:E{horizon}:{[p.name for p in matches]}")
        models[horizon] = matches[0]
    return models


def _locked_artifact(path: Path, *, expected_sha256: str, label: str) -> dict:
    observed = _sha256(path).lower()
    if observed != expected_sha256.lower():
        raise RuntimeError(
            f"FROZEN_ARTIFACT_SHA256_MISMATCH:{label}:expected={expected_sha256}:observed={observed}"
        )
    return {"path": str(path), "sha256": observed, "expected_sha256": expected_sha256, "verified": True}


def frozen_model_manifest(training_root: Path, panel_root: Path) -> dict:
    manifest = _read_manifest(panel_root)
    signal_features = list(manifest.get("signal", {}).get("features", []))
    exit_features = list(manifest.get("stop_execution", {}).get("features", []))
    if len(signal_features) != SIGNAL_FEATURE_COUNT:
        raise RuntimeError(f"FROZEN_SIGNAL_FEATURE_SCHEMA_MISMATCH:{len(signal_features)}")
    if len(exit_features) != EXIT_FEATURE_COUNT:
        raise RuntimeError(f"FROZEN_EXIT_FEATURE_SCHEMA_MISMATCH:{len(exit_features)}")
    required_h = sorted({m.h for m in known.TOP10})
    signal_root = training_root / "signal" / "models"
    signal = {h: _last_wf_model(signal_root, h) for h in required_h}
    max_exit = max(m.d - 1 for m in known.TOP10 if m.mode == "LEARNED_EXIT")
    exit_root = training_root / "e1-30-learned-exit-20260809"
    exit_models = _exit_models(exit_root, max_exit)
    signal_lock = {h: _locked_artifact(path, expected_sha256=EXPECTED_SIGNAL_SHA256[h], label=f"H{h}") for h, path in signal.items()}
    exit_lock = {h: _locked_artifact(path, expected_sha256=EXPECTED_EXIT_SHA256[h], label=f"E{h}") for h, path in exit_models.items()}
    return {
        "contract_id": CONTRACT_ID,
        "model_usage": "FROZEN_INFERENCE_ONLY_NO_FIT_NO_RETRAINING",
        "requested_historical_forward_start": str(REQUESTED_HISTORICAL_FORWARD_START.date()),
        "effective_true_forward_start": "DERIVED_FROM_EXIT_TRAINING_LABEL_CUTOFF",
        "signal_panel": str(panel_root / "signal-panel.parquet"),
        "stop_execution_panel": str(panel_root / "stop-execution-panel.parquet"),
        "signal_features": signal_features,
        "exit_features": exit_features,
        "signal_models": {
            str(h): {
                **signal_lock[h],
                "freeze_source": "LATEST_AVAILABLE_ALREADY_TRAINED_WF_007",
                "development_full_artifact_available": False,
                "frozen_wf_test_window": "2023-02-09..2023-08-10",
                "training_data_no_later_than": "2023-02-08",
            }
            for h, path in signal.items()
        },
        "exit_models": {
            str(h): {**exit_lock[h], "kind": "DEVELOPMENT_FULL"}
            for h, path in exit_models.items()
        },
        "feature_projection_only": True,
        "target_columns_read": False,
        "final_holdout_opened": False,
    }


def exit_development_training_cutoff_audit(training_root: Path, frozen: dict) -> dict:
    """Prove that each frozen DEVELOPMENT_FULL exit model predates deployment.

    A forward target is only available after its last horizon session.  Checking
    the decision date alone would therefore be insufficient.
    """
    root = training_root / "e1-30-learned-exit-20260809"
    summary_path = root / "learned-exit-training-summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"EXIT_TRAINING_SUMMARY_MISSING:{summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("frozen_validation_opened") is not False:
        raise RuntimeError("EXIT_TRAINING_FROZEN_VALIDATION_WAS_OPENED")
    signal_meta = summary.get("input_manifests", {}).get("primary", {}).get("signal", {})
    signal_path = Path(str(signal_meta.get("path", "")))
    if not signal_path.is_file():
        raise FileNotFoundError(f"EXIT_TRAINING_SIGNAL_PANEL_MISSING:{signal_path}")
    horizons = sorted(int(h) for h in frozen["exit_models"])
    targets = [f"gross_excess_return_{h}" for h in horizons]
    columns = ["decision_date", "holdout_locked", *targets]
    frame = pd.read_parquet(signal_path, columns=columns)
    frame["decision_date"] = pd.to_datetime(frame["decision_date"]).dt.normalize()
    sessions = sorted(pd.Timestamp(x) for x in frame["decision_date"].dropna().unique())
    session_index = {date: index for index, date in enumerate(sessions)}
    rows = []
    violations = []
    for horizon in horizons:
        selected = summary.get("selection", {}).get(str(horizon), {})
        expected_path = Path(str(frozen["exit_models"][str(horizon)]["path"])).resolve()
        observed_path = Path(str(selected.get("development_full_model", ""))).resolve()
        if observed_path != expected_path:
            raise RuntimeError(f"EXIT_DEVELOPMENT_FULL_MODEL_PROVENANCE_MISMATCH:E{horizon}")
        if selected.get("development_full_model_usage") != "FORWARD_SHADOW_ONLY_NOT_OOS_EVIDENCE":
            raise RuntimeError(f"EXIT_DEVELOPMENT_FULL_USAGE_CONTRACT_MISMATCH:E{horizon}")
        valid = (~frame["holdout_locked"].fillna(False).astype(bool)) & np.isfinite(pd.to_numeric(frame[f"gross_excess_return_{horizon}"], errors="coerce"))
        if not valid.any():
            raise RuntimeError(f"EXIT_DEVELOPMENT_FULL_TRAINING_ROWS_MISSING:E{horizon}")
        last_decision = pd.Timestamp(frame.loc[valid, "decision_date"].max())
        position = session_index.get(last_decision)
        completion = sessions[position + horizon] if position is not None and position + horizon < len(sessions) else None
        row = {
            "exit_horizon": horizon,
            "model_path": str(observed_path),
            "last_training_decision_date": str(last_decision.date()),
            "last_training_label_completion_date": None if completion is None else str(pd.Timestamp(completion).date()),
            "requested_historical_forward_start": str(REQUESTED_HISTORICAL_FORWARD_START.date()),
            "decision_before_requested_start": bool(last_decision < REQUESTED_HISTORICAL_FORWARD_START),
            "label_completion_before_requested_start": bool(completion is not None and pd.Timestamp(completion) < REQUESTED_HISTORICAL_FORWARD_START),
        }
        rows.append(row)
        if completion is None:
            violations.append(row)
    if violations:
        raise RuntimeError("EXIT_DEVELOPMENT_FULL_LABEL_COMPLETION_UNRESOLVED:" + json.dumps(violations, sort_keys=True))
    latest_completion = max(pd.Timestamp(row["last_training_label_completion_date"]) for row in rows)
    next_positions = [index for index, date in enumerate(sessions) if pd.Timestamp(date) > latest_completion]
    if not next_positions:
        raise RuntimeError(f"EXIT_DEVELOPMENT_FULL_NO_SESSION_AFTER_LABEL_CUTOFF:{latest_completion.date()}")
    effective_start = pd.Timestamp(sessions[next_positions[0]])
    if effective_start > REPLAY_END:
        raise RuntimeError(
            f"EXIT_DEVELOPMENT_FULL_EFFECTIVE_FORWARD_AFTER_REPLAY_END:{effective_start.date()}>{REPLAY_END.date()}"
        )
    audit = {
        "status": "PASSED_EXACT_TOP10_EFFECTIVE_FORWARD_START_DETERMINED",
        "training_summary": str(summary_path), "training_summary_sha256": _sha256(summary_path),
        "training_signal_panel": str(signal_path), "training_signal_panel_sha256": _sha256(signal_path),
        "frozen_validation_opened": False, "rows": rows,
        "requested_historical_forward_start": str(REQUESTED_HISTORICAL_FORWARD_START.date()),
        "latest_training_label_completion_date": str(latest_completion.date()),
        "effective_true_forward_start": str(effective_start.date()),
        "effective_start_selection": "NEXT_AVAILABLE_SESSION_AFTER_LATEST_REQUIRED_EXIT_LABEL_COMPLETION",
        "historical_2023_08_11_to_effective_start": "DIAGNOSTIC_ONLY_NOT_EXACT_TOP10_TRUE_OOS",
    }
    return audit


def _assert_panel_columns(path: Path, features: list[str]) -> None:
    columns = set(pq.ParquetFile(path).schema.names)
    missing = sorted(set(["decision_date", "ticker", "holdout_locked", *features]) - columns)
    if missing:
        raise RuntimeError(f"FROZEN_LIVE_PANEL_COLUMNS_MISSING:{path}:{missing}")


def feature_provenance_audit(panel_root: Path, builder_source: Path, frozen: dict) -> dict:
    """Fail closed on feature-code structures that can consume future sessions.

    This is a structural provenance check, deliberately distinct from the
    point-in-time-universe audit.  It validates the stored panels against their
    manifest and proves that the feature-producing functions do not contain a
    negative shift; label functions may, and must remain isolated.
    """
    if not builder_source.is_file():
        raise FileNotFoundError(f"FEATURE_PROVENANCE_SOURCE_MISSING:{builder_source}")
    builder_sha256 = _sha256(builder_source)
    if builder_sha256 != EXPECTED_FEATURE_BUILDER_SHA256:
        raise RuntimeError(
            "FEATURE_PROVENANCE_BUILDER_SHA256_MISMATCH:"
            f"expected={EXPECTED_FEATURE_BUILDER_SHA256}:observed={builder_sha256}"
        )
    manifest = _read_manifest(panel_root)
    source = builder_source.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(builder_source))
    functions = {node.name: ast.get_source_segment(source, node) or "" for node in tree.body if isinstance(node, ast.FunctionDef)}
    required = ("_daily_features", "_minute_daily", "_stop_panel", "_signal_labels", "_future_path_labels")
    missing = [name for name in required if not functions.get(name)]
    if missing:
        raise RuntimeError(f"FEATURE_PROVENANCE_FUNCTIONS_MISSING:{missing}")
    future_shift = re.compile(r"\.shift\(\s*-\s*\d+")
    feature_functions = ("_daily_features", "_minute_daily", "_stop_panel")
    leaking = [name for name in feature_functions if future_shift.search(functions[name])]
    if leaking:
        raise RuntimeError(f"FEATURE_PROVENANCE_NEGATIVE_SHIFT_IN_FEATURE_PATH:{leaking}")
    label_future_paths = [name for name in ("_signal_labels", "_future_path_labels") if future_shift.search(functions[name])]
    if set(label_future_paths) != {"_signal_labels"}:
        raise RuntimeError(f"FEATURE_PROVENANCE_LABEL_ISOLATION_UNEXPECTED:{label_future_paths}")
    signal_path = panel_root / "signal-panel.parquet"
    stop_path = panel_root / "stop-execution-panel.parquet"
    observed = {"signal": _sha256(signal_path), "stop_execution": _sha256(stop_path)}
    expected = {
        "signal": str(manifest.get("signal", {}).get("sha256", "")).lower(),
        "stop_execution": str(manifest.get("stop_execution", {}).get("sha256", "")).lower(),
    }
    if not all(expected.values()) or observed != expected:
        raise RuntimeError(f"FEATURE_PROVENANCE_PANEL_HASH_MISMATCH:expected={expected}:observed={observed}")
    if list(manifest.get("signal", {}).get("features", [])) != list(frozen["signal_features"]):
        raise RuntimeError("FEATURE_PROVENANCE_SIGNAL_SCHEMA_MISMATCH")
    if list(manifest.get("stop_execution", {}).get("features", [])) != list(frozen["exit_features"]):
        raise RuntimeError("FEATURE_PROVENANCE_EXIT_SCHEMA_MISMATCH")
    return {
        "status": "PASSED_STRUCTURAL_CAUSAL_FEATURE_PROVENANCE_AUDIT",
        "builder_source": str(builder_source), "builder_source_sha256": builder_sha256,
        "expected_builder_source_sha256": EXPECTED_FEATURE_BUILDER_SHA256,
        "builder_source_sha256_verified": True,
        "feature_functions_checked": list(feature_functions),
        "negative_shift_in_feature_path": False,
        "future_labels_isolated_from_inference_features": True,
        "panel_hashes_verified_against_manifest": True,
        "raw_prefix_rebuild_proof": False,
        "raw_prefix_rebuild_note": "Structural provenance audit; a separately reproducible prefix rebuild remains the stronger optional proof.",
    }


def _scan_batches(path: Path, columns: list[str], batch_size: int = 100_000):
    _assert_panel_columns(path, [x for x in columns if x not in {"decision_date", "ticker", "holdout_locked"}])
    yield from pq.ParquetFile(path).iter_batches(columns=columns, batch_size=batch_size)


def _normalise_frame(batch: pa.RecordBatch, features: list[str]) -> pd.DataFrame:
    frame = batch.to_pandas()
    frame["decision_date"] = pd.to_datetime(frame["decision_date"]).dt.tz_localize(None).dt.normalize()
    if frame["holdout_locked"].fillna(False).astype(bool).any():
        raise RuntimeError("FINAL_FROZEN_HOLDOUT_PANEL_ROWS_REJECTED")
    return frame.loc[
        (frame["decision_date"] >= REPLAY_START) & (frame["decision_date"] <= REPLAY_END),
        ["decision_date", "ticker", *features],
    ].copy()


def _write_batches(path: Path, frames: Iterable[pd.DataFrame]) -> int:
    writer: pq.ParquetWriter | None = None
    rows = 0
    try:
        for frame in frames:
            if frame.empty:
                continue
            table = pa.Table.from_pandas(frame, preserve_index=False)
            if writer is None:
                path.parent.mkdir(parents=True, exist_ok=True)
                writer = pq.ParquetWriter(path, table.schema, compression="zstd")
            writer.write_table(table)
            rows += len(frame)
    finally:
        if writer is not None:
            writer.close()
    if rows == 0:
        raise RuntimeError(f"FROZEN_INFERENCE_EMPTY:{path}")
    return rows


def build_frozen_entry_predictions(panel_root: Path, frozen: dict, output: Path) -> dict:
    panel = panel_root / "signal-panel.parquet"
    features = list(frozen["signal_features"])
    models = {int(h): joblib.load(meta["path"]) for h, meta in frozen["signal_models"].items()}
    columns = ["decision_date", "ticker", "holdout_locked", *features]

    def frames():
        for batch in _scan_batches(panel, columns):
            source = _normalise_frame(batch, features)
            if source.empty:
                continue
            x = source[features].to_numpy(dtype=float, copy=False)
            for horizon, model in models.items():
                scores = np.asarray(model.predict(x), dtype=float)
                yield pd.DataFrame({
                    "decision_date": source["decision_date"].to_numpy(),
                    "ticker": source["ticker"].astype(str).to_numpy(),
                    "fold_id": "FROZEN_MODEL_LIVE",
                    "horizon_sessions": horizon,
                    "predicted_net_excess_return": scores,
                    "family": "V5_FROZEN_WF_007",
                    "holdout_locked": False,
                })

    rows = _write_batches(output, frames())
    frame = pd.read_parquet(output, columns=["decision_date", "ticker", "horizon_sessions"])
    audit = {
        "path": str(output), "rows": rows,
        "min_date": str(pd.Timestamp(frame.decision_date.min()).date()),
        "max_date": str(pd.Timestamp(frame.decision_date.max()).date()),
        "horizons": sorted(int(x) for x in frame.horizon_sessions.unique()),
        "target_columns_read": False, "models_fit": False,
    }
    return audit


def _next_session_map(prices: pd.DataFrame) -> dict[pd.Timestamp, pd.Timestamp]:
    dates = sorted(pd.Timestamp(x) for x in prices.loc[prices.ticker.eq("URTH"), "date"].unique())
    return {date: dates[i + 1] for i, date in enumerate(dates[:-1])}


def _all_valid_ranked_candidates(payload: tuple, policy, threshold: float) -> list[str]:
    tickers, _scores, neg_scores, group_size = payload
    limit = max(1, math.ceil(group_size * policy.top_fraction))
    passing = int(np.searchsorted(neg_scores, -threshold, side="right")) if len(neg_scores) else 0
    # Do not cap at max_names here: positions already open can cause any later
    # valid candidate to be purchased by the actual replay.
    return [str(ticker) for ticker in tickers[:min(limit, passing)]]


def _exit_needs(predictions: Path, prices: pd.DataFrame, frozen_policies: dict[str, dict]) -> set[tuple[pd.Timestamp, str, int]]:
    next_session = _next_session_map(prices)
    all_dates = sorted(next_session) + ([max(next_session.values())] if next_session else [])
    needs: set[tuple[pd.Timestamp, str, int]] = set()
    learned = [m for m in known.TOP10 if m.mode == "LEARNED_EXIT"]
    for model in learned:
        bridge = frozen_policies[model.model_id]
        policy = search._policy_from_dict(dict(bridge["frozen_entry_policy"]))
        threshold = float(bridge["frozen_resolved_threshold"])
        hpred = pd.read_parquet(predictions, filters=[("horizon_sessions", "=", model.h)])
        hpred["decision_date"] = pd.to_datetime(hpred["decision_date"]).dt.normalize()
        prepared = search.prepare_signals(hpred.rename(columns={"horizon_sessions": "horizon", "predicted_net_excess_return": "score"})[
            ["decision_date", "ticker", "fold_id", "horizon", "score"]
        ])
        for date, payload in prepared["by_date"].items():
            candidates = _all_valid_ranked_candidates(payload, policy, threshold)
            entry = next_session.get(pd.Timestamp(date))
            if entry is None:
                continue
            try:
                entry_index = bisect_right(all_dates, entry) - 1
            except TypeError:
                continue
            # Replay removes already held names only *after* this ranked set has
            # been formed.  Every valid candidate can therefore move up into an
            # actual purchase; restricting this to max_names would silently
            # turn a learned-exit policy into a fixed-exit fallback.
            for ticker in candidates:
                for offset in range(policy.holding_days - 1):
                    index = entry_index + offset
                    if index >= len(all_dates):
                        break
                    needs.add((pd.Timestamp(all_dates[index]), str(ticker), policy.holding_days - 1 - offset))
    return needs


def build_frozen_exit_predictions(panel_root: Path, frozen: dict, needs: set[tuple[pd.Timestamp, str, int]], output: Path) -> dict:
    if not needs:
        raise RuntimeError("FROZEN_EXIT_NO_REQUIRED_DECISIONS")
    panel = panel_root / "stop-execution-panel.parquet"
    features = list(frozen["exit_features"])
    models = {int(h): joblib.load(meta["path"]) for h, meta in frozen["exit_models"].items()}
    requested: dict[tuple[pd.Timestamp, str], set[int]] = {}
    for date, ticker, horizon in needs:
        requested.setdefault((pd.Timestamp(date), str(ticker)), set()).add(int(horizon))
    columns = ["decision_date", "ticker", "holdout_locked", *features]
    emitted: set[tuple[pd.Timestamp, str, int]] = set()

    def frames():
        for batch in _scan_batches(panel, columns):
            source = _normalise_frame(batch, features)
            if source.empty:
                continue
            wanted = [requested.get((pd.Timestamp(r.decision_date), str(r.ticker)), set()) for r in source[["decision_date", "ticker"]].itertuples(index=False)]
            relevant = [i for i, values in enumerate(wanted) if values]
            if not relevant:
                continue
            sub = source.iloc[relevant].reset_index(drop=True)
            wanted = [wanted[i] for i in relevant]
            x = sub[features].to_numpy(dtype=float, copy=False)
            for horizon, estimator in models.items():
                indexes = [i for i, values in enumerate(wanted) if horizon in values]
                if not indexes:
                    continue
                values = np.asarray(estimator.predict(x[indexes]), dtype=float)
                out = pd.DataFrame({
                    "decision_date": sub.iloc[indexes]["decision_date"].to_numpy(),
                    "ticker": sub.iloc[indexes]["ticker"].astype(str).to_numpy(),
                    "exit_horizon_sessions": horizon,
                    "predicted_continuation_excess": values,
                    "holdout_locked": False,
                })
                emitted.update((pd.Timestamp(r.decision_date), str(r.ticker), int(r.exit_horizon_sessions)) for r in out.itertuples(index=False))
                yield out

    rows = _write_batches(output, frames())
    missing = len(needs - emitted)
    if missing:
        raise RuntimeError(f"FROZEN_EXIT_FEATURE_COVERAGE_MISSING:{missing}")
    return {"path": str(output), "rows": rows, "required_decisions": len(needs), "missing_decisions": missing, "target_columns_read": False, "models_fit": False}


def _at(curve: pd.DataFrame, date: pd.Timestamp) -> pd.Series:
    part = curve.loc[pd.to_datetime(curve.date) <= date]
    if part.empty:
        raise RuntimeError(f"LIVE_REPLAY_CUTOFF_NOT_REACHED:{date.date()}")
    return part.iloc[-1]


def _cagr(start: float, end: float, begin: pd.Timestamp, finish: pd.Timestamp) -> float:
    years = max((finish - begin).days / 365.2425, 1.0 / 365.2425)
    return (end / start) ** (1.0 / years) - 1.0 if start > 0 and end > 0 else float("nan")


def _segment(curve: pd.DataFrame, begin: pd.Timestamp, end: pd.Timestamp, label: str) -> dict:
    start_row = _at(curve, begin)
    end_row = _at(curve, end)
    start_date = pd.Timestamp(start_row.date)
    end_date = pd.Timestamp(end_row.date)
    strategy_cagr = _cagr(float(start_row.strategy_value), float(end_row.strategy_value), start_date, end_date)
    urth_cagr = _cagr(float(start_row.urth_value), float(end_row.urth_value), start_date, end_date)
    return {
        "classification": label,
        "start": str(start_date.date()), "end": str(end_date.date()),
        "strategy_growth_factor": float(end_row.strategy_value) / float(start_row.strategy_value),
        "urth_growth_factor": float(end_row.urth_value) / float(start_row.urth_value),
        "strategy_cagr": strategy_cagr, "urth_cagr": urth_cagr,
        "cagr_excess": strategy_cagr - urth_cagr,
    }


def _assert_exact_learned_exit(result: dict, model_id: str, run_label: str) -> None:
    metrics = result["metrics"]
    required = int(metrics.get("required_exit_decisions", 0))
    available = int(metrics.get("available_exit_decisions", 0))
    coverage = float(metrics.get("exit_decision_coverage", 0.0))
    missing = int(metrics.get("learned_exit_missing_prediction_count", 0))
    delisting = int(metrics.get("delisting_exit_decisions", 0))
    if required != available + delisting or coverage != 1.0 or missing != 0:
        raise RuntimeError(
            f"LEARNED_EXIT_COVERAGE_INCOMPLETE:{model_id}:{run_label}:"
            f"required={required}:available={available}:delisting={delisting}:coverage={coverage}:missing={missing}"
        )


def _report(summary: dict) -> str:
    lines = [
        "# Frozen Top-10 Live Replay", "",
        f"Status: `{summary['status']}`", "",
        "This suite performed inference only. Every model binary was SHA-256 locked; no model was fitted, no entry policy was searched, and no threshold was recalibrated.", "",
        "## Interpretation", "",
        f"- Historical regime backcast: {summary['backcast_window'][0]} to {summary['backcast_window'][1]} — diagnostic only; not causal OOS.",
        f"- True forward replay: {summary['true_forward_window'][0]} to {summary['true_forward_window'][1]} — separate fresh account, frozen model and policy, no retraining.",
        "- Learned exits: every required decision is covered either by a frozen model prediction or by the explicit delisting contract; no alias or forward-fill resolution was used.",
        f"- Feature provenance: `{summary['feature_provenance']['status']}`.",
        "- Final holdout remains closed; point-in-time universe is not verified, so promotion remains blocked.", "",
        "## Frozen identities (original order; not re-ranked)", "",
        "| Rank | Model | Backcast CAGR-X | True-forward CAGR-X | Forward growth | Forward trades |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for row in summary["results"]:
        lines.append(
            f"| {row['rank']} | {row['model_id']} | {row['backcast']['cagr_excess'] * 100:.2f}% | "
            f"{row['true_forward']['cagr_excess'] * 100:.2f}% | {row['true_forward']['strategy_growth_factor']:.3f}x | {row['true_forward_trade_count']} |"
        )
    return "\n".join(lines) + "\n"


def run(a: argparse.Namespace) -> dict:
    panel_root = Path(a.panel_root)
    training_root = Path(a.training_root)
    output = Path(a.output_root)
    output.mkdir(parents=True, exist_ok=True)
    frozen = frozen_model_manifest(training_root, panel_root)
    _write_json(output / "frozen_model_manifest.json", frozen)
    exit_training_audit = exit_development_training_cutoff_audit(training_root, frozen)
    _write_json(output / "exit_development_training_cutoff_audit.json", exit_training_audit)
    effective_true_forward_start = pd.Timestamp(exit_training_audit["effective_true_forward_start"])
    feature_audit = feature_provenance_audit(panel_root, Path(a.feature_builder_source), frozen)
    _write_json(output / "feature_provenance_audit.json", feature_audit)
    bridges = {m.model_id: known.load_known_bridge(Path(a.known_profit_root), m) for m in known.TOP10}
    _write_json(output / "frozen_policy_manifest.json", {"contract_id": CONTRACT_ID, "models": list(bridges.values())})
    entry_path = output / "frozen-entry-predictions.parquet"
    entry_audit = build_frozen_entry_predictions(panel_root, frozen, entry_path)
    entry = pd.read_parquet(entry_path)
    entry["decision_date"] = pd.to_datetime(entry["decision_date"]).dt.normalize()
    entry["horizon"] = entry["horizon_sessions"].astype(int)
    entry["score"] = pd.to_numeric(entry["predicted_net_excess_return"], errors="coerce")
    entry = entry.dropna(subset=["score"])
    tickers = set(entry.ticker.astype(str).unique())
    prices, price_audit = load_price_panel(Path(a.daily_store_root), tickers)
    raw_needs = _exit_needs(entry_path, prices, bridges)
    needs = {
        need for need in raw_needs
        if not (
            need[1] in DELISTING_RULES
            and pd.Timestamp(need[0]) > pd.Timestamp(DELISTING_RULES[need[1]]["series_end"])
        )
    }
    delisting_need_count = sum(
        1 for decision_day, ticker, _ in raw_needs
        if ticker in DELISTING_RULES and pd.Timestamp(decision_day) > pd.Timestamp(DELISTING_RULES[ticker]["series_end"])
    )
    exit_path = output / "frozen-exit-predictions.parquet"
    exit_audit = build_frozen_exit_predictions(panel_root, frozen, needs, exit_path)
    provider = LearnedExitProvider(exit_path)
    tax = ProfitTaxConfig(allowance_eur=a.tax_allowance_eur, church_tax_rate=a.church_tax_rate, benchmark_partial_exemption_rate=a.benchmark_partial_exemption)
    configure_replay(provider, tax)
    results = []
    backcast_end_date: pd.Timestamp | None = None
    for model in known.TOP10:
        bridge = bridges[model.model_id]
        policy = search._policy_from_dict(dict(bridge["frozen_entry_policy"]))
        signals = entry.loc[entry.horizon.eq(model.h), ["decision_date", "ticker", "fold_id", "horizon", "score"]].copy()
        result = replay(signals, prices, policy, CostModel(20), TaxConfig(False), initial=a.initial_capital, resolved_threshold=float(bridge["frozen_resolved_threshold"]), delisting_rules=DELISTING_RULES)
        forward_result = replay(signals, prices, policy, CostModel(20), TaxConfig(False), start=effective_true_forward_start, end=REPLAY_END, initial=a.initial_capital, resolved_threshold=float(bridge["frozen_resolved_threshold"]), delisting_rules=DELISTING_RULES)
        if model.mode == "LEARNED_EXIT":
            _assert_exact_learned_exit(result, model.model_id, "CONTINUOUS_BACKCAST_TO_FORWARD")
            _assert_exact_learned_exit(forward_result, model.model_id, "ISOLATED_TRUE_FORWARD")
        curve = result["curve"]
        backcast_end = pd.Timestamp(curve.loc[pd.to_datetime(curve.date) < effective_true_forward_start, "date"].max())
        backcast_end_date = backcast_end if backcast_end_date is None else backcast_end_date
        backcast = _segment(curve, pd.Timestamp(curve.date.min()), backcast_end, "HISTORICAL_REGIME_BACKCAST_NOT_CAUSAL_OOS")
        continuous_forward = _segment(curve, backcast_end, pd.Timestamp(curve.date.max()), "CONTINUOUS_CONTEXT_DIAGNOSTIC_ONLY")
        forward_curve = forward_result["curve"]
        forward = _segment(forward_curve, pd.Timestamp(forward_curve.date.min()), pd.Timestamp(forward_curve.date.max()), "TRUE_FORWARD_FRESH_ACCOUNT_FROZEN_MODEL_OOS")
        results.append({
            "rank": model.rank, "model_id": model.model_id, "mode": model.mode, "h": model.h, "d": model.d, "n": model.n,
            "known_reference_median_cagr_excess": model.known_median_cagr_excess,
            "backcast": backcast, "continuous_forward_context": continuous_forward, "true_forward": forward,
            "continuous_trade_count": int(result["metrics"].get("trade_count", 0)),
            "true_forward_trade_count": int(forward_result["metrics"].get("trade_count", 0)),
            "metrics": result["metrics"],
            "true_forward_metrics": forward_result["metrics"],
            "frozen_policy_id": policy.policy_id, "frozen_resolved_threshold": float(bridge["frozen_resolved_threshold"]),
            "delisting_rules": DELISTING_RULES,
        })
    pd.DataFrame([{**{k: v for k, v in row.items() if k not in {"backcast", "continuous_forward_context", "true_forward", "metrics", "true_forward_metrics"}},
                   **{f"backcast_{k}": v for k, v in row["backcast"].items()},
                   **{f"continuous_forward_{k}": v for k, v in row["continuous_forward_context"].items()},
                   **{f"true_forward_{k}": v for k, v in row["true_forward"].items()},
                   **{f"metric_{k}": v for k, v in row["metrics"].items() if isinstance(v, (int, float, str, bool))},
                   **{f"true_forward_metric_{k}": v for k, v in row["true_forward_metrics"].items() if isinstance(v, (int, float, str, bool))}}
                  for row in results]).to_csv(output / "live_replay_results.csv", index=False)
    summary = {
        "contract_id": CONTRACT_ID, "status": "FROZEN_MODEL_LIVE_REPLAY_COMPLETE", "final_holdout_opened": False,
        "model_retraining": "FORBIDDEN_AND_NOT_PERFORMED", "policy_reoptimization": "FORBIDDEN_AND_NOT_PERFORMED",
        "requested_historical_forward_start": str(REQUESTED_HISTORICAL_FORWARD_START.date()),
        "backcast_window": [str(REPLAY_START.date()), str(backcast_end_date.date())],
        "true_forward_window": [str(effective_true_forward_start.date()), str(REPLAY_END.date())],
        "backcast_interpretation": "HISTORICAL_REGIME_BACKCAST_NOT_CAUSAL_OOS",
        "forward_interpretation": "TRUE_FORWARD_FROZEN_MODEL_OOS",
        "point_in_time_universe_verified": False, "promotion_eligible": False,
        "frozen_models": frozen, "exit_development_training_cutoff": exit_training_audit,
        "feature_provenance": feature_audit,
        "entry_inference": entry_audit, "exit_inference": exit_audit,
        "price_audit": price_audit, "results": results,
        "delisting_contract": {
            "status": "APPLIED",
            "synthetic_alias_resolutions": 0,
            "forward_fill_resolutions": 0,
            "excluded_missing_exit_keys": delisting_need_count,
            "rules": DELISTING_RULES,
        },
    }
    _write_json(output / "live_replay_summary.json", summary)
    (output / "REPORT.md").write_text(_report(summary), encoding="utf-8")
    return summary


def self_test() -> None:
    assert REQUESTED_HISTORICAL_FORWARD_START == pd.Timestamp("2023-08-11")
    curve = pd.DataFrame({"date": pd.to_datetime(["2023-08-10", "2026-07-24"]), "strategy_value": [100.0, 121.0], "urth_value": [100.0, 110.0]})
    segment = _segment(curve, pd.Timestamp("2023-08-10"), pd.Timestamp("2026-07-24"), "TEST")
    assert segment["cagr_excess"] > 0
    assert len({m.model_id for m in known.TOP10}) == 10
    assert all(m.h in {11, 24, 28} for m in known.TOP10)
    candidate_policy = Policy(28, 0.95, 1.0, 1, 21, "LEARNED_EXIT")
    candidate_payload = (np.array(["HELD", "REPLACEMENT"]), np.array([2.0, 1.0]), np.array([-2.0, -1.0]), 2)
    assert _all_valid_ranked_candidates(candidate_payload, candidate_policy, 0.0) == ["HELD", "REPLACEMENT"]
    source = Path(__file__).read_text(encoding="utf-8")
    assert source.count(".fit(") == 1  # this assertion is the only occurrence
    assert "target_columns_read\": False" in source
    assert source.count("min(limit, passing, policy.max_names)") == 1  # this assertion only
    assert "LEARNED_EXIT_COVERAGE_INCOMPLETE" in source
    assert "NEXT_AVAILABLE_SESSION_AFTER_LATEST_REQUIRED_EXIT_LABEL_COMPLETION" in source
    assert len(EXPECTED_SIGNAL_SHA256) == 3 and len(EXPECTED_EXIT_SHA256) == 20
    assert len(EXPECTED_FEATURE_BUILDER_SHA256) == 64
    dates = pd.to_datetime(["2026-07-22", "2026-07-23", "2026-07-24"])
    prices = pd.DataFrame([
        {"date": date, "ticker": ticker, "open": 100.0, "close": 100.0}
        for date in dates for ticker in ("URTH", "TEST")
    ])
    signals = pd.DataFrame([{
        "decision_date": dates[0], "ticker": "TEST", "fold_id": "SYNTHETIC",
        "horizon": 28, "score": 1.0,
    }])
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "exit.parquet"
        pd.DataFrame([
            {"decision_date": dates[1], "ticker": "TEST", "exit_horizon_sessions": 3, "predicted_continuation_excess": 0.01, "holdout_locked": False},
            {"decision_date": dates[2], "ticker": "TEST", "exit_horizon_sessions": 2, "predicted_continuation_excess": 0.01, "holdout_locked": False},
        ]).to_parquet(path, index=False)
        configure_replay(LearnedExitProvider(path), ProfitTaxConfig())
        censored = replay(signals, prices, Policy(28, 0.0, 1.0, 1, 4, "LEARNED_EXIT"), CostModel(20), TaxConfig(False), resolved_threshold=0.0)
        metrics = censored["metrics"]
        assert metrics["required_exit_decisions"] == metrics["available_exit_decisions"] == 2
        assert metrics["learned_exit_missing_prediction_count"] == 0
        assert metrics["learned_exit_fallback_positions"] == 0
        assert metrics["learned_exit_horizon_boundary_count"] == 1
    print("TOP10_FROZEN_LIVE_REPLAY_SELF_TEST_OK")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Frozen Top-10 live inference and chronological replay")
    parser.add_argument("--panel-root")
    parser.add_argument("--training-root")
    parser.add_argument("--feature-builder-source")
    parser.add_argument("--known-profit-root")
    parser.add_argument("--daily-store-root")
    parser.add_argument("--output-root", default="artifacts/top10-frozen-live-replay")
    parser.add_argument("--initial-capital", type=float, default=10_000.0)
    parser.add_argument("--tax-allowance-eur", type=float, default=1000.0)
    parser.add_argument("--church-tax-rate", type=float, default=0.0)
    parser.add_argument("--benchmark-partial-exemption", type=float, default=0.30)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test(); return
    missing = [name for name in ("panel_root", "training_root", "feature_builder_source", "known_profit_root", "daily_store_root") if not getattr(args, name)]
    if missing:
        raise SystemExit("missing required args: " + ", ".join("--" + x.replace("_", "-") for x in missing))
    summary = run(args)
    print(json.dumps({"contract_id": summary["contract_id"], "status": summary["status"], "output_root": args.output_root}, indent=2))


if __name__ == "__main__":
    main()
