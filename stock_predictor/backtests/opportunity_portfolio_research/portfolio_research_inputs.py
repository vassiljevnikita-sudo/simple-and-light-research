from __future__ import annotations

import json
from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq

REQUIRED_PREDICTION_COLUMNS = ("decision_date", "ticker", "fold_id", "horizon_sessions", "predicted_net_excess_return", "family")
_HOLDOUT_ROLE_COLUMNS = ("dataset_role", "split", "evaluation_split", "fold_type", "fold_role", "stage")
_HOLDOUT_ROLE_VALUES = {"FINAL_FROZEN_HOLDOUT", "FINAL_HOLDOUT", "HOLDOUT", "FROZEN_HOLDOUT"}


def assert_no_final_frozen_holdout(frame: pd.DataFrame, *, source: str = "predictions") -> None:
    """Fail closed if an input explicitly contains final/frozen holdout rows."""
    if "holdout_locked" in frame.columns:
        values = frame["holdout_locked"].astype(str).str.strip().str.lower()
        if values.isin({"true", "1", "yes", "y"}).any():
            raise ValueError(f"FINAL_FROZEN_HOLDOUT_ENTRY_PREDICTIONS_REJECTED:{source}:holdout_locked")
    for column in _HOLDOUT_ROLE_COLUMNS:
        if column not in frame.columns:
            continue
        values = frame[column].astype(str).str.strip().str.upper()
        if values.isin(_HOLDOUT_ROLE_VALUES).any():
            raise ValueError(f"FINAL_FROZEN_HOLDOUT_ENTRY_PREDICTIONS_REJECTED:{source}:{column}")

def load_predictions(path: Path) -> tuple[pd.DataFrame, dict]:
    cols = list(REQUIRED_PREDICTION_COLUMNS)
    available = set(pq.ParquetFile(path).schema.names)
    optional_guard_cols = [c for c in ("holdout_locked", *_HOLDOUT_ROLE_COLUMNS) if c in available]
    frame = pd.read_parquet(path, columns=cols + optional_guard_cols)
    if frame.empty:
        raise ValueError("V5 prediction artifact is empty")
    assert_no_final_frozen_holdout(frame, source=str(path))
    if frame["family"].astype(str).str.upper().eq("V2_RANKING").any():
        frame = frame.loc[~frame["family"].astype(str).str.upper().eq("V2_RANKING")].copy()
    frame["decision_date"] = pd.to_datetime(frame["decision_date"]).dt.normalize()
    frame["horizon"] = frame["horizon_sessions"].astype(int)
    frame["score"] = pd.to_numeric(frame["predicted_net_excess_return"], errors="coerce")
    frame = frame.dropna(subset=["decision_date", "ticker", "fold_id", "horizon", "score"])
    frame["ticker"] = frame["ticker"].astype(str)
    frame["fold_id"] = frame["fold_id"].astype(str)
    horizons = sorted(frame["horizon"].unique().tolist())
    if horizons == list(range(1, 31)):
        signal_contract = "V5_SELECTED:H1-H30"
    else:
        signal_contract = "V5_SELECTED:" + "/".join(f"H{int(h)}" for h in horizons)
    audit = {
        "path": str(path), "rows": int(len(frame)), "v2_rows_excluded": int(len(pd.read_parquet(path, columns=["family"])) - len(frame)),
        "horizons": horizons, "tickers": int(frame["ticker"].nunique()),
        "folds": sorted(frame["fold_id"].unique()), "signal_contract": signal_contract,
        "family_used_for_selection": False,
    }
    return frame[["decision_date", "ticker", "fold_id", "horizon", "score"]], audit

def _manifest_paths(daily_root: Path) -> dict[str, Path]:
    manifest = json.loads((daily_root / "manifest.json").read_text(encoding="utf-8"))
    result = {}
    for key, meta in manifest.get("groups", {}).items():
        if not isinstance(meta, dict) or not meta.get("output"):
            continue
        ticker = str(meta.get("ticker", key.split("/")[-2] if "/" in key else "")).upper()
        result.setdefault(ticker, daily_root / Path(str(meta["output"]).replace("artifacts/daily-parquet/", "")))
    return result

def load_price_panel(daily_root: Path, tickers: set[str], benchmark: str = "URTH") -> tuple[pd.DataFrame, dict]:
    paths = _manifest_paths(daily_root)
    wanted = set(map(str.upper, tickers)) | {benchmark}
    frames = []
    missing = []
    for ticker in sorted(wanted):
        path = paths.get(ticker)
        if not path or not path.is_file():
            missing.append(ticker); continue
        frame = pd.read_parquet(path, columns=["session_date", "open", "close"])
        frame["date"] = pd.to_datetime(frame.pop("session_date")).dt.normalize()
        frame["ticker"] = ticker
        frame = frame.dropna(subset=["date", "open", "close"])
        frame = frame[(frame["open"] > 0) & (frame["close"] > 0)]
        frames.append(frame[["date", "ticker", "open", "close"]])
    if not frames or benchmark not in {str(x).upper() for x in pd.concat(frames)["ticker"].unique()}:
        raise FileNotFoundError(f"benchmark price data unavailable: {benchmark}")
    panel = pd.concat(frames, ignore_index=True).sort_values(["date", "ticker"]).reset_index(drop=True)
    audit = {"daily_root": str(daily_root), "benchmark": benchmark, "tickers_loaded": int(panel["ticker"].nunique()), "missing_tickers": missing, "rows": int(len(panel)), "source": "canonical_daily_parquet"}
    return panel, audit
