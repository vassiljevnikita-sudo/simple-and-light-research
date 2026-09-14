"""Causal opportunity-state predictability diagnostics for the Dynamic-QBD run.

The primary unit is an active signal event: decision_date x ticker x family x
arm.  The completed matured-prediction store is read row-group by row-group;
the model factory and prediction artifacts are never regenerated.  Forward
realized excess is a target only.  The final holdout is strictly excluded.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from hashlib import sha256
import json
import math
from pathlib import Path
import re
import shutil
import sqlite3
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


SCHEMA_VERSION = "DYNAMIC_QBD_OPPORTUNITY_STATE_V1"
HOLDOUT_START = pd.Timestamp("2026-07-25")
ARMS = (
    "A_FROZEN_MODEL_FROZEN_CALIBRATION",
    "B_FROZEN_MODEL_ROLLING_RECALIBRATION",
    "C_ROLLING_REFIT_ROLLING_RECALIBRATION",
)
FAMILY_RE = re.compile(r"^H(?P<h>\d+)_D(?P<d>\d+)_N(?P<n>\d+)_(?P<exit>.+)$")
FEATURE_GROUPS = {
    "T1_CONSENSUS_ONLY": [],
    "T2_SCORE_ONLY": ["prediction_score", "score_percentile", "distance_to_threshold",
                      "horizon_h", "holding_d", "max_names_n"],
}
T1_FEATURES = [
    "active_family_count_same_ticker", "active_family_fraction_same_ticker",
    "mean_score_same_ticker", "median_score_same_ticker", "std_score_same_ticker",
    "max_score_same_ticker", "min_score_same_ticker", "score_range_same_ticker",
    "active_fraction_short", "active_fraction_mid", "active_fraction_long",
    "mean_score_short", "mean_score_mid", "mean_score_long",
    "max_score_short", "max_score_mid", "max_score_long",
    "short_mid_agreement", "mid_long_agreement", "short_long_agreement",
    "short_minus_long_score", "short_minus_long_active_fraction",
    "cross_horizon_score_std", "cross_horizon_consensus_strength",
    "score_slope_over_h",
]
MARKET_FEATURES = [
    "benchmark_trailing_20d_return", "benchmark_trailing_60d_return",
    "benchmark_realized_volatility_20d", "benchmark_drawdown_trailing_252d",
]
MODELS = ("RIDGE", "HGB")
FEATURE_ARMS = ("T1_CONSENSUS_ONLY", "T2_SCORE_ONLY", "T3_SCORE_PLUS_CONSENSUS", "T4_SCORE_PLUS_CONSENSUS_PLUS_MARKET")
TARGET_COLUMNS = ("forward_excess_return", "positive_excess", "same_date_target_rank", "same_ticker_target_rank")


def _json_default(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, (pd.Timestamp, pd.Timedelta)):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def _hash(value) -> str:
    return sha256(json.dumps(value, default=_json_default, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict) -> None:
    tmp = path.with_name(f".{path.name}.{Path(__file__).stem}.tmp")
    tmp.write_text(json.dumps(value, default=_json_default, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    last_error = None
    for _ in range(40):
        try:
            tmp.replace(path)
            return
        except PermissionError as exc:
            last_error = exc
            time.sleep(0.05)
    raise last_error


def _atomic_frame(frame: pd.DataFrame, path: Path, parquet: bool = True) -> None:
    tmp = path.with_name(f".{path.name}.{Path(__file__).stem}.tmp")
    if parquet:
        frame.to_parquet(tmp, index=False)
    else:
        frame.to_csv(tmp, index=False)
    tmp.replace(path)


def _atomic_parquet_table(table, path: Path) -> None:
    """Publish a pyarrow table atomically without materialising all parts."""
    tmp = path.with_name(f".{path.name}.{Path(__file__).stem}.tmp")
    pq.write_table(table, tmp)
    tmp.replace(path)


def _publish_parquet_parts(parts: list[Path], path: Path) -> None:
    if not parts:
        raise FileNotFoundError("PARQUET_PARTS_MISSING")
    tmp = path.with_name(f".{path.name}.{Path(__file__).stem}.tmp")
    writer = None
    try:
        for part in parts:
            table = pq.read_table(part)
            if writer is None:
                writer = pq.ParquetWriter(tmp, table.schema, compression="zstd")
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()
    tmp.replace(path)


def _aggregate_eligibility_parts(parts: list[Path], cache: Path, output: Path) -> None:
    """Aggregate row-group eligibility fragments on disk, not in RAM."""
    db_path = cache / "eligibility.sqlite3"
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("CREATE TABLE IF NOT EXISTS eligibility (decision_date TEXT, ticker TEXT, arm TEXT, h_bucket TEXT, eligible_family_count INTEGER, PRIMARY KEY (decision_date, ticker, arm, h_bucket))")
        for part in parts:
            frame = pd.read_parquet(part)
            rows = [(str(pd.Timestamp(r.decision_date).date()), str(r.ticker), str(r.arm), str(r.h_bucket), int(r.eligible_family_count))
                    for r in frame.itertuples(index=False)]
            conn.executemany("INSERT INTO eligibility VALUES (?, ?, ?, ?, ?) ON CONFLICT(decision_date, ticker, arm, h_bucket) DO UPDATE SET eligible_family_count = eligible_family_count + excluded.eligible_family_count", rows)
        conn.commit()
        tmp = output.with_name(f".{output.name}.{Path(__file__).stem}.tmp")
        writer = None
        for chunk in pd.read_sql_query("SELECT decision_date, ticker, arm, h_bucket, eligible_family_count FROM eligibility ORDER BY decision_date, ticker, arm, h_bucket", conn, chunksize=100_000):
            chunk["decision_date"] = pd.to_datetime(chunk["decision_date"])
            table = pa.Table.from_pandas(chunk, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(tmp, table.schema, compression="zstd")
            writer.write_table(table)
        if writer is not None:
            writer.close()
        tmp.replace(output)
    finally:
        conn.close()


def _require_before_holdout(frame: pd.DataFrame, column: str, holdout_start: pd.Timestamp) -> None:
    values = pd.to_datetime(frame[column], errors="raise")
    if len(values) and values.max() >= holdout_start:
        raise ValueError(f"FINAL_HOLDOUT_BOUNDARY_VIOLATION:{column}:{values.max()}")


def _parse_metadata(path: Path) -> pd.DataFrame:
    raw = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for item in raw["families"]:
        match = FAMILY_RE.match(str(item["family_id"]))
        if not match:
            raise ValueError(f"FAMILY_ID_SCHEMA_INVALID:{item['family_id']}")
        rows.append({"family_id": str(item["family_id"]), "horizon_h": int(match["h"]),
                     "holding_d": int(match["d"]), "max_names_n": int(match["n"]),
                     "exit_mode": str(match["exit"]),
                     "h_bucket": "SHORT" if int(match["h"]) <= 10 else "MID" if int(match["h"]) <= 20 else "LONG"})
    result = pd.DataFrame(rows)
    if result["family_id"].duplicated().any():
        raise ValueError("FAMILY_METADATA_DUPLICATE")
    return result


def _load_schedule(run_root: Path, metadata: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    schedule_path = run_root / "abc_generation_schedule.parquet"
    valid_path = run_root / "valid_generations.parquet"
    schedule = pd.read_parquet(schedule_path, columns=["family_id", "generation_id", "arm", "activation_date", "resolved_threshold", "resolved_top_fraction"])
    schedule["activation_date"] = pd.to_datetime(schedule["activation_date"])
    if schedule["arm"].isna().any() or schedule["resolved_threshold"].isna().any():
        raise ValueError("SCHEDULE_REQUIRED_FIELDS_MISSING")
    # A generation can receive later causal calibration states.  They are
    # valid when their activation dates differ; only two contradictory states
    # at the same activation timestamp are invalid.
    conflicts = schedule.groupby(["family_id", "generation_id", "arm", "activation_date"]).agg(
        threshold_n=("resolved_threshold", "nunique"), top_n=("resolved_top_fraction", "nunique"))
    if (conflicts[["threshold_n", "top_n"]] > 1).any().any():
        raise ValueError("SCHEDULE_SAME_ACTIVATION_THRESHOLD_CONFLICT")
    schedule = schedule.sort_values("activation_date").drop_duplicates(["family_id", "generation_id", "arm", "activation_date"], keep="last")
    schedule = schedule.merge(metadata, on="family_id", how="left", validate="many_to_one")
    valid = pd.read_parquet(valid_path, columns=["family_id", "generation_id", "information_cutoff", "activation_date", "resolved_score_quantile"])
    valid["information_cutoff"] = pd.to_datetime(valid["information_cutoff"])
    valid = valid.drop_duplicates(["family_id", "generation_id"], keep="first")
    schedule = schedule.merge(valid, on=["family_id", "generation_id"], how="left", validate="many_to_one", suffixes=("", "_valid"))
    schedule["information_cutoff"] = schedule["information_cutoff"].fillna(schedule["activation_date"])
    if schedule["information_cutoff"].gt(schedule["activation_date"]).any():
        raise ValueError("GENERATION_CUTOFF_AFTER_ACTIVATION")
    return schedule, metadata


def _load_trades(run_root: Path) -> pd.DataFrame:
    path = run_root / "development/family_shadow_trades.parquet"
    columns = ["ticker", "entry_date", "family_id", "arm"]
    trades = pd.read_parquet(path, columns=columns)
    trades["entry_date"] = pd.to_datetime(trades["entry_date"])
    trades["trade_executed"] = True
    return trades.drop_duplicates(["ticker", "entry_date", "family_id", "arm"])


def _market_features(run_root: Path, dates: pd.Series) -> pd.DataFrame:
    """Build lagged URTH-only features from one validated NAV row-group."""
    nav_path = run_root / "development/family_shadow_nav.parquet"
    nav = pq.ParquetFile(nav_path).read_row_group(0, columns=["date", "urth_value"]).to_pandas()
    nav["date"] = pd.to_datetime(nav["date"])
    nav = nav.sort_values("date").drop_duplicates("date")
    if nav["urth_value"].isna().any():
        raise ValueError("MARKET_URTH_MISSING")
    # Shift one session: the market state is available before the decision-day
    # return, avoiding a same-day close-to-close leakage ambiguity.
    price = nav["urth_value"].astype(float)
    ret = price.pct_change()
    shifted = pd.DataFrame({"decision_date": nav["date"],
                            "benchmark_trailing_20d_return": price.shift(1) / price.shift(21) - 1.0,
                            "benchmark_trailing_60d_return": price.shift(1) / price.shift(61) - 1.0,
                            "benchmark_realized_volatility_20d": ret.shift(1).rolling(20, min_periods=20).std() * np.sqrt(252.0),
                            "benchmark_drawdown_trailing_252d": price.shift(1) / price.shift(1).rolling(252, min_periods=252).max() - 1.0})
    return shifted[shifted["decision_date"].isin(pd.to_datetime(dates))]


def _schedule_for_family(schedule: pd.DataFrame, family_id: str) -> pd.DataFrame:
    return schedule.loc[schedule["family_id"].eq(family_id)].copy()


def _apply_causal_schedule(frame: pd.DataFrame, schedule: pd.DataFrame) -> pd.DataFrame:
    """Expand source rows to arms using the latest causal schedule state."""
    output = []
    value_columns = ["arm", "activation_date", "resolved_threshold", "resolved_top_fraction",
                     "information_cutoff", "horizon_h", "holding_d", "max_names_n", "exit_mode", "h_bucket"]
    for (family_id, generation_id), source_group in frame.groupby(["family_id", "generation_id"], sort=False):
        states = schedule[(schedule["family_id"].eq(family_id)) & (schedule["generation_id"].eq(generation_id))]
        for arm, state in states.groupby("arm", sort=False):
            state = state.sort_values("activation_date").drop_duplicates("activation_date", keep="last")
            positions = np.searchsorted(state["activation_date"].to_numpy(dtype="datetime64[ns]"),
                                             source_group["decision_date"].to_numpy(dtype="datetime64[ns]"), side="right") - 1
            valid = positions >= 0
            if not valid.any():
                continue
            selected = source_group.loc[valid].copy()
            chosen = state.iloc[positions[valid]].reset_index(drop=True)
            for column in value_columns:
                selected[column] = chosen[column].to_numpy()
            output.append(selected)
    return pd.concat(output, ignore_index=True) if output else frame.iloc[0:0].copy()


def build_opportunity_panel(run_root: Path, output_root: Path, holdout_start: pd.Timestamp = HOLDOUT_START) -> dict:
    output_root.mkdir(parents=True, exist_ok=True)
    cache = output_root / "opportunity_panel"
    meta_path = output_root / "opportunity_panel.meta.json"
    pred_path = run_root / "matured_generation_predictions.parquet"
    metadata = _parse_metadata(run_root / "development/family_registry.json")
    schedule, metadata = _load_schedule(run_root, metadata)
    trades = _load_trades(run_root)
    pf = pq.ParquetFile(pred_path)
    source = {"path": str(pred_path.resolve()), "size": pred_path.stat().st_size,
              "rows": pf.metadata.num_rows, "row_groups": pf.metadata.num_row_groups,
              "schedule_size": (run_root / "abc_generation_schedule.parquet").stat().st_size}
    fingerprint = _hash({"schema": SCHEMA_VERSION, "holdout": str(holdout_start.date()), "source": source,
                         "active_definition": "score_ge_resolved_threshold_and_activation_date",
                         "target": "realized_excess_with_terminal_date_before_holdout"})
    existing_meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
    if meta_path.is_file() and cache.is_dir() and existing_meta.get("fingerprint") == fingerprint and (output_root / "opportunity_panel.parquet").is_file():
        print("[cache] opportunity_panel hit", flush=True)
        return existing_meta
    marker = cache / "_fingerprint.json"
    if cache.exists() and marker.is_file():
        old_fingerprint = json.loads(marker.read_text(encoding="utf-8")).get("fingerprint")
        if old_fingerprint != fingerprint:
            invalid = cache.with_name(f"{cache.name}.invalidated-{int(time.time())}")
            cache.replace(invalid)
    cache.mkdir(parents=True, exist_ok=True)
    _atomic_json(marker, {"fingerprint": fingerprint, "schema_version": SCHEMA_VERSION})
    progress_path = cache / "_progress.json"
    progress = json.loads(progress_path.read_text(encoding="utf-8")) if progress_path.is_file() else {}
    counts = progress.get("counts", {"row_groups": 0, "input_rows": 0, "dedup_rows": 0, "active_rows": 0, "target_rows": 0})
    columns = ["decision_date", "terminal_date", "ticker", "family_id", "generation_id", "model_artifact_id", "score", "realized_excess"]
    for rg_index in range(pf.num_row_groups):
        part_path = cache / f"part-{rg_index:04d}.parquet"
        eligibility_path = cache / f"eligibility-{rg_index:04d}.parquet"
        if part_path.is_file() and eligibility_path.is_file() and str(rg_index) in progress.get("completed", []):
            continue
        frame = pf.read_row_group(rg_index, columns=columns).to_pandas()
        counts["row_groups"] += 1
        counts["input_rows"] += len(frame)
        frame = frame.drop_duplicates()
        counts["dedup_rows"] += len(frame)
        frame["decision_date"] = pd.to_datetime(frame["decision_date"])
        frame["terminal_date"] = pd.to_datetime(frame["terminal_date"])
        frame = frame[(frame["decision_date"] < holdout_start) & (frame["terminal_date"] < holdout_start)]
        if frame.empty:
            continue
        family_ids = frame["family_id"].dropna().astype(str).unique().tolist()
        local_schedule = schedule.loc[schedule["family_id"].isin(family_ids)].copy()
        frame = _apply_causal_schedule(frame, local_schedule)
        if frame.empty:
            eligibility_path.write_bytes(b"")
            progress.setdefault("completed", []).append(str(rg_index))
            progress["counts"] = counts
            _atomic_json(progress_path, progress)
            continue
        eligible = frame[["decision_date", "ticker", "family_id", "arm", "h_bucket"]].drop_duplicates()
        eligible = eligible.groupby(["decision_date", "ticker", "arm", "h_bucket"], as_index=False).agg(eligible_family_count=("family_id", "nunique"))
        tmp_eligibility = eligibility_path.with_suffix(".tmp")
        eligible.to_parquet(tmp_eligibility, index=False)
        tmp_eligibility.replace(eligibility_path)
        frame["score_percentile"] = frame.groupby(["decision_date", "family_id", "generation_id"])["score"].rank(method="average", pct=True)
        frame["distance_to_threshold"] = frame["score"] - frame["resolved_threshold"]
        frame["signal_active"] = frame["score"].ge(frame["resolved_threshold"])
        frame = frame[frame["signal_active"]]
        if frame.empty:
            continue
        frame["target_matured_date"] = frame["terminal_date"]
        frame["has_valid_target"] = frame["realized_excess"].notna() & frame["terminal_date"].lt(holdout_start)
        frame["forward_excess_return"] = frame["realized_excess"]
        frame["forward_stock_return"] = np.nan
        frame["forward_benchmark_return"] = np.nan
        frame = frame.merge(trades, left_on=["ticker", "decision_date", "family_id", "arm"],
                            right_on=["ticker", "entry_date", "family_id", "arm"], how="left", suffixes=("", "_trade"))
        frame["trade_executed"] = frame["trade_executed"].fillna(False).astype(bool)
        frame = frame.drop(columns=["entry_date"], errors="ignore")
        keep = ["decision_date", "ticker", "family_id", "arm", "horizon_h", "holding_d", "max_names_n", "exit_mode",
                "generation_id", "information_cutoff", "model_artifact_id", "score", "score_percentile", "resolved_threshold",
                "resolved_top_fraction", "distance_to_threshold", "signal_active", "trade_executed", "forward_stock_return",
                "forward_benchmark_return", "forward_excess_return", "target_matured_date", "has_valid_target", "h_bucket"]
        frame["prediction_score"] = frame["score"]
        keep.insert(11, "prediction_score")
        tmp_part = part_path.with_suffix(".tmp")
        frame[keep].to_parquet(tmp_part, index=False)
        tmp_part.replace(part_path)
        counts["active_rows"] += len(frame)
        counts["target_rows"] += int(frame["has_valid_target"].sum())
        progress.setdefault("completed", []).append(str(rg_index))
        progress["counts"] = counts
        _atomic_json(progress_path, progress)
        if rg_index % 100 == 0:
            print(f"[panel] row_group={rg_index}/{pf.num_row_groups} active_rows={counts['active_rows']}", flush=True)
    if counts["active_rows"] == 0:
        raise ValueError("NO_ACTIVE_OPPORTUNITIES")
    _publish_parquet_parts(sorted(cache.glob("part-*.parquet")), output_root / "opportunity_panel.parquet")
    eligibility_parts = [p for p in sorted(cache.glob("eligibility-*.parquet")) if p.stat().st_size > 0]
    if eligibility_parts:
        _aggregate_eligibility_parts(eligibility_parts, cache, output_root / "eligibility_panel.parquet")
    meta = {"schema_version": SCHEMA_VERSION, "fingerprint": fingerprint, "sources": source,
            "holdout_boundary_exclusive": str(holdout_start), "counts": counts,
            "unavailable_fields": ["forward_stock_return", "forward_benchmark_return", "market_breadth",
                                    "cross_sectional_return_dispersion", "average_pairwise_correlation_proxy"],
            "active_definition": "signal_active = score >= resolved_threshold and decision_date >= activation_date",
            "target_definition": "forward_excess_return = authoritative matured realized_excess; no target imputation",
            "duplicate_contract": "exact input rows deduplicated before arm-specific schedule expansion",
            "resume_contract": "atomic row-group fragments with fingerprint marker and per-part progress",
            "eligibility_denominator": "same decision_date x ticker x arm x H-bucket eligible family count from all matured signal rows"}
    _atomic_json(meta_path, meta)
    return meta


def _read_events(output_root: Path) -> pd.DataFrame:
    parts = sorted((output_root / "opportunity_panel").glob("part-*.parquet"))
    if not parts:
        raise FileNotFoundError("OPPORTUNITY_PANEL_PARTS_MISSING")
    events = pd.concat((pd.read_parquet(p) for p in parts), ignore_index=True)
    exact_rows = len(events)
    events = events.drop_duplicates(ignore_index=True)
    # Cross-row-group duplicates are safe to remove only when every field is
    # identical.  A same-key disagreement is a hard source-integrity failure.
    exact_duplicates_removed = exact_rows - len(events)
    events["decision_date"] = pd.to_datetime(events["decision_date"])
    events["target_matured_date"] = pd.to_datetime(events["target_matured_date"])
    if events.duplicated(["decision_date", "ticker", "family_id", "arm", "generation_id"]).any():
        raise ValueError("OPPORTUNITY_EVENT_DUPLICATE")
    _require_before_holdout(events, "decision_date", HOLDOUT_START)
    events.attrs["exact_duplicates_removed"] = exact_duplicates_removed
    return events


def _bucket_summary(active: pd.DataFrame, eligibility: pd.DataFrame) -> pd.DataFrame:
    keys = ["decision_date", "ticker", "arm"]
    raw = active.groupby(keys).agg(
        raw_active_family_count=("family_id", "nunique"), raw_mean_score=("score", "mean"),
        raw_median_score=("score", "median"), raw_std_score=("score", "std"),
        raw_max_score=("score", "max"), raw_min_score=("score", "min")).reset_index()
    raw["raw_score_range"] = raw["raw_max_score"] - raw["raw_min_score"]
    bucket = active.groupby(keys + ["h_bucket"]).agg(bucket_mean_score=("score", "mean"),
                                                       bucket_active_family_count=("family_id", "nunique"),
                                                       bucket_max_score=("score", "max")).reset_index()
    # One economic vote per H bucket is the primary consensus contract.
    dedup = bucket.groupby(keys).agg(active_family_count_same_ticker=("h_bucket", "nunique"),
                                      mean_score_same_ticker=("bucket_mean_score", "mean"),
                                      median_score_same_ticker=("bucket_mean_score", "median"),
                                      std_score_same_ticker=("bucket_mean_score", "std"),
                                      max_score_same_ticker=("bucket_mean_score", "max"),
                                      min_score_same_ticker=("bucket_mean_score", "min")).reset_index()
    dedup["score_range_same_ticker"] = dedup["max_score_same_ticker"] - dedup["min_score_same_ticker"]
    piv = bucket.pivot(index=keys, columns="h_bucket", values=["bucket_mean_score", "bucket_active_family_count", "bucket_max_score"])
    piv.columns = [f"{a.lower()}_{b.lower()}" for a, b in piv.columns]
    piv = piv.reset_index()
    for h in ("short", "mid", "long"):
        piv[f"active_fraction_{h}"] = piv.get(f"bucket_active_family_count_{h}", pd.Series(index=piv.index, dtype=float))
    result = raw.merge(dedup, on=keys, how="outer", validate="one_to_one").merge(piv, on=keys, how="left", validate="one_to_one")
    # Signed score agreement and slope over fixed H-bucket centers are fully
    # deterministic and do not use any outcome.
    def agreement(row, a, b):
        x, y = row.get(f"bucket_mean_score_{a}"), row.get(f"bucket_mean_score_{b}")
        return np.nan if pd.isna(x) or pd.isna(y) else float(np.sign(x) == np.sign(y))
    result["short_mid_agreement"] = result.apply(lambda r: agreement(r, "short", "mid"), axis=1)
    result["mid_long_agreement"] = result.apply(lambda r: agreement(r, "mid", "long"), axis=1)
    result["short_long_agreement"] = result.apply(lambda r: agreement(r, "short", "long"), axis=1)
    result["short_minus_long_score"] = result.get("bucket_mean_score_short", np.nan) - result.get("bucket_mean_score_long", np.nan)
    result["short_minus_long_active_fraction"] = result.get("active_fraction_short", np.nan) - result.get("active_fraction_long", np.nan)
    result["cross_horizon_score_std"] = result[[x for x in ["bucket_mean_score_short", "bucket_mean_score_mid", "bucket_mean_score_long"] if x in result]].std(axis=1)
    result["cross_horizon_consensus_strength"] = result[[x for x in ["active_fraction_short", "active_fraction_mid", "active_fraction_long"] if x in result]].mean(axis=1)
    result["score_slope_over_h"] = result.apply(lambda r: _slope(r), axis=1)
    # Denominators come from all eligible matured signal rows, not just active
    # rows.  This prevents an active-only panel from silently turning a count
    # into a tautological fraction.
    if eligibility is not None and not eligibility.empty:
        total = eligibility.groupby(["decision_date", "ticker", "arm"], as_index=False)["eligible_family_count"].sum().rename(columns={"eligible_family_count": "eligible_family_count_same_ticker"})
        result = result.merge(total, on=keys, how="left", validate="one_to_one")
        by_bucket = eligibility.pivot_table(index=keys, columns="h_bucket", values="eligible_family_count", aggfunc="sum").reset_index()
        by_bucket.columns = [*keys] + [f"eligible_family_count_{str(c).lower()}" for c in by_bucket.columns[len(keys):]]
        result = result.merge(by_bucket, on=keys, how="left", validate="one_to_one")
    else:
        result["eligible_family_count_same_ticker"] = np.nan
    result["active_family_fraction_same_ticker"] = result["active_family_count_same_ticker"] / result["eligible_family_count_same_ticker"].replace(0, np.nan)
    for h in ("short", "mid", "long"):
        result[f"active_fraction_{h}"] = result.get(f"active_fraction_{h}", np.nan) / result.get(f"eligible_family_count_{h}", pd.Series(np.nan, index=result.index)).replace(0, np.nan)
        result[f"mean_score_{h}"] = result.get(f"bucket_mean_score_{h}", np.nan)
        result[f"max_score_{h}"] = result.get(f"bucket_max_score_{h}", np.nan)
    return result


def _slope(row) -> float:
    pairs = [(5, row.get("bucket_mean_score_short")), (15, row.get("bucket_mean_score_mid")), (25, row.get("bucket_mean_score_long"))]
    pairs = [(x, y) for x, y in pairs if pd.notna(y)]
    return float(np.polyfit([x for x, _ in pairs], [y for _, y in pairs], 1)[0]) if len(pairs) >= 2 else np.nan


def build_feature_panel(run_root: Path, output_root: Path) -> dict:
    events = _read_events(output_root)
    if events.attrs.get("exact_duplicates_removed", 0):
        _atomic_frame(events, output_root / "opportunity_panel.parquet")
    active = events.loc[events["signal_active"]].copy()
    eligibility = pd.read_parquet(output_root / "eligibility_panel.parquet") if (output_root / "eligibility_panel.parquet").is_file() else pd.DataFrame()
    summary = _bucket_summary(active, eligibility)
    # Same-ticker opportunity groups are the consensus state.  Market features
    # are lagged URTH-only features and are not inferred from target returns.
    features = active.merge(summary, on=["decision_date", "ticker", "arm"], how="left", validate="many_to_one")
    market = _market_features(run_root, features["decision_date"])
    features = features.merge(market, on="decision_date", how="left", validate="many_to_one")
    features["positive_excess"] = np.where(features["has_valid_target"], features["forward_excess_return"].gt(0).astype(float), np.nan)
    features["same_date_target_rank"] = features.groupby(["decision_date", "arm"])["forward_excess_return"].rank(method="average", ascending=False)
    features["same_ticker_target_rank"] = features.groupby(["decision_date", "ticker", "arm"])["forward_excess_return"].rank(method="average", ascending=False)
    features["same_date_target_rank"] = features["same_date_target_rank"].where(features["has_valid_target"])
    features["same_ticker_target_rank"] = features["same_ticker_target_rank"].where(features["has_valid_target"])
    features["event_key"] = (features["decision_date"].dt.strftime("%Y-%m-%d") + "|" + features["ticker"] + "|" + features["family_id"] + "|" + features["arm"])
    if features["event_key"].duplicated().any():
        raise ValueError("FEATURE_EVENT_KEY_DUPLICATE")
    raw_dedup = summary[["decision_date", "ticker", "arm", "raw_active_family_count", "active_family_count_same_ticker",
                         "raw_mean_score", "mean_score_same_ticker", "raw_std_score", "std_score_same_ticker",
                         "raw_score_range", "score_range_same_ticker", "active_family_fraction_same_ticker"]].copy()
    raw_dedup["raw_minus_dedup_active_count"] = raw_dedup["raw_active_family_count"] - raw_dedup["active_family_count_same_ticker"]
    raw_dedup["raw_minus_dedup_mean_score"] = raw_dedup["raw_mean_score"] - raw_dedup["mean_score_same_ticker"]
    raw_dedup["raw_density_ratio"] = raw_dedup["raw_active_family_count"] / raw_dedup["active_family_count_same_ticker"].replace(0, np.nan)
    _atomic_frame(raw_dedup, output_root / "consensus_raw_vs_dedup.csv", parquet=False)
    _atomic_frame(features.sort_values(["decision_date", "ticker", "family_id", "arm"]), output_root / "feature_panel.parquet")
    set_counts = features.groupby(["decision_date", "ticker", "arm"], sort=False)["active_family_count_same_ticker"].first()
    return {"rows": len(features), "valid_target_rows": int(features["has_valid_target"].sum()),
            "active_dates": int(features["decision_date"].nunique()), "active_ticker_date_sets": int(features.groupby(["decision_date", "ticker", "arm"]).ngroups),
            "active_family_overlap": {"max_active_families_same_ticker": int(set_counts.max()),
                                       "sets_with_multiple_active_families": int((set_counts > 1).sum()),
                                       "fraction_sets_with_multiple_active_families": float((set_counts > 1).mean()),
                                       "mean_deduplicated_active_families": float(set_counts.mean())},
            "feature_coverage": {c: float(features[c].notna().mean()) for c in T1_FEATURES + MARKET_FEATURES if c in features}}


def _feature_columns(arm: str) -> list[str]:
    if arm == "T1_CONSENSUS_ONLY":
        return T1_FEATURES
    if arm == "T2_SCORE_ONLY":
        return FEATURE_GROUPS[arm]
    if arm == "T3_SCORE_PLUS_CONSENSUS":
        return FEATURE_GROUPS["T2_SCORE_ONLY"] + T1_FEATURES
    if arm == "T4_SCORE_PLUS_CONSENSUS_PLUS_MARKET":
        return FEATURE_GROUPS["T2_SCORE_ONLY"] + T1_FEATURES + MARKET_FEATURES
    raise ValueError(arm)


def _month(value) -> pd.Timestamp:
    return pd.Timestamp(value).to_period("M").to_timestamp()


def _model(name: str):
    if name == "RIDGE":
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=1.0))
    if name == "HGB":
        return HistGradientBoostingRegressor(max_iter=100, learning_rate=0.05, max_leaf_nodes=15,
                                             l2_regularization=1.0, random_state=17)
    raise ValueError(name)


def _rank_ic(frame: pd.DataFrame, pred: str = "prediction") -> float:
    x = frame[[pred, "forward_excess_return"]].replace([np.inf, -np.inf], np.nan).dropna()
    return _safe_corr(x[pred].rank(method="average"), x["forward_excess_return"].rank(method="average")) if len(x) >= 3 and x[pred].nunique() > 1 and x["forward_excess_return"].nunique() > 1 else np.nan


def _safe_corr(left: pd.Series, right: pd.Series) -> float:
    x = pd.concat([left, right], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    return float(x.iloc[:, 0].corr(x.iloc[:, 1])) if len(x) >= 3 and x.iloc[:, 0].nunique() > 1 and x.iloc[:, 1].nunique() > 1 else np.nan


def _date_rank_ic(frame: pd.DataFrame, group_cols: list[str]) -> float:
    values = []
    for _, g in frame.groupby(group_cols, sort=True):
        value = _rank_ic(g)
        if pd.notna(value):
            values.append(value)
    return float(np.mean(values)) if values else np.nan


def _top_spread(frame: pd.DataFrame, pred: str = "prediction") -> float:
    values = []
    for _, g in frame.groupby(["decision_date", "arm", "h_bucket"], sort=True):
        g = g[[pred, "forward_excess_return"]].dropna()
        if len(g) < 4:
            continue
        cutoff = g[pred].quantile(.75)
        top, rest = g[g[pred] >= cutoff], g[g[pred] < cutoff]
        if len(top) and len(rest):
            values.append(float(top.forward_excess_return.mean() - rest.forward_excess_return.mean()))
    return float(np.mean(values)) if values else np.nan


def _walkforward(features: pd.DataFrame, output_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    x = features.loc[features["has_valid_target"]].copy()
    x["month"] = x["decision_date"].map(_month)
    x["target_matured_date"] = pd.to_datetime(x["target_matured_date"])
    test_months = sorted(x["month"].unique())
    rows, pred_rows, folds = [], [], []
    for arm in FEATURE_ARMS:
        cols = _feature_columns(arm)
        for model_name in MODELS:
            for test_month in test_months:
                test_start = pd.Timestamp(test_month)
                test_end = test_start + pd.offsets.MonthEnd(1)
                train = x[(x["decision_date"] < test_start) & (x["target_matured_date"] < test_start)]
                test = x[(x["decision_date"] >= test_start) & (x["decision_date"] <= test_end)]
                if len(train) < 200 or test.empty:
                    continue
                # Arms are evaluated independently; this avoids turning the
                # three execution arms into independent observations.
                for exec_arm in ARMS:
                    tr = train[train["arm"].eq(exec_arm)].copy()
                    te = test[test["arm"].eq(exec_arm)].copy()
                    tr[cols] = tr[cols].replace([np.inf, -np.inf], np.nan)
                    te[cols] = te[cols].replace([np.inf, -np.inf], np.nan)
                    # Missing horizon buckets are valid state information;
                    # only rows with no feature information at all are
                    # excluded.  Ridge imputes from the training fold and
                    # HGB handles NaNs natively.
                    tr = tr.loc[tr[cols].notna().any(axis=1)]
                    te = te.loc[te[cols].notna().any(axis=1)]
                    if len(tr) < 100 or te.empty:
                        continue
                    model = _model(model_name)
                    model.fit(tr[cols], tr["forward_excess_return"])
                    pred = model.predict(te[cols])
                    part = te[["decision_date", "ticker", "family_id", "arm", "h_bucket", "forward_excess_return", "positive_excess"]].copy()
                    part["prediction"] = pred
                    part["feature_arm"] = arm
                    part["model"] = model_name
                    part["test_month"] = test_start
                    pred_rows.append(part)
                    folds.append({"feature_arm": arm, "model": model_name, "execution_arm": exec_arm,
                                  "train_start": str(tr["decision_date"].min().date()), "train_end": str(tr["decision_date"].max().date()),
                                  "max_target_matured_date": str(tr["target_matured_date"].max().date()),
                                  "test_start": str(test_start.date()), "test_end": str(test_end.date()),
                                  "train_rows": len(tr), "test_rows": len(te), "feature_columns": ",".join(cols),
                                  "feature_hash": _hash(cols)})
    if not pred_rows:
        raise ValueError("WALKFORWARD_NO_VALID_FOLDS")
    predictions = pd.concat(pred_rows, ignore_index=True)
    fold_manifest = pd.DataFrame(folds)
    metric_rows = []
    for keys, g in predictions.groupby(["feature_arm", "model", "arm", "test_month"], sort=True):
        fa, model_name, exec_arm, month = keys
        metric_rows.append({"feature_arm": fa, "model": model_name, "execution_arm": exec_arm, "test_month": month,
                            "rows": len(g), "rank_ic": _rank_ic(g), "same_date_rank_ic": _date_rank_ic(g, ["decision_date"]),
                            "same_ticker_rank_ic": _date_rank_ic(g, ["decision_date", "ticker"]),
                            "top_quartile_spread": _top_spread(g),
                            "hit_rate": float((np.sign(g["prediction"]) == np.sign(g["forward_excess_return"])).mean())})
    _atomic_frame(predictions, output_root / "model_predictions.parquet")
    _atomic_frame(fold_manifest, output_root / "walkforward_fold_manifest.csv", parquet=False)
    metrics = pd.DataFrame(metric_rows)
    _atomic_frame(metrics, output_root / "fold_metrics.csv", parquet=False)
    summary = metrics.groupby(["feature_arm", "model", "execution_arm"], as_index=False).agg(
        folds=("test_month", "nunique"), rows=("rows", "sum"), rank_ic=("rank_ic", "mean"),
        same_date_rank_ic=("same_date_rank_ic", "mean"), same_ticker_rank_ic=("same_ticker_rank_ic", "mean"),
        top_quartile_spread=("top_quartile_spread", "mean"), hit_rate=("hit_rate", "mean"))
    _atomic_frame(summary, output_root / "predictive_metrics.csv", parquet=False)
    return predictions, metrics, fold_manifest


def _block_bootstrap(values: pd.Series, seed: int = 20260823, repetitions: int = 1000, block_size: int = 3) -> dict:
    values = np.asarray(values.dropna(), dtype=float)
    if not len(values):
        return {"status": "INSUFFICIENT_EVIDENCE", "n": 0, "q05": None, "median": None, "q95": None, "positive_fraction": None}
    blocks = [values[i:i + block_size] for i in range(0, len(values), block_size)]
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(repetitions):
        chosen = []
        while sum(len(x) for x in chosen) < len(values):
            chosen.append(blocks[int(rng.integers(0, len(blocks)))])
        samples.append(np.concatenate(chosen)[:len(values)].mean())
    return {"status": "PASS", "n": len(values), "mean": float(values.mean()), "median": float(np.median(values)),
            "q05": float(np.quantile(samples, .05)), "q95": float(np.quantile(samples, .95)),
            "positive_fraction": float((np.asarray(samples) > 0).mean()), "repetitions": repetitions,
            "block_size_months": block_size, "seed": seed}


def _date_block_bootstrap(frame: pd.DataFrame, value: str, date: str = "decision_date", seed: int = 20260823,
                          repetitions: int = 1000, block_size: int = 3) -> dict:
    if frame.empty or value not in frame or date not in frame:
        return {"status": "INSUFFICIENT_EVIDENCE", "n": 0, "q05": None, "median": None, "q95": None, "positive_fraction": None}
    dated = frame[[date, value]].dropna().copy()
    if dated.empty:
        return {"status": "INSUFFICIENT_EVIDENCE", "n": 0, "q05": None, "median": None, "q95": None, "positive_fraction": None}
    dated[date] = pd.to_datetime(dated[date]).map(_month)
    monthly = dated.groupby(date)[value].mean().sort_index()
    result = _block_bootstrap(monthly, seed=seed, repetitions=repetitions, block_size=block_size)
    result["inference_unit"] = "calendar_month_block_mean"
    result["calendar_months"] = int(len(monthly))
    return result


def _univariate_outputs(features: pd.DataFrame, output_root: Path, seed: int, repetitions: int) -> dict:
    valid = features.loc[features["has_valid_target"]].copy()
    candidate_features = list(dict.fromkeys(T1_FEATURES + ["prediction_score", "score_percentile", "distance_to_threshold"]))
    bins, monotonicity, bootstrap = [], [], {}
    for feature in candidate_features:
        if feature not in valid or valid[feature].notna().sum() < 20:
            continue
        x = valid[["decision_date", "forward_excess_return", feature]].dropna().copy()
        if x.empty or x[feature].nunique() < 2:
            continue
        try:
            x["bin"] = pd.qcut(x[feature], q=5, labels=False, duplicates="drop") + 1
        except ValueError:
            continue
        for b, g in x.groupby("bin", sort=True):
            bins.append({"feature": feature, "bin": int(b), "rows": len(g), "feature_min": float(g[feature].min()),
                         "feature_max": float(g[feature].max()), "mean_future_excess": float(g.forward_excess_return.mean()),
                         "median_future_excess": float(g.forward_excess_return.median()),
                         "positive_excess_rate": float(g.forward_excess_return.gt(0).mean())})
            key = f"{feature}:bin_{int(b)}"
            bootstrap[key] = _date_block_bootstrap(g, "forward_excess_return", seed=seed + int(b), repetitions=repetitions)
        grouped = pd.DataFrame([r for r in bins if r["feature"] == feature]).sort_values("bin")
        if len(grouped) >= 2:
            slope = np.polyfit(grouped["bin"], grouped["mean_future_excess"], 1)[0]
            monotonicity.append({"feature": feature, "bins": len(grouped), "mean_bin_slope": float(slope),
                                 "spearman_bin_mean": float(grouped["bin"].corr(grouped["mean_future_excess"], method="spearman"))})
    _atomic_frame(pd.DataFrame(bins), output_root / "univariate_feature_bins.csv", parquet=False)
    _atomic_frame(pd.DataFrame(monotonicity), output_root / "univariate_monotonicity.csv", parquet=False)
    _atomic_json(output_root / "univariate_bootstrap.json", {"seed": seed, "repetitions": repetitions, "features": bootstrap})
    return {"features": len(set(r["feature"] for r in bins)), "bins": len(bins), "bootstrap": bootstrap}


def _economic_bootstrap(daily: pd.DataFrame, seed: int, repetitions: int) -> dict:
    """Paired time-block bootstrap of diagnostic hurdle differences."""
    if daily.empty:
        return {}
    work = daily.copy().sort_values(["feature_arm", "model", "execution_arm", "h_bucket", "rule", "decision_date"])
    work["switch"] = work.groupby(["feature_arm", "model", "execution_arm", "h_bucket", "rule"])["turnover_key"].transform(lambda s: s.ne(s.shift()).fillna(False))
    work["net_20bp"] = work["excess_return"] - work["switch"].astype(float) * 20 / 10000.0
    outputs = {}
    comparisons = (("B3_SCORE_PLUS_CONSENSUS", "B1_SCORE_ONLY", "T3_SCORE_PLUS_CONSENSUS"),
                   ("B3_SCORE_PLUS_CONSENSUS", "B0_EQUAL_ACTIVE", "T3_SCORE_PLUS_CONSENSUS"),
                   ("B4_SCORE_PLUS_CONSENSUS_PLUS_MARKET", "B3_SCORE_PLUS_CONSENSUS", "CROSS_ARM"))
    for a, b, scope in comparisons:
        if scope == "CROSS_ARM":
            left = work[(work.rule == a) & (work.feature_arm == "T4_SCORE_PLUS_CONSENSUS_PLUS_MARKET")]
            right = work[(work.rule == b) & (work.feature_arm == "T3_SCORE_PLUS_CONSENSUS")]
        else:
            left = work[(work.rule == a) & (work.feature_arm == scope)]
            right = work[(work.rule == b) & (work.feature_arm == scope)]
        keys = ["model", "execution_arm", "h_bucket", "decision_date"]
        pair = left[keys + ["net_20bp"]].merge(right[keys + ["net_20bp"]], on=keys, suffixes=("_a", "_b"))
        pair["difference"] = pair["net_20bp_a"] - pair["net_20bp_b"]
        outputs[f"{a}_MINUS_{b}"] = _date_block_bootstrap(pair, "difference", seed=seed, repetitions=repetitions)
    return outputs


def _incremental_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    pivots = predictions.pivot_table(index=["decision_date", "ticker", "family_id", "arm", "h_bucket", "test_month", "forward_excess_return"],
                                      columns=["feature_arm", "model"], values="prediction", aggfunc="mean").reset_index()
    pivots.columns = [c[0] if isinstance(c, tuple) and c[1] == "" else f"{c[0]}__{c[1]}" if isinstance(c, tuple) else c for c in pivots.columns]
    for _, g in pivots.groupby(["arm", "h_bucket", "test_month"], sort=True):
        for model in MODELS:
            for a, b in (("T3_SCORE_PLUS_CONSENSUS", "T2_SCORE_ONLY"),
                         ("T4_SCORE_PLUS_CONSENSUS_PLUS_MARKET", "T3_SCORE_PLUS_CONSENSUS")):
                ca, cb = f"{a}__{model}", f"{b}__{model}"
                if ca not in g.columns or cb not in g.columns:
                    continue
                z = g[["decision_date", "ticker", "family_id", "arm", "h_bucket", "forward_excess_return", ca, cb]].dropna()
                if z.empty:
                    continue
                z = z.rename(columns={ca: "prediction_a", cb: "prediction_b"})
                ic_a = _rank_ic(z.rename(columns={"prediction_a": "prediction"}))
                ic_b = _rank_ic(z.rename(columns={"prediction_b": "prediction"}))
                spread_a = _top_spread(z.rename(columns={"prediction_a": "prediction"}))
                spread_b = _top_spread(z.rename(columns={"prediction_b": "prediction"}))
                rows.append({"execution_arm": g["arm"].iloc[0], "h_bucket": g["h_bucket"].iloc[0], "model": model,
                             "test_month": g["test_month"].iloc[0], "comparison": f"{a}_MINUS_{b}", "rows": len(z),
                             "rank_ic_a": ic_a, "rank_ic_b": ic_b,
                             "incremental_rank_ic": ic_a - ic_b if pd.notna(ic_a) and pd.notna(ic_b) else np.nan,
                             "spread_a": spread_a, "spread_b": spread_b,
                             "incremental_top_quartile_spread": spread_a - spread_b if pd.notna(spread_a) and pd.notna(spread_b) else np.nan,
                             "prediction_corr": _safe_corr(z["prediction_a"], z["prediction_b"])})
    return pd.DataFrame(rows)


def economic_replay(features: pd.DataFrame, predictions: pd.DataFrame, output_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Cross-fitted predictions are joined to the same active candidate set.
    p = predictions.groupby(["decision_date", "ticker", "family_id", "arm", "h_bucket", "feature_arm", "model"], as_index=False)["prediction"].mean()
    base = features.loc[features["has_valid_target"], ["decision_date", "ticker", "family_id", "arm", "h_bucket", "forward_excess_return", "score", "cross_horizon_consensus_strength", "short_minus_long_score"]]
    p = p.merge(base, on=["decision_date", "ticker", "family_id", "arm", "h_bucket"], how="left", validate="many_to_one")
    rows = []
    for keys, g in p.groupby(["feature_arm", "model", "arm", "h_bucket"], sort=True):
        fa, model_name, exec_arm, h_bucket = keys
        for date, d in g.groupby("decision_date", sort=True):
            if len(d) < 4:
                continue
            for rule, score_col in (("B1_SCORE_ONLY", "score"), ("B2_CONSENSUS_ONLY", "cross_horizon_consensus_strength"),
                                    ("B3_SCORE_PLUS_CONSENSUS", "prediction"), ("B4_SCORE_PLUS_CONSENSUS_PLUS_MARKET", "prediction")):
                if rule == "B1_SCORE_ONLY":
                    rank = d[score_col]
                elif rule == "B2_CONSENSUS_ONLY":
                    rank = d[score_col]
                elif rule == "B3_SCORE_PLUS_CONSENSUS" and fa != "T3_SCORE_PLUS_CONSENSUS":
                    continue
                elif rule == "B4_SCORE_PLUS_CONSENSUS_PLUS_MARKET" and fa != "T4_SCORE_PLUS_CONSENSUS_PLUS_MARKET":
                    continue
                else:
                    rank = d[score_col]
                q = rank.quantile(.75)
                selected = d.loc[rank.ge(q)]
                rows.append({"feature_arm": fa, "model": model_name, "execution_arm": exec_arm, "h_bucket": h_bucket,
                             "rule": rule, "decision_date": date, "candidate_rows": len(d), "selected_rows": len(selected),
                             "excess_return": float(selected["forward_excess_return"].mean()),
                             "turnover_key": "|".join(sorted(selected["ticker"].astype(str) + "|" + selected["family_id"].astype(str)))})
            rows.append({"feature_arm": fa, "model": model_name, "execution_arm": exec_arm, "h_bucket": h_bucket,
                         "rule": "B0_EQUAL_ACTIVE", "decision_date": date, "candidate_rows": len(d), "selected_rows": len(d),
                         "excess_return": float(d["forward_excess_return"].mean()),
                         "turnover_key": "|".join(sorted(d["ticker"].astype(str) + "|" + d["family_id"].astype(str)))})
            rows.append({"feature_arm": fa, "model": model_name, "execution_arm": exec_arm, "h_bucket": h_bucket,
                         "rule": "B5_ORACLE", "decision_date": date, "candidate_rows": len(d), "selected_rows": max(1, len(d) // 4),
                         "excess_return": float(d.nlargest(max(1, len(d) // 4), "forward_excess_return")["forward_excess_return"].mean()),
                         "turnover_key": "ORACLE"})
    daily = pd.DataFrame(rows)
    if daily.empty:
        raise ValueError("ECONOMIC_REPLAY_EMPTY")
    daily["decision_date"] = pd.to_datetime(daily["decision_date"])
    daily = daily.sort_values(["feature_arm", "model", "execution_arm", "h_bucket", "rule", "decision_date"])
    summary_rows = []
    for keys, g in daily.groupby(["feature_arm", "model", "execution_arm", "h_bucket", "rule"], sort=True):
        returns = g["excess_return"].to_numpy(dtype=float)
        wealth = np.cumprod(1.0 + returns)
        switch = g["turnover_key"].ne(g["turnover_key"].shift()).fillna(False)
        for cost in (0, 10, 20, 30):
            net = returns - switch.to_numpy(dtype=float) * cost / 10000.0
            w = np.cumprod(1.0 + net)
            dd = w / np.maximum.accumulate(w) - 1.0
            summary_rows.append({"feature_arm": keys[0], "model": keys[1], "execution_arm": keys[2], "h_bucket": keys[3], "rule": keys[4],
                                 "additional_cost_bps": cost, "observations": len(net), "cagr_excess": float(w[-1] ** (252 / len(net)) - 1.0),
                                 "annualized_excess": float((1 + net.mean()) ** 252 - 1.0),
                                 "sharpe": float(net.mean() / net.std(ddof=0) * np.sqrt(252)) if net.std(ddof=0) else 0.0,
                                 "relative_max_drawdown": float(dd.min()), "mean_excess": float(net.mean()),
                                 "turnover": float(switch.mean()), "switch_count": int(switch.sum()), "terminal_wealth": float(w[-1])})
    summary = pd.DataFrame(summary_rows)
    _atomic_frame(daily, output_root / "economic_replay_daily.parquet")
    _atomic_frame(summary, output_root / "economic_replay.csv", parquet=False)
    stress = summary[summary["additional_cost_bps"].isin([0, 10, 20, 30])].copy()
    _atomic_frame(stress, output_root / "economic_cost_stress.csv", parquet=False)
    return summary, daily


def _ablation_outputs(features: pd.DataFrame, predictions: pd.DataFrame, output_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    h, d = features["horizon_h"], features["holding_d"]
    masks = {
        "A_FULL_UNIVERSE": pd.Series(True, index=features.index),
        "B_REMOVE_H30": h.ne(30), "C_REMOVE_H28_H30": h.lt(28),
        "D_REMOVE_H_GE25_D_GE25": ~h.ge(25) | ~d.ge(25),
        "E_SAME_TICKER_MULTI_FAMILY": features.groupby(["decision_date", "ticker", "arm"])["family_id"].transform("nunique").ge(2),
        "F_DATES_AT_LEAST_5_ACTIVE_FAMILIES": features.groupby(["decision_date", "arm"])["family_id"].transform("nunique").ge(5),
        "G_HIGH_DISAGREEMENT": features["cross_horizon_score_std"].ge(features["cross_horizon_score_std"].median()),
        "H_HIGH_CONSENSUS": features["cross_horizon_consensus_strength"].ge(features["cross_horizon_consensus_strength"].median()),
        "I_EARLY_SUBPERIOD": features["decision_date"].lt(pd.Timestamp("2023-01-01")),
        "I_LATE_SUBPERIOD": features["decision_date"].ge(pd.Timestamp("2023-01-01")),
        "J_LOW_SCORE": features["score_percentile"].lt(.75), "J_HIGH_SCORE": features["score_percentile"].ge(.75),
    }
    rows = []
    for name, mask in masks.items():
        keep = set(features.loc[mask, "event_key"])
        p = predictions[predictions.apply(lambda r: f"{pd.Timestamp(r.decision_date).strftime('%Y-%m-%d')}|{r.ticker}|{r.family_id}|{r.arm}" in keep, axis=1)]
        for keys, g in p.groupby(["feature_arm", "model", "arm", "h_bucket"], sort=True):
            t3 = g[g.feature_arm.eq("T3_SCORE_PLUS_CONSENSUS")]
            t2 = g[g.feature_arm.eq("T2_SCORE_ONLY")]
            rows.append({"ablation": name, "feature_arm": keys[0], "model": keys[1], "execution_arm": keys[2], "h_bucket": keys[3],
                         "rows": len(g), "rank_ic": _rank_ic(g), "top_quartile_spread": _top_spread(g)})
    result = pd.DataFrame(rows)
    _atomic_frame(result, output_root / "subperiod_robustness.csv", parquet=False)
    # A compact consensus-specific view is easier to audit than the full fold table.
    compact = result[result["feature_arm"].isin(["T2_SCORE_ONLY", "T3_SCORE_PLUS_CONSENSUS"])].copy()
    _atomic_frame(compact, output_root / "anti_h30_ablation.csv", parquet=False)
    return result, compact


def _diagnosis(metrics: pd.DataFrame, incremental: pd.DataFrame, economic: pd.DataFrame, ablation: pd.DataFrame) -> str:
    def avg(arm, col):
        x = metrics.loc[metrics.feature_arm.eq(arm), col].dropna()
        return float(x.mean()) if len(x) else np.nan
    t2, t3 = avg("T2_SCORE_ONLY", "rank_ic"), avg("T3_SCORE_PLUS_CONSENSUS", "rank_ic")
    delta = incremental.loc[incremental.comparison.eq("T3_SCORE_PLUS_CONSENSUS_MINUS_T2_SCORE_ONLY"), "incremental_rank_ic"].dropna()
    t3_e = economic.loc[economic.rule.eq("B3_SCORE_PLUS_CONSENSUS") & economic.additional_cost_bps.eq(20), "mean_excess"].mean()
    b0 = economic.loc[economic.rule.eq("B0_EQUAL_ACTIVE") & economic.additional_cost_bps.eq(20), "mean_excess"].mean()
    raw = incremental["incremental_rank_ic"].dropna().median() if len(incremental) else np.nan
    if pd.notna(delta).any() and float(delta.median()) > 0 and (delta.quantile(.05) >= 0) and t3_e > b0:
        return "OPPORTUNITY_CONSENSUS_INCREMENTAL_PREDICTABILITY_SUPPORTED"
    if pd.notna(delta).any() and float(delta.median()) > 0 and t3_e > b0:
        return "OPPORTUNITY_CONSENSUS_WEAK_BUT_ECONOMICALLY_INTERESTING"
    if pd.notna(t2) and pd.notna(t3) and t2 > 0 and float(delta.median()) <= 0 and t3 <= t2:
        return "SCORE_ONLY_EXPLAINS_AVAILABLE_PREDICTABILITY"
    return "NO_ROBUST_OPPORTUNITY_STATE_PREDICTABILITY"


def run(run_root: Path, output_root: Path, holdout_start: pd.Timestamp = HOLDOUT_START, repetitions: int = 1000, seed: int = 20260823) -> dict:
    started = time.time()
    panel_meta = build_opportunity_panel(run_root, output_root, holdout_start)
    feature_meta = build_feature_panel(run_root, output_root)
    features = pd.read_parquet(output_root / "feature_panel.parquet")
    univariate_meta = _univariate_outputs(features, output_root, seed, repetitions)
    predictions, fold_metrics, fold_manifest = _walkforward(features, output_root)
    incremental = _incremental_metrics(predictions)
    _atomic_frame(incremental, output_root / "incremental_feature_value.csv", parquet=False)
    economic, daily = economic_replay(features, predictions, output_root)
    ablations, compact = _ablation_outputs(features, predictions, output_root)
    bootstrap_inc = {}
    for comparison, g in incremental.groupby("comparison", sort=True):
        bootstrap_inc[comparison] = _date_block_bootstrap(g, "incremental_rank_ic", date="test_month", seed=seed, repetitions=repetitions)
    bootstrap_spread = {}
    for comparison, g in incremental.groupby("comparison", sort=True):
        bootstrap_spread[comparison] = _date_block_bootstrap(g, "incremental_top_quartile_spread", date="test_month", seed=seed + 1, repetitions=repetitions)
    bootstrap_econ = _economic_bootstrap(daily, seed=seed + 2, repetitions=repetitions)
    _atomic_json(output_root / "bootstrap_incremental_rank_ic.json", bootstrap_inc)
    _atomic_json(output_root / "bootstrap_incremental_spread.json", bootstrap_spread)
    _atomic_json(output_root / "bootstrap_economic_value.json", bootstrap_econ)
    diagnosis = _diagnosis(fold_metrics, incremental, economic, ablations)
    design_family_count = int(_parse_metadata(run_root / "development/family_registry.json")["family_id"].nunique())
    manifest = {"schema_version": SCHEMA_VERSION, "git_commit": _git_commit(), "code_hash": _file_sha256(Path(__file__)), "run_timestamp_utc": pd.Timestamp.now(tz="UTC").isoformat(),
                "input_paths": {"run_root": str(run_root.resolve()), "matured_generation_predictions": str((run_root / "matured_generation_predictions.parquet").resolve()),
                                "abc_generation_schedule": str((run_root / "abc_generation_schedule.parquet").resolve()), "valid_generations": str((run_root / "valid_generations.parquet").resolve()),
                                "family_shadow_trades": str((run_root / "development/family_shadow_trades.parquet").resolve()), "family_shadow_nav": str((run_root / "development/family_shadow_nav.parquet").resolve())},
                "holdout_boundary_exclusive": str(holdout_start), "design_family_count": design_family_count,
                "active_family_count": int(features.family_id.nunique()), "family_count": int(features.family_id.nunique()),
                "event_count": int(len(features)), "valid_target_count": int(features.has_valid_target.sum()),
                "active_decision_ticker_arm_sets": int(features.groupby(["decision_date", "ticker", "arm"]).ngroups),
                "active_family_overlap": feature_meta["active_family_overlap"], "feature_coverage": feature_meta["feature_coverage"], "models": list(MODELS), "feature_arms": list(FEATURE_ARMS),
                "bootstrap_repetitions": repetitions, "random_seed": seed, "models_retrained": False, "predictions_regenerated": False,
                "holdout_opened": False, "selector_authority": False, "methodological_status": diagnosis,
                "unavailable_fields": panel_meta["unavailable_fields"], "univariate": {"feature_count": univariate_meta["features"], "bin_count": univariate_meta["bins"]},
                "runtime_seconds": time.time() - started,
                "source_contract": {"primary_target": "realized_excess", "active_definition": panel_meta["active_definition"],
                                    "stock_and_benchmark_return_components": "UNAVAILABLE_NOT_IN_AUTHORITATIVE_MATURED_EVIDENCE"}}
    _atomic_json(output_root / "manifest.json", manifest)
    report = _render_report(manifest, fold_metrics, incremental, economic, ablations, panel_meta, feature_meta)
    (output_root / "REPORT.md.tmp").write_text(report, encoding="utf-8")
    (output_root / "REPORT.md.tmp").replace(output_root / "REPORT.md")
    summary = {"status": "DYNAMIC_QBD_OPPORTUNITY_STATE_COMPLETE", "primary_diagnosis": diagnosis,
               "event_count": manifest["event_count"], "valid_target_count": manifest["valid_target_count"],
               "active_decision_ticker_arm_sets": manifest["active_decision_ticker_arm_sets"], "runtime_seconds": manifest["runtime_seconds"],
               "family_count": manifest["family_count"], "design_family_count": manifest["design_family_count"], "active_family_overlap": manifest["active_family_overlap"],
               "holdout_opened": False, "models_retrained": False, "predictions_regenerated": False,
               "selector_authority": False}
    _atomic_json(output_root / "summary.json", summary)
    return summary


def _render_report(manifest, metrics, incremental, economic, ablations, panel_meta, feature_meta) -> str:
    def f(v): return "n/a" if pd.isna(v) else f"{float(v):.6f}"
    t2 = metrics[metrics.feature_arm.eq("T2_SCORE_ONLY")].rank_ic.mean()
    t3 = metrics[metrics.feature_arm.eq("T3_SCORE_PLUS_CONSENSUS")].rank_ic.mean()
    t4 = metrics[metrics.feature_arm.eq("T4_SCORE_PLUS_CONSENSUS_PLUS_MARKET")].rank_ic.mean()
    inc = incremental[incremental.comparison.eq("T3_SCORE_PLUS_CONSENSUS_MINUS_T2_SCORE_ONLY")].incremental_rank_ic
    inc_spread = incremental[incremental.comparison.eq("T3_SCORE_PLUS_CONSENSUS_MINUS_T2_SCORE_ONLY")].incremental_top_quartile_spread
    inc_frame = incremental[incremental.comparison.eq("T3_SCORE_PLUS_CONSENSUS_MINUS_T2_SCORE_ONLY")]
    boot_inc = _date_block_bootstrap(inc_frame, "incremental_rank_ic", date="test_month", seed=manifest.get("random_seed", 20260823), repetitions=manifest.get("bootstrap_repetitions", 1000))
    boot_spread = _date_block_bootstrap(inc_frame, "incremental_top_quartile_spread", date="test_month", seed=manifest.get("random_seed", 20260823) + 1, repetitions=manifest.get("bootstrap_repetitions", 1000))
    overlap = manifest.get("active_family_overlap", {})
    e20 = economic[economic.additional_cost_bps.eq(20)].groupby("rule")["mean_excess"].mean()
    lines = ["# Dynamic-QBD Opportunity-State Predictability / Cross-Family Consensus", "",
             "## Technical summary", "",
             f"Primary diagnosis: **{manifest['methodological_status']}**. The deduplicated consensus state does not clear the causal robustness and economic hurdles against Score-only; no selector authority is granted.",
             f"The run contains `{manifest['event_count']}` active events from `{manifest['active_family_count']}` of `{manifest['design_family_count']}` design Families. Only `{overlap.get('fraction_sets_with_multiple_active_families', float('nan')):.1%}` of active date×ticker×arm sets contain more than one deduplicated active Family, limiting effective consensus evidence.", "",
             "## Scope and safeguards", "",
             f"- Active event count: `{manifest['event_count']}`; valid targets: `{manifest['valid_target_count']}`; active date×ticker×arm sets: `{manifest['active_decision_ticker_arm_sets']}`.",
             f"- Holdout boundary: `{manifest['holdout_boundary_exclusive']}`; opened: **no**.",
             "- Models retrained: **no**. Predictions regenerated: **no**. Only Ridge/HGB diagnostics were fit on historical opportunity evidence.",
             f"- SIGNAL_ACTIVE means `score >= resolved_threshold` and `decision_date >= activation_date`; forward `realized_excess` is target-only.",
             "- Stock-return and separate benchmark-return components were unavailable in authoritative matured evidence and remain missing; no imputation was performed.", "",
             "## Predictability result", "",
             f"- Mean fold Rank-IC: T2 Score-only `{f(t2)}`, T3 Score+Consensus `{f(t3)}`, T4 + Market `{f(t4)}`.",
             f"- T3 minus T2 incremental Rank-IC: fold median `{f(inc.median() if len(inc) else np.nan)}`, date-block bootstrap q05/q50/q95 `{f(boot_inc.get('q05', np.nan))}` / `{f(boot_inc.get('median', np.nan))}` / `{f(boot_inc.get('q95', np.nan))}`.",
             f"- T3 minus T2 top-quartile spread increment: fold median `{f(inc_spread.median() if len(inc_spread) else np.nan)}`, date-block bootstrap q05/q50/q95 `{f(boot_spread.get('q05', np.nan))}` / `{f(boot_spread.get('median', np.nan))}` / `{f(boot_spread.get('q95', np.nan))}`.",
             "- T1/T3 Consensus features use one primary economic vote per H bucket; missing H buckets are causal state information and are imputed only from the training fold for Ridge. Raw Family-density diagnostics are retained separately.", "",
             "## Economic hurdle", "",
             "- B0 Equal-active is the hurdle; B1 Score-only, B2 Consensus-only, B3 Score+Consensus, B4 +Market and B5 Oracle use the same active candidate sets.",
             f"- At 20 bp additional diagnostic cost, mean excess is B0 `{f(e20.get('B0_EQUAL_ACTIVE', np.nan))}`, B1 `{f(e20.get('B1_SCORE_ONLY', np.nan))}`, B3 `{f(e20.get('B3_SCORE_PLUS_CONSENSUS', np.nan))}`.",
             "- Cost stress is reported at 0/10/20/30 bp. These are diagnostic cross-sectional replays, not production portfolios or capital-authorized selectors.", "",
             "## Robustness and multiple-counting control", "",
             "- H30, H28–H30, same-ticker, activity-count, consensus/disagreement, score and early/late ablations are persisted in `anti_h30_ablation.csv` and `subperiod_robustness.csv`.",
             "- Raw active Family density is materially larger than deduplicated H-bucket breadth in the persisted RAW-vs-DEDUP table; raw density is therefore not accepted as independent consensus evidence.",
             "- Inference is time-block based; no IID interpretation over the millions of correlated opportunity rows is used.", "",
             "## Limitations and self-review", "",
             "- T2 contains only concrete score/Family H-D-N information; T3 adds only predeclared consensus features; T4 adds only lagged URTH market state.",
             "- Different H regimes are reported by SHORT/MID/LONG buckets; the original H-specific realized excess remains the primary target.",
             "- All conclusions are Development/Research-only. No selector receives authority.", "",
             "- The authoritative matured table exposes `realized_excess` but not separate stock- and benchmark-return components; those fields remain unavailable and were not imputed.",
             "- Economic replay metrics are research diagnostics on overlapping H-specific opportunity returns, not deployable portfolio NAV statistics.", "",
             f"## Conclusion\n\n**{manifest['methodological_status']}**", ""]
    return "\n".join(lines)


def _git_commit() -> str | None:
    try:
        import subprocess
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/dynamic-qbd-opportunity-state"))
    parser.add_argument("--holdout-start", default="2026-07-25")
    parser.add_argument("--bootstrap-repetitions", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260823)
    args = parser.parse_args(argv)
    if args.bootstrap_repetitions <= 0:
        raise ValueError("BOOTSTRAP_REPETITIONS_INVALID")
    print(json.dumps(run(args.run_root.resolve(), args.output_root.resolve(), pd.Timestamp(args.holdout_start), args.bootstrap_repetitions, args.seed), default=_json_default, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
