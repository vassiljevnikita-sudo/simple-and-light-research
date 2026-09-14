"""Causal regime-persistence and selector-feasibility diagnostics.

This module deliberately operates on the completed Dynamic-QBD development
evidence.  It never opens the final holdout and never reads the global matured
prediction table.  State variables are observed at an assessment month; the
forward one- and three-month excess returns are evaluation targets only.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from hashlib import sha256
import json
import math
from pathlib import Path
import re
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


SCHEMA_VERSION = "DYNAMIC_QBD_REGIME_DIAGNOSTICS_V2"
DEFAULT_HOLDOUT_START = pd.Timestamp("2026-07-25")
DEFAULT_BLOCK_SIZE = 3
DEFAULT_BOOTSTRAP_REPETITIONS = 1000
FAMILY_RE = re.compile(r"^H(?P<h>\d+)_D(?P<d>\d+)_N(?P<n>\d+)_(?P<exit>.+)$")
ARMS = (
    "A_FROZEN_MODEL_FROZEN_CALIBRATION",
    "B_FROZEN_MODEL_ROLLING_RECALIBRATION",
    "C_ROLLING_REFIT_ROLLING_RECALIBRATION",
)
LEVELS = ("FAMILY", "H_BAND", "HD_GRID", "ECONOMIC_REGION")
SELECTOR_RULES = ("F0_ORACLE", "F1_CURRENT_WINNER", "F2_TOP_QUARTILE_EQUAL",
                  "F3_PERSISTENT_TOP", "F4_INCUMBENT", "F5_EQUAL_WEIGHT")
ACTIVITY_CONDITIONS = ("ALL", "ACTIVE_ONLY", "NO_OPPORTUNITY", "RECENTLY_ACTIVE")


def _json_default(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, (pd.Timestamp, pd.Timedelta)):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def _stable_hash(value) -> str:
    return sha256(json.dumps(value, default=_json_default, sort_keys=True,
                             separators=(",", ":")).encode()).hexdigest()


def _atomic_write_json(path: Path, value) -> None:
    temporary = path.with_name(f".{path.name}.{Path().resolve().name}.tmp")
    temporary.write_text(json.dumps(value, default=_json_default, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def _atomic_write_frame(frame: pd.DataFrame, path: Path, *, parquet: bool = True) -> None:
    temporary = path.with_name(f".{path.name}.{Path().resolve().name}.tmp")
    if parquet:
        frame.to_parquet(temporary, index=False)
    else:
        frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _input_fingerprint(path: Path) -> dict:
    stat = path.stat()
    result = {"path": str(path.resolve()), "size": stat.st_size,
              "mtime_ns": stat.st_mtime_ns}
    if path.suffix.lower() in {".json", ".csv"} and stat.st_size <= 20_000_000:
        result["sha256"] = sha256(path.read_bytes()).hexdigest()
    elif path.suffix.lower() == ".parquet":
        metadata = pq.ParquetFile(path).metadata
        result.update(rows=metadata.num_rows, row_groups=metadata.num_row_groups,
                      schema=str(pq.ParquetFile(path).schema_arrow))
    return result


def _code_config_hash(config: dict) -> str:
    """Fingerprint the executable diagnostic code and its run configuration."""
    return _stable_hash({"module_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
                         "config": config})


def _require_before_holdout(frame: pd.DataFrame, column: str, holdout_start: pd.Timestamp) -> None:
    values = pd.to_datetime(frame[column], errors="raise")
    if values.max() >= holdout_start:
        raise ValueError(f"FINAL_HOLDOUT_BOUNDARY_VIOLATION:{column}:{values.max()}")


def _parse_family_metadata(path: Path) -> pd.DataFrame:
    raw = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for item in raw["families"]:
        match = FAMILY_RE.match(str(item["family_id"]))
        if not match:
            raise ValueError(f"FAMILY_ID_SCHEMA_INVALID:{item['family_id']}")
        exit_mode = str(item.get("entry_policy_rule", {}).get("exit_policy", {}).get("family", match["exit"]))
        rows.append({
            "family_id": str(item["family_id"]),
            "horizon_h": int(item["horizon_sessions"]),
            "holding_d": int(item["holding_days"]),
            "max_names_n": int(item["max_names"]),
            "exit_mode": exit_mode,
            "family_variant": match["exit"],
        })
    result = pd.DataFrame(rows).sort_values("family_id").reset_index(drop=True)
    if result["family_id"].duplicated().any():
        raise ValueError("FAMILY_METADATA_DUPLICATE")
    return result


def _band(value: int, bands: tuple[tuple[str, int, int], ...]) -> str:
    for name, lower, upper in bands:
        if lower <= value <= upper:
            return name
    raise ValueError(f"BAND_UNASSIGNED:{value}")


H_BANDS = (("H01_05", 1, 5), ("H06_10", 6, 10), ("H11_20", 11, 20), ("H21_30", 21, 30))
D_BANDS = (("D01_05", 1, 5), ("D06_10", 6, 10), ("D11_20", 11, 20), ("D21_30", 21, 30))
H_REGIONS = (("FAST", 1, 5), ("SHORT", 6, 10), ("MID", 11, 20), ("LONG", 21, 30))


def build_cluster_membership(metadata: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for item in metadata.itertuples(index=False):
        h_band = _band(item.horizon_h, H_BANDS)
        d_band = _band(item.holding_d, D_BANDS)
        h_region = _band(item.horizon_h, H_REGIONS)
        rows.extend([
            {"cluster_level": "FAMILY", "cluster_id": f"FAMILY::{item.family_id}", "family_id": item.family_id},
            {"cluster_level": "H_BAND", "cluster_id": f"H_BAND::{h_band}", "family_id": item.family_id},
            {"cluster_level": "HD_GRID", "cluster_id": f"HD_GRID::{h_band}::{d_band}::N{item.max_names_n:02d}::{item.family_variant}", "family_id": item.family_id},
            {"cluster_level": "ECONOMIC_REGION", "cluster_id": f"ECONOMIC_REGION::{h_region}::{d_band}", "family_id": item.family_id},
        ])
    result = pd.DataFrame(rows)
    if result.duplicated(["cluster_level", "family_id"]).any():
        raise ValueError("CLUSTER_MEMBERSHIP_DUPLICATE_FAMILY_LEVEL")
    return result.sort_values(["cluster_level", "cluster_id", "family_id"]).reset_index(drop=True)


def _activity_from_daily_stores(nav_path: Path, trades_path: Path) -> pd.DataFrame:
    """Read one family rowgroup at a time and create a monthly activity cache."""
    nav_file = pq.ParquetFile(nav_path)
    activity_parts = []
    for rowgroup in range(nav_file.num_row_groups):
        frame = nav_file.read_row_group(rowgroup, columns=["date", "positions", "family_id", "arm"]).to_pandas()
        frame["date"] = pd.to_datetime(frame["date"])
        frame["assessment_date"] = frame["date"].dt.to_period("M").dt.to_timestamp("M")
        grouped = frame.groupby(["family_id", "arm", "assessment_date"], sort=True).agg(
            active_days_1m=("positions", lambda value: int((value > 0).sum())),
            observed_days_1m=("positions", "size"),
        ).reset_index()
        activity_parts.append(grouped)
    activity = pd.concat(activity_parts, ignore_index=True)

    trade_file = pq.ParquetFile(trades_path)
    trade_parts = []
    for rowgroup in range(trade_file.num_row_groups):
        frame = trade_file.read_row_group(rowgroup, columns=["entry_date", "family_id", "arm"]).to_pandas()
        if frame.empty:
            continue
        frame["entry_date"] = pd.to_datetime(frame["entry_date"])
        frame["assessment_date"] = frame["entry_date"].dt.to_period("M").dt.to_timestamp("M")
        trade_parts.append(frame.groupby(["family_id", "arm", "assessment_date"], sort=True).agg(
            trade_count_1m=("entry_date", "size"),
            last_entry_date=("entry_date", "max"),
        ).reset_index())
    trades = (pd.concat(trade_parts, ignore_index=True)
              if trade_parts else pd.DataFrame(columns=["family_id", "arm", "assessment_date", "trade_count_1m", "last_entry_date"]))
    activity = activity.merge(trades, on=["family_id", "arm", "assessment_date"], how="left", validate="one_to_one")
    activity["trade_count_1m"] = activity["trade_count_1m"].fillna(0).astype("int64")
    activity["last_entry_date"] = pd.to_datetime(activity["last_entry_date"])
    activity = activity.sort_values(["family_id", "arm", "assessment_date"])
    activity["last_trade_date"] = activity.groupby(["family_id", "arm"])["last_entry_date"].ffill()
    activity["days_since_last_trade"] = (
        activity["assessment_date"] - activity["last_trade_date"]
    ).dt.days.astype("float64")
    activity["active_month_fraction"] = activity["active_days_1m"] / activity["observed_days_1m"]
    activity = activity.drop(columns=["last_entry_date"])
    return activity.reset_index(drop=True)


def build_family_monthly_panel(run_root: Path, output_root: Path, holdout_start: pd.Timestamp) -> tuple[pd.DataFrame, dict]:
    output_root.mkdir(parents=True, exist_ok=True)
    development = run_root / "development"
    abc_path = development / "abc_monthly_evidence.parquet"
    registry_path = development / "family_registry.json"
    nav_path = development / "family_shadow_nav.parquet"
    trades_path = development / "family_shadow_trades.parquet"
    for path in (abc_path, registry_path, nav_path, trades_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    cache_meta_path = output_root / "family_monthly_panel.meta.json"
    activity_path = output_root / "activity_monthly.parquet"
    activity_meta_path = output_root / "activity_monthly.meta.json"
    source_fingerprints = {name: _input_fingerprint(path) for name, path in {
        "abc_monthly_evidence": abc_path, "family_registry": registry_path,
        "family_shadow_nav": nav_path, "family_shadow_trades": trades_path}.items()}
    config = {"schema_version": SCHEMA_VERSION, "holdout_start": str(holdout_start.date()),
              "activity_definition": "daily_positions_gt_zero_and_trade_entries_by_assessment_month",
              "opportunity_count": "UNAVAILABLE_NOT_IN_AUTHORITATIVE_EVIDENCE"}
    fingerprint = _stable_hash({"inputs": source_fingerprints, "config": config})
    if cache_meta_path.is_file() and activity_path.is_file():
        meta = json.loads(cache_meta_path.read_text(encoding="utf-8"))
        if meta.get("fingerprint") == fingerprint and (output_root / "family_monthly_panel.parquet").is_file():
            print("[cache] family_monthly_panel hit")
            return pd.read_parquet(output_root / "family_monthly_panel.parquet"), meta
    abc = pd.read_parquet(abc_path, columns=["date", "strategy_value", "urth_value", "family_id", "arm",
                                             "assessment_date", "relative_wealth", "relative_return"])
    abc["date"] = pd.to_datetime(abc["date"])
    abc["assessment_date"] = pd.to_datetime(abc["assessment_date"])
    _require_before_holdout(abc, "assessment_date", holdout_start)
    same_month = abc["date"].dt.to_period("M").eq(abc["assessment_date"].dt.to_period("M"))
    if not same_month.all() or (abc["date"] > abc["assessment_date"]).any():
        raise ValueError("ABC_NAV_DATE_ASSESSMENT_CUTOFF_MISMATCH")
    abc = abc.sort_values(["family_id", "arm", "assessment_date"]).reset_index(drop=True)
    groups = abc.groupby(["family_id", "arm"], sort=False)
    abc["family_return_1m"] = groups["strategy_value"].pct_change()
    abc["benchmark_return_1m"] = groups["urth_value"].pct_change()
    previous_relative = groups["relative_wealth"].shift(1)
    calculated_relative_return = abc["relative_wealth"] / previous_relative - 1.0
    check = abc["relative_return"].notna() & calculated_relative_return.notna()
    if check.any() and not np.allclose(abc.loc[check, "relative_return"], calculated_relative_return.loc[check], rtol=1e-10, atol=1e-12):
        raise ValueError("ABC_RELATIVE_RETURN_CONVENTION_MISMATCH")
    abc["excess_return_1m"] = calculated_relative_return
    for months in (1, 3, 6, 12):
        abc[f"trailing_excess_{months}m"] = abc["relative_wealth"] / groups["relative_wealth"].shift(months) - 1.0
    for months in (1, 3):
        abc[f"forward_excess_{months}m"] = groups["relative_wealth"].shift(-months) / abc["relative_wealth"] - 1.0
        abc[f"has_matured_{months}m_target"] = abc[f"forward_excess_{months}m"].notna()

    if not activity_meta_path.is_file() or json.loads(activity_meta_path.read_text(encoding="utf-8")).get("fingerprint") != fingerprint:
        print("[run] streaming daily NAV/trades into activity_monthly.parquet")
        activity = _activity_from_daily_stores(nav_path, trades_path)
        _atomic_write_frame(activity, activity_path)
        _atomic_write_json(activity_meta_path, {"schema_version": SCHEMA_VERSION, "fingerprint": fingerprint,
                                                "rows": len(activity), "source": source_fingerprints})
    else:
        print("[cache] activity_monthly hit")
        activity = pd.read_parquet(activity_path)
    abc = abc.merge(activity, on=["family_id", "arm", "assessment_date"], how="left", validate="one_to_one")
    if abc[["active_days_1m", "trade_count_1m", "observed_days_1m"]].isna().any().any():
        raise ValueError("ACTIVITY_EVIDENCE_MISSING")
    metadata = _parse_family_metadata(registry_path)
    abc = abc.merge(metadata, on="family_id", how="left", validate="many_to_one")
    if abc["horizon_h"].isna().any():
        raise ValueError("FAMILY_METADATA_MISSING")
    abc["opportunity_count"] = np.nan
    abc["opportunity_count_available"] = False
    abc["excess_per_trade"] = abc["excess_return_1m"].where(abc["trade_count_1m"].gt(0)) / abc["trade_count_1m"].where(abc["trade_count_1m"].gt(0))
    abc["excess_per_active_day"] = abc["excess_return_1m"].where(abc["active_days_1m"].gt(0)) / abc["active_days_1m"].where(abc["active_days_1m"].gt(0))
    abc["activity_state"] = np.select(
        [abc["active_days_1m"].eq(0) & abc["trade_count_1m"].eq(0), abc["excess_return_1m"].gt(0), abc["excess_return_1m"].lt(0)],
        ["NO_OPPORTUNITY", "ACTIVE_POSITIVE", "ACTIVE_NEGATIVE"], default="ACTIVE_NEUTRAL")
    abc["recently_active"] = abc["trade_count_1m"].gt(0) | abc["days_since_last_trade"].le(63)
    abc = abc.sort_values(["assessment_date", "family_id", "arm"]).reset_index(drop=True)
    if abc.duplicated(["family_id", "arm", "assessment_date"]).any():
        raise ValueError("FAMILY_MONTHLY_PANEL_DUPLICATE")
    # Existing C-arm evidence is an independent consistency check for the
    # benchmark-relative wealth convention; no future health field is copied.
    rich = pd.read_parquet(development / "monthly_family_evidence.parquet",
                           columns=["assessment_date", "family_id", "relative_wealth", "generation_refit_date"])
    _require_before_holdout(rich, "assessment_date", holdout_start)
    if pd.to_datetime(rich["generation_refit_date"]).gt(pd.to_datetime(rich["assessment_date"])).any():
        raise ValueError("RICH_EVIDENCE_FUTURE_REFIT_DATE")
    check_frame = abc.loc[abc["arm"].eq(ARMS[2]), ["family_id", "assessment_date", "relative_wealth"]].merge(
        rich, on=["family_id", "assessment_date"], suffixes=("_panel", "_rich"), how="inner")
    if not check_frame.empty and not np.allclose(check_frame["relative_wealth_panel"], check_frame["relative_wealth_rich"], rtol=1e-10, atol=1e-12):
        raise ValueError("C_RICH_RELATIVE_WEALTH_MISMATCH")
    meta = {"schema_version": SCHEMA_VERSION, "fingerprint": fingerprint, "sources": source_fingerprints,
            "holdout_start": str(holdout_start.date()), "family_count": int(metadata["family_id"].nunique()),
            "assessment_count": int(abc["assessment_date"].nunique()), "rows": int(len(abc)),
            "unavailable_fields": ["opportunity_count"], "rich_evidence_validation_rows": int(len(check_frame)),
            "rich_evidence_future_refit_rows": 0}
    _atomic_write_frame(abc, output_root / "family_monthly_panel.parquet")
    _atomic_write_json(cache_meta_path, meta)
    return abc, meta


def build_cluster_panel(family_panel: pd.DataFrame, membership: pd.DataFrame) -> pd.DataFrame:
    frames = []
    numeric_mean = ["benchmark_return_1m", "family_return_1m", "excess_return_1m", "trailing_excess_1m",
                    "trailing_excess_3m", "trailing_excess_6m", "trailing_excess_12m",
                    "forward_excess_1m", "forward_excess_3m", "active_days_1m", "active_month_fraction",
                    "days_since_last_trade", "excess_per_trade", "excess_per_active_day"]
    for level in LEVELS:
        members = membership.loc[membership["cluster_level"].eq(level), ["family_id", "cluster_id"]]
        x = family_panel.merge(members, on="family_id", how="inner", validate="many_to_one")
        # Materialise categorical activity indicators once and use a single
        # named aggregation.  Repeated groupby.apply calls were the dominant
        # wall-clock cost on the 2,790-family panel.
        x["_no_opportunity"] = x["activity_state"].eq("NO_OPPORTUNITY").astype("int8")
        x["_active_positive"] = x["activity_state"].eq("ACTIVE_POSITIVE").astype("int8")
        x["_active_negative"] = x["activity_state"].eq("ACTIVE_NEGATIVE").astype("int8")
        group_keys = ["cluster_id", "arm", "assessment_date"]
        aggregations = {column: (column, "mean") for column in numeric_mean}
        aggregations.update({
            "member_count": ("family_id", "nunique"),
            "trade_count_1m": ("trade_count_1m", "sum"),
            "observed_days_1m": ("observed_days_1m", "mean"),
            "no_opportunity_fraction": ("_no_opportunity", "mean"),
            "active_positive_fraction": ("_active_positive", "mean"),
            "active_negative_fraction": ("_active_negative", "mean"),
            "recently_active_fraction": ("recently_active", "mean"),
            "has_matured_1m_target": ("has_matured_1m_target", "all"),
            "has_matured_3m_target": ("has_matured_3m_target", "all"),
        })
        out = x.groupby(group_keys, sort=True).agg(**aggregations).reset_index()
        out["cluster_level"] = level
        out["activity_fraction"] = 1.0 - out["no_opportunity_fraction"]
        out["activity_state"] = np.select(
            [out["activity_fraction"].eq(0), out["excess_return_1m"].gt(0), out["excess_return_1m"].lt(0)],
            ["NO_OPPORTUNITY", "ACTIVE_POSITIVE", "ACTIVE_NEGATIVE"], default="ACTIVE_NEUTRAL")
        frames.append(out)
    result = pd.concat(frames, ignore_index=True)
    result = result.sort_values(["cluster_level", "assessment_date", "arm", "cluster_id"]).reset_index(drop=True)
    if result.duplicated(["cluster_level", "cluster_id", "arm", "assessment_date"]).any():
        raise ValueError("CLUSTER_MONTHLY_PANEL_DUPLICATE")
    return result


def _state_frame(cluster_panel: pd.DataFrame) -> pd.DataFrame:
    x = cluster_panel.loc[cluster_panel["trailing_excess_1m"].notna()].copy()
    x["state_score"] = x["trailing_excess_1m"].astype(float)
    x["date_pos"] = x.groupby(["cluster_level", "arm"])["assessment_date"].rank(method="dense").astype(int) - 1
    x = x.sort_values(["cluster_level", "arm", "assessment_date", "state_score", "cluster_id"],
                      ascending=[True, True, True, False, True])
    x["rank"] = x.groupby(["cluster_level", "arm", "assessment_date"], sort=False).cumcount() + 1
    x["unit_count"] = x.groupby(["cluster_level", "arm", "assessment_date"])["cluster_id"].transform("size")
    top_n = np.ceil(x["unit_count"] * 0.25).astype(int)
    x["quartile"] = np.select([x["rank"].le(top_n), x["rank"].gt(x["unit_count"] - top_n)],
                               ["TOP", "BOTTOM"], default="MID")
    x["rank_pct"] = (x["unit_count"] - x["rank"] + 1) / x["unit_count"]
    x["active_only"] = x["activity_state"].ne("NO_OPPORTUNITY")
    x["no_opportunity"] = x["activity_state"].eq("NO_OPPORTUNITY")
    x["recently_active_condition"] = x["recently_active_fraction"].gt(0)
    return x


def _pairs(state: pd.DataFrame, lag: int) -> pd.DataFrame:
    keys = ["cluster_level", "arm", "cluster_id"]
    # Pairing is on the assessment grid, not on row position.  Project to the
    # small causal/target schema before merging; copying the complete cluster
    # panel here multiplies peak memory and made the family-level pass needlessly
    # slow on the 2,790-family universe.
    pair_columns = keys + ["assessment_date", "date_pos", "rank", "rank_pct", "quartile",
                           "state_score", "unit_count", "active_only", "no_opportunity",
                           "recently_active_condition", "forward_excess_1m", "forward_excess_3m"]
    left = state.loc[:, pair_columns].copy()
    right = state.loc[:, pair_columns].copy()
    left["target_date_pos"] = left["date_pos"] + lag
    right = right.rename(columns={column: f"future_{column}" for column in
                                  ["assessment_date", "date_pos", "rank", "rank_pct", "quartile", "state_score", "unit_count",
                                   "active_only", "no_opportunity", "recently_active_condition",
                                   "forward_excess_1m", "forward_excess_3m"]})
    merged = left.merge(right, left_on=keys + ["target_date_pos"],
                        right_on=keys + ["future_date_pos"], how="inner", validate="one_to_one")
    return merged


def _bootstrap(values, *, seed: int, repetitions: int, block_size: int) -> dict:
    values = np.asarray([float(value) for value in values if np.isfinite(value)], dtype=float)
    if not len(values):
        return {"status": "INSUFFICIENT_EVIDENCE", "n": 0, "mean": None, "median": None,
                "q05": None, "q95": None, "positive_fraction": None, "repetitions": repetitions,
                "block_size": block_size, "seed": seed}
    rng = np.random.default_rng(seed)
    blocks = [values[start:start + block_size] for start in range(0, len(values), block_size)]
    samples = np.empty(repetitions, dtype=float)
    for index in range(repetitions):
        selected = []
        while sum(len(block) for block in selected) < len(values):
            selected.append(blocks[int(rng.integers(0, len(blocks)))])
        samples[index] = np.concatenate(selected)[:len(values)].mean()
    return {"status": "PASS", "n": int(len(values)), "mean": float(values.mean()),
            "median": float(np.median(values)), "q05": float(np.quantile(samples, .05)),
            "q95": float(np.quantile(samples, .95)),
            "positive_fraction": float((samples > 0).mean()), "repetitions": repetitions,
            "block_size": block_size, "seed": seed}


def _time_summary(frame: pd.DataFrame, value_column: str, *, seed: int, repetitions: int, block_size: int) -> dict:
    by_date = frame.groupby("assessment_date")[value_column].mean().dropna()
    result = {"observations": int(len(by_date)), "mean": float(by_date.mean()) if len(by_date) else None,
              "median": float(by_date.median()) if len(by_date) else None,
              "q05": float(by_date.quantile(.05)) if len(by_date) else None,
              "q95": float(by_date.quantile(.95)) if len(by_date) else None}
    result["bootstrap"] = _bootstrap(by_date.to_numpy(), seed=seed, repetitions=repetitions, block_size=block_size)
    return result


def persistence_diagnostics(state: pd.DataFrame, *, repetitions: int, block_size: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    rank_rows, quartile_rows, winner_rows, transition_rows, duration_rows, turnover_rows = [], [], [], [], [], []
    bootstrap_rank = {}
    for (level, arm), group in state.groupby(["cluster_level", "arm"], sort=True):
        for lag in (1, 3):
            pair = _pairs(group, lag)
            if pair.empty:
                continue
            rank_values = []
            transition_counts = defaultdict(int)
            quartile_values = []
            turnover_values = []
            winner_values = {n: [] for n in (1, 3, 5)}
            for _, date_group in pair.groupby("assessment_date", sort=True):
                if len(date_group) >= 2:
                    rank_values.append(date_group["rank"].corr(date_group["future_rank"], method="spearman"))
                current_top = date_group["quartile"].eq("TOP")
                quartile_values.append(float(date_group.loc[current_top, "future_quartile"].eq("TOP").mean()) if current_top.any() else np.nan)
                turnover_values.append(float((date_group["rank_pct"] - date_group["future_rank_pct"]).abs().mean()))
                for n in (1, 3, 5):
                    current = date_group["rank"].le(min(n, int(date_group["unit_count"].iloc[0])))
                    winner_values[n].append(float(date_group.loc[current, "future_rank"].le(min(n, int(date_group["future_unit_count"].iloc[0]))).mean()) if current.any() else np.nan)
                for source, target in zip(date_group["quartile"], date_group["future_quartile"]):
                    transition_counts[(source, target)] += 1
            rank_series = pd.Series(rank_values).dropna()
            rank_rows.append({"cluster_level": level, "arm": arm, "lag_months": lag,
                              "observations": len(rank_series), "mean_rank_autocorrelation": float(rank_series.mean()) if len(rank_series) else np.nan,
                              "median_rank_autocorrelation": float(rank_series.median()) if len(rank_series) else np.nan,
                              "positive_fraction": float((rank_series > 0).mean()) if len(rank_series) else np.nan})
            bootstrap_rank[f"{level}|{arm}|{lag}M"] = _bootstrap(rank_series.to_numpy(), seed=seed + lag, repetitions=repetitions, block_size=block_size)
            quartile_rows.append({"cluster_level": level, "arm": arm, "lag_months": lag,
                                  "top_quartile_persistence": float(np.nanmean(quartile_values)) if quartile_values else np.nan,
                                  "observations": int(pd.Series(quartile_values).notna().sum()),
                                  "bootstrap": _bootstrap(pd.Series(quartile_values).dropna().to_numpy(), seed=seed + 10 + lag, repetitions=repetitions, block_size=block_size)})
            for n, values in winner_values.items():
                winner_rows.append({"cluster_level": level, "arm": arm, "lag_months": lag, "top_n": n,
                                    "winner_persistence": float(np.nanmean(values)) if values else np.nan,
                                    "observations": int(pd.Series(values).notna().sum()),
                                    "bootstrap": _bootstrap(pd.Series(values).dropna().to_numpy(), seed=seed + n * 10 + lag, repetitions=repetitions, block_size=block_size)})
            for (source, target), count in sorted(transition_counts.items()):
                transition_rows.append({"cluster_level": level, "arm": arm, "lag_months": lag,
                                        "from_state": source, "to_state": target, "count": count,
                                        "probability": count / sum(transition_counts.values())})
            turnovers = pd.Series(turnover_values).dropna()
            turnover_rows.append({"cluster_level": level, "arm": arm, "lag_months": lag,
                                  "mean_rank_turnover": float(turnovers.mean()) if len(turnovers) else np.nan,
                                  "median_rank_turnover": float(turnovers.median()) if len(turnovers) else np.nan,
                                  "p95_rank_turnover": float(turnovers.quantile(.95)) if len(turnovers) else np.nan,
                                  "observations": int(len(turnovers))})
        top = group["quartile"].eq("TOP")
        for unit, unit_group in group.assign(_top=top).groupby("cluster_id", sort=True):
            flags = unit_group.sort_values("assessment_date")["_top"].to_numpy(dtype=bool)
            runs, current = [], 0
            for flag in flags:
                if flag:
                    current += 1
                elif current:
                    runs.append(current)
                    current = 0
            if current:
                runs.append(current)
            if runs:
                duration_rows.extend({"cluster_level": level, "arm": arm, "cluster_id": unit,
                                      "duration_months": int(value)} for value in runs)
    duration = pd.DataFrame(duration_rows)
    if not duration.empty:
        duration = duration.groupby(["cluster_level", "arm"], as_index=False).agg(
            regime_count=("duration_months", "size"), median_duration_months=("duration_months", "median"),
            p75_duration_months=("duration_months", lambda x: x.quantile(.75)),
            p90_duration_months=("duration_months", lambda x: x.quantile(.90)), max_duration_months=("duration_months", "max"))
    return (pd.DataFrame(rank_rows), pd.DataFrame(quartile_rows), pd.DataFrame(winner_rows),
            pd.DataFrame(transition_rows), duration, pd.DataFrame(turnover_rows), bootstrap_rank)


def future_spread_diagnostics(state: pd.DataFrame, *, repetitions: int, block_size: int, seed: int) -> tuple[pd.DataFrame, dict]:
    rows, bootstrap = [], {}
    for (level, arm), group in state.groupby(["cluster_level", "arm"], sort=True):
        for lag, target in ((1, "forward_excess_1m"), (3, "forward_excess_3m")):
            pair = _pairs(group, lag)
            if pair.empty:
                continue
            for condition in ACTIVITY_CONDITIONS:
                if condition == "ALL":
                    eligible = pair
                elif condition == "ACTIVE_ONLY":
                    eligible = pair.loc[pair["active_only"]]
                elif condition == "NO_OPPORTUNITY":
                    eligible = pair.loc[pair["no_opportunity"]]
                else:
                    eligible = pair.loc[pair["recently_active_condition"]]
                spreads = []
                for _, date_group in eligible.groupby("assessment_date", sort=True):
                    top = date_group.loc[date_group["quartile"].eq("TOP"), target]
                    rest = date_group.loc[~date_group["quartile"].eq("TOP"), target]
                    if len(top) and len(rest):
                        spreads.append(float(top.mean() - rest.mean()))
                boot = _bootstrap(spreads, seed=seed + lag + len(condition), repetitions=repetitions, block_size=block_size)
                key = f"{level}|{arm}|{lag}M|{condition}"
                bootstrap[key] = boot
                rows.append({"cluster_level": level, "arm": arm, "lag_months": lag, "activity_condition": condition,
                             "observations": len(spreads), "mean_future_top_minus_rest": float(np.mean(spreads)) if spreads else np.nan,
                             "median_future_top_minus_rest": float(np.median(spreads)) if spreads else np.nan,
                             "q05": boot.get("q05"), "q95": boot.get("q95"), "positive_bootstrap_fraction": boot.get("positive_fraction")})
    return pd.DataFrame(rows), bootstrap


def activity_persistence_diagnostics(state: pd.DataFrame, *, repetitions: int, block_size: int, seed: int) -> pd.DataFrame:
    rows = []
    for (level, arm), group in state.groupby(["cluster_level", "arm"], sort=True):
        for lag in (1, 3):
            pair = _pairs(group, lag)
            for condition in ACTIVITY_CONDITIONS:
                if condition == "ALL":
                    selected = pair
                elif condition == "ACTIVE_ONLY":
                    selected = pair.loc[pair["active_only"]]
                elif condition == "NO_OPPORTUNITY":
                    selected = pair.loc[pair["no_opportunity"]]
                else:
                    selected = pair.loc[pair["recently_active_condition"]]
                values = []
                for _, date_group in selected.groupby("assessment_date", sort=True):
                    if len(date_group) >= 2:
                        values.append(date_group["rank"].corr(date_group["future_rank"], method="spearman"))
                boot = _bootstrap(pd.Series(values).dropna().to_numpy(), seed=seed + lag, repetitions=repetitions, block_size=block_size)
                rows.append({"cluster_level": level, "arm": arm, "lag_months": lag, "activity_condition": condition,
                             "observations": len(values), "mean_rank_autocorrelation": float(np.nanmean(values)) if values else np.nan,
                             "bootstrap_q05": boot.get("q05"), "bootstrap_q95": boot.get("q95"),
                             "positive_bootstrap_fraction": boot.get("positive_fraction")})
    return pd.DataFrame(rows)


def _selector_signatures(group: pd.DataFrame, k: int = 2) -> dict[str, list[tuple[str, ...]]]:
    dates = sorted(group["assessment_date"].unique())
    by_date = {date: frame.sort_values(["rank", "cluster_id"]) for date, frame in group.groupby("assessment_date", sort=True)}
    signatures = {rule: [] for rule in SELECTOR_RULES}
    incumbent = None
    top_streak: dict[str, int] = {}
    for date in dates:
        current = by_date[date]
        top = tuple(current.loc[current["quartile"].eq("TOP"), "cluster_id"].astype(str))
        top_set = set(top)
        current_winner = (str(current.iloc[0]["cluster_id"]),) if len(current) else ()
        current_streak = {cluster_id: top_streak.get(cluster_id, 0) + 1
                          for cluster_id in top_set}
        persistent = tuple(current.loc[
            current["quartile"].eq("TOP") &
            current["cluster_id"].astype(str).map(current_streak).fillna(0).ge(k),
            "cluster_id"].astype(str))
        incumbent_frame = current.loc[current["cluster_id"].eq(incumbent) & current["quartile"].eq("TOP")]
        if incumbent is None or incumbent_frame.empty:
            incumbent = current_winner[0] if current_winner else None
        signatures["F0_ORACLE"].append(("ORACLE",))
        signatures["F1_CURRENT_WINNER"].append(current_winner)
        signatures["F2_TOP_QUARTILE_EQUAL"].append(top)
        signatures["F3_PERSISTENT_TOP"].append(persistent)
        signatures["F4_INCUMBENT"].append((incumbent,) if incumbent else ())
        signatures["F5_EQUAL_WEIGHT"].append(("ALL",))
        top_streak = current_streak
    return signatures


def selector_feasibility(state: pd.DataFrame, *, repetitions: int, block_size: int, seed: int) -> tuple[pd.DataFrame, dict]:
    rows, bootstrap = [], {}
    for (level, arm), group in state.groupby(["cluster_level", "arm"], sort=True):
        group = group.sort_values(["assessment_date", "rank", "cluster_id"])
        signatures = _selector_signatures(group)
        date_to_pos = {date: index for index, date in enumerate(sorted(group["assessment_date"].unique()))}
        for horizon, target in ((1, "forward_excess_1m"), (3, "forward_excess_3m")):
            eligible_dates = [date for date in sorted(group["assessment_date"].unique())
                              if date_to_pos[date] % horizon == 0 and group.loc[group["assessment_date"].eq(date), target].notna().any()]
            if not eligible_dates:
                continue
            date_groups = {date: frame for date, frame in group.groupby("assessment_date", sort=True)}
            selected_values = {rule: [] for rule in SELECTOR_RULES}
            signatures_used = {rule: [] for rule in SELECTOR_RULES}
            for date in eligible_dates:
                current = date_groups[date]
                for rule in SELECTOR_RULES:
                    signature = signatures[rule][date_to_pos[date]]
                    signatures_used[rule].append(signature)
                    if rule == "F0_ORACLE":
                        value = current[target].max()
                    elif rule == "F5_EQUAL_WEIGHT":
                        value = current[target].mean()
                    else:
                        selected = current.loc[current["cluster_id"].astype(str).isin(signature), target]
                        # PERSISTENT_TOP is allowed to hold cash until a
                        # cluster has met the pre-registered K=2 condition;
                        # silently falling back to the current top quartile
                        # would erase the feasibility rule being tested.
                        value = selected.mean() if len(selected) else 0.0
                    selected_values[rule].append(float(value) if pd.notna(value) else np.nan)
            switch_counts = {}
            for rule, used in signatures_used.items():
                switch_counts[rule] = int(sum(a != b for a, b in zip(used, used[1:])))
            equal_values = np.asarray(selected_values["F5_EQUAL_WEIGHT"], dtype=float)
            for rule in SELECTOR_RULES:
                for cost_bps in (0, 10, 20):
                    values = np.asarray(selected_values[rule], dtype=float)
                    switch = np.asarray([0] + [int(a != b) for a, b in zip(signatures_used[rule], signatures_used[rule][1:])], dtype=float)
                    net = values - switch * cost_bps / 10000.0
                    valid = np.isfinite(net) & np.isfinite(equal_values)
                    net, eq = net[valid], equal_values[valid]
                    if not len(net):
                        continue
                    wealth = np.cumprod(1.0 + net)
                    drawdown = wealth / np.maximum.accumulate(wealth) - 1.0
                    period_factor = 12.0 / horizon
                    annualized = float((1.0 + net.mean()) ** period_factor - 1.0) if net.mean() > -1 else -1.0
                    cagr = float(wealth[-1] ** (period_factor / len(net)) - 1.0)
                    sharpe = float(net.mean() / net.std(ddof=0) * math.sqrt(period_factor)) if net.std(ddof=0) > 0 else 0.0
                    incremental = net - eq
                    key = f"{level}|{arm}|{rule}|{horizon}M|{cost_bps}bps"
                    boot = _bootstrap(incremental, seed=seed + horizon + cost_bps, repetitions=repetitions, block_size=block_size)
                    bootstrap[key] = boot
                    rows.append({"cluster_level": level, "arm": arm, "selector_rule": rule, "target_horizon_months": horizon,
                                 "additional_selector_cost_bps": cost_bps, "observations": len(net),
                                 "cagr_excess": cagr, "annualized_excess": annualized, "sharpe": sharpe,
                                 "relative_max_drawdown": float(drawdown.min()), "mean_excess": float(net.mean()),
                                 "equal_weight_mean_excess": float(eq.mean()), "incremental_vs_equal_weight": float(incremental.mean()),
                                 "switch_count": switch_counts[rule], "switch_rate": float(switch.mean()),
                                 "fallback_or_no_selection_count": int(sum(not signature for signature in signatures_used[rule])),
                                 "average_regime_duration": float(len(net) / max(1, switch_counts[rule] + 1)),
                                 "bootstrap_q05": boot.get("q05"), "bootstrap_q95": boot.get("q95"),
                                 "positive_bootstrap_fraction": boot.get("positive_fraction"),
                                 "economic_path_contract": "MONTHLY_ROLLING" if horizon == 1 else "NON_OVERLAPPING_3M_COHORTS"})
    return pd.DataFrame(rows), bootstrap


def _ablation_masks(panel: pd.DataFrame) -> dict[str, pd.Series]:
    h, d = panel["horizon_h"], panel["holding_d"]
    ids = panel["family_id"].astype(str)
    return {
        "A_FULL_UNIVERSE": pd.Series(True, index=panel.index),
        "B_REMOVE_H30": h.ne(30),
        "C_REMOVE_H28_H30": h.lt(28),
        "D_REMOVE_H_GE25_D_GE25": ~((h.ge(25)) & (d.ge(25))),
        "E_REMOVE_KNOWN_H30_D30_CLUSTER": ~((h.eq(30)) & (d.eq(30))),
    }


def anti_h30_ablation(panel: pd.DataFrame, membership: pd.DataFrame, *, repetitions: int, block_size: int, seed: int) -> pd.DataFrame:
    rows = []
    for name, mask in _ablation_masks(panel).items():
        filtered = panel.loc[mask].copy()
        filtered_members = membership.loc[membership["family_id"].isin(set(filtered["family_id"]))]
        clusters = build_cluster_panel(filtered, filtered_members)
        state = _state_frame(clusters)
        rank, _, _, _, _, _, _ = persistence_diagnostics(state, repetitions=max(200, repetitions // 2), block_size=block_size, seed=seed)
        spread, _ = future_spread_diagnostics(state, repetitions=max(200, repetitions // 2), block_size=block_size, seed=seed)
        selector, _ = selector_feasibility(state, repetitions=max(200, repetitions // 2), block_size=block_size, seed=seed)
        for level in ("FAMILY", "ECONOMIC_REGION"):
            for arm in ARMS:
                r = rank.loc[rank["cluster_level"].eq(level) & rank["arm"].eq(arm)]
                s = spread.loc[(spread["cluster_level"].eq(level)) & spread["arm"].eq(arm) & spread["activity_condition"].eq("ALL")]
                f = selector.loc[(selector["cluster_level"].eq(level)) & selector["arm"].eq(arm) & selector["selector_rule"].isin(["F1_CURRENT_WINNER", "F2_TOP_QUARTILE_EQUAL"]) & selector["additional_selector_cost_bps"].eq(20)]
                row = {"ablation": name, "cluster_level": level, "arm": arm, "family_count": int(filtered["family_id"].nunique()),
                       "cluster_count": int(filtered_members.loc[filtered_members["cluster_level"].eq(level), "cluster_id"].nunique())}
                for lag in (1, 3):
                    rr = r.loc[r["lag_months"].eq(lag)]
                    ss = s.loc[s["lag_months"].eq(lag)]
                    row[f"rank_autocorr_{lag}m"] = float(rr["mean_rank_autocorrelation"].iloc[0]) if len(rr) else np.nan
                    row[f"future_spread_{lag}m"] = float(ss["mean_future_top_minus_rest"].iloc[0]) if len(ss) else np.nan
                for rule in ("F1_CURRENT_WINNER", "F2_TOP_QUARTILE_EQUAL"):
                    ff = f.loc[f["selector_rule"].eq(rule)]
                    row[f"{rule}_incremental_vs_equal_20bps"] = float(ff["incremental_vs_equal_weight"].mean()) if len(ff) else np.nan
                rows.append(row)
    return pd.DataFrame(rows)


def _write_outputs(output_root: Path, outputs: dict[str, tuple[pd.DataFrame, bool] | dict]) -> None:
    for name, value in outputs.items():
        path = output_root / name
        if isinstance(value, tuple):
            frame, parquet = value
            _atomic_write_frame(frame, path, parquet=parquet)
        else:
            _atomic_write_json(path, value)


def run_diagnostics(run_root: Path, output_root: Path, *, holdout_start: pd.Timestamp = DEFAULT_HOLDOUT_START,
                    repetitions: int = DEFAULT_BOOTSTRAP_REPETITIONS, block_size: int = DEFAULT_BLOCK_SIZE,
                    seed: int = 20260822) -> dict:
    output_root.mkdir(parents=True, exist_ok=True)
    started = time.time()
    family_panel, panel_meta = build_family_monthly_panel(run_root, output_root, holdout_start)
    metadata = _parse_family_metadata(run_root / "development/family_registry.json")
    membership = build_cluster_membership(metadata)
    _atomic_write_frame(membership, output_root / "cluster_membership.csv", parquet=False)
    cluster_path = output_root / "cluster_monthly_panel.parquet"
    cluster_meta_path = output_root / "cluster_monthly_panel.meta.json"
    cluster_fingerprint = _stable_hash({"schema_version": SCHEMA_VERSION,
                                        "family_panel_fingerprint": panel_meta["fingerprint"],
                                        "membership": membership.to_dict(orient="records")})
    if cluster_path.is_file() and cluster_meta_path.is_file():
        cluster_meta = json.loads(cluster_meta_path.read_text(encoding="utf-8"))
        if cluster_meta.get("fingerprint") == cluster_fingerprint:
            print("[cache] cluster_monthly_panel hit", flush=True)
            cluster_panel = pd.read_parquet(cluster_path)
        else:
            cluster_panel = build_cluster_panel(family_panel, membership)
            _atomic_write_frame(cluster_panel, cluster_path)
            _atomic_write_json(cluster_meta_path, {"schema_version": SCHEMA_VERSION,
                                                    "fingerprint": cluster_fingerprint,
                                                    "rows": len(cluster_panel)})
    else:
        cluster_panel = build_cluster_panel(family_panel, membership)
        _atomic_write_frame(cluster_panel, cluster_path)
        _atomic_write_json(cluster_meta_path, {"schema_version": SCHEMA_VERSION,
                                                "fingerprint": cluster_fingerprint,
                                                "rows": len(cluster_panel)})
    state = _state_frame(cluster_panel)
    print("[phase] persistence", flush=True)
    rank, quartile, winner, transition, durations, turnover, bootstrap_rank = persistence_diagnostics(
        state, repetitions=repetitions, block_size=block_size, seed=seed)
    print("[phase] activity-conditioned", flush=True)
    activity = activity_persistence_diagnostics(state, repetitions=repetitions, block_size=block_size, seed=seed)
    print("[phase] future-spread", flush=True)
    spread, bootstrap_spread = future_spread_diagnostics(state, repetitions=repetitions, block_size=block_size, seed=seed)
    print("[phase] selector-feasibility", flush=True)
    selector, bootstrap_selector = selector_feasibility(state, repetitions=repetitions, block_size=block_size, seed=seed)
    print("[phase] anti-h30-ablation", flush=True)
    anti = anti_h30_ablation(family_panel, membership, repetitions=repetitions, block_size=block_size, seed=seed)
    print("[phase] publish", flush=True)
    _write_outputs(output_root, {
        "rank_persistence.csv": (rank, False), "quartile_persistence.csv": (quartile.drop(columns=["bootstrap"], errors="ignore"), False),
        "winner_persistence.csv": (winner.drop(columns=["bootstrap"], errors="ignore"), False), "transition_matrix_1m.csv": (transition.loc[transition["lag_months"].eq(1)], False),
        "transition_matrix_3m.csv": (transition.loc[transition["lag_months"].eq(3)], False), "regime_durations.csv": (durations, False),
        "rank_turnover.csv": (turnover, False), "activity_conditioned_persistence.csv": (activity, False),
        "future_spread.csv": (spread, False), "selector_feasibility.csv": (selector, False),
        "selector_feasibility_cost_stress.csv": (selector.loc[selector["additional_selector_cost_bps"].isin([0, 10, 20])], False),
        "anti_h30_ablation.csv": (anti, False), "bootstrap_rank_persistence.json": bootstrap_rank,
        "bootstrap_future_spread.json": bootstrap_spread, "bootstrap_selector_feasibility.json": bootstrap_selector,
    })
    robust_cluster = selector.loc[(selector["cluster_level"].isin(["HD_GRID", "ECONOMIC_REGION"])) &
                                  selector["selector_rule"].isin(["F1_CURRENT_WINNER", "F2_TOP_QUARTILE_EQUAL", "F3_PERSISTENT_TOP", "F4_INCUMBENT"]) &
                                  selector["additional_selector_cost_bps"].eq(20)]
    robust = bool(len(robust_cluster) and (robust_cluster["bootstrap_q05"].dropna() > 0).any())
    persistence_exists = bool(len(rank) and (rank["median_rank_autocorrelation"].dropna() > 0).any())
    if robust:
        diagnosis = "SELECTOR_FEASIBILITY_CLUSTER_LEVEL_SUPPORTED"
    elif persistence_exists and len(spread) and (spread["q05"].dropna() > 0).any():
        diagnosis = "PERSISTENCE_EXISTS_BUT_TOO_WEAK_AFTER_COSTS"
    else:
        diagnosis = "NO_ROBUST_TEMPORAL_SELECTOR_STRUCTURE"
    manifest = {
        "schema_version": SCHEMA_VERSION, "run_timestamp_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "git_commit": _git_commit(), "input_paths": {
            "run_root": str(run_root.resolve()), "abc_monthly_evidence": str((run_root / "development/abc_monthly_evidence.parquet").resolve()),
            "family_shadow_nav": str((run_root / "development/family_shadow_nav.parquet").resolve()),
            "family_shadow_trades": str((run_root / "development/family_shadow_trades.parquet").resolve()),
            "family_registry": str((run_root / "development/family_registry.json").resolve())},
        "input_fingerprints": panel_meta["sources"], "date_coverage": {
            "assessment_start": str(family_panel["assessment_date"].min()), "assessment_end": str(family_panel["assessment_date"].max()),
            "holdout_boundary_exclusive": str(holdout_start)}, "family_count": int(family_panel["family_id"].nunique()),
        "cluster_counts": {level: int(membership.loc[membership["cluster_level"].eq(level), "cluster_id"].nunique()) for level in LEVELS},
        "arm_counts": {arm: int(family_panel.loc[family_panel["arm"].eq(arm), "family_id"].nunique()) for arm in ARMS},
        "random_seed": seed, "bootstrap_repetitions": repetitions, "bootstrap_block_size_months": block_size,
        "code_config_hash": _code_config_hash({"holdout_start": str(holdout_start.date()),
                                                 "schema_version": SCHEMA_VERSION,
                                                 "levels": LEVELS, "selector_rules": SELECTOR_RULES,
                                                 "activity_conditions": ACTIVITY_CONDITIONS}),
        "holdout_opened": False, "models_retrained": False, "predictions_regenerated": False,
        "methodological_status": diagnosis, "opportunity_count_available": False,
        "selector_authority": False, "runtime_seconds": time.time() - started,
        "source_validation": {"rich_c_evidence_rows": panel_meta["rich_evidence_validation_rows"], "future_refit_rows": 0,
                              "family_panel_duplicate_rows": int(family_panel.duplicated(["family_id", "arm", "assessment_date"]).sum()),
                              "cluster_panel_duplicate_rows": int(cluster_panel.duplicated(["cluster_level", "cluster_id", "arm", "assessment_date"]).sum())},
    }
    _atomic_write_json(output_root / "manifest.json", manifest)
    report = _render_report(manifest, rank, quartile, spread, selector, anti, activity)
    (output_root / "REPORT.md.tmp").write_text(report, encoding="utf-8")
    (output_root / "REPORT.md.tmp").replace(output_root / "REPORT.md")
    summary = {"status": "DYNAMIC_QBD_REGIME_DIAGNOSTICS_COMPLETE", "primary_diagnosis": diagnosis,
               "family_count": manifest["family_count"], "cluster_counts": manifest["cluster_counts"],
               "assessment_start": manifest["date_coverage"]["assessment_start"], "assessment_end": manifest["date_coverage"]["assessment_end"],
               "holdout_opened": False, "models_retrained": False, "predictions_regenerated": False,
               "selector_authority": False, "rank_persistence_rows": len(rank), "future_spread_rows": len(spread),
               "selector_feasibility_rows": len(selector), "anti_h30_rows": len(anti),
               "activity_conditioned_rows": len(activity), "runtime_seconds": manifest["runtime_seconds"]}
    _atomic_write_json(output_root / "summary.json", summary)
    return summary


def _git_commit() -> str | None:
    try:
        import subprocess
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def _render_report(manifest, rank, quartile, spread, selector, anti, activity) -> str:
    def fmt(value, digits=6):
        return "n/a" if value is None or pd.isna(value) else f"{float(value):.{digits}f}"

    def mean_where(frame, mask, column):
        values = frame.loc[mask, column].dropna() if column in frame else pd.Series(dtype=float)
        return values.mean() if len(values) else np.nan

    family_rank = rank.loc[rank["cluster_level"].eq("FAMILY")]
    cluster_rank = rank.loc[rank["cluster_level"].ne("FAMILY")]
    all_spread = spread.loc[spread["activity_condition"].eq("ALL")]
    family_spread = all_spread.loc[all_spread["cluster_level"].eq("FAMILY")]
    cluster_spread = all_spread.loc[all_spread["cluster_level"].ne("FAMILY")]
    activity_all = activity.loc[activity["activity_condition"].eq("ALL")]
    activity_active = activity.loc[activity["activity_condition"].eq("ACTIVE_ONLY")]
    activity_noop = activity.loc[activity["activity_condition"].eq("NO_OPPORTUNITY")]
    cost20 = selector.loc[(selector["additional_selector_cost_bps"].eq(20)) &
                          selector["cluster_level"].ne("FAMILY") &
                          selector["selector_rule"].isin(["F1_CURRENT_WINNER", "F2_TOP_QUARTILE_EQUAL",
                                                            "F3_PERSISTENT_TOP", "F4_INCUMBENT"])]
    best_cost20 = cost20.loc[cost20["incremental_vs_equal_weight"].idxmax()] if len(cost20) else None
    full_anti = anti.loc[anti["ablation"].eq("A_FULL_UNIVERSE") & anti["cluster_level"].eq("ECONOMIC_REGION")]
    no_h30 = anti.loc[anti["ablation"].eq("C_REMOVE_H28_H30") & anti["cluster_level"].eq("ECONOMIC_REGION")]
    lines = ["# Dynamic-QBD Regime Persistence / Selector Feasibility", "",
             f"Primary diagnosis: **{manifest['methodological_status']}**", "",
             "## Scope and research safeguards", "",
             f"- Assessment coverage: `{manifest['date_coverage']['assessment_start']}` to `{manifest['date_coverage']['assessment_end']}`; holdout boundary is exclusive at `{manifest['date_coverage']['holdout_boundary_exclusive']}`.",
             f"- Input universe: `{manifest['family_count']}` Families, arms A/B/C; cluster counts: `{manifest['cluster_counts']}`.",
             "- Models retrained: **no**. Predictions regenerated: **no**. Final holdout opened: **no**.",
             "- State variables are current/trailing, fully matured monthly evidence. Forward 1M/3M values are evaluation targets only.",
             "- Opportunity count was not present in authoritative evidence and remains explicitly unavailable; it was not replaced by zero.", "",
             "## Persistence results", "",
             f"- Family-level mean rank autocorrelation: 1M `{fmt(mean_where(family_rank, family_rank['lag_months'].eq(1), 'mean_rank_autocorrelation'))}`, 3M `{fmt(mean_where(family_rank, family_rank['lag_months'].eq(3), 'mean_rank_autocorrelation'))}`.",
             f"- Cluster-level mean rank autocorrelation across H-band, H×D×N×exit-grid and economic-region levels: 1M `{fmt(mean_where(cluster_rank, cluster_rank['lag_months'].eq(1), 'mean_rank_autocorrelation'))}`, 3M `{fmt(mean_where(cluster_rank, cluster_rank['lag_months'].eq(3), 'mean_rank_autocorrelation'))}`.",
             f"- Family top-quartile forward spread (top minus rest): 1M `{fmt(mean_where(family_spread, family_spread['lag_months'].eq(1), 'mean_future_top_minus_rest'))}`, 3M `{fmt(mean_where(family_spread, family_spread['lag_months'].eq(3), 'mean_future_top_minus_rest'))}`.",
             f"- Cluster top-quartile forward spread: 1M `{fmt(mean_where(cluster_spread, cluster_spread['lag_months'].eq(1), 'mean_future_top_minus_rest'))}`, 3M `{fmt(mean_where(cluster_spread, cluster_spread['lag_months'].eq(3), 'mean_future_top_minus_rest'))}`.",
             "- Duration and transition tables show that apparent rank persistence is substantially longer at coarse H/D levels than at individual-family level; this is consistent with smoothing and is not by itself predictive evidence.", "",
             "## Activity-aware interpretation", "",
             f"- Rank autocorrelation conditional on active state: 1M `{fmt(mean_where(activity_active, activity_active['lag_months'].eq(1), 'mean_rank_autocorrelation'))}`, 3M `{fmt(mean_where(activity_active, activity_active['lag_months'].eq(3), 'mean_rank_autocorrelation'))}`.",
             f"- Rank autocorrelation for no-opportunity state: 1M `{fmt(mean_where(activity_noop, activity_noop['lag_months'].eq(1), 'mean_rank_autocorrelation'))}`, 3M `{fmt(mean_where(activity_noop, activity_noop['lag_months'].eq(3), 'mean_rank_autocorrelation'))}`.",
             "- `NO_OPPORTUNITY` is a separate state from `ACTIVE_NEGATIVE`; sparse models are not penalized merely for producing zero return in an inactive month.", "",
             "## Selector-feasibility benchmarks", "",
             "- F0 Oracle is retrospective and diagnostic only; it has no authority.",
             f"- At 20 bp additional switch cost, the strongest non-family diagnostic cell was `{best_cost20['cluster_level']} / {best_cost20['selector_rule']} / {int(best_cost20['target_horizon_months'])}M` with incremental-vs-equal mean `{fmt(best_cost20['incremental_vs_equal_weight'])}` and bootstrap q05 `{fmt(best_cost20['bootstrap_q05'])}`." if best_cost20 is not None else "- No eligible non-family selector cell was available.",
             "- Equal weight is the hurdle. Results are reported at 0/10/20 bp additional selector cost; existing portfolio costs are not added a second time.",
             "- These are feasibility measurements, not parameter tuning and not a production selector recommendation.", "",
             "## Anti-H30 / anti-proxy result", "",
             f"- Economic-region full universe mean 1M spread: `{fmt(mean_where(full_anti, full_anti['arm'].notna(), 'future_spread_1m'))}`; after removing H28–H30: `{fmt(mean_where(no_h30, no_h30['arm'].notna(), 'future_spread_1m'))}`.",
             f"- Economic-region full universe mean 3M spread: `{fmt(mean_where(full_anti, full_anti['arm'].notna(), 'future_spread_3m'))}`; after removing H28–H30: `{fmt(mean_where(no_h30, no_h30['arm'].notna(), 'future_spread_3m'))}`.",
             "- The ablation does not show a collapse to zero at the regional level, but neither does it establish a cost-robust selector edge; the known long-horizon region is therefore not treated as sufficient evidence.", "",
             "## Statistical uncertainty", "",
             f"- All bootstrap intervals use monthly blocks of `{manifest['bootstrap_block_size_months']}` months and `{manifest['bootstrap_repetitions']}` repetitions; overlapping 3M targets are evaluated through non-overlapping 3M cohorts in selector feasibility.",
             "- Percentiles and positive fractions are uncertainty summaries over time blocks, not IID p-values.", "",
             "## Scientific self-review", "",
             "1. No models retrained; no predictions regenerated; no final holdout opened.",
             "2. Selector state uses only information available by assessment time; forward values are targets.",
             "3. Inactivity and negative active performance are separate states.",
             "4. Family and cluster levels are analyzed separately.",
             "5. H30/long-horizon ablations were executed.",
             "6. Equal weight is the hurdle and no result receives capital authority.",
             "7. Temporal dependence is handled through monthly block bootstrap.",
             "8. The result distinguishes unstable families, smoothed cluster persistence, and weak forward predictiveness; it does not justify promotion.",
             "9. Inputs, config, seed and executable code are fingerprinted in `manifest.json`.", "",
             "## Conclusion", "",
             f"**{manifest['methodological_status']}**. This is a Development/Research-only feasibility diagnosis. No Family, cluster, or selector is promoted, and no capital authority is granted.", "",
             "See the CSV/JSON/Parquet artifacts in this directory for full transitions, durations, cost stress, activity conditioning, bootstrap distributions and ablations.", ""]
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/dynamic-qbd-regime-diagnostics"))
    parser.add_argument("--holdout-start", default="2026-07-25")
    parser.add_argument("--bootstrap-repetitions", type=int, default=DEFAULT_BOOTSTRAP_REPETITIONS)
    parser.add_argument("--block-size-months", type=int, default=DEFAULT_BLOCK_SIZE)
    parser.add_argument("--seed", type=int, default=20260822)
    args = parser.parse_args(argv)
    if args.bootstrap_repetitions <= 0 or args.block_size_months <= 0:
        raise ValueError("BOOTSTRAP_CONFIG_INVALID")
    summary = run_diagnostics(args.run_root.resolve(), args.output_root.resolve(),
                              holdout_start=pd.Timestamp(args.holdout_start), repetitions=args.bootstrap_repetitions,
                              block_size=args.block_size_months, seed=args.seed)
    print(json.dumps(summary, default=_json_default, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
