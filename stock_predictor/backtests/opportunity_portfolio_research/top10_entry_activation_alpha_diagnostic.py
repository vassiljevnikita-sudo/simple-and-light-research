"""Diagnose Top-10 entry inactivity without retraining or regenerating predictions.

Contract: TOP10_ENTRY_ACTIVATION_AND_ALPHA_DIAGNOSTIC_V1

This module is intentionally read-only with respect to model/prediction artifacts. It
compares historical walk-forward activation with causal expanding-live activation and
then evaluates cross-sectional ranking skill against subsequently realized market
returns. Realized returns are evaluation-only and never feed calibration or prediction.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from .portfolio_research_inputs import assert_no_final_frozen_holdout, load_predictions, load_price_panel


CONTRACT_ID = "TOP10_ENTRY_ACTIVATION_AND_ALPHA_DIAGNOSTIC_V1"
LIVE_START = pd.Timestamp("2023-08-11")
LIVE_END = pd.Timestamp("2026-07-24")
ROUNDTRIP_COST = 0.002
QUANTILES = (0.90, 0.95, 0.975, 0.99)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    tmp.replace(path)


def _markdown(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "(no rows)"
    columns = [str(c) for c in frame.columns]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(lines)


def _causal_entry(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    required = {"decision_date", "ticker", "horizon_sessions", "predicted_net_excess_return", "model_training_cutoff"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"CAUSAL_ENTRY_COLUMNS_MISSING:{missing}")
    assert_no_final_frozen_holdout(frame, source=str(path))
    frame = frame.copy()
    frame["decision_date"] = pd.to_datetime(frame["decision_date"]).dt.normalize()
    frame["ticker"] = frame["ticker"].astype(str)
    frame["horizon"] = frame["horizon_sessions"].astype(int)
    frame["score"] = pd.to_numeric(frame["predicted_net_excess_return"], errors="coerce")
    frame["model_training_cutoff"] = pd.to_datetime(frame["model_training_cutoff"]).dt.normalize()
    if frame.duplicated(["decision_date", "ticker", "horizon"]).any():
        raise RuntimeError("CAUSAL_ENTRY_DUPLICATE_KEYS")
    if (frame["model_training_cutoff"] >= frame["decision_date"]).any():
        raise RuntimeError("CAUSAL_ENTRY_TRAINING_CUTOFF_NOT_PRIOR")
    observed = (frame["decision_date"].min(), frame["decision_date"].max())
    if observed != (LIVE_START, LIVE_END):
        raise RuntimeError(f"CAUSAL_ENTRY_DATE_COVERAGE_MISMATCH:{observed}")
    if set(frame["horizon"].unique()) != {11, 24, 28}:
        raise RuntimeError(f"CAUSAL_ENTRY_HORIZON_MISMATCH:{sorted(frame['horizon'].unique())}")
    return frame


def _contracts(manifest_path: Path) -> tuple[pd.DataFrame, dict[str, dict]]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    models = payload.get("models", [])
    if len(models) != 10:
        raise RuntimeError(f"FROZEN_TOP10_COUNT_MISMATCH:{len(models)}")
    rows = []
    by_model = {}
    for model in models:
        policy = model["frozen_entry_policy"]
        row = {
            "model_id": str(model["model_id"]),
            "horizon": int(policy["horizon"]),
            "score_quantile": float(policy["score_quantile"]),
            "top_fraction": float(policy["top_fraction"]),
        }
        by_model[row["model_id"]] = row
        rows.append(row)
    unique = pd.DataFrame(rows).drop_duplicates(["horizon", "score_quantile", "top_fraction"]).sort_values(
        ["horizon", "score_quantile", "top_fraction"]
    )
    unique["contract_id"] = unique.apply(
        lambda r: f"H{int(r.horizon):02d}_Q{str(float(r.score_quantile)).replace('.', '_')}_T{str(float(r.top_fraction)).replace('.', '_')}",
        axis=1,
    )
    return unique.reset_index(drop=True), by_model


def _historical_thresholds(summary_path: Path) -> dict[tuple[int, str, float], float]:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    hist = payload.get("historical_reproduction", {})
    if hist.get("status") != "PASSED":
        raise RuntimeError("HISTORICAL_PREDICTION_REPRODUCTION_NOT_PASSED")
    result = {}
    for row in hist.get("entry_rows", []):
        horizon = int(row["horizon"])
        fold_id = str(row["fold_id"])
        for q, values in row.get("threshold_crossings", {}).items():
            result[(horizon, fold_id, float(q))] = float(values["expected_threshold"])
    if not result:
        raise RuntimeError("HISTORICAL_THRESHOLDS_MISSING")
    return result


def _threshold_column(q: float) -> str:
    return f"resolved_threshold_q{str(float(q)).replace('.', '_')}"


def _daily_activation(frame: pd.DataFrame, threshold_by_row: pd.Series, *, source: str,
                      contract_id: str, horizon: int, score_quantile: float,
                      top_fraction: float) -> pd.DataFrame:
    x = frame[["decision_date", "ticker", "score"]].copy()
    x["threshold"] = pd.to_numeric(threshold_by_row, errors="coerce").to_numpy()
    if x["threshold"].isna().any() or x["score"].isna().any():
        raise RuntimeError(f"ACTIVATION_NAN:{source}:{contract_id}")
    grouped = x.groupby("decision_date", sort=True)
    base = grouped["score"].agg(universe_size="size", score_min="min", score_median="median", score_max="max")
    q = grouped["score"].quantile([0.90, 0.95, 0.975, 0.99]).unstack()
    q.columns = ["score_p90", "score_p95", "score_p97_5", "score_p99"]
    threshold = grouped["threshold"].first().rename("threshold")
    if not grouped["threshold"].nunique(dropna=False).eq(1).all():
        raise RuntimeError(f"THRESHOLD_NOT_UNIQUE_PER_DATE:{source}:{contract_id}")
    crossing = x.assign(cross=x["score"].ge(x["threshold"])).groupby("decision_date")["cross"].sum().rename("crossing_count")
    out = base.join(q).join(threshold).join(crossing).reset_index()
    out["crossing_count"] = out["crossing_count"].astype(int)
    out["crossing_fraction"] = out["crossing_count"] / out["universe_size"]
    out["max_margin"] = out["score_max"] - out["threshold"]
    out["max_ratio"] = np.where(out["threshold"].abs() > 1e-15, out["score_max"] / out["threshold"], np.nan)
    out["threshold_over_p99"] = np.where(out["score_p99"].abs() > 1e-15, out["threshold"] / out["score_p99"], np.nan)
    out["top_fraction_limit"] = np.maximum(1, np.ceil(out["universe_size"] * float(top_fraction))).astype(int)
    out["threshold_passing_count"] = out["crossing_count"]
    out["final_eligible_count"] = np.minimum(out["top_fraction_limit"], out["crossing_count"])
    out["relative_candidate_count"] = out["top_fraction_limit"]
    out["relative_candidates_but_no_crossing"] = (out["relative_candidate_count"] > 0) & (out["crossing_count"] == 0)
    out.insert(0, "source", source)
    out.insert(1, "contract_id", contract_id)
    out.insert(2, "horizon", int(horizon))
    out.insert(3, "score_quantile", float(score_quantile))
    out.insert(4, "top_fraction", float(top_fraction))
    return out


def _activation_tables(causal: pd.DataFrame, historical: pd.DataFrame, contracts: pd.DataFrame,
                       historical_thresholds: dict[tuple[int, str, float], float]) -> pd.DataFrame:
    rows = []
    for contract in contracts.itertuples(index=False):
        c = causal.loc[causal["horizon"].eq(contract.horizon)].copy()
        column = _threshold_column(contract.score_quantile)
        if column not in c.columns:
            raise RuntimeError(f"CAUSAL_THRESHOLD_COLUMN_MISSING:{column}")
        rows.append(_daily_activation(
            c, c[column], source="CAUSAL_EXPANDING_LIVE", contract_id=contract.contract_id,
            horizon=contract.horizon, score_quantile=contract.score_quantile, top_fraction=contract.top_fraction,
        ))

        h = historical.loc[historical["horizon"].eq(contract.horizon)].copy()
        if "fold_id" not in h:
            raise RuntimeError("HISTORICAL_FOLD_ID_MISSING")
        # WF_000 is the prequential calibration-only history.  It has no
        # published OOS threshold crossing record and must not be treated as
        # an OOS fold in this activation comparison.
        historical_fold_keys = {
            fold for (horizon, fold, quantile) in historical_thresholds
            if horizon == int(contract.horizon) and quantile == float(contract.score_quantile)
        }
        h = h.loc[h["fold_id"].astype(str).isin(historical_fold_keys)].copy()
        if h.empty:
            raise RuntimeError(f"HISTORICAL_OOS_FOLDS_EMPTY:{contract.contract_id}")
        mapped = h["fold_id"].map(
            lambda fold: historical_thresholds.get((int(contract.horizon), str(fold), float(contract.score_quantile)))
        )
        if mapped.isna().any():
            examples = h.loc[mapped.isna(), "fold_id"].astype(str).drop_duplicates().head(5).tolist()
            raise RuntimeError(f"HISTORICAL_THRESHOLD_MAPPING_MISSING:{contract.contract_id}:{examples}")
        rows.append(_daily_activation(
            h, mapped, source="HISTORICAL_WF", contract_id=contract.contract_id,
            horizon=contract.horizon, score_quantile=contract.score_quantile, top_fraction=contract.top_fraction,
        ))
    return pd.concat(rows, ignore_index=True)


def _period_activation(daily: pd.DataFrame, frequency: str) -> pd.DataFrame:
    x = daily.copy()
    x["period"] = x["decision_date"].dt.to_period(frequency).astype(str)
    keys = ["source", "contract_id", "horizon", "score_quantile", "top_fraction", "period"]
    rows = []
    for key, g in x.groupby(keys, sort=True):
        source, contract_id, horizon, q, top, period = key
        rows.append({
            "source": source, "contract_id": contract_id, "horizon": horizon, "score_quantile": q,
            "top_fraction": top, "period": period, "days": len(g),
            "days_with_crossing": int(g["crossing_count"].gt(0).sum()),
            "crossing_day_rate": float(g["crossing_count"].gt(0).mean()),
            "total_crossings": int(g["crossing_count"].sum()),
            "total_final_eligible": int(g["final_eligible_count"].sum()),
            "median_universe_size": float(g["universe_size"].median()),
            "median_score_max": float(g["score_max"].median()),
            "median_score_p99": float(g["score_p99"].median()),
            "median_threshold": float(g["threshold"].median()),
            "median_max_margin": float(g["max_margin"].median()),
            "median_max_ratio": float(g["max_ratio"].median()),
            "median_threshold_over_p99": float(g["threshold_over_p99"].median()),
            "days_relative_candidates_but_no_crossing": int(g["relative_candidates_but_no_crossing"].sum()),
        })
    return pd.DataFrame(rows)


def _price_views(prices: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[pd.Timestamp], dict[pd.Timestamp, int]]:
    p = prices.copy()
    p["date"] = pd.to_datetime(p["date"]).dt.normalize()
    p["ticker"] = p["ticker"].astype(str)
    p = p.drop_duplicates(["date", "ticker"], keep="last")
    stock = p[["date", "ticker", "open", "close"]].copy()
    urth = stock.loc[stock["ticker"].eq("URTH"), ["date", "open", "close"]].sort_values("date")
    sessions = [pd.Timestamp(x) for x in urth["date"].tolist()]
    index = {d: i for i, d in enumerate(sessions)}
    return stock, urth, sessions, index


def _attach_realized(frame: pd.DataFrame, stock: pd.DataFrame, urth: pd.DataFrame,
                     sessions: list[pd.Timestamp], session_index: dict[pd.Timestamp, int], horizon: int) -> pd.DataFrame:
    x = frame.loc[frame["horizon"].eq(horizon), ["decision_date", "ticker", "score"]].copy()
    if x.empty:
        return x
    positions = x["decision_date"].map(session_index)
    valid = positions.notna() & positions.map(lambda i: int(i) + horizon < len(sessions) if pd.notna(i) else False)
    x = x.loc[valid].copy()
    positions = positions.loc[valid].astype(int)
    x["entry_date"] = [sessions[i + 1] for i in positions]
    x["terminal_date"] = [sessions[i + horizon] for i in positions]

    entry = stock[["date", "ticker", "open"]].rename(columns={"date": "entry_date", "open": "stock_entry_open"})
    terminal = stock[["date", "ticker", "close"]].rename(columns={"date": "terminal_date", "close": "stock_terminal_close"})
    x = x.merge(entry, on=["entry_date", "ticker"], how="left").merge(terminal, on=["terminal_date", "ticker"], how="left")
    ue = urth[["date", "open"]].rename(columns={"date": "entry_date", "open": "urth_entry_open"})
    ut = urth[["date", "close"]].rename(columns={"date": "terminal_date", "close": "urth_terminal_close"})
    x = x.merge(ue, on="entry_date", how="left").merge(ut, on="terminal_date", how="left")
    finite = np.isfinite(x[["stock_entry_open", "stock_terminal_close", "urth_entry_open", "urth_terminal_close"]]).all(axis=1)
    positive = (x[["stock_entry_open", "stock_terminal_close", "urth_entry_open", "urth_terminal_close"]] > 0).all(axis=1)
    x["realized_price_available"] = finite & positive
    good = x["realized_price_available"]
    x["realized_stock_return"] = np.nan
    x["realized_urth_return"] = np.nan
    x["realized_gross_relative_excess"] = np.nan
    x["realized_net_excess_20bps"] = np.nan
    x.loc[good, "realized_stock_return"] = x.loc[good, "stock_terminal_close"] / x.loc[good, "stock_entry_open"] - 1.0
    x.loc[good, "realized_urth_return"] = x.loc[good, "urth_terminal_close"] / x.loc[good, "urth_entry_open"] - 1.0
    x.loc[good, "realized_gross_relative_excess"] = (
        (x.loc[good, "stock_terminal_close"] / x.loc[good, "stock_entry_open"])
        / (x.loc[good, "urth_terminal_close"] / x.loc[good, "urth_entry_open"]) - 1.0
    )
    x.loc[good, "realized_net_excess_20bps"] = x.loc[good, "realized_gross_relative_excess"] - ROUNDTRIP_COST
    x["score_percentile"] = x.groupby("decision_date")["score"].rank(method="average", pct=True)
    return x


def _spearman(frame: pd.DataFrame) -> float:
    g = frame.loc[frame["realized_price_available"], ["score", "realized_net_excess_20bps"]].dropna()
    if len(g) < 3 or g["score"].nunique() < 2 or g["realized_net_excess_20bps"].nunique() < 2:
        return float("nan")
    return float(g["score"].rank(method="average").corr(g["realized_net_excess_20bps"].rank(method="average")))


def _alpha_row(g: pd.DataFrame, source: str, horizon: int, period: str) -> dict:
    valid = g.loc[g["realized_price_available"] & g["realized_net_excess_20bps"].notna()].copy()
    row = {
        "source": source, "horizon": horizon, "period": period,
        "candidate_rows": len(g), "evaluated_rows": len(valid),
        "missing_realized_price_rows": int(len(g) - len(valid)),
        "spearman": _spearman(valid),
    }
    for pct, label in ((0.995, "top_0_5pct"), (0.99, "top_1pct"), (0.95, "top_5pct")):
        top = valid.loc[valid["score_percentile"].ge(pct)]
        row[f"{label}_rows"] = len(top)
        row[f"{label}_mean_realized_net_excess"] = float(top["realized_net_excess_20bps"].mean()) if len(top) else float("nan")
        row[f"{label}_hit_rate"] = float(top["realized_net_excess_20bps"].gt(0).mean()) if len(top) else float("nan")
    decile_top = valid.loc[valid["score_percentile"].ge(0.9), "realized_net_excess_20bps"]
    decile_bottom = valid.loc[valid["score_percentile"].le(0.1), "realized_net_excess_20bps"]
    row["top_decile_mean"] = float(decile_top.mean()) if len(decile_top) else float("nan")
    row["bottom_decile_mean"] = float(decile_bottom.mean()) if len(decile_bottom) else float("nan")
    row["top_minus_bottom_decile_spread"] = row["top_decile_mean"] - row["bottom_decile_mean"]
    return row


def _alpha_tables(causal: pd.DataFrame, historical: pd.DataFrame, prices: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    stock, urth, sessions, session_index = _price_views(prices)
    monthly = []
    summary = []
    for source, frame in (("HISTORICAL_WF", historical), ("CAUSAL_EXPANDING_LIVE", causal)):
        for horizon in (11, 24, 28):
            x = _attach_realized(frame, stock, urth, sessions, session_index, horizon)
            if x.empty:
                continue
            summary.append(_alpha_row(x, source, horizon, "ALL"))
            x["period"] = x["decision_date"].dt.to_period("M").astype(str)
            for period, g in x.groupby("period", sort=True):
                monthly.append(_alpha_row(g, source, horizon, str(period)))
    return pd.DataFrame(monthly), pd.DataFrame(summary)


def _aggregate_activation(daily: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, g in daily.groupby(["source", "contract_id", "horizon", "score_quantile", "top_fraction"], sort=True):
        source, contract_id, horizon, q, top = key
        rows.append({
            "source": source, "contract_id": contract_id, "horizon": horizon, "score_quantile": q, "top_fraction": top,
            "days": len(g), "crossing_days": int(g["crossing_count"].gt(0).sum()),
            "crossing_day_rate": float(g["crossing_count"].gt(0).mean()),
            "total_crossings": int(g["crossing_count"].sum()),
            "total_final_eligible": int(g["final_eligible_count"].sum()),
            "median_score_max": float(g["score_max"].median()),
            "median_score_p99": float(g["score_p99"].median()),
            "median_threshold": float(g["threshold"].median()),
            "median_max_ratio": float(g["max_ratio"].median()),
            "median_threshold_over_p99": float(g["threshold_over_p99"].median()),
        })
    return pd.DataFrame(rows)


def _diagnose(activation: pd.DataFrame, alpha: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for contract_id, g in activation.groupby("contract_id", sort=True):
        hist = g.loc[g["source"].eq("HISTORICAL_WF")]
        live = g.loc[g["source"].eq("CAUSAL_EXPANDING_LIVE")]
        if hist.empty or live.empty:
            continue
        h = int(live.iloc[0]["horizon"])
        ah = alpha.loc[(alpha["source"].eq("HISTORICAL_WF")) & alpha["horizon"].eq(h)]
        al = alpha.loc[(alpha["source"].eq("CAUSAL_EXPANDING_LIVE")) & alpha["horizon"].eq(h)]
        if ah.empty or al.empty:
            diagnosis = "INCONCLUSIVE"
            hist_top1 = live_top1 = hist_spread = live_spread = float("nan")
        else:
            hist_top1 = float(ah.iloc[0]["top_1pct_mean_realized_net_excess"])
            live_top1 = float(al.iloc[0]["top_1pct_mean_realized_net_excess"])
            hist_spread = float(ah.iloc[0]["top_minus_bottom_decile_spread"])
            live_spread = float(al.iloc[0]["top_minus_bottom_decile_spread"])
            hist_cross = float(hist.iloc[0]["crossing_day_rate"])
            live_cross = float(live.iloc[0]["crossing_day_rate"])
            activation_collapse = hist_cross >= 0.01 and live_cross < max(0.005, 0.25 * hist_cross)
            alpha_historically_positive = np.isfinite(hist_top1) and hist_top1 > 0
            live_ranking_positive = np.isfinite(live_top1) and live_top1 > 0 and np.isfinite(live_spread) and live_spread > 0
            alpha_decay = alpha_historically_positive and np.isfinite(live_top1) and live_top1 <= 0 and np.isfinite(live_spread) and live_spread <= 0
            scale_ratio = (
                float(live.iloc[0]["median_score_p99"]) / float(hist.iloc[0]["median_score_p99"])
                if abs(float(hist.iloc[0]["median_score_p99"])) > 1e-15 else float("nan")
            )
            threshold_pressure = (
                float(live.iloc[0]["median_threshold_over_p99"]) > 1.05
                and float(live.iloc[0]["median_threshold_over_p99"]) > 1.25 * max(float(hist.iloc[0]["median_threshold_over_p99"]), 1e-12)
            )
            if activation_collapse and alpha_decay:
                diagnosis = "BOTH_CALIBRATION_AND_ALPHA_DECAY"
            elif activation_collapse and live_ranking_positive and threshold_pressure and np.isfinite(scale_ratio) and scale_ratio < 0.75:
                diagnosis = "SCORE_SCALE_DRIFT"
            elif activation_collapse and live_ranking_positive:
                diagnosis = "CALIBRATION_TOO_RESTRICTIVE"
            elif not activation_collapse and alpha_decay:
                diagnosis = "RANKING_ALPHA_DECAY"
            elif live_cross < 0.02 and live_cross > 0 and live_ranking_positive:
                diagnosis = "HEALTHY_BUT_RARE_SIGNAL"
            else:
                diagnosis = "INCONCLUSIVE"
        rows.append({
            "contract_id": contract_id, "horizon": h,
            "historical_crossing_day_rate": float(hist.iloc[0]["crossing_day_rate"]),
            "live_crossing_day_rate": float(live.iloc[0]["crossing_day_rate"]),
            "historical_median_threshold_over_p99": float(hist.iloc[0]["median_threshold_over_p99"]),
            "live_median_threshold_over_p99": float(live.iloc[0]["median_threshold_over_p99"]),
            "historical_top1_mean_realized_net_excess": hist_top1,
            "live_top1_mean_realized_net_excess": live_top1,
            "historical_top_minus_bottom_decile_spread": hist_spread,
            "live_top_minus_bottom_decile_spread": live_spread,
            "diagnosis": diagnosis,
            "diagnosis_is_heuristic": True,
        })
    return pd.DataFrame(rows)


def _actual_trade_examples(trades_path: Path | None, causal: pd.DataFrame, contracts_by_model: dict[str, dict]) -> pd.DataFrame:
    if trades_path is None or not trades_path.is_file():
        return pd.DataFrame()
    trades = pd.read_csv(trades_path)
    if trades.empty:
        return pd.DataFrame()
    trades["entry_date"] = pd.to_datetime(trades["entry_date"]).dt.normalize()
    dates = sorted(pd.Timestamp(x) for x in causal["decision_date"].drop_duplicates())
    prior = {dates[i + 1]: dates[i] for i in range(len(dates) - 1)}
    rows = []
    for t in trades.itertuples(index=False):
        model_id = str(t.model_id)
        spec = contracts_by_model.get(model_id)
        if spec is None:
            continue
        decision = prior.get(pd.Timestamp(t.entry_date))
        if decision is None:
            continue
        day = causal.loc[(causal["decision_date"].eq(decision)) & causal["horizon"].eq(spec["horizon"])].copy()
        stock = day.loc[day["ticker"].eq(str(t.ticker))]
        if stock.empty:
            continue
        column = _threshold_column(spec["score_quantile"])
        score = float(stock.iloc[0]["score"])
        threshold = float(stock.iloc[0][column])
        ordered = day["score"].sort_values(ascending=False).to_numpy()
        second = float(ordered[1]) if len(ordered) > 1 else float("nan")
        percentile = float(day["score"].rank(method="average", pct=True).loc[stock.index[0]])
        rows.append({
            "model_id": model_id, "ticker": str(t.ticker), "decision_date": decision,
            "entry_date": pd.Timestamp(t.entry_date), "horizon": spec["horizon"],
            "score": score, "threshold": threshold, "score_minus_threshold": score - threshold,
            "score_percentile": percentile, "second_best_score": second, "score_minus_second_best": score - second,
            "realized_trade_excess": float(getattr(t, "excess_return", np.nan)),
            "exit_reason": str(getattr(t, "exit_reason", "")),
        })
    return pd.DataFrame(rows)


def run(args: argparse.Namespace) -> dict:
    output = Path(args.output_root)
    output.mkdir(parents=True, exist_ok=True)

    causal = _causal_entry(Path(args.causal_entry_predictions))
    historical, historical_audit = load_predictions(Path(args.historical_entry_predictions))
    if not set(historical["horizon"].unique()).issuperset({11, 24, 28}):
        raise RuntimeError("HISTORICAL_ENTRY_HORIZONS_MISSING")
    contracts, contracts_by_model = _contracts(Path(args.frozen_policy_manifest))
    thresholds = _historical_thresholds(Path(args.historical_reproduction_summary))

    daily = _activation_tables(causal, historical, contracts, thresholds)
    daily.to_parquet(output / "entry_activation_daily.parquet", index=False)
    monthly = _period_activation(daily, "M")
    monthly.to_csv(output / "entry_activation_monthly.csv", index=False)
    quarterly = _period_activation(daily, "Q")
    quarterly.to_csv(output / "entry_activation_quarterly.csv", index=False)

    tickers = set(causal["ticker"].astype(str)) | set(historical.loc[historical["horizon"].isin([11, 24, 28]), "ticker"].astype(str))
    prices, price_audit = load_price_panel(Path(args.daily_store_root), tickers)
    alpha_monthly, alpha_summary = _alpha_tables(causal, historical, prices)
    alpha_monthly.to_csv(output / "entry_alpha_monthly.csv", index=False)
    alpha_summary.to_csv(output / "entry_alpha_horizon_summary.csv", index=False)

    activation_summary = _aggregate_activation(daily)
    diagnosis = _diagnose(activation_summary, alpha_summary)
    trade_examples = _actual_trade_examples(Path(args.causal_trades) if args.causal_trades else None, causal, contracts_by_model)
    trade_examples.to_csv(output / "actual_crossing_examples.csv", index=False)

    result = {
        "contract_id": CONTRACT_ID,
        "status": "COMPLETE",
        "interpretation": "DIAGNOSTIC_ONLY_NO_POLICY_REOPTIMIZATION",
        "no_model_training": True,
        "no_prediction_generation": True,
        "realized_returns_used_for_evaluation_only": True,
        "final_holdout_opened": False,
        "live_start": str(LIVE_START.date()),
        "live_end": str(LIVE_END.date()),
        "activation_contracts": contracts.to_dict(orient="records"),
        "activation_summary": activation_summary.to_dict(orient="records"),
        "alpha_horizon_summary": alpha_summary.to_dict(orient="records"),
        "diagnoses": diagnosis.to_dict(orient="records"),
        "historical_prediction_audit": historical_audit,
        "price_audit": price_audit,
        "realized_return_contract": {
            "entry": "NEXT_SESSION_OPEN",
            "terminal": "HORIZON_SESSION_CLOSE",
            "gross_excess": "STOCK_GROWTH_DIVIDED_BY_URTH_GROWTH_MINUS_ONE",
            "net_excess": "GROSS_RELATIVE_EXCESS_MINUS_20_BPS",
            "right_censored_rows": "EXCLUDED",
        },
    }
    _write_json(output / "diagnostic_summary.json", result)

    report_table = diagnosis.copy()
    for c in ["historical_crossing_day_rate", "live_crossing_day_rate", "historical_top1_mean_realized_net_excess", "live_top1_mean_realized_net_excess"]:
        if c in report_table:
            report_table[c] = report_table[c].map(lambda x: "nan" if pd.isna(x) else f"{float(x):.4%}")
    report = [
        "# Top-10 Entry Activation and Alpha Diagnostic",
        "",
        f"Status: **{result['status']}**",
        "",
        "This audit reuses existing historical and causal-expanding predictions. It does not fit models, regenerate predictions, or optimize policies.",
        "Realized future returns are attached only after prediction for evaluation and never feed thresholds or model inputs.",
        "",
        "## Diagnosis by frozen activation contract",
        "",
        _markdown(report_table),
        "",
        "## Interpretation labels",
        "",
        "- `SCORE_SCALE_DRIFT`: activation collapsed while relative ranking still works and current scores sit materially below calibration scale.",
        "- `CALIBRATION_TOO_RESTRICTIVE`: activation collapsed but high-ranked names retain positive realized alpha.",
        "- `RANKING_ALPHA_DECAY`: activation is not the main issue; realized ranking skill deteriorated.",
        "- `BOTH_CALIBRATION_AND_ALPHA_DECAY`: activation collapsed and ranking alpha also failed.",
        "- `HEALTHY_BUT_RARE_SIGNAL`: rare activation with positive ranking evidence.",
        "- `INCONCLUSIVE`: evidence does not satisfy conservative heuristic gates.",
        "",
        "Final holdout remained closed.",
    ]
    (output / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return result


def self_test() -> None:
    dates = pd.to_datetime(["2025-01-02", "2025-01-03"])
    frame = pd.DataFrame({
        "decision_date": [dates[0]] * 4,
        "ticker": ["A", "B", "C", "D"],
        "score": [0.1, 0.2, 0.3, 0.4],
    })
    daily = _daily_activation(
        frame, pd.Series([0.35] * 4), source="TEST", contract_id="TEST", horizon=11,
        score_quantile=0.975, top_fraction=0.25,
    )
    assert int(daily.iloc[0]["crossing_count"]) == 1
    assert int(daily.iloc[0]["top_fraction_limit"]) == 1
    assert int(daily.iloc[0]["final_eligible_count"]) == 1
    assert abs(float(daily.iloc[0]["max_margin"]) - 0.05) < 1e-12
    source = Path(__file__).read_text(encoding="utf-8").split("def self_test", 1)[0]
    forbidden_fit = ".fit("
    assert forbidden_fit not in source and "joblib" not in source and "sklearn" not in source
    print("TOP10_ENTRY_ACTIVATION_ALPHA_DIAGNOSTIC_SELF_TEST_OK")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--causal-entry-predictions")
    parser.add_argument("--historical-entry-predictions")
    parser.add_argument("--historical-reproduction-summary")
    parser.add_argument("--frozen-policy-manifest")
    parser.add_argument("--daily-store-root")
    parser.add_argument("--causal-trades")
    parser.add_argument("--output-root", default="artifacts/top10-entry-activation-alpha-diagnostic")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    required = [
        "causal_entry_predictions", "historical_entry_predictions", "historical_reproduction_summary",
        "frozen_policy_manifest", "daily_store_root",
    ]
    missing = [name for name in required if not getattr(args, name)]
    if missing:
        parser.error("required arguments missing: " + ", ".join("--" + name.replace("_", "-") for name in missing))
    result = run(args)
    print(json.dumps({
        "status": result["status"], "contracts": len(result["activation_contracts"]),
        "diagnoses": result["diagnoses"], "no_model_training": result["no_model_training"],
        "no_prediction_generation": result["no_prediction_generation"],
    }, indent=2, default=str))


if __name__ == "__main__":
    main()
