"""Monthly causal family evidence panel."""
from __future__ import annotations

from collections import defaultdict
import numpy as np
import pandas as pd

from .dynamic_qbd_wealth_metrics import wealth_path_metrics_at_indices


PERFORMANCE_FEATURES = (
    "excess_1m", "excess_3m", "excess_6m", "excess_12m", "ewma_excess",
    "recent_minus_long_run_excess", "positive_period_fraction",
)
DOWNSIDE_FEATURES = (
    "relative_sortino", "relative_downside_deviation", "relative_max_drawdown", "cdar_95",
    "expected_shortfall_95", "current_drawdown_depth", "drawdown_duration_sessions",
    "max_drawdown_duration_sessions", "median_recovery_duration_sessions",
    "time_under_water_fraction", "capital_impairment_area",
)
GENERATION_HEALTH_FEATURES = (
    "rank_ic", "top_score_realised_excess", "calibration_error", "score_spread", "prediction_residual_drift",
    "top_vs_median_realised_excess",
)

STRUCTURAL_ROBUSTNESS_FEATURES = (
    "trade_count", "sample_sufficiency_weight", "independent_winners", "ticker_concentration",
    "leave_best_trade_out_excess", "leave_best_ticker_out_excess", "fold_stability",
)


def _trailing_excess(relative: pd.Series, sessions: int) -> float:
    if len(relative) <= sessions:
        return float("nan")
    return float(relative.iloc[-1] / relative.iloc[-sessions - 1] - 1.0)


def build_monthly_market_state(prices: pd.DataFrame) -> pd.DataFrame:
    """Causal market-state features available at each observed month end."""
    required={"date","ticker","close"}
    missing=required-set(prices)
    if missing:
        raise ValueError(f"MARKET_STATE_PRICE_COLUMNS_MISSING:{sorted(missing)}")
    frame=prices.copy(); frame["date"]=pd.to_datetime(frame["date"])
    close=frame.pivot_table(index="date",columns="ticker",values="close",aggfunc="last").sort_index()
    if "URTH" not in close:
        raise ValueError("MARKET_STATE_URTH_MISSING")
    returns=close.pct_change(fill_method=None)
    state=pd.DataFrame(index=close.index)
    state["benchmark_trend"]=close["URTH"]/close["URTH"].rolling(63,min_periods=20).mean()-1.0
    state["market_volatility"]=returns["URTH"].rolling(21,min_periods=10).std(ddof=0)
    twenty_day=close.pct_change(20,fill_method=None)
    stocks=twenty_day.drop(columns=["URTH"],errors="ignore")
    state["market_breadth"]=stocks.gt(0).mean(axis=1)
    state["cross_sectional_dispersion"]=returns.drop(columns=["URTH"],errors="ignore").std(axis=1,ddof=0)
    if "volume" in frame:
        volume=frame.pivot_table(index="date",columns="ticker",values="volume",aggfunc="last").sort_index()
        state["liquidity_state"]=np.log1p(volume.drop(columns=["URTH"],errors="ignore")).median(axis=1)
    else:
        state["liquidity_state"]=close.drop(columns=["URTH"],errors="ignore").notna().mean(axis=1)
    state=state.reset_index().rename(columns={"date":"assessment_date"})
    state=state.groupby(state["assessment_date"].dt.to_period("M"),sort=True).tail(1)
    return state.reset_index(drop=True)


def build_monthly_family_evidence(
    family_nav: pd.DataFrame,
    generation_schedule: pd.DataFrame,
    *,
    matured_predictions: pd.DataFrame | None = None,
    trades: pd.DataFrame | None = None,
) -> pd.DataFrame:
    required = {"date", "family_id", "strategy_value", "urth_value"}
    missing = required - set(family_nav)
    if missing:
        raise ValueError(f"FAMILY_NAV_COLUMNS_MISSING:{sorted(missing)}")
    nav = family_nav.copy()
    nav["date"] = pd.to_datetime(nav["date"])
    schedule = generation_schedule.copy()
    schedule["activation_date"] = pd.to_datetime(schedule["activation_date"])
    trades_by_family = {}
    if trades is not None and not trades.empty:
        trade_source = trades.copy()
        trade_source["exit_date"] = pd.to_datetime(trade_source["exit_date"])
        trades_by_family = {str(key): group for key, group in trade_source.groupby("family_id", sort=False)}
    rows = []
    for family_id, family in nav.groupby("family_id", sort=True):
        family = family.sort_values("date").reset_index(drop=True)
        family["relative_wealth"] = family["strategy_value"] / family["urth_value"]
        month_ends = family.groupby(family["date"].dt.to_period("M"), sort=True).tail(1)
        risk_by_index = wealth_path_metrics_at_indices(family, month_ends.index)
        family_schedule = schedule.loc[schedule["family_id"].eq(family_id)].sort_values("activation_date")
        family_prediction_frame = None
        family_prediction_terminal_dates = None
        if matured_predictions is not None and (not hasattr(matured_predictions, "read_family") or matured_predictions.row_count):
            family_prediction_frame = (matured_predictions.read_family(str(family_id))
                                       if hasattr(matured_predictions, "read_family") else matured_predictions)
            if "generation_id" not in family_prediction_frame:
                raise ValueError("GENERATION_ID_REQUIRED_FOR_CURRENT_FIT_EVIDENCE")
            family_prediction_terminal_dates = pd.to_datetime(family_prediction_frame["terminal_date"])
        family_trade_frame = trades_by_family.get(str(family_id)) if trades_by_family else None
        for endpoint_index, end_row in zip(month_ends.index, month_ends.itertuples(index=False)):
            cutoff = pd.Timestamp(end_row.date)
            history = family.iloc[: int(endpoint_index) + 1]
            active = family_schedule.loc[family_schedule["activation_date"].le(cutoff)]
            if active.empty:
                continue
            generation = active.iloc[-1]
            relative = history["relative_wealth"]
            daily_excess = relative.pct_change().dropna()
            monthly_relative=history.groupby(history["date"].dt.to_period("M"),sort=True).tail(1)["relative_wealth"]
            monthly_excess=monthly_relative.pct_change().dropna()
            risk = risk_by_index[int(endpoint_index)]
            record = {
                "assessment_date": cutoff,
                "information_cutoff": cutoff,
                "family_id": family_id,
                "generation_id": generation["generation_id"],
                "generation_refit_date": generation.get("refit_timestamp", generation["activation_date"]),
                "resolved_threshold": float(generation["resolved_threshold"]),
                "nav": float(history["strategy_value"].iloc[-1]),
                "benchmark_nav": float(history["urth_value"].iloc[-1]),
                "relative_wealth": float(relative.iloc[-1]),
                "excess_1m": _trailing_excess(relative, 21),
                "excess_3m": _trailing_excess(relative, 63),
                "excess_6m": _trailing_excess(relative, 126),
                "excess_12m": _trailing_excess(relative, 252),
                "ewma_excess": float(daily_excess.ewm(span=63, adjust=False).mean().iloc[-1]) if len(daily_excess) else 0.0,
                "recent_minus_long_run_excess": float(daily_excess.tail(21).mean() - daily_excess.mean()) if len(daily_excess) else 0.0,
                "positive_period_fraction": float(monthly_excess.gt(0).mean()) if len(monthly_excess) else 0.0,
                **risk,
            }
            if family_trade_frame is not None and not family_trade_frame.empty:
                ft = family_trade_frame.loc[family_trade_frame["exit_date"].le(cutoff)]
                record["trade_count"] = int(len(ft))
                record["matured_trade_count"] = int(len(ft))
                contribution_column = next((name for name in (
                    "excess_return", "net_excess_return", "net_pnl", "pnl"
                ) if name in ft), None)
                contributions = (ft[contribution_column].astype(float) if contribution_column
                                 else pd.Series(0.0, index=ft.index, dtype=float))
                positive = contributions.clip(lower=0)
                record["top_trade_contribution"] = float(positive.max() / positive.sum()) if positive.sum() > 0 else 0.0
                by_ticker = ft.assign(_c=positive).groupby("ticker")["_c"].sum()
                record["top_ticker_contribution"] = float(by_ticker.max() / by_ticker.sum()) if by_ticker.sum() > 0 else 0.0
                ticker_total = ft.assign(_c=contributions).groupby("ticker")["_c"].sum()
                record["independent_winners"] = int(ticker_total.gt(0).sum())
                absolute_total = float(ticker_total.abs().sum())
                record["ticker_concentration"] = float(ticker_total.abs().max() / absolute_total) if absolute_total else 0.0
                record["leave_best_trade_out_excess"] = float(
                    (contributions.sum() - contributions.max()) / max(1, len(contributions) - 1)
                ) if len(contributions) else 0.0
                best_ticker = str(ticker_total.idxmax()) if len(ticker_total) else ""
                without_best_ticker = contributions.loc[ft["ticker"].astype(str).ne(best_ticker)]
                record["leave_best_ticker_out_excess"] = float(without_best_ticker.mean()) if len(without_best_ticker) else 0.0
                record["sample_sufficiency_weight"] = float(min(1.0, np.sqrt(len(ft) / 30.0)))
            else:
                record.update(trade_count=0, matured_trade_count=0, top_trade_contribution=0.0,
                              top_ticker_contribution=0.0, independent_winners=0, ticker_concentration=0.0,
                              leave_best_trade_out_excess=0.0, leave_best_ticker_out_excess=0.0,
                              sample_sufficiency_weight=0.0)
            record.update(matured_prediction_count=0, rank_ic=np.nan, top_score_realised_excess=np.nan,
                          calibration_error=np.nan, score_spread=np.nan, prediction_residual_drift=np.nan,
                          top_vs_median_realised_excess=np.nan,
                          fold_stability=float(generation.get("selection_oos_positive_fold_fraction",np.nan)),
                          selection_oos_fold_count=int(generation.get("selection_oos_fold_count",0)),
                          generation_stability=np.nan)
            if family_prediction_frame is not None:
                family_mp = family_prediction_frame.loc[family_prediction_terminal_dates.le(cutoff)]
                mp = family_mp.loc[family_mp["generation_id"].eq(generation["generation_id"])]
                record["matured_prediction_count"] = int(len(mp))
                if len(mp) >= 2:
                    record["rank_ic"] = float(mp["score"].corr(mp["realized_excess"], method="spearman"))
                    top = mp.loc[mp["score"].ge(mp["score"].quantile(.9)), "realized_excess"]
                    record["top_score_realised_excess"] = float(top.mean()) if len(top) else np.nan
                    median_band = mp.loc[mp["score"].between(mp["score"].quantile(.45), mp["score"].quantile(.55)),
                                         "realized_excess"]
                    record["top_vs_median_realised_excess"] = float(top.mean() - median_band.mean()) if len(top) and len(median_band) else np.nan
                    record["calibration_error"] = float((mp["score"] - mp["realized_excess"]).abs().mean())
                    record["score_spread"] = float(mp["score"].quantile(.9) - mp["score"].quantile(.5))
                    record["prediction_residual_drift"] = float((mp["realized_excess"] - mp["score"]).tail(63).mean())
                if "fold_id" in family_mp and len(family_mp):
                    fold_means=family_mp.groupby("fold_id")["realized_excess"].mean()
                    record["fold_stability"]=float(fold_means.gt(0).mean())
                if len(family_mp) and family_mp["generation_id"].nunique()>1:
                    generation_means=family_mp.groupby("generation_id")["realized_excess"].mean()
                    record["generation_stability"]=float(generation_means.gt(0).mean())
            rows.append(record)
    panel = pd.DataFrame(rows)
    if not panel.empty:
        panel = panel.sort_values(["assessment_date", "family_id"]).reset_index(drop=True)
    return panel
