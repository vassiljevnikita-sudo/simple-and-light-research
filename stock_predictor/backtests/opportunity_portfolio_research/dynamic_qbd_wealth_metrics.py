"""Benchmark-relative downside and capital-impairment diagnostics."""
from __future__ import annotations

import math
import numpy as np
import pandas as pd


def wealth_path_metrics_at_indices(
    curve: pd.DataFrame,
    endpoint_indices,
    *,
    alpha: float = 0.05,
) -> dict[int, dict]:
    """Calculate prefix wealth metrics without rebuilding every prefix frame.

    ``build_monthly_family_evidence`` needs the same historical metrics as
    :func:`wealth_path_metrics` at many increasing month-end cutoffs.  The
    former implementation copied, sorted and rescanned the whole history for
    every cutoff.  This helper keeps the exact prefix definition but derives
    the reusable wealth, drawdown and duration arrays once.  Quantile-tail
    metrics still use the same prefix values and ``ceil(n * alpha)`` rule as
    the scalar implementation.
    """
    required = {"date", "strategy_value", "urth_value"}
    missing = required - set(curve)
    if missing or curve.empty:
        raise ValueError(f"WEALTH_CURVE_INVALID:{sorted(missing)}")
    x = curve.loc[:, ["date", "strategy_value", "urth_value"]].copy().sort_values("date")
    dates = pd.to_datetime(x["date"]).reset_index(drop=True)
    relative = (
        x["strategy_value"].astype(float).to_numpy()
        / x["urth_value"].astype(float).to_numpy()
    )
    count = len(relative)
    returns = pd.Series(relative).pct_change().dropna().to_numpy(dtype=float)
    peak = np.maximum.accumulate(relative)
    drawdown = relative / peak - 1.0
    underwater = drawdown < 0

    current_duration = np.zeros(count, dtype=np.int64)
    for index in range(1, count):
        current_duration[index] = current_duration[index - 1] + 1 if underwater[index] else 0
    if count and underwater[0]:
        current_duration[0] = 1
    longest_duration = np.maximum.accumulate(current_duration) if count else current_duration
    underwater_count = np.cumsum(underwater, dtype=np.int64)
    impairment = np.cumsum(np.maximum(-drawdown, 0.0), dtype=float)
    impairment_5 = np.cumsum(drawdown <= -0.05, dtype=np.int64)
    impairment_10 = np.cumsum(drawdown <= -0.10, dtype=np.int64)

    recovery_durations = []
    underwater_start = None
    completed_recoveries_by_index = {}
    for index, value in enumerate(drawdown):
        if value < 0:
            if underwater_start is None:
                underwater_start = index
        elif underwater_start is not None:
            recovery_durations.append(index - underwater_start)
            completed_recoveries_by_index[index] = tuple(recovery_durations)
            underwater_start = None

    endpoints = sorted({int(index) for index in endpoint_indices})
    if any(index < 0 or index >= count for index in endpoints):
        raise IndexError("WEALTH_ENDPOINT_INDEX_OUT_OF_RANGE")
    result = {}
    for endpoint in endpoints:
        prefix_returns = returns[:endpoint]
        prefix_drawdown = drawdown[: endpoint + 1]
        downside = np.minimum(prefix_returns, 0.0)
        downside_deviation = float(np.sqrt(np.mean(np.square(downside)))) if len(prefix_returns) else 0.0
        annual_excess = float(prefix_returns.mean() * 252) if len(prefix_returns) else 0.0
        sortino = (annual_excess / (downside_deviation * np.sqrt(252))
                   if downside_deviation > 0 else (np.inf if annual_excess > 0 else 0.0))
        tail_count = max(1, int(np.ceil(len(prefix_returns) * alpha))) if len(prefix_returns) else 0
        losses = np.sort(prefix_returns[prefix_returns < 0])
        expected_shortfall = float(losses[:tail_count].mean()) if len(losses) and tail_count else 0.0
        dd_tail_count = max(1, int(np.ceil(len(prefix_drawdown) * alpha)))
        dd_losses = np.sort(prefix_drawdown[prefix_drawdown < 0])
        cdar = float(dd_losses[:dd_tail_count].mean()) if len(dd_losses) else 0.0
        recoveries = completed_recoveries_by_index.get(endpoint)
        if recoveries is None:
            # The fallback is only used between recovery endpoints.  The
            # stored tuples are cumulative, so retain the latest one.
            prior = [index for index in completed_recoveries_by_index if index <= endpoint]
            recoveries = completed_recoveries_by_index[max(prior)] if prior else ()
        result[endpoint] = {
            "relative_sortino": float(sortino),
            "relative_downside_deviation": downside_deviation,
            "relative_max_drawdown": float(np.min(prefix_drawdown)),
            "cdar_95": cdar,
            "expected_shortfall_95": expected_shortfall,
            "current_drawdown_depth": float(drawdown[endpoint]),
            "drawdown_duration_sessions": int(current_duration[endpoint]),
            "max_drawdown_duration_sessions": int(longest_duration[endpoint]),
            "median_recovery_duration_sessions": float(np.median(recoveries)) if recoveries else 0.0,
            "time_under_water_fraction": float(underwater_count[endpoint] / (endpoint + 1)),
            "capital_impairment_area": float(impairment[endpoint]),
            "capital_impairment_gt_5pct_fraction": float(impairment_5[endpoint] / (endpoint + 1)),
            "capital_impairment_gt_10pct_fraction": float(impairment_10[endpoint] / (endpoint + 1)),
            "relative_wealth": float(relative[endpoint]),
            "period_days": int((dates.iloc[endpoint] - dates.iloc[0]).days),
        }
    return result


def wealth_path_metrics(curve: pd.DataFrame, *, alpha: float = 0.05) -> dict:
    required = {"date", "strategy_value", "urth_value"}
    missing = required - set(curve)
    if missing or curve.empty:
        raise ValueError(f"WEALTH_CURVE_INVALID:{sorted(missing)}")
    x = curve.loc[:, ["date", "strategy_value", "urth_value"]].copy().sort_values("date")
    x["relative_wealth"] = x["strategy_value"].astype(float) / x["urth_value"].astype(float)
    returns = x["relative_wealth"].pct_change().dropna()
    downside = returns.clip(upper=0)
    downside_deviation = float(np.sqrt(np.mean(np.square(downside)))) if len(downside) else 0.0
    annual_excess = float(returns.mean() * 252) if len(returns) else 0.0
    sortino = annual_excess / (downside_deviation * math.sqrt(252)) if downside_deviation > 0 else (math.inf if annual_excess > 0 else 0.0)
    peak = x["relative_wealth"].cummax()
    drawdown = x["relative_wealth"] / peak - 1.0
    losses = returns[returns < 0].sort_values()
    tail_count = max(1, int(math.ceil(len(returns) * alpha))) if len(returns) else 0
    expected_shortfall = float(losses.iloc[:tail_count].mean()) if len(losses) and tail_count else 0.0
    dd_losses = drawdown[drawdown < 0].sort_values()
    cdar_count = max(1, int(math.ceil(len(drawdown) * alpha)))
    cdar = float(dd_losses.iloc[:cdar_count].mean()) if len(dd_losses) else 0.0
    longest = current = 0
    recovery_durations = []
    underwater_start = None
    dates = pd.to_datetime(x["date"])
    for i, value in enumerate(drawdown):
        if value < 0:
            current += 1
            longest = max(longest, current)
            if underwater_start is None:
                underwater_start = i
        else:
            if underwater_start is not None:
                recovery_durations.append(i - underwater_start)
            underwater_start = None
            current = 0
    impairment = -drawdown.clip(upper=0)
    return {
        "relative_sortino": float(sortino),
        "relative_downside_deviation": downside_deviation,
        "relative_max_drawdown": float(drawdown.min()),
        "cdar_95": cdar,
        "expected_shortfall_95": expected_shortfall,
        "current_drawdown_depth": float(drawdown.iloc[-1]),
        "drawdown_duration_sessions": int(current),
        "max_drawdown_duration_sessions": int(longest),
        "median_recovery_duration_sessions": float(np.median(recovery_durations)) if recovery_durations else 0.0,
        "time_under_water_fraction": float(drawdown.lt(0).mean()),
        "capital_impairment_area": float(impairment.sum()),
        "capital_impairment_gt_5pct_fraction": float(drawdown.le(-0.05).mean()),
        "capital_impairment_gt_10pct_fraction": float(drawdown.le(-0.10).mean()),
        "relative_wealth": float(x["relative_wealth"].iloc[-1]),
        "period_days": int((dates.iloc[-1] - dates.iloc[0]).days),
    }
