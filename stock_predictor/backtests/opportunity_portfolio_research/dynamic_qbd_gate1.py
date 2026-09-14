"""Performance-only temporal predictability gate for Dynamic-QBD families."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

from .dynamic_qbd_evidence import PERFORMANCE_FEATURES
from .top10_qbd_overfitting_diagnostics import block_bootstrap_mean
from .dynamic_qbd_family_surface import structural_plateau_id_from_family_id


def collapse_policy_variants_to_plateaus(panel: pd.DataFrame) -> pd.DataFrame:
    """Make H/D/exit plateaus, rather than N variants, the inference rows."""
    x = panel.copy()
    x["plateau_id"] = x["family_id"].map(structural_plateau_id_from_family_id)
    if x["plateau_id"].equals(x["family_id"].astype(str)):
        return x
    numeric = [column for column in x.select_dtypes(include=[np.number]).columns if column != "family_id"]
    stable = [column for column in ("generation_id", "generation_refit_date", "information_cutoff") if column in x]
    aggregated = x.groupby(["assessment_date", "plateau_id"], as_index=False)[numeric].median()
    if stable:
        provenance = x.sort_values(["assessment_date", "family_id"]).groupby(
            ["assessment_date", "plateau_id"], as_index=False
        )[stable].first()
        aggregated = aggregated.merge(provenance, on=["assessment_date", "plateau_id"], how="left")
    aggregated["family_id"] = aggregated.pop("plateau_id")
    return aggregated


@dataclass(frozen=True)
class Gate1Result:
    status: str
    predictions: pd.DataFrame
    monthly_rank_metrics: pd.DataFrame
    baseline_comparison: pd.DataFrame
    bootstrap: dict
    horizon_status: dict[int, str]


def add_forward_targets(panel: pd.DataFrame) -> pd.DataFrame:
    x = panel.copy().sort_values(["family_id", "assessment_date"])
    for months in (1, 3):
        x[f"future_{months}m_excess"] = x.groupby("family_id")["relative_wealth"].shift(-months) / x["relative_wealth"] - 1.0
        x[f"future_{months}m_available_at"] = x.groupby("family_id")["assessment_date"].shift(-months)
    return x


def _fit_ridge(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, ridge: float = 1e-4) -> np.ndarray:
    mean = np.nanmean(train_x, axis=0)
    scale = np.nanstd(train_x, axis=0)
    scale[~np.isfinite(scale) | (scale == 0)] = 1.0
    tx = np.nan_to_num((train_x - mean) / scale)
    vx = np.nan_to_num((test_x - mean) / scale)
    design = np.column_stack([np.ones(len(tx)), tx])
    beta = np.linalg.solve(design.T @ design + np.eye(design.shape[1]) * ridge, design.T @ train_y)
    return np.column_stack([np.ones(len(vx)), vx]) @ beta


def run_gate1(panel: pd.DataFrame, *, minimum_train_months: int = 24, block_size: int = 3, switch_cost_bps: float = 10.0) -> Gate1Result:
    forbidden = set(panel.columns) & {"rank_ic", "calibration_error", "prediction_residual_drift", "market_regime", "ticker"}
    panel = collapse_policy_variants_to_plateaus(panel)
    features = [x for x in PERFORMANCE_FEATURES if x in panel]
    if not features:
        raise ValueError("GATE1_PERFORMANCE_FEATURES_MISSING")
    x = add_forward_targets(panel)
    dates = tuple(sorted(pd.to_datetime(x["assessment_date"]).unique()))
    predictions = []
    for target_months in (1, 3):
        target = f"future_{target_months}m_excess"
        for index, assessment in enumerate(dates):
            if index < minimum_train_months:
                continue
            available_at=pd.to_datetime(x[f"future_{target_months}m_available_at"])
            train = x.loc[pd.to_datetime(x["assessment_date"]).lt(assessment) & available_at.le(assessment) & x[target].notna()]
            test = x.loc[pd.to_datetime(x["assessment_date"]).eq(assessment) & x[target].notna()]
            if train.empty or test.empty:
                continue
            predicted = _fit_ridge(train[features].to_numpy(float), train[target].to_numpy(float), test[features].to_numpy(float))
            for row, value in zip(test.itertuples(index=False), predicted):
                predictions.append({"assessment_date": assessment, "family_id": row.family_id, "target_months": target_months,
                                    "predicted_excess": float(value), "realized_excess": float(getattr(row, target))})
    prediction_frame = pd.DataFrame(predictions)
    rank_rows = []
    baseline_rows = []
    prior_selector = {1: None, 3: None}
    static_winner = {}
    if not prediction_frame.empty:
        for (months, assessment), group in prediction_frame.groupby(["target_months", "assessment_date"]):
            ranked = group.sort_values("predicted_excess", ascending=False)
            rank_ic = ranked["predicted_excess"].corr(ranked["realized_excess"], method="spearman") if len(ranked) > 1 else np.nan
            top = float(ranked.iloc[0]["realized_excess"])
            bottom = float(ranked.iloc[-1]["realized_excess"])
            equal = float(ranked["realized_excess"].mean())
            rank_rows.append({"assessment_date": assessment, "target_months": months, "rank_ic": rank_ic,
                              "predicted_top_excess": top, "top_minus_bottom": top-bottom,
                              "direction_accuracy":float(((ranked["predicted_excess"]>0)==(ranked["realized_excess"]>0)).mean()),
                              "magnitude_rmse":float(np.sqrt(np.mean(np.square(ranked["predicted_excess"]-ranked["realized_excess"]))))})
            source=x.loc[pd.to_datetime(x["assessment_date"]).eq(assessment)].set_index("family_id")
            if months not in static_winner:
                history=x.loc[pd.to_datetime(x["assessment_date"]).lt(assessment)]
                static_scores=history.groupby("family_id")["excess_12m"].mean().dropna()
                static_winner[months]=str(static_scores.idxmax()) if len(static_scores) else str(ranked.iloc[0].family_id)
            realized=dict(zip(ranked["family_id"].astype(str),ranked["realized_excess"].astype(float)))
            def winner(feature):
                available=source.loc[source.index.astype(str).isin(realized)]
                values=available[feature].dropna() if feature in available else pd.Series(dtype=float)
                return str(values.idxmax()) if len(values) else None
            available_source=source.loc[source.index.astype(str).isin(realized)]
            forecast_columns=[name for name in ("excess_1m","excess_3m","excess_6m","excess_12m") if name in available_source]
            if forecast_columns:
                forecast_average=available_source[forecast_columns].rank(pct=True).mean(axis=1).dropna()
                forecast_winner=str(forecast_average.idxmax()) if len(forecast_average) else None
            else:
                forecast_winner=None
            selected=str(ranked.iloc[0].family_id); prior=prior_selector[months]
            switch_cost=(switch_cost_bps/10000.0) if prior is not None and prior!=selected else 0.0
            selector_net=top-switch_cost; prior_excess=realized.get(prior,equal) if prior else equal
            oracle=max(realized.values())
            row={"assessment_date":assessment,"target_months":months,"selected_family_id":selected,
                 "selector_excess":selector_net,"selector_gross_excess":top,"switch_cost":switch_cost,
                 "benchmark_only_excess":0.0,"equal_weight_excess":equal,
                 "static_best_development_excess":realized.get(static_winner[months],equal),
                 "previous_incumbent_held_excess":prior_excess,
                 "do_nothing_keep_incumbent_excess":prior_excess,
                 "trailing_1m_winner_excess":realized.get(winner("excess_1m"),equal),
                 "trailing_3m_winner_excess":realized.get(winner("excess_3m"),equal),
                 "trailing_6m_winner_excess":realized.get(winner("excess_6m"),equal),
                 "long_run_historical_best_excess":realized.get(winner("excess_12m"),equal),
                 "simple_forecast_averaging_excess":realized.get(forecast_winner,equal),
                 "oracle_excess":oracle,"oracle_regret":oracle-selector_net,
                 "captured_oracle_alpha":selector_net/oracle if oracle>0 else 0.0,
                 "incremental_vs_equal":selector_net-equal}
            switched=float(prior is not None and prior!=selected)
            for bps in (0,10,20,30,50):
                row[f"selector_excess_{bps}bps"]=top-switched*bps/10000.0
                row[f"incremental_vs_equal_{bps}bps"]=row[f"selector_excess_{bps}bps"]-equal
            baseline_rows.append(row); prior_selector[months]=selected
    ranks = pd.DataFrame(rank_rows)
    baselines = pd.DataFrame(baseline_rows)
    bootstraps = {}
    horizon_status = {}
    for months in (1, 3):
        evidence = baselines.loc[baselines["target_months"].eq(months), "incremental_vs_equal"].tolist() if not baselines.empty else []
        bootstrap = block_bootstrap_mean(evidence, block_size=block_size, repetitions=1000,
                                         seed=17+months) if evidence else {"mean": 0.0, "lower": 0.0, "upper": 0.0}
        rank_values = ranks.loc[ranks["target_months"].eq(months), "rank_ic"] if not ranks.empty else pd.Series(dtype=float)
        rank_positive = bool(len(rank_values) and float(rank_values.median()) > 0)
        incremental_positive = bool(evidence and float(bootstrap.get("q05", -1)) > 0)
        stressed = baselines.loc[baselines["target_months"].eq(months), "incremental_vs_equal_30bps"] if not baselines.empty else pd.Series(dtype=float)
        cost_stress_positive = bool(len(stressed) and float(stressed.mean()) > 0)
        passed = rank_positive and incremental_positive and cost_stress_positive
        horizon_status[months] = f"GATE1_{months}M_PASS" if passed else f"GATE1_{months}M_FAIL"
        bootstraps[str(months)] = bootstrap | {"rank_ic_median_positive": rank_positive,
                                               "cost_stress_30bps_positive": cost_stress_positive}
    passed_horizons = [months for months, value in horizon_status.items() if value.endswith("PASS")]
    status = "GATE1_PASS" if len(passed_horizons) == 2 else (
        "GATE1_PARTIAL_PASS_LIMITED_SELECTOR_AUTHORITY" if passed_horizons
        else "GATE1_FAIL_NO_PERFORMANCE_SELECTOR_AUTHORITY"
    )
    bootstraps["forbidden_inputs_present_but_not_consumed"] = sorted(forbidden)
    bootstraps["selector_authority_target_months"] = passed_horizons
    economic_summary = {}
    if not baselines.empty:
        for months, group in baselines.groupby("target_months"):
            ordered=group.sort_values("assessment_date").reset_index(drop=True)
            economic=ordered if int(months)==1 else ordered.iloc[::3].reset_index(drop=True)
            returns = economic["selector_excess"].astype(float)
            wealth = (1.0 + returns).cumprod()
            drawdown = wealth / wealth.cummax() - 1.0
            downside = returns.loc[returns.lt(0)]
            economic_summary[str(int(months))] = {
                "selected_family_cagr_excess": float(wealth.iloc[-1] ** ((12.0/int(months)) / len(returns)) - 1.0),
                "selected_family_relative_sortino": float(returns.mean() / downside.std(ddof=0)) if len(downside) and downside.std(ddof=0) > 0 else 0.0,
                "selected_family_relative_max_drawdown": float(drawdown.min()),
                "after_cost_cumulative_excess": float(wealth.iloc[-1] - 1.0),
                "oracle_regret_mean": float(economic["oracle_regret"].mean()),
                "captured_oracle_alpha_mean": float(economic["captured_oracle_alpha"].mean()),
                "economic_path_contract": "MONTHLY_SELECTOR_REPLAY" if int(months)==1 else "NON_OVERLAPPING_3M_COHORTS",
                "economic_observations": int(len(economic)),
            }
    bootstraps["economic_summary"] = economic_summary
    return Gate1Result(status, prediction_frame, ranks, baselines, bootstraps, horizon_status)
