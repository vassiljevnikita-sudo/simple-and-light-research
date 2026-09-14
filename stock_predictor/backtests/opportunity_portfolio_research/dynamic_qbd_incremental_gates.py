"""Time-aware Gate 1B, Gate 2 and Gate 3 incremental ablations."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

from .dynamic_qbd_evidence import PERFORMANCE_FEATURES, DOWNSIDE_FEATURES, GENERATION_HEALTH_FEATURES
from .dynamic_qbd_gate1 import _fit_ridge, add_forward_targets, collapse_policy_variants_to_plateaus
from .top10_qbd_overfitting_diagnostics import block_bootstrap_mean

MARKET_STATE_FEATURES = ("benchmark_trend", "market_volatility", "market_breadth", "cross_sectional_dispersion",
                         "liquidity_state")


@dataclass(frozen=True)
class IncrementalGateResult:
    gate: str
    status: str
    horizon_status: dict[int, str]
    monthly_metrics: pd.DataFrame
    bootstrap: dict


def run_incremental_gate(panel: pd.DataFrame, *, gate: str, added_features, base_features=None,
                         base_features_by_horizon: dict[int, tuple[str, ...]] | None = None,
                         minimum_train_months: int = 24, block_size: int = 3) -> IncrementalGateResult:
    x = add_forward_targets(collapse_policy_variants_to_plateaus(panel))
    added = [column for column in added_features if column in x and x[column].notna().any()]
    if not added:
        return IncrementalGateResult(gate, f"{gate}_NOT_RUN_INPUT_MISSING", {1:f"{gate}_1M_NOT_RUN",3:f"{gate}_3M_NOT_RUN"},
                                     pd.DataFrame(), {"reason":"INPUT_MISSING","base_features":[],
                                                      "available_added_features":added})
    dates = tuple(sorted(pd.to_datetime(x["assessment_date"]).unique()))
    rows = []
    feature_contract = {}
    for months in (1, 3):
        requested_base=tuple((base_features_by_horizon or {}).get(months,base_features or PERFORMANCE_FEATURES))
        base=[column for column in requested_base if column in x and x[column].notna().any()]
        incremental=[column for column in added if column not in base]
        feature_contract[str(months)]={"base":base,"incremental":incremental,"target":"FUTURE_EXCESS_RETURN"}
        if not base or not incremental:
            continue
        target = f"future_{months}m_excess"
        available = pd.to_datetime(x[f"future_{months}m_available_at"])
        for index, assessment in enumerate(dates):
            if index < minimum_train_months:
                continue
            train = x.loc[pd.to_datetime(x["assessment_date"]).lt(assessment) & available.le(assessment) & x[target].notna()]
            test = x.loc[pd.to_datetime(x["assessment_date"]).eq(assessment) & x[target].notna()]
            if train.empty or test.empty:
                continue
            base_prediction = _fit_ridge(train[base].to_numpy(float), train[target].to_numpy(float), test[base].to_numpy(float))
            full_prediction = _fit_ridge(train[base+incremental].to_numpy(float), train[target].to_numpy(float), test[base+incremental].to_numpy(float))
            realized = test[target].to_numpy(float)
            base_top = int(np.argmax(base_prediction)); full_top = int(np.argmax(full_prediction))
            rows.append({"assessment_date":assessment,"target_months":months,
                         "base_top_excess":float(realized[base_top]),"full_top_excess":float(realized[full_top]),
                         "incremental_top_excess":float(realized[full_top]-realized[base_top]),
                         "base_rank_ic":pd.Series(base_prediction).corr(pd.Series(realized),method="spearman"),
                         "full_rank_ic":pd.Series(full_prediction).corr(pd.Series(realized),method="spearman"),
                         "incremental_rank_ic":pd.Series(full_prediction).corr(pd.Series(realized),method="spearman")-
                                               pd.Series(base_prediction).corr(pd.Series(realized),method="spearman")})
    metrics = pd.DataFrame(rows)
    horizon_status, bootstraps = {}, {}
    for months in (1, 3):
        evidence = metrics.loc[metrics["target_months"].eq(months), "incremental_top_excess"].dropna() if not metrics.empty else pd.Series(dtype=float)
        bootstrap = block_bootstrap_mean(evidence.tolist(),block_size=block_size,repetitions=1000,seed=31+months) if len(evidence) else {}
        rank = metrics.loc[metrics["target_months"].eq(months), "incremental_rank_ic"].dropna() if not metrics.empty else pd.Series(dtype=float)
        passed = bool(len(evidence) and len(rank) and float(bootstrap.get("q05",-1))>0 and float(rank.median())>0)
        horizon_status[months] = f"{gate}_{months}M_PASS" if passed else f"{gate}_{months}M_FAIL"
        bootstraps[str(months)] = bootstrap | {"median_incremental_rank_ic":float(rank.median()) if len(rank) else None}
    passes = sum(value.endswith("PASS") for value in horizon_status.values())
    status = f"{gate}_PASS" if passes==2 else (f"{gate}_PARTIAL_PASS" if passes else f"{gate}_FAIL")
    bootstraps["feature_contract"] = feature_contract
    return IncrementalGateResult(gate,status,horizon_status,metrics,bootstraps)


def run_gate1b(panel: pd.DataFrame, **kwargs) -> IncrementalGateResult:
    return run_incremental_gate(panel,gate="GATE1B",added_features=DOWNSIDE_FEATURES,**kwargs)


def run_gate2(panel: pd.DataFrame, *, include_downside: bool | dict[int, bool] = False, **kwargs) -> IncrementalGateResult:
    accepted={months:(bool(include_downside.get(months,False)) if isinstance(include_downside,dict) else bool(include_downside)) for months in (1,3)}
    bases={months:PERFORMANCE_FEATURES+(DOWNSIDE_FEATURES if accepted[months] else ()) for months in (1,3)}
    return run_incremental_gate(panel,gate="GATE2",base_features_by_horizon=bases,
                                added_features=GENERATION_HEALTH_FEATURES,**kwargs)


def run_gate3(panel: pd.DataFrame, *, include_downside: bool | dict[int, bool] = False, **kwargs) -> IncrementalGateResult:
    accepted={months:(bool(include_downside.get(months,False)) if isinstance(include_downside,dict) else bool(include_downside)) for months in (1,3)}
    bases={months:PERFORMANCE_FEATURES+(DOWNSIDE_FEATURES if accepted[months] else ())+GENERATION_HEALTH_FEATURES for months in (1,3)}
    return run_incremental_gate(panel,gate="GATE3",base_features_by_horizon=bases,
                                added_features=MARKET_STATE_FEATURES,**kwargs)
