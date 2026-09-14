"""Frozen, serialisable research contract for TOP10_QBD_ROUTER_V1.

This module deliberately contains no data access or broker integration.  It is
the stable boundary shared by the router, replay, tests and freeze gate.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, is_dataclass
from datetime import date
from enum import Enum
from typing import Any, Mapping

from .top10_qbd_policy import RouterArm


class ExpertType(str, Enum):
    STOCK = "STOCK"
    BENCHMARK = "BENCHMARK"
    CASH = "CASH"


class LifecycleState(str, Enum):
    SHADOW = "SHADOW"
    CANDIDATE = "CANDIDATE"
    ENSEMBLE = "ENSEMBLE"
    CHAMPION = "CHAMPION"
    PROBATION = "PROBATION"
    SUSPENDED = "SUSPENDED"


class HealthLevel(str, Enum):
    PASS = "PASS"
    WATCH = "WATCH"
    FAIL = "FAIL"


class ActivityLevel(str, Enum):
    NORMAL = "NORMAL"
    WATCH = "WATCH"
    ANOMALOUS = "ANOMALOUS"
    CRITICAL = "CRITICAL"


def _json_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, date):
        return value.isoformat()
    if is_dataclass(value):
        return {k: _json_value(v) for k, v in asdict(value).items()}
    if isinstance(value, Mapping):
        return {str(k): _json_value(v) for k, v in sorted(value.items(), key=lambda x: str(x[0]))}
    if isinstance(value, (tuple, list)):
        return [_json_value(v) for v in value]
    return value


def canonical_json(value: Any) -> str:
    return json.dumps(_json_value(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def sha256_fingerprint(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class FrozenQbdRouterPolicyV1:
    contract_id: str = "TOP10_QBD_ROUTER_V1"
    policy_version: str = "TOP10_QBD_ROUTER_V1_POLICY_2026_01"
    router_arm: RouterArm = RouterArm.K_UNCERTAINTY_LCB
    assessment_cadence_sessions: int = 1
    allocation_cadence_sessions: int = 1
    alpha_discount: float = 0.97
    softmax_eta: float = 4.0
    min_matured_observations: int = 20
    min_eligible_opportunities: int = 10
    promotion_lcb_margin: float = 0.0005
    probation_lcb_margin: float = -0.0005
    suspension_lcb_margin: float = -0.002
    reactivation_lcb_margin: float = 0.0002
    min_champion_sessions: int = 10
    activity_prior_a: float = 1.0
    activity_prior_b: float = 9.0
    activity_watch_zero_prob: float = 0.25
    activity_anomalous_zero_prob: float = 0.05
    activity_critical_zero_prob: float = 0.01
    regime_shrinkage_k: float = 20.0
    changepoint_enabled: bool = True
    changepoint_config: Mapping[str, Any] = None  # type: ignore[assignment]
    stock_roundtrip_bps: float = 20.0
    benchmark_roundtrip_bps: float = 5.0
    switch_cost_bps: float = 10.0
    fallback_order: tuple[str, ...] = ("MSCI_WORLD", "CASH")
    final_holdout_start: date = date(2026, 7, 25)
    final_holdout_end: date = date(2026, 12, 31)

    def __post_init__(self) -> None:
        if self.changepoint_config is None:
            object.__setattr__(self, "changepoint_config", {"hazard": 0.02, "decay": 0.90, "alert": 0.65})
        if self.final_holdout_start > self.final_holdout_end:
            raise ValueError("final holdout boundaries are invalid")
        if self.router_arm is None or any(not x for x in self.fallback_order):
            raise ValueError("policy identity and fallback order are required")
        if "LIVE" in canonical_json(self).upper() or "BROKER" in canonical_json(self).upper():
            raise ValueError("live trading is not representable by the research contract")

    def to_json(self) -> str:
        return canonical_json(self)

    @property
    def policy_hash(self) -> str:
        return sha256_fingerprint(self)


def default_policy(arm: RouterArm = RouterArm.K_UNCERTAINTY_LCB) -> FrozenQbdRouterPolicyV1:
    return FrozenQbdRouterPolicyV1(router_arm=arm)
