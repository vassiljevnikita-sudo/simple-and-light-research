"""Typed health snapshots and deterministic health calculations."""
from __future__ import annotations
from dataclasses import dataclass
from datetime import date
from statistics import mean, median
from .top10_qbd_router_contracts import ActivityLevel, HealthLevel
from .top10_qbd_shadow_ledger import MaturedOutcome

@dataclass(frozen=True)
class StructuralHealth:
    level: HealthLevel; reason_codes: tuple[str, ...]; checked_at: date
@dataclass(frozen=True)
class ActivityPosterior:
    a: float; b: float; eligible_opportunities: int; actual_trades: int
    opportunity_time_since_last_trade: int; last_trade_date: date | None
@dataclass(frozen=True)
class ActivityHealth:
    level: ActivityLevel; zero_trade_probability: float; posterior: ActivityPosterior; evaluated_at: date
@dataclass(frozen=True)
class AlphaHealth:
    discounted_net_excess: float; mean_net_excess: float | None; median_net_excess: float | None
    relative_drawdown: float; effective_n: float; lcb_net_excess: float | None; evaluated_at: date
@dataclass(frozen=True)
class RegimeScore:
    regime_id: str; local_score: float | None; global_score: float; rho: float; shrunk_score: float; effective_local_n: float
@dataclass(frozen=True)
class ChangePointState:
    change_probability: float; expected_run_length: float; forgetting_multiplier: float
    evaluated_at: date; opaque_detector_state: dict[str, object]
@dataclass(frozen=True)
class ExpertHealthSnapshot:
    expert_id: str; as_of: date; structural: StructuralHealth; activity: ActivityHealth
    alpha: AlphaHealth; regime: RegimeScore; uncertainty_score: float
    changepoint: ChangePointState | None; matured_evidence_cursor: str

def alpha_from_outcomes(previous: AlphaHealth, matured: tuple[MaturedOutcome, ...], at: date, discount: float = .97) -> AlphaHealth:
    vals = [x.net_excess for x in matured]
    if not vals: return previous
    prior = previous.discounted_net_excess
    discounted = prior * discount + vals[-1] * (1 - discount)
    n = previous.effective_n + len(vals)
    all_values = vals if previous.mean_net_excess is None else [previous.mean_net_excess] * max(1, int(previous.effective_n)) + vals
    m = mean(all_values); med = median(all_values)
    variance = sum((x - m) ** 2 for x in all_values) / max(1, len(all_values) - 1)
    lcb = m - 1.96 * (variance / max(1.0, n)) ** .5
    peak = max(0.0, m); drawdown = min(0.0, med - peak)
    return AlphaHealth(discounted, m, med, drawdown, n, lcb, at)
