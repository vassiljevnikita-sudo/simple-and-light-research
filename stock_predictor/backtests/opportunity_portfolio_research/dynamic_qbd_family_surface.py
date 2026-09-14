"""Dynamic-QBD adapter over the existing H1-H30/HxD surface."""
from __future__ import annotations

from itertools import product
import re

from .portfolio_policy_contracts import Policy
from .portfolio_policy_search import SEARCH_QUANTILES, SEARCH_TOP_FRACTIONS
from .dynamic_qbd_generation_contracts import ExpertFamilySpec
from .dynamic_qbd_runtime_resources import configure_memory_peak

# Dynamic-QBD is a local research suite with potentially large ProcessPool
# fan-out. Apply the memory ceiling as soon as the canonical H×D×N surface is
# imported so both pipeline and Development 2016-2025 entrypoints inherit the same
# process-tree contract before any heavy panel/model allocation starts.
configure_memory_peak()

DYNAMIC_QBD_HORIZONS = tuple(range(1, 31))
DYNAMIC_QBD_MAX_NAMES = tuple(range(1, 7))


def structural_plateau_id(horizon: int, holding_days: int, exit_mode: str) -> str:
    """Ex-ante three-session neighborhoods; never derived from realized performance."""
    h0 = ((int(horizon) - 1) // 3) * 3 + 1
    d0 = ((int(holding_days) - 1) // 3) * 3 + 1
    return f"H{h0:02d}-{min(30,h0+2):02d}_D{d0:02d}-{min(30,d0+2):02d}_{exit_mode}"


def structural_plateau_id_from_family_id(family_id: str) -> str:
    match = re.fullmatch(r"H(\d{2})_D(\d{2})_N\d{2}_(FIXED|LEARNED_EXIT)", str(family_id))
    if not match:
        return str(family_id)
    return structural_plateau_id(int(match.group(1)), int(match.group(2)), match.group(3))


def dynamic_qbd_cells():
    return tuple((h, d) for h in DYNAMIC_QBD_HORIZONS for d in range(1, h + 1))


def dynamic_entry_grid(horizon: int, holding_days: int, *, sleeve: float = .50):
    if (horizon, holding_days) not in set(dynamic_qbd_cells()):
        raise ValueError("DYNAMIC_QBD_INVALID_H_D_CELL")
    return tuple(Policy(horizon, q, top, n, holding_days, "FIXED", 0.0, "IGNORE_NEW", "EQUAL_ACTIVE", sleeve)
                 for q, top, n in product(SEARCH_QUANTILES, SEARCH_TOP_FRACTIONS, DYNAMIC_QBD_MAX_NAMES))


def assert_surface_contract() -> None:
    if len(dynamic_qbd_cells()) != 465:
        raise AssertionError("DYNAMIC_QBD_SURFACE_NOT_465_CELLS")
    if {p.max_names for p in dynamic_entry_grid(30, 30)} != set(range(1, 7)):
        raise AssertionError("DYNAMIC_QBD_N1_N6_INCOMPLETE")


def build_family_specs(*, feature_schema_sha256: str, model_family: str = "RIDGE_HGB_FROZEN_RULE",
                       training_window_sessions: int = 504, calibration_window_sessions: int = 252,
                       purge_sessions: int = 30,
                       refit_cadence: str = "MONTH_END", random_seed: int = 17,
                       include_learned_exit: bool = False, score_quantile: float = .975,
                       top_fraction: float = .005,
                       cost_contract: dict | None = None,
                       tax_contract: dict | None = None) -> tuple[ExpertFamilySpec, ...]:
    cost_contract = dict(cost_contract or {"roundtrip_bps": 20.0})
    tax_contract = dict(tax_contract or {"mode": "PRE_TAX", "enabled": False})
    families = []
    for horizon, holding in dynamic_qbd_cells():
        exit_modes = ("FIXED", "LEARNED_EXIT") if include_learned_exit and holding > 1 else ("FIXED",)
        for max_names in DYNAMIC_QBD_MAX_NAMES:
            for exit_mode in exit_modes:
                family_id = f"H{horizon:02d}_D{holding:02d}_N{max_names:02d}_{exit_mode}"
                families.append(ExpertFamilySpec(
                    family_id, horizon, holding, max_names,
                    {"score_quantile": float(score_quantile), "top_fraction": float(top_fraction),
                     "minimum_active_dates": 8, "selection": "FROZEN_FAMILY_POLICY",
                     "sleeve_contract":"ENTRY_NOTIONAL_CAP_MARK_TO_MARKET_DRIFT_DIAGNOSTIC",
                     "structural_plateau_id": structural_plateau_id(horizon, holding, exit_mode),
                     "optional_policy_optimization_design": {
                         "score_quantiles": SEARCH_QUANTILES,
                         "top_fractions": SEARCH_TOP_FRACTIONS,
                         "arms": ("B2", "C2"),
                     }},
                    {"family": exit_mode, "execution": "NEXT_SESSION_OPEN", "replacement": "IGNORE_NEW"},
                    feature_schema_sha256, model_family,
                    {"primitive": "V5_POINT_IN_TIME_WALK_FORWARD", "purge_sessions": int(purge_sessions)},
                    {"rule": "FROZEN_CAUSAL_SELECTION", "candidate_models": ("RIDGE", "HGB")},
                    training_window_sessions, calibration_window_sessions,
                    {"method": "PRIOR_OOS_DAILY_TOP_SCORE_QUANTILE", "score_quantile": float(score_quantile)},
                    refit_cadence, {"ticker": "URTH", "idle_capital": "URTH"},
                    cost_contract, tax_contract, random_seed,
                ))
    return tuple(families)
