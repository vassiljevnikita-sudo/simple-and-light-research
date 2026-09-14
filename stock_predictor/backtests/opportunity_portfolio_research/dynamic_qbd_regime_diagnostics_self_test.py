"""Focused causal and reproducibility tests for regime diagnostics."""
from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from .dynamic_qbd_regime_diagnostics import (
    ARMS,
    _ablation_masks,
    _bootstrap,
    _require_before_holdout,
    _selector_signatures,
    _state_frame,
    activity_persistence_diagnostics,
    build_cluster_membership,
    build_cluster_panel,
    build_family_monthly_panel,
    future_spread_diagnostics,
    persistence_diagnostics,
    selector_feasibility,
)


def _registry(families):
    return {"schema_version": "TEST", "family_registry_hash": "test", "families": [
        {"family_id": family_id, "horizon_sessions": h, "holding_days": d, "max_names": n,
         "entry_policy_rule": {"exit_policy": {"family": "FIXED"}}
         } for family_id, h, d, n in families]}


def _synthetic_run(root: Path) -> None:
    development = root / "development"
    development.mkdir(parents=True)
    families = [("H01_D01_N01_FIXED", 1, 1, 1), ("H30_D30_N01_FIXED", 30, 30, 1)]
    (development / "family_registry.json").write_text(json.dumps(_registry(families)), encoding="utf-8")
    dates = pd.date_range("2025-01-31", periods=8, freq="ME")
    arms = ARMS
    abc_rows, nav_rows, trade_rows, rich_rows = [], [], [], []
    for family_index, (family_id, _, _, _) in enumerate(families):
        for arm_index, arm in enumerate(arms):
            strategy = 10000.0
            benchmark = 10000.0
            for month_index, assessment in enumerate(dates):
                strategy *= 1.0 + (0.01 if family_index == 0 else -0.002) + arm_index * .0005
                benchmark *= 1.0 + .002
                abc_rows.append({"date": assessment, "strategy_value": strategy, "urth_value": benchmark,
                                 "family_id": family_id, "arm": arm, "assessment_date": assessment,
                                 "relative_wealth": strategy / benchmark, "relative_return": np.nan})
                if arm_index == 2:
                    rich_rows.append({"assessment_date": assessment, "family_id": family_id,
                                      "relative_wealth": strategy / benchmark,
                                      "generation_refit_date": assessment - pd.Timedelta(days=5)})
                for day in pd.date_range(assessment - pd.offsets.MonthBegin(1) + pd.Timedelta(days=1), assessment, freq="D"):
                    nav_rows.append({"date": day, "positions": 1 if family_index == 0 else 0,
                                     "family_id": family_id, "arm": arm})
            if family_index == 0:
                trade_rows.append({"entry_date": dates[0] + pd.Timedelta(days=1), "family_id": family_id, "arm": arm})
    abc = pd.DataFrame(abc_rows).sort_values(["family_id", "arm", "assessment_date"])
    abc["relative_return"] = abc.groupby(["family_id", "arm"])["relative_wealth"].pct_change()
    abc.to_parquet(development / "abc_monthly_evidence.parquet", index=False)
    pd.DataFrame(nav_rows).to_parquet(development / "family_shadow_nav.parquet", index=False)
    pd.DataFrame(trade_rows).to_parquet(development / "family_shadow_trades.parquet", index=False)
    pd.DataFrame(rich_rows).to_parquet(development / "monthly_family_evidence.parquet", index=False)


def main() -> int:
    with tempfile.TemporaryDirectory() as folder:
        run_root = Path(folder) / "run"
        _synthetic_run(run_root)
        output = Path(folder) / "out"
        panel, meta = build_family_monthly_panel(run_root, output, pd.Timestamp("2026-07-25"))
        assert len(panel) == 2 * 3 * 8
        assert panel.duplicated(["family_id", "arm", "assessment_date"]).sum() == 0
        first = panel.loc[panel["family_id"].eq("H01_D01_N01_FIXED") & panel["arm"].eq(ARMS[0])].sort_values("assessment_date")
        assert pd.isna(first.iloc[-1]["forward_excess_1m"]) and pd.isna(first.iloc[-2]["forward_excess_3m"])
        assert first.iloc[1]["forward_excess_1m"] == first.iloc[2]["excess_return_1m"]
        assert set(panel.loc[panel["family_id"].eq("H30_D30_N01_FIXED"), "activity_state"]) == {"NO_OPPORTUNITY"}
        membership = build_cluster_membership(pd.DataFrame([
            {"family_id": "H01_D01_N01_FIXED", "horizon_h": 1, "holding_d": 1, "max_names_n": 1, "exit_mode": "FIXED", "family_variant": "FIXED"},
            {"family_id": "H30_D30_N01_FIXED", "horizon_h": 30, "holding_d": 30, "max_names_n": 1, "exit_mode": "FIXED", "family_variant": "FIXED"},
        ]))
        assert len(membership) == 8 and membership.duplicated(["cluster_level", "family_id"]).sum() == 0
        clusters = build_cluster_panel(panel, membership)
        assert clusters.duplicated(["cluster_level", "cluster_id", "arm", "assessment_date"]).sum() == 0
        state = _state_frame(clusters)
        rank, quartile, winner, transition, duration, turnover, _ = persistence_diagnostics(state, repetitions=40, block_size=2, seed=11)
        activity = activity_persistence_diagnostics(state, repetitions=40, block_size=2, seed=11)
        spread, _ = future_spread_diagnostics(state, repetitions=40, block_size=2, seed=11)
        selector, _ = selector_feasibility(state, repetitions=40, block_size=2, seed=11)
        assert len(rank) and len(quartile) and len(winner) and len(transition) and len(duration) and len(turnover)
        assert set(activity["activity_condition"]) >= {"ALL", "ACTIVE_ONLY", "NO_OPPORTUNITY"}
        assert len(spread) and len(selector)
        assert np.isfinite(selector.select_dtypes(include=[np.number]).to_numpy(dtype=float)).all()
        signature_state = state.loc[
            state["cluster_level"].eq("ECONOMIC_REGION") & state["arm"].eq(ARMS[0])]
        signatures = _selector_signatures(signature_state)
        assert signatures["F3_PERSISTENT_TOP"][0] == ()
        assert _bootstrap([1, 2, 3], seed=5, repetitions=50, block_size=2) == _bootstrap([1, 2, 3], seed=5, repetitions=50, block_size=2)
        masks = _ablation_masks(panel)
        assert masks["B_REMOVE_H30"].all() is not True
        assert panel.loc[masks["B_REMOVE_H30"], "horizon_h"].max() < 30
        try:
            _require_before_holdout(pd.DataFrame({"assessment_date": [datetime(2026, 7, 25)]}), "assessment_date", pd.Timestamp("2026-07-25"))
            raise AssertionError("holdout boundary accepted")
        except ValueError as exc:
            assert "FINAL_HOLDOUT_BOUNDARY_VIOLATION" in str(exc)
        print("DYNAMIC_QBD_REGIME_DIAGNOSTICS_SELF_TEST_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
