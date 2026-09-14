"""Focused contract tests for the causal opportunity-state diagnostics."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from .dynamic_qbd_opportunity_state import (
    ARMS,
    HOLDOUT_START,
    _block_bootstrap,
    _feature_columns,
    _require_before_holdout,
    build_feature_panel,
    build_opportunity_panel,
)


def _fixture(root: Path) -> None:
    run = root / "run"
    (run / "development").mkdir(parents=True)
    families = ["H01_D01_N01_FIXED", "H11_D01_N01_FIXED"]
    json_path = run / "development/family_registry.json"
    json_path.write_text(json.dumps({"families": [{"family_id": x} for x in families]}), encoding="utf-8")
    generation = "g1"
    schedule_rows = []
    valid_rows = []
    for family in families:
        for arm in ARMS:
            schedule_rows.append({"family_id": family, "generation_id": generation, "arm": arm,
                                  "activation_date": "2021-01-01", "resolved_threshold": 0.5,
                                  "resolved_top_fraction": 0.005})
        valid_rows.append({"family_id": family, "generation_id": generation,
                           "information_cutoff": "2020-12-31", "activation_date": "2021-01-01",
                           "resolved_score_quantile": 0.995})
    pd.DataFrame(schedule_rows).to_parquet(run / "abc_generation_schedule.parquet", index=False)
    pd.DataFrame(valid_rows).to_parquet(run / "valid_generations.parquet", index=False)
    rows = []
    for date in ["2021-01-04", "2021-01-05", "2021-01-06", "2021-01-07"]:
        for family, score, excess in [(families[0], 0.8, 0.03), (families[1], 0.7, -0.01)]:
            rows.append({"decision_date": date, "terminal_date": "2021-01-20", "ticker": "AAA",
                         "family_id": family, "generation_id": generation, "model_artifact_id": "m1",
                         "score": score, "realized_excess": excess})
    rows.append(dict(rows[0]))  # exact duplicate must be removed before arm expansion
    rows.append({"decision_date": "2021-01-04", "terminal_date": "2021-01-20", "ticker": "BBB",
                 "family_id": families[0], "generation_id": generation, "model_artifact_id": "m1",
                 "score": 0.1, "realized_excess": 0.0})  # inactive, never a candidate
    pd.DataFrame(rows).to_parquet(run / "matured_generation_predictions.parquet", index=False)
    pd.DataFrame(columns=["ticker", "entry_date", "family_id", "arm"]).to_parquet(run / "development/family_shadow_trades.parquet", index=False)
    dates = pd.date_range("2020-01-01", "2022-12-31", freq="D")
    pd.DataFrame({"date": dates, "urth_value": np.linspace(100, 120, len(dates))}).to_parquet(run / "development/family_shadow_nav.parquet", index=False)


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="dynamic-qbd-opportunity-state-selftest-") as temp:
        root = Path(temp)
        _fixture(root)
        run = root / "run"
        out = root / "out"
        meta = build_opportunity_panel(run, out)
        assert meta["counts"]["active_rows"] == 24, meta
        panel = pd.read_parquet(out / "opportunity_panel.parquet")
        assert len(panel) == 24
        assert panel[["forward_stock_return", "forward_benchmark_return"]].isna().all().all()
        assert panel["forward_excess_return"].notna().all()
        assert panel["decision_date"].max() < HOLDOUT_START
        assert panel.duplicated(["decision_date", "ticker", "family_id", "arm", "generation_id"]).sum() == 0
        feature_meta = build_feature_panel(run, out)
        features = pd.read_parquet(out / "feature_panel.parquet")
        assert feature_meta["rows"] == 24
        assert features["same_ticker_target_rank"].notna().all()
        assert set(features["active_family_count_same_ticker"].dropna()) == {2}, features[["decision_date", "ticker", "arm", "active_family_count_same_ticker"]].drop_duplicates().to_dict("records")
        assert set(features["eligible_family_count_same_ticker"].dropna()) == {2}, features[["decision_date", "ticker", "arm", "eligible_family_count_same_ticker"]].drop_duplicates().to_dict("records")
        assert _feature_columns("T2_SCORE_ONLY") == ["prediction_score", "score_percentile", "distance_to_threshold", "horizon_h", "holding_d", "max_names_n"]
        assert _block_bootstrap(pd.Series([1.0, 2.0, 3.0]), seed=7, repetitions=20) == _block_bootstrap(pd.Series([1.0, 2.0, 3.0]), seed=7, repetitions=20)
        try:
            _require_before_holdout(pd.DataFrame({"x": [HOLDOUT_START]}), "x", HOLDOUT_START)
        except ValueError as exc:
            assert "FINAL_HOLDOUT_BOUNDARY_VIOLATION" in str(exc)
        else:
            raise AssertionError("holdout boundary did not fail closed")
    print("dynamic_qbd_opportunity_state_self_test: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
