"""Deterministic smoke contract for dynamic_qbd_fold_event_counterfactual.

This file is intentionally not auto-executed by the implementation workflow.
The user runs it locally.
"""
from __future__ import annotations

from datetime import date

import pandas as pd

from .dynamic_qbd_fold_event_counterfactual import (
    _event_rows,
    _segment_metrics,
    _state_fingerprint,
)


def _audit_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "horizon": 3,
                "assessment_date": "2020-08-31",
                "f_refit": True,
                "candidate_id": "RIDGE_A1",
                "model_family": "RIDGE",
                "parameters_json": '{"alpha": 1.0}',
                "model_artifact_id": "M1",
                "generation_id": "G1",
                "prediction_artifact_path": "unused-1",
                "fold_count": 2,
                "evidence_fingerprint": "E2",
                "resolved_threshold": 0.02,
                "resolved_top_fraction": 0.01,
                "calibration_observations": 100,
                "train_start": "2018-01-02",
                "train_end": "2020-01-31",
            },
            {
                "horizon": 3,
                "assessment_date": "2020-09-30",
                "f_refit": False,
                "candidate_id": "RIDGE_A1",
                "model_family": "RIDGE",
                "parameters_json": '{"alpha": 1.0}',
                "model_artifact_id": "M_MONTH",
                "generation_id": "G_MONTH",
                "prediction_artifact_path": "unused-month",
                "fold_count": 2,
                "evidence_fingerprint": "E2",
                "resolved_threshold": 0.019,
                "resolved_top_fraction": 0.01,
                "calibration_observations": 100,
                "train_start": "2018-02-01",
                "train_end": "2020-02-28",
            },
            {
                "horizon": 3,
                "assessment_date": "2021-02-26",
                "f_refit": True,
                "candidate_id": "RIDGE_A1",
                "model_family": "RIDGE",
                "parameters_json": '{"alpha": 1.0}',
                "model_artifact_id": "M2",
                "generation_id": "G2",
                "prediction_artifact_path": "unused-2",
                "fold_count": 3,
                "evidence_fingerprint": "E3",
                "resolved_threshold": 0.018,
                "resolved_top_fraction": 0.01,
                "calibration_observations": 110,
                "train_start": "2018-07-02",
                "train_end": "2020-07-31",
            },
            {
                "horizon": 3,
                "assessment_date": "2021-08-31",
                "f_refit": True,
                "candidate_id": "RIDGE_A1",
                "model_family": "RIDGE",
                "parameters_json": '{"alpha": 1.0}',
                "model_artifact_id": "M3",
                "generation_id": "G3",
                "prediction_artifact_path": "unused-3",
                "fold_count": 4,
                "evidence_fingerprint": "E4",
                "resolved_threshold": 0.017,
                "resolved_top_fraction": 0.01,
                "calibration_observations": 120,
                "train_start": "2019-01-02",
                "train_end": "2021-01-29",
            },
        ]
    )


def main() -> None:
    market_dates = tuple(
        pd.bdate_range("2020-08-31", "2021-12-31")
    )
    events = _event_rows(
        _audit_fixture(),
        market_dates,
        date(2021, 12, 31),
    )
    assert len(events) == 2
    assert events[0]["event_date"] == pd.Timestamp("2021-02-26")
    assert events[0]["interval_end"] < events[1]["event_date"]
    assert events[0]["old_model_artifact_id"] == "M1"
    assert events[0]["new_model_artifact_id"] == "M2"
    assert events[0]["fold_count_increment"] == 1
    assert events[1]["old_model_artifact_id"] == "M2"
    assert events[1]["new_model_artifact_id"] == "M3"

    prior = {
        "as_of": "2021-02-25",
        "initial_value": 10000.0,
        "cash": 0.0,
        "urth_units": 1.0,
        "positions": {},
        "pending_orders": {},
        "transaction_cost_eur": 10.0,
        "threshold": 0.02,
        "trades": [{"id": 1}],
        "curve": [
            {
                "date": pd.Timestamp("2021-02-25"),
                "strategy_value": 11000.0,
                "urth_value": 10800.0,
            }
        ],
    }
    fingerprint_a = _state_fingerprint(prior)
    fingerprint_b = _state_fingerprint(dict(prior))
    assert fingerprint_a == fingerprint_b

    result = {
        "curve": pd.DataFrame(
            [
                prior["curve"][0],
                {
                    "date": pd.Timestamp("2021-02-26"),
                    "strategy_value": 11100.0,
                    "urth_value": 10900.0,
                },
                {
                    "date": pd.Timestamp("2021-03-01"),
                    "strategy_value": 11200.0,
                    "urth_value": 11000.0,
                },
            ]
        ),
        "trades": [{"id": 1}, {"id": 2}],
        "replay_state": {
            "transaction_cost_eur": 14.0,
        },
    }
    metrics = _segment_metrics(result, prior, initial=10000.0)
    assert metrics["segment_sessions"] == 2
    assert metrics["trade_count"] == 1
    assert abs(metrics["cost_eur"] - 4.0) < 1e-12
    assert metrics["terminal_value"] == 11200.0

    print("DYNAMIC_QBD_FOLD_EVENT_COUNTERFACTUAL_SELF_TEST_PASS")


if __name__ == "__main__":
    main()
