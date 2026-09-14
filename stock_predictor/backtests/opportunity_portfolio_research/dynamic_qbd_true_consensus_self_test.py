"""Focused contract tests for the true-consensus diagnostic."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from .dynamic_qbd_true_consensus import (
    HOLDOUT_START,
    _bootstrap,
    _build_sets,
    _candidate_panel,
    _prepare_events,
    _selection_rows,
)


def _fixture() -> pd.DataFrame:
    rows = []
    for date, ticker, scores in [("2022-01-03", "AAA", {5: .8, 7: .7, 15: .2}),
                                 ("2022-02-03", "BBB", {5: .1, 15: .6})]:
        for h, score in scores.items():
            rows.append({"decision_date": date, "ticker": ticker, "arm": "C",
                         "family_id": f"H{h:02d}_D01_N01_FIXED", "horizon_h": h,
                         "holding_d": 1, "max_names_n": 1, "prediction_score": score,
                         "score_percentile": score, "distance_to_threshold": score - .1,
                         "h_bucket": "SHORT" if h <= 10 else "MID", "forward_excess_return": .1 * score,
                         "target_matured_date": "2022-03-01", "has_valid_target": True})
    return pd.DataFrame(rows)


class TrueConsensusSelfTest(unittest.TestCase):
    def test_dedup_multi_bucket_contract(self):
        events = _prepare_events(_fixture())
        sets, features, buckets = _build_sets(events)
        self.assertEqual(len(sets), 2)
        self.assertEqual(int(sets.loc[sets.ticker.eq("AAA"), "active_family_count_raw"].iloc[0]), 3)
        self.assertEqual(int(sets.loc[sets.ticker.eq("AAA"), "dedup_h_bucket_count"].iloc[0]), 2)
        self.assertEqual(sets.loc[sets.ticker.eq("AAA"), "bucket_combination"].iloc[0], "SHORT_MID")
        self.assertEqual(int(sets.loc[sets.ticker.eq("BBB"), "dedup_h_bucket_count"].iloc[0]), 2)
        candidates = _candidate_panel(events, sets, features, buckets)
        self.assertFalse(candidates.duplicated(["decision_date", "ticker", "arm", "h_bucket"]).any())
        self.assertEqual(len(candidates), 4)

    def test_holdout_and_bucket_validation_fail_closed(self):
        bad = _fixture().copy()
        bad.loc[0, "decision_date"] = str(HOLDOUT_START.date())
        with self.assertRaises(ValueError):
            _prepare_events(bad)
        bad = _fixture().copy()
        bad.loc[0, "h_bucket"] = "LONG"
        with self.assertRaises(ValueError):
            _prepare_events(bad)

    def test_paired_candidates_and_bootstrap_reproducible(self):
        pred = []
        for bucket, score, target in [("SHORT", .8, .2), ("MID", .2, .1)]:
            for arm, value in [("T2_SCORE_ONLY", score), ("T3_TRUE_CONSENSUS", score + .01)]:
                pred.append({"set_key": "2022-01-03|AAA|C", "decision_date": pd.Timestamp("2022-01-03"),
                             "ticker": "AAA", "arm": "C", "h_bucket": bucket, "bucket_realized_excess": target,
                             "dedup_h_bucket_count": 2, "bucket_combination": "SHORT_MID", "set_type": "S2_TWO_BUCKETS",
                             "prediction": value, "feature_arm": arm, "model": "RIDGE", "test_month": pd.Timestamp("2022-01-01")})
        out = _selection_rows(pd.DataFrame(pred))
        self.assertEqual(len(out), 1)
        self.assertEqual(out.t2_selected_candidate.iloc[0], "SHORT")
        self.assertEqual(out.t3_selected_candidate.iloc[0], "SHORT")
        frame = pd.DataFrame({"test_month": pd.to_datetime(["2022-01-01", "2022-02-01", "2022-03-01", "2022-04-01"]), "delta": [1., -1., 2., -2.]})
        self.assertEqual(_bootstrap(frame, "delta", 7, repetitions=50), _bootstrap(frame, "delta", 7, repetitions=50))


if __name__ == "__main__":
    unittest.main()
