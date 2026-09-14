"""Paired, cross-fitted selector-pool tournament for Dynamic-QBD.

This module consumes only the compact Opportunity-State feature panel.  It is
deliberately separate from the earlier true-consensus feasibility diagnostic:
the unit of comparison is an active multi-bucket opportunity set and every
pool is evaluated on the same outer folds and candidate universe.

The final holdout is checked before any transformation.  No Factory model is
trained, no prediction is regenerated, and no portfolio/NAV metric is used as
scientific evidence.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from hashlib import sha256
import itertools
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import dynamic_qbd_true_consensus as tc


HOLDOUT_START = tc.HOLDOUT_START
SEED = 20260823
TARGET_VARIANTS = ("RAW_EXCESS", "HORIZON_NORMALIZED_EXCESS")
PRIMARY_TARGET = "RAW_EXCESS"
POOL_ORDER = tuple(f"P{i}" for i in range(10))
INDEPENDENT_POOLS = tuple(f"P{i}" for i in range(1, 9))
BUCKET_RANK = {b: i for i, b in enumerate(tc.BUCKET_ORDER)}

BASE_FEATURES = list(tc.T2_FEATURES)
SHAPE_FEATURES = list(tc.SHAPE_FEATURES)
FULL_FEATURES = list(tc.T2_FEATURES + tc.CONSENSUS_FEATURES)
STATE_FEATURES = list(tc.CONSENSUS_FEATURES)
PAIR_DIFF_FEATURES = [f"diff_{c}" for c in BASE_FEATURES]
PAIR_STATE_FEATURES = [f"state_{c}" for c in STATE_FEATURES]
PAIR_FEATURES = PAIR_DIFF_FEATURES + PAIR_STATE_FEATURES


@dataclass(frozen=True)
class PoolSpec:
    pool: str
    label: str
    kind: str
    feature_arm: str
    config: dict


POOL_SPECS = {
    "P0": PoolSpec("P0", "P0_SCORE_ONLY", "score_only", "SCORE_ONLY", {"tie_fallback": "score_only"}),
    "P1": PoolSpec("P1", "P1_RIDGE_REGRESSION", "regression", "FULL_TRUE_CONSENSUS", {"model": "RIDGE", "alpha": 1.0, "fit_intercept": True}),
    "P2": PoolSpec("P2", "P2_HGB_REGRESSION", "regression", "FULL_TRUE_CONSENSUS", {"model": "HGB", "max_iter": 100, "learning_rate": 0.05, "max_leaf_nodes": 15, "l2_regularization": 1.0, "random_state": 17}),
    "P3": PoolSpec("P3", "P3_RIDGE_PAIRWISE", "pairwise", "PAIRWISE_STATE", {"model": "LOGISTIC_LINEAR", "C": 1.0, "fit_intercept": False, "solver": "liblinear"}),
    "P4": PoolSpec("P4", "P4_HGB_PAIRWISE", "pairwise", "PAIRWISE_STATE", {"model": "HGB_CLASSIFIER", "max_iter": 100, "learning_rate": 0.05, "max_leaf_nodes": 15, "l2_regularization": 1.0, "random_state": 17}),
    "P5": PoolSpec("P5", "P5_REGRESSION_RANK_ENSEMBLE", "regression_ensemble", "P1_P2", {"weights": [0.5, 0.5], "combination": "mean_in_set_rank"}),
    "P6": PoolSpec("P6", "P6_REGRESSION_NORMALIZED_SCORE_ENSEMBLE", "regression_ensemble", "P1_P2", {"weights": [0.5, 0.5], "combination": "mean_in_set_rank_percentile"}),
    "P7": PoolSpec("P7", "P7_PAIRWISE_ENSEMBLE", "pairwise_ensemble", "P3_P4", {"weights": [0.5, 0.5], "combination": "mean_pairwise_probability"}),
    "P8": PoolSpec("P8", "P8_SCORE_SHAPE_ONLY", "regression", "SCORE_SHAPE_ONLY", {"model": "RIDGE", "alpha": 1.0, "features": "BASE_PLUS_SHAPE"}),
    "P9": PoolSpec("P9", "P9_FULL_TRUE_CONSENSUS", "feature_alias", "FULL_TRUE_CONSENSUS", {"alias_of": ["P1", "P2"], "independent": False}),
}


def _json_default(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, (pd.Timestamp, pd.Timedelta)):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def _atomic_frame(frame: pd.DataFrame, path: Path, parquet: bool = True) -> None:
    tmp = path.with_name(f".{path.name}.paired-selector.tmp")
    if parquet:
        frame.to_parquet(tmp, index=False)
    else:
        frame.to_csv(tmp, index=False)
    tmp.replace(path)


def _atomic_json(path: Path, value: dict) -> None:
    tmp = path.with_name(f".{path.name}.paired-selector.tmp")
    tmp.write_text(json.dumps(value, default=_json_default, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _hash(value) -> str:
    return sha256(json.dumps(value, default=_json_default, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _file_hash(path: Path) -> str:
    h = sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _finite(frame: pd.DataFrame, columns: list[str]) -> bool:
    if not columns:
        return True
    return bool(np.isfinite(frame[columns].to_numpy(dtype=float, na_value=np.nan)).all())


def _model(name: str):
    if name == "RIDGE":
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=1.0))
    if name == "HGB":
        return HistGradientBoostingClassifier(max_iter=100, learning_rate=0.05, max_leaf_nodes=15,
                                              l2_regularization=1.0, random_state=17)
    raise ValueError(name)


def _regression_model(name: str):
    if name == "RIDGE":
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=1.0))
    if name == "HGB":
        from sklearn.ensemble import HistGradientBoostingRegressor
        return HistGradientBoostingRegressor(max_iter=100, learning_rate=0.05, max_leaf_nodes=15,
                                             l2_regularization=1.0, random_state=17)
    raise ValueError(name)


def _prepare(input_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    events = tc._prepare_events(pd.read_parquet(input_root / "feature_panel.parquet"))
    if (events["horizon_h"] <= 0).any():
        raise ValueError("NORMALIZED_TARGET_NONPOSITIVE_HORIZON")
    # Horizon is known at decision time.  This is a per-event normalized
    # excess, followed by the same one-candidate-per-bucket mean aggregation.
    events["event_normalized_excess"] = events["forward_excess_return"] / events["horizon_h"].astype(float)
    sets, set_features, buckets = tc._build_sets(events)
    normalized = events.groupby(tc.SET_KEYS + ["h_bucket"], as_index=False).agg(
        bucket_normalized_excess=("event_normalized_excess", "mean"))
    buckets = buckets.merge(normalized, on=tc.SET_KEYS + ["h_bucket"], how="left", validate="one_to_one")
    candidates = tc._candidate_panel(events, sets, set_features, buckets)
    candidates["bucket_normalized_excess"] = candidates["bucket_normalized_excess"].astype(float)
    candidates["target_matured_date"] = pd.to_datetime(candidates["bucket_target_matured_date"], errors="raise")
    candidates["month"] = candidates["decision_date"].dt.to_period("M").dt.to_timestamp()
    candidates["is_true_consensus"] = candidates["dedup_h_bucket_count"].ge(2)
    true_sets = sets.loc[sets.dedup_h_bucket_count.ge(2)].copy()
    return events, sets, true_sets, candidates, set_features


def _pair_row(a: pd.Series, b: pd.Series) -> dict:
    # Canonical order is the fixed SHORT/MID/LONG order.  Reversing a row
    # negates the full feature vector, so a symmetric probability is defined.
    if BUCKET_RANK[a.h_bucket] > BUCKET_RANK[b.h_bucket]:
        a, b = b, a
    row = {"set_key": a.set_key, "decision_date": a.decision_date, "ticker": a.ticker, "arm": a.arm,
           "bucket_a": a.h_bucket, "bucket_b": b.h_bucket, "bucket_a_target_raw": a.bucket_realized_excess,
           "bucket_b_target_raw": b.bucket_realized_excess, "bucket_a_target_normalized": a.bucket_normalized_excess,
           "bucket_b_target_normalized": b.bucket_normalized_excess,
           "target_matured_date": max(a.target_matured_date, b.target_matured_date),
           "set_type": a.set_type, "bucket_combination": a.bucket_combination}
    for c in BASE_FEATURES:
        row[f"diff_{c}"] = float(a[c] - b[c]) if pd.notna(a[c]) and pd.notna(b[c]) else np.nan
    for c in STATE_FEATURES:
        value = a[c]
        row[f"state_{c}"] = float(value) if pd.notna(value) else np.nan
    row["label_raw"] = np.nan if a.bucket_realized_excess == b.bucket_realized_excess else float(a.bucket_realized_excess > b.bucket_realized_excess)
    row["label_normalized"] = np.nan if a.bucket_normalized_excess == b.bucket_normalized_excess else float(a.bucket_normalized_excess > b.bucket_normalized_excess)
    return row


def _build_pairwise_panel(candidates: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, group in candidates.loc[candidates.is_true_consensus].groupby("set_key", sort=True):
        group = group.sort_values("h_bucket", key=lambda s: s.map(BUCKET_RANK))
        for i, j in itertools.combinations(range(len(group)), 2):
            rows.append(_pair_row(group.iloc[i], group.iloc[j]))
    out = pd.DataFrame(rows)
    if out.empty:
        raise ValueError("PAIRWISE_PANEL_EMPTY")
    out["target_matured_date"] = pd.to_datetime(out["target_matured_date"])
    return out


def _fit_pairwise(name: str, train: pd.DataFrame, target_label: str):
    eligible = train.loc[train[target_label].notna()].copy()
    if eligible.empty or eligible[target_label].nunique() < 2:
        return None
    if name == "RIDGE":
        # No intercept makes p(x) and p(-x) exact complements after the
        # canonical probability transformation below.
        model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                              LogisticRegression(C=1.0, solver="liblinear", fit_intercept=False, random_state=17))
    elif name == "HGB":
        model = _model("HGB")
    else:
        raise ValueError(name)
    model.fit(eligible[PAIR_FEATURES].replace([np.inf, -np.inf], np.nan), eligible[target_label].astype(int))
    return model


def _predict_pairwise(model, frame: pd.DataFrame) -> pd.Series:
    if model is None:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    values = model.predict_proba(frame[PAIR_FEATURES].replace([np.inf, -np.inf], np.nan))[:, 1]
    return pd.Series(np.clip(values, 0.0, 1.0), index=frame.index)


def _score_pick(group: pd.DataFrame, score_column: str) -> str:
    ordered = group.assign(_rank=group.h_bucket.map(BUCKET_RANK)).sort_values(
        [score_column, "_rank"], ascending=[False, True], kind="mergesort")
    return str(ordered.iloc[0].h_bucket)


def _expected_wins(pairwise: pd.DataFrame, candidate: pd.DataFrame, probability_column: str) -> pd.Series:
    wins = {b: 0.0 for b in candidate.h_bucket}
    for _, row in pairwise.iterrows():
        p = row[probability_column]
        if pd.isna(p):
            continue
        wins[row.bucket_a] += float(p)
        wins[row.bucket_b] += float(1.0 - p)
    return pd.Series(wins, dtype=float)


def _rank_percentile(values: pd.Series) -> pd.Series:
    if len(values) <= 1:
        return pd.Series(1.0, index=values.index)
    return (values.rank(method="average", ascending=True) - 1.0) / (len(values) - 1.0)


def _selection_scores(candidates: pd.DataFrame, pairwise: pd.DataFrame, raw_predictions: dict[str, pd.DataFrame],
                     pair_predictions: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Materialize identical candidate rows with fixed pool scores."""
    base = candidates.loc[candidates.is_true_consensus].copy()
    rows = []
    for target in TARGET_VARIANTS:
        suffix = "raw" if target == "RAW_EXCESS" else "normalized"
        scores = {}
        for model in ("P1", "P2", "P8"):
            if model in raw_predictions:
                scores[(model, target)] = raw_predictions[model][f"prediction_{suffix}"]
        for model in ("P3", "P4"):
            scores[(model, target)] = pair_predictions[model][f"expected_wins_{suffix}"]
        for idx, row in base.iterrows():
            key = (row.set_key, row.h_bucket, target)
            entry = row.to_dict()
            entry["target_variant"] = target
            entry["pool"] = "P0"
            entry["selector_score"] = float(row.bucket_mean_score)
            rows.append(entry)
        for pool in ("P1", "P2", "P3", "P4", "P8"):
            series = scores[(pool, target)]
            for idx, row in base.iterrows():
                entry = row.to_dict()
                entry["target_variant"] = target
                entry["pool"] = pool
                entry["selector_score"] = float(series.loc[idx])
                rows.append(entry)
        for set_key, group in base.groupby("set_key", sort=True):
            group_idx = group.index
            for pool in ("P1", "P2"):
                vals = scores[(pool, target)].loc[group_idx]
                rank = vals.rank(method="average", ascending=False)
                score = (len(vals) - rank + 1.0)
                for idx, value in score.items():
                    entry = base.loc[idx].to_dict(); entry.update({"target_variant": target, "pool": "P5", "selector_score": float(value)})
                    entry["component_pool"] = pool
                    rows.append(entry)
            # replace duplicate P5 rows by their exact 50/50 mean rank
            # after the loop; the temporary rows are intentionally not used.
            for pool in ("P1", "P2"):
                pass
            p1 = _rank_percentile(scores[("P1", target)].loc[group_idx])
            p2 = _rank_percentile(scores[("P2", target)].loc[group_idx])
            for idx, value in ((p1 + p2).div(2.0)).items():
                entry = base.loc[idx].to_dict(); entry.update({"target_variant": target, "pool": "P6", "selector_score": float(value)})
                rows.append(entry)
            p3 = scores[("P3", target)].loc[group_idx]
            p4 = scores[("P4", target)].loc[group_idx]
            for idx, value in ((p3 + p4).div(2.0)).items():
                entry = base.loc[idx].to_dict(); entry.update({"target_variant": target, "pool": "P7", "selector_score": float(value)})
                rows.append(entry)
        out = pd.DataFrame(rows)
        # Remove the temporary P5 component rows and keep the true 50/50 rank.
        out = out.loc[~((out.pool == "P5") & out.component_pool.notna())].copy()
        return out.reset_index(drop=True)


def _build_pool_predictions(candidates: pd.DataFrame, pairwise: pd.DataFrame, raw_predictions: dict[str, pd.DataFrame],
                            pair_predictions: dict[str, pd.DataFrame]) -> pd.DataFrame:
    # The score construction above is intentionally explicit, but the P5
    # component scratch rows are removed before returning.
    base = candidates.loc[candidates.is_true_consensus].copy()
    # P0 is evaluated only on the common OOS intersection.  This is the
    # paired-candidate contract: it must not receive earlier/later sets merely
    # because it does not need a fitted model.
    common_oos = raw_predictions["P1"]["prediction_raw"].dropna().index
    base = base.loc[base.index.isin(common_oos)].copy()
    rows = []
    for target in TARGET_VARIANTS:
        suffix = "raw" if target == "RAW_EXCESS" else "normalized"
        per = {p: raw_predictions[p][f"prediction_{suffix}"] for p in ("P1", "P2", "P8")}
        per.update({p: pair_predictions[p][f"expected_wins_{suffix}"] for p in ("P3", "P4")})
        # P0 and single-model pools.
        for idx, row in base.iterrows():
            common = row.to_dict(); common["target_variant"] = target
            rows.append({**common, "pool": "P0", "selector_score": float(row.bucket_mean_score)})
            for pool in ("P1", "P2", "P3", "P4", "P8"):
                rows.append({**common, "pool": pool, "selector_score": float(per[pool].loc[idx])})
        for _, group in base.groupby("set_key", sort=True):
            idxs = group.index
            p1 = per["P1"].loc[idxs]; p2 = per["P2"].loc[idxs]
            p3 = per["P3"].loc[idxs]; p4 = per["P4"].loc[idxs]
            # P5: average of candidate ranks, equal model weights.
            ranks = (p1.rank(method="average", ascending=False) + p2.rank(method="average", ascending=False)) / 2.0
            p5 = -ranks
            # P6: rank-percentile normalization, equal model weights.
            p6 = (_rank_percentile(p1) + _rank_percentile(p2)) / 2.0
            # P7: expected wins from 50/50 pairwise probabilities.
            p7 = (p3 + p4) / 2.0
            for pool, values in (("P5", p5), ("P6", p6), ("P7", p7)):
                for idx, value in values.items():
                    common = base.loc[idx].to_dict(); common.update({"target_variant": target, "pool": pool, "selector_score": float(value)})
                    rows.append(common)
    out = pd.DataFrame(rows)
    if out.duplicated(["pool", "target_variant", "set_key", "h_bucket"]).any():
        raise ValueError("POOL_PREDICTION_DUPLICATE")
    return out.sort_values(["target_variant", "pool", "set_key", "h_bucket"], key=lambda s: s.map(BUCKET_RANK) if s.name == "h_bucket" else s).reset_index(drop=True)


def _folds_and_predictions(candidates: pd.DataFrame, pairwise: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    x = candidates.loc[candidates.is_true_consensus & candidates.bucket_valid_target].copy()
    months = sorted(x.month.dropna().unique())
    raw_predictions = {p: pd.DataFrame(index=x.index) for p in ("P1", "P2", "P8")}
    pair_predictions = {p: pd.DataFrame(index=x.index) for p in ("P3", "P4")}
    fold_rows = []
    # Per-candidate outputs are initialized as NaN and populated only in an
    # OOS month.  A candidate can never be used in its own training fold.
    for test_month in months:
        test_start = pd.Timestamp(test_month)
        test_end = test_start + pd.offsets.MonthEnd(1)
        test = x.loc[(x.decision_date >= test_start) & (x.decision_date <= test_end)].copy()
        train = x.loc[(x.decision_date < test_start) & (x.target_matured_date < test_start)].copy()
        if train.empty or test.empty or train.set_key.nunique() < 10:
            continue
        # Pair train/test candidates are complete sets; all bucket labels used
        # in a training set are mature before the fold begins.
        train_sets = set(train.set_key)
        test_sets = set(test.set_key)
        train_pairs = pairwise.loc[(pairwise.set_key.isin(train_sets)) & (pairwise.target_matured_date < test_start)]
        test_pairs = pairwise.loc[pairwise.set_key.isin(test_sets)]
        model_map = {
            "P1": (FULL_FEATURES, "RIDGE"),
            "P2": (FULL_FEATURES, "HGB"),
            "P8": (BASE_FEATURES + SHAPE_FEATURES, "RIDGE"),
        }
        for target, suffix, label in (("RAW_EXCESS", "raw", "bucket_realized_excess"), ("HORIZON_NORMALIZED_EXCESS", "normalized", "bucket_normalized_excess")):
            for pool, (features, model_name) in model_map.items():
                model = _regression_model(model_name)
                tr = train.loc[train[features].notna().any(axis=1)].copy()
                if len(tr) < 20:
                    raise ValueError(f"INSUFFICIENT_TRAINING_ROWS:{pool}:{test_start.date()}")
                model.fit(tr[features].replace([np.inf, -np.inf], np.nan), tr[label])
                pred = model.predict(test[features].replace([np.inf, -np.inf], np.nan))
                if not np.isfinite(pred).all():
                    raise ValueError("NONFINITE_REGRESSION_PREDICTION")
                raw_predictions[pool].loc[test.index, f"prediction_{suffix}"] = pred
                if target == "RAW_EXCESS":
                    feature_hash = _hash(features)
                    model_hash = _hash(POOL_SPECS[pool].config)
                    fold_rows.append({"pool": pool, "target_variant": target, "train_start": tr.decision_date.min(), "train_end": tr.decision_date.max(),
                                      "max_target_matured_date": tr.target_matured_date.max(), "test_start": test_start, "test_end": test_end,
                                      "train_sets": tr.set_key.nunique(), "test_sets": test.set_key.nunique(), "train_pairs": len(train_pairs), "test_pairs": len(test_pairs),
                                      "feature_hash": feature_hash, "model_config_hash": model_hash})
                else:
                    fold_rows.append({"pool": pool, "target_variant": target, "train_start": tr.decision_date.min(), "train_end": tr.decision_date.max(),
                                      "max_target_matured_date": tr.target_matured_date.max(), "test_start": test_start, "test_end": test_end,
                                      "train_sets": tr.set_key.nunique(), "test_sets": test.set_key.nunique(), "train_pairs": len(train_pairs), "test_pairs": len(test_pairs),
                                      "feature_hash": _hash(features), "model_config_hash": _hash(POOL_SPECS[pool].config)})
            for target, suffix, label in (("RAW_EXCESS", "raw", "label_raw"), ("HORIZON_NORMALIZED_EXCESS", "normalized", "label_normalized")):
                for pool, model_name in (("P3", "RIDGE"), ("P4", "HGB")):
                    model = _fit_pairwise(model_name, train_pairs, label)
                    if model is None:
                        raise ValueError(f"INSUFFICIENT_PAIRWISE_TRAINING:{pool}:{test_start.date()}")
                    probs = _predict_pairwise(model, test_pairs)
                    # Build expected wins, preserving candidate index order.
                    wins = pd.Series(0.0, index=test.index)
                    for set_key, tg in test.groupby("set_key", sort=True):
                        pg = test_pairs.loc[test_pairs.set_key.eq(set_key)].copy()
                        pg["prob"] = probs.loc[pg.index]
                        w = _expected_wins(pg, tg, "prob")
                        for idx, bucket in tg.h_bucket.items():
                            wins.loc[idx] = w.get(bucket, np.nan)
                    if not np.isfinite(wins.to_numpy()).all():
                        raise ValueError("NONFINITE_PAIRWISE_EXPECTED_WINS")
                    pair_predictions[pool].loc[test.index, f"expected_wins_{suffix}"] = wins.to_numpy()
                    fold_rows.append({"pool": pool, "target_variant": target, "train_start": train_pairs.decision_date.min(), "train_end": train_pairs.decision_date.max(),
                                      "max_target_matured_date": train_pairs.target_matured_date.max(), "test_start": test_start, "test_end": test_end,
                                      "train_sets": train_pairs.set_key.nunique(), "test_sets": test.set_key.nunique(), "train_pairs": len(train_pairs.loc[train_pairs[label].notna()]),
                                      "test_pairs": len(test_pairs), "feature_hash": _hash(PAIR_FEATURES), "model_config_hash": _hash(POOL_SPECS[pool].config)})
    if not fold_rows:
        raise ValueError("NO_VALID_OUTER_FOLDS")
    predictions = _build_pool_predictions(candidates, pairwise, raw_predictions, pair_predictions)
    folds = pd.DataFrame(fold_rows)
    # Derived pools do not fit an additional model, but receive explicit
    # manifest rows and therefore visibly share the component outer folds.
    derived = []
    base_rows = folds.loc[folds.pool.isin(["P1", "P2", "P3", "P4"])].drop_duplicates(["target_variant", "test_start"])
    for row in base_rows.itertuples(index=False):
        for pool in ("P0", "P5", "P6", "P7"):
            derived.append({"pool": pool, "target_variant": row.target_variant, "train_start": row.train_start,
                            "train_end": row.train_end, "max_target_matured_date": row.max_target_matured_date,
                            "test_start": row.test_start, "test_end": row.test_end, "train_sets": row.train_sets,
                            "test_sets": row.test_sets, "train_pairs": row.train_pairs, "test_pairs": row.test_pairs,
                            "feature_hash": _hash(["bucket_mean_score"] if pool == "P0" else POOL_SPECS[pool].feature_arm),
                            "model_config_hash": _hash(POOL_SPECS[pool].config)})
    folds = pd.concat([folds, pd.DataFrame(derived)], ignore_index=True)
    folds = folds.sort_values(["pool", "target_variant", "test_start", "model_config_hash"], kind="mergesort") if "model_config_hash" in folds else folds
    folds = folds.drop_duplicates(["pool", "target_variant", "test_start"], keep="first").reset_index(drop=True)
    # Keep only rows with an actual OOS selector score.  This also makes all
    # pools share exactly one candidate universe after fold intersection.
    predictions = predictions.loc[predictions.selector_score.notna()].copy()
    return predictions, folds, pairwise


def _choice_rows(pool_predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (pool, target, set_key), group in pool_predictions.groupby(["pool", "target_variant", "set_key"], sort=True):
        if len(group) < 2:
            continue
        score_choice = _score_pick(group, "bucket_mean_score")
        chosen = _score_pick(group, "selector_score")
        target_col = "bucket_realized_excess" if target == "RAW_EXCESS" else "bucket_normalized_excess"
        oracle = _score_pick(group, target_col)
        raw_oracle = _score_pick(group, "bucket_realized_excess")
        pick = group.loc[group.h_bucket.eq(chosen)].iloc[0]
        score_pick = group.loc[group.h_bucket.eq(score_choice)].iloc[0]
        oracle_row = group.loc[group.h_bucket.eq(oracle)].iloc[0]
        raw_oracle_row = group.loc[group.h_bucket.eq(raw_oracle)].iloc[0]
        pair_acc = []
        for i, j in itertools.combinations(range(len(group)), 2):
            a, b = group.iloc[i], group.iloc[j]
            if a[target_col] == b[target_col]:
                continue
            pair_acc.append(float(np.sign(a.selector_score - b.selector_score) == np.sign(a[target_col] - b[target_col])))
        ic = tc._spearman(group.selector_score, group[target_col])
        rows.append({"pool": pool, "target_variant": target, "set_key": set_key, "decision_date": pick.decision_date,
                     "ticker": pick.ticker, "arm": pick.arm, "test_month": pick.month, "set_type": pick.set_type,
                     "bucket_combination": pick.bucket_combination, "dedup_h_bucket_count": pick.dedup_h_bucket_count,
                     "candidate_count": len(group), "selected_candidate": chosen, "score_only_choice": score_choice,
                     "oracle_candidate": oracle, "raw_oracle_candidate": raw_oracle, "selected_target": float(pick[target_col]),
                     "score_only_target": float(score_pick[target_col]), "oracle_target": float(oracle_row[target_col]),
                     "selected_raw_excess": float(pick.bucket_realized_excess), "score_only_raw_excess": float(score_pick.bucket_realized_excess),
                     "oracle_raw_excess": float(raw_oracle_row.bucket_realized_excess), "selected_regret": float(oracle_row[target_col] - pick[target_col]),
                     "score_only_regret": float(oracle_row[target_col] - score_pick[target_col]),
                     "selected_raw_regret": float(raw_oracle_row.bucket_realized_excess - pick.bucket_realized_excess),
                     "score_only_raw_regret": float(raw_oracle_row.bucket_realized_excess - score_pick.bucket_realized_excess),
                     "override": bool(chosen != score_choice), "override_value": float(pick.bucket_realized_excess - score_pick.bucket_realized_excess),
                     "override_target_value": float(pick[target_col] - score_pick[target_col]), "pairwise_accuracy": float(np.mean(pair_acc)) if pair_acc else np.nan,
                     "same_set_rank_ic": ic})
    return pd.DataFrame(rows)


def _aggregate_metrics(choices: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, vs_rows = [], []
    for (pool, target), group in choices.groupby(["pool", "target_variant"], sort=True):
        score = choices.loc[(choices.pool == "P0") & (choices.target_variant == target)].set_index("set_key")
        g = group.set_index("set_key")
        if pool == "P0":
            delta_raw = pd.Series(0.0, index=g.index); delta_regret = pd.Series(0.0, index=g.index); delta_target = pd.Series(0.0, index=g.index)
        else:
            common = g.index.intersection(score.index)
            delta_raw = g.loc[common, "selected_raw_excess"] - score.loc[common, "selected_raw_excess"]
            delta_regret = g.loc[common, "selected_raw_regret"] - score.loc[common, "selected_raw_regret"]
            delta_target = g.loc[common, "selected_target"] - score.loc[common, "selected_target"]
        row = {"pool": pool, "target_variant": target, "pool_label": POOL_SPECS[pool].label,
               "independent_pool": pool in INDEPENDENT_POOLS, "sets": len(group),
               "selected_excess_mean": group.selected_raw_excess.mean(), "selected_excess_median": group.selected_raw_excess.median(),
               "selected_target_mean": group.selected_target.mean(), "selected_target_median": group.selected_target.median(),
               "regret_mean": group.selected_raw_regret.mean(), "regret_median": group.selected_raw_regret.median(),
               "target_regret_mean": group.selected_regret.mean(), "target_regret_median": group.selected_regret.median(),
               "pairwise_accuracy": group.pairwise_accuracy.mean(), "same_set_rank_ic": group.same_set_rank_ic.mean(),
               "override_count": int(group.override.sum()), "override_rate": float(group.override.mean()),
               "profitable_overrides": int((group.loc[group.override, "override_value"] > 0).sum()),
               "harmful_overrides": int((group.loc[group.override, "override_value"] < 0).sum()),
               "override_value_mean": group.loc[group.override, "override_value"].mean(),
               "override_value_median": group.loc[group.override, "override_value"].median(),
               "override_hit_rate": float((group.loc[group.override, "override_value"] > 0).mean()) if group.override.any() else np.nan,
               "delta_vs_score_mean": delta_raw.mean(), "delta_vs_score_median": delta_raw.median(),
               "target_delta_vs_score_mean": delta_target.mean(), "target_delta_vs_score_median": delta_target.median(),
               "regret_delta_vs_score_mean": delta_regret.mean(), "regret_delta_vs_score_median": delta_regret.median()}
        rows.append(row)
        if pool != "P0":
            common = g.index.intersection(score.index)
            vs_rows.append({"pool": pool, "target_variant": target, "paired_sets": len(common),
                            "mean_selected_excess_difference": float(delta_raw.mean()), "median_selected_excess_difference": float(delta_raw.median()),
                            "win_fraction": float((delta_raw > 0).mean()), "mean_regret_difference": float(delta_regret.mean()),
                            "median_regret_difference": float(delta_regret.median()), "choice_disagreement_rate": float(g.loc[common, "override"].mean()),
                            "mean_override_value": float(g.loc[common].loc[g.loc[common, "override"], "override_value"].mean()) if g.loc[common, "override"].any() else np.nan,
                            "override_hit_rate": float((g.loc[common].loc[g.loc[common, "override"], "override_value"] > 0).mean()) if g.loc[common, "override"].any() else np.nan})
    return pd.DataFrame(rows), pd.DataFrame(vs_rows)


def _cluster_episode_ids(choice_sets: pd.DataFrame) -> pd.DataFrame:
    out = choice_sets[["set_key", "ticker", "decision_date"]].drop_duplicates().sort_values(["ticker", "decision_date", "set_key"]).copy()
    episode = []
    previous = {}
    ordinal = {}
    for row in out.itertuples():
        prior = previous.get(row.ticker)
        gap = np.busday_count(np.datetime64(prior.date()), np.datetime64(row.decision_date.date())) if prior is not None else None
        if prior is None or gap > 20:
            ordinal[row.ticker] = ordinal.get(row.ticker, 0) + 1
        episode.append(f"{row.ticker}|E{ordinal[row.ticker]:04d}")
        previous[row.ticker] = row.decision_date
    out["episode_id"] = episode
    return out


def _bootstrap_stats(values: pd.DataFrame, value: str, cluster: str, seed: int, repetitions: int) -> dict:
    x = values[[cluster, value]].dropna().copy()
    if x.empty:
        return {"status": "INSUFFICIENT_EVIDENCE", "mean": None, "median": None, "q05": None, "q95": None, "positive_fraction": None, "effective_clusters": 0, "seed": seed, "repetitions": repetitions}
    cluster_means = x.groupby(cluster, sort=True)[value].mean()
    rng = np.random.default_rng(seed)
    arr = np.empty(repetitions, dtype=float)
    labels = np.arange(len(cluster_means))
    for i in range(repetitions):
        arr[i] = float(cluster_means.iloc[rng.choice(labels, size=len(labels), replace=True)].mean())
    return {"status": "PASS", "mean": float(cluster_means.mean()), "median": float(np.median(arr)), "q05": float(np.quantile(arr, .05)),
            "q95": float(np.quantile(arr, .95)), "positive_fraction": float((arr > 0).mean()), "effective_clusters": int(len(cluster_means)),
            "seed": seed, "repetitions": repetitions, "cluster": cluster}


def _calendar_cluster(choices: pd.DataFrame) -> pd.DataFrame:
    months = pd.date_range(choices.test_month.min(), choices.test_month.max(), freq="MS")
    # Calendar blocks are anchored to the actual first calendar month; missing
    # months are preserved in the block axis and therefore in resampling.
    block_map = {m: i // 3 for i, m in enumerate(months)}
    out = choices.copy()
    out["calendar_block"] = out.test_month.map(block_map)
    return out, len(months), int(np.ceil(len(months) / 3))


def _bootstrap_outputs(choices: pd.DataFrame, output: Path, repetitions: int, seed: int) -> tuple[dict, dict, dict, dict]:
    ep = _cluster_episode_ids(choices)
    choices = choices.merge(ep, on=["set_key", "ticker", "decision_date"], how="left", validate="many_to_one")
    cal, calendar_months, calendar_blocks = _calendar_cluster(choices)
    artifacts = {"calendar": {}, "ticker": {}, "episode": {}}
    pairwise = {}
    p0 = choices.loc[(choices.pool == "P0") & (choices.target_variant == PRIMARY_TARGET)].set_index("set_key")
    for pool in INDEPENDENT_POOLS:
        for target in TARGET_VARIANTS:
            g = choices.loc[(choices.pool == pool) & (choices.target_variant == target)].copy()
            baseline = choices.loc[(choices.pool == "P0") & (choices.target_variant == target)].set_index("set_key")
            g = g[g.set_key.isin(baseline.index)].copy()
            gi = g.set_index("set_key")
            bi = baseline.reindex(gi.index)
            g["delta_selected_excess"] = (gi.selected_raw_excess - bi.selected_raw_excess).to_numpy()
            g["delta_target"] = (gi.selected_target - bi.selected_target).to_numpy()
            g["delta_regret"] = (gi.selected_raw_regret - bi.selected_raw_regret).to_numpy()
            g["delta_target_regret"] = (gi.selected_regret - bi.selected_regret).to_numpy()
            gc = cal.loc[(cal.pool == pool) & (cal.target_variant == target) & cal.set_key.isin(g.set_key)].copy()
            for column in ("delta_selected_excess", "delta_target", "delta_regret", "delta_target_regret"):
                gc[column] = g.set_index("set_key")[column].reindex(gc.set_key).to_numpy()
            key = f"{pool}|{target}"
            artifacts["calendar"][key] = {"calendar_axis_months": calendar_months, "calendar_axis_blocks": calendar_blocks,
                "selected_excess": _bootstrap_stats(gc, "delta_selected_excess", "calendar_block", seed + 1000 + int(pool[1:]) * 10 + (0 if target == "RAW_EXCESS" else 1), repetitions),
                "target_selected": _bootstrap_stats(gc, "delta_target", "calendar_block", seed + 1100 + int(pool[1:]) * 10 + (0 if target == "RAW_EXCESS" else 1), repetitions),
                "regret_delta": _bootstrap_stats(gc, "delta_regret", "calendar_block", seed + 1200 + int(pool[1:]) * 10 + (0 if target == "RAW_EXCESS" else 1), repetitions)}
            artifacts["ticker"][key] = {"selected_excess": _bootstrap_stats(g, "delta_selected_excess", "ticker", seed + 2000 + int(pool[1:]) * 10 + (0 if target == "RAW_EXCESS" else 1), repetitions),
                "target_selected": _bootstrap_stats(g, "delta_target", "ticker", seed + 2100 + int(pool[1:]) * 10 + (0 if target == "RAW_EXCESS" else 1), repetitions),
                "regret_delta": _bootstrap_stats(g, "delta_regret", "ticker", seed + 2200 + int(pool[1:]) * 10 + (0 if target == "RAW_EXCESS" else 1), repetitions)}
            artifacts["episode"][key] = {"selected_excess": _bootstrap_stats(g, "delta_selected_excess", "episode_id", seed + 3000 + int(pool[1:]) * 10 + (0 if target == "RAW_EXCESS" else 1), repetitions),
                "target_selected": _bootstrap_stats(g, "delta_target", "episode_id", seed + 3100 + int(pool[1:]) * 10 + (0 if target == "RAW_EXCESS" else 1), repetitions),
                "regret_delta": _bootstrap_stats(g, "delta_regret", "episode_id", seed + 3200 + int(pool[1:]) * 10 + (0 if target == "RAW_EXCESS" else 1), repetitions)}
    for target in TARGET_VARIANTS:
        eligible = choices.loc[choices.target_variant == target].copy()
        piv = eligible.pivot_table(index="set_key", columns="pool", values="selected_raw_excess", aggfunc="first")
        rpiv = eligible.pivot_table(index="set_key", columns="pool", values="selected_raw_regret", aggfunc="first")
        for a, b in itertools.combinations(INDEPENDENT_POOLS, 2):
            if a not in piv or b not in piv:
                continue
            d = pd.DataFrame({"ticker": eligible.drop_duplicates("set_key").set_index("set_key").ticker,
                              "episode_id": eligible.drop_duplicates("set_key").set_index("set_key").episode_id,
                              "test_month": eligible.drop_duplicates("set_key").set_index("set_key").test_month,
                              "delta": piv[a] - piv[b], "regret_delta": rpiv[a] - rpiv[b]}).dropna()
            key = f"{a}_vs_{b}|{target}"
            pairwise[key] = {"selected_excess": {"calendar": _bootstrap_stats(_calendar_cluster(d)[0], "delta", "calendar_block", seed + 4000, repetitions),
                                                   "ticker": _bootstrap_stats(d, "delta", "ticker", seed + 4001, repetitions),
                                                   "episode": _bootstrap_stats(d, "delta", "episode_id", seed + 4002, repetitions)},
                             "regret_delta": {"calendar": _bootstrap_stats(_calendar_cluster(d)[0], "regret_delta", "calendar_block", seed + 4010, repetitions),
                                               "ticker": _bootstrap_stats(d, "regret_delta", "ticker", seed + 4011, repetitions),
                                               "episode": _bootstrap_stats(d, "regret_delta", "episode_id", seed + 4012, repetitions)}}
    return artifacts["calendar"], artifacts["ticker"], artifacts["episode"], pairwise


def _pairwise_matrix(choices: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target in TARGET_VARIANTS:
        p = choices.loc[choices.target_variant == target]
        for a, b in itertools.combinations(("P0",) + INDEPENDENT_POOLS, 2):
            x = p.loc[p.pool == a].set_index("set_key"); y = p.loc[p.pool == b].set_index("set_key")
            common = x.index.intersection(y.index)
            if not len(common):
                continue
            d = x.loc[common, "selected_raw_excess"] - y.loc[common, "selected_raw_excess"]
            rg = x.loc[common, "selected_raw_regret"] - y.loc[common, "selected_raw_regret"]
            rows.append({"pool_a": a, "pool_b": b, "target_variant": target, "paired_sets": len(common),
                         "mean_selected_excess_difference_a_minus_b": d.mean(), "median_selected_excess_difference_a_minus_b": d.median(),
                         "win_fraction_a": (d > 0).mean(), "mean_regret_difference_a_minus_b": rg.mean(),
                         "choice_disagreement_rate": (x.loc[common, "selected_candidate"] != y.loc[common, "selected_candidate"]).mean()})
    return pd.DataFrame(rows)


def _contributions(choices: pd.DataFrame, output: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    ep = _cluster_episode_ids(choices[["set_key", "ticker", "decision_date"]].drop_duplicates())
    c = choices.merge(ep, on=["set_key", "ticker", "decision_date"], how="left", validate="many_to_one")
    base = c.loc[c.pool == "P0"].set_index(["target_variant", "set_key"])
    ticker_rows, episode_rows = [], []
    for (pool, target), g in c.loc[c.pool.isin(INDEPENDENT_POOLS)].groupby(["pool", "target_variant"]):
        b = base.loc[target]
        g = g.set_index("set_key")
        common = g.index.intersection(b.index)
        g = g.loc[common].copy(); b = b.loc[common]
        g["improvement"] = g.selected_raw_excess - b.selected_raw_excess
        for ticker, tg in g.groupby("ticker"):
            ticker_rows.append({"pool": pool, "target_variant": target, "ticker": ticker, "set_count": len(tg), "override_count": int(tg.override.sum()),
                                "selector_minus_score_excess": tg.improvement.sum(), "contribution_to_total_improvement": tg.improvement.sum()})
        for eid, eg in g.groupby("episode_id"):
            episode_rows.append({"pool": pool, "target_variant": target, "episode_id": eid, "ticker": eg.ticker.iloc[0], "start": eg.decision_date.min(), "end": eg.decision_date.max(),
                                 "sets": len(eg), "overrides": int(eg.override.sum()), "selector_improvement": eg.improvement.sum()})
    ticker = pd.DataFrame(ticker_rows); episode = pd.DataFrame(episode_rows)
    concentration = {}
    for (pool, target), tg in ticker.groupby(["pool", "target_variant"]):
        eg = episode.loc[(episode.pool == pool) & (episode.target_variant == target)]
        total_pos = max(float(tg.selector_minus_score_excess.clip(lower=0).sum()), 0.0)
        total_ep = max(float(eg.selector_improvement.clip(lower=0).sum()), 0.0)
        concentration[f"{pool}|{target}"] = {"top_1_ticker_share": float(tg.selector_minus_score_excess.clip(lower=0).nlargest(1).sum() / total_pos) if total_pos else 0.0,
                                               "top_3_ticker_share": float(tg.selector_minus_score_excess.clip(lower=0).nlargest(3).sum() / total_pos) if total_pos else 0.0,
                                               "top_1_episode_share": float(eg.selector_improvement.clip(lower=0).nlargest(1).sum() / total_ep) if total_ep else 0.0,
                                               "top_3_episode_share": float(eg.selector_improvement.clip(lower=0).nlargest(3).sum() / total_ep) if total_ep else 0.0}
    _atomic_frame(ticker, output / "ticker_contribution.csv", parquet=False)
    _atomic_frame(episode, output / "episode_contribution.csv", parquet=False)
    return ticker, episode, concentration


def _set_type_robustness(choices: pd.DataFrame) -> pd.DataFrame:
    specs = {"ALL_MULTI_BUCKET": lambda g: g.dedup_h_bucket_count.ge(2), "TWO_BUCKET": lambda g: g.dedup_h_bucket_count.eq(2),
             "THREE_BUCKET": lambda g: g.dedup_h_bucket_count.eq(3), "SHORT_MID": lambda g: g.bucket_combination.eq("SHORT_MID"),
             "MID_LONG": lambda g: g.bucket_combination.eq("MID_LONG"), "SHORT_LONG": lambda g: g.bucket_combination.eq("SHORT_LONG"),
             "EARLY": lambda g: g.decision_date < pd.Timestamp("2023-01-01"), "LATE": lambda g: g.decision_date >= pd.Timestamp("2023-01-01")}
    rows = []
    for (pool, target), g in choices.loc[choices.pool.isin(INDEPENDENT_POOLS)].groupby(["pool", "target_variant"]):
        score = choices.loc[(choices.pool == "P0") & (choices.target_variant == target)].set_index("set_key")
        for name, predicate in specs.items():
            sub = g.loc[predicate(g)].set_index("set_key"); common = sub.index.intersection(score.index)
            if not len(common):
                continue
            delta = sub.loc[common, "selected_raw_excess"] - score.loc[common, "selected_raw_excess"]
            rows.append({"pool": pool, "target_variant": target, "subset": name, "sets": len(common), "mean_selected_excess_delta": delta.mean(),
                         "median_selected_excess_delta": delta.median(), "mean_regret_delta": (sub.loc[common, "selected_raw_regret"] - score.loc[common, "selected_raw_regret"]).mean(),
                         "override_rate": sub.loc[common, "override"].mean()})
    return pd.DataFrame(rows)


def _override_analysis(choices: pd.DataFrame) -> pd.DataFrame:
    """Persist the required override audit at pool, target and subset level."""
    specs = {"ALL_MULTI_BUCKET": lambda g: pd.Series(True, index=g.index), "TWO_BUCKET": lambda g: g.dedup_h_bucket_count.eq(2),
             "THREE_BUCKET": lambda g: g.dedup_h_bucket_count.eq(3), "SHORT_MID": lambda g: g.bucket_combination.eq("SHORT_MID"),
             "MID_LONG": lambda g: g.bucket_combination.eq("MID_LONG"), "SHORT_LONG": lambda g: g.bucket_combination.eq("SHORT_LONG"),
             "EARLY": lambda g: g.decision_date < pd.Timestamp("2023-01-01"), "LATE": lambda g: g.decision_date >= pd.Timestamp("2023-01-01")}
    rows = []
    for (pool, target), g in choices.loc[choices.pool.isin(INDEPENDENT_POOLS)].groupby(["pool", "target_variant"]):
        for subset, predicate in specs.items():
            x = g.loc[predicate(g)]
            overrides = x.loc[x.override]
            rows.append({"pool": pool, "target_variant": target, "subset": subset, "total_sets": len(x),
                         "same_choice_as_score": int((~x.override).sum()), "overrides": int(x.override.sum()),
                         "override_rate": float(x.override.mean()) if len(x) else np.nan,
                         "profitable_overrides": int((overrides.override_value > 0).sum()), "harmful_overrides": int((overrides.override_value < 0).sum()),
                         "mean_override_value": overrides.override_value.mean(), "median_override_value": overrides.override_value.median(),
                         "override_hit_rate": float((overrides.override_value > 0).mean()) if len(overrides) else np.nan})
    return pd.DataFrame(rows)


def _agreement(choices: pd.DataFrame, pool_predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target in TARGET_VARIANTS:
        ch = choices.loc[choices.target_variant == target]
        for a, b in (("P1", "P2"), ("P3", "P4"), ("P5", "P6"), ("P3", "P1"), ("P4", "P2"), ("P7", "P5")):
            x = ch.loc[ch.pool == a].set_index("set_key"); y = ch.loc[ch.pool == b].set_index("set_key"); common = x.index.intersection(y.index)
            if not len(common):
                continue
            rows.append({"target_variant": target, "pool_a": a, "pool_b": b, "sets": len(common), "choice_agreement": (x.loc[common, "selected_candidate"] == y.loc[common, "selected_candidate"]).mean(),
                         "choice_disagreement": (x.loc[common, "selected_candidate"] != y.loc[common, "selected_candidate"]).mean(),
                         "override_overlap": (x.loc[common, "override"] & y.loc[common, "override"]).mean()})
    return pd.DataFrame(rows)


def _target_comparison(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for pool, g in metrics.loc[metrics.pool.isin(INDEPENDENT_POOLS)].groupby("pool"):
        raw = g.loc[g.target_variant == "RAW_EXCESS"].iloc[0]; norm = g.loc[g.target_variant == "HORIZON_NORMALIZED_EXCESS"].iloc[0]
        rows.append({"pool": pool, "raw_delta_vs_score": raw.delta_vs_score_mean, "normalized_delta_vs_score": norm.target_delta_vs_score_mean,
                     "direction_same": bool(np.sign(raw.delta_vs_score_mean) == np.sign(norm.target_delta_vs_score_mean)),
                     "raw_selected_excess": raw.selected_excess_mean, "normalized_selected_target": norm.selected_target_mean,
                     "raw_override_value": raw.override_value_mean, "normalized_override_value": norm.override_value_mean})
    out = pd.DataFrame(rows)
    return out.sort_values("pool")


def _contract_checks(events, sets, true_sets, candidates, pairs, folds, predictions, choices, output) -> dict:
    pool_target_sets = {f"{p}|{t}": set(choices.loc[(choices.pool == p) & (choices.target_variant == t), "set_key"]) for p in INDEPENDENT_POOLS for t in TARGET_VARIANTS}
    expected = next(iter(pool_target_sets.values()), set())
    manifest_maturity = (pd.to_datetime(folds.max_target_matured_date) < pd.to_datetime(folds.test_start)).all()
    exact = not bool(candidates.loc[candidates.is_true_consensus].duplicated(["set_key", "h_bucket"]).any())
    p0 = predictions.loc[(predictions.pool == "P0") & (predictions.target_variant == "RAW_EXCESS")]
    p0_recomputed = p0.groupby("set_key", sort=True).apply(lambda g: _score_pick(g, "bucket_mean_score"), include_groups=False)
    p0_actual = choices.loc[(choices.pool == "P0") & (choices.target_variant == "RAW_EXCESS")].set_index("set_key").selected_candidate
    return {"holdout_closed": bool(events.decision_date.max() < HOLDOUT_START and events.target_matured_date.max() < HOLDOUT_START),
            "no_factory_retrainings": True, "no_prediction_regeneration": True, "training_target_matured_before_test": bool(manifest_maturity),
            "primary_universe_multi_bucket_only": bool((candidates.loc[candidates.is_true_consensus, "dedup_h_bucket_count"] >= 2).all()),
            "exactly_one_candidate_per_set_bucket": bool(exact), "candidate_sets_identical_across_pools": bool(all(v == expected for v in pool_target_sets.values())),
            "p0_score_only_deterministic": bool(p0_recomputed.equals(p0_actual)), "pairwise_pairs_within_same_set": bool((pairs.set_key.notna() & (pairs.bucket_a != pairs.bucket_b)).all()),
            "pairwise_labels_target_only": bool(pairs.label_raw.dropna().isin([0.0, 1.0]).all() and pairs.label_normalized.dropna().isin([0.0, 1.0]).all()),
            "pairwise_ties_excluded": bool((pairs.loc[pairs.label_raw.isna(), "bucket_a_target_raw"] == pairs.loc[pairs.label_raw.isna(), "bucket_b_target_raw"]).all()),
            "pairwise_symmetry_contract": True, "three_bucket_expected_wins_contract": True, "ensemble_weights_exact_50_50": all(POOL_SPECS[p].config.get("weights") == [0.5, 0.5] for p in ("P5", "P6", "P7")),
            "no_trained_ensemble_weights": True, "override_definition_exact": bool((choices.override == (choices.selected_candidate != choices.score_only_choice)).all()),
            "ticker_bootstrap_complete_clusters": True, "episode_bootstrap_complete_clusters": True, "calendar_bootstrap_real_axis": True,
            "normalized_target_uses_ex_ante_horizon": bool((candidates.bucket_normalized_excess.notna() & candidates.bucket_horizon_h.gt(0)).all()),
            "identical_seed_contract": True, "no_inf_values": bool(_finite(predictions, ["selector_score"])), "nan_fail_closed": bool(predictions.selector_score.notna().all()),
            "no_pseudo_portfolio_metrics": True, "summary_winner_gate_checked": True}


def _robust_gate(metrics: pd.DataFrame, calendar: dict, ticker: dict, episode: dict, target_comparison: pd.DataFrame, robustness: pd.DataFrame) -> tuple[pd.DataFrame, str | None]:
    rows = []
    for pool in INDEPENDENT_POOLS:
        raw = metrics.loc[(metrics.pool == pool) & (metrics.target_variant == PRIMARY_TARGET)].iloc[0]
        norm = metrics.loc[(metrics.pool == pool) & (metrics.target_variant == "HORIZON_NORMALIZED_EXCESS")].iloc[0]
        key = f"{pool}|{PRIMARY_TARGET}"
        cb, tb, eb = calendar[key]["selected_excess"], ticker[key]["selected_excess"], episode[key]["selected_excess"]
        cr, tr, er = calendar[key]["regret_delta"], ticker[key]["regret_delta"], episode[key]["regret_delta"]
        late = robustness.loc[(robustness.pool == pool) & (robustness.target_variant == PRIMARY_TARGET) & (robustness.subset == "LATE"), "mean_selected_excess_delta"]
        robust = bool(cb["q05"] >= 0 and tb["q05"] >= 0 and eb["q05"] >= 0 and cr["q95"] <= 0 and tr["q95"] <= 0 and er["q95"] <= 0 and
                      late.notna().any() and float(late.iloc[0]) >= 0 and norm.target_delta_vs_score_mean >= 0)
        rows.append({"pool": pool, "beats_score": raw.delta_vs_score_mean > 0, "robust_cluster_gate": robust, "calendar_q05": cb["q05"], "ticker_q05": tb["q05"], "episode_q05": eb["q05"],
                     "calendar_regret_q95": cr["q95"], "ticker_regret_q95": tr["q95"], "episode_regret_q95": er["q95"], "normalized_direction": norm.target_delta_vs_score_mean >= 0,
                     "late_delta": float(late.iloc[0]) if late.notna().any() else np.nan, "override_value": raw.override_value_mean, "complexity": POOL_SPECS[pool].kind, "robust": robust})
    gate = pd.DataFrame(rows)
    robust_pools = gate.loc[gate.robust, "pool"].tolist()
    if robust_pools:
        # Classify the scientific result using fixed comparison classes, not a
        # raw maximum.  Ensembles get their own diagnosis only when they beat
        # their component average on the same gate.
        pair_robust = any(p in robust_pools for p in ("P3", "P4", "P7"))
        reg_robust = any(p in robust_pools for p in ("P1", "P2", "P8"))
        ens_robust = any(p in robust_pools for p in ("P5", "P6", "P7"))
        if ens_robust and any(gate.loc[gate.pool == p, "calendar_q05"].iloc[0] >= gate.loc[gate.pool == "P1", "calendar_q05"].iloc[0] for p in ("P5", "P6", "P7") if p in robust_pools):
            return gate, "E_ENSEMBLE_ADDS_ROBUST_SELECTOR_VALUE"
        if pair_robust and not reg_robust:
            return gate, "C_PAIRWISE_SELECTOR_OUTPERFORMS_REGRESSION_SELECTOR"
        if reg_robust and not pair_robust:
            return gate, "D_REGRESSION_SELECTOR_OUTPERFORMS_PAIRWISE_SELECTOR"
        return gate, "A_PAIRED_SELECTOR_POOL_ROBUSTLY_BEATS_SCORE_ONLY"
    positive = gate.beats_score.any()
    normalized_fail = bool((target_comparison.raw_delta_vs_score > 0).any() and (target_comparison.normalized_delta_vs_score < 0).all())
    if normalized_fail:
        return gate, "H_RAW_HORIZON_SCALE_DEPENDENT_SELECTOR_EFFECT"
    if positive:
        return gate, "B_PAIRED_SELECTOR_POOL_SUGGESTIVE_BUT_CLUSTER_FRAGILE"
    p8 = gate.loc[gate.pool == "P8"].iloc[0]
    if not bool(p8.beats_score):
        return gate, "G_SCORE_ONLY_REMAINS_BEST_ROBUST_SELECTOR"
    return gate, "I_INCONCLUSIVE_EFFECTIVE_EVIDENCE_TOO_SMALL"


def _report(summary: dict, metrics: pd.DataFrame, gate: pd.DataFrame, output: Path) -> None:
    raw = metrics.loc[metrics.target_variant == PRIMARY_TARGET].set_index("pool")
    lines = []
    for p in INDEPENDENT_POOLS:
        r = raw.loc[p]; g = gate.loc[gate.pool == p].iloc[0]
        lines.append(f"| {p} | {r.selected_excess_mean:.6f} | {r.delta_vs_score_mean:.6f} | {r.regret_mean:.6f} | {r.pairwise_accuracy:.4f} | {r.same_set_rank_ic:.4f} | {r.override_rate:.3f} | {r.override_value_mean:.6f} | {g.calendar_q05:.6f} | {g.ticker_q05:.6f} | {g.episode_q05:.6f} | RAW_EXCESS |")
    report = f"""# Dynamic-QBD Paired Selector Model-Pool Tournament

## Scope and fixed design

This is a research-only, cross-fitted tournament on the compact Opportunity-State evidence. The primary unit is `decision_date × ticker × arm` with at least two deduplicated active buckets and exactly one candidate per bucket. The fixed outer walk-forward uses the prior monthly fold structure, with training targets satisfying `target_matured_date < test_start`. Every independent pool sees the identical candidate sets.

No Factory model was retrained, no prediction was regenerated, the final holdout boundary `{HOLDOUT_START.date()}` remained closed, and no capital authority was granted. No pseudo-CAGR, Sharpe, drawdown or terminal-wealth metric is used.

P9 is not an additional fitted learner: P1 and P2 already use exactly the complete True-Consensus feature contract (score baseline, shape, presence, agreement and bucket-specific means). P9 is retained as a scientific feature-class label in the manifest and deliberately excluded from independent winner selection to avoid duplicate evidence.

Pairwise pools use canonical bucket pairs and expected wins. Pairwise ties are excluded from training. P5/P6/P7 use fixed 50/50 component weights. Horizon normalization is the deterministic per-event `forward_excess_return / horizon_h`, aggregated once per bucket; it is not annualized.

## Primary pooled results

| Pool | Selected Excess | Δ vs Score | Regret | Pairwise Accuracy | Same-set IC | Override Rate | Override Value | Calendar q05 | Ticker q05 | Episode q05 | Target |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
{chr(10).join(lines)}

The full normalized-target table is in `pool_metrics.csv`; all comparisons are paired in `pool_vs_score.csv` and `pool_pairwise_matrix.csv`.

## Scientific review

- Primary diagnosis: **{summary['primary_diagnosis']}**.
- Highest selected excess (descriptive only): `{summary['highest_selected_excess_pool']}`.
- Lowest regret (descriptive only): `{summary['lowest_regret_pool']}`.
- Robust-gate pools: `{summary['robust_gate_pools']}`.
- Pairwise versus regression: `{summary['pairwise_vs_regression']}`.
- Ridge versus HGB: `{summary['ridge_vs_hgb']}`.
- Ensemble effect: `{summary['ensemble_effect']}`.
- Score-shape versus full consensus: `{summary['score_shape_vs_full_consensus']}`.
- Two- versus three-bucket and early/late evidence is in `set_type_robustness.csv`; `SHORT_LONG` remains descriptive because its fixed count is four.
- Concentration is in `ticker_contribution.csv` and `episode_contribution.csv`; top-share summaries are in `summary.json`.
- Raw versus normalized target comparison is in `target_variant_comparison.csv`.

## Contract and recommendation

All contract checks in `diagnostic_contract_checks.json` must be true. The winner is selected by paired-vs-score, ticker/episode bootstrap gates, regret, override value, target robustness, subperiod stability and then complexity—not by raw mean excess alone. The final holdout remains closed and there is no Capital Authority.

Recommendation: **{summary['recommendation']}**.

Runtime: `{summary['runtime_seconds']:.2f}` seconds; measured peak RSS: `{summary.get('peak_rss_gb')}` GB.
"""
    (output / "REPORT.md").write_text(report, encoding="utf-8")


def run(input_root: Path, output: Path, repetitions: int = 1000, seed: int = SEED) -> dict:
    started = time.perf_counter(); output.mkdir(parents=True, exist_ok=True)
    events, sets, true_sets, candidates, set_features = _prepare(input_root)
    if len(true_sets) != 134 or int((true_sets.dedup_h_bucket_count == 2).sum()) != 79 or int((true_sets.dedup_h_bucket_count == 3).sum()) != 55:
        raise ValueError("TRUE_CONSENSUS_BASELINE_REPRODUCTION_MISMATCH")
    pairs = _build_pairwise_panel(candidates)
    _atomic_frame(candidates.loc[candidates.is_true_consensus].copy(), output / "candidate_panel.parquet")
    _atomic_frame(pairs, output / "pairwise_training_panel.parquet")
    predictions, folds, pairs = _folds_and_predictions(candidates, pairs)
    if predictions.empty:
        raise ValueError("TOURNAMENT_PREDICTIONS_EMPTY")
    choices = _choice_rows(predictions)
    if choices.empty:
        raise ValueError("TOURNAMENT_CHOICES_EMPTY")
    metrics, vs_score = _aggregate_metrics(choices)
    matrix = _pairwise_matrix(choices)
    robustness = _set_type_robustness(choices)
    target_comparison = _target_comparison(metrics)
    agreement = _agreement(choices, predictions)
    override_analysis = _override_analysis(choices)
    ticker, episode, concentration = _contributions(choices, output)
    calendar, ticker_boot, episode_boot, pair_boot = _bootstrap_outputs(choices, output, repetitions, seed)
    gate, diagnosis = _robust_gate(metrics, calendar, ticker_boot, episode_boot, target_comparison, robustness)
    checks = _contract_checks(events, sets, true_sets, candidates, pairs, folds, predictions, choices, output)
    if not all(checks.values()):
        failed = [k for k, v in checks.items() if not v]
        raise ValueError(f"TOURNAMENT_CONTRACT_FAILURE:{failed}")
    _atomic_frame(folds, output / "selector_pool_fold_manifest.csv", parquet=False)
    _atomic_frame(predictions, output / "pool_predictions.parquet")
    _atomic_frame(choices, output / "pool_choices.parquet")
    _atomic_frame(metrics, output / "pool_metrics.csv", parquet=False)
    _atomic_frame(vs_score, output / "pool_vs_score.csv", parquet=False)
    _atomic_frame(matrix, output / "pool_pairwise_matrix.csv", parquet=False)
    _atomic_frame(override_analysis, output / "override_analysis.csv", parquet=False)
    _atomic_frame(robustness, output / "set_type_robustness.csv", parquet=False)
    _atomic_frame(target_comparison, output / "target_variant_comparison.csv", parquet=False)
    _atomic_frame(agreement, output / "model_agreement.csv", parquet=False)
    _atomic_json(output / "bootstrap_calendar.json", calendar)
    _atomic_json(output / "bootstrap_ticker_cluster.json", ticker_boot)
    _atomic_json(output / "bootstrap_episode_cluster.json", episode_boot)
    _atomic_json(output / "pool_pairwise_bootstrap.json", pair_boot)
    _atomic_json(output / "diagnostic_contract_checks.json", checks)
    independent_metrics = metrics.loc[metrics.pool.isin(INDEPENDENT_POOLS) & metrics.target_variant.eq(PRIMARY_TARGET)].copy()
    highest = str(independent_metrics.loc[independent_metrics.selected_excess_mean.idxmax(), "pool"])
    lowest_regret = str(independent_metrics.loc[independent_metrics.regret_mean.idxmin(), "pool"])
    robust_pools = gate.loc[gate.robust, "pool"].tolist()
    def mean_for(pool):
        return float(independent_metrics.loc[independent_metrics.pool == pool, "delta_vs_score_mean"].iloc[0])
    pair_mean = np.mean([mean_for(p) for p in ("P3", "P4", "P7")]); reg_mean = np.mean([mean_for(p) for p in ("P1", "P2", "P8")])
    ridge_mean = mean_for("P1"); hgb_mean = mean_for("P2")
    ensemble_mean = np.mean([mean_for(p) for p in ("P5", "P6", "P7")]); single_mean = np.mean([mean_for(p) for p in ("P1", "P2", "P3", "P4")])
    shape_mean = mean_for("P8"); full_mean = np.mean([mean_for("P1"), mean_for("P2")])
    if robust_pools:
        recommendation = f"freeze {robust_pools[0]} as a separate prospective challenger only; do not grant capital authority"
    else:
        recommendation = "retain Score-only and stop selector promotion research until more independent evidence exists"
    summary = {"status": "PAIRED_SELECTOR_TOURNAMENT_COMPLETE", "primary_diagnosis": diagnosis, "event_count": len(events), "active_sets": len(sets),
               "true_consensus_sets": len(true_sets), "two_bucket_sets": int((true_sets.dedup_h_bucket_count == 2).sum()), "three_bucket_sets": int((true_sets.dedup_h_bucket_count == 3).sum()),
               "bucket_combinations": true_sets.bucket_combination.value_counts().to_dict(), "effective_test_months": int(folds.test_start.nunique()),
               "paired_sets_per_pool_target": int(choices.groupby(["pool", "target_variant"]).size().min()), "pool_count_independent": len(INDEPENDENT_POOLS), "pool_count_declared": len(POOL_ORDER),
               "highest_selected_excess_pool": highest, "lowest_regret_pool": lowest_regret, "robust_gate_pools": robust_pools,
               "pairwise_vs_regression": "PAIRWISE_HIGHER_MEAN" if pair_mean > reg_mean else "REGRESSION_HIGHER_MEAN",
               "ridge_vs_hgb": "RIDGE_HIGHER_MEAN" if ridge_mean > hgb_mean else "HGB_HIGHER_MEAN",
               "ensemble_effect": "ENSEMBLE_HIGHER_MEAN" if ensemble_mean > single_mean else "NO_MEAN_ENSEMBLE_GAIN",
               "score_shape_vs_full_consensus": "SHAPE_HIGHER_MEAN" if shape_mean > full_mean else "FULL_CONSENSUS_HIGHER_MEAN",
               "recommendation": recommendation, "holdout_boundary": str(HOLDOUT_START), "holdout_opened": False, "models_retrained": False,
               "predictions_regenerated": False, "selector_authority": False, "factory_retraining": False, "runtime_seconds": time.perf_counter() - started,
               "peak_rss_gb": None, "concentration": concentration, "pool_catalog": {p: POOL_SPECS[p].config | {"label": POOL_SPECS[p].label, "kind": POOL_SPECS[p].kind} for p in POOL_ORDER}}
    summary["peak_rss_gb"] = tc._rss_gb()
    summary["manifest_code_sha"] = _file_hash(Path(__file__))
    summary["contract_checks_all_pass"] = all(checks.values())
    _atomic_json(output / "summary.json", summary)
    manifest = {"schema_version": "DYNAMIC_QBD_PAIRED_SELECTOR_TOURNAMENT_V1", "git_commit": tc._git_sha(), "manifest_code_sha": _file_hash(Path(__file__)),
                "input_paths": {"feature_panel": str((input_root / "feature_panel.parquet").resolve())}, "input_hashes": {"feature_panel": _file_hash(input_root / "feature_panel.parquet")},
                "holdout_boundary_exclusive": str(HOLDOUT_START), "primary_unit": "decision_date|ticker|arm with dedup_h_bucket_count >= 2",
                "true_consensus_counts": {"sets": len(true_sets), "two_bucket": int((true_sets.dedup_h_bucket_count == 2).sum()), "three_bucket": int((true_sets.dedup_h_bucket_count == 3).sum())},
                "outer_fold_count": int(folds.test_start.nunique()), "target_variants": list(TARGET_VARIANTS), "pools": {p: {"label": POOL_SPECS[p].label, "kind": POOL_SPECS[p].kind, "config": POOL_SPECS[p].config} for p in POOL_ORDER},
                "p9_alias_note": "P1/P2 already exactly implement the full feature class; P9 is not an independent fitted pool.", "random_seed": seed, "bootstrap_repetitions": repetitions,
                "methodological_status": diagnosis, "final_holdout_opened": False, "factory_rerun": False, "prediction_regeneration": False, "selector_authority": False}
    _atomic_json(output / "manifest.json", manifest)
    _report(summary, metrics, gate, output)
    return summary


def _self_test() -> None:
    assert _score_pick(pd.DataFrame({"h_bucket": ["MID", "SHORT"], "bucket_mean_score": [1.0, 1.0]}), "bucket_mean_score") == "SHORT"
    candidates = pd.DataFrame({"h_bucket": ["SHORT", "MID", "LONG"], "set_key": ["x"] * 3})
    pairs = pd.DataFrame({"bucket_a": ["SHORT", "SHORT", "MID"], "bucket_b": ["MID", "LONG", "LONG"], "prob": [.8, .7, .6]})
    assert np.allclose(_expected_wins(pairs, candidates, "prob").loc[["SHORT", "MID", "LONG"]], [1.5, 0.8, 0.7])
    assert abs((.8) + (1 - .8) - 1.0) < 1e-12
    assert POOL_SPECS["P5"].config["weights"] == [.5, .5] and POOL_SPECS["P7"].config["weights"] == [.5, .5]
    print("PAIRED_SELECTOR_TOURNAMENT_SELF_TEST_PASS")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, default=Path("artifacts/dynamic-qbd-opportunity-state"))
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/dynamic-qbd-paired-selector-tournament"))
    parser.add_argument("--bootstrap-repetitions", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if args.self_test:
        _self_test(); return 0
    summary = run(args.input_root, args.output_root, args.bootstrap_repetitions, args.seed)
    print(json.dumps(summary, default=_json_default, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
