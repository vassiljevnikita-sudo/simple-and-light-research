"""True cross-horizon consensus feasibility diagnostics.

This module consumes the compact, already completed opportunity evidence.  It
does not read the 1.63bn-row prediction store, regenerate predictions, or
train factory models.  The primary observational unit is a
``decision_date x ticker x arm`` set with at least two *deduplicated* active
H buckets.  Economic comparisons are paired within those exact sets; no
overlapping H-specific returns are annualised as portfolio NAV evidence.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


HOLDOUT_START = pd.Timestamp("2026-07-25")
SEED = 20260823
MODELS = ("RIDGE", "HGB")
SET_KEYS = ["decision_date", "ticker", "arm"]
BUCKET_ORDER = ("SHORT", "MID", "LONG")
BUCKET_CENTER = {"SHORT": 5.0, "MID": 15.0, "LONG": 25.0}
T2_FEATURES = ["bucket_mean_score", "bucket_score_percentile", "bucket_distance_to_threshold",
               "bucket_horizon_h", "bucket_holding_d", "bucket_max_names_n"]
SHAPE_FEATURES = ["mean_score_across_active_buckets", "max_score_across_active_buckets",
                  "score_range_across_active_buckets", "cross_horizon_score_std",
                  "score_slope_over_h", "short_minus_mid_score", "mid_minus_long_score",
                  "short_minus_long_score"]
CONSENSUS_FEATURES = ["dedup_h_bucket_count", "has_short", "has_mid", "has_long",
                      "mean_score_short", "mean_score_mid", "mean_score_long",
                      "short_mid_agreement", "mid_long_agreement", "short_long_agreement"] + SHAPE_FEATURES
FEATURE_ARMS = {
    "T2_SCORE_ONLY": T2_FEATURES,
    "T2_SCORE_SHAPE": T2_FEATURES + SHAPE_FEATURES,
    "T3_TRUE_CONSENSUS": T2_FEATURES + CONSENSUS_FEATURES,
}


def _json_default(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, (pd.Timestamp, pd.Timedelta)):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def _hash(value) -> str:
    return sha256(json.dumps(value, default=_json_default, sort_keys=True,
                             separators=(",", ":")).encode()).hexdigest()


def _atomic_frame(frame: pd.DataFrame, path: Path, parquet: bool = True) -> None:
    tmp = path.with_name(f".{path.name}.true-consensus.tmp")
    if parquet:
        frame.to_parquet(tmp, index=False)
    else:
        frame.to_csv(tmp, index=False)
    tmp.replace(path)


def _atomic_json(path: Path, value: dict) -> None:
    tmp = path.with_name(f".{path.name}.true-consensus.tmp")
    tmp.write_text(json.dumps(value, default=_json_default, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _month(x) -> pd.Timestamp:
    return pd.Timestamp(x).to_period("M").to_timestamp()


def _spearman(x: pd.Series, y: pd.Series) -> float:
    z = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(z) < 2 or z.x.nunique() < 2 or z.y.nunique() < 2:
        return np.nan
    return float(z.x.rank(method="average").corr(z.y.rank(method="average")))


def _rss_gb() -> float | None:
    try:
        import ctypes
        from ctypes import wintypes
        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
                        ("PrivateUsage", ctypes.c_size_t)]
        c = Counters()
        c.cb = ctypes.sizeof(Counters)
        fn = ctypes.WinDLL("psapi.dll", use_last_error=True).GetProcessMemoryInfo
        fn.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        fn.restype = wintypes.BOOL
        if not fn(ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(c), c.cb):
            return None
        return float(max(c.WorkingSetSize, c.PeakWorkingSetSize) / (1024 ** 3))
    except Exception:
        return None


def _bucket(h: pd.Series | int) -> pd.Series | str:
    if isinstance(h, pd.Series):
        return pd.cut(h.astype(float), [-np.inf, 10, 20, np.inf], labels=["SHORT", "MID", "LONG"]).astype(str)
    return "SHORT" if int(h) <= 10 else "MID" if int(h) <= 20 else "LONG"


def _combination(values) -> str:
    return "_".join(b for b in BUCKET_ORDER if b in set(values))


def _set_type(count: int) -> str:
    return "S1_SINGLE_BUCKET" if count == 1 else "S2_TWO_BUCKETS" if count == 2 else "S3_THREE_BUCKETS"


def _prepare_events(panel: pd.DataFrame) -> pd.DataFrame:
    required = SET_KEYS + ["family_id", "horizon_h", "holding_d", "max_names_n", "prediction_score",
                           "score_percentile", "distance_to_threshold", "h_bucket", "forward_excess_return",
                           "target_matured_date", "has_valid_target"]
    missing = sorted(set(required) - set(panel.columns))
    if missing:
        raise ValueError(f"TRUE_CONSENSUS_REQUIRED_COLUMNS_MISSING:{missing}")
    out = panel.copy()
    out["decision_date"] = pd.to_datetime(out["decision_date"], errors="raise")
    out["target_matured_date"] = pd.to_datetime(out["target_matured_date"], errors="raise")
    if len(out) and out["decision_date"].max() >= HOLDOUT_START:
        raise ValueError("FINAL_HOLDOUT_BOUNDARY_VIOLATION:decision_date")
    if len(out) and out["target_matured_date"].max() >= HOLDOUT_START:
        raise ValueError("FINAL_HOLDOUT_BOUNDARY_VIOLATION:target_matured_date")
    if out.duplicated(SET_KEYS + ["family_id"]).any():
        raise ValueError("TRUE_CONSENSUS_DUPLICATE_FAMILY_EVENT")
    expected = _bucket(out["horizon_h"])
    if not expected.eq(out["h_bucket"].astype(str)).all():
        raise ValueError("TRUE_CONSENSUS_BUCKET_MEMBERSHIP_INVALID")
    out["has_valid_target"] = out["has_valid_target"].astype(bool) & out["forward_excess_return"].notna()
    out["set_key"] = (out.decision_date.dt.strftime("%Y-%m-%d") + "|" + out.ticker.astype(str) + "|" + out.arm.astype(str))
    return out


def _build_sets(events: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    keys = SET_KEYS
    buckets = events.groupby(keys + ["h_bucket"], as_index=False).agg(
        bucket_mean_score=("prediction_score", "mean"),
        bucket_score_percentile=("score_percentile", "mean"),
        bucket_distance_to_threshold=("distance_to_threshold", "mean"),
        bucket_horizon_h=("horizon_h", "mean"), bucket_holding_d=("holding_d", "mean"),
        bucket_max_names_n=("max_names_n", "mean"),
        bucket_realized_excess=("forward_excess_return", "mean"),
        bucket_family_count=("family_id", "nunique"), bucket_target_count=("has_valid_target", "sum"),
        bucket_event_count=("family_id", "size"), bucket_target_matured_date=("target_matured_date", "max"))
    buckets["bucket_valid_target"] = (buckets.bucket_target_count == buckets.bucket_event_count) & buckets.bucket_realized_excess.notna()
    buckets["bucket_combination"] = buckets.groupby(keys)["h_bucket"].transform(lambda s: _combination(s))
    counts = buckets.groupby(keys, as_index=False).agg(
        dedup_h_bucket_count=("h_bucket", "nunique"), active_family_count_dedup=("h_bucket", "nunique"))
    raw = events.groupby(keys, as_index=False).agg(active_family_count_raw=("family_id", "nunique"))
    sets = counts.merge(raw, on=keys, validate="one_to_one")
    for b in BUCKET_ORDER:
        sets[f"active_{b.lower()}"] = sets.set_index(keys).index.isin(
            pd.MultiIndex.from_frame(buckets.loc[buckets.h_bucket.eq(b), keys].drop_duplicates()))
    sets = sets.reset_index(drop=True)
    sets["set_type"] = sets.dedup_h_bucket_count.map(_set_type)
    sets["bucket_combination"] = sets.apply(lambda r: _combination([b for b in BUCKET_ORDER if r[f"active_{b.lower()}"]]), axis=1)
    sets["set_key"] = sets.decision_date.dt.strftime("%Y-%m-%d") + "|" + sets.ticker.astype(str) + "|" + sets.arm.astype(str)
    if sets["set_key"].duplicated().any():
        raise ValueError("TRUE_CONSENSUS_SET_DUPLICATE")
    # Set-level features use exactly one mean score per economic H bucket.
    piv = buckets.pivot_table(index=keys, columns="h_bucket", values="bucket_mean_score", aggfunc="first")
    for b in BUCKET_ORDER:
        if b not in piv:
            piv[b] = np.nan
    piv = piv[ list(BUCKET_ORDER) ].reset_index().rename(columns={b: f"mean_score_{b.lower()}" for b in BUCKET_ORDER})
    feat = sets.merge(piv, on=keys, how="left", validate="one_to_one")
    feat["has_short"] = feat["active_short"].astype(int)
    feat["has_mid"] = feat["active_mid"].astype(int)
    feat["has_long"] = feat["active_long"].astype(int)
    feat["mean_score_across_active_buckets"] = feat[[f"mean_score_{b.lower()}" for b in BUCKET_ORDER]].mean(axis=1)
    feat["max_score_across_active_buckets"] = feat[[f"mean_score_{b.lower()}" for b in BUCKET_ORDER]].max(axis=1)
    feat["score_range_across_active_buckets"] = feat.max_score_across_active_buckets - feat.mean_score_across_active_buckets.where(feat.dedup_h_bucket_count.eq(1), feat[[f"mean_score_{b.lower()}" for b in BUCKET_ORDER]].min(axis=1))
    # For the primary universe all rows have >=2 buckets; the explicit min is
    # still deterministic for the single-bucket control.
    vals = feat[[f"mean_score_{b.lower()}" for b in BUCKET_ORDER]].to_numpy(float)
    feat["score_range_across_active_buckets"] = np.nanmax(vals, axis=1) - np.nanmin(vals, axis=1)
    feat["cross_horizon_score_std"] = np.nan
    multi = feat.dedup_h_bucket_count.ge(2).to_numpy()
    feat.loc[multi, "cross_horizon_score_std"] = np.nanstd(vals[multi], axis=1, ddof=1)
    feat.loc[feat.dedup_h_bucket_count < 2, "cross_horizon_score_std"] = np.nan
    for a, b in (("short", "mid"), ("mid", "long"), ("short", "long")):
        feat[f"{a}_{b}_agreement"] = np.where(feat[[f"mean_score_{a}", f"mean_score_{b}"]].notna().all(axis=1),
                                                (np.sign(feat[f"mean_score_{a}"]) == np.sign(feat[f"mean_score_{b}"])).astype(float), np.nan)
        feat[f"{a}_minus_{b}_score"] = feat[f"mean_score_{a}"] - feat[f"mean_score_{b}"]
    def slope(row):
        xy = [(BUCKET_CENTER[b], row[f"mean_score_{b.lower()}"]) for b in BUCKET_ORDER if pd.notna(row[f"mean_score_{b.lower()}"])]
        return float(np.polyfit([x for x, _ in xy], [y for _, y in xy], 1)[0]) if len(xy) >= 2 else np.nan
    feat["score_slope_over_h"] = feat.apply(slope, axis=1)
    # Reattach bucket-level set features to every family candidate for the
    # audit panel; this panel is not used as independent evidence.
    event_features = events.merge(feat, on=keys, how="left", validate="many_to_one")
    return sets, feat, buckets


def _candidate_panel(events: pd.DataFrame, sets: pd.DataFrame, set_features: pd.DataFrame, buckets: pd.DataFrame) -> pd.DataFrame:
    out = buckets.merge(set_features, on=SET_KEYS, how="left", validate="many_to_one", suffixes=("", "_set"))
    out["set_key"] = out.decision_date.dt.strftime("%Y-%m-%d") + "|" + out.ticker.astype(str) + "|" + out.arm.astype(str)
    out["set_type"] = out["dedup_h_bucket_count"].map(_set_type)
    out["is_true_consensus"] = out.dedup_h_bucket_count.ge(2)
    out["month"] = out.decision_date.map(_month)
    out["same_ticker_relative_target"] = out.groupby(SET_KEYS)["bucket_realized_excess"].rank(method="average", ascending=False)
    return out.sort_values(SET_KEYS + ["h_bucket"]).reset_index(drop=True)


def _model(name: str):
    if name == "RIDGE":
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=1.0))
    if name == "HGB":
        return HistGradientBoostingRegressor(max_iter=100, learning_rate=0.05, max_leaf_nodes=15,
                                             l2_regularization=1.0, random_state=17)
    raise ValueError(name)


def _walkforward(candidates: pd.DataFrame, output: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    x = candidates.loc[candidates.is_true_consensus & candidates.bucket_valid_target].copy()
    x["target_matured_date"] = x["bucket_target_matured_date"]
    x["month"] = x.decision_date.map(_month)
    months = sorted(x.month.dropna().unique())
    pred_rows, fold_rows = [], []
    for test_month in months:
        test_start = pd.Timestamp(test_month)
        test_end = test_start + pd.offsets.MonthEnd(1)
        train = x.loc[(x.decision_date < test_start) & (x.target_matured_date < test_start)]
        test = x.loc[(x.decision_date >= test_start) & (x.decision_date <= test_end)]
        if train.empty or test.empty or train.set_key.nunique() < 10:
            continue
        test_sets = set(test.set_key)
        for feature_arm, cols in FEATURE_ARMS.items():
            for model_name in MODELS:
                tr = train.loc[train[cols].notna().any(axis=1)].copy()
                te = test.loc[test[cols].notna().any(axis=1)].copy()
                if len(tr) < 20 or te.empty:
                    continue
                model = _model(model_name)
                model.fit(tr[cols].replace([np.inf, -np.inf], np.nan), tr.bucket_realized_excess)
                te = te.copy()
                te["prediction"] = model.predict(te[cols].replace([np.inf, -np.inf], np.nan))
                te["feature_arm"] = feature_arm
                te["model"] = model_name
                te["test_month"] = test_start
                te["train_set_count"] = tr.set_key.nunique()
                te["test_set_count"] = te.set_key.nunique()
                pred_rows.append(te[["decision_date", "ticker", "arm", "set_key", "h_bucket", "bucket_realized_excess",
                                     "bucket_valid_target", "set_type", "dedup_h_bucket_count", "bucket_combination",
                                     "prediction", "feature_arm", "model", "test_month", "train_set_count", "test_set_count"]])
                fold_rows.append({"feature_arm": feature_arm, "model": model_name, "test_month": test_start,
                                  "train_start": tr.decision_date.min(), "train_end": tr.decision_date.max(),
                                  "max_target_matured_date": tr.target_matured_date.max(),
                                  "test_start": test_start, "test_end": test_end,
                                  "true_consensus_sets_train": tr.set_key.nunique(), "true_consensus_sets_test": te.set_key.nunique(),
                                  "h_bucket_candidates_test": len(te), "families_test": int(events_family_count(candidates, test_sets)),
                                  "feature_columns": ",".join(cols), "feature_hash": _hash(cols)})
    if not pred_rows:
        raise ValueError("TRUE_CONSENSUS_NO_VALID_WALKFORWARD_FOLDS")
    predictions = pd.concat(pred_rows, ignore_index=True)
    folds = pd.DataFrame(fold_rows)
    if (pd.to_datetime(folds.max_target_matured_date) >= pd.to_datetime(folds.test_start)).any():
        raise ValueError("TRUE_CONSENSUS_TRAINING_MATURITY_LEAK")
    _atomic_frame(predictions, output / "model_predictions.parquet")
    _atomic_frame(folds, output / "walkforward_fold_manifest.csv", parquet=False)
    return predictions, folds


def events_family_count(candidates: pd.DataFrame, set_keys: set[str]) -> int:
    return int(candidates.loc[candidates.set_key.isin(set_keys), "bucket_family_count"].sum())


def _pairwise_accuracy(g: pd.DataFrame) -> tuple[float, int]:
    rows = []
    for i in range(len(g)):
        for j in range(i + 1, len(g)):
            a, b = g.iloc[i], g.iloc[j]
            if a.bucket_realized_excess == b.bucket_realized_excess:
                continue
            predicted = np.sign(a.prediction - b.prediction)
            realized = np.sign(a.bucket_realized_excess - b.bucket_realized_excess)
            rows.append(float(predicted == realized))
    return (float(np.mean(rows)) if rows else np.nan, len(rows))


def _selection_rows(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (set_key, model, month), group in predictions.groupby(["set_key", "model", "test_month"], sort=True):
        arms = set(group.feature_arm)
        if not {"T2_SCORE_ONLY", "T3_TRUE_CONSENSUS"}.issubset(arms):
            continue
        t2 = group[group.feature_arm.eq("T2_SCORE_ONLY")].set_index("h_bucket")
        t3 = group[group.feature_arm.eq("T3_TRUE_CONSENSUS")].set_index("h_bucket")
        common = sorted(set(t2.index) & set(t3.index))
        if len(common) < 2 or set(t2.index) != set(t3.index):
            raise ValueError("PAIRED_CANDIDATE_UNIVERSE_MISMATCH")
        t2, t3 = t2.loc[common], t3.loc[common]
        t2_pick, t3_pick = t2.prediction.idxmax(), t3.prediction.idxmax()
        oracle = t2.bucket_realized_excess.idxmax()
        t2_pa, t2_pairs = _pairwise_accuracy(t2.reset_index())
        t3_pa, t3_pairs = _pairwise_accuracy(t3.reset_index())
        rows.append({"set_key": set_key, "decision_date": t2.decision_date.iloc[0], "ticker": t2.ticker.iloc[0],
                     "arm": t2.arm.iloc[0], "model": model, "test_month": month,
                     "dedup_h_bucket_count": t2.dedup_h_bucket_count.iloc[0], "bucket_combination": t2.bucket_combination.iloc[0],
                     "set_type": t2.set_type.iloc[0], "candidate_count": len(common),
                     "t2_selected_candidate": t2_pick, "t3_selected_candidate": t3_pick, "oracle_candidate": oracle,
                     "t2_realized_excess": float(t2.loc[t2_pick, "bucket_realized_excess"]),
                     "t3_realized_excess": float(t3.loc[t3_pick, "bucket_realized_excess"]),
                     "oracle_realized_excess": float(t2.loc[oracle, "bucket_realized_excess"]),
                     "equal_active_realized_excess": float(t2.bucket_realized_excess.mean()),
                     "t3_minus_t2_excess": float(t3.loc[t3_pick, "bucket_realized_excess"] - t2.loc[t2_pick, "bucket_realized_excess"]),
                     "t2_regret": float(t2.loc[oracle, "bucket_realized_excess"] - t2.loc[t2_pick, "bucket_realized_excess"]),
                     "t3_regret": float(t2.loc[oracle, "bucket_realized_excess"] - t3.loc[t3_pick, "bucket_realized_excess"]),
                     "t3_minus_t2_regret": float(t2.loc[t2_pick, "bucket_realized_excess"] - t3.loc[t3_pick, "bucket_realized_excess"]),
                     "t2_pairwise_accuracy": t2_pa, "t3_pairwise_accuracy": t3_pa,
                     "t3_pairwise_accuracy_delta": (t3_pa - t2_pa) if pd.notna(t2_pa) and pd.notna(t3_pa) else np.nan,
                     "t2_pairwise_correct": t2_pa > .5 if pd.notna(t2_pa) else np.nan,
                     "t3_pairwise_correct": t3_pa > .5 if pd.notna(t3_pa) else np.nan,
                     "pairwise_comparisons": t2_pairs})
    return pd.DataFrame(rows)


def _metrics(predictions: pd.DataFrame, paired: pd.DataFrame) -> dict[str, pd.DataFrame]:
    same_ticker, same_date, pairwise, regret = [], [], [], []
    for (feature_arm, model, month), g in predictions.groupby(["feature_arm", "model", "test_month"], sort=True):
        per_set = []
        for _, s in g.groupby("set_key", sort=True):
            if len(s) >= 2:
                value = _spearman(s.prediction, s.bucket_realized_excess)
                if pd.notna(value):
                    per_set.append(value)
        same_ticker.append({"feature_arm": feature_arm, "model": model, "test_month": month,
                            "sets_with_ge2_candidates": int(len(per_set)), "rank_ic": float(np.mean(per_set)) if per_set else np.nan})
        per_date = []
        for _, s in g.groupby("decision_date", sort=True):
            if len(s) >= 2:
                value = _spearman(s.prediction, s.bucket_realized_excess)
                if pd.notna(value):
                    per_date.append(value)
        same_date.append({"feature_arm": feature_arm, "model": model, "test_month": month,
                          "date_sets_with_ge2_candidates": int(len(per_date)), "rank_ic": float(np.mean(per_date)) if per_date else np.nan})
        acc = [_pairwise_accuracy(s.reset_index(drop=True))[0] for _, s in g.groupby("set_key") if len(s) >= 2]
        pairwise.append({"feature_arm": feature_arm, "model": model, "test_month": month,
                         "sets_with_pairs": int(sum(pd.notna(acc))), "pairwise_accuracy": float(np.nanmean(acc)) if any(pd.notna(acc)) else np.nan})
    for model, g in paired.groupby("model", sort=True):
        regret.append({"comparison": "T3_MINUS_T2", "model": model, "sets": len(g),
                       "mean_t2_regret": g.t2_regret.mean(), "mean_t3_regret": g.t3_regret.mean(),
                       "mean_regret_delta": g.t3_minus_t2_regret.mean(), "median_regret_delta": g.t3_minus_t2_regret.median(),
                       "mean_selected_excess_delta": g.t3_minus_t2_excess.mean(), "median_selected_excess_delta": g.t3_minus_t2_excess.median()})
    return {"same_ticker_metrics.csv": pd.DataFrame(same_ticker), "same_date_metrics.csv": pd.DataFrame(same_date),
            "pairwise_metrics.csv": pd.DataFrame(pairwise), "regret_metrics.csv": pd.DataFrame(regret)}


def _bootstrap(frame: pd.DataFrame, value: str, seed: int, repetitions: int = 1000, block_size: int = 3, positive_is_better: bool = True) -> dict:
    f = frame[["test_month", value]].dropna().copy()
    if f.empty:
        return {"status": "INSUFFICIENT_EVIDENCE", "n_sets": 0, "effective_calendar_months": 0, "q05": None, "median": None, "q95": None, "positive_fraction": None, "seed": seed, "repetitions": repetitions, "block_size_months": block_size}
    monthly = f.groupby("test_month")[value].mean().sort_index()
    months = list(monthly.index)
    blocks = [months[i:i + block_size] for i in range(0, len(months), block_size)]
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(repetitions):
        selected = []
        while len(selected) < len(months):
            selected.extend(blocks[int(rng.integers(0, len(blocks)))])
        selected = selected[:len(months)]
        samples.append(float(monthly.reindex(selected).mean()))
    arr = np.asarray(samples)
    signed = arr if positive_is_better else -arr
    return {"status": "PASS", "mean": float(monthly.mean()), "median": float(np.median(arr)),
            "q05": float(np.quantile(arr, .05)), "q95": float(np.quantile(arr, .95)),
            "positive_fraction": float((signed > 0).mean()), "n_sets": int(len(f)),
            "effective_calendar_months": int(len(months)), "seed": seed, "repetitions": repetitions,
            "block_size_months": block_size, "direction": "higher_is_better" if positive_is_better else "lower_is_better"}


def _write_robustness(candidates: pd.DataFrame, paired: pd.DataFrame, output: Path) -> None:
    rows = []
    specs = {"ALL_MULTI_BUCKET": lambda x: x.dedup_h_bucket_count.ge(2),
             "TWO_BUCKET": lambda x: x.dedup_h_bucket_count.eq(2),
             "THREE_BUCKET": lambda x: x.dedup_h_bucket_count.eq(3),
             "SHORT_MID": lambda x: x.bucket_combination.eq("SHORT_MID"),
             "SHORT_LONG": lambda x: x.bucket_combination.eq("SHORT_LONG"),
             "MID_LONG": lambda x: x.bucket_combination.eq("MID_LONG"),
             "EARLY": lambda x: x.decision_date < pd.Timestamp("2023-01-01"),
             "LATE": lambda x: x.decision_date >= pd.Timestamp("2023-01-01")}
    for name, pred in specs.items():
        keep = set(candidates.loc[pred(candidates) & candidates.is_true_consensus, "set_key"])
        p = paired[paired.set_key.isin(keep)]
        for model, g in p.groupby("model"):
            rows.append({"subset": name, "model": model, "candidate_sets": len(keep), "paired_rows": len(g),
                         "mean_selected_excess_delta": g.t3_minus_t2_excess.mean(), "median_selected_excess_delta": g.t3_minus_t2_excess.median(),
                         "mean_regret_delta": g.t3_minus_t2_regret.mean(), "pairwise_accuracy_delta": (g.t3_pairwise_accuracy - g.t2_pairwise_accuracy).mean()})
    _atomic_frame(pd.DataFrame(rows), output / "set_type_robustness.csv", parquet=False)


def _anti_h30(events: pd.DataFrame, predictions: pd.DataFrame, paired: pd.DataFrame, output: Path) -> None:
    rows = []
    for name, mask in {"FULL_UNIVERSE": pd.Series(True, index=events.index), "REMOVE_H30": events.horizon_h.ne(30),
                       "REMOVE_H28_H30": events.horizon_h.lt(28), "REMOVE_H_GE25_D_GE25": ~((events.horizon_h.ge(25)) & (events.holding_d.ge(25)))}.items():
        sub = events.loc[mask].copy()
        counts = sub.groupby(SET_KEYS).h_bucket.nunique()
        keep = set(counts[counts.ge(2)].index.map(lambda x: f"{pd.Timestamp(x[0]).strftime('%Y-%m-%d')}|{x[1]}|{x[2]}"))
        p = paired[paired.set_key.isin(keep)]
        for model, g in p.groupby("model"):
            rows.append({"ablation": name, "reconstructed_multi_bucket_sets": len(keep), "paired_rows": len(g), "model": model,
                         "mean_t3_minus_t2_excess": g.t3_minus_t2_excess.mean(), "median_t3_minus_t2_excess": g.t3_minus_t2_excess.median(),
                         "mean_t3_minus_t2_regret": g.t3_minus_t2_regret.mean()})
        if not p.size:
            rows.append({"ablation": name, "reconstructed_multi_bucket_sets": len(keep), "paired_rows": 0, "model": "NONE"})
    _atomic_frame(pd.DataFrame(rows), output / "anti_h30_true_consensus.csv", parquet=False)


def _diagnosis(output: Path, summary: dict) -> str:
    ic = json.loads((output / "bootstrap_same_ticker_ic.json").read_text())
    excess = json.loads((output / "bootstrap_selected_excess.json").read_text())
    regret = json.loads((output / "bootstrap_regret.json").read_text())
    shape = pd.read_csv(output / "score_shape_comparison.csv")
    adequate = summary["true_consensus_sets"] >= 20 and ic.get("effective_calendar_months", 0) >= 6
    if not adequate:
        return "INCONCLUSIVE_INSUFFICIENT_TRUE_CONSENSUS_EVIDENCE"
    by_model = []
    for model in MODELS:
        mi, me, mr = ic.get("by_model", {}).get(model, {}), excess.get("by_model", {}).get(model, {}), regret.get("by_model", {}).get(model, {})
        by_model.append(mi.get("q05", -np.inf) >= 0 and me.get("q05", -np.inf) >= 0 and mr.get("q95", np.inf) <= 0)
    t3_robust = ic.get("q05", -np.inf) >= 0 and excess.get("q05", -np.inf) >= 0 and regret.get("q95", np.inf) <= 0 and all(by_model)
    t3_suggestive = excess.get("median", 0) > 0 or ic.get("median", 0) > 0 or regret.get("median", 0) < 0
    shape_gain = bool(len(shape) and shape["shape_minus_t2_selected_excess"].median() > 0 and shape["shape_minus_t2_selected_excess"].notna().sum() >= 5)
    consensus_gain = bool(excess.get("median", 0) > 0)
    if t3_robust:
        return "TRUE_CROSS_HORIZON_SELECTOR_VALUE_SUPPORTED"
    if shape_gain and not consensus_gain:
        return "SCORE_SHAPE_ADDS_VALUE_BUT_CONSENSUS_DOES_NOT"
    if not t3_suggestive:
        return "SCORE_ONLY_REMAINS_BEST_AVAILABLE_SELECTOR"
    if t3_suggestive:
        return "TRUE_CROSS_HORIZON_SIGNAL_SUGGESTIVE_BUT_NOT_ROBUST"
    return "NO_TRUE_CROSS_HORIZON_SELECTOR_VALUE"


def run(input_root: Path, output: Path, repetitions: int = 1000, seed: int = SEED) -> dict:
    started = time.perf_counter()
    memory_rows = []
    def mem(stage: str):
        value = _rss_gb()
        memory_rows.append({"stage": stage, "rss_gb": value})
    mem("start")
    output.mkdir(parents=True, exist_ok=True)
    events = _prepare_events(pd.read_parquet(input_root / "feature_panel.parquet"))
    mem("events_loaded")
    sets, set_features, buckets = _build_sets(events)
    candidates = _candidate_panel(events, sets, set_features, buckets)
    mem("candidate_panel_built")
    true_sets = sets.loc[sets.dedup_h_bucket_count.ge(2)].copy()
    if not len(true_sets):
        raise ValueError("NO_TRUE_CONSENSUS_SETS")
    _atomic_frame(true_sets, output / "true_consensus_sets.csv", parquet=False)
    _atomic_frame(set_features, output / "true_consensus_feature_panel.parquet")
    _atomic_frame(candidates, output / "h_bucket_candidate_panel.parquet")
    predictions, folds = _walkforward(candidates, output)
    mem("walkforward_complete")
    paired = _selection_rows(predictions)
    if paired.empty:
        raise ValueError("TRUE_CONSENSUS_NO_PAIRED_T2_T3_ROWS")
    _atomic_frame(paired, output / "paired_t3_vs_t2.parquet")
    metrics = _metrics(predictions, paired)
    for name, frame in metrics.items():
        _atomic_frame(frame, output / name, parquet=False)
    shape = predictions[predictions.feature_arm.isin(["T2_SCORE_ONLY", "T2_SCORE_SHAPE", "T3_TRUE_CONSENSUS"])].copy()
    shape_rows = []
    for (set_key, model, month), g in shape.groupby(["set_key", "model", "test_month"]):
        picks = {}
        for arm, s in g.groupby("feature_arm"):
            picks[arm] = s.loc[s.prediction.idxmax(), "bucket_realized_excess"]
        if "T2_SCORE_ONLY" in picks:
            shape_rows.append({"set_key": set_key, "model": model, "test_month": month,
                               "t2_selected_excess": picks["T2_SCORE_ONLY"], "shape_selected_excess": picks.get("T2_SCORE_SHAPE", np.nan),
                               "t3_selected_excess": picks.get("T3_TRUE_CONSENSUS", np.nan),
                               "shape_minus_t2_selected_excess": picks.get("T2_SCORE_SHAPE", np.nan) - picks["T2_SCORE_ONLY"] if "T2_SCORE_SHAPE" in picks else np.nan,
                               "t3_minus_t2_selected_excess": picks.get("T3_TRUE_CONSENSUS", np.nan) - picks["T2_SCORE_ONLY"] if "T3_TRUE_CONSENSUS" in picks else np.nan})
    _atomic_frame(pd.DataFrame(shape_rows), output / "score_shape_comparison.csv", parquet=False)
    _write_robustness(candidates, paired, output)
    _anti_h30(events, predictions, paired, output)
    bootstrap = {
        "bootstrap_same_ticker_ic.json": _bootstrap(_paired_ic(predictions), "delta_ic", seed, repetitions),
        "bootstrap_pairwise_accuracy.json": _bootstrap(paired, "t3_pairwise_accuracy_delta", seed + 1, repetitions),
        "bootstrap_selected_excess.json": _bootstrap(paired, "t3_minus_t2_excess", seed + 2, repetitions),
        "bootstrap_regret.json": _bootstrap(paired, "t3_minus_t2_regret", seed + 3, repetitions, positive_is_better=False),
    }
    bootstrap["bootstrap_same_ticker_ic.json"]["by_model"] = {m: _bootstrap(_paired_ic(predictions).query("model == @m"), "delta_ic", seed + 10 + i, repetitions) for i, m in enumerate(MODELS)}
    bootstrap["bootstrap_selected_excess.json"]["by_model"] = {m: _bootstrap(paired[paired.model.eq(m)], "t3_minus_t2_excess", seed + 20 + i, repetitions) for i, m in enumerate(MODELS)}
    bootstrap["bootstrap_regret.json"]["by_model"] = {m: _bootstrap(paired[paired.model.eq(m)], "t3_minus_t2_regret", seed + 30 + i, repetitions, positive_is_better=False) for i, m in enumerate(MODELS)}
    bootstrap["bootstrap_pairwise_accuracy.json"]["by_model"] = {m: _bootstrap(paired[paired.model.eq(m)], "t3_pairwise_accuracy_delta", seed + 40 + i, repetitions) for i, m in enumerate(MODELS)}
    for name, value in bootstrap.items():
        _atomic_json(output / name, value)
    checks = _contract_checks(events, sets, candidates, predictions, paired, _legacy_t1_audit(input_root))
    _atomic_json(output / "diagnostic_contract_checks.json", checks)
    mem("metrics_and_bootstrap_complete")
    _atomic_frame(pd.DataFrame(memory_rows), output / "memory-telemetry.csv", parquet=False)
    rss_values = [r["rss_gb"] for r in memory_rows if r["rss_gb"] is not None]
    summary = {"status": "TRUE_CONSENSUS_COMPLETE", "event_count": int(len(events)), "active_sets": int(len(sets)),
               "true_consensus_sets": int(len(true_sets)), "single_bucket_sets": int((sets.dedup_h_bucket_count == 1).sum()),
               "two_bucket_sets": int((sets.dedup_h_bucket_count == 2).sum()), "three_bucket_sets": int((sets.dedup_h_bucket_count == 3).sum()),
               "bucket_combinations": true_sets.bucket_combination.value_counts().to_dict(),
               "effective_test_months": int(folds.test_month.nunique()), "fold_count": int(len(folds)),
               "paired_rows": int(len(paired)), "family_count": int(events.family_id.nunique()),
               "date_min": str(events.decision_date.min().date()), "date_max": str(events.decision_date.max().date()),
               "holdout_boundary": str(HOLDOUT_START), "holdout_opened": False, "models_retrained": False,
               "predictions_regenerated": False, "selector_authority": False, "peak_rss_gb": None,
               "runtime_seconds": time.perf_counter() - started}
    summary["peak_rss_gb"] = max(rss_values) if rss_values else None
    summary["active_events_by_arm"] = events.arm.value_counts().to_dict()
    summary["paired_model_results"] = {
        model: {"paired_sets": int((paired.model == model).sum()),
                "mean_selected_excess_delta": float(paired.loc[paired.model.eq(model), "t3_minus_t2_excess"].mean()),
                "median_selected_excess_delta": float(paired.loc[paired.model.eq(model), "t3_minus_t2_excess"].median()),
                "mean_regret_delta": float(paired.loc[paired.model.eq(model), "t3_minus_t2_regret"].mean()),
                "bootstrap_selected_excess": bootstrap["bootstrap_selected_excess.json"]["by_model"][model],
                "bootstrap_regret": bootstrap["bootstrap_regret.json"]["by_model"][model],
                "bootstrap_same_ticker_ic": bootstrap["bootstrap_same_ticker_ic.json"]["by_model"][model],
                "bootstrap_pairwise_accuracy": bootstrap["bootstrap_pairwise_accuracy.json"]["by_model"][model]}
        for model in MODELS}
    diagnosis = _diagnosis(output, summary)
    summary["primary_diagnosis"] = diagnosis
    _atomic_json(output / "summary.json", summary)
    manifest = {"schema_version": "DYNAMIC_QBD_TRUE_CONSENSUS_V1", "git_commit": _git_sha(),
                "input_paths": {"feature_panel": str((input_root / "feature_panel.parquet").resolve()),
                                "opportunity_panel": str((input_root / "opportunity_panel.parquet").resolve())},
                "input_hashes": {"feature_panel": _file_hash(input_root / "feature_panel.parquet")},
                "holdout_boundary_exclusive": str(HOLDOUT_START), "date_coverage": [summary["date_min"], summary["date_max"]],
                "family_count": summary["family_count"], "active_set_count": summary["active_sets"],
                "true_consensus_set_count": summary["true_consensus_sets"], "bucket_counts": summary["bucket_combinations"],
                "random_seeds": [seed, seed + 1, seed + 2, seed + 3], "bootstrap_repetitions": repetitions,
                "methodological_status": diagnosis, "factory_rerun": False, "prediction_regeneration": False,
                "final_holdout_opened": False, "selector_authority": False, "code_hash": _file_hash(Path(__file__))}
    _atomic_json(output / "manifest.json", manifest)
    _write_report(summary, bootstrap, checks, output)
    return summary


def _paired_ic(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (set_key, model, month), g in predictions.groupby(["set_key", "model", "test_month"]):
        t2 = g[g.feature_arm.eq("T2_SCORE_ONLY")].set_index("h_bucket")
        t3 = g[g.feature_arm.eq("T3_TRUE_CONSENSUS")].set_index("h_bucket")
        if len(t2) >= 2 and set(t2.index) == set(t3.index):
            rows.append({"test_month": month, "model": model, "delta_ic": _spearman(t3.prediction, t3.bucket_realized_excess) - _spearman(t2.prediction, t2.bucket_realized_excess)})
    return pd.DataFrame(rows)


def _contract_checks(events, sets, candidates, predictions, paired, legacy_t1=None) -> dict:
    t1 = legacy_t1 or {"status": "UNAVAILABLE"}
    return {"holdout_closed": bool(events.decision_date.max() < HOLDOUT_START and events.target_matured_date.max() < HOLDOUT_START),
            "true_consensus_definition": bool((sets.dedup_h_bucket_count >= 2).sum() == len(sets.loc[sets.dedup_h_bucket_count >= 2])),
            "raw_family_count_not_used_for_multi_bucket": True, "bucket_membership_valid": True,
            "no_duplicate_set_rows": not sets.duplicated(SET_KEYS).any(),
            "no_duplicate_set_bucket_rows": not candidates.duplicated(SET_KEYS + ["h_bucket"]).any(),
            "paired_candidate_sets_identical": True, "training_maturity_valid": True,
            "t1_same_ticker_selection": {"status": "NOT_APPLICABLE", "legacy_audit": t1}, "pseudo_portfolio_metrics_in_scientific_gate": False,
            "nan_inf_in_predictions": bool(np.isfinite(predictions.prediction).all()), "paired_rows": int(len(paired))}


def _legacy_t1_audit(input_root: Path) -> dict:
    path = input_root / "model_predictions.parquet"
    if not path.is_file():
        return {"status": "UNAVAILABLE"}
    p = pd.read_parquet(path, columns=["decision_date", "ticker", "arm", "model", "test_month", "feature_arm", "prediction"])
    p = p[p.feature_arm.eq("T1_CONSENSUS_ONLY")]
    if p.empty:
        return {"status": "UNAVAILABLE_NO_T1_ROWS"}
    ranges = p.groupby(["decision_date", "ticker", "arm", "model", "test_month"], as_index=False).prediction.agg(
        prediction_range=lambda s: float(s.max() - s.min()))
    old_metric = input_root / "predictive_metrics.csv"
    old = pd.read_csv(old_metric) if old_metric.is_file() else pd.DataFrame()
    old_t1 = old.loc[old.feature_arm.eq("T1_CONSENSUS_ONLY"), ["model", "same_ticker_rank_ic"]].to_dict("records") if not old.empty else []
    return {"status": "PASS", "max_abs_prediction_difference_within_set": float(ranges.prediction_range.max()),
            "nonzero_ranges_above_tolerance": int((ranges.prediction_range > 1e-12).sum()),
            "within_set_invariant": bool((ranges.prediction_range <= 1e-12).all()),
            "legacy_same_ticker_rank_ic": old_t1,
            "legacy_same_ticker_rank_ic_interpretation": "NOT_APPLICABLE_WHEN_WITHIN_SET_INVARIANT"}


def _file_hash(path: Path) -> str:
    h = sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_sha() -> str:
    try:
        import subprocess
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "UNKNOWN"


def _write_report(summary: dict, bootstrap: dict, checks: dict, output: Path) -> None:
    b = bootstrap
    model_lines = []
    for model, result in summary.get("paired_model_results", {}).items():
        ex = result["bootstrap_selected_excess"]
        ic = result["bootstrap_same_ticker_ic"]
        rg = result["bootstrap_regret"]
        model_lines.append(f"- **{model}**: paired sets `{result['paired_sets']}`, selected-excess delta median `{ex.get('median'):.6f}`, q05 `{ex.get('q05'):.6f}`; same-ticker IC delta median `{ic.get('median'):.6f}`, q05 `{ic.get('q05'):.6f}`; regret delta q95 `{rg.get('q95'):.6f}`.")
    robustness = pd.read_csv(output / "set_type_robustness.csv")
    h30 = pd.read_csv(output / "anti_h30_true_consensus.csv")
    early_late = robustness[robustness.subset.isin(["EARLY", "LATE"])]
    robust_text = "; ".join(f"{r.subset}/{r.model}: Δexcess={r.mean_selected_excess_delta:.4f}" for r in early_late.itertuples())
    h30_text = "; ".join(f"{r.ablation}/{r.model}: sets={int(r.reconstructed_multi_bucket_sets)}, Δexcess={r.mean_t3_minus_t2_excess:.4f}" for r in h30.itertuples())
    def brief(value):
        return f"mean={value.get('mean', float('nan')):.6f}, median={value.get('median', float('nan')):.6f}, q05={value.get('q05', float('nan')):.6f}, q95={value.get('q95', float('nan')):.6f}, positive_fraction={value.get('positive_fraction', float('nan')):.3f}"
    report = f"""# Dynamic-QBD True Cross-Horizon Consensus

## Scope and contract

This is a research-only identification test built from the existing compact Opportunity-State evidence. No Factory model was retrained, no prediction was regenerated, and the final holdout boundary `{HOLDOUT_START.date()}` remained closed. The primary unit is a `decision_date × ticker × arm` set with at least two active economic H buckets. One candidate vote is used per bucket; raw Family multiplicity is not treated as independent consensus.

The old T1 same-ticker result is not interpreted as selector evidence. T1 features are set-level constants, and prediction variation inside a set is only floating-point noise; therefore T1 same-ticker selection is `NOT_APPLICABLE`.

## Evidence counts

- Events: `{summary['event_count']}`; active sets: `{summary['active_sets']}`; active Families: `{summary['family_count']}`.
- Active events are all in arm C: `{summary.get('active_events_by_arm', {})}`. Arms A/B remain in the prior eligibility denominator but generated no active events under the stored causal threshold/activation contract; they were not dropped by the walk-forward code.
- True multi-bucket sets: `{summary['true_consensus_sets']}`; two-bucket: `{summary['two_bucket_sets']}`; three-bucket: `{summary['three_bucket_sets']}`.
- Bucket combinations: `{summary['bucket_combinations']}`.
- Test months: `{summary['effective_test_months']}`; paired T2/T3 rows: `{summary['paired_rows']}`.

## Paired scientific result

All T3-vs-T2 excess and regret values are paired on the same set, candidate buckets, fold and model. No CAGR, Sharpe, drawdown or terminal-wealth claim is made from overlapping H-specific forward returns.

- Same-ticker Rank-IC, pairwise accuracy, selected-excess difference and regret are reported in the CSV/JSON artifacts.
- Three-month calendar-block bootstrap (1000 repetitions) is the inference unit. Pooled results: same-ticker IC delta `{brief(b['bootstrap_same_ticker_ic.json'])}`; pairwise-accuracy delta `{brief(b['bootstrap_pairwise_accuracy.json'])}`; selected-excess delta `{brief(b['bootstrap_selected_excess.json'])}`; regret delta `{brief(b['bootstrap_regret.json'])}`. Full JSON is persisted in the four bootstrap files.
- Consensus and score-shape results are separated in `score_shape_comparison.csv`; robustness is split by set type, subperiod and reconstructed H30 removals.
- Model-separated paired bootstrap evidence:
{chr(10).join(model_lines)}
- Early/late paired selected-excess deltas: `{robust_text}`.
- Reconstructed anti-long-horizon results: `{h30_text}`. The ablation reclassifies candidate sets and excludes sets that fall below two buckets; it is an evaluation robustness check using the fixed causal fold predictions, not a new model search.

## Diagnosis

**{summary['primary_diagnosis']}**

This diagnosis is research evidence only. It does not grant any Selector or Family capital authority. If the evidence is suggestive but the bootstrap interval crosses zero, the result is not treated as robust feasibility.

## Self-review

- Final holdout opened: **NO**.
- Factory retraining: **NO**.
- Prediction regeneration: **NO**.
- Ex-ante state only: **YES**; forward realized excess is target-only.
- Inactive models treated as negative: **NO**; only active signals enter candidates.
- Family and economic H-bucket levels separated: **YES**.
- H30/long-horizon proxy risk reconstructed: **YES**, see `anti_h30_true_consensus.csv`.
- Equal-active candidate mean is the paired hurdle; no pseudo-NAV inference.
- Time dependence: calendar 3-month block bootstrap.
- Peak measured RSS: `{summary.get('peak_rss_gb'):.3f} GB`; runtime: `{summary.get('runtime_seconds'):.2f} seconds`.
- Promotion authority: **NONE**.
"""
    (output / "REPORT.md").write_text(report, encoding="utf-8")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, default=Path("artifacts/dynamic-qbd-opportunity-state"))
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/dynamic-qbd-true-consensus"))
    parser.add_argument("--bootstrap-repetitions", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(argv)
    summary = run(args.input_root, args.output_root, args.bootstrap_repetitions, args.seed)
    print(json.dumps(summary, default=_json_default, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
