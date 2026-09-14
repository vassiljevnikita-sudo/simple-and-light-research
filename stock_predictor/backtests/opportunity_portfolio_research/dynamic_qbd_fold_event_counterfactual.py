"""Event-level counterfactual diagnosis for Dynamic-QBD fold-clock refits.

This Development-only follow-up consumes a completed Fold-Clock surface validation.
It does not refit the 1,936 source models. For every non-initial Fold-Clock
event it branches the exact same pre-event portfolio state:

OLD -- keep the incumbent entry model and incumbent calibration for one more
       Fold-Clock interval.
NEW -- activate the already-fitted Fold-Clock model and its calibration.

NEW becomes the canonical state for the next event. This measures the immediate
causal value of each refresh while preserving the actual Fold-Clock history.

The primary suite intentionally uses FIXED exits only. Counterfactual OLD entry
signals can create positions not present in the original A/F/M run; reusing the
original learned-exit provider for those positions would create missing exit
predictions and confound entry-refit diagnosis with exit-model coverage.

No activation gate is learned here. Ex-ante event features and post-event signal
shift diagnostics are reported separately. No recipe switching, performance
trigger, threshold search, promotion, capital authority or final holdout access
is permitted.
"""
from __future__ import annotations

import os
os.environ.setdefault("DYNAMIC_QBD_MEMORY_LIMIT_GB", "90")
os.environ.setdefault("DYNAMIC_QBD_MEMORY_SOFT_TARGET_GB", "84")

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date
import json
import math
import multiprocessing as mp
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .contract_fingerprints import stable_hash
from .cost_contracts import CostModel
from .portfolio_policy_contracts import Policy
from .tax_contracts import TaxConfig
from .cpu_topology import set_current_process_affinity
from .dynamic_qbd_fold_clock_surface_validation_2016_2025 import (
    ARM_F,
    CLOSED_HOLDOUT_START,
    HOLDOUT_CONTRACT,
    SCHEMA_VERSION as SOURCE_SCHEMA_VERSION,
    LEGACY_SCHEMA_VERSIONS as SOURCE_LEGACY_SCHEMA_VERSIONS,
    _family_schedule,
    _schedule_from_audit,
    _sha256_file,
    _write_json,
)
from .dynamic_qbd_h1_30_adapter import FEATURE_COLUMNS
from .dynamic_qbd_portfolio_replay import replay_family
from .dynamic_qbd_runtime_resources import active_cpu_contract, configure_cpu_peak, cpu_budget
from .dynamic_qbd_runtime_telemetry import NWInfoSampler
from .dynamic_qbd_family_surface import build_family_specs, structural_plateau_id_from_family_id


AUTHORITY = "SHADOW_ONLY_NO_PROMOTION_NO_HOLDOUT"
SCHEMA_VERSION = "DYNAMIC_QBD_FOLD_EVENT_COUNTERFACTUAL_V1"
COUNTERFACTUAL_OLD = "OLD_KEEP_INCUMBENT_ONE_MORE_FOLD_INTERVAL"
COUNTERFACTUAL_NEW = "NEW_ACTIVATE_FOLD_CLOCK_REFIT"
DEFAULT_HORIZON_PROCESS_WORKERS = 24

EX_ANTE_FEATURE_FIELDS = (
    "incumbent_model_age_days",
    "fold_count_increment",
    "threshold_delta",
    "threshold_pct_delta",
    "calibration_observation_delta",
    "train_start_shift_days",
    "train_end_shift_days",
)
POST_EVENT_DIAGNOSTIC_FIELDS = (
    "post_event_score_spearman",
    "post_event_daily_top_overlap_median",
    "post_event_threshold_pass_jaccard_median",
)


def _feature_schema_fingerprint(signal_panel: Path) -> str:
    schema = pq.ParquetFile(signal_panel).schema_arrow
    return stable_hash([
        (name, str(schema.field(name).type))
        for name in FEATURE_COLUMNS if name in schema.names
    ])


def _resolve_source_prediction(path_value: str, source_root: Path) -> Path:
    path = Path(str(path_value))
    if path.is_file():
        return path
    normalized = str(path_value).replace("\\", "/")
    marker = "/entry-models/"
    if marker in normalized:
        suffix = normalized.split(marker, 1)[1]
        candidate = source_root / "entry-models" / Path(suffix)
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"COUNTERFACTUAL_SOURCE_PREDICTION_MISSING:{path_value}")


def _source_contract(source_root: Path, signal_panel: Path) -> tuple[dict, dict]:
    summary_path = source_root / "summary.json"
    contract_path = source_root / "run-contract.json"
    entry_path = source_root / "entry-fit-audit.parquet"
    prices_path = source_root / "inputs" / "prices.parquet"
    required = (summary_path, contract_path, entry_path, prices_path)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"COUNTERFACTUAL_SOURCE_SURFACE_VALIDATION_ARTIFACTS_MISSING:{missing}"
        )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    checks = {
        "source_status_complete": summary.get("status") == "COMPLETE",
        "source_schema_supported": summary.get("schema_version") in ({SOURCE_SCHEMA_VERSION} | set(SOURCE_LEGACY_SCHEMA_VERSIONS)),
        "source_holdout_closed": summary.get("final_holdout_opened") is False,
        "source_promotion_disabled": summary.get("promotion_allowed") is False,
        "source_holdout_contract_matches": summary.get("holdout_contract") == HOLDOUT_CONTRACT,
        "source_development_end_pre_holdout":
            date.fromisoformat(summary["development_end"]) < CLOSED_HOLDOUT_START,
        "source_family_count_5400": int(summary.get("family_count", 0)) == 5400,
        "source_fold_clock_fit_count_positive":
            int(summary.get("fold_clock_fit_count_total", 0)) > 30,
        "signal_panel_hash_matches_source":
            _sha256_file(signal_panel) == str(contract.get("signal_panel_sha256")),
    }
    failed = sorted(key for key, value in checks.items() if not value)
    if failed:
        raise AssertionError(f"COUNTERFACTUAL_SOURCE_CONTRACT_FAILED:{failed}")
    return summary, {"checks": checks, "source_run_contract": contract}


def _market_interval_end(
    market_dates: tuple[pd.Timestamp, ...],
    *,
    event_date: pd.Timestamp,
    next_event_date: pd.Timestamp | None,
    development_end: pd.Timestamp,
) -> pd.Timestamp:
    dates = pd.DatetimeIndex(market_dates)
    eligible = (
        dates[dates <= development_end]
        if next_event_date is None
        else dates[dates < next_event_date]
    )
    eligible = eligible[eligible >= event_date]
    if not len(eligible):
        raise ValueError(f"COUNTERFACTUAL_EMPTY_EVENT_INTERVAL:{event_date.date()}")
    return pd.Timestamp(eligible[-1]).normalize()


def _prior_market_date(
    market_dates: tuple[pd.Timestamp, ...], event_date: pd.Timestamp
) -> pd.Timestamp:
    dates = pd.DatetimeIndex(market_dates)
    eligible = dates[dates < event_date]
    if not len(eligible):
        raise ValueError(f"COUNTERFACTUAL_NO_PRIOR_MARKET_DATE:{event_date.date()}")
    return pd.Timestamp(eligible[-1]).normalize()


def _event_rows(
    horizon_audit: pd.DataFrame,
    market_dates: tuple[pd.Timestamp, ...],
    development_end: date,
) -> list[dict]:
    source = horizon_audit.sort_values("assessment_date").reset_index(drop=True)
    refits = source.loc[source["f_refit"].astype(bool)].reset_index(drop=True)
    if len(refits) < 2:
        return []
    rows = []
    for index in range(1, len(refits)):
        old = refits.iloc[index - 1]
        new = refits.iloc[index]
        event_date = pd.Timestamp(new["assessment_date"]).normalize()
        next_event = (
            pd.Timestamp(refits.iloc[index + 1]["assessment_date"]).normalize()
            if index + 1 < len(refits) else None
        )
        interval_end = _market_interval_end(
            market_dates,
            event_date=event_date,
            next_event_date=next_event,
            development_end=pd.Timestamp(development_end),
        )
        if (
            str(old["candidate_id"]) != str(new["candidate_id"])
            or str(old["model_family"]) != str(new["model_family"])
            or str(old["parameters_json"]) != str(new["parameters_json"])
        ):
            raise AssertionError(
                f"COUNTERFACTUAL_RECIPE_DRIFT:H{int(new['horizon'])}:{event_date.date()}"
            )
        if str(old["model_artifact_id"]) == str(new["model_artifact_id"]):
            raise AssertionError(
                f"COUNTERFACTUAL_REFIT_REUSED_MODEL:H{int(new['horizon'])}:{event_date.date()}"
            )
        old_fold_count = int(old["fold_count"])
        new_fold_count = int(new["fold_count"])
        if new_fold_count <= old_fold_count:
            raise AssertionError(
                f"COUNTERFACTUAL_NONEXPANDING_FOLD_EVENT:H{int(new['horizon'])}:{event_date.date()}"
            )
        old_threshold = float(old["resolved_threshold"])
        new_threshold = float(new["resolved_threshold"])
        old_train_start = pd.Timestamp(old["train_start"])
        new_train_start = pd.Timestamp(new["train_start"])
        old_train_end = pd.Timestamp(old["train_end"])
        new_train_end = pd.Timestamp(new["train_end"])
        rows.append({
            "event_id":
                f"H{int(new['horizon']):02d}_{event_date.date()}_F{new_fold_count}",
            "horizon": int(new["horizon"]),
            "event_index": index,
            "event_date": event_date,
            "interval_end": interval_end,
            "next_event_date": next_event,
            "old_assessment_date": pd.Timestamp(old["assessment_date"]).normalize(),
            "new_assessment_date": event_date,
            "old_model_artifact_id": str(old["model_artifact_id"]),
            "new_model_artifact_id": str(new["model_artifact_id"]),
            "old_generation_id": str(old["generation_id"]),
            "new_generation_id": str(new["generation_id"]),
            "old_prediction_artifact_path": str(old["prediction_artifact_path"]),
            "new_prediction_artifact_path": str(new["prediction_artifact_path"]),
            "candidate_id": str(new["candidate_id"]),
            "model_family": str(new["model_family"]),
            "parameters_json": str(new["parameters_json"]),
            "recipe_identity_unchanged": True,
            "old_fold_count": old_fold_count,
            "new_fold_count": new_fold_count,
            "fold_count_increment": new_fold_count - old_fold_count,
            "old_evidence_fingerprint": str(old["evidence_fingerprint"]),
            "new_evidence_fingerprint": str(new["evidence_fingerprint"]),
            "old_threshold": old_threshold,
            "new_threshold": new_threshold,
            "threshold_delta": new_threshold - old_threshold,
            "threshold_pct_delta": (
                new_threshold / old_threshold - 1.0
                if abs(old_threshold) > 1e-15 else float("nan")
            ),
            "old_top_fraction": float(old["resolved_top_fraction"]),
            "new_top_fraction": float(new["resolved_top_fraction"]),
            "old_calibration_observations": int(old["calibration_observations"]),
            "new_calibration_observations": int(new["calibration_observations"]),
            "calibration_observation_delta":
                int(new["calibration_observations"])
                - int(old["calibration_observations"]),
            "incumbent_model_age_days":
                int((event_date - pd.Timestamp(old["assessment_date"])).days),
            "old_train_start": old_train_start,
            "new_train_start": new_train_start,
            "train_start_shift_days":
                int((new_train_start - old_train_start).days),
            "old_train_end": old_train_end,
            "new_train_end": new_train_end,
            "train_end_shift_days":
                int((new_train_end - old_train_end).days),
        })
    return rows


def _read_interval_features(
    signal_panel: Path,
    *,
    event_date: pd.Timestamp,
    interval_end: pd.Timestamp,
) -> pd.DataFrame:
    columns = ["decision_date", "ticker", *FEATURE_COLUMNS]
    filters = [
        ("decision_date", ">", event_date.to_pydatetime()),
        ("decision_date", "<=", interval_end.to_pydatetime()),
    ]
    frame = pd.read_parquet(signal_panel, columns=columns, filters=filters)
    frame["decision_date"] = pd.to_datetime(frame["decision_date"]).dt.normalize()
    if frame.empty:
        raise ValueError(
            f"COUNTERFACTUAL_INTERVAL_FEATURES_EMPTY:"
            f"{event_date.date()}:{interval_end.date()}"
        )
    if frame.duplicated(["decision_date", "ticker"]).any():
        raise ValueError("COUNTERFACTUAL_INTERVAL_FEATURE_DUPLICATE_KEYS")
    return frame


def _old_interval_signals(
    *,
    event: Mapping[str, Any],
    signal_panel: Path,
    source_root: Path,
    output_root: Path,
) -> pd.DataFrame:
    cache = (
        output_root
        / "old-counterfactual-signals"
        / f"{event['event_id']}-OLD.parquet"
    )
    if cache.is_file():
        return pd.read_parquet(cache)
    prediction_path = _resolve_source_prediction(
        str(event["old_prediction_artifact_path"]), source_root
    )
    model_path = prediction_path.with_name("model.joblib")
    if not model_path.is_file():
        raise FileNotFoundError(
            f"COUNTERFACTUAL_INCUMBENT_MODEL_MISSING:{model_path}"
        )
    frame = _read_interval_features(
        signal_panel,
        event_date=pd.Timestamp(event["event_date"]),
        interval_end=pd.Timestamp(event["interval_end"]),
    )
    model = joblib.load(model_path)
    scores = model.predict(frame[list(FEATURE_COLUMNS)].to_numpy(float))
    signals = frame[["decision_date", "ticker"]].copy()
    signals["score"] = np.asarray(scores, dtype=float)
    if not np.isfinite(signals["score"].to_numpy(float)).all():
        raise AssertionError(
            f"COUNTERFACTUAL_NONFINITE_OLD_SCORE:{event['event_id']}"
        )
    cache.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache.with_name(cache.name + f".{os.getpid()}.tmp")
    signals.to_parquet(temporary, index=False)
    os.replace(temporary, cache)
    return signals


def _post_event_signal_diagnostics(
    event: Mapping[str, Any],
    *,
    old_signals: pd.DataFrame,
    new_f_signals: pd.DataFrame,
) -> dict[str, Any]:
    start = pd.Timestamp(event["event_date"])
    end = pd.Timestamp(event["interval_end"])
    decision_dates = pd.to_datetime(new_f_signals["decision_date"])
    new = new_f_signals.loc[
        decision_dates.gt(start) & decision_dates.le(end),
        ["decision_date", "ticker", "score"],
    ].copy()
    new["decision_date"] = pd.to_datetime(new["decision_date"]).dt.normalize()
    old = old_signals[["decision_date", "ticker", "score"]].copy()
    merged = old.merge(
        new,
        on=["decision_date", "ticker"],
        suffixes=("_old", "_new"),
        validate="one_to_one",
    )
    if merged.empty:
        return {
            "post_event_common_signal_rows": 0,
            "post_event_score_spearman": float("nan"),
            "post_event_daily_top_overlap_median": float("nan"),
            "post_event_threshold_pass_jaccard_median": float("nan"),
        }
    spearman = merged["score_old"].corr(
        merged["score_new"], method="spearman"
    )
    top_overlaps = []
    threshold_overlaps = []
    old_threshold = float(event["old_threshold"])
    new_threshold = float(event["new_threshold"])
    top_fraction = float(event["new_top_fraction"])
    for _, day in merged.groupby("decision_date", sort=True):
        take = max(1, int(math.ceil(len(day) * top_fraction)))
        old_top = set(
            day.nlargest(take, "score_old")["ticker"].astype(str)
        )
        new_top = set(
            day.nlargest(take, "score_new")["ticker"].astype(str)
        )
        union = old_top | new_top
        top_overlaps.append(
            len(old_top & new_top) / len(union) if union else 1.0
        )
        old_pass = set(
            day.loc[day["score_old"].ge(old_threshold), "ticker"].astype(str)
        )
        new_pass = set(
            day.loc[day["score_new"].ge(new_threshold), "ticker"].astype(str)
        )
        pass_union = old_pass | new_pass
        threshold_overlaps.append(
            len(old_pass & new_pass) / len(pass_union)
            if pass_union else 1.0
        )
    return {
        "post_event_common_signal_rows": int(len(merged)),
        "post_event_score_spearman":
            float(spearman) if pd.notna(spearman) else float("nan"),
        "post_event_daily_top_overlap_median":
            float(np.median(top_overlaps)),
        "post_event_threshold_pass_jaccard_median":
            float(np.median(threshold_overlaps)),
    }


def _policy(family) -> Policy:
    return Policy(
        horizon=int(family.horizon_sessions),
        score_quantile=float(
            family.entry_policy_rule.get("score_quantile", .975)
        ),
        top_fraction=float(
            family.entry_policy_rule.get("top_fraction", .005)
        ),
        max_names=int(family.max_names),
        holding_days=int(family.holding_days),
        exit_family="FIXED",
        exit_value=0.0,
        replacement=str(
            family.exit_policy.get("replacement", "IGNORE_NEW")
        ),
        allocation=str(
            family.entry_policy_rule.get("allocation", "EQUAL_ACTIVE")
        ),
        sleeve=float(family.entry_policy_rule.get("sleeve", .50)),
    )


def _state_fingerprint(state: Mapping[str, Any]) -> str:
    return stable_hash({
        "as_of": state.get("as_of"),
        "cash": state.get("cash"),
        "urth_units": state.get("urth_units"),
        "positions": state.get("positions"),
        "pending_orders": state.get("pending_orders"),
        "transaction_cost_eur": state.get("transaction_cost_eur"),
        "threshold": state.get("threshold"),
    })


def _segment_metrics(
    result: dict,
    prior_state: Mapping[str, Any],
    *,
    initial: float,
) -> dict[str, float | int]:
    prior_curve = list(prior_state.get("curve", []))
    prior_len = len(prior_curve)
    curve = result["curve"].reset_index(drop=True)
    segment = curve.iloc[prior_len:].copy()
    if segment.empty:
        raise ValueError("COUNTERFACTUAL_SEGMENT_CURVE_EMPTY")
    if prior_curve:
        start_strategy = float(prior_curve[-1]["strategy_value"])
        start_urth = float(prior_curve[-1]["urth_value"])
    else:
        start_strategy = float(initial)
        start_urth = float(initial)
    end_strategy = float(segment.iloc[-1]["strategy_value"])
    end_urth = float(segment.iloc[-1]["urth_value"])
    strategy_return = end_strategy / start_strategy - 1.0
    benchmark_return = end_urth / start_urth - 1.0
    relative = np.concatenate([
        np.asarray([1.0]),
        (segment["strategy_value"].to_numpy(float) / start_strategy)
        / (segment["urth_value"].to_numpy(float) / start_urth),
    ])
    peak = np.maximum.accumulate(relative)
    relative_maxdd = float(np.min(relative / peak - 1.0))
    prior_trades = len(prior_state.get("trades", []))
    trade_count = int(len(result.get("trades", [])) - prior_trades)
    prior_cost = float(
        prior_state.get("transaction_cost_eur", 0.0)
    )
    end_cost = float(
        result["replay_state"].get(
            "transaction_cost_eur", prior_cost
        )
    )
    return {
        "terminal_value": end_strategy,
        "strategy_return": strategy_return,
        "benchmark_return": benchmark_return,
        "excess_return": strategy_return - benchmark_return,
        "relative_max_drawdown": relative_maxdd,
        "trade_count": trade_count,
        "cost_eur": end_cost - prior_cost,
        "segment_sessions": int(len(segment)),
    }


def _old_shared_schedule(
    event: Mapping[str, Any], horizon: int
) -> pd.DataFrame:
    return pd.DataFrame([{
        "activation_date": pd.Timestamp(
            event["old_assessment_date"]
        ),
        "family_id": f"H{horizon:02d}_SHARED_OLD",
        "generation_id": str(event["old_generation_id"]),
        "model_artifact_id": str(
            event["old_model_artifact_id"]
        ),
        "resolved_threshold": float(event["old_threshold"]),
        "resolved_top_fraction": float(event["old_top_fraction"]),
        "entry_policy_id":
            f"OLD_KEEP_INCUMBENT:{event['event_id']}",
        "exit_policy_id": "FIXED_SHARED",
    }])


def _prefix_state(
    *,
    family,
    f_signals: pd.DataFrame,
    f_schedule: pd.DataFrame,
    prices: pd.DataFrame,
    evaluation_start: date,
    prefix_end: pd.Timestamp,
    initial: float,
) -> dict:
    result = replay_family(
        signals=f_signals,
        authoritative_signals=f_signals,
        prices=prices,
        policy=_policy(family),
        generation_schedule=_family_schedule(
            f_schedule,
            family=family,
            arm=ARM_F,
            exit_generation_id="",
        ),
        cost=CostModel(
            float(family.cost_contract.get("roundtrip_bps", 20.0))
        ),
        tax=TaxConfig(enabled=False),
        start=pd.Timestamp(evaluation_start),
        end=prefix_end,
        initial=initial,
    )
    return result["replay_state"]


def _branch_family_event(
    *,
    family,
    event: Mapping[str, Any],
    state: dict,
    old_signals: pd.DataFrame,
    f_signals: pd.DataFrame,
    f_schedule: pd.DataFrame,
    prices: pd.DataFrame,
    initial: float,
) -> tuple[dict, dict]:
    fingerprint = _state_fingerprint(state)
    common = {
        "prices": prices,
        "policy": _policy(family),
        "cost": CostModel(
            float(family.cost_contract.get("roundtrip_bps", 20.0))
        ),
        "tax": TaxConfig(enabled=False),
        "start": pd.Timestamp(event["event_date"]),
        "end": pd.Timestamp(event["interval_end"]),
        "initial": initial,
        "resume_state": state,
    }
    old_shared = _old_shared_schedule(
        event, int(family.horizon_sessions)
    )
    old_result = replay_family(
        signals=old_signals,
        authoritative_signals=old_signals,
        generation_schedule=_family_schedule(
            old_shared,
            family=family,
            arm=COUNTERFACTUAL_OLD,
            exit_generation_id="",
        ),
        **common,
    )
    new_result = replay_family(
        signals=f_signals,
        authoritative_signals=f_signals,
        generation_schedule=_family_schedule(
            f_schedule,
            family=family,
            arm=COUNTERFACTUAL_NEW,
            exit_generation_id="",
        ),
        **common,
    )
    old_metrics = _segment_metrics(
        old_result, state, initial=initial
    )
    new_metrics = _segment_metrics(
        new_result, state, initial=initial
    )
    row = {
        "event_id": str(event["event_id"]),
        "horizon": int(family.horizon_sessions),
        "event_date":
            pd.Timestamp(event["event_date"]).date().isoformat(),
        "interval_end":
            pd.Timestamp(event["interval_end"]).date().isoformat(),
        "event_year": int(pd.Timestamp(event["event_date"]).year),
        "family_id": str(family.family_id),
        "structural_plateau_id":
            structural_plateau_id_from_family_id(
                family.family_id
            ),
        "holding_days": int(family.holding_days),
        "max_names": int(family.max_names),
        "exit_mode": "FIXED",
        "branch_start_state_fingerprint": fingerprint,
        "old_terminal_value": old_metrics["terminal_value"],
        "new_terminal_value": new_metrics["terminal_value"],
        "new_minus_old_terminal_value_eur": float(
            new_metrics["terminal_value"]
            - old_metrics["terminal_value"]
        ),
        "old_strategy_return": old_metrics["strategy_return"],
        "new_strategy_return": new_metrics["strategy_return"],
        "new_minus_old_strategy_return": float(
            new_metrics["strategy_return"]
            - old_metrics["strategy_return"]
        ),
        "old_excess_return": old_metrics["excess_return"],
        "new_excess_return": new_metrics["excess_return"],
        "new_minus_old_excess_return": float(
            new_metrics["excess_return"]
            - old_metrics["excess_return"]
        ),
        "old_relative_max_drawdown":
            old_metrics["relative_max_drawdown"],
        "new_relative_max_drawdown":
            new_metrics["relative_max_drawdown"],
        "new_minus_old_relative_maxdrawdown": float(
            new_metrics["relative_max_drawdown"]
            - old_metrics["relative_max_drawdown"]
        ),
        "old_trade_count": old_metrics["trade_count"],
        "new_trade_count": new_metrics["trade_count"],
        "new_minus_old_trade_count": int(
            new_metrics["trade_count"]
            - old_metrics["trade_count"]
        ),
        "old_cost_eur": old_metrics["cost_eur"],
        "new_cost_eur": new_metrics["cost_eur"],
        "new_minus_old_cost_eur": float(
            new_metrics["cost_eur"]
            - old_metrics["cost_eur"]
        ),
        "segment_sessions": int(new_metrics["segment_sessions"]),
    }
    return row, new_result["replay_state"]


def _horizon_checkpoint_paths(
    root: Path,
    horizon: int,
) -> tuple[Path, Path, Path]:
    checkpoint = root / "checkpoints"
    return (
        checkpoint
        / f"H{horizon:02d}-event-family.parquet",
        checkpoint
        / f"H{horizon:02d}-event-ledger.parquet",
        checkpoint / f"H{horizon:02d}.json",
    )


def _load_horizon_checkpoint(
    root: Path,
    horizon: int,
    run_hash: str,
):
    family_path, ledger_path, meta_path = (
        _horizon_checkpoint_paths(root, horizon)
    )
    if not all(
        path.is_file()
        for path in (
            family_path,
            ledger_path,
            meta_path,
        )
    ):
        return None
    try:
        meta = json.loads(
            meta_path.read_text(encoding="utf-8")
        )
        if (
            meta.get("semantic_run_contract_hash")
            != run_hash
            or not meta.get("complete")
        ):
            return None
        family = pd.read_parquet(family_path)
        ledger = pd.read_parquet(ledger_path)
        if (
            len(family) != int(meta["family_rows"])
            or len(ledger) != int(meta["event_rows"])
        ):
            return None
        return family, ledger
    except Exception:
        return None


def _save_horizon_checkpoint(
    root: Path,
    horizon: int,
    run_hash: str,
    family: pd.DataFrame,
    ledger: pd.DataFrame,
) -> None:
    family_path, ledger_path, meta_path = (
        _horizon_checkpoint_paths(root, horizon)
    )
    family_path.parent.mkdir(
        parents=True, exist_ok=True
    )
    family.to_parquet(family_path, index=False)
    ledger.to_parquet(ledger_path, index=False)
    _write_json(meta_path, {
        "schema_version":
            "DYNAMIC_QBD_FOLD_EVENT_COUNTERFACTUAL_HORIZON_CHECKPOINT_V1",
        "semantic_run_contract_hash": run_hash,
        "horizon": int(horizon),
        "complete": True,
        "family_rows": int(len(family)),
        "event_rows": int(len(ledger)),
    })


def _run_horizon(
    *,
    horizon: int,
    horizon_audit: pd.DataFrame,
    families: tuple[Any, ...],
    source_root: Path,
    signal_panel: Path,
    output_root: Path,
    prices_path: Path,
    market_dates: tuple[pd.Timestamp, ...],
    development_start: date,
    development_end: date,
    initial: float,
    run_hash: str,
    logical_processor: int | None,
) -> dict[str, Any]:
    started = pd.Timestamp.utcnow()
    affinity_pinned = (
        bool(
            set_current_process_affinity(
                [int(logical_processor)]
            )
        )
        if logical_processor is not None
        else False
    )
    f_signal_path = (
        source_root
        / "authoritative-signals"
        / f"H{horizon:02d}-{ARM_F}.parquet"
    )
    if not f_signal_path.is_file():
        raise FileNotFoundError(
            f"COUNTERFACTUAL_SOURCE_F_SIGNALS_MISSING:"
            f"{f_signal_path}"
        )
    f_signals = pd.read_parquet(f_signal_path)
    f_signals["decision_date"] = pd.to_datetime(
        f_signals["decision_date"]
    ).dt.normalize()
    prices = pd.read_parquet(prices_path)
    f_schedule = _schedule_from_audit(
        horizon_audit, ARM_F
    )
    evaluation_start = pd.Timestamp(
        horizon_audit.sort_values("assessment_date").iloc[0]["assessment_date"]
    ).date()
    events = _event_rows(
        horizon_audit,
        market_dates,
        development_end,
    )
    if not events:
        raise AssertionError(
            f"COUNTERFACTUAL_NO_NONINITIAL_EVENTS:H{horizon}"
        )

    ledger_rows = []
    signal_paths: dict[str, Path] = {}
    for event in events:
        old_signals = _old_interval_signals(
            event=event,
            signal_panel=signal_panel,
            source_root=source_root,
            output_root=output_root,
        )
        diagnostics = _post_event_signal_diagnostics(
            event,
            old_signals=old_signals,
            new_f_signals=f_signals,
        )
        ledger_rows.append({
            **{
                key: value
                for key, value in event.items()
                if not key.endswith(
                    "_prediction_artifact_path"
                )
            },
            **diagnostics,
            "diagnostic_contract":
                "EX_ANTE_FIELDS_ACTIVATION_SAFE__POST_EVENT_FIELDS_DIAGNOSTIC_ONLY",
        })
        signal_paths[str(event["event_id"])] = (
            output_root
            / "old-counterfactual-signals"
            / f"{event['event_id']}-OLD.parquet"
        )
        del old_signals

    prefix_end = _prior_market_date(
        market_dates,
        pd.Timestamp(events[0]["event_date"]),
    )
    family_rows: list[dict] = []
    states: dict[str, dict] = {}

    # Build each family's canonical pre-first-event state once. Event-first
    # processing below then loads each OLD signal interval only once and shares
    # the prepared signal frame across every D/N family in this horizon.
    for family_index, family in enumerate(families):
        states[str(family.family_id)] = _prefix_state(
            family=family,
            f_signals=f_signals,
            f_schedule=f_schedule,
            prices=prices,
            evaluation_start=evaluation_start,
            prefix_end=prefix_end,
            initial=initial,
        )
        if (
            (family_index + 1) % 25 == 0
            or family_index + 1 == len(families)
        ):
            print(
                f"[fold-event] H{horizon:02d} prefix states "
                f"{family_index + 1}/{len(families)}",
                flush=True,
            )

    for event_index, event in enumerate(events):
        old_signals = pd.read_parquet(
            signal_paths[str(event["event_id"])]
        )
        for family in families:
            family_id = str(family.family_id)
            row, next_state = _branch_family_event(
                family=family,
                event=event,
                state=states[family_id],
                old_signals=old_signals,
                f_signals=f_signals,
                f_schedule=f_schedule,
                prices=prices,
                initial=initial,
            )
            family_rows.append(row)
            states[family_id] = next_state
        del old_signals
        print(
            f"[fold-event] H{horizon:02d} event "
            f"{event_index + 1}/{len(events)} "
            f"{event['event_id']} families={len(families)}",
            flush=True,
        )
    family_frame = pd.DataFrame(
        family_rows
    ).sort_values(
        ["event_date", "family_id"]
    ).reset_index(drop=True)
    ledger = pd.DataFrame(
        ledger_rows
    ).sort_values(
        "event_date"
    ).reset_index(drop=True)
    expected = len(families) * len(events)
    if len(family_frame) != expected:
        raise AssertionError(
            f"COUNTERFACTUAL_HORIZON_ROWCOUNT:"
            f"H{horizon}:{len(family_frame)}!={expected}"
        )
    _save_horizon_checkpoint(
        output_root,
        horizon,
        run_hash,
        family_frame,
        ledger,
    )
    elapsed = (
        pd.Timestamp.utcnow() - started
    ).total_seconds()
    return {
        "horizon": int(horizon),
        "family": family_frame,
        "ledger": ledger,
        "logical_processor": logical_processor,
        "affinity_pinned": affinity_pinned,
        "compute_seconds": float(elapsed),
    }


def _contract_audit(
    *,
    source_summary: dict,
    source_entry_audit: pd.DataFrame,
    event_ledger: pd.DataFrame,
    family_rows: pd.DataFrame,
    fixed_family_count: int,
) -> dict[str, Any]:
    refits = source_entry_audit.loc[
        source_entry_audit["f_refit"].astype(bool)
    ]
    expected_events = int(
        len(refits)
        - source_entry_audit["horizon"].nunique()
    )
    expected_family_rows = int(sum(
        len(event_ledger.loc[event_ledger["horizon"].eq(h)])
        * family_rows.loc[
            family_rows["horizon"].eq(h), "family_id"
        ].nunique()
        for h in sorted(event_ledger["horizon"].unique())
    ))
    checks = {
        "final_holdout_opened_false":
            source_summary.get(
                "final_holdout_opened"
            ) is False,
        "promotion_allowed_false":
            source_summary.get(
                "promotion_allowed"
            ) is False,
        "source_status_complete":
            source_summary.get("status") == "COMPLETE",
        "all_30_horizons_present":
            int(
                source_entry_audit[
                    "horizon"
                ].nunique()
            ) == 30,
        "event_count_matches_noninitial_fold_refits":
            len(event_ledger) == expected_events,
        "event_family_row_coverage_complete":
            len(family_rows) == expected_family_rows,
        "same_recipe_within_each_event":
            bool(event_ledger["recipe_identity_unchanged"].astype(bool).all()),
        "only_real_fold_expansion_events":
            bool(
                (
                    event_ledger[
                        "new_fold_count"
                    ].astype(int)
                    >
                    event_ledger[
                        "old_fold_count"
                    ].astype(int)
                ).all()
            ),
        "new_model_differs_from_incumbent":
            bool(
                (
                    event_ledger[
                        "new_model_artifact_id"
                    ].astype(str)
                    !=
                    event_ledger[
                        "old_model_artifact_id"
                    ].astype(str)
                ).all()
            ),
        "new_fit_date_equals_event_date":
            bool(
                pd.to_datetime(
                    event_ledger[
                        "new_assessment_date"
                    ]
                ).dt.normalize().equals(
                    pd.to_datetime(
                        event_ledger["event_date"]
                    ).dt.normalize()
                )
            ),
        "incumbent_fit_precedes_event":
            bool(
                (
                    pd.to_datetime(
                        event_ledger[
                            "old_assessment_date"
                        ]
                    )
                    <
                    pd.to_datetime(
                        event_ledger["event_date"]
                    )
                ).all()
            ),
        "fixed_exit_only":
            bool(
                family_rows[
                    "exit_mode"
                ].eq("FIXED").all()
            ),
        "fixed_family_coverage_complete":
            int(
                family_rows[
                    "family_id"
                ].nunique()
            ) == fixed_family_count,
        "no_activation_gate_learned": True,
        "post_event_diagnostics_not_activation_authority":
            True,
        "counterfactual_new_is_canonical_fold_clock_path":
            True,
    }
    failed = sorted(
        key for key, value in checks.items()
        if not value
    )
    if failed:
        raise AssertionError(
            f"COUNTERFACTUAL_CONTRACT_AUDIT_FAILED:"
            f"{failed}"
        )
    return {
        "schema_version":
            "DYNAMIC_QBD_FOLD_EVENT_COUNTERFACTUAL_CONTRACT_AUDIT_V1",
        "status": "PASS",
        "checks": checks,
        "expected_noninitial_event_count":
            expected_events,
        "fixed_family_count": fixed_family_count,
        "expected_event_family_rows": expected_family_rows,
        "authority": AUTHORITY,
    }


def _event_summary(
    family_rows: pd.DataFrame,
    ledger: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    keys = [
        "event_id",
        "horizon",
        "event_date",
        "interval_end",
        "event_year",
    ]
    for values, frame in family_rows.groupby(
        keys, sort=True
    ):
        row = dict(zip(keys, values))
        excess = frame[
            "new_minus_old_excess_return"
        ].to_numpy(float)
        dd = frame[
            "new_minus_old_relative_maxdrawdown"
        ].to_numpy(float)
        row.update({
            "family_count":
                int(frame["family_id"].nunique()),
            "median_new_minus_old_excess_return":
                float(np.median(excess)),
            "median_new_minus_old_terminal_value_eur":
                float(
                    frame[
                        "new_minus_old_terminal_value_eur"
                    ].median()
                ),
            "median_new_minus_old_relative_maxdrawdown":
                float(np.median(dd)),
            "median_new_minus_old_trade_count":
                float(
                    frame[
                        "new_minus_old_trade_count"
                    ].median()
                ),
            "median_new_minus_old_cost_eur":
                float(
                    frame[
                        "new_minus_old_cost_eur"
                    ].median()
                ),
            "positive_family_fraction":
                float(np.mean(excess > 0)),
            "risk_not_worse_family_fraction":
                float(np.mean(dd >= 0)),
        })
        if (
            row[
                "median_new_minus_old_excess_return"
            ] > 0
            and
            row[
                "median_new_minus_old_relative_maxdrawdown"
            ] >= 0
        ):
            diagnosis = "RETURN_AND_RISK_HELPFUL"
        elif (
            row[
                "median_new_minus_old_excess_return"
            ] > 0
        ):
            diagnosis = "RETURN_HELPFUL_RISK_WORSE"
        elif (
            row[
                "median_new_minus_old_relative_maxdrawdown"
            ] >= 0
        ):
            diagnosis = "RETURN_WORSE_RISK_HELPFUL"
        else:
            diagnosis = "RETURN_AND_RISK_HARMFUL"
        row["event_diagnosis"] = diagnosis
        rows.append(row)
    result = pd.DataFrame(rows)
    duplicate_metadata = [
        column for column in ("event_date", "interval_end")
        if column in ledger.columns
    ]
    ledger_extra = ledger.drop(columns=duplicate_metadata)
    return result.merge(
        ledger_extra,
        on=["event_id", "horizon"],
        how="left",
        validate="one_to_one",
    )


def _plateau_event_summary(
    family_rows: pd.DataFrame,
) -> pd.DataFrame:
    numeric = [
        "new_minus_old_terminal_value_eur",
        "new_minus_old_excess_return",
        "new_minus_old_relative_maxdrawdown",
        "new_minus_old_trade_count",
        "new_minus_old_cost_eur",
    ]
    keys = [
        "event_id",
        "horizon",
        "event_date",
        "event_year",
        "structural_plateau_id",
    ]
    grouped = family_rows.groupby(
        keys, as_index=False
    )[numeric].median()
    counts = family_rows.groupby(
        keys, as_index=False
    )["family_id"].nunique().rename(
        columns={"family_id": "family_count"}
    )
    return grouped.merge(
        counts, on=keys
    )


def _aggregate_summary(
    event_summary: pd.DataFrame,
    key: str,
) -> pd.DataFrame:
    rows = []
    for value, frame in event_summary.groupby(
        key, sort=True
    ):
        excess = frame[
            "median_new_minus_old_excess_return"
        ].to_numpy(float)
        dd = frame[
            "median_new_minus_old_relative_maxdrawdown"
        ].to_numpy(float)
        rows.append({
            key: value,
            "event_count": int(len(frame)),
            "median_event_excess_return_delta":
                float(np.median(excess)),
            "positive_event_fraction":
                float(np.mean(excess > 0)),
            "median_event_relative_maxdrawdown_delta":
                float(np.median(dd)),
            "risk_not_worse_event_fraction":
                float(np.mean(dd >= 0)),
            "return_and_risk_helpful_fraction":
                float(
                    np.mean(
                        frame[
                            "event_diagnosis"
                        ].eq(
                            "RETURN_AND_RISK_HELPFUL"
                        )
                    )
                ),
            "return_and_risk_harmful_fraction":
                float(
                    np.mean(
                        frame[
                            "event_diagnosis"
                        ].eq(
                            "RETURN_AND_RISK_HARMFUL"
                        )
                    )
                ),
        })
    return pd.DataFrame(rows)


def _spearman_associations(
    event_summary: pd.DataFrame,
) -> dict[str, Any]:
    target = "median_new_minus_old_excess_return"
    result: dict[str, Any] = {
        "target": target,
        "authority":
            "DIAGNOSTIC_ONLY_NO_ACTIVATION_GATE",
        "ex_ante_features": {},
        "post_event_diagnostic_features": {},
    }
    for field in EX_ANTE_FEATURE_FIELDS:
        pair = (
            event_summary[
                [field, target]
            ]
            .replace(
                [np.inf, -np.inf], np.nan
            )
            .dropna()
        )
        value = (
            pair[field].corr(
                pair[target], method="spearman"
            )
            if len(pair) >= 3 else np.nan
        )
        result["ex_ante_features"][field] = {
            "spearman":
                None if pd.isna(value)
                else float(value),
            "n": int(len(pair)),
            "activation_safe": True,
        }
    for field in POST_EVENT_DIAGNOSTIC_FIELDS:
        pair = (
            event_summary[
                [field, target]
            ]
            .replace(
                [np.inf, -np.inf], np.nan
            )
            .dropna()
        )
        value = (
            pair[field].corr(
                pair[target], method="spearman"
            )
            if len(pair) >= 3 else np.nan
        )
        result[
            "post_event_diagnostic_features"
        ][field] = {
            "spearman":
                None if pd.isna(value)
                else float(value),
            "n": int(len(pair)),
            "activation_safe": False,
        }
    return result


def _report(summary: dict) -> str:
    harmful_h = sorted(
        summary["horizon_summary"],
        key=lambda row:
            row["median_event_excess_return_delta"],
    )[:5]
    helpful_h = sorted(
        summary["horizon_summary"],
        key=lambda row:
            row["median_event_excess_return_delta"],
        reverse=True,
    )[:5]
    weak_years = sorted(
        summary["year_summary"],
        key=lambda row:
            row["median_event_excess_return_delta"],
    )[:3]
    lines = [
        "# Dynamic-QBD Fold-Event Counterfactual",
        "",
        f"Status: **{summary['status']}**",
        f"Authority: {AUTHORITY}; final holdout remains closed.",
        "",
        "## Contract",
        "",
        "- NEW and OLD branch from the exact same pre-event portfolio state.",
        "- NEW follows the actual Fold-Clock generation and becomes canonical for the next event.",
        "- OLD keeps the prior model and prior calibration only for the current Fold interval.",
        "- Entry recipe is identical; only model generation/calibration age differs.",
        "- FIXED exits only; no learned-exit coverage confound.",
        "- No activation threshold or selector is fitted by this suite.",
        "",
        "## Aggregate",
        "",
        f"- Non-initial Fold events: {summary['event_count']}.",
        f"- Fixed families: {summary['fixed_family_count']}.",
        f"- Event-family rows: {summary['event_family_rows']}.",
        f"- Positive median event fraction: {summary['positive_event_fraction']:.2%}.",
        f"- Return-and-risk helpful fraction: {summary['return_and_risk_helpful_fraction']:.2%}.",
        f"- Return-and-risk harmful fraction: {summary['return_and_risk_harmful_fraction']:.2%}.",
        "",
        "## Most harmful horizons",
        "",
    ]
    for row in harmful_h:
        lines.append(
            f"- H{int(row['horizon']):02d}: "
            f"{row['median_event_excess_return_delta']:+.4%}; "
            f"positive events "
            f"{row['positive_event_fraction']:.1%}."
        )
    lines.extend([
        "",
        "## Most helpful horizons",
        "",
    ])
    for row in helpful_h:
        lines.append(
            f"- H{int(row['horizon']):02d}: "
            f"{row['median_event_excess_return_delta']:+.4%}; "
            f"positive events "
            f"{row['positive_event_fraction']:.1%}."
        )
    lines.extend([
        "",
        "## Weakest event years",
        "",
    ])
    for row in weak_years:
        lines.append(
            f"- {int(row['event_year'])}: "
            f"{row['median_event_excess_return_delta']:+.4%}; "
            f"positive events "
            f"{row['positive_event_fraction']:.1%}."
        )
    lines.extend([
        "",
        "Post-event signal-shift fields use future interval behavior and are diagnostic only. They are forbidden as activation inputs.",
        "",
        "No result in this report grants promotion or capital authority.",
    ])
    return "\n".join(lines) + "\n"


def run_counterfactual(
    *,
    source_surface_validation_root: str | Path,
    signal_panel: str | Path,
    output_root: str | Path,
    code_commit: str,
    initial: float = 10000.0,
    horizon_workers: int =
        DEFAULT_HORIZON_PROCESS_WORKERS,
    nwinfo_interval_seconds: float = 60.0,
    nwinfo_executable: str | Path | None = None,
) -> dict:
    source_root = Path(source_surface_validation_root)
    signal_panel = Path(signal_panel)
    output_root = Path(output_root)
    output_root.mkdir(
        parents=True, exist_ok=True
    )
    if not signal_panel.is_file():
        raise FileNotFoundError(signal_panel)

    budget = cpu_budget(
        .80,
        enforce_capacity_fraction=True,
    )
    worker_count = int(
        budget["target_logical_processors"]
    )
    resources = configure_cpu_peak(
        target_fraction=.80,
        process_workers=worker_count,
        native_threads_per_worker=1,
        memory_limit_gb=90.0,
        memory_soft_target_gb=84.0,
        enforce_capacity_fraction=True,
    )
    selected = [
        int(x)
        for x in resources.get(
            "selected_logical_processors", []
        )
    ]
    horizon_process_count = max(
        1,
        min(
            int(horizon_workers),
            len(selected) if selected
            else worker_count,
            30,
        ),
    )

    source_summary, source_audit = (
        _source_contract(
            source_root, signal_panel
        )
    )
    development_start = date.fromisoformat(
        source_summary["development_start"]
    )
    development_end = date.fromisoformat(
        source_summary["development_end"]
    )
    if not math.isclose(
        float(initial),
        float(source_summary["initial_value_eur"]),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "COUNTERFACTUAL_INITIAL_VALUE_MUST_MATCH_SOURCE_SURFACE_VALIDATION"
        )
    source_entry = pd.read_parquet(
        source_root / "entry-fit-audit.parquet"
    ).copy()
    source_entry["prediction_artifact_path"] = [
        str(
            _resolve_source_prediction(
                value, source_root
            )
        )
        for value in source_entry[
            "prediction_artifact_path"
        ].astype(str)
    ]
    if int(
        source_entry["horizon"].nunique()
    ) != 30:
        raise AssertionError(
            "COUNTERFACTUAL_SOURCE_ENTRY_AUDIT_HORIZONS_NOT_30"
        )

    feature_schema = (
        _feature_schema_fingerprint(
            signal_panel
        )
    )
    all_families = build_family_specs(
        feature_schema_sha256=feature_schema,
        include_learned_exit=True,
        score_quantile=float(
            source_summary["score_quantile"]
        ),
        top_fraction=float(
            source_summary["top_fraction"]
        ),
    )
    fixed_families = tuple(
        family
        for family in all_families
        if str(
            family.exit_policy.get(
                "family", "FIXED"
            )
        ) == "FIXED"
    )
    if len(fixed_families) != 2790:
        raise AssertionError(
            f"COUNTERFACTUAL_FIXED_FAMILY_COUNT:"
            f"{len(fixed_families)}"
        )
    family_by_h = {
        h: tuple(
            family
            for family in fixed_families
            if int(
                family.horizon_sessions
            ) == h
        )
        for h in range(1, 31)
    }

    prices_path = (
        source_root
        / "inputs"
        / "prices.parquet"
    )
    prices_index = pd.read_parquet(
        prices_path,
        columns=["date", "ticker"],
    )
    market_dates = tuple(
        pd.to_datetime(
            prices_index.loc[
                prices_index[
                    "ticker"
                ].eq("URTH"),
                "date",
            ]
        )
        .drop_duplicates()
        .sort_values()
        .tolist()
    )
    del prices_index

    semantic_contract = {
        "schema_version": SCHEMA_VERSION,
        "authority": AUTHORITY,
        "code_commit": code_commit,
        "source_surface_validation_schema":
            source_summary["schema_version"],
        "source_semantic_run_contract_hash":
            source_summary[
                "semantic_run_contract_hash"
            ],
        "source_code_commit":
            source_summary["code_commit"],
        "source_summary_sha256":
            _sha256_file(
                source_root / "summary.json"
            ),
        "signal_panel_sha256":
            _sha256_file(signal_panel),
        "development_start":
            development_start,
        "development_end":
            development_end,
        "holdout_contract":
            HOLDOUT_CONTRACT,
        "final_holdout_opened": False,
        "promotion_allowed": False,
        "fixed_family_count":
            len(fixed_families),
        "counterfactual": {
            "old": COUNTERFACTUAL_OLD,
            "new": COUNTERFACTUAL_NEW,
            "new_becomes_next_event_canonical_state":
                True,
            "same_pre_event_portfolio_state":
                True,
            "same_frozen_recipe": True,
            "new_model_available_after_event_cutoff":
                True,
            "learned_exit_included": False,
        },
        "diagnostics": {
            "ex_ante_fields":
                list(EX_ANTE_FEATURE_FIELDS),
            "post_event_fields":
                list(
                    POST_EVENT_DIAGNOSTIC_FIELDS
                ),
            "activation_gate_optimized":
                False,
        },
        "execution_semantics": {
            "scheduler":
                "HORIZON_PROCESS_POOL_V3_ONE_CPU_PER_HORIZON_TASK",
            "horizon_process_workers":
                horizon_process_count,
            "family_workers_per_horizon": 1,
            "cpu_target_fraction": .80,
            "native_threads_per_worker": 1,
            "memory_limit_gib": 90.0,
            "memory_soft_target_gib": 84.0,
        },
    }
    run_hash = stable_hash(
        semantic_contract
    )
    run_contract_path = (
        output_root / "run-contract.json"
    )
    if run_contract_path.is_file():
        prior = json.loads(
            run_contract_path.read_text(
                encoding="utf-8"
            )
        )
        if (
            prior.get(
                "semantic_run_contract_hash"
            ) != run_hash
        ):
            raise ValueError(
                "COUNTERFACTUAL_RESTART_RUN_CONTRACT_MISMATCH"
            )
    else:
        _write_json(
            run_contract_path,
            {
                **semantic_contract,
                "code_commit": code_commit,
                "semantic_run_contract_hash":
                    run_hash,
            },
        )

    sampler = NWInfoSampler(
        output_root,
        interval_seconds=
            nwinfo_interval_seconds,
        executable=nwinfo_executable,
    ).start()
    try:
        all_family: list[
            pd.DataFrame
        ] = []
        all_ledger: list[
            pd.DataFrame
        ] = []
        pending = []
        for h in range(1, 31):
            checkpoint = (
                _load_horizon_checkpoint(
                    output_root,
                    h,
                    run_hash,
                )
            )
            if checkpoint is not None:
                family, ledger = checkpoint
                all_family.append(family)
                all_ledger.append(ledger)
                print(
                    f"[fold-event] checkpoint "
                    f"H{h:02d} hit",
                    flush=True,
                )
            else:
                pending.append(h)

        if pending:
            context = mp.get_context(
                "spawn"
            )
            with ProcessPoolExecutor(
                max_workers=
                    horizon_process_count,
                mp_context=context,
            ) as pool:
                futures = {
                    pool.submit(
                        _run_horizon,
                        horizon=h,
                        horizon_audit=
                            source_entry.loc[
                                source_entry[
                                    "horizon"
                                ].eq(h)
                            ].copy(),
                        families=family_by_h[h],
                        source_root=source_root,
                        signal_panel=signal_panel,
                        output_root=output_root,
                        prices_path=prices_path,
                        market_dates=market_dates,
                        development_start=
                            development_start,
                        development_end=
                            development_end,
                        initial=initial,
                        run_hash=run_hash,
                        logical_processor=(
                            selected[
                                index
                                % len(selected)
                            ]
                            if selected else None
                        ),
                    ): h
                    for index, h
                    in enumerate(pending)
                }
                for future in as_completed(
                    futures
                ):
                    result = future.result()
                    all_family.append(
                        result["family"]
                    )
                    all_ledger.append(
                        result["ledger"]
                    )
                    print(
                        f"[fold-event] "
                        f"H{int(result['horizon']):02d} "
                        f"complete cpu="
                        f"{result.get('logical_processor')}",
                        flush=True,
                    )

        family_rows = pd.concat(
            all_family,
            ignore_index=True,
        )
        ledger = pd.concat(
            all_ledger,
            ignore_index=True,
        )
        if (
            family_rows.empty
            or ledger.empty
        ):
            raise AssertionError(
                "COUNTERFACTUAL_NO_EVENT_RESULTS"
            )

        event_summary = _event_summary(
            family_rows, ledger
        )
        plateau_summary = (
            _plateau_event_summary(
                family_rows
            )
        )
        horizon_summary = (
            _aggregate_summary(
                event_summary, "horizon"
            )
        )
        year_summary = _aggregate_summary(
            event_summary, "event_year"
        )
        associations = (
            _spearman_associations(
                event_summary
            )
        )
        contract_audit = (
            _contract_audit(
                source_summary=
                    source_summary,
                source_entry_audit=
                    source_entry,
                event_ledger=ledger,
                family_rows=family_rows,
                fixed_family_count=
                    len(fixed_families),
            )
        )

        family_rows.to_parquet(
            output_root
            / "event-family-counterfactual.parquet",
            index=False,
        )
        ledger.to_csv(
            output_root
            / "event-ledger.csv",
            index=False,
        )
        event_summary.to_csv(
            output_root
            / "event-summary.csv",
            index=False,
        )
        plateau_summary.to_csv(
            output_root
            / "event-plateau-summary.csv",
            index=False,
        )
        horizon_summary.to_csv(
            output_root
            / "horizon-summary.csv",
            index=False,
        )
        year_summary.to_csv(
            output_root
            / "year-summary.csv",
            index=False,
        )
        risk_problem = (
            event_summary.loc[
                event_summary[
                    "event_diagnosis"
                ].isin([
                    "RETURN_AND_RISK_HARMFUL",
                    "RETURN_HELPFUL_RISK_WORSE",
                ])
            ].copy()
        )
        risk_problem.to_csv(
            output_root
            / "risk-problem-events.csv",
            index=False,
        )
        _write_json(
            output_root
            / "feature-associations.json",
            associations,
        )
        _write_json(
            output_root
            / "contract-audit.json",
            contract_audit,
        )
        _write_json(
            output_root
            / "source-contract-audit.json",
            source_audit,
        )

        summary = {
            "schema_version":
                SCHEMA_VERSION,
            "status": "COMPLETE",
            "authority": AUTHORITY,
            "code_commit": code_commit,
            "semantic_run_contract_hash":
                run_hash,
            "source_surface_validation_code_commit":
                source_summary[
                    "code_commit"
                ],
            "source_surface_validation_semantic_run_contract_hash":
                source_summary[
                    "semantic_run_contract_hash"
                ],
            "development_start":
                development_start.isoformat(),
            "development_end":
                development_end.isoformat(),
            "holdout_contract":
                HOLDOUT_CONTRACT,
            "final_holdout_opened":
                False,
            "promotion_allowed": False,
            "activation_gate_optimized":
                False,
            "learned_exit_included":
                False,
            "fixed_family_count":
                int(
                    family_rows[
                        "family_id"
                    ].nunique()
                ),
            "event_count":
                int(
                    event_summary[
                        "event_id"
                    ].nunique()
                ),
            "event_family_rows":
                int(len(family_rows)),
            "positive_event_fraction":
                float(
                    np.mean(
                        event_summary[
                            "median_new_minus_old_excess_return"
                        ].to_numpy(float)
                        > 0
                    )
                ),
            "return_and_risk_helpful_fraction":
                float(
                    np.mean(
                        event_summary[
                            "event_diagnosis"
                        ].eq(
                            "RETURN_AND_RISK_HELPFUL"
                        )
                    )
                ),
            "return_and_risk_harmful_fraction":
                float(
                    np.mean(
                        event_summary[
                            "event_diagnosis"
                        ].eq(
                            "RETURN_AND_RISK_HARMFUL"
                        )
                    )
                ),
            "median_event_excess_return_delta":
                float(
                    event_summary[
                        "median_new_minus_old_excess_return"
                    ].median()
                ),
            "median_event_relative_maxdrawdown_delta":
                float(
                    event_summary[
                        "median_new_minus_old_relative_maxdrawdown"
                    ].median()
                ),
            "horizon_summary":
                horizon_summary.to_dict(
                    orient="records"
                ),
            "year_summary":
                year_summary.to_dict(
                    orient="records"
                ),
            "feature_associations":
                associations,
            "execution_resources": {
                **active_cpu_contract(),
                "horizon_process_workers":
                    horizon_process_count,
                "family_workers_per_horizon":
                    1,
            },
            "contract_audit":
                contract_audit,
        }
        _write_json(
            output_root / "summary.json",
            summary,
        )
        (
            output_root / "REPORT.md"
        ).write_text(
            _report(summary),
            encoding="utf-8",
        )
        _write_json(
            output_root / "manifest.json",
            {
                "schema_version":
                    SCHEMA_VERSION,
                "status": "COMPLETE",
                "authority": AUTHORITY,
                "code_commit": code_commit,
                "compact_remote_results": [
                    "summary.json",
                    "REPORT.md",
                    "contract-audit.json",
                    "source-contract-audit.json",
                    "event-ledger.csv",
                    "event-summary.csv",
                    "event-plateau-summary.csv",
                    "horizon-summary.csv",
                    "year-summary.csv",
                    "risk-problem-events.csv",
                    "feature-associations.json",
                    "run-contract.json",
                    "nwinfo-summary.json",
                ],
                "large_local_results": [
                    "event-family-counterfactual.parquet",
                    "old-counterfactual-signals/",
                    "checkpoints/",
                    "nwinfo-sensors.jsonl",
                ],
            },
        )
        return summary
    finally:
        sampler.stop()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__
    )
    parser.add_argument(
        "--source-surface-validation-root",
        required=True,
    )
    parser.add_argument(
        "--signal-panel",
        required=True,
    )
    parser.add_argument(
        "--output-root",
        required=True,
    )
    parser.add_argument(
        "--code-commit",
        required=True,
    )
    parser.add_argument(
        "--initial",
        type=float,
        default=10000.0,
    )
    parser.add_argument(
        "--horizon-workers",
        type=int,
        default=
            DEFAULT_HORIZON_PROCESS_WORKERS,
    )
    parser.add_argument(
        "--nwinfo-interval-seconds",
        type=float,
        default=60.0,
    )
    parser.add_argument(
        "--nwinfo-executable",
        help=(
            "Exact nwinfo.exe path; "
            "alternatively set NWINFO_EXE."
        ),
    )
    args = parser.parse_args()
    summary = run_counterfactual(
        source_surface_validation_root=
            args.source_surface_validation_root,
        signal_panel=args.signal_panel,
        output_root=args.output_root,
        code_commit=args.code_commit,
        initial=args.initial,
        horizon_workers=
            args.horizon_workers,
        nwinfo_interval_seconds=
            args.nwinfo_interval_seconds,
        nwinfo_executable=
            args.nwinfo_executable,
    )
    print(
        json.dumps(
            {
                "status":
                    summary["status"],
                "event_count":
                    summary["event_count"],
                "fixed_family_count":
                    summary[
                        "fixed_family_count"
                    ],
                "median_event_excess_return_delta":
                    summary[
                        "median_event_excess_return_delta"
                    ],
                "execution_resources":
                    summary[
                        "execution_resources"
                    ],
            },
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
