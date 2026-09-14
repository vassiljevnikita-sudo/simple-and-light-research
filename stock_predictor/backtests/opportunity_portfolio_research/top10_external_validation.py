from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from . import portfolio_policy_search as search
from .cost_contracts import CostModel
from .tax_contracts import TaxConfig
from .portfolio_policy_contracts import policy_dict
from .portfolio_research_inputs import load_predictions, load_price_panel
from .learned_exit_qbd_profit import ProfitTaxConfig
from .learned_exit_qbd_provider import LearnedExitProvider
from .learned_exit_qbd_replay import configure_replay, replay as qbd_replay
from .learned_exit_qbd_suite import _aggregate

CONTRACT_ID = "TOP10_EXTERNAL_VALIDATION_FROZEN_ENTRY_V2"
RAW_EXPECTED_MIN_UTC = pd.Timestamp("2016-01-01T00:00:00Z")
RAW_EXPECTED_MAX_UTC = pd.Timestamp("2026-07-24T23:59:00Z")
BACKWARD_END = pd.Timestamp("2020-08-07")
KNOWN_START = pd.Timestamp("2020-08-10")
KNOWN_END = pd.Timestamp("2023-08-10")
FORWARD_START = pd.Timestamp("2023-08-11")
FORWARD_END = pd.Timestamp("2026-07-24")
INITIAL_CAPITAL_EUR = 10_000.0
KNOWN_TOLERANCE = 5e-4


@dataclass(frozen=True)
class FrozenModel:
    rank: int
    mode: str
    h: int
    d: int
    n: int
    known_median_cagr_excess: float

    @property
    def model_id(self) -> str:
        mode = "L" if self.mode == "LEARNED_EXIT" else "F"
        return f"R{self.rank:02d}_{mode}_H{self.h:02d}_D{self.d:02d}_N{self.n}"


TOP10: tuple[FrozenModel, ...] = (
    FrozenModel(1, "LEARNED_EXIT", 11, 3, 1, 0.9761),
    FrozenModel(2, "FIXED", 24, 5, 1, 0.5038),
    FrozenModel(3, "LEARNED_EXIT", 28, 21, 1, 0.3069),
    FrozenModel(4, "LEARNED_EXIT", 28, 21, 5, 0.3069),
    FrozenModel(5, "LEARNED_EXIT", 28, 21, 4, 0.3069),
    FrozenModel(6, "LEARNED_EXIT", 28, 21, 6, 0.3069),
    FrozenModel(7, "LEARNED_EXIT", 28, 21, 2, 0.3069),
    FrozenModel(8, "LEARNED_EXIT", 28, 21, 3, 0.3069),
    FrozenModel(9, "LEARNED_EXIT", 24, 21, 5, 0.0899),
    FrozenModel(10, "LEARNED_EXIT", 24, 21, 6, 0.0899),
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    tmp.replace(path)


def _parse_raw_timestamp(row: dict) -> pd.Timestamp | None:
    for key in ("timestamp_utc", "timestamp", "t", "time"):
        if key not in row or row[key] in (None, ""):
            continue
        try:
            ts = pd.Timestamp(row[key])
        except Exception:
            continue
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        return ts
    return None


def audit_raw_minute_coverage(raw_root: Path) -> dict:
    """Read record timestamps; filenames are never used as coverage evidence."""
    files = sorted(raw_root.rglob("*.jsonl.gz"))
    if not files:
        raise RuntimeError(f"NO_ALPACA_MINUTE_GZ:{raw_root}")
    observed_min: pd.Timestamp | None = None
    observed_max: pd.Timestamp | None = None
    rows = parsed = invalid_json = missing_timestamp = 0
    nonempty_files = 0
    for path in files:
        file_parsed = 0
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if not line.strip():
                    continue
                rows += 1
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    invalid_json += 1
                    continue
                if not isinstance(payload, dict):
                    missing_timestamp += 1
                    continue
                ts = _parse_raw_timestamp(payload)
                if ts is None:
                    missing_timestamp += 1
                    continue
                parsed += 1
                file_parsed += 1
                observed_min = ts if observed_min is None or ts < observed_min else observed_min
                observed_max = ts if observed_max is None or ts > observed_max else observed_max
        if file_parsed:
            nonempty_files += 1
    if observed_min is None or observed_max is None:
        raise RuntimeError("RAW_MINUTE_NO_PARSEABLE_TIMESTAMPS")
    exact = observed_min == RAW_EXPECTED_MIN_UTC and observed_max == RAW_EXPECTED_MAX_UTC
    audit = {
        "contract_id": CONTRACT_ID,
        "raw_root": str(raw_root),
        "files_seen": len(files),
        "nonempty_files": nonempty_files,
        "rows_seen": rows,
        "timestamps_parsed": parsed,
        "invalid_json_rows": invalid_json,
        "missing_timestamp_rows": missing_timestamp,
        "observed_min_utc": observed_min.isoformat(),
        "observed_max_utc": observed_max.isoformat(),
        "expected_min_utc": RAW_EXPECTED_MIN_UTC.isoformat(),
        "expected_max_utc": RAW_EXPECTED_MAX_UTC.isoformat(),
        "coverage_exact": exact,
        "coverage_derived_from_payloads": True,
        "filename_dates_are_not_coverage_evidence": True,
    }
    if not exact:
        raise RuntimeError(
            "RAW_MINUTE_COVERAGE_MISMATCH:"
            f"observed={observed_min.isoformat()}..{observed_max.isoformat()} "
            f"expected={RAW_EXPECTED_MIN_UTC.isoformat()}..{RAW_EXPECTED_MAX_UTC.isoformat()}"
        )
    return audit


def audit_daily_store_coverage(daily_root: Path, tickers: set[str]) -> dict:
    """Verify actual local daily-bar coverage before any external replay.

    The minute audit establishes raw availability.  This separate gate prevents
    an apparently complete raw archive from masking a truncated derived daily
    store.  IPO-era ticker gaps remain visible in the per-ticker audit, while
    the benchmark itself must span both external endpoints.
    """
    prices, loader_audit = load_price_panel(daily_root, tickers)
    by_ticker = (
        prices.groupby("ticker", as_index=False)["date"]
        .agg(first_date="min", last_date="max", rows="size")
        .sort_values("ticker")
    )
    if prices.empty:
        raise RuntimeError("DAILY_STORE_EMPTY")
    urth = by_ticker.loc[by_ticker["ticker"].eq("URTH")]
    if len(urth) != 1:
        raise RuntimeError("DAILY_STORE_URTH_MISSING")
    first = pd.Timestamp(prices["date"].min()).normalize()
    last = pd.Timestamp(prices["date"].max()).normalize()
    urth_first = pd.Timestamp(urth.iloc[0]["first_date"]).normalize()
    urth_last = pd.Timestamp(urth.iloc[0]["last_date"]).normalize()
    required_first_session = pd.Timestamp("2016-01-04")
    required_last_session = pd.Timestamp("2026-07-24")
    exact_span = first <= required_first_session and last >= required_last_session
    benchmark_span = urth_first <= required_first_session and urth_last >= required_last_session
    audit = {
        "daily_root": str(daily_root),
        "coverage_derived_from_daily_parquet_values": True,
        "store_first_session": str(first.date()),
        "store_last_session": str(last.date()),
        "urth_first_session": str(urth_first.date()),
        "urth_last_session": str(urth_last.date()),
        "required_first_session": str(required_first_session.date()),
        "required_last_session": str(required_last_session.date()),
        "store_span_sufficient": exact_span,
        "benchmark_span_sufficient": benchmark_span,
        "ticker_coverage": [
            {
                "ticker": str(row.ticker),
                "first_date": str(pd.Timestamp(row.first_date).date()),
                "last_date": str(pd.Timestamp(row.last_date).date()),
                "rows": int(row.rows),
            }
            for row in by_ticker.itertuples(index=False)
        ],
        "loader_audit": loader_audit,
    }
    if not exact_span or not benchmark_span:
        raise RuntimeError(
            "DAILY_STORE_COVERAGE_MISMATCH:"
            f"store={first.date()}..{last.date()};"
            f"urth={urth_first.date()}..{urth_last.date()}"
        )
    return audit


def _known_cell_path(root: Path, model: FrozenModel) -> Path:
    return root / "cells" / model.mode.lower() / f"H{model.h:02d}_D{model.d:02d}_N{model.n}.json"


def load_known_bridge(root: Path, model: FrozenModel) -> dict:
    path = _known_cell_path(root, model)
    if not path.is_file():
        raise RuntimeError(f"KNOWN_CELL_MISSING:{model.model_id}:{path}")
    raw = path.read_bytes()
    payload = json.loads(raw)
    if payload.get("status") != "COMPLETE":
        raise RuntimeError(f"KNOWN_CELL_NOT_COMPLETE:{model.model_id}")
    observed = (
        str(payload.get("mode")),
        int(payload.get("prediction_horizon", -1)),
        int(payload.get("holding_days", -1)),
        int(payload.get("max_names", -1)),
    )
    expected = (model.mode, model.h, model.d, model.n)
    if observed != expected:
        raise RuntimeError(f"KNOWN_CELL_IDENTITY_MISMATCH:{model.model_id}:{observed}!={expected}")
    summary = payload.get("profit_summary", {})
    median_x = float(summary.get("median_active_cagr_excess", float("nan")))
    if not math.isfinite(median_x) or abs(median_x - model.known_median_cagr_excess) > KNOWN_TOLERANCE:
        raise RuntimeError(
            f"KNOWN_MEDIAN_MISMATCH:{model.model_id}:"
            f"observed={median_x:.8f}:expected={model.known_median_cagr_excess:.8f}"
        )
    terminal = float(summary["fold_compounded_terminal_value"])
    bench_terminal = float(summary["fold_compounded_urth_terminal_value"])
    initial = float(payload.get("initial_capital", INITIAL_CAPITAL_EUR))
    if initial <= 0:
        initial = INITIAL_CAPITAL_EUR
    outer = list(payload.get("outer_rows", []))
    if outer:
        starts = pd.to_datetime([x["fold_start"] for x in outer]).date
        ends = pd.to_datetime([x["fold_end"] for x in outer]).date
        if min(starts) < KNOWN_START.date() or max(ends) > KNOWN_END.date():
            raise RuntimeError(f"KNOWN_BRIDGE_OUTSIDE_LOCKED_WINDOW:{model.model_id}")
    raw_policy = payload.get("final_policy")
    raw_threshold = payload.get("final_threshold")
    if not isinstance(raw_policy, dict) or raw_threshold is None:
        raise RuntimeError(f"KNOWN_FROZEN_ENTRY_POLICY_MISSING:{model.model_id}")
    policy = search._policy_from_dict(raw_policy)
    observed_policy_identity = (
        policy.horizon,
        policy.holding_days,
        policy.max_names,
        policy.exit_family,
    )
    expected_policy_identity = (model.h, model.d, model.n, model.mode)
    if observed_policy_identity != expected_policy_identity:
        raise RuntimeError(
            f"KNOWN_FROZEN_ENTRY_POLICY_IDENTITY_MISMATCH:{model.model_id}:"
            f"{observed_policy_identity}!={expected_policy_identity}"
        )
    threshold = float(raw_threshold)
    if not math.isfinite(threshold):
        raise RuntimeError(f"KNOWN_FROZEN_THRESHOLD_INVALID:{model.model_id}")
    return {
        "model_id": model.model_id,
        "source_path": str(path),
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "median_active_cagr_excess": median_x,
        "growth_factor": terminal / initial,
        "benchmark_growth_factor": bench_terminal / initial,
        "active_folds": int(summary.get("active_folds", 0)),
        "outer_folds": int(summary.get("outer_folds", 0)),
        "terminal_value_eur": terminal,
        "benchmark_terminal_value_eur": bench_terminal,
        "frozen_entry_policy": policy_dict(policy),
        "frozen_resolved_threshold": threshold,
        "entry_policy_usage": "FROZEN_FROM_KNOWN_OOS_FINAL_FIT_NO_EXTERNAL_REOPTIMIZATION",
        "immutable": True,
    }


def _assert_prediction_window(pred: pd.DataFrame, *, segment: str) -> dict:
    if pred.empty:
        raise RuntimeError(f"{segment}_PREDICTIONS_EMPTY")
    dates = pd.to_datetime(pred["decision_date"])
    if getattr(dates.dt, "tz", None) is not None:
        dates = dates.dt.tz_convert(None)
    lo, hi = dates.min(), dates.max()
    if segment == "BACKWARD":
        if hi.date() > BACKWARD_END.date():
            raise RuntimeError(f"BACKWARD_PREDICTION_LEAKS_KNOWN:{hi}")
    elif segment == "FORWARD":
        if hi.date() > FORWARD_END.date():
            raise RuntimeError(f"FORWARD_PREDICTION_AFTER_RAW_MAX:{hi}")
        if hi.date() < FORWARD_START.date():
            raise RuntimeError(f"FORWARD_PREDICTION_MISSING_TARGET_PERIOD:{hi}")
    else:
        raise ValueError(segment)
    return {"rows": len(pred), "min_decision_date": str(lo.date()), "max_decision_date": str(hi.date())}


def _filter_outer_segment(rows: Iterable[dict], *, segment: str) -> list[dict]:
    kept: list[dict] = []
    for raw in rows:
        row = dict(raw)
        start = pd.Timestamp(row["fold_start"])
        end = pd.Timestamp(row["fold_end"])
        cal = pd.Timestamp(row["calibration_end"])
        if not cal < start:
            raise RuntimeError(f"NON_CAUSAL_PORTFOLIO_FOLD:{segment}:{row.get('fold_id')}:{cal}>={start}")
        if segment == "BACKWARD":
            if start.date() >= KNOWN_START.date():
                continue
            if end.date() > BACKWARD_END.date():
                raise RuntimeError(f"BACKWARD_FOLD_OVERLAPS_KNOWN:{row.get('fold_id')}:{start}..{end}")
            kept.append(row)
        elif segment == "FORWARD":
            if end.date() < FORWARD_START.date():
                continue
            if start.date() < FORWARD_START.date():
                raise RuntimeError(f"FORWARD_FOLD_STRADDLES_KNOWN:{row.get('fold_id')}:{start}..{end}")
            if end.date() > FORWARD_END.date():
                raise RuntimeError(f"FORWARD_FOLD_AFTER_RAW_MAX:{row.get('fold_id')}:{end}")
            kept.append(row)
        else:
            raise ValueError(segment)
    if not kept:
        raise RuntimeError(f"NO_EXTERNAL_OUTER_FOLDS:{segment}")
    return kept


def _segment_summary(rows: list[dict], initial: float) -> dict:
    summary = _aggregate(rows, initial=initial)
    summary["growth_factor"] = float(summary["fold_compounded_terminal_value"]) / initial
    summary["benchmark_growth_factor"] = float(summary["fold_compounded_urth_terminal_value"]) / initial
    summary["first_fold_start"] = min(str(x["fold_start"]) for x in rows)
    summary["last_fold_end"] = max(str(x["fold_end"]) for x in rows)
    return summary


def _fmt_pct(value: object) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    return f"{number * 100.0:.2f}%"


def _report(summary: dict) -> str:
    """Write a compact, decision-oriented report without re-ranking the Top 10."""
    lines = [
        "# Frozen Top-10 External Validation",
        "",
        f"Status: `{summary['status']}`",
        "",
        "The ten identities, their entry policies, costs, taxes and Known-OOS results were frozen before this run. "
        "Backward and Forward results did not select a new policy.",
        "",
        "## Data and validity gates",
        "",
        f"- Raw-minute coverage: {summary['raw_coverage']['observed_min_utc']} to {summary['raw_coverage']['observed_max_utc']}",
        f"- Daily store: {summary['daily_store_coverage']['store_first_session']} to {summary['daily_store_coverage']['store_last_session']}",
        f"- URTH: {summary['daily_store_coverage']['urth_first_session']} to {summary['daily_store_coverage']['urth_last_session']}",
        "- Final holdout: closed",
        "- Point-in-time universe: not verified; survivorship bias remains a promotion block.",
        "",
        "## CAGR-excess retention",
        "",
        "| Rank | Model | Back CAGR-X | Known CAGR-X | Forward CAGR-X | Back retention | Forward retention | Status |",
        "|---:|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in sorted(summary["results"], key=lambda item: int(item["rank_2020_2023"])):
        label = f"{row['mode']} H{int(row['h']):02d}/D{int(row['d']):02d}/N{int(row['n'])}"
        lines.append(
            f"| {row['rank_2020_2023']} | {label} | {_fmt_pct(row['back_median_cagr_excess'])} | "
            f"{_fmt_pct(row['known_median_cagr_excess'])} | {_fmt_pct(row['forward_median_cagr_excess'])} | "
            f"{_fmt_pct(row['back_retention'])} | {_fmt_pct(row['forward_retention'])} | {row['stability_status']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation contract",
            "",
            "- `STABLE`: both external medians positive and both retentions at least 50%.",
            "- `DECAY`: both positive and weaker retention at least 25%, but below 50%.",
            "- `COLLAPSE`: sign reversal or weaker retention below 25%.",
            "- `NO_ACTIVITY`: no active fold in an external segment.",
            "",
            "Full-period wealth is chained multiplicatively across Backward, Known and Forward windows. "
            "Fold tax ledgers are independent research approximations, not a single continuous real-depot tax simulation.",
            "",
        ]
    )
    return "\n".join(lines)


def classify(known: float, back: dict, forward: dict) -> dict:
    if known <= 0:
        return {"status": "UNCLASSIFIED", "back_retention": None, "forward_retention": None}
    if int(back.get("active_folds", 0)) == 0 or int(forward.get("active_folds", 0)) == 0:
        return {"status": "NO_ACTIVITY", "back_retention": None, "forward_retention": None}
    bx = float(back["median_active_cagr_excess"])
    fx = float(forward["median_active_cagr_excess"])
    br = bx / known
    fr = fx / known
    if bx > 0 and fx > 0 and min(br, fr) >= 0.50:
        status = "STABLE"
    elif bx > 0 and fx > 0 and min(br, fr) >= 0.25:
        status = "DECAY"
    else:
        status = "COLLAPSE"
    return {"status": status, "back_retention": br, "forward_retention": fr}


def _run_external_cell(
    pred: pd.DataFrame,
    prices: pd.DataFrame,
    provider: LearnedExitProvider,
    tax: ProfitTaxConfig,
    model: FrozenModel,
    known_bridge: dict,
    *,
    segment: str,
    initial: float,
) -> tuple[dict, list[dict]]:
    configure_replay(provider, tax)
    policy = search._policy_from_dict(dict(known_bridge["frozen_entry_policy"]))
    threshold = float(known_bridge["frozen_resolved_threshold"])
    hpred = pred.loc[pred["horizon"].eq(model.h)].copy()
    fold_data = search._prepare_fold_data(hpred)
    outer_rows: list[dict] = []
    # The policy was selected on the immutable Known window.  It is therefore
    # replayed unchanged in every external fold; no score/top-fraction/grid
    # selection may consume Backward or Forward performance.
    for index, fold in enumerate(fold_data):
        if index == 0:
            continue
        calibration_end = pd.Timestamp(fold_data[index - 1]["end"])
        if calibration_end >= pd.Timestamp(fold["start"]):
            raise RuntimeError(
                f"NON_CAUSAL_PORTFOLIO_FOLD:{segment}:{fold['fold_id']}:"
                f"{calibration_end}>={fold['start']}"
            )
        result = qbd_replay(
            fold["signals"],
            prices,
            policy,
            CostModel(20),
            TaxConfig(False),
            start=fold["start"],
            end=fold["end"],
            initial=initial,
            resolved_threshold=threshold,
            prepared_signals=fold["prepared"],
        )
        outer_rows.append(
            {
                "horizon": model.h,
                "fold_id": str(fold["fold_id"]),
                "fold_start": str(pd.Timestamp(fold["start"]).date()),
                "fold_end": str(pd.Timestamp(fold["end"]).date()),
                "calibration_end": str(calibration_end.date()),
                "policy_id": policy.policy_id,
                **policy_dict(policy),
                "threshold": threshold,
                "entry_policy_selection": "FROZEN_KNOWN_FINAL_FIT",
                **result["metrics"],
            }
        )
    rows = _filter_outer_segment(outer_rows, segment=segment)
    return _segment_summary(rows, initial), rows


def _load_external_inputs(v5_root: Path, learned_root: Path, daily_root: Path, segment: str):
    pred, pred_audit = load_predictions(v5_root)
    pred_window = _assert_prediction_window(pred, segment=segment)
    needed_h = {m.h for m in TOP10}
    missing = sorted(needed_h - set(int(x) for x in pred.horizon.unique()))
    if missing:
        raise RuntimeError(f"{segment}_V5_HORIZONS_MISSING:{missing}")
    tickers = set(pred.loc[pred.horizon.isin(needed_h), "ticker"].unique())
    prices, price_audit = load_price_panel(daily_root, tickers)
    provider = LearnedExitProvider(learned_root)
    return pred, prices, provider, {
        "prediction_audit": pred_audit,
        "prediction_window": pred_window,
        "price_audit": price_audit,
        "learned_exit_provider_audit": provider.audit.__dict__,
    }


def run(a: argparse.Namespace) -> dict:
    out = Path(a.output_root)
    out.mkdir(parents=True, exist_ok=True)
    raw_audit = audit_raw_minute_coverage(Path(a.raw_minute_root))
    _write_json(out / "raw_coverage_audit.json", raw_audit)
    _write_json(
        out / "frozen_top10.json",
        {"contract_id": CONTRACT_ID, "models": [asdict(m) | {"model_id": m.model_id} for m in TOP10]},
    )

    known = {m.model_id: load_known_bridge(Path(a.known_profit_root), m) for m in TOP10}
    _write_json(
        out / "known_bridge_manifest.json",
        {
            "contract_id": CONTRACT_ID,
            "locked_window": [str(KNOWN_START.date()), str(KNOWN_END.date())],
            "models": list(known.values()),
        },
    )

    back_pred, back_prices, back_provider, back_audit = _load_external_inputs(
        Path(a.backward_v5_predictions),
        Path(a.backward_learned_exit_predictions),
        Path(a.daily_store_root),
        "BACKWARD",
    )
    fwd_pred, fwd_prices, fwd_provider, fwd_audit = _load_external_inputs(
        Path(a.forward_v5_predictions),
        Path(a.forward_learned_exit_predictions),
        Path(a.daily_store_root),
        "FORWARD",
    )
    all_tickers = set(back_pred["ticker"].unique()) | set(fwd_pred["ticker"].unique())
    daily_audit = audit_daily_store_coverage(Path(a.daily_store_root), all_tickers)
    _write_json(out / "daily_store_coverage_audit.json", daily_audit)

    tax = ProfitTaxConfig(
        allowance_eur=a.tax_allowance_eur,
        church_tax_rate=a.church_tax_rate,
        benchmark_partial_exemption_rate=a.benchmark_partial_exemption,
    )
    rows: list[dict] = []
    outer_dump: dict[str, dict] = {}
    for i, model in enumerate(TOP10, 1):
        print(f"[top10-external] {i}/10 {model.model_id}", flush=True)
        k = known[model.model_id]
        bsum, brows = _run_external_cell(
            back_pred,
            back_prices,
            back_provider,
            tax,
            model,
            k,
            segment="BACKWARD",
            initial=a.initial_capital,
        )
        fsum, frows = _run_external_cell(
            fwd_pred,
            fwd_prices,
            fwd_provider,
            tax,
            model,
            k,
            segment="FORWARD",
            initial=a.initial_capital,
        )
        cls = classify(model.known_median_cagr_excess, bsum, fsum)
        strategy_g = bsum["growth_factor"] * k["growth_factor"] * fsum["growth_factor"]
        benchmark_g = (
            bsum["benchmark_growth_factor"]
            * k["benchmark_growth_factor"]
            * fsum["benchmark_growth_factor"]
        )
        row = {
                "rank_2020_2023": model.rank,
                "model_id": model.model_id,
                "mode": model.mode,
                "h": model.h,
                "d": model.d,
                "n": model.n,
                "back_median_cagr_excess": bsum["median_active_cagr_excess"],
                "known_median_cagr_excess": model.known_median_cagr_excess,
                "forward_median_cagr_excess": fsum["median_active_cagr_excess"],
                "back_retention": cls["back_retention"],
                "forward_retention": cls["forward_retention"],
                "stability_status": cls["status"],
                "back_q25_cagr_excess": bsum["q25_active_cagr_excess"],
                "forward_q25_cagr_excess": fsum["q25_active_cagr_excess"],
                "back_worst_cagr_excess": bsum["worst_active_cagr_excess"],
                "forward_worst_cagr_excess": fsum["worst_active_cagr_excess"],
                "back_active_folds": bsum["active_folds"],
                "back_outer_folds": bsum["outer_folds"],
                "forward_active_folds": fsum["active_folds"],
                "forward_outer_folds": fsum["outer_folds"],
                "back_positive_active_fold_fraction": bsum["positive_active_fold_fraction"],
                "forward_positive_active_fold_fraction": fsum["positive_active_fold_fraction"],
                "back_growth_factor": bsum["growth_factor"],
                "known_growth_factor": k["growth_factor"],
                "forward_growth_factor": fsum["growth_factor"],
                "full_strategy_growth_factor": strategy_g,
                "full_benchmark_growth_factor": benchmark_g,
                "full_terminal_value_eur": a.initial_capital * strategy_g,
                "full_benchmark_terminal_value_eur": a.initial_capital * benchmark_g,
                "full_wealth_excess_eur": a.initial_capital * (strategy_g - benchmark_g),
                "back_trades": bsum["total_trades"],
                "forward_trades": fsum["total_trades"],
                "back_mean_turnover": bsum["mean_turnover"],
                "forward_mean_turnover": fsum["mean_turnover"],
                "back_tax_paid_eur_fold_sum": bsum["total_tax_paid_eur"],
                "forward_tax_paid_eur_fold_sum": fsum["total_tax_paid_eur"],
                "known_source_sha256": k["source_sha256"],
                "frozen_entry_policy": json.dumps(k["frozen_entry_policy"], sort_keys=True),
                "frozen_resolved_threshold": k["frozen_resolved_threshold"],
        }
        rows.append(row)
        outer_dump[model.model_id] = {"backward": brows, "forward": frows}

    pd.DataFrame(rows).sort_values("rank_2020_2023").to_csv(out / "retention_matrix.csv", index=False)
    pd.DataFrame(rows).sort_values("full_terminal_value_eur", ascending=False).to_csv(
        out / "full_period_wealth.csv", index=False
    )
    _write_json(out / "external_outer_rows.json", outer_dump)
    summary = {
        "contract_id": CONTRACT_ID,
        "status": "COMPLETE",
        "raw_coverage": raw_audit,
        "daily_store_coverage": daily_audit,
        "backward_target_end": str(BACKWARD_END.date()),
        "known_bridge": [str(KNOWN_START.date()), str(KNOWN_END.date())],
        "forward_target": [str(FORWARD_START.date()), str(FORWARD_END.date())],
        "top10_frozen_before_external_results": True,
        "entry_policy_frozen_from_known_window": True,
        "external_entry_policy_optimization": "FORBIDDEN",
        "known_bridge_recomputed": False,
        "known_bridge_loaded_immutable": True,
        "wealth_is_multiplicative": True,
        "fold_tax_ledgers_are_independent": True,
        "promotion_eligible": False,
        "point_in_time_universe_verified": False,
        "survivorship_bias_promotion_block": True,
        "classification": {
            "STABLE": "both external medians > 0 and both retention ratios >= 0.50",
            "DECAY": "both external medians > 0 and minimum retention >= 0.25 but < 0.50",
            "COLLAPSE": "otherwise, including sign flip or retention < 0.25",
            "NO_ACTIVITY": "at least one external segment has zero active folds",
            "UNCLASSIFIED": "known median CAGR excess <= 0",
        },
        "backward_input_audit": back_audit,
        "forward_input_audit": fwd_audit,
        "tax_config": tax.__dict__,
        "results": rows,
    }
    _write_json(out / "evaluation_summary.json", summary)
    (out / "REPORT.md").write_text(_report(summary), encoding="utf-8")
    return summary


def self_test() -> None:
    known = 1.0
    back = {"active_folds": 2, "median_active_cagr_excess": 0.6}
    forward = {"active_folds": 2, "median_active_cagr_excess": 0.5}
    assert classify(known, back, forward)["status"] == "STABLE"
    assert classify(known, {**back, "median_active_cagr_excess": 0.49}, forward)["status"] == "DECAY"
    assert classify(known, {**back, "median_active_cagr_excess": 0.24}, forward)["status"] == "COLLAPSE"
    assert classify(known, {"active_folds": 0, "median_active_cagr_excess": 0.0}, forward)["status"] == "NO_ACTIVITY"

    growth = 1.20 * 2.558489 * 1.10
    assert abs(INITIAL_CAPITAL_EUR * growth - 33_772.0548) < 1e-6

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        samples = {
            "2016-01-01.jsonl.gz": ["2016-01-01T00:00:00Z", "2016-01-01T00:01:00Z"],
            "2026-07-24.jsonl.gz": ["2026-07-24T23:58:00Z", "2026-07-24T23:59:00Z"],
            "2026-07-26.jsonl.gz": [],
        }
        for name, stamps in samples.items():
            with gzip.open(root / name, "wt", encoding="utf-8") as fh:
                for ts in stamps:
                    fh.write(json.dumps({"t": ts, "S": "TEST"}) + "\n")
        audit = audit_raw_minute_coverage(root)
        assert audit["coverage_exact"] is True
        assert audit["observed_max_utc"] == RAW_EXPECTED_MAX_UTC.isoformat()

    assert len(TOP10) == 10 and TOP10[0].model_id == "R01_L_H11_D03_N1"
    assert TOP10[1].mode == "FIXED" and TOP10[-1].n == 6
    print("TOP10_EXTERNAL_VALIDATION_SELF_TEST_OK")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Frozen Top-10 backward/forward external validation")
    p.add_argument("--raw-minute-root", required=False)
    p.add_argument("--backward-v5-predictions", required=False)
    p.add_argument("--backward-learned-exit-predictions", required=False)
    p.add_argument("--forward-v5-predictions", required=False)
    p.add_argument("--forward-learned-exit-predictions", required=False)
    p.add_argument("--known-profit-root", required=False)
    p.add_argument("--daily-store-root", required=False)
    p.add_argument("--output-root", default="artifacts/top10-external-validation")
    p.add_argument("--entry-budget", type=int, default=12)
    p.add_argument("--max-workers", type=int, default=8)
    p.add_argument("--initial-capital", type=float, default=INITIAL_CAPITAL_EUR)
    p.add_argument("--tax-allowance-eur", type=float, default=1000.0)
    p.add_argument("--church-tax-rate", type=float, default=0.0)
    p.add_argument("--benchmark-partial-exemption", type=float, default=0.30)
    p.add_argument("--self-test", action="store_true")
    return p.parse_args()


def main() -> None:
    a = parse_args()
    if a.self_test:
        self_test()
        return
    required = (
        "raw_minute_root",
        "backward_v5_predictions",
        "backward_learned_exit_predictions",
        "forward_v5_predictions",
        "forward_learned_exit_predictions",
        "known_profit_root",
        "daily_store_root",
    )
    missing = [x for x in required if not getattr(a, x)]
    if missing:
        raise SystemExit("missing required args: " + ", ".join("--" + x.replace("_", "-") for x in missing))
    result = run(a)
    print(json.dumps({"contract_id": result["contract_id"], "status": result["status"], "output_root": a.output_root}, indent=2))


if __name__ == "__main__":
    main()
