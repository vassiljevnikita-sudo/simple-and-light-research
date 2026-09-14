"""Retrospective, maturity-aware validation of Top-10 entry thresholds.

Contract: TOP10_RETROSPECTIVE_THRESHOLD_VALIDATION_V1

This suite never changes a live threshold and never uses future outcomes to make a
forward decision. At each historical assessment date it looks backward only at
predictions whose H-session outcome was already fully realized by that date. It asks
whether the then-used threshold rejected top-ranked names that subsequently showed
positive realized net excess.

This is diagnostic evidence only. It does not fit models, regenerate predictions,
optimize thresholds, or select a replacement policy from the evaluation period.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .portfolio_research_inputs import load_price_panel
from .top10_entry_activation_alpha_diagnostic import (
    LIVE_END,
    LIVE_START,
    _attach_realized,
    _causal_entry,
    _contracts,
    _markdown,
    _price_views,
    _threshold_column,
    _write_json,
)

CONTRACT_ID = "TOP10_RETROSPECTIVE_THRESHOLD_VALIDATION_V1"
PRIMARY_LOOKBACK_SESSIONS = 126
SENSITIVITY_LOOKBACKS = (63, 126, 252)
ASSESS_EVERY_SESSIONS = 21
MIN_MATURED_DECISION_DAYS = 42
MIN_SHADOW_CANDIDATES = 30


def _historical_crossing_baseline(path: Path) -> dict[str, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "COMPLETE":
        raise RuntimeError(f"ENTRY_DIAGNOSTIC_NOT_COMPLETE:{payload.get('status')}")
    if payload.get("final_holdout_opened"):
        raise RuntimeError("FINAL_HOLDOUT_MUST_REMAIN_CLOSED")
    result: dict[str, float] = {}
    for row in payload.get("activation_summary", []):
        if row.get("source") != "HISTORICAL_WF":
            continue
        result[str(row["contract_id"])] = float(row["crossing_day_rate"])
    if not result:
        raise RuntimeError("HISTORICAL_CROSSING_BASELINE_MISSING")
    return result


def _assessment_dates(causal: pd.DataFrame) -> list[pd.Timestamp]:
    dates = [pd.Timestamp(x) for x in sorted(causal["decision_date"].drop_duplicates())]
    if not dates or dates[0] != LIVE_START or dates[-1] != LIVE_END:
        raise RuntimeError("CAUSAL_ASSESSMENT_DATE_COVERAGE_MISMATCH")
    anchors = dates[ASSESS_EVERY_SESSIONS - 1 :: ASSESS_EVERY_SESSIONS]
    if dates[-1] not in anchors:
        anchors.append(dates[-1])
    return anchors


def _prepare_contract_rows(causal: pd.DataFrame, prices: pd.DataFrame, horizon: int,
                           score_quantile: float, top_fraction: float) -> pd.DataFrame:
    base = causal.loc[causal["horizon"].eq(int(horizon))].copy()
    threshold_col = _threshold_column(score_quantile)
    if threshold_col not in base.columns:
        raise RuntimeError(f"CAUSAL_THRESHOLD_COLUMN_MISSING:{threshold_col}")
    if base.groupby("decision_date")[threshold_col].nunique(dropna=False).ne(1).any():
        raise RuntimeError(f"CAUSAL_THRESHOLD_NOT_UNIQUE_PER_DATE:{threshold_col}")

    base["daily_rank"] = base.groupby("decision_date")["score"].rank(method="first", ascending=False)
    universe = base.groupby("decision_date")["ticker"].transform("size")
    base["top_fraction_limit"] = np.maximum(1, np.ceil(universe * float(top_fraction))).astype(int)
    base["relative_candidate"] = base["daily_rank"].le(base["top_fraction_limit"])
    base["threshold"] = pd.to_numeric(base[threshold_col], errors="coerce")
    if base["threshold"].isna().any():
        raise RuntimeError(f"CAUSAL_THRESHOLD_NAN:{threshold_col}")
    base["accepted_by_threshold"] = base["score"].ge(base["threshold"])
    base["accepted_relative_candidate"] = base["relative_candidate"] & base["accepted_by_threshold"]
    base["shadow_rejected_candidate"] = base["relative_candidate"] & ~base["accepted_by_threshold"]

    stock, urth, sessions, session_index = _price_views(prices)
    realized = _attach_realized(base, stock, urth, sessions, session_index, int(horizon))
    flags = base[[
        "decision_date", "ticker", "threshold", "daily_rank", "top_fraction_limit",
        "relative_candidate", "accepted_by_threshold", "accepted_relative_candidate",
        "shadow_rejected_candidate",
    ]]
    realized = realized.merge(flags, on=["decision_date", "ticker"], how="left", validate="one_to_one")
    if realized[["relative_candidate", "accepted_by_threshold", "shadow_rejected_candidate"]].isna().any().any():
        raise RuntimeError("RETROSPECTIVE_FLAG_MERGE_FAILED")
    return realized


def _classify(*, historical_crossing_rate: float, recent_crossing_rate: float,
              matured_decision_days: int, shadow_count: int, shadow_mean: float) -> str:
    if matured_decision_days < MIN_MATURED_DECISION_DAYS or shadow_count < MIN_SHADOW_CANDIDATES:
        return "INSUFFICIENT_MATURED_EVIDENCE"
    collapse = (
        historical_crossing_rate >= 0.01
        and recent_crossing_rate < max(0.005, 0.25 * historical_crossing_rate)
    )
    if not collapse:
        return "NO_RETROSPECTIVE_RESTRICTION_EVIDENCE"
    if np.isfinite(shadow_mean) and shadow_mean > 0:
        return "RETROSPECTIVE_TOO_RESTRICTIVE_EVIDENCE"
    return "RARE_SIGNAL_WITHOUT_POSITIVE_SHADOW_ALPHA"


def _one_window(frame: pd.DataFrame, assessment_date: pd.Timestamp, lookback_sessions: int,
                historical_crossing_rate: float, contract_id: str, horizon: int,
                score_quantile: float, top_fraction: float) -> dict:
    matured = frame.loc[
        frame["realized_price_available"]
        & frame["terminal_date"].le(assessment_date)
        & frame["decision_date"].le(assessment_date)
    ].copy()
    matured_dates = [pd.Timestamp(x) for x in sorted(matured["decision_date"].drop_duplicates())]
    if not matured_dates:
        return {
            "contract_id": contract_id, "horizon": horizon, "score_quantile": score_quantile,
            "top_fraction": top_fraction, "assessment_date": assessment_date,
            "lookback_sessions": lookback_sessions, "matured_decision_days": 0,
            "verdict": "INSUFFICIENT_MATURED_EVIDENCE",
        }
    window_dates = matured_dates[-int(lookback_sessions):]
    window = matured.loc[matured["decision_date"].isin(window_dates)].copy()

    daily = window.groupby("decision_date", sort=True).agg(
        crossing=("accepted_by_threshold", "max"),
        threshold=("threshold", "first"),
        score_p99=("score", lambda s: float(s.quantile(0.99))),
        score_max=("score", "max"),
    ).reset_index()
    shadow = window.loc[window["shadow_rejected_candidate"]].copy()
    accepted = window.loc[window["accepted_relative_candidate"]].copy()
    relative = window.loc[window["relative_candidate"]].copy()

    shadow_mean = float(shadow["realized_net_excess_20bps"].mean()) if len(shadow) else float("nan")
    shadow_median = float(shadow["realized_net_excess_20bps"].median()) if len(shadow) else float("nan")
    shadow_hit = float(shadow["realized_net_excess_20bps"].gt(0).mean()) if len(shadow) else float("nan")
    accepted_mean = float(accepted["realized_net_excess_20bps"].mean()) if len(accepted) else float("nan")
    relative_mean = float(relative["realized_net_excess_20bps"].mean()) if len(relative) else float("nan")
    recent_crossing_rate = float(daily["crossing"].mean()) if len(daily) else float("nan")
    pressure = np.where(daily["score_p99"].abs() > 1e-15, daily["threshold"] / daily["score_p99"], np.nan)
    verdict = _classify(
        historical_crossing_rate=float(historical_crossing_rate),
        recent_crossing_rate=recent_crossing_rate,
        matured_decision_days=len(window_dates),
        shadow_count=len(shadow),
        shadow_mean=shadow_mean,
    )
    return {
        "contract_id": contract_id,
        "horizon": int(horizon),
        "score_quantile": float(score_quantile),
        "top_fraction": float(top_fraction),
        "assessment_date": assessment_date,
        "lookback_sessions": int(lookback_sessions),
        "window_start_decision_date": min(window_dates),
        "window_end_decision_date": max(window_dates),
        "latest_matured_terminal_date": pd.Timestamp(window["terminal_date"].max()),
        "matured_decision_days": len(window_dates),
        "evaluated_rows": int(len(window)),
        "historical_crossing_day_rate": float(historical_crossing_rate),
        "recent_crossing_day_rate": recent_crossing_rate,
        "activation_ratio_vs_historical": (
            recent_crossing_rate / historical_crossing_rate if historical_crossing_rate > 0 else float("nan")
        ),
        "median_threshold_over_p99": float(np.nanmedian(pressure)) if np.isfinite(pressure).any() else float("nan"),
        "relative_candidate_count": int(len(relative)),
        "accepted_relative_candidate_count": int(len(accepted)),
        "shadow_rejected_candidate_count": int(len(shadow)),
        "shadow_mean_realized_net_excess": shadow_mean,
        "shadow_median_realized_net_excess": shadow_median,
        "shadow_hit_rate": shadow_hit,
        "accepted_mean_realized_net_excess": accepted_mean,
        "all_relative_candidates_mean_realized_net_excess": relative_mean,
        "verdict": verdict,
        "verdict_is_heuristic": True,
    }


def run(args: argparse.Namespace) -> dict:
    output = Path(args.output_root)
    output.mkdir(parents=True, exist_ok=True)

    causal = _causal_entry(Path(args.causal_entry_predictions))
    contracts, _ = _contracts(Path(args.frozen_policy_manifest))
    historical_baseline = _historical_crossing_baseline(Path(args.entry_diagnostic_summary))
    tickers = set(causal["ticker"].astype(str))
    prices, price_audit = load_price_panel(Path(args.daily_store_root), tickers)
    anchors = _assessment_dates(causal)

    prepared: dict[tuple[int, float, float], pd.DataFrame] = {}
    rows: list[dict] = []
    for contract in contracts.itertuples(index=False):
        if contract.contract_id not in historical_baseline:
            raise RuntimeError(f"HISTORICAL_BASELINE_MISSING:{contract.contract_id}")
        key = (int(contract.horizon), float(contract.score_quantile), float(contract.top_fraction))
        if key not in prepared:
            prepared[key] = _prepare_contract_rows(
                causal, prices, int(contract.horizon), float(contract.score_quantile), float(contract.top_fraction)
            )
        frame = prepared[key]
        for assessment_date in anchors:
            for lookback in SENSITIVITY_LOOKBACKS:
                rows.append(_one_window(
                    frame, assessment_date, lookback,
                    historical_baseline[contract.contract_id], contract.contract_id,
                    int(contract.horizon), float(contract.score_quantile), float(contract.top_fraction),
                ))

    windows = pd.DataFrame(rows)
    windows.to_csv(output / "retrospective_threshold_windows.csv", index=False)
    windows.to_parquet(output / "retrospective_threshold_windows.parquet", index=False)

    latest_date = pd.Timestamp(windows["assessment_date"].dropna().max())
    primary = windows.loc[
        pd.to_datetime(windows["assessment_date"]).eq(latest_date)
        & windows["lookback_sessions"].eq(PRIMARY_LOOKBACK_SESSIONS)
    ].copy().sort_values(["horizon", "score_quantile", "top_fraction"])
    primary.to_csv(output / "latest_primary_126_session_assessment.csv", index=False)

    summary = {
        "contract_id": CONTRACT_ID,
        "status": "COMPLETE",
        "interpretation": "RETROSPECTIVE_DIAGNOSTIC_ONLY_NO_THRESHOLD_CHANGE",
        "no_model_training": True,
        "no_prediction_generation": True,
        "no_threshold_optimization": True,
        "no_forward_outcome_use": True,
        "final_holdout_opened": False,
        "assessment_contract": {
            "primary_lookback_sessions": PRIMARY_LOOKBACK_SESSIONS,
            "sensitivity_lookbacks_sessions": list(SENSITIVITY_LOOKBACKS),
            "assessment_every_sessions": ASSESS_EVERY_SESSIONS,
            "outcome_visibility_rule": "terminal_date <= assessment_date",
            "shadow_definition": "top_fraction relative candidate rejected by then-used threshold",
            "replacement_threshold_selected": False,
        },
        "latest_assessment_date": str(latest_date.date()),
        "latest_primary_assessment": primary.to_dict(orient="records"),
        "price_audit": price_audit,
    }
    _write_json(output / "retrospective_threshold_summary.json", summary)

    report = primary[[
        "contract_id", "horizon", "matured_decision_days", "historical_crossing_day_rate",
        "recent_crossing_day_rate", "shadow_rejected_candidate_count",
        "shadow_mean_realized_net_excess", "shadow_hit_rate", "median_threshold_over_p99", "verdict",
    ]].copy()
    for column in ("historical_crossing_day_rate", "recent_crossing_day_rate", "shadow_mean_realized_net_excess", "shadow_hit_rate"):
        report[column] = report[column].map(lambda x: "nan" if pd.isna(x) else f"{float(x):.4%}")
    text = [
        "# Top-10 Retrospective Threshold Validation",
        "",
        "Status: **COMPLETE**",
        "",
        "This is a backward-looking, maturity-aware diagnostic. At each historical assessment date, only outcomes whose terminal date had already occurred are visible.",
        "It does not fit models, regenerate predictions, optimize thresholds, or choose a replacement threshold from the evaluation period.",
        "",
        f"Primary window: last {PRIMARY_LOOKBACK_SESSIONS} fully evaluable decision sessions.",
        "63 and 252 sessions are sensitivity views only and must not be used to pick the best-looking result.",
        "",
        "## Latest primary assessment",
        "",
        _markdown(report),
        "",
        "`RETROSPECTIVE_TOO_RESTRICTIVE_EVIDENCE` means the old activation rate collapsed while top-fraction names rejected by the threshold had positive subsequently realized net excess in the fully matured backward window.",
        "It is evidence for a calibration review, not authorization to change the live threshold.",
        "",
        "Final holdout remained closed.",
    ]
    (output / "REPORT.md").write_text("\n".join(text) + "\n", encoding="utf-8")
    return summary


def self_test() -> None:
    assert _classify(
        historical_crossing_rate=0.05, recent_crossing_rate=0.0,
        matured_decision_days=126, shadow_count=100, shadow_mean=0.02,
    ) == "RETROSPECTIVE_TOO_RESTRICTIVE_EVIDENCE"
    assert _classify(
        historical_crossing_rate=0.05, recent_crossing_rate=0.0,
        matured_decision_days=126, shadow_count=100, shadow_mean=-0.01,
    ) == "RARE_SIGNAL_WITHOUT_POSITIVE_SHADOW_ALPHA"
    assert _classify(
        historical_crossing_rate=0.05, recent_crossing_rate=0.03,
        matured_decision_days=126, shadow_count=100, shadow_mean=0.02,
    ) == "NO_RETROSPECTIVE_RESTRICTION_EVIDENCE"
    assert _classify(
        historical_crossing_rate=0.05, recent_crossing_rate=0.0,
        matured_decision_days=20, shadow_count=10, shadow_mean=0.02,
    ) == "INSUFFICIENT_MATURED_EVIDENCE"
    print("TOP10_RETROSPECTIVE_THRESHOLD_VALIDATION_SELF_TEST_OK")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--causal-entry-predictions")
    parser.add_argument("--frozen-policy-manifest")
    parser.add_argument("--entry-diagnostic-summary")
    parser.add_argument("--daily-store-root")
    parser.add_argument("--output-root", default="artifacts/top10-retrospective-threshold-validation")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    required = ("causal_entry_predictions", "frozen_policy_manifest", "entry_diagnostic_summary", "daily_store_root")
    missing = [name for name in required if not getattr(args, name)]
    if missing:
        parser.error("required arguments missing: " + ", ".join("--" + x.replace("_", "-") for x in missing))
    print(json.dumps(run(args), indent=2, default=str))


if __name__ == "__main__":
    main()
