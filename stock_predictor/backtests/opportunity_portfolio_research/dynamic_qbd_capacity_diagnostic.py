"""Causal max-names diagnostics using an already frozen R1 replay input set."""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import pandas as pd

from .cost_contracts import CostModel
from .portfolio_policy_contracts import Policy
from .tax_contracts import TaxConfig
from .dynamic_qbd_portfolio_replay import replay_family


def _signals(monthly_models: Path, artifact_ids: set[str]) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for path in monthly_models.rglob("model-predictions.parquet"):
        frame = pd.read_parquet(path, columns=["decision_date", "ticker", "score", "model_artifact_id"])
        frame = frame.loc[frame["model_artifact_id"].astype(str).isin(artifact_ids)]
        if not frame.empty:
            parts.append(frame)
    if not parts:
        raise FileNotFoundError("CAPACITY_DIAGNOSTIC_R1_GENERATION_PREDICTIONS_MISSING")
    return pd.concat(parts, ignore_index=True).drop_duplicates(["decision_date", "ticker", "model_artifact_id"])


def _summary(n: int, result: dict) -> dict:
    metrics = result["metrics"]
    trades = pd.DataFrame(result["trades"])
    cvna = trades.loc[trades["ticker"].eq("CVNA")] if not trades.empty else trades
    rank = trades.groupby("entry_score_rank", sort=True).agg(
        trades=("ticker", "size"), total_buy_notional=("buy_notional", "sum"),
        total_excess_return=("excess_return", "sum"), mean_excess_return=("excess_return", "mean"),
    ).reset_index() if not trades.empty else pd.DataFrame()
    return {"max_names": n, "terminal_value": float(metrics["terminal_value"]),
            "terminal_relative_return": float(metrics["terminal_value"] / metrics["urth_terminal_value"] - 1.0),
            "trade_count": int(metrics["trade_count"]), "relative_max_drawdown": float(metrics["worst_relative_drawdown"]),
            "cvna_trade_count": int(len(cvna)), "cvna_total_buy_notional": float(cvna["buy_notional"].sum()) if len(cvna) else 0.0,
            "cvna_max_buy_notional": float(cvna["buy_notional"].max()) if len(cvna) else 0.0,
            "trades": trades, "rank": rank}


def run(*, experiment_root: str | Path, distributions_path: str | Path) -> dict:
    root = Path(experiment_root)
    experiment_summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    schedule = pd.read_csv(root / "monthly-generation-schedule.csv")
    r1 = schedule.loc[schedule["arm"].eq("R1_ROLLING_RECIPE_MONTHLY_REFIT")].copy()
    if r1.empty:
        raise ValueError("CAPACITY_DIAGNOSTIC_R1_SCHEDULE_MISSING")
    signals = _signals(root / "monthly-models", set(r1["model_artifact_id"].astype(str)))
    prices = pd.read_parquet(root / "inputs" / "prices.parquet")
    distributions = pd.read_parquet(distributions_path)
    start = pd.Timestamp(r1["activation_date"].min())
    # The price input may deliberately contain observations beyond the sealed
    # development experiment.  A diagnostic must replay the recorded interval,
    # never silently extend it to the data-store maximum.
    evaluation = experiment_summary.get("evaluation", {})
    end = pd.Timestamp(evaluation.get("end"))
    if pd.isna(end):
        raise ValueError("CAPACITY_DIAGNOSTIC_EVALUATION_END_MISSING")
    base = Policy(horizon=3, score_quantile=.75, top_fraction=.01, max_names=1, holding_days=2, sleeve=.5)
    summaries = []
    rank_parts = []
    trade_parts = []
    for n in range(1, 7):
        result = replay_family(signals=signals, prices=prices, policy=replace(base, max_names=n), cost=CostModel(20.0),
                               tax=TaxConfig(False), generation_schedule=r1, start=start, end=end,
                               initial=10000.0, distributions=distributions)
        item = _summary(n, result)
        rank = item.pop("rank"); trades = item.pop("trades")
        if not rank.empty:
            rank["max_names"] = n; rank_parts.append(rank)
        if not trades.empty:
            trades["max_names"] = n; trade_parts.append(trades)
        summaries.append(item)
    frame = pd.DataFrame(summaries).sort_values("max_names")
    frame["marginal_terminal_value_eur"] = frame["terminal_value"].diff()
    frame["marginal_trade_count"] = frame["trade_count"].diff()
    recorded = pd.read_csv(root / "max-names-capacity-ablation.csv").set_index("max_names")
    reproduced = frame.set_index("max_names")
    shared = reproduced.index.intersection(recorded.index)
    mismatch = (reproduced.loc[shared, "terminal_value"] - recorded.loc[shared, "terminal_value"]).abs()
    if mismatch.empty or float(mismatch.max()) > 1e-8:
        raise AssertionError(f"CAPACITY_DIAGNOSTIC_REPLAY_MISMATCH:{float(mismatch.max()) if len(mismatch) else 'NO_SHARED_ROWS'}")
    output = root / "capacity-diagnostic"
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "capacity-summary.csv", index=False)
    pd.concat(rank_parts, ignore_index=True).to_csv(output / "contribution-by-entry-rank.csv", index=False)
    pd.concat(trade_parts, ignore_index=True).to_parquet(output / "capacity-trades.parquet", index=False)
    rank_frame = pd.concat(rank_parts, ignore_index=True)
    rank_1_3 = rank_frame.loc[rank_frame["entry_score_rank"].isin((1, 2, 3))].copy()
    rank_1_3.to_csv(output / "contribution-by-entry-rank-1-to-3.csv", index=False)
    payload = {"schema_version": "DYNAMIC_QBD_CAPACITY_DIAGNOSTIC_V1",
               "status": "COMPLETE_LEGACY_MODEL_CACHE_INPUTS_DIAGNOSTIC_ONLY",
               "authority": "DIAGNOSTIC_ONLY_NO_RECIPE_SELECTION_NO_PROMOTION_NO_HOLDOUT",
               "invariance_contract": "ONLY_MAX_NAMES_CHANGED_R1_SCORES_SCHEDULE_THRESHOLDS_SLEEVE_COSTS_EXECUTION_FIXED",
               "reproduction_contract": {"recorded_capacity_ablation_terminal_value_max_abs_error_eur": float(mismatch.max())},
               "source_model_provenance": "LEGACY_CACHE_NO_GENERATION_MANIFESTS_NOT_VALID_FOR_FINAL_CLAIM",
               "capacity": frame.to_dict(orient="records")}
    (output / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    n3 = frame.loc[frame["max_names"].eq(3)].iloc[0]
    lines = ["# N-capacity diagnostic", "", "Diagnostic only; legacy model files have no generation manifests.", "",
             "## Invariance", "", "Only `max_names` changed. Terminal-value reproduction against the recorded ablation is exact within EUR 1e-8.", "",
             "## Marginal capacity", "", frame.to_csv(index=False, lineterminator="\n"), "",
             "## N3 score-rank contribution", "", rank_1_3.loc[rank_1_3["max_names"].eq(3)].to_csv(index=False, lineterminator="\n"), "",
             "## CVNA exposure", "", frame[["max_names", "cvna_trade_count", "cvna_total_buy_notional", "cvna_max_buy_notional"]].to_csv(index=False, lineterminator="\n"), "",
             f"N3 terminal value: EUR {float(n3['terminal_value']):.2f}."]
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    return payload


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", required=True)
    parser.add_argument("--distributions", required=True)
    args = parser.parse_args(argv)
    print(json.dumps(run(experiment_root=args.experiment_root, distributions_path=args.distributions), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
