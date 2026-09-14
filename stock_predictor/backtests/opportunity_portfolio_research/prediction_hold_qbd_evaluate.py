from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path
from statistics import median
from typing import Iterable

import numpy as np
import pandas as pd

CONTRACT_ID = "PREDICTION_HOLD_QBD_SURFACE_V1"
PRIMARY_METRIC = "median_active_cagr_excess"
HEATMAP_METRICS = (
    "median_active_cagr_excess",
    "q25_active_cagr_excess",
    "positive_active_fold_fraction",
    "trade_count",
    "robust_gate_pass",
    "local_robust_neighbor_fraction",
    "design_space_pass",
)


def expected_cells(prediction_min: int = 1, prediction_max: int = 30, hold_max: int | None = None) -> list[tuple[int, int]]:
    lo, hi = int(prediction_min), int(prediction_max)
    if lo < 1 or hi < lo:
        raise ValueError(f"INVALID_PREDICTION_RANGE:{lo}:{hi}")
    out: list[tuple[int, int]] = []
    for horizon in range(lo, hi + 1):
        upper = horizon if hold_max is None else min(horizon, int(hold_max))
        out.extend((horizon, hold) for hold in range(1, max(0, upper) + 1))
    return out


def _aggregate_cell(group: pd.DataFrame) -> dict:
    values = pd.to_numeric(group["cagr_excess"], errors="coerce").fillna(0.0).astype(float)
    trades = pd.to_numeric(group["trade_count"], errors="coerce").fillna(0).astype(int)
    active = trades > 0
    active_values = values.loc[active]
    active_list, all_list = active_values.tolist(), values.tolist()
    turnover = pd.to_numeric(group["turnover"], errors="coerce").dropna() if "turnover" in group else pd.Series(dtype=float)
    drawdown = pd.to_numeric(group["worst_relative_drawdown"], errors="coerce").dropna() if "worst_relative_drawdown" in group else pd.Series(dtype=float)
    folds, active_folds = len(group), int(active.sum())
    row = {
        "prediction_horizon": int(group["prediction_horizon"].iloc[0]),
        "holding_days": int(group["holding_days"].iloc[0]),
        "folds": int(folds),
        "active_folds": active_folds,
        "inactive_folds": int(folds - active_folds),
        "active_fold_fraction": active_folds / max(1, folds),
        "positive_fold_fraction": int((values > 0).sum()) / max(1, folds),
        "positive_active_fold_fraction": int((active_values > 0).sum()) / max(1, active_folds),
        "median_cagr_excess": float(median(all_list)) if all_list else 0.0,
        "q25_cagr_excess": float(np.quantile(all_list, .25)) if all_list else 0.0,
        "worst_cagr_excess": float(min(all_list)) if all_list else 0.0,
        "median_active_cagr_excess": float(median(active_list)) if active_list else 0.0,
        "q25_active_cagr_excess": float(np.quantile(active_list, .25)) if active_list else 0.0,
        "worst_active_cagr_excess": float(min(active_list)) if active_list else 0.0,
        "trade_count": int(trades.sum()),
        "median_turnover": float(turnover.median()) if not turnover.empty else 0.0,
        "worst_relative_drawdown": float(drawdown.min()) if not drawdown.empty else 0.0,
    }
    row["robust_gate_pass"] = bool(
        active_folds >= 3
        and row["positive_active_fold_fraction"] >= .60
        and row["median_active_cagr_excess"] > 0
        and row["trade_count"] >= 8
    )
    return row


def aggregate_outer_results(rows: Iterable[dict] | pd.DataFrame) -> pd.DataFrame:
    frame = rows.copy() if isinstance(rows, pd.DataFrame) else pd.DataFrame(list(rows))
    required = {"prediction_horizon", "holding_days", "cagr_excess", "trade_count"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"QBD_OUTER_COLUMNS_MISSING:{','.join(missing)}")
    frame["prediction_horizon"] = pd.to_numeric(frame["prediction_horizon"], errors="raise").astype(int)
    frame["holding_days"] = pd.to_numeric(frame["holding_days"], errors="raise").astype(int)
    invalid = frame[(frame["prediction_horizon"] < 1) | (frame["holding_days"] < 1) | (frame["holding_days"] > frame["prediction_horizon"])]
    if not invalid.empty:
        raise ValueError(f"QBD_INVALID_CELL_IN_OUTER_RESULTS:{invalid[['prediction_horizon','holding_days']].drop_duplicates().to_dict('records')}")
    result = [_aggregate_cell(g) for _, g in frame.groupby(["prediction_horizon", "holding_days"], sort=True)]
    return pd.DataFrame(result).sort_values(["prediction_horizon", "holding_days"]).reset_index(drop=True)


def _neighbors(cell: tuple[int, int], allowed: set[tuple[int, int]]) -> list[tuple[int, int]]:
    h, d = cell
    return sorted(
        (h + dh, d + dd)
        for dh in (-1, 0, 1)
        for dd in (-1, 0, 1)
        if (dh or dd) and (h + dh, d + dd) in allowed
    )


def add_local_stability(cells: pd.DataFrame, expected: Iterable[tuple[int, int]]) -> pd.DataFrame:
    allowed = set(expected)
    lookup = {(int(r.prediction_horizon), int(r.holding_days)): r._asdict() for r in cells.itertuples(index=False)}
    rows = []
    for cell, center in lookup.items():
        wanted = _neighbors(cell, allowed)
        present = [n for n in wanted if n in lookup]
        neighbor_rows = [lookup[n] for n in present]
        vals = [float(x[PRIMARY_METRIC]) for x in neighbor_rows]
        robust_fraction = sum(bool(x["robust_gate_pass"]) for x in neighbor_rows) / max(1, len(neighbor_rows))
        coverage = len(present) / max(1, len(wanted)) if wanted else 1.0
        rows.append({
            "prediction_horizon": cell[0],
            "holding_days": cell[1],
            "expected_neighbor_count": len(wanted),
            "local_neighbor_count": len(present),
            "local_neighbor_coverage": coverage,
            "local_robust_neighbor_fraction": robust_fraction,
            "local_median_neighbor_cagr_excess": float(median(vals)) if vals else 0.0,
            "local_worst_neighbor_cagr_excess": float(min(vals)) if vals else 0.0,
            "local_median_abs_delta_cagr_excess": float(median([abs(v - float(center[PRIMARY_METRIC])) for v in vals])) if vals else 0.0,
            "local_stability_pass": bool(
                center["robust_gate_pass"] and coverage >= .999999 and len(present) >= 2
                and robust_fraction >= .50 and (float(median(vals)) if vals else 0.0) > 0
            ),
        })
    return cells.merge(pd.DataFrame(rows), on=["prediction_horizon", "holding_days"], how="left")


def _components(candidates: set[tuple[int, int]]) -> list[list[tuple[int, int]]]:
    remaining, out = set(candidates), []
    while remaining:
        start = min(remaining); remaining.remove(start)
        queue, component = deque([start]), []
        while queue:
            cell = queue.popleft(); component.append(cell)
            for n in _neighbors(cell, candidates):
                if n in remaining:
                    remaining.remove(n); queue.append(n)
        out.append(sorted(component))
    return out


def assign_design_space(cells: pd.DataFrame, minimum_plateau_cells: int = 3) -> tuple[pd.DataFrame, pd.DataFrame]:
    result = cells.copy()
    candidates = {(int(r.prediction_horizon), int(r.holding_days)) for r in result.itertuples(index=False) if bool(r.local_stability_pass)}
    lookup = result.set_index(["prediction_horizon", "holding_days"])
    cell_to_plateau, plateau_rows = {}, []
    for pid, component in enumerate(_components(candidates), 1):
        for cell in component:
            cell_to_plateau[cell] = pid
        block = lookup.loc[component]
        plateau_rows.append({
            "plateau_id": pid,
            "cells": len(component),
            "prediction_min": min(h for h, _ in component),
            "prediction_max": max(h for h, _ in component),
            "hold_min": min(d for _, d in component),
            "hold_max": max(d for _, d in component),
            "worst_cell_median_active_cagr_excess": float(block[PRIMARY_METRIC].min()),
            "median_cell_median_active_cagr_excess": float(block[PRIMARY_METRIC].median()),
            "median_cell_q25_active_cagr_excess": float(block["q25_active_cagr_excess"].median()),
            "minimum_positive_active_fold_fraction": float(block["positive_active_fold_fraction"].min()),
            "total_trades": int(block["trade_count"].sum()),
        })
    plateau = pd.DataFrame(plateau_rows)
    if not plateau.empty:
        plateau = plateau.sort_values(
            ["worst_cell_median_active_cagr_excess", "median_cell_q25_active_cagr_excess", "minimum_positive_active_fold_fraction", "cells", "median_cell_median_active_cagr_excess"],
            ascending=False,
        ).reset_index(drop=True)
        plateau["plateau_rank"] = np.arange(1, len(plateau) + 1)
    size_by_id = {int(r.plateau_id): int(r.cells) for r in plateau.itertuples(index=False)} if not plateau.empty else {}
    rank_by_id = {int(r.plateau_id): int(r.plateau_rank) for r in plateau.itertuples(index=False)} if not plateau.empty else {}
    ids = [cell_to_plateau.get((int(r.prediction_horizon), int(r.holding_days))) for r in result.itertuples(index=False)]
    result["plateau_id"] = ids
    result["plateau_size"] = [size_by_id.get(x, 0) if x is not None else 0 for x in ids]
    result["plateau_rank"] = [rank_by_id.get(x) if x is not None else None for x in ids]
    result["design_space_pass"] = result["plateau_size"] >= int(minimum_plateau_cells)
    return result, plateau


def validate_surface_coverage(status: pd.DataFrame, expected: Iterable[tuple[int, int]]) -> dict:
    expected_set = set(expected)
    if status.empty:
        observed, complete, duplicates = set(), set(), []
    else:
        frame = status.copy()
        frame[["prediction_horizon", "holding_days"]] = frame[["prediction_horizon", "holding_days"]].astype(int)
        dup = frame.duplicated(["prediction_horizon", "holding_days"], keep=False)
        duplicates = frame.loc[dup, ["prediction_horizon", "holding_days"]].drop_duplicates().to_dict("records")
        observed = set(zip(frame.prediction_horizon, frame.holding_days))
        ok = frame["status"].astype(str).eq("COMPLETE")
        complete = set(zip(frame.loc[ok, "prediction_horizon"], frame.loc[ok, "holding_days"]))
    missing, failed, unexpected = sorted(expected_set-observed), sorted(expected_set-complete), sorted(observed-expected_set)
    return {
        "expected_cells": len(expected_set), "observed_cells": len(observed), "complete_cells": len(complete & expected_set),
        "missing_cells": [{"prediction_horizon": h, "holding_days": d} for h, d in missing],
        "failed_or_incomplete_cells": [{"prediction_horizon": h, "holding_days": d} for h, d in failed],
        "unexpected_cells": [{"prediction_horizon": h, "holding_days": d} for h, d in unexpected],
        "duplicate_cells": duplicates,
        "surface_complete": not missing and not failed and not unexpected and not duplicates,
    }


def evaluate_surface(outer: pd.DataFrame, status: pd.DataFrame, *, prediction_min=1, prediction_max=30, hold_max=None, minimum_plateau_cells=3):
    expected = expected_cells(prediction_min, prediction_max, hold_max)
    coverage = validate_surface_coverage(status, expected)
    cells = aggregate_outer_results(outer) if not outer.empty else pd.DataFrame()
    if not cells.empty:
        cells = add_local_stability(cells, expected)
        cells, plateau = assign_design_space(cells, minimum_plateau_cells)
    else:
        plateau = pd.DataFrame()
    aggregates = {(int(r.prediction_horizon), int(r.holding_days)) for r in cells.itertuples(index=False)} if not cells.empty else set()
    complete = {(int(r.prediction_horizon), int(r.holding_days)) for r in status.itertuples(index=False) if str(r.status) == "COMPLETE"} if not status.empty else set()
    parity = aggregates == complete
    best = plateau.iloc[0].to_dict() if not plateau.empty and int(plateau.iloc[0]["cells"]) >= minimum_plateau_cells else None
    summary = {
        "contract_id": CONTRACT_ID, "primary_response": PRIMARY_METRIC,
        "final_holdout_opened": False, "final_holdout_locked": True, "interpolation_used": False,
        "surface_coverage": coverage, "aggregate_complete_cell_parity": parity,
        "aggregated_cells": len(cells),
        "robust_cells": int(cells["robust_gate_pass"].sum()) if not cells.empty else 0,
        "locally_stable_cells": int(cells["local_stability_pass"].sum()) if not cells.empty else 0,
        "design_space_cells": int(cells["design_space_pass"].sum()) if not cells.empty else 0,
        "plateaus": len(plateau), "minimum_plateau_cells": int(minimum_plateau_cells), "best_plateau": best,
        "qbd_complete": bool(coverage["surface_complete"] and parity),
    }
    return cells, plateau, summary


def load_cell_artifacts(root: Path):
    status_rows, outer_rows, final_rows = [], [], []
    for path in sorted((Path(root) / "cells").glob("H*_D*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        h, d, state = int(payload["prediction_horizon"]), int(payload["holding_days"]), str(payload.get("status", "UNKNOWN"))
        status_rows.append({"prediction_horizon": h, "holding_days": d, "status": state, "error": payload.get("error"), "artifact": str(path)})
        if state != "COMPLETE":
            continue
        outer_rows.extend({**row, "prediction_horizon": h, "holding_days": d} for row in payload.get("outer_rows", []))
        if payload.get("final_policy"):
            final_rows.append({"prediction_horizon": h, "holding_days": d, **payload["final_policy"], "resolved_threshold": payload.get("final_threshold")})
    return pd.DataFrame(status_rows), pd.DataFrame(outer_rows), pd.DataFrame(final_rows)


def write_evaluation_artifacts(root: Path, *, prediction_min=1, prediction_max=30, hold_max=None, minimum_plateau_cells=3) -> dict:
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    status, outer, final = load_cell_artifacts(root)
    cells, plateau, summary = evaluate_surface(outer, status, prediction_min=prediction_min, prediction_max=prediction_max, hold_max=hold_max, minimum_plateau_cells=minimum_plateau_cells)
    status.to_csv(root / "qbd_cell_status.csv", index=False)
    outer.to_csv(root / "qbd_outer_fold_results.csv", index=False)
    final.to_csv(root / "qbd_final_policies.csv", index=False)
    cells.to_csv(root / "qbd_surface_cells.csv", index=False)
    plateau.to_csv(root / "qbd_plateaus.csv", index=False)
    (cells[cells["design_space_pass"]] if not cells.empty else pd.DataFrame()).to_csv(root / "qbd_design_space.csv", index=False)
    heatmaps = []
    if not cells.empty:
        predictions = list(range(int(prediction_min), int(prediction_max)+1))
        max_hold = max((d for _, d in expected_cells(prediction_min, prediction_max, hold_max)), default=0)
        for metric in HEATMAP_METRICS:
            pivot = cells.pivot(index="holding_days", columns="prediction_horizon", values=metric).reindex(index=range(1, max_hold+1), columns=predictions)
            path = root / f"heatmap_{metric}.csv"; pivot.to_csv(path, index_label="holding_days"); heatmaps.append(path.name)
    summary["heatmap_files"] = heatmaps
    (root / "qbd_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--output-root", default="artifacts/prediction-hold-qbd-surface")
    p.add_argument("--prediction-min", type=int, default=1); p.add_argument("--prediction-max", type=int, default=30)
    p.add_argument("--hold-max", type=int, default=None); p.add_argument("--minimum-plateau-cells", type=int, default=3)
    a = p.parse_args()
    summary = write_evaluation_artifacts(Path(a.output_root), prediction_min=a.prediction_min, prediction_max=a.prediction_max, hold_max=a.hold_max, minimum_plateau_cells=a.minimum_plateau_cells)
    print(json.dumps(summary, indent=2, sort_keys=True, default=str))
    return 0 if summary["qbd_complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
