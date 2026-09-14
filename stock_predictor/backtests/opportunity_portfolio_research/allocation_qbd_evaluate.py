from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .portfolio_allocation_weights import CONTRACT_ID, DEFAULT_TREATMENTS
from .prediction_hold_qbd_evaluate import _aggregate_cell


BASELINE = "EQUAL_ACTIVE"


def treatment_slug(value: str) -> str:
    return str(value).replace(":", "_").replace(".", "p").replace("/", "_")


def load_primary_phase1_design_space(path: Path) -> list[tuple[int, int]]:
    frame = pd.read_csv(Path(path))
    required = {"prediction_horizon", "holding_days", "plateau_id", "plateau_size"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"ALLOCATION_QBD_PHASE1_COLUMNS_MISSING:{','.join(missing)}")
    if "design_space_pass" in frame:
        passed = frame["design_space_pass"].astype(str).str.lower().isin(("true", "1", "yes"))
        frame = frame.loc[passed].copy()
    if frame.empty:
        raise ValueError("ALLOCATION_QBD_PHASE1_DESIGN_SPACE_EMPTY")
    sizes = frame.groupby("plateau_id", dropna=True)["plateau_size"].max()
    if sizes.empty:
        raise ValueError("ALLOCATION_QBD_PHASE1_PLATEAU_MISSING")
    max_size = float(sizes.max())
    candidate_ids = sorted(sizes.loc[sizes.eq(max_size)].index.tolist())
    if "plateau_rank" in frame and len(candidate_ids) > 1:
        ranks = frame.loc[frame["plateau_id"].isin(candidate_ids)].groupby("plateau_id")["plateau_rank"].min()
        plateau_id = ranks.sort_values().index[0]
    else:
        plateau_id = candidate_ids[0]
    block = frame.loc[frame["plateau_id"].eq(plateau_id), ["prediction_horizon", "holding_days"]].copy()
    block = block.astype(int).drop_duplicates().sort_values(["prediction_horizon", "holding_days"])
    cells = [tuple(x) for x in block.itertuples(index=False, name=None)]
    if len(cells) != int(max_size):
        raise ValueError(f"ALLOCATION_QBD_PHASE1_PLATEAU_SIZE_MISMATCH:declared={int(max_size)}:rows={len(cells)}")
    return cells


def expected_cells(design_cells: Iterable[tuple[int, int]], treatments: Iterable[str] = DEFAULT_TREATMENTS) -> list[tuple[int, int, str]]:
    cells = sorted({(int(h), int(d)) for h, d in design_cells})
    tx = tuple(dict.fromkeys(str(x) for x in treatments))
    if not cells or not tx or BASELINE not in tx:
        raise ValueError("ALLOCATION_QBD_EXPECTED_CELLS_INVALID")
    return [(h, d, t) for h, d in cells for t in tx]


def validate_surface_coverage(status: pd.DataFrame, expected: Iterable[tuple[int, int, str]]) -> dict:
    wanted = set(expected)
    keys = ["prediction_horizon", "holding_days", "allocation"]
    if status.empty:
        observed, complete, duplicates = set(), set(), []
    else:
        frame = status.copy()
        frame["prediction_horizon"] = frame["prediction_horizon"].astype(int)
        frame["holding_days"] = frame["holding_days"].astype(int)
        frame["allocation"] = frame["allocation"].astype(str)
        dup = frame.duplicated(keys, keep=False)
        duplicates = frame.loc[dup, keys].drop_duplicates().to_dict("records")
        observed = set(map(tuple, frame[keys].itertuples(index=False, name=None)))
        ok = frame["status"].astype(str).eq("COMPLETE")
        complete = set(map(tuple, frame.loc[ok, keys].itertuples(index=False, name=None)))
    missing = sorted(wanted - observed)
    failed = sorted(wanted - complete)
    unexpected = sorted(observed - wanted)
    pack = lambda rows: [{"prediction_horizon": int(h), "holding_days": int(d), "allocation": str(a)} for h, d, a in rows]
    return {
        "expected_cells": len(wanted), "observed_cells": len(observed), "complete_cells": len(complete & wanted),
        "missing_cells": pack(missing), "failed_or_incomplete_cells": pack(failed),
        "unexpected_cells": pack(unexpected), "duplicate_cells": duplicates,
        "surface_complete": not missing and not failed and not unexpected and not duplicates,
    }


def aggregate_outer_results(rows: pd.DataFrame) -> pd.DataFrame:
    required = {"prediction_horizon", "holding_days", "allocation", "cagr_excess", "trade_count"}
    missing = sorted(required - set(rows.columns))
    if missing:
        raise ValueError(f"ALLOCATION_QBD_OUTER_COLUMNS_MISSING:{','.join(missing)}")
    out = []
    for (_h, _d, allocation), group in rows.groupby(["prediction_horizon", "holding_days", "allocation"], sort=True):
        row = _aggregate_cell(group)
        row["allocation"] = str(allocation)
        out.append(row)
    return pd.DataFrame(out).sort_values(["prediction_horizon", "holding_days", "allocation"]).reset_index(drop=True)


def add_paired_baseline(cells: pd.DataFrame) -> pd.DataFrame:
    result = cells.copy()
    baseline = result.loc[result["allocation"].eq(BASELINE), [
        "prediction_horizon", "holding_days", "median_active_cagr_excess", "q25_active_cagr_excess", "worst_active_cagr_excess",
    ]].rename(columns={
        "median_active_cagr_excess": "baseline_median_active_cagr_excess",
        "q25_active_cagr_excess": "baseline_q25_active_cagr_excess",
        "worst_active_cagr_excess": "baseline_worst_active_cagr_excess",
    })
    result = result.merge(baseline, on=["prediction_horizon", "holding_days"], how="left")
    if result["baseline_median_active_cagr_excess"].isna().any():
        raise ValueError("ALLOCATION_QBD_BASELINE_CELL_MISSING")
    result["delta_median_active_cagr_excess"] = result["median_active_cagr_excess"] - result["baseline_median_active_cagr_excess"]
    result["delta_q25_active_cagr_excess"] = result["q25_active_cagr_excess"] - result["baseline_q25_active_cagr_excess"]
    result["beats_equal_active"] = result["delta_median_active_cagr_excess"] > 0.0
    result.loc[result["allocation"].eq(BASELINE), "beats_equal_active"] = False
    return result


def summarize_treatments(cells: pd.DataFrame, design_cell_count: int) -> pd.DataFrame:
    rows = []
    for allocation, group in cells.groupby("allocation", sort=True):
        deltas = group["delta_median_active_cagr_excess"].astype(float)
        medians = group["median_active_cagr_excess"].astype(float)
        robust_fraction = float(group["robust_gate_pass"].astype(bool).mean())
        positive_fraction = float((medians > 0.0).mean())
        if allocation == BASELINE:
            beat_fraction = 0.0
            treatment_pass = bool(len(group) == design_cell_count and robust_fraction >= 0.50 and positive_fraction >= 0.50)
        else:
            beat_fraction = float((deltas > 0.0).mean())
            treatment_pass = bool(len(group) == design_cell_count and robust_fraction >= 0.50 and beat_fraction >= 0.60 and float(deltas.median()) > 0.0)
        rows.append({
            "allocation": allocation, "cells": int(len(group)), "expected_cells": int(design_cell_count),
            "complete_across_phase1_design_space": bool(len(group) == design_cell_count),
            "robust_cells": int(group["robust_gate_pass"].astype(bool).sum()), "robust_cell_fraction": robust_fraction,
            "positive_cell_fraction": positive_fraction, "median_cell_median_active_cagr_excess": float(medians.median()),
            "q25_cell_median_active_cagr_excess": float(np.quantile(medians, .25)), "worst_cell_median_active_cagr_excess": float(medians.min()),
            "median_delta_vs_equal_active": float(deltas.median()), "q25_delta_vs_equal_active": float(np.quantile(deltas, .25)),
            "worst_delta_vs_equal_active": float(deltas.min()), "beat_equal_active_cell_fraction": beat_fraction,
            "total_trades": int(group["trade_count"].sum()), "median_turnover": float(group["median_turnover"].median()),
            "allocation_qbd_pass": treatment_pass, "baseline_reference": allocation == BASELINE,
        })
    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary
    summary = summary.sort_values(
        ["allocation_qbd_pass", "q25_delta_vs_equal_active", "median_delta_vs_equal_active", "robust_cell_fraction", "worst_delta_vs_equal_active"],
        ascending=[False, False, False, False, False],
    ).reset_index(drop=True)
    summary["treatment_rank"] = np.arange(1, len(summary) + 1)
    return summary


def evaluate_surface(outer: pd.DataFrame, status: pd.DataFrame, design_cells: list[tuple[int, int]], treatments: Iterable[str]):
    wanted = expected_cells(design_cells, treatments)
    coverage = validate_surface_coverage(status, wanted)
    cells = aggregate_outer_results(outer) if not outer.empty else pd.DataFrame()
    if not cells.empty:
        cells = add_paired_baseline(cells)
        treatment = summarize_treatments(cells, len(design_cells))
    else:
        treatment = pd.DataFrame()
    aggregates = set(map(tuple, cells[["prediction_horizon", "holding_days", "allocation"]].itertuples(index=False, name=None))) if not cells.empty else set()
    complete = set(map(tuple, status.loc[status["status"].astype(str).eq("COMPLETE"), ["prediction_horizon", "holding_days", "allocation"]].itertuples(index=False, name=None))) if not status.empty else set()
    parity = aggregates == complete
    passing = treatment.loc[treatment["allocation_qbd_pass"].astype(bool)] if not treatment.empty else pd.DataFrame()
    nonbaseline_passing = passing.loc[~passing["baseline_reference"].astype(bool)] if not passing.empty else pd.DataFrame()
    best = nonbaseline_passing.iloc[0].to_dict() if not nonbaseline_passing.empty else None
    summary = {
        "contract_id": CONTRACT_ID, "phase1_design_cells": len(design_cells),
        "treatments": list(dict.fromkeys(str(x) for x in treatments)), "expected_surface_cells": len(wanted),
        "surface_coverage": coverage, "aggregate_complete_cell_parity": bool(parity), "aggregated_cells": len(cells),
        "passing_treatments": int(treatment["allocation_qbd_pass"].sum()) if not treatment.empty else 0,
        "best_nonbaseline_treatment": best,
        "selection_principle": "paired broad stability across frozen Phase-1 H/D design space; not maximum CAGR",
        "final_holdout_opened": False, "final_holdout_locked": True, "interpolation_used": False,
        "v45_exit_overlay_used": False, "qbd_complete": bool(coverage["surface_complete"] and parity),
    }
    return cells, treatment, summary


def load_cell_artifacts(root: Path):
    status_rows, outer_rows, final_rows = [], [], []
    for path in sorted((Path(root) / "cells").glob("H*_D*__*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        h, d, allocation = int(payload["prediction_horizon"]), int(payload["holding_days"]), str(payload["allocation"])
        state = str(payload.get("status", "UNKNOWN"))
        status_rows.append({"prediction_horizon": h, "holding_days": d, "allocation": allocation, "status": state, "error": payload.get("error"), "artifact": str(path)})
        if state != "COMPLETE":
            continue
        outer_rows.extend({**row, "prediction_horizon": h, "holding_days": d, "allocation": allocation} for row in payload.get("outer_rows", []))
        if payload.get("final_policy"):
            final_rows.append({"prediction_horizon": h, "holding_days": d, "allocation": allocation, **payload["final_policy"], "resolved_threshold": payload.get("final_threshold")})
    return pd.DataFrame(status_rows), pd.DataFrame(outer_rows), pd.DataFrame(final_rows)


def _write_heatmaps(root: Path, cells: pd.DataFrame) -> None:
    heat = root / "heatmaps"
    heat.mkdir(parents=True, exist_ok=True)
    for allocation, group in cells.groupby("allocation", sort=True):
        slug = treatment_slug(allocation)
        for metric in ("median_active_cagr_excess", "delta_median_active_cagr_excess", "robust_gate_pass"):
            table = group.pivot(index="prediction_horizon", columns="holding_days", values=metric)
            table.to_csv(heat / f"{metric}__{slug}.csv")


def _write_markdown(root: Path, summary: dict, treatment: pd.DataFrame) -> None:
    lines = [
        "# Phase 2 — Allocation QbD Results", "", f"- Contract: `{CONTRACT_ID}`",
        f"- Frozen Phase-1 H/D cells: **{summary['phase1_design_cells']}**", f"- Allocation treatments: **{len(summary['treatments'])}**",
        f"- Surface cells: **{summary['surface_coverage']['complete_cells']}/{summary['expected_surface_cells']}**",
        "- Final Holdout: **closed**", "- Interpolation: **not used**", "- V4.5 overlay: **not used**", "", "## Treatment ranking", "",
    ]
    if treatment.empty:
        lines.append("No complete treatment results.")
    else:
        cols = ["allocation", "treatment_rank", "robust_cell_fraction", "beat_equal_active_cell_fraction", "median_delta_vs_equal_active", "q25_delta_vs_equal_active", "allocation_qbd_pass"]
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
        for row in treatment[cols].itertuples(index=False, name=None):
            lines.append("| " + " | ".join(str(value) for value in row) + " |")
    lines += ["", "Selection is based on paired stability across the frozen Phase-1 design space, not an isolated maximum-CAGR cell.", ""]
    (root / "ALLOCATION_QBD_RESULTS_SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def write_evaluation_artifacts(root: Path, *, phase1_design_space: Path, treatments: Iterable[str] = DEFAULT_TREATMENTS) -> dict:
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    design_cells = load_primary_phase1_design_space(Path(phase1_design_space))
    tx = tuple(dict.fromkeys(str(x) for x in treatments))
    status, outer, final = load_cell_artifacts(root)
    cells, treatment, summary = evaluate_surface(outer, status, design_cells, tx)
    status.to_csv(root / "allocation_qbd_cell_status.csv", index=False)
    outer.to_csv(root / "allocation_qbd_outer_fold_results.csv", index=False)
    final.to_csv(root / "allocation_qbd_final_policies.csv", index=False)
    cells.to_csv(root / "allocation_qbd_surface_cells.csv", index=False)
    treatment.to_csv(root / "allocation_qbd_treatment_summary.csv", index=False)
    paired = cells.loc[~cells["allocation"].eq(BASELINE)].copy() if not cells.empty else pd.DataFrame()
    paired.to_csv(root / "allocation_qbd_paired_vs_equal.csv", index=False)
    passing_names = set(treatment.loc[treatment["allocation_qbd_pass"].astype(bool), "allocation"].astype(str)) if not treatment.empty else set()
    design = cells.loc[cells["allocation"].isin(passing_names)].copy() if not cells.empty else pd.DataFrame()
    design.to_csv(root / "allocation_qbd_design_space.csv", index=False)
    if not cells.empty:
        _write_heatmaps(root, cells)
    (root / "allocation_qbd_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    _write_markdown(root, summary, treatment)
    return summary
