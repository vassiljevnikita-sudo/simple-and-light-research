from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .allocation_qbd_evaluate import load_primary_phase1_design_space
from .prediction_hold_qbd_evaluate import _aggregate_cell
from .replacement_qbd_contract import (
    ALLOCATION_FIXED,
    BASELINE,
    CONTRACT_ID,
    DEFAULT_TREATMENTS,
    PHASE2_CONTRACT_ID,
)


def treatment_slug(value: str) -> str:
    return str(value).replace(":", "_").replace(".", "p").replace("/", "_")


def _as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    return series.astype(str).str.lower().isin(("true", "1", "yes"))


def load_phase2_allocation_lock(
    summary_path: Path,
    treatment_summary_path: Path,
) -> str:
    summary = json.loads(Path(summary_path).read_text(encoding="utf-8"))
    if summary.get("contract_id") != PHASE2_CONTRACT_ID:
        raise ValueError("REPLACEMENT_QBD_PHASE2_CONTRACT_MISMATCH")
    if summary.get("qbd_complete") is not True:
        raise ValueError("REPLACEMENT_QBD_PHASE2_NOT_COMPLETE")
    if summary.get("final_holdout_opened") is not False:
        raise ValueError("REPLACEMENT_QBD_PHASE2_HOLDOUT_WAS_OPENED")
    if summary.get("interpolation_used") is not False:
        raise ValueError("REPLACEMENT_QBD_PHASE2_INTERPOLATION_USED")
    if summary.get("v45_exit_overlay_used") is not False:
        raise ValueError("REPLACEMENT_QBD_PHASE2_V45_USED")

    treatment = pd.read_csv(Path(treatment_summary_path))
    required = {"allocation", "allocation_qbd_pass", "baseline_reference"}
    missing = sorted(required - set(treatment.columns))
    if missing:
        raise ValueError(
            f"REPLACEMENT_QBD_PHASE2_TREATMENT_COLUMNS_MISSING:{','.join(missing)}"
        )
    passed = treatment.loc[_as_bool(treatment["allocation_qbd_pass"])].copy()
    if len(passed) != 1:
        raise ValueError(
            f"REPLACEMENT_QBD_PHASE2_PASS_COUNT_INVALID:{len(passed)}"
        )
    row = passed.iloc[0]
    if str(row["allocation"]) != ALLOCATION_FIXED:
        raise ValueError(
            f"REPLACEMENT_QBD_PHASE2_ALLOCATION_LOCK_MISMATCH:{row['allocation']}"
        )
    if not bool(_as_bool(pd.Series([row["baseline_reference"]])).iloc[0]):
        raise ValueError("REPLACEMENT_QBD_PHASE2_WINNER_NOT_BASELINE")
    if summary.get("best_nonbaseline_treatment") is not None:
        raise ValueError("REPLACEMENT_QBD_PHASE2_NONBASELINE_WINNER_PRESENT")
    return ALLOCATION_FIXED


def expected_cells(
    design_cells: Iterable[tuple[int, int]],
    treatments: Iterable[str] = DEFAULT_TREATMENTS,
) -> list[tuple[int, int, str]]:
    cells = sorted({(int(h), int(d)) for h, d in design_cells})
    tx = tuple(dict.fromkeys(str(x) for x in treatments))
    if not cells or not tx or BASELINE not in tx:
        raise ValueError("REPLACEMENT_QBD_EXPECTED_CELLS_INVALID")
    return [(h, d, treatment) for h, d in cells for treatment in tx]


def validate_surface_coverage(
    status: pd.DataFrame,
    expected: Iterable[tuple[int, int, str]],
) -> dict:
    wanted = set(expected)
    keys = ["prediction_horizon", "holding_days", "replacement"]
    if status.empty:
        observed, complete, duplicates = set(), set(), []
    else:
        frame = status.copy()
        frame["prediction_horizon"] = frame["prediction_horizon"].astype(int)
        frame["holding_days"] = frame["holding_days"].astype(int)
        frame["replacement"] = frame["replacement"].astype(str)
        dup = frame.duplicated(keys, keep=False)
        duplicates = frame.loc[dup, keys].drop_duplicates().to_dict("records")
        observed = set(map(tuple, frame[keys].itertuples(index=False, name=None)))
        ok = frame["status"].astype(str).eq("COMPLETE")
        complete = set(
            map(tuple, frame.loc[ok, keys].itertuples(index=False, name=None))
        )
    missing = sorted(wanted - observed)
    failed = sorted(wanted - complete)
    unexpected = sorted(observed - wanted)

    def pack(rows):
        return [
            {
                "prediction_horizon": int(h),
                "holding_days": int(d),
                "replacement": str(replacement),
            }
            for h, d, replacement in rows
        ]

    return {
        "expected_cells": len(wanted),
        "observed_cells": len(observed),
        "complete_cells": len(complete & wanted),
        "missing_cells": pack(missing),
        "failed_or_incomplete_cells": pack(failed),
        "unexpected_cells": pack(unexpected),
        "duplicate_cells": duplicates,
        "surface_complete": not missing and not failed and not unexpected and not duplicates,
    }


def aggregate_outer_results(rows: pd.DataFrame) -> pd.DataFrame:
    required = {
        "prediction_horizon",
        "holding_days",
        "replacement",
        "cagr_excess",
        "trade_count",
    }
    missing = sorted(required - set(rows.columns))
    if missing:
        raise ValueError(
            f"REPLACEMENT_QBD_OUTER_COLUMNS_MISSING:{','.join(missing)}"
        )
    out = []
    for (_h, _d, replacement), group in rows.groupby(
        ["prediction_horizon", "holding_days", "replacement"], sort=True
    ):
        row = _aggregate_cell(group)
        row["replacement"] = str(replacement)
        out.append(row)
    return (
        pd.DataFrame(out)
        .sort_values(["prediction_horizon", "holding_days", "replacement"])
        .reset_index(drop=True)
    )


def add_paired_baseline(cells: pd.DataFrame) -> pd.DataFrame:
    result = cells.copy()
    baseline = result.loc[
        result["replacement"].eq(BASELINE),
        [
            "prediction_horizon",
            "holding_days",
            "median_active_cagr_excess",
            "q25_active_cagr_excess",
            "worst_active_cagr_excess",
            "median_turnover",
        ],
    ].rename(
        columns={
            "median_active_cagr_excess": "baseline_median_active_cagr_excess",
            "q25_active_cagr_excess": "baseline_q25_active_cagr_excess",
            "worst_active_cagr_excess": "baseline_worst_active_cagr_excess",
            "median_turnover": "baseline_median_turnover",
        }
    )
    result = result.merge(
        baseline,
        on=["prediction_horizon", "holding_days"],
        how="left",
    )
    if result["baseline_median_active_cagr_excess"].isna().any():
        raise ValueError("REPLACEMENT_QBD_BASELINE_CELL_MISSING")
    result["delta_median_active_cagr_excess"] = (
        result["median_active_cagr_excess"]
        - result["baseline_median_active_cagr_excess"]
    )
    result["delta_q25_active_cagr_excess"] = (
        result["q25_active_cagr_excess"]
        - result["baseline_q25_active_cagr_excess"]
    )
    result["delta_turnover_vs_ignore_new"] = (
        result["median_turnover"] - result["baseline_median_turnover"]
    )
    result["beats_ignore_new"] = result["delta_median_active_cagr_excess"] > 0.0
    result.loc[result["replacement"].eq(BASELINE), "beats_ignore_new"] = False
    return result


def summarize_treatments(cells: pd.DataFrame, design_cell_count: int) -> pd.DataFrame:
    rows = []
    for replacement, group in cells.groupby("replacement", sort=True):
        deltas = group["delta_median_active_cagr_excess"].astype(float)
        q25_deltas = group["delta_q25_active_cagr_excess"].astype(float)
        medians = group["median_active_cagr_excess"].astype(float)
        turnover_delta = group["delta_turnover_vs_ignore_new"].astype(float)
        robust_fraction = float(group["robust_gate_pass"].astype(bool).mean())
        positive_fraction = float((medians > 0.0).mean())
        if replacement == BASELINE:
            beat_fraction = 0.0
            treatment_pass = bool(
                len(group) == design_cell_count
                and robust_fraction >= 0.50
                and positive_fraction >= 0.50
            )
        else:
            beat_fraction = float((deltas > 0.0).mean())
            treatment_pass = bool(
                len(group) == design_cell_count
                and robust_fraction >= 0.50
                and beat_fraction >= 0.60
                and float(deltas.median()) > 0.0
                and float(np.quantile(deltas, 0.25)) >= 0.0
            )
        rows.append(
            {
                "replacement": replacement,
                "cells": int(len(group)),
                "expected_cells": int(design_cell_count),
                "complete_across_phase1_design_space": bool(
                    len(group) == design_cell_count
                ),
                "robust_cells": int(group["robust_gate_pass"].astype(bool).sum()),
                "robust_cell_fraction": robust_fraction,
                "positive_cell_fraction": positive_fraction,
                "median_cell_median_active_cagr_excess": float(medians.median()),
                "q25_cell_median_active_cagr_excess": float(
                    np.quantile(medians, 0.25)
                ),
                "worst_cell_median_active_cagr_excess": float(medians.min()),
                "median_delta_vs_ignore_new": float(deltas.median()),
                "q25_delta_vs_ignore_new": float(np.quantile(deltas, 0.25)),
                "worst_delta_vs_ignore_new": float(deltas.min()),
                "beat_ignore_new_cell_fraction": beat_fraction,
                "median_q25_delta_vs_ignore_new": float(q25_deltas.median()),
                "median_turnover_delta_vs_ignore_new": float(turnover_delta.median()),
                "total_trades": int(group["trade_count"].sum()),
                "median_turnover": float(group["median_turnover"].median()),
                "replacement_qbd_pass": treatment_pass,
                "baseline_reference": replacement == BASELINE,
            }
        )
    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary
    summary = summary.sort_values(
        [
            "replacement_qbd_pass",
            "q25_delta_vs_ignore_new",
            "median_delta_vs_ignore_new",
            "robust_cell_fraction",
            "worst_delta_vs_ignore_new",
        ],
        ascending=[False, False, False, False, False],
    ).reset_index(drop=True)
    summary["treatment_rank"] = np.arange(1, len(summary) + 1)
    return summary


def evaluate_surface(
    outer: pd.DataFrame,
    status: pd.DataFrame,
    design_cells: list[tuple[int, int]],
    treatments: Iterable[str],
):
    wanted = expected_cells(design_cells, treatments)
    coverage = validate_surface_coverage(status, wanted)
    cells = aggregate_outer_results(outer) if not outer.empty else pd.DataFrame()
    if not cells.empty:
        cells = add_paired_baseline(cells)
        treatment = summarize_treatments(cells, len(design_cells))
    else:
        treatment = pd.DataFrame()
    aggregates = (
        set(
            map(
                tuple,
                cells[
                    ["prediction_horizon", "holding_days", "replacement"]
                ].itertuples(index=False, name=None),
            )
        )
        if not cells.empty
        else set()
    )
    complete = (
        set(
            map(
                tuple,
                status.loc[
                    status["status"].astype(str).eq("COMPLETE"),
                    ["prediction_horizon", "holding_days", "replacement"],
                ].itertuples(index=False, name=None),
            )
        )
        if not status.empty
        else set()
    )
    parity = aggregates == complete
    passing = (
        treatment.loc[treatment["replacement_qbd_pass"].astype(bool)]
        if not treatment.empty
        else pd.DataFrame()
    )
    nonbaseline_passing = (
        passing.loc[~passing["baseline_reference"].astype(bool)]
        if not passing.empty
        else pd.DataFrame()
    )
    best = (
        nonbaseline_passing.iloc[0].to_dict()
        if not nonbaseline_passing.empty
        else None
    )
    summary = {
        "contract_id": CONTRACT_ID,
        "phase1_design_cells": len(design_cells),
        "phase2_allocation_fixed": ALLOCATION_FIXED,
        "phase2_allocation_frozen": True,
        "treatments": list(dict.fromkeys(str(x) for x in treatments)),
        "expected_surface_cells": len(wanted),
        "surface_coverage": coverage,
        "aggregate_complete_cell_parity": bool(parity),
        "aggregated_cells": len(cells),
        "passing_treatments": int(
            treatment["replacement_qbd_pass"].sum()
        )
        if not treatment.empty
        else 0,
        "best_nonbaseline_treatment": best,
        "selection_principle": (
            "paired broad stability across frozen Phase-1 H/D design space with "
            "Phase-2 EQUAL_ACTIVE fixed; not maximum CAGR"
        ),
        "final_holdout_opened": False,
        "final_holdout_locked": True,
        "interpolation_used": False,
        "v45_exit_overlay_used": False,
        "qbd_complete": bool(coverage["surface_complete"] and parity),
    }
    return cells, treatment, summary


def load_cell_artifacts(root: Path):
    status_rows, outer_rows, final_rows = [], [], []
    for path in sorted((Path(root) / "cells").glob("H*_D*__*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        h = int(payload["prediction_horizon"])
        d = int(payload["holding_days"])
        replacement = str(payload["replacement"])
        state = str(payload.get("status", "UNKNOWN"))
        status_rows.append(
            {
                "prediction_horizon": h,
                "holding_days": d,
                "replacement": replacement,
                "status": state,
                "error": payload.get("error"),
                "artifact": str(path),
            }
        )
        if state != "COMPLETE":
            continue
        outer_rows.extend(
            {
                **row,
                "prediction_horizon": h,
                "holding_days": d,
                "replacement": replacement,
            }
            for row in payload.get("outer_rows", [])
        )
        if payload.get("final_policy"):
            final_rows.append(
                {
                    "prediction_horizon": h,
                    "holding_days": d,
                    "replacement": replacement,
                    **payload["final_policy"],
                    "resolved_threshold": payload.get("final_threshold"),
                }
            )
    return (
        pd.DataFrame(status_rows),
        pd.DataFrame(outer_rows),
        pd.DataFrame(final_rows),
    )


def _write_heatmaps(root: Path, cells: pd.DataFrame) -> None:
    heat = root / "heatmaps"
    heat.mkdir(parents=True, exist_ok=True)
    for replacement, group in cells.groupby("replacement", sort=True):
        slug = treatment_slug(replacement)
        for metric in (
            "median_active_cagr_excess",
            "delta_median_active_cagr_excess",
            "delta_turnover_vs_ignore_new",
            "robust_gate_pass",
        ):
            table = group.pivot(
                index="prediction_horizon",
                columns="holding_days",
                values=metric,
            )
            table.to_csv(heat / f"{metric}__{slug}.csv")


def write_evaluation_artifacts(
    root: Path,
    *,
    design_space_path: Path,
    phase2_summary_path: Path,
    phase2_treatment_path: Path,
    treatments: Iterable[str],
) -> dict:
    design_cells = load_primary_phase1_design_space(design_space_path)
    allocation = load_phase2_allocation_lock(
        phase2_summary_path, phase2_treatment_path
    )
    if allocation != ALLOCATION_FIXED:
        raise ValueError("REPLACEMENT_QBD_PHASE2_ALLOCATION_NOT_FROZEN")
    status, outer, final = load_cell_artifacts(root)
    cells, treatment, summary = evaluate_surface(
        outer, status, design_cells, treatments
    )
    root = Path(root)
    status.to_csv(root / "replacement_qbd_cell_status.csv", index=False)
    outer.to_csv(root / "replacement_qbd_outer_results.csv", index=False)
    final.to_csv(root / "replacement_qbd_final_policy.csv", index=False)
    cells.to_csv(root / "replacement_qbd_cells.csv", index=False)
    treatment.to_csv(root / "replacement_qbd_treatment_summary.csv", index=False)
    _write_heatmaps(root, cells)
    summary = {
        **summary,
        "phase1_design_space_path": str(design_space_path),
        "phase2_summary_path": str(phase2_summary_path),
        "phase2_treatment_summary_path": str(phase2_treatment_path),
        "phase2_allocation_fixed": allocation,
    }
    (root / "replacement_qbd_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    return summary
