from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import multiprocessing as mp
import os
from pathlib import Path
from typing import Iterator

import pandas as pd

from . import portfolio_policy_walk_forward as multicore_walk_forward, portfolio_policy_search as search
from .allocation_qbd_evaluate import load_primary_phase1_design_space
from .affinity_coordinator_pool import AffinityCoordinatorPool
from .portfolio_research_inputs import load_predictions, load_price_panel
from .portfolio_replay_fragment_cache import build_fragment_namespace
from .portfolio_replay_process_backend import backend_stats, install_multicore_backend, shutdown_multicore_backend
from .portfolio_resilient_process_pool import install_resilient_process_pool, restore_process_pool_runner
from .replacement_qbd_evaluate import load_phase2_allocation_lock
from . import replacement_qbd_surface as phase3_surface
from .concentration_replacement_qbd_contract import (
    ALLOCATION_FIXED, BASELINE_REPLACEMENT, CONTRACT_ID, DEFAULT_MAX_NAMES,
    DEFAULT_REPLACEMENTS, EXIT_FAMILY_FIXED, EXIT_VALUE_FIXED, SLEEVE_FIXED,
    parse_max_names, parse_replacement_treatment,
)
from .concentration_replacement_qbd_evaluate import load_phase3_provenance, treatment_slug, write_evaluation_artifacts

ENTRY_GRID_SIZE = len(search.SEARCH_QUANTILES) * len(search.SEARCH_TOP_FRACTIONS)
_ORIGINAL_COORDINATOR_EXECUTOR = multicore_walk_forward.ThreadPoolExecutor


@contextmanager
def concentration_search_contract(horizon: int, holding_days: int, max_names: int, replacement: str) -> Iterator[None]:
    """Reuse Phase-3 search hardening while making max_names an explicit fixed cell factor."""
    h, d, n = int(horizon), int(holding_days), parse_max_names(max_names)
    tx = parse_replacement_treatment(replacement)
    old_names = search.SEARCH_MAX_NAMES
    old_size = phase3_surface.ENTRY_GRID_SIZE
    old_contract = phase3_surface.CONTRACT_ID
    search.SEARCH_MAX_NAMES = (n,)
    phase3_surface.ENTRY_GRID_SIZE = ENTRY_GRID_SIZE
    phase3_surface.CONTRACT_ID = CONTRACT_ID
    try:
        with phase3_surface.replacement_search_contract(h, d, tx):
            original_choose = search.choose_policy

            def choose_fixed(*args, **kwargs):
                chosen, result, leaderboard, meta = original_choose(*args, **kwargs)
                if int(chosen.max_names) != n:
                    raise RuntimeError(f"CONCENTRATION_QBD_MAX_NAMES_ESCAPED_CELL:N{n}:actual={chosen.max_names}")
                return chosen, result, leaderboard, dict(meta) | {
                    "max_names_fixed": n,
                    "max_names_removed_from_inner_search": True,
                    "phase3_replacement_reopened": True,
                }

            search.choose_policy = choose_fixed
            search._window_checkpoint_key = lambda requested_horizon, fold, history_end, budget: search._cache_key(
                CONTRACT_ID, "outer_window", h, d, n, tx, ALLOCATION_FIXED,
                int(requested_horizon), str(fold), str(pd.Timestamp(history_end)), int(budget)
            )
            search._final_fit_checkpoint_key = lambda requested_horizon, history_end, budget: search._cache_key(
                CONTRACT_ID, "final_fit", h, d, n, tx, ALLOCATION_FIXED,
                int(requested_horizon), str(pd.Timestamp(history_end)), int(budget)
            )
            search._horizon_checkpoint_key = lambda requested_horizon, budget: search._cache_key(
                CONTRACT_ID, "cell", h, d, n, tx, ALLOCATION_FIXED, int(requested_horizon), int(budget)
            )
            yield
    finally:
        search.SEARCH_MAX_NAMES = old_names
        phase3_surface.ENTRY_GRID_SIZE = old_size
        phase3_surface.CONTRACT_ID = old_contract


def concentration_grid(horizon: int, holding_days: int, max_names: int, replacement: str):
    with concentration_search_contract(horizon, holding_days, max_names, replacement):
        return list(search.grid(int(horizon)))


def _cell_path(root: Path, h: int, d: int, n: int, replacement: str) -> Path:
    return root / "cells" / f"H{h:02d}_D{d:02d}__N{n}__{treatment_slug(replacement)}.json"


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    tmp.replace(path)


def _load_complete(path: Path, h: int, d: int, n: int, replacement: str) -> dict | None:
    if not path.is_file(): return None
    try: p = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError): return None
    valid = (
        p.get("contract_id") == CONTRACT_ID and p.get("status") == "COMPLETE"
        and int(p.get("prediction_horizon", -1)) == h and int(p.get("holding_days", -1)) == d
        and int(p.get("max_names", -1)) == n and str(p.get("replacement")) == replacement
        and p.get("max_names_removed_from_inner_search") is True and p.get("phase3_replacement_reopened") is True
        and str(p.get("allocation")) == ALLOCATION_FIXED and abs(float(p.get("sleeve", -1)) - SLEEVE_FIXED) <= 1e-12
        and str(p.get("exit_family")) == EXIT_FAMILY_FIXED and abs(float(p.get("exit_value", 99)) - EXIT_VALUE_FIXED) <= 1e-12
        and p.get("final_holdout_opened") is False and p.get("v45_exit_overlay_used") is False
    )
    return p if valid else None


def _run_cell(hpred, prices, *, h: int, d: int, n: int, replacement: str, budget: int, workers: int, cache_path, namespace):
    with concentration_search_contract(h, d, n, replacement):
        outer, _history, meta = multicore_walk_forward.run_walk_forward(
            hpred, prices, horizons=(h,), budget=budget, max_workers=workers,
            fragment_cache_path=cache_path, fragment_namespace=namespace,
        )
        final_policy, threshold = phase3_surface._validate_cell(h, d, replacement, outer, meta)
    if int(final_policy["max_names"]) != n or any(int(row.get("max_names", -1)) != n for row in outer):
        raise RuntimeError(f"CONCENTRATION_QBD_POLICY_ESCAPED_MAX_NAMES:N{n}")
    return {
        "contract_id": CONTRACT_ID, "status": "COMPLETE", "prediction_horizon": h,
        "holding_days": d, "max_names": n, "replacement": replacement,
        "entry_grid_size": ENTRY_GRID_SIZE, "max_names_removed_from_inner_search": True,
        "phase1_holding_and_prediction_frozen": True, "phase2_allocation_frozen": True,
        "phase3_replacement_reopened": True, "allocation": ALLOCATION_FIXED, "sleeve": SLEEVE_FIXED,
        "exit_family": EXIT_FAMILY_FIXED, "exit_value": EXIT_VALUE_FIXED,
        "v45_exit_overlay_used": False, "final_holdout_opened": False,
        "roundtrip_cost_stress_in_phase4": False, "tax_stress_in_phase4": False,
        "entry_grid_dimensions": {"score_quantile": list(search.SEARCH_QUANTILES), "top_fraction": list(search.SEARCH_TOP_FRACTIONS), "max_names": [n]},
        "outer_rows": outer, "final_policy": final_policy, "final_threshold": threshold,
        "search_coverage": meta.get("search_coverage", {}).get(h, []), "performance": meta.get("performance", {}),
    }


def parse_max_names_values(raw: str | None) -> tuple[int, ...]:
    if not raw or not raw.strip(): return DEFAULT_MAX_NAMES
    return tuple(dict.fromkeys(parse_max_names(x.strip()) for x in raw.split(",") if x.strip()))


def parse_replacements(raw: str | None) -> tuple[str, ...]:
    if not raw or not raw.strip(): return DEFAULT_REPLACEMENTS
    values = tuple(dict.fromkeys(parse_replacement_treatment(x.strip()) for x in raw.split(",") if x.strip()))
    return values if BASELINE_REPLACEMENT in values else (BASELINE_REPLACEMENT,) + values


def _self_test() -> None:
    assert ENTRY_GRID_SIZE == 12
    assert {p.max_names for p in concentration_grid(24, 5, 4, "IGNORE_NEW")} == {4}
    with concentration_search_contract(24, 5, 1, "IGNORE_NEW"):
        a = search._horizon_checkpoint_key(24, ENTRY_GRID_SIZE)
    with concentration_search_contract(24, 5, 2, "IGNORE_NEW"):
        b = search._horizon_checkpoint_key(24, ENTRY_GRID_SIZE)
    with concentration_search_contract(24, 5, 1, "REPLACE_WEAKEST"):
        c = search._horizon_checkpoint_key(24, ENTRY_GRID_SIZE)
    assert len({a, b, c}) == 3
    print("CONCENTRATION_REPLACEMENT_QBD_SURFACE_SELF_TEST_PASS")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase-4 Concentration x Replacement QbD")
    p.add_argument("--v5-predictions", default=""); p.add_argument("--daily-store-root", default="")
    p.add_argument("--phase1-design-space", default="artifacts/prediction-hold-qbd-surface/qbd_design_space.csv")
    p.add_argument("--phase2-summary", default="artifacts/allocation-qbd/allocation_qbd_summary.json")
    p.add_argument("--phase2-treatment-summary", default="artifacts/allocation-qbd/allocation_qbd_treatment_summary.csv")
    p.add_argument("--phase3-summary", default="artifacts/replacement-qbd/replacement_qbd_summary.json")
    p.add_argument("--phase3-treatment-summary", default="artifacts/replacement-qbd/replacement_qbd_treatment_summary.csv")
    p.add_argument("--output-root", default="artifacts/concentration-replacement-qbd")
    p.add_argument("--max-names", default="1,2,3,4,5"); p.add_argument("--replacements", default="")
    p.add_argument("--stage-a-budget", type=int, default=ENTRY_GRID_SIZE); p.add_argument("--max-workers", type=int, default=24)
    p.add_argument("--coordinator-threads", type=int, default=4); p.add_argument("--force", action="store_true")
    p.add_argument("--stop-on-error", action="store_true"); p.add_argument("--fine-grained-fragment-cache", action="store_true")
    p.add_argument("--self-test", action="store_true")
    return p.parse_args()


def main(args: argparse.Namespace | None = None) -> int:
    args = args or parse_args()
    if args.self_test: _self_test(); return 0
    if not args.v5_predictions or not args.daily_store_root: raise ValueError("--v5-predictions and --daily-store-root are required")
    if not 1 <= int(args.max_workers) <= 24 or not 1 <= int(args.coordinator_threads) <= 8: raise ValueError("worker range invalid")
    if load_phase2_allocation_lock(Path(args.phase2_summary), Path(args.phase2_treatment_summary)) != ALLOCATION_FIXED:
        raise RuntimeError("CONCENTRATION_QBD_PHASE2_ALLOCATION_NOT_EQUAL_ACTIVE")
    load_phase3_provenance(Path(args.phase3_summary), Path(args.phase3_treatment_summary))
    names, replacements = parse_max_names_values(args.max_names), parse_replacements(args.replacements)
    design = load_primary_phase1_design_space(Path(args.phase1_design_space))
    requested = [(h,d,n,tx) for h,d in design for n in names for tx in replacements]
    horizons = sorted({h for h,_,_,_ in requested}); root = Path(args.output_root); root.mkdir(parents=True, exist_ok=True)
    predictions, prediction_audit = load_predictions(Path(args.v5_predictions))
    missing = sorted(set(horizons) - set(int(x) for x in predictions.horizon.unique()))
    if missing: raise RuntimeError(f"CONCENTRATION_QBD_PREDICTION_HORIZONS_MISSING:{missing}")
    tickers = set(predictions.loc[predictions.horizon.isin(horizons), "ticker"].unique())
    prices, price_audit = load_price_panel(Path(args.daily_store_root), tickers)
    for name in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS","BLIS_NUM_THREADS"): os.environ[name]="1"
    os.environ["OPPORTUNITY_WINDOW_PIPELINE"] = str(args.coordinator_threads)
    os.environ.setdefault("OPPORTUNITY_TELEMETRY_PATH", str(root / "concentration_replacement_qbd_multicore_telemetry.jsonl"))
    fine_cache = bool(args.fine_grained_fragment_cache); install_multicore_backend(args.max_workers); install_resilient_process_pool()
    multicore_walk_forward.ThreadPoolExecutor = AffinityCoordinatorPool
    failed=[]; reused=0; complete=0
    try:
        for h in horizons:
            hpred = predictions.loc[predictions.horizon.eq(h)].copy()
            ns_base = build_fragment_namespace(search._research_input_fingerprint(hpred, prices)) if fine_cache else None
            cache_path = root / "concentration_replacement_qbd_fragment_cache.sqlite3" if fine_cache else None
            for _h,d,n,tx in [x for x in requested if x[0] == h]:
                path = _cell_path(root,h,d,n,tx)
                if not args.force and _load_complete(path,h,d,n,tx) is not None:
                    reused += 1; complete += 1; print(f"[concentration-qbd] H{h:02d}/D{d:02d}/N{n}/{tx}: RESUME", flush=True); continue
                print(f"[concentration-qbd] H{h:02d}/D{d:02d}/N{n}/{tx}: measure {complete+len(failed)+1}/{len(requested)}; entry_grid={ENTRY_GRID_SIZE}", flush=True)
                namespace = f"{ns_base}:{CONTRACT_ID}:{h}:{d}:{n}:{tx}" if ns_base else None
                try:
                    payload = _run_cell(hpred, prices, h=h,d=d,n=n,replacement=tx,budget=args.stage_a_budget,workers=args.max_workers,cache_path=cache_path,namespace=namespace)
                    _write_json(path,payload); complete += 1
                except Exception as exc:
                    failure={"contract_id":CONTRACT_ID,"status":"FAILED","prediction_horizon":h,"holding_days":d,"max_names":n,"replacement":tx,"error":f"{type(exc).__name__}:{exc}","allocation":ALLOCATION_FIXED,"sleeve":SLEEVE_FIXED,"exit_family":EXIT_FAMILY_FIXED,"exit_value":EXIT_VALUE_FIXED,"max_names_removed_from_inner_search":True,"phase3_replacement_reopened":True,"v45_exit_overlay_used":False,"final_holdout_opened":False}
                    _write_json(path,failure); failed.append(failure)
                    if args.stop_on_error: raise
        evaluation = write_evaluation_artifacts(root, phase1_design_space=Path(args.phase1_design_space), phase3_summary=Path(args.phase3_summary), phase3_treatment_summary=Path(args.phase3_treatment_summary), max_names_values=names, replacements=replacements)
        summary={"contract_id":CONTRACT_ID,"status":"COMPLETE" if evaluation["qbd_complete"] and not failed else "INCOMPLETE","phase1_design_cells":len(design),"phase2_allocation_fixed":ALLOCATION_FIXED,"phase3_replacement_reopened":True,"max_names_values":list(names),"replacements":list(replacements),"requested_surface_cells":len(requested),"completed_cells_this_or_prior_run":complete,"reused_complete_cells":reused,"failed_cells_this_run":failed,"entry_grid_size_per_cell":ENTRY_GRID_SIZE,"max_names_removed_from_inner_search":True,"cost_and_tax_stress_deferred":True,"final_holdout_opened":False,"final_holdout_locked":True,"interpolation_used":False,"prediction_audit":prediction_audit,"price_audit":price_audit,"backend":backend_stats(),"evaluation":evaluation}
        _write_json(root / "concentration_replacement_qbd_run_summary.json", summary); print(json.dumps(summary,indent=2,sort_keys=True,default=str)); return 0 if summary["status"]=="COMPLETE" else 2
    finally:
        multicore_walk_forward.ThreadPoolExecutor = _ORIGINAL_COORDINATOR_EXECUTOR; restore_process_pool_runner(); shutdown_multicore_backend()


if __name__ == "__main__":
    mp.freeze_support(); raise SystemExit(main())
