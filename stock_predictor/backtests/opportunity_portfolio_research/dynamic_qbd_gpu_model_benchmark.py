"""Isolated CPU versus heterogeneous OpenCL model-fit benchmark.

This harness has no authority to alter a causal run.  It measures a
semantics-matched double-precision Ridge normal-equation path on every OpenCL
GPU and reports sklearn HistGradientBoosting as unsupported unless an exactly
equivalent backend exists.  Outputs are diagnostic development artifacts.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import date
import json
import os
from pathlib import Path
import time
from typing import Any, Mapping

import numpy as np
import pandas as pd

from stock_predictor.v5.train import DeterministicEncoder
from .candidate_oos import (CandidateOosFactory, load_primary_candidate_registry,
                            read_development_panel)
from .qbd_training_selection_contracts import FoldPolicy, TargetContract


SCHEMA_VERSION = "DQBD_GPU_MODEL_BENCHMARK_V1"
HOLDOUT_BOUNDARY = date(2026, 7, 25)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
                    encoding="utf-8")


def opencl_devices() -> list[dict[str, Any]]:
    try:
        import pyopencl as cl
    except ImportError:
        return []
    rows = []
    for platform_index, platform in enumerate(cl.get_platforms()):
        for device_index, device in enumerate(platform.get_devices(device_type=cl.device_type.GPU)):
            rows.append({
                "platform_index": platform_index,
                "device_index": device_index,
                "platform": platform.name.strip(),
                "vendor": device.vendor.strip(),
                "name": device.name.strip(),
                "global_memory_gib": float(device.global_mem_size) / (1024 ** 3),
                "compute_units": int(device.max_compute_units),
                "double_precision": "cl_khr_fp64" in str(device.extensions).split(),
            })
    return rows


def _device(row: Mapping[str, Any]):
    import pyopencl as cl
    platform = cl.get_platforms()[int(row["platform_index"])]
    return platform.get_devices(device_type=cl.device_type.GPU)[int(row["device_index"])]


def _prepare_matrix(*, signal_panel: Path, feature_schema_path: Path,
                    hyperparameter_space: Path, horizon: int, fold_id: str,
                    development_end: date, max_rows: int | None,
                    scratch_root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    feature_schema = json.loads(feature_schema_path.read_text(encoding="utf-8"))
    candidates = load_primary_candidate_registry(hyperparameter_space)
    factory = CandidateOosFactory(
        signal_panel=signal_panel, feature_schema=feature_schema, candidates=candidates,
        output_root=scratch_root / "unused-candidate-store",
        fold_policy=FoldPolicy(), target_contract=TargetContract(cost_model={"roundtrip_bps": 20.0}),
        development_end=development_end, holdout_boundary=HOLDOUT_BOUNDARY,
        random_seed=17, model_training_contract_hash="GPU_BENCHMARK_NO_RUN_AUTHORITY")
    folds = {fold.fold_id: fold for fold in factory.fold_specs(
        horizon=horizon, development_end=development_end,
        holdout_boundary=HOLDOUT_BOUNDARY)}
    if fold_id not in folds:
        raise ValueError(f"GPU_BENCHMARK_FOLD_NOT_FOUND:{fold_id}")
    fold = folds[fold_id]
    frame = factory._load(horizon, development_end=development_end,
                          holdout_boundary=HOLDOUT_BOUNDARY)  # diagnostic harness
    date_values = pd.to_datetime(frame["decision_date"]).dt.date.astype(str)
    train = frame.loc[date_values.isin(fold.train_dates)].copy()
    sessions = sorted(set(date_values))
    session_index = {value: index for index, value in enumerate(sessions)}
    eligible_dates = [value for value in pd.to_datetime(train["decision_date"]).dt.date.astype(str)
                      if session_index[value] + horizon < len(sessions)]
    label_column = factory.target_contract.target_column(horizon)
    labels = read_development_panel(
        path=signal_panel, columns=["decision_date", "ticker", label_column],
        development_end=development_end, holdout_boundary=HOLDOUT_BOUNDARY,
        decision_dates=eligible_dates)
    train = train.merge(labels, on=["decision_date", "ticker"], how="inner",
                        validate="one_to_one")
    if max_rows and len(train) > max_rows:
        # Deterministic evenly-spaced sample retains the temporal span.
        positions = np.linspace(0, len(train) - 1, int(max_rows), dtype=np.int64)
        train = train.iloc[positions].copy()
    rows = []
    y_reg = []
    y_positive = []
    y_downside = []
    for raw in train.to_dict("records"):
        snapshot = {name: raw[name] for name in factory.feature_columns
                    if name in raw and pd.notna(raw[name])}
        outcome = float(raw[label_column])
        rows.append({"decision_date": str(raw["decision_date"]),
                     "ticker": str(raw["ticker"]), "isin": str(raw["ticker"]),
                     "sector": str(raw.get("sector", "Unknown")),
                     "sub_industry": str(raw.get("sub_industry", "Unknown")),
                     "feature_snapshot": snapshot})
        y_reg.append(outcome)
        y_positive.append(int(outcome > .002))
        y_downside.append(int(outcome < -.03))
    started = time.perf_counter()
    encoder = DeterministicEncoder().fit(rows)
    matrix = np.asarray(encoder.transform(rows).values, dtype=np.float64, order="C")
    encoding_seconds = time.perf_counter() - started
    metadata = {"horizon": horizon, "fold_id": fold_id, "rows": int(matrix.shape[0]),
                "features": int(matrix.shape[1]), "encoding_seconds": encoding_seconds,
                "development_end": development_end.isoformat(),
                "sampled": bool(max_rows and len(eligible_dates) > max_rows)}
    return (matrix, np.asarray(y_reg, dtype=np.float64),
            np.asarray(y_positive, dtype=np.int32),
            np.asarray(y_downside, dtype=np.int32), metadata)


def cpu_benchmark(matrix: np.ndarray, y_reg: np.ndarray, y_positive: np.ndarray,
                  y_downside: np.ndarray, *, ridge_parameters: Mapping[str, Any],
                  hgb_parameters: Mapping[str, Any], progress=None) -> tuple[dict, np.ndarray]:
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    from sklearn.linear_model import LogisticRegression, Ridge

    started = time.perf_counter()
    ridge = Ridge(alpha=float(ridge_parameters["alpha"]), random_state=17).fit(matrix, y_reg)
    ridge_seconds = time.perf_counter() - started
    reference = ridge.predict(matrix[: min(10_000, len(matrix))])

    started = time.perf_counter()
    if progress:
        progress("CPU_RIDGE_COMPLETE")
    LogisticRegression(C=float(ridge_parameters["positive_C"]), max_iter=2000,
                       random_state=17).fit(matrix, y_positive)
    positive_seconds = time.perf_counter() - started
    started = time.perf_counter()
    if progress:
        progress("CPU_POSITIVE_LOGISTIC_COMPLETE")
    LogisticRegression(C=float(ridge_parameters["downside_C"]), max_iter=2000,
                       random_state=17).fit(matrix, y_downside)
    downside_seconds = time.perf_counter() - started
    if progress:
        progress("CPU_RIDGE_BUNDLE_COMPLETE")

    common = {"learning_rate": float(hgb_parameters["learning_rate"]),
              "max_iter": int(hgb_parameters["max_iter"]),
              "max_leaf_nodes": int(hgb_parameters["max_leaf_nodes"]),
              "l2_regularization": float(hgb_parameters["l2_regularization"]),
              "random_state": 17}
    started = time.perf_counter()
    HistGradientBoostingRegressor(**common).fit(matrix, y_reg)
    hgb_reg_seconds = time.perf_counter() - started
    if progress:
        progress("CPU_HGB_REGRESSOR_COMPLETE")
    started = time.perf_counter()
    HistGradientBoostingClassifier(**common).fit(matrix, y_positive)
    hgb_positive_seconds = time.perf_counter() - started
    if progress:
        progress("CPU_HGB_POSITIVE_COMPLETE")
    started = time.perf_counter()
    HistGradientBoostingClassifier(**common).fit(matrix, y_downside)
    hgb_downside_seconds = time.perf_counter() - started
    if progress:
        progress("CPU_HGB_BUNDLE_COMPLETE")
    return ({"ridge_regressor_seconds": ridge_seconds,
             "logistic_positive_seconds": positive_seconds,
             "logistic_downside_seconds": downside_seconds,
             "ridge_bundle_seconds": ridge_seconds + positive_seconds + downside_seconds,
             "hgb_regressor_seconds": hgb_reg_seconds,
             "hgb_positive_seconds": hgb_positive_seconds,
             "hgb_downside_seconds": hgb_downside_seconds,
             "hgb_bundle_seconds": hgb_reg_seconds + hgb_positive_seconds + hgb_downside_seconds},
            np.asarray(reference, dtype=np.float64))


_OPENCL_SOURCE = r"""
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void gram_rhs(__global const double *x, __global const double *y,
                       __global double *gram, __global double *rhs,
                       const int rows, const int cols) {
    int i = get_global_id(0);
    int j = get_global_id(1);
    if (i >= cols || j >= cols) return;
    double total = 0.0;
    for (int row = 0; row < rows; ++row)
        total += x[row * cols + i] * x[row * cols + j];
    gram[i * cols + j] = total;
    if (j == 0) {
        double target = 0.0;
        for (int row = 0; row < rows; ++row)
            target += x[row * cols + i] * y[row];
        rhs[i] = target;
    }
}
"""


def _opencl_engine(device_row: Mapping[str, Any]):
    import pyopencl as cl

    device = _device(device_row)
    context = cl.Context([device])
    queue = cl.CommandQueue(context)
    program = cl.Program(context, _OPENCL_SOURCE).build()
    return context, queue, program


def gpu_ridge(matrix: np.ndarray, target: np.ndarray, *, alpha: float,
              device_row: Mapping[str, Any], engine=None) -> tuple[dict, np.ndarray]:
    import pyopencl as cl

    context, queue, program = engine or _opencl_engine(device_row)
    started = time.perf_counter()
    x_mean = matrix.mean(axis=0)
    y_mean = float(target.mean())
    centered_x = np.ascontiguousarray(matrix - x_mean, dtype=np.float64)
    centered_y = np.ascontiguousarray(target - y_mean, dtype=np.float64)
    rows, cols = centered_x.shape
    flags = cl.mem_flags
    x_buffer = cl.Buffer(context, flags.READ_ONLY | flags.COPY_HOST_PTR, hostbuf=centered_x)
    y_buffer = cl.Buffer(context, flags.READ_ONLY | flags.COPY_HOST_PTR, hostbuf=centered_y)
    gram = np.empty((cols, cols), dtype=np.float64)
    rhs = np.empty(cols, dtype=np.float64)
    gram_buffer = cl.Buffer(context, flags.WRITE_ONLY, gram.nbytes)
    rhs_buffer = cl.Buffer(context, flags.WRITE_ONLY, rhs.nbytes)
    program.gram_rhs(queue, (cols, cols), None, x_buffer, y_buffer,
                     gram_buffer, rhs_buffer, np.int32(rows), np.int32(cols))
    cl.enqueue_copy(queue, gram, gram_buffer)
    cl.enqueue_copy(queue, rhs, rhs_buffer).wait()
    transfer_and_kernel_seconds = time.perf_counter() - started
    solve_started = time.perf_counter()
    gram.flat[:: cols + 1] += float(alpha)
    coefficients = np.linalg.solve(gram, rhs)
    intercept = y_mean - float(x_mean @ coefficients)
    solve_seconds = time.perf_counter() - solve_started
    predictions = matrix[: min(10_000, len(matrix))] @ coefficients + intercept
    total_seconds = time.perf_counter() - started
    return ({"device": dict(device_row), "rows": rows, "features": cols,
             "transfer_center_kernel_seconds": transfer_and_kernel_seconds,
             "cpu_small_solve_seconds": solve_seconds, "total_seconds": total_seconds},
            np.asarray(predictions, dtype=np.float64))


def _pin_and_verify_cpu(cpu: int) -> None:
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        kernel32 = ctypes.windll.kernel32
        kernel32.GetCurrentProcess.argtypes = []
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        kernel32.SetProcessAffinityMask.argtypes = [wintypes.HANDLE, ctypes.c_size_t]
        kernel32.SetProcessAffinityMask.restype = wintypes.BOOL
        kernel32.GetProcessAffinityMask.argtypes = [
            wintypes.HANDLE, ctypes.POINTER(ctypes.c_size_t),
            ctypes.POINTER(ctypes.c_size_t)]
        kernel32.GetProcessAffinityMask.restype = wintypes.BOOL
        process = kernel32.GetCurrentProcess()
        requested = 1 << int(cpu)
        if not kernel32.SetProcessAffinityMask(process, requested):
            raise OSError("GPU_BENCHMARK_CPU_AFFINITY_SET_FAILED")
        process_mask = ctypes.c_size_t()
        system_mask = ctypes.c_size_t()
        if not kernel32.GetProcessAffinityMask(
                process, ctypes.byref(process_mask), ctypes.byref(system_mask)):
            raise OSError("GPU_BENCHMARK_CPU_AFFINITY_READ_FAILED")
        if int(process_mask.value) != requested:
            raise RuntimeError(
                f"GPU_BENCHMARK_CPU_AFFINITY_MISMATCH:{process_mask.value}:{requested}")
        return
    import psutil
    process = psutil.Process()
    process.cpu_affinity([int(cpu)])
    if process.cpu_affinity() != [int(cpu)]:
        raise RuntimeError("GPU_BENCHMARK_CPU_AFFINITY_MISMATCH")


def run(args: argparse.Namespace) -> dict:
    if args.cpu_affinity is not None:
        _pin_and_verify_cpu(int(args.cpu_affinity))
    output = Path(args.output_root)
    def progress(phase: str) -> None:
        _write_json(output / "progress.json", {
            "schema_version": SCHEMA_VERSION, "phase": phase,
            "pid": os.getpid(), "cpu_affinity": args.cpu_affinity,
        })

    progress("LOADING_REAL_FOLD")
    matrix, y_reg, y_positive, y_downside, data = _prepare_matrix(
        signal_panel=Path(args.signal_panel), feature_schema_path=Path(args.feature_schema),
        hyperparameter_space=Path(args.hyperparameter_space), horizon=args.horizon,
        fold_id=args.fold_id, development_end=date.fromisoformat(args.development_end),
        max_rows=args.max_rows, scratch_root=output)
    progress("REAL_FOLD_MATRIX_READY")
    hyperparameters = json.loads(Path(args.hyperparameter_space).read_text(encoding="utf-8"))["families"]
    ridge_parameters = hyperparameters["RIDGE_LOGISTIC"][args.ridge_candidate_index]
    hgb_parameters = hyperparameters["HIST_GRADIENT_BOOSTING"][args.hgb_candidate_index]
    cpu, reference = cpu_benchmark(
        matrix, y_reg, y_positive, y_downside, ridge_parameters=ridge_parameters,
        hgb_parameters=hgb_parameters, progress=progress)
    devices = [row for row in opencl_devices() if row["double_precision"]]
    progress("BUILDING_PERSISTENT_GPU_ENGINES")
    engines = [(row, _opencl_engine(row)) for row in devices]
    gpu_results = []
    predictions = []
    for row, engine in engines:
        progress(f"GPU_RIDGE_START:{row['name']}")
        result, prediction = gpu_ridge(
            matrix, y_reg, alpha=float(ridge_parameters["alpha"]), device_row=row,
            engine=engine)
        delta = prediction - reference
        result["equivalence"] = {
            "max_abs_prediction_delta": float(np.max(np.abs(delta))),
            "mean_abs_prediction_delta": float(np.mean(np.abs(delta))),
            "prediction_correlation": float(np.corrcoef(prediction, reference)[0, 1]),
            "allclose_rtol_1e_8_atol_1e_10": bool(np.allclose(
                prediction, reference, rtol=1e-8, atol=1e-10)),
        }
        gpu_results.append(result)
        predictions.append(prediction)
        progress(f"GPU_RIDGE_COMPLETE:{row['name']}")
    dual = None
    if len(engines) >= 2:
        started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(gpu_ridge, matrix, y_reg,
                                   alpha=float(ridge_parameters["alpha"]),
                                   device_row=row, engine=engine)
                       for row, engine in engines[:2]]
            pair = [future.result()[0] for future in futures]
        dual = {"wall_seconds_for_two_independent_ridge_jobs": time.perf_counter() - started,
                "devices": [row["name"] for row, _ in engines[:2]], "jobs": pair,
                "cpu_two_job_estimate_seconds": 2.0 * cpu["ridge_regressor_seconds"]}
    result = {"schema_version": SCHEMA_VERSION,
              "authority": "DIAGNOSTIC_ONLY_NO_CAUSAL_RUN_MUTATION",
              "data": data, "canonical_candidates": {
                  "ridge_index": args.ridge_candidate_index,
                  "ridge_parameters": ridge_parameters,
                  "hgb_index": args.hgb_candidate_index,
                  "hgb_parameters": hgb_parameters,
              }, "cpu": cpu, "gpu_ridge": gpu_results,
              "dual_gpu_throughput": dual,
              "hgb_gpu_status": "UNSUPPORTED_SAME_SEMANTICS_SKLEARN_HIST_GRADIENT_BOOSTING",
              "integration_gate": {
                  "prediction_equivalence_required": True,
                  "artifact_fingerprint_change_requires_versioned_contract": True,
                  "eligible_for_current_run": False}}
    _write_json(output / "summary.json", result)
    progress("COMPLETE")
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--signal-panel", required=True)
    parser.add_argument("--feature-schema", required=True)
    parser.add_argument("--hyperparameter-space", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--horizon", type=int, default=11)
    parser.add_argument("--fold-id", default="WF01")
    parser.add_argument("--development-end", default="2025-12-31")
    parser.add_argument("--max-rows", type=int, default=20000)
    parser.add_argument("--ridge-candidate-index", type=int, default=0)
    parser.add_argument("--hgb-candidate-index", type=int, default=0)
    parser.add_argument("--cpu-affinity", type=int, default=3)
    args = parser.parse_args(argv)
    result = run(args)
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
