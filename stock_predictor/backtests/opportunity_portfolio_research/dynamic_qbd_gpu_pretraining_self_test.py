"""Host integration self-test for the two GPU-backed bundle components."""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import time

import numpy as np

from stock_predictor.v5 import train
from .dynamic_qbd_gpu_pretraining import (
    GPU_EXECUTION_BACKEND, GPU_HGB_EXECUTION_BACKEND,
    GPUSectionQueueTimeout, exclusive_gpu_device,
    configure_gpu_pretraining, gpu_section_queue_snapshot,
    gpu_workload_allowed, opencl_fp64_devices,
)


def main() -> int:
    root = Path(tempfile.mkdtemp(prefix="dqbd-gpu-pretraining-self-test-"))
    os.environ["DQBD_ENABLE_GPU_PRETRAINING"] = "1"
    contract = configure_gpu_pretraining(root, require=True)
    rows = []
    for index in range(48):
        value = float(index) / 10.0
        rows.append({
            "feature_snapshot": {"x": value, "bucket": str(index % 4)},
            "labels": {"1": {"y_excess_net": value / 100.0,
                               "y_positive_edge": int(index % 3 == 0),
                               "y_downside": int(index % 7 == 0)}},
        })
    os.environ["DQBD_FIT_SHARED_ID"] = "GPU_SELF_TEST_H01_F01"
    os.environ["DQBD_GPU_JOB_ID"] = "GPU_SELF_TEST_RIDGE"
    bundle = train.build_bundle(
        "RIDGE_LOGISTIC", {"alpha": 1.0, "positive_C": 1.0,
                            "downside_C": 1.0}, 17).fit(rows, 1)
    if bundle.execution_backend != GPU_EXECUTION_BACKEND:
        raise AssertionError("GPU_RIDGE_BACKEND_NOT_SELECTED")
    if not bundle.gpu_fit_diagnostics or not bundle.gpu_fit_diagnostics.get("device"):
        raise AssertionError("GPU_RIDGE_DIAGNOSTICS_MISSING")
    if not bundle.gpu_fit_diagnostics.get("shared_gram_cache"):
        raise AssertionError("GPU_RIDGE_SHARED_GRAM_NOT_ENABLED")
    second_ridge = train.build_bundle(
        "RIDGE_LOGISTIC", {"alpha": 10.0, "positive_C": 1.0,
                            "downside_C": 1.0}, 17).fit(rows, 1)
    if not second_ridge.gpu_fit_diagnostics.get("shared_gram_cache_hit"):
        raise AssertionError("GPU_RIDGE_SHARED_GRAM_CACHE_MISS")
    devices = opencl_fp64_devices()
    if len(devices) >= 2:
        os.environ["DQBD_GPU_WORKER_SLOT"] = "1"
        os.environ["DQBD_FIT_SHARED_ID"] = "GPU_SELF_TEST_H01_F02"
        os.environ["DQBD_GPU_JOB_ID"] = "GPU_SELF_TEST_RIDGE_DEVICE_1"
        second_gpu_ridge = train.build_bundle(
            "RIDGE_LOGISTIC", {"alpha": 25.0, "positive_C": 1.0,
                                "downside_C": 1.0}, 17).fit(rows, 1)
        if second_gpu_ridge.execution_backend != GPU_EXECUTION_BACKEND:
            raise AssertionError("GPU_RIDGE_DEVICE_1_BACKEND_NOT_SELECTED")
    os.environ["DQBD_FORCE_EXECUTION_BACKEND"] = GPU_HGB_EXECUTION_BACKEND
    for slot, device in enumerate(devices[:2]):
        if not gpu_workload_allowed(device, "HGB"):
            if gpu_workload_allowed(device, "RIDGE"):
                continue
            raise AssertionError("GPU_DISPLAY_POLICY_BLOCKED_ALL_WORKLOADS")
        os.environ["DQBD_GPU_WORKER_SLOT"] = str(slot)
        os.environ["DQBD_GPU_JOB_ID"] = f"GPU_SELF_TEST_HGB_{slot}"
        hgb = train.build_bundle(
            "HIST_GRADIENT_BOOSTING", {"learning_rate": .05, "max_iter": 3,
                                       "max_leaf_nodes": 5,
                                       "l2_regularization": 1.0}, 17).fit(rows, 1)
        if hgb.execution_backend != GPU_HGB_EXECUTION_BACKEND:
            raise AssertionError("GPU_HGB_BACKEND_NOT_SELECTED")
        if not hgb.gpu_fit_diagnostics or not hgb.gpu_fit_diagnostics.get("device"):
            raise AssertionError("GPU_HGB_DIAGNOSTICS_MISSING")
        identity = f"{device['vendor']} {device['name']}".upper()
        expected_bin = 15 if ("AMD" in identity or
                              "ADVANCED MICRO DEVICES" in identity) else 63
        if hgb.gpu_fit_diagnostics["kernel_parameters"]["max_bin"] != expected_bin:
            raise AssertionError("GPU_HGB_DEVICE_KERNEL_POLICY_MISMATCH")
        if hgb.gpu_fit_diagnostics.get("device_lock_scope") != "PER_ESTIMATOR_FIT":
            raise AssertionError("GPU_HGB_LOCK_SCOPE_NOT_PIPELINED")
    if contract["device_count"] < 2:
        raise AssertionError("TWO_GPU_CONTRACT_NOT_SATISFIED")
    event_files = sorted((root / "gpu-device-events").glob("*.jsonl"))
    if len(event_files) < 2:
        raise AssertionError("GPU_DEVICE_EVENT_STREAMS_MISSING")
    for path in event_files[:2]:
        events = [
            json.loads(line) for line in path.read_text(
                encoding="utf-8").splitlines() if line.strip()
        ]
        kinds = {str(event.get("event")) for event in events}
        if not {"ACQUIRE", "RELEASE"} <= kinds:
            raise AssertionError("GPU_DEVICE_SECTION_EVENTS_INCOMPLETE")
        if any(
            not str(event.get("job_id") or "").startswith("GPU_SELF_TEST_")
            for event in events if event.get("event") == "ACQUIRE"
        ):
            raise AssertionError("GPU_DEVICE_SECTION_JOB_ID_MISSING")
        if not any(
            float(event.get("duration_seconds", 0.0)) >= 0.0
            for event in events if event.get("event") == "RELEASE"
        ):
            raise AssertionError("GPU_DEVICE_SECTION_DURATION_MISSING")
    queue_events = root / "gpu-section-queue-events.jsonl"
    if not queue_events.is_file():
        raise AssertionError("GPU_SECTION_READY_QUEUE_EVENTS_MISSING")
    queued = [
        json.loads(line) for line in queue_events.read_text(
            encoding="utf-8").splitlines() if line.strip()
    ]
    if not queued or any(
        not str(event.get("job_id") or "").startswith("GPU_SELF_TEST_")
        for event in queued
    ):
        raise AssertionError("GPU_SECTION_READY_QUEUE_WORKLOADS_MISSING")
    backlog = gpu_section_queue_snapshot(root, devices, cache_seconds=0.0)
    if any(int(row.get("ticket_count", 0)) != 0
           for row in backlog.get("devices", ())):
        raise AssertionError("GPU_SECTION_READY_QUEUE_NOT_DRAINED")
    if len(devices) >= 2:
        one_device = gpu_section_queue_snapshot(
            root, devices[:1], cache_seconds=60.0)
        two_devices = gpu_section_queue_snapshot(
            root, devices[:2], cache_seconds=60.0)
        if len(one_device.get("devices", ())) != 1:
            raise AssertionError("GPU_SECTION_CACHE_ONE_DEVICE_IDENTITY_LOST")
        if len(two_devices.get("devices", ())) != 2:
            raise AssertionError("GPU_SECTION_CACHE_DEVICE_SET_COLLISION")

    # Fault injection: a live foreign section blocks one physical adapter
    # until the short timeout expires. The next acquisition must recover in
    # the same process after the blocker is removed; a restart is forbidden.
    recovery_device = devices[0]
    recovery_queue = (
        root / "gpu-section-ready-queue"
        / f"platform-{int(recovery_device['platform_index'])}-device-"
        f"{int(recovery_device['device_index'])}")
    recovery_queue.mkdir(parents=True, exist_ok=True)
    blocker = recovery_queue / "_active-section.token"
    blocker.write_text(json.dumps({
        "schema_version": "DQBD_GPU_SECTION_QUEUE_V1",
        "ticket_id": "SELF_TEST_BLOCKER",
        "pid": os.getpid(),
        "claimed_at": time.time(),
        "platform_index": int(recovery_device["platform_index"]),
        "device_index": int(recovery_device["device_index"]),
        "name": str(recovery_device.get("name", "")),
    }), encoding="utf-8")
    old_timeout = os.environ.get("DQBD_GPU_SECTION_QUEUE_TIMEOUT_SECONDS")
    os.environ["DQBD_GPU_SECTION_QUEUE_TIMEOUT_SECONDS"] = "0.1"
    try:
        try:
            with exclusive_gpu_device(root, recovery_device):
                raise AssertionError("GPU_QUEUE_TIMEOUT_INJECTION_NOT_TRIGGERED")
        except GPUSectionQueueTimeout:
            pass
        if not blocker.is_file():
            raise AssertionError("GPU_QUEUE_TIMEOUT_BLOCKER_DISAPPEARED")
        blocker.unlink()
        with exclusive_gpu_device(root, recovery_device):
            pass
    finally:
        if old_timeout is None:
            os.environ.pop("DQBD_GPU_SECTION_QUEUE_TIMEOUT_SECONDS", None)
        else:
            os.environ["DQBD_GPU_SECTION_QUEUE_TIMEOUT_SECONDS"] = old_timeout
        if blocker.is_file():
            blocker.unlink()
    active_tokens = list(
        (root / "gpu-section-ready-queue").glob(
            "platform-*-device-*/*active-section.token"))
    if active_tokens:
        raise AssertionError("GPU_SECTION_ACTIVE_TOKEN_NOT_RELEASED")
    os.environ.pop("DQBD_FIT_SHARED_ID", None)
    os.environ.pop("DQBD_GPU_JOB_ID", None)
    os.environ.pop("DQBD_FORCE_EXECUTION_BACKEND", None)
    print("DQBD_GPU_PRETRAINING_SELF_TEST_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
