"""Small extensions for the v40.1 Dynamic-QBD execution hardening.

Kept separate from the core hardening module so the remaining protections can
be reviewed independently: execution-repair accounting and the final runtime
JSON/temp writers. No scientific or scheduler policy is changed.
"""
from __future__ import annotations

import importlib.abc
import importlib.machinery
import os
import sys
import time
from functools import wraps

from . import dynamic_qbd_manifested_job_coordinator as coordinator
from .dynamic_qbd_attempt_fenced_store import (
    AttemptFencedManifestedJobStore,
)
from .dynamic_qbd_runtime_hardening import (
    _safe_write_json,
    _scheduler_health_error,
)


_RUNNER_MODULE = (
    "stock_predictor.backtests.opportunity_portfolio_research."
    "dynamic_qbd_causal_model_store_run"
)
_NON_WORKER_RECOVERY_MARKERS = (
    "DQBD_STALE_DISPATCHER_ACTIVITY",
    "DQBD_LANE_REPLACEMENT_FAILED",
)
_INSTALLED = False
_RUNNER_HOOK_INSTALLED = False


def _patch_execution_recovery_accounting() -> None:
    cls = AttemptFencedManifestedJobStore
    if getattr(cls.requeue_execution_failure, "_dqbd_v40_1_accounting", False):
        return
    original = cls.requeue_execution_failure

    @wraps(original)
    def requeue_execution_failure(
        self, job_id: str, owner: str, *, reason: str,
        preferred_executor: str | None = None,
        count_worker_failure: bool = True,
    ) -> bool:
        message = str(reason)
        if any(marker in message for marker in _NON_WORKER_RECOVERY_MARKERS):
            count_worker_failure = False
        return original(
            self,
            job_id,
            owner,
            reason=message,
            preferred_executor=preferred_executor,
            count_worker_failure=count_worker_failure,
        )

    requeue_execution_failure._dqbd_v40_1_accounting = True
    cls.requeue_execution_failure = requeue_execution_failure


def _patch_ram_spill_writer() -> None:
    """Make the last fixed ``.tmp`` RAM spill advisory and collision-safe."""
    cls = coordinator.RamWorkerPauseController
    if getattr(cls._spill, "_dqbd_v40_1_unique_temp", False):
        return

    def spill(self, *, event: str, usage: float) -> None:
        try:
            self.spill_root.mkdir(parents=True, exist_ok=True)
            processes = []
            for process in self._processes(self.pool):
                pid = getattr(process, "pid", None)
                if not pid:
                    continue
                try:
                    rss = self._rss_bytes(int(pid))
                except OSError:
                    rss = None
                processes.append({
                    "pid": int(pid),
                    "rss_bytes": rss,
                    "paused": False,
                })
            payload = {
                "schema_version": "DQBD_RAM_SPILL_RESUME_V2_SINGLE_LANE",
                "event": str(event),
                "timestamp": time.time(),
                "controller_pid": os.getpid(),
                "system_usage_fraction": float(usage),
                "processes": processes,
                "active_lanes": (
                    self.pool.active_lanes()
                    if hasattr(self.pool, "active_lanes") else []
                ),
                "scheduler": (
                    self.scheduler.telemetry()
                    if hasattr(self.scheduler, "telemetry") else {}
                ),
                "evaluation_not_opened": True,
            }
            _safe_write_json(self.spill_path, payload)
        except OSError as exc:
            # Reclaim state is in memory/SQLite; the spill is operator and
            # resume telemetry. A sharing/disk write failure must not block the
            # actual single-lane recovery decision.
            _scheduler_health_error(self.scheduler, "telemetry", exc)

    spill._dqbd_v40_1_unique_temp = True
    cls._spill = spill


def _patch_runner_module(module) -> None:
    """Replace only the runner's atomic JSON publisher."""
    if getattr(module, "_dqbd_runtime_writer_hardened", False):
        return
    module._json = _safe_write_json
    module._dqbd_runtime_writer_hardened = True


class _RunnerPatchLoader(importlib.abc.Loader):
    def __init__(self, loader) -> None:
        self.loader = loader

    def create_module(self, spec):
        create = getattr(self.loader, "create_module", None)
        return create(spec) if callable(create) else None

    def exec_module(self, module) -> None:
        self.loader.exec_module(module)
        _patch_runner_module(module)


class _RunnerPatchFinder(importlib.abc.MetaPathFinder):
    """Patch the Step-9 runner after its module body defines ``_json``.

    ``python -m ...dynamic_qbd_causal_model_store_run`` initializes the package
    before resolving the submodule spec. Installing this finder from package
    bootstrap therefore wraps the ordinary source loader without changing
    module resolution or code identity. The finder removes itself after the
    one target spec is wrapped.
    """

    def find_spec(self, fullname, path=None, target=None):
        global _RUNNER_HOOK_INSTALLED
        if fullname != _RUNNER_MODULE:
            return None
        try:
            sys.meta_path.remove(self)
        except ValueError:
            pass
        _RUNNER_HOOK_INSTALLED = False
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is None:
            return spec
        spec.loader = _RunnerPatchLoader(spec.loader)
        return spec


def _install_runner_writer_hook() -> None:
    global _RUNNER_HOOK_INSTALLED
    existing = sys.modules.get(_RUNNER_MODULE)
    if existing is not None:
        _patch_runner_module(existing)
        return
    if _RUNNER_HOOK_INSTALLED:
        return
    sys.meta_path.insert(0, _RunnerPatchFinder())
    _RUNNER_HOOK_INSTALLED = True


def install_runtime_hardening_extensions() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    _patch_execution_recovery_accounting()
    _patch_ram_spill_writer()
    _install_runner_writer_hook()
    _INSTALLED = True
