"""Low-overhead host telemetry for long Dynamic-QBD local runs.

NWinfo is optional research telemetry. Failure to discover or sample it never
changes the scientific result; the suite records the failure and continues.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import threading
from typing import Any


def _resolve_nwinfo_executable(explicit: str | Path | None = None) -> str | None:
    """Resolve NWinfo without assuming one installer-specific location.

    Resolution order is explicit CLI path, NWINFO_EXE environment override,
    then PATH discovery. Explicit/env paths may be either absolute paths or
    command names resolvable through PATH.
    """
    candidates = [explicit, os.environ.get("NWINFO_EXE"), "nwinfo", "nwinfo.exe"]
    for candidate in candidates:
        if candidate is None or not str(candidate).strip():
            continue
        raw = str(candidate).strip().strip('"')
        path = Path(raw).expanduser()
        if path.is_file():
            return str(path.resolve())
        discovered = shutil.which(raw)
        if discovered:
            return str(Path(discovered).resolve())
    return None


class NWInfoSampler:
    def __init__(
        self,
        root: str | Path,
        *,
        interval_seconds: float = 60.0,
        executable: str | Path | None = None,
    ):
        self.root = Path(root)
        self.interval_seconds = max(10.0, float(interval_seconds))
        self.executable = _resolve_nwinfo_executable(executable)
        self.sensor_path = self.root / "nwinfo-sensors.jsonl"
        self.system_path = self.root / "nwinfo-system.json"
        self.summary_path = self.root / "nwinfo-summary.json"
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._samples = 0
        self._failures = 0
        self._last_error: str | None = None

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def _run(self, args: list[str], *, timeout: float = 20.0) -> dict[str, Any]:
        if not self.executable:
            raise FileNotFoundError("NWINFO_EXECUTABLE_NOT_FOUND")
        completed = subprocess.run(
            [self.executable, *args],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            encoding="utf-8",
            errors="replace",
        )
        payload: Any = None
        parse_error = None
        stdout = completed.stdout.strip()
        if stdout:
            try:
                payload = json.loads(stdout)
            except json.JSONDecodeError as exc:
                parse_error = f"{type(exc).__name__}:{exc}"
        return {
            "timestamp_utc": self._now(),
            "returncode": int(completed.returncode),
            "command": args,
            "payload": payload,
            "stdout_unparsed": None if payload is not None else stdout[-20000:],
            "stderr": completed.stderr.strip()[-10000:],
            "parse_error": parse_error,
        }

    def _write_summary(self) -> None:
        payload = {
            "schema_version": "DYNAMIC_QBD_NWINFO_TELEMETRY_V2_EXPLICIT_EXECUTABLE",
            "available": bool(self.executable),
            "executable": self.executable,
            "sample_interval_seconds": self.interval_seconds,
            "samples": int(self._samples),
            "failures": int(self._failures),
            "last_error": self._last_error,
            "telemetry_authority": "DIAGNOSTIC_ONLY_FAIL_OPEN",
        }
        self.summary_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    def capture_system_snapshot(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        if not self.executable:
            self._last_error = "NWINFO_EXECUTABLE_NOT_FOUND"
            self._failures += 1
            self._write_summary()
            return
        try:
            record = self._run(["--format=json", "--cp=UTF8", "--sys", "--cpu"])
            self.system_path.write_text(
                json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            if record["returncode"] != 0:
                self._failures += 1
                self._last_error = f"NWINFO_SYSTEM_RETURN_CODE:{record['returncode']}"
        except Exception as exc:
            self._failures += 1
            self._last_error = f"{type(exc).__name__}:{exc}"
        self._write_summary()

    def sample_once(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        if not self.executable:
            return
        try:
            record = self._run(
                ["--format=json", "--cp=UTF8", "--sensors=CPU,DIMM,IMC"],
                timeout=min(30.0, max(10.0, self.interval_seconds * 0.75)),
            )
            with self._lock:
                with self.sensor_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, sort_keys=True) + "\n")
                self._samples += 1
                if record["returncode"] != 0:
                    self._failures += 1
                    self._last_error = f"NWINFO_SENSOR_RETURN_CODE:{record['returncode']}"
        except Exception as exc:
            with self._lock:
                self._failures += 1
                self._last_error = f"{type(exc).__name__}:{exc}"
        self._write_summary()

    def _loop(self) -> None:
        self.sample_once()
        while not self._stop.wait(self.interval_seconds):
            self.sample_once()

    def start(self) -> "NWInfoSampler":
        self.root.mkdir(parents=True, exist_ok=True)
        self.capture_system_snapshot()
        if self.executable and self._thread is None:
            self._thread = threading.Thread(
                target=self._loop, name="dynamic-qbd-nwinfo", daemon=True
            )
            self._thread.start()
        return self

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(2.0, self.interval_seconds + 2.0))
            self._thread = None
        self._write_summary()
        return json.loads(self.summary_path.read_text(encoding="utf-8"))

    def __enter__(self) -> "NWInfoSampler":
        return self.start()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()
