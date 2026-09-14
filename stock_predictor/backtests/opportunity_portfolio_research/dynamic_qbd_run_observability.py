"""Compile a compact, time-ordered observability package for a QBD run.

This is deliberately operational evidence, not an economic evaluation.  It
reads manifested SQLite timestamps and GPU acquire events without opening
holdout data or materializing prediction payloads.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sqlite3
import statistics
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from .dynamic_qbd_wealth_metrics import wealth_path_metrics
from .german_retail_tax_engine import TaxLedger
from .next_open_portfolio_replay import _metrics
from .tax_contracts import TaxConfig


SEEDS = ("SHORT", "PRIMARY", "LONG")
JOB_COLUMNS = (
    "job_id", "kind", "state", "attempt", "last_error", "started_at",
    "finished_at", "payload",
)


def _iso(epoch: float | None) -> str | None:
    if epoch is None:
        return None
    return datetime.fromtimestamp(float(epoch), timezone.utc).isoformat().replace(
        "+00:00", "Z")


def _parse_payload(raw: str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except (TypeError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _job_dimensions(job_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    values = dict(payload)
    match = re.search(r"candidate_oos_fold:(\d+):([^:]+):", job_id)
    if match:
        values.setdefault("horizon", int(match.group(1)))
        values.setdefault("fold_id", match.group(2))
    return {
        "cutoff": values.get("cutoff"),
        "horizon": values.get("horizon"),
        "fold_id": values.get("fold_id"),
        "candidate_id": values.get("candidate_id"),
        "model_family_key": values.get("model_family_key"),
        "portfolio_family_key": values.get("portfolio_family_key"),
    }


def _load_gpu_events(paths: Iterable[Path]) -> dict[str, dict[str, Any]]:
    events: dict[str, dict[str, Any]] = {}
    for path in paths:
        if not path.is_file():
            continue
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                job_id = row.get("job_id")
                if job_id:
                    events[str(job_id)] = {
                        "executor": "GPU",
                        "workload": row.get("workload"),
                        "device": row.get("name"),
                        "device_index": row.get("platform_index"),
                    }
    return events


def _git_sha(repo_root: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo_root,
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


ECONOMIC_COLUMNS = [
    "seed", "family_id", "as_of", "path_start", "path_end", "source_run_contract_hash",
    "initial_value", "terminal_value", "urth_terminal_value", "cagr", "urth_cagr",
    "cagr_excess", "total_return", "urth_total_return", "total_return_excess",
    "max_drawdown", "urth_max_drawdown", "worst_relative_drawdown",
    "relative_max_drawdown", "expected_shortfall_95", "relative_downside_deviation",
    "turnover", "trade_count", "total_cost_eur", "period_days", "average_positions",
]


def _latest_replay_segments(checkpoint_root: Path) -> list[tuple[str, str, Path]]:
    latest: dict[tuple[str, str], Path] = {}
    for seed in SEEDS:
        root = checkpoint_root / seed.lower() / "portfolio-replay"
        if not root.is_dir():
            continue
        for nav_path in root.rglob("nav.parquet"):
            family_id = nav_path.parent.parent.name
            key = (seed, family_id)
            current = latest.get(key)
            if current is None or nav_path.parent.name > current.parent.name:
                latest[key] = nav_path
    return [(seed, family_id, path) for (seed, family_id), path in sorted(latest.items())]


def _economic_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    numeric = (
        "terminal_value", "cagr", "urth_cagr", "cagr_excess", "max_drawdown",
        "relative_max_drawdown", "expected_shortfall_95", "relative_downside_deviation",
        "turnover", "trade_count", "total_cost_eur",
    )
    summary: dict[str, Any] = {}
    for seed in SEEDS:
        seed_rows = [row for row in rows if row["seed"] == seed]
        values: dict[str, dict[str, float | int]] = {}
        for field in numeric:
            observed = [float(row[field]) for row in seed_rows if row.get(field) is not None]
            if not observed:
                continue
            values[field] = {
                "min": float(min(observed)),
                "median": float(statistics.median(observed)),
                "mean": float(statistics.fmean(observed)),
                "max": float(max(observed)),
            }
        summary[seed] = {"row_count": len(seed_rows), "metrics": values}
    return summary


def compile_economic_results(*, checkpoint_root: Path, output_dir: Path) -> dict[str, Any]:
    """Compile existing completed replay prefixes using canonical replay metrics.

    The current Step-9 run did not finish its causal arm materialization.  This
    therefore reports the latest available replay prefix per family and never
    labels those rows as S0/S1/S2/S3 or Oracle results.
    """
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for seed, family_id, nav_path in _latest_replay_segments(checkpoint_root):
        state_path = nav_path.parent / "replay-state.json"
        manifest_path = nav_path.parent / "replay-manifest.json"
        try:
            curve = pd.read_parquet(nav_path)
            state = json.loads(state_path.read_text(encoding="utf-8"))
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if curve.empty:
                raise ValueError("EMPTY_NAV_CURVE")
            initial = float(state.get("initial_value", curve["strategy_value"].iloc[0]))
            trades = state.get("trades", [])
            costs = {"transaction_cost_eur": float(state.get("transaction_cost_eur", 0.0))}
            tax = TaxLedger.from_snapshot(TaxConfig(enabled=False), state.get("tax_ledger"))
            canonical = _metrics(
                curve=curve,
                initial=initial,
                trades=trades,
                costs=costs,
                tax=tax,
                benchmark_initial=initial,
            )
            risk = wealth_path_metrics(curve)
            rows.append({
                "seed": seed,
                "family_id": family_id,
                "as_of": str(state.get("as_of", nav_path.parent.name)),
                "path_start": str(curve["date"].iloc[0]),
                "path_end": str(curve["date"].iloc[-1]),
                "source_run_contract_hash": manifest.get("run_contract_hash"),
                "initial_value": float(canonical["initial_value"]),
                "terminal_value": float(canonical["terminal_value"]),
                "urth_terminal_value": float(canonical["urth_terminal_value"]),
                "cagr": float(canonical["cagr"]),
                "urth_cagr": float(canonical["urth_cagr"]),
                "cagr_excess": float(canonical["cagr_excess"]),
                "total_return": float(canonical["total_return"]),
                "urth_total_return": float(canonical["urth_total_return"]),
                "total_return_excess": float(canonical["total_return_excess"]),
                "max_drawdown": float(canonical["max_drawdown"]),
                "urth_max_drawdown": float(canonical["urth_max_drawdown"]),
                "worst_relative_drawdown": float(canonical["worst_relative_drawdown"]),
                "relative_max_drawdown": float(risk["relative_max_drawdown"]),
                "expected_shortfall_95": float(risk["expected_shortfall_95"]),
                "relative_downside_deviation": float(risk["relative_downside_deviation"]),
                "turnover": float(canonical["turnover"]),
                "trade_count": int(canonical["trade_count"]),
                "total_cost_eur": float(canonical["total_cost_eur"]),
                "period_days": int(canonical["period_days"]),
                "average_positions": float(canonical["average_positions"]),
            })
        except (OSError, ValueError, KeyError, TypeError, ArithmeticError) as exc:
            errors.append({"seed": seed, "family_id": family_id, "error": f"{type(exc).__name__}:{exc}"})
    economic_path = output_dir / "economic-segment-results.csv"
    with economic_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ECONOMIC_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return {
        "status": "PARTIAL_SEGMENT_DIAGNOSTIC" if rows else "NO_ECONOMIC_SEGMENTS",
        "causal_arm_labels_available": False,
        "latest_prefix_per_family": True,
        "rows": len(rows),
        "errors": errors,
        "summary_by_seed": _economic_summary(rows),
        "file": economic_path.name,
    }


def compile_observability(
    *, checkpoint_root: Path, output_dir: Path, run_root: Path,
    repo_root: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    gpu_events = _load_gpu_events((
        run_root / "_shared-compute" / "gpu-section-queue-events.jsonl",
        checkpoint_root / "_shared-compute" / "gpu-section-queue-events.jsonl",
    ))
    lifecycle_path = output_dir / "job-lifecycle-timeline.csv"
    event_path = output_dir / "job-completion-events.csv"
    minute_path = output_dir / "job-completion-by-minute.csv"
    fieldnames = [
        "event_time_utc", "event_type", "seed", "job_id", "kind", "state",
        "attempt", "started_at_utc", "finished_at_utc", "duration_seconds",
        "cutoff", "horizon", "fold_id", "candidate_id", "model_family_key",
        "portfolio_family_key", "executor", "workload", "device",
        "device_index", "last_error",
    ]
    minute_counts: defaultdict[tuple[str, str], Counter] = defaultdict(Counter)
    seed_summary: dict[str, dict[str, Any]] = {}
    all_errors: list[dict[str, Any]] = []
    lifecycle_rows: list[dict[str, Any]] = []
    event_rows_buffer: list[dict[str, Any]] = []
    total_rows = 0
    event_rows = 0

    for seed in SEEDS:
            db_path = checkpoint_root / seed.lower() / "run-state" / "jobs.sqlite3"
            if not db_path.is_file():
                seed_summary[seed] = {"status": "DB_MISSING"}
                continue
            counts = Counter()
            completed_times: list[float] = []
            connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
            cursor = connection.cursor()
            state_counts = Counter({
                str(state): int(count)
                for state, count in cursor.execute(
                    "SELECT state, COUNT(*) FROM jobs GROUP BY state")
            })
            query = (
                "SELECT job_id,kind,state,attempt,last_error,started_at,"
                "finished_at,payload FROM jobs WHERE started_at IS NOT NULL "
                "OR finished_at IS NOT NULL OR state IN ('FAILED','RUNNING') "
                "ORDER BY COALESCE(started_at, finished_at), job_id"
            )
            for raw in cursor.execute(query):
                total_rows += 1
                job_id, kind, state, attempt, last_error, started, finished, payload_raw = raw
                payload = _parse_payload(payload_raw)
                dimensions = _job_dimensions(str(job_id), payload)
                gpu = gpu_events.get(str(job_id), {})
                executor = gpu.get("executor") or (
                    "CPU" if kind != "candidate_oos_fold" else "UNKNOWN")
                duration = (
                    max(0.0, float(finished) - float(started))
                    if started is not None and finished is not None else None)
                if state == "COMPLETE":
                    counts["COMPLETE"] += 1
                    if finished is not None:
                        completed_times.append(float(finished))
                else:
                    counts[str(state)] += 1
                if last_error:
                    all_errors.append({
                        "seed": seed, "job_id": str(job_id), "kind": kind,
                        "state": state, "attempt": attempt,
                        "last_error": last_error,
                    })
                common = {
                    "seed": seed, "job_id": str(job_id), "kind": kind,
                    "state": state, "attempt": attempt,
                    "started_at_utc": _iso(started),
                    "finished_at_utc": _iso(finished),
                    "duration_seconds": duration,
                    **dimensions,
                    "executor": executor,
                    "workload": gpu.get("workload"),
                    "device": gpu.get("device"),
                    "device_index": gpu.get("device_index"),
                    "last_error": last_error,
                }
                lifecycle_rows.append({
                    "event_time_utc": _iso(finished or started),
                    "event_type": "COMPLETE" if state == "COMPLETE" else state,
                    **common,
                })
                if started is not None:
                    event = dict(common)
                    event.update({
                        "event_time_utc": _iso(started),
                        "event_type": "START",
                    })
                    event_rows_buffer.append(event)
                    event_rows += 1
                    minute_counts[(seed, _iso(started)[:16])]["started_jobs"] += 1
                if finished is not None:
                    event = dict(common)
                    event.update({
                        "event_time_utc": _iso(finished),
                        "event_type": state,
                    })
                    event_rows_buffer.append(event)
                    event_rows += 1
                    minute_counts[(seed, _iso(finished)[:16])][
                        "completed_jobs" if state == "COMPLETE" else "failed_jobs"] += 1
            connection.close()
            incomplete = bool(
                state_counts.get("PENDING")
                or state_counts.get("RUNNING")
                or state_counts.get("FAILED")
            )
            seed_summary[seed] = {
                "status": "INCOMPLETE" if incomplete else "COMPLETE",
                "state_counts": dict(state_counts),
                "state_counts_observed": dict(counts),
                "observed_started_or_finished_rows": sum(counts.values()),
                "first_completion_utc": _iso(min(completed_times)) if completed_times else None,
                "last_completion_utc": _iso(max(completed_times)) if completed_times else None,
            }

    def _event_sort_key(row: dict[str, Any]) -> tuple[str, str, str]:
        return (
            str(row.get("event_time_utc") or ""),
            str(row.get("seed") or ""),
            str(row.get("job_id") or ""),
        )

    lifecycle_rows.sort(key=_event_sort_key)
    event_rows_buffer.sort(key=lambda row: (
        str(row.get("event_time_utc") or ""),
        0 if row.get("event_type") == "START" else 1,
        str(row.get("seed") or ""),
        str(row.get("job_id") or ""),
    ))
    with lifecycle_path.open("w", newline="", encoding="utf-8") as lifecycle:
        writer = csv.DictWriter(lifecycle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(lifecycle_rows)
    with event_path.open("w", newline="", encoding="utf-8") as event_file:
        writer = csv.DictWriter(event_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(event_rows_buffer)

    with minute_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "minute_utc", "seed", "started_jobs", "completed_jobs", "failed_jobs",
        ])
        writer.writeheader()
        for (seed, minute), counts in sorted(
                minute_counts.items(), key=lambda item: (item[0][1], item[0][0])):
            writer.writerow({
                "minute_utc": minute + ":00Z", "seed": seed,
                "started_jobs": counts.get("started_jobs", 0),
                "completed_jobs": counts.get("completed_jobs", 0),
                "failed_jobs": counts.get("failed_jobs", 0),
            })

    economic = compile_economic_results(
        checkpoint_root=checkpoint_root, output_dir=output_dir)
    result = {
        "schema_version": "DQBD_RUN_OBSERVABILITY_V1",
        "status": "INCOMPLETE_FAILED" if all_errors else "INCOMPLETE",
        "evaluation_opened": False,
        "holdout_reads": 0,
        "checkpoint_root": str(checkpoint_root),
        "run_root": str(run_root),
        "compiled_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "code_sha_at_compilation": _git_sha(repo_root),
        "seeds": seed_summary,
        "error_count": len(all_errors),
        "errors": all_errors,
        "timeline_rows": total_rows,
        "event_rows": event_rows,
        "gpu_event_job_count": len(gpu_events),
        "economic_results": economic,
        "files": {
            "job_lifecycle_timeline": lifecycle_path.name,
            "job_completion_events": event_path.name,
            "job_completion_by_minute": minute_path.name,
            "compiled_results": "compiled-results.json",
            "economic_segment_results": economic["file"],
        },
    }
    serialized = json.dumps(result, indent=2, sort_keys=True) + "\n"
    (output_dir / "partial-results.json").write_text(serialized, encoding="utf-8")
    (output_dir / "compiled-results.json").write_text(serialized, encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--repo-root", required=True, type=Path)
    args = parser.parse_args()
    result = compile_observability(
        checkpoint_root=args.checkpoint_root, output_dir=args.output_dir,
        run_root=args.run_root, repo_root=args.repo_root)
    print(json.dumps({
        "status": result["status"], "timeline_rows": result["timeline_rows"],
        "event_rows": result["event_rows"], "error_count": result["error_count"],
        "output_dir": str(args.output_dir),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
