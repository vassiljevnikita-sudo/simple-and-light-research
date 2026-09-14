from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np
import pandas as pd

from . import portfolio_research_cli as runner
from . import portfolio_policy_search as search
from . import portfolio_policy_walk_forward as multicore_walk_forward
from .affinity_coordinator_pool import AffinityCoordinatorPool
from .portfolio_evidence_expansion import run_evidence_expansion
from .portfolio_replay_process_backend import backend_stats, install_multicore_backend, shutdown_multicore_backend
from .portfolio_replay_result_cache import ReplayResultStore
from .portfolio_resilient_process_pool import install_resilient_process_pool, restore_process_pool_runner
from .portfolio_research_validation_runtime import install_research_runtime_v2, postprocess_research_artifacts

_ORIGINAL_RUNNER_REPLAY = runner.replay
_FINAL_REPLAY_MEMO: dict[tuple, dict] = {}
_FRAME_FINGERPRINTS: dict[tuple[int, tuple[str, ...]], str] = {}
_FINAL_RESULT_STORE: ReplayResultStore | None = None
_FINAL_REPLAY_STATS = {
    "memory_hits": 0,
    "persistent_hits": 0,
    "misses": 0,
    "writes": 0,
    "memory_entries": 0,
    "persistent_enabled": False,
    "persistent_path": None,
}


def _frame_fingerprint(frame, wanted: tuple[str, ...]) -> str | None:
    if frame is None:
        return None
    cols = tuple(c for c in wanted if c in frame.columns)
    cache_key = (id(frame), cols)
    cached = _FRAME_FINGERPRINTS.get(cache_key)
    if cached is not None:
        return cached
    h = hashlib.sha256()
    h.update(str(tuple(frame.shape)).encode("utf-8"))
    h.update(str(cols).encode("utf-8"))
    if cols:
        values = pd.util.hash_pandas_object(frame[list(cols)], index=False, categorize=True).to_numpy(dtype=np.uint64)
        h.update(values.tobytes())
    digest = h.hexdigest()
    _FRAME_FINGERPRINTS[cache_key] = digest
    return digest


def _memory_replay_key(
    signals, prices, policy, cost, tax_config, start, end, initial,
    regime, resolved_threshold, prepared_signals,
) -> tuple:
    return (
        id(signals), id(prices), policy.policy_id, float(cost.roundtrip_bps),
        tuple(sorted(asdict(tax_config).items())),
        str(pd.Timestamp(start)) if start is not None else None,
        str(pd.Timestamp(end)) if end is not None else None,
        float(initial), id(regime) if regime is not None else None,
        float(resolved_threshold).hex() if resolved_threshold is not None else None,
        id(prepared_signals) if prepared_signals is not None else None,
    )


def _persistent_replay_key(signals, policy, cost, tax_config, start, end, initial, regime, resolved_threshold) -> str:
    payload = {
        "signal_fingerprint": _frame_fingerprint(signals, ("decision_date", "ticker", "fold_id", "horizon", "score")),
        "policy_id": policy.policy_id,
        "roundtrip_bps": float(cost.roundtrip_bps),
        "tax": asdict(tax_config),
        "start": str(pd.Timestamp(start)) if start is not None else None,
        "end": str(pd.Timestamp(end)) if end is not None else None,
        "initial": float(initial),
        "regime_fingerprint": _frame_fingerprint(regime, ("date", "regime")),
        "resolved_threshold": float(resolved_threshold).hex() if resolved_threshold is not None else None,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _memoized_runner_replay(
    signals, prices, policy, cost, tax_config, start=None, end=None, initial=10000.0,
    regime=None, resolved_threshold=None, prepared_signals=None,
):
    memory_key = _memory_replay_key(
        signals, prices, policy, cost, tax_config, start, end, initial,
        regime, resolved_threshold, prepared_signals,
    )
    cached = _FINAL_REPLAY_MEMO.get(memory_key)
    if cached is not None:
        _FINAL_REPLAY_STATS["memory_hits"] += 1
        return cached

    persistent_key = None
    if _FINAL_RESULT_STORE is not None:
        persistent_key = _persistent_replay_key(
            signals, policy, cost, tax_config, start, end, initial, regime, resolved_threshold
        )
        found, cached = _FINAL_RESULT_STORE.get(persistent_key)
        if found:
            _FINAL_REPLAY_MEMO[memory_key] = cached
            _FINAL_REPLAY_STATS["persistent_hits"] += 1
            _FINAL_REPLAY_STATS["memory_entries"] = len(_FINAL_REPLAY_MEMO)
            return cached

    _FINAL_REPLAY_STATS["misses"] += 1
    result = _ORIGINAL_RUNNER_REPLAY(
        signals, prices, policy, cost, tax_config,
        start=start, end=end, initial=initial, regime=regime,
        resolved_threshold=resolved_threshold, prepared_signals=prepared_signals,
    )
    _FINAL_REPLAY_MEMO[memory_key] = result
    _FINAL_REPLAY_STATS["memory_entries"] = len(_FINAL_REPLAY_MEMO)
    if _FINAL_RESULT_STORE is not None and persistent_key is not None:
        _FINAL_RESULT_STORE.put(persistent_key, result)
        _FINAL_REPLAY_STATS["writes"] += 1
    return result


def _run_walk_forward_with_backend(*args, **kwargs):
    global _FINAL_RESULT_STORE
    multicore_walk_forward.ThreadPoolExecutor = AffinityCoordinatorPool
    outer, history, meta = multicore_walk_forward.run_walk_forward(*args, **kwargs)
    meta = dict(meta)
    performance = dict(meta.get("performance", {}))
    fragment_info = dict(performance.get("persistent_fragment_cache", {}))
    if fragment_info.get("enabled") and fragment_info.get("path") and fragment_info.get("namespace"):
        search_cache = Path(str(fragment_info["path"]))
        final_cache = search_cache.with_name("final_replay_cache.sqlite3")
        _FINAL_RESULT_STORE = ReplayResultStore(final_cache, str(fragment_info["namespace"]))
        cache_stats = _FINAL_RESULT_STORE.stats()
        _FINAL_REPLAY_STATS["persistent_enabled"] = True
        _FINAL_REPLAY_STATS["persistent_path"] = str(final_cache)
        _FINAL_REPLAY_STATS["persistent_entries_loaded_at_start"] = cache_stats["entries_loaded_at_start"]
        print(
            f"[multicore] final replay cache: {cache_stats['entries_loaded_at_start']} reusable full results loaded from {final_cache}",
            flush=True,
        )
    performance["execution_backend"] = backend_stats()
    performance["final_replay_reuse"] = _FINAL_REPLAY_STATS
    meta["performance"] = performance
    return outer, history, meta


def main() -> int:
    global _FINAL_RESULT_STORE
    args = runner.parse_args()
    runner.HARD_MAX_WORKERS = 12
    workers = max(1, min(int(args.max_workers), runner.HARD_MAX_WORKERS))
    install_multicore_backend(workers)
    install_resilient_process_pool()
    install_research_runtime_v2()
    runner.run_walk_forward = _run_walk_forward_with_backend
    runner.replay = _memoized_runner_replay
    try:
        result = runner.main(args)
        if not args.self_test:
            # Classify the official Development artifacts first. Expanded evidence is
            # strictly post-selection and never feeds back into search/readiness.
            postprocess_research_artifacts(args.output_root)

            # Release the search pool before starting the evidence pool. This keeps
            # one 12-process pool + four lightweight coordinators, never two 12-worker
            # pools at the same time.
            shutdown_multicore_backend()
            expansion = run_evidence_expansion(
                args.v5_predictions,
                args.daily_store_root,
                args.output_root,
                max_workers=workers,
                coordinator_threads=4,
            )
            print(f"[evidence] expansion summary={expansion}", flush=True)
        return result
    finally:
        if _FINAL_RESULT_STORE is not None:
            print(f"[multicore] persistent final replay stats={_FINAL_RESULT_STORE.stats()}", flush=True)
            _FINAL_RESULT_STORE.close()
            _FINAL_RESULT_STORE = None
        print(f"[multicore] final replay reuse stats={_FINAL_REPLAY_STATS}", flush=True)
        runner.replay = _ORIGINAL_RUNNER_REPLAY
        restore_process_pool_runner()
        shutdown_multicore_backend()


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())
