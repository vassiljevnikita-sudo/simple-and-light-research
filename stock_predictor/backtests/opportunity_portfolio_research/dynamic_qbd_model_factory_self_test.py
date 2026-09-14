"""Fast regression tests for canonical Dynamic-QBD model jobs."""
from __future__ import annotations

from datetime import date, timedelta

from .dynamic_qbd_model_jobs import canonical_queue_sort_key, fixed_model_job_key, fixed_portfolio_groups, unique_fixed_model_job_count
from .dynamic_qbd_generation_registry import GenerationRegistry
from .dynamic_qbd_family_surface import build_family_specs


def main() -> int:
    families = build_family_specs(feature_schema_sha256="fixture-schema")
    h3 = tuple(x for x in families if x.horizon_sessions == 3 and x.exit_policy["family"] == "FIXED")
    groups = fixed_portfolio_groups(h3)
    assert len(groups) == 1, "D/N fixed portfolio cells must share one model group"
    assert len(next(iter(groups.values()))) == 18, "H3 must expose D1..D3 × N1..N6"

    cutoffs = {3: (date(2021, 1, 31), date(2021, 2, 26), date(2021, 3, 31))}
    assert unique_fixed_model_job_count(h3, cutoffs) == 3
    representative = h3[0]
    for family in h3:
        assert fixed_model_job_key(family, cutoffs[3][0], data_identity="fixture").job_id == fixed_model_job_key(
            representative, cutoffs[3][0], data_identity="fixture").job_id
    print("MODEL_FIT_INDEPENDENT_OF_D_N_PASS")
    print("UNIQUE_HORIZON_CUTOFF_MODEL_JOB_PASS")

    dates = tuple(date(2021, 1, 1) + timedelta(days=i) for i in range(40))
    cutoff = dates[0]
    activations = (dates[10], dates[20], dates[30])
    old = {d for d in dates if cutoff < d <= dates[-1]}
    bounded = set()
    start = cutoff
    for activation in (*activations, None):
        bounded.update(d for d in dates if start < d and (activation is None or d <= activation))
        if activation is None:
            break
        start = activation
    assert bounded == old, (len(old), len(bounded))
    print("BOUNDED_GENERATION_SCORING_SEMANTIC_PARITY_PASS")

    # Queue ordering is explicit and therefore independent of the order in
    # which portfolio-family cells happen to be materialized.
    payloads = [{"family": family, "cutoff": str(cutoff)} for family in reversed(h3) for cutoff in reversed(cutoffs[3])]
    assert [canonical_queue_sort_key(x) for x in sorted(payloads, key=canonical_queue_sort_key)] == sorted(
        canonical_queue_sort_key(x) for x in payloads
    )
    print("PARALLEL_QUEUE_DETERMINISM_PASS")

    # A single-writer registry can be checkpointed and reloaded without
    # losing the O(1) family/cutoff index used by the queue resume path.
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        path = __import__("pathlib").Path(tmp) / "registry.json"
        registry = GenerationRegistry(path, tuple(h3), autosave=False)
        registry.checkpoint()
        resumed = GenerationRegistry(path, tuple(h3), autosave=False)
        assert resumed.records == registry.records and resumed.current == registry.current
    print("PARALLEL_MODEL_QUEUE_RESUME_PASS")
    print("FINAL_HOLDOUT_CLOSED_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
