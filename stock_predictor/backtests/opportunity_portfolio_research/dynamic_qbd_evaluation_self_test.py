"""Synthetic contract tests for the frozen Step-8 evaluation layer."""
from __future__ import annotations

from datetime import date
import json
from pathlib import Path

import numpy as np

from .dynamic_qbd_evaluation_bootstrap import moving_calendar_block_bootstrap
from .dynamic_qbd_orchestrator_contract import default_pseudolive_experiment_contract
from .dynamic_qbd_run_gate_contract import default_run_gate_contract


BASELINE = "e6b72df4ecb0e595293df62099860781436181dc"


def main() -> int:
    gate = default_run_gate_contract(default_pseudolive_experiment_contract(baseline_commit=BASELINE))
    artifact = Path(__file__).resolve().parents[3] / "research" / "DQBD_CAUSAL_MODEL_STORE_V1_RUN_GATES.json"
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    assert gate.contract_hash == payload["contract_hash"]
    evaluation = gate.evaluation
    assert evaluation.seed_definitions["PRIMARY"] == "2021-02-26"
    assert "S2_MINUS_S0" in evaluation.contrast_definitions
    assert "summary.json" in evaluation.result_schema

    a = np.linspace(.001, .002, 63)
    b = a + .001
    benchmark = np.linspace(.0002, .0004, 63)
    first = moving_calendar_block_bootstrap(a, b, benchmark=benchmark, calendar_days=90)
    second = moving_calendar_block_bootstrap(a, b, benchmark=benchmark, calendar_days=90)
    changed = moving_calendar_block_bootstrap(a, b + .0005, benchmark=benchmark, calendar_days=90)
    assert first == second
    assert first["observed_contrast"] > 0
    assert first["confidence_interval"] == second["confidence_interval"]
    assert first["confidence_interval"] != changed["confidence_interval"]
    assert first["paired_path_length"] == len(a)
    assert first["implementation"] == "DQBD_MOVING_CALENDAR_BLOCK_BOOTSTRAP_V1"
    assert first["metric"] == "cagr_excess"
    assert first["confidence_level"] == .95
    assert first["observed_contrast"] == first["observed_arm_b_cagr_excess"] - first["observed_arm_a_cagr_excess"]
    assert first["observed_arm_a_cagr_excess"] != float(np.mean(a - benchmark))

    for kwargs in ({"confidence_level": .90}, {"block_size": 20},
                   {"repetitions": 1999}, {"seed": 213}):
        try:
            moving_calendar_block_bootstrap(a, b, **kwargs)
        except ValueError as exc:
            assert str(exc) == "DQBD_BOOTSTRAP_PARAMETERS_NOT_FROZEN"
        else:
            raise AssertionError(f"unfrozen bootstrap parameter accepted: {kwargs}")

    # Explicit temporal dependence: the helper rejects an IID-style short input.
    try:
        moving_calendar_block_bootstrap([.1] * 20, [.1] * 20)
    except ValueError as exc:
        assert str(exc) == "DQBD_BOOTSTRAP_PAIRED_SERIES_INVALID"
    else:
        raise AssertionError("short non-block input was accepted")

    # Development boundary remains closed by the upstream contract.
    assert date(2026, 7, 25) > gate.development_end
    print("DQBD_EVALUATION_CONTRACT_SELF_TEST_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
