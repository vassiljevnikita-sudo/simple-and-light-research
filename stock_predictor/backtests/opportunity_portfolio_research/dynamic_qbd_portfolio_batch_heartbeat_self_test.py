"""Regression check for advisory seed heartbeat during v40.1 batches."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile

from .dynamic_qbd_batched_portfolio_runtime import _write_seed_activity_heartbeat


def run_self_test() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "seed-bank-heartbeat.json"
        _write_seed_activity_heartbeat(
            path,
            owner="dqbd-step9-test-primary",
            state="ACTIVE",
        )
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["schema_version"] == "DQBD_SEED_BANK_ACTIVITY_V1"
        assert payload["state"] == "ACTIVE"
        assert payload["owner"] == "dqbd-step9-test-primary"
        assert payload["activity_source"] == "PORTFOLIO_BATCH_PARENT"
        assert float(payload["updated_at_epoch"]) > 0.0
        assert not tuple(path.parent.glob(path.name + ".*.tmp"))

    print("dqbd portfolio batch heartbeat self-test: PASS")


if __name__ == "__main__":
    run_self_test()
