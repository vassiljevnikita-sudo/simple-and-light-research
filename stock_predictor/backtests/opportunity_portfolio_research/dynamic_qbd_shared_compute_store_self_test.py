"""Deterministic checks for cross-seed physical-compute sharing."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
from pathlib import Path
import tempfile
import time

from .dynamic_qbd_shared_compute_store import (
    compute_identity_key, compute_key_lock, materialize_immutable_tree,
    shared_store_contract,
)


def _locked_worker(payload: tuple[str, int]) -> dict:
    root, worker = payload
    identity = {"horizon": 11, "fold": "WF03", "candidate": "HGB_01"}
    requested = time.monotonic()
    with compute_key_lock(root, "candidate-oos", identity) as key:
        entered = time.monotonic()
        time.sleep(.20)
        exited = time.monotonic()
    return {"worker": worker, "key": key, "requested": requested,
            "entered": entered, "exited": exited}


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="dqbd-shared-compute-") as raw:
        root = Path(raw)
        contract_a = shared_store_contract(root)
        contract_b = shared_store_contract(root)
        assert contract_a == contract_b

        identity = {"horizon": 11, "cutoff": "2022-08-31",
                    "training_dates_hash": "A", "recipe": "HGB_01"}
        same = dict(identity)
        changed = dict(identity, training_dates_hash="B")
        assert compute_identity_key("generation", identity) == compute_identity_key("generation", same)
        assert compute_identity_key("generation", identity) != compute_identity_key("generation", changed)

        context = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=2, mp_context=context) as pool:
            rows = list(pool.map(_locked_worker, ((str(root), 0), (str(root), 1))))
        rows.sort(key=lambda row: row["entered"])
        assert rows[0]["key"] == rows[1]["key"]
        assert rows[1]["entered"] >= rows[0]["exited"]

        source = root / "producer" / "artifact"
        source.mkdir(parents=True)
        (source / "model.joblib").write_bytes(b"immutable-model")
        (source / "manifest.json").write_text(
            json.dumps({"sha": "test"}), encoding="utf-8")
        short = root / "views" / "short"
        primary = root / "views" / "primary"
        assert not short.exists() and not primary.exists()
        first = materialize_immutable_tree(source, short)
        second = materialize_immutable_tree(source, primary)
        again = materialize_immutable_tree(source, short)
        assert first["status"] == "MATERIALIZED"
        assert second["status"] == "MATERIALIZED"
        assert again["status"] == "EXISTS"
        assert (short / "model.joblib").read_bytes() == b"immutable-model"
        assert (primary / "model.joblib").read_bytes() == b"immutable-model"
    print("DYNAMIC_QBD_SHARED_COMPUTE_STORE_SELF_TEST_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
