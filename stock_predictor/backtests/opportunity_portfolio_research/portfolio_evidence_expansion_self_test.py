from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from .portfolio_evidence_expansion import EvidenceCheckpoint, _key, _sign_pvalue, _wilson


def main() -> int:
    assert abs(_sign_pvalue(2, 3) - 0.5) < 1e-12
    low, high = _wilson(2, 3)
    assert 0.2 < low < 0.3 and 0.9 < high < 1.0
    job = {
        "horizon": 10, "fold_id": "WF1", "policy_id": "p", "threshold": 0.1,
        "start": "2020-01-01", "end": "2020-02-01", "cost_bps": 20,
        "tax_world": "PRE_TAX", "diagnostic_type": "outer",
    }
    assert _key(job) == _key(dict(job))
    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "checkpoint.jsonl"
        checkpoint = EvidenceCheckpoint(path)
        checkpoint.append("a", {"value": 1})
        assert EvidenceCheckpoint(path).get("a")["value"] == 1
    frame = pd.DataFrame({"horizon": [5], "fold_id": ["WF1"]})
    assert len(frame) == 1
    print("EVIDENCE_EXPANSION_SELF_TEST_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
