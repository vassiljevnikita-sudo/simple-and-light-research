"""Stable fingerprints shared by Opportunity-Portfolio research contracts."""
from __future__ import annotations

import hashlib
import json


def stable_hash(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()
