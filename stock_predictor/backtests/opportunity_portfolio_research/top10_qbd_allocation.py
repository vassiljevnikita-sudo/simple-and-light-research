from __future__ import annotations
import math
def allocation_weights(members, scores, *, mode="softmax", eta=4.0, fallback="MSCI_WORLD"):
    members = tuple(members)
    if not members: return {fallback: 1.0}
    if mode == "equal": return {x: 1.0/len(members) for x in members}
    z = [math.exp(max(-50.0, min(50.0, eta * scores.get(x, 0.0)))) for x in members]
    total = sum(z)
    return {x: v/total for x, v in zip(members, z)}
def validate_weights(weights, suspended=()):
    if any(not math.isfinite(v) or v < 0 for v in weights.values()) or abs(sum(weights.values())-1)>1e-9: raise ValueError("invalid allocation weights")
    if any(weights.get(x, 0) != 0 for x in suspended): raise ValueError("suspended expert allocated")
