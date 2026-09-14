"""Explicit adapter boundary for the frozen V4 controller.

The controller remains an immutable dependency.  The adapter validates its
contract once and exposes only a causal, scalar gate to the new router.
"""
from __future__ import annotations
from .top10_adaptation_gate_controller_v4 import _validate_contract
from .top10_qbd_expert_registry import V4_CONTROLLER_ID, V4_CONTROLLER_SHA256

class FrozenV4ControllerAdapter:
    controller_id=V4_CONTROLLER_ID
    controller_hash=V4_CONTROLLER_SHA256
    def __init__(self): _validate_contract()
    def causal_gate(self, matured_quality: tuple[float, ...]) -> float:
        if not matured_quality: return 0.0
        positive=sum(x > 0 for x in matured_quality)
        return max(0.0,min(1.0,positive/len(matured_quality)))
