"""Causal EWMA change detector.  The state is serialisable and has no router import."""
from __future__ import annotations
from datetime import date
from .top10_qbd_health_metrics import ChangePointState

def update_changepoint(previous: ChangePointState, matured_net_excess_observation: float, config, evaluated_at: date) -> ChangePointState:
    decay = float(config.get("decay", .90)); hazard = float(config.get("hazard", .02)); alert = float(config.get("alert", .65))
    baseline = float(previous.opaque_detector_state.get("mean", 0.0))
    scale = float(previous.opaque_detector_state.get("scale", .001))
    surprise = max(0.0, min(1.0, (baseline - matured_net_excess_observation) / max(scale, 1e-9)))
    probability = min(1.0, hazard + (1.0 - hazard) * (previous.change_probability * decay + surprise * (1.0 - decay)))
    return ChangePointState(probability, 1.0 / max(1e-9, hazard + probability * .1), max(.25, 1.0 - probability), evaluated_at,
                            {"mean": decay * baseline + (1-decay) * matured_net_excess_observation, "scale": scale, "alert": alert})
