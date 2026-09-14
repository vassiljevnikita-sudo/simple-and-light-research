from __future__ import annotations
from .top10_qbd_health_metrics import RegimeScore

def shrink_regime(*, regime_id: str, local_score: float | None, global_score: float, effective_local_n: float, k: float) -> RegimeScore:
    rho = effective_local_n / (effective_local_n + k) if local_score is not None else 0.0
    shrunk = rho * (local_score or 0.0) + (1.0 - rho) * global_score
    return RegimeScore(regime_id, local_score, global_score, rho, shrunk, effective_local_n)
