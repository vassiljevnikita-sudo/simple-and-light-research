from __future__ import annotations

import pandas as pd
from .market_regime_contracts import RegimeContract

def classify_urth(urth: pd.DataFrame, contract: RegimeContract = RegimeContract()) -> pd.DataFrame:
    x = urth.sort_values("date").set_index("date")["close"].astype(float)
    ret63 = x.pct_change(63); ret126 = x.pct_change(126); ma200 = x.rolling(200, min_periods=50).mean()
    distance = x / ma200 - 1.0
    drawdown = x / x.cummax() - 1.0
    vol = x.pct_change().rolling(20, min_periods=10).std()
    regime = pd.Series("TRANSITION", index=x.index)
    regime[(ret126 >= contract.bull_return_126) & (distance >= contract.trend_distance)] = "STRONG_BULL"
    regime[(ret126 <= contract.bear_return_126) & (distance <= -contract.trend_distance)] = "STRONG_BEAR"
    sideways = ret126.abs() < abs(contract.bull_return_126) / 2
    sideways &= distance.abs() < contract.trend_distance * 1.5
    sideways &= vol < contract.transition_vol
    regime[sideways] = "SIDEWAYS"
    out = pd.DataFrame({"date": x.index, "urth_close": x.values, "ret63": ret63.values, "ret126": ret126.values,
                        "distance_ma200": distance.values, "trailing_drawdown": drawdown.values, "vol20": vol.values,
                        "regime": regime.values})
    out["regime"] = out["regime"].where(out["ret126"].notna() & out["distance_ma200"].notna(), "TRANSITION")
    return out
