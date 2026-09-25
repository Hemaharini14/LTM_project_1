"""
Serving side of the India delay model - see train_india_model.py for why.

Short version: DelayNetV2 has 0.20% Indian training data, four airports, and
placeholder constants where its strongest features should be. On real Indian
departures it scores AUC 0.594. This scores 0.723 on the same flights.

Routing is by carrier or by airport, not by country lookup, because both are
things the caller already has. A flight only reaches this model when it is
plainly an Indian operation; everything else stays on DelayNetV2, which is
better at what it was actually trained on.

If the artifact is missing - nobody has run the trainer - covers() returns
False and every caller falls through to DelayNetV2 unchanged.
"""
from __future__ import annotations

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from train_india_model import (  # noqa: E402
    FEATURES, INDIA_CARRIERS, MODEL_PATH,
)

# Airports that make a flight an Indian operation even under a foreign
# designator - a codeshare out of Delhi is still a Delhi departure.
INDIA_AIRPORTS = {
    "DEL", "BOM", "BLR", "MAA", "HYD", "CCU", "GOI", "GOX", "PNQ", "AMD", "JAI",
    "COK", "TRV", "LKO", "PAT", "NAG", "IXC", "IXB", "IXM", "IXE", "IXR", "IXZ",
    "VNS", "BBI", "GAU", "SXR", "ATQ", "RPR", "IDR", "BHO", "VTZ", "TIR", "CJB",
    "HBX", "STV", "UDR", "JDH", "DED", "DIB", "IMF", "AGR",
}

_cache: dict = {}


def _bundle() -> dict | None:
    """The saved model, loaded once. None when it has never been trained."""
    if "b" not in _cache:
        try:
            import joblib
            _cache["b"] = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None
        except Exception as e:
            print(f"[india_delay_model] could not load {MODEL_PATH}: {e}")
            _cache["b"] = None
    return _cache["b"]


def covers(origin_airport: str, destination_airport: str, carrier_code: str) -> bool:
    """Should this flight be scored by the India model rather than DelayNetV2?"""
    if _bundle() is None:
        return False
    o = (origin_airport or "").upper()
    d = (destination_airport or "").upper()
    c = (carrier_code or "").upper()
    return c in INDIA_CARRIERS or o in INDIA_AIRPORTS or d in INDIA_AIRPORTS


def predict(carrier_code: str, origin_airport: str, destination_airport: str,
            weekday, month, scheduled_elapsed_time: float,
            origin_precip_in: float, origin_pressure: float,
            origin_visibility: float, origin_wind_speed: float,
            scheduled_hour: int = 12,
            origin_hourly_congestion: float | None = None) -> float | None:
    """Calibrated probability of departing 15+ minutes late, or None.

    Takes no temperature and no previous-leg delay on purpose: both are
    constants on every Indian row in the training data, so they carry no
    signal and passing them would only imply otherwise.
    """
    bundle = _bundle()
    if bundle is None:
        return None
    import pandas as pd

    row = pd.DataFrame([{
        "origin_airport": (origin_airport or "").upper(),
        "destination_airport": (destination_airport or "").upper(),
        "carrier_code": (carrier_code or "").upper(),
        "scheduled_hour": int(scheduled_hour),
        "weekday": int(weekday),
        "month": int(month),
        "origin_hourly_congestion": (float(origin_hourly_congestion)
                                     if origin_hourly_congestion is not None else 0.0),
        "origin_precip_in": float(origin_precip_in),
        "origin_visibility": float(origin_visibility),
        "origin_wind_speed": float(origin_wind_speed),
        "origin_pressure": float(origin_pressure),
        "scheduled_elapsed_time": float(scheduled_elapsed_time),
    }])
    try:
        return float(bundle["model"].predict_proba(row[FEATURES])[0, 1])
    except Exception as e:
        print(f"[india_delay_model] scoring failed, falling back: {e}")
        return None


def info() -> dict | None:
    """What the saved model was trained on, for the admin page and the README."""
    b = _bundle()
    if not b:
        return None
    return {k: b.get(k) for k in
            ("trained_rows", "historical_rows", "collected_rows", "base_rate", "cv_auc")}


if __name__ == "__main__":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    print("model info:", info())
    print()
    for o, d, c in [("DEL", "BOM", "6E"), ("BOM", "DEL", "AI"), ("MAA", "DEL", "6E"),
                    ("ORD", "DEN", "UA"), ("JFK", "LAX", "AA")]:
        if not covers(o, d, c):
            print(f"  {c} {o}->{d}  not an India route - stays on DelayNetV2")
            continue
        for hour in (6, 12, 19):
            p = predict(c, o, d, weekday=4, month=10, scheduled_elapsed_time=130,
                        origin_precip_in=0.0, origin_pressure=29.9,
                        origin_visibility=10.0, origin_wind_speed=8.0,
                        scheduled_hour=hour, origin_hourly_congestion=0.0)
            print(f"  {c} {o}->{d} {hour:02d}:00  {p:.1%}")
