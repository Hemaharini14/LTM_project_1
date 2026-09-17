"""
Inference wrapper for the trained flight+weather disruption model (v2).
Loads the model + encoder once, then call predict_delay_probability(...)
with a flight + weather snapshot to get a delay probability.
"""
import os
import sys
import pandas as pd
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model_preprocessing import FlightWeatherEncoder, CAT_COLS, CONT_COLS
from delay_model_v2 import DelayNetV2

ARTIFACT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "artifacts")
_ENCODER = None
_MODEL = None


def _load():
    global _ENCODER, _MODEL
    if _ENCODER is None:
        _ENCODER = FlightWeatherEncoder.load(os.path.join(ARTIFACT_DIR, "delay_encoder_v2.joblib"))
    if _MODEL is None:
        ckpt = torch.load(os.path.join(ARTIFACT_DIR, "delay_model_v2.pt"), map_location="cpu")
        _MODEL = DelayNetV2(ckpt["vocab_sizes"], ckpt["n_continuous"])
        _MODEL.load_state_dict(ckpt["model_state"])
        _MODEL.eval()
    return _ENCODER, _MODEL


def predict_delay_probability(carrier_code: str, origin_airport: str, destination_airport: str,
                               weekday, month, scheduled_elapsed_time: float,
                               origin_temp_f: float, origin_precip_in: float, origin_pressure: float,
                               origin_visibility: float, origin_wind_speed: float,
                               origin_temp_known: bool = True) -> float:
    """Returns a probability in [0, 1] that the flight departs 15+ minutes late.

    origin_temp_known should be False when origin_temp_f is a filled-in
    placeholder rather than a real reading (e.g. India routes, which have no
    temperature data at all - see clean_india_flights.py), so the model can
    discount it instead of treating it as observed weather.
    """
    encoder, model = _load()
    row = pd.DataFrame([{
        "carrier_code": str(carrier_code),
        "origin_airport": str(origin_airport),
        "destination_airport": str(destination_airport),
        "weekday": str(weekday),
        "month": str(month),
        "scheduled_elapsed_time": scheduled_elapsed_time,
        "origin_temp_f": origin_temp_f, "origin_temp_known": int(bool(origin_temp_known)),
        "origin_precip_in": origin_precip_in,
        "origin_pressure": origin_pressure, "origin_visibility": origin_visibility,
        "origin_wind_speed": origin_wind_speed,
    }])
    x_cat = torch.tensor(encoder.transform_cat(row))
    x_cont = torch.tensor(encoder.transform_cont(row))
    with torch.no_grad():
        logit = model(x_cat, x_cont)
        prob = torch.sigmoid(logit).item()
    return prob


def risk_label(prob: float) -> str:
    if prob >= 0.65:
        return "High"
    elif prob >= 0.45:
        return "Moderate"
    return "Low"


if __name__ == "__main__":
    # Example: a flight with poor origin-side weather (low visibility, some
    # precipitation) should generally score higher risk than clear conditions.
    p_bad_weather = predict_delay_probability(
        carrier_code="WN", origin_airport="ORD", destination_airport="LGA",
        weekday="5", month="12", scheduled_elapsed_time=140,
        origin_temp_f=28, origin_precip_in=0.4, origin_pressure=29.5,
        origin_visibility=2.0, origin_wind_speed=22,
    )
    p_clear_weather = predict_delay_probability(
        carrier_code="WN", origin_airport="ORD", destination_airport="LGA",
        weekday="5", month="12", scheduled_elapsed_time=140,
        origin_temp_f=45, origin_precip_in=0.0, origin_pressure=30.2,
        origin_visibility=10.0, origin_wind_speed=5,
    )
    print(f"Bad weather scenario:  {p_bad_weather:.3f} ({risk_label(p_bad_weather)})")
    print(f"Clear weather scenario:{p_clear_weather:.3f} ({risk_label(p_clear_weather)})")