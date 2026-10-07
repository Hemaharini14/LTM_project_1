"""
Inference wrapper for the trained flight+weather disruption model (v2).
Loads the model + encoder once, then call predict_delay_probability(...)
with a flight + weather snapshot to get a delay probability.
"""
import json
import os
import sys
import joblib
import pandas as pd
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model_preprocessing import FlightWeatherEncoder, CAT_COLS, CONT_COLS
from delay_model_v2 import DelayNetV2
from config import CONGESTION_LOOKUP_PATH

ARTIFACT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "artifacts")
CALIBRATOR_PATH = os.path.join(ARTIFACT_DIR, "calibrator_v2.joblib")
_ENCODER = None
_MODEL = None
_CONGESTION_LOOKUP = None
_CALIBRATOR = None
_CALIBRATOR_LOAD_ATTEMPTED = False


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


def _load_calibrator():
    """The raw model output is NOT a calibrated probability - train_delay_model_v2.py applies
    pos_weight to the loss (so the model doesn't ignore the rare "Delayed" class), which
    systematically inflates every raw score. evaluate_delay_model_v2.py fits a real isotonic
    mapping (raw score -> actual historical delay rate at that score, on real held-out flights)
    and saves it here. Falls back to the uncalibrated raw score (never crashes) if that script
    hasn't been run yet - same "always works, this is a bonus layer" pattern as everywhere else
    in this app, though risk_label()'s thresholds assume the calibrated scale."""
    global _CALIBRATOR, _CALIBRATOR_LOAD_ATTEMPTED
    if not _CALIBRATOR_LOAD_ATTEMPTED:
        _CALIBRATOR_LOAD_ATTEMPTED = True
        if os.path.exists(CALIBRATOR_PATH):
            _CALIBRATOR = joblib.load(CALIBRATOR_PATH)
        else:
            print("[predict_delay_v2] No calibrator found - run evaluate_delay_model_v2.py to "
                  "get real (not inflated) probabilities. Using raw uncalibrated scores for now.")
    return _CALIBRATOR


def _typical_congestion(origin_airport: str, scheduled_hour: int) -> float:
    """Real historical average congestion for this airport+hour (precomputed
    by build_unified_flight_dataset.py), used only when a caller doesn't have
    an actual historical row to read the real value from directly - e.g. a
    hypothetical/ad-hoc chat query. Falls back to the overall training mean,
    never to a fabricated guess."""
    global _CONGESTION_LOOKUP
    if _CONGESTION_LOOKUP is None:
        try:
            with open(CONGESTION_LOOKUP_PATH) as f:
                _CONGESTION_LOOKUP = json.load(f)
        except FileNotFoundError:
            _CONGESTION_LOOKUP = {}
    key = f"{origin_airport}|{scheduled_hour}"
    return _CONGESTION_LOOKUP.get(key, _CONGESTION_LOOKUP.get("__default__", 0.0))


def predict_delay_probability(carrier_code: str, origin_airport: str, destination_airport: str,
                               weekday, month, scheduled_elapsed_time: float,
                               origin_temp_f: float, origin_precip_in: float, origin_pressure: float,
                               origin_visibility: float, origin_wind_speed: float,
                               origin_temp_known: bool = True,
                               scheduled_hour: int = 12, is_holiday: bool = False,
                               origin_hourly_congestion: float | None = None,
                               prev_leg_arrival_delay: float | None = None,
                               leg_of_day: int | None = None,
                               scheduled_turnaround_min: float | None = None,
                               dest_weather: dict | None = None,
                               distance_miles: float | None = None) -> float:
    """Returns a probability in [0, 1] that the flight departs 15+ minutes late.

    origin_temp_known should be False when origin_temp_f is a filled-in
    placeholder rather than a real reading (e.g. India routes, which have no
    temperature data at all - see clean_india_flights.py), so the model can
    discount it instead of treating it as observed weather.

    scheduled_hour/is_holiday default to a neutral midday/non-holiday guess
    when the caller genuinely doesn't know the exact departure time or date
    (e.g. the chatbot's "typical week/month" queries) - pass the real values
    whenever they're available (the web form, or a real historical row) for
    a more accurate result. origin_hourly_congestion, if not given, is looked
    up as this airport+hour's real historical average rather than guessed.
    route_frequency needs no parameter - the encoder derives it internally
    from origin/destination against the routes it was actually trained on.
    """
    encoder, model = _load()
    if origin_hourly_congestion is None:
        origin_hourly_congestion = _typical_congestion(str(origin_airport).upper(), int(scheduled_hour))
    is_weekend = 1 if int(weekday) in (5, 6) else 0

    # Indian routes go to a model trained on them. This network saw 10,634
    # Indian rows out of 5.4M - 0.20% - across four airports, with placeholder
    # constants in temperature and previous-leg delay, the two features it
    # weights most. On real Indian departures it scores AUC 0.594 while a
    # logistic regression on the origin airport alone scores 0.701. The
    # dedicated model scores 0.723 on the same flights; see
    # train_india_model.py. Everything else stays here, where the 5.4M rows
    # are actually relevant.
    #
    # covers() is False when the artifact has never been trained, so this is a
    # no-op on a fresh checkout rather than an import-time dependency.
    try:
        from india_delay_model import covers as _india_covers, predict as _india_predict
        if _india_covers(origin_airport, destination_airport, carrier_code):
            india_prob = _india_predict(
                carrier_code=carrier_code, origin_airport=origin_airport,
                destination_airport=destination_airport, weekday=weekday, month=month,
                scheduled_elapsed_time=scheduled_elapsed_time,
                origin_precip_in=origin_precip_in, origin_pressure=origin_pressure,
                origin_visibility=origin_visibility, origin_wind_speed=origin_wind_speed,
                scheduled_hour=scheduled_hour,
                origin_hourly_congestion=origin_hourly_congestion)
            if india_prob is not None:
                return india_prob
    except Exception as e:
        print(f"[predict_delay_v2] India model unavailable, using DelayNetV2: {e}")

    # Features added with the aircraft-rotation work. A caller checking a flight in
    # advance cannot know how late the inbound aircraft ran, so these default to
    # "unknown" rather than to a value that reads as good news.
    #
    # prev_leg_known=0 is paired with leg_of_day=0 deliberately: in training those
    # always co-occur (no prior leg IS the first leg of the day), so serving them
    # together keeps us on distributions the model actually saw. Passing a real
    # leg_of_day while claiming the prior leg is unknown would be a combination that
    # never appears in training - exactly the kind of mismatch that made
    # origin_temp_known produce wildly different scores for identical input.
    prev_leg_known = 1 if prev_leg_arrival_delay is not None else 0
    feats = {
        "prev_leg_arrival_delay": float(prev_leg_arrival_delay or 0.0),
        "prev_leg_known": prev_leg_known,
        "leg_of_day": int(leg_of_day) if leg_of_day is not None else (0 if not prev_leg_known else 1),
    }
    dest_weather = dest_weather or {}
    feats["dest_weather_known"] = 1 if dest_weather else 0

    # Unlike the other optional fields, distance is always computable with no
    # caller effort - real airport coordinates exist for virtually every route -
    # so a missing value is filled with the real great-circle distance rather
    # than going straight to the training-mean fallback below. Training itself
    # uses the carrier's own REPORTED distance (BTS/India real data), which this
    # approximates closely but is not identical to (no accounting for routing) -
    # see reference_data.distance_miles_between.
    if distance_miles is None:
        from reference_data import distance_miles_between
        distance_miles = distance_miles_between(origin_airport, destination_airport)

    # Anything still unknown is filled with the encoder's own training mean for that
    # column, which the same encoder then scales to exactly 0 - "no signal" - instead
    # of a made-up reading. The *_known flags above tell the model which rows those are.
    # Harmless no-op against a model trained before this column existed - the loaded
    # encoder only reads the columns it was actually fit with (self.cont_cols).
    for col, supplied in [("scheduled_turnaround_min", scheduled_turnaround_min),
                          ("dest_temp_f", dest_weather.get("temp_f")),
                          ("dest_precip_in", dest_weather.get("precip_in")),
                          ("dest_pressure", dest_weather.get("pressure")),
                          ("dest_visibility", dest_weather.get("visibility")),
                          ("dest_wind_speed", dest_weather.get("wind_speed")),
                          ("distance_miles", distance_miles)]:
        feats[col] = float(supplied) if supplied is not None else encoder.cont_mean.get(col, 0.0)

    row = pd.DataFrame([{
        "carrier_code": str(carrier_code),
        "origin_airport": str(origin_airport),
        "destination_airport": str(destination_airport),
        "weekday": str(weekday),
        "month": str(month),
        "scheduled_hour": str(scheduled_hour),
        "scheduled_elapsed_time": scheduled_elapsed_time,
        "origin_temp_f": origin_temp_f, "origin_temp_known": int(bool(origin_temp_known)),
        "origin_precip_in": origin_precip_in,
        "origin_pressure": origin_pressure, "origin_visibility": origin_visibility,
        "origin_wind_speed": origin_wind_speed,
        "is_weekend": is_weekend, "is_holiday": int(bool(is_holiday)),
        "origin_hourly_congestion": origin_hourly_congestion,
        **feats,
    }])
    x_cat = torch.tensor(encoder.transform_cat(row))
    x_cont = torch.tensor(encoder.transform_cont(row))
    with torch.no_grad():
        logit = model(x_cat, x_cont)
        raw_prob = torch.sigmoid(logit).item()
    calibrator = _load_calibrator()
    if calibrator is not None:
        return float(calibrator.predict([raw_prob])[0])
    return raw_prob


def risk_label(prob: float) -> str:
    """Thresholds are set against the CALIBRATED probability scale (see _load_calibrator) -
    checked against real held-out flights: below 0.15 the actual delay rate is under the
    ~18% overall average (genuinely low), 0.15-0.30 covers roughly the average up to ~2x
    it, and 0.30+ real flights in that bucket were actually delayed a third of the time or
    more - see evaluate_delay_model_v2.py's calibration_buckets output for the real numbers
    behind these cutoffs."""
    if prob >= 0.30:
        return "High"
    elif prob >= 0.15:
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