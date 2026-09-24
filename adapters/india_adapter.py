"""
Bridges the India catalogue into Module 1's feature space.

The trained classifier (predict_delay_v2) expects a real flight+weather row:
carrier, route, weekday, month, departure hour, block time, origin weather,
holiday/weekend flags, airport congestion, aircraft rotation. A catalogue entry
has only schedule and fare, so this fills the rest from training-set defaults
and RECORDS WHICH ONES, because a prediction built mostly from defaults is a
weaker claim than one built from observed conditions and the caller should be
able to say so.

Two tiers of honesty come out of this:

  * Routes the model actually trained on (the real India data covers
    BLR/BOM/CCU/DEL/HYD) get a genuine score - real carrier, real route, real
    congestion history.
  * Routes it never saw (Chennai, Pune, Goa, Jaipur...) still return a number,
    because the encoder maps unseen categories to its "unknown" index rather
    than failing - but `modelled` comes back False so nothing downstream
    presents it as grounded.

Swap `load_catalog()` for a live flight API and the rest of the pipeline is
unchanged - that is the whole point of keeping this layer separate.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import date, datetime
from functools import lru_cache

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from predict_delay_v2 import predict_delay_probability, risk_label  # noqa: E402
from recovery_tools import get_dataset_index  # noqa: E402
from feature_engineering import is_holiday_date  # noqa: E402

CATALOG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "india_catalog.json"
)

# Training-set defaults for everything the catalogue cannot supply. These are
# the same neutral clear-weather values the web form pre-fills with, so a
# catalogue flight is scored on exactly the assumptions the UI already states.
DEFAULTS = {
    "origin_temp_f": 82.0,          # India rows carry no real temperature at all
    "origin_temp_known": False,     # ...so the model is told to discount it
    "origin_precip_in": 0.0,
    "origin_pressure": 29.9,
    "origin_visibility": 10.0,
    "origin_wind_speed": 8.0,
}


@lru_cache(maxsize=1)
def load_catalog() -> dict:
    with open(CATALOG_PATH, encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def _modelled_airports() -> frozenset[str]:
    """Airports the classifier actually trained on."""
    try:
        return frozenset(get_dataset_index()["airports"])
    except Exception:
        return frozenset()


def city_to_iata(city: str) -> str | None:
    cities = load_catalog()["cities"]
    if city in cities:
        return cities[city]
    lowered = {k.lower(): v for k, v in cities.items()}
    return lowered.get((city or "").strip().lower())


def find_flights(from_city: str, to_city: str) -> list[dict]:
    """Catalogue flights for a city pair, cheapest first."""
    cat = load_catalog()
    a, b = (from_city or "").strip().lower(), (to_city or "").strip().lower()
    out = [
        f for f in cat["flights"]
        if f["from_city"].lower() == a and f["to_city"].lower() == b
    ]
    return sorted(out, key=lambda f: f["fare_inr"])


def hotels_for(city: str) -> list[dict]:
    """Hotel tiers for a city, cheapest first."""
    cat = load_catalog()
    for name, tiers in cat["hotels"].items():
        if name.lower() == (city or "").strip().lower():
            return sorted(tiers, key=lambda h: h["price_per_night"])
    return []


def attractions_for(city: str) -> list[dict]:
    cat = load_catalog()
    for name, items in cat["attractions"].items():
        if name.lower() == (city or "").strip().lower():
            return list(items)
    return []


def to_model_features(flight: dict, travel_date: date) -> tuple[dict, list[str]]:
    """Catalogue entry -> Module 1 kwargs, plus the list of defaulted features."""
    hour = int(flight["depart"].split(":")[0])
    iso = travel_date.isoformat()

    features = {
        "carrier_code": flight["carrier_code"],
        "origin_airport": flight["from"],
        "destination_airport": flight["to"],
        "weekday": travel_date.weekday(),
        "month": travel_date.month,
        "scheduled_elapsed_time": float(flight["duration_min"]),
        "scheduled_hour": hour,
        "is_holiday": is_holiday_date(iso),
        **DEFAULTS,
    }
    # Weather is the whole of what we substitute; everything else above is real
    # schedule data straight off the catalogue entry.
    defaulted = sorted(DEFAULTS.keys())
    return features, defaulted


def score_flight(flight: dict, travel_date: date) -> dict:
    """Runs Module 1 on a catalogue flight.

    Returns the probability alongside `modelled`, which is False when this route
    is outside the trained data - the number is still produced (the encoder
    handles unseen categories) but it is not evidence.
    """
    features, defaulted = to_model_features(flight, travel_date)
    prob = predict_delay_probability(**features)
    known = _modelled_airports()
    modelled = bool(known) and flight["from"] in known and flight["to"] in known

    return {
        "disruption_risk": round(float(prob), 4),
        "risk_label": risk_label(prob),
        "modelled": modelled,
        "defaulted_features": defaulted,
    }


if __name__ == "__main__":
    import io

    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    d = date(2026, 10, 12)
    for pair in [("Chennai", "Delhi"), ("Mumbai", "Delhi"), ("Bengaluru", "Mumbai")]:
        flights = find_flights(*pair)
        print(f"\n{pair[0]} -> {pair[1]}  ({len(flights)} options)")
        for f in flights:
            s = score_flight(f, d)
            tag = "modelled" if s["modelled"] else "UNMODELLED route"
            print(f"  {f['number']:8s} {f['depart']}  Rs{f['fare_inr']:>6,}  "
                  f"risk {s['disruption_risk']:.3f} {s['risk_label']:<8s} [{tag}]")
