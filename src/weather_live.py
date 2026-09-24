"""
Real forecast weather for both ends of a flight, from Open-Meteo.

The model has ten weather inputs and serving supplied almost none of them.
Origin weather came from form fields most people leave blank, so it fell back
to placeholders with origin_temp_known=0; destination weather was never passed
at all, so dest_weather_known was 0 on every single prediction and all five
dest_* features sat at their training means. This fills both, needs no API key
and has no quota, and forecasts 16 days out - which covers the trip planner's
horizon as well as today's departures.

UNITS ARE THE WHOLE PROBLEM HERE, and getting them wrong would bias every
prediction rather than break anything visibly. Both traps were found by
comparing against the training distribution rather than assumed:

  pressure    Open-Meteo's `surface_pressure` is station pressure at field
              elevation. Delhi sits at 237m, so it reads 28.92 inHg while the
              training column (METAR altimeter, median 29.57, 1st percentile
              24.37) is sea-level. Feeding station pressure would tell the
              model every Delhi departure sits in a deep low. `pressure_msl`
              is the matching field and reads 29.69.

  visibility  the training column is METAR-derived and caps at 10 miles -
              its median AND its 99th percentile are both exactly 10.00.
              Open-Meteo reports true optical range, which was 24 to 49 miles
              on a clear Delhi day. That is not "better data", it is off the
              end of everything the model has seen, so it is clamped.

Everything returns None rather than raising: a forecast that cannot be
fetched leaves the caller on its existing placeholder path.
"""
from __future__ import annotations

import os
import sys
import time
from datetime import date, datetime

import httpx

try:
    import truststore
    truststore.inject_into_ssl()
except ImportError:
    pass

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from reference_data import AIRPORT_COORDS  # noqa: E402

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
HOURLY = "temperature_2m,precipitation,pressure_msl,visibility,wind_speed_10m"

# The training column tops out here (median and p99 are both 10.00), so a real
# 40-mile optical range is out of distribution rather than merely unusual.
MAX_VISIBILITY_MI = 10.0

# Open-Meteo publishes 16 days; beyond that there is no forecast to have.
MAX_FORECAST_DAYS = 16

_HPA_TO_INHG = 0.02953
_M_TO_MI = 1 / 1609.34

_cache: dict[tuple, dict | None] = {}
_CACHE_TTL_S = 1800.0
_cache_times: dict[tuple, float] = {}


def _fetch(lat: float, lon: float, on: date) -> dict | None:
    try:
        r = httpx.get(FORECAST_URL, params={
            "latitude": lat, "longitude": lon, "hourly": HOURLY,
            "temperature_unit": "fahrenheit", "wind_speed_unit": "mph",
            "precipitation_unit": "inch",
            "start_date": on.isoformat(), "end_date": on.isoformat(),
        }, timeout=12)
        r.raise_for_status()
        return r.json().get("hourly")
    except Exception as e:
        print(f"[weather_live] forecast failed for {lat},{lon} on {on}: {e}")
        return None


def airport_weather(iata: str, when: datetime | None = None) -> dict | None:
    """Forecast conditions at an airport for a given hour.

    Returns the five fields the model names, in the units it was trained on,
    or None when the airport has no published coordinates or the day is beyond
    the forecast horizon.
    """
    coords = AIRPORT_COORDS.get((iata or "").upper())
    if not coords:
        return None
    when = when or datetime.now()
    on, hour = when.date(), when.hour

    days_out = (on - date.today()).days
    if days_out < 0 or days_out > MAX_FORECAST_DAYS:
        return None

    key = ((iata or "").upper(), on.isoformat(), hour)
    if key in _cache and (time.time() - _cache_times.get(key, 0)) < _CACHE_TTL_S:
        return _cache[key]

    hourly = _fetch(coords[0], coords[1], on)
    result = None
    if hourly:
        try:
            i = min(max(hour, 0), len(hourly["temperature_2m"]) - 1)

            def at(field, default=None):
                v = hourly.get(field, [None] * (i + 1))[i]
                return default if v is None else float(v)

            pressure_hpa = at("pressure_msl")
            visibility_m = at("visibility")
            result = {
                "temp_f": round(at("temperature_2m", 70.0), 1),
                "precip_in": round(at("precipitation", 0.0), 3),
                # Sea-level, to match the training column - see module docstring.
                "pressure": round(pressure_hpa * _HPA_TO_INHG, 2) if pressure_hpa else 29.92,
                # Clamped, for the same reason.
                "visibility": (round(min(visibility_m * _M_TO_MI, MAX_VISIBILITY_MI), 1)
                               if visibility_m is not None else 10.0),
                "wind_speed": round(at("wind_speed_10m", 8.0), 1),
                "source": "open-meteo",
            }
        except (KeyError, IndexError, TypeError, ValueError) as e:
            print(f"[weather_live] could not read forecast for {iata}: {e}")
            result = None

    _cache[key] = result
    _cache_times[key] = time.time()
    return result


def route_weather(origin_iata: str, destination_iata: str,
                  when: datetime | None = None) -> tuple[dict | None, dict | None]:
    """Both ends at once. Either side may be None independently."""
    return airport_weather(origin_iata, when), airport_weather(destination_iata, when)


if __name__ == "__main__":
    import io
    from datetime import timedelta
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

    now = datetime.now()
    print(f"{'airport':8s} {'when':12s} {'temp':>7s} {'precip':>8s} {'pressure':>9s} "
          f"{'vis':>6s} {'wind':>7s}")
    for iata in ["DEL", "BOM", "MAA", "BLR", "JFK", "ORD"]:
        w = airport_weather(iata, now + timedelta(hours=6))
        if not w:
            print(f"{iata:8s} no coordinates or outside forecast range")
            continue
        print(f"{iata:8s} {'+6h':12s} {w['temp_f']:6.1f}F {w['precip_in']:7.3f}in "
              f"{w['pressure']:8.2f}in {w['visibility']:5.1f}mi {w['wind_speed']:6.1f}mph")

    print()
    for days in (0, 7, 15, 20):
        w = airport_weather("DEL", now + timedelta(days=days))
        print(f"DEL +{days:2}d  " + (f"{w['temp_f']}F, {w['pressure']}inHg, vis {w['visibility']}mi"
                                      if w else "outside the 16-day forecast horizon"))
