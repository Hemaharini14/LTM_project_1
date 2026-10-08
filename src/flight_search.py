"""
Real-time flight search wrapped around the EXISTING delay model.

    Aviationstack (aviationstack.py)  ->  normalised flight
    Open-Meteo     (weather_live.py)  ->  departure / arrival conditions
    predict_delay_v2.predict_delay_probability  ->  probability (unchanged model,
                                                    encoder and calibrator)
    delay_duration.estimate_delay_duration      ->  historical "if delayed" length

Nothing here trains, edits or re-implements the model. This layer only adapts
what the APIs return to the arguments predict_delay_probability already takes,
and declines - with a stated reason - when an input it needs does not exist,
instead of substituting a default. In particular:

  * no weather forecast for the departure airport  -> no prediction
  * no way to know the scheduled duration          -> no prediction
  * route not in the trained data                  -> "not modelled"
  * the model has no delay-minutes output, so none is claimed. The only minutes
    shown beyond what the airline itself reports are labelled as the historical
    median of flights that WERE delayed, which is a statistic, not a forecast.
"""
from __future__ import annotations

import os
import re
import sys
from datetime import date, datetime, timedelta
from datetime import date as Date
from zoneinfo import ZoneInfo

from pydantic import BaseModel, Field, field_validator, model_validator

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import aviationstack
from delay_duration import estimate_delay_duration
from feature_engineering import is_holiday_date
from model_metrics import load_model_metrics
from predict_delay_v2 import predict_delay_probability, risk_label
from recovery_tools import is_route_covered, lookup_flight_by_number
from weather_live import airport_weather, MAX_FORECAST_DAYS

_IATA = re.compile(r"^[A-Za-z]{3}$")
_FLIGHT = re.compile(r"^[A-Za-z0-9]{2}\s?\d{1,4}[A-Za-z]?$")


class FlightSearchError(Exception):
    """An error with an HTTP status and a message that is safe to show a user."""

    def __init__(self, code: str, message: str, status: int = 400):
        super().__init__(message)
        self.code, self.message, self.status = code, message, status


# ------------------------------------------------------------------ request models
class SearchQuery(BaseModel):
    origin: str | None = None
    destination: str | None = None
    flight_number: str | None = None
    date: Date | None = None

    @field_validator("origin", "destination", mode="before")
    @classmethod
    def _airport(cls, v):
        v = (v or "").strip()
        if not v:
            return None
        # Accept "MAA - Chennai" as typed from the datalist.
        v = v.split("-")[0].split(" ")[0].strip()
        if not _IATA.match(v):
            raise ValueError(f"'{v}' is not a 3-letter IATA airport code (e.g. MAA, LHR).")
        return v.upper()

    @field_validator("flight_number", mode="before")
    @classmethod
    def _flight(cls, v):
        v = (v or "").strip()
        if not v:
            return None
        if not _FLIGHT.match(v):
            raise ValueError(f"'{v}' is not a flight number like EK543 or 6E 2134.")
        return v.replace(" ", "").upper()

    @field_validator("date", mode="before")
    @classmethod
    def _date(cls, v):
        if v in (None, ""):
            return None
        try:
            return datetime.strptime(str(v), "%Y-%m-%d").date()
        except ValueError:
            raise ValueError("date must be YYYY-MM-DD.")

    @model_validator(mode="after")
    def _enough(self):
        if not (self.flight_number or self.origin or self.destination):
            raise ValueError("Enter a flight number, or at least a departure or arrival airport.")
        return self


class PredictRequest(BaseModel):
    """The normalised flight, exactly as /api/flights/search returned it."""
    airline_iata: str = Field(min_length=2, max_length=3)
    origin_iata: str = Field(min_length=3, max_length=3)
    destination_iata: str = Field(min_length=3, max_length=3)
    scheduled_departure: str
    scheduled_arrival: str | None = None
    flight_number: str | None = None
    origin_timezone: str | None = None
    destination_timezone: str | None = None
    status: str | None = None

    @field_validator("airline_iata", "origin_iata", "destination_iata")
    @classmethod
    def _upper(cls, v):
        return v.strip().upper()

    @field_validator("scheduled_departure")
    @classmethod
    def _dep(cls, v):
        if _local_dt(v) is None:
            raise ValueError("scheduled_departure must be an ISO date-time.")
        return v


# ------------------------------------------------------------------ helpers
def _local_dt(s: str | None) -> datetime | None:
    """Aviationstack stamps LOCAL airport time with a '+00:00' suffix, so the
    clock reading is right and the offset is not - the offset is discarded."""
    if not s or len(s) < 16:
        return None
    try:
        return datetime.strptime(s[:16], "%Y-%m-%dT%H:%M")
    except ValueError:
        return None


def _elapsed_minutes(f: dict) -> tuple[float | None, str | None]:
    """(minutes, how it was obtained). From the API's own scheduled times and
    airport time zones when both exist; otherwise from the 2019 catalogue's
    record of that exact flight number on that exact route; otherwise unknown."""
    dep, arr = _local_dt(f.get("scheduled_departure")), _local_dt(f.get("scheduled_arrival"))
    otz, dtz = f.get("origin_timezone"), f.get("destination_timezone")
    if dep and arr and otz and dtz:
        try:
            utc_dep = dep.replace(tzinfo=ZoneInfo(otz)).astimezone(ZoneInfo("UTC"))
            utc_arr = arr.replace(tzinfo=ZoneInfo(dtz)).astimezone(ZoneInfo("UTC"))
            minutes = (utc_arr - utc_dep).total_seconds() / 60
            if 20 <= minutes <= 24 * 60:
                return minutes, "airline schedule (departure and arrival times, converted for time zones)"
        except Exception:
            pass
    carrier, number = f.get("airline_iata"), f.get("flight_number")
    if carrier and number:
        rec = lookup_flight_by_number(carrier, number)
        if (rec and rec["origin_airport"] == f.get("origin_iata")
                and rec["destination_airport"] == f.get("destination_iata")):
            return float(rec["scheduled_elapsed_time"]), (
                f"2019 catalogue: this flight number flew this route {rec['occurrences']} times")
    return None, None


def _threshold() -> float | None:
    """The calibrated decision threshold the evaluation script measured on real
    held-out flights (metrics_v2.json) - not a number chosen here."""
    m = load_model_metrics() or {}
    t = m.get("recommended_threshold")
    return float(t) if t is not None else None


def _model_name(carrier: str, origin: str, dest: str) -> str:
    try:
        from india_delay_model import covers
        if covers(origin, dest, carrier):
            return "India route model (DelayNetV2 is not used for these routes)"
    except Exception:
        pass
    return "DelayNetV2"


def flight_id(f: dict, day: date) -> str | None:
    code = f.get("flight_iata") or (f"{f.get('airline_iata') or ''}{f.get('flight_number') or ''}" or None)
    if not code or not f.get("origin_iata"):
        return None
    return f"{code}_{f['origin_iata']}_{day.isoformat()}"


# ------------------------------------------------------------------ prediction
def predict_flight(f: dict) -> dict:
    """Run the existing model for one normalised flight. Always returns a dict;
    `status` says whether a prediction exists and `reason` says why not."""
    def decline(status: str, reason: str, **extra):
        return {"status": status, "reason": reason, "delay_probability": None,
                "predicted": None, "model": None, **extra}

    if (f.get("status") or "").lower() == "cancelled":
        return decline("unavailable", "This flight is cancelled; there is nothing to predict.")

    carrier, origin, dest = f.get("airline_iata"), f.get("origin_iata"), f.get("destination_iata")
    dep = _local_dt(f.get("scheduled_departure"))
    if not (carrier and origin and dest and dep):
        return decline("unavailable", "The API did not return the airline, airports and scheduled "
                                       "departure the model needs.")

    if not is_route_covered(origin, dest):
        return decline("not_modeled", f"{origin} to {dest} is not in the flight data the model was "
                                       "trained on, so any number would be unfounded.")

    elapsed, elapsed_src = _elapsed_minutes(f)
    if elapsed is None:
        return decline("unavailable", "The scheduled flight duration is not available from the API "
                                       "or the historical catalogue, so a reliable prediction can't be made.")

    days_out = (dep.date() - date.today()).days
    if days_out > MAX_FORECAST_DAYS:
        return decline("unavailable", f"The weather forecast only reaches {MAX_FORECAST_DAYS} days ahead; "
                                       "the model needs weather at the departure airport.")
    origin_wx = airport_weather(origin, dep)
    if origin_wx is None:
        return decline("unavailable", f"No weather forecast could be retrieved for {origin} at "
                                       "departure time, and the model is not run on assumed weather.")
    arr_dt = dep + timedelta(minutes=elapsed)
    dest_wx = airport_weather(dest, arr_dt)   # optional input: the model has a 'known' flag for it

    prob = predict_delay_probability(
        carrier_code=carrier, origin_airport=origin, destination_airport=dest,
        weekday=dep.weekday(), month=dep.month, scheduled_elapsed_time=elapsed,
        origin_temp_f=origin_wx["temp_f"], origin_temp_known=True,
        origin_precip_in=origin_wx["precip_in"], origin_pressure=origin_wx["pressure"],
        origin_visibility=origin_wx["visibility"], origin_wind_speed=origin_wx["wind_speed"],
        scheduled_hour=dep.hour, is_holiday=is_holiday_date(dep.date().isoformat()),
        dest_weather=dest_wx,
    )
    threshold = _threshold()
    tier = risk_label(prob)
    delayed = (prob >= threshold) if threshold is not None else (tier == "High")

    duration = estimate_delay_duration(carrier, origin, dest, dep.hour) if (delayed or tier != "Low") else None
    return {
        "status": "ok", "reason": None,
        "delay_probability": round(prob, 4),
        "predicted": "DELAYED" if delayed else "ON TIME",
        "risk_tier": tier,
        "threshold": threshold,
        "threshold_note": ("Decision threshold from this model's own held-out evaluation "
                           "(maximises F1)." if threshold is not None
                           else "No evaluation threshold on file; using the app's High-risk tier."),
        "model": _model_name(carrier, origin, dest),
        # NOT a prediction for this flight - see the module docstring.
        "typical_delay_if_delayed": ({
            "median_min": duration["median_min"], "p90_min": duration["p90_min"],
            "sample_size": duration["sample_size"], "basis": duration["basis_label"],
            "top_cause": duration["top_cause"],
        } if duration else None),
        "inputs": {
            "airline": carrier, "origin": origin, "destination": dest,
            "weekday": dep.strftime("%A"), "month": dep.strftime("%B"),
            "scheduled_hour": dep.hour, "holiday": is_holiday_date(dep.date().isoformat()),
            "scheduled_duration_min": round(elapsed), "duration_source": elapsed_src,
            "previous_leg_delay": "unknown (not published before departure)",
        },
        "weather": {"origin": origin_wx, "destination": dest_wx,
                    "destination_note": None if dest_wx else
                    "Arrival-airport forecast unavailable; the model was told it is unknown."},
    }


# ------------------------------------------------------------------ search
_API_ERRORS = {
    "no_key": ("missing_api_key", 503, "Live flight data isn't configured: AVIATIONSTACK_API_KEY is not set on the server."),
    "invalid_key": ("invalid_api_key", 503, "The Aviationstack API key was rejected. Check AVIATIONSTACK_API_KEY."),
    "quota_exhausted": ("quota_exhausted", 429, "This month's Aviationstack allowance is used up, so no new live searches can be made."),
    "rate_limited": ("rate_limited", 429, "Aviationstack is rate-limiting requests. Wait a minute and try again."),
    "plan_restricted": ("plan_restricted", 502, "Aviationstack's free plan doesn't include this lookup."),
    "api_unreachable": ("api_unreachable", 502, "Couldn't reach Aviationstack (timeout or network error). Try again shortly."),
    "api_error": ("api_error", 502, "Aviationstack returned an error for this search."),
}


def search_flights(q: SearchQuery, limit: int = 10) -> dict:
    today = date.today()
    day = q.date or today
    if day < today:
        raise FlightSearchError("invalid_date", "Pick today or a future date; past dates aren't "
                                                "available on the live data plan.")
    if day == today:
        rows = aviationstack.search_live(q.origin, q.destination, q.flight_number, limit=50)
        mode = "live"
    else:
        if not q.origin:
            raise FlightSearchError("origin_required", "For a future date, enter the departure airport "
                                                       "(the schedule is looked up by airport).")
        rows = aviationstack.search_future(q.origin, day, q.destination, q.flight_number, limit=50)
        mode = "schedule"

    info = aviationstack.last_call_info()
    if rows is None:
        code, status, msg = _API_ERRORS.get(info["status"], _API_ERRORS["api_error"])
        raise FlightSearchError(code, msg, status)

    if mode == "live":
        rows = [r for r in rows if not _local_dt(r["scheduled_departure"])
                or _local_dt(r["scheduled_departure"]).date() == day]
    rows = rows[:limit]
    if not rows:
        raise FlightSearchError("no_flights", "No flights matched that search. Check the airports, "
                                              "flight number and date.", 404)

    flights = []
    for r in rows:
        r = dict(r)
        r["id"] = flight_id(r, day)
        r["duration_min"] = (lambda e: round(e) if e else None)(
            _elapsed_minutes(r)[0] if r.get("scheduled_arrival") else None)
        r["prediction"] = predict_flight(r)
        flights.append(r)

    return {
        "query": {"origin": q.origin, "destination": q.destination,
                  "flight_number": q.flight_number, "date": day.isoformat()},
        "count": len(flights),
        "flights": flights,
        "meta": {
            "mode": mode,
            "cached": info["cached"], "cache_age_s": info["age_s"], "stale": info["stale"],
            "notice": ("Data retrieved from cache" + (f" ({info['age_s']}s old)" if info["age_s"] else "")
                       + (" - the live API quota is exhausted, so this may be out of date." if info["stale"] else ""))
            if info["cached"] else None,
            "quota": aviationstack.usage(),
        },
    }


def flight_details(fid: str) -> dict:
    m = re.match(r"^([A-Za-z0-9]{3,7})_([A-Za-z]{3})_(\d{4}-\d{2}-\d{2})$", fid or "")
    if not m:
        raise FlightSearchError("invalid_flight_id", "That flight link is not valid.")
    code, origin, day = m.group(1), m.group(2), m.group(3)
    res = search_flights(SearchQuery(origin=origin, flight_number=code, date=day), limit=5)
    for f in res["flights"]:
        if f["id"] and f["id"].upper() == fid.upper():
            return {"flight": f, "meta": res["meta"]}
    return {"flight": res["flights"][0], "meta": res["meta"]}


def optional_fare(f: dict) -> dict | None:
    """Real Google Flights fare via the existing SerpApi integration (quota-capped,
    disk-cached). Aviationstack carries no prices, so this is the only source."""
    try:
        from serpapi_prices import real_flight_price
        day = (f.get("scheduled_departure") or "")[:10]
        return real_flight_price(f["origin_iata"], f["destination_iata"], day, adults=1)
    except Exception as e:
        print(f"[flight_search] fare lookup skipped: {e}")
        return None
