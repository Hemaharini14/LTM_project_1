"""
Plain-language reasons for a delay probability - "evening air-traffic congestion
at Delhi", not "+16 pts".

How a reason is found: the flight is scored again with ONE condition moved to a
calm baseline (a 10:00 departure, clear weather, a Tuesday, no holiday) by the
same predict_delay_probability call that produced its probability - so Indian
routes are explained by the India model and everything else by DelayNetV2,
always matching the number shown. A condition is named as a reason only if
removing it lowers the risk noticeably. Whatever those conditions don't explain
is the airline's normal level on that route, and is said so in words.

These describe what the model responds to, which is not proof of physical cause;
the wording stays with things the inputs actually are (time of day, weather,
day, holiday, the route itself).
"""
from __future__ import annotations

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from predict_delay_v2 import predict_delay_probability, risk_label
from reference_data import AIRPORT_CITY

_CALM = {"temp_f": 60.0, "precip_in": 0.0, "pressure": 29.92, "visibility": 10.0, "wind_speed": 8.0}
_UNKNOWN = {"temp_f": 55.0, "precip_in": 0.0, "pressure": 29.9, "visibility": 10.0, "wind_speed": 8.0}
_DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MIN_EFFECT = 0.02   # a condition must move the probability by 2+ percentage points to be named


def _city(code: str) -> str:
    return AIRPORT_CITY.get((code or "").upper(), (code or "").upper())


def delay_reasons(carrier_code: str, origin_airport: str, destination_airport: str,
                  weekday: int, month: int, scheduled_hour: int, scheduled_elapsed_time: float,
                  weather: dict | None = None, is_holiday: bool = False,
                  dest_weather: dict | None = None, temp_known: bool | None = None) -> dict:
    """{"probability", "risk", "reasons": [plain sentences, biggest first], "helping": [...],
    "summary": one sentence}. weather is the departure-airport conditions the probability
    used (None = unknown, which is then not offered as a reason). temp_known overrides the
    model's temperature-known flag so the probability matches a caller that sets it
    differently (the Delay Check page keeps it False even with a forecast)."""
    origin, dest = _city(origin_airport), _city(destination_airport)
    hour = max(0, min(23, int(scheduled_hour)))

    def score(hour_=hour, weekday_=weekday, wx=weather, holiday_=is_holiday):
        w = wx or _UNKNOWN
        return predict_delay_probability(
            carrier_code=carrier_code, origin_airport=origin_airport, destination_airport=destination_airport,
            weekday=weekday_, month=month, scheduled_elapsed_time=scheduled_elapsed_time,
            origin_temp_f=w["temp_f"], origin_temp_known=(wx is not None) if temp_known is None else temp_known,
            origin_precip_in=w["precip_in"], origin_pressure=w["pressure"],
            origin_visibility=w["visibility"], origin_wind_speed=w["wind_speed"],
            scheduled_hour=hour_, is_holiday=holiday_, dest_weather=dest_weather)

    base = score()
    found = []   # (effect, reason when it adds risk, wording when it lowers risk)

    if hour != 10:
        if hour >= 17:
            up = f"Evening air-traffic congestion at {origin} - delays from earlier flights build up by this time of day"
        elif hour >= 12:
            up = f"Afternoon traffic build-up at {origin} as the day's delays start to accumulate"
        elif hour < 6:
            up = f"Very early departure from {origin}"
        else:
            up = f"Morning peak traffic at {origin}"
        down = (f"Early-morning departure - the first flights of the day at {origin} usually leave on time"
                if hour < 10 else f"Quieter time of day at {origin}")
        found.append((base - score(hour_=10), up, down,
                      "the evening departure" if hour >= 17 else "the afternoon departure" if hour >= 12
                      else "the very early departure" if hour < 6 else "the morning-peak departure"))

    if weather:
        bits = []
        if weather.get("precip_in", 0) >= 0.01:
            bits.append("rain")
        if weather.get("visibility", 10) < 3:
            bits.append("fog/very poor visibility")
        elif weather.get("visibility", 10) < 6:
            bits.append("reduced visibility")
        if weather.get("wind_speed", 0) >= 15:
            bits.append("strong winds")
        what = (", ".join(bits[:-1]) + " and " + bits[-1]) if len(bits) > 1 else (bits[0] if bits else "the weather")
        found.append((base - score(wx=_CALM), f"Forecast {what} at {origin} around departure",
                      f"Clear weather forecast at {origin}", f"the weather at {origin}"))

    if weekday != 1:
        found.append((base - score(weekday_=1), f"Busy {_DAYS[weekday]} travel",
                      f"{_DAYS[weekday]}s are a quieter day to fly", f"{_DAYS[weekday]} travel"))
    if is_holiday:
        found.append((base - score(holiday_=False), "Holiday travel rush", "Holiday timing", "holiday travel"))

    ranked = sorted(found, key=lambda f: -f[0])
    reasons = [f[1] for f in ranked if f[0] >= MIN_EFFECT]
    small = [f[3] for f in ranked if f[0] >= MIN_EFFECT]
    helping = [f[2] for f in found if f[0] <= -MIN_EFFECT]
    explained = sum(f[0] for f in found if f[0] >= MIN_EFFECT)
    risk = risk_label(base)

    # What the named conditions don't account for is this airline's normal level here.
    if risk != "Low" and (not reasons or explained < base * 0.5):
        reasons.append(f"{origin}-{dest} is a busy route with regular congestion, so this airline's flights "
                       f"here are often late even on a normal day")

    if risk == "Low":
        # A low score has no "reasons for delay"; at most something nudges it up a little.
        summary = ("Delays are unlikely" + (f" - only {' and '.join(small[:2])} add{'s' if len(small[:2]) == 1 else ''} "
                                            f"a little risk." if small else
                                            f": {helping[0][0].lower() + helping[0][1:]}." if helping else
                                            " - nothing in this flight's timing or weather raises the risk."))
        reasons = []
    else:
        summary = "Mainly " + reasons[0][0].lower() + reasons[0][1:] + "." if reasons else ""
    return {"probability": base, "risk": risk, "reasons": reasons, "helping": helping, "summary": summary}
