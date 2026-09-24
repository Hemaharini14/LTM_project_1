"""
Why THIS flight is risky - attribution against the trained model itself.

delay_duration.py answers a historical question: "flights like this one, when
they were late, ran N minutes, and those minutes were attributed to X." That is
a property of the route's past, not of the flight in front of you. It does not
move when you change the weather, because it cannot.

This answers the other question, the one people actually ask: given the
conditions entered for THIS flight, which of them are pushing the risk up?

Method: counterfactual ablation. Re-score the flight with one condition reset
to a neutral baseline and measure how far the probability moves. If dropping
visibility back to 10 miles takes the risk from 41% to 23%, then poor
visibility is worth +18 points on this flight, according to the model that
produced the number - not according to a table of national averages.

That makes the explanation genuinely conditional: change the weather in the
form and the reasons change with it. It is also honest about its own limits -
these are the model's sensitivities, which is a claim about the model, not a
claim about physical causation.

One forward pass per factor (a dozen), batched, so the whole explanation costs
about as much as a single prediction.
"""
from __future__ import annotations

import os
import sys

import pandas as pd
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from predict_delay_v2 import (  # noqa: E402
    _load, _load_calibrator, _typical_congestion, predict_delay_probability,
)

# Neutral = benign conditions at a quiet time. Each factor is measured against
# these, so "contribution" means "how much worse than a calm midday departure".
NEUTRAL = {
    "origin_visibility": 10.0,
    "origin_precip_in": 0.0,
    "origin_wind_speed": 8.0,
    "origin_pressure": 29.9,
    "origin_temp_f": 60.0,
    "scheduled_hour": 11,
    "is_holiday": 0,
    "is_weekend": 0,
}

# Human labels, plus how to phrase the observed value.
FACTORS = [
    ("origin_visibility", "Visibility", lambda v: f"{v:g} mi"),
    ("origin_precip_in", "Precipitation", lambda v: f"{v:g} in"),
    ("origin_wind_speed", "Wind speed", lambda v: f"{v:g} mph"),
    ("origin_temp_f", "Temperature", lambda v: f"{v:g}°F"),
    ("origin_pressure", "Air pressure", lambda v: f"{v:g} inHg"),
    ("scheduled_hour", "Departure time", lambda v: f"{int(v):02d}:00"),
    ("origin_hourly_congestion", "Airport congestion", lambda v: f"{v:.2f} vs typical"),
    ("is_holiday", "Holiday travel", lambda v: "yes" if v else "no"),
    ("is_weekend", "Weekend", lambda v: "yes" if v else "no"),
    ("scheduled_elapsed_time", "Flight length", lambda v: f"{int(v)} min"),
]


def _build_row(f: dict) -> dict:
    return {
        "carrier_code": str(f["carrier_code"]),
        "origin_airport": str(f["origin_airport"]),
        "destination_airport": str(f["destination_airport"]),
        "weekday": str(f["weekday"]),
        "month": str(f["month"]),
        "scheduled_hour": str(int(f["scheduled_hour"])),
        "scheduled_elapsed_time": float(f["scheduled_elapsed_time"]),
        "origin_temp_f": float(f["origin_temp_f"]),
        "origin_temp_known": int(bool(f.get("origin_temp_known", True))),
        "origin_precip_in": float(f["origin_precip_in"]),
        "origin_pressure": float(f["origin_pressure"]),
        "origin_visibility": float(f["origin_visibility"]),
        "origin_wind_speed": float(f["origin_wind_speed"]),
        "is_weekend": int(f["is_weekend"]),
        "is_holiday": int(bool(f["is_holiday"])),
        "origin_hourly_congestion": float(f["origin_hourly_congestion"]),
        "leg_of_day": int(f.get("leg_of_day", 0)),
        "scheduled_turnaround_min": float(f["scheduled_turnaround_min"]),
        "prev_leg_arrival_delay": float(f.get("prev_leg_arrival_delay", 0.0)),
        "prev_leg_known": int(f.get("prev_leg_known", 0)),
        "dest_temp_f": float(f["dest_temp_f"]),
        "dest_precip_in": float(f["dest_precip_in"]),
        "dest_pressure": float(f["dest_pressure"]),
        "dest_visibility": float(f["dest_visibility"]),
        "dest_wind_speed": float(f["dest_wind_speed"]),
        "dest_weather_known": int(f.get("dest_weather_known", 0)),
    }


def explain(carrier_code: str, origin_airport: str, destination_airport: str,
            weekday: int, month: int, scheduled_elapsed_time: float,
            origin_temp_f: float, origin_precip_in: float, origin_pressure: float,
            origin_visibility: float, origin_wind_speed: float,
            origin_temp_known: bool = True, scheduled_hour: int = 12,
            is_holiday: bool = False,
            origin_hourly_congestion: float | None = None,
            top_n: int = 5) -> dict:
    """Ranked per-flight drivers of this prediction.

    Each entry's `impact` is the probability points this condition adds versus
    the neutral baseline. Positive pushes risk up, negative pulls it down.
    """
    encoder, model = _load()
    calibrator = _load_calibrator()

    if origin_hourly_congestion is None:
        origin_hourly_congestion = _typical_congestion(str(origin_airport).upper(), int(scheduled_hour))

    base = {
        "carrier_code": carrier_code, "origin_airport": origin_airport,
        "destination_airport": destination_airport, "weekday": weekday, "month": month,
        "scheduled_elapsed_time": scheduled_elapsed_time,
        "origin_temp_f": origin_temp_f, "origin_temp_known": origin_temp_known,
        "origin_precip_in": origin_precip_in, "origin_pressure": origin_pressure,
        "origin_visibility": origin_visibility, "origin_wind_speed": origin_wind_speed,
        "scheduled_hour": scheduled_hour, "is_holiday": int(bool(is_holiday)),
        "is_weekend": 1 if int(weekday) in (5, 6) else 0,
        "origin_hourly_congestion": origin_hourly_congestion,
        "scheduled_turnaround_min": encoder.cont_mean.get("scheduled_turnaround_min", 0.0),
    }
    for c in ["dest_temp_f", "dest_precip_in", "dest_pressure", "dest_visibility", "dest_wind_speed"]:
        base[c] = encoder.cont_mean.get(c, 0.0)

    # Row 0 is the flight as entered; each subsequent row neutralises one factor.
    rows = [_build_row(base)]
    probed = []
    for key, label, fmt in FACTORS:
        if key == "origin_hourly_congestion":
            neutral_val = _typical_congestion("__none__", 12)      # overall mean
        elif key == "scheduled_elapsed_time":
            neutral_val = encoder.cont_mean.get("scheduled_elapsed_time", scheduled_elapsed_time)
        else:
            neutral_val = NEUTRAL.get(key)
        if neutral_val is None:
            continue
        variant = dict(base)
        variant[key] = neutral_val
        if key == "scheduled_hour":
            variant["origin_hourly_congestion"] = _typical_congestion(
                str(origin_airport).upper(), int(neutral_val))
        rows.append(_build_row(variant))
        probed.append((key, label, fmt, base[key], neutral_val))

    df = pd.DataFrame(rows)
    with torch.no_grad():
        logits = model(torch.tensor(encoder.transform_cat(df)),
                       torch.tensor(encoder.transform_cont(df)))
        raw = torch.sigmoid(logits).numpy()
    probs = calibrator.predict(raw) if calibrator is not None else raw

    actual = float(probs[0])
    drivers = []
    for i, (key, label, fmt, observed, neutral_val) in enumerate(probed, start=1):
        impact = actual - float(probs[i])
        # Below ~1 point is noise in a calibrated probability, not a finding.
        if abs(impact) < 0.01:
            continue
        drivers.append({
            "factor": label,
            "value": fmt(observed),
            "impact": round(impact, 4),
            "direction": "increases" if impact > 0 else "reduces",
        })

    drivers.sort(key=lambda d: -abs(d["impact"]))
    raised = [d for d in drivers if d["impact"] > 0][:top_n]
    lowered = [d for d in drivers if d["impact"] < 0][:2]

    return {
        "probability": round(actual, 4),
        "drivers": raised,
        "protective": lowered,
        "summary": _summary(raised),
    }


def _summary(raised: list[dict]) -> str:
    if not raised:
        return "No single condition stands out - the risk reflects this route and schedule generally."
    lead = raised[0]
    text = (f"Mainly {lead['factor'].lower()} ({lead['value']}), "
            f"adding {lead['impact'] * 100:.0f} points on its own")
    if len(raised) > 1:
        text += f", then {raised[1]['factor'].lower()} ({raised[1]['value']})"
    return text + "."


if __name__ == "__main__":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

    scenarios = [
        ("Blizzard, evening", dict(origin_visibility=0.2, origin_precip_in=1.5,
                                    origin_wind_speed=45, origin_temp_f=8,
                                    origin_pressure=28.7, scheduled_hour=19)),
        ("Clear, early morning", dict(origin_visibility=10, origin_precip_in=0.0,
                                       origin_wind_speed=5, origin_temp_f=70,
                                       origin_pressure=30.1, scheduled_hour=6)),
        ("Clear but late evening", dict(origin_visibility=10, origin_precip_in=0.0,
                                         origin_wind_speed=6, origin_temp_f=72,
                                         origin_pressure=30.0, scheduled_hour=21)),
    ]
    for name, kw in scenarios:
        r = explain(carrier_code="AA", origin_airport="ORD", destination_airport="DEN",
                    weekday=4, month=1, scheduled_elapsed_time=160, **kw)
        print(f"\n{name}: risk {r['probability']:.1%}")
        print(f"  {r['summary']}")
        for d in r["drivers"]:
            print(f"    +{d['impact'] * 100:5.1f} pts  {d['factor']:<20s} {d['value']}")
        for d in r["protective"]:
            print(f"    {d['impact'] * 100:6.1f} pts  {d['factor']:<20s} {d['value']}")
