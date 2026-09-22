"""
Builds the real "how long, and why" lookup that the delay model can't answer.

The trained model (delay_model_v2) is a binary classifier: it predicts the
PROBABILITY a flight departs 15+ minutes late, and nothing about how late or
what caused it. The raw monthly files carry both - departure_delay in minutes
and a full attribution across carrier / weather / national-aviation-system /
security / late-arriving-aircraft - but clean_monthly_flights.py collapses all
of that into the is_delayed boolean and drops the rest.

This script goes back to the raw files and precomputes, per group, the real
distribution of delay length and the real average attribution of those minutes,
over flights that actually were delayed. Same precomputed-real-lookup pattern
as congestion_lookup.json: no model, no guessing, just conditional statistics
over real history, so a lookup at request time is a dict hit.

Three tiers, most specific first, so a quiet route still gets a real answer
from a broader group rather than a fabricated one:
    "CARRIER|ORIGIN-DEST"  -> this airline on this route
    "ORIGIN-DEST"          -> this route, any airline
    "ORIGIN|HOUR"          -> this airport at this time of day
    "__default__"          -> every delayed flight in the data

Groups with fewer than MIN_SAMPLES real delayed flights are dropped rather than
reported, since a median over three flights isn't worth showing.

NOTE: cause attribution is US-only. The India source (Dataset.csv) records
delay minutes but no cause breakdown, so India routes get duration with
causes omitted rather than causes invented.

Run: python build_delay_duration_lookup.py
Output: outputs/delay_duration_lookup.json
"""
import json
import os
import sys

import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import MONTHLY_FLIGHT_WEATHER_PATHS, OUTPUT_DIR

LOOKUP_PATH = os.path.join(OUTPUT_DIR, "delay_duration_lookup.json")

DELAY_THRESHOLD_MIN = 15      # same definition of "delayed" the classifier trains on
MIN_SAMPLES = 30

CAUSE_COLS = {
    "delay_carrier": "carrier",
    "delay_weather": "weather",
    "delay_national_aviation_system": "air traffic / airport",
    "delay_security": "security",
    "delay_late_aircarft_arrival": "late arriving aircraft",
}

USE_COLS = ["carrier_code", "origin_airport", "destination_airport",
            "scheduled_departure_dt", "departure_delay"] + list(CAUSE_COLS)


def _summarize(group: pd.DataFrame) -> dict:
    """Real distribution of delay length, plus where those minutes actually went."""
    delays = group["departure_delay"]
    total_cause_min = sum(group[c].sum() for c in CAUSE_COLS)
    causes = {}
    if total_cause_min > 0:
        for col, label in CAUSE_COLS.items():
            share = group[col].sum() / total_cause_min
            if share >= 0.01:
                causes[label] = round(float(share), 3)
    return {
        "median_min": int(delays.median()),
        "p90_min": int(delays.quantile(0.90)),
        "sample_size": int(len(group)),
        # ordered so the caller can just take the first as "the usual reason"
        "causes": dict(sorted(causes.items(), key=lambda kv: -kv[1])),
    }


def main():
    frames = []
    for path in MONTHLY_FLIGHT_WEATHER_PATHS:
        if not os.path.exists(path):
            print(f"  skipping missing {os.path.basename(path)}")
            continue
        df = pd.read_csv(path, usecols=USE_COLS, low_memory=False)
        df = df[df["departure_delay"] >= DELAY_THRESHOLD_MIN]
        frames.append(df)
        print(f"  {os.path.basename(path)}: {len(df):,} delayed flights")

    if not frames:
        raise SystemExit("No raw monthly files found - nothing to build.")

    df = pd.concat(frames, ignore_index=True)
    for col in CAUSE_COLS:
        df[col] = df[col].fillna(0.0)
    df["route"] = df["origin_airport"] + "-" + df["destination_airport"]
    df["hour"] = pd.to_datetime(df["scheduled_departure_dt"], errors="coerce").dt.hour
    print(f"\nTotal real delayed flights: {len(df):,}")

    lookup = {}

    for (carrier, route), g in df.groupby(["carrier_code", "route"]):
        if len(g) >= MIN_SAMPLES:
            lookup[f"{carrier}|{route}"] = _summarize(g)
    print(f"  carrier+route groups: {len(lookup):,}")

    before = len(lookup)
    for route, g in df.groupby("route"):
        if len(g) >= MIN_SAMPLES:
            lookup[route] = _summarize(g)
    print(f"  route groups:         {len(lookup) - before:,}")

    before = len(lookup)
    for (origin, hour), g in df.groupby(["origin_airport", "hour"]):
        if len(g) >= MIN_SAMPLES:
            lookup[f"{origin}|{int(hour)}"] = _summarize(g)
    print(f"  airport+hour groups:  {len(lookup) - before:,}")

    lookup["__default__"] = _summarize(df)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(LOOKUP_PATH, "w") as f:
        json.dump(lookup, f)
    size_mb = os.path.getsize(LOOKUP_PATH) / 1024 / 1024
    print(f"\nSaved {len(lookup):,} groups to {LOOKUP_PATH} ({size_mb:.1f} MB)")
    d = lookup["__default__"]
    print(f"Overall: median {d['median_min']} min, p90 {d['p90_min']} min, "
          f"top cause {next(iter(d['causes']), 'n/a')}")


if __name__ == "__main__":
    main()
