"""
Data-grounded tools for the Module 2 recovery agent. Every function here
reads from your actual cleaned CSVs - nothing is invented by an LLM.

- search_alternative_flights: real historical flights on the same route,
  scored by the trained Module 1 model
- search_hotel_options: real hotel pricing tiers from cleaned_hotel_bookings.csv
- get_destination_weather: live conditions from cleaned_global_weather.csv
"""
import os
import sys
import json
import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import HOTELS_CLEAN_PATH, MONTHLY_FLIGHT_WEATHER_CLEAN_PATH, WEATHER_CLEAN_PATH, OUTPUT_DIR
from predict_delay_v2 import predict_delay_probability, risk_label

_FLIGHTS_DF = None
_HOTELS_DF = None
_WEATHER_DF = None

DATASET_INDEX_PATH = os.path.join(OUTPUT_DIR, "dataset_index.json")
_DATASET_INDEX = None


def _load_flights():
    global _FLIGHTS_DF
    if _FLIGHTS_DF is None:
        _FLIGHTS_DF = pd.read_csv(MONTHLY_FLIGHT_WEATHER_CLEAN_PATH, low_memory=False)
    return _FLIGHTS_DF


def _load_hotels():
    global _HOTELS_DF
    if _HOTELS_DF is None:
        _HOTELS_DF = pd.read_csv(HOTELS_CLEAN_PATH)
    return _HOTELS_DF


def _load_weather():
    global _WEATHER_DF
    if _WEATHER_DF is None:
        _WEATHER_DF = pd.read_csv(WEATHER_CLEAN_PATH)
    return _WEATHER_DF


def get_dataset_index() -> dict:
    """
    Set of carrier codes and airport codes actually present in the trained
    flight+weather dataset, cached to a small JSON file after the first
    (slow, full-column) scan so the web app doesn't pay that cost on every
    restart. Used to tell a user honestly when a route/airport simply isn't
    covered by this project's (2019 US domestic) data, instead of silently
    returning a meaningless prediction for it.
    """
    global _DATASET_INDEX
    if _DATASET_INDEX is not None:
        return _DATASET_INDEX

    if os.path.exists(DATASET_INDEX_PATH):
        with open(DATASET_INDEX_PATH, "r") as f:
            _DATASET_INDEX = json.load(f)
        return _DATASET_INDEX

    cols = pd.read_csv(MONTHLY_FLIGHT_WEATHER_CLEAN_PATH, low_memory=False,
                        usecols=["carrier_code", "origin_airport", "destination_airport"])
    airports = sorted(set(cols["origin_airport"].unique()) | set(cols["destination_airport"].unique()))
    carriers = sorted(cols["carrier_code"].unique())
    _DATASET_INDEX = {"carriers": carriers, "airports": airports}
    with open(DATASET_INDEX_PATH, "w") as f:
        json.dump(_DATASET_INDEX, f)
    return _DATASET_INDEX


def is_route_covered(origin_airport: str, destination_airport: str) -> bool:
    """True only if BOTH airports appear somewhere in the training data."""
    index = get_dataset_index()
    airports = set(index["airports"])
    return (origin_airport or "").upper() in airports and (destination_airport or "").upper() in airports


def search_alternative_flights(origin_airport: str, destination_airport: str,
                                exclude_carrier: str, weekday: int, priority: str = "cost",
                                top_n: int = 5) -> list[dict]:
    """
    Finds real historical flights on the same route (excluding the disrupted
    carrier), scores each with the trained Module 1 model using that row's
    OWN recorded weather (so the risk score reflects real conditions that
    were actually observed for that flight, not synthetic guesses).
    """
    df = _load_flights()
    subset = df[
        (df["origin_airport"] == origin_airport) &
        (df["destination_airport"] == destination_airport) &
        (df["carrier_code"] != exclude_carrier)
    ].copy()
    if subset.empty:
        return []

    same_day = subset[subset["weekday"] == weekday]
    pool = same_day if len(same_day) >= 3 else subset
    pool = pool.sample(n=min(len(pool), 30), random_state=42)

    options = []
    for _, row in pool.iterrows():
        prob = predict_delay_probability(
            carrier_code=row["carrier_code"], origin_airport=row["origin_airport"],
            destination_airport=row["destination_airport"], weekday=row["weekday"],
            month=row["month"], scheduled_elapsed_time=row["scheduled_elapsed_time"],
            origin_temp_f=row["origin_temp_f"], origin_precip_in=row["origin_precip_in"],
            origin_pressure=row["origin_pressure"], origin_visibility=row["origin_visibility"],
            origin_wind_speed=row["origin_wind_speed"],
            dest_temp_f=row["dest_temp_f"], dest_precip_in=row["dest_precip_in"],
            dest_pressure=row["dest_pressure"], dest_visibility=row["dest_visibility"],
            dest_wind_speed=row["dest_wind_speed"],
        )
        options.append({
            "carrier": row["carrier_code"],
            "flight_number": str(row.get("flight_number", "")),
            "route": row["route"],
            "departure_time": str(row["scheduled_departure_dt"])[11:16] if pd.notna(row.get("scheduled_departure_dt")) else None,
            "arrival_time": str(row["scheduled_arrival_dt"])[11:16] if pd.notna(row.get("scheduled_arrival_dt")) else None,
            "scheduled_elapsed_time": int(row["scheduled_elapsed_time"]),
            "delay_probability": round(prob, 3),
            "risk_label": risk_label(prob),
        })

    if priority == "time":
        options.sort(key=lambda o: (o["delay_probability"], o["scheduled_elapsed_time"]))
    elif priority == "comfort":
        options.sort(key=lambda o: o["delay_probability"])
    else:  # cost - shorter flights assumed cheaper as a proxy (no real fare data)
        options.sort(key=lambda o: (o["scheduled_elapsed_time"], o["delay_probability"]))

    return options[:top_n]


def search_hotel_options(nightly_budget: float, priority: str = "cost", top_n: int = 5) -> list[dict]:
    """Recommends hotel/room-type tiers using real ADR distributions."""
    df = _load_hotels()
    grouped = df.groupby(["hotel", "reserved_room_type"]).agg(
        median_adr=("adr", "median"),
        cancellation_rate=("is_canceled", "mean"),
        n_bookings=("adr", "count"),
    ).reset_index()
    grouped = grouped[grouped["n_bookings"] >= 30]

    if priority == "cost":
        candidates = grouped[grouped["median_adr"] <= nightly_budget * 1.1].sort_values("median_adr")
    elif priority == "comfort":
        candidates = grouped.sort_values("median_adr", ascending=False)
    else:  # time -> most reliable (lowest cancellation) segment
        candidates = grouped.sort_values("cancellation_rate")

    if candidates.empty:
        candidates = grouped.sort_values("median_adr")

    return [
        {
            "hotel_type": row["hotel"],
            "room_type": row["reserved_room_type"],
            "median_nightly_rate_usd": round(float(row["median_adr"]), 2),
            "cancellation_rate": round(float(row["cancellation_rate"]), 3),
        }
        for _, row in candidates.head(top_n).iterrows()
    ]


def get_destination_weather(location_name: str) -> dict | None:
    """Looks up the latest live conditions for a destination, if available."""
    df = _load_weather()
    match = df[df["location_name"].str.lower() == location_name.lower()]
    if match.empty:
        return None
    row = match.iloc[0]
    return {
        "location": row["location_name"],
        "temperature_celsius": round(float(row["temperature_celsius"]), 1),
        "condition": row["condition_text"],
        "wind_kph": round(float(row["wind_kph"]), 1),
        "precip_mm": round(float(row["precip_mm"]), 1),
        "visibility_km": round(float(row["visibility_km"]), 1),
        "last_updated": row["last_updated"],
    }