"""
Data-grounded tools for the Module 2 recovery agent. Every function here
reads from your actual cleaned CSVs (or the indexed SQLite copy of the big
one) - nothing is invented by an LLM.

- search_alternative_flights: real historical flights on the same route,
  scored by the trained Module 1 model
- search_hotel_options: real hotel pricing tiers from cleaned_hotel_bookings.csv
- get_destination_weather: live conditions from WeatherAPI.com (falls back to
  cleaned_global_weather.csv)
"""
import os
import sqlite3
import sys
import json
import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import HOTELS_CLEAN_PATH, UNIFIED_FLIGHT_WEATHER_CLEAN_PATH, WEATHER_CLEAN_PATH, OUTPUT_DIR, FLIGHT_CATALOG_DB_PATH
from predict_delay_v2 import predict_delay_probability, risk_label
from weatherapi_live import live_weather

_HOTELS_DF = None
_WEATHER_DF = None

DATASET_INDEX_PATH = os.path.join(OUTPUT_DIR, "dataset_index.json")
_DATASET_INDEX = None


def _catalog_conn() -> sqlite3.Connection:
    """One short-lived read-only connection per call, not a cached global -
    SQLite connections are cheap to open and this keeps every query trivially
    safe across gunicorn's multiple worker processes, unlike a single shared
    in-memory DataFrame would be. Falls back to None if the indexed file
    hasn't been built yet (see build_flight_catalog_db.py); callers treat
    that exactly like "no real data for this route" rather than crashing.
    """
    if not os.path.exists(FLIGHT_CATALOG_DB_PATH):
        return None
    conn = sqlite3.connect(f"file:{FLIGHT_CATALOG_DB_PATH}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


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
    covered by this project's trained data (2019 US domestic + real 2019/2020 India domestic
    flights - see clean_india_flights.py), instead of silently returning a meaningless prediction for it.
    """
    global _DATASET_INDEX
    if _DATASET_INDEX is not None:
        return _DATASET_INDEX

    if os.path.exists(DATASET_INDEX_PATH):
        with open(DATASET_INDEX_PATH, "r") as f:
            _DATASET_INDEX = json.load(f)
        return _DATASET_INDEX

    cols = pd.read_csv(UNIFIED_FLIGHT_WEATHER_CLEAN_PATH, low_memory=False,
                        usecols=["carrier_code", "origin_airport", "destination_airport"])
    airports = sorted(set(cols["origin_airport"].unique()) | set(cols["destination_airport"].unique()))
    carriers = sorted(cols["carrier_code"].unique())
    _DATASET_INDEX = {"carriers": carriers, "airports": airports}
    with open(DATASET_INDEX_PATH, "w") as f:
        json.dump(_DATASET_INDEX, f)
    return _DATASET_INDEX


def is_route_covered(origin_airport: str, destination_airport: str) -> bool:
    """True when some trained model can actually score this route.

    Two sources now, not one. The unified dataset index is the original: both
    airports must appear in it. But the India model is trained on its own
    historical rows plus the real outcomes collected since, which covers
    airports the unified dataset never had - Chennai among them. Before this,
    a Chennai departure fell through to the reference-schedule branch and got
    no prediction at all, even though a model that can score it exists.
    """
    o = (origin_airport or "").upper()
    d = (destination_airport or "").upper()
    airports = set(get_dataset_index()["airports"])
    if o in airports and d in airports:
        return True
    try:
        from india_delay_model import INDIA_AIRPORTS
        # india_delay_model.covers() is permissive by design (o-in-India OR
        # d-in-India OR carrier-is-Indian) for its real job: picking which
        # backend scores a flight that's already known to need the India model.
        # Reused as-is here, it wrongly marked DEL->SIN "covered" just because
        # DEL is a real India airport - any India departure to anywhere on Earth
        # passed. This function answers a different question - "does a real
        # prediction exist for this route at all" - so both ends must be real
        # India domestic airports, not just one.
        return o in INDIA_AIRPORTS and d in INDIA_AIRPORTS
    except Exception:
        return False


def get_carriers_for_route(origin_airport: str, destination_airport: str) -> list[str]:
    """Real carrier codes that actually have historical flights on this exact route -
    not just individually-covered airports (a route between two covered airports may
    still have zero real flights, e.g. never-observed pairings)."""
    conn = _catalog_conn()
    if conn is None:
        return []
    try:
        rows = conn.execute(
            "SELECT DISTINCT carrier_code FROM flights WHERE origin_airport = ? AND destination_airport = ?",
            ((origin_airport or "").upper(), (destination_airport or "").upper()),
        ).fetchall()
    finally:
        conn.close()
    return sorted(r["carrier_code"] for r in rows)


def lookup_flight_by_number(carrier_code: str, flight_number: str) -> dict | None:
    """
    Looks up a real flight's route and typical duration by carrier + flight number, from
    every 2019 occurrence of that flight in the training data. A flight number can recur many
    times (different dates) but usually flies the same route, so this returns the most common
    (origin, destination) pairing along with how many times it was actually observed - callers
    should show that occurrence count so the user knows this is a real historical lookup, not a
    live schedule. Returns None if this carrier+flight number was never seen.
    """
    wanted = str(flight_number).strip()
    if wanted.endswith(".0"):
        wanted = wanted[:-2]
    conn = _catalog_conn()
    if conn is None:
        return None
    try:
        rows = conn.execute(
            "SELECT origin_airport, destination_airport, scheduled_elapsed_time "
            "FROM flights WHERE carrier_code = ? AND flight_no_str = ?",
            ((carrier_code or "").upper(), wanted),
        ).fetchall()
    finally:
        conn.close()
    if not rows:
        return None

    subset = pd.DataFrame(rows, columns=["origin_airport", "destination_airport", "scheduled_elapsed_time"])
    grouped = subset.groupby(["origin_airport", "destination_airport"]).agg(
        scheduled_elapsed_time=("scheduled_elapsed_time", "median"),
        occurrences=("scheduled_elapsed_time", "count"),
    ).reset_index().sort_values("occurrences", ascending=False)
    top = grouped.iloc[0]
    return {
        "origin_airport": top["origin_airport"],
        "destination_airport": top["destination_airport"],
        "scheduled_elapsed_time": round(float(top["scheduled_elapsed_time"])),
        "occurrences": int(top["occurrences"]),
        "total_seen": int(len(subset)),
    }


def _aggregate_by_flight(options: list[dict]) -> list[dict]:
    """The sampled pool is historical ROWS, so the same real flight (same carrier +
    number + route) usually appears several times on different dates, each with its
    own recorded weather and therefore its own delay score. Returned as-is that shows
    the traveler three identical-looking cards with three different percentages.

    Collapse them to one entry per real flight and average the score across its real
    occurrences: keeping the single best instance instead would be cherry-picking that
    flight's luckiest day and would systematically understate the risk. occurrences is
    carried through so callers can show how much real history backs the number, same as
    lookup_flight_by_number does.
    """
    groups = {}
    for o in options:
        key = (o["carrier"], o["flight_number"], o["route"])
        groups.setdefault(key, []).append(o)

    merged = []
    for rows in groups.values():
        base = dict(rows[0])
        mean_prob = sum(r["delay_probability"] for r in rows) / len(rows)
        base["delay_probability"] = round(mean_prob, 3)
        base["risk_label"] = risk_label(mean_prob)
        base["scheduled_elapsed_time"] = int(
            sorted(r["scheduled_elapsed_time"] for r in rows)[len(rows) // 2])
        base["occurrences"] = len(rows)
        merged.append(base)
    return merged


def search_alternative_flights(origin_airport: str, destination_airport: str,
                                exclude_carrier: str, weekday: int, priority: str = "cost",
                                top_n: int = 5) -> list[dict]:
    """
    Finds real historical flights on the same route (excluding the disrupted
    carrier), scores each with the trained Module 1 model using that row's
    OWN recorded weather (so the risk score reflects real conditions that
    were actually observed for that flight, not synthetic guesses).
    """
    conn = _catalog_conn()
    if conn is None:
        return []
    try:
        subset = pd.read_sql_query(
            "SELECT * FROM flights WHERE origin_airport = ? AND destination_airport = ? "
            "AND carrier_code != ?",
            conn, params=(origin_airport, destination_airport, exclude_carrier),
        )
    finally:
        conn.close()
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
            origin_temp_f=row["origin_temp_f"], origin_temp_known=bool(row["origin_temp_known"]),
            origin_precip_in=row["origin_precip_in"],
            origin_pressure=row["origin_pressure"], origin_visibility=row["origin_visibility"],
            origin_wind_speed=row["origin_wind_speed"],
            # Real per-row values (not defaults/lookups) since we have the actual
            # historical record - see predict_delay_probability's docstring.
            scheduled_hour=int(row["scheduled_hour"]), is_holiday=bool(row["is_holiday"]),
            origin_hourly_congestion=row["origin_hourly_congestion"],
            # Same real historical record also carries arrival-airport weather -
            # None (not a guess) when this particular row never had it (e.g. an
            # India row), same meaning as origin_temp_known above.
            dest_weather=({"temp_f": row["dest_temp_f"], "precip_in": row["dest_precip_in"],
                           "pressure": row["dest_pressure"], "visibility": row["dest_visibility"],
                           "wind_speed": row["dest_wind_speed"]}
                          if bool(row.get("dest_weather_known", 0)) else None),
            # Real per-row distance from the catalogue (build_unified_flight_dataset.py
            # computes it from real airport coordinates for any row that didn't already
            # report one) - .get() stays defensive against a stale cached catalogue
            # file from before this column existed, where it falls back to
            # predict_delay_probability's own real coordinate lookup instead.
            distance_miles=row.get("distance_miles"),
        )
        fn = row.get("flight_number", "")
        options.append({
            "carrier": row["carrier_code"],
            # stored as a float in the CSV, so a bare str() renders "403.0"
            "flight_number": str(int(fn)) if pd.notna(fn) and str(fn).replace(".", "").isdigit() else str(fn),
            "route": row["route"],
            "departure_time": str(row["scheduled_departure_dt"])[11:16] if pd.notna(row.get("scheduled_departure_dt")) else None,
            "arrival_time": str(row["scheduled_arrival_dt"])[11:16] if pd.notna(row.get("scheduled_arrival_dt")) else None,
            "scheduled_elapsed_time": int(row["scheduled_elapsed_time"]),
            "delay_probability": round(prob, 3),
            "risk_label": risk_label(prob),
        })

    options = _aggregate_by_flight(options)

    if priority == "time":
        options.sort(key=lambda o: (o["delay_probability"], o["scheduled_elapsed_time"]))
    elif priority == "comfort":
        options.sort(key=lambda o: o["delay_probability"])
    else:  # cost - shorter flights assumed cheaper as a proxy (no real fare data)
        options.sort(key=lambda o: (o["scheduled_elapsed_time"], o["delay_probability"]))

    return options[:top_n]


def _hotel_tier_groups():
    """Every real hotel/room-type tier with enough bookings (>=30) to trust its median -
    shared by search_hotel_options (pre-filtered by a guessed nightly budget) and
    itinerary_optimizer.py (which searches ALL real tiers against the actual total
    budget constraint directly, rather than a pre-filtered guess)."""
    df = _load_hotels()
    grouped = df.groupby(["hotel", "reserved_room_type"]).agg(
        median_adr=("adr", "median"),
        cancellation_rate=("is_canceled", "mean"),
        n_bookings=("adr", "count"),
    ).reset_index()
    return grouped[grouped["n_bookings"] >= 30]


def _hotel_row_to_dict(row) -> dict:
    return {
        "hotel_type": row["hotel"],
        "room_type": row["reserved_room_type"],
        "median_nightly_rate_usd": round(float(row["median_adr"]), 2),
        "cancellation_rate": round(float(row["cancellation_rate"]), 3),
        # Real, dataset-grounded stand-in for a "review": the share of past bookings of
        # this tier that were NOT cancelled - a genuine reliability signal, not a fabricated
        # guest opinion.
        "reliability_pct": round((1 - float(row["cancellation_rate"])) * 100, 1),
    }


def get_all_hotel_tiers() -> list[dict]:
    """Every real hotel tier, unfiltered by any guessed budget - for itinerary_optimizer.py
    to search against the traveler's actual total budget constraint directly."""
    grouped = _hotel_tier_groups()
    return [_hotel_row_to_dict(row) for _, row in grouped.iterrows()]


def search_hotel_options(nightly_budget: float, priority: str = "cost", top_n: int = 5) -> list[dict]:
    """Recommends hotel/room-type tiers using real ADR distributions."""
    grouped = _hotel_tier_groups()

    if priority == "cost":
        candidates = grouped[grouped["median_adr"] <= nightly_budget * 1.1].sort_values("median_adr")
    elif priority == "comfort":
        candidates = grouped.sort_values("median_adr", ascending=False)
    else:  # time -> most reliable (lowest cancellation) segment
        candidates = grouped.sort_values("cancellation_rate")

    if candidates.empty:
        candidates = grouped.sort_values("median_adr")

    return [_hotel_row_to_dict(row) for _, row in candidates.head(top_n).iterrows()]


def get_destination_weather(location_name: str) -> dict | None:
    """Real current conditions for a destination - a genuinely live call
    (WeatherAPI.com) first, falling back to GlobalWeatherRepository.csv (a
    one-time downloaded snapshot that only gets staler - see clean_weather.py)
    only if the key is missing or the call fails, so this never goes from
    "a real but possibly stale reading" to nothing at all."""
    if not location_name:
        return None
    live = live_weather(location_name)
    if live:
        return live

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