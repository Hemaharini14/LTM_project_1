"""
Converts outputs/cleaned_flight_weather_unified.csv (5.4M rows, ~864MB) into
an indexed SQLite file, so recovery_tools.py can query the three things it
actually needs (a route's carriers, a flight number's route, alternatives on
a route) without holding the whole file in memory.

The in-memory version needed ~2-3GB of real RAM just for this one preload,
which rules out most free hosting tiers - a web app that queries an indexed
864MB file on disk instead needs only the RAM for whatever small result set
one query returns (typically dozens to a few hundred rows), the same gain
every database exists to provide.

Streamed in chunks on both sides - read with pandas' chunksize so the
MIGRATION itself never holds all 5.4M rows in memory either, written with
executemany so SQLite's insert stays fast - so the conversion needs no more
RAM than the final app does.

Run after build_unified_flight_dataset.py (or build_bts_dataset.py) changes
the source CSV - this is a derived, regenerable file, not committed to git.
Run: python build_flight_catalog_db.py
"""
import os
import sqlite3
import sys
import time

import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import UNIFIED_FLIGHT_WEATHER_CLEAN_PATH, FLIGHT_CATALOG_DB_PATH

CHUNK_ROWS = 200_000

# Exactly the columns recovery_tools.py's three catalog functions actually
# read (get_carriers_for_route, lookup_flight_by_number, search_alternative_
# flights, including every field predict_delay_probability takes from a real
# row) - narrower than the full CSV, which also carries training-only fields
# (is_delayed itself isn't needed for serving a real alternative's risk score,
# the model recomputes it fresh from the conditions).
COLUMNS = [
    "carrier_code", "flight_number", "origin_airport", "destination_airport",
    "route", "weekday", "month", "scheduled_hour", "is_holiday",
    "scheduled_departure_dt", "scheduled_arrival_dt", "scheduled_elapsed_time",
    "origin_temp_f", "origin_temp_known", "origin_precip_in", "origin_pressure",
    "origin_visibility", "origin_wind_speed", "origin_hourly_congestion",
    "dest_temp_f", "dest_precip_in", "dest_pressure", "dest_visibility",
    "dest_wind_speed", "dest_weather_known", "distance_miles",
]


def run():
    if os.path.exists(FLIGHT_CATALOG_DB_PATH):
        os.remove(FLIGHT_CATALOG_DB_PATH)

    conn = sqlite3.connect(FLIGHT_CATALOG_DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")  # readers never block on a concurrent writer (there isn't one at runtime, but build-safe)

    print(f"Reading {UNIFIED_FLIGHT_WEATHER_CLEAN_PATH} in {CHUNK_ROWS:,}-row chunks...")
    t0 = time.time()
    total = 0
    first = True
    for chunk in pd.read_csv(UNIFIED_FLIGHT_WEATHER_CLEAN_PATH, usecols=COLUMNS,
                             chunksize=CHUNK_ROWS, low_memory=False):
        # Precomputed once here instead of per-request - lookup_flight_by_number
        # used to call this per call and it was the single slowest thing in the
        # app (~70s across 5.4M rows); now it's a plain indexed column.
        fn = chunk["flight_number"]
        chunk["flight_no_str"] = fn.astype(str).str.replace(r"\.0$", "", regex=True).str.strip()

        chunk.to_sql("flights", conn, if_exists="append", index=False)
        total += len(chunk)
        print(f"  {total:,} rows written...", end="\r", flush=True)
    print(f"\n{total:,} rows written in {time.time() - t0:.0f}s. Building indexes...")

    t0 = time.time()
    # Covers get_carriers_for_route and search_alternative_flights (both filter
    # on origin+destination first).
    conn.execute("CREATE INDEX idx_route ON flights(origin_airport, destination_airport)")
    # Covers lookup_flight_by_number.
    conn.execute("CREATE INDEX idx_flightno ON flights(carrier_code, flight_no_str)")
    conn.commit()
    print(f"Indexes built in {time.time() - t0:.0f}s.")

    size_mb = os.path.getsize(FLIGHT_CATALOG_DB_PATH) / 1_000_000
    print(f"Saved -> {FLIGHT_CATALOG_DB_PATH} ({size_mb:.0f} MB on disk, "
          f"nothing held in memory at serving time beyond one query's result)")
    conn.close()


if __name__ == "__main__":
    run()
