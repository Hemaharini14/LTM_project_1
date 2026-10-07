"""
Combines the cleaned US (clean_monthly_flights.py) and India
(clean_india_flights.py) flight+weather tables into one unified training
set (config.UNIFIED_FLIGHT_WEATHER_CLEAN_PATH), on the shared origin-weather
schema both sources actually have real values for.

India rows have no real temperature reading at all (origin_temp_known=0).
Rather than fabricate a plausible-looking value, missing temperature is
filled with the *known* (US) rows' mean origin_temp_f - after the
encoder's zero-mean/unit-std scaling this becomes exactly 0, i.e. "no
signal", and origin_temp_known is fed alongside it so the model can learn
to actually rely on that flag instead of a fake reading.
"""
import json
import pandas as pd
from config import (
    MONTHLY_FLIGHT_WEATHER_CLEAN_PATH, INDIA_FLIGHT_WEATHER_CLEAN_PATH,
    UNIFIED_FLIGHT_WEATHER_CLEAN_PATH, CONGESTION_LOOKUP_PATH,
)
from reference_data import distance_miles_between

SHARED_COLS = [
    "carrier_code", "origin_airport", "destination_airport", "weekday", "month",
    "scheduled_departure_dt", "scheduled_arrival_dt", "scheduled_elapsed_time",
    "origin_temp_f", "origin_temp_known", "origin_precip_in", "origin_pressure",
    "origin_visibility", "origin_wind_speed", "is_delayed", "route", "flight_number",
    "scheduled_hour", "is_weekend", "is_holiday", "origin_hourly_congestion",
    # Aircraft rotation (US only - India has no tail numbers), see clean_monthly_flights
    "leg_of_day", "scheduled_turnaround_min", "prev_leg_arrival_delay", "prev_leg_known",
    # Destination weather (US only), carried with its own known-flag
    "dest_temp_f", "dest_precip_in", "dest_pressure", "dest_visibility", "dest_wind_speed",
    "dest_weather_known",
    # Real reported route distance - both sources have it (BTS in miles already,
    # India converted from km in clean_india_flights.py).
    "distance_miles",
]

# Numeric columns only one source has, OR that can occasionally be missing even
# where both sources normally have it (a handful of unmapped BTS/India rows).
# Filled with the KNOWN rows' mean so that after the encoder's zero-mean scaling
# they land at exactly 0 ("no signal"); distance_miles has no *_known flag of its
# own, so a filled value there is indistinguishable from "no signal" by design -
# same reasoning as scheduled_turnaround_min, which has never had one either.
PARTIAL_COLS = ["scheduled_turnaround_min", "dest_temp_f", "dest_precip_in",
                "dest_pressure", "dest_visibility", "dest_wind_speed", "distance_miles"]


def run(us_path: str = MONTHLY_FLIGHT_WEATHER_CLEAN_PATH,
        out_path: str = UNIFIED_FLIGHT_WEATHER_CLEAN_PATH,
        congestion_path: str = CONGESTION_LOOKUP_PATH) -> pd.DataFrame:
    print("=" * 60)
    print("UNIFIED US + INDIA FLIGHT+WEATHER DATASET")
    print("=" * 60)

    us_df = pd.read_csv(us_path, low_memory=False)
    if "flight_number" not in us_df.columns:
        us_df["flight_number"] = ""
    india_df = pd.read_csv(INDIA_FLIGHT_WEATHER_CLEAN_PATH, low_memory=False)
    print(f"US rows: {len(us_df):,} | India rows: {len(india_df):,}")

    # A column can be entirely absent from one source's FILE (not just NaN on
    # some rows) - e.g. the original 2019 US cleaning predates distance_miles.
    # Add it as all-missing rather than letting `df[SHARED_COLS]` KeyError, so
    # PARTIAL_COLS' fillna(mean) below still has a real column to work with.
    for df, label in ((us_df, "us_path"), (india_df, "INDIA_FLIGHT_WEATHER_CLEAN_PATH")):
        missing = [c for c in SHARED_COLS if c not in df.columns]
        if missing:
            print(f"[build_unified_flight_dataset] {label} has no {missing} - "
                  f"filling as missing (not a guess; PARTIAL_COLS means it one way or another).")
            for c in missing:
                df[c] = pd.NA

    combined = pd.concat([us_df[SHARED_COLS], india_df[SHARED_COLS]], ignore_index=True)

    known_temp_mean = combined.loc[combined["origin_temp_known"] == 1, "origin_temp_f"].mean()
    n_unknown = int((combined["origin_temp_known"] == 0).sum())
    combined["origin_temp_f"] = combined["origin_temp_f"].fillna(known_temp_mean)

    # distance_miles specifically: when an entire SOURCE lacks the column (the
    # original 2019 US cleaning predates it), PARTIAL_COLS' fillna(mean) below
    # would stamp millions of rows with one unrelated constant (India's mean
    # route length) regardless of the row's real route - verified directly:
    # that produced the exact same "800.16mi" on a FLL-LCK row and a LAX-JFK
    # row alike. A real distance is computable instead, from the same real
    # airport coordinates serving time already uses (reference_data.distance_
    # miles_between) - so every row gets its own real route length, and only
    # a genuine coordinate gap (rare) falls through to the mean fallback below.
    missing_distance = combined["distance_miles"].isna()
    if missing_distance.any():
        n_before = int(missing_distance.sum())
        computed = combined.loc[missing_distance].apply(
            lambda r: distance_miles_between(r["origin_airport"], r["destination_airport"]), axis=1)
        combined.loc[missing_distance, "distance_miles"] = computed
        n_resolved = int(computed.notna().sum())
        print(f"[build_unified_flight_dataset] Computed real great-circle distance for "
              f"{n_resolved:,} of {n_before:,} rows with no reported distance "
              f"(real airport coordinates) - {n_before - n_resolved:,} fall back to the mean "
              f"below, airport(s) with no published coordinates.")

    for col in PARTIAL_COLS:
        combined[col] = pd.to_numeric(combined[col], errors="coerce")
        combined[col] = combined[col].fillna(combined[col].mean())
    print(f"Filled {n_unknown:,} rows with unknown origin temperature "
          f"(origin_temp_known=0) using the known-rows mean ({known_temp_mean:.1f}F).")

    combined.to_csv(out_path, index=False)
    print(f"\nUnified shape: {combined.shape[0]:,} rows x {combined.shape[1]} cols")
    print(f"Saved -> {out_path}")
    print("\nOverall delay rate (>15 min):", round(combined["is_delayed"].mean(), 3))
    print("Routes covered:", combined["route"].nunique())

    # Real historical average congestion per airport+hour, for predict_delay_v2.py
    # to look up when an ad-hoc caller has no actual historical row to read it
    # from directly - see _typical_congestion() there.
    congestion_means = combined.groupby(["origin_airport", "scheduled_hour"])["origin_hourly_congestion"].mean()
    congestion_lookup = {f"{airport}|{hour}": round(float(v), 2) for (airport, hour), v in congestion_means.items()}
    congestion_lookup["__default__"] = round(float(combined["origin_hourly_congestion"].mean()), 2)
    with open(congestion_path, "w") as f:
        json.dump(congestion_lookup, f)
    print(f"Saved congestion lookup ({len(congestion_lookup):,} airport+hour entries) -> {congestion_path}")
    return combined


if __name__ == "__main__":
    run()
