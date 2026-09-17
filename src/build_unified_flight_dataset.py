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
import pandas as pd
from config import (
    MONTHLY_FLIGHT_WEATHER_CLEAN_PATH, INDIA_FLIGHT_WEATHER_CLEAN_PATH,
    UNIFIED_FLIGHT_WEATHER_CLEAN_PATH,
)

SHARED_COLS = [
    "carrier_code", "origin_airport", "destination_airport", "weekday", "month",
    "scheduled_departure_dt", "scheduled_arrival_dt", "scheduled_elapsed_time",
    "origin_temp_f", "origin_temp_known", "origin_precip_in", "origin_pressure",
    "origin_visibility", "origin_wind_speed", "is_delayed", "route", "flight_number",
]


def run() -> pd.DataFrame:
    print("=" * 60)
    print("UNIFIED US + INDIA FLIGHT+WEATHER DATASET")
    print("=" * 60)

    us_df = pd.read_csv(MONTHLY_FLIGHT_WEATHER_CLEAN_PATH, low_memory=False)
    if "flight_number" not in us_df.columns:
        us_df["flight_number"] = ""
    india_df = pd.read_csv(INDIA_FLIGHT_WEATHER_CLEAN_PATH, low_memory=False)
    print(f"US rows: {len(us_df):,} | India rows: {len(india_df):,}")

    combined = pd.concat([us_df[SHARED_COLS], india_df[SHARED_COLS]], ignore_index=True)

    known_temp_mean = combined.loc[combined["origin_temp_known"] == 1, "origin_temp_f"].mean()
    n_unknown = int((combined["origin_temp_known"] == 0).sum())
    combined["origin_temp_f"] = combined["origin_temp_f"].fillna(known_temp_mean)
    print(f"Filled {n_unknown:,} rows with unknown origin temperature "
          f"(origin_temp_known=0) using the known-rows mean ({known_temp_mean:.1f}F).")

    combined.to_csv(UNIFIED_FLIGHT_WEATHER_CLEAN_PATH, index=False)
    print(f"\nUnified shape: {combined.shape[0]:,} rows x {combined.shape[1]} cols")
    print(f"Saved -> {UNIFIED_FLIGHT_WEATHER_CLEAN_PATH}")
    print("\nOverall delay rate (>15 min):", round(combined["is_delayed"].mean(), 3))
    print("Routes covered:", combined["route"].nunique())
    return combined


if __name__ == "__main__":
    run()
