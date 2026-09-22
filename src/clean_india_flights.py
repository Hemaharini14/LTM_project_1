"""
Preprocessing for dataset/Dataset.csv - real 2019/early-2020 India domestic
flights (BLR/BOM/CCU/DEL/HYD) merged with a single hourly weather snapshot
per flight (World Weather Online-style fields: windspeedKmph, precipMM,
pressure in hPa, visibility in km, humidity, cloudcover).

Two structural gaps vs. the US monthly flight+weather data
(clean_monthly_flights.py), both handled by omission rather than by
inventing a number:
  - No destination-side weather at all (only one snapshot per flight,
    treated as the origin's conditions since "From" is the recorded route
    origin). -> dest_* columns simply don't exist for these rows; the
    unified schema is origin-weather-only so nothing needs to be faked.
  - No temperature field of any kind. -> origin_temp_f is left NaN and
    origin_temp_known=0 for every India row, so training can impute/mask it
    instead of pretending a value was observed.
"""
import pandas as pd
from config import INDIA_FLIGHT_WEATHER_RAW_PATH, INDIA_FLIGHT_WEATHER_CLEAN_PATH
from feature_engineering import add_calendar_features, add_origin_congestion

# Real IATA carrier codes for the airline names as they appear in this file.
AIRLINE_TO_CARRIER_CODE = {
    "air asia": "AK",
    "indigo": "6E",
    "air india": "AI",
    "spicejet": "SG",
    "vistara": "UK",
    "go air": "G8",
}

KMPH_TO_MPH = 0.621371
KM_TO_MILES = 0.621371
MM_TO_INCHES = 1 / 25.4
HPA_TO_INHG = 0.0295299830714

DELAY_THRESHOLD_MINUTES = 15


def load_raw() -> pd.DataFrame:
    return pd.read_csv(INDIA_FLIGHT_WEATHER_RAW_PATH)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    before = len(df)
    df = df.drop_duplicates()
    removed_dupes = before - len(df)

    date = pd.to_datetime(df["Used Date"], format="%d-%m-%Y")
    departure_dt = date + pd.to_timedelta(df["Scheduled Departure"] + ":00")
    arrival_dt = date + pd.to_timedelta(df["Scheduled Arrival"] + ":00")
    # Arrival clock-time earlier than departure clock-time means the flight
    # lands after midnight (overnight service) - real schedules do this,
    # it isn't a data error.
    arrival_dt = arrival_dt.where(arrival_dt >= departure_dt, arrival_dt + pd.Timedelta(days=1))

    carrier_code = df["Airline"].str.strip().str.lower().map(AIRLINE_TO_CARRIER_CODE)
    unmapped = df["Airline"][carrier_code.isna()].unique().tolist()
    if unmapped:
        print(f"[clean_india_flights] WARNING: no carrier code mapping for airline(s): {unmapped} "
              f"- these rows will be dropped.")

    out = pd.DataFrame({
        "carrier_code": carrier_code,
        "origin_airport": df["From"].str.upper(),
        "destination_airport": df["To"].str.upper(),
        "weekday": date.dt.dayofweek,
        "month": date.dt.month,
        "scheduled_departure_dt": departure_dt,
        "scheduled_arrival_dt": arrival_dt,
        "scheduled_elapsed_time": (arrival_dt - departure_dt).dt.total_seconds() / 60,
        "departure_delay": pd.to_numeric(df["Departure Delay"], errors="coerce"),
        "is_cancelled": df["Status"] == 0,
        "origin_temp_f": pd.NA,
        "origin_temp_known": 0,
        "origin_precip_in": pd.to_numeric(df["weather__hourly__precipMM"], errors="coerce") * MM_TO_INCHES,
        "origin_pressure": pd.to_numeric(df["weather__hourly__pressure"], errors="coerce") * HPA_TO_INHG,
        "origin_visibility": pd.to_numeric(df["weather__hourly__visibility"], errors="coerce") * KM_TO_MILES,
        "origin_wind_speed": pd.to_numeric(df["weather__hourly__windspeedKmph"], errors="coerce") * KMPH_TO_MPH,
    })
    out = out.dropna(subset=["carrier_code"])

    # Computed before the cancellation filter below so congestion reflects real
    # total scheduled traffic (cancelled flights still occupied a scheduled slot).
    out = add_calendar_features(out)
    out = add_origin_congestion(out)

    n_cancelled = int(out["is_cancelled"].sum())
    print(f"[clean_india_flights] Detected {n_cancelled:,} cancelled flights "
          f"({n_cancelled / max(len(out), 1):.2%} of rows), set aside from delay labeling.")
    out = out[~out["is_cancelled"]].copy()

    before_delay_drop = len(out)
    out = out.dropna(subset=["departure_delay"])
    removed_no_delay_value = before_delay_drop - len(out)

    out["is_delayed"] = (out["departure_delay"] > DELAY_THRESHOLD_MINUTES).astype(int)

    # This source has no tail numbers and no destination weather, so the rotation
    # and dest-weather features genuinely cannot be computed. Flag them unknown
    # rather than inventing values - build_unified_flight_dataset fills the
    # numeric side with the known-rows mean, which scales to a neutral 0.
    out["leg_of_day"] = 0
    out["scheduled_turnaround_min"] = pd.NA
    out["prev_leg_arrival_delay"] = 0.0
    out["prev_leg_known"] = 0
    out["dest_weather_known"] = 0
    for c in ["dest_temp_f", "dest_precip_in", "dest_pressure", "dest_visibility", "dest_wind_speed"]:
        out[c] = pd.NA
    out["route"] = out["origin_airport"] + "-" + out["destination_airport"]
    out["flight_number"] = ""

    out = out.reset_index(drop=True)
    print(f"[clean_india_flights] Removed {removed_dupes:,} exact duplicate rows.")
    print(f"[clean_india_flights] Removed {removed_no_delay_value:,} rows with no usable departure_delay value.")
    return out


def run() -> pd.DataFrame:
    print("=" * 60)
    print("DATASET.CSV (INDIA DOMESTIC FLIGHTS) — PREPROCESSING")
    print("=" * 60)
    df = load_raw()
    print(f"Raw shape: {df.shape[0]:,} rows x {df.shape[1]} cols")

    df_clean = clean(df)
    df_clean.to_csv(INDIA_FLIGHT_WEATHER_CLEAN_PATH, index=False)
    print(f"\nClean shape: {df_clean.shape[0]:,} rows x {df_clean.shape[1]} cols")
    print(f"Saved -> {INDIA_FLIGHT_WEATHER_CLEAN_PATH}")
    print("\nOverall delay rate (>15 min):", round(df_clean["is_delayed"].mean(), 3))
    print("\nRoutes:", sorted(df_clean["route"].unique().tolist()))
    return df_clean


if __name__ == "__main__":
    run()
