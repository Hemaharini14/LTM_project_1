"""
Preprocessing for the "Historical Flight Delay and Weather Data USA"
monthly CSVs (05-2019.csv ... 12-2019.csv).

Unlike Airlines.csv, each row here already has real weather observations
captured at BOTH the origin and destination airport at flight time
(temperature, precipitation, pressure, visibility, wind), plus real delay
causes and real calendar dates. This makes it the stronger candidate for
the core disruption prediction model going forward.

Destination-side weather (dest_*) is KEPT, carried alongside a
dest_weather_known flag. The India data merged in alongside this has no
destination weather at all, so rather than dropping the column for everyone
(which is what used to happen, discarding a 99.8%-populated real signal),
India rows carry the flag at 0 - the same approach origin_temp_known already
uses for their missing temperature. Nothing is fabricated either way.

Also derives aircraft-rotation features from tail_number - see
_add_rotation_features for why "late arriving aircraft", the single largest
real cause of delay minutes, was previously invisible to the model.
"""
import pandas as pd
import numpy as np
from config import MONTHLY_FLIGHT_WEATHER_PATHS, MONTHLY_FLIGHT_WEATHER_CLEAN_PATH
from feature_engineering import add_calendar_features, add_origin_congestion

ORIGIN_WEATHER_RENAME = {
    "STATION_x": "origin_station",
    "HourlyDryBulbTemperature_x": "origin_temp_f",
    "HourlyPrecipitation_x": "origin_precip_in",
    "HourlyStationPressure_x": "origin_pressure",
    "HourlyVisibility_x": "origin_visibility",
    "HourlyWindSpeed_x": "origin_wind_speed",
}
def _add_rotation_features(df):
    """Aircraft-rotation features, derived from tail_number.

    "Late arriving aircraft" is the single largest cause of delay minutes in this
    data (~40%), and the model had no visibility into it at all. Measured on one
    month: when the same aircraft's previous leg arrived 45-90 min late, this
    flight was delayed 83.7% of the time, against 9.2% when it arrived early.

    Three features, and the distinction between them matters at serving time:

      leg_of_day, scheduled_turnaround_min
          Pure schedule structure, so they're knowable the moment a ticket goes
          on sale. Both carry real signal on their own - first leg of the day
          runs 10.3% late, seventh 28.7%; under-30-min turnarounds 27.4% against
          16.9% at 60-90 min.

      prev_leg_arrival_delay (+ prev_leg_known)
          By far the strongest, but only observable once that earlier leg has
          actually landed. A traveller checking next week cannot know it, so it
          ships with a known-flag and a neutral 0 when unavailable - the same
          approach origin_temp_known already uses for India's missing
          temperature. The flag is what stops the model reading "unknown" as
          "the inbound was on time".

    Computed strictly causally: legs are ordered by scheduled departure within
    (tail_number, date) and only EARLIER legs are ever read, so no flight sees
    its own outcome or a later one.
    """
    if "tail_number" not in df.columns:
        df["leg_of_day"] = 0
        df["scheduled_turnaround_min"] = 0.0
        df["prev_leg_arrival_delay"] = 0.0
        df["prev_leg_known"] = 0
        return df

    df = df.sort_values(["tail_number", "scheduled_departure_dt"]).copy()
    has_tail = df["tail_number"].notna()
    grouped = df[has_tail].groupby(["tail_number", "date"], sort=False)

    df.loc[has_tail, "leg_of_day"] = grouped.cumcount()
    prev_arr_delay = grouped["arrival_delay"].shift(1)
    prev_sched_arr = grouped["scheduled_arrival_dt"].shift(1)

    turnaround = (pd.to_datetime(df.loc[has_tail, "scheduled_departure_dt"], errors="coerce")
                  - pd.to_datetime(prev_sched_arr, errors="coerce")).dt.total_seconds() / 60
    # Negative or absurd gaps mean the pairing is wrong (overnight, data error),
    # so treat those as "no usable prior leg" rather than feeding noise in.
    turnaround = turnaround.where((turnaround >= 0) & (turnaround <= 1440))

    df.loc[has_tail, "scheduled_turnaround_min"] = turnaround
    df.loc[has_tail, "prev_leg_arrival_delay"] = prev_arr_delay

    df["prev_leg_known"] = df["prev_leg_arrival_delay"].notna().astype(int)
    df["prev_leg_arrival_delay"] = df["prev_leg_arrival_delay"].fillna(0.0)
    df["leg_of_day"] = df["leg_of_day"].fillna(0).astype(int)
    # No prior leg: the aircraft started its day here, so there is no turnaround
    # to be tight. Median of real turnarounds is the neutral stand-in.
    median_turn = df["scheduled_turnaround_min"].median()
    df["scheduled_turnaround_min"] = df["scheduled_turnaround_min"].fillna(median_turn)

    known = int(df["prev_leg_known"].sum())
    print(f"[clean_monthly_flights] Rotation features: {known:,} of {len(df):,} flights "
          f"({known/len(df):.1%}) have a real prior leg that day.")
    return df


DEST_WEATHER_RENAME = {
    "STATION_y": "dest_station",
    "HourlyDryBulbTemperature_y": "dest_temp_f",
    "HourlyPrecipitation_y": "dest_precip_in",
    "HourlyStationPressure_y": "dest_pressure",
    "HourlyVisibility_y": "dest_visibility",
    "HourlyWindSpeed_y": "dest_wind_speed",
}
NUMERIC_WEATHER_COLS = [v for k, v in {**ORIGIN_WEATHER_RENAME, **DEST_WEATHER_RENAME}.items()
                        if "station" not in v]

DATETIME_COLS = ["scheduled_departure_dt", "scheduled_arrival_dt",
                  "actual_departure_dt", "actual_arrival_dt"]

DELAY_CAUSE_COLS = ["delay_carrier", "delay_weather", "delay_national_aviation_system",
                     "delay_security", "delay_late_aircarft_arrival"]

DELAY_THRESHOLD_MINUTES = 15


def load_all_months() -> pd.DataFrame:
    frames = []
    for path in MONTHLY_FLIGHT_WEATHER_PATHS:
        try:
            df = pd.read_csv(path, low_memory=False)
            print(f"  Loaded {path.split(chr(92))[-1].split('/')[-1]} -> {len(df):,} rows")
            frames.append(df)
        except FileNotFoundError:
            print(f"  WARNING: {path} not found — skipping this month.")
    if not frames:
        raise FileNotFoundError("No monthly flight-weather files were found. "
                                 "Check MONTHLY_FLIGHT_WEATHER_PATHS in config.py.")
    combined = pd.concat(frames, ignore_index=True)
    return combined


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    before = len(df)
    df = df.drop_duplicates()
    removed_dupes = before - len(df)

    df = df.rename(columns={**ORIGIN_WEATHER_RENAME, **DEST_WEATHER_RENAME})

    for col in DATETIME_COLS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    for col in NUMERIC_WEATHER_COLS + ["departure_delay", "arrival_delay", "scheduled_elapsed_time",
                                       "distance_miles"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Computed before the cancellation filter below so congestion reflects real
    # total scheduled traffic (cancelled flights still occupied a scheduled slot).
    df = add_calendar_features(df)
    df = add_origin_congestion(df)

    before_cancel_split = len(df)
    if "cancelled_code" in df.columns:
        raw_col = df["cancelled_code"]
        numeric_col = pd.to_numeric(raw_col, errors="coerce")
        pct_numeric = numeric_col.notna().mean()
        print(f"[clean_monthly_flights] cancelled_code sample values: "
              f"{raw_col.dropna().unique()[:8].tolist()}")
        print(f"[clean_monthly_flights] cancelled_code is {pct_numeric:.0%} numeric-parseable.")

        if pct_numeric > 0.9:
            df["is_cancelled"] = numeric_col.fillna(0) != 0
        else:
            not_cancelled_tokens = {"", "0", "nan", "none", "n", "false", "0.0"}
            normalized = raw_col.astype(str).str.strip().str.lower()
            df["is_cancelled"] = raw_col.notna() & (~normalized.isin(not_cancelled_tokens))
    else:
        df["is_cancelled"] = False
    n_cancelled = int(df["is_cancelled"].sum())
    print(f"[clean_monthly_flights] Detected {n_cancelled:,} cancelled flights "
          f"({n_cancelled / max(before_cancel_split,1):.2%} of rows).")
    df_flown = df[~df["is_cancelled"]].copy()

    before_delay_drop = len(df_flown)
    df_flown = df_flown.dropna(subset=["departure_delay"])
    removed_no_delay_value = before_delay_drop - len(df_flown)

    df_flown["is_delayed"] = (df_flown["departure_delay"] > DELAY_THRESHOLD_MINUTES).astype(int)

    for col in NUMERIC_WEATHER_COLS + ["distance_miles"]:
        if col in df_flown.columns:
            missing = df_flown[col].isna().sum()
            if missing > 0:
                df_flown[col] = df_flown[col].fillna(df_flown[col].median())

    df_flown = _add_rotation_features(df_flown)

    df_flown["route"] = df_flown["origin_airport"].astype(str) + "-" + df_flown["destination_airport"].astype(str)
    if "delay_weather" in df_flown.columns:
        df_flown["weather_caused_delay"] = (pd.to_numeric(df_flown["delay_weather"], errors="coerce").fillna(0) > 0).astype(int)

    # Real temperature is always present for this source (unlike the India data
    # merged in later) - flag so the model can weight known vs unknown temperature.
    df_flown["origin_temp_known"] = 1
    # Destination weather was previously computed and then dropped, because the
    # India source has none. It's 99.8% populated here and arrival-airport
    # conditions genuinely matter, so keep it with a known-flag - exactly the
    # pattern origin_temp_known already uses - and let India rows carry 0.
    df_flown["dest_weather_known"] = 1
    df_flown = df_flown.drop(columns=[c for c in ["dest_station"] if c in df_flown.columns])

    df_flown = df_flown.reset_index(drop=True)

    print(f"[clean_monthly_flights] Removed {removed_dupes:,} exact duplicate rows.")
    print(f"[clean_monthly_flights] {n_cancelled:,} cancelled flights set aside (is_cancelled=True), not used for delay labeling.")
    print(f"[clean_monthly_flights] Removed {removed_no_delay_value:,} rows with no usable departure_delay value.")
    return df_flown


def run() -> pd.DataFrame:
    print("=" * 60)
    print("MONTHLY FLIGHT + WEATHER DATA (2019) — PREPROCESSING")
    print("=" * 60)
    df = load_all_months()
    print(f"\nCombined raw shape: {df.shape[0]:,} rows x {df.shape[1]} cols")
    print(f"Columns with nulls (top 10):\n{df.isnull().sum().sort_values(ascending=False).head(10)}")

    df_clean = clean(df)
    df_clean.to_csv(MONTHLY_FLIGHT_WEATHER_CLEAN_PATH, index=False)
    print(f"\nClean shape: {df_clean.shape[0]:,} rows x {df_clean.shape[1]} cols")
    print(f"Saved -> {MONTHLY_FLIGHT_WEATHER_CLEAN_PATH}")

    print("\nOverall delay rate (>15 min):", round(df_clean["is_delayed"].mean(), 3))
    if "month" in df_clean.columns:
        print("\nDelay rate by month:")
        print(df_clean.groupby("month")["is_delayed"].mean())
    if "weather_caused_delay" in df_clean.columns:
        print("\nShare of delays attributed to weather:", round(df_clean["weather_caused_delay"].mean(), 3))
    return df_clean


if __name__ == "__main__":
    run()