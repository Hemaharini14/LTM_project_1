"""
Preprocessing for GlobalWeatherRepository.csv.

This dataset has last_updated_epoch/last_updated fields suggesting it's a
rolling CURRENT-conditions snapshot (one row per location, refreshed daily),
not historical data. That means it likely does NOT line up by date with the
2019 flight data, so it is NOT merged into the historical disruption model.

Instead, this is cleaned into a lookup table keyed by location_name, meant
for the recovery agent to check *live* destination weather when planning a
trip in real time (e.g. "is it currently raining at the destination,
should the agent weight comfort-priority alternatives more heavily").

Run the date-range check below first to confirm this assumption:
    py -c "import pandas as pd; df=pd.read_csv('dataset/GlobalWeatherRepository.csv'); print(df['last_updated'].min(), df['last_updated'].max())"
If it turns out to span many years/dates (not just recent days), tell me
and I'll rewrite this to join historically instead.
"""
import pandas as pd
from config import WEATHER_RAW_PATH, WEATHER_CLEAN_PATH

# Keep only the columns actually useful for trip-recovery decisions;
# air quality / astronomy fields (moon phase, sunrise, etc.) aren't relevant
# to flight/hotel disruption planning and are dropped to keep the file small.
KEEP_COLS = [
    "country", "location_name", "latitude", "longitude", "timezone",
    "last_updated", "temperature_celsius", "condition_text",
    "wind_kph", "wind_direction", "precip_mm", "humidity",
    "visibility_km", "uv_index",
]


def load_raw() -> pd.DataFrame:
    return pd.read_csv(WEATHER_RAW_PATH)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    before = len(df)
    df = df.drop_duplicates()
    removed_dupes = before - len(df)

    available_cols = [c for c in KEEP_COLS if c in df.columns]
    missing_cols = [c for c in KEEP_COLS if c not in df.columns]
    if missing_cols:
        print(f"[clean_weather] NOTE: expected columns not found, skipping: {missing_cols}")
    df = df[available_cols].copy()

    df["last_updated"] = pd.to_datetime(df["last_updated"], errors="coerce")

    # Keep only the most recent reading per location (this is a snapshot
    # dataset, so multiple rows per location likely represent refresh history)
    df = df.sort_values("last_updated").drop_duplicates(subset=["location_name"], keep="last")

    df = df.reset_index(drop=True)
    print(f"[clean_weather] Removed {removed_dupes:,} exact duplicate rows.")
    print(f"[clean_weather] Reduced to latest reading per location: {len(df):,} unique locations.")
    return df


def run() -> pd.DataFrame:
    print("=" * 60)
    print("GLOBALWEATHERREPOSITORY.CSV — PREPROCESSING")
    print("=" * 60)
    df = load_raw()
    print(f"Raw shape: {df.shape[0]:,} rows x {df.shape[1]} cols")

    if "last_updated" in df.columns:
        parsed = pd.to_datetime(df["last_updated"], errors="coerce")
        print(f"Date range: {parsed.min()} to {parsed.max()}")
        span_days = (parsed.max() - parsed.min()).days if parsed.notna().any() else None
        if span_days is not None and span_days < 90:
            print("-> Confirms this looks like a CURRENT-conditions snapshot, not historical "
                  "data matching the 2019 flights. Treating it as a live lookup source only.")
        elif span_days is not None:
            print(f"-> Spans {span_days} days — wider than expected for a pure snapshot. "
                  f"If this actually covers 2019, tell me and I'll rework this into a "
                  f"historical join instead of a live lookup.")

    df_clean = clean(df)
    df_clean.to_csv(WEATHER_CLEAN_PATH, index=False)
    print(f"Saved -> {WEATHER_CLEAN_PATH}")
    return df_clean


if __name__ == "__main__":
    run()