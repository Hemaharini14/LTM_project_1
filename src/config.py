"""
Central config: paths and constants shared across the preprocessing scripts.
Adjust DATA_DIR below if your dataset/ folder lives somewhere else relative
to where you run main.py from.
"""
import os

# Assumes this file lives in a subfolder (e.g. src/) directly under the
# project root that also contains dataset/. Adjust if your layout differs.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(PROJECT_ROOT, "dataset")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")

AIRLINES_RAW_PATH = os.path.join(DATA_DIR, "Airlines.csv")
AIRLINES_CLEAN_PATH = os.path.join(OUTPUT_DIR, "cleaned_airlines.csv")

HOTELS_RAW_PATH = os.path.join(DATA_DIR, "hotel_bookings.csv")
HOTELS_CLEAN_PATH = os.path.join(OUTPUT_DIR, "cleaned_hotel_bookings.csv")

# "Historical Flight Delay and Weather Data USA" — monthly files.
# Each row is a real flight with real weather observations (temperature,
# precipitation, pressure, visibility, wind) already captured at BOTH the
# origin and destination airport at flight time, plus real delay-cause
# breakdowns and real dates. This supersedes Airlines.csv as the primary
# source for the disruption prediction model.
MONTHLY_FLIGHT_WEATHER_FILES = [
    "05-2019.csv", "06-2019.csv", "07-2019.csv", "08-2019.csv",
    "09-2019.csv", "10-2019.csv", "11-2019.csv", "12-2019.csv",
]
MONTHLY_FLIGHT_WEATHER_PATHS = [os.path.join(DATA_DIR, f) for f in MONTHLY_FLIGHT_WEATHER_FILES]
MONTHLY_FLIGHT_WEATHER_CLEAN_PATH = os.path.join(OUTPUT_DIR, "cleaned_flight_weather_2019.csv")

# Real 2019/early-2020 India domestic flights (BLR/BOM/CCU/DEL/HYD) with a
# single origin-side weather snapshot per flight (windspeed/precip/pressure/
# visibility/cloudcover, no temperature, no destination-side weather) — see
# clean_india_flights.py. Merged with the US data above (origin-weather
# columns only, since that's the largest common feature set both sources
# actually have real values for) into one unified training set.
INDIA_FLIGHT_WEATHER_RAW_PATH = os.path.join(DATA_DIR, "Dataset.csv")
INDIA_FLIGHT_WEATHER_CLEAN_PATH = os.path.join(OUTPUT_DIR, "cleaned_india_flights.csv")

# Final unified table (US + India, origin-weather schema) that the model is
# actually trained on — see build_unified_flight_dataset.py.
UNIFIED_FLIGHT_WEATHER_CLEAN_PATH = os.path.join(OUTPUT_DIR, "cleaned_flight_weather_unified.csv")

# Real historical average origin-airport-hourly-congestion, keyed by
# "AIRPORT|HOUR" (plus a "__default__" overall-mean fallback), precomputed by
# build_unified_flight_dataset.py so single ad-hoc predictions (predict_delay_v2.py)
# can look up a real typical value instead of guessing when the caller doesn't
# have an exact historical row to read it from directly.
CONGESTION_LOOKUP_PATH = os.path.join(OUTPUT_DIR, "congestion_lookup.json")

# US BTS "Reporting Carrier On-Time Performance" (transtats.bts.gov), recent
# years, with Open-Meteo weather at both ends - see build_bts_dataset.py. Kept
# under separate names so the 2019-trained model and its files stay intact
# until the BTS model is compared against it.
BTS_TRAIN_YEARS = [2025]
BTS_CLEAN_PATH = os.path.join(OUTPUT_DIR, "cleaned_bts_flights.csv")
BTS_TEST_CLEAN_PATH = os.path.join(OUTPUT_DIR, "cleaned_bts_test_flights.csv")
BTS_UNIFIED_CLEAN_PATH = os.path.join(OUTPUT_DIR, "cleaned_flight_weather_bts_unified.csv")
BTS_CONGESTION_LOOKUP_PATH = os.path.join(OUTPUT_DIR, "congestion_lookup_bts.json")

# GlobalWeatherRepository.csv — appears to be a rolling current-conditions
# snapshot (has last_updated_epoch), NOT historical data matching the 2019
# flights. Treated as a separate "live destination weather lookup" source
# for the recovery agent, not merged into the historical delay model.
WEATHER_RAW_PATH = os.path.join(DATA_DIR, "GlobalWeatherRepository.csv")
WEATHER_CLEAN_PATH = os.path.join(OUTPUT_DIR, "cleaned_global_weather.csv")

# Indexed, disk-backed copy of UNIFIED_FLIGHT_WEATHER_CLEAN_PATH - see
# build_flight_catalog_db.py. recovery_tools.py queries this instead of
# loading the full 864MB/5.4M-row CSV into memory at startup; a deploy host
# with ~512MB RAM can run the whole app this way instead of needing ~2-3GB
# for that one in-memory DataFrame.
FLIGHT_CATALOG_DB_PATH = os.path.join(OUTPUT_DIR, "flight_catalog.db")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)