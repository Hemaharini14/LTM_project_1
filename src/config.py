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

# GlobalWeatherRepository.csv — appears to be a rolling current-conditions
# snapshot (has last_updated_epoch), NOT historical data matching the 2019
# flights. Treated as a separate "live destination weather lookup" source
# for the recovery agent, not merged into the historical delay model.
WEATHER_RAW_PATH = os.path.join(DATA_DIR, "GlobalWeatherRepository.csv")
WEATHER_CLEAN_PATH = os.path.join(OUTPUT_DIR, "cleaned_global_weather.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)