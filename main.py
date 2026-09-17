"""
Runs the full Step-1 (data preprocessing) pipeline end to end:
  1. Clean Airlines.csv                 -> outputs/cleaned_airlines.csv
  2. Clean hotel_bookings.csv           -> outputs/cleaned_hotel_bookings.csv
  3. Clean monthly flight+weather 2019  -> outputs/cleaned_flight_weather_2019.csv
  4. Clean Dataset.csv (India domestic) -> outputs/cleaned_india_flights.csv
  5. Merge US + India                   -> outputs/cleaned_flight_weather_unified.csv
  6. Clean GlobalWeatherRepository.csv  -> outputs/cleaned_global_weather.csv
  7. Generate EDA plots (Airlines + hotels) -> outputs/plots/*.png

Run from the project root:
    python main.py
"""
import sys
import os

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import clean_airlines
import clean_hotels
import clean_monthly_flights
import clean_india_flights
import build_unified_flight_dataset
import clean_weather
import eda_visuals


def main():
    airlines_df = clean_airlines.run()
    print()
    hotels_df = clean_hotels.run()
    print()

    try:
        flights_weather_df = clean_monthly_flights.run()
    except Exception as e:
        print(f"[main] Skipped monthly flight+weather step due to error: {e}")
        flights_weather_df = None
    print()

    unified_df = None
    try:
        india_flights_df = clean_india_flights.run()
        print()
        if flights_weather_df is not None:
            unified_df = build_unified_flight_dataset.run()
    except Exception as e:
        print(f"[main] Skipped India flight+weather step due to error: {e}")
        india_flights_df = None
    print()

    try:
        weather_df = clean_weather.run()
    except Exception as e:
        print(f"[main] Skipped GlobalWeatherRepository step due to error: {e}")
        weather_df = None
    print()

    eda_visuals.run()

    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)
    print(f"Cleaned Airlines rows:            {len(airlines_df):,}")
    print(f"Cleaned Hotel Bookings rows:       {len(hotels_df):,}")
    if flights_weather_df is not None:
        print(f"Cleaned Flight+Weather 2019 rows:  {len(flights_weather_df):,}")
    if india_flights_df is not None:
        print(f"Cleaned India Flight+Weather rows: {len(india_flights_df):,}")
    if unified_df is not None:
        print(f"Unified US+India training rows:    {len(unified_df):,}")
    if weather_df is not None:
        print(f"Cleaned Global Weather locations:  {len(weather_df):,}")
    print("Next step: model training (disruption prediction) — see project README.")


if __name__ == "__main__":
    main()