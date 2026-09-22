"""
Builds the training table for the delay-DURATION model.

The classifier's unified table (build_unified_flight_dataset.py) deliberately
drops departure_delay - it only needs the is_delayed boolean. The cleaned
intermediate files still carry the real minutes though, so this reads those
directly rather than re-cleaning the raw monthlies, and never touches the
unified table the calibrated classifier depends on.

Trained on DELAYED flights only, so the target is "given this flight is late,
how late". That composes cleanly with the existing classifier rather than
duplicating it:
    P(delay)              <- delay_model_v2 (calibrated)
    E[minutes | delayed]  <- this model
Mixing on-time flights in would just make the regressor spend its capacity
re-learning the classification boundary it already has a better answer for.

Run: python build_delay_duration_dataset.py
Output: outputs/delay_duration_training.csv
"""
import os
import sys

import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import (MONTHLY_FLIGHT_WEATHER_CLEAN_PATH, INDIA_FLIGHT_WEATHER_CLEAN_PATH,
                     OUTPUT_DIR)
from build_unified_flight_dataset import SHARED_COLS

TRAINING_PATH = os.path.join(OUTPUT_DIR, "delay_duration_training.csv")
DELAY_THRESHOLD_MIN = 15
# Real but useless to learn from: a handful of multi-day disruptions would
# dominate the loss without being predictable from schedule and weather.
MAX_DELAY_MIN = 600

COLS = SHARED_COLS + ["departure_delay"]


def run() -> pd.DataFrame:
    frames = []
    for path, label in [(MONTHLY_FLIGHT_WEATHER_CLEAN_PATH, "US"),
                        (INDIA_FLIGHT_WEATHER_CLEAN_PATH, "India")]:
        df = pd.read_csv(path, low_memory=False)
        if "flight_number" not in df.columns:
            df["flight_number"] = ""
        df = df[COLS]
        delayed = df[df["departure_delay"] >= DELAY_THRESHOLD_MIN]
        print(f"{label}: {len(df):,} rows -> {len(delayed):,} delayed")
        frames.append(delayed)

    combined = pd.concat(frames, ignore_index=True)

    known_temp_mean = combined.loc[combined["origin_temp_known"] == 1, "origin_temp_f"].mean()
    combined["origin_temp_f"] = combined["origin_temp_f"].fillna(known_temp_mean)

    before = len(combined)
    combined = combined[combined["departure_delay"] <= MAX_DELAY_MIN]
    print(f"Dropped {before - len(combined):,} delays over {MAX_DELAY_MIN} min as unlearnable outliers.")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    combined.to_csv(TRAINING_PATH, index=False)
    d = combined["departure_delay"]
    print(f"\nSaved {len(combined):,} rows -> {TRAINING_PATH}")
    print(f"Delay minutes: median {d.median():.0f} | mean {d.mean():.1f} | "
          f"p90 {d.quantile(.9):.0f} | max {d.max():.0f}")
    return combined


if __name__ == "__main__":
    run()
