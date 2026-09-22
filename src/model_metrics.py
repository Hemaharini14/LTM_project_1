"""
Surfaces "is the delay prediction actually working?" in the admin panel with
real numbers, not a claim:
  - load_model_metrics(): the real held-out validation performance from the
    last run of evaluate_delay_model_v2.py (AUC, accuracy, precision/recall/F1) -
    computed on the 15% of real historical flights the model never trained on.
  - sample_validation_flights(): a handful of those real held-out flights,
    each with its real recorded conditions, its real actual outcome, AND a
    fresh live prediction from whichever model is currently loaded (recomputed
    now, not a frozen number from evaluation time) - so you can see concretely
    where the model agrees with reality and where it doesn't, not just a
    single aggregate score.
Both return None / [] (never fabricated placeholder numbers) if
evaluate_delay_model_v2.py hasn't been run yet.
"""
import os
import sys
import json
import random

import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import OUTPUT_DIR
from predict_delay_v2 import predict_delay_probability, risk_label

ARTIFACT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "artifacts")
METRICS_PATH = os.path.join(ARTIFACT_DIR, "metrics_v2.json")
VALIDATION_SAMPLE_PATH = os.path.join(OUTPUT_DIR, "validation_sample.csv")


def load_model_metrics() -> dict | None:
    if not os.path.exists(METRICS_PATH):
        return None
    with open(METRICS_PATH) as f:
        return json.load(f)


def sample_validation_flights(n: int = 6) -> list[dict]:
    """n real held-out flights (never seen in training) with their real actual
    outcome and a live re-prediction, so the admin panel can show concrete
    predicted-vs-actual examples rather than just a headline metric."""
    if not os.path.exists(VALIDATION_SAMPLE_PATH):
        return []
    df = pd.read_csv(VALIDATION_SAMPLE_PATH)
    rows = df.sample(n=min(n, len(df))).to_dict("records")

    out = []
    for row in rows:
        prob = predict_delay_probability(
            carrier_code=row["carrier_code"], origin_airport=row["origin_airport"],
            destination_airport=row["destination_airport"], weekday=row["weekday"],
            month=row["month"], scheduled_elapsed_time=row["scheduled_elapsed_time"],
            origin_temp_f=row["origin_temp_f"], origin_temp_known=bool(row["origin_temp_known"]),
            origin_precip_in=row["origin_precip_in"], origin_pressure=row["origin_pressure"],
            origin_visibility=row["origin_visibility"], origin_wind_speed=row["origin_wind_speed"],
            scheduled_hour=row["scheduled_hour"], is_holiday=bool(row["is_holiday"]),
            origin_hourly_congestion=row["origin_hourly_congestion"],
        )
        actual_delayed = bool(row["is_delayed"])
        predicted_label = risk_label(prob)
        # A prediction "agrees" with reality if High/Moderate lined up with an actual
        # delay, or Low lined up with on-time - the same real threshold semantics
        # used everywhere else in the app (see predict_delay_v2.risk_label).
        predicted_delayed = predicted_label in ("High", "Moderate")
        out.append({
            "carrier_code": row["carrier_code"], "origin_airport": row["origin_airport"],
            "destination_airport": row["destination_airport"], "weekday": row["weekday"],
            "month": row["month"],
            "predicted_probability": round(prob, 3), "predicted_label": predicted_label,
            "actual_outcome": "Delayed" if actual_delayed else "On-time",
            "agrees": predicted_delayed == actual_delayed,
        })
    return out
