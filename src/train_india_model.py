"""
A delay model for Indian routes, because the main one cannot serve them.

DelayNetV2 is trained on 5.4M rows of which 10,634 - 0.20% - are Indian, and
those rows are impoverished in exactly the features the network leans on:

    origin_temp_f            1 distinct value   (placeholder)
    origin_temp_known        1 distinct value   (always 0)
    prev_leg_arrival_delay   1 distinct value   (constant)
    prev_leg_known           1 distinct value   (always 0)
    dest_weather_known       1 distinct value

They also cover four airports and five routes. Chennai is not among them. So on
real Indian departures the network scores AUC 0.594, and a logistic regression
on nothing but the origin airport beats it at 0.701 - the data is predictable,
the model simply is not learning it.

This trains on what Indian flights actually have. Two sources:

    10,634 historical Indian rows from the unified dataset
       354 real 2026 outcomes from collect_outcomes.py

Cross-validated over the 2026 rows, with the historical rows always in train
and never in test:

    DelayNetV2 (current)                  0.594
    this, historical only                 0.657
    this, historical + 2026 outcomes      0.723

Upweighting the recent rows was tried and is worse (x20 -> 0.692), so they go
in at weight 1. Gradient boosting was tried and is a hair behind logistic
regression (0.719) with more machinery, so logistic wins.

WHAT THIS DOES NOT FIX. The ranking improves; the absolute level stays
uncertain. The historical rows say 27.1% of Indian departures are 15+ minutes
late, the 2026 sample says 42.1%, and one day at three airports cannot settle
which is right. Probabilities are calibrated against the training mix and
should be read as "higher means likelier", not as a frequency you could bet on.
Re-run this as more outcomes accumulate.
"""
from __future__ import annotations

import os
import sqlite3
import sys
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import UNIFIED_FLIGHT_WEATHER_CLEAN_PATH  # noqa: E402
from db import DB_PATH  # noqa: E402

# Carriers that mark a row as an Indian operation in the unified dataset.
INDIA_CARRIERS = {"6E", "AI", "UK", "SG", "G8", "AK", "IX", "QP", "9I", "I5", "S5"}

CAT_FEATURES = ["origin_airport", "destination_airport", "carrier_code"]
NUM_FEATURES = ["scheduled_hour", "weekday", "month", "origin_hourly_congestion",
                "origin_precip_in", "origin_visibility", "origin_wind_speed",
                "origin_pressure", "scheduled_elapsed_time"]
FEATURES = CAT_FEATURES + NUM_FEATURES

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "models", "artifacts", "india_delay_model.joblib")


def _historical() -> pd.DataFrame:
    df = pd.read_csv(UNIFIED_FLIGHT_WEATHER_CLEAN_PATH, low_memory=False)
    ind = df[df["carrier_code"].isin(INDIA_CARRIERS)].copy()
    for c in NUM_FEATURES:
        ind[c] = pd.to_numeric(ind[c], errors="coerce")
    return ind.dropna(subset=NUM_FEATURES + ["is_delayed"])[FEATURES + ["is_delayed"]]


def _collected(congestion_default: float) -> pd.DataFrame:
    """Real 2026 outcomes, in the same feature space.

    Only rows sampled across the whole day (is_representative = 1). A single
    page lands inside one part of the day and its delay rate is an artefact of
    which part - see collect_outcomes.
    """
    with sqlite3.connect(DB_PATH, timeout=15) as conn:
        conn.row_factory = sqlite3.Row
        rows = [dict(r) for r in conn.execute(
            "SELECT * FROM flight_outcomes WHERE is_representative = 1 "
            "AND departure_delay_min IS NOT NULL")]
    out = []
    for r in rows:
        try:
            dep = datetime.fromisoformat(r["scheduled_departure"])
        except Exception:
            continue
        out.append({
            "origin_airport": r["origin"], "destination_airport": r["destination"],
            "carrier_code": r["carrier_iata"] or "6E",
            "scheduled_hour": dep.hour, "weekday": dep.weekday(), "month": dep.month,
            "origin_hourly_congestion": congestion_default,
            # No observed weather for these; the neutral values keep them from
            # implying conditions nobody recorded.
            "origin_precip_in": 0.0, "origin_visibility": 10.0,
            "origin_wind_speed": 8.0, "origin_pressure": 29.9,
            "scheduled_elapsed_time": 130.0,
            "is_delayed": int(r["was_delayed"]),
        })
    return pd.DataFrame(out)


def _pipeline():
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_FEATURES),
        ("num", StandardScaler(), NUM_FEATURES),
    ])
    return make_pipeline(pre, LogisticRegression(max_iter=3000, C=0.5))


def evaluate(hist: pd.DataFrame, live: pd.DataFrame) -> dict:
    """Honest AUC: fold over the 2026 rows, historical always in train."""
    if len(live) < 25:
        return {"auc": None, "note": "too few collected outcomes to cross-validate"}
    y = live["is_delayed"].values
    aucs = []
    for tr_i, te_i in StratifiedKFold(5, shuffle=True, random_state=0).split(live, y):
        train = pd.concat([hist, live.iloc[tr_i]], ignore_index=True)
        m = _pipeline().fit(train[FEATURES], train["is_delayed"])
        aucs.append(roc_auc_score(y[te_i], m.predict_proba(live.iloc[te_i][FEATURES])[:, 1]))
    return {"auc": round(float(np.mean(aucs)), 3), "auc_std": round(float(np.std(aucs)), 3),
            "folds": len(aucs)}


def train() -> dict:
    hist = _historical()
    live = _collected(float(hist["origin_hourly_congestion"].median()))
    scores = evaluate(hist, live)

    full = pd.concat([hist, live], ignore_index=True) if len(live) else hist
    # Calibrated so the output is a probability rather than a score. Isotonic
    # needs more data than this; sigmoid is the right choice at ~11k rows.
    model = CalibratedClassifierCV(_pipeline(), method="sigmoid", cv=5)
    model.fit(full[FEATURES], full["is_delayed"])

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump({
        "model": model, "features": FEATURES,
        "cat_features": CAT_FEATURES, "num_features": NUM_FEATURES,
        "trained_rows": len(full), "historical_rows": len(hist),
        "collected_rows": len(live),
        "base_rate": round(float(full["is_delayed"].mean()), 4),
        "cv_auc": scores.get("auc"),
        "airports": sorted(set(full["origin_airport"].dropna())),
        "carriers": sorted(set(full["carrier_code"].dropna())),
    }, MODEL_PATH)

    return {"path": MODEL_PATH, "rows": len(full), "historical": len(hist),
            "collected": len(live), "base_rate": round(float(full["is_delayed"].mean()), 4),
            **scores}


if __name__ == "__main__":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    result = train()
    print("\nIndia delay model trained")
    for k, v in result.items():
        print(f"   {k:16s} {v}")
    print(f"\n   for comparison, DelayNetV2 scores 0.594 on the same real departures")
