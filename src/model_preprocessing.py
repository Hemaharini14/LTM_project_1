"""
Generalized categorical/continuous encoder for the disruption prediction
model trained on the flight+weather 2019 dataset. Same pattern as the
earlier Airlines-only encoder, but parameterized so it works with the
richer feature set (weather + calendar fields) without rewriting the class.
"""
import pandas as pd
import numpy as np
import joblib
import os

# Categorical fields -> each gets its own embedding table.
# scheduled_hour (0-23) is categorical rather than a scaled continuous value
# because time-of-day effects aren't linear (e.g. evening cascading delays
# are qualitatively different from a quiet mid-morning slot) - same reasoning
# as treating weekday/month as embeddings rather than raw numbers.
CAT_COLS = ["carrier_code", "origin_airport", "destination_airport", "weekday", "month", "scheduled_hour"]

# Continuous fields -> scaled (zero mean, unit std) and fed directly into the MLP.
# Origin-side weather only: the India data merged in (see
# clean_india_flights.py) has no destination-side weather at all, so
# dest_* fields were dropped from the whole pipeline rather than faked.
# origin_temp_known (1/0) rides alongside origin_temp_f so the model can
# tell a real US reading apart from the India rows' filled-in placeholder.
#
# is_weekend/is_holiday/origin_hourly_congestion: real calendar and traffic
# features derived straight from the data itself - see feature_engineering.py.
# route_frequency: NOT a raw column - derived at fit/transform time below from
# how often each route appears in the TRAINING split only (see fit()), so an
# unseen-in-training route never leaks a val/test-only signal.
CONT_COLS = [
    "scheduled_elapsed_time",
    "origin_temp_f", "origin_temp_known",
    "origin_precip_in", "origin_pressure", "origin_visibility", "origin_wind_speed",
    "is_weekend", "is_holiday", "origin_hourly_congestion", "route_frequency",
    # Aircraft rotation. prev_leg_arrival_delay is the strongest single feature in
    # the data (9% delay rate when the inbound arrived early vs 84% when it was
    # 45-90 min late) but is only observable same-day, so prev_leg_known rides
    # alongside it exactly as origin_temp_known does. leg_of_day and
    # scheduled_turnaround_min are pure schedule structure and always knowable.
    "leg_of_day", "scheduled_turnaround_min", "prev_leg_arrival_delay", "prev_leg_known",
    # Destination-side weather, previously computed and thrown away.
    "dest_temp_f", "dest_precip_in", "dest_pressure", "dest_visibility",
    "dest_wind_speed", "dest_weather_known",
]

TARGET_COL = "is_delayed"


class FlightWeatherEncoder:
    def __init__(self, cat_cols=None, cont_cols=None):
        self.cat_cols = cat_cols or CAT_COLS
        self.cont_cols = cont_cols or CONT_COLS
        self.vocab = {}
        self.cont_mean = {}
        self.cont_std = {}
        self.route_freq = {}  # route -> log1p(training-set occurrence count)

    def _with_route_frequency(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adds route_frequency if it's a configured feature and not already a
        column - derived from origin/destination so callers never need to pass
        a redundant "route" string. Unseen-in-training routes get 0.0 (a real
        signal: "this exact pairing wasn't observed"), not a guessed value."""
        if "route_frequency" not in self.cont_cols or "route_frequency" in df.columns:
            return df
        df = df.copy()
        route = df["route"] if "route" in df.columns else (
            df["origin_airport"].astype(str) + "-" + df["destination_airport"].astype(str))
        df["route_frequency"] = route.map(self.route_freq).fillna(0.0)
        return df

    def fit(self, df: pd.DataFrame):
        for col in self.cat_cols:
            uniques = sorted(df[col].astype(str).unique().tolist())
            self.vocab[col] = {v: i + 1 for i, v in enumerate(uniques)}  # 0 = unknown/unseen
        if "route_frequency" in self.cont_cols:
            route = df["route"] if "route" in df.columns else (
                df["origin_airport"].astype(str) + "-" + df["destination_airport"].astype(str))
            self.route_freq = np.log1p(route.value_counts()).to_dict()
        df = self._with_route_frequency(df)
        for col in self.cont_cols:
            self.cont_mean[col] = float(df[col].mean())
            self.cont_std[col] = float(df[col].std() or 1.0)
        return self

    def vocab_sizes(self):
        return {col: len(v) + 1 for col, v in self.vocab.items()}

    def transform_cat(self, df: pd.DataFrame) -> np.ndarray:
        arrs = []
        for col in self.cat_cols:
            mapping = self.vocab[col]
            arrs.append(df[col].astype(str).map(lambda x: mapping.get(x, 0)).to_numpy())
        return np.stack(arrs, axis=1).astype(np.int64)

    def transform_cont(self, df: pd.DataFrame) -> np.ndarray:
        df = self._with_route_frequency(df)
        arrs = []
        for col in self.cont_cols:
            vals = (df[col].to_numpy(dtype=np.float32) - self.cont_mean[col]) / self.cont_std[col]
            arrs.append(vals)
        return np.stack(arrs, axis=1).astype(np.float32)

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "FlightWeatherEncoder":
        return joblib.load(path)


def load_clean_flight_weather(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    for col in CAT_COLS:
        if col in df.columns:
            df[col] = df[col].astype(str)
    return df