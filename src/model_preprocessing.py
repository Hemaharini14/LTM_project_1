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

# Categorical fields -> each gets its own embedding table
CAT_COLS = ["carrier_code", "origin_airport", "destination_airport", "weekday", "month"]

# Continuous fields -> scaled (zero mean, unit std) and fed directly into the MLP.
# Origin-side weather only: the India data merged in (see
# clean_india_flights.py) has no destination-side weather at all, so
# dest_* fields were dropped from the whole pipeline rather than faked.
# origin_temp_known (1/0) rides alongside origin_temp_f so the model can
# tell a real US reading apart from the India rows' filled-in placeholder.
CONT_COLS = [
    "scheduled_elapsed_time",
    "origin_temp_f", "origin_temp_known",
    "origin_precip_in", "origin_pressure", "origin_visibility", "origin_wind_speed",
]

TARGET_COL = "is_delayed"


class FlightWeatherEncoder:
    def __init__(self, cat_cols=None, cont_cols=None):
        self.cat_cols = cat_cols or CAT_COLS
        self.cont_cols = cont_cols or CONT_COLS
        self.vocab = {}
        self.cont_mean = {}
        self.cont_std = {}

    def fit(self, df: pd.DataFrame):
        for col in self.cat_cols:
            uniques = sorted(df[col].astype(str).unique().tolist())
            self.vocab[col] = {v: i + 1 for i, v in enumerate(uniques)}  # 0 = unknown/unseen
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