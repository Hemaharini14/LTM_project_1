"""
Data preprocessing for Airlines.csv (539,383 historical US domestic flights).

Steps:
  1. Load + basic integrity checks (dtypes, nulls, duplicates)
  2. Handle duplicates / invalid rows
  3. Feature engineering: time-of-day bucket, weekend flag, route column
  4. Save cleaned dataset + print a summary report
"""
import pandas as pd
import numpy as np
from config import AIRLINES_RAW_PATH, AIRLINES_CLEAN_PATH


def time_bucket(minutes: int) -> str:
    hour = minutes // 60
    if 5 <= hour < 12:
        return "Morning"
    elif 12 <= hour < 17:
        return "Afternoon"
    elif 17 <= hour < 21:
        return "Evening"
    return "Night"


def load_raw(path: str = AIRLINES_RAW_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def inspect(df: pd.DataFrame) -> dict:
    report = {
        "n_rows": len(df),
        "n_cols": df.shape[1],
        "n_duplicates": int(df.duplicated().sum()),
        "null_counts": df.isnull().sum().to_dict(),
        "dtypes": df.dtypes.astype(str).to_dict(),
    }
    return report


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 1. Remove exact duplicate rows
    before = len(df)
    df = df.drop_duplicates()
    removed_dupes = before - len(df)

    # 2. Sanity-check value ranges (Airlines.csv is normally clean, but we
    #    guard against corrupted rows defensively rather than assuming)
    df = df[(df["Time"] >= 0) & (df["Time"] < 1440)]
    df = df[df["Length"] > 0]
    df = df[df["DayOfWeek"].between(1, 7)]
    df = df[df["Delay"].isin([0, 1])]

    # 3. Feature engineering
    df["route"] = df["AirportFrom"] + "-" + df["AirportTo"]
    df["time_of_day"] = df["Time"].apply(time_bucket)
    df["is_weekend"] = df["DayOfWeek"].isin([6, 7]).astype(int)
    df["departure_hour"] = df["Time"] // 60

    df = df.reset_index(drop=True)
    print(f"[clean_airlines] Removed {removed_dupes} exact duplicate rows.")
    print(f"[clean_airlines] {before - len(df) - removed_dupes} additional rows dropped on range checks.")
    return df


def run() -> pd.DataFrame:
    print("=" * 60)
    print("AIRLINES.CSV — PREPROCESSING")
    print("=" * 60)
    df = load_raw()
    report = inspect(df)
    print(f"Raw shape: {report['n_rows']:,} rows x {report['n_cols']} cols")
    print(f"Duplicate rows: {report['n_duplicates']}")
    nulls = {k: v for k, v in report["null_counts"].items() if v > 0}
    print(f"Columns with nulls: {nulls if nulls else 'None'}")

    df_clean = clean(df)
    df_clean.to_csv(AIRLINES_CLEAN_PATH, index=False)
    print(f"Clean shape: {df_clean.shape[0]:,} rows x {df_clean.shape[1]} cols")
    print(f"Saved -> {AIRLINES_CLEAN_PATH}")

    print("\nDelay rate by time_of_day:")
    print(df_clean.groupby("time_of_day")["Delay"].mean().sort_values(ascending=False))
    print("\nDelay rate: weekend vs weekday:")
    print(df_clean.groupby("is_weekend")["Delay"].mean())
    return df_clean


if __name__ == "__main__":
    run()