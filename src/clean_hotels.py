"""
Data preprocessing for hotel_bookings.csv (119,390 hotel booking records).

Steps:
  1. Load + basic integrity checks (dtypes, nulls, duplicates)
  2. Handle missing values (children, country, agent, company)
  3. Remove invalid/outlier rows (negative or absurd adr)
  4. Feature engineering: total_nights, has_children, is_family, is_domestic-style flags
  5. Save cleaned dataset + print a summary report
"""
import pandas as pd
import numpy as np
from config import HOTELS_RAW_PATH, HOTELS_CLEAN_PATH


def load_raw(path: str = HOTELS_RAW_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def inspect(df: pd.DataFrame) -> dict:
    return {
        "n_rows": len(df),
        "n_cols": df.shape[1],
        "n_duplicates": int(df.duplicated().sum()),
        "null_counts": df.isnull().sum().to_dict(),
    }


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 1. Remove exact duplicate rows (hotel_bookings.csv is known to have
    #    many, since it's a widely-used teaching dataset with repeat records)
    before = len(df)
    df = df.drop_duplicates()
    removed_dupes = before - len(df)

    # 2. Missing value handling
    df["children"] = pd.to_numeric(df["children"], errors="coerce").fillna(0).astype(int)
    df["country"] = df["country"].fillna("Unknown")
    df["agent"] = df["agent"].fillna(0)
    df["company"] = df["company"].fillna(0)

    # 3. Remove invalid adr rows (negative rates are data errors; extreme
    #    outliers above $2000/night are dropped rather than capped, since
    #    they're rare enough (<0.01%) that capping would distort the segment
    #    medians used later for hotel recommendations)
    before_adr = len(df)
    df = df[(df["adr"] >= 0) & (df["adr"] <= 2000)]
    removed_adr_outliers = before_adr - len(df)

    # 4. Drop rows with zero guests entirely (adults+children+babies == 0),
    #    a known data-quality issue in this dataset
    before_guests = len(df)
    df = df[(df["adults"] + df["children"] + df["babies"]) > 0]
    removed_zero_guest = before_guests - len(df)

    # 5. Feature engineering
    df["total_nights"] = df["stays_in_weekend_nights"] + df["stays_in_week_nights"]
    df = df[df["total_nights"] > 0]  # a stay must have at least 1 night
    df["has_children"] = ((df["children"] + df["babies"]) > 0).astype(int)
    df["total_guests"] = df["adults"] + df["children"] + df["babies"]
    df["estimated_stay_cost"] = (df["adr"] * df["total_nights"]).round(2)

    df = df.reset_index(drop=True)
    print(f"[clean_hotels] Removed {removed_dupes} exact duplicate rows.")
    print(f"[clean_hotels] Removed {removed_adr_outliers} rows with invalid/outlier adr.")
    print(f"[clean_hotels] Removed {removed_zero_guest} rows with zero total guests.")
    return df


def run() -> pd.DataFrame:
    print("=" * 60)
    print("HOTEL_BOOKINGS.CSV — PREPROCESSING")
    print("=" * 60)
    df = load_raw()
    report = inspect(df)
    print(f"Raw shape: {report['n_rows']:,} rows x {report['n_cols']} cols")
    print(f"Duplicate rows: {report['n_duplicates']:,}")
    nulls = {k: v for k, v in report["null_counts"].items() if v > 0}
    print(f"Columns with nulls: {nulls if nulls else 'None'}")

    df_clean = clean(df)
    df_clean.to_csv(HOTELS_CLEAN_PATH, index=False)
    print(f"Clean shape: {df_clean.shape[0]:,} rows x {df_clean.shape[1]} cols")
    print(f"Saved -> {HOTELS_CLEAN_PATH}")

    print("\nMedian ADR by hotel type:")
    print(df_clean.groupby("hotel")["adr"].median())
    print("\nCancellation rate by hotel type:")
    print(df_clean.groupby("hotel")["is_canceled"].mean())
    return df_clean


if __name__ == "__main__":
    run()