"""
Shared derived-feature helpers for the flight+weather cleaners
(clean_monthly_flights.py, clean_india_flights.py) so both sources compute
the same real, data-derived features identically. Nothing here is
fabricated - every feature is a direct function of the real scheduled
departure timestamp, the real weekday, or the real scheduled traffic
already present in the dataset.

Holiday dates are the same kind of hand-curated, well-known public
reference data as reference_data.py / intl_reference.py - real official
holidays, not sourced from any private dataset.
"""
import pandas as pd

US_FEDERAL_HOLIDAYS_2019 = {
    "2019-05-27",  # Memorial Day
    "2019-07-04",  # Independence Day
    "2019-09-02",  # Labor Day
    "2019-10-14",  # Columbus Day
    "2019-11-11",  # Veterans Day
    "2019-11-28",  # Thanksgiving Day
    "2019-12-25",  # Christmas Day
}
# Covers the India data's real date range (Jan 2019 - Jan 2020).
INDIA_NATIONAL_HOLIDAYS = {
    "2019-01-01",  # New Year's Day
    "2019-01-26",  # Republic Day
    "2019-08-15",  # Independence Day
    "2019-10-02",  # Gandhi Jayanti
    "2019-10-27",  # Diwali
    "2019-12-25",  # Christmas Day
    "2020-01-01",  # New Year's Day
    "2020-01-26",  # Republic Day
}
ALL_HOLIDAYS = US_FEDERAL_HOLIDAYS_2019 | INDIA_NATIONAL_HOLIDAYS


def is_holiday_date(date_str: str) -> bool:
    """date_str must be 'YYYY-MM-DD'. Used wherever only a date (not a full
    training dataframe) is available, e.g. the web form's travel_date."""
    return date_str in ALL_HOLIDAYS


def add_calendar_features(df: pd.DataFrame, departure_col: str = "scheduled_departure_dt") -> pd.DataFrame:
    """Adds scheduled_hour (0-23), is_weekend, and is_holiday from a real
    scheduled-departure timestamp. Mutates and returns df."""
    dt = df[departure_col]
    df["scheduled_hour"] = dt.dt.hour
    df["is_weekend"] = df["weekday"].isin([5, 6]).astype(int)
    df["is_holiday"] = dt.dt.strftime("%Y-%m-%d").isin(ALL_HOLIDAYS).astype(int)
    return df


def add_origin_congestion(df: pd.DataFrame, departure_col: str = "scheduled_departure_dt") -> pd.DataFrame:
    """Real congestion proxy: how many flights (from this same source's raw
    data, cancelled or not) are scheduled to depart the same origin airport
    within the same calendar hour. Computed purely from this dataset's own
    scheduled departures - not a live/external feed, not fabricated.
    Mutates and returns df."""
    hour_bucket = df[departure_col].dt.floor("h")
    df["origin_hourly_congestion"] = df.groupby(["origin_airport", hour_bucket])[departure_col].transform("count")
    return df
