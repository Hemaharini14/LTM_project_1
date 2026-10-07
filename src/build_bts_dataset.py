"""
Builds a training set from the US BTS "Reporting Carrier On-Time Performance"
files, with weather at BOTH ends of every flight.

The 2019 monthly files the model was first trained on are themselves BTS data
joined to NOAA station readings - this is the same source, brought forward to
the years the app is actually used in. Raw BTS has no weather at all, so it is
added here from Open-Meteo's historical-forecast archive:

  why that archive  it is the stored output of the same forecast models
                    weather_live.py queries at serving time, so the model
                    trains on the kind of number it will later be given. The
                    2019 NOAA columns were station observations the app can
                    never supply in advance. It is also the only Open-Meteo
                    archive with visibility.

  which hour        origin weather at the scheduled DEPARTURE hour, destination
                    weather at the scheduled ARRIVAL hour - both knowable
                    before the flight. Never the actual times: a late flight
                    lands later, and weather keyed to that would leak the label.

  units             converted exactly as weather_live.py does (sea-level
                    pressure in inHg, visibility clamped at 10 mi), so training
                    and serving cannot drift apart.

Flights at an airport with no published coordinates are dropped rather than
given a median reading - the counts are printed.

Everything is cached under dataset/bts/, so a rerun only fetches what is
missing. Run:
    python build_bts_dataset.py                 # train years (BTS_TRAIN_YEARS)
    python build_bts_dataset.py --test 2026     # out-of-time test months
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
import zipfile

import httpx
import numpy as np
import pandas as pd

try:
    import truststore
    truststore.inject_into_ssl()
except ImportError:
    pass

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import (DATA_DIR, OUTPUT_DIR, INDIA_FLIGHT_WEATHER_CLEAN_PATH,  # noqa: E402
                    BTS_TRAIN_YEARS, BTS_CLEAN_PATH, BTS_TEST_CLEAN_PATH,
                    BTS_UNIFIED_CLEAN_PATH, BTS_CONGESTION_LOOKUP_PATH)
from clean_monthly_flights import clean  # noqa: E402
import build_unified_flight_dataset  # noqa: E402

BTS_DIR = os.path.join(DATA_DIR, "bts")
RAW_DIR = os.path.join(BTS_DIR, "raw")
MONTH_DIR = os.path.join(BTS_DIR, "months")
WEATHER_DIR = os.path.join(BTS_DIR, "weather")
COORDS_PATH = os.path.join(BTS_DIR, "airport_coords.json")

BTS_URL = ("https://transtats.bts.gov/PREZIP/"
           "On_Time_Reporting_Carrier_On_Time_Performance_1987_present_{year}_{month}.zip")
OURAIRPORTS_URL = "https://davidmegginson.github.io/ourairports-data/airports.csv"
WEATHER_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
HOURLY = "temperature_2m,precipitation,pressure_msl,visibility,wind_speed_10m"

# Same conversions as weather_live.py - see its docstring for why each matters.
_HPA_TO_INHG = 0.02953
_M_TO_MI = 1 / 1609.34
MAX_VISIBILITY_MI = 10.0

BTS_COLUMNS = {
    "FlightDate": "date", "Reporting_Airline": "carrier_code",
    "Flight_Number_Reporting_Airline": "flight_number", "Tail_Number": "tail_number",
    "Origin": "origin_airport", "Dest": "destination_airport",
    "Month": "month", "DayOfWeek": "weekday",
    "CRSDepTime": "crs_dep", "CRSArrTime": "crs_arr",
    "DepDelay": "departure_delay", "ArrDelay": "arrival_delay",
    "Cancelled": "cancelled_code", "Diverted": "diverted",
    "CRSElapsedTime": "scheduled_elapsed_time", "Distance": "distance_miles",
    "CarrierDelay": "delay_carrier", "WeatherDelay": "delay_weather",
    "NASDelay": "delay_national_aviation_system", "SecurityDelay": "delay_security",
    "LateAircraftDelay": "delay_late_aircarft_arrival",
}


def _client() -> httpx.Client:
    return httpx.Client(timeout=300, follow_redirects=True)


# ------------------------------------------------------------------ BTS months

def fetch_month(year: int, month: int) -> str | None:
    """Downloads one BTS month and keeps only the columns used. Returns the CSV
    path, or None if BTS hasn't published that month yet."""
    out = os.path.join(MONTH_DIR, f"{year}-{month:02d}.csv")
    if os.path.exists(out):
        return out
    os.makedirs(MONTH_DIR, exist_ok=True)
    url = BTS_URL.format(year=year, month=month)
    print(f"  downloading BTS {year}-{month:02d} ...", flush=True)
    for attempt in range(3):
        try:
            with _client() as c:
                r = c.get(url)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            break
        except httpx.HTTPError as e:
            print(f"    attempt {attempt + 1} failed: {e}")
            time.sleep(10)
    else:
        raise RuntimeError(f"could not download {url}")
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        name = next(n for n in z.namelist() if n.lower().endswith(".csv"))
        with z.open(name) as f:
            df = pd.read_csv(f, usecols=list(BTS_COLUMNS), low_memory=False,
                             dtype={"CRSDepTime": str, "CRSArrTime": str})
    df.rename(columns=BTS_COLUMNS).to_csv(out, index=False)
    print(f"    {len(df):,} flights -> {out}")
    return out


def _local_dt(date: pd.Series, hhmm: pd.Series) -> pd.Series:
    """BTS schedule times are local 'hhmm' strings; 2400 means midnight next day."""
    hhmm = hhmm.astype(str).str.zfill(4)
    hours = hhmm.str[:2].astype(int)
    minutes = hhmm.str[2:].astype(int)
    return pd.to_datetime(date) + pd.to_timedelta(hours * 60 + minutes, unit="m")


def load_month(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False, dtype={"crs_dep": str, "crs_arr": str})
    df = df[df["diverted"].fillna(0) == 0].drop(columns=["diverted"])
    # A handful of BTS rows have no scheduled duration; the model needs it.
    df = df.dropna(subset=["scheduled_elapsed_time", "crs_dep", "crs_arr"])
    df["weekday"] = df["weekday"].astype(int) - 1      # BTS 1=Mon -> 0=Mon, as in 2019 data
    df["scheduled_departure_dt"] = _local_dt(df["date"], df["crs_dep"])
    arr = _local_dt(df["date"], df["crs_arr"])
    # Arrival time is on the departure date in BTS; an earlier clock time means it
    # landed the next day (red-eyes).
    df["scheduled_arrival_dt"] = arr.where(arr >= df["scheduled_departure_dt"], arr + pd.Timedelta(days=1))
    for col in ["delay_carrier", "delay_weather", "delay_national_aviation_system",
                "delay_security", "delay_late_aircarft_arrival"]:
        df[col] = df[col].fillna(0)
    return df.drop(columns=["crs_dep", "crs_arr"])


# ------------------------------------------------------------------ weather

def airport_coords() -> dict[str, tuple[float, float]]:
    if os.path.exists(COORDS_PATH):
        with open(COORDS_PATH) as f:
            return {k: tuple(v) for k, v in json.load(f).items()}
    print("  downloading airport coordinates (OurAirports) ...")
    with _client() as c:
        r = c.get(OURAIRPORTS_URL)
    r.raise_for_status()
    every = pd.read_csv(io.StringIO(r.text), usecols=["iata_code", "latitude_deg", "longitude_deg",
                                                      "type", "keywords"])
    ap = every[every["iata_code"].notna() & every["type"].str.contains("airport")]
    # A few IATA codes appear twice (a closed field reusing the code); keep the larger one.
    rank = {"large_airport": 0, "medium_airport": 1, "small_airport": 2}
    ap = ap.assign(r=ap["type"].map(rank).fillna(3)).sort_values("r").drop_duplicates("iata_code")
    coords = {row.iata_code: (row.latitude_deg, row.longitude_deg) for row in ap.itertuples()}
    # A renamed airport can lose its old code to the keywords field - Palm Beach is
    # listed as DJT with "PBI" only in keywords, while BTS still reports PBI. Fill
    # such gaps from medium/large airports' keywords, never overriding a real code.
    big = every[every["type"].isin(["large_airport", "medium_airport"]) & every["keywords"].notna()]
    for row in big.itertuples():
        for token in str(row.keywords).replace(";", ",").split(","):
            token = token.strip()
            if len(token) == 3 and token.isalpha() and token.isupper() and token not in coords:
                coords[token] = (row.latitude_deg, row.longitude_deg)
    os.makedirs(BTS_DIR, exist_ok=True)
    with open(COORDS_PATH, "w") as f:
        json.dump(coords, f)
    return coords


def fetch_weather(iata: str, lat: float, lon: float, start: str, end: str) -> pd.DataFrame | None:
    """Hourly local-time weather for one airport, cached per airport and range."""
    path = os.path.join(WEATHER_DIR, f"{iata}_{start}_{end}.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    os.makedirs(WEATHER_DIR, exist_ok=True)
    params = {"latitude": lat, "longitude": lon, "hourly": HOURLY, "timezone": "auto",
              "start_date": start, "end_date": end, "temperature_unit": "fahrenheit",
              "wind_speed_unit": "mph", "precipitation_unit": "inch"}
    for attempt in range(6):
        try:
            with _client() as c:
                r = c.get(WEATHER_URL, params=params)
            if r.status_code == 429:
                # Free tier is rate limited per minute/hour/day - wait and retry.
                wait = 60 * (attempt + 1)
                print(f"    rate limited on {iata}, waiting {wait}s")
                time.sleep(wait)
                continue
            r.raise_for_status()
            h = r.json()["hourly"]
            break
        except (httpx.HTTPError, KeyError, ValueError) as e:   # ValueError: empty/non-JSON body
            print(f"    weather {iata} attempt {attempt + 1} failed: {e}")
            time.sleep(15)
    else:
        return None
    w = pd.DataFrame({
        "hour": h["time"],
        "temp_f": h["temperature_2m"],
        "precip_in": h["precipitation"],
        "pressure": pd.Series(h["pressure_msl"], dtype=float) * _HPA_TO_INHG,
        "visibility": (pd.Series(h["visibility"], dtype=float) * _M_TO_MI).clip(upper=MAX_VISIBILITY_MI),
        "wind_speed": h["wind_speed_10m"],
    }).drop_duplicates("hour")          # the repeated hour when DST ends
    w.to_csv(path, index=False)
    return w


def attach_weather(df: pd.DataFrame, weather: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """origin_* at the scheduled departure hour, dest_* at the scheduled arrival hour."""
    long = pd.concat([w.assign(airport=a) for a, w in weather.items()], ignore_index=True)
    long["hour"] = pd.to_datetime(long["hour"])
    fields = ["temp_f", "precip_in", "pressure", "visibility", "wind_speed"]
    for side, airport_col, time_col in [("origin", "origin_airport", "scheduled_departure_dt"),
                                         ("dest", "destination_airport", "scheduled_arrival_dt")]:
        renamed = long.rename(columns={"airport": airport_col, "hour": f"_{side}_hour",
                                       **{f: f"{side}_{f}" for f in fields}})
        df[f"_{side}_hour"] = df[time_col].dt.floor("h")
        df = df.merge(renamed, on=[airport_col, f"_{side}_hour"], how="left").drop(columns=[f"_{side}_hour"])
    return df


# ------------------------------------------------------------------ pipeline

def build(year_months: list[tuple[int, int]], out_path: str) -> pd.DataFrame:
    paths = [(y, m, fetch_month(y, m)) for y, m in year_months]
    paths = [(y, m, p) for y, m, p in paths if p]
    if not paths:
        raise RuntimeError("BTS has published none of the requested months yet.")
    print(f"BTS months available: {', '.join(f'{y}-{m:02d}' for y, m, _ in paths)}")

    coords = airport_coords()
    airports = set()
    for _, _, p in paths:
        a = pd.read_csv(p, usecols=["origin_airport", "destination_airport"])
        airports |= set(a["origin_airport"]) | set(a["destination_airport"])
    missing = sorted(a for a in airports if a not in coords)
    print(f"{len(airports)} airports; {len(missing)} without coordinates: {missing}")

    # One request per airport covers the whole span (+1 day for overnight arrivals).
    start = f"{paths[0][0]}-{paths[0][1]:02d}-01"
    last = pd.Timestamp(year=paths[-1][0], month=paths[-1][1], day=1) + pd.offsets.MonthEnd(0) + pd.Timedelta(days=1)
    end = last.strftime("%Y-%m-%d")
    weather = {}
    for i, a in enumerate(sorted(airports - set(missing)), 1):
        w = fetch_weather(a, *coords[a], start, end)
        if w is not None:
            weather[a] = w
        if i % 25 == 0:
            print(f"  weather {i}/{len(airports) - len(missing)} airports", flush=True)
    print(f"Weather fetched for {len(weather)} airports")

    frames = []
    for y, m, p in paths:
        df = attach_weather(load_month(p), weather)
        wcols = [c for c in df.columns if c.startswith(("origin_", "dest_")) and c not in
                 ("origin_airport",)]
        no_wx = df[wcols].isna().any(axis=1)
        print(f"[{y}-{m:02d}] {len(df):,} flights, dropping {int(no_wx.sum()):,} with no weather at one end")
        frames.append(clean(df[~no_wx]))
    out = pd.concat(frames, ignore_index=True)
    out.to_csv(out_path, index=False)
    print(f"\nSaved {len(out):,} flights -> {out_path}  (delay rate {out['is_delayed'].mean():.1%})")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", type=int, help="build out-of-time test months for this year instead")
    args = ap.parse_args()
    if args.test:
        build([(args.test, m) for m in range(1, 13)], BTS_TEST_CLEAN_PATH)
        return
    build([(y, m) for y in BTS_TRAIN_YEARS for m in range(1, 13)], BTS_CLEAN_PATH)
    # Same US + India merge as the 2019 model, so the India fallback path is unchanged.
    build_unified_flight_dataset.run(us_path=BTS_CLEAN_PATH, out_path=BTS_UNIFIED_CLEAN_PATH,
                                     congestion_path=BTS_CONGESTION_LOOKUP_PATH)


if __name__ == "__main__":
    main()
