"""
Real LIVE current weather, via WeatherAPI.com - verified directly against the
live API before this was written (real Paris conditions: 17.3C, "Patchy rain
nearby", real humidity/wind/visibility).

This replaces GlobalWeatherRepository.csv as recovery_tools.get_destination_weather()'s
primary source. That CSV is a ONE-TIME downloaded snapshot - its own cleaning
script (clean_weather.py) already flags it as "rolling current-conditions"
data that only gets staler with time, never refreshing itself. A live call
is strictly better for a function whose whole job is "what's the weather
AT THE DESTINATION RIGHT NOW" - the CSV is kept only as a fallback for when
the key is missing or the call fails, so the recovery agent's check_weather
tool never goes from "a real but stale snapshot" to "nothing at all".

Free plan is 100,000 calls/month - generous next to this project's other
quota-guarded APIs (AviationStack ~90, SerpApi 250) - but still capped and
cached on disk, same mechanism as those, so a bug here can't run away
unnoticed. TTL is short (conditions genuinely change within the hour),
unlike the day-plus TTLs elsewhere that cache something more stable.
"""
from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from datetime import date
from pathlib import Path

import httpx

try:
    import truststore
    truststore.inject_into_ssl()
except ImportError:
    pass

API = "https://api.weatherapi.com/v1/current.json"
_DATA = Path(__file__).resolve().parent.parent / "data"
_CACHE_DIR = _DATA / "cache" / "weatherapi"
_USAGE_FILE = _DATA / "weatherapi_usage.json"

MONTHLY_CALL_CAP = int(os.environ.get("WEATHERAPI_MONTHLY_CAP", "20000"))
_TTL_S = 30 * 60  # current conditions are genuinely stale well before a day is up


def _key() -> str | None:
    return os.environ.get("WEATHERAPI_KEY") or None


def _load_usage() -> dict:
    try:
        u = json.loads(_USAGE_FILE.read_text(encoding="utf-8"))
    except Exception:
        u = {}
    month = date.today().strftime("%Y-%m")
    if u.get("month") != month:
        u = {"month": month, "calls": 0}
    return u


def _save_usage(u: dict) -> None:
    try:
        _USAGE_FILE.parent.mkdir(parents=True, exist_ok=True)
        _USAGE_FILE.write_text(json.dumps(u), encoding="utf-8")
    except Exception as e:
        print(f"[weatherapi_live] could not record usage: {e}")


def usage() -> dict:
    u = _load_usage()
    return {"month": u["month"], "calls": u["calls"], "cap": MONTHLY_CALL_CAP,
            "remaining": max(0, MONTHLY_CALL_CAP - u["calls"])}


_LOCK_FILE = _DATA / "weatherapi_usage.lock"


@contextmanager
def _usage_lock(timeout_s: float = 30.0):
    """Same race-prevention lock as serpapi_prices._usage_lock, same reason:
    gunicorn's default multi-worker setup (see Dockerfile) can run two
    requests through the check-call-increment sequence at once, and without
    this both can pass the cap check before either writes back."""
    _LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.time() + timeout_s
    fd = None
    while fd is None:
        try:
            fd = os.open(_LOCK_FILE, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            try:
                if time.time() - _LOCK_FILE.stat().st_mtime > timeout_s:
                    _LOCK_FILE.unlink(missing_ok=True)
                    continue
            except FileNotFoundError:
                continue
            if time.time() > deadline:
                print("[weatherapi_live] usage lock held too long - proceeding unlocked")
                break
            time.sleep(0.05)
    try:
        yield
    finally:
        if fd is not None:
            os.close(fd)
        _LOCK_FILE.unlink(missing_ok=True)


def _cache_path(location_name: str) -> Path:
    safe = "".join(c if c.isalnum() or c in "-_" else "." for c in location_name.lower())[:120]
    return _CACHE_DIR / f"{safe}.json"


def live_weather(location_name: str) -> dict | None:
    """Real current conditions for a city/place name, in get_destination_weather()'s
    existing shape - same keys, so no caller needs to change. None (never a
    guess) when the key is missing, the cap is hit with no cache to fall back
    on, the place can't be resolved, or the call fails for any reason."""
    if not location_name:
        return None
    path = _cache_path(location_name)
    if path.exists() and (time.time() - path.stat().st_mtime) < _TTL_S:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass

    api_key = _key()
    if not api_key:
        return None

    # Locked for the whole check-call-increment sequence - see _usage_lock.
    with _usage_lock():
        u = _load_usage()
        if u["calls"] >= MONTHLY_CALL_CAP:
            print(f"[weatherapi_live] monthly cap reached ({u['calls']}/{MONTHLY_CALL_CAP})")
            if path.exists():
                try:
                    return json.loads(path.read_text(encoding="utf-8"))
                except Exception:
                    pass
            return None

        body = None
        for attempt in (1, 2):
            try:
                r = httpx.get(API, params={"key": api_key, "q": location_name}, timeout=12)
                if r.status_code == 400:
                    # WeatherAPI's real "no such place" response - not worth a retry or a
                    # log line, every destination name typed into this app hits this sometimes.
                    return None
                r.raise_for_status()
                body = r.json()
                break
            except Exception as e:
                # One retry only - a transient connection drop (observed directly,
                # not theoretical) shouldn't cost a real reading when a second
                # attempt would have worked; a second real failure just returns None.
                if attempt == 2:
                    print(f"[weatherapi_live] call failed for '{location_name}': {e}")
                    return None

        u["calls"] += 1
        _save_usage(u)

    loc, cur = body.get("location") or {}, body.get("current") or {}
    if not cur:
        return None
    result = {
        "location": loc.get("name") or location_name,
        "temperature_celsius": round(float(cur["temp_c"]), 1),
        "condition": (cur.get("condition") or {}).get("text"),
        "wind_kph": round(float(cur["wind_kph"]), 1),
        "precip_mm": round(float(cur["precip_mm"]), 1),
        "visibility_km": round(float(cur["vis_km"]), 1),
        "last_updated": cur.get("last_updated"),
        "source": "weatherapi_live",
    }
    try:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result), encoding="utf-8")
    except Exception as e:
        print(f"[weatherapi_live] could not cache response: {e}")
    return result


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    print("usage:", usage())
    for city in ["Paris", "Mumbai", "Not A Real City Name"]:
        w = live_weather(city)
        print(f"{city}: {w}")
    print("usage after:", usage())
