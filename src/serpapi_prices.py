"""
Real flight and hotel PRICES, via SerpApi's google_flights / google_hotels
engines - verified directly against the live API before this was written
(real $219 JFK->LAX fare, real $154/night LA hotel with rating 4.1/3424
reviews, real photos). This is the one source in this project that can
answer "how much", which is why every other flight/hotel card elsewhere
says "no fare data available" - nothing else here ever could.

SerpApi does not sell flight/hotel data directly; it returns Google's own
Flights/Hotels results, which is why real prices, ratings and photos are
available at all without a travel-industry contract.

THE QUOTA IS THE DESIGN CONSTRAINT, same principle as aviationstack.py and
copying its exact mechanism: a disk cache (keyed by engine + real search
params) and a hard monthly cap, because the free plan allows only 250
searches/month and that allowance is SHARED across every SerpApi engine on
the key, not just these two. Everything here returns None/[] rather than
raising when the key is missing, the cap is hit, or the call fails - callers
fall back to the existing "no fare data" language, never a guess.
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

API = "https://serpapi.com/search.json"
_DATA = Path(__file__).resolve().parent.parent / "data"
_CACHE_DIR = _DATA / "cache" / "serpapi"
_USAGE_FILE = _DATA / "serpapi_usage.json"

# Deliberately well below the real 250/month cap - this key's quota is shared
# with every other SerpApi engine anyone on this key ever calls, not just these
# two, so leaving headroom matters more than squeezing out the last few calls.
MONTHLY_CALL_CAP = int(os.environ.get("SERPAPI_MONTHLY_CAP", "200"))

# A trip is usually planned days ahead of being re-checked, and a price that's
# a few hours stale is still a real, useful price - cached generously to make
# the limited quota stretch across a classroom's worth of testing.
_TTL_S = 6 * 3600


def _key() -> str | None:
    return os.environ.get("SERPAPI_KEY") or None


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
        print(f"[serpapi_prices] could not record usage: {e}")


def usage() -> dict:
    u = _load_usage()
    return {"month": u["month"], "calls": u["calls"], "cap": MONTHLY_CALL_CAP,
            "remaining": max(0, MONTHLY_CALL_CAP - u["calls"])}


_LOCK_FILE = _DATA / "serpapi_usage.lock"


@contextmanager
def _usage_lock(timeout_s: float = 30.0):
    """Exclusive-create spinlock around the whole check-call-increment sequence.

    Without this, two requests landing on two different gunicorn workers (the
    Dockerfile runs several by default) can both read calls=199 before either
    writes back, both pass the cap check, and both spend a real paid call -
    silently letting MONTHLY_CALL_CAP be exceeded under any real concurrency.
    os.O_EXCL is atomic on both Windows and POSIX, so no extra dependency.
    A lock older than the timeout is assumed abandoned (a crashed holder)
    and stolen rather than deadlocking the app forever.
    """
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
                print("[serpapi_prices] usage lock held too long - proceeding unlocked "
                      "(worse case: one over-cap call, not a crash)")
                break
            time.sleep(0.05)
    try:
        yield
    finally:
        if fd is not None:
            os.close(fd)
        _LOCK_FILE.unlink(missing_ok=True)


def _cache_path(engine: str, params: dict) -> Path:
    stamp = "_".join(f"{k}-{v}" for k, v in sorted(params.items()) if k != "api_key")
    safe = "".join(c if c.isalnum() or c in "-_" else "." for c in stamp)[:150]
    return _CACHE_DIR / f"{engine}__{safe}.json"


def _get(engine: str, params: dict) -> dict | None:
    """One real SerpApi search, served from disk when possible, refused past
    the monthly cap (a stale cached answer is still served if one exists)."""
    path = _cache_path(engine, params)
    if path.exists() and (time.time() - path.stat().st_mtime) < _TTL_S:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass

    api_key = _key()
    if not api_key:
        return None

    # Locked for the whole check-call-increment sequence, not just the file
    # writes - see _usage_lock's docstring for why the check alone isn't safe.
    with _usage_lock():
        u = _load_usage()
        if u["calls"] >= MONTHLY_CALL_CAP:
            print(f"[serpapi_prices] monthly cap reached ({u['calls']}/{MONTHLY_CALL_CAP}) "
                  f"- serving nothing new until {u['month']} rolls over")
            if path.exists():
                try:
                    return json.loads(path.read_text(encoding="utf-8"))
                except Exception:
                    pass
            return None

        try:
            r = httpx.get(API, params={"engine": engine, **params, "api_key": api_key}, timeout=20)
            r.raise_for_status()
            body = r.json()
        except Exception as e:
            print(f"[serpapi_prices] {engine} call failed: {e}")
            return None

        if body.get("error"):
            print(f"[serpapi_prices] {engine} returned an error: {body['error']}")
            return None

        u["calls"] += 1
        _save_usage(u)
    try:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(body), encoding="utf-8")
    except Exception as e:
        print(f"[serpapi_prices] could not cache response: {e}")
    return body


def real_flight_price(origin_iata: str, destination_iata: str, outbound_date: str,
                      return_date: str | None = None, adults: int = 1) -> dict | None:
    """Real Google Flights price for this route/date, or None if unavailable.

    `type=2` (one way) when no return_date, `type=1` (round trip) otherwise -
    SerpApi requires return_date for a round trip and rejects it for one-way.
    Picks the cheapest of best_flights/other_flights, which both carry real
    Google Flights results - "best" is Google's own ranking, not just price.
    """
    params = {
        "departure_id": (origin_iata or "").upper(), "arrival_id": (destination_iata or "").upper(),
        "outbound_date": outbound_date, "currency": "USD", "adults": max(1, adults),
        "hl": "en", "gl": "us",
        "type": "1" if return_date else "2",
    }
    if return_date:
        params["return_date"] = return_date
    body = _get("google_flights", params)
    if not body:
        return None
    options = (body.get("best_flights") or []) + (body.get("other_flights") or [])
    if not options:
        return None
    cheapest = min(options, key=lambda o: o.get("price") or float("inf"))
    if cheapest.get("price") is None:
        return None
    legs = cheapest.get("flights") or [{}]
    first_leg, last_leg = legs[0], legs[-1]
    # Real scheduled clock times for the WHOLE journey - first leg's departure,
    # last leg's arrival - not a layover's intermediate times. Both already
    # come back as real local "YYYY-MM-DD HH:MM" strings, nothing computed.
    return {
        "price_usd": cheapest["price"],
        "airline": first_leg.get("airline"),
        "flight_number": first_leg.get("flight_number"),
        "duration_min": cheapest.get("total_duration"),
        "stops": max(0, len(legs) - 1),
        "departure_time": (first_leg.get("departure_airport") or {}).get("time"),
        "arrival_time": (last_leg.get("arrival_airport") or {}).get("time"),
        "booking_token": cheapest.get("booking_token"),
        "source": "serpapi_google_flights",
    }


def real_hotel_prices(city: str, check_in: str, check_out: str,
                      adults: int = 1, limit: int = 5) -> list[dict]:
    """Real Google Hotels results near `city` for these real dates - price,
    rating, review count, star class and a real photo, when Google has them.
    Empty list (never a guess) if the key/quota/lookup isn't available."""
    if not city:
        return []
    body = _get("google_hotels", {
        "q": city, "check_in_date": check_in, "check_out_date": check_out,
        "adults": max(1, adults), "currency": "USD", "hl": "en", "gl": "us",
    })
    if not body:
        return []
    out = []
    for p in (body.get("properties") or [])[:limit]:
        rate = p.get("rate_per_night") or {}
        total = p.get("total_rate") or {}
        if rate.get("extracted_lowest") is None:
            continue
        images = p.get("images") or []
        out.append({
            "name": p.get("name"),
            "nightly_rate_usd": rate["extracted_lowest"],
            "total_rate_usd": total.get("extracted_lowest"),
            "rating": p.get("overall_rating"),
            "reviews": p.get("reviews"),
            "hotel_class": p.get("hotel_class"),
            "amenities": (p.get("amenities") or [])[:4],
            "thumbnail": images[0]["thumbnail"] if images else None,
            "link": p.get("link"),
            "lat": (p.get("gps_coordinates") or {}).get("latitude"),
            "lon": (p.get("gps_coordinates") or {}).get("longitude"),
            "source": "serpapi_google_hotels",
        })
    return out


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    print("usage:", usage())
    f = real_flight_price("JFK", "LAX", "2026-12-10")
    print("flight:", json.dumps(f, indent=2) if f else "none")
    hs = real_hotel_prices("Los Angeles", "2026-12-10", "2026-12-13")
    print(f"hotels: {len(hs)}")
    for h in hs[:3]:
        print(f"  {h['name']} - ${h['nightly_rate_usd']}/night, {h['rating']} ({h['reviews']} reviews)")
    print("usage after:", usage())
