"""
Ignav (ignav.com): real flight fares, per itinerary, for every flight price in the app.

    POST https://ignav.com/api/fares/one-way   (X-Api-Key header)

Every itinerary carries a real price together with the flight it belongs to -
airline, flight number(s), local departure/arrival times, stops, duration - so a
traveller can pick a specific flight and the trip cost follows that pick. That is
what SerpApi's cheapest-of-the-day number could not do.

PRICES ARE FOR THE WHOLE PARTY. `price.amount` is the total for the passengers in
the request (Ignav docs; also checked: 3 adults = $325 where 1 adult = $103 on the
same route and day). Callers must never multiply it by the traveller count again;
`price_per_person_usd` is provided for display.

Requests use market "US", so amounts come back in USD - the currency the rest of
the app holds money in (src/currency.py converts for display). An itinerary in
any other currency is dropped rather than converted at a guessed rate.

Quota: 1,000 free requests, then paid per request. A disk cache (default 6h) and a
monthly cap (IGNAV_MONTHLY_CAP) sit in front of every call, the same guards as
aviationstack.py and serpapi_prices.py. If Ignav is unavailable, the public
helpers fall back to SerpApi (serpapi_prices.py) and say so in `source`.
"""
from __future__ import annotations

import json
import os
import threading
import time
from datetime import date
from pathlib import Path

import httpx

try:
    import truststore
    truststore.inject_into_ssl()
except ImportError:
    pass

API = "https://ignav.com/api"
_DATA = Path(__file__).resolve().parent.parent / "data"
_CACHE_DIR = _DATA / "cache" / "ignav"
_USAGE_FILE = _DATA / "ignav_usage.json"
MONTHLY_CALL_CAP = int(os.environ.get("IGNAV_MONTHLY_CAP", "900"))
CACHE_TTL_S = float(os.environ.get("IGNAV_CACHE_TTL_S", str(6 * 3600)))
SOURCE_LABEL = "Ignav"
_lock = threading.Lock()


def _key() -> str | None:
    return os.environ.get("IGNAV_API_KEY") or None


def _usage() -> dict:
    try:
        u = json.loads(_USAGE_FILE.read_text(encoding="utf-8"))
    except Exception:
        u = {}
    month = date.today().strftime("%Y-%m")
    return u if u.get("month") == month else {"month": month, "calls": 0}


def usage() -> dict:
    u = _usage()
    return {**u, "cap": MONTHLY_CALL_CAP, "remaining": max(0, MONTHLY_CALL_CAP - u["calls"])}


def _cache_path(path: str, body: dict) -> Path:
    stamp = "_".join(f"{k}-{v}" for k, v in sorted(body.items()))
    safe = "".join(c if c.isalnum() or c in "-_" else "." for c in stamp)[:140]
    return _CACHE_DIR / f"{path.strip('/').replace('/', '.')}__{safe}.json"


def _post(path: str, body: dict) -> dict | None:
    """One Ignav call: disk cache first, refused past the monthly cap, never raises."""
    cpath = _cache_path(path, body)
    if cpath.exists() and time.time() - cpath.stat().st_mtime < CACHE_TTL_S:
        try:
            return json.loads(cpath.read_text(encoding="utf-8"))
        except Exception:
            pass
    key = _key()
    if not key:
        return None
    with _lock:
        u = _usage()
        if u["calls"] >= MONTHLY_CALL_CAP:
            print(f"[ignav] monthly cap reached ({u['calls']}/{MONTHLY_CALL_CAP})")
            return None
        try:
            r = httpx.post(f"{API}/{path.lstrip('/')}", json=body,
                           headers={"X-Api-Key": key}, timeout=45)
            u["calls"] += 1
            try:
                _USAGE_FILE.parent.mkdir(parents=True, exist_ok=True)
                _USAGE_FILE.write_text(json.dumps(u), encoding="utf-8")
            except Exception:
                pass
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            # The key travels in a header, but never let it reach a log either way.
            print(f"[ignav] {path} failed: {str(e).replace(key, '***')}")
            return None
    try:
        cpath.parent.mkdir(parents=True, exist_ok=True)
        cpath.write_text(json.dumps(data), encoding="utf-8")
    except Exception:
        pass
    return data


def _hhmm(ts: str | None) -> str | None:
    """'2026-10-22T23:45:00' -> '2026-10-22 23:45' (the format the templates slice)."""
    return ts[:16].replace("T", " ") if ts and len(ts) >= 16 else None


def _normalise(it: dict, adults: int) -> dict | None:
    price = it.get("price") or {}
    if price.get("amount") is None or (price.get("currency") or "USD") != "USD":
        return None
    out = it.get("outbound") or {}
    segs = out.get("segments") or []
    if not segs:
        return None
    first, last = segs[0], segs[-1]
    numbers = [f"{s.get('marketing_carrier_code') or ''}{s.get('flight_number') or ''}" for s in segs]
    total = float(price["amount"])
    return {
        "id": it.get("ignav_id"),
        "price_usd": total,                                  # whole party
        "price_per_person_usd": round(total / max(adults, 1), 2),
        "adults": adults,
        "price_status": price.get("status"),
        "airline": out.get("carrier") or first.get("operating_carrier_name"),
        "carrier_code": first.get("marketing_carrier_code"),
        "flight_number": numbers[0],
        "flight_numbers": numbers,
        "departure_time": _hhmm(first.get("departure_time_local")),
        "arrival_time": _hhmm(last.get("arrival_time_local")),
        "duration_min": out.get("duration_minutes"),
        "stops": max(0, len(segs) - 1),
        "via": [s.get("arrival_airport") for s in segs[:-1]],
        "aircraft": first.get("aircraft"),
        "self_transfer": bool(it.get("requires_self_transfer")),
        "cabin": it.get("cabin_class"),
        "source": "ignav",
        "source_label": SOURCE_LABEL,
    }


def one_way_options(origin: str, destination: str, on: str, adults: int = 1,
                    limit: int = 10) -> list[dict] | None:
    """Real priced itineraries for one leg, cheapest first. None if Ignav could not
    be asked (no key, cap, error); [] if it answered with no flights."""
    if not (origin and destination and on):
        return None
    adults = max(1, int(adults or 1))
    body = _post("fares/one-way", {
        "origin": origin.upper(), "destination": destination.upper(),
        "departure_date": str(on)[:10], "adults": adults,
        "cabin_class": "economy", "market": "US",
    })
    if body is None:
        return None
    rows = [r for r in (_normalise(it, adults) for it in body.get("itineraries") or []) if r]
    rows.sort(key=lambda r: (r["price_usd"], r["duration_min"] or 0))
    return rows[:limit]


def _serpapi_fallback(origin, destination, on, adults):
    try:
        from serpapi_prices import real_flight_price as serp
        fare = serp(origin, destination, on, adults=adults)
    except Exception as e:
        print(f"[ignav] SerpApi fallback skipped: {e}")
        return None
    if fare:
        fare = {**fare, "source_label": "Google Flights via SerpApi (Ignav unavailable)",
                "price_per_person_usd": round(fare["price_usd"] / max(adults, 1), 2), "adults": adults}
    return fare


def real_flight_price(origin_iata: str, destination_iata: str, outbound_date: str,
                      return_date: str | None = None, adults: int = 1) -> dict | None:
    """Cheapest real fare for this leg and date (whole party), same shape the app
    already used: price_usd, airline, flight_number, stops, departure_time,
    arrival_time - plus source_label. return_date is accepted for compatibility;
    each leg is priced as its own one-way search."""
    options = one_way_options(origin_iata, destination_iata, outbound_date, adults, limit=1)
    if options:
        return options[0]
    if options is None:
        return _serpapi_fallback(origin_iata, destination_iata, outbound_date, adults)
    return None


def price_for_flight(origin_iata: str, destination_iata: str, on: str,
                     flight_iata: str, adults: int = 1) -> dict | None:
    """The fare of one specific flight (e.g. '6E2369') if Ignav lists it that day."""
    wanted = (flight_iata or "").replace(" ", "").upper()
    for opt in one_way_options(origin_iata, destination_iata, on, adults, limit=100) or []:
        if wanted and wanted in [n.upper() for n in opt["flight_numbers"]]:
            return {**opt, "same_flight": True}
    return None
