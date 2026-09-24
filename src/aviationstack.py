"""
AviationStack: real schedules, and real delay minutes before a flight departs.

This fills the hole OpenSky structurally cannot. Receiver networks report what
has already flown, so nothing in opensky.py can say which airframe is due to
operate a future flight - every live prediction therefore sends
prev_leg_known=0, the model's strongest feature unavailable. AviationStack
publishes the schedule itself, and against it, an estimated departure. A flight
that has not left yet already carries a delay figure.

Three things are available on the free plan and used here:

  scheduled_departures()  today's not-yet-departed flights from an airport,
                          each with departure.delay in minutes
  future_schedule()       real flight numbers and times for a date up to
                          several days out (flightsFuture)
  flight_now()            one flight's live status, delay and icao24

Not available on the free plan: historical flight_date, airlines, routes. Do
not add callers for those - they return function_access_restricted.

THE QUOTA IS THE DESIGN CONSTRAINT. The free tier allows roughly 100 calls a
MONTH, not a minute - about three a day. That is far too little to call on a
page view, so two guards sit in front of every request:

  a disk cache   keyed by endpoint and parameters, with a TTL per endpoint.
                 A schedule for next Tuesday does not change hourly.
  a budget file  counts calls in the current month and refuses past the cap,
                 so the app degrades to "no live data" instead of silently
                 failing once the allowance is gone.

Everything returns None or an empty list rather than raising, so every caller
keeps working when the key is missing, the quota is spent, or the API is down.
"""
from __future__ import annotations

import json
import os
import time
from datetime import date, datetime
from pathlib import Path

import httpx
from dotenv import load_dotenv

try:
    import truststore
    truststore.inject_into_ssl()
except ImportError:
    pass

load_dotenv()

API = "https://api.aviationstack.com/v1"
_DATA = Path(__file__).resolve().parent.parent / "data"
_CACHE_DIR = _DATA / "cache" / "aviationstack"
_USAGE_FILE = _DATA / "aviationstack_usage.json"

# Deliberately below the plan's 100 so a miscount cannot silently exhaust it and
# leave nothing for the daily collector, which matters more than any page view.
MONTHLY_CALL_CAP = int(os.environ.get("AVIATIONSTACK_MONTHLY_CAP", "90"))

# How long each endpoint's answer stays useful. A future timetable is stable for
# days; a live delay board is worth minutes.
_TTL_S = {
    "flightsFuture": 7 * 24 * 3600,
    "timetable": 3600,
    "flights": 900,
}

# Separate from the monthly allowance: the free plan also rejects calls made
# too close together with a 429, which is how a batch job that loops over
# airports loses most of its results. Requests are spaced, and a 429 is waited
# out once rather than counted as a failure.
_MIN_INTERVAL_S = 1.5
_last_call = 0.0
# When the plan says nothing is left in the current window it also says when
# that window resets. Calling before then only burns monthly quota on a
# guaranteed 429, so requests are held until it passes.
_blocked_until = 0.0


def _key() -> str | None:
    return os.environ.get("AVIATIONSTACK_API_KEY") or None


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
        print(f"[aviationstack] could not record usage: {e}")


def usage() -> dict:
    """Calls spent this month, for the UI and for the collector to check."""
    u = _load_usage()
    local_left = max(0, MONTHLY_CALL_CAP - u["calls"])
    return {"month": u["month"], "calls": u["calls"], "cap": MONTHLY_CALL_CAP,
            "remaining": local_left,
            "plan_quota": u.get("quota_limit"),
            "reported_remaining": u.get("reported_remaining")}


def _cache_path(endpoint: str, params: dict) -> Path:
    stamp = "_".join(f"{k}-{v}" for k, v in sorted(params.items()) if k != "access_key")
    safe = "".join(c if c.isalnum() or c in "-_" else "." for c in stamp)[:120]
    return _CACHE_DIR / f"{endpoint}__{safe}.json"


def _get(endpoint: str, params: dict, ttl: float | None = None) -> dict | None:
    """One API call, served from disk when possible and refused past the cap."""
    ttl = _TTL_S.get(endpoint, 900) if ttl is None else ttl
    path = _cache_path(endpoint, params)
    if path.exists() and (time.time() - path.stat().st_mtime) < ttl:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass

    api_key = _key()
    if not api_key:
        return None

    u = _load_usage()
    if u["calls"] >= MONTHLY_CALL_CAP:
        print(f"[aviationstack] monthly cap reached ({u['calls']}/{MONTHLY_CALL_CAP}) "
              f"- serving nothing until {u['month']} rolls over")
        # A stale cached answer beats no answer at all once the budget is gone.
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                pass
        return None

    global _last_call, _blocked_until
    body = None
    for attempt in (1, 2):
        wait = _blocked_until - time.time()
        if wait > 0:
            if wait > 75:
                print(f"[aviationstack] rate window closed for another {wait:.0f}s - skipping")
                return None
            time.sleep(wait + 1)
        gap = time.time() - _last_call
        if gap < _MIN_INTERVAL_S:
            time.sleep(_MIN_INTERVAL_S - gap)
        try:
            r = httpx.get(f"{API}/{endpoint}",
                          params={"access_key": api_key, **params}, timeout=30)
            _last_call = time.time()
            u["calls"] += 1
            # The response states the real position - x-quota-limit is the
            # month's allowance, x-rate-limit-remaining what is left. Prefer
            # those over a local tally, which drifts whenever a call is made
            # from anywhere else (a script, a second process, the dashboard).
            try:
                if "x-quota-limit" in r.headers:
                    u["quota_limit"] = int(r.headers["x-quota-limit"])
                if "x-rate-limit-remaining" in r.headers:
                    u["reported_remaining"] = int(r.headers["x-rate-limit-remaining"])
            except (TypeError, ValueError):
                pass
            _save_usage(u)
            try:
                if int(r.headers.get("x-rate-limit-remaining", "1")) <= 0:
                    _blocked_until = float(r.headers.get("x-rate-limit-reset", 0))
            except (TypeError, ValueError):
                pass
            if r.status_code == 429 and attempt == 1:
                wait = max(5.0, min(_blocked_until - time.time() + 1, 70.0))
                print(f"[aviationstack] rate limited, waiting {wait:.0f}s")
                time.sleep(wait)
                continue
            r.raise_for_status()
            body = r.json()
            break
        except Exception as e:
            print(f"[aviationstack] {endpoint} failed: {_redact(e)}")
            return None
    if body is None:
        return None

    if isinstance(body, dict) and "error" in body:
        err = body["error"]
        print(f"[aviationstack] {endpoint} refused: {err.get('code')} {err.get('message')}")
        return None

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(body), encoding="utf-8")
    except Exception:
        pass
    return body


def _redact(e: object) -> str:
    """httpx puts the failing URL in its message, and the key is a query param.

    An exception printed to a log or a console is exactly how a key leaks, so
    it is replaced here rather than trusted not to appear.
    """
    text = str(e)
    api_key = _key()
    return text.replace(api_key, "***") if api_key else text


def _clean(v):
    return v.strip() if isinstance(v, str) else v


def scheduled_departures(origin_iata: str, limit: int = 100) -> list[dict]:
    """Flights leaving this airport today that have NOT departed yet.

    `delay` here is the airline's own estimate against its own schedule, which
    is why this is worth having: it exists before the aircraft moves.
    """
    body = _get("flights", {"dep_iata": (origin_iata or "").upper(),
                            "flight_status": "scheduled", "limit": limit})
    out = []
    for f in (body or {}).get("data", []):
        dep, arr = f.get("departure") or {}, f.get("arrival") or {}
        fl = f.get("flight") or {}
        out.append({
            "flight_iata": _clean(fl.get("iata")),
            "flight_number": _clean(fl.get("number")),
            "carrier_iata": _clean((f.get("airline") or {}).get("iata")),
            "origin": _clean(dep.get("iata")),
            "destination": _clean(arr.get("iata")),
            "scheduled_departure": dep.get("scheduled"),
            "estimated_departure": dep.get("estimated"),
            "departure_delay_min": dep.get("delay"),
            "arrival_delay_min": arr.get("delay"),
            "status": f.get("flight_status"),
            "icao24": ((f.get("aircraft") or {}).get("icao24") or "").lower() or None,
        })
    return out


def flight_now(flight_iata: str) -> dict | None:
    """Live record for one flight designator (e.g. "6E6376"), or None.

    Codeshares mean one designator can return several rows for the same metal;
    the operating record is preferred, since its delay is the real one.
    """
    body = _get("flights", {"flight_iata": (flight_iata or "").replace(" ", "").upper(),
                            "limit": 5})
    rows = (body or {}).get("data") or []
    if not rows:
        return None
    rows.sort(key=lambda f: 1 if (f.get("flight") or {}).get("codeshared") else 0)
    f = rows[0]
    dep, arr = f.get("departure") or {}, f.get("arrival") or {}
    return {
        "flight_iata": _clean((f.get("flight") or {}).get("iata")),
        "status": f.get("flight_status"),
        "origin": _clean(dep.get("iata")),
        "destination": _clean(arr.get("iata")),
        "scheduled_departure": dep.get("scheduled"),
        "estimated_departure": dep.get("estimated"),
        "actual_departure": dep.get("actual"),
        "departure_delay_min": dep.get("delay"),
        "arrival_delay_min": arr.get("delay"),
        "terminal": _clean(dep.get("terminal")),
        "gate": _clean(dep.get("gate")),
        "icao24": ((f.get("aircraft") or {}).get("icao24") or "").lower() or None,
        "operated_by": _clean((f.get("airline") or {}).get("name")),
    }


def future_schedule(origin_iata: str, on: str | date, limit: int = 100,
                    offset: int = 0) -> tuple[list[dict], int]:
    """Real scheduled departures for a future date (flightsFuture).

    No delay figures - the day has not happened - but real flight numbers,
    times and aircraft types, which is what a plan booked ahead needs and what
    a generated catalogue can only invent.

    Returns (rows, total). A page is 100 rows and a big airport has well over a
    thousand, ordered by departure time - so reading only the first pages gets
    a night's worth of flights rather than a day's. Callers wanting a spread
    step `offset` across `total` instead of paging from the start.
    """
    when = on.isoformat() if isinstance(on, date) else str(on)
    # No `limit` here: flightsFuture 400s on it, unlike the flights endpoint.
    # The whole day comes back and is trimmed locally instead.
    # Uppercase, unlike the flights endpoint's dep_iata: flightsFuture rejects a
    # lowercase code with a validation error that surfaces as a bare 400.
    params = {"iataCode": (origin_iata or "").upper(), "type": "departure", "date": when}
    if offset:
        params["offset"] = offset
    body = _get("flightsFuture", params)
    total = ((body or {}).get("pagination") or {}).get("total") or 0
    out = []
    for f in (body or {}).get("data", [])[:limit]:
        dep, arr = f.get("departure") or {}, f.get("arrival") or {}
        fl, al = f.get("flight") or {}, f.get("airline") or {}
        out.append({
            "flight_iata": (_clean(fl.get("iataNumber")) or "").upper() or None,
            "flight_number": _clean(fl.get("number")),
            "carrier_iata": (_clean(al.get("iataCode")) or "").upper() or None,
            "carrier_name": _clean(al.get("name")),
            "origin": (_clean(dep.get("iataCode")) or "").upper() or None,
            "destination": (_clean(arr.get("iataCode")) or "").upper() or None,
            "scheduled_departure": _clean(dep.get("scheduledTime")),
            "scheduled_arrival": _clean(arr.get("scheduledTime")),
            "terminal": _clean(dep.get("terminal")) or None,
            "aircraft_model": _clean((f.get("aircraft") or {}).get("modelText")),
            "weekday": f.get("weekday"),
        })
    return out, total


if __name__ == "__main__":
    import io
    import sys
    from datetime import timedelta
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

    print("usage:", usage())
    if not _key():
        print("AVIATIONSTACK_API_KEY not set - nothing to test")
        sys.exit(0)

    deps = scheduled_departures("DEL", limit=20)
    print(f"\n{len(deps)} DEL flights scheduled but not yet departed")
    for d in deps[:6]:
        print(f"   {d['flight_iata'] or '?':9s} -> {d['destination'] or '?':4s} "
              f"sched {str(d['scheduled_departure'])[11:16]}  delay={d['departure_delay_min']}")

    fut, total = future_schedule("DEL", date.today() + timedelta(days=7), limit=20)
    print(f"\n{len(fut)} of {total} DEL departures scheduled 7 days out")
    for f in fut[:6]:
        print(f"   {f['flight_iata'] or '?':9s} -> {f['destination'] or '?':4s} "
              f"{str(f['scheduled_departure'])[11:16]}  {f['aircraft_model']}")

    print("\nusage after:", usage())
