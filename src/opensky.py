"""
Live aircraft movements from the OpenSky Network.

This exists to fill the one input the trained model wants and the web form
cannot supply: how late the inbound aircraft was. That feature is worth about
+0.19 AUC on held-out data (0.71 -> 0.90), but every live prediction currently
sends prev_leg_known=0 because nothing knows where the aircraft has been.

OpenSky records REAL OBSERVED movements from volunteer ADS-B receivers -
icao24 (the aircraft), callsign, and firstSeen/lastSeen timestamps for each
leg. What it does NOT have is schedules, so "delay" is only computable by
subtracting an observed time from a scheduled one the caller supplies.

Two things are therefore offered:

Because identification depends on having already seen the flight depart, all of
this describes flights that have flown - useful for reconciling a past
prediction against what happened, not for scoring one that has not left yet.

  prior_leg(...)       the aircraft's previous leg and when it actually landed.
                       With a scheduled arrival, that becomes real delay minutes.
  turnaround_min(...)  the observed ground time between that landing and the
                       next departure. Needs no schedule at all, and is the
                       mechanism the delay actually propagates through - a
                       compressed turnaround is how a late inbound becomes a
                       late outbound.

Everything degrades to None rather than raising: no credentials, no coverage,
or a rate limit all mean "we could not observe this", and the caller keeps its
prev_leg_known=0 path.
"""
from __future__ import annotations

import os
import sys
import time

import httpx
from dotenv import load_dotenv

try:
    import truststore
    truststore.inject_into_ssl()
except ImportError:
    pass

load_dotenv()
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

TOKEN_URL = ("https://auth.opensky-network.org/auth/realms/opensky-network"
             "/protocol/openid-connect/token")
API = "https://opensky-network.org/api"

# OpenSky keys airports by ICAO; the rest of this project speaks IATA.
IATA_TO_ICAO = {
    "DEL": "VIDP", "BOM": "VABB", "BLR": "VOBL", "MAA": "VOMM", "HYD": "VOHS",
    "CCU": "VECC", "GOI": "VOGO", "PNQ": "VAPO", "AMD": "VAAH", "JAI": "VIJP",
    "COK": "VOCI", "TRV": "VOTV", "LKO": "VILK", "PAT": "VEPT", "NAG": "VANP",
    # US majors, for the trained-data routes
    "ATL": "KATL", "ORD": "KORD", "DFW": "KDFW", "DEN": "KDEN", "LAX": "KLAX",
    "JFK": "KJFK", "EWR": "KEWR", "SFO": "KSFO", "SEA": "KSEA", "BOS": "KBOS",
    "LAS": "KLAS", "MCO": "KMCO", "PHX": "KPHX", "MIA": "KMIA", "IAH": "KIAH",
}

# Airlines file flight plans under ICAO codes, not the IATA codes passengers see.
# AI 472 flies as AIC472. Without this the callsign never matches.
IATA_TO_ICAO_AIRLINE = {
    "AI": "AIC", "6E": "IGO", "UK": "VTI", "SG": "SEJ", "QP": "AKJ", "G8": "GOW",
    "IX": "AXB", "9I": "LLR",
    "AA": "AAL", "DL": "DAL", "UA": "UAL", "WN": "SWA", "B6": "JBU", "AS": "ASA",
    "NK": "NKS", "F9": "FFT", "HA": "HAL", "G4": "AAY", "OO": "SKW", "MQ": "ENY",
}

# Typical scheduled ground time in the training data (median 50 min). Used only
# to translate an observed turnaround into "how late was the inbound", and
# reported as an inference, never as a measurement.
TYPICAL_TURNAROUND_MIN = 50.0

# How far either side of a scheduled push the live check is worth attempting.
# Receivers report what has ALREADY flown, so a departure days out has no
# observable aircraft assigned to it yet and the search window around it lies
# entirely in the future - the lookup can only come back empty.
LIVE_WINDOW_H = 6

_CLIENT = httpx.Client(timeout=45)
_token: dict = {"value": None, "expires": 0.0}


def _bearer() -> str | None:
    """Cached OAuth2 token. OpenSky issues 30-minute tokens; refresh a minute early."""
    cid = os.environ.get("OPENSKY_CLIENT_ID")
    sec = os.environ.get("OPENSKY_CLIENT_SECRET")
    if not cid or not sec:
        return None
    if _token["value"] and time.time() < _token["expires"]:
        return _token["value"]
    try:
        r = _CLIENT.post(TOKEN_URL, data={
            "grant_type": "client_credentials", "client_id": cid, "client_secret": sec,
        })
        r.raise_for_status()
        body = r.json()
        _token["value"] = body["access_token"]
        _token["expires"] = time.time() + int(body.get("expires_in", 1800)) - 60
        return _token["value"]
    except Exception as e:
        print(f"[opensky] auth failed: {e}")
        return None


def _get(path: str, params: dict) -> list | None:
    tok = _bearer()
    if not tok:
        return None
    try:
        r = _CLIENT.get(f"{API}{path}", params=params, headers={"Authorization": f"Bearer {tok}"})
        if r.status_code == 404:
            return []            # OpenSky's "nothing observed here", not an error
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"[opensky] {path} failed: {e}")
        return None


def icao_for(iata: str) -> str | None:
    return IATA_TO_ICAO.get((iata or "").upper())


def find_aircraft(callsign: str, origin_iata: str, around_ts: int,
                   window_h: int = 6) -> str | None:
    """The icao24 flying this callsign out of this airport, from real departures.

    Callsigns are the operational identifier ("AIC472"), not the marketed flight
    number ("AI 472"), and OpenSky pads them - so matching is on the stripped
    prefix rather than equality.
    """
    icao = icao_for(origin_iata)
    if not icao:
        return None
    flights = _get("/flights/departure", {
        "airport": icao, "begin": around_ts - window_h * 3600, "end": around_ts + window_h * 3600,
    })
    if not flights:
        return None
    want = (callsign or "").replace(" ", "").upper()
    for f in flights:
        cs = (f.get("callsign") or "").strip().upper()
        if cs and (cs == want or cs.startswith(want)):
            return f.get("icao24")
    return None


def prior_leg(icao24: str, before_ts: int, lookback_h: int = 18) -> dict | None:
    """The leg this aircraft flew immediately before `before_ts`.

    lastSeen is when it was last observed on that leg - effectively the landing,
    which is what a downstream delay propagates from.
    """
    flights = _get("/flights/aircraft", {
        "icao24": (icao24 or "").lower(),
        "begin": before_ts - lookback_h * 3600, "end": before_ts,
    })
    if not flights:
        return None
    earlier = [f for f in flights if f.get("lastSeen") and f["lastSeen"] < before_ts]
    if not earlier:
        return None
    last = max(earlier, key=lambda f: f["lastSeen"])
    return {
        "icao24": last.get("icao24"),
        "callsign": (last.get("callsign") or "").strip(),
        "from": last.get("estDepartureAirport"),
        "to": last.get("estArrivalAirport"),
        "actual_arrival_ts": last["lastSeen"],
        "actual_departure_ts": last.get("firstSeen"),
    }


def turnaround_min(prior: dict, departure_ts: int) -> float | None:
    """Observed ground time between the inbound landing and this departure.

    Schedule-free, which matters because OpenSky has no schedules. A short
    turnaround behind a late inbound is exactly how delay propagates.
    """
    if not prior or not prior.get("actual_arrival_ts"):
        return None
    mins = (departure_ts - prior["actual_arrival_ts"]) / 60.0
    return round(mins, 1) if 0 <= mins <= 1440 else None


def prior_leg_delay_min(prior: dict, scheduled_arrival_ts: int | None) -> float | None:
    """Real minutes late for the inbound leg - needs a scheduled arrival to subtract.

    Returns None without one rather than guessing, since OpenSky cannot supply it.
    """
    if not prior or not scheduled_arrival_ts:
        return None
    return round((prior["actual_arrival_ts"] - scheduled_arrival_ts) / 60.0, 1)


def observed_rotation(callsign: str, origin_iata: str, departure_ts: int) -> dict:
    """Everything observable about this flight's inbound aircraft.

    The shape predict_delay_probability wants: prev_leg_arrival_delay when a
    schedule was available, the turnaround either way, and `observed` so callers
    know whether anything was actually seen.
    """
    icao24 = find_aircraft(callsign, origin_iata, departure_ts)
    if not icao24:
        return {"observed": False, "reason": "aircraft not identified from live departures"}

    prior = prior_leg(icao24, departure_ts)
    if not prior:
        return {"observed": False, "icao24": icao24, "reason": "no earlier leg observed today"}

    return {
        "observed": True,
        "icao24": icao24,
        "prior_leg": prior,
        "turnaround_min": turnaround_min(prior, departure_ts),
    }


def callsign_for(carrier_iata: str, flight_number: str) -> str | None:
    """IATA flight designator -> the ICAO callsign OpenSky files it under."""
    icao = IATA_TO_ICAO_AIRLINE.get((carrier_iata or "").upper())
    num = "".join(ch for ch in str(flight_number or "") if ch.isdigit())
    return f"{icao}{num}" if icao and num else None


# An inbound leg is only the inbound if it landed where this flight departs and
# did so recently. Beyond this, the aircraft is simply somewhere else earlier in
# its day and the gap measures nothing.
MAX_INBOUND_GAP_H = 6


def rotation_from_icao24(icao24: str, origin_iata: str,
                         scheduled_departure_ts: int) -> dict:
    """Inbound-leg view for an airframe someone else has already identified.

    live_rotation() has to find the aircraft by looking up this flight's own
    departure, which means the departure must already have happened - that is
    why it only ever works retrospectively. A schedule source that names the
    airframe removes that step, and this is the half that remains.

    Two guards, both learned from the data rather than assumed:

      the leg must LAND here   the most recent leg is often one that departed
                               this airport, meaning the aircraft is still away
                               and the real inbound has not flown yet.
      it must land RECENTLY    for a flight tonight, the newest observable leg
                               is this morning's, giving a 14-hour "turnaround"
                               that describes nothing.
    """
    if not icao24:
        return {"observed": False, "reason": "no airframe identified for this flight"}

    prior = prior_leg(icao24, scheduled_departure_ts)
    if not prior:
        return {"observed": False, "icao24": icao24,
                "reason": "no earlier leg observed for this aircraft"}

    here = icao_for(origin_iata)
    landed_at = (prior.get("to") or "").upper() or None
    if here and landed_at and landed_at != here:
        return {"observed": False, "icao24": icao24,
                "reason": f"this aircraft's last observed leg ended at {landed_at}, "
                          f"not {origin_iata} - its inbound has not flown yet"}

    gap = turnaround_min(prior, scheduled_departure_ts)
    if gap is None or gap > MAX_INBOUND_GAP_H * 60:
        return {"observed": False, "icao24": icao24,
                "reason": ("the aircraft's last observed leg was too long before "
                           "departure to be the inbound - it has more flying to do first")}

    inferred = round(max(0.0, TYPICAL_TURNAROUND_MIN - gap), 1)
    return {
        "observed": True,
        "icao24": icao24,
        "prior_leg": f"{prior.get('from') or '?'} -> {prior.get('to') or '?'}",
        "inbound_landed_ts": prior["actual_arrival_ts"],
        "ground_time_min": gap,
        "inferred_inbound_delay_min": inferred,
        "basis": ("airframe from the airline schedule, inbound landing observed by "
                  f"receivers; delay inferred against a {TYPICAL_TURNAROUND_MIN:.0f} min "
                  "typical turnaround"),
    }


def live_rotation(carrier_iata: str, flight_number: str, origin_iata: str,
                   scheduled_departure_ts: int) -> dict:
    """The aircraft that operated this flight, and how its inbound leg went.

    RETROSPECTIVE ONLY, and that is a hard limit rather than a gap to close
    here. The aircraft is identified by finding this flight's own departure in
    the receiver feed, so the departure has to have happened: a DEL query
    covering the next six hours returns nothing at all, because no receiver has
    seen those flights yet. OpenSky carries no schedules and no tail
    assignments, so nothing in it can say which airframe is *due* to operate a
    future flight. Callers predicting a future departure will get observed=False
    and should keep their prev_leg_known=0 path.

    Deliberately measures the gap to the SCHEDULED departure, not the actual
    one. Actual departure is unknown before the fact, and using it would leak
    the outcome - a flight that leaves late has a *longer* observed turnaround,
    which would teach exactly the wrong lesson.

    Returns an inferred inbound delay: if an aircraft that normally gets ~50
    minutes on the ground only has 20 left before its scheduled push, roughly
    30 minutes of that has already been eaten by a late inbound.

    Skipped entirely more than LIVE_WINDOW_H ahead of the scheduled departure,
    where the answer is structurally unavailable rather than merely missing.
    """
    cs = callsign_for(carrier_iata, flight_number)
    if not cs:
        return {"observed": False, "reason": "no ICAO callsign known for this carrier"}

    # Asking about a flight that has not been flown towards yet is not a failed
    # observation, it is a question that cannot be answered, and saying "not
    # seen" would misreport it as the former. Skip the two API calls too.
    hours_out = (scheduled_departure_ts - time.time()) / 3600.0
    if hours_out > LIVE_WINDOW_H:
        return {
            "observed": False, "too_early": True, "callsign": cs, "hours_out": round(hours_out, 1),
            "reason": (f"departure is {hours_out / 24:.0f} day(s) out"
                       if hours_out >= 24 else f"departure is {hours_out:.0f}h out"),
        }

    icao24 = find_aircraft(cs, origin_iata, scheduled_departure_ts)
    if not icao24:
        return {"observed": False, "callsign": cs,
                "reason": "aircraft not seen departing this airport near that time"}

    prior = prior_leg(icao24, scheduled_departure_ts)
    if not prior:
        return {"observed": False, "callsign": cs, "icao24": icao24,
                "reason": "no earlier leg observed for this aircraft today"}

    gap = turnaround_min(prior, scheduled_departure_ts)
    if gap is None:
        return {"observed": False, "callsign": cs, "icao24": icao24,
                "reason": "inbound landing outside a usable window"}

    inferred_delay = round(max(0.0, TYPICAL_TURNAROUND_MIN - gap), 1)
    return {
        "observed": True,
        "callsign": cs,
        "icao24": icao24,
        "prior_leg": f"{prior.get('from') or '?'} -> {prior.get('to') or '?'}",
        "inbound_landed_ts": prior["actual_arrival_ts"],
        "ground_time_min": gap,
        "inferred_inbound_delay_min": inferred_delay,
        # honest about which of these is measured and which is derived
        "basis": ("observed inbound landing; inbound delay inferred from a "
                  f"{TYPICAL_TURNAROUND_MIN:.0f} min typical turnaround"),
    }


# Live positions change every few seconds, but a page refresh does not need a
# fresh fetch - and every fetch spends from a 4,000/day credit budget shared by
# all viewers. A short TTL keyed on the rounded box means a dozen people looking
# at the same route cost one call.
_TRAFFIC_TTL_S = 20.0
_traffic_cache: dict[tuple, tuple[float, list]] = {}


def bbox_for_route(origin_iata: str, dest_iata: str, pad_deg: float = 1.5) -> tuple | None:
    """A lat/lon box covering both airports and the airspace between them.

    Returns None when either airport is outside AIRPORT_COORDS, which is the
    caller's cue to skip the map rather than draw an empty one.
    """
    from reference_data import AIRPORT_COORDS
    a = AIRPORT_COORDS.get((origin_iata or "").upper())
    b = AIRPORT_COORDS.get((dest_iata or "").upper())
    if not a or not b:
        return None
    lats, lons = (a[0], b[0]), (a[1], b[1])
    return (min(lats) - pad_deg, min(lons) - pad_deg,
            max(lats) + pad_deg, max(lons) + pad_deg)


def live_traffic(bbox: tuple, highlight_callsign: str | None = None) -> list[dict]:
    """Aircraft currently airborne (or taxiing) inside `bbox`.

    Unlike everything else in this module, this needs no schedule and no tail
    assignment - it is simply where the transponders are right now, which is the
    one thing the network can always answer. Empty list on any failure.
    """
    key = tuple(round(v, 1) for v in bbox)
    hit = _traffic_cache.get(key)
    now = time.time()
    if hit and now - hit[0] < _TRAFFIC_TTL_S:
        rows = hit[1]
    else:
        lamin, lomin, lamax, lomax = bbox
        body = _get("/states/all", {"lamin": lamin, "lomin": lomin,
                                    "lamax": lamax, "lomax": lomax})
        if not body or not isinstance(body, dict):
            return []
        rows = body.get("states") or []
        _traffic_cache[key] = (now, rows)

    want = (highlight_callsign or "").replace(" ", "").upper()
    out = []
    for r in rows:
        # State vectors are positional: 0 icao24, 1 callsign, 5 lon, 6 lat,
        # 7 barometric altitude, 8 on_ground, 9 velocity m/s, 10 true track.
        lat, lon = r[6], r[5]
        if lat is None or lon is None:
            continue
        cs = (r[1] or "").strip().upper()
        out.append({
            "icao24": r[0],
            "callsign": cs or None,
            "lat": round(lat, 4),
            "lon": round(lon, 4),
            "heading": round(r[10], 1) if r[10] is not None else None,
            "altitude_ft": round(r[7] * 3.28084) if r[7] is not None else None,
            "speed_kt": round(r[9] * 1.94384) if r[9] is not None else None,
            "on_ground": bool(r[8]),
            "is_yours": bool(want and cs and (cs == want or cs.startswith(want))),
        })
    return out


if __name__ == "__main__":
    import io
    from datetime import datetime, timezone

    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    ts = int(time.time()) - 2 * 86400

    for iata in ["DEL", "BOM", "BLR", "MAA"]:
        deps = _get("/flights/departure", {
            "airport": icao_for(iata), "begin": ts - 7200, "end": ts,
        }) or []
        print(f"\n{iata}: {len(deps)} departures observed in a 2h window")
        for f in deps[:2]:
            cs = (f.get("callsign") or "?").strip()
            ac = f.get("icao24")
            p = prior_leg(ac, f["firstSeen"])
            when = datetime.fromtimestamp(f["firstSeen"], timezone.utc).strftime("%H:%M")
            if p:
                turn = turnaround_min(p, f["firstSeen"])
                print(f"  {cs:9s} dep {when}Z  aircraft {ac}  <- prior leg "
                      f"{p['from']}->{p['to']}, turnaround {turn} min")
            else:
                print(f"  {cs:9s} dep {when}Z  aircraft {ac}  <- no prior leg observed")
