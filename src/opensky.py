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


def live_rotation(carrier_iata: str, flight_number: str, origin_iata: str,
                   scheduled_departure_ts: int) -> dict:
    """Pre-departure view of the aircraft due to operate this flight.

    Deliberately measures the gap to the SCHEDULED departure, not the actual
    one. Actual departure is unknown before the fact, and using it would leak
    the outcome - a flight that leaves late has a *longer* observed turnaround,
    which would teach exactly the wrong lesson.

    Returns an inferred inbound delay: if an aircraft that normally gets ~50
    minutes on the ground only has 20 left before its scheduled push, roughly
    30 minutes of that has already been eaten by a late inbound.
    """
    cs = callsign_for(carrier_iata, flight_number)
    if not cs:
        return {"observed": False, "reason": "no ICAO callsign known for this carrier"}

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
