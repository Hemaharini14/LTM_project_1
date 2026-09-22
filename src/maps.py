"""
Free, keyless geocoding via OpenStreetMap's Nominatim - used to place an
AI-suggested hotel name on a map, and to center sightseeing.py's real
Geoapify POI search on a city. Nominatim's usage policy caps requests at
~1/second and requires a real User-Agent - respected here with a small delay
between calls. If a specific place name doesn't resolve (LLM-suggested names
sometimes don't match Nominatim's index exactly), we fall back to just the
destination city so the user still gets a map, honestly centered on the area
rather than the exact (unverified) property.
"""
import math
import os
import time
import httpx
from dotenv import load_dotenv

try:
    import truststore
    truststore.inject_into_ssl()
except ImportError:
    pass

load_dotenv()

from reference_data import AIRPORT_COORDS

_NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
_GEOAPIFY_ROUTING_URL = "https://api.geoapify.com/v1/routing"
_HEADERS = {"User-Agent": "SmartRouteAI/1.0 (educational travel-planning project)"}
_last_call = 0.0
_EARTH_RADIUS_KM = 6371.0


def _throttle():
    global _last_call
    elapsed = time.monotonic() - _last_call
    if elapsed < 1.1:
        time.sleep(1.1 - elapsed)
    _last_call = time.monotonic()


def _geocode_once(query: str) -> dict | None:
    _throttle()
    try:
        r = httpx.get(_NOMINATIM_URL, params={"q": query, "format": "json", "limit": 1},
                      headers=_HEADERS, timeout=10)
        data = r.json()
        if data:
            return {"lat": float(data[0]["lat"]), "lon": float(data[0]["lon"]),
                    "display_name": data[0]["display_name"]}
    except Exception:
        pass
    return None


def geocode_city(city: str) -> dict | None:
    """Real (lat, lon) for a city name - shared by anything that needs to center a
    search on a place (e.g. sightseeing.py's Geoapify radius search)."""
    return _geocode_once(city)


def geocode_hotel(hotel_name: str, destination_city: str) -> dict | None:
    """Tries "hotel name, city" first, then just the city as a fallback so the map still shows
    something useful. Returns None (skip the map entirely) only if even the city fails."""
    if destination_city:
        result = _geocode_once(f"{hotel_name}, {destination_city}")
        if result:
            result["approximate"] = False
            return result
        result = _geocode_once(destination_city)
        if result:
            result["approximate"] = True
            return result
    return _geocode_once(hotel_name)


def osm_embed_url(lat: float, lon: float, delta: float = 0.01) -> str:
    """A no-API-key embeddable OpenStreetMap iframe centered on (lat, lon) with a marker."""
    bbox = f"{lon - delta},{lat - delta},{lon + delta},{lat + delta}"
    return f"https://www.openstreetmap.org/export/embed.html?bbox={bbox}&marker={lat},{lon}"


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Real great-circle distance between two real coordinate pairs - used to find
    the nearest airport, never to invent a distance for a place we haven't geocoded."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
    return 2 * _EARTH_RADIUS_KM * math.asin(math.sqrt(a))


def nearest_supported_airport(place: str, allowed_codes) -> dict | None:
    """Geocodes `place` (Nominatim - free, real) and returns the closest airport,
    by real great-circle distance, among `allowed_codes` (the finite set this
    project actually has flight/hotel data for - see reference_data.AIRPORT_CITY).
    This is deliberately NOT "nearest airport on Earth" - an arbitrary city could
    resolve to an airport with zero real flight data, which would be worse than
    just asking the user for a code. Returns None if the place doesn't geocode or
    none of allowed_codes have known coordinates."""
    geo = geocode_city(place)
    if not geo:
        return None
    candidates = [(code, *AIRPORT_COORDS[code]) for code in allowed_codes if code in AIRPORT_COORDS]
    if not candidates:
        return None
    best_code, best_km = None, None
    for code, alat, alon in candidates:
        km = _haversine_km(geo["lat"], geo["lon"], alat, alon)
        if best_km is None or km < best_km:
            best_code, best_km = code, km
    return {"airport_code": best_code, "distance_km": round(best_km, 1),
            "place_lat": geo["lat"], "place_lon": geo["lon"], "place_name": geo["display_name"]}


def route_between(from_lat: float, from_lon: float, to_lat: float, to_lon: float,
                   mode: str = "drive") -> dict | None:
    """Real routed distance/duration between two real coordinate pairs, via Geoapify's
    Routing API (the same free key sightseeing.py uses). Returns None - never a guessed
    number - if no key is configured, the call fails, or no route exists (Geoapify
    legitimately 400s with "No path could be found" for some pairs).

    mode is a Geoapify travel mode; only road modes are valid. There is no 'train' mode,
    and 'bus' just routes a bus-shaped vehicle over the same roads as 'drive' (verified:
    identical distance and time), so it is NOT a real bus service - see transport_modes.py.
    """
    api_key = os.environ.get("GEOAPIFY_API_KEY")
    if not api_key:
        return None
    try:
        resp = httpx.get(_GEOAPIFY_ROUTING_URL, params={
            "waypoints": f"{from_lat},{from_lon}|{to_lat},{to_lon}",
            "mode": mode,
            "apiKey": api_key,
        }, timeout=15)
        resp.raise_for_status()
        features = resp.json().get("features", [])
        if not features:
            return None
        props = features[0]["properties"]
        return {"distance_km": round(props["distance"] / 1000, 1),
                "duration_min": round(props["time"] / 60)}
    except Exception as e:
        print(f"[maps] Geoapify routing failed ({mode}): {e}")
        return None


def route_airport_to_place(airport_code: str, place_lat: float, place_lon: float) -> dict | None:
    """Real driving distance/duration from `airport_code` to (place_lat, place_lon).
    Returns None if the airport has no known coordinates or the routing call fails."""
    coords = AIRPORT_COORDS.get((airport_code or "").upper())
    if not coords:
        return None
    return route_between(coords[0], coords[1], place_lat, place_lon, mode="drive")
