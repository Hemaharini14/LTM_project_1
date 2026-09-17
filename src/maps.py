"""
Free, keyless geocoding via OpenStreetMap's Nominatim, used only to place an
AI-suggested hotel name on a map. Nominatim's usage policy caps requests at
~1/second and requires a real User-Agent - respected here with a small delay
between calls. If a specific place name doesn't resolve (LLM-suggested names
sometimes don't match Nominatim's index exactly), we fall back to just the
destination city so the user still gets a map, honestly centered on the area
rather than the exact (unverified) property.
"""
import time
import httpx

_NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
_HEADERS = {"User-Agent": "SmartRouteAI/1.0 (educational travel-planning project)"}
_last_call = 0.0


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
