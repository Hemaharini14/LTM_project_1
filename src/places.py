"""
Real, verifiable hotels from the Foursquare Places API.

This replaces the weakest data in the trip planner. Until now the named hotels
shown to the traveller came from llm_utils.suggest_destination_content() - an
LLM listing properties from memory, correctly labelled "AI-suggested,
unverified" because a model can be wrong about a real business. Foursquare
returns actual venues: real name, real street address, real coordinates, and
(usually) a real website and phone number, so the traveller can confirm the
place exists before booking.

What it deliberately does NOT return is a rating or a price. Both are Premium
fields on Foursquare - verified directly against the live API, they answer
HTTP 429 "no API credits remaining" on the free tier while name/address/
location return 200. So this module has no review data, and the UI must not
imply otherwise. Nightly cost continues to come from the real ADR medians in
cleaned_hotel_bookings.csv (a price tier, not this specific property's rate).

Without FOURSQUARE_API_KEY set, or on any failure, this returns an empty list
so callers fall through to the previous AI-suggested behaviour - the same
"always works, better when configured" pattern as sightseeing.py.
"""
import os
import sys
import threading

import httpx
from dotenv import load_dotenv

try:
    import truststore
    truststore.inject_into_ssl()
except ImportError:
    pass

load_dotenv()

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from maps import osm_embed_url

SEARCH_URL = "https://places-api.foursquare.com/places/search"
# Geoapify's purpose-built geocoding autocomplete. Used instead of Nominatim
# because Nominatim's usage policy explicitly forbids building autocomplete on
# it, and because this one takes a type filter - which is what keeps a query
# like "newyork" off a Tokyo hairdresser and on an actual city.
AUTOCOMPLETE_URL = "https://api.geoapify.com/v1/geocode/autocomplete"
# Foursquare pins behaviour to a dated API version; sending it explicitly keeps
# the response shape stable instead of silently drifting when they ship changes.
API_VERSION = "2025-06-17"

# Free-tier fields only. Adding "rating" or "price" here makes the whole call
# fail with 429 on a free account - see the module docstring.
FIELDS = "fsq_place_id,name,location,latitude,longitude,website,tel"

# One pooled client for the whole process. Typeahead fires a request per
# keystroke, and opening a fresh TLS connection each time cost ~2.3s per call
# against ~0.7s on a warm pool - the single biggest reason the dropdown felt slow.
_CLIENT = httpx.Client(timeout=8)

# Successful lookups only: caching a transient failure would pin an empty
# dropdown for that prefix until restart. Bounded so it can't grow unbounded.
_AUTOCOMPLETE_CACHE: dict[str, list[dict]] = {}
_CACHE_MAX = 512


def _search(query: str, near: str, limit: int) -> list[dict]:
    api_key = os.environ.get("FOURSQUARE_API_KEY")
    if not api_key or not near:
        return []
    try:
        resp = httpx.get(SEARCH_URL, headers={
            "Authorization": f"Bearer {api_key}",
            "X-Places-Api-Version": API_VERSION,
        }, params={"query": query, "near": near, "limit": limit, "fields": FIELDS}, timeout=15)
        resp.raise_for_status()
        return resp.json().get("results", [])
    except Exception as e:
        print(f"[places] Foursquare '{query}' lookup failed for '{near}': {e}")
        return []


def _to_dict(place: dict, note: str) -> dict | None:
    name = place.get("name")
    if not name:
        return None
    loc = place.get("location", {})
    lat, lon = place.get("latitude"), place.get("longitude")
    return {
        "name": name,
        # keys named to match the AI-suggested shape the templates already render
        "area": loc.get("locality") or loc.get("region") or "",
        "note": note,
        "address": loc.get("formatted_address") or loc.get("address") or "",
        "website": place.get("website"),
        "tel": place.get("tel"),
        "map_url": osm_embed_url(lat, lon) if lat is not None and lon is not None else None,
        # real coordinates for this exact venue, so the pin is the property itself
        "map_approximate": False,
        "source": "foursquare",
    }


def autocomplete_places(query: str, limit: int = 6) -> list[dict]:
    """City suggestions for a partial name, for the location pickers in the trip form.

    type=city restricts results to actual settlements, so the traveller can only pick
    a real city - they can no longer submit free text that resolves to a shop, which
    is how a Boston -> "newyork" trip once ended up planned around Tokyo. Each result
    carries its real coordinates, and the formatted label is unambiguous enough to
    re-resolve cleanly downstream.
    """
    api_key = os.environ.get("GEOAPIFY_API_KEY")
    query = (query or "").strip()
    if not api_key or len(query) < 2:
        return []

    key = f"{query.lower()}|{limit}"
    if key in _AUTOCOMPLETE_CACHE:
        return _AUTOCOMPLETE_CACHE[key]

    try:
        resp = _CLIENT.get(AUTOCOMPLETE_URL, params={
            "text": query, "type": "city", "limit": limit, "apiKey": api_key,
        })
        resp.raise_for_status()
        out = []
        for f in resp.json().get("features", []):
            p = f.get("properties", {})
            label = p.get("formatted")
            if label:
                out.append({"label": label, "lat": p.get("lat"), "lon": p.get("lon")})
        if out:
            if len(_AUTOCOMPLETE_CACHE) >= _CACHE_MAX:
                _AUTOCOMPLETE_CACHE.clear()
            _AUTOCOMPLETE_CACHE[key] = out
        return out
    except Exception as e:
        print(f"[places] autocomplete failed for '{query}': {e}")
        return []


def _warm_connection_pool():
    """Open the TLS connection once at startup so the traveller's first keystroke
    doesn't pay for the handshake (~2s cold vs ~0.5s warm). Best-effort and in the
    background - a failure here just means the first lookup is slow, as before."""
    try:
        _CLIENT.get(AUTOCOMPLETE_URL, params={
            "text": "a", "type": "city", "limit": 1,
            "apiKey": os.environ.get("GEOAPIFY_API_KEY", ""),
        })
    except Exception:
        pass


if os.environ.get("GEOAPIFY_API_KEY"):
    threading.Thread(target=_warm_connection_pool, daemon=True).start()


def find_real_hotels(city: str, limit: int = 4) -> list[dict]:
    """Real hotels in `city` with real addresses. No rating and no price - those are
    Premium fields this free tier can't read (see module docstring)."""
    out = []
    for place in _search("hotel", city, limit):
        d = _to_dict(place, "Real venue (Foursquare) - no rating or price on the free tier")
        if d:
            out.append(d)
    return out


def find_real_food_spots(city: str, limit: int = 4) -> list[dict]:
    """Real restaurants in `city`, same caveats as find_real_hotels."""
    out = []
    for place in _search("restaurant", city, limit):
        d = _to_dict(place, "Real venue (Foursquare)")
        if d:
            out.append(d)
    return out


if __name__ == "__main__":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    for h in find_real_hotels("Bengaluru", 5):
        print(f"{h['name'][:32]:34s} {h['address'][:46]:48s} tel={h['tel']}")
