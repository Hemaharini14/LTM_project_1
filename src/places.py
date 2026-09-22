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
# Foursquare pins behaviour to a dated API version; sending it explicitly keeps
# the response shape stable instead of silently drifting when they ship changes.
API_VERSION = "2025-06-17"

# Free-tier fields only. Adding "rating" or "price" here makes the whole call
# fail with 429 on a free account - see the module docstring.
FIELDS = "fsq_place_id,name,location,latitude,longitude,website,tel"


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
