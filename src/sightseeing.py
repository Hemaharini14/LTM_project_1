"""
Real sightseeing points-of-interest via the Geoapify Places API - genuine
attractions (name + real address), not an LLM guess. Sits between the
hand-curated FOOD_SPOTS-style data (intl_reference.py) and the LLM fallback
(llm_utils.suggest_destination_content): try curated first, then this real
API, and only fall back to an LLM guess (always labeled unverified) if
neither has anything for the destination.

Requires a free GEOAPIFY_API_KEY (see .env.example) - without one, or if the
lookup fails for any reason, this returns an empty list so callers fall
through to the next tier rather than crashing.
"""
import os
import httpx

from maps import geocode_city

PLACES_URL = "https://api.geoapify.com/v2/places"
# Verified against the real API - "tourism.museum" and "heritage" are NOT valid
# Geoapify categories despite seeming plausible; museums live under entertainment.
CATEGORIES = "tourism.sights,tourism.attraction,entertainment.museum"
SEARCH_RADIUS_M = 8000

# Geoapify's leaf category (last segment) -> a short human label for the "note" field.
_CATEGORY_LABELS = {
    "museum": "Museum", "monument": "Monument", "memorial": "Memorial",
    "castle": "Castle", "ruines": "Historic ruins", "fort": "Fort",
    "artwork": "Public artwork", "attraction": "Attraction", "sights": "Sight",
}


def _label_for(categories: list[str]) -> str:
    for cat in reversed(categories or []):
        leaf = cat.split(".")[-1]
        if leaf in _CATEGORY_LABELS:
            return _CATEGORY_LABELS[leaf]
    return "Real point of interest (Geoapify)"


def get_real_sightseeing(city: str, limit: int = 4) -> list[dict]:
    """Real attractions near `city`, in the same {"name", "area", "note"} shape
    intl_reference.FOOD_SPOTS uses, so templates/tools can treat them the same way."""
    api_key = os.environ.get("GEOAPIFY_API_KEY")
    if not api_key or not city:
        return []

    geo = geocode_city(city)
    if not geo:
        return []

    try:
        resp = httpx.get(PLACES_URL, params={
            "categories": CATEGORIES,
            "filter": f"circle:{geo['lon']},{geo['lat']},{SEARCH_RADIUS_M}",
            "bias": f"proximity:{geo['lon']},{geo['lat']}",
            "limit": limit,
            "apiKey": api_key,
        }, timeout=10)
        resp.raise_for_status()
        features = resp.json().get("features", [])
    except Exception as e:
        print(f"[sightseeing] Geoapify lookup failed for '{city}': {e}")
        return []

    spots = []
    for f in features:
        props = f.get("properties", {})
        name = props.get("name")
        if not name:
            continue
        # Real popularity signal, not a guess: Geoapify passes through the OSM
        # wikipedia/wikidata tags, and a place only carries those if it's notable
        # enough for someone to have written an encyclopedia entry about it. So
        # "has wiki" separates the landmarks everyone visits from the genuinely
        # lesser-known local spots, without asking an LLM to rank real places.
        well_known = bool(props.get("wiki_and_media"))
        spots.append({
            "name": name,
            "area": props.get("suburb") or props.get("district") or props.get("city") or city,
            "note": _label_for(props.get("categories", [])),
            "well_known": well_known,
            "lat": props.get("lat"),
            "lon": props.get("lon"),
        })
    return spots


def split_common_and_hidden(spots: list[dict]) -> tuple[list[dict], list[dict]]:
    """Real landmarks (have a wikipedia/wikidata entry) vs genuinely lesser-known real
    places (don't). Both lists are real Geoapify places either way - this only groups
    them. LLM-suggested spots have no wiki flag, so they fall into 'hidden' by default;
    callers that mix sources should label them as unverified regardless."""
    common = [s for s in spots if s.get("well_known")]
    hidden = [s for s in spots if not s.get("well_known")]
    return common, hidden
