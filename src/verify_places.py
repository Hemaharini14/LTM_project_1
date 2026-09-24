"""
Confirm that a named place actually exists near a city.

This exists because of a gap between two sources that are each wrong on their
own. Geoapify's Places API returns real, correctly-categorised POIs, but its
coverage of a hill station like Ooty is thin and noisy - fourteen named
results, most of them minor, with Ooty Lake, the Botanical Garden and
Doddabetta missing entirely. An LLM, asked the same question, names exactly
the spots a visitor would actually go to, because those are documented
everywhere - but it can also produce a confident name for a place that does
not exist, and nothing downstream could tell the difference.

So: let the model propose, and make a geocoder confirm. A name survives only
if a real place by roughly that name sits near the city. Two witnesses are
tried, either of which is sufficient:

  the geocoder   Geoapify resolves "<name>, <city>" to something nearby whose
                 own name shares the distinctive words of the request.
  Wikipedia      a notable article with a similar title is geotagged within
                 10km (geosearch is free, keyless, and caps at that radius).

The name check is the part that matters. Without it a geocoder answers almost
any query by falling back to the city centroid, which "confirms" everything:
"Doddabetta Peak" came back as Udhagamandalam, and so did an entirely invented
temple. Comparison therefore ignores generic place-type words - temple, beach,
museum, falls - and matches on what is actually distinctive, so
"Kapaleeshwarar Temple" no longer passes by resolving to the unrelated
Kachaleeswarar Temple down the road.

This is deliberately conservative: it drops real places it cannot confirm
(Wenlock Downs, which no source resolves cleanly) rather than pass an
unconfirmed name through. A confirmed spot is a real one; an absent spot is
not a claim that it isn't.
"""
from __future__ import annotations

import os
import re
from concurrent.futures import ThreadPoolExecutor
from math import asin, cos, radians, sin, sqrt

import httpx

try:
    # Same reason as llm_utils/maps: this network TLS-intercepts HTTPS, and
    # without the OS trust store both lookups fail verification - which this
    # module swallows as "unconfirmed", silently dropping every real place.
    import truststore
    truststore.inject_into_ssl()
except ImportError:
    pass

GEOCODE_URL = "https://api.geoapify.com/v1/geocode/search"
WIKI_URL = "https://en.wikipedia.org/w/api.php"
WIKI_RADIUS_M = 10000          # the API's hard maximum
MAX_DISTANCE_KM = 60.0         # a day trip from the city still counts as "near"
_HEADERS = {"User-Agent": "SmartRouteAI/1.0 (educational travel-planning project)"}

# Words that say what KIND of place something is rather than which one. Matching
# on these lets any temple confirm any other temple, so they are excluded and
# similarity is judged on the distinctive remainder.
_GENERIC = {
    "the", "of", "and", "a", "an", "in", "at", "near", "city", "town", "old", "new",
    "temple", "church", "basilica", "cathedral", "mosque", "shrine", "monastery",
    "beach", "lake", "falls", "waterfall", "river", "hill", "hills", "peak", "point",
    "park", "garden", "gardens", "museum", "gallery", "fort", "palace", "castle",
    "railway", "station", "road", "street", "market", "bazaar", "tower", "bridge",
    "national", "state", "government", "memorial", "monument", "centre", "center",
    "house", "view", "viewpoint", "valley", "dam", "zoo", "aquarium", "resort",
}

_verify_cache: dict[tuple, dict | None] = {}


def _tokens(s: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]+", (s or "").lower())
            if w not in _GENERIC and len(w) > 2}


def _similar(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / min(len(ta), len(tb))


def _km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    dlat, dlon = radians(lat2 - lat1), radians(lon2 - lon1)
    h = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    return 2 * 6371.0 * asin(sqrt(h))


def _by_geocoder(name: str, city: str, lat: float, lon: float) -> dict | None:
    api_key = os.environ.get("GEOAPIFY_API_KEY")
    if not api_key:
        return None
    try:
        r = httpx.get(GEOCODE_URL, params={
            "text": f"{name}, {city}", "bias": f"proximity:{lon},{lat}",
            "limit": 3, "format": "json", "apiKey": api_key,
        }, timeout=12)
        r.raise_for_status()
        results = r.json().get("results") or []
    except Exception:
        return None

    for hit in results:
        got = hit.get("name") or hit.get("address_line1") or ""
        if hit.get("lat") is None or hit.get("lon") is None:
            continue
        d = _km(lat, lon, hit["lat"], hit["lon"])
        if d <= MAX_DISTANCE_KM and _similar(name, got) >= 0.5:
            return {"confirmed_by": "geocoder", "matched_name": got,
                    "distance_km": round(d, 1), "lat": hit["lat"], "lon": hit["lon"]}
    return None


_wiki_cache: dict[tuple, list] = {}


def _wiki_nearby(lat: float, lon: float) -> list:
    key = (round(lat, 2), round(lon, 2))
    if key in _wiki_cache:
        return _wiki_cache[key]
    try:
        r = httpx.get(WIKI_URL, params={
            "action": "query", "list": "geosearch", "gscoord": f"{lat}|{lon}",
            "gsradius": WIKI_RADIUS_M, "gslimit": 100, "format": "json",
        }, headers=_HEADERS, timeout=15)
        r.raise_for_status()
        hits = r.json().get("query", {}).get("geosearch", [])
    except Exception:
        hits = []
    _wiki_cache[key] = hits
    return hits


def _by_wikipedia(name: str, lat: float, lon: float) -> dict | None:
    for h in _wiki_nearby(lat, lon):
        if _similar(name, h.get("title", "")) >= 0.6:
            return {"confirmed_by": "wikipedia", "matched_name": h["title"],
                    "distance_km": round(h.get("dist", 0) / 1000.0, 1),
                    "lat": h.get("lat"), "lon": h.get("lon")}
    return None


def confirm_place(name: str, city: str, lat: float, lon: float) -> dict | None:
    """A real place by roughly this name near (lat, lon), or None.

    The geocoder is tried first because it covers far more than Wikipedia does;
    Wikipedia catches landmarks the geocoder resolves only to their district.
    """
    if not name or not city:
        return None
    key = (name.strip().lower(), round(lat, 2), round(lon, 2))
    if key in _verify_cache:
        return _verify_cache[key]
    found = _by_geocoder(name, city, lat, lon) or _by_wikipedia(name, lat, lon)
    _verify_cache[key] = found
    return found


def confirm_all(names: list[str], city: str, lat: float, lon: float,
                max_workers: int = 6) -> list[dict]:
    """Confirm several names at once, preserving the order given.

    Each check is a network round-trip of a couple of seconds, so eight names
    would be twenty seconds run one after another - and this sits in the
    request path of a trip plan. They are independent, so they run together.
    """
    if not names:
        return []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        results = list(pool.map(lambda n: confirm_place(n, city, lat, lon), names))
    return [{"name": n, **r} for n, r in zip(names, results) if r]


if __name__ == "__main__":
    import io
    import sys
    from dotenv import load_dotenv
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

    cases = {
        "Ooty": (11.4127, 76.7031,
                 ["Ooty Lake", "Government Botanical Garden", "Doddabetta Peak",
                  "Rose Garden Ooty", "Nilgiri Mountain Railway", "Pykara Falls",
                  "Completely Fictional Temple of Zorblax", "Grand Imaginary Aquarium"]),
        "Chennai": (13.0827, 80.2707,
                    ["Marina Beach", "Kapaleeshwarar Temple", "Fort St. George",
                     "Invented Palace of Nowhere"]),
    }
    for city, (lat, lon, names) in cases.items():
        kept = confirm_all(names, city, lat, lon)
        keep = {k["name"] for k in kept}
        print(f"\n{city}: {len(kept)}/{len(names)} confirmed")
        for k in kept:
            print(f"   OK       {k['name'][:34]:36s} {k['confirmed_by']:9s} "
                  f"-> {k['matched_name'][:30]:32s} {k['distance_km']:4.1f} km")
        for n in names:
            if n not in keep:
                print(f"   dropped  {n[:34]:36s}")
