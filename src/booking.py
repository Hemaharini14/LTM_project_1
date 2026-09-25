"""
Deep links from a recommended flight to a site that can actually sell it.

The recovery plan names a better flight and then leaves you to go and find it.
This builds the search URL that lands on that route, that date and that party
size, so the next step is one click rather than retyping everything.

DELIBERATELY NOT AN LLM CALL, even though this hangs off the agent. A URL is a
fixed grammar: get a character wrong and the link 404s or silently drops the
passenger count, and asking a model to emit one each time buys nothing but
latency and a new way to be wrong. The agent gets this as a TOOL instead - it
decides when a booking link is worth offering and for which flight, and the
link itself is assembled by code that was checked against the live sites.

Each format below was verified by request: the parameters survive the redirect
chain rather than being dropped on arrival. Google Flights was tried and
rejected - it bounces a query-string search to /travel/flights/unsupported.

These are SEARCH links, not bookings. Nothing here reserves a seat, sends a
payment or passes any personal detail - the airline and the site handle all of
that. Fares are not ours to quote either, which is why no price is carried
across: the catalogue's fares are estimates (see build_catalog) and putting one
next to a booking button would read as a quote.
"""
from __future__ import annotations

from datetime import date, datetime
from urllib.parse import quote

# Ordered: the first is offered as the main action, the rest as alternatives.
# All three are Indian OTAs that sell domestic and international tickets.
SITES = ("ixigo", "makemytrip", "goibibo")

DISPLAY_NAME = {
    "ixigo": "ixigo",
    "makemytrip": "MakeMyTrip",
    "goibibo": "Goibibo",
}

MAX_ADULTS = 9          # every one of these sites caps a single booking here


def _as_date(when: str | date | datetime | None) -> date:
    if isinstance(when, datetime):
        return when.date()
    if isinstance(when, date):
        return when
    if when:
        for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y"):
            try:
                return datetime.strptime(str(when)[:10], fmt).date()
            except ValueError:
                continue
    return date.today()


def split_route(route: str) -> tuple[str, str] | None:
    """"ORD-DEN" -> ("ORD", "DEN"). None if it is not a route string."""
    parts = [p.strip().upper() for p in str(route or "").replace("→", "-").split("-")]
    parts = [p for p in parts if len(p) == 3 and p.isalpha()]
    return (parts[0], parts[1]) if len(parts) >= 2 else None


def booking_url(origin: str, destination: str, when: str | date | None,
                adults: int = 1, children: int = 0, infants: int = 0,
                site: str = "ixigo") -> str | None:
    """A search URL on `site` for this route, date and party.

    Returns None rather than a broken link when the airports are not proper
    IATA codes - a three-letter code is the one thing every one of these
    sites requires.
    """
    o, d = (origin or "").strip().upper(), (destination or "").strip().upper()
    if not (len(o) == 3 and o.isalpha() and len(d) == 3 and d.isalpha()):
        return None

    day = _as_date(when)
    adults = max(1, min(int(adults or 1), MAX_ADULTS))
    children = max(0, min(int(children or 0), MAX_ADULTS))
    infants = max(0, min(int(infants or 0), adults))   # one lap infant per adult

    if site == "makemytrip":
        return (f"https://www.makemytrip.com/flight/search"
                f"?itinerary={o}-{d}-{day.strftime('%d/%m/%Y')}"
                f"&tripType=O&paxType=A-{adults}_C-{children}_I-{infants}"
                f"&cabinClass=E")
    if site == "goibibo":
        return (f"https://www.goibibo.com/flights/air-{o}-{d}-"
                f"{day.strftime('%Y%m%d')}--{adults}-{children}-{infants}-E-D/")
    # ixigo, the default
    return (f"https://www.ixigo.com/search/result/flight"
            f"?from={o}&to={d}&date={day.strftime('%d%m%Y')}"
            f"&adults={adults}&children={children}&infants={infants}"
            f"&class=e&source=SmartRouteAI")


def booking_links(origin: str, destination: str, when: str | date | None,
                  adults: int = 1, children: int = 0, infants: int = 0) -> list[dict]:
    """Every site's link for one flight, ready to render as buttons."""
    out = []
    for site in SITES:
        url = booking_url(origin, destination, when, adults, children, infants, site)
        if url:
            out.append({"site": site, "name": DISPLAY_NAME[site], "url": url})
    return out


def links_for_flight(flight: dict, when: str | date | None, adults: int = 1,
                     children: int = 0, infants: int = 0) -> list[dict]:
    """Links for one of search_alternative_flights()'s dicts.

    Those carry the route as a single "ORD-DEN" string rather than two fields,
    so it is split here instead of at every call site.
    """
    pair = split_route(flight.get("route", ""))
    if not pair:
        return []
    return booking_links(pair[0], pair[1], when, adults, children, infants)


def describe_party(adults: int, children: int = 0, infants: int = 0) -> str:
    """"2 adults, 1 child" - for the line above the booking buttons."""
    bits = [f"{adults} adult{'s' if adults != 1 else ''}"]
    if children:
        bits.append(f"{children} child{'ren' if children != 1 else ''}")
    if infants:
        bits.append(f"{infants} infant{'s' if infants != 1 else ''}")
    return ", ".join(bits)


if __name__ == "__main__":
    import io
    import sys
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

    print("split_route:", split_route("ORD-DEN"), split_route("DEL → BOM"), split_route("nonsense"))
    print("party:", describe_party(2, 1, 1))
    print()
    for link in booking_links("DEL", "BOM", "2026-10-02", adults=2, children=1):
        print(f"  {link['name']:12s} {link['url']}")
    print()
    print("bad codes ->", booking_url("XX1", "BOM", "2026-10-02"))
    print("infants capped to adults ->",
          "infants=1" in (booking_url("DEL", "BOM", "2026-10-02", adults=1, infants=5) or ""))
