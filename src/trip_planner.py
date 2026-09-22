"""
Module 4 (web app): end-to-end budget trip planning.

The flight/hotel/sightseeing-budget choice itself is genuinely optimized by
itinerary_optimizer.py - jointly searching real flight options and real
hotel tiers against the traveler's actual total budget, rather than
guessing a fixed % split per category and patching afterward. This module
wraps that result with the rest of the trip content: named food spots,
real/AI sightseeing highlights, a day-by-day plan, and (if an LLM is
configured) AI-suggested named hotels with a map. No fabricated
flights/hotels/prices - those always come from the real cleaned datasets,
the trained delay model, or deterministic math. Sightseeing tries real
points-of-interest (sightseeing.py's Geoapify lookup) first, then an LLM
guess (labelled unverified), and only falls back to the fully generic
SIGHTSEEING_TEMPLATE below if neither has anything for the destination.
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from recovery_tools import get_destination_weather, is_route_covered, get_dataset_index
from budget_optimizer import validate_hotel_budget
from intl_reference import food_spots_for, INTL_AIRPORT_CODES
from llm_utils import suggest_destination_content
from maps import geocode_hotel, osm_embed_url, nearest_supported_airport, route_airport_to_place
from sightseeing import get_real_sightseeing, split_common_and_hidden
from places import find_real_hotels
from itinerary_optimizer import optimize_itinerary

# Comfort level -> multiplier applied to that category's real/baseline share
LEVEL_MULTIPLIER = {"budget": 0.7, "standard": 1.0, "luxury": 1.4}

# comfort level -> priority keyword understood by recovery_tools
LEVEL_TO_PRIORITY = {"budget": "cost", "standard": "time", "luxury": "comfort"}

# How many real sightseeing spots each pace actually schedules per day. This is what
# makes the "light / moderate / packed" choice mean something: before, every level got
# a hard-coded 2 spots a day and the setting only swapped the generic fallback wording.
SPOTS_PER_DAY = {"light": 1, "moderate": 2, "packed": 4}

# sightseeing level -> generic morning/afternoon/evening wording, used only when
# neither real (Geoapify) nor AI-suggested spots exist for the destination.
SIGHTSEEING_TEMPLATE = {
    "light": {"morning": "Free morning / rest", "afternoon": "One guided landmark or museum visit",
              "evening": "Leisure walk nearby"},
    "moderate": {"morning": "Landmark or museum visit", "afternoon": "Local market or neighborhood exploration",
                 "evening": "Leisure time / optional activity"},
    "packed": {"morning": "Early landmark visit", "afternoon": "Museum or cultural site, then local market",
               "evening": "Evening excursion or show"},
}


def plan_budget_trip(
    origin_airport: str,
    destination_airport: str,
    destination_city: str,
    total_budget: float,
    days: int,
    start_weekday: int,
    return_weekday: int,
    stay_comfort: str = "standard",
    travel_comfort: str = "standard",
    food_comfort: str = "standard",
    sightseeing_level: str = "moderate",
    transport_mode: str = "",
) -> dict:
    days = max(int(days), 1)
    nights = max(days - 1, 1)

    # The city picker submits a qualified label ("Chennai, TN, India"). The geocoding
    # APIs want that full string, but the curated tables and the weather CSV key on a
    # bare city name - passing the label made them all miss, which silently dropped
    # curated food spots and fired the expensive LLM fallback instead.
    city_key = (destination_city or "").split(",")[0].strip()

    # If the traveler typed a destination city but left the airport blank, auto-resolve
    # the nearest airport this project actually has real flight/hotel data for - real
    # great-circle distance among that finite known set (see maps.nearest_supported_airport),
    # never "nearest airport on Earth" which could land on one with zero real coverage here.
    # Either way, also geocode the destination place so we can report a real driving
    # time from whichever airport is used to reach it (maps.route_airport_to_place).
    auto_resolved = None
    destination_geo = None
    if destination_city:
        if not destination_airport:
            allowed_airports = set(get_dataset_index()["airports"]) | INTL_AIRPORT_CODES
            auto_resolved = nearest_supported_airport(destination_city, allowed_airports)
            if auto_resolved:
                destination_airport = auto_resolved["airport_code"]
                destination_geo = auto_resolved
        else:
            destination_geo = nearest_supported_airport(destination_city, {destination_airport.upper()})

    destination_airport_transfer = None
    if destination_geo and destination_airport:
        route = route_airport_to_place(destination_airport, destination_geo["place_lat"], destination_geo["place_lon"])
        if route:
            destination_airport_transfer = {"airport_code": destination_airport.upper(), **route}

    # The joint flight/hotel/sightseeing search below optimizes for ONE priority -
    # stay_comfort drives it, since the real hotel-vs-sightseeing budget trade-off
    # is the actual engine here. travel_comfort/food_comfort still scale their own
    # real/baseline shares (see itinerary_optimizer.py's food_multiplier param).
    optimize_priority = LEVEL_TO_PRIORITY.get(stay_comfort, "time")

    result = optimize_itinerary(
        origin_airport=origin_airport, destination_airport=destination_airport,
        total_budget=total_budget, days=days,
        start_weekday=start_weekday, return_weekday=return_weekday,
        priority=optimize_priority,
        food_multiplier=LEVEL_MULTIPLIER.get(food_comfort, 1.0),
        transport_multiplier=LEVEL_MULTIPLIER.get(travel_comfort, 1.0),
    )

    route_covered = is_route_covered(origin_airport, destination_airport)
    route_reference = (not result["has_real_flight_data"]) and bool(result["outbound_flights"] or result["return_flights"])
    outbound_flights = result["outbound_flights"]
    return_flights = result["return_flights"]
    budget = result["budget_breakdown"]

    if result["feasible"]:
        hotel_options = result["hotel_options"]
        nightly_equivalent = budget["hotel_cost"] / nights
    else:
        # No real hotel tier fits this budget even alone with real necessities -
        # report the cheapest real option and the honest shortfall, same spirit
        # as budget_optimizer.reallocate_budget's shortfall reporting.
        hotel_options = [result["cheapest_hotel"]] if result["cheapest_hotel"] else []
        nightly_equivalent = result["cheapest_hotel"]["median_nightly_rate_usd"] if result["cheapest_hotel"] else 0.0
    hotel_reality_check = validate_hotel_budget(budget["hotel_cost"] or nightly_equivalent * nights, nights,
                                                 priority=optimize_priority)

    weather = get_destination_weather(city_key) if city_key else None

    # How many real spots a day this pace actually schedules - drives both how many
    # we request and how the day plan below is filled.
    per_day = SPOTS_PER_DAY.get(sightseeing_level, 2)

    # The LLM is the LAST-RESORT source for food/sightseeing/hotels, so it must only
    # run when a real source actually came up empty. Calling it up front cost ~9.5s on
    # every single plan for a fallback that curated data, Geoapify and Foursquare
    # usually make unnecessary. Lazy + memoised: at most one call, only if needed.
    _ai_cache = {}

    def ai_data():
        if "v" not in _ai_cache:
            _ai_cache["v"] = (suggest_destination_content(
                destination_city, sightseeing_count=min(days * per_day, 40))
                if destination_city else None)
        return _ai_cache["v"]

    food_spots = food_spots_for(city_key)
    food_source = "curated" if food_spots else "none"
    if not food_spots:
        ai = ai_data()
        if ai:
            food_spots = ai.get("food_spots", [])
            if food_spots:
                food_source = "ai"

    # Capped so a long trip doesn't fire an oversized API request; Geoapify simply
    # returns fewer than asked if the city doesn't have that many real POIs.
    # Over-fetch: Geoapify returns the NEAREST matches, so asking for exactly what the
    # itinerary needs gets whatever happens to sit closest to the centre (often a cluster
    # of minor statues) rather than the city's actual landmarks. Pull a wider pool, then
    # lead with the wiki-backed well-known places and keep the rest as hidden gems.
    pool_size = min(max(days * per_day * 3, 20), 60)
    _pool = get_real_sightseeing(destination_city, limit=pool_size) if destination_city else []
    _pool_common, _pool_hidden = split_common_and_hidden(_pool)
    sightseeing_spots = _pool_common + _pool_hidden
    sightseeing_source = "real" if sightseeing_spots else "none"
    if not sightseeing_spots:
        ai = ai_data()
        if ai:
            sightseeing_spots = ai.get("sightseeing", [])
            if sightseeing_spots:
                sightseeing_source = "ai"

    # Real named venues first (Foursquare: real name, address and coordinates), and
    # only fall back to the LLM's remembered property names - which can be wrong about
    # a real business - if no key is configured or the city returns nothing. Neither
    # source carries a rating or a real nightly price for the specific property; the
    # cost shown elsewhere is the real ADR tier median. See places.py.
    suggested_hotels = find_real_hotels(destination_city, limit=3) if destination_city else []
    hotel_source = "real" if suggested_hotels else "none"

    if not suggested_hotels:
        _ai = ai_data()
        suggested_hotels_raw = (_ai.get("hotels", []) if _ai else [])[:3]
        hotel_source = "ai" if suggested_hotels_raw else "none"
        for h in suggested_hotels_raw:
            geo = geocode_hotel(h["name"], destination_city)
            suggested_hotels.append({
                **h,
                "map_url": osm_embed_url(geo["lat"], geo["lon"]) if geo else None,
                "map_approximate": geo.get("approximate", True) if geo else None,
                "source": "ai",
            })

    template = SIGHTSEEING_TEMPLATE.get(sightseeing_level, SIGHTSEEING_TEMPLATE["moderate"])
    sightseeing_per_day = round(budget["sightseeing_cost"] / days, 2)
    day_plan = []
    # Only full days actually visit spots - day 1 is arrival and the last day is
    # checkout, and both overwrite their activities below. Advancing the index on
    # those days too would spend the best (wiki-backed) landmarks on slots that are
    # then discarded, which is why day 2 used to open on a minor statue.
    sightseeing_day = 0
    for day_num in range(1, days + 1):
        food_pick = food_spots[(day_num - 1) % len(food_spots)] if food_spots else None
        dinner_line = f"Dinner at {food_pick['name']} ({food_pick['area']})" if food_pick else "Evening leisure walk nearby"

        is_full_day = day_num != 1 and day_num != days
        day_spots = []
        if sightseeing_spots and is_full_day:
            i = sightseeing_day * per_day
            # Take this day's slice; wrap only if the destination genuinely has fewer
            # real spots than the chosen pace needs (Geoapify returns what exists, and
            # never pads), so a packed trip to a small city repeats rather than invents.
            day_spots = [sightseeing_spots[(i + n) % len(sightseeing_spots)]
                         for n in range(min(per_day, len(sightseeing_spots)))]
            sightseeing_day += 1

        if day_spots:
            default_morning = f"{day_spots[0]['name']} ({day_spots[0]['area']})"
            rest = day_spots[1:]
            default_afternoon = (", then ".join(f"{s['name']} ({s['area']})" for s in rest)
                                 if rest else template["afternoon"])
        else:
            default_morning, default_afternoon = template["morning"], template["afternoon"]

        if day_num == 1:
            arrive_line = ("Arrive by car, check in" if transport_mode == "car"
                           else "Arrive, transfer to hotel, check in")
            morning, afternoon, evening = arrive_line, "Settle in, short neighborhood walk", dinner_line
            spend = round(sightseeing_per_day / 2, 2)
        elif day_num == days:
            breakfast_line = f"Breakfast at {food_pick['name']}" if food_pick else "Breakfast, last-minute shopping"
            # Wording has to match how they're actually travelling - telling a
            # driver to transfer to the airport is the same mistake as showing
            # them flight cards.
            depart_line = {
                "car": "Set off on the drive home",
                "flight": "Transfer to airport for return flight",
            }.get(transport_mode, "Head home")
            morning, afternoon, evening = breakfast_line, "Check out, pack", depart_line
            spend = round(sightseeing_per_day / 2, 2)
        else:
            morning, afternoon = default_morning, default_afternoon
            evening = dinner_line if food_pick else template["evening"]
            spend = sightseeing_per_day

        day_plan.append({
            "day": day_num, "morning": morning, "afternoon": afternoon, "evening": evening,
            "food_pick": food_pick, "suggested_spend_usd": spend,
        })

    # Real landmarks vs genuinely lesser-known real places - see sightseeing.py.
    # (AI-fallback spots carry no wiki flag, so they land in "hidden" and stay labelled
    # unverified by sightseeing_source below.)
    _common, _hidden = split_common_and_hidden(sightseeing_spots)

    food_per_day = round(budget["food_cost"] / days, 2)
    transport_per_day = round(budget["transport_cost"] / days, 2)

    return {
        "route_covered": route_covered,
        "route_reference": route_reference,
        "budget_breakdown": budget,
        "nights": nights,
        "destination_geocoded": destination_geo is not None,
        "auto_resolved_destination_airport": auto_resolved["airport_code"] if auto_resolved else None,
        "destination_airport_transfer": destination_airport_transfer,
        "outbound_flights": outbound_flights,
        "return_flights": return_flights,
        "hotel_options": hotel_options,
        "hotel_reality_check": hotel_reality_check,
        "suggested_hotels": suggested_hotels,
        "hotel_source": hotel_source,
        "destination_weather": weather,
        "food_source": food_source,
        "sightseeing_source": sightseeing_source,
        "spots_per_day": per_day,
        "sightseeing_common": _common,
        "sightseeing_hidden": _hidden,
        "daily_food_budget_usd": food_per_day,
        "daily_transport_budget_usd": transport_per_day,
        "day_plan": day_plan,
        # Genuine optimization metadata (itinerary_optimizer.py) - how many real
        # flight x hotel combinations were actually checked against the real
        # budget constraint, and whether one was actually found to fit.
        "optimizer_feasible": result["feasible"],
        "combinations_considered": result["combinations_considered"],
        "optimizer_shortfall_usd": result.get("shortfall_usd"),
    }
