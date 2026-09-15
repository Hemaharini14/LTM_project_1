"""
Module 4 (web app): end-to-end budget trip planning.

Combines the real-data tools already built for the recovery agent
(recovery_tools.py, budget_optimizer.py) into a single "plan a trip within
my budget" flow: outbound + return flight options, a hotel tier, a
category-wise cost breakdown, and a day-by-day activity template scaled to
the traveler's comfort levels. No fabricated flights/hotels/prices - those
always come from the real cleaned datasets or deterministic math. The
day-by-day plan is a generic template (no point-of-interest dataset exists
in this project), labelled as such rather than invented as real places.
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from recovery_tools import (
    search_alternative_flights, search_hotel_options, get_destination_weather, is_route_covered,
)
from budget_optimizer import suggest_realistic_hotel_budget, validate_hotel_budget
from intl_reference import get_reference_flights, food_spots_for
from llm_utils import suggest_food_and_sightseeing

# Base share of total budget per category before comfort-level adjustment
BASE_SHARE = {
    "flight_cost": 0.30,
    "hotel_cost": 0.30,
    "food_cost": 0.15,
    "transport_cost": 0.10,
    "sightseeing_cost": 0.15,
}

# Comfort level -> multiplier applied to that category's base share
LEVEL_MULTIPLIER = {"budget": 0.7, "standard": 1.0, "luxury": 1.4}

# comfort level -> priority keyword understood by recovery_tools
LEVEL_TO_PRIORITY = {"budget": "cost", "standard": "time", "luxury": "comfort"}

# sightseeing level -> generic morning/afternoon/evening activity template
# (no points-of-interest dataset exists in this project, so these are
# intentionally generic - named food picks, where available, come from
# intl_reference.food_spots_for() instead, kept separate from this template)
SIGHTSEEING_TEMPLATE = {
    "light": {"morning": "Free morning / rest", "afternoon": "One guided landmark or museum visit",
              "evening": "Leisure walk nearby"},
    "moderate": {"morning": "Landmark or museum visit", "afternoon": "Local market or neighborhood exploration",
                 "evening": "Leisure time / optional activity"},
    "packed": {"morning": "Early landmark visit", "afternoon": "Museum or cultural site, then local market",
               "evening": "Evening excursion or show"},
}


def _budget_split(total_budget: float, stay: str, travel: str, food: str, sightseeing: str) -> dict:
    levels = {
        "flight_cost": travel, "transport_cost": travel,
        "hotel_cost": stay,
        "food_cost": food,
        "sightseeing_cost": sightseeing,
    }
    weighted = {
        cat: BASE_SHARE[cat] * LEVEL_MULTIPLIER.get(levels[cat], 1.0)
        for cat in BASE_SHARE
    }
    total_weight = sum(weighted.values())
    return {cat: round(total_budget * (w / total_weight), 2) for cat, w in weighted.items()}


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
) -> dict:
    days = max(int(days), 1)
    nights = max(days - 1, 1)

    budget = _budget_split(total_budget, stay_comfort, travel_comfort, food_comfort, sightseeing_level)
    travel_priority = LEVEL_TO_PRIORITY.get(travel_comfort, "time")
    stay_priority = LEVEL_TO_PRIORITY.get(stay_comfort, "time")

    route_covered = is_route_covered(origin_airport, destination_airport)
    route_reference = False
    if route_covered:
        outbound_flights = search_alternative_flights(
            origin_airport=origin_airport, destination_airport=destination_airport,
            exclude_carrier="", weekday=start_weekday, priority=travel_priority, top_n=3,
        )
        return_flights = search_alternative_flights(
            origin_airport=destination_airport, destination_airport=origin_airport,
            exclude_carrier="", weekday=return_weekday, priority=travel_priority, top_n=3,
        )
    else:
        # Not in the trained (2019 US domestic) dataset - fall back to curated
        # reference schedules for known international routes, if any. These
        # carry no delay-risk score; the model has no signal for them.
        outbound_flights = get_reference_flights(origin_airport, destination_airport)
        return_flights = get_reference_flights(destination_airport, origin_airport)
        route_reference = bool(outbound_flights or return_flights)

    nightly_hotel_budget = budget["hotel_cost"] / nights
    hotel_options = search_hotel_options(nightly_hotel_budget, priority=stay_priority, top_n=3)
    hotel_reality_check = validate_hotel_budget(budget["hotel_cost"], nights, priority=stay_priority)

    weather = get_destination_weather(destination_city) if destination_city else None

    food_spots = food_spots_for(destination_city)
    sightseeing_spots = []
    food_source = "curated" if food_spots else "none"
    if not food_spots and destination_city:
        ai_data = suggest_food_and_sightseeing(destination_city)
        if ai_data:
            food_spots = ai_data.get("food_spots", [])
            sightseeing_spots = ai_data.get("sightseeing", [])
            if food_spots:
                food_source = "ai"
    sightseeing_source = "ai" if sightseeing_spots else "none"

    template = SIGHTSEEING_TEMPLATE.get(sightseeing_level, SIGHTSEEING_TEMPLATE["moderate"])
    sightseeing_per_day = round(budget["sightseeing_cost"] / days, 2)
    day_plan = []
    for day_num in range(1, days + 1):
        food_pick = food_spots[(day_num - 1) % len(food_spots)] if food_spots else None
        dinner_line = f"Dinner at {food_pick['name']} ({food_pick['area']})" if food_pick else "Evening leisure walk nearby"

        if sightseeing_spots:
            i = (day_num - 1) * 2
            spot_a = sightseeing_spots[i % len(sightseeing_spots)]
            spot_b = sightseeing_spots[(i + 1) % len(sightseeing_spots)] if len(sightseeing_spots) > 1 else None
            default_morning = f"{spot_a['name']} ({spot_a['area']})"
            default_afternoon = f"{spot_b['name']} ({spot_b['area']})" if spot_b else template["afternoon"]
        else:
            default_morning, default_afternoon = template["morning"], template["afternoon"]

        if day_num == 1:
            morning, afternoon, evening = "Arrive, transfer to hotel, check in", "Settle in, short neighborhood walk", dinner_line
            spend = round(sightseeing_per_day / 2, 2)
        elif day_num == days:
            breakfast_line = f"Breakfast at {food_pick['name']}" if food_pick else "Breakfast, last-minute shopping"
            morning, afternoon, evening = breakfast_line, "Check out, pack", "Transfer to airport for return flight"
            spend = round(sightseeing_per_day / 2, 2)
        else:
            morning, afternoon = default_morning, default_afternoon
            evening = dinner_line if food_pick else template["evening"]
            spend = sightseeing_per_day

        day_plan.append({
            "day": day_num, "morning": morning, "afternoon": afternoon, "evening": evening,
            "food_pick": food_pick, "suggested_spend_usd": spend,
        })

    food_per_day = round(budget["food_cost"] / days, 2)
    transport_per_day = round(budget["transport_cost"] / days, 2)

    return {
        "route_covered": route_covered,
        "route_reference": route_reference,
        "budget_breakdown": {**budget, "total": round(sum(budget.values()), 2)},
        "nights": nights,
        "outbound_flights": outbound_flights,
        "return_flights": return_flights,
        "hotel_options": hotel_options,
        "hotel_reality_check": hotel_reality_check,
        "destination_weather": weather,
        "food_source": food_source,
        "sightseeing_source": sightseeing_source,
        "daily_food_budget_usd": food_per_day,
        "daily_transport_budget_usd": transport_per_day,
        "day_plan": day_plan,
    }
