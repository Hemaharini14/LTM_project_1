"""
Genuine budget-constrained itinerary optimization - the actual "engine" part
of trip_planner.py. The old approach guessed a fixed % of the total budget
for each category, then independently searched within that guess (e.g. hotel
tiers filtered to a made-up nightly budget) - if nothing real matched, it
silently fell back to the cheapest tier without telling the user or
adjusting anything else.

This instead jointly searches the REAL discrete choices we actually have:
  - Real historical flight options (delay_probability from the trained
    model, scheduled_elapsed_time) - no real fare data exists for flights at
    all, so flight_cost stays a nominal budget line regardless of which real
    flight is chosen, same honesty as before.
  - Real hotel tiers (median_nightly_rate_usd, reliability_pct from
    cleaned_hotel_bookings.csv) - THIS is where real optimization applies:
    hotel_cost becomes an actual affordable real tier's real price, and
    whatever's left over (after real necessities) becomes the real
    sightseeing budget - not a fixed guessed percentage.

Food/transport have no real per-tier dataset the way hotels do, so they're
held at a fixed realistic share of the budget rather than searched.

If NO real hotel tier fits the budget at all, this reports the shortfall
honestly (same spirit as budget_optimizer.reallocate_budget) instead of
pretending something fits.
"""
import itertools

from recovery_tools import search_alternative_flights, get_all_hotel_tiers
from intl_reference import get_reference_flights

FOOD_SHARE = 0.12
TRANSPORT_SHARE = 0.08
# No real fare data exists for any flight in this project - flight_cost is a
# nominal placeholder line, unaffected by which real flight is chosen below.
FLIGHT_SHARE = 0.20

MAX_ALTERNATIVES = 2


def _score(ob: dict | None, rb: dict | None, hotel: dict, leftover: float, priority: str) -> float:
    """Higher is better. Every input is real: delay_probability from the trained model
    (only for whichever leg actually has real data - see optimize_itinerary), real ADR
    data for the hotel, and leftover from the traveler's own real total budget minus
    real/baseline necessities."""
    risks = [x["delay_probability"] for x in (ob, rb) if x and x.get("delay_probability") is not None]
    avg_delay_risk = sum(risks) / len(risks) if risks else 0.0
    if priority == "time":
        return -avg_delay_risk * 100 + leftover * 0.01
    elif priority == "comfort":
        return -avg_delay_risk * 50 + hotel["median_nightly_rate_usd"] * 0.5 + leftover * 0.02
    else:  # cost - maximize real discretionary money left over after real necessities
        return leftover - avg_delay_risk * 20


def _leg_candidates(origin: str, destination: str, weekday: int, priority: str) -> tuple[list[dict], bool]:
    """Real historical flights for ONE direction if this exact pairing has any (the two
    directions of a route are NOT symmetric in this project's real data - e.g. Dataset.csv
    only ever recorded DEL->HYD, never HYD->DEL, a genuine gap in the source, not a bug),
    else a curated cross-border reference schedule if one exists, else empty. Returns
    (candidates, is_real) so the caller can tell which case it got."""
    real = search_alternative_flights(origin_airport=origin, destination_airport=destination,
                                       exclude_carrier="", weekday=weekday, priority=priority, top_n=5)
    if real:
        return real, True
    return get_reference_flights(origin, destination), False


def optimize_itinerary(origin_airport: str, destination_airport: str,
                        total_budget: float, days: int,
                        start_weekday: int, return_weekday: int,
                        priority: str = "cost",
                        food_multiplier: float = 1.0, transport_multiplier: float = 1.0) -> dict:
    days = max(int(days), 1)
    nights = max(days - 1, 1)

    outbound_candidates, has_real_outbound = _leg_candidates(
        origin_airport, destination_airport, start_weekday, priority)
    return_candidates, has_real_return = _leg_candidates(
        destination_airport, origin_airport, return_weekday, priority)
    has_real_flight_data = has_real_outbound or has_real_return

    hotel_tiers = get_all_hotel_tiers()

    food_cost = round(total_budget * FOOD_SHARE * food_multiplier, 2)
    transport_cost = round(total_budget * TRANSPORT_SHARE * transport_multiplier, 2)
    flight_cost = round(total_budget * FLIGHT_SHARE, 2)
    remaining = total_budget - food_cost - transport_cost - flight_cost

    # Only the leg(s) with real data enter the joint search (so real delay-risk scoring
    # actually drives the pairing); a leg with no real data falls back to its reference
    # schedule (or None) as a single fixed placeholder, contributing no risk to scoring.
    ob_pool = outbound_candidates if has_real_outbound else [outbound_candidates[0] if outbound_candidates else None]
    rb_pool = return_candidates if has_real_return else [return_candidates[0] if return_candidates else None]

    combos = []
    for ob, rb, hotel in itertools.product(ob_pool, rb_pool, hotel_tiers):
        hotel_total = round(hotel["median_nightly_rate_usd"] * nights, 2)
        leftover = round(remaining - hotel_total, 2)
        if leftover < 0:
            continue
        combos.append({
            "outbound": ob, "return": rb, "hotel": hotel,
            "hotel_total_usd": hotel_total, "sightseeing_cost": leftover,
            "score": _score(ob, rb, hotel, leftover, priority),
        })

    if not combos:
        cheapest_hotel = min(hotel_tiers, key=lambda h: h["median_nightly_rate_usd"]) if hotel_tiers else None
        shortfall = (round(cheapest_hotel["median_nightly_rate_usd"] * nights - remaining, 2)
                     if cheapest_hotel else None)
        return {
            "feasible": False,
            "has_real_flight_data": has_real_flight_data,
            "combinations_considered": 0,
            "shortfall_usd": shortfall,
            "cheapest_hotel": cheapest_hotel,
            "outbound_flights": outbound_candidates,
            "return_flights": return_candidates,
            "budget_breakdown": {
                "flight_cost": flight_cost, "hotel_cost": 0.0, "food_cost": food_cost,
                "transport_cost": transport_cost, "sightseeing_cost": 0.0,
                "total": round(flight_cost + food_cost + transport_cost, 2),
            },
        }

    combos.sort(key=lambda c: c["score"], reverse=True)
    best = combos[0]
    # Alternatives with a genuinely different hotel tier, so the runner-ups aren't
    # just the same hotel paired with a slightly different flight time.
    alternatives = []
    seen_hotels = {best["hotel"]["hotel_type"] + best["hotel"]["room_type"]}
    for c in combos[1:]:
        key = c["hotel"]["hotel_type"] + c["hotel"]["room_type"]
        if key in seen_hotels:
            continue
        seen_hotels.add(key)
        alternatives.append(c)
        if len(alternatives) >= MAX_ALTERNATIVES:
            break

    def _leg_result(has_real, best_val, alt_vals, fallback):
        if not has_real:
            return fallback
        vals = [best_val] + alt_vals
        # Dedupe while preserving order (alternatives can repeat a flight across hotel tiers).
        seen, out = set(), []
        for v in vals:
            key = (v.get("carrier"), v.get("flight_number"), v.get("route"))
            if key not in seen:
                seen.add(key)
                out.append(v)
        return out

    return {
        "feasible": True,
        "has_real_flight_data": has_real_flight_data,
        "combinations_considered": len(combos),
        "outbound_flights": _leg_result(has_real_outbound, best["outbound"],
                                         [c["outbound"] for c in alternatives], outbound_candidates),
        "return_flights": _leg_result(has_real_return, best["return"],
                                       [c["return"] for c in alternatives], return_candidates),
        "hotel_options": [best["hotel"]] + [c["hotel"] for c in alternatives],
        "alternatives": [{"hotel": c["hotel"], "sightseeing_cost": c["sightseeing_cost"]} for c in alternatives],
        "budget_breakdown": {
            "flight_cost": flight_cost,
            "hotel_cost": best["hotel_total_usd"],
            "food_cost": food_cost,
            "transport_cost": transport_cost,
            "sightseeing_cost": best["sightseeing_cost"],
            "total": round(flight_cost + best["hotel_total_usd"] + food_cost
                            + transport_cost + best["sightseeing_cost"], 2),
        },
    }
