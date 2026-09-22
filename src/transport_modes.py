"""
Compares how to actually get from A to B, for the budget trip planner's
"pick your mode of transport" step.

Only modes with a real data source are offered:

  car     - real road distance and driving time from the Geoapify Routing API
            (same free key as sightseeing.py). Cost is NOT a looked-up price -
            no fare data source exists - it's fuel arithmetic from the real
            distance using the two named assumptions below, reported with the
            basis string so the traveler can see exactly what it assumes.

  flight  - real scheduled flight time and a real trained-model delay risk for
            the nearest covered airports (recovery_tools). There is no fare data
            anywhere in this project, so flight cost is None, never a guess.

  train /
  bus     - reported as unavailable, with the reason. Geoapify has no train mode
            at all, and its "bus" mode is just a bus-shaped vehicle on the same
            roads (it returns identical numbers to car - verified), not a real
            bus service with stops, timetable or fare. Presenting either as a
            real option would mean inventing the times and the prices, so this
            says "no data source" instead. Wire in a real rail/coach API and
            they can join the comparison honestly.
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from maps import geocode_city, route_between, nearest_supported_airport
from recovery_tools import get_dataset_index, search_alternative_flights
from intl_reference import INTL_AIRPORT_CODES, get_reference_flights

# Named, auditable assumptions behind the car cost estimate. These are NOT looked
# up from any dataset - change them to match local fuel prices. Anything derived
# from them is reported as an estimate with its basis attached, never as a price.
FUEL_LITRES_PER_100KM = 7.5
FUEL_COST_PER_LITRE_USD = 1.20


def _car_option(origin_geo: dict, dest_geo: dict) -> dict:
    route = route_between(origin_geo["lat"], origin_geo["lon"], dest_geo["lat"], dest_geo["lon"], mode="drive")
    if not route:
        return {"mode": "car", "available": False,
                "note": "No drivable route found between these two places."}

    litres = route["distance_km"] * FUEL_LITRES_PER_100KM / 100
    return {
        "mode": "car",
        "available": True,
        "distance_km": route["distance_km"],
        "duration_hours": round(route["duration_min"] / 60, 2),
        "estimated_cost_usd": round(litres * FUEL_COST_PER_LITRE_USD, 2),
        "cost_basis": (f"fuel only, at {FUEL_LITRES_PER_100KM} L/100km and "
                       f"${FUEL_COST_PER_LITRE_USD:.2f}/L - excludes tolls, parking and wear"),
        "note": "Real road route and driving time (Geoapify).",
    }


def _flight_option(origin_place: str, dest_place: str, weekday: int) -> dict:
    allowed = set(get_dataset_index()["airports"]) | INTL_AIRPORT_CODES
    o = nearest_supported_airport(origin_place, allowed)
    d = nearest_supported_airport(dest_place, allowed)
    if not o or not d:
        return {"mode": "flight", "available": False,
                "note": "Couldn't match these places to airports with real flight data."}
    if o["airport_code"] == d["airport_code"]:
        return {"mode": "flight", "available": False,
                "note": f"Both places use the same airport ({o['airport_code']}) - flying makes no sense here."}

    flights = search_alternative_flights(origin_airport=o["airport_code"],
                                          destination_airport=d["airport_code"],
                                          exclude_carrier="", weekday=weekday,
                                          priority="time", top_n=1)
    reference_only = False
    if not flights:
        flights = get_reference_flights(o["airport_code"], d["airport_code"])
        reference_only = True
    if not flights:
        return {"mode": "flight", "available": False,
                "note": f"No flight data covers {o['airport_code']} to {d['airport_code']}."}

    best = flights[0]
    return {
        "mode": "flight",
        "available": True,
        "origin_airport": o["airport_code"],
        "destination_airport": d["airport_code"],
        "distance_km": None,
        "duration_hours": round(best["scheduled_elapsed_time"] / 60, 2),
        # No fare data exists anywhere in this project - see the module docstring.
        "estimated_cost_usd": None,
        "cost_basis": None,
        "delay_probability": best.get("delay_probability"),
        "risk_label": best.get("risk_label"),
        "carrier": best.get("carrier"),
        "flight_number": best.get("flight_number"),
        "reference_only": reference_only,
        "note": ("Reference schedule only - this route isn't in the trained data, so no delay risk."
                 if reference_only else
                 "Real scheduled flight time and trained-model delay risk. Airport transfers not included."),
    }


def _unavailable(mode: str, reason: str, unsupported: bool = False) -> dict:
    """unsupported=True means this project has no data source for the mode at all,
    as opposed to a mode we can price but that doesn't work for this particular
    route. The UI hides the former (a permanent "no train data" card is noise on
    every search) and shows the latter, which is real information about the trip."""
    return {"mode": mode, "available": False, "unsupported": unsupported, "note": reason}


def compare_transport_modes(origin_place: str, destination_place: str, weekday: int = 0) -> list[dict]:
    """Every way of making this trip that we have real data for, plus an honest
    'no data source' entry for the ones we don't. Never invents a time or a price."""
    origin_geo = geocode_city(origin_place) if origin_place else None
    dest_geo = geocode_city(destination_place) if destination_place else None

    options = []
    if origin_geo and dest_geo:
        options.append(_car_option(origin_geo, dest_geo))
    else:
        missing = origin_place if not origin_geo else destination_place
        options.append(_unavailable("car", f"Couldn't find '{missing}' on the map."))

    options.append(_flight_option(origin_place, destination_place, weekday))
    options.append(_unavailable(
        "train", "No rail timetable or fare data source is configured, so a train time "
                 "or price here would be invented rather than looked up.", unsupported=True))
    options.append(_unavailable(
        "bus", "No coach timetable or fare data source is configured. Road routing can only "
               "time a bus-sized vehicle driving the route, which is not a real bus service.",
        unsupported=True))
    return options


def describe_mode(origin_place: str, destination_place: str, mode: str, weekday: int = 0) -> dict | None:
    """Just the one mode the traveller picked, so the itinerary can show its real
    numbers (and so the plan doesn't show flights to someone who chose to drive)."""
    for opt in compare_transport_modes(origin_place, destination_place, weekday):
        if opt["mode"] == mode:
            return opt
    return None


if __name__ == "__main__":
    for opt in compare_transport_modes("Bengaluru", "Chennai", weekday=2):
        if opt["available"]:
            cost = f"${opt['estimated_cost_usd']}" if opt.get("estimated_cost_usd") is not None else "no fare data"
            print(f"{opt['mode']:7s} {opt['duration_hours']:>5.2f} h  {cost:>14s}  {opt['note']}")
        else:
            print(f"{opt['mode']:7s} unavailable - {opt['note']}")
