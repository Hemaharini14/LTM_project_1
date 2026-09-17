"""
The sidebar chatbot: a real LangGraph ReAct agent (not a scripted bot) that
can call the same grounded tools the rest of the app uses - it looks up
real flights, real hotel tiers, real destination weather, and checks
whether a route is even in the dataset before answering, rather than
guessing. Requires an LLM key (see llm_utils.py / .env.example); with none
configured it returns a plain explanation instead of crashing.
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from llm_utils import get_llm, suggest_destination_content
from recovery_tools import (
    search_alternative_flights, search_hotel_options, get_destination_weather,
    is_route_covered, get_dataset_index,
)
from intl_reference import get_reference_flights, food_spots_for
from predict_delay_v2 import predict_delay_probability, risk_label

_DEFAULT_WEATHER = {"temp_f": 55, "precip_in": 0.0, "pressure": 29.9, "visibility": 10.0, "wind_speed": 8}

SYSTEM_PROMPT = (
    "You are the SmartRouteAI travel assistant, embedded as a sidebar chatbot. "
    "You help with two things: checking flight delay risk, and planning a trip within a budget. "
    "The underlying dataset is US domestic flights from 2019 plus real India domestic flights "
    "(BLR/BOM/CCU/DEL/HYD, 2019-2020) - no international/cross-border routes, no real-time fares, "
    "no named hotels or points of interest (only hotel price TIERS and generic sightseeing "
    "budgeting). Always call check_route_coverage before quoting flight data for a route, and "
    "tell the user plainly when a route/country isn't covered instead of guessing. Never invent "
    "a flight, price, hotel name, or attraction that a tool didn't return. Keep answers short and "
    "concrete. If check_route_coverage says a route isn't covered, still call find_reference_flights "
    "before telling the user there's no flight info at all - cross-border routes (India<->Singapore/"
    "Malaysia/UAE) have a curated reference schedule (carrier and typical timing only, explicitly "
    "NOT a delay-risk prediction) even though the trained model has no risk score for them. "
    "Distinguish these two cases clearly to the user."
)


@tool
def check_route_coverage(origin_airport: str, destination_airport: str) -> str:
    """Check whether both airports (IATA codes) exist in the training dataset at all."""
    covered = is_route_covered(origin_airport, destination_airport)
    if covered:
        return f"{origin_airport.upper()} and {destination_airport.upper()} are both covered by the dataset."
    index = get_dataset_index()
    return (f"NOT COVERED: this dataset only contains 2019 US domestic airports "
            f"({len(index['airports'])} of them). At least one of {origin_airport}/{destination_airport} "
            f"is not in it - tell the user we have no flight data for this route/country.")


@tool
def check_flight_delay_risk(carrier_code: str, origin_airport: str, destination_airport: str,
                             month: int, weekday: int, scheduled_elapsed_time: float = 120) -> str:
    """Predict delay risk for a flight using typical (clear-weather) conditions, since exact
    weather isn't available in chat. carrier_code/origin_airport/destination_airport are IATA
    codes, month is 1-12, weekday is 0=Monday..6=Sunday."""
    if not is_route_covered(origin_airport, destination_airport):
        return "This route isn't covered by the dataset - no prediction can be made."
    prob = predict_delay_probability(
        carrier_code=carrier_code, origin_airport=origin_airport, destination_airport=destination_airport,
        weekday=weekday, month=month, scheduled_elapsed_time=scheduled_elapsed_time,
        origin_temp_f=_DEFAULT_WEATHER["temp_f"], origin_temp_known=False,
        origin_precip_in=_DEFAULT_WEATHER["precip_in"],
        origin_pressure=_DEFAULT_WEATHER["pressure"], origin_visibility=_DEFAULT_WEATHER["visibility"],
        origin_wind_speed=_DEFAULT_WEATHER["wind_speed"],
    )
    return f"Estimated delay probability {prob:.0%} ({risk_label(prob)} risk), assuming typical clear weather."


@tool
def find_flights(origin_airport: str, destination_airport: str, weekday: int, priority: str = "cost") -> str:
    """Find real historical flight options for a route, scored by the trained delay model.
    priority is 'cost', 'time', or 'comfort'. If this returns "not covered", call
    find_reference_flights next before concluding there's nothing to show."""
    if not is_route_covered(origin_airport, destination_airport):
        return "NOT COVERED by the trained model - call find_reference_flights before answering."
    options = search_alternative_flights(origin_airport, destination_airport, exclude_carrier="",
                                          weekday=weekday, priority=priority, top_n=3)
    if not options:
        return "No matching historical flights found for that route/day combination."
    return "\n".join(
        f"{o['carrier']} {o['flight_number']} · {o['route']} · {o['scheduled_elapsed_time']}min · "
        f"{o['risk_label']} risk ({o['delay_probability']:.0%}) - no fare data available"
        for o in options
    )


@tool
def find_reference_flights(origin_airport: str, destination_airport: str) -> str:
    """For routes NOT covered by the trained model (e.g. India/Singapore/Malaysia): look up a
    curated reference schedule (real airlines, typical duration/time-of-day). This is NOT a
    delay-risk prediction - always say so. Returns empty if no reference data exists either."""
    options = get_reference_flights(origin_airport, destination_airport)
    if not options:
        return "No reference schedule either - genuinely no flight data at all for this route."
    return "\n".join(
        f"{o['carrier']} · {o['route']} · ~{o['scheduled_elapsed_time']}min · "
        f"typical departure: {o['typical_departure']} - REFERENCE SCHEDULE ONLY, not a delay-risk prediction"
        for o in options
    )


@tool
def find_hotels(nightly_budget: float, priority: str = "cost") -> str:
    """Find hotel price tiers (not named properties - the dataset has no hotel names) within a
    nightly budget. priority is 'cost', 'time' (reliability), or 'comfort'."""
    options = search_hotel_options(nightly_budget, priority=priority, top_n=3)
    if not options:
        return "No hotel tier matched that nightly budget."
    return "\n".join(
        f"{h['hotel_type']} · room {h['room_type']} · ~${h['median_nightly_rate_usd']}/night · "
        f"{h['cancellation_rate']:.0%} historical cancellation rate"
        for h in options
    )


@tool
def check_destination_weather(city: str) -> str:
    """Look up current live weather for a destination city, if available."""
    weather = get_destination_weather(city)
    if not weather:
        return f"No live weather data available for '{city}'."
    return f"{weather['condition']}, {weather['temperature_celsius']}°C, updated {weather['last_updated']}."


@tool
def find_food_and_sightseeing(city: str) -> str:
    """Find food spots and sightseeing highlights for a city. Tries the curated (verified) list
    first; if nothing curated exists, falls back to an LLM suggestion - always tell the user which
    kind it is (curated real data vs. AI-suggested/unverified) using the label this tool returns."""
    curated = food_spots_for(city)
    if curated:
        lines = [f"{s['name']} ({s['area']}) - {s['note']}" for s in curated]
        return "CURATED, VERIFIED food spots:\n" + "\n".join(lines)
    ai_data = suggest_destination_content(city)
    if not ai_data or not (ai_data.get("food_spots") or ai_data.get("sightseeing")):
        return f"No curated data and no AI suggestions available for '{city}'."
    lines = [f"AI-SUGGESTED (unverified) food spot: {s['name']} ({s['area']}) - {s['note']}"
             for s in ai_data.get("food_spots", [])]
    lines += [f"AI-SUGGESTED (unverified) sightseeing: {s['name']} ({s['area']}) - {s['note']}"
              for s in ai_data.get("sightseeing", [])]
    return "\n".join(lines)


_TOOLS = [check_route_coverage, check_flight_delay_risk, find_flights, find_reference_flights,
          find_hotels, check_destination_weather, find_food_and_sightseeing]
_AGENT = None


def _get_agent():
    global _AGENT
    if _AGENT is not None:
        return _AGENT
    llm = get_llm()
    if llm is None:
        return None
    from langgraph.prebuilt import create_react_agent
    _AGENT = create_react_agent(llm, _TOOLS)
    return _AGENT


def run_chat(history: list[dict], message: str) -> str:
    """history: list of {"role": "user"|"assistant", "content": str}. Returns the reply text."""
    agent = _get_agent()
    if agent is None:
        return (
            "The chatbot needs an LLM key to reason with. Set LLM_PROVIDER and the matching API "
            "key (OpenAI, Anthropic, or Grok) in a .env file in the project root, then restart "
            "the app - see .env.example."
        )

    messages = [SystemMessage(content=SYSTEM_PROMPT)]
    for turn in history[-10:]:
        cls = HumanMessage if turn.get("role") == "user" else AIMessage
        messages.append(cls(content=turn.get("content", "")))
    messages.append(HumanMessage(content=message))

    try:
        result = agent.invoke({"messages": messages})
        final = result["messages"][-1]
        return final.content
    except Exception as e:
        return f"Sorry, the assistant hit an error: {e}"
