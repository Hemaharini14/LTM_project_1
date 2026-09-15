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

from llm_utils import get_llm
from recovery_tools import (
    search_alternative_flights, search_hotel_options, get_destination_weather,
    is_route_covered, get_dataset_index,
)
from predict_delay_v2 import predict_delay_probability, risk_label

_DEFAULT_WEATHER = {"temp_f": 55, "precip_in": 0.0, "pressure": 29.9, "visibility": 10.0, "wind_speed": 8}

SYSTEM_PROMPT = (
    "You are the SmartRouteAI travel assistant, embedded as a sidebar chatbot. "
    "You help with two things: checking flight delay risk, and planning a trip within a budget. "
    "The underlying dataset is US domestic flights from 2019 only (no international routes, no "
    "real-time fares, no named hotels or points of interest - only hotel price TIERS and generic "
    "sightseeing budgeting). Always call check_route_coverage before quoting flight data for a "
    "route, and tell the user plainly when a route/country isn't covered instead of guessing. "
    "Never invent a flight, price, hotel name, or attraction that a tool didn't return. Keep "
    "answers short and concrete."
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
        dest_temp_f=_DEFAULT_WEATHER["temp_f"], dest_precip_in=_DEFAULT_WEATHER["precip_in"],
        dest_pressure=_DEFAULT_WEATHER["pressure"], dest_visibility=_DEFAULT_WEATHER["visibility"],
        dest_wind_speed=_DEFAULT_WEATHER["wind_speed"],
        origin_temp_f=_DEFAULT_WEATHER["temp_f"], origin_precip_in=_DEFAULT_WEATHER["precip_in"],
        origin_pressure=_DEFAULT_WEATHER["pressure"], origin_visibility=_DEFAULT_WEATHER["visibility"],
        origin_wind_speed=_DEFAULT_WEATHER["wind_speed"],
    )
    return f"Estimated delay probability {prob:.0%} ({risk_label(prob)} risk), assuming typical clear weather."


@tool
def find_flights(origin_airport: str, destination_airport: str, weekday: int, priority: str = "cost") -> str:
    """Find real historical flight options for a route. priority is 'cost', 'time', or 'comfort'."""
    if not is_route_covered(origin_airport, destination_airport):
        return "This route isn't covered by the dataset - no flights to show."
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


_TOOLS = [check_route_coverage, check_flight_delay_risk, find_flights, find_hotels, check_destination_weather]
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
