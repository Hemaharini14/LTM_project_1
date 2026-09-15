"""
Module 2: Recovery Planning, built as an explicit LangGraph state machine.

Flow:
  check_risk -> [branch: Low risk -> END | Moderate/High -> continue]
             -> search_flights -> search_hotels -> check_weather
             -> reallocate_budget -> generate_recommendation -> END

Every tool call is grounded in real cleaned data (recovery_tools.py) or a
deterministic formula (budget_optimizer.py). The LLM (if configured) is
only used for the final narrative - it never invents flights, hotels, or
prices; those come from the state populated by earlier nodes.
"""
import os
import sys
import json
from typing import TypedDict, Optional
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
load_dotenv()

from langgraph.graph import StateGraph, END

from predict_delay_v2 import predict_delay_probability, risk_label
from recovery_tools import search_alternative_flights, search_hotel_options, get_destination_weather
from budget_optimizer import reallocate_budget, summarize_budget
from llm_utils import get_llm


class TripState(TypedDict, total=False):
    # --- inputs ---
    carrier_code: str
    origin_airport: str
    destination_airport: str
    destination_city: str  # for the live weather lookup (may differ from airport code)
    weekday: int
    month: int
    scheduled_elapsed_time: float
    origin_temp_f: float
    origin_precip_in: float
    origin_pressure: float
    origin_visibility: float
    origin_wind_speed: float
    dest_temp_f: float
    dest_precip_in: float
    dest_pressure: float
    dest_visibility: float
    dest_wind_speed: float
    priority: str  # 'cost' | 'time' | 'comfort'
    budget: dict  # flight_cost, hotel_cost, food_cost, transport_cost, sightseeing_cost

    # --- populated as the graph runs ---
    delay_probability: float
    risk_label: str
    alternative_flights: list
    hotel_options: list
    destination_weather: Optional[dict]
    extra_cost: float
    new_budget: dict
    budget_shortfall: float
    trim_log: list
    recommendation_text: str
    mode: str


# ---------------------------------------------------------------- NODES

def check_risk(state: TripState) -> TripState:
    prob = predict_delay_probability(
        carrier_code=state["carrier_code"], origin_airport=state["origin_airport"],
        destination_airport=state["destination_airport"], weekday=state["weekday"],
        month=state["month"], scheduled_elapsed_time=state["scheduled_elapsed_time"],
        origin_temp_f=state["origin_temp_f"], origin_precip_in=state["origin_precip_in"],
        origin_pressure=state["origin_pressure"], origin_visibility=state["origin_visibility"],
        origin_wind_speed=state["origin_wind_speed"],
        dest_temp_f=state["dest_temp_f"], dest_precip_in=state["dest_precip_in"],
        dest_pressure=state["dest_pressure"], dest_visibility=state["dest_visibility"],
        dest_wind_speed=state["dest_wind_speed"],
    )
    return {"delay_probability": prob, "risk_label": risk_label(prob)}


def route_on_risk(state: TripState) -> str:
    """Conditional edge: Low risk skips straight to the end."""
    return "continue" if state["risk_label"] in ("Moderate", "High") else "stop"


def no_action_needed(state: TripState) -> TripState:
    return {
        "recommendation_text": (
            f"Delay risk is {state['delay_probability']:.0%} (Low) - "
            f"no recovery action needed. Proceed with your original booking."
        ),
        "mode": "no-action",
    }


def search_flights_node(state: TripState) -> TripState:
    alts = search_alternative_flights(
        origin_airport=state["origin_airport"], destination_airport=state["destination_airport"],
        exclude_carrier=state["carrier_code"], weekday=state["weekday"],
        priority=state.get("priority", "cost"), top_n=3,
    )
    return {"alternative_flights": alts}


def search_hotels_node(state: TripState) -> TripState:
    nightly_budget = state.get("budget", {}).get("hotel_cost", 100) / 3
    hotels = search_hotel_options(nightly_budget, state.get("priority", "cost"), top_n=3)
    return {"hotel_options": hotels}


def check_weather_node(state: TripState) -> TripState:
    city = state.get("destination_city")
    weather = get_destination_weather(city) if city else None
    return {"destination_weather": weather}


def reallocate_budget_node(state: TripState) -> TripState:
    budget = state.get("budget", {})
    alts = state.get("alternative_flights", [])
    extra_cost = 0.0
    if alts:
        # No real fare data exists; use a simple proxy (10% of flight budget
        # per risk tier avoided) rather than a fabricated price difference.
        extra_cost = round(budget.get("flight_cost", 0) * 0.1, 2)
    new_budget, shortfall, trim_log = reallocate_budget(budget, extra_cost, state.get("priority", "cost"))
    return {
        "extra_cost": extra_cost,
        "new_budget": summarize_budget(new_budget),
        "budget_shortfall": shortfall,
        "trim_log": trim_log,
    }


def generate_recommendation_node(state: TripState) -> TripState:
    llm = get_llm()
    best_flight = state["alternative_flights"][0] if state.get("alternative_flights") else None
    best_hotel = state["hotel_options"][0] if state.get("hotel_options") else None

    if llm is None:
        parts = [f"Delay risk is {state['delay_probability']:.0%} ({state['risk_label']})."]
        if best_flight:
            parts.append(f"Alternative: {best_flight['carrier']} flight, "
                         f"{best_flight['risk_label'].lower()} risk ({best_flight['delay_probability']:.0%}).")
        if best_hotel:
            parts.append(f"Hotel option: {best_hotel['hotel_type']} room {best_hotel['room_type']}, "
                         f"~${best_hotel['median_nightly_rate_usd']}/night.")
        if state.get("destination_weather"):
            w = state["destination_weather"]
            parts.append(f"Destination weather: {w['condition']}, {w['temperature_celsius']}°C.")
        parts.append(f"Budget adjusted by ${state.get('extra_cost', 0):.2f} under '{state.get('priority')}' priority.")
        return {"recommendation_text": " ".join(parts), "mode": "deterministic (no LLM key configured)"}

    from langchain_core.messages import SystemMessage, HumanMessage
    system = ("You are a travel recovery assistant. Summarize the disruption and recommendation "
              "in under 100 words using ONLY the data provided below. Do not invent flights, "
              "hotels, or prices beyond what's given.")
    human = json.dumps({
        "delay_probability": state["delay_probability"], "risk_label": state["risk_label"],
        "best_alternative_flight": best_flight, "best_hotel_option": best_hotel,
        "destination_weather": state.get("destination_weather"),
        "budget_adjustment": {"extra_cost": state.get("extra_cost"), "new_budget": state.get("new_budget")},
        "priority": state.get("priority"),
    })
    try:
        result = llm.invoke([SystemMessage(content=system), HumanMessage(content=human)])
        return {"recommendation_text": result.content, "mode": f"LLM ({os.getenv('LLM_PROVIDER')})"}
    except Exception as e:
        return {"recommendation_text": f"[LLM error: {e}] Falling back to raw data above.",
                "mode": "error-fallback"}


# ---------------------------------------------------------------- GRAPH

def build_graph():
    graph = StateGraph(TripState)
    graph.add_node("check_risk", check_risk)
    graph.add_node("no_action_needed", no_action_needed)
    graph.add_node("search_flights", search_flights_node)
    graph.add_node("search_hotels", search_hotels_node)
    graph.add_node("check_weather", check_weather_node)
    graph.add_node("reallocate_budget", reallocate_budget_node)
    graph.add_node("generate_recommendation", generate_recommendation_node)

    graph.set_entry_point("check_risk")
    graph.add_conditional_edges("check_risk", route_on_risk, {
        "stop": "no_action_needed",
        "continue": "search_flights",
    })
    graph.add_edge("no_action_needed", END)
    graph.add_edge("search_flights", "search_hotels")
    graph.add_edge("search_hotels", "check_weather")
    graph.add_edge("check_weather", "reallocate_budget")
    graph.add_edge("reallocate_budget", "generate_recommendation")
    graph.add_edge("generate_recommendation", END)

    return graph.compile()


if __name__ == "__main__":
    app = build_graph()
    sample_trip = {
        "carrier_code": "WN", "origin_airport": "LAX", "destination_airport": "SFO",
        "destination_city": "San Francisco", "weekday": 5, "month": 12,
        "scheduled_elapsed_time": 90,
        "origin_temp_f": 55, "origin_precip_in": 0.3, "origin_pressure": 29.6,
        "origin_visibility": 3.0, "origin_wind_speed": 20,
        "dest_temp_f": 58, "dest_precip_in": 0.1, "dest_pressure": 29.8,
        "dest_visibility": 6.0, "dest_wind_speed": 12,
        "priority": "time",
        "budget": {"flight_cost": 150, "hotel_cost": 400, "food_cost": 250,
                   "transport_cost": 150, "sightseeing_cost": 150},
    }
    result = app.invoke(sample_trip)
    print(json.dumps({k: v for k, v in result.items() if k not in
                       ("origin_temp_f","origin_precip_in","origin_pressure","origin_visibility",
                        "origin_wind_speed","dest_temp_f","dest_precip_in","dest_pressure",
                        "dest_visibility","dest_wind_speed")}, indent=2, default=str))