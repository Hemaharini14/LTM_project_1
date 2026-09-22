"""
Module 2: Recovery Planning.

check_risk always runs first, deterministically - the real trained model
decides whether there's anything to recover from at all, and an LLM never
gets to skip or second-guess that number. Low risk stops here immediately,
same as before: no LLM call, no tool calls, just the real risk score.

For Moderate/High risk, an LLM agent (if an LLM is configured - see
llm_utils.py) decides WHICH of the real recovery tools to actually use and
in what order, instead of always running all four regardless of relevance:
search real alternative flights, search real hotel tiers, check real live
destination weather, reallocate the real budget. For example it can skip
hotel search for what looks like a same-day trip, or skip the weather check
if it wouldn't change anything. Every tool still only returns real data
from recovery_tools.py / budget_optimizer.py - the agent decides *whether*
to call a tool and narrates the outcome, it never invents what a tool
returns. Each tool's real result is also captured into a shared dict
(_collected, inside _agentic_recovery) so the caller gets the same
structured fields (alternative_flights, hotel_options, etc.) regardless of
what order the agent called things in - this is what keeps
templates/flight_delay.html unchanged even though the flow is now agentic.

Without an LLM key configured, falls back to _deterministic_recovery(): the
exact previous fixed sequence (call all four tools, deterministic summary)
- the recovery flow always works, agentic behavior is a bonus layer on top,
same principle as everywhere else in this app.
"""
import os
import sys
import json
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
load_dotenv()

from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage

from predict_delay_v2 import predict_delay_probability, risk_label
from recovery_tools import search_alternative_flights, search_hotel_options, get_destination_weather
from budget_optimizer import reallocate_budget, summarize_budget
from llm_utils import get_llm


def _check_risk(trip: dict) -> tuple[float, str]:
    prob = predict_delay_probability(
        carrier_code=trip["carrier_code"], origin_airport=trip["origin_airport"],
        destination_airport=trip["destination_airport"], weekday=trip["weekday"],
        month=trip["month"], scheduled_elapsed_time=trip["scheduled_elapsed_time"],
        origin_temp_f=trip["origin_temp_f"], origin_temp_known=trip.get("origin_temp_known", True),
        origin_precip_in=trip["origin_precip_in"],
        origin_pressure=trip["origin_pressure"], origin_visibility=trip["origin_visibility"],
        origin_wind_speed=trip["origin_wind_speed"],
        scheduled_hour=trip.get("scheduled_hour", 12), is_holiday=trip.get("is_holiday", False),
    )
    return prob, risk_label(prob)


def _deterministic_recovery(trip: dict, delay_probability: float, label: str) -> dict:
    """The exact previous fixed pipeline - used when no LLM is configured, so the
    recovery flow always works even without agentic behavior."""
    alts = search_alternative_flights(
        origin_airport=trip["origin_airport"], destination_airport=trip["destination_airport"],
        exclude_carrier=trip["carrier_code"], weekday=trip["weekday"],
        priority=trip.get("priority", "cost"), top_n=3,
    )
    nightly_budget = trip.get("budget", {}).get("hotel_cost", 100) / 3
    hotels = search_hotel_options(nightly_budget, trip.get("priority", "cost"), top_n=3)
    city = trip.get("destination_city")
    weather = get_destination_weather(city) if city else None
    budget = trip.get("budget", {})
    # No real fare data exists; use a simple proxy (10% of flight budget per
    # risk tier avoided) rather than a fabricated price difference.
    extra_cost = round(budget.get("flight_cost", 0) * 0.1, 2) if alts else 0.0
    new_budget, shortfall, trim_log = reallocate_budget(budget, extra_cost, trip.get("priority", "cost"))

    parts = [f"Delay risk is {delay_probability:.0%} ({label})."]
    if alts:
        parts.append(f"Alternative: {alts[0]['carrier']} flight, "
                      f"{alts[0]['risk_label'].lower()} risk ({alts[0]['delay_probability']:.0%}).")
    if hotels:
        parts.append(f"Hotel option: {hotels[0]['hotel_type']} room {hotels[0]['room_type']}, "
                      f"~${hotels[0]['median_nightly_rate_usd']}/night.")
    if weather:
        parts.append(f"Destination weather: {weather['condition']}, {weather['temperature_celsius']}°C.")
    parts.append(f"Budget adjusted by ${extra_cost:.2f} under '{trip.get('priority')}' priority.")

    return {
        "delay_probability": delay_probability, "risk_label": label,
        "alternative_flights": alts, "hotel_options": hotels, "destination_weather": weather,
        "extra_cost": extra_cost, "new_budget": summarize_budget(new_budget),
        "budget_shortfall": shortfall, "trim_log": trim_log,
        "recommendation_text": " ".join(parts), "mode": "deterministic (no LLM key configured)",
    }


def _summarize(delay_probability: float, label: str, data: dict, trip: dict) -> str:
    """Deterministic narration built only from real tool output - used when no LLM is
    configured and when the agent call fails, so there is always a real recommendation."""
    parts = [f"Delay risk is {delay_probability:.0%} ({label})."]
    alts = data.get("alternative_flights") or []
    if alts:
        a = alts[0]
        parts.append(f"Best alternative: {a['carrier']} {a['flight_number']}, "
                     f"{a['risk_label'].lower()} risk ({a['delay_probability']:.0%}).")
    else:
        parts.append("No real alternative flights were found for this route.")
    hotels = data.get("hotel_options") or []
    if hotels:
        h = hotels[0]
        parts.append(f"Hotel option: {h['hotel_type']} room {h['room_type']}, "
                     f"~${h['median_nightly_rate_usd']}/night.")
    weather = data.get("destination_weather")
    if weather:
        parts.append(f"Destination weather: {weather['condition']}, "
                     f"{weather['temperature_celsius']}°C.")
    if data.get("extra_cost"):
        parts.append(f"Budget adjusted by ${data['extra_cost']:.2f} under "
                     f"'{trip.get('priority', 'cost')}' priority.")
    return " ".join(parts)


def _agentic_recovery(trip: dict, delay_probability: float, label: str) -> dict:
    """LLM agent decides which real tools to call. Each tool stashes its real return
    value into `collected` (a closure variable) as a side effect, in addition to
    returning a text summary for the agent to reason over - so the final structured
    result is built from what actually happened, not from the agent's own account
    of what it did."""
    # A ReAct loop is several sequential calls, so the per-call cap has to be looser
    # than the single-shot narration default, plus one retry for a transient blip.
    llm = get_llm(timeout=40, max_retries=1)
    collected = {"alternative_flights": None, "hotel_options": None, "destination_weather": None,
                 "new_budget": None, "budget_shortfall": None, "trim_log": None, "extra_cost": 0.0,
                 "alternative_search_weekday": None, "hotel_search_nightly_budget": None}

    @tool
    def search_alternatives(weekday_offset: int = 0, priority: str = "") -> str:
        """Real alternative flights on this route, scored by the trained delay model.
        weekday_offset (-3..3): search a different real day if the first result is all
        high-risk. priority: 'time'|'cost'|'comfort', defaults to the trip's own."""
        weekday = (trip["weekday"] + max(-3, min(3, weekday_offset))) % 7
        use_priority = priority if priority in ("time", "cost", "comfort") else trip.get("priority", "cost")
        alts = search_alternative_flights(
            origin_airport=trip["origin_airport"], destination_airport=trip["destination_airport"],
            exclude_carrier=trip["carrier_code"], weekday=weekday, priority=use_priority, top_n=3,
        )
        collected["alternative_flights"] = alts
        collected["alternative_search_weekday"] = weekday
        if not alts:
            return f"No real alternative flights found for this route on weekday={weekday}."
        return ("\n".join(f"{a['carrier']} {a['flight_number']} - {a['risk_label']} risk "
                           f"({a['delay_probability']:.0%}), {a['scheduled_elapsed_time']}min" for a in alts)
                + f"\n(real historical flights for weekday={weekday}, ranked by '{use_priority}')")

    @tool
    def search_hotels(max_nightly_budget: float = -1.0, priority: str = "") -> str:
        """Real hotel tiers, only if an overnight stay is actually needed.
        max_nightly_budget: nightly cap to search against (-1 = the trip's own).
        priority: 'time'|'cost'|'comfort', defaults to the trip's own."""
        default_nightly = trip.get("budget", {}).get("hotel_cost", 100) / 3
        nightly_budget = max_nightly_budget if max_nightly_budget >= 0 else default_nightly
        use_priority = priority if priority in ("time", "cost", "comfort") else trip.get("priority", "cost")
        hotels = search_hotel_options(nightly_budget, use_priority, top_n=3)
        collected["hotel_options"] = hotels
        collected["hotel_search_nightly_budget"] = round(nightly_budget, 2)
        if not hotels:
            return f"No real hotel tier fits ${nightly_budget:.2f}/night."
        return ("\n".join(f"{h['hotel_type']} {h['room_type']} ~${h['median_nightly_rate_usd']}/night, "
                           f"{h['reliability_pct']}% reliable" for h in hotels)
                + f"\n(searched against ${nightly_budget:.2f}/night, ranked by '{use_priority}')")

    @tool
    def check_weather() -> str:
        """Real live weather at the destination city, if it would change the plan."""
        city = trip.get("destination_city")
        weather = get_destination_weather(city) if city else None
        collected["destination_weather"] = weather
        if not weather:
            return f"No live weather data available for '{city}'."
        return f"{weather['condition']}, {weather['temperature_celsius']}°C, updated {weather['last_updated']}."

    @tool
    def reallocate_trip_budget() -> str:
        """Reallocate the real budget for this disruption's extra cost. Call after
        search_alternatives (extra cost is zero if no alternative was found)."""
        budget = trip.get("budget", {})
        extra_cost = round(budget.get("flight_cost", 0) * 0.1, 2) if collected.get("alternative_flights") else 0.0
        new_budget, shortfall, trim_log = reallocate_budget(budget, extra_cost, trip.get("priority", "cost"))
        collected["extra_cost"] = extra_cost
        collected["new_budget"] = summarize_budget(new_budget)
        collected["budget_shortfall"] = shortfall
        collected["trim_log"] = trim_log
        return (f"Extra cost: ${extra_cost:.2f}. Shortfall: ${shortfall:.2f}. "
                + ("; ".join(trim_log) if trim_log else "No trimming needed."))

    tools = [search_alternatives, search_hotels, check_weather, reallocate_trip_budget]
    from langgraph.prebuilt import create_react_agent
    agent = create_react_agent(llm, tools)

    # Kept deliberately terse: a ReAct loop re-sends this prompt AND every tool schema
    # on every step, so wording here is multiplied by the number of steps. An earlier
    # verbose version cost ~1080 tokens/step and blew the provider's 8000 TPM budget.
    system = (
        "Travel disruption recovery agent. The trained model predicts a real "
        f"{delay_probability:.0%} delay risk ({label}). "
        "Call only the tools that matter here: skip hotels unless a late/long flight makes "
        "an overnight stay plausible, skip weather unless it would change the plan. "
        "After search_alternatives, call reallocate_trip_budget. "
        "Never invent a flight, hotel, price or weather value - report only what a tool returned. "
        f"Priority is '{trip.get('priority', 'cost')}': for 'time' or 'comfort' recommend the "
        "LOWEST delay_probability returned (NOT the shortest flight); for 'cost' prefer the "
        "shortest duration, as no real fare data exists. "
        "If every option comes back high-risk, you may retry search_alternatives with a "
        "different weekday_offset. "
        "Answer in under 80 words: the recommendation only, no preamble and no list of the "
        "tools you called."
    )
    trip_context = json.dumps({
        "priority": trip.get("priority"), "destination_city": trip.get("destination_city"),
        "budget": trip.get("budget"), "scheduled_hour": trip.get("scheduled_hour", 12),
        "scheduled_elapsed_time": trip.get("scheduled_elapsed_time"), "weekday": trip.get("weekday"),
    })
    try:
        # recursion_limit caps agent+tool node visits (~2 per step), bounding worst-case
        # token spend so one indecisive run can't exhaust the provider's per-minute budget.
        result = agent.invoke({"messages": [SystemMessage(content=system),
                                             HumanMessage(content=f"Trip context: {trip_context}")]},
                              {"recursion_limit": 10})
        recommendation_text = result["messages"][-1].content
        mode = f"agentic ({os.getenv('LLM_PROVIDER')})"
    except Exception as e:
        # The agent is a bonus layer, not the product: if the LLM times out or errors,
        # summarise the real tool results we already collected rather than showing the
        # traveler a raw exception string where the recommendation should be.
        print(f"[recovery_graph] agent failed, using deterministic summary: {e}")
        recommendation_text = _summarize(delay_probability, label, collected, trip)
        mode = "deterministic (LLM unavailable)"

    return {
        "delay_probability": delay_probability, "risk_label": label,
        "alternative_flights": collected["alternative_flights"] or [],
        "hotel_options": collected["hotel_options"] or [],
        "destination_weather": collected["destination_weather"],
        "extra_cost": collected["extra_cost"],
        "new_budget": collected["new_budget"],
        "budget_shortfall": collected["budget_shortfall"],
        "trim_log": collected["trim_log"] or [],
        "alternative_search_weekday": collected["alternative_search_weekday"],
        "hotel_search_nightly_budget": collected["hotel_search_nightly_budget"],
        "recommendation_text": recommendation_text,
        "mode": mode,
    }


def run_recovery(trip: dict) -> dict:
    delay_probability, label = _check_risk(trip)
    if label not in ("Moderate", "High"):
        return {
            "delay_probability": delay_probability, "risk_label": label,
            "recommendation_text": (f"Delay risk is {delay_probability:.0%} (Low) - "
                                     f"no recovery action needed. Proceed with your original booking."),
            "mode": "no-action",
        }
    llm = get_llm()
    if llm is None:
        return _deterministic_recovery(trip, delay_probability, label)
    return _agentic_recovery(trip, delay_probability, label)


class _RecoveryApp:
    """Thin shim so api.py's existing `_recovery_app.invoke(state)` call site (from when this
    was a compiled LangGraph StateGraph) doesn't need to change."""
    def invoke(self, state: dict) -> dict:
        return run_recovery(state)


def build_graph():
    return _RecoveryApp()


if __name__ == "__main__":
    sample_trip = {
        "carrier_code": "WN", "origin_airport": "LAX", "destination_airport": "SFO",
        "destination_city": "San Francisco", "weekday": 5, "month": 12,
        "scheduled_elapsed_time": 90,
        "origin_temp_f": 55, "origin_temp_known": True, "origin_precip_in": 0.3, "origin_pressure": 29.6,
        "origin_visibility": 3.0, "origin_wind_speed": 20,
        "priority": "time",
        "budget": {"flight_cost": 150, "hotel_cost": 400, "food_cost": 250,
                   "transport_cost": 150, "sightseeing_cost": 150},
    }
    result = run_recovery(sample_trip)
    print(json.dumps(result, indent=2, default=str))
