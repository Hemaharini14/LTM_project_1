"""
Trip planning pipeline, as a LangGraph state machine.

    select_flight -> check_disruption -> [recovery] -> allocate_budget -> assemble

Nothing here re-implements the existing modules; it sequences them:

  Module 1  predict_delay_v2      scores every candidate flight (via the
                                  India adapter, which fills the features the
                                  catalogue cannot supply and records which)
  Module 2  recovery principle    when the chosen flight is High risk, search
                                  the alternatives and take the lowest real
                                  risk that still fits - the same "compare real
                                  delay_probability, pick the lowest" rule
                                  recovery_graph's agent follows
  Module 3  budget_optimizer      validate_hotel_budget grounds the nightly
                                  rate against real ADR medians, so "this fits"
                                  is checked against actual market pricing
                                  rather than only against the catalogue

Budget rules, applied in the order the spec sets out: default 3 nights, then
trim in increasing order of pain - drop paid attractions first, then reduce
nights to a floor of 1, then downgrade the hotel tier. If the cheapest possible
plan still exceeds the budget we return it anyway, flagged OVER_BUDGET, because
a traveller is better served by "this is the least it can cost" than by a
refusal.
"""
from __future__ import annotations

import os
import sys
from datetime import date
from typing import Any, Literal, TypedDict

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langgraph.graph import StateGraph, END  # noqa: E402

from adapters.india_adapter import (  # noqa: E402
    find_flights, hotels_for, attractions_for, score_flight,
)
from budget_optimizer import validate_hotel_budget  # noqa: E402

# Calibrated scale (see predict_delay_v2.risk_label): >=0.30 is High, i.e. about
# twice the ~18% base rate. That is the point at which looking for an
# alternative is worth the traveller's attention.
RISK_THRESHOLD = 0.30
DEFAULT_NIGHTS = 3
MIN_NIGHTS = 1

Status = Literal["READY", "OVER_BUDGET", "NO_OPTIONS"]


class TripState(TypedDict, total=False):
    # request
    from_city: str
    to_city: str
    travel_date: date
    travelers: int
    budget: float
    # working
    candidates: list[dict]
    flight: dict | None
    disruption: dict | None
    was_rerouted: bool
    hotel: dict | None
    nights: int
    attractions: list[dict]
    trim_log: list[str]
    status: Status
    message: str


# ----------------------------------------------------------------- nodes

def select_flight(state: TripState) -> TripState:
    """Cheapest catalogue flight for the route, with the rest kept as fallbacks."""
    candidates = find_flights(state["from_city"], state["to_city"])
    if not candidates:
        return {**state, "candidates": [], "flight": None, "status": "NO_OPTIONS",
                "message": f"No flights found between {state['from_city']} and {state['to_city']}."}
    return {**state, "candidates": candidates, "flight": candidates[0], "was_rerouted": False}


def check_disruption(state: TripState) -> TripState:
    """Module 1 on the selected flight."""
    if not state.get("flight"):
        return state
    return {**state, "disruption": score_flight(state["flight"], state["travel_date"])}


def recovery(state: TripState) -> TripState:
    """Module 2's rule: score every alternative, take the genuinely lowest risk.

    Only swaps if an alternative is materially safer (>=5 points of real
    probability), so we don't reroute a traveller onto a different flight for a
    rounding difference - and never onto one that busts the flight budget.
    """
    current = state["flight"]
    current_risk = state["disruption"]["disruption_risk"]
    budget, travelers = state["budget"], state["travelers"]

    best, best_score = current, state["disruption"]
    for alt in state["candidates"]:
        if alt["number"] == current["number"]:
            continue
        if alt["fare_inr"] * travelers > budget:
            continue                                   # cannot fund it anyway
        scored = score_flight(alt, state["travel_date"])
        if scored["disruption_risk"] < best_score["disruption_risk"] - 0.05:
            best, best_score = alt, scored

    rerouted = best["number"] != current["number"]
    return {
        **state, "flight": best, "disruption": best_score, "was_rerouted": rerouted,
        "trim_log": state.get("trim_log", []) + (
            [f"Rerouted to {best['number']} - real delay risk "
             f"{best_score['disruption_risk']:.0%} vs {current_risk:.0%}"] if rerouted else []
        ),
    }


def allocate_budget(state: TripState) -> TripState:
    """Module 3: fit hotel + attractions into what the flight leaves behind."""
    travelers, budget = state["travelers"], state["budget"]
    flight_total = state["flight"]["fare_inr"] * travelers
    remaining = budget - flight_total

    tiers = hotels_for(state["to_city"])           # cheapest first
    sights = attractions_for(state["to_city"])[:5]
    trim: list[str] = list(state.get("trim_log", []))

    # Start generous: best tier we could afford at the default stay, all sights.
    nights = DEFAULT_NIGHTS
    chosen = tiers[0] if tiers else None
    for t in tiers:
        if t["price_per_night"] * nights <= remaining * 0.75:
            chosen = t
    selected = list(sights)

    def total_of(hotel, n, sels):
        h = hotel["price_per_night"] * n if hotel else 0
        return flight_total + h + sum(s["est_cost"] for s in sels) * travelers

    # Trim in increasing order of pain: paid sights, then nights, then tier.
    if chosen:
        while total_of(chosen, nights, selected) > budget and any(s["est_cost"] > 0 for s in selected):
            drop = max(selected, key=lambda s: s["est_cost"])
            selected.remove(drop)
            trim.append(f"Dropped {drop['name']} (Rs{drop['est_cost']}/person)")

        while total_of(chosen, nights, selected) > budget and nights > MIN_NIGHTS:
            nights -= 1
            trim.append(f"Reduced stay to {nights} night{'s' if nights != 1 else ''}")

        idx = tiers.index(chosen)
        while total_of(chosen, nights, selected) > budget and idx > 0:
            idx -= 1
            chosen = tiers[idx]
            trim.append(f"Downgraded to {chosen['tier']}")

    total = total_of(chosen, nights, selected)
    status: Status = "READY" if total <= budget else "OVER_BUDGET"

    return {**state, "hotel": chosen, "nights": nights, "attractions": selected,
            "trim_log": trim, "status": status}


def assemble(state: TripState) -> TripState:
    """Final human-readable summary. Never asserts more than the data supports."""
    if state.get("status") == "NO_OPTIONS":
        return state

    f, h = state["flight"], state.get("hotel")
    travelers = state["travelers"]
    total = (f["fare_inr"] * travelers
             + (h["price_per_night"] * state["nights"] if h else 0)
             + sum(a["est_cost"] for a in state["attractions"]) * travelers)

    bits = [f"{f['airline']} {f['number']} departs {f['depart']}"]
    if state.get("was_rerouted"):
        bits.append("rerouted to a lower-risk flight")
    if h:
        bits.append(f"{state['nights']} night{'s' if state['nights'] != 1 else ''} at {h['name']}")
    if state["attractions"]:
        bits.append(f"{len(state['attractions'])} attractions")

    if state["status"] == "OVER_BUDGET":
        bits.append(f"cheapest possible plan still Rs{total - state['budget']:,.0f} over budget")
    elif not state["disruption"]["modelled"]:
        # Say it here rather than let the number stand unqualified.
        bits.append("delay risk is indicative - this route is outside the trained data")

    return {**state, "message": ", ".join(bits) + "."}


# ----------------------------------------------------------------- graph

def _needs_recovery(state: TripState) -> str:
    if not state.get("flight") or not state.get("disruption"):
        return "end"
    return "recovery" if state["disruption"]["disruption_risk"] > RISK_THRESHOLD else "allocate"


def build_trip_graph():
    g = StateGraph(TripState)
    g.add_node("select_flight", select_flight)
    g.add_node("check_disruption", check_disruption)
    g.add_node("recovery", recovery)
    g.add_node("allocate_budget", allocate_budget)
    g.add_node("assemble", assemble)

    g.set_entry_point("select_flight")
    g.add_conditional_edges(
        "select_flight",
        lambda s: "end" if not s.get("flight") else "check",
        {"check": "check_disruption", "end": END},
    )
    g.add_conditional_edges(
        "check_disruption", _needs_recovery,
        {"recovery": "recovery", "allocate": "allocate_budget", "end": END},
    )
    g.add_edge("recovery", "allocate_budget")
    g.add_edge("allocate_budget", "assemble")
    g.add_edge("assemble", END)
    return g.compile()


_GRAPH = None


def plan_trip(from_city: str, to_city: str, travel_date: date,
              travelers: int, budget: float) -> dict:
    """Runs the pipeline and returns the API response body."""
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_trip_graph()

    out: dict[str, Any] = _GRAPH.invoke({
        "from_city": from_city, "to_city": to_city, "travel_date": travel_date,
        "travelers": travelers, "budget": float(budget),
        "was_rerouted": False, "trim_log": [], "attractions": [], "nights": DEFAULT_NIGHTS,
    })

    route_label = f"{from_city} → {to_city}"
    if out.get("status") == "NO_OPTIONS" or not out.get("flight"):
        return {"status": "NO_OPTIONS", "route_label": route_label, "flight": None,
                "hotel": None, "attractions": [], "total": 0, "budget": float(budget),
                "under_budget": True,
                "message": out.get("message", "No flights found for this route.")}

    f, h = out["flight"], out.get("hotel")
    travelers = out["travelers"]
    hotel_total = h["price_per_night"] * out["nights"] if h else 0
    sights_total = sum(a["est_cost"] for a in out["attractions"]) * travelers
    total = f["fare_inr"] * travelers + hotel_total + sights_total

    # Real ADR check (Module 3): is this nightly rate plausible for a real stay?
    reality = validate_hotel_budget(hotel_total, out["nights"]) if h else None

    return {
        "status": out["status"],
        "route_label": route_label,
        "flight": {
            "number": f["number"], "airline": f["airline"],
            "depart": f["depart"], "arrive": f["arrive"],
            "price_per_person": f["fare_inr"],
            "price_total": f["fare_inr"] * travelers,
            "disruption_risk": out["disruption"]["disruption_risk"],
            "was_rerouted": bool(out.get("was_rerouted")),
            # extra, additive: lets the UI mark an unmodelled route honestly
            "risk_is_modelled": out["disruption"]["modelled"],
        },
        "hotel": ({"name": h["name"], "nights": out["nights"],
                   "price_per_night": h["price_per_night"], "price_total": hotel_total,
                   "market_check": reality.get("verdict") if reality else None}
                  if h else None),
        "attractions": [{"name": a["name"], "est_cost": a["est_cost"]} for a in out["attractions"]],
        "total": total,
        "budget": float(budget),
        "under_budget": total <= budget,
        "message": out.get("message", ""),
        "adjustments": out.get("trim_log", []),
    }
