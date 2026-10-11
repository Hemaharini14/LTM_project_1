"""
Tests for POST /api/plan-trip.

These go through the Flask endpoint rather than calling the graph directly, so
the contract the frontend depends on - field names, status values, 422 shape -
is what's actually under test.
"""
import os
import sys
from datetime import date

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

import api  # noqa: E402

DATE = "2026-10-12"


@pytest.fixture(scope="module")
def client():
    api.app.config["TESTING"] = True
    with api.app.test_client() as c:
        yield c


def plan(client, **overrides):
    body = {"from_city": "Chennai", "to_city": "Delhi", "date": DATE,
            "travelers": 2, "budget": 25000}
    body.update(overrides)
    return client.post("/api/plan-trip", json=body)


# ------------------------------------------------------------------ 1
def test_chennai_delhi_is_ready_and_within_budget(client):
    r = plan(client)
    assert r.status_code == 200
    d = r.get_json()

    assert d["status"] == "READY"
    assert d["route_label"] == "Chennai → Delhi"
    assert d["total"] <= d["budget"], "READY must never exceed the budget"
    assert d["under_budget"] is True

    f = d["flight"]
    assert f["price_total"] == f["price_per_person"] * 2
    assert 0.0 <= f["disruption_risk"] <= 1.0
    # total must reconcile exactly with its parts
    sights = sum(a["est_cost"] for a in d["attractions"]) * 2
    assert d["total"] == f["price_total"] + d["hotel"]["price_total"] + sights


# ------------------------------------------------------------------ 2
def test_tight_budget_downgrades_stay_or_drops_sights(client):
    roomy = plan(client).get_json()
    tight = plan(client, budget=12000).get_json()

    assert tight["status"] in ("READY", "OVER_BUDGET")
    # something must have given: fewer nights, a cheaper room, or fewer sights
    gave_way = (
        tight["hotel"]["nights"] < roomy["hotel"]["nights"]
        or tight["hotel"]["price_per_night"] < roomy["hotel"]["price_per_night"]
        or len(tight["attractions"]) < len(roomy["attractions"])
    )
    assert gave_way, "a tighter budget must trim the plan"
    assert tight["adjustments"], "trimming should be explained, not silent"


# ------------------------------------------------------------------ 3
def test_impossible_budget_reports_over_budget_with_cheapest_plan(client):
    d = plan(client, budget=2000).get_json()

    assert d["status"] == "OVER_BUDGET"
    assert d["under_budget"] is False
    assert d["total"] > d["budget"]
    # still returns a usable plan rather than refusing
    assert d["flight"] is not None
    assert d["hotel"]["nights"] == 1, "should bottom out at the minimum stay"


# ------------------------------------------------------------------ 4
def test_high_risk_flight_is_rerouted():
    """A flight over the threshold must be swapped for a materially safer one.

    Exercises the recovery node directly with a constructed high-risk flight
    rather than hoping a catalogue route happens to score above 0.30. It used
    to go through the endpoint, which made it a test of whatever the model
    currently predicts: when the India model replaced DelayNetV2 on these
    routes every Indian flight dropped below the threshold, nothing rerouted,
    and the test failed while the recovery logic it covers was untouched.
    A better model should not break a test of the branch it feeds.
    """
    from trip_graph import RISK_THRESHOLD, recovery

    risky = {"number": "6E 999", "airline": "Test Air", "carrier_code": "6E",
             "from": "BOM", "to": "DEL", "from_city": "Mumbai", "to_city": "Delhi",
             "depart": "19:00", "arrive": "21:10", "duration_min": 130, "fare_inr": 5000}
    safe = {**risky, "number": "6E 111", "depart": "06:00", "arrive": "08:10"}

    state = {
        "from_city": "Mumbai", "to_city": "Delhi",
        # a date object, not the string the endpoint takes - the adapter calls
        # .isoformat() on it downstream
        "travel_date": date.fromisoformat(DATE),
        "travelers": 1, "budget": 25000,
        "flight": risky, "candidates": [risky, safe],
        "disruption": {"disruption_risk": 0.62, "risk_is_modelled": True},
        "was_rerouted": False, "trim_log": [],
    }
    out = recovery(state)

    assert out["disruption"]["disruption_risk"] <= RISK_THRESHOLD or not out["was_rerouted"],         "a reroute must land at or below the threshold, never swap risk for risk"
    if out["was_rerouted"]:
        assert out["flight"]["number"] != risky["number"]
        # the node logs to trim_log; assemble is what renames it to adjustments
        assert any("Rerouted" in a for a in out["trim_log"])


def test_reroute_invariant_holds_on_real_routes(client):
    """Whatever the model says today, a rerouted flight is never still risky."""
    for src, dst in [("Mumbai", "Delhi"), ("Delhi", "Mumbai"), ("Kolkata", "Delhi")]:
        d = plan(client, from_city=src, to_city=dst).get_json()
        flight = d.get("flight")
        if flight and flight["was_rerouted"]:
            assert flight["disruption_risk"] <= 0.30, f"{src}->{dst} rerouted but still risky"


# ------------------------------------------------------------------ 5
def test_unknown_route_returns_no_options(client):
    d = plan(client, from_city="Atlantis").get_json()

    assert d["status"] == "NO_OPTIONS"
    assert d["flight"] is None
    assert d["total"] == 0
    assert d["message"]


# ------------------------------------------------------- input validation
@pytest.mark.parametrize("bad,why", [
    ({"travelers": 0}, "travelers below one"),
    ({"budget": 0}, "zero budget"),
    ({"budget": -500}, "negative budget"),
    ({"date": "12-10-2026"}, "wrong date format"),
    ({"date": "not-a-date"}, "unparseable date"),
    ({"to_city": "Chennai"}, "origin equals destination"),
    ({"from_city": ""}, "missing origin"),
])
def test_invalid_input_returns_422(client, bad, why):
    r = plan(client, **bad)
    assert r.status_code == 422, f"expected 422 for {why}"
    body = r.get_json()
    assert body["details"], "422 should say what was wrong"


# ------------------------------------------------------------------ travellers survive the 2-step form
def test_trip_planner_carries_travellers_from_compare_to_plan(client):
    """The page is two posts: 'compare' transport modes, then the chosen mode's own
    form posts 'plan'. Every field must ride along in that second form - travelers
    was once missing, so every plan silently priced a party of one."""
    import re
    with client.session_transaction() as s:
        s["user_id"] = 1
        s["user_name"] = "t"
    first = client.post("/budget-trip", data={
        "stage": "compare", "origin_place": "Chennai, TN, India", "destination_place": "New Delhi, India",
        "destination_city": "New Delhi, India", "start_date": "2026-10-22", "days": "3",
        "travelers": "10", "total_budget": "200000"}).get_data(as_text=True)
    forms = re.findall(r'<form method="post"[^>]*>(.*?)</form>', first, re.S)
    plan_forms = [f for f in forms if 'name="stage" value="plan"' in f]
    assert plan_forms, "the compare step should offer at least one transport mode"
    assert all('name="travelers" value="10"' in f for f in plan_forms)


# ------------------------------------------------------------------ the plan respects the budget
def test_hotel_pick_fits_what_the_budget_leaves(monkeypatch):
    import api as appmod
    plan = {"nights": 2, "rooms": 1, "travelers": 2,
            "budget_breakdown": {"food_cost": 50.0, "transport_cost": 30.0},
            "real_outbound_price": {"price_usd": 200.0, "airline": "X"},
            "real_return_price": {"price_usd": 200.0, "airline": "X"},
            "real_priced_hotels": [{"name": n, "nightly_rate_usd": r}
                                   for n, r in (("A", 20.0), ("B", 60.0), ("C", 90.0), ("D", 400.0))]}
    inputs = {"total_budget": 700.0, "travelers": 2, "days": 3, "stay_comfort": "luxury"}
    with appmod.app.test_request_context():
        appmod.session["currency"] = "USD"
        cb = appmod._cost_breakdown(plan, inputs)
    # 700 - 400 flights - 80 allowances = 220 left -> 110/room/night: C (90) is the best that fits, not D (400)
    hotel = next(i for i in cb["items"] if i["label"] == "Hotel")
    assert plan["default_hotel_name"] == "C" and hotel["usd"] == 180.0
    assert cb["fit"]["possible"] and cb["fit"]["fits"]
    assert sum(i["usd"] for i in cb["items"] if i["label"] != "Left over") <= 700.0


def test_impossible_budget_reports_the_real_minimum():
    import api as appmod
    plan = {"nights": 2, "rooms": 1, "travelers": 1,
            "budget_breakdown": {"food_cost": 5.0, "transport_cost": 3.0},
            "real_outbound_price": {"price_usd": 120.0}, "real_return_price": {"price_usd": 110.0},
            "real_priced_hotels": [{"name": "Cheap", "nightly_rate_usd": 4.0}]}
    with appmod.app.test_request_context():
        appmod.session["currency"] = "USD"
        cb = appmod._cost_breakdown(plan, {"total_budget": 45.0, "travelers": 1, "days": 3,
                                           "stay_comfort": "standard"})
    assert cb["fit"]["possible"] is False
    # flights 230 + cheapest hotel 8; food/transport are shares of what is left after
    # travel, and travel alone already exceeds the 45 budget, so they add nothing
    assert cb["fit"]["minimum_usd"] == 120 + 110 + 8
    assert plan["default_hotel_name"] == "Cheap"



def test_car_trip_costs_the_drive_and_reserves_nothing_for_flights():
    import api as appmod
    plan = {"nights": 2, "rooms": 1, "travelers": 2, "budget_breakdown": {},
            "selected_transport": {"mode": "car", "available": True, "distance_km": 500,
                                   "duration_hours": 8, "estimated_cost_usd": 40.0},
            "real_priced_hotels": [{"name": "H", "nightly_rate_usd": 20.0}]}
    inputs = {"total_budget": 300.0, "travelers": 2, "days": 3, "transport_mode": "car",
              "stay_comfort": "budget", "food_comfort": "standard", "travel_comfort": "standard"}
    with appmod.app.test_request_context():
        appmod.session["currency"] = "USD"
        cb = appmod._cost_breakdown(plan, inputs)
    labels = [i["label"] for i in cb["items"]]
    assert "Flights" not in labels and labels[0] == "Car (fuel)"
    car = cb["items"][0]
    assert car["usd"] == 80.0, "fuel there and back"
    food = next(i for i in cb["items"] if i["label"] == "Food")
    assert food["usd"] == (300 - 80) * appmod.FOOD_SHARE_OF_REMAINING
    assert cb["fit"]["fits"]
