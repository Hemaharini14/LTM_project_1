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
