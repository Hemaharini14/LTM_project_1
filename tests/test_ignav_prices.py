"""
Tests for src/ignav_prices.py. The HTTP call is replaced by a stand-in shaped like
Ignav's documented /fares/one-way response, so these check OUR handling: party-total
pricing, currency safety, exact-flight matching, caching, fallback and key secrecy.
"""
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import ignav_prices as ig  # noqa: E402

KEY = "ignav_TEST_KEY_never_logged"


def itin(amount, code, num, dep="2026-10-22T23:45:00", arr="2026-10-23T02:40:00", currency="USD", segs=1):
    seg = {"marketing_carrier_code": code, "flight_number": num, "departure_airport": "MAA",
           "departure_time_local": dep, "arrival_airport": "DEL", "arrival_time_local": arr,
           "duration_minutes": 175}
    return {"price": {"amount": amount, "currency": currency, "status": "verified"},
            "outbound": {"carrier": "IndiGo", "duration_minutes": 175,
                         "segments": [seg] * segs},
            "cabin_class": "economy", "ignav_id": f"id-{code}{num}"}


class Resp:
    def __init__(self, body, status=200):
        self.body, self.status_code = body, status

    def json(self):
        return self.body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"401 Unauthorized for url https://ignav.com/api ... {KEY}")


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    monkeypatch.setattr(ig, "_CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(ig, "_USAGE_FILE", tmp_path / "usage.json")
    monkeypatch.setattr(ig, "_key", lambda: KEY)


def fake(monkeypatch, body, status=200):
    calls = []

    def post(url, json=None, headers=None, timeout=None):
        calls.append({"url": url, "json": json, "headers": headers})
        return Resp(body, status)
    monkeypatch.setattr(ig.httpx, "post", post)
    return calls


def test_price_is_the_party_total_not_multiplied(monkeypatch):
    calls = fake(monkeypatch, {"itineraries": [itin(325, "6E", "2369")]})
    o = ig.one_way_options("MAA", "DEL", "2026-10-22", adults=3)[0]
    assert o["price_usd"] == 325 and o["price_per_person_usd"] == round(325 / 3, 2)
    assert calls[0]["json"]["adults"] == 3 and calls[0]["headers"]["X-Api-Key"] == KEY
    assert o["flight_number"] == "6E2369" and o["departure_time"] == "2026-10-22 23:45"
    assert o["stops"] == 0 and o["source_label"] == "Ignav"


def test_cheapest_first_and_non_usd_dropped(monkeypatch):
    fake(monkeypatch, {"itineraries": [itin(400, "AI", "1"), itin(300, "6E", "2"),
                                       itin(10, "XX", "9", currency="INR")]})
    opts = ig.one_way_options("MAA", "DEL", "2026-10-22")
    assert [o["flight_number"] for o in opts] == ["6E2", "AI1"]


def test_exact_flight_match(monkeypatch):
    fake(monkeypatch, {"itineraries": [itin(300, "6E", "2"), itin(350, "AI", "538")]})
    f = ig.price_for_flight("MAA", "DEL", "2026-10-22", "AI 538")
    assert f["price_usd"] == 350 and f["same_flight"] is True
    assert ig.price_for_flight("MAA", "DEL", "2026-10-22", "UK999") is None


def test_connections_are_counted(monkeypatch):
    fake(monkeypatch, {"itineraries": [itin(500, "EK", "543", segs=2)]})
    assert ig.one_way_options("MAA", "LHR", "2026-12-15")[0]["stops"] == 1


def test_identical_search_is_cached(monkeypatch):
    calls = fake(monkeypatch, {"itineraries": [itin(300, "6E", "2")]})
    ig.one_way_options("MAA", "DEL", "2026-10-22")
    ig.real_flight_price("MAA", "DEL", "2026-10-22")
    assert len(calls) == 1


def test_failure_falls_back_to_serpapi_and_says_so(monkeypatch, capsys):
    fake(monkeypatch, {}, status=401)
    import serpapi_prices
    monkeypatch.setattr(serpapi_prices, "real_flight_price",
                        lambda *a, **k: {"price_usd": 120.0, "airline": "IndiGo"})
    fare = ig.real_flight_price("MAA", "DEL", "2026-10-22", adults=2)
    assert fare["price_usd"] == 120.0 and "SerpApi" in fare["source_label"]
    assert KEY not in capsys.readouterr().out, "the key must never be printed"


def test_no_flights_is_none_without_fallback(monkeypatch):
    fake(monkeypatch, {"itineraries": []})
    import serpapi_prices

    def must_not_call(*a, **k):
        raise AssertionError("an answered-but-empty search is not a failure")
    monkeypatch.setattr(serpapi_prices, "real_flight_price", must_not_call)
    assert ig.real_flight_price("MAA", "DEL", "2026-10-22") is None


def test_monthly_cap_refuses(monkeypatch):
    calls = fake(monkeypatch, {"itineraries": [itin(300, "6E", "2")]})
    monkeypatch.setattr(ig, "MONTHLY_CALL_CAP", 0)
    assert ig.one_way_options("MAA", "DEL", "2026-10-22") is None and not calls


def test_real_key_not_in_source():
    real = ""
    env = os.path.join(ROOT, ".env")
    if os.path.exists(env):
        for line in open(env, encoding="utf-8"):
            if line.startswith("IGNAV_API_KEY="):
                real = line.split("=", 1)[1].strip()
    if not real:
        pytest.skip("no real key configured")
    for folder in ("src", "templates", "static"):
        for root, _, files in os.walk(os.path.join(ROOT, folder)):
            for f in files:
                if f.endswith((".py", ".html", ".js")):
                    assert real not in open(os.path.join(root, f), encoding="utf-8", errors="ignore").read()
