"""
Tests for the live flight search layer (src/flight_search.py + /api/flights/*).

Aviationstack and the weather provider are replaced with small stand-ins shaped like
their documented responses - the point is to test OUR handling of them (field
mapping, declining on missing inputs, error codes, key secrecy), never to supply
data to the product. The delay model itself is the real one unless a test says
otherwise.
"""
import os
import sys
from datetime import date, timedelta

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

import api  # noqa: E402
import aviationstack  # noqa: E402
import flight_search as fs  # noqa: E402

TODAY = date.today().isoformat()
SECRET = "SECRET-KEY-DO-NOT-LEAK-123"

CLEAR_WX = {"temp_f": 80.0, "precip_in": 0.0, "pressure": 29.9, "visibility": 10.0,
            "wind_speed": 6.0, "source": "open-meteo"}


def api_row(**over):
    row = {
        "flight_date": TODAY, "flight_status": "scheduled",
        "departure": {"airport": "Chennai International", "timezone": "Asia/Kolkata", "iata": "MAA",
                      "terminal": "1", "gate": None, "delay": 5,
                      "scheduled": f"{TODAY}T16:45:00+00:00", "estimated": None, "actual": None},
        "arrival": {"airport": "Indira Gandhi International", "timezone": "Asia/Kolkata", "iata": "DEL",
                    "terminal": None, "gate": None, "baggage": None, "delay": None,
                    "scheduled": f"{TODAY}T19:40:00+00:00", "estimated": None, "actual": None},
        "airline": {"name": "IndiGo", "iata": "6E", "icao": "IGO"},
        "flight": {"number": "698", "iata": "6E698", "icao": "IGO698", "codeshared": None},
        "aircraft": None,
    }
    row.update(over)
    return row


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    """Fresh cache dir + usage file per test; key set unless a test removes it."""
    monkeypatch.setattr(aviationstack, "_CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(aviationstack, "_USAGE_FILE", tmp_path / "usage.json")
    monkeypatch.setattr(aviationstack, "_key", lambda: SECRET)
    monkeypatch.setattr(aviationstack, "_last_call", 0.0)
    monkeypatch.setattr(aviationstack, "_MIN_INTERVAL_S", 0)


@pytest.fixture
def client():
    api.app.config["TESTING"] = True
    with api.app.test_client() as c:
        with c.session_transaction() as s:
            s["user_id"] = 1
            s["user_name"] = "t"
        yield c


class FakeResponse:
    def __init__(self, body, status=200, headers=None):
        self._body, self.status_code, self.headers = body, status, headers or {}

    def json(self):
        return self._body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


def fake_http(monkeypatch, body, status=200):
    calls = []

    def _get(url, params=None, timeout=None):
        calls.append(params)
        return FakeResponse(body, status)
    monkeypatch.setattr(aviationstack.httpx, "get", _get)
    return calls


def with_weather(monkeypatch, wx=CLEAR_WX):
    monkeypatch.setattr(fs, "airport_weather", lambda iata, when=None: wx)


# ------------------------------------------------------------ normalisation
def test_missing_api_fields_stay_none_not_invented():
    n = aviationstack.normalize_live(api_row())
    assert n["aircraft"] is None and n["gate_departure"] is None
    assert n["terminal_arrival"] is None and n["baggage_belt"] is None
    assert n["airline_name"] == "IndiGo" and n["flight_iata"] == "6E698"


def test_duration_uses_both_airport_time_zones():
    f = aviationstack.normalize_live(api_row())
    f["origin_timezone"], f["destination_timezone"] = "Asia/Kolkata", "Europe/London"
    f["scheduled_departure"], f["scheduled_arrival"] = f"{TODAY}T04:15:00+00:00", f"{TODAY}T07:30:00+00:00"
    minutes, how = fs._elapsed_minutes(f)
    # The clock difference (195 min) is NOT the duration across time zones.
    assert minutes != 195 and 20 <= minutes <= 24 * 60 and "time zones" in how


# ------------------------------------------------------------ search endpoint
def test_search_returns_real_fields_and_prediction(client, monkeypatch):
    fake_http(monkeypatch, {"data": [api_row()]})
    with_weather(monkeypatch)
    r = client.get("/api/flights/search?origin=MAA&destination=DEL")
    assert r.status_code == 200
    d = r.get_json()
    f = d["flights"][0]
    assert f["airline_name"] == "IndiGo" and f["flight_iata"] == "6E698"
    assert f["duration_min"] == 175
    p = f["prediction"]
    assert p["status"] == "ok" and 0 <= p["delay_probability"] <= 1
    assert p["predicted"] in ("DELAYED", "ON TIME")
    assert d["meta"]["cached"] is False


def test_identical_search_is_served_from_cache_and_says_so(client, monkeypatch):
    calls = fake_http(monkeypatch, {"data": [api_row()]})
    with_weather(monkeypatch)
    client.get("/api/flights/search?origin=MAA&destination=DEL")
    second = client.get("/api/flights/search?origin=MAA&destination=DEL").get_json()
    assert len(calls) == 1, "the second identical search must not spend API quota"
    assert second["meta"]["cached"] is True
    assert "cache" in second["meta"]["notice"].lower()


def test_no_flights_found_is_404_with_message(client, monkeypatch):
    fake_http(monkeypatch, {"data": []})
    r = client.get("/api/flights/search?origin=MAA&destination=DEL")
    assert r.status_code == 404 and r.get_json()["error"]["code"] == "no_flights"


def test_missing_api_key_is_a_clear_503(client, monkeypatch):
    monkeypatch.setattr(aviationstack, "_key", lambda: None)
    r = client.get("/api/flights/search?origin=MAA&destination=DEL")
    assert r.status_code == 503
    assert r.get_json()["error"]["code"] == "missing_api_key"


def test_rate_limit_is_429(client, monkeypatch):
    fake_http(monkeypatch, {"error": {"code": "rate_limit_reached", "message": "slow down"}})
    r = client.get("/api/flights/search?origin=MAA&destination=DEL")
    assert r.status_code == 429 and r.get_json()["error"]["code"] == "rate_limited"


def test_api_down_is_502_not_a_crash(client, monkeypatch):
    def boom(*a, **k):
        raise TimeoutError("timed out")
    monkeypatch.setattr(aviationstack.httpx, "get", boom)
    r = client.get("/api/flights/search?origin=MAA&destination=DEL")
    assert r.status_code == 502 and r.get_json()["error"]["code"] == "api_unreachable"


@pytest.mark.parametrize("qs", [
    "origin=XX", "origin=M4A", "flight_number=!!", "", "origin=MAA&date=15-12-2026",
])
def test_invalid_input_is_400(client, qs):
    r = client.get("/api/flights/search?" + qs)
    assert r.status_code == 400 and r.get_json()["error"]["code"] == "invalid_request"


def test_past_date_rejected(client):
    past = (date.today() - timedelta(days=2)).isoformat()
    r = client.get(f"/api/flights/search?origin=MAA&date={past}")
    assert r.status_code == 400 and r.get_json()["error"]["code"] == "invalid_date"


def test_future_date_needs_an_origin(client):
    future = (date.today() + timedelta(days=10)).isoformat()
    r = client.get(f"/api/flights/search?flight_number=6E698&date={future}")
    assert r.status_code == 400 and r.get_json()["error"]["code"] == "origin_required"


def test_endpoints_require_login():
    api.app.config["TESTING"] = True
    with api.app.test_client() as anon:
        assert anon.get("/api/flights/search?origin=MAA").status_code == 401
        assert anon.post("/api/flights/predict-delay", json={}).status_code == 401


# ------------------------------------------------------------ prediction integrity
def test_no_weather_means_no_prediction_not_a_default(monkeypatch):
    monkeypatch.setattr(fs, "airport_weather", lambda iata, when=None: None)

    def must_not_run(*a, **k):
        raise AssertionError("model must not be called without departure weather")
    monkeypatch.setattr(fs, "predict_delay_probability", must_not_run)
    p = fs.predict_flight(aviationstack.normalize_live(api_row()))
    assert p["status"] == "unavailable" and p["delay_probability"] is None
    assert "weather" in p["reason"].lower()


def test_unknown_duration_means_no_prediction(monkeypatch):
    with_weather(monkeypatch)
    f = aviationstack.normalize_live(api_row())
    f["scheduled_arrival"] = None          # API gave no arrival time
    f["flight_number"] = "99999"           # and the catalogue has never seen it
    p = fs.predict_flight(f)
    assert p["status"] == "unavailable" and "duration" in p["reason"].lower()


def test_unmodelled_route_is_declined(monkeypatch):
    with_weather(monkeypatch)
    f = aviationstack.normalize_live(api_row())
    f["origin_iata"], f["destination_iata"] = "ZZZ", "YYY"
    assert fs.predict_flight(f)["status"] == "not_modeled"


@pytest.mark.parametrize("prob,expected", [(0.60, "DELAYED"), (0.35, "DELAYED"), (0.10, "ON TIME")])
def test_label_uses_the_models_own_threshold(monkeypatch, prob, expected):
    with_weather(monkeypatch)
    monkeypatch.setattr(fs, "predict_delay_probability", lambda **k: prob)
    monkeypatch.setattr(fs, "_threshold", lambda: 0.35)
    p = fs.predict_flight(aviationstack.normalize_live(api_row()))
    assert p["predicted"] == expected and p["delay_probability"] == round(prob, 4)
    assert p["threshold"] == 0.35


def test_model_receives_the_real_weather_and_schedule(monkeypatch):
    with_weather(monkeypatch)
    seen = {}

    def spy(**k):
        seen.update(k)
        return 0.2
    monkeypatch.setattr(fs, "predict_delay_probability", spy)
    fs.predict_flight(aviationstack.normalize_live(api_row()))
    assert seen["origin_temp_f"] == 80.0 and seen["origin_wind_speed"] == 6.0
    assert seen["scheduled_hour"] == 16 and seen["scheduled_elapsed_time"] == 175
    assert seen["carrier_code"] == "6E" and seen["origin_airport"] == "MAA"


def test_no_minutes_forecast_is_claimed():
    # The model is a probability-only classifier: nothing may present delay
    # minutes as a prediction for this flight.
    src = open(os.path.join(ROOT, "src", "flight_search.py"), encoding="utf-8").read()
    assert '"estimated_delay_minutes"' not in src and '"expected_delay_minutes"' not in src


# ------------------------------------------------------------ predict endpoint
def test_predict_endpoint_validates_and_returns_structure(client, monkeypatch):
    with_weather(monkeypatch)
    body = {"airline_iata": "6E", "origin_iata": "MAA", "destination_iata": "DEL",
            "scheduled_departure": f"{TODAY}T16:45:00+00:00",
            "scheduled_arrival": f"{TODAY}T19:40:00+00:00",
            "origin_timezone": "Asia/Kolkata", "destination_timezone": "Asia/Kolkata"}
    r = client.post("/api/flights/predict-delay", json=body)
    assert r.status_code == 200
    d = r.get_json()
    assert set(d) == {"flight", "weather", "prediction"}
    assert d["prediction"]["status"] == "ok"
    bad = client.post("/api/flights/predict-delay", json={**body, "origin_iata": "M"})
    assert bad.status_code == 400


# ------------------------------------------------------------ secrecy + rendering
def test_api_key_never_appears_in_responses_or_frontend(client, monkeypatch):
    fake_http(monkeypatch, {"data": [api_row()]})
    with_weather(monkeypatch)
    blobs = [client.get("/api/flights/search?origin=MAA&destination=DEL").get_data(as_text=True),
             client.get("/flight-search").get_data(as_text=True),
             client.get("/static/js/flight_search.js").get_data(as_text=True)]
    for b in blobs:
        assert SECRET not in b and "access_key" not in b


def test_search_page_renders_with_form(client):
    html = client.get("/flight-search").get_data(as_text=True)
    assert 'id="fs-form"' in html and "flight_search.js" in html and 'id="airport-list"' in html


def test_frontend_never_uses_innerhtml():
    js = open(os.path.join(ROOT, "static", "js", "flight_search.js"), encoding="utf-8").read()
    assert "innerHTML" not in js


# ------------------------------------------------------------ "why this score"
def test_drivers_follow_the_weather_and_only_for_delaynet(monkeypatch):
    storm = {"temp_f": 34.0, "precip_in": 0.5, "pressure": 29.4, "visibility": 1.5,
             "wind_speed": 25.0, "source": "open-meteo"}
    with_weather(monkeypatch, storm)
    us = aviationstack.normalize_live(api_row(
        departure={"iata": "ATL", "timezone": "America/New_York", "scheduled": f"{TODAY}T19:00:00+00:00"},
        arrival={"iata": "LGA", "timezone": "America/New_York", "scheduled": f"{TODAY}T21:20:00+00:00"},
        airline={"name": "Delta", "iata": "DL"}, flight={"number": "1", "iata": "DL1"}))
    p = fs.predict_flight(us)
    assert p["status"] == "ok" and p["model"] == "DelayNetV2"
    assert p["why"]["drivers"], "a stormy flight must have named drivers"
    assert p["why"]["drivers"][0]["factor"] in ("Precipitation", "Visibility")

    india = fs.predict_flight(aviationstack.normalize_live(api_row()))
    assert india["why"] is None and "India route model" in india["why_note"]
