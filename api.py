"""
SmartRouteAI web app: login/signup (with an admin role), a two-option
dashboard, a flight-delay checker with budget-based alternative recovery,
an end-to-end budget trip planner, clickable trip history, an admin panel,
and a sidebar chatbot backed by a real LangGraph tool-calling agent. Wraps
the existing ML modules in src/ (delay prediction, recovery graph, budget
optimizer, trip planner, chat agent) behind a Flask UI and persists users +
trip history to SQLite (src/db.py).

Run from the project root:
    python api.py
"""
import os
import re
import sys
import json
import threading
from datetime import date, datetime, timedelta
from functools import wraps

from flask import (Flask, render_template, request, redirect, url_for, session,
                   flash, send_file, send_from_directory)
from werkzeug.security import generate_password_hash, check_password_hash
import markdown as _markdown
import bleach

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(PROJECT_ROOT, "src"))

import db
from predict_delay_v2 import predict_delay_probability, risk_label
from delay_duration import estimate_delay_duration
from explain_delay import explain as explain_delay
from opensky import (bbox_for_route, callsign_for, live_rotation, live_traffic,
                     rotation_from_icao24)
from aviationstack import flight_now
from weather_live import airport_weather
from booking import describe_party, links_for_flight
from serpapi_prices import real_flight_price, real_hotel_prices
from recovery_graph import build_graph
from recovery_tools import get_dataset_index, is_route_covered, lookup_flight_by_number
from trip_planner import plan_budget_trip
import flight_search as fs
from pydantic import ValidationError
from trip_graph import plan_trip as run_trip_graph
from transport_modes import compare_transport_modes, describe_mode
from places import autocomplete_places
from maps import nearest_supported_airport
from reference_data import carrier_label, airport_label, CARRIER_NAMES
from intl_reference import INTL_AIRPORT_CODES, get_reference_flights
from llm_utils import narrate_trip_plan, llm_status
from model_metrics import load_model_metrics, sample_validation_flights
from chat_agent import run_chat
from feature_engineering import is_holiday_date
import currency as currency_utils

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-key-change-in-production")
app.jinja_env.globals["carrier_name"] = carrier_label
app.jinja_env.globals["airport_label"] = airport_label

_MARKDOWN_ALLOWED_TAGS = ["p", "strong", "em", "ul", "ol", "li", "table", "thead", "tbody", "tr",
                          "th", "td", "h1", "h2", "h3", "h4", "br", "code", "pre", "blockquote", "hr", "a"]


def _render_markdown(text: str) -> str:
    """Renders LLM narration (which comes back as Markdown - headers, bold, tables) as HTML,
    so it displays as formatted text instead of a wall of raw ** and | characters. Sanitized
    with bleach since this is model output, not something we've reviewed line by line."""
    if not text:
        return ""
    html = _markdown.markdown(text, extensions=["tables", "sane_lists"])
    return bleach.clean(html, tags=_MARKDOWN_ALLOWED_TAGS, attributes={"a": ["href", "title"]}, strip=True)


app.jinja_env.filters["render_markdown"] = _render_markdown
# Costs are computed in USD internally (the training data and hotel ADR tiers
# are dollar-denominated); DEFAULT_CURRENCY is only what they are DISPLAYED in.
app.jinja_env.filters["money"] = lambda usd: currency_utils.format_money(
    usd, session.get("currency", currency_utils.DEFAULT_CURRENCY))

db.init_db()
_recovery_app = build_graph()


def _preload_flight_data():
    """Touch the flight catalogue once at startup, off the critical path.

    This used to mean reading the whole ~5.4M-row/864MB CSV into memory
    (~27s, and ~2-3GB of RAM held for the life of the process - more than
    most free hosting tiers have at all). The catalogue is now an indexed
    SQLite file (build_flight_catalog_db.py) that recovery_tools.py queries
    directly per request, so there's no multi-second load or standing memory
    cost to warm here - this just opens the file once so the very first real
    request isn't the one paying for the OS to fault it in from disk.
    """
    try:
        from recovery_tools import _catalog_conn
        conn = _catalog_conn()
        if conn is None:
            print("[api] flight_catalog.db not found - run src/build_flight_catalog_db.py "
                  "(alternatives/trip-planner search will return no results until then).")
        else:
            conn.execute("SELECT 1 FROM flights LIMIT 1")
            conn.close()
    except Exception as e:
        print(f"[api] flight-catalog preload skipped: {e}")


threading.Thread(target=_preload_flight_data, daemon=True).start()

# Sign up (or already be registered) with one of these emails to get admin access.
ADMIN_EMAILS = {e.strip().lower() for e in os.environ.get("ADMIN_EMAILS", "aihemaharini@gmail.com").split(",")}

DEFAULT_WEATHER = {
    "temp_f": 55, "precip_in": 0.0, "pressure": 29.9, "visibility": 10.0, "wind_speed": 8,
}
ASSUMED_RECOVERY_HOTEL_NIGHTS = 3  # recovery_graph doesn't track trip length; matches its own /3 assumption


def _float_field(form, key: str, default: float) -> float:
    """form.get(key, default) only falls back when the field is absent, not when it's
    submitted as an empty string (e.g. a number input the user cleared) - this covers both."""
    raw = (form.get(key) or "").strip()
    return float(raw) if raw else float(default)


def _int_field(form, key: str, default: int) -> int:
    raw = (form.get(key) or "").strip()
    return int(raw) if raw else int(default)


def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if not session.get("user_id"):
            return redirect(url_for("login", next=request.path))
        return view(*args, **kwargs)
    return wrapped


def admin_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if not session.get("is_admin"):
            flash("Admin access required.", "error")
            return redirect(url_for("home"))
        return view(*args, **kwargs)
    return wrapped


@app.context_processor
def inject_user():
    user_id = session.get("user_id")
    return {
        "current_user_name": session.get("user_name") if user_id else None,
        "is_admin": session.get("is_admin", False),
        "current_currency": session.get("currency", currency_utils.DEFAULT_CURRENCY),
        "supported_currencies": currency_utils.SUPPORTED_CURRENCIES,
    }


def _dropdown_options():
    index = get_dataset_index()
    all_airports = sorted(set(index["airports"]) | INTL_AIRPORT_CODES)
    all_carriers = sorted(set(index["carriers"]) | set(CARRIER_NAMES.keys()))
    return {
        "carriers": [{"code": c, "label": carrier_label(c)} for c in all_carriers],
        "airports": [{"code": a, "label": airport_label(a)} for a in all_airports],
    }


def _format_ts(iso_str: str) -> str:
    try:
        return datetime.fromisoformat(iso_str).strftime("%b %d, %Y · %I:%M %p UTC")
    except (ValueError, TypeError):
        return iso_str or ""


# ---------------------------------------------------------------- health

@app.route("/health")
def health():
    """Liveness/readiness probe for a process manager or container orchestrator -
    no login, no page render, just real component checks. Each check is wrapped
    separately so one failing dependency still reports which one, rather than a
    bare 500. Deliberately does NOT call the live weather/flight APIs - those
    have their own quotas and failure modes, and a health check spending them
    would make the probe itself a cause of outages."""
    checks = {}

    try:
        from predict_delay_v2 import predict_delay_probability
        predict_delay_probability(
            carrier_code="AA", origin_airport="JFK", destination_airport="LAX",
            weekday=1, month=6, scheduled_elapsed_time=360,
            origin_temp_f=70, origin_temp_known=True, origin_precip_in=0,
            origin_pressure=29.9, origin_visibility=10, origin_wind_speed=8,
        )
        checks["delay_model"] = "ok"
    except Exception as e:
        checks["delay_model"] = f"error: {e}"

    try:
        import sqlite3
        sqlite3.connect(db.DB_PATH, timeout=5).close()
        checks["database"] = "ok"
    except Exception as e:
        checks["database"] = f"error: {e}"

    healthy = all(v == "ok" for v in checks.values())
    return checks, 200 if healthy else 503


# ---------------------------------------------------------------- auth

SHOWCASE_INDEX = os.path.join(PROJECT_ROOT, "static", "showcase", "index.html")


@app.route("/")
def index():
    """Cinematic showcase for logged-out visitors, dashboard for signed-in users.

    The showcase is a Vite/React/Three.js build (web/, built into
    static/showcase/) served as a static file - it has no server state of its
    own and its CTAs link straight into the real Flask pages. If it hasn't been
    built, fall back to the original server-rendered landing page so the app
    still works from a fresh clone without running npm.
    """
    if session.get("user_id"):
        return redirect(url_for("home"))
    if os.path.exists(SHOWCASE_INDEX):
        return send_file(SHOWCASE_INDEX)
    return render_template("landing.html")


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "")

        if not name or not email or not password:
            flash("Please fill in all fields.", "error")
        elif len(password) < 6:
            flash("Password must be at least 6 characters.", "error")
        elif db.get_user_by_email(email):
            flash("An account with that email already exists.", "error")
        else:
            is_admin = email.lower() in ADMIN_EMAILS
            user_id = db.create_user(name, email, generate_password_hash(password), is_admin=is_admin)
            session["user_id"] = user_id
            session["user_name"] = name
            session["is_admin"] = is_admin
            session["currency"] = currency_utils.DEFAULT_CURRENCY
            return redirect(url_for("home"))
    return render_template("signup.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "")
        user = db.get_user_by_email(email)
        if user and check_password_hash(user["password_hash"], password):
            session["user_id"] = user["id"]
            session["user_name"] = user["name"]
            session["is_admin"] = bool(user["is_admin"])
            session["currency"] = user["preferred_currency"] or currency_utils.DEFAULT_CURRENCY
            return redirect(request.args.get("next") or url_for("home"))
        flash("Invalid email or password.", "error")
    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/set-currency", methods=["POST"])
@login_required
def set_currency():
    currency = request.form.get("currency", currency_utils.DEFAULT_CURRENCY).upper()
    if currency not in currency_utils.SUPPORTED_CURRENCIES:
        currency = currency_utils.DEFAULT_CURRENCY
    session["currency"] = currency
    db.update_user_currency(session["user_id"], currency)
    return redirect(request.form.get("next") or url_for("home"))


# ---------------------------------------------------------------- dashboard

@app.route("/home")
@login_required
def home():
    return render_template("home.html")


@app.route("/history")
@login_required
def history():
    checks, trips = db.get_recent_trips(session["user_id"], limit=15)

    flight_history = []
    for row in checks:
        inputs = json.loads(row["inputs_json"])
        recommendation = json.loads(row["recommendation_json"]) if row["recommendation_json"] else None
        flight_history.append({
            "id": row["id"],
            "carrier": inputs.get("carrier_code"),
            "origin": inputs.get("origin_airport"),
            "destination": inputs.get("destination_airport"),
            "travel_date": inputs.get("travel_date"),
            "probability": row["delay_probability"],
            "risk_label": row["risk_label"],
            "recommendation_text": recommendation.get("recommendation_text") if recommendation else None,
            "created_at": _format_ts(row["created_at"]),
        })

    trip_history = []
    for row in trips:
        inputs = json.loads(row["inputs_json"])
        trip_history.append({
            "id": row["id"],
            "origin": row["origin"],
            "destination": row["destination"],
            "total_budget": row["total_budget"],
            "days": row["days"],
            "start_date": inputs.get("start_date"),
            "return_date": inputs.get("return_date"),
            "stay_comfort": inputs.get("stay_comfort"),
            "travel_comfort": inputs.get("travel_comfort"),
            "food_comfort": inputs.get("food_comfort"),
            "sightseeing_level": inputs.get("sightseeing_level"),
            "created_at": _format_ts(row["created_at"]),
        })

    return render_template("history.html", flight_history=flight_history, trip_history=trip_history)


@app.route("/history/flight/<int:check_id>")
@login_required
def history_flight_detail(check_id):
    row = db.get_flight_check_by_id(check_id)
    if not row or (row["user_id"] != session["user_id"] and not session.get("is_admin")):
        flash("That flight check couldn't be found.", "error")
        return redirect(url_for("history"))
    inputs = json.loads(row["inputs_json"])
    recommendation = json.loads(row["recommendation_json"]) if row["recommendation_json"] else None
    result = {"probability": row["delay_probability"], "label": row["risk_label"], "inputs": inputs}
    # The plan itself renders on its own page now, so a saved check links to it
    # rather than carrying the markup - otherwise a stored plan silently vanished
    # when that block moved out of flight_delay.html.
    return render_template("flight_delay.html", result=result, recommendation=None,
                            saved_plan_id=check_id if recommendation else None,
                            from_history=True, **_dropdown_options())


@app.route("/history/flight/<int:check_id>/alternatives")
@login_required
def history_flight_alternatives(check_id):
    """A plan that was saved with an earlier check, read-only."""
    row = db.get_flight_check_by_id(check_id)
    if not row or (row["user_id"] != session["user_id"] and not session.get("is_admin")):
        flash("That flight check couldn't be found.", "error")
        return redirect(url_for("history"))
    recommendation = json.loads(row["recommendation_json"]) if row["recommendation_json"] else None
    if not recommendation:
        flash("No alternative plan was saved with that check.", "error")
        return redirect(url_for("history_flight_detail", check_id=check_id))
    inputs = json.loads(row["inputs_json"])
    return render_template("alternatives.html", recommendation=recommendation,
                            flight=inputs,
                            risk={"probability": row["delay_probability"],
                                  "label": row["risk_label"]})


@app.route("/history/trip/<int:trip_id>")
@login_required
def history_trip_detail(trip_id):
    row = db.get_budget_trip_by_id(trip_id)
    if not row or (row["user_id"] != session["user_id"] and not session.get("is_admin")):
        flash("That trip plan couldn't be found.", "error")
        return redirect(url_for("history"))
    plan = json.loads(row["plan_json"])
    form_data = json.loads(row["inputs_json"])
    return render_template("budget_trip.html", plan=plan, form_data=form_data,
                            from_history=True, **_dropdown_options())


# ---------------------------------------------------------------- flight delay check

@app.route("/flight-delay", methods=["GET", "POST"])
@login_required
def flight_delay():
    result = None
    recommendation = None

    if request.method == "POST":
        stage = request.form.get("stage", "check")

        if stage == "check":
            travel_date = request.form.get("travel_date") or date.today().isoformat()
            weekday = datetime.strptime(travel_date, "%Y-%m-%d").weekday()
            carrier_code = request.form.get("carrier_code", "").strip().upper()
            flight_number = request.form.get("flight_number", "").strip()
            origin_airport = request.form.get("origin_airport", "").strip().upper()
            destination_airport = request.form.get("destination_airport", "").strip().upper()
            scheduled_elapsed_time = _float_field(request.form, "scheduled_elapsed_time", 120)
            departure_time_str = request.form.get("departure_time", "").strip()
            scheduled_hour = int(departure_time_str.split(":")[0]) if departure_time_str else 12
            # Minutes matter: reconciling a 19:15 departure that left at 19:23
            # against hour-granularity would read 23 min late instead of 8.
            scheduled_minute = (int(departure_time_str.split(":")[1])
                                if departure_time_str and ":" in departure_time_str else 0)

            flight_lookup = None
            if flight_number:
                flight_lookup = lookup_flight_by_number(carrier_code, flight_number)
                if flight_lookup:
                    origin_airport = flight_lookup["origin_airport"]
                    destination_airport = flight_lookup["destination_airport"]
                    scheduled_elapsed_time = flight_lookup["scheduled_elapsed_time"]

            if not origin_airport or not destination_airport:
                flash("Enter a flight number we have on record, or fill in the origin/destination airports manually.", "error")
                return render_template("flight_delay.html", result=None, recommendation=None,
                                        **_dropdown_options())

            inputs = {
                "carrier_code": carrier_code,
                "flight_number": flight_number,
                "origin_airport": origin_airport,
                "destination_airport": destination_airport,
                "destination_city": (request.form.get("destination_city", "").strip()
                                 or request.form.get("destination_place", "").strip()),
                "travel_date": travel_date,
                "weekday": weekday,
                "month": datetime.strptime(travel_date, "%Y-%m-%d").month,
                "scheduled_elapsed_time": scheduled_elapsed_time,
                "scheduled_hour": scheduled_hour,
                "is_holiday": is_holiday_date(travel_date),
                "origin_temp_f": _float_field(request.form, "origin_temp_f", DEFAULT_WEATHER["temp_f"]),
                # True only if the user actually typed a real reading - if this field was left
                # blank (using DEFAULT_WEATHER's placeholder), it's a guess, same as the chatbot's
                # guessed weather, and must be flagged the same way. origin_temp_known was 100%
                # correlated with US-vs-India in training, so the model leans on it heavily -
                # mismatching it between two paths using the identical guessed number produces
                # very different (wrong) predictions for the same real input.
                "origin_temp_known": bool((request.form.get("origin_temp_f") or "").strip()),
                "weather_source": "entered" if (request.form.get("origin_temp_f") or "").strip() else None,
                "origin_precip_in": _float_field(request.form, "origin_precip_in", DEFAULT_WEATHER["precip_in"]),
                "origin_pressure": _float_field(request.form, "origin_pressure", DEFAULT_WEATHER["pressure"]),
                "origin_visibility": _float_field(request.form, "origin_visibility", DEFAULT_WEATHER["visibility"]),
                "origin_wind_speed": _float_field(request.form, "origin_wind_speed", DEFAULT_WEATHER["wind_speed"]),
            }

            # Nobody fills in five weather fields, so they defaulted to
            # placeholders on almost every check. A real forecast is free,
            # keyless and available 16 days out, so blank fields are filled
            # from it instead.
            if not inputs["origin_temp_known"]:
                try:
                    forecast = airport_weather(
                        inputs["origin_airport"],
                        datetime.strptime(f"{travel_date} {scheduled_hour:02d}:00",
                                          "%Y-%m-%d %H:%M"))
                    if forecast:
                        for form_field, key in [
                            ("origin_temp_f", "temp_f"), ("origin_precip_in", "precip_in"),
                            ("origin_pressure", "pressure"), ("origin_visibility", "visibility"),
                            ("origin_wind_speed", "wind_speed"),
                        ]:
                            if not (request.form.get(form_field) or "").strip():
                                inputs[form_field] = forecast[key]
                        inputs["weather_source"] = "forecast"
                        # origin_temp_known stays False on purpose. In training it
                        # was 100% correlated with US-vs-India, so it encodes which
                        # dataset a row came from more than whether anyone read a
                        # thermometer - and setting it True here cost 0.611 -> 0.570.
                except Exception as e:
                    print(f"[api] forecast lookup skipped: {e}")

            # Destination weather, at the SCHEDULED arrival time - never the actual
            # one, which would leak the outcome (a late flight lands later). This
            # used to be withheld entirely: an earlier check against 139 real but
            # unrepresentative outcomes showed it hurting (0.713 -> 0.603), so the
            # model only ever saw dest_weather_known=0 in production and effectively
            # never trained on that signal at serving time either. Re-measured
            # properly on 4M real BTS flights the retrained model never trained on
            # (see src/compare_delay_models.py), arrival weather helps more than
            # departure weather does (0.773 -> 0.861 AUC with both ends supplied).
            # None on any failure - the model falls back to its own training mean
            # for dest_* with dest_weather_known=0, same as before, never a guess.
            inputs["dest_weather"] = None
            try:
                dep_dt = datetime.strptime(f"{travel_date} {scheduled_hour:02d}:{scheduled_minute:02d}",
                                           "%Y-%m-%d %H:%M")
                arr_dt = dep_dt + timedelta(minutes=inputs["scheduled_elapsed_time"] or 0)
                inputs["dest_weather"] = airport_weather(inputs["destination_airport"], arr_dt)
            except Exception as e:
                print(f"[api] destination forecast lookup skipped: {e}")

            # Live aircraft rotation: find the real aircraft due to fly this, see
            # where it has been today, and how much ground time is left before the
            # SCHEDULED push. That fills prev_leg_arrival_delay, which the form
            # cannot supply and which the model weights heavily.
            rotation = None
            airline_estimate = None
            if flight_number and carrier_code:
                try:
                    sched_ts = int(datetime.strptime(
                        f"{travel_date} {scheduled_hour:02d}:{scheduled_minute:02d}",
                        "%Y-%m-%d %H:%M").timestamp())

                    # The airline's schedule names the airframe, which receivers
                    # cannot do for a flight that has not departed. That removes
                    # the step that made the rotation check retrospective, so it
                    # is tried first and the receiver-only path stays as fallback.
                    # Only for today. The free plan's flights endpoint takes no
                    # date, so it always answers with today's instance of this
                    # number - presenting that against a departure next week
                    # would show one day's delay as another's.
                    live = (flight_now(f"{carrier_code}{flight_number}")
                            if travel_date == date.today().isoformat() else None)
                    if live:
                        airline_estimate = {
                            "delay_min": live.get("departure_delay_min"),
                            "status": live.get("status"),
                            "scheduled": live.get("scheduled_departure"),
                            "estimated": live.get("estimated_departure"),
                            "terminal": live.get("terminal"),
                            "gate": live.get("gate"),
                        }
                    if live and live.get("icao24"):
                        rotation = rotation_from_icao24(
                            live["icao24"], inputs["origin_airport"], sched_ts)
                    else:
                        rotation = live_rotation(carrier_code, flight_number,
                                                  inputs["origin_airport"], sched_ts)
                except Exception as e:
                    print(f"[api] live rotation lookup skipped: {e}")

            covered = is_route_covered(inputs["origin_airport"], inputs["destination_airport"])
            reference_flights = [] if covered else get_reference_flights(
                inputs["origin_airport"], inputs["destination_airport"])

            lookup_note = None
            if flight_number and flight_lookup:
                lookup_note = (f"Found flight {carrier_code} {flight_number}: {origin_airport} → "
                               f"{destination_airport}, seen {flight_lookup['occurrences']} time(s) "
                               f"on this route in our 2019 records.")
            elif flight_number and not flight_lookup:
                lookup_note = (f"Flight {carrier_code} {flight_number} isn't in our 2019 records"
                               + (f" — using the route you entered manually ({origin_airport} → "
                                  f"{destination_airport}) instead." if request.form.get("origin_airport") else "."))

            if not covered and not reference_flights:
                result = {"no_data": True, "inputs": inputs, "lookup_note": lookup_note}
            elif not covered:
                result = {"reference_only": True, "inputs": inputs, "reference_flights": reference_flights,
                          "lookup_note": lookup_note}
            else:
                prob = predict_delay_probability(
                    carrier_code=inputs["carrier_code"], origin_airport=inputs["origin_airport"],
                    destination_airport=inputs["destination_airport"], weekday=inputs["weekday"],
                    month=inputs["month"], scheduled_elapsed_time=inputs["scheduled_elapsed_time"],
                    origin_temp_f=inputs["origin_temp_f"], origin_temp_known=inputs["origin_temp_known"],
                    origin_precip_in=inputs["origin_precip_in"],
                    origin_pressure=inputs["origin_pressure"], origin_visibility=inputs["origin_visibility"],
                    origin_wind_speed=inputs["origin_wind_speed"],
                    scheduled_hour=inputs["scheduled_hour"], is_holiday=inputs["is_holiday"],
                    # Only supplied when a real aircraft was actually observed;
                    # None keeps the model on its prev_leg_known=0 path.
                    prev_leg_arrival_delay=(rotation.get("inferred_inbound_delay_min")
                                            if rotation and rotation.get("observed") else None),
                    dest_weather=inputs["dest_weather"],
                )
                label = risk_label(prob)
                result = {"probability": prob, "label": label, "inputs": inputs,
                          "lookup_note": lookup_note, "rotation": rotation,
                          "airline_estimate": airline_estimate}
                # Real price + real scheduled departure/arrival clock time (Google
                # Flights, via SerpApi) for this exact route/date - the SAME call
                # already used on the Trip Planner and recovery pages, now here too
                # since this is the highest-traffic page in the app. Quota note:
                # SerpApi's free plan is 250 searches/month shared across this whole
                # project; this adds one call per DISTINCT route+date checked (the
                # 6h cache absorbs repeat checks of the same flight, which is most
                # real usage during testing/a demo) - watch data/serpapi_usage.json
                # if checks of many different routes start happening often.
                try:
                    result["real_price"] = real_flight_price(
                        inputs["origin_airport"], inputs["destination_airport"],
                        travel_date, adults=1)
                except Exception as e:
                    print(f"[api] real flight price skipped: {e}")
                    result["real_price"] = None
                # "How likely" comes from the trained model; "how long and why" is a
                # real historical statistic for flights like this one that actually
                # were delayed (delay_duration.py) - the classifier can't say either.
                if label != "Low":
                    result["duration"] = estimate_delay_duration(
                        inputs["carrier_code"], inputs["origin_airport"],
                        inputs["destination_airport"], inputs["scheduled_hour"])
                # Why THIS flight scores what it does, attributed against the
                # conditions actually entered - see explain_delay.py. Distinct
                # from `duration`, which is the route's history and does not
                # respond to the weather on this form.
                result["why"] = explain_delay(
                    carrier_code=inputs["carrier_code"],
                    origin_airport=inputs["origin_airport"],
                    destination_airport=inputs["destination_airport"],
                    weekday=inputs["weekday"], month=inputs["month"],
                    scheduled_elapsed_time=inputs["scheduled_elapsed_time"],
                    origin_temp_f=inputs["origin_temp_f"],
                    origin_temp_known=inputs["origin_temp_known"],
                    origin_precip_in=inputs["origin_precip_in"],
                    origin_pressure=inputs["origin_pressure"],
                    origin_visibility=inputs["origin_visibility"],
                    origin_wind_speed=inputs["origin_wind_speed"],
                    scheduled_hour=inputs["scheduled_hour"],
                    is_holiday=inputs["is_holiday"])
                db.save_flight_check(session["user_id"], inputs, prob, label)
                session["last_flight_check"] = inputs
                session["last_flight_risk"] = {"probability": prob, "label": label}

    return render_template("flight_delay.html", result=result, recommendation=recommendation,
                            **_dropdown_options())


@app.route("/flight-delay/alternatives", methods=["GET", "POST"])
@login_required
def flight_alternatives():
    """The recovery plan, on its own page.

    This used to render underneath the delay result on /flight-delay, so the
    form and then the plan stacked below a number you had already read. It
    reads the same session state the old "recover" stage did.
    """
    inputs = session.get("last_flight_check")
    risk = session.get("last_flight_risk")
    if not inputs or not risk:
        flash("Check a flight first, then we can plan around its delay.", "error")
        return redirect(url_for("flight_delay"))

    recommendation = None
    if request.method == "POST":
        display_currency = session.get("currency", currency_utils.DEFAULT_CURRENCY)

        def _money_field(name, default):
            # The labels carry the display currency, so what was typed has to
            # come back to USD before the optimiser sees it as a constraint.
            return currency_utils.to_usd(_float_field(request.form, name, default),
                                          display_currency)

        budget = {
            "flight_cost": _money_field("flight_cost", 150),
            "hotel_cost": _money_field("hotel_cost", 400),
            "food_cost": _money_field("food_cost", 250),
            "transport_cost": _money_field("transport_cost", 150),
            "sightseeing_cost": _money_field("sightseeing_cost", 150),
        }
        priority = request.form.get("priority", "cost")

        # Party size, carried through to the booking site so nobody retypes it.
        # Infants are capped at one per adult, which is what every airline and
        # every one of these sites enforces anyway.
        adults = max(1, min(_int_field(request.form, "adults", 1), 9))
        children = max(0, min(_int_field(request.form, "children", 0), 8))
        infants = max(0, min(_int_field(request.form, "infants", 0), adults))
        session["party"] = {"adults": adults, "children": children, "infants": infants}

        # Categories the traveller has already paid for - these must not be
        # trimmed to fund the disruption (see budget_optimizer.reallocate_budget).
        committed = [c for c in request.form.getlist("committed") if c.endswith("_cost")]

        planned_spots = []
        for i in (1, 2, 3):
            name = request.form.get(f"spot_name_{i}", "").strip()
            if name:
                planned_spots.append({"name": name,
                                      "cost_usd": _money_field(f"spot_cost_{i}", 0)})

        state = {**inputs, "priority": priority, "budget": budget,
                 "committed": committed, "planned_spots": planned_spots,
                 "adults": adults, "children": children, "infants": infants,
                 "stay_name": request.form.get("stay_name", "").strip()}
        # get(), not pop(): the recovery agent's own booking_link tool and the
        # real-price lookup both read trip["travel_date"] too - popping it here
        # silently left both with no date to search against.
        travel_date = state.get("travel_date")
        outcome = _recovery_app.invoke(state)
        outcome["assumed_hotel_nights"] = ASSUMED_RECOVERY_HOTEL_NIGHTS

        # Every option gets a link, not only the one the agent singled out - the
        # agent's pick is a recommendation, not a restriction on what you may book.
        for f in outcome.get("alternative_flights") or []:
            f["booking"] = links_for_flight(f, travel_date, adults, children, infants)
        recommendation = outcome
        db.save_flight_check(session["user_id"], inputs, risk["probability"],
                             risk["label"], outcome)

    party = session.get("party") or {"adults": 1, "children": 0, "infants": 0}
    return render_template("alternatives.html", recommendation=recommendation,
                            flight=inputs, risk=risk, party=party,
                            party_label=describe_party(**party))


# ---------------------------------------------------------------- budget trip planner

@app.route("/budget-trip", methods=["GET", "POST"])
@login_required
def budget_trip():
    plan = None
    form_data = {}
    transport_options = None

    if request.method == "POST":
        form_data = request.form.to_dict()
        # to_dict() keeps only the LAST value of a repeated field - the interest
        # checkboxes submit several "interests" values, so this needs getlist()
        # or every box but the last would silently uncheck itself on re-render.
        form_data["interests"] = request.form.getlist("interests")
        start_date_str = request.form.get("start_date") or date.today().isoformat()
        days = _int_field(request.form, "days", 5)
        start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
        return_date = start_date + timedelta(days=days)
        stage = request.form.get("stage", "plan")

        # Stage 1: just compare how to get there. Only modes with a real data
        # source come back available - see transport_modes.py.
        if stage == "compare":
            transport_options = compare_transport_modes(
                origin_place=request.form.get("origin_place", "").strip(),
                destination_place=request.form.get("destination_place", "").strip(),
                weekday=start_date.weekday(),
            )
            return render_template("budget_trip.html", plan=None, form_data=form_data,
                                    transport_options=transport_options, **_dropdown_options())

        inputs = {
            "origin_place": request.form.get("origin_place", "").strip(),
            "destination_place": request.form.get("destination_place", "").strip(),
            "transport_mode": request.form.get("transport_mode", "").strip(),
            "origin_airport": request.form.get("origin_airport", "").strip().upper(),
            "destination_airport": request.form.get("destination_airport", "").strip().upper(),
            "destination_city": (request.form.get("destination_city", "").strip()
                                 or request.form.get("destination_place", "").strip()),
            # The field is labelled in the display currency, so what was typed
            # has to come back to USD before it constrains anything - the
            # optimiser and the hotel tiers are dollar-denominated.
            "total_budget": currency_utils.to_usd(
                _float_field(request.form, "total_budget", 1000),
                session.get("currency", currency_utils.DEFAULT_CURRENCY)),
            "days": days,
            "start_date": start_date_str,
            "return_date": return_date.strftime("%Y-%m-%d"),
            "stay_comfort": request.form.get("stay_comfort", "standard"),
            "travel_comfort": request.form.get("travel_comfort", "standard"),
            "food_comfort": request.form.get("food_comfort", "standard"),
            "sightseeing_level": request.form.get("sightseeing_level", "moderate"),
            # Capped: the hotel tiers are per-room medians, and beyond a dozen
            # people a trip is a group booking this planner cannot price.
            "travelers": max(1, min(_int_field(request.form, "travelers", 1), 12)),
        }

        if not inputs["origin_airport"] and inputs["origin_place"]:
            allowed = set(get_dataset_index()["airports"]) | INTL_AIRPORT_CODES
            resolved = nearest_supported_airport(inputs["origin_place"], allowed)
            if resolved:
                inputs["origin_airport"] = resolved["airport_code"]

        plan = plan_budget_trip(
            origin_airport=inputs["origin_airport"],
            destination_airport=inputs["destination_airport"],
            destination_city=inputs["destination_city"],
            total_budget=inputs["total_budget"],
            days=inputs["days"],
            start_weekday=start_date.weekday(),
            return_weekday=return_date.weekday(),
            stay_comfort=inputs["stay_comfort"],
            travel_comfort=inputs["travel_comfort"],
            food_comfort=inputs["food_comfort"],
            sightseeing_level=inputs["sightseeing_level"],
            transport_mode=inputs["transport_mode"],
            travelers=inputs["travelers"],
        )
        # Real OTA search links for each flight option shown on the Trip Planner
        # page - same builder the recovery agent already uses (booking.py), so
        # "Book flight" on this page and on the recovery plan go through the
        # identical real URL grammar. One real date per leg (outbound uses the
        # real start date, return the real return date), real party size.
        for leg_flights, leg_date in ((plan.get("outbound_flights"), inputs["start_date"]),
                                      (plan.get("return_flights"), inputs["return_date"])):
            for f in (leg_flights or []):
                try:
                    f["booking"] = links_for_flight(f, leg_date, adults=inputs["travelers"])
                except Exception as e:
                    print(f"[api] booking link skipped for {f.get('route')}: {e}")
                    f["booking"] = []

        # Real market prices (SerpApi -> Google Flights/Hotels) - the one real
        # price source anywhere in this project. Quota-guarded (see
        # serpapi_prices.py); None/[] when the key/quota/lookup isn't
        # available, same as every other optional enrichment here, so the page
        # falls back to its existing "no fare data" language rather than
        # breaking when this is unavailable.
        if plan.get("route_covered") or plan.get("route_reference"):
            try:
                plan["real_outbound_price"] = real_flight_price(
                    inputs["origin_airport"], inputs["destination_airport"],
                    inputs["start_date"], adults=inputs["travelers"])
            except Exception as e:
                print(f"[api] real flight price skipped: {e}")
                plan["real_outbound_price"] = None
            try:
                plan["real_return_price"] = real_flight_price(
                    inputs["destination_airport"], inputs["origin_airport"],
                    inputs["return_date"], adults=inputs["travelers"])
            except Exception as e:
                print(f"[api] real return price skipped: {e}")
                plan["real_return_price"] = None
        try:
            plan["real_priced_hotels"] = real_hotel_prices(
                inputs["destination_city"] or inputs["destination_place"],
                inputs["start_date"], inputs["return_date"],
                adults=inputs["travelers"])
        except Exception as e:
            print(f"[api] real hotel prices skipped: {e}")
            plan["real_priced_hotels"] = []

        plan["transport_mode"] = inputs["transport_mode"]
        # Real numbers for the mode actually chosen, so a traveller who picked the
        # car sees their drive rather than flights they never asked for.
        if inputs["transport_mode"] and inputs["origin_place"] and inputs["destination_place"]:
            plan["selected_transport"] = describe_mode(
                inputs["origin_place"], inputs["destination_place"],
                inputs["transport_mode"], start_date.weekday())
        if plan.get("auto_resolved_destination_airport"):
            inputs["destination_airport"] = plan["auto_resolved_destination_airport"]
        narrative, mode = narrate_trip_plan(inputs, plan)
        plan["recommendation_text"] = narrative
        plan["recommendation_mode"] = mode
        db.save_budget_trip(session["user_id"], inputs, plan)

    return render_template("budget_trip.html", plan=plan, form_data=form_data,
                            transport_options=transport_options, **_dropdown_options())


# ---------------------------------------------------------------- admin

@app.route("/admin")
@login_required
@admin_required
def admin():
    stats = db.get_activity_stats()
    users = db.get_all_users()
    checks = [
        {**dict(row), "created_at": _format_ts(row["created_at"])}
        for row in db.get_all_flight_checks(limit=25)
    ]
    trips = [
        {**dict(row), "created_at": _format_ts(row["created_at"])}
        for row in db.get_all_budget_trips(limit=25)
    ]
    return render_template("admin.html", stats=stats, users=users, checks=checks, trips=trips,
                            llm=llm_status(), model_metrics=load_model_metrics(),
                            validation_examples=sample_validation_flights(6))


def _cors(resp):
    """Open CORS for the planner API so a frontend served from a dev server
    (vite on 5173, live-server on 5500) can call it. Read-only endpoints only."""
    resp.headers["Access-Control-Allow-Origin"] = request.headers.get("Origin", "*")
    resp.headers["Access-Control-Allow-Methods"] = "POST, GET, OPTIONS"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return resp


@app.route("/api/plan-trip", methods=["POST", "OPTIONS"])
def api_plan_trip():
    """Budget-checked itinerary: flight, hotel, attractions, total.

    Runs the LangGraph pipeline in src/trip_graph.py, which sequences the three
    existing modules - the trained classifier scores every candidate flight, the
    recovery rule swaps a High-risk pick for a genuinely safer one, and the
    budget optimiser fits the stay and sights into what is left.

    422 on invalid input rather than silently coercing: a zero budget or a
    reversed date is a mistake worth surfacing, not something to guess around.
    """
    if request.method == "OPTIONS":
        return _cors(app.make_response(("", 204)))

    body = request.get_json(silent=True) or {}
    errors = []

    from_city = str(body.get("from_city", "")).strip()
    to_city = str(body.get("to_city", "")).strip()
    if not from_city or not to_city:
        errors.append("from_city and to_city are required.")
    elif from_city.lower() == to_city.lower():
        errors.append("from_city and to_city must differ.")

    try:
        travelers = int(body.get("travelers", 1))
        if travelers < 1:
            errors.append("travelers must be at least 1.")
    except (TypeError, ValueError):
        travelers = 1
        errors.append("travelers must be a whole number.")

    try:
        budget = float(body.get("budget", 0))
        if budget <= 0:
            errors.append("budget must be greater than zero.")
    except (TypeError, ValueError):
        budget = 0.0
        errors.append("budget must be a number.")

    raw_date = str(body.get("date", "")).strip()
    travel_date = None
    try:
        travel_date = datetime.strptime(raw_date, "%Y-%m-%d").date()
    except ValueError:
        errors.append("date must be in YYYY-MM-DD format.")

    if errors:
        return _cors(app.make_response(({"error": "Invalid request", "details": errors}, 422)))

    result = run_trip_graph(from_city, to_city, travel_date, travelers, budget)
    return _cors(app.make_response((result, 200)))


@app.route("/frontend/<path:filename>")
def frontend_asset(filename):
    """Static assets for the planner page."""
    return send_from_directory(os.path.join(PROJECT_ROOT, "frontend"), filename)


@app.route("/plan")
def plan_page():
    """Vanilla-JS planner UI. Separate from the 3D showcase at /, which stays."""
    return send_file(os.path.join(PROJECT_ROOT, "frontend", "index.html"))


@app.route("/api/live-traffic")
def api_live_traffic():
    """Aircraft currently flying the corridor between two airports.

    The OpenSky credentials stay server-side - the browser asks this endpoint
    for a route, never the network directly. Positions are real transponder
    reports, which is the one question the network can answer about a flight
    that has not departed: not which airframe is assigned to it, but what is
    in that airspace right now.

    Unknown airport codes return an empty list rather than an error, so the map
    simply does not draw instead of breaking the page around it.
    """
    origin = (request.args.get("origin") or "").strip().upper()[:4]
    dest = (request.args.get("dest") or "").strip().upper()[:4]
    box = bbox_for_route(origin, dest)
    if not box:
        return {"aircraft": [], "route": None,
                "note": "no published coordinates for one of these airports"}

    from reference_data import AIRPORT_COORDS
    mine = callsign_for(request.args.get("carrier", ""), request.args.get("flight", ""))
    aircraft = live_traffic(box, highlight_callsign=mine)
    return {
        "aircraft": aircraft,
        "route": {
            "origin": {"iata": origin, "lat": AIRPORT_COORDS[origin][0],
                       "lon": AIRPORT_COORDS[origin][1]},
            "destination": {"iata": dest, "lat": AIRPORT_COORDS[dest][0],
                            "lon": AIRPORT_COORDS[dest][1]},
        },
        "your_callsign": mine,
        "count": len(aircraft),
    }


@app.route("/api/delay-prediction")
def api_delay_prediction():
    """Pre-departure delay prediction as a plain JSON contract:
    flight_number, date, origin, destination, scheduled_departure,
    delay_probability, expected_delay_minutes, risk_severity, prediction_type.

    GET /api/delay-prediction?flight_number=AA101&date=2026-10-10
    GET /api/delay-prediction?carrier=AA&origin=JFK&destination=LAX&date=2026-10-10&departure_time=08:00

    flight_number alone only resolves a route if that exact carrier+number
    appears in the training catalogue (2019 US flights) - this project has no
    live flight-number-to-route schedule lookup (that would need AviationStack,
    which is quota-limited - see booking.py/aviationstack.py elsewhere in the
    app). Pass origin/destination explicitly for anything else; the response
    says which path was used.

    risk_severity is this app's own real, calibration-checked three tiers
    (LOW/MODERATE/HIGH - see predict_delay_v2.risk_label), not an arbitrary
    four-way percentage split - a fourth "VERY HIGH" cutoff would be a number
    nobody measured, so none is offered here. expected_delay_minutes is 0 for
    LOW (a delay is unlikely enough that there's nothing real to plan around);
    for MODERATE/HIGH it's the real median minutes among flights like this one
    that WERE actually delayed (delay_duration.py) - a conditional statistic,
    not a promise about this specific flight.
    """
    flight_number_raw = (request.args.get("flight_number") or "").strip().upper()
    carrier_code = (request.args.get("carrier") or "").strip().upper()
    flight_number = ""
    if flight_number_raw:
        m = re.match(r"^([A-Z]{2,3})(\d+)$", flight_number_raw)
        if m:
            carrier_code = carrier_code or m.group(1)
            flight_number = m.group(2)
        else:
            flight_number = flight_number_raw

    date_str = (request.args.get("date") or "").strip()
    try:
        travel_date = datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        return {"error": "date is required, as YYYY-MM-DD"}, 422

    departure_time = (request.args.get("departure_time") or "12:00").strip()
    try:
        dep_hour, dep_minute = (int(x) for x in departure_time.split(":")[:2])
    except ValueError:
        return {"error": "departure_time must be HH:MM"}, 422

    origin = (request.args.get("origin") or "").strip().upper()
    destination = (request.args.get("destination") or "").strip().upper()
    # Optional override; stays None (not a guessed default) until the fallback
    # chain below resolves it from a catalogue lookup or the final 120min default.
    _dur_raw = (request.args.get("scheduled_elapsed_time") or "").strip()
    try:
        scheduled_elapsed_time = float(_dur_raw) if _dur_raw else None
    except ValueError:
        return {"error": "scheduled_elapsed_time must be a number"}, 422
    route_source = "entered" if (origin and destination) else None

    if (not origin or not destination) and carrier_code and flight_number:
        lookup = lookup_flight_by_number(carrier_code, flight_number)
        if lookup:
            origin, destination = lookup["origin_airport"], lookup["destination_airport"]
            scheduled_elapsed_time = scheduled_elapsed_time or lookup["scheduled_elapsed_time"]
            route_source = "2019 training catalogue"

    if not origin or not destination:
        return {"error": ("Could not resolve a route. Pass origin and destination directly, "
                          "or a flight_number this app's 2019 catalogue actually has.")}, 422
    if not carrier_code:
        return {"error": "carrier is required (either directly, or embedded in flight_number, e.g. AA101)."}, 422

    scheduled_elapsed_time = scheduled_elapsed_time or 120
    dep_dt = datetime.combine(travel_date, datetime.min.time()).replace(hour=dep_hour, minute=dep_minute)

    if not is_route_covered(origin, destination):
        reference = get_reference_flights(origin, destination)
        return {
            "flight_number": flight_number_raw or None, "date": date_str,
            "origin": origin, "destination": destination,
            "scheduled_departure": departure_time,
            "delay_probability": None, "expected_delay_minutes": None,
            "risk_severity": "NOT_MODELED",
            "prediction_type": "pre_departure",
            "note": ("This route isn't in the trained dataset, so no real delay-risk number "
                    "exists for it." + (" A reference schedule (not a risk prediction) exists: "
                    f"{reference[0]['carrier']}, ~{reference[0]['scheduled_elapsed_time']}min."
                    if reference else " No reference schedule exists either.")),
        }, 200

    origin_weather = None
    try:
        origin_weather = airport_weather(origin, dep_dt)
    except Exception as e:
        print(f"[api] delay-prediction origin forecast skipped: {e}")
    dest_weather = None
    try:
        arr_dt = dep_dt + timedelta(minutes=scheduled_elapsed_time)
        dest_weather = airport_weather(destination, arr_dt)
    except Exception as e:
        print(f"[api] delay-prediction destination forecast skipped: {e}")

    prob = predict_delay_probability(
        carrier_code=carrier_code, origin_airport=origin, destination_airport=destination,
        weekday=travel_date.weekday(), month=travel_date.month,
        scheduled_elapsed_time=scheduled_elapsed_time,
        origin_temp_f=(origin_weather or DEFAULT_WEATHER)["temp_f"],
        origin_temp_known=origin_weather is not None,
        origin_precip_in=(origin_weather or DEFAULT_WEATHER)["precip_in"],
        origin_pressure=(origin_weather or DEFAULT_WEATHER)["pressure"],
        origin_visibility=(origin_weather or DEFAULT_WEATHER)["visibility"],
        origin_wind_speed=(origin_weather or DEFAULT_WEATHER)["wind_speed"],
        scheduled_hour=dep_hour, is_holiday=is_holiday_date(date_str),
        dest_weather=dest_weather,
    )
    label = risk_label(prob)

    expected_delay_minutes = 0
    if label != "Low":
        duration = estimate_delay_duration(carrier_code, origin, destination, dep_hour)
        expected_delay_minutes = duration["median_min"] if duration else None

    return {
        "flight_number": flight_number_raw or None,
        "date": date_str,
        "origin": origin,
        "destination": destination,
        "scheduled_departure": departure_time,
        "delay_probability": round(prob, 4),
        "expected_delay_minutes": expected_delay_minutes,
        "risk_severity": label.upper(),
        "prediction_type": "pre_departure",
        # Beyond the requested contract, kept because silently dropping how the
        # number was produced is how a real prediction gets mistaken for a live
        # schedule lookup - see this route's own docstring.
        "route_source": route_source,
        "weather_source": "forecast" if origin_weather else "no forecast available (clear-weather default used)",
    }, 200


# ---------------------------------------------------------------- live flight search
# Real-time flights (Aviationstack) + weather (Open-Meteo) -> the existing delay
# model. See src/flight_search.py. Login is required on all of it: every uncached
# search spends part of a ~100-calls-a-month API allowance.

def _api_login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if not session.get("user_id"):
            return {"error": {"code": "login_required", "message": "Please log in."}}, 401
        return view(*args, **kwargs)
    return wrapped


def _validation_message(e: ValidationError) -> str:
    return "; ".join(err["msg"].removeprefix("Value error, ") for err in e.errors())


def _with_display_fare(flight: dict) -> dict:
    fare = fs.optional_fare(flight)
    flight["fare"] = ({
        "display": currency_utils.format_money(
            fare["price_usd"], session.get("currency", currency_utils.DEFAULT_CURRENCY)),
        "airline": fare.get("airline"), "stops": fare.get("stops"),
        "source": "Google Flights via SerpApi (cheapest option on this route and date, per adult)",
        "same_flight": False,
    } if fare else None)
    return flight


@app.route("/flight-search")
@login_required
def flight_search_page():
    return render_template("flight_search.html", **_dropdown_options())


@app.route("/api/flights/search")
@_api_login_required
def api_flights_search():
    try:
        q = fs.SearchQuery(origin=request.args.get("origin"),
                           destination=request.args.get("destination"),
                           flight_number=request.args.get("flight_number"),
                           date=request.args.get("date"))
    except ValidationError as e:
        return {"error": {"code": "invalid_request", "message": _validation_message(e)}}, 400
    try:
        return fs.search_flights(q), 200
    except fs.FlightSearchError as e:
        return {"error": {"code": e.code, "message": e.message}}, e.status
    except Exception as e:  # never a stack trace to the browser
        print(f"[api] flight search failed: {e}")
        return {"error": {"code": "internal_error",
                          "message": "Something went wrong while searching. Please try again."}}, 500


@app.route("/api/flights/<flight_id>")
@_api_login_required
def api_flight_details(flight_id):
    try:
        out = fs.flight_details(flight_id)
        if request.args.get("include_price") == "1":
            _with_display_fare(out["flight"])
        return out, 200
    except fs.FlightSearchError as e:
        return {"error": {"code": e.code, "message": e.message}}, e.status
    except Exception as e:
        print(f"[api] flight details failed: {e}")
        return {"error": {"code": "internal_error",
                          "message": "Couldn't load that flight. Please try again."}}, 500


@app.route("/api/flights/predict-delay", methods=["POST"])
@_api_login_required
def api_flights_predict():
    try:
        req = fs.PredictRequest(**(request.get_json(silent=True) or {}))
    except ValidationError as e:
        return {"error": {"code": "invalid_request", "message": _validation_message(e)}}, 400
    flight = {
        "airline_iata": req.airline_iata, "origin_iata": req.origin_iata,
        "destination_iata": req.destination_iata,
        "scheduled_departure": req.scheduled_departure, "scheduled_arrival": req.scheduled_arrival,
        "flight_number": req.flight_number, "origin_timezone": req.origin_timezone,
        "destination_timezone": req.destination_timezone, "status": req.status,
    }
    try:
        prediction = fs.predict_flight(flight)
    except Exception as e:
        print(f"[api] predict-delay failed: {e}")
        return {"error": {"code": "model_error",
                          "message": "The delay model couldn't score this flight."}}, 500
    return {"flight": flight, "weather": prediction.get("weather"), "prediction": prediction}, 200


@app.route("/api/showcase-flight")
def api_showcase_flight():
    """Real model output for the showcase landing page.

    The showcase is the logged-out entry point, so this is deliberately public
    and read-only: it runs the same trained model and the same historical
    duration/cause lookup the product uses, on a caller-supplied route, and
    returns exactly what those produce. No login, no writes, no free text
    reaching anything but an airport-code lookup.

    Inputs are clamped to codes the dataset actually covers - an uncovered route
    would otherwise get a confident-looking score the model has no basis for.
    """
    index = get_dataset_index()
    known = set(index["airports"]) | INTL_AIRPORT_CODES
    carriers = set(index["carriers"])

    carrier = (request.args.get("carrier", "AI") or "").upper()[:3]
    origin = (request.args.get("from", "MAA") or "").upper()[:4]
    dest = (request.args.get("to", "DEL") or "").upper()[:4]
    hour = max(0, min(23, _int_field(request.args, "hour", 10)))

    if origin not in known or dest not in known or origin == dest:
        return {"available": False,
                "reason": f"No trained flight data covers {origin} to {dest}."}, 200
    if carrier not in carriers:
        carrier = sorted(carriers)[0]

    prob = predict_delay_probability(
        carrier_code=carrier, origin_airport=origin, destination_airport=dest,
        weekday=4, month=10, scheduled_elapsed_time=150,
        origin_temp_f=DEFAULT_WEATHER["temp_f"], origin_precip_in=DEFAULT_WEATHER["precip_in"],
        origin_pressure=DEFAULT_WEATHER["pressure"], origin_visibility=DEFAULT_WEATHER["visibility"],
        origin_wind_speed=DEFAULT_WEATHER["wind_speed"], scheduled_hour=hour,
    )
    duration = estimate_delay_duration(carrier, origin, dest, hour)
    metrics = load_model_metrics() or {}

    return {
        "available": True,
        "carrier": carrier, "from": origin, "to": dest,
        "delay_probability": round(prob, 4),
        "risk_label": risk_label(prob),
        # None when the lookup hasn't been built - the UI must not invent one
        "median_delay_min": duration["median_min"] if duration else None,
        "p90_delay_min": duration["p90_min"] if duration else None,
        "top_cause": duration["top_cause"] if duration else None,
        "causes": duration["causes"] if duration else {},
        "sample_size": duration["sample_size"] if duration else None,
        # Which tier answered. "overall" means we had nothing for this route and
        # fell back to the national average - the UI must not present that as
        # specific to the flight.
        "basis": duration["basis"] if duration else None,
        "basis_label": duration["basis_label"] if duration else None,
        "model_auc": metrics.get("auc"),
    }


@app.route("/api/places")
@login_required
def api_places():
    """City suggestions for the trip form's location pickers. Cities only - see
    places.autocomplete_places for why free text was a problem."""
    return {"results": autocomplete_places(request.args.get("q", ""))}


# ---------------------------------------------------------------- chatbot

@app.route("/api/chat", methods=["POST"])
@login_required
def api_chat():
    data = request.get_json(silent=True) or {}
    message = (data.get("message") or "").strip()
    if not message:
        return {"reply": "Type a question and I'll help."}

    history_key = "chat_history"
    chat_history = session.get(history_key, [])
    reply = run_chat(chat_history, message)
    chat_history.append({"role": "user", "content": message})
    chat_history.append({"role": "assistant", "content": reply})
    session[history_key] = chat_history[-20:]
    return {"reply": reply}


if __name__ == "__main__":
    # use_reloader=False: the "flight/" venv folder lives inside this project
    # root, and Werkzeug's file watcher recursively stats everything under
    # the working directory, which causes a spurious restart loop against it.
    app.run(debug=True, use_reloader=False)
