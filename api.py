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
import sys
import json
from datetime import date, datetime, timedelta
from functools import wraps

from flask import Flask, render_template, request, redirect, url_for, session, flash
from werkzeug.security import generate_password_hash, check_password_hash
import markdown as _markdown
import bleach

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(PROJECT_ROOT, "src"))

import db
from predict_delay_v2 import predict_delay_probability, risk_label
from recovery_graph import build_graph
from recovery_tools import get_dataset_index, is_route_covered, lookup_flight_by_number
from trip_planner import plan_budget_trip
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
app.jinja_env.filters["money"] = lambda usd: currency_utils.format_money(usd, session.get("currency", "USD"))

db.init_db()
_recovery_app = build_graph()

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
        "current_currency": session.get("currency", "USD"),
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


# ---------------------------------------------------------------- auth

@app.route("/")
def index():
    if session.get("user_id"):
        return redirect(url_for("home"))
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
            session["currency"] = "USD"
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
            session["currency"] = user["preferred_currency"] or "USD"
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
    currency = request.form.get("currency", "USD").upper()
    if currency not in currency_utils.SUPPORTED_CURRENCIES:
        currency = "USD"
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
    return render_template("flight_delay.html", result=result, recommendation=recommendation,
                            from_history=True, **_dropdown_options())


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
                "origin_precip_in": _float_field(request.form, "origin_precip_in", DEFAULT_WEATHER["precip_in"]),
                "origin_pressure": _float_field(request.form, "origin_pressure", DEFAULT_WEATHER["pressure"]),
                "origin_visibility": _float_field(request.form, "origin_visibility", DEFAULT_WEATHER["visibility"]),
                "origin_wind_speed": _float_field(request.form, "origin_wind_speed", DEFAULT_WEATHER["wind_speed"]),
            }

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
                )
                label = risk_label(prob)
                result = {"probability": prob, "label": label, "inputs": inputs, "lookup_note": lookup_note}
                db.save_flight_check(session["user_id"], inputs, prob, label)
                session["last_flight_check"] = inputs
                session["last_flight_risk"] = {"probability": prob, "label": label}

        elif stage == "recover":
            inputs = session.get("last_flight_check")
            risk = session.get("last_flight_risk")
            if not inputs or not risk:
                flash("Please check your flight again before requesting alternatives.", "error")
                return redirect(url_for("flight_delay"))

            budget = {
                "flight_cost": _float_field(request.form, "flight_cost", 150),
                "hotel_cost": _float_field(request.form, "hotel_cost", 400),
                "food_cost": _float_field(request.form, "food_cost", 250),
                "transport_cost": _float_field(request.form, "transport_cost", 150),
                "sightseeing_cost": _float_field(request.form, "sightseeing_cost", 150),
            }
            priority = request.form.get("priority", "cost")

            state = {**inputs, "priority": priority, "budget": budget}
            state.pop("travel_date", None)
            outcome = _recovery_app.invoke(state)
            outcome["assumed_hotel_nights"] = ASSUMED_RECOVERY_HOTEL_NIGHTS
            recommendation = outcome
            result = {"probability": risk["probability"], "label": risk["label"], "inputs": inputs}
            db.save_flight_check(session["user_id"], inputs, risk["probability"], risk["label"], outcome)

    return render_template("flight_delay.html", result=result, recommendation=recommendation,
                            **_dropdown_options())


# ---------------------------------------------------------------- budget trip planner

@app.route("/budget-trip", methods=["GET", "POST"])
@login_required
def budget_trip():
    plan = None
    form_data = {}
    transport_options = None

    if request.method == "POST":
        form_data = request.form.to_dict()
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
            "total_budget": _float_field(request.form, "total_budget", 1000),
            "days": days,
            "start_date": start_date_str,
            "return_date": return_date.strftime("%Y-%m-%d"),
            "stay_comfort": request.form.get("stay_comfort", "standard"),
            "travel_comfort": request.form.get("travel_comfort", "standard"),
            "food_comfort": request.form.get("food_comfort", "standard"),
            "sightseeing_level": request.form.get("sightseeing_level", "moderate"),
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
        )
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
