"""
The sidebar chatbot: a real LangGraph ReAct agent (not a scripted bot) that
can call the same grounded tools the rest of the app uses - it looks up
real flights, real hotel tiers, real destination weather, and checks
whether a route is even in the dataset before answering, rather than
guessing. Requires an LLM key (see llm_utils.py / .env.example); with none
configured it returns a plain explanation instead of crashing.
"""
import contextvars
import re
import time
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from llm_utils import get_llm, suggest_destination_content
from recovery_tools import (
    search_alternative_flights, search_hotel_options, get_destination_weather,
    is_route_covered, get_dataset_index, get_carriers_for_route,
)
from intl_reference import get_reference_flights, food_spots_for
from predict_delay_v2 import predict_delay_probability, risk_label
from reference_data import carrier_label
from sightseeing import get_real_sightseeing
from itinerary_optimizer import optimize_itinerary
import currency as currency_utils

# The traveller's display currency for this request. Tools take amounts in it and
# return amounts written in it; the data underneath stays in USD (src/currency.py).
_CUR = contextvars.ContextVar("chat_currency", default=currency_utils.DEFAULT_CURRENCY)


def _fmt(usd: float) -> str:
    return currency_utils.format_money(usd, _CUR.get())


def _usd(amount: float) -> float:
    return currency_utils.to_usd(amount, _CUR.get())

_DEFAULT_WEATHER = {"temp_f": 55, "precip_in": 0.0, "pressure": 29.9, "visibility": 10.0, "wind_speed": 8}

SYSTEM_PROMPT = (
    "You are the SmartRouteAI travel assistant, embedded as a sidebar chatbot. "
    "You help with two things: checking flight delay risk, and planning a trip within a budget. "
    "The underlying dataset is US domestic flights from 2019 plus real India domestic flights "
    "(BLR/BOM/CCU/DEL/HYD, 2019-2020) - no international/cross-border routes in the delay model. "
    "Real named hotels with live nightly rates come from find_hotels, which needs a city - ask "
    "for the city before suggesting hotels. Always call check_route_coverage before quoting flight data for a route, and "
    "tell the user plainly when a route/country isn't covered instead of guessing. Never invent "
    "a flight, price, hotel name, or attraction that a tool didn't return. Keep answers short and "
    "concrete. If check_route_coverage says a route isn't covered, still call find_reference_flights "
    "before telling the user there's no flight info at all - cross-border routes (India<->Singapore/"
    "Malaysia/UAE) have a curated reference schedule (carrier and typical timing only, explicitly "
    "NOT a delay-risk prediction) even though the trained model has no risk score for them. "
    "Distinguish these two cases clearly to the user. "
    "For open-ended questions like 'when should I fly', 'what's the best day', 'which airline is "
    "safest', or 'why is this risky' - don't just call check_flight_delay_risk once and stop. Use "
    "compare_departure_times / compare_weekdays / compare_carriers_on_route to actually investigate "
    "multiple real scenarios via the trained model and give a reasoned recommendation, and use "
    "explain_weather_impact to explain WHY a risk is elevated using the model's own real "
    "before/after numbers - never assert a reason (e.g. 'it's risky because of rain') that you "
    "didn't get from a tool call. "
    "For 'plan a trip within my budget' requests (origin, destination, a total budget, and trip "
    "length), use optimize_trip_budget or compare_trip_priorities - these jointly search real "
    "flight options and real hotel tiers against the actual budget and report a real feasible "
    "combination (or an honest shortfall), rather than find_hotels alone which only searches "
    "hotels in isolation. Use compare_trip_priorities specifically when the user's preferences "
    "are mixed or fuzzy (e.g. 'cheap but not a sketchy hotel') so you can explain the real "
    "trade-offs across cost/time/comfort before recommending one."
)


@tool
def check_route_coverage(origin_airport: str, destination_airport: str) -> str:
    """Check whether both airports (IATA codes) exist in the training dataset at all."""
    covered = is_route_covered(origin_airport, destination_airport)
    if covered:
        return f"{origin_airport.upper()} and {destination_airport.upper()} are both covered by the dataset."
    index = get_dataset_index()
    return (f"NOT COVERED: this dataset only contains 2019 US domestic airports plus real India "
            f"domestic airports ({len(index['airports'])} total). At least one of "
            f"{origin_airport}/{destination_airport} "
            f"is not in it - tell the user we have no flight data for this route/country.")


@tool
def check_flight_delay_risk(carrier_code: str, origin_airport: str, destination_airport: str,
                             month: int, weekday: int, scheduled_elapsed_time: float = 120,
                             scheduled_hour: int = 12, is_holiday: bool = False) -> str:
    """Predict delay risk for a flight using typical (clear-weather) conditions, since exact
    weather isn't available in chat. carrier_code/origin_airport/destination_airport are IATA
    codes, month is 1-12, weekday is 0=Monday..6=Sunday, scheduled_hour is 0-23 (pass the real
    departure hour if the user mentioned a time - e.g. "evening flight" -> ~18 - this measurably
    affects the real model's prediction). Set is_holiday=True if the user's date is a known
    US/India public holiday."""
    if not is_route_covered(origin_airport, destination_airport):
        return "This route isn't covered by the dataset - no prediction can be made."
    prob = predict_delay_probability(
        carrier_code=carrier_code, origin_airport=origin_airport, destination_airport=destination_airport,
        weekday=weekday, month=month, scheduled_elapsed_time=scheduled_elapsed_time,
        origin_temp_f=_DEFAULT_WEATHER["temp_f"], origin_temp_known=False,
        origin_precip_in=_DEFAULT_WEATHER["precip_in"],
        origin_pressure=_DEFAULT_WEATHER["pressure"], origin_visibility=_DEFAULT_WEATHER["visibility"],
        origin_wind_speed=_DEFAULT_WEATHER["wind_speed"],
        scheduled_hour=scheduled_hour, is_holiday=is_holiday,
    )
    return f"Estimated delay probability {prob:.0%} ({risk_label(prob)} risk), assuming typical clear weather."


def _typical_risk(carrier_code, origin_airport, destination_airport, weekday, month,
                   scheduled_elapsed_time, scheduled_hour=12, is_holiday=False) -> float:
    """Shared helper for the comparison tools below - one real call to the trained
    model under typical (clear-weather) conditions, varying only what's asked."""
    return predict_delay_probability(
        carrier_code=carrier_code, origin_airport=origin_airport, destination_airport=destination_airport,
        weekday=weekday, month=month, scheduled_elapsed_time=scheduled_elapsed_time,
        origin_temp_f=_DEFAULT_WEATHER["temp_f"], origin_temp_known=False,
        origin_precip_in=_DEFAULT_WEATHER["precip_in"],
        origin_pressure=_DEFAULT_WEATHER["pressure"], origin_visibility=_DEFAULT_WEATHER["visibility"],
        origin_wind_speed=_DEFAULT_WEATHER["wind_speed"],
        scheduled_hour=scheduled_hour, is_holiday=is_holiday,
    )


@tool
def compare_departure_times(carrier_code: str, origin_airport: str, destination_airport: str,
                             month: int, weekday: int, scheduled_elapsed_time: float = 120) -> str:
    """Compare delay risk across different times of day for the SAME route/carrier/day, by
    calling the real trained model once per hour (typical weather each time) - use this when
    the user asks "when should I fly" or "is morning better than evening" instead of guessing.
    Returns each hour's real predicted risk, ranked best to worst."""
    if not is_route_covered(origin_airport, destination_airport):
        return "This route isn't covered by the dataset - no comparison can be made."
    hours = [3, 6, 9, 12, 15, 18, 21, 23]
    scored = sorted(
        ((h, _typical_risk(carrier_code, origin_airport, destination_airport, weekday, month,
                            scheduled_elapsed_time, scheduled_hour=h)) for h in hours),
        key=lambda x: x[1]
    )
    return "\n".join(f"{h:02d}:00 - {p:.0%} ({risk_label(p)} risk)" for h, p in scored)


@tool
def compare_weekdays(carrier_code: str, origin_airport: str, destination_airport: str,
                      month: int, scheduled_elapsed_time: float = 120, scheduled_hour: int = 12) -> str:
    """Compare delay risk across the 7 days of the week for the SAME route/carrier, by calling
    the real trained model once per weekday (typical weather each time) - use this when the user
    asks "what's the best day to fly". Returns each weekday's real predicted risk, ranked."""
    if not is_route_covered(origin_airport, destination_airport):
        return "This route isn't covered by the dataset - no comparison can be made."
    day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    scored = sorted(
        ((d, _typical_risk(carrier_code, origin_airport, destination_airport, d, month,
                            scheduled_elapsed_time, scheduled_hour=scheduled_hour)) for d in range(7)),
        key=lambda x: x[1]
    )
    return "\n".join(f"{day_names[d]} - {p:.0%} ({risk_label(p)} risk)" for d, p in scored)


@tool
def compare_carriers_on_route(origin_airport: str, destination_airport: str,
                               weekday: int, month: int, scheduled_elapsed_time: float = 120) -> str:
    """Compare delay risk across every carrier that REALLY flies this exact route, by calling
    the real trained model once per carrier (typical weather each time) - use this when the user
    asks "which airline is best/safest" for a route. Only compares carriers with real historical
    flights on this route, not every carrier in the whole dataset."""
    if not is_route_covered(origin_airport, destination_airport):
        return "This route isn't covered by the dataset - no comparison can be made."
    carriers = get_carriers_for_route(origin_airport, destination_airport)
    if not carriers:
        return "No carrier has real historical flights on this exact route pairing."
    scored = sorted(
        ((c, _typical_risk(c, origin_airport, destination_airport, weekday, month, scheduled_elapsed_time))
         for c in carriers),
        key=lambda x: x[1]
    )
    return "\n".join(f"{carrier_label(c)} - {p:.0%} ({risk_label(p)} risk)" for c, p in scored)


@tool
def explain_weather_impact(carrier_code: str, origin_airport: str, destination_airport: str,
                            weekday: int, month: int, scheduled_elapsed_time: float,
                            origin_temp_f: float, origin_precip_in: float, origin_pressure: float,
                            origin_visibility: float, origin_wind_speed: float) -> str:
    """Explains how much of a flight's delay risk comes from weather specifically, by calling the
    real trained model twice: once with the given weather values, once with a clear-weather
    baseline, and reporting the real difference. Use this after you have real or live weather
    numbers (e.g. from check_destination_weather, converted to °F/inHg/mph/miles) to explain WHY
    a risk is elevated, instead of asserting a reason the model didn't actually give you."""
    if not is_route_covered(origin_airport, destination_airport):
        return "This route isn't covered by the dataset - no explanation can be made."
    # Both calls use the SAME origin_temp_known so only the weather numbers vary - flipping that
    # flag between calls would confound the comparison, since it was 100% correlated with
    # US-vs-India in training and the model can pick up on it as more than just "is temp known".
    common = dict(carrier_code=carrier_code, origin_airport=origin_airport, destination_airport=destination_airport,
                  weekday=weekday, month=month, scheduled_elapsed_time=scheduled_elapsed_time, origin_temp_known=True)
    with_weather = predict_delay_probability(
        **common, origin_temp_f=origin_temp_f, origin_precip_in=origin_precip_in,
        origin_pressure=origin_pressure, origin_visibility=origin_visibility, origin_wind_speed=origin_wind_speed,
    )
    baseline = predict_delay_probability(
        **common, origin_temp_f=_DEFAULT_WEATHER["temp_f"], origin_precip_in=_DEFAULT_WEATHER["precip_in"],
        origin_pressure=_DEFAULT_WEATHER["pressure"], origin_visibility=_DEFAULT_WEATHER["visibility"],
        origin_wind_speed=_DEFAULT_WEATHER["wind_speed"],
    )
    delta = with_weather - baseline
    if delta == 0:
        impact = "This weather has no effect on risk."
    else:
        impact = f"This weather {'raises' if delta > 0 else 'lowers'} risk by {abs(delta):.0%} points."
    return (f"With the given weather: {with_weather:.0%} ({risk_label(with_weather)}). "
            f"Clear-weather baseline for this same flight: {baseline:.0%} ({risk_label(baseline)}). "
            f"{impact}")


@tool
def find_flights(origin_airport: str, destination_airport: str, weekday: int, priority: str = "cost") -> str:
    """Find real historical flight options for a route, scored by the trained delay model.
    priority is 'cost', 'time', or 'comfort'. If this returns "not covered", call
    find_reference_flights next before concluding there's nothing to show."""
    if not is_route_covered(origin_airport, destination_airport):
        return "NOT COVERED by the trained model - call find_reference_flights before answering."
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
def find_reference_flights(origin_airport: str, destination_airport: str) -> str:
    """For routes NOT covered by the trained model (e.g. India/Singapore/Malaysia): look up a
    curated reference schedule (real airlines, typical duration/time-of-day). This is NOT a
    delay-risk prediction - always say so. Returns empty if no reference data exists either."""
    options = get_reference_flights(origin_airport, destination_airport)
    if not options:
        return "No reference schedule either - genuinely no flight data at all for this route."
    return "\n".join(
        f"{o['carrier']} · {o['route']} · ~{o['scheduled_elapsed_time']}min · "
        f"typical departure: {o['typical_departure']} - REFERENCE SCHEDULE ONLY, not a delay-risk prediction"
        for o in options
    )


@tool
def find_hotels(nightly_budget: float, city: str = "", check_in: str = "", nights: int = 1) -> str:
    """Find REAL named hotels in a city at or under a nightly budget (per room, in the USER'S
    currency), cheapest first, from live Google Hotels rates. city is required for real
    hotels - ask the user for it if they haven't said. check_in is YYYY-MM-DD (defaults to
    7 days from today); nights defaults to 1."""
    from datetime import date, timedelta
    from serpapi_prices import real_hotel_prices
    if not (city or "").strip():
        return ("Real hotel rates need a city. Ask the user which city they want to stay in - "
                "don't suggest hotels without one.")
    try:
        start = date.fromisoformat(check_in) if check_in else date.today() + timedelta(days=7)
    except ValueError:
        start = date.today() + timedelta(days=7)
    end = start + timedelta(days=max(int(nights or 1), 1))
    cap_usd = _usd(nightly_budget)
    fitting = real_hotel_prices(city, start.isoformat(), end.isoformat(), adults=2, limit=6,
                                cheapest_first=True, max_price_usd=max(int(cap_usd), 1))
    fitting = [h for h in fitting if h["nightly_rate_usd"] <= cap_usd + 1e-9]
    if fitting:
        lines = [f"Real hotels in {city} at or under {_fmt(cap_usd)}/night (Google Hotels, "
                 f"{start.isoformat()} for {(end - start).days} night(s), 2 adults per room, cheapest first):"]
        lines += [f"- {h['name']}: {_fmt(h['nightly_rate_usd'])}/night"
                  + (f", rating {h['rating']}" if h.get("rating") else "") for h in fitting]
        return "\n".join(lines)
    # A looser price filter surfaces the cheap end of the market; an unfiltered
    # search only returns Google's most relevant page, which can miss it.
    cheapest = (real_hotel_prices(city, start.isoformat(), end.isoformat(), adults=2, limit=3,
                                  cheapest_first=True, max_price_usd=max(int(cap_usd * 5), 15))
                or real_hotel_prices(city, start.isoformat(), end.isoformat(), adults=2, limit=3,
                                     cheapest_first=True))
    if not cheapest:
        return f"No live hotel rates could be retrieved for {city} right now."
    return (f"NO hotel in {city} was found at or under {_fmt(cap_usd)}/night for {start.isoformat()}. "
            f"The cheapest real options are: "
            + "; ".join(f"{h['name']} {_fmt(h['nightly_rate_usd'])}/night" for h in cheapest)
            + ". Tell the user plainly that their budget is below these; do not call these within budget.")


def _format_optimizer_result(result: dict, priority: str) -> str:
    if not result["feasible"]:
        cheapest = result.get("cheapest_hotel")
        return (f"NOT FEASIBLE at priority={priority}: even the cheapest real hotel tier "
                f"({cheapest['hotel_type'] + ' ' + cheapest['room_type'] if cheapest else 'n/a'}) "
                f"leaves a real shortfall of {_fmt(result['shortfall_usd'])} against this budget.")
    b, hotel = result["budget_breakdown"], result["hotel_options"][0]
    lines = [f"priority={priority}: checked {result['combinations_considered']} real flight x hotel combinations."]
    lines.append(f"Hotel: {hotel['hotel_type']} room {hotel['room_type']}, ~{_fmt(hotel['median_nightly_rate_usd'])}/night, "
                 f"{hotel['reliability_pct']}% historical reliability - real total {_fmt(b['hotel_cost'])}.")
    for label, legs in (("Outbound", result["outbound_flights"]), ("Return", result["return_flights"])):
        if legs and legs[0].get("delay_probability") is not None:
            f = legs[0]
            lines.append(f"{label}: {carrier_label(f['carrier'])} - {f['delay_probability']:.0%} delay risk ({f['risk_label']}).")
        elif legs and legs[0].get("reference_only"):
            lines.append(f"{label}: reference schedule only ({carrier_label(legs[0]['carrier'])}), no real delay-risk data.")
        else:
            lines.append(f"{label}: no real or reference flight data for this leg.")
    lines.append(f"Budget split: flight {_fmt(b['flight_cost'])}, hotel {_fmt(b['hotel_cost'])}, food {_fmt(b['food_cost'])}, "
                 f"transport {_fmt(b['transport_cost'])}, sightseeing {_fmt(b['sightseeing_cost'])} (total {_fmt(b['total'])}).")
    return "\n".join(lines)


@tool
def optimize_trip_budget(origin_airport: str, destination_airport: str, total_budget: float,
                          days: int, start_weekday: int, return_weekday: int,
                          priority: str = "cost") -> str:
    """Run the real budget-constrained itinerary optimizer for ONE priority ('cost', 'time', or
    'comfort'): jointly searches real flight options and real hotel tiers against the traveler's
    actual total budget and returns the best-scoring real combination, or an honest shortfall if
    nothing fits. Use compare_trip_priorities instead if the user's preferences are mixed or
    unclear (e.g. wants both cheap AND comfortable)."""
    result = optimize_itinerary(origin_airport, destination_airport, _usd(total_budget), days,
                                 start_weekday, return_weekday, priority=priority)
    return _format_optimizer_result(result, priority)


@tool
def compare_trip_priorities(origin_airport: str, destination_airport: str, total_budget: float,
                             days: int, start_weekday: int, return_weekday: int) -> str:
    """Run the real itinerary optimizer for all three priorities (cost, time, comfort) on the
    SAME route/budget/dates and compare the real results - use this when a user describes mixed
    or fuzzy preferences (e.g. "cheap but not a sketchy hotel", "don't want to fly at night") so
    you can explain the real trade-offs between options before recommending one, instead of
    guessing which single priority best matches what they said."""
    sections = []
    for p in ("cost", "time", "comfort"):
        result = optimize_itinerary(origin_airport, destination_airport, _usd(total_budget), days,
                                     start_weekday, return_weekday, priority=p)
        sections.append(_format_optimizer_result(result, p))
    return "\n\n".join(sections)


@tool
def check_destination_weather(city: str) -> str:
    """Look up current live weather for a destination city, if available."""
    weather = get_destination_weather(city)
    if not weather:
        return f"No live weather data available for '{city}'."
    return f"{weather['condition']}, {weather['temperature_celsius']}°C, updated {weather['last_updated']}."


@tool
def find_food_and_sightseeing(city: str) -> str:
    """Find food spots and sightseeing highlights for a city. Tries curated (verified) food, then
    REAL sightseeing points-of-interest (Geoapify), then falls back to an LLM suggestion for
    whatever's still missing - always tell the user which kind it is (curated/real data vs.
    AI-suggested/unverified) using the label this tool returns."""
    lines = []
    curated_food = food_spots_for(city)
    if curated_food:
        lines += [f"CURATED, VERIFIED food spot: {s['name']} ({s['area']}) - {s['note']}" for s in curated_food]

    real_sights = get_real_sightseeing(city)
    if real_sights:
        lines += [f"REAL sightseeing (Geoapify): {s['name']} ({s['area']}) - {s['note']}" for s in real_sights]

    if not curated_food or not real_sights:
        ai_data = suggest_destination_content(city)
        if ai_data:
            if not curated_food:
                lines += [f"AI-SUGGESTED (unverified) food spot: {s['name']} ({s['area']}) - {s['note']}"
                          for s in ai_data.get("food_spots", [])]
            if not real_sights:
                lines += [f"AI-SUGGESTED (unverified) sightseeing: {s['name']} ({s['area']}) - {s['note']}"
                          for s in ai_data.get("sightseeing", [])]

    if not lines:
        return f"No curated data, no real POI data, and no AI suggestions available for '{city}'."
    return "\n".join(lines)


_TOOLS = [check_route_coverage, check_flight_delay_risk,
          compare_departure_times, compare_weekdays, compare_carriers_on_route, explain_weather_impact,
          find_flights, find_reference_flights,
          find_hotels, optimize_trip_budget, compare_trip_priorities,
          check_destination_weather, find_food_and_sightseeing]
_AGENT = None


def _get_agent():
    global _AGENT
    if _AGENT is not None:
        return _AGENT
    # Longer timeout and SDK-level retries: a free Groq key allows ~8,000 tokens a
    # MINUTE and one agent turn makes several calls of ~2,500 tokens each, so a 429
    # is routine. The OpenAI client honours the server's Retry-After on 429, which
    # turns "rate limit reached" into a few seconds' wait instead of an error.
    llm = get_llm(timeout=45, max_retries=3)
    if llm is None:
        return None
    from langgraph.prebuilt import create_react_agent
    _AGENT = create_react_agent(llm, _TOOLS)
    return _AGENT


def run_chat(history: list[dict], message: str, currency: str | None = None) -> str:
    """history: list of {"role": "user"|"assistant", "content": str}. Returns the reply text.
    currency: the traveller's display currency (e.g. "INR"); budgets are asked for and
    amounts quoted in it, and tool arguments are taken in it."""
    code = currency if currency in currency_utils.SUPPORTED_CURRENCIES else currency_utils.DEFAULT_CURRENCY
    _CUR.set(code)
    agent = _get_agent()
    if agent is None:
        return (
            "The chatbot needs an LLM key to reason with. Set LLM_PROVIDER and the matching API "
            "key (OpenAI, Anthropic, or Grok) in a .env file in the project root, then restart "
            "the app - see .env.example."
        )

    info = currency_utils.SUPPORTED_CURRENCIES[code]
    messages = [SystemMessage(content=SYSTEM_PROMPT + (
        f" The user's currency is {code} ({info['symbol']}, {info['name']}). Ask for budgets in {code}, "
        f"pass amounts to tools in {code} exactly as the user gave them (tools convert internally), and "
        f"quote every amount in {code} exactly as the tools write it. Never ask for, convert to, or quote "
        f"US dollars unless the user's currency is USD."))]
    # Every call in the agent loop resends this history, so it is the cheapest
    # place to save tokens: the last 6 turns, long replies clipped.
    for turn in history[-6:]:
        cls = HumanMessage if turn.get("role") == "user" else AIMessage
        text = turn.get("content", "") or ""
        messages.append(cls(content=text if len(text) <= 700 else text[:700] + " …"))
    messages.append(HumanMessage(content=message))

    for attempt in (1, 2):
        try:
            result = agent.invoke({"messages": messages}, config={"recursion_limit": 12})
            final = result["messages"][-1]
            return final.content
        except Exception as e:
            wait = _rate_limit_wait(e)
            if wait is not None and attempt == 1 and wait <= 30:
                time.sleep(wait + 1)
                continue
            print(f"[chat_agent] {type(e).__name__}: {e}")
            return _friendly_error(e, wait)
    return _friendly_error(None, None)


def _rate_limit_wait(e) -> float | None:
    """Seconds the provider asked us to wait, if this is a rate-limit error."""
    text = str(e)
    if "429" not in text and "rate limit" not in text.lower() and "rate_limit" not in text:
        return None
    m = re.search(r"try again in ([\d.]+)\s*(ms|s)", text)
    if not m:
        return 10.0
    v = float(m.group(1))
    return v / 1000 if m.group(2) == "ms" else v


def _friendly_error(e, wait) -> str:
    """What the user sees - never the raw provider payload (it carries org IDs)."""
    if wait is not None:
        return ("I'm getting a lot of questions right now and the AI service asked me to slow "
                "down (free-tier limit). Please wait about a minute and ask again.")
    text = str(e or "")
    if "401" in text or "invalid_api_key" in text or "Incorrect API key" in text:
        return "The chatbot's AI key was rejected. Check the API key in the .env file."
    if "timed out" in text.lower() or "timeout" in text.lower():
        return "The AI service took too long to answer. Please try again."
    return "Sorry, something went wrong while answering. Please try again in a moment."
