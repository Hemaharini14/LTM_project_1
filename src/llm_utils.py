"""
Shared LLM access for the "final reasoning" step of both recommendation
flows (delay-recovery in recovery_graph.py, and the budget trip planner).
The LLM only ever narrates data that was already retrieved deterministically
from real datasets/tools - it never invents flights, hotels, or prices.

Configure via .env (see .env.example) - pick one provider:
    LLM_PROVIDER=openai       OPENAI_API_KEY=sk-...          OPENAI_MODEL=gpt-4o-mini (default)
    LLM_PROVIDER=anthropic    ANTHROPIC_API_KEY=sk-ant-...   ANTHROPIC_MODEL=claude-sonnet-5 (default)
    LLM_PROVIDER=grok         GROK_API_KEY=xai-...           GROK_MODEL=grok-4 (default)
    LLM_PROVIDER=groq         GROQ_API_KEY=gsk_...           GROQ_MODEL=llama-3.3-70b-versatile (default)
Grok (xAI, x.ai) and Groq (groq.com) are two different companies with similar
names - check your key's prefix if unsure ("xai-" vs "gsk_"). Both APIs are
OpenAI-compatible, so both are served via ChatOpenAI + a custom base_url
rather than a separate SDK - no extra package needed for either.

With no key configured, every caller falls back to a deterministic,
template-based summary instead of failing - the web app always works,
the LLM narration is a bonus layer on top of it.

On networks that TLS-intercept HTTPS (common on corporate networks - Windows
trusts the corporate root CA, but Python's bundled certifi list doesn't),
plain requests to api.openai.com fail with "certificate verify failed" /
"Connection error" even with a valid key. truststore.inject_into_ssl() makes
Python's ssl module use the OS certificate store (same trust as curl/the
browser) instead, which fixes that without touching any real certificates.
"""
import os
import json
from dotenv import load_dotenv

try:
    import truststore
    truststore.inject_into_ssl()
except ImportError:
    pass

load_dotenv()


def get_llm(timeout: int = 12, max_retries: int = 0):
    """timeout is per HTTP call. The default suits a single narration call; the recovery
    agent (recovery_graph.py) passes a longer one because a ReAct loop makes several
    sequential calls and a 12s cap made the whole plan fall back on one slow response."""
    provider = os.getenv("LLM_PROVIDER", "").lower()
    try:
        if provider == "openai" and os.getenv("OPENAI_API_KEY"):
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0.3,
                               timeout=timeout, max_retries=max_retries)
        if provider == "anthropic" and os.getenv("ANTHROPIC_API_KEY"):
            from langchain_anthropic import ChatAnthropic
            return ChatAnthropic(model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5"), temperature=0.3,
                                  timeout=timeout, max_retries=max_retries)
        if provider == "grok" and os.getenv("GROK_API_KEY"):
            # xAI's Grok API is OpenAI-compatible - reuse ChatOpenAI pointed at api.x.ai.
            # NOT the same service as Groq (groq.com) below - easy to mix up, different keys/endpoints.
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(model=os.getenv("GROK_MODEL", "grok-4"),
                               api_key=os.getenv("GROK_API_KEY"),
                               base_url=os.getenv("GROK_BASE_URL", "https://api.x.ai/v1"),
                               temperature=0.3, timeout=timeout, max_retries=max_retries)
        if provider == "groq" and os.getenv("GROQ_API_KEY"):
            # Groq (groq.com) - fast-inference host for open models (Llama, etc). Also
            # OpenAI-compatible. Keys look like "gsk_..." - that prefix means Groq, not xAI Grok.
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
                               api_key=os.getenv("GROQ_API_KEY"),
                               base_url=os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1"),
                               temperature=0.3, timeout=timeout, max_retries=max_retries)
    except Exception as e:
        print(f"[llm_utils] LLM init failed, using deterministic narrative: {e}")
    return None


def llm_status() -> dict:
    """For the UI: is an LLM actually configured right now?"""
    provider = os.getenv("LLM_PROVIDER", "").lower()
    key_present = bool(os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
                       or os.getenv("GROK_API_KEY") or os.getenv("GROQ_API_KEY"))
    return {"configured": get_llm() is not None, "provider": provider or None, "key_present": key_present}


def _narration_digest(inputs: dict, plan: dict) -> dict:
    """The few dozen numbers a 120-word summary can actually use.

    Keeps the top option from each list rather than every option, and the
    counts rather than the contents - "4 sightseeing spots" is all the summary
    needs to say, where the spots themselves are thousands of tokens.
    """
    def top(key, fields):
        rows = plan.get(key) or []
        return {f: rows[0].get(f) for f in fields if f in rows[0]} if rows else None

    return {
        "trip": {
            "from": inputs.get("origin_airport"), "to": inputs.get("destination_airport"),
            "city": inputs.get("destination_city"), "days": inputs.get("days"),
            "nights": plan.get("nights"), "budget_usd": inputs.get("total_budget"),
        },
        "outbound": top("outbound_flights",
                        ["carrier", "flight_number", "risk_label", "price_usd"]),
        "return": top("return_flights",
                      ["carrier", "flight_number", "risk_label", "price_usd"]),
        "hotel": top("hotel_options", ["hotel_type", "median_nightly_rate_usd"]),
        "budget_breakdown": plan.get("budget_breakdown"),
        "daily_food_budget_usd": plan.get("daily_food_budget_usd"),
        "daily_transport_budget_usd": plan.get("daily_transport_budget_usd"),
        "counts": {
            "sightseeing_well_known": len(plan.get("sightseeing_common") or []),
            "sightseeing_lesser_known": len(plan.get("sightseeing_hidden") or []),
            "days_planned": len(plan.get("day_plan") or []),
        },
        "within_budget": plan.get("optimizer_feasible"),
        "shortfall_usd": plan.get("optimizer_shortfall_usd"),
    }


def narrate_trip_plan(inputs: dict, plan: dict) -> tuple[str, str]:
    """Returns (narrative_text, mode) for the budget trip planner, mirroring
    recovery_graph.py's grounded-narration pattern (real data in, narration out)."""
    llm = get_llm()
    best_out = plan["outbound_flights"][0] if plan.get("outbound_flights") else None
    best_ret = plan["return_flights"][0] if plan.get("return_flights") else None
    best_hotel = plan["hotel_options"][0] if plan.get("hotel_options") else None

    def deterministic() -> str:
        parts = [f"Trip from {inputs['origin_airport']} to {inputs['destination_airport']}, "
                 f"{inputs['days']} day(s), budget ${inputs['total_budget']:.0f}."]
        if best_out:
            parts.append(f"Outbound: {best_out['carrier']} {best_out['flight_number']} "
                         f"({best_out['risk_label'].lower()} delay risk).")
        if best_ret:
            parts.append(f"Return: {best_ret['carrier']} {best_ret['flight_number']} "
                         f"({best_ret['risk_label'].lower()} delay risk).")
        if best_hotel:
            parts.append(f"Stay: {best_hotel['hotel_type']} tier, ~${best_hotel['median_nightly_rate_usd']}/night.")
        parts.append(f"Budget split across flight/hotel/food/transport/sightseeing totals "
                     f"${plan['budget_breakdown']['total']:.0f}.")
        return " ".join(parts)

    if llm is None:
        return deterministic(), "deterministic (no LLM key configured)"

    from langchain_core.messages import SystemMessage, HumanMessage
    system = ("You are a travel planning assistant. Summarize this trip plan and explain the "
              "reasoning behind the budget split in under 120 words, using ONLY the data given "
              "below. Do not invent flights, hotels, prices, or points of interest beyond what's given.")
    # Send a digest, not the plan. Serialised whole, a plan runs to several
    # thousand tokens - most of it sightseeing lists and per-day itineraries the
    # summary never mentions - and on an 8,000 tokens-per-minute tier a single
    # narration could exceed the entire minute's budget on its own and come back
    # 429. Everything the summary can actually talk about is here.
    human = json.dumps(_narration_digest(inputs, plan), default=str)
    try:
        result = llm.invoke([SystemMessage(content=system), HumanMessage(content=human)])
        return result.content, f"LLM ({os.getenv('LLM_PROVIDER')})"
    except Exception as e:
        # The plan itself is complete and real either way - narration is the only
        # thing lost. Showing an API error blob where the summary should be makes
        # a working plan look broken, so degrade to the deterministic wording and
        # say plainly why, keeping the detail in the log for whoever is debugging.
        print(f"[llm_utils] narration failed, using deterministic summary: {e}")
        reason = "rate limit" if "429" in str(e) or "rate_limit" in str(e) else "unavailable"
        return deterministic(), f"deterministic (AI narration {reason})"


def suggest_destination_content(destination_city: str, sightseeing_count: int = 4) -> dict | None:
    """
    Fallback for destinations with no curated entry in intl_reference.py's
    FOOD_SPOTS or no real Geoapify sightseeing data (sightseeing.py). Asks the
    configured LLM for well-known real food spots, sightseeing highlights, AND
    named hotels people search for in the area. This is genuinely different
    from every other piece of data in this app: it is NOT grounded in a
    dataset, and an LLM can be wrong about real places. Returns None if no LLM
    is configured or the call/parse fails - callers must fall back to the
    generic template in that case, and the UI must always label this output
    "AI-suggested, unverified" rather than presenting it as checked data.
    Deliberately does NOT ask for prices, photos, or reviews for the hotels -
    those would need to be fabricated (an LLM can't know a real current rate,
    has no real photo, and has no real guest's opinion), so we only ask for
    names/areas to search by.

    sightseeing_count: raise this for a multi-day trip so the day-by-day plan
    (trip_planner.py) doesn't run out of distinct real spots and start
    repeating after 2 days - the model is told to still omit rather than pad
    with guesses if it doesn't actually know that many.
    """
    llm = get_llm()
    if llm is None or not destination_city:
        return None

    from langchain_core.messages import SystemMessage, HumanMessage
    system = (
        "You suggest well-known, real, verifiable food spots, sightseeing highlights, and named "
        "hotels for a travel destination. Respond with ONLY compact JSON, no prose, no markdown "
        'fences, in exactly this shape: {"food_spots": [{"name": str, "area": str, "note": str}, '
        '...], "sightseeing": [{"name": str, "area": str, "note": str}, ...], "hotels": '
        '[{"name": str, "area": str, "note": str}, ...]}. Give at most 4 food/hotel items each, and '
        f"up to {sightseeing_count} sightseeing items ONLY IF you genuinely know that many distinct "
        "real ones - fewer genuine items is always better than padding the list. Real hotel "
        "brand/property names only (no price, no rating - you don't have reliable current data for "
        "those). If you are not confident something is real, omit it rather than guessing."
    )
    try:
        result = llm.invoke([SystemMessage(content=system), HumanMessage(content=destination_city)])
        text = result.content.strip()
        if text.startswith("```"):
            text = text.strip("`").split("\n", 1)[-1]
        data = json.loads(text)
        if not isinstance(data, dict) or "food_spots" not in data or "sightseeing" not in data:
            return None
        data.setdefault("hotels", [])
        return data
    except Exception as e:
        print(f"[llm_utils] suggest_destination_content failed, using generic template: {e}")
        return None
