"""
Shared LLM access for the "final reasoning" step of both recommendation
flows (delay-recovery in recovery_graph.py, and the budget trip planner).
The LLM only ever narrates data that was already retrieved deterministically
from real datasets/tools - it never invents flights, hotels, or prices.

Configure via .env (see .env.example) - pick one provider:
    LLM_PROVIDER=openai       OPENAI_API_KEY=sk-...          OPENAI_MODEL=gpt-4o-mini (default)
    LLM_PROVIDER=anthropic    ANTHROPIC_API_KEY=sk-ant-...   ANTHROPIC_MODEL=claude-sonnet-5 (default)
    LLM_PROVIDER=grok         GROK_API_KEY=xai-...           GROK_MODEL=grok-4 (default)
Grok's API is OpenAI-compatible, so it's served via ChatOpenAI + a custom base_url
rather than a separate SDK - no extra package needed.

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


def get_llm():
    provider = os.getenv("LLM_PROVIDER", "").lower()
    try:
        if provider == "openai" and os.getenv("OPENAI_API_KEY"):
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0.3,
                               timeout=12, max_retries=0)
        if provider == "anthropic" and os.getenv("ANTHROPIC_API_KEY"):
            from langchain_anthropic import ChatAnthropic
            return ChatAnthropic(model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5"), temperature=0.3,
                                  timeout=12, max_retries=0)
        if provider == "grok" and os.getenv("GROK_API_KEY"):
            # xAI's Grok API is OpenAI-compatible - reuse ChatOpenAI pointed at api.x.ai.
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(model=os.getenv("GROK_MODEL", "grok-4"),
                               api_key=os.getenv("GROK_API_KEY"),
                               base_url=os.getenv("GROK_BASE_URL", "https://api.x.ai/v1"),
                               temperature=0.3, timeout=12, max_retries=0)
    except Exception as e:
        print(f"[llm_utils] LLM init failed, using deterministic narrative: {e}")
    return None


def llm_status() -> dict:
    """For the UI: is an LLM actually configured right now?"""
    provider = os.getenv("LLM_PROVIDER", "").lower()
    key_present = bool(os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY") or os.getenv("GROK_API_KEY"))
    return {"configured": get_llm() is not None, "provider": provider or None, "key_present": key_present}


def narrate_trip_plan(inputs: dict, plan: dict) -> tuple[str, str]:
    """Returns (narrative_text, mode) for the budget trip planner, mirroring
    recovery_graph.generate_recommendation_node's grounded-narration pattern."""
    llm = get_llm()
    best_out = plan["outbound_flights"][0] if plan.get("outbound_flights") else None
    best_ret = plan["return_flights"][0] if plan.get("return_flights") else None
    best_hotel = plan["hotel_options"][0] if plan.get("hotel_options") else None

    if llm is None:
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
        return " ".join(parts), "deterministic (no LLM key configured)"

    from langchain_core.messages import SystemMessage, HumanMessage
    system = ("You are a travel planning assistant. Summarize this trip plan and explain the "
              "reasoning behind the budget split in under 120 words, using ONLY the data given "
              "below. Do not invent flights, hotels, prices, or points of interest beyond what's given.")
    human = json.dumps({"trip_inputs": inputs, "plan": plan}, default=str)
    try:
        result = llm.invoke([SystemMessage(content=system), HumanMessage(content=human)])
        return result.content, f"LLM ({os.getenv('LLM_PROVIDER')})"
    except Exception as e:
        return f"[LLM error: {e}] Falling back to raw data below.", "error-fallback"


def suggest_food_and_sightseeing(destination_city: str) -> dict | None:
    """
    Fallback for destinations with no curated entry in intl_reference.py's
    FOOD_SPOTS. Asks the configured LLM for well-known real food spots and
    sightseeing highlights. This is genuinely different from every other
    piece of data in this app: it is NOT grounded in a dataset, and an LLM
    can be wrong about real places. Returns None if no LLM is configured or
    the call/parse fails - callers must fall back to the generic template
    in that case, and the UI must always label this output "AI-suggested,
    unverified" rather than presenting it as checked data.
    """
    llm = get_llm()
    if llm is None or not destination_city:
        return None

    from langchain_core.messages import SystemMessage, HumanMessage
    system = (
        "You suggest well-known, real, verifiable food spots and sightseeing highlights for a "
        "travel destination. Respond with ONLY compact JSON, no prose, no markdown fences, in "
        'exactly this shape: {"food_spots": [{"name": str, "area": str, "note": str}, ...], '
        '"sightseeing": [{"name": str, "area": str, "note": str}, ...]}. Give at most 4 items per '
        "list. If you are not confident this is a real place or you don't have reliable knowledge "
        'of it, return {"food_spots": [], "sightseeing": []} instead of guessing.'
    )
    try:
        result = llm.invoke([SystemMessage(content=system), HumanMessage(content=destination_city)])
        text = result.content.strip()
        if text.startswith("```"):
            text = text.strip("`").split("\n", 1)[-1]
        data = json.loads(text)
        if not isinstance(data, dict) or "food_spots" not in data or "sightseeing" not in data:
            return None
        return data
    except Exception as e:
        print(f"[llm_utils] suggest_food_and_sightseeing failed, using generic template: {e}")
        return None
