"""
Priority-aware budget reallocation. When a disruption forces an extra cost,
this trims flexible categories first, respecting per-category floors so
nothing goes below a sane minimum.

Also includes a real-data check: suggest_realistic_hotel_budget() and
validate_hotel_budget() ground the hotel_cost line against actual ADR
(average daily rate) medians from cleaned_hotel_bookings.csv, so the
budget module isn't just abstract math on whatever number a user types in.
"""
import os
import sys
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import HOTELS_CLEAN_PATH

CATEGORIES = ["flight_cost", "hotel_cost", "food_cost", "transport_cost", "sightseeing_cost"]

MIN_FRACTION = {
    "flight_cost": 0.95, "hotel_cost": 0.90, "food_cost": 0.55,
    "transport_cost": 0.55, "sightseeing_cost": 0.35,
}

PRIORITY_TRIM_ORDER = {
    "cost": ["sightseeing_cost", "food_cost", "transport_cost", "hotel_cost", "flight_cost"],
    "time": ["sightseeing_cost", "transport_cost", "food_cost", "hotel_cost", "flight_cost"],
    "comfort": ["transport_cost", "sightseeing_cost", "food_cost", "hotel_cost", "flight_cost"],
}

# priority -> which real hotel tier to anchor against
PRIORITY_HOTEL_TIER = {"cost": "Resort Hotel", "time": "City Hotel", "comfort": "City Hotel"}

_HOTELS_DF = None


def _load_hotels():
    global _HOTELS_DF
    if _HOTELS_DF is None:
        _HOTELS_DF = pd.read_csv(HOTELS_CLEAN_PATH)
    return _HOTELS_DF


def reallocate_budget(current_budget: dict, extra_cost: float, priority: str = "cost"):
    order = PRIORITY_TRIM_ORDER.get(priority, PRIORITY_TRIM_ORDER["cost"])
    new_budget = dict(current_budget)
    remaining = extra_cost
    trim_log = []

    for cat in order:
        if remaining <= 0:
            break
        original = current_budget.get(cat, 0.0)
        floor = original * MIN_FRACTION[cat]
        available = max(0.0, new_budget[cat] - floor)
        take = min(available, remaining)
        if take > 0:
            new_budget[cat] -= take
            remaining -= take
            trim_log.append(f"Trimmed ${take:.2f} from {cat.replace('_cost','')}")

    shortfall = max(0.0, remaining)
    if shortfall > 0:
        trim_log.append(f"${shortfall:.2f} shortfall remains after trimming to category floors.")
    return new_budget, shortfall, trim_log


def summarize_budget(budget: dict) -> dict:
    total = sum(budget.get(c, 0.0) for c in CATEGORIES)
    return {**{c: round(budget.get(c, 0.0), 2) for c in CATEGORIES}, "total": round(total, 2)}


def suggest_realistic_hotel_budget(nights: int, priority: str = "cost") -> dict:
    """
    Uses real ADR medians from cleaned_hotel_bookings.csv to suggest a
    realistic total hotel cost for a stay of this length, anchored to the
    hotel tier that matches the traveler's priority.
    """
    df = _load_hotels()
    tier = PRIORITY_HOTEL_TIER.get(priority, "City Hotel")
    tier_rows = df[df["hotel"] == tier]
    median_adr = float(tier_rows["adr"].median())
    cancellation_rate = float(tier_rows["is_canceled"].mean())
    suggested_total = round(median_adr * nights, 2)
    return {
        "hotel_tier": tier,
        "median_nightly_rate_usd": round(median_adr, 2),
        "nights": nights,
        "suggested_total_usd": suggested_total,
        "historical_cancellation_rate": round(cancellation_rate, 3),
    }


def validate_hotel_budget(user_hotel_cost: float, nights: int, priority: str = "cost") -> dict:
    """
    Flags whether a user-entered hotel_cost is realistic compared to real
    historical pricing for a stay of this length - catches budgets that are
    unrealistically low before a disruption even happens.
    """
    suggestion = suggest_realistic_hotel_budget(nights, priority)
    suggested_total = suggestion["suggested_total_usd"]
    ratio = user_hotel_cost / suggested_total if suggested_total > 0 else None

    if ratio is None:
        verdict = "unknown"
    elif ratio < 0.5:
        verdict = "unrealistically low"
    elif ratio < 0.85:
        verdict = "tight but plausible"
    elif ratio <= 1.5:
        verdict = "reasonable"
    else:
        verdict = "generous"

    return {
        "user_hotel_cost": user_hotel_cost,
        "suggested_total_usd": suggested_total,
        "ratio_to_real_median": round(ratio, 2) if ratio is not None else None,
        "verdict": verdict,
        "based_on": suggestion,
    }