"""
Standalone demo of Module 3 (Budget Optimization). Run this on its own to
see the reallocation engine and the real-ADR budget validation working
without needing the full recovery graph.

Run: python test_budget_optimizer.py
"""
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from budget_optimizer import (
    reallocate_budget, summarize_budget,
    suggest_realistic_hotel_budget, validate_hotel_budget,
)

SAMPLE_BUDGET = {
    "flight_cost": 150, "hotel_cost": 400, "food_cost": 250,
    "transport_cost": 150, "sightseeing_cost": 150,
}

print("=" * 60)
print("1. REALLOCATION ACROSS ALL 3 PRIORITY MODES")
print("=" * 60)
for priority in ["cost", "time", "comfort"]:
    new_budget, shortfall, trim_log = reallocate_budget(SAMPLE_BUDGET, extra_cost=150, priority=priority)
    print(f"\nPriority: {priority}")
    print("New budget:", summarize_budget(new_budget))
    for line in trim_log:
        print(" -", line)

print("\n" + "=" * 60)
print("2. SHORTFALL EDGE CASE (extra cost too large to fully absorb)")
print("=" * 60)
new_budget, shortfall, trim_log = reallocate_budget(SAMPLE_BUDGET, extra_cost=900, priority="cost")
print("New budget:", summarize_budget(new_budget))
print("Shortfall:", shortfall)
for line in trim_log:
    print(" -", line)

print("\n" + "=" * 60)
print("3. REAL-ADR-GROUNDED HOTEL BUDGET SUGGESTION")
print("=" * 60)
for priority in ["cost", "comfort"]:
    suggestion = suggest_realistic_hotel_budget(nights=4, priority=priority)
    print(f"\nPriority: {priority}")
    print(suggestion)

print("\n" + "=" * 60)
print("4. VALIDATING A USER-ENTERED HOTEL BUDGET AGAINST REAL DATA")
print("=" * 60)
for user_amount in [50, 320, 900]:
    result = validate_hotel_budget(user_hotel_cost=user_amount, nights=4, priority="cost")
    print(f"\nUser entered ${user_amount} for 4 nights:")
    print(f"  Verdict: {result['verdict']}")
    print(f"  Real median suggests: ${result['suggested_total_usd']}")
    print(f"  Ratio: {result['ratio_to_real_median']}")