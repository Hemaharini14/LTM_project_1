"""
Replace the catalogue's invented flights with real published schedules.

data/india_catalog.json shipped with 202 generated flights, and says so in its
own _note: "Schedules and fares are illustrative, not a live airline feed -
swap india_adapter's source for a real API to replace them." This is that
swap, for the schedule half of it.

AviationStack's flightsFuture returns the real timetable for a future date:
operating flight numbers, scheduled departure and arrival, aircraft type.

COST, because it is most of the design. A page is 100 rows and Delhi files
about 1,500 departures a day, so one call sees a fourteenth of the airport -
and pages run in departure order, so the first one is all night flights.
SAMPLES_PER_AIRPORT spreads offsets across the day instead, which means four
calls per airport: forty for all ten cities, against a ~100/month allowance.
The plan also enforces a short rolling rate window, so a full build spends
about twenty minutes mostly waiting. Run it deliberately, not on a schedule.
Results cache for a week - a published timetable does not move hourly.

FARES ARE NOT REPLACED, because nothing here knows them. AviationStack sells
schedules, not prices; no free source gives real fares. The existing estimates
are carried across and marked fare_source="estimated" so a real schedule can
never be mistaken for a real price. Every flight this writes carries
schedule_source="aviationstack" for the same reason - a reader can tell which
half is observed.

Codeshares are collapsed. One 06:05 Chennai-Delhi sold under four designators
is one aeroplane, and leaving all four in would let the planner offer the same
seat four times over, which is the duplicate-flight bug that already had to be
fixed once in recovery_tools.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import date, datetime, timedelta

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from aviationstack import future_schedule, usage  # noqa: E402

# Pages are 100 rows, ordered by departure time, and Delhi alone files ~1,500
# a day. Reading pages 1..n therefore returns a night's flights, not a day's -
# so offsets are spread evenly across the total instead. Each sample is one
# call against a ~100/month allowance, which is why the default is small.
SAMPLES_PER_AIRPORT = 4

CATALOG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "india_catalog.json")

# Median one-way economy fares by block time, used only to carry a plausible
# number where the old catalogue had one. Explicitly an estimate.
def _estimate_fare_inr(duration_min: int) -> int:
    return int(round((2200 + 18 * max(duration_min, 45)) / 100.0) * 100)


# The carriers that actually operate Indian domestic services. flightsFuture,
# unlike the flights endpoint, carries no `codeshared` block, so there is no
# field saying which designator is the metal - and a Delhi-Mumbai slot lists
# Lufthansa, Singapore Airlines and Japan Airlines alongside the operator.
# Matching the route's own country is the available signal.
INDIAN_OPERATORS = {"AI", "6E", "UK", "SG", "IX", "QP", "G8", "9I", "I5", "S5"}

# Freight and mail. Real departures, but not seats - offering one in a trip
# plan would be worse than offering a generated flight, because it looks real.
CARGO_CARRIERS = {"D0", "BZ", "FX", "5X", "CK", "RU", "QY", "MB", "K4", "3S"}


def _pick_operator(candidates: list[tuple]) -> tuple:
    """Of the designators sharing one departure slot, the one to publish.

    Prefers an Indian operator, accepts any other passenger carrier, and
    returns (None, None) when the slot is cargo only.
    """
    passenger = [(f, d) for f, d in candidates
                 if (f.get("carrier_iata") or "").upper() not in CARGO_CARRIERS]
    if not passenger:
        return (None, None)
    for f, d in passenger:
        if (f.get("carrier_iata") or "").upper() in INDIAN_OPERATORS:
            return (f, d)
    return passenger[0]


def _load() -> dict:
    with open(CATALOG_PATH, encoding="utf-8") as fh:
        return json.load(fh)


def _hhmm(stamp: str | None) -> str | None:
    if not stamp:
        return None
    try:
        return datetime.fromisoformat(stamp.replace(" ", "T")).strftime("%H:%M")
    except Exception:
        return stamp[11:16] if len(stamp) >= 16 else None


def _duration(dep: str | None, arr: str | None) -> int | None:
    try:
        d = datetime.fromisoformat((dep or "").replace(" ", "T"))
        a = datetime.fromisoformat((arr or "").replace(" ", "T"))
        mins = (a - d).total_seconds() / 60.0
        if mins < 0:                       # crossed midnight
            mins += 24 * 60
        return int(mins) if 30 <= mins <= 900 else None
    except Exception:
        return None


def build(on: date | None = None, cities: dict | None = None) -> dict:
    """Fetch real schedules for every catalogue city and rewrite the flight list."""
    cat = _load()
    cities = cities or cat["cities"]
    iata_to_city = {v: k for k, v in cities.items()}
    on = on or (date.today() + timedelta(days=7))

    real: list[dict] = []
    seen: set[tuple] = set()
    per_airport: dict[str, int] = {}

    for city, iata in cities.items():
        first, total = future_schedule(iata, on)
        rows = list(first)
        if total > len(first) and SAMPLES_PER_AIRPORT > 1:
            step = max(1, total // SAMPLES_PER_AIRPORT)
            for n in range(1, SAMPLES_PER_AIRPORT):
                more, _ = future_schedule(iata, on, offset=min(n * step, max(total - 100, 0)))
                rows.extend(more)
        # Group first, choose second. Every designator selling a departure shares
        # its times, so taking the first row seen picks whoever the feed happened
        # to list - which put DHL and Japan Airlines on Delhi-Mumbai domestic.
        slots: dict[tuple, list] = {}
        for f in rows:
            dest = (f.get("destination") or "").upper()
            # Only routes between catalogue cities - the planner cannot price
            # or stay anywhere else.
            if dest not in iata_to_city or dest == iata:
                continue
            dep_hhmm = _hhmm(f.get("scheduled_departure"))
            arr_hhmm = _hhmm(f.get("scheduled_arrival"))
            dur = _duration(f.get("scheduled_departure"), f.get("scheduled_arrival"))
            if not (dep_hhmm and arr_hhmm and dur):
                continue
            slots.setdefault((iata, dest, dep_hhmm, arr_hhmm), []).append((f, dur))

        kept = 0
        for sig, candidates in slots.items():
            if sig in seen:
                continue
            f, dur = _pick_operator(candidates)
            if f is None:            # cargo-only slot, nothing a traveller can book
                continue
            seen.add(sig)
            dest = sig[1]
            dep_hhmm, arr_hhmm = sig[2], sig[3]
            carrier = (f.get("carrier_iata") or "").upper()
            number = str(f.get("flight_number") or "").strip()
            real.append({
                "number": f"{carrier} {number}".strip(),
                "airline": (f.get("carrier_name") or "").title() or carrier,
                "carrier_code": carrier,
                "from_city": city,
                "to_city": iata_to_city[dest],
                "from": iata,
                "to": dest,
                "depart": dep_hhmm,
                "arrive": arr_hhmm,
                "duration_min": dur,
                "fare_inr": _estimate_fare_inr(dur),
                "aircraft": f.get("aircraft_model"),
                "schedule_source": "aviationstack",
                "fare_source": "estimated",
            })
            kept += 1
        per_airport[iata] = kept

    return {"flights": real, "per_airport": per_airport, "for_date": on.isoformat(),
            "sampled_per_airport": SAMPLES_PER_AIRPORT, "usage": usage()}


def write(result: dict) -> dict:
    """Merge real flights into the catalogue, keeping generated ones only where
    a route has no real coverage at all."""
    cat = _load()
    real = result["flights"]
    real_routes = {(f["from"], f["to"]) for f in real}
    kept_generated = [f for f in cat["flights"]
                      if (f["from"], f["to"]) not in real_routes]
    for f in kept_generated:
        f.setdefault("schedule_source", "generated")
        f.setdefault("fare_source", "estimated")

    cat["flights"] = real + kept_generated
    cat["_note"] = (
        f"Flight schedules with schedule_source='aviationstack' are real published "
        f"timetables fetched for {result['for_date']} - flight numbers, times and "
        f"aircraft types are genuine. Routes with no live coverage keep the original "
        f"generated entries, marked schedule_source='generated'. FARES ARE ESTIMATES "
        f"IN BOTH CASES (fare_source='estimated') - no free source supplies real "
        f"prices. Hotel tiers and attraction costs remain approximate."
    )
    with open(CATALOG_PATH, "w", encoding="utf-8") as fh:
        json.dump(cat, fh, indent=2, ensure_ascii=False)
    return {"real": len(real), "generated_kept": len(kept_generated),
            "total": len(cat["flights"])}


if __name__ == "__main__":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

    only = sys.argv[1:] or None
    cat = _load()
    cities = ({c: i for c, i in cat["cities"].items() if i in only} if only else None)
    if only:
        print(f"limited to {only}: {len(cities)} cities x {SAMPLES_PER_AIRPORT} samples "
              f"= {len(cities) * SAMPLES_PER_AIRPORT} API calls")

    res = build(cities=cities)
    print(f"\nreal flights found for {res['for_date']}:")
    for iata, n in sorted(res["per_airport"].items(), key=lambda kv: -kv[1]):
        print(f"   {iata}  {n:4} catalogue-route departures")
    print(f"\n   {len(res['flights'])} total after collapsing codeshares")
    if res["flights"]:
        f = res["flights"][0]
        print(f"   e.g. {f['number']:9s} {f['from']}->{f['to']} {f['depart']}-{f['arrive']} "
              f"({f['duration_min']} min, {f['aircraft']})")
    print(f"   usage: {res['usage']}")

    if res["flights"] and "--write" in sys.argv:
        print("\n", write(res))
    elif res["flights"]:
        print("\n   (dry run - pass --write to update data/india_catalog.json)")
