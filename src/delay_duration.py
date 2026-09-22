"""
Answers "how long, and why" for a flight the model has flagged as risky.

predict_delay_v2 answers only "how likely" - it's a binary classifier, so a
22% score says nothing about whether that delay would be 20 minutes or four
hours, which is exactly the difference between waiting at the gate and losing
a day of the trip. This reads the precomputed lookup built by
build_delay_duration_lookup.py from ~1M real delayed flights.

What it returns is a real conditional statistic, NOT a model prediction: "among
real flights like this one that were actually delayed, the median delay was N
minutes and the minutes were mostly attributed to X". It does not condition on
the weather the traveller typed in, the way the probability does. Callers must
present it that way - "when this flight is delayed it's usually ~43 min", not
"your flight will be 43 minutes late".

Returns None if the lookup hasn't been built, so callers degrade to showing
probability alone rather than crashing.
"""
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import OUTPUT_DIR

LOOKUP_PATH = os.path.join(OUTPUT_DIR, "delay_duration_lookup.json")
_LOOKUP = None
_LOAD_ATTEMPTED = False

# How specific the matched group was, for display - a traveller should be able to
# tell "this exact airline on this exact route" from "every flight in the dataset".
_BASIS_LABEL = {
    "carrier_route": "this airline on this route",
    "route": "this route, all airlines",
    "airport_hour": "this airport at this time of day",
    "overall": "all delayed flights in the dataset",
}


def _load() -> dict:
    global _LOOKUP, _LOAD_ATTEMPTED
    if not _LOAD_ATTEMPTED:
        _LOAD_ATTEMPTED = True
        try:
            with open(LOOKUP_PATH) as f:
                _LOOKUP = json.load(f)
        except FileNotFoundError:
            print("[delay_duration] no lookup found - run build_delay_duration_lookup.py "
                  "to enable duration and cause. Showing probability only for now.")
            _LOOKUP = None
    return _LOOKUP


def estimate_delay_duration(carrier_code: str, origin_airport: str,
                             destination_airport: str, scheduled_hour: int = 12) -> dict | None:
    """Real delay length and cause attribution for flights like this one, from the
    most specific group with enough real history to be worth reporting."""
    lookup = _load()
    if not lookup:
        return None

    carrier = (carrier_code or "").upper()
    route = f"{(origin_airport or '').upper()}-{(destination_airport or '').upper()}"
    candidates = [
        (f"{carrier}|{route}", "carrier_route"),
        (route, "route"),
        (f"{(origin_airport or '').upper()}|{int(scheduled_hour)}", "airport_hour"),
        ("__default__", "overall"),
    ]

    for key, basis in candidates:
        hit = lookup.get(key)
        if hit:
            causes = hit.get("causes", {})
            top = next(iter(causes), None)
            return {
                "median_min": hit["median_min"],
                "p90_min": hit["p90_min"],
                "sample_size": hit["sample_size"],
                "causes": causes,
                "top_cause": top,
                "top_cause_share": causes.get(top) if top else None,
                "basis": basis,
                "basis_label": _BASIS_LABEL[basis],
            }
    return None


if __name__ == "__main__":
    for args in [("AA", "ORD", "DEN", 19), ("WN", "LAX", "SFO", 7), ("XX", "ZZZ", "YYY", 3)]:
        d = estimate_delay_duration(*args)
        if d:
            print(f"{args[0]} {args[1]}->{args[2]} @{args[3]}h: median {d['median_min']}min, "
                  f"p90 {d['p90_min']}min, n={d['sample_size']:,}, "
                  f"mostly {d['top_cause']} ({d['top_cause_share']:.0%}) [{d['basis_label']}]")
