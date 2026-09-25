"""
Score real 2026 departures with the trained model and see if it was right.

Until now "is the model correct" could only be answered from held-out rows of
the same historical US dataset it learned from. That measures fit to its own
past. collect_outcomes.py records what actually happened to real flights this
year, which makes a genuine out-of-sample test possible for the first time.

WHAT THIS CAN AND CANNOT CONCLUDE, because the two are easy to confuse:

  AUC - trustworthy here. It asks whether delayed flights score above on-time
        ones, and rank order is unaffected by how the sample was selected, as
        long as both classes are present. This is the headline number.

  calibration - NOT trustworthy here, and deliberately not acted on. Whether
        "30%" means three-in-ten depends on the base rate being real, and this
        sample's base rate is not: it is one page of many and arrived carrying
        almost no on-time flights. Reported for information, flagged, and never
        written back into the calibrator. Refitting on it would drag every
        prediction upward to match a delay rate that is an artefact of how the
        rows were chosen.

Conditions are also not the app's best case: these flights carry no weather
reading, so scoring uses the same defaults a user who leaves the weather blank
would get. That is a fair test of the serving path, and a pessimistic test of
the model - it is being asked to work from schedule, route and carrier alone.
"""
from __future__ import annotations

import os
import sqlite3
import sys
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from db import DB_PATH  # noqa: E402
from predict_delay_v2 import predict_delay_probability, risk_label  # noqa: E402

DEFAULT_WEATHER = {"temp_f": 70.0, "precip_in": 0.0, "pressure": 29.92,
                   "visibility": 10.0, "wind_speed": 8.0}


MIN_REPRESENTATIVE = 50


def _rows() -> list[dict]:
    """Outcomes to score, preferring the ones sampled across the whole day.

    Rows with is_representative=1 come from several pages spread over the day;
    the rest are a single page, which lands inside one part of it. Mixing them
    drags the base rate toward whichever slice happened to be read - the two
    groups here differ by 14 points (56.1% against 42.1%), so the biased rows
    are dropped as soon as there are enough good ones to stand on.
    """
    with sqlite3.connect(DB_PATH, timeout=15) as conn:
        conn.row_factory = sqlite3.Row
        good = [dict(r) for r in conn.execute(
            "SELECT * FROM flight_outcomes WHERE departure_delay_min IS NOT NULL "
            "AND is_representative = 1")]
        if len(good) >= MIN_REPRESENTATIVE:
            return good
        return [dict(r) for r in conn.execute(
            "SELECT * FROM flight_outcomes WHERE departure_delay_min IS NOT NULL")]


def _elapsed_minutes(row: dict) -> float:
    """Scheduled block time, from the two scheduled timestamps when both exist."""
    try:
        dep = datetime.fromisoformat(row["scheduled_departure"])
        arr = datetime.fromisoformat(row["scheduled_arrival"])
        mins = (arr - dep).total_seconds() / 60.0
        if 20 <= mins <= 1200:
            return mins
    except Exception:
        pass
    return 130.0


def _auc(scores: list[float], labels: list[int]) -> float | None:
    """Rank-based AUC (Mann-Whitney), no sklearn dependency.

    Ties get the average rank, which matters because many flights here share a
    score when they share a route and hour.
    """
    pos = sum(labels)
    neg = len(labels) - pos
    if pos == 0 or neg == 0:
        return None
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    rank_sum = sum(r for r, l in zip(ranks, labels) if l == 1)
    return (rank_sum - pos * (pos + 1) / 2.0) / (pos * neg)


def evaluate() -> dict:
    rows = _rows()
    if len(rows) < 10:
        return {"error": f"only {len(rows)} outcomes stored - run collect_outcomes.py first"}

    scores, labels, scored = [], [], []
    for r in rows:
        try:
            dep = datetime.fromisoformat(r["scheduled_departure"])
        except Exception:
            continue
        try:
            prob = predict_delay_probability(
                carrier_code=r["carrier_iata"] or "AI",
                origin_airport=r["origin"], destination_airport=r["destination"],
                weekday=dep.weekday(), month=dep.month,
                scheduled_elapsed_time=_elapsed_minutes(r),
                origin_temp_f=DEFAULT_WEATHER["temp_f"],
                origin_precip_in=DEFAULT_WEATHER["precip_in"],
                origin_pressure=DEFAULT_WEATHER["pressure"],
                origin_visibility=DEFAULT_WEATHER["visibility"],
                origin_wind_speed=DEFAULT_WEATHER["wind_speed"],
                origin_temp_known=False,
                scheduled_hour=dep.hour,
                is_holiday=False,
            )
        except Exception as e:
            print(f"[check_model] could not score {r['flight_iata']}: {e}")
            continue
        scores.append(float(prob))
        labels.append(int(r["was_delayed"]))
        scored.append((r, float(prob)))

    if not scores:
        return {"error": "nothing could be scored"}

    auc = _auc(scores, labels)
    actual_rate = sum(labels) / len(labels)
    mean_pred = sum(scores) / len(scores)

    # Does a higher score actually mean a higher chance of being late?
    ordered = sorted(scored, key=lambda t: t[1])
    third = max(1, len(ordered) // 3)
    bands = []
    for name, chunk in [("lowest third", ordered[:third]),
                        ("middle third", ordered[third:2 * third]),
                        ("highest third", ordered[2 * third:])]:
        if not chunk:
            continue
        late = sum(1 for r, _ in chunk if r["was_delayed"])
        bands.append({
            "band": name, "n": len(chunk),
            "mean_predicted": round(sum(p for _, p in chunk) / len(chunk), 3),
            "actually_late": round(late / len(chunk), 3),
        })

    # Per carrier, because "is it right?" is usually asked about one airline.
    # Its own predicted-vs-actual is the only thing that answers it: intuition
    # about an airline is about how late it USUALLY runs, which is a different
    # question from how often it crosses the 15-minute line.
    by_carrier = {}
    for row, prob in scored:
        by_carrier.setdefault(row["carrier_iata"] or "??", []).append(
            (prob, int(row["was_delayed"]), row["departure_delay_min"]))
    carriers = []
    for code, vals in sorted(by_carrier.items(), key=lambda kv: -len(kv[1])):
        if len(vals) < 3:
            continue
        late = sorted(v[2] for v in vals)
        carriers.append({
            "carrier": code, "n": len(vals),
            "mean_predicted": round(sum(v[0] for v in vals) / len(vals), 3),
            "actually_late": round(sum(v[1] for v in vals) / len(vals), 3),
            "median_minutes_late": late[len(late) // 2],
        })

    flagged = sum(1 for s in scores if risk_label(s) != "Low")
    return {
        "by_carrier": carriers,
        "flights_scored": len(scores),
        "auc": round(auc, 3) if auc is not None else None,
        "mean_predicted": round(mean_pred, 3),
        "actual_delayed_rate": round(actual_rate, 3),
        "flagged_not_low": flagged,
        "bands": bands,
    }


if __name__ == "__main__":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    res = evaluate()
    if "error" in res:
        print(res["error"])
        sys.exit(0)

    print(f"\nScored {res['flights_scored']} real departures collected from AviationStack.\n")
    print(f"  AUC (does it rank delays above on-time?)   {res['auc']}")
    print(f"  mean predicted probability                 {res['mean_predicted']}")
    print(f"  actual share delayed in this sample        {res['actual_delayed_rate']}")
    print(f"  flagged above Low risk                     {res['flagged_not_low']}/{res['flights_scored']}")
    if res.get("by_carrier"):
        print("\n  by carrier (predicted vs what really happened):")
        print(f"     {'':5s} {'n':>4s} {'predicted':>10s} {'actual':>8s} {'median late':>12s}")
        for c in res["by_carrier"]:
            print(f"     {c['carrier']:5s} {c['n']:>4} {c['mean_predicted']:>10.1%} "
                  f"{c['actually_late']:>8.1%} {c['median_minutes_late']:>9} min")
        print("     a carrier can sit just under the 15-minute line and still score")
        print("     a high probability of crossing it - those are different questions")

    print("\n  by predicted band:")
    for b in res["bands"]:
        print(f"     {b['band']:14s} n={b['n']:4}  predicted {b['mean_predicted']:.3f}"
              f"   actually late {b['actually_late']:.3f}")

    try:
        from india_delay_model import info as _india_info
        bundle = _india_info()
    except Exception:
        bundle = None
    if bundle and bundle.get("collected_rows"):
        print("\n  WARNING: the India model was trained on these same collected rows")
        print(f"  ({bundle['collected_rows']} of them), so the AUC above is in-sample and")
        print("  flatters it. The honest out-of-sample figure is the cross-validated")
        print(f"  {bundle.get('cv_auc')} from train_india_model.py, where the 2026 rows are")
        print("  held out fold by fold. Collect more outcomes and retrain to separate them.")

    print("\n  AUC is the number to trust. 0.5 is a coin flip; above it means the")
    print("  model puts delayed flights higher.")
    print("")
    print("  These rows are sampled across the whole day (is_representative=1),")
    print("  so the base rate here is a real estimate. It is still one day at a")
    print("  few airports - accumulate more before refitting anything on it.")
