"""
Record what actually happened, so the model can be checked against 2026.

Everything the delay model knows came from a historical US dataset. Whether it
is still right - on Indian routes, on this year's operations - has never been
testable here, because nothing in the project observed a real outcome. The
calibration figures quoted so far are held-out numbers from that same training
data, which says how well the model fits its own past, not how it does now.

AviationStack reports departures that have already left with both the schedule
and the actual time, so `delay` is a measured outcome rather than a forecast.
One call returns up to 100 of them. On a ~100-call monthly allowance, a single
daily call is affordable and yields a few thousand real outcomes a month.

TWO THINGS MAKE THIS SAMPLE UNSAFE TO TREAT AS THE TRUTH ABOUT DELAYS, and
both are handled rather than hidden:

  codeshares   one aircraft appears several times under different designators.
               MS9579, VS8901 and BA8252 were the same 13:35 to Patna. Counted
               naively, one late departure becomes three, and whichever
               airlines codeshare most look worst. Rows are collapsed onto the
               operating flight.

  selection    a page of 100 is not a random sample of the 1,805 available, and
               the first page observed carried no on-time flight at all - every
               row had a delay of at least a minute. Whether that is ordering
               or a reporting bias is not knowable from here, so the raw rate
               in this table is NOT the delay rate, and `is_representative` is
               stored as 0 to stop a later reader assuming otherwise.

So this builds evidence, not ground truth. It is enough to answer "does the
model rank these flights correctly" (AUC survives a biased sample as long as
both classes appear); it is NOT enough to refit the calibrator, which needs
the base rate to be real. Anyone tempted should read check_model.py first.
"""
from __future__ import annotations

import os
import sqlite3
import sys
from datetime import date, datetime, timezone

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from aviationstack import _get, usage  # noqa: E402
from db import DB_PATH  # noqa: E402

DELAY_THRESHOLD_MIN = 15      # the same cutoff the model was trained against


def _ensure_table() -> None:
    with sqlite3.connect(DB_PATH, timeout=15) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS flight_outcomes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                collected_at TEXT NOT NULL,
                flight_date TEXT NOT NULL,
                carrier_iata TEXT,
                flight_number TEXT,
                flight_iata TEXT,
                origin TEXT,
                destination TEXT,
                scheduled_departure TEXT,
                scheduled_arrival TEXT,
                actual_departure TEXT,
                departure_delay_min INTEGER,
                arrival_delay_min INTEGER,
                was_delayed INTEGER,
                icao24 TEXT,
                is_representative INTEGER DEFAULT 0,
                UNIQUE(flight_date, flight_iata, scheduled_departure)
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_outcomes_route "
                     "ON flight_outcomes(origin, destination)")


def _operating_row(rows: list[dict]) -> dict:
    """Of several codeshare records for one departure, the operating one.

    A codeshared record carries a `codeshared` block pointing at the real
    operator; the record without one is the metal.
    """
    for f in rows:
        if not (f.get("flight") or {}).get("codeshared"):
            return f
    return rows[0]


def collect(origin_iata: str = "DEL", limit: int = 100) -> dict:
    """Fetch and store today's completed departures from one airport."""
    _ensure_table()
    body = _get("flights", {"dep_iata": (origin_iata or "").upper(),
                            "flight_status": "landed", "limit": limit})
    if not body:
        return {"stored": 0, "reason": "no data (no key, quota spent, or API down)"}

    rows = body.get("data") or []

    # Group codeshares: same origin, same scheduled minute, same destination is
    # one physical departure however many designators are sold against it.
    groups: dict[tuple, list] = {}
    for f in rows:
        dep, arr = f.get("departure") or {}, f.get("arrival") or {}
        groups.setdefault((dep.get("scheduled"), arr.get("iata")), []).append(f)

    stored = skipped = 0
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with sqlite3.connect(DB_PATH, timeout=15) as conn:
        for rows_for_departure in groups.values():
            f = _operating_row(rows_for_departure)
            dep, arr = f.get("departure") or {}, f.get("arrival") or {}
            fl, al = f.get("flight") or {}, f.get("airline") or {}
            delay = dep.get("delay")
            if delay is None:
                skipped += 1
                continue
            try:
                conn.execute("""
                    INSERT OR IGNORE INTO flight_outcomes
                    (collected_at, flight_date, carrier_iata, flight_number, flight_iata,
                     origin, destination, scheduled_departure, scheduled_arrival,
                     actual_departure, departure_delay_min, arrival_delay_min,
                     was_delayed, icao24, is_representative)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,0)
                """, (
                    now, f.get("flight_date") or str(date.today()),
                    (al.get("iata") or "").upper() or None,
                    fl.get("number"), (fl.get("iata") or "").upper() or None,
                    (dep.get("iata") or "").upper() or None,
                    (arr.get("iata") or "").upper() or None,
                    dep.get("scheduled"), arr.get("scheduled"), dep.get("actual"),
                    int(delay), arr.get("delay"),
                    1 if int(delay) >= DELAY_THRESHOLD_MIN else 0,
                    ((f.get("aircraft") or {}).get("icao24") or "").lower() or None,
                ))
                stored += conn.total_changes and 1 or 0
            except Exception as e:
                print(f"[collect_outcomes] row skipped: {e}")
                skipped += 1

        total = conn.execute("SELECT COUNT(*) FROM flight_outcomes").fetchone()[0]
        late = conn.execute("SELECT COUNT(*) FROM flight_outcomes WHERE was_delayed=1").fetchone()[0]

    return {
        "origin": origin_iata.upper(),
        "returned": len(rows),
        "distinct_departures": len(groups),
        "skipped_no_delay_figure": skipped,
        "table_total": total,
        "table_delayed": late,
        # Stated, not computed into anything: see the module docstring.
        "raw_delayed_share": round(late / total, 3) if total else None,
        "usage": usage(),
    }


if __name__ == "__main__":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    airport = sys.argv[1] if len(sys.argv) > 1 else "DEL"
    result = collect(airport)
    print(f"\ncollected from {result.get('origin', airport)}")
    for k, v in result.items():
        if k != "origin":
            print(f"   {k:26s} {v}")
    print("\nNote: raw_delayed_share is NOT the delay rate - the sample is one "
          "page of many and\n      carries a selection bias. It is stored with "
          "is_representative=0 for that reason.")
