# SmartRouteAI

Flight delay prediction and budget trip planning, over real flight, weather and
hotel data.

## Running it

```bash
flight/Scripts/activate          # Windows venv used by this project
python api.py                    # http://127.0.0.1:5000
```

Flask serves everything on one port — the API, the product pages, and both
frontends. There is no separate frontend server to start.

| URL | What it is |
|---|---|
| `/` | Cinematic 3D showcase (React + Three.js, built from `web/`) |
| `/plan` | Vanilla-JS trip planner UI |
| `/flight-delay` | Delay checker — risk, expected duration, cause breakdown |
| `/budget-trip` | Full planner — transport comparison, hotels, sightseeing, itinerary |
| `/admin` | Model accuracy, calibration check, predicted-vs-actual samples |

## `POST /api/plan-trip`

```bash
curl -X POST http://127.0.0.1:5000/api/plan-trip \
  -H "Content-Type: application/json" \
  -d '{"from_city":"Chennai","to_city":"Delhi","date":"2026-10-12","travelers":2,"budget":25000}'
```

```json
{
  "status": "READY",
  "route_label": "Chennai → Delhi",
  "flight": { "number": "6E 319", "airline": "IndiGo", "depart": "11:05", "arrive": "14:22",
              "price_per_person": 4200, "price_total": 8400,
              "disruption_risk": 0.2988, "was_rerouted": false, "risk_is_modelled": false },
  "hotel": { "name": "Delhi City Centre Hotel", "nights": 3,
             "price_per_night": 2400, "price_total": 7200 },
  "attractions": [{ "name": "India Gate", "est_cost": 0 }],
  "total": 19200, "budget": 25000, "under_budget": true,
  "message": "IndiGo 6E 319 departs 11:05, 3 nights at Delhi City Centre Hotel, 5 attractions."
}
```

`status` is `READY`, `OVER_BUDGET` (cheapest possible plan returned anyway) or
`NO_OPTIONS`. Invalid input returns **422** with a `details` list — travellers
below 1, budget at or below 0, a malformed date, or origin equal to destination.

### Pipeline

A LangGraph state machine (`src/trip_graph.py`) sequencing the three modules:

```
select_flight → check_disruption ─(risk > 0.30)→ recovery ─┐
                       │                                    ├→ allocate_budget → assemble
                       └────────────────────────────────────┘
```

| Module | Code | Role |
|---|---|---|
| 1 — Disruption prediction | `src/predict_delay_v2.py` | Calibrated classifier, held-out **AUC 0.873** |
| 2 — Recovery planning | `src/recovery_graph.py` | LangGraph agent over real recovery tools |
| 3 — Budget optimization | `src/budget_optimizer.py` | Trim ordering + floors, ADR-grounded pricing |

Budget trimming runs in increasing order of pain: drop paid attractions, then
reduce nights (floor 1), then downgrade the hotel tier. Every adjustment is
reported back in `adjustments`.

### What is real, and what is not

The disruption risk is genuine — your trained model scoring the schedule, which
is why it varies by departure hour (early flights score Low, evening ones High,
because later flights inherit the day's accumulated lateness).

The **catalogue is not a live feed**. `data/india_catalog.json` holds generated
flight numbers, times and fares for 10 Indian cities, because no fare data
exists anywhere in this project. `adapters/india_adapter.py` is the single swap
point for a real flight API.

Two fields keep this legible rather than hidden:

- `risk_is_modelled` — `false` when the route is outside the trained data. The
  real India coverage is BLR/BOM/CCU/DEL/HYD; **Chennai is not in it**, so
  Chennai routes return a number that is indicative, not evidence.
- `basis` on the delay-duration lookup — says when a figure is the overall
  average rather than that route's own history.

## Tests

```bash
python -m pytest tests/ -q
```

Covers the five contract behaviours (ready-and-within-budget, tight-budget
trimming, impossible budget, high-risk reroute, unknown route) plus input
validation.

## Rebuilding generated assets

```bash
python src/build_delay_duration_lookup.py   # delay duration + cause lookup
python src/evaluate_delay_model_v2.py       # metrics + probability calibrator
cd web && npm install && npm run build      # 3D showcase → static/showcase/
```

Model artifacts and datasets are gitignored; these regenerate them. The app
degrades gracefully when they are absent rather than failing.

## Configuration

Copy `.env.example` to `.env`. All keys are optional — each feature falls back
to a working default without one.

| Key | Enables |
|---|---|
| `LLM_PROVIDER` + matching key | Recovery agent reasoning, chatbot, AI suggestions |
| `GEOAPIFY_API_KEY` | Sightseeing, routing, city autocomplete |
| `FOURSQUARE_API_KEY` | Real hotel names and addresses |
