# SmartRouteAI

Flight delay prediction and budget trip planning, over real flight, weather and
schedule data.

Two products share one Flask app and one trained model:

- **Delay check** — score a specific flight, see which conditions are driving
  the risk, check the live aircraft, and plan around a delay if there is one.
- **Trip planner** — pick flights and a hotel against a budget, with real
  sightseeing, real transport comparison and a day-by-day itinerary.

Everything is priced in Indian rupees by default and the app is built for
Indian routes, though the delay model was trained on a US dataset (see
[Honest limits](#honest-limits)).

## Running it

```bash
flight/Scripts/activate          # the venv this project uses (Windows)
python api.py                    # http://127.0.0.1:5000
```

Or in a container (`Dockerfile`, production entry point is `gunicorn`, not the
Flask dev server above):

```bash
docker build -t smartrouteai .
docker run -p 5000:5000 --env-file .env smartrouteai
```

Model artifacts, datasets and the database are gitignored and NOT baked into
the image — mount them as a volume, or run the training scripts once inside
the running container. `GET /health` reports whether the delay model and
database actually loaded, for a process manager or orchestrator to probe.

Flask serves everything on one port — the API, the product pages and both
frontends. There is no separate frontend server.

| URL | What it is |
|---|---|
| `/` | Cinematic 3D showcase (React + Three.js, built from `web/`) |
| `/flight-delay` | Delay checker — risk, driving conditions, live aircraft, corridor map |
| `/flight-delay/alternatives` | Recovery plan — alternative flights with booking links |
| `/budget-trip` | Full planner — transport, hotels, sightseeing, itinerary |
| `/plan` | Vanilla-JS trip planner UI over `POST /api/plan-trip` |
| `/admin` | Model accuracy, calibration, predicted-vs-actual samples |
| `/health` | Liveness/readiness check — model + database status, no login |

> **Cold start is now a few seconds, not ~90.** The flight catalogue (5.4M
> rows) used to be loaded whole into memory, costing both that ~90s and
> ~2-3GB of standing RAM for the life of the process - more than most free
> hosting tiers have at all. It's now an indexed SQLite file
> (`src/build_flight_catalog_db.py`) queried per request instead, so nothing
> is loaded until a query actually needs it. Transport tables still aren't
> warmed at startup.

---

## What the agent and the LLM actually do

This is the part most worth reading, because the honest answer is narrower than
"it's an AI app" suggests — and that narrowness is deliberate.

### The rule

**The LLM never invents a fact that reaches the user as truth.** It chooses
which tools to call and narrates what those tools really returned. Every
number on every page — delay risk, flight times, hotel rates, distances — comes
from a trained model, a dataset or a real API, never from the model's memory.

There is exactly one place the LLM is asked to *supply* facts
(`suggest_destination_content`, for sightseeing in cities the POI APIs cover
badly), and every name it produces is confirmed against a geocoder before it is
shown. Anything unconfirmed is dropped. See
[`src/verify_places.py`](src/verify_places.py).

### Where an LLM genuinely runs

| Where | Kind | What it decides |
|---|---|---|
| [`src/recovery_graph.py`](src/recovery_graph.py) | **ReAct agent**, 5 tools | Whether you need a different flight, a hotel, or just a budget shuffle — and in which order to find out |
| [`src/chat_agent.py`](src/chat_agent.py) | **ReAct agent**, 13 tools | Which analysis answers your question ("is Tuesday better?", "which carrier is safest on this route?") |
| [`src/llm_utils.py`](src/llm_utils.py) | Single call | Narrating a finished trip plan in prose |
| [`src/llm_utils.py`](src/llm_utils.py) | Single call | Proposing well-known sightseeing spots — **then verified** |

Both agents are LangGraph `create_react_agent` loops. The tools return real
values *and* stash them in a closure, so the structured result is built from
what the tools actually returned rather than from the agent's account of what
it did. If the agent hallucinates in its narration, the cards beside it still
show the real data.

**The recovery agent's five tools:** search alternative flights (scored by the
trained model), search hotel tiers, check destination weather, reallocate the
budget, and build a booking link for the flight it recommends.

### Where an LLM deliberately does *not* run

| Component | Why not |
|---|---|
| [`src/trip_graph.py`](src/trip_graph.py) | The `/api/plan-trip` pipeline is a **deterministic** LangGraph state machine. Zero LLM calls. Same input, same plan, every time. |
| [`src/predict_delay_v2.py`](src/predict_delay_v2.py) | A trained PyTorch classifier. An LLM cannot estimate a probability. |
| [`src/explain_delay.py`](src/explain_delay.py) | Per-flight explanations come from counterfactual ablation against the model itself, not from asking a model to speculate. |
| [`src/booking.py`](src/booking.py) | A URL is a fixed grammar. One wrong character 404s or silently drops the passenger count. |
| [`src/itinerary_optimizer.py`](src/itinerary_optimizer.py) | Real combinatorial search over flights × hotels against a real budget constraint. |

**A LangGraph state machine is not an agent.** `trip_graph.py` uses the same
library as the recovery agent but makes no model calls at all — it is a
flowchart with a conditional edge. Worth distinguishing, because "built on
LangGraph" is often read as "agentic" when it is not.

### Without an API key

Every LLM path degrades rather than fails. The recovery agent falls back to
`_deterministic_recovery()` — the same tools, called in a fixed order, with
template narration. The chatbot reports that it is not configured. Trip
narration falls back to a generated summary. **The app works fully with no LLM
key at all**; the LLM is a reasoning layer on top of working software, not the
thing holding it up.

---

## The delay model

A PyTorch binary classifier (`DelayNetV2`) predicting whether a flight departs
**15+ minutes late**, trained on US BTS "Reporting Carrier On-Time Performance"
flights (2025) with real weather at **both** the departure and arrival airport,
joined from Open-Meteo at the scheduled time of each — see
[`src/build_bts_dataset.py`](src/build_bts_dataset.py). This replaced an earlier
version trained on 2019 US flights with origin weather only; that version and
its numbers are kept in `models/artifacts_pre_bts/` for comparison.

Raw scores are not probabilities — training used `pos_weight` to handle class
imbalance, which inflates them — so an **isotonic regression calibrator** is
fitted on held-out data and applied at serving. Without it, a raw 0.42 meant an
actual delay rate near 13%.

Risk labels: **High ≥ 0.30**, **Moderate ≥ 0.15**, otherwise Low.

### How well it works

There are **two models**, and which one scores a flight depends on the route.

### DelayNetV2 — US and international routes

The held-out number is **AUC 0.873**, but it splits sharply by whether the
inbound aircraft's delay is known, and serving almost never knows it:

| Held-out subgroup | AUC | n | Actual delay rate |
|---|---|---|---|
| `prior_leg_known` | 0.897 | 772,472 | 24.2% |
| `prior_leg_unknown` — **what serving sends** | **0.717** | 261,046 | 11.6% |

Held-out data still rewards a model for fitting its own training period, so
the number that actually matters is the **out-of-time test**: trained on 2025
flights only, scored against **4,014,657 real BTS flights from Jan–Jul 2026**
that didn't exist yet when training happened —
[`src/compare_delay_models.py`](src/compare_delay_models.py).

| Model | AUC, all flights | AUC, checked in advance |
|---|---|---|
| Previous model (2019 data, origin weather only, as it used to serve) | 0.773 | 0.656 |
| Previous model, given arrival weather too | 0.833 | 0.686 |
| **Current model (BTS 2025, both ends)** | **0.861** | **0.701** |

Most of that gain is arrival weather itself (0.773 → 0.833); training on more
recent data adds the rest (0.833 → 0.861). The earlier version of this project
measured destination weather on 139 real but unrepresentative outcomes and
found it hurt (0.713 → 0.603), so it was deliberately withheld from serving.
Re-measured on 4 million real out-of-time flights, that conclusion doesn't
hold — withholding it was a real cost, not a safeguard — so both the web form
and the recovery agent now fetch a forecast for the destination airport at the
scheduled arrival time (never the actual one, which would leak the outcome)
and pass it through. The calibrated scale tracks reality closely on this test:
flights predicted 30–40% risk were actually late 34.2% of the time; flights
predicted 65%+ were actually late 92.9% of the time.

### The India model — why it had to exist

DelayNetV2 saw 10,634 Indian rows out of 6.89M (**0.15%**), across four airports,
with Chennai absent entirely. Worse, the features it weights most are
*constants* on every Indian row:

```
origin_temp_f            1 distinct value   (placeholder)
prev_leg_arrival_delay   1 distinct value   (constant)
prev_leg_known           1 distinct value   (always 0)
```

On real Indian departures it scored **AUC 0.594** — and a logistic regression
on nothing but the origin airport scored **0.701**. The data was predictable;
the model was not learning it. [`src/train_india_model.py`](src/train_india_model.py)
trains on what Indian flights actually have:

| Model | AUC on real 2026 departures |
|---|---|
| DelayNetV2 | 0.594 |
| India model, historical rows only | 0.657 |
| **India model, historical + collected outcomes** | **0.723** |

Cross-validated with the 2026 rows held out fold by fold. Indian routes are
routed to it automatically; everything else stays on DelayNetV2, which is
better at what it was trained on.

### What it still gets wrong

It **under-predicts**, and most at the low end:

```
lowest predicted third    predicted 11%   actually late 19%
middle third              predicted 30%   actually late 32%
highest third             predicted 53%   actually late 75%
```

So read the ranking, not the level. A 4% is meaningfully safer than a 30%, but
it is not a 4-in-100 promise. The absolute scale is unsettled because the two
training sources disagree about the base rate — historical Indian rows say
27.1% of departures are 15+ minutes late, the 2026 sample says 42.1% — and one
day at three airports cannot decide which is right. Re-run the trainer as
outcomes accumulate.

One thing that makes a correct number feel wrong: IndiGo's **median** real
departure is 9 minutes late. The model reports the chance of crossing **15**
minutes, so "likely to be late" and "a flight you would call on time" are both
true at once.

### Two different datasets, two different jobs

The model's weights come from the BTS 2025 training set above. The **live
catalogue of real historical flights** that recovery/alternatives/chatbot
search over — real carriers, flight numbers, scheduled times — is a separate
file, still the original 2019 US + India dataset
(`outputs/cleaned_flight_weather_unified.csv`, converted to an indexed
SQLite file — `outputs/flight_catalog.db`, `build_flight_catalog_db.py` —
queried per request instead of loaded whole into memory; see "Cold start"
above). Each of those rows also carries its own real origin
*and* destination weather, which is now passed to the model too
([`src/recovery_tools.py`](src/recovery_tools.py)) — it was already in the
file, just not used. Moving that catalogue itself to BTS 2025 is a larger,
separate change not yet made.

---

## Where the data comes from

| Source | Key needed | Supplies |
|---|---|---|
| Trained dataset | — | Historical flights, weather, hotel ADR tiers |
| [Open-Meteo](https://open-meteo.com) | **no** | Forecast weather, 16 days out, fills blank form fields |
| [OpenSky Network](https://opensky-network.org) | yes | Live aircraft positions, inbound-leg turnaround |
| [AviationStack](https://aviationstack.com) | yes | Real schedules, pre-departure delay, airframe id |
| [Geoapify](https://geoapify.com) | yes | Sightseeing POIs, routing, city autocomplete, place confirmation |
| [Foursquare](https://foursquare.com/developers) | yes | Real hotel names and addresses |
| Wikipedia geosearch | **no** | Second witness when confirming a place exists |
| [Frankfurter](https://frankfurter.app) | **no** | Live USD→INR rates |

**AviationStack's free plan allows ~100 calls per *month*.** That is the
binding design constraint, not a footnote: every request goes through a disk
cache and a hard monthly cap, and the app degrades to schedule-and-weather
rather than failing when the allowance is gone.

### What is real and what is estimated

The project labels this rather than blurring it:

- **Flight schedules** — real for routes built by
  [`src/build_catalog.py`](src/build_catalog.py) (`schedule_source:
  "aviationstack"`), generated elsewhere (`"generated"`).
- **Fares are always estimates** (`fare_source: "estimated"`). No free source
  supplies real prices. A real flight number beside an invented fare is exactly
  what gets believed, so the data says so.
- **Sightseeing** — real POIs from Geoapify, or LLM-proposed names that a
  geocoder confirmed. Unconfirmed names are dropped.
- **Delay risk** — always the trained model. `risk_is_modelled: false` marks
  routes outside the trained coverage, where the number is indicative rather
  than evidence.

---

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
  "flight": { "number": "6E 319", "airline": "IndiGo", "depart": "11:05",
              "price_total": 8400, "disruption_risk": 0.2988,
              "was_rerouted": false, "risk_is_modelled": false },
  "hotel": { "name": "Delhi City Centre Hotel", "nights": 3, "price_total": 7200 },
  "total": 19200, "budget": 25000, "under_budget": true
}
```

`status` is `READY`, `OVER_BUDGET` (cheapest possible plan returned anyway) or
`NO_OPTIONS`. Invalid input returns **422** with a `details` list.

The pipeline ([`src/trip_graph.py`](src/trip_graph.py)) — deterministic, no LLM:

```
select_flight → check_disruption ─(risk > 0.30)→ recovery ─┐
                       │                                    ├→ allocate_budget → assemble
                       └────────────────────────────────────┘
```

Budget trimming runs in increasing order of pain: drop paid attractions, then
reduce nights (floor 1), then downgrade the hotel tier. Every adjustment is
reported in `adjustments`.

---

## Project layout

```
api.py                  Flask app — all routes, both products
src/                    Model, agents, tools, data clients (45 modules)
  predict_delay_v2.py     the trained classifier + calibration, and the
                          router that sends Indian routes to the model below
  train_india_model.py    trains the India model (DelayNetV2 is 0.20% Indian)
  india_delay_model.py    serves it, and decides which routes it covers
  explain_delay.py        per-flight "why", by counterfactual ablation
  recovery_graph.py       ReAct recovery agent (5 tools)
  chat_agent.py           ReAct chatbot (13 tools)
  trip_graph.py           deterministic /api/plan-trip pipeline
  booking.py              deep links to ixigo / MakeMyTrip / Goibibo
  aviationstack.py        schedules + pre-departure delay (quota-guarded)
  opensky.py              live aircraft positions and rotation
  weather_live.py         Open-Meteo forecast, in training units
  verify_places.py        confirm an LLM-proposed place really exists
  collect_outcomes.py     record real outcomes for validation
  check_model.py          score the model against those outcomes
adapters/               catalogue → model feature space
templates/              Jinja2 product pages
static/                 CSS, JS, built 3D showcase
web/                    React + Three.js showcase source
frontend/               vanilla-JS planner
tests/                  pytest contract tests
```

## Tests

```bash
python -m pytest tests/ -q          # 12 tests
```

Covers the `/api/plan-trip` contract (ready-and-within-budget, tight-budget
trimming, impossible budget, high-risk reroute, unknown route) plus input
validation.

Several modules are also runnable directly as self-tests:

```bash
python src/weather_live.py      # forecast at six airports
python src/booking.py           # link building, including rejection cases
python src/verify_places.py     # real vs fabricated place names
python src/check_model.py       # model vs real collected outcomes
```

## Rebuilding generated assets

```bash
python src/build_bts_dataset.py             # download BTS + weather, build training set
python src/build_bts_dataset.py --test 2026 # out-of-time test set, for comparison only
python src/train_delay_model_v2.py          # trains DelayNetV2 (DELAY_ARTIFACT_DIR to avoid overwriting live)
python src/evaluate_delay_model_v2.py       # metrics + probability calibrator
python src/compare_delay_models.py          # compares all artifact dirs against the out-of-time test
python src/train_india_model.py             # India delay model (AUC 0.594 -> 0.723)
python src/build_delay_duration_lookup.py   # delay duration + cause lookup
python src/build_catalog.py DEL BOM --write # real schedules into the catalogue
python src/build_flight_catalog_db.py       # indexed SQLite copy for recovery_tools.py (run after the above)
cd web && npm install && npm run build      # 3D showcase → static/showcase/
```

Model artifacts and datasets are gitignored; these regenerate them. The app
degrades gracefully when they are absent.

> `build_catalog.py` costs ~4 AviationStack calls per airport and waits out a
> rolling rate window. A full ten-city build is ~40 of the ~100 monthly calls
> and about twenty minutes. Run it deliberately, not on a schedule.

## Configuration

Copy `.env.example` to `.env`. **Every key is optional** — each feature falls
back to something that works without one.

| Key | Enables |
|---|---|
| `LLM_PROVIDER` + matching key | Recovery agent, chatbot, narration, place suggestions |
| `GEOAPIFY_API_KEY` | Sightseeing, routing, autocomplete, place confirmation |
| `FOURSQUARE_API_KEY` | Real hotel names and addresses |
| `OPENSKY_CLIENT_ID` / `_SECRET` | Live aircraft check and corridor map |
| `AVIATIONSTACK_API_KEY` | Real schedules, pre-departure delay |
| `SECRET_KEY` | Flask session signing |
| `ADMIN_EMAILS` | Admin dashboard access |

`.env` is gitignored. Exception text from the API clients is redacted, because
httpx puts the failing URL — key included — into its error messages.

---

## Honest limits

Worth knowing before trusting a number:

- **Two models, and the Indian one is young.** DelayNetV2 covers US and
  international routes, trained on 6.89M BTS 2025 rows; Indian routes go to a
  model trained on 10,988. It ranks far better than DelayNetV2 did there
  (0.723 against 0.594)
  but under-predicts, especially at the low end. Read the ordering, not the
  level.
- **Fares are never real.** See above.
- **The live aircraft check is retrospective without AviationStack.** Receiver
  networks report aircraft that have already flown, so OpenSky alone cannot say
  which airframe will operate a future flight. AviationStack's schedule closes
  that gap for today's flights only.
- **Collected outcomes are one day at three airports.** They are now sampled
  across the day rather than from a single page — which moved the measured AUC
  from 0.713 to 0.594 and the delay rate from 56% to 42%, so the sampling
  mattered more than anything else measured here. Rows carry
  `is_representative`; older single-page rows are kept but not scored. The
  calibrator is still *not* refitted on any of it: 354 rows from one day is
  evidence, not a distribution.
- **Cold start is now a few seconds**, not the ~90s an earlier version took.
