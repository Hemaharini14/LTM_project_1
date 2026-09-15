"""
Hand-curated public reference data for India/Singapore/Malaysia routes and
famous food destinations - added because the trained delay-prediction
model only knows 2019 US domestic flights and has zero signal for these
countries. This is the same kind of well-known, verifiable public
reference data as reference_data.py's airport-city map (real airports,
real airlines that fly these routes, real famous food streets/hawker
centres) - NOT model output, and NOT presented as if it were. Typical
durations/departure windows are indicative public schedule knowledge, not
live or historical per-flight data, and there is deliberately no delay-risk
score attached to any of it.
"""

# Airports added beyond the trained dataset - used to extend the dropdown
# and to tell "international, reference-only" apart from "no data at all".
INTL_AIRPORT_CODES = {"DEL", "BOM", "BLR", "MAA", "CCU", "HYD", "COK", "GOI",
                       "SIN", "KUL", "PEN", "LGK", "JHB", "BKI"}

# Only these resolve against the live weather dataset (outputs/cleaned_global_weather.csv)
# under this exact spelling - used to suggest a working destination_city value.
WEATHER_CITY_HINT = {"DEL": "New Delhi", "SIN": "Singapore", "KUL": "Kuala Lumpur"}

# (origin, destination) IATA codes -> typical carriers/duration/time-of-day.
# Indicative only - no fare, no real-time schedule, no delay model behind it.
INTL_ROUTES = {
    ("DEL", "SIN"): [{"carrier": "AI", "typical_duration_min": 330, "typical_departure": "Morning"},
                      {"carrier": "SQ", "typical_duration_min": 330, "typical_departure": "Night"}],
    ("SIN", "DEL"): [{"carrier": "AI", "typical_duration_min": 300, "typical_departure": "Afternoon"},
                      {"carrier": "SQ", "typical_duration_min": 300, "typical_departure": "Morning"}],
    ("DEL", "KUL"): [{"carrier": "AI", "typical_duration_min": 330, "typical_departure": "Afternoon"},
                      {"carrier": "MH", "typical_duration_min": 330, "typical_departure": "Night"}],
    ("KUL", "DEL"): [{"carrier": "MH", "typical_duration_min": 340, "typical_departure": "Morning"},
                      {"carrier": "AI", "typical_duration_min": 340, "typical_departure": "Afternoon"}],
    ("BOM", "SIN"): [{"carrier": "AI", "typical_duration_min": 330, "typical_departure": "Night"},
                      {"carrier": "SQ", "typical_duration_min": 330, "typical_departure": "Afternoon"}],
    ("SIN", "BOM"): [{"carrier": "SQ", "typical_duration_min": 320, "typical_departure": "Morning"},
                      {"carrier": "AI", "typical_duration_min": 320, "typical_departure": "Night"}],
    ("MAA", "SIN"): [{"carrier": "SQ", "typical_duration_min": 240, "typical_departure": "Morning"},
                      {"carrier": "AI", "typical_duration_min": 240, "typical_departure": "Evening"}],
    ("SIN", "MAA"): [{"carrier": "SQ", "typical_duration_min": 240, "typical_departure": "Afternoon"}],
    ("BLR", "SIN"): [{"carrier": "AI", "typical_duration_min": 255, "typical_departure": "Morning"},
                      {"carrier": "SQ", "typical_duration_min": 255, "typical_departure": "Evening"}],
    ("SIN", "BLR"): [{"carrier": "SQ", "typical_duration_min": 250, "typical_departure": "Afternoon"}],
    ("CCU", "SIN"): [{"carrier": "AI", "typical_duration_min": 260, "typical_departure": "Morning"}],
    ("SIN", "CCU"): [{"carrier": "AI", "typical_duration_min": 260, "typical_departure": "Afternoon"}],
    ("SIN", "KUL"): [{"carrier": "SQ", "typical_duration_min": 95, "typical_departure": "Morning"},
                      {"carrier": "AK", "typical_duration_min": 95, "typical_departure": "Afternoon"},
                      {"carrier": "MH", "typical_duration_min": 95, "typical_departure": "Evening"}],
    ("KUL", "SIN"): [{"carrier": "MH", "typical_duration_min": 95, "typical_departure": "Morning"},
                      {"carrier": "AK", "typical_duration_min": 95, "typical_departure": "Night"}],
    ("KUL", "PEN"): [{"carrier": "AK", "typical_duration_min": 60, "typical_departure": "Morning"},
                      {"carrier": "MH", "typical_duration_min": 60, "typical_departure": "Evening"}],
    ("PEN", "KUL"): [{"carrier": "MH", "typical_duration_min": 60, "typical_departure": "Afternoon"}],
    ("DEL", "BOM"): [{"carrier": "6E", "typical_duration_min": 130, "typical_departure": "Morning"},
                      {"carrier": "AI", "typical_duration_min": 135, "typical_departure": "Evening"},
                      {"carrier": "UK", "typical_duration_min": 130, "typical_departure": "Afternoon"}],
    ("BOM", "DEL"): [{"carrier": "6E", "typical_duration_min": 130, "typical_departure": "Afternoon"},
                      {"carrier": "AI", "typical_duration_min": 135, "typical_departure": "Morning"}],
}


def get_reference_flights(origin_airport: str, destination_airport: str) -> list[dict]:
    """Common flight-option shape (same keys search_alternative_flights returns) but with
    delay_probability=None and reference_only=True - so templates can reuse the same cards
    while clearly not presenting this as model output."""
    origin, destination = (origin_airport or "").upper(), (destination_airport or "").upper()
    entries = INTL_ROUTES.get((origin, destination), [])
    return [
        {
            "carrier": e["carrier"],
            "flight_number": "",
            "route": f"{origin}-{destination}",
            "departure_time": None,
            "arrival_time": None,
            "typical_departure": e.get("typical_departure"),
            "scheduled_elapsed_time": e["typical_duration_min"],
            "delay_probability": None,
            "risk_label": "Not modeled",
            "reference_only": True,
        }
        for e in entries
    ]


# Hand-curated, well-known, real food spots - not from any trained dataset.
FOOD_SPOTS = {
    "delhi": [
        {"name": "Chandni Chowk (Paranthe Wali Gali)", "area": "Old Delhi", "note": "Legendary stuffed-paratha lane"},
        {"name": "Karim's", "area": "Jama Masjid, Old Delhi", "note": "Historic Mughlai kebabs and curries"},
        {"name": "Khan Market food strip", "area": "Central Delhi", "note": "Mix of cafes and Indian eateries"},
    ],
    "new delhi": [],  # filled via alias below
    "mumbai": [
        {"name": "Mohammed Ali Road", "area": "South Mumbai", "note": "Famous street-food stretch, kebabs & malpua"},
        {"name": "Juhu Beach chaat stalls", "area": "Juhu", "note": "Classic pav bhaji, bhel puri by the beach"},
        {"name": "Bademiya", "area": "Colaba", "note": "Iconic late-night kebab rolls"},
    ],
    "bengaluru": [
        {"name": "VV Puram Food Street", "area": "Basavanagudi", "note": "South Indian street-food lane"},
        {"name": "MTR (Mavalli Tiffin Room)", "area": "Lalbagh Road", "note": "Historic South Indian breakfast institution"},
    ],
    "chennai": [
        {"name": "Murugan Idli Shop", "area": "Multiple outlets", "note": "Iconic soft idlis and filter coffee"},
        {"name": "Marina Beach food stalls", "area": "Marina Beach", "note": "Sundal, sugarcane juice, evening snacks"},
    ],
    "kolkata": [
        {"name": "Park Street", "area": "Central Kolkata", "note": "Historic dining strip"},
        {"name": "New Market food stalls", "area": "New Market", "note": "Kathi rolls and Mughlai snacks"},
    ],
    "singapore": [
        {"name": "Maxwell Food Centre", "area": "Chinatown", "note": "Famous hawker centre, Tian Tian chicken rice"},
        {"name": "Lau Pa Sat", "area": "Downtown Core", "note": "Historic Victorian hawker market, satay street at night"},
        {"name": "Newton Food Centre", "area": "Newton", "note": "Seafood and grilled skewers hawker centre"},
    ],
    "kuala lumpur": [
        {"name": "Jalan Alor", "area": "Bukit Bintang", "note": "KL's best-known street-food strip"},
        {"name": "Petaling Street", "area": "Chinatown KL", "note": "Hawker stalls and night market food"},
    ],
    "penang": [
        {"name": "Gurney Drive Hawker Centre", "area": "Gurney Drive", "note": "Famous char kway teow and assam laksa"},
        {"name": "Chulia Street", "area": "George Town", "note": "Night hawker stalls in the old town"},
    ],
}
FOOD_SPOTS["new delhi"] = FOOD_SPOTS["delhi"]

_FOOD_CITY_ALIASES = {
    "bombay": "mumbai", "bangalore": "bengaluru", "madras": "chennai", "calcutta": "kolkata",
}


def food_spots_for(city: str) -> list[dict]:
    if not city:
        return []
    key = _FOOD_CITY_ALIASES.get(city.strip().lower(), city.strip().lower())
    return FOOD_SPOTS.get(key, [])
