"""
Static public reference data (IATA carrier/airport codes -> human names).
This is NOT derived from the project's private datasets - it's the same
kind of lookup table any flight-booking UI ships with, used only to make
codes readable in dropdowns. Only carriers that actually appear in
outputs/cleaned_flight_weather_unified.csv are offered in the dropdown; the
airport map covers common US hubs and falls back to the raw code for the
rest (there is no airport-name dataset in this project to draw from).
"""

CARRIER_NAMES = {
    "AA": "American Airlines",
    "AS": "Alaska Airlines",
    "B6": "JetBlue Airways",
    "DL": "Delta Air Lines",
    "F9": "Frontier Airlines",
    "G4": "Allegiant Air",
    "HA": "Hawaiian Airlines",
    "NK": "Spirit Airlines",
    "UA": "United Airlines",
    "WN": "Southwest Airlines",
    "CO": "Continental Airlines",
    "US": "US Airways",
    "VX": "Virgin America",
    "OO": "SkyWest Airlines",
    "MQ": "Envoy Air",
    "EV": "ExpressJet",
    "YX": "Republic Airways",
    "9E": "Endeavor Air",
    # International (India / Singapore / Malaysia) - reference-only routes, see intl_reference.py
    "AI": "Air India",
    "6E": "IndiGo",
    "UK": "Vistara (merged into Air India, Nov 2024)",
    "SG": "SpiceJet",
    "QP": "Akasa Air",
    "G8": "GoAir (now Go First)",
    "SQ": "Singapore Airlines",
    "TR": "Scoot",
    "MH": "Malaysia Airlines",
    "AK": "AirAsia",
    "EK": "Emirates",
}

AIRPORT_CITY = {
    "ATL": "Atlanta", "LAX": "Los Angeles", "ORD": "Chicago O'Hare", "DFW": "Dallas/Fort Worth",
    "DEN": "Denver", "JFK": "New York JFK", "LGA": "New York LaGuardia", "EWR": "Newark",
    "SFO": "San Francisco", "SEA": "Seattle", "LAS": "Las Vegas", "MCO": "Orlando",
    "CLT": "Charlotte", "PHX": "Phoenix", "MIA": "Miami", "IAH": "Houston Intercontinental",
    "BOS": "Boston", "MSP": "Minneapolis-St Paul", "DTW": "Detroit", "PHL": "Philadelphia",
    "BWI": "Baltimore", "SLC": "Salt Lake City", "IAD": "Washington Dulles", "DCA": "Washington Reagan",
    "HNL": "Honolulu", "PDX": "Portland", "STL": "St. Louis", "TPA": "Tampa",
    "SAN": "San Diego", "BNA": "Nashville", "AUS": "Austin", "MDW": "Chicago Midway",
    "FLL": "Fort Lauderdale", "HOU": "Houston Hobby", "OAK": "Oakland", "SJC": "San Jose",
    "SMF": "Sacramento", "RDU": "Raleigh-Durham", "CLE": "Cleveland", "PIT": "Pittsburgh",
    "CVG": "Cincinnati", "IND": "Indianapolis", "CMH": "Columbus", "MCI": "Kansas City",
    "SAT": "San Antonio", "MSY": "New Orleans", "ANC": "Anchorage", "OGG": "Maui",
    "RSW": "Fort Myers", "PBI": "West Palm Beach", "JAX": "Jacksonville", "OMA": "Omaha",
    # International (reference-only routes, not in the trained dataset - see intl_reference.py)
    "DEL": "Delhi", "BOM": "Mumbai", "BLR": "Bengaluru", "MAA": "Chennai", "CCU": "Kolkata",
    "HYD": "Hyderabad", "COK": "Kochi", "GOI": "Goa",
    # Tamil Nadu, Karnataka, Andhra Pradesh - reference-only, see intl_reference.py
    "CJB": "Coimbatore", "IXM": "Madurai", "TRZ": "Tiruchirapalli",
    "IXE": "Mangalore", "HBX": "Hubli",
    "VGA": "Vijayawada", "VTZ": "Visakhapatnam", "TIR": "Tirupati",
    "SIN": "Singapore",
    "KUL": "Kuala Lumpur", "PEN": "Penang", "LGK": "Langkawi", "JHB": "Johor Bahru", "BKI": "Kota Kinabalu",
    "DXB": "Dubai",
}


def carrier_label(code: str) -> str:
    code = (code or "").upper()
    name = CARRIER_NAMES.get(code)
    return f"{name} ({code})" if name else code


def airport_label(code: str) -> str:
    code = (code or "").upper()
    city = AIRPORT_CITY.get(code)
    return f"{city} ({code})" if city else code
