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


# Real published IATA airport coordinates (lat, lon) for every code in
# AIRPORT_CITY above - public reference data, same spirit as the city-name
# table, not derived from any dataset in this project. Rounded to ~1km
# precision, which is all "nearest supported airport to this city" or
# "airport -> destination driving route" (see maps.py) actually needs.
AIRPORT_COORDS = {
    "ATL": (33.6407, -84.4277), "LAX": (33.9416, -118.4085), "ORD": (41.9742, -87.9073),
    "DFW": (32.8998, -97.0403), "DEN": (39.8561, -104.6737), "JFK": (40.6413, -73.7781),
    "LGA": (40.7769, -73.8740), "EWR": (40.6895, -74.1745), "SFO": (37.6213, -122.3790),
    "SEA": (47.4502, -122.3088), "LAS": (36.0840, -115.1537), "MCO": (28.4312, -81.3081),
    "CLT": (35.2144, -80.9473), "PHX": (33.4352, -112.0101), "MIA": (25.7959, -80.2870),
    "IAH": (29.9902, -95.3368), "BOS": (42.3656, -71.0096), "MSP": (44.8848, -93.2223),
    "DTW": (42.2124, -83.3534), "PHL": (39.8744, -75.2424), "BWI": (39.1774, -76.6684),
    "SLC": (40.7899, -111.9791), "IAD": (38.9531, -77.4565), "DCA": (38.8512, -77.0402),
    "HNL": (21.3245, -157.9251), "PDX": (45.5898, -122.5951), "STL": (38.7487, -90.3700),
    "TPA": (27.9755, -82.5332), "SAN": (32.7338, -117.1933), "BNA": (36.1263, -86.6774),
    "AUS": (30.1975, -97.6664), "MDW": (41.7868, -87.7522), "FLL": (26.0726, -80.1527),
    "HOU": (29.6454, -95.2789), "OAK": (37.7126, -122.2197), "SJC": (37.3626, -121.9291),
    "SMF": (38.6954, -121.5908), "RDU": (35.8776, -78.7875), "CLE": (41.4117, -81.8498),
    "PIT": (40.4915, -80.2329), "CVG": (39.0489, -84.6678), "IND": (39.7173, -86.2944),
    "CMH": (39.9980, -82.8919), "MCI": (39.2976, -94.7139), "SAT": (29.5337, -98.4698),
    "MSY": (29.9934, -90.2580), "ANC": (61.1743, -149.9982), "OGG": (20.8986, -156.4305),
    "RSW": (26.5362, -81.7552), "PBI": (26.6832, -80.0956), "JAX": (30.4941, -81.6879),
    "OMA": (41.3032, -95.8941),
    "DEL": (28.5562, 77.1000), "BOM": (19.0896, 72.8656), "BLR": (13.1986, 77.7066),
    "MAA": (12.9941, 80.1709), "CCU": (22.6547, 88.4467), "HYD": (17.2403, 78.4294),
    "COK": (10.1520, 76.4019), "GOI": (15.3808, 73.8314), "CJB": (11.0300, 77.0434),
    "IXM": (9.8345, 78.0934), "TRZ": (10.7654, 78.7097), "IXE": (12.9613, 74.8900),
    "HBX": (15.3617, 75.0849), "VGA": (16.5304, 80.7968), "VTZ": (17.7211, 83.2245),
    "TIR": (13.6325, 79.5433),
    "SIN": (1.3644, 103.9915), "KUL": (2.7456, 101.7099), "PEN": (5.2971, 100.2769),
    "LGK": (6.3297, 99.7286), "JHB": (1.6412, 103.6695), "BKI": (5.9372, 116.0511),
    "DXB": (25.2532, 55.3657),
}


def carrier_label(code: str) -> str:
    code = (code or "").upper()
    name = CARRIER_NAMES.get(code)
    return f"{name} ({code})" if name else code


def airport_label(code: str) -> str:
    code = (code or "").upper()
    city = AIRPORT_CITY.get(code)
    return f"{city} ({code})" if city else code
